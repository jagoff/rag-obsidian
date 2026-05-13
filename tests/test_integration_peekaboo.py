"""Tests para `rag.integrations.peekaboo` — Fase 1 wrapper.

Cubre los paths sin invocar el CLI real ni cargar granite MLX-VLM:
- Gate `RAG_PEEKABOO_ENABLE` (off / on).
- Binary detection (override env, missing).
- Modos inválidos.
- subprocess: success / timeout / non-zero / TCC-denied stderr / empty PNG.
- Caption: VLM empty, VLM exception, happy path.
- `keep_image=True` preserva el archivo; `keep_image=False` lo borra.

Cero dependencia de `peekaboo` instalado, cero load de modelos MLX.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rag.integrations import peekaboo as pk


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Reset env vars del módulo entre tests para evitar leak."""
    for var in ("RAG_PEEKABOO_ENABLE", "RAG_PEEKABOO_BIN", "RAG_PEEKABOO_TIMEOUT_SECS"):
        monkeypatch.delenv(var, raising=False)
    yield


def _fake_completed(returncode: int = 0, stderr: str = "", stdout: str = ""):
    return subprocess.CompletedProcess(
        args=["peekaboo"], returncode=returncode, stdout=stdout, stderr=stderr,
    )


# --- gate ---

def test_disabled_short_circuits(monkeypatch):
    monkeypatch.delenv("RAG_PEEKABOO_ENABLE", raising=False)
    called = []
    monkeypatch.setattr(pk, "_capture_png", lambda **kw: called.append(kw) or (None, "should_not_be_called"))
    out = pk.capture_and_caption(mode="frontmost")
    assert out["ok"] is False
    assert out["error"] == "peekaboo_disabled"
    assert called == []
    assert out["took_ms"] >= 0


def test_enabled_off_value(monkeypatch):
    monkeypatch.setenv("RAG_PEEKABOO_ENABLE", "no")
    assert pk._is_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_enabled_on_values(monkeypatch, value):
    monkeypatch.setenv("RAG_PEEKABOO_ENABLE", value)
    assert pk._is_enabled() is True


# --- binary resolution ---

def test_binary_override_missing(monkeypatch, tmp_path):
    ghost = tmp_path / "nope-peekaboo"
    monkeypatch.setenv("RAG_PEEKABOO_BIN", str(ghost))
    assert pk._resolve_binary() is None


def test_binary_override_existing(monkeypatch, tmp_path):
    fake = tmp_path / "peekaboo"
    fake.write_text("#!/bin/sh\necho fake\n")
    fake.chmod(0o755)
    monkeypatch.setenv("RAG_PEEKABOO_BIN", str(fake))
    assert pk._resolve_binary() == str(fake)


def test_binary_via_path_missing(monkeypatch):
    monkeypatch.delenv("RAG_PEEKABOO_BIN", raising=False)
    monkeypatch.setattr(pk.shutil, "which", lambda _: None)
    assert pk._resolve_binary() is None


# --- mode validation ---

def test_invalid_mode_short_circuits(monkeypatch):
    monkeypatch.setattr(pk, "_resolve_binary", lambda: "/usr/bin/peekaboo-fake")
    path, err = pk._capture_png(mode="bogus")
    assert path is None
    assert err is not None and err.startswith("invalid_mode")


# --- subprocess paths ---

def test_capture_no_binary(monkeypatch):
    monkeypatch.setattr(pk, "_resolve_binary", lambda: None)
    path, err = pk._capture_png(mode="frontmost")
    assert path is None
    assert err == "peekaboo_not_installed"


def test_capture_timeout(monkeypatch):
    monkeypatch.setattr(pk, "_resolve_binary", lambda: "/usr/bin/peekaboo-fake")

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="peekaboo", timeout=10)

    monkeypatch.setattr(pk.subprocess, "run", raise_timeout)
    path, err = pk._capture_png(mode="frontmost")
    assert path is None
    assert err == "peekaboo_timeout"


def test_capture_tcc_denied(monkeypatch):
    monkeypatch.setattr(pk, "_resolve_binary", lambda: "/usr/bin/peekaboo-fake")
    monkeypatch.setattr(
        pk.subprocess, "run",
        lambda *a, **k: _fake_completed(
            returncode=1,
            stderr="Error: Screen recording permission is required.",
        ),
    )
    path, err = pk._capture_png(mode="frontmost")
    assert path is None
    assert err is not None and err.startswith("tcc_denied:")


def test_capture_non_zero_exit(monkeypatch):
    monkeypatch.setattr(pk, "_resolve_binary", lambda: "/usr/bin/peekaboo-fake")
    monkeypatch.setattr(
        pk.subprocess, "run",
        lambda *a, **k: _fake_completed(returncode=2, stderr="some unrelated failure"),
    )
    path, err = pk._capture_png(mode="frontmost")
    assert path is None
    assert err is not None and err.startswith("peekaboo_failed")


def test_capture_empty_png(monkeypatch):
    """peekaboo reporta success pero no escribió bytes."""
    monkeypatch.setattr(pk, "_resolve_binary", lambda: "/usr/bin/peekaboo-fake")

    def fake_run(cmd, **kw):
        # cmd[-1] es el path destino — dejarlo como archivo vacío.
        out_path = Path(cmd[cmd.index("--path") + 1])
        out_path.write_bytes(b"")
        return _fake_completed(returncode=0)

    monkeypatch.setattr(pk.subprocess, "run", fake_run)
    path, err = pk._capture_png(mode="frontmost")
    assert path is None
    assert err == "peekaboo_empty_output"


def test_capture_success(monkeypatch):
    monkeypatch.setattr(pk, "_resolve_binary", lambda: "/usr/bin/peekaboo-fake")

    def fake_run(cmd, **kw):
        out_path = Path(cmd[cmd.index("--path") + 1])
        out_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")
        return _fake_completed(returncode=0)

    monkeypatch.setattr(pk.subprocess, "run", fake_run)
    path, err = pk._capture_png(mode="frontmost", retina=True)
    assert err is None
    assert path is not None and path.exists()
    assert path.stat().st_size > 0
    path.unlink(missing_ok=True)


# --- caption ---

def test_caption_vlm_empty(monkeypatch, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"png")
    monkeypatch.setattr("rag.ocr._vlm_describe", lambda *a, **k: "")
    text, err = pk._caption(img)
    assert text == ""
    assert err == "vlm_empty"


def test_caption_vlm_exception(monkeypatch, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"png")

    def boom(*a, **k):
        raise RuntimeError("mlx blew up")

    monkeypatch.setattr("rag.ocr._vlm_describe", boom)
    text, err = pk._caption(img)
    assert text == ""
    assert err is not None and err.startswith("vlm_error")


def test_caption_happy(monkeypatch, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"png")
    monkeypatch.setattr("rag.ocr._vlm_describe", lambda *a, **k: "  Captura de Safari mostrando GitHub.  ")
    text, err = pk._caption(img, prompt="describe")
    assert err is None
    assert text == "Captura de Safari mostrando GitHub."


# --- capture_and_caption integration ---

def test_capture_and_caption_happy(monkeypatch):
    """End-to-end del wrapper con subprocess + VLM mockeados."""
    monkeypatch.setenv("RAG_PEEKABOO_ENABLE", "1")
    monkeypatch.setattr(pk, "_resolve_binary", lambda: "/usr/bin/peekaboo-fake")

    written_paths = []

    def fake_run(cmd, **kw):
        out_path = Path(cmd[cmd.index("--path") + 1])
        out_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        written_paths.append(out_path)
        return _fake_completed(returncode=0)

    monkeypatch.setattr(pk.subprocess, "run", fake_run)
    monkeypatch.setattr("rag.ocr._vlm_describe", lambda *a, **k: "ventana de Terminal con prompt zsh")

    out = pk.capture_and_caption(mode="frontmost", keep_image=False)
    assert out["ok"] is True
    assert "Terminal" in out["caption"]
    assert out["mode"] == "frontmost"
    assert out["image_path"] is None
    assert out["error"] is None
    assert out["took_ms"] >= 0
    # PNG deletada porque keep_image=False
    assert all(not p.exists() for p in written_paths)


def test_capture_and_caption_keeps_image(monkeypatch):
    monkeypatch.setenv("RAG_PEEKABOO_ENABLE", "1")
    monkeypatch.setattr(pk, "_resolve_binary", lambda: "/usr/bin/peekaboo-fake")

    def fake_run(cmd, **kw):
        out_path = Path(cmd[cmd.index("--path") + 1])
        out_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        return _fake_completed(returncode=0)

    monkeypatch.setattr(pk.subprocess, "run", fake_run)
    monkeypatch.setattr("rag.ocr._vlm_describe", lambda *a, **k: "caption")

    out = pk.capture_and_caption(mode="frontmost", keep_image=True)
    assert out["ok"] is True
    assert out["image_path"] is not None
    persisted = Path(out["image_path"])
    try:
        assert persisted.exists() and persisted.stat().st_size > 0
        # Permisos 0600 — privacy invariant.
        assert oct(persisted.stat().st_mode)[-3:] == "600"
    finally:
        persisted.unlink(missing_ok=True)


def test_capture_and_caption_tcc_denied_surfaces(monkeypatch):
    """Error TCC del subprocess propaga sin disparar VLM."""
    monkeypatch.setenv("RAG_PEEKABOO_ENABLE", "1")
    monkeypatch.setattr(pk, "_resolve_binary", lambda: "/usr/bin/peekaboo-fake")
    monkeypatch.setattr(
        pk.subprocess, "run",
        lambda *a, **k: _fake_completed(
            returncode=1, stderr="Error: Screen recording permission is required.",
        ),
    )

    vlm_calls = []
    monkeypatch.setattr(
        "rag.ocr._vlm_describe",
        lambda *a, **k: vlm_calls.append((a, k)) or "should_not_be_called",
    )

    out = pk.capture_and_caption(mode="frontmost")
    assert out["ok"] is False
    assert out["error"].startswith("tcc_denied")
    assert out["caption"] == ""
    assert vlm_calls == [], "VLM no debe invocarse cuando capture falla"


def test_capture_and_caption_invalid_mode(monkeypatch):
    monkeypatch.setenv("RAG_PEEKABOO_ENABLE", "1")
    out = pk.capture_and_caption(mode="totally-bogus")
    assert out["ok"] is False
    assert out["error"].startswith("invalid_mode")


def test_capture_and_caption_disabled_skips_subprocess(monkeypatch):
    """Default OFF: ni siquiera tocamos `_capture_png`."""
    called = {"n": 0}

    def trip(*a, **k):
        called["n"] += 1
        return (None, "should_not_run")

    monkeypatch.setattr(pk, "_capture_png", trip)
    out = pk.capture_and_caption(mode="frontmost")
    assert called["n"] == 0
    assert out["error"] == "peekaboo_disabled"
