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
    for var in (
        "RAG_PEEKABOO_ENABLE", "RAG_PEEKABOO_BIN", "RAG_PEEKABOO_TIMEOUT_SECS",
        "RAG_SCREEN_OBSERVE", "RAG_SCREEN_QUIET_HOURS", "RAG_SCREEN_APP_DENY",
    ):
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


# ── Fase 2 — passive observer ────────────────────────────────────────────────


from datetime import datetime  # noqa: E402


# --- quiet hours ---


@pytest.mark.parametrize("spec,expected", [
    (None, None),
    ("", None),
    ("bogus", None),
    ("22:00", None),  # missing dash
    ("22:00-07:00", (1320, 420)),
    ("00:00-23:59", (0, 1439)),
    ("99:99-00:00", None),  # out of range
])
def test_parse_quiet_hours(spec, expected):
    assert pk._parse_quiet_hours(spec) == expected


@pytest.mark.parametrize("hh,mm,spec,expected", [
    # wrap window 22:00-07:00
    (23, 30, "22:00-07:00", True),
    (3, 0, "22:00-07:00", True),
    (12, 0, "22:00-07:00", False),
    (22, 0, "22:00-07:00", True),
    (7, 0, "22:00-07:00", False),  # exclusive end
    # non-wrap window 09:00-17:00
    (12, 0, "09:00-17:00", True),
    (8, 59, "09:00-17:00", False),
    (17, 0, "09:00-17:00", False),
    # no spec → never quiet
    (3, 0, "", False),
    (3, 0, None, False),
])
def test_in_quiet_hours(hh, mm, spec, expected):
    now = datetime(2026, 5, 13, hh, mm)
    assert pk._in_quiet_hours(now, spec) is expected


# --- denylist ---


def test_app_denylist_empty(monkeypatch):
    monkeypatch.delenv("RAG_SCREEN_APP_DENY", raising=False)
    assert pk._app_denylist() == frozenset()


def test_app_denylist_csv(monkeypatch):
    monkeypatch.setenv("RAG_SCREEN_APP_DENY", "1Password, Banking ,, Messages")
    assert pk._app_denylist() == frozenset({"1password", "banking", "messages"})


# --- simhash ---


def test_simhash_deterministic():
    assert pk._simhash64("hello") == pk._simhash64("hello")
    assert pk._simhash64("hello") != pk._simhash64("hellp")


def test_simhash_fits_signed_int64():
    """SQLite INTEGER es signed 64-bit. Verificamos que el output cabe."""
    for sample in ["", "x", "y" * 1000, "café 🌶️ unicode"]:
        v = pk._simhash64(sample)
        assert -(2**63) <= v < 2**63


# --- observe_once gates ---


def test_observe_disabled_default(monkeypatch):
    monkeypatch.delenv("RAG_SCREEN_OBSERVE", raising=False)
    out = pk.observe_once()
    assert out["ok"] is False
    assert out["skipped_reason"] == "observe_disabled"


def test_observe_enabled_but_capture_gate_off(monkeypatch):
    monkeypatch.setenv("RAG_SCREEN_OBSERVE", "1")
    monkeypatch.delenv("RAG_PEEKABOO_ENABLE", raising=False)
    out = pk.observe_once()
    assert out["skipped_reason"] == "peekaboo_disabled"


def test_observe_quiet_hours_blocks(monkeypatch):
    monkeypatch.setenv("RAG_SCREEN_OBSERVE", "1")
    monkeypatch.setenv("RAG_PEEKABOO_ENABLE", "1")
    monkeypatch.setenv("RAG_SCREEN_QUIET_HOURS", "00:00-23:59")  # all day
    captured = {"n": 0}
    monkeypatch.setattr(pk, "_capture_with_meta", lambda **kw: captured.__setitem__("n", captured["n"] + 1) or (None, {}, "should_not_be_called"))
    out = pk.observe_once(now=datetime(2026, 5, 13, 12, 0))
    assert out["skipped_reason"] == "quiet_hours"
    assert captured["n"] == 0, "no debe llamar al capture si quiet hours"


def test_observe_app_denied_post_capture(monkeypatch, tmp_path):
    """App detectada en denylist → PNG borrada, sin VLM call, sin DB write."""
    monkeypatch.setenv("RAG_SCREEN_OBSERVE", "1")
    monkeypatch.setenv("RAG_PEEKABOO_ENABLE", "1")
    monkeypatch.setenv("RAG_SCREEN_APP_DENY", "1Password,Banking")

    fake_png = tmp_path / "fake.png"
    fake_png.write_bytes(b"\x89PNG")

    def fake_capture(**kw):
        return (fake_png, {"app_name": "1Password", "window_title": "vault"}, None)

    monkeypatch.setattr(pk, "_capture_with_meta", fake_capture)
    monkeypatch.setattr("rag.ocr._vlm_describe", lambda *a, **k: pytest.fail("VLM should not run"))

    out = pk.observe_once()
    assert out["skipped_reason"] == "app_denied"
    assert out["app_name"] == "1Password"
    assert not fake_png.exists(), "PNG debe borrarse"


# --- observe_once dedup + insert ---


def _mock_telemetry(monkeypatch, tmp_path):
    """Setea DB_PATH a tmp_path y ensure-tablas. Returns sqlite3.Connection abierta."""
    import sqlite3
    import rag as _rag
    monkeypatch.setattr(_rag, "DB_PATH", tmp_path)
    con = sqlite3.connect(str(tmp_path / "telemetry.db"))
    _rag._ensure_telemetry_tables(con)
    con.commit()
    return con


def test_observe_dedup_title_skips_vlm(monkeypatch, tmp_path):
    """Si el title del frontmost matchea la última observación de la misma
    app en los últimos 60s → skip caption, no insert."""
    monkeypatch.setenv("RAG_SCREEN_OBSERVE", "1")
    monkeypatch.setenv("RAG_PEEKABOO_ENABLE", "1")

    con = _mock_telemetry(monkeypatch, tmp_path)
    import time as _time
    now_ts = int(_time.time())
    con.execute(
        "INSERT INTO rag_screen_observations "
        "(ts, app_name, window_title, caption, caption_simhash, took_ms, capture_mode) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (now_ts, "Ghostty", "fer — ~/repos/rag", "previous caption", 123, 200, "frontmost"),
    )
    con.commit()
    con.close()

    fake_png = tmp_path / "fake.png"
    fake_png.write_bytes(b"\x89PNG")

    def fake_capture(**kw):
        return (fake_png, {"app_name": "Ghostty", "window_title": "fer — ~/repos/rag"}, None)

    monkeypatch.setattr(pk, "_capture_with_meta", fake_capture)
    monkeypatch.setattr("rag.ocr._vlm_describe", lambda *a, **k: pytest.fail("VLM should not run on dedup hit"))

    out = pk.observe_once()
    assert out["skipped_reason"] == "dedup_title"
    assert not fake_png.exists()


def test_observe_inserts_new_row(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_SCREEN_OBSERVE", "1")
    monkeypatch.setenv("RAG_PEEKABOO_ENABLE", "1")

    _mock_telemetry(monkeypatch, tmp_path)
    fake_png = tmp_path / "fake.png"
    fake_png.write_bytes(b"\x89PNG")

    def fake_capture(**kw):
        return (fake_png, {"app_name": "Code", "window_title": "server.py — rag"}, None)

    monkeypatch.setattr(pk, "_capture_with_meta", fake_capture)
    monkeypatch.setattr("rag.ocr._vlm_describe", lambda *a, **k: "Visual Studio Code mostrando server.py de FastAPI")

    out = pk.observe_once(now=datetime(2026, 5, 13, 14, 30))
    assert out["ok"] is True
    assert out["observation_id"] is not None
    assert out["app_name"] == "Code"
    assert out["window_title"] == "server.py — rag"
    assert "FastAPI" in out["caption"]
    assert out["skipped_reason"] is None
    assert not fake_png.exists()

    # Verificá row escrita.
    import sqlite3
    con = sqlite3.connect(str(tmp_path / "telemetry.db"))
    row = con.execute(
        "SELECT app_name, window_title, caption, capture_mode FROM rag_screen_observations "
        "WHERE id = ?",
        (out["observation_id"],),
    ).fetchone()
    con.close()
    assert row == ("Code", "server.py — rag", "Visual Studio Code mostrando server.py de FastAPI", "frontmost")


def test_observe_vlm_empty_no_insert(monkeypatch, tmp_path):
    """VLM devolvió string vacío → no escribimos row (ruido)."""
    monkeypatch.setenv("RAG_SCREEN_OBSERVE", "1")
    monkeypatch.setenv("RAG_PEEKABOO_ENABLE", "1")

    _mock_telemetry(monkeypatch, tmp_path)
    fake_png = tmp_path / "fake.png"
    fake_png.write_bytes(b"\x89PNG")

    def fake_capture(**kw):
        return (fake_png, {"app_name": "App", "window_title": "blank"}, None)

    monkeypatch.setattr(pk, "_capture_with_meta", fake_capture)
    monkeypatch.setattr("rag.ocr._vlm_describe", lambda *a, **k: "")

    out = pk.observe_once()
    assert out["skipped_reason"] == "vlm_empty"
    assert out["observation_id"] is None

    import sqlite3
    con = sqlite3.connect(str(tmp_path / "telemetry.db"))
    cnt = con.execute("SELECT COUNT(*) FROM rag_screen_observations").fetchone()[0]
    con.close()
    assert cnt == 0


def test_observe_capture_tcc_denied(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_SCREEN_OBSERVE", "1")
    monkeypatch.setenv("RAG_PEEKABOO_ENABLE", "1")
    _mock_telemetry(monkeypatch, tmp_path)
    monkeypatch.setattr(pk, "_capture_with_meta", lambda **kw: (None, {}, "tcc_denied: blah"))
    monkeypatch.setattr("rag.ocr._vlm_describe", lambda *a, **k: pytest.fail("VLM should not run"))
    out = pk.observe_once()
    assert out["ok"] is False
    assert out["error"].startswith("tcc_denied")
    assert out["observation_id"] is None


# --- state-file opt-in pattern (Fase 2g) ---


def test_observe_state_disabled_default(monkeypatch, tmp_path):
    """State file no existe → _observe_state_enabled() False."""
    fake_state = tmp_path / "screen_observe_enabled"
    monkeypatch.setattr(pk, "_OBSERVE_STATE_FILE", fake_state)
    assert pk._observe_state_enabled() is False


def test_observe_state_set_creates_file(monkeypatch, tmp_path):
    fake_state = tmp_path / "subdir" / "screen_observe_enabled"
    monkeypatch.setattr(pk, "_OBSERVE_STATE_FILE", fake_state)
    pk._observe_state_set(True)
    assert fake_state.is_file()
    assert pk._observe_state_enabled() is True


def test_observe_state_set_idempotent(monkeypatch, tmp_path):
    fake_state = tmp_path / "screen_observe_enabled"
    monkeypatch.setattr(pk, "_OBSERVE_STATE_FILE", fake_state)
    pk._observe_state_set(True)
    pk._observe_state_set(True)  # no-op, no error
    assert fake_state.is_file()
    pk._observe_state_set(False)
    pk._observe_state_set(False)  # no-op, no error
    assert not fake_state.is_file()


def test_observe_state_set_false_removes(monkeypatch, tmp_path):
    fake_state = tmp_path / "screen_observe_enabled"
    fake_state.write_text("")
    monkeypatch.setattr(pk, "_OBSERVE_STATE_FILE", fake_state)
    assert pk._observe_state_enabled() is True
    pk._observe_state_set(False)
    assert not fake_state.exists()
    assert pk._observe_state_enabled() is False


def test_supervisor_plist_injects_env_when_enabled(monkeypatch, tmp_path):
    """Plist generator agrega RAG_PEEKABOO_ENABLE+RAG_SCREEN_OBSERVE
    cuando el state file existe."""
    fake_state = tmp_path / "screen_observe_enabled"
    fake_state.write_text("")
    monkeypatch.setattr(pk, "_OBSERVE_STATE_FILE", fake_state)
    from rag.plists.persistent import _supervisor_plist
    out = _supervisor_plist("/usr/local/bin/rag")
    assert "RAG_PEEKABOO_ENABLE" in out
    assert "RAG_SCREEN_OBSERVE" in out


def test_supervisor_plist_skips_env_when_disabled(monkeypatch, tmp_path):
    fake_state = tmp_path / "screen_observe_enabled_NOT_THERE"
    monkeypatch.setattr(pk, "_OBSERVE_STATE_FILE", fake_state)
    from rag.plists.persistent import _supervisor_plist
    out = _supervisor_plist("/usr/local/bin/rag")
    assert "RAG_PEEKABOO_ENABLE" not in out
    assert "RAG_SCREEN_OBSERVE" not in out


def test_query_last_observation_within_window(monkeypatch, tmp_path):
    """Verifica el helper _query_last_observation contra una DB con rows."""
    con = _mock_telemetry(monkeypatch, tmp_path)
    import time as _time
    now_ts = int(_time.time())
    rows = [
        (now_ts - 30, "Ghostty", "shell-A", "cap A", 1, 100, "frontmost"),  # in window
        (now_ts - 90, "Ghostty", "shell-B", "cap B", 2, 100, "frontmost"),  # outside 60s
        (now_ts - 10, "Safari",  "tab-X",   "cap X", 3, 100, "frontmost"),  # different app
    ]
    con.executemany(
        "INSERT INTO rag_screen_observations "
        "(ts, app_name, window_title, caption, caption_simhash, took_ms, capture_mode) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()

    hit = pk._query_last_observation(con, "Ghostty", within_seconds=60)
    assert hit is not None
    assert hit["window_title"] == "shell-A"

    miss = pk._query_last_observation(con, "Ghostty", within_seconds=10)
    assert miss is None  # the only recent Ghostty row is 30s old

    miss_app = pk._query_last_observation(con, "Nonexistent", within_seconds=3600)
    assert miss_app is None

    con.close()
