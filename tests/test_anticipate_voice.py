"""Tests for rag_anticipate.voice — voice brief utility (Phase 2 stub).

Cubre:
1. text_to_audio empty text → None
2. text_to_audio successfully writes file (mock urlopen)
3. text_to_audio cache hit (no http call si file existe)
4. text_to_audio HTTP error → None
5. text_to_audio timeout → None
6. text_to_audio response empty bytes → None
7. is_tts_available True con mock 200
8. is_tts_available False con connection refused
9. cleanup_old_briefs no dir → 0
10. cleanup_old_briefs deletes old, keeps recent
11. cleanup_old_briefs handles permission errors silently
12. _tts_endpoint respect RAG_WEB_BASE_URL env

Mockea `urllib.request.urlopen` con MagicMock devolviendo bytes. No depende
del web server real ni de macOS `say`.
"""

from __future__ import annotations

import os
import time
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_anticipate import voice


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_voice_dir(tmp_path, monkeypatch):
    """Aísla el VOICE_OUT_DIR a un tmp_path para no tocar ~/.local."""
    target = tmp_path / "voice"
    monkeypatch.setattr(voice, "VOICE_OUT_DIR", target)
    return target


def _mock_response(payload: bytes, status: int = 200) -> MagicMock:
    """Construye un MagicMock que se comporta como urllib response (CM)."""
    resp = MagicMock()
    resp.read.return_value = payload
    resp.status = status
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm


# ── 1. text_to_audio: empty text ─────────────────────────────────────────────


def test_text_to_audio_empty_text_returns_none(tmp_voice_dir):
    assert voice.text_to_audio("") is None
    assert voice.text_to_audio("   ") is None
    assert voice.text_to_audio("\n\t  \n") is None
    # No file ni dir creado para inputs vacíos.
    assert not tmp_voice_dir.exists() or not any(tmp_voice_dir.iterdir())


# ── 2. text_to_audio: success path ───────────────────────────────────────────


def test_text_to_audio_success_writes_file(tmp_voice_dir):
    payload = b"RIFF\x00\x00\x00\x00WAVEfmt fakebytes"
    with patch("urllib.request.urlopen", return_value=_mock_response(payload)) as mock_open:
        result = voice.text_to_audio("hola mundo")
    assert result is not None
    assert result.exists()
    assert result.read_bytes() == payload
    assert result.parent == tmp_voice_dir
    assert result.name.startswith("brief-")
    assert result.suffix == ".wav"
    assert mock_open.call_count == 1


# ── 3. text_to_audio: cache hit ──────────────────────────────────────────────


def test_text_to_audio_cache_hit_skips_http(tmp_voice_dir):
    payload = b"first-render-bytes"
    with patch("urllib.request.urlopen", return_value=_mock_response(payload)) as mock_open:
        first = voice.text_to_audio("texto cacheado")
    assert first is not None
    assert first.exists()
    assert mock_open.call_count == 1

    # Segunda llamada con el mismo texto: NO debe llamar HTTP, devuelve el path.
    with patch("urllib.request.urlopen") as mock_open_2:
        second = voice.text_to_audio("texto cacheado")
    assert second == first
    assert mock_open_2.call_count == 0


# ── 4. text_to_audio: HTTP error ─────────────────────────────────────────────


def test_text_to_audio_http_error_returns_none(tmp_voice_dir):
    err = urllib.error.HTTPError(
        url="http://127.0.0.1:8765/api/tts",
        code=500,
        msg="Internal Server Error",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=err):
        result = voice.text_to_audio("texto que rompe el server")
    assert result is None
    # No file written.
    assert tmp_voice_dir.exists()
    assert list(tmp_voice_dir.glob("brief-*.wav")) == []


# ── 5. text_to_audio: timeout ────────────────────────────────────────────────


def test_text_to_audio_timeout_returns_none(tmp_voice_dir):
    import socket
    with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
        result = voice.text_to_audio("texto que tarda demasiado")
    assert result is None
    assert list(tmp_voice_dir.glob("brief-*.wav")) == []


# ── 6. text_to_audio: empty bytes response ───────────────────────────────────


def test_text_to_audio_empty_bytes_returns_none(tmp_voice_dir):
    with patch("urllib.request.urlopen", return_value=_mock_response(b"")):
        result = voice.text_to_audio("respuesta vacía")
    assert result is None
    assert list(tmp_voice_dir.glob("brief-*.wav")) == []


# ── 7. is_tts_available: true con mock 200 ───────────────────────────────────


def test_is_tts_available_true_on_200():
    with patch("urllib.request.urlopen", return_value=_mock_response(b"ok", status=200)):
        assert voice.is_tts_available() is True


# ── 8. is_tts_available: false on connection refused ─────────────────────────


def test_is_tts_available_false_on_connection_refused():
    err = urllib.error.URLError("Connection refused")
    with patch("urllib.request.urlopen", side_effect=err):
        assert voice.is_tts_available() is False


# ── 9. cleanup_old_briefs: no dir → 0 ────────────────────────────────────────


def test_cleanup_old_briefs_no_dir_returns_zero(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert voice.cleanup_old_briefs(out_dir=missing) == 0


# ── 10. cleanup_old_briefs: deletes old, keeps recent ────────────────────────


def test_cleanup_old_briefs_deletes_old_keeps_recent(tmp_voice_dir):
    tmp_voice_dir.mkdir(parents=True, exist_ok=True)
    old = tmp_voice_dir / "brief-old00000000.wav"
    new = tmp_voice_dir / "brief-new00000000.wav"
    old.write_bytes(b"old")
    new.write_bytes(b"new")

    now = time.time()
    # 30 días viejo, max_age=7d → debe borrarse.
    os.utime(old, (now - 30 * 86400, now - 30 * 86400))
    # 1 día viejo, max_age=7d → debe quedarse.
    os.utime(new, (now - 1 * 86400, now - 1 * 86400))

    deleted = voice.cleanup_old_briefs(max_age_days=7, out_dir=tmp_voice_dir)
    assert deleted == 1
    assert not old.exists()
    assert new.exists()


# ── 11. cleanup_old_briefs: permission error silent ──────────────────────────


def test_cleanup_old_briefs_handles_permission_errors_silently(tmp_voice_dir):
    tmp_voice_dir.mkdir(parents=True, exist_ok=True)
    a = tmp_voice_dir / "brief-aaaaaaaaaaaa.wav"
    b = tmp_voice_dir / "brief-bbbbbbbbbbbb.wav"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    now = time.time()
    os.utime(a, (now - 30 * 86400, now - 30 * 86400))
    os.utime(b, (now - 30 * 86400, now - 30 * 86400))

    real_unlink = Path.unlink

    def flaky_unlink(self, *args, **kwargs):
        if self.name == a.name:
            raise PermissionError("denied")
        return real_unlink(self, *args, **kwargs)

    with patch.object(Path, "unlink", flaky_unlink):
        # Debe no romper aunque uno falle.
        deleted = voice.cleanup_old_briefs(max_age_days=7, out_dir=tmp_voice_dir)

    # `b` se eliminó OK; `a` falló silenciosamente.
    assert deleted == 1
    assert a.exists()
    assert not b.exists()


# ── 12. _tts_endpoint respects RAG_WEB_BASE_URL ──────────────────────────────


def test_tts_endpoint_default(monkeypatch):
    monkeypatch.delenv("RAG_WEB_BASE_URL", raising=False)
    assert voice._tts_endpoint() == "http://127.0.0.1:8765/api/tts"


def test_tts_endpoint_respects_env(monkeypatch):
    monkeypatch.setenv("RAG_WEB_BASE_URL", "http://example.local:9999")
    assert voice._tts_endpoint() == "http://example.local:9999/api/tts"


# ── Bonus: dedup_key estable (mismo texto → mismo path) ──────────────────────


def test_text_to_audio_path_deterministic(tmp_voice_dir):
    """Mismo texto produce el mismo path (hash sha256[:12])."""
    payload = b"deterministic"
    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        p1 = voice.text_to_audio("frase determinística")
    # Borrar para forzar segundo render con mismo path computado.
    assert p1 is not None
    p1.unlink()
    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        p2 = voice.text_to_audio("frase determinística")
    assert p2 == p1
