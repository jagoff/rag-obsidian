"""Tests para `POST /api/wa/send_voice` (Voz Espejo, Feature #5).

Stub `tts_text_to_opus` para no depender de `say` / `ffmpeg` reales.
Stub `bridge_client.send_ptt` para no llamar al bridge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_web_server = pytest.importorskip("web.server")
app = _web_server.app

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_send_voice_rejects_invalid_jid(client):
    """JID sin @ → 400."""
    r = client.post("/api/wa/send_voice", json={"jid": "noatsign", "text": "hola"})
    assert r.status_code == 400


def test_send_voice_rejects_empty_text(client):
    """Empty text → 422 del pydantic min_length."""
    r = client.post("/api/wa/send_voice", json={"jid": "x@y", "text": ""})
    assert r.status_code == 422


def test_send_voice_rejects_too_long_text(client):
    """Text > 2000 → 422."""
    r = client.post("/api/wa/send_voice",
                    json={"jid": "x@y", "text": "a" * 3000})
    assert r.status_code == 422


def test_send_voice_tts_failed(client, monkeypatch, tmp_path):
    """tts_text_to_opus devuelve None → ok=False, error_kind=tts_failed."""
    from rag.integrations.whatsapp import voice as _voice
    monkeypatch.setattr(_voice, "tts_text_to_opus", lambda text, voice="Mónica": None)

    r = client.post("/api/wa/send_voice",
                    json={"jid": "5491155@s.whatsapp.net", "text": "hola, cómo estás"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error_kind"] == "tts_failed"


def test_send_voice_happy_path(client, monkeypatch, tmp_path):
    """tts_text_to_opus + send_ptt funcionan → ok=True, message_id."""
    from rag.integrations.whatsapp import voice as _voice, bridge_client as _bc

    fake_ogg = tmp_path / "fake.ogg"
    fake_ogg.write_bytes(b"fake-ogg-bytes")

    monkeypatch.setattr(_voice, "tts_text_to_opus", lambda text, voice="Mónica": fake_ogg)
    monkeypatch.setattr(_voice, "cleanup", lambda *paths: None)
    monkeypatch.setattr(_bc, "send_ptt",
                        lambda jid, path, reply_to=None: {"message_id": "msg-123"})

    r = client.post("/api/wa/send_voice",
                    json={"jid": "5491155@s.whatsapp.net",
                          "text": "hola, te paso el deck"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["message_id"] == "msg-123"
    assert body["voice"] == "Mónica"


def test_send_voice_bridge_error(client, monkeypatch, tmp_path):
    """bridge.send_ptt tira BridgeError → ok=False, error_kind=bridge_error."""
    from rag.integrations.whatsapp import voice as _voice, bridge_client as _bc

    fake_ogg = tmp_path / "fake.ogg"
    fake_ogg.write_bytes(b"fake")
    monkeypatch.setattr(_voice, "tts_text_to_opus", lambda text, voice="Mónica": fake_ogg)
    monkeypatch.setattr(_voice, "cleanup", lambda *paths: None)

    def raise_bridge(jid, path, reply_to=None):
        raise _bc.BridgeError("connection refused")
    monkeypatch.setattr(_bc, "send_ptt", raise_bridge)

    r = client.post("/api/wa/send_voice",
                    json={"jid": "x@y", "text": "hola muchachos"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error_kind"] == "bridge_error"


def test_voice_healthcheck_ok(client, monkeypatch):
    """Cuando say + ffmpeg están instalados → ok=True, missing=[]."""
    import shutil
    monkeypatch.setattr(shutil, "which",
                        lambda cmd: f"/usr/bin/{cmd}")
    r = client.get("/api/wa/voice/healthcheck")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["say"] is True
    assert body["ffmpeg"] is True
    assert body["voice_default"] == "Mónica"
    assert body["missing"] == []


def test_voice_healthcheck_missing_say(client, monkeypatch):
    """Cuando say falta → ok=False, missing=['say']."""
    import shutil
    def fake_which(cmd):
        return None if cmd == "say" else "/opt/homebrew/bin/ffmpeg"
    monkeypatch.setattr(shutil, "which", fake_which)
    # También bloqueamos el fallback path absoluto de ffmpeg.
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    r = client.get("/api/wa/voice/healthcheck")
    body = r.json()
    assert body["ok"] is False
    assert "say" in body["missing"]


def test_to_opus_uses_ffmpeg_from_path(monkeypatch, tmp_path):
    """to_opus debe usar el ffmpeg resuelto por PATH, no un path hardcodeado."""
    import subprocess

    from rag.integrations.whatsapp import voice as _voice

    input_audio = tmp_path / "voice.webm"
    input_audio.write_bytes(b"audio")
    calls = []

    monkeypatch.setattr(_voice.shutil, "which",
                        lambda cmd: "/custom/bin/ffmpeg" if cmd == "ffmpeg" else None)

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"x" * 300)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(_voice.subprocess, "run", fake_run)

    out = _voice.to_opus(input_audio)

    assert out is not None
    assert calls[0][0] == "/custom/bin/ffmpeg"


def test_send_voice_custom_voice(client, monkeypatch, tmp_path):
    """`voice` param se pasa a tts_text_to_opus."""
    from rag.integrations.whatsapp import voice as _voice, bridge_client as _bc

    voice_used = []
    fake_ogg = tmp_path / "fake.ogg"
    fake_ogg.write_bytes(b"fake")
    def fake_tts(text, voice="Mónica"):
        voice_used.append(voice)
        return fake_ogg
    monkeypatch.setattr(_voice, "tts_text_to_opus", fake_tts)
    monkeypatch.setattr(_voice, "cleanup", lambda *paths: None)
    monkeypatch.setattr(_bc, "send_ptt",
                        lambda jid, path, reply_to=None: {"message_id": "m"})

    r = client.post("/api/wa/send_voice",
                    json={"jid": "x@y", "text": "hola che", "voice": "Pablo"})
    assert r.status_code == 200
    assert r.json()["voice"] == "Pablo"
    assert voice_used == ["Pablo"]
