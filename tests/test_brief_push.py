"""Tests para `_brief_push_to_whatsapp`: push automático de morning/today/digest
al chat de WhatsApp del usuario.

El helper reusa `_ambient_config()` como gate (jid + enabled) y
`_ambient_whatsapp_send` como transport — los tests monkeypatch ambos para
quedarse offline y validar el comportamiento de:
  - no-op cuando no hay config
  - send con jid correcto + body que incluye título, path, narrativa
  - rewrite de citaciones a obsidian:// URLs
  - log appendea evento con `whatsapp_sent` bool
"""
import json

import pytest

import rag


@pytest.fixture
def cfg_set(monkeypatch, tmp_path):
    cfg_path = tmp_path / "ambient.json"
    monkeypatch.setattr(rag, "AMBIENT_CONFIG_PATH", cfg_path)
    monkeypatch.setattr(
        rag, "AMBIENT_LOG_PATH", tmp_path / "ambient.jsonl"
    )

    def _set(jid: str | None, enabled: bool = True):
        if jid is None:
            if cfg_path.exists():
                cfg_path.unlink()
            return
        cfg_path.write_text(json.dumps({"jid": jid, "enabled": enabled}))

    return _set


@pytest.fixture
def captured(monkeypatch):
    calls: list[dict] = []

    def fake_send(jid, text):
        calls.append({"jid": jid, "text": text})
        return True

    monkeypatch.setattr(rag, "_ambient_whatsapp_send", fake_send)
    return calls


def test_no_config_is_silent_noop(cfg_set, captured):
    cfg_set(None)
    sent = rag._brief_push_to_whatsapp("Morning", "05-Reviews/x.md", "hola")
    assert sent is False
    assert captured == []


def test_disabled_config_skips_send(cfg_set, captured):
    cfg_set("123@g.us", enabled=False)
    sent = rag._brief_push_to_whatsapp("Morning", "05-Reviews/x.md", "hola")
    assert sent is False
    assert captured == []


def test_sends_to_jid_with_title_and_path_and_body(cfg_set, captured):
    cfg_set("123@g.us")
    sent = rag._brief_push_to_whatsapp(
        "Morning 2026-04-15", "05-Reviews/2026-04-15.md", "Hola, hoy enfocate."
    )
    assert sent is True
    assert len(captured) == 1
    assert captured[0]["jid"] == "123@g.us"
    text = captured[0]["text"]
    assert "Morning 2026-04-15" in text
    assert "05-Reviews/2026-04-15.md" in text
    assert "Hola, hoy enfocate." in text


def test_citations_rewritten_to_obsidian_urls(cfg_set, captured):
    cfg_set("123@g.us")
    rag._brief_push_to_whatsapp(
        "Morning",
        "05-Reviews/x.md",
        "Mirá [Foo](02-Areas/Foo.md) y [bar/baz.md] hoy.",
    )
    text = captured[0]["text"]
    # Markdown link wrappers ya no aparecen — fueron reemplazados.
    assert "[Foo](" not in text
    assert "[bar/baz.md]" not in text
    assert "obsidian://open?vault=" in text
    assert "02-Areas/Foo.md" in text
    assert "bar/baz.md" in text


def test_logs_brief_push_event(cfg_set, captured, tmp_path):
    cfg_set("123@g.us")
    rag._brief_push_to_whatsapp(
        "Morning 2026-04-15", "05-Reviews/x.md", "hola"
    )
    log_path = rag.AMBIENT_LOG_PATH
    assert log_path.exists()
    lines = [
        json.loads(l) for l in log_path.read_text().splitlines() if l.strip()
    ]
    push_events = [e for e in lines if e.get("cmd") == "brief_push"]
    assert len(push_events) == 1
    assert push_events[0]["title"] == "Morning 2026-04-15"
    assert push_events[0]["path"] == "05-Reviews/x.md"
    assert push_events[0]["whatsapp_sent"] is True


def test_send_failure_logged_as_false(cfg_set, monkeypatch, tmp_path):
    cfg_set("123@g.us")
    monkeypatch.setattr(rag, "_ambient_whatsapp_send", lambda j, t: False)
    sent = rag._brief_push_to_whatsapp("X", "y.md", "z")
    assert sent is False
    log_path = rag.AMBIENT_LOG_PATH
    push_events = [
        json.loads(l) for l in log_path.read_text().splitlines() if l.strip()
        if json.loads(l).get("cmd") == "brief_push"
    ]
    assert push_events[0]["whatsapp_sent"] is False
