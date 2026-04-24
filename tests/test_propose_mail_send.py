"""Tests para `propose_mail_send` chat tool + `/api/mail/send` endpoint
(2026-04-24, request del user para el slash `/mail` en el web chat).

Cubre el mismo contrato que `test_whatsapp_send_draft.py` para el equivalente
de WhatsApp, adaptado a la semántica de Gmail (`to` es un email no un JID,
hay subject + body + cc/bcc en lugar de message_text plano).

Cubre:

1. `propose_mail_send` — happy path, validación del email, body vacío
   permitido. Siempre devuelve `needs_clarification=True` para forzar
   la confirmación click-to-send (acción destructiva).
2. `/api/mail/send` — happy path 200, to inválido 400, body vacío 400,
   Gmail API rechaza 502.
3. Tool registration — `propose_mail_send` en `CHAT_TOOLS`,
   `PROPOSAL_TOOL_NAMES`, NO en `PARALLEL_SAFE`. Addendum menciona
   la tool y la palabra "confirmación".
"""
from __future__ import annotations

import json

import pytest

import rag
from fastapi.testclient import TestClient
import web.server as _server
from web import tools as _tools


_client = TestClient(_server.app)


# ── 1. propose_mail_send ───────────────────────────────────────────────


def test_propose_mail_send_happy_path():
    raw = rag.propose_mail_send("foo@bar.com", "Hola", "qué tal")
    payload = json.loads(raw)
    assert payload["kind"] == "mail"
    assert payload["needs_clarification"] is True
    assert payload["proposal_id"].startswith("prop-")
    fields = payload["fields"]
    assert fields["to"] == "foo@bar.com"
    assert fields["subject"] == "Hola"
    assert fields["body"] == "qué tal"
    assert fields["error"] is None
    assert fields["cc"] is None
    assert fields["bcc"] is None


def test_propose_mail_send_strips_whitespace():
    raw = rag.propose_mail_send("  foo@bar.com  ", "  Hola  ", "body")
    fields = json.loads(raw)["fields"]
    assert fields["to"] == "foo@bar.com"
    assert fields["subject"] == "Hola"


def test_propose_mail_send_empty_subject_defaults():
    raw = rag.propose_mail_send("foo@bar.com", "", "body")
    fields = json.loads(raw)["fields"]
    assert fields["subject"] == "(sin asunto)"


def test_propose_mail_send_invalid_email_surfaces_error():
    raw = rag.propose_mail_send("no-arroba", "subject", "body")
    fields = json.loads(raw)["fields"]
    assert fields["error"] == "invalid_email"
    # La card igual se renderiza para que el user edite el destinatario.
    assert json.loads(raw)["needs_clarification"] is True


def test_propose_mail_send_empty_to_surfaces_error():
    raw = rag.propose_mail_send("", "subject", "body")
    fields = json.loads(raw)["fields"]
    assert fields["error"] == "empty_to"


def test_propose_mail_send_empty_body_allowed():
    """Body vacío NO es error — la UI tiene textarea editable y el user
    puede tipear el cuerpo después de revisar el draft del LLM."""
    raw = rag.propose_mail_send("foo@bar.com", "subject", "")
    fields = json.loads(raw)["fields"]
    assert fields["error"] is None
    assert fields["body"] == ""


def test_propose_mail_send_with_cc_bcc():
    raw = rag.propose_mail_send(
        "foo@bar.com", "subj", "body",
        cc="cc1@x.com,cc2@y.com",
        bcc="bcc@z.com",
    )
    fields = json.loads(raw)["fields"]
    assert fields["cc"] == "cc1@x.com,cc2@y.com"
    assert fields["bcc"] == "bcc@z.com"


def test_propose_mail_send_always_needs_clarification_even_on_happy():
    """Mismo invariante que propose_whatsapp_send: NUNCA auto-send.
    El user confirma con click — esto bloquea el camino de un tool-call
    rogue del LLM que dispare un mail real sin ver al user."""
    raw = rag.propose_mail_send("foo@bar.com", "subj", "body")
    payload = json.loads(raw)
    assert payload["needs_clarification"] is True
    # No debería haber ningún campo `created: True`.
    assert "created" not in payload or payload["created"] is False


# ── 2. /api/mail/send endpoint ──────────────────────────────────────────


def test_mail_send_endpoint_happy_path(monkeypatch):
    captured = {}
    def _fake_send(to, subject, body, cc=None, bcc=None):
        captured["to"] = to
        captured["subject"] = subject
        captured["body"] = body
        captured["cc"] = cc
        captured["bcc"] = bcc
        return {"ok": True, "message_id": "msg-123", "thread_id": "thr-456"}
    monkeypatch.setattr(rag, "_send_gmail", _fake_send)

    resp = _client.post("/api/mail/send", json={
        "to": "foo@bar.com",
        "subject": "Hola",
        "body": "qué tal",
        "proposal_id": "prop-abc",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["to"] == "foo@bar.com"
    assert body["message_id"] == "msg-123"
    assert captured["to"] == "foo@bar.com"
    assert captured["subject"] == "Hola"


def test_mail_send_endpoint_strips_to_whitespace(monkeypatch):
    captured = {}
    def _fake_send(to, subject, body, cc=None, bcc=None):
        captured["to"] = to
        return {"ok": True, "message_id": "x"}
    monkeypatch.setattr(rag, "_send_gmail", _fake_send)
    resp = _client.post("/api/mail/send", json={
        "to": "  foo@bar.com  ",
        "subject": "s",
        "body": "b",
    })
    assert resp.status_code == 200
    assert captured["to"] == "foo@bar.com"


def test_mail_send_endpoint_subject_empty_uses_placeholder(monkeypatch):
    captured = {}
    def _fake_send(to, subject, body, cc=None, bcc=None):
        captured["subject"] = subject
        return {"ok": True}
    monkeypatch.setattr(rag, "_send_gmail", _fake_send)
    resp = _client.post("/api/mail/send", json={
        "to": "foo@bar.com",
        "subject": "   ",
        "body": "body",
    })
    assert resp.status_code == 200
    assert captured["subject"] == "(sin asunto)"


def test_mail_send_endpoint_rejects_invalid_to():
    resp = _client.post("/api/mail/send", json={
        "to": "not-an-email",
        "subject": "s",
        "body": "b",
    })
    assert resp.status_code == 400
    assert "to" in resp.json()["detail"].lower() or "@" in resp.json()["detail"]


def test_mail_send_endpoint_rejects_empty_to():
    resp = _client.post("/api/mail/send", json={
        "to": "  ",
        "subject": "s",
        "body": "b",
    })
    assert resp.status_code == 400


def test_mail_send_endpoint_rejects_empty_body():
    resp = _client.post("/api/mail/send", json={
        "to": "foo@bar.com",
        "subject": "s",
        "body": "   ",  # whitespace-only → strip → empty
    })
    assert resp.status_code == 400
    assert "body" in resp.json()["detail"].lower() or "vac" in resp.json()["detail"].lower()


def test_mail_send_endpoint_gmail_error_returns_502(monkeypatch):
    def _fake_send(*a, **kw):
        return {"ok": False, "error": "creds revoked"}
    monkeypatch.setattr(rag, "_send_gmail", _fake_send)
    resp = _client.post("/api/mail/send", json={
        "to": "foo@bar.com",
        "subject": "s",
        "body": "b",
    })
    assert resp.status_code == 502
    assert "creds" in resp.json()["detail"]


def test_mail_send_endpoint_logs_audit_event(monkeypatch):
    """No persistimos el body por privacidad, pero sí queremos un evento
    en el ambient log con to + proposal_id + message_id para
    correlacionar con el turno del chat."""
    monkeypatch.setattr(rag, "_send_gmail", lambda *a, **kw: {
        "ok": True, "message_id": "m1", "thread_id": "t1",
    })
    captured_events = []
    monkeypatch.setattr(rag, "_ambient_log_event", lambda ev: captured_events.append(ev))
    resp = _client.post("/api/mail/send", json={
        "to": "foo@bar.com",
        "subject": "subj",
        "body": "body",
        "proposal_id": "prop-xyz",
    })
    assert resp.status_code == 200
    audit = next((e for e in captured_events if e.get("cmd") == "mail_user_send"), None)
    assert audit is not None
    assert audit["to"] == "foo@bar.com"
    assert audit["proposal_id"] == "prop-xyz"
    assert audit["message_id"] == "m1"
    # NO debería loggear el body en plaintext (privacy).
    assert "body" not in audit or audit.get("body") in (None, "")


# ── 3. Tool registration invariants ────────────────────────────────────


def test_propose_mail_send_is_registered():
    assert _tools.propose_mail_send in _tools.CHAT_TOOLS
    assert "propose_mail_send" in _tools.TOOL_FNS
    # Critical: emite SSE proposal event para que la UI muestre la card.
    assert "propose_mail_send" in _tools.PROPOSAL_TOOL_NAMES


def test_propose_mail_send_is_NOT_parallel_safe():
    """Mismo razonamiento que propose_whatsapp_send: aunque el tool no
    envía, el send a tercero debe ser aislado en su round del LLM."""
    assert "propose_mail_send" not in _tools.PARALLEL_SAFE


def test_tool_addendum_mentions_mail_send():
    addendum = _tools._WEB_TOOL_ADDENDUM
    assert "propose_mail_send" in addendum
    # Debe enseñar que es destructivo y necesita confirmación.
    assert "confirmaci" in addendum.lower()
