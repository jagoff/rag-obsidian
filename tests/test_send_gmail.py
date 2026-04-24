"""Tests for `_send_gmail` + `rag send-mail` CLI.

Origin: user report 2026-04-24 — "quiero enviar un mail desde wsp pero no
lo envía". El WhatsApp listener antes solo LEÍA mails (/mail [N]); no
había tool para enviar. Este archivo testea la capa Python que el
listener invoca como subprocess (`rag send-mail --to X --subject Y --body
Z --json`).

Cobertura:
  - `_send_gmail` con Gmail API mockeada (happy path, error, sin creds).
  - Validación de args (to inválido, body vacío, subject default).
  - `rag send-mail` CLI (exit codes + JSON output).
"""
from __future__ import annotations

import base64
import email
import json

import pytest
from click.testing import CliRunner

import rag


# ── 1. _send_gmail helper ──────────────────────────────────────────────────


class _FakeSendResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def execute(self):
        return self._payload


class _FakeMessages:
    def __init__(self, response: dict):
        self._response = response
        self.calls: list[dict] = []

    def send(self, userId: str, body: dict):
        self.calls.append({"userId": userId, "body": body})
        return _FakeSendResponse(self._response)


class _FakeUsers:
    def __init__(self, msgs: _FakeMessages):
        self._msgs = msgs

    def messages(self):
        return self._msgs


class _FakeGmailService:
    def __init__(self, response: dict | None = None):
        self._msgs = _FakeMessages(response or {"id": "fake123", "threadId": "t456"})

    def users(self):
        return _FakeUsers(self._msgs)


def test_send_gmail_no_creds(monkeypatch):
    """Sin creds gmail-send el helper devuelve ok=False sin crashear."""
    monkeypatch.setattr(rag, "_gmail_send_service", lambda: None)
    result = rag._send_gmail("fer@ejemplo.com", "test", "hola")
    assert result["ok"] is False
    assert "creds" in result["error"]


def test_send_gmail_invalid_to_empty():
    result = rag._send_gmail("", "subject", "body")
    assert result["ok"] is False
    assert "to vacío" in result["error"]


def test_send_gmail_invalid_to_no_at(monkeypatch):
    """`to` sin @ se rechaza antes de pegarle al API."""
    monkeypatch.setattr(rag, "_gmail_send_service", lambda: _FakeGmailService())
    result = rag._send_gmail("no-es-un-email", "x", "y")
    assert result["ok"] is False
    assert "email válido" in result["error"]


def test_send_gmail_happy_path(monkeypatch):
    svc = _FakeGmailService(response={"id": "msg_abc", "threadId": "th_xyz"})
    monkeypatch.setattr(rag, "_gmail_send_service", lambda: svc)

    result = rag._send_gmail(
        to="fer@ejemplo.com",
        subject="test",
        body="hola mundo",
    )
    assert result["ok"] is True
    assert result["message_id"] == "msg_abc"
    assert result["thread_id"] == "th_xyz"

    # Verificá que el raw payload tenga headers + body correctos.
    call = svc._msgs.calls[0]
    raw = call["body"]["raw"]
    decoded = base64.urlsafe_b64decode(raw).decode("utf-8")
    msg = email.message_from_string(decoded)
    assert msg["To"] == "fer@ejemplo.com"
    assert msg["Subject"] == "test"
    payload = msg.get_payload(decode=True).decode("utf-8")
    assert "hola mundo" in payload
    # Message-ID + Date se setean automáticos.
    assert msg["Message-ID"]
    assert msg["Date"]


def test_send_gmail_empty_subject_gets_default(monkeypatch):
    """subject vacío se reemplaza con '(sin asunto)' — evita mail sin subject."""
    svc = _FakeGmailService()
    monkeypatch.setattr(rag, "_gmail_send_service", lambda: svc)
    rag._send_gmail(to="fer@ejemplo.com", subject="", body="hi")

    call = svc._msgs.calls[0]
    raw = call["body"]["raw"]
    decoded = base64.urlsafe_b64decode(raw).decode("utf-8")
    msg = email.message_from_string(decoded)
    assert msg["Subject"] == "(sin asunto)"


def test_send_gmail_with_cc_bcc(monkeypatch):
    svc = _FakeGmailService()
    monkeypatch.setattr(rag, "_gmail_send_service", lambda: svc)
    rag._send_gmail(
        to="fer@ejemplo.com",
        subject="subj",
        body="body",
        cc="cc1@x.com, cc2@x.com",
        bcc="hidden@x.com",
    )
    call = svc._msgs.calls[0]
    raw = call["body"]["raw"]
    decoded = base64.urlsafe_b64decode(raw).decode("utf-8")
    msg = email.message_from_string(decoded)
    assert msg["Cc"] == "cc1@x.com, cc2@x.com"
    assert msg["Bcc"] == "hidden@x.com"


def test_send_gmail_api_exception_becomes_error(monkeypatch):
    class _BoomMessages:
        def send(self, **_kw):
            raise RuntimeError("429 rate limited")
    class _BoomSvc:
        def users(self):
            class _U:
                def messages(self_inner):
                    return _BoomMessages()
            return _U()
    monkeypatch.setattr(rag, "_gmail_send_service", lambda: _BoomSvc())

    result = rag._send_gmail("fer@ejemplo.com", "x", "y")
    assert result["ok"] is False
    assert "rate limited" in result["error"] or "RuntimeError" in result["error"]


# ── 2. `rag send-mail` CLI ─────────────────────────────────────────────────


def test_cli_send_mail_ok(monkeypatch):
    """Exit 0 + JSON con ok=true cuando el envío funciona."""
    svc = _FakeGmailService(response={"id": "CLI_OK", "threadId": "CLI_TH"})
    monkeypatch.setattr(rag, "_gmail_send_service", lambda: svc)

    runner = CliRunner()
    result = runner.invoke(rag.cli, [
        "send-mail",
        "--to", "fer@ejemplo.com",
        "--subject", "cli test",
        "--body", "desde cli",
        "--json",
    ])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output.strip())
    assert parsed["ok"] is True
    assert parsed["message_id"] == "CLI_OK"


def test_cli_send_mail_failure_exit_1(monkeypatch):
    """Gmail API error → exit 1, JSON con ok=false."""
    class _BoomSvc:
        def users(self):
            class _U:
                def messages(self_inner):
                    class _M:
                        def send(self_m, **_kw):
                            raise RuntimeError("quota exceeded")
                    return _M()
            return _U()
    monkeypatch.setattr(rag, "_gmail_send_service", lambda: _BoomSvc())

    runner = CliRunner()
    result = runner.invoke(rag.cli, [
        "send-mail", "--to", "x@x.com", "--subject", "s",
        "--body", "b", "--json",
    ])
    assert result.exit_code == 1
    parsed = json.loads(result.output.strip())
    assert parsed["ok"] is False
    assert "quota" in parsed["error"]


def test_cli_send_mail_bad_to_exit_2(monkeypatch):
    """--to sin @ → exit 2 (argumentos inválidos)."""
    # No hace falta mockear el service; la validación pre-vuela.
    runner = CliRunner()
    result = runner.invoke(rag.cli, [
        "send-mail", "--to", "not-an-email",
        "--subject", "s", "--body", "b", "--json",
    ])
    assert result.exit_code == 2
    parsed = json.loads(result.output.strip())
    assert parsed["ok"] is False
    assert "--to inválido" in parsed["error"]


def test_cli_send_mail_body_from_stdin(monkeypatch):
    """Si --body no viene, lee de stdin."""
    svc = _FakeGmailService()
    monkeypatch.setattr(rag, "_gmail_send_service", lambda: svc)

    runner = CliRunner()
    result = runner.invoke(rag.cli, [
        "send-mail", "--to", "fer@ejemplo.com", "--subject", "s", "--json",
    ], input="body desde stdin\nlínea 2")
    assert result.exit_code == 0

    # Body debe haber llegado al raw payload.
    raw = svc._msgs.calls[0]["body"]["raw"]
    decoded = base64.urlsafe_b64decode(raw).decode("utf-8")
    msg = email.message_from_string(decoded)
    payload = msg.get_payload(decode=True).decode("utf-8")
    assert "body desde stdin" in payload
    assert "línea 2" in payload


def test_cli_send_mail_human_output_on_success(monkeypatch):
    """Sin --json el output es human-readable (no JSON)."""
    svc = _FakeGmailService()
    monkeypatch.setattr(rag, "_gmail_send_service", lambda: svc)

    runner = CliRunner()
    result = runner.invoke(rag.cli, [
        "send-mail", "--to", "fer@ejemplo.com", "--body", "hola",
    ])
    assert result.exit_code == 0
    # No debería ser JSON válido cuando no se pasa --json.
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.output.strip())
    assert "enviado" in result.output.lower()
