"""Tests para `_handle_chat_create_intent` (Fase 3).

El flow replica el del web /api/chat pero adaptado al loop del CLI
`rag chat`: single-round tool-decide → tool exec → chip Rich. Returns
True si el tool dispara, False si el LLM rehusa o hay error.

Cubre: happy path (tool disparó + evento creado), needs_clarification,
created:false, tool desconocido, LLM sin tool_calls, exception en la
llamada a ollama.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import rag


def _fake_ollama_response(tool_calls: list) -> MagicMock:
    """Shape a minimal ollama.ChatResponse with .message.tool_calls."""
    msg = MagicMock()
    msg.tool_calls = tool_calls or []
    msg.content = ""
    resp = MagicMock()
    resp.message = msg
    return resp


def _fake_tool_call(name: str, args: dict) -> MagicMock:
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = args
    return tc


# ── happy path: tool fired + created ────────────────────────────────────────


def test_handle_chat_create_intent_happy_reminder(monkeypatch):
    """LLM llama propose_reminder → rag._create_reminder creado → chip
    printea OK → returns True."""
    # Mock propose_reminder's dependencies so _create_reminder "succeeds"
    # without touching osascript.
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda txt, now=None, **kw: datetime(2026, 4, 22, 10, 0, 0),
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)
    monkeypatch.setattr(
        rag, "_create_reminder",
        lambda *a, **kw: (True, "x-apple-reminderkit://ABC"),
    )
    # Mock warmup so it doesn't block / schedule threads.
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "test-model")

    # Mock the ollama client's chat() to return a tool_call for propose_reminder.
    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response([
        _fake_tool_call("propose_reminder", {"title": "llamar a mama", "when": "mañana 10am"}),
    ])
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    handled, created = rag._handle_chat_create_intent("recordame llamar a mama mañana 10am")
    assert handled is True
    # Reminder creado → created_info lleva el id para el /undo del CLI.
    assert created is not None
    assert created["kind"] == "reminder"
    assert created["reminder_id"] == "x-apple-reminderkit://ABC"
    # Verify ollama was called with the propose tools.
    _, kwargs = fake_client.chat.call_args
    tool_names = [fn.__name__ for fn in kwargs["tools"]]
    assert "propose_reminder" in tool_names
    assert "propose_calendar_event" in tool_names


def test_handle_chat_create_intent_happy_event_allday(monkeypatch):
    """All-day event auto-detection end-to-end in the CLI flow."""
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda txt, now=None, **kw: datetime(2026, 4, 22, 0, 0, 0),
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)
    monkeypatch.setattr(
        rag, "_create_calendar_event",
        lambda *a, **kw: (True, "UID-123"),
    )
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "test-model")

    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response([
        _fake_tool_call("propose_calendar_event", {
            "title": "Grecia viene a casa", "start": "el miercoles",
        }),
    ])
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    handled, created = rag._handle_chat_create_intent("el miercoles viene Grecia, agendalo")
    assert handled is True
    # Events no exponen un undo — created_info debe ser None para que el
    # CLI no ofrezca `/undo` (matches web UX).
    assert created is None


# ── LLM chose not to call tool ──────────────────────────────────────────────


def test_handle_chat_create_intent_no_toolcall_returns_false(monkeypatch):
    """Si el LLM no emite tool_calls → return False para que el caller
    caiga al path de chat normal."""
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "test-model")
    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response([])  # no tools
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    handled, created = rag._handle_chat_create_intent("recordame algo impreciso")
    assert handled is False
    assert created is None


# ── needs_clarification path ────────────────────────────────────────────────


def test_handle_chat_create_intent_needs_clarification(monkeypatch):
    """propose_reminder detectó fecha ambigua → chip de warning, but
    returns True porque el handler resolvió el turn (el caller no debe
    fallthrough)."""
    monkeypatch.setattr(
        rag, "_parse_natural_datetime", lambda *a, **kw: None,
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "test-model")

    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response([
        _fake_tool_call("propose_reminder", {"title": "x", "when": "un día"}),
    ])
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    handled, created = rag._handle_chat_create_intent("recordame x un día de estos")
    assert handled is True
    # needs_clarification → no create → no undo offered.
    assert created is None


# ── create failed (Apple disabled, etc) ─────────────────────────────────────


def test_handle_chat_create_intent_create_failed(monkeypatch):
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda *a, **kw: datetime(2026, 4, 22, 0, 0),
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)
    monkeypatch.setattr(
        rag, "_create_calendar_event",
        lambda *a, **kw: (False, "Apple integration deshabilitada"),
    )
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "test-model")

    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response([
        _fake_tool_call("propose_calendar_event", {
            "title": "x", "start": "miercoles",
        }),
    ])
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    # Error is caught + reported but handler resolved this turn →
    # handled=True (don't fallthrough to query-RAG), no undo affordance.
    handled, created = rag._handle_chat_create_intent("creá evento miercoles")
    assert handled is True
    assert created is None


# ── unknown tool name ──────────────────────────────────────────────────────


def test_handle_chat_create_intent_unknown_tool(monkeypatch):
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "test-model")
    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response([
        _fake_tool_call("some_random_tool", {}),
    ])
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    handled, created = rag._handle_chat_create_intent("recordame x")
    assert handled is False
    assert created is None


# ── ollama.chat threw ──────────────────────────────────────────────────────


def test_handle_chat_create_intent_ollama_exception(monkeypatch):
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "test-model")
    fake_client = MagicMock()
    fake_client.chat.side_effect = RuntimeError("ollama caído")
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    handled, created = rag._handle_chat_create_intent("recordame x mañana 10am")
    assert handled is False
    assert created is None


# ── command-r arg wrapping ─────────────────────────────────────────────────


def test_handle_chat_create_intent_commandr_args_unwrap(monkeypatch):
    """command-r envuelve los args como {tool_name, parameters: {...}}.
    El handler desenvuelve y llama el tool con los args correctos."""
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda txt, now=None, **kw: datetime(2026, 4, 22, 10, 0),
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)
    create_mock = MagicMock(return_value=(True, "ID"))
    monkeypatch.setattr(rag, "_create_reminder", create_mock)
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "test-model")

    # command-r style: {"parameters": {"title": ..., "when": ...}}
    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response([
        _fake_tool_call("propose_reminder", {
            "tool_name": "propose_reminder",
            "parameters": {"title": "llamar", "when": "mañana 10am"},
        }),
    ])
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    handled, created = rag._handle_chat_create_intent("recordame llamar mañana 10am")
    assert handled is True
    assert created is not None and created["reminder_id"] == "ID"
    # Verifica que el tool REAL fue llamado con los args desenvuelto.
    args, _ = create_mock.call_args
    assert args[0] == "llamar"
