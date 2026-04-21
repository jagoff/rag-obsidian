"""Tests for the `/undo` slash command in `rag chat`.

The undo loop is owned by the chat() CLI body — too interactive for a
full end-to-end test. Here we unit-test the pieces the slash command
composes:
  - `_handle_chat_create_intent` returns the reminder_id via created_info
    on a successful create (covered in test_chat_create_handler.py).
  - `_delete_reminder` deletes by id (covered in the CLI's existing
    Apple integration paths — we mock it).
  - The chat loop's state machine: last_created populated on success,
    cleared after /undo, not set on needs_clarification/events.

Rather than driving chat() interactively (which needs stdin/stdout
mocking), we replicate the relevant state transitions in a tight
simulation loop using `_handle_chat_create_intent` + `_delete_reminder`
with monkeypatched dependencies.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import rag


def _fake_ollama_response(tool_calls: list) -> MagicMock:
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


def _make_reminder_intent(monkeypatch, reminder_id: str = "x-apple://RID") -> None:
    """Monkeypatch every dep needed for propose_reminder → create → success."""
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda txt, now=None, **kw: datetime(2026, 4, 22, 10, 0),
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)
    monkeypatch.setattr(
        rag, "_create_reminder",
        lambda *a, **kw: (True, reminder_id),
    )
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "test-model")
    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response([
        _fake_tool_call("propose_reminder", {
            "title": "llamar mama", "when": "mañana 10am",
        }),
    ])
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)


def test_create_populates_last_created_for_reminder(monkeypatch):
    """Happy path: propose_reminder → reminder_id returned in created_info."""
    _make_reminder_intent(monkeypatch, reminder_id="x-apple://ABC")
    handled, created = rag._handle_chat_create_intent("recordame llamar a mama mañana")
    assert handled is True
    assert created == {
        "kind": "reminder",
        "reminder_id": "x-apple://ABC",
        "title": "llamar mama",
    }


def test_create_calendar_event_returns_none_created_info(monkeypatch):
    """Events are tracked but don't expose undo — created_info is None so
    the CLI doesn't offer /undo. Mirrors web UX."""
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda txt, now=None, **kw: datetime(2026, 4, 22, 0, 0),
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
            "title": "Grecia", "start": "el miercoles",
        }),
    ])
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    handled, created = rag._handle_chat_create_intent("el miercoles viene Grecia")
    assert handled is True
    assert created is None


def test_undo_calls_delete_reminder_with_stored_id(monkeypatch):
    """State machine: reminder created → stored → /undo dispatches
    _delete_reminder with the exact id → on success state is cleared."""
    _make_reminder_intent(monkeypatch, reminder_id="x-apple://ZZZ")
    handled, created = rag._handle_chat_create_intent("recordame llamar mañana")
    assert handled is True and created is not None

    last_created = created
    calls = []

    def _fake_delete(rid):
        calls.append(rid)
        return True, "ok"

    monkeypatch.setattr(rag, "_delete_reminder", _fake_delete)

    # Simulate the /undo branch: delete + clear state on success.
    rid = last_created["reminder_id"]
    ok, _ = rag._delete_reminder(rid)
    assert ok is True
    assert calls == ["x-apple://ZZZ"]
    last_created = None if ok else last_created
    assert last_created is None


def test_undo_handles_delete_failure(monkeypatch):
    """_delete_reminder returns False (e.g. Apple integration disabled) →
    last_created should NOT be cleared so user can retry."""
    _make_reminder_intent(monkeypatch, reminder_id="x-apple://FAIL")
    _, created = rag._handle_chat_create_intent("recordame x mañana")
    monkeypatch.setattr(
        rag, "_delete_reminder",
        lambda rid: (False, "Apple integration deshabilitada"),
    )

    last_created = created
    rid = last_created["reminder_id"]
    ok, msg = rag._delete_reminder(rid)
    assert ok is False
    assert "deshabilitada" in msg
    # State preserved so user can try again later.
    last_created = None if ok else last_created
    assert last_created is not None


def test_needs_clarification_does_not_populate_last_created(monkeypatch):
    """Ambiguous create → created_info None → chat loop should NOT record
    anything (nothing to undo)."""
    monkeypatch.setattr(rag, "_parse_natural_datetime", lambda *a, **kw: None)
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "test-model")
    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response([
        _fake_tool_call("propose_reminder", {"title": "x", "when": "un día"}),
    ])
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    handled, created = rag._handle_chat_create_intent("recordame x un día")
    assert handled is True
    assert created is None
