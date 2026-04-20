"""Tests para _delete_reminder + _delete_calendar_event.

Alimentan el flujo auto-create + undo: tras crear un reminder/event la UI
muestra un toast con "Deshacer" que hace DELETE al endpoint
correspondiente, el cual llama a estos helpers.

Mockeamos `_osascript` + `_apple_enabled`.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import rag


def _capture_osascript(monkeypatch, return_val="ok"):
    m = MagicMock(return_value=return_val)
    monkeypatch.setattr(rag, "_osascript", m)
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    return m


# ── _delete_reminder ────────────────────────────────────────────────────────


def test_delete_reminder_ok(monkeypatch):
    m = _capture_osascript(monkeypatch, return_val="ok")
    ok, msg = rag._delete_reminder("x-apple-reminderkit://REMCDReminder/ABC-123")
    assert ok is True
    assert "borrada" in msg or "eliminada" in msg or "deshecha" in msg
    script = m.call_args[0][0]
    assert 'tell application "Reminders"' in script
    assert "delete" in script
    assert "ABC-123" in script


def test_delete_reminder_escapes_id_quotes(monkeypatch):
    m = _capture_osascript(monkeypatch, return_val="ok")
    rag._delete_reminder('id-with-"quotes"')
    script = m.call_args[0][0]
    assert r'\"quotes\"' in script


def test_delete_reminder_empty_id():
    ok, msg = rag._delete_reminder("")
    assert ok is False


def test_delete_reminder_apple_disabled(monkeypatch):
    monkeypatch.setattr(rag, "_apple_enabled", lambda: False)
    ok, msg = rag._delete_reminder("x")
    assert ok is False
    assert "deshabilitada" in msg


def test_delete_reminder_osascript_error(monkeypatch):
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: "err: not found")
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    ok, msg = rag._delete_reminder("x")
    assert ok is False
    assert "not found" in msg


def test_delete_reminder_empty_osascript_output(monkeypatch):
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: "")
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    ok, msg = rag._delete_reminder("x")
    assert ok is False


# ── _delete_calendar_event ──────────────────────────────────────────────────


def test_delete_event_ok(monkeypatch):
    m = _capture_osascript(monkeypatch, return_val="ok")
    ok, msg = rag._delete_calendar_event("UID-ABC-123")
    assert ok is True
    script = m.call_args[0][0]
    assert 'tell application "Calendar"' in script
    assert "delete" in script
    assert "UID-ABC-123" in script
    # Iteration over writable calendars (not `whose uid is ...`) because
    # Calendar.app's `whose uid` query is unreliable — verified 2026-04-20.
    assert "writable is true" in script
    assert "repeat with" in script


def test_delete_event_escapes_uid_quotes(monkeypatch):
    m = _capture_osascript(monkeypatch, return_val="ok")
    rag._delete_calendar_event('uid-with-"quotes"')
    script = m.call_args[0][0]
    assert r'\"quotes\"' in script


def test_delete_event_empty_uid():
    ok, msg = rag._delete_calendar_event("")
    assert ok is False


def test_delete_event_apple_disabled(monkeypatch):
    monkeypatch.setattr(rag, "_apple_enabled", lambda: False)
    ok, msg = rag._delete_calendar_event("x")
    assert ok is False
    assert "deshabilitada" in msg


def test_delete_event_osascript_error(monkeypatch):
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: "err: event not found")
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    ok, msg = rag._delete_calendar_event("x")
    assert ok is False
    assert "event not found" in msg
