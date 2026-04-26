"""Tests para los EventKit fast-paths de Calendar y Reminders
(audit 2026-04-25 R2-Calendar #2 + R2-Calendar #3).

`_create_calendar_event`, `_delete_calendar_event` y `_delete_reminder`
ahora intentan EventKit primero (lookup indexado, ~50-100ms) y caen al
AppleScript path (lento, ~5s+) solo si EventKit no carga o falla.

No podemos testear el path EventKit real sin un macOS con permisos +
Calendar/Reminders accessible. Testeamos:
1. El fallback AppleScript SÍ se invoca cuando EventKit falla
2. Los timeouts del fallback son los nuevos reducidos (8s create, 5s delete)
3. EventKit está documentado en docstring (anti-drift)
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import rag


@pytest.fixture(autouse=True)
def _force_applescript(monkeypatch):
    """Mismo patrón que test_delete_helpers — fuerza fallback."""
    try:
        import objc  # noqa: PLC0415

        def _force_failure(name):
            raise LookupError(f"forced fail in tests: {name}")

        monkeypatch.setattr(objc, "lookUpClass", _force_failure)
    except ImportError:
        pass


def test_create_calendar_event_falls_back_to_applescript(monkeypatch):
    """Si EventKit no carga, _create_calendar_event cae al AppleScript
    path con timeout 8s (no 20s)."""
    captured: dict = {}

    def _spy_osascript(script, *, timeout=None):
        captured["script"] = script
        captured["timeout"] = timeout
        return "ABC-123-UID"

    monkeypatch.setattr(rag, "_osascript", _spy_osascript)
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)

    start = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)
    ok, uid = rag._create_calendar_event("Test event", start)

    assert ok is True
    assert uid == "ABC-123-UID"
    # Timeout reducido del audit (20s→8s)
    assert captured["timeout"] == 8.0
    # AppleScript verdadero, no nuestro spy de EventKit
    assert 'tell application "Calendar"' in captured["script"]


def test_create_calendar_event_falls_back_with_recurrence(monkeypatch):
    """Eventos con recurrence siempre usan AppleScript (skip EventKit
    porque EKRecurrenceRule es complejo). Verifica que igual usa el
    timeout reducido."""
    captured: dict = {}

    def _spy_osascript(script, *, timeout=None):
        captured["script"] = script
        captured["timeout"] = timeout
        return "RECUR-UID"

    monkeypatch.setattr(rag, "_osascript", _spy_osascript)
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)

    ok, _ = rag._create_calendar_event(
        "Reunion semanal",
        datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc),
        recurrence={"freq": "WEEKLY", "interval": 1},
    )
    assert ok is True
    assert captured["timeout"] == 8.0
    # AppleScript debería incluir el RRULE
    assert "FREQ=WEEKLY" in captured["script"] or "recurrence" in captured["script"].lower()


def test_delete_reminder_uses_reduced_timeout(monkeypatch):
    """Fallback AppleScript de _delete_reminder ahora es 5s (no 15s)."""
    captured: dict = {}

    def _spy_osascript(script, *, timeout=None):
        captured["timeout"] = timeout
        return "ok"

    monkeypatch.setattr(rag, "_osascript", _spy_osascript)
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)

    rag._delete_reminder("x-apple-reminderkit://REMCDReminder/ABC")
    assert captured["timeout"] == 5.0


def test_eventkit_documented_in_docstrings():
    """Anti-drift: las 3 funciones deben mencionar EventKit en su
    docstring para que un dev futuro entienda por qué hay 2 paths."""
    assert "EventKit" in (rag._create_calendar_event.__doc__ or "")
    assert "EventKit" in (rag._delete_calendar_event.__doc__ or "")
    assert "EventKit" in (rag._delete_reminder.__doc__ or "")
