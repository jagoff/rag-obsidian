"""Tests para `_create_calendar_event` (fase 1.4).

Crea un evento en Calendar.app vía AppleScript. Signature:
  _create_calendar_event(title, start, end=None, *, calendar=None,
                          location=None, notes=None, all_day=False,
                          recurrence=None) → (ok, uid_or_error)

Contrato:
  - `start` requerido (datetime). `end` default = start + 1h.
  - `calendar=None` → primer calendario escribible.
  - `all_day=True` → ignora horas, usa date literal.
  - `recurrence={freq,interval,byday?}` → RRULE string (supported por
    Calendar.app, a diferencia de Reminders).

Todos los tests mockean `_osascript` y capturan el script generado.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import rag


def _capture_osascript(monkeypatch, return_val="ABC-DEF-UUID-123"):
    m = MagicMock(return_value=return_val)
    monkeypatch.setattr(rag, "_osascript", m)
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    return m


# ── Happy path ──────────────────────────────────────────────────────────────


def test_create_event_basic(monkeypatch):
    m = _capture_osascript(monkeypatch)
    start = datetime(2026, 4, 25, 14, 0, 0)
    end = datetime(2026, 4, 25, 15, 0, 0)
    ok, uid = rag._create_calendar_event("reunión con Juan", start, end)
    assert ok is True
    assert uid == "ABC-DEF-UUID-123"
    script = m.call_args[0][0]
    assert "reunión con Juan" in script
    assert "make new event" in script
    assert "summary" in script
    # Start date construction
    assert "set year of _s to 2026" in script
    assert "set month of _s to 4" in script
    assert "set day of _s to 25" in script
    assert "set hours of _s to 14" in script
    # End date construction
    assert "set year of _e to 2026" in script
    assert "set hours of _e to 15" in script


def test_create_event_default_end_is_start_plus_1h(monkeypatch):
    """Si no se pasa end, debe calcularse start + 1h."""
    m = _capture_osascript(monkeypatch)
    start = datetime(2026, 4, 25, 14, 30, 0)
    rag._create_calendar_event("x", start)
    script = m.call_args[0][0]
    # End hours should be 15, minutes 30.
    assert "set hours of _e to 15" in script
    assert "set minutes of _e to 30" in script


def test_create_event_explicit_calendar(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rag._create_calendar_event(
        "x", datetime(2026, 4, 25, 14, 0), calendar="Trabajo",
    )
    script = m.call_args[0][0]
    assert 'name is "Trabajo"' in script


def test_create_event_default_calendar_fallback(monkeypatch):
    """Sin calendar explícito → primer calendario escribible."""
    m = _capture_osascript(monkeypatch)
    rag._create_calendar_event("x", datetime(2026, 4, 25, 14, 0))
    script = m.call_args[0][0]
    # Debe contener alguna forma de fallback a default (first writable calendar).
    assert "writable" in script or "first calendar" in script


# ── Opcionales ──────────────────────────────────────────────────────────────


def test_create_event_with_location(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rag._create_calendar_event(
        "x", datetime(2026, 4, 25, 14, 0), location="Palermo Soho",
    )
    script = m.call_args[0][0]
    assert "location" in script
    assert "Palermo Soho" in script


def test_create_event_with_notes(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rag._create_calendar_event(
        "x", datetime(2026, 4, 25, 14, 0), notes="Preparar slides antes",
    )
    script = m.call_args[0][0]
    assert "description" in script
    assert "Preparar slides antes" in script


def test_create_event_all_day(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rag._create_calendar_event(
        "vacaciones", datetime(2026, 7, 10, 0, 0), all_day=True,
    )
    script = m.call_args[0][0]
    assert "allday event" in script


def test_create_event_with_recurrence(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rec = {"freq": "WEEKLY", "interval": 1, "byday": ["MO", "WE"]}
    rag._create_calendar_event(
        "standup", datetime(2026, 4, 27, 10, 0), recurrence=rec,
    )
    script = m.call_args[0][0]
    assert "FREQ=WEEKLY" in script
    assert "BYDAY=MO,WE" in script
    # Calendar.app sí soporta recurrence nativo (a diferencia de Reminders).
    assert "recurrence" in script


# ── Escaping ────────────────────────────────────────────────────────────────


def test_create_event_quotes_in_title_escaped(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rag._create_calendar_event(
        'reunión "urgente"', datetime(2026, 4, 25, 14, 0),
    )
    script = m.call_args[0][0]
    assert r'\"urgente\"' in script


def test_create_event_quotes_in_location_escaped(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rag._create_calendar_event(
        "x", datetime(2026, 4, 25, 14, 0), location='"Bar" en esquina',
    )
    script = m.call_args[0][0]
    assert r'\"Bar\"' in script


# ── Errores ─────────────────────────────────────────────────────────────────


def test_create_event_empty_title():
    ok, msg = rag._create_calendar_event("", datetime(2026, 4, 25, 14, 0))
    assert ok is False
    assert "título" in msg or "titulo" in msg or "vacío" in msg


def test_create_event_apple_disabled(monkeypatch):
    monkeypatch.setattr(rag, "_apple_enabled", lambda: False)
    ok, msg = rag._create_calendar_event("x", datetime(2026, 4, 25, 14, 0))
    assert ok is False
    assert "deshabilitada" in msg


def test_create_event_end_before_start_fails():
    start = datetime(2026, 4, 25, 15, 0)
    end = datetime(2026, 4, 25, 14, 0)
    ok, msg = rag._create_calendar_event("x", start, end)
    assert ok is False


def test_create_event_osascript_error_surfaces(monkeypatch):
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: "err: calendar not found")
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    ok, msg = rag._create_calendar_event(
        "x", datetime(2026, 4, 25, 14, 0), calendar="Nonexistent",
    )
    assert ok is False
    assert "calendar not found" in msg


def test_create_event_empty_osascript_output(monkeypatch):
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: "")
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    ok, msg = rag._create_calendar_event("x", datetime(2026, 4, 25, 14, 0))
    assert ok is False
