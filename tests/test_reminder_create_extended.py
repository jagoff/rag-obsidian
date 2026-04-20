"""Tests para `_create_reminder` mejorado (fase 1.3).

La variante original solo soportaba `due_token="tomorrow"` hardcodeada a
09:00. La mejora agrega:
  - `due_dt: datetime` — fecha/hora arbitraria (gana sobre due_token).
  - `priority: 1|5|9 | None` — alta/media/baja/ninguna.
  - `notes: str | None` — cuerpo del recordatorio.
  - `recurrence: {freq, interval, byday?} | None` — best-effort vía JXA;
    si AppleScript no acepta el property, el reminder igual se crea
    (sin recurrence) y se retorna `(True, id)` — documentado.

Todos los tests mockean `_osascript`; capturan el script generado para
verificar la construcción determinista.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import rag


def _capture_osascript(monkeypatch):
    """Helper: patch `_osascript` con un MagicMock que captura el script
    pasado y devuelve un id fake por default. Retorna el mock."""
    m = MagicMock(return_value="x-apple-reminderkit://REMCDReminder/ABC-123")
    monkeypatch.setattr(rag, "_osascript", m)
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    return m


# ── Backward compat: due_token="tomorrow" sigue funcionando ─────────────────


def test_create_reminder_due_token_tomorrow_unchanged(monkeypatch):
    m = _capture_osascript(monkeypatch)
    ok, rid = rag._create_reminder("comprar pan", due_token="tomorrow")
    assert ok is True
    assert rid.startswith("x-apple-reminderkit://")
    script = m.call_args[0][0]
    # Scripted tomorrow block must still appear.
    assert "(current date) + (1 * days)" in script
    assert "comprar pan" in script


# ── Nuevo: due_dt con datetime arbitrario ───────────────────────────────────


def test_create_reminder_with_due_dt(monkeypatch):
    m = _capture_osascript(monkeypatch)
    dt = datetime(2026, 4, 25, 14, 30, 0)
    ok, rid = rag._create_reminder("reunión con Juan", due_dt=dt)
    assert ok is True
    script = m.call_args[0][0]
    # AppleScript date construction: set year/month/day/hours/minutes.
    # Day set to 1 first to avoid month-overflow overruns.
    assert "set day of _d to 1" in script
    assert "set year of _d to 2026" in script
    assert "set month of _d to 4" in script
    assert "set day of _d to 25" in script
    assert "set hours of _d to 14" in script
    assert "set minutes of _d to 30" in script
    assert "reunión con Juan" in script


def test_due_dt_wins_over_due_token(monkeypatch):
    """Si pasan ambos, due_dt debe ganar (más preciso)."""
    m = _capture_osascript(monkeypatch)
    dt = datetime(2026, 4, 25, 14, 30, 0)
    rag._create_reminder("x", due_token="tomorrow", due_dt=dt)
    script = m.call_args[0][0]
    assert "set year of _d to 2026" in script
    # due_token="tomorrow" block should NOT appear when due_dt is given.
    assert "(current date) + (1 * days)" not in script


# ── Nuevo: priority ─────────────────────────────────────────────────────────


def test_create_reminder_priority_high(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rag._create_reminder("urgent", priority=1)
    script = m.call_args[0][0]
    assert "priority of _r" in script and "1" in script


def test_create_reminder_priority_medium(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rag._create_reminder("soon", priority=5)
    script = m.call_args[0][0]
    assert "priority of _r" in script
    assert "set priority of _r to 5" in script


def test_create_reminder_priority_omitted_when_none(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rag._create_reminder("normal")
    script = m.call_args[0][0]
    assert "priority of _r" not in script


# ── Nuevo: notes (body) ─────────────────────────────────────────────────────


def test_create_reminder_with_notes(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rag._create_reminder("llamar a Juan", notes="Sobre el proyecto X")
    script = m.call_args[0][0]
    assert "body" in script
    assert "Sobre el proyecto X" in script


def test_create_reminder_notes_escaped(monkeypatch):
    """Dobles comillas en notes no deben romper el AppleScript."""
    m = _capture_osascript(monkeypatch)
    rag._create_reminder("x", notes='tiene "comillas" acá')
    script = m.call_args[0][0]
    # Las comillas del input deben escaparse.
    assert r'\"comillas\"' in script


# ── Existing: list_name ─────────────────────────────────────────────────────


def test_create_reminder_with_list_name(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rag._create_reminder("x", list_name="Trabajo")
    script = m.call_args[0][0]
    assert 'of list "Trabajo"' in script


def test_create_reminder_list_name_escaped(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rag._create_reminder("x", list_name='"weird" list')
    script = m.call_args[0][0]
    assert r'\"weird\" list' in script


# ── Nuevo: recurrence (best-effort) ─────────────────────────────────────────


def test_create_reminder_recurrence_included_best_effort(monkeypatch):
    """El script debe TRY setear recurrence dentro de su propio try/on error
    (AppleScript en Reminders.app no siempre acepta recurrence — igual
    queremos que el reminder se cree aunque recurrence falle).
    """
    m = _capture_osascript(monkeypatch)
    rec = {"freq": "WEEKLY", "interval": 1, "byday": ["MO"]}
    ok, rid = rag._create_reminder("pagar", recurrence=rec)
    assert ok is True
    script = m.call_args[0][0]
    # Debe intentar construir la RRULE string y setearla.
    assert "FREQ=WEEKLY" in script
    assert "BYDAY=MO" in script
    # Debe estar dentro de su propio try block para no tumbar la creación.
    # Buscamos patrón: try ... recurrence ... on error ... end try (anidado).
    assert script.count("try") >= 2  # outer + recurrence inner


# ── Errores ─────────────────────────────────────────────────────────────────


def test_create_reminder_empty_name_fails():
    ok, msg = rag._create_reminder("")
    assert ok is False
    assert "vacío" in msg


def test_create_reminder_apple_disabled(monkeypatch):
    monkeypatch.setattr(rag, "_apple_enabled", lambda: False)
    ok, msg = rag._create_reminder("x")
    assert ok is False
    assert "deshabilitada" in msg


def test_create_reminder_osascript_error_surfaces(monkeypatch):
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: "err: list not found")
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    ok, msg = rag._create_reminder("x", list_name="Nonexistent")
    assert ok is False
    assert "list not found" in msg


def test_create_reminder_empty_osascript_output(monkeypatch):
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: "")
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    ok, msg = rag._create_reminder("x")
    assert ok is False


def test_create_reminder_name_quote_escape(monkeypatch):
    m = _capture_osascript(monkeypatch)
    rag._create_reminder('tiene "comillas"')
    script = m.call_args[0][0]
    # Comillas escapadas para que el AppleScript string literal no se rompa.
    assert r'\"comillas\"' in script
