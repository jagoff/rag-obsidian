"""Tests for `rag_anticipate.quiet_hours` — gate global del Anticipatory Agent.

Cubre:
- `is_night`: dentro/fuera de ventana, bordes inclusive/exclusive,
  wrap-around midnight, custom ventana via env var.
- `is_in_meeting`: sin events, evento en curso, evento futuro, evento
  pasado, eventos malformados.
- `is_focus_state`: sin state, state focus-like, state non-focus.
- `is_quiet_now`: combinación de señales, bypass via env var, reason
  string contractual.

Mockea `rag._fetch_calendar_today` y `rag._read_state` con
`monkeypatch.setattr` — la implementación lee ambos a través del módulo
`rag`, así que el patch en el módulo es suficiente.
"""

from __future__ import annotations

from datetime import datetime

import pytest

import rag
from rag_anticipate.quiet_hours import (
    is_focus_state,
    is_in_meeting,
    is_night,
    is_quiet_now,
)


# ── Fixture: bypass_off ──────────────────────────────────────────────────────
# Asegura que la env var de bypass no esté activa en tests que prueban
# que el gate efectivamente cierra. La fixture `monkeypatch` ya scopea
# por test, no hay leak entre runs.

@pytest.fixture(autouse=True)
def _no_bypass(monkeypatch):
    monkeypatch.delenv("RAG_ANTICIPATE_BYPASS_QUIET", raising=False)
    # Asegurar defaults de night window también — algún test previo
    # podría haber dejado override puesto.
    monkeypatch.delenv("RAG_ANTICIPATE_QUIET_NIGHT_START", raising=False)
    monkeypatch.delenv("RAG_ANTICIPATE_QUIET_NIGHT_END", raising=False)


# ── is_night ─────────────────────────────────────────────────────────────────

def test_is_night_true_at_23h():
    """23:00 cae dentro de la ventana nocturna default (22:00 → 08:00)."""
    assert is_night(datetime(2026, 4, 25, 23, 0)) is True


def test_is_night_true_at_3am():
    """03:00 (madrugada) está dentro del wrap-around — el caso clásico
    que validamos no se rompa por la lógica de cruzar medianoche."""
    assert is_night(datetime(2026, 4, 25, 3, 0)) is True


def test_is_night_false_at_10am():
    """10:00 mediodía — claramente fuera. Sanity check obvio."""
    assert is_night(datetime(2026, 4, 25, 10, 0)) is False


def test_is_night_false_at_20h():
    """20:00 — todavía no entra en quiet hours (default arranca 22:00)."""
    assert is_night(datetime(2026, 4, 25, 20, 0)) is False


def test_is_night_true_at_22h_exact_inclusive():
    """Borde inclusive: 22:00 exacto YA es noche. Si esto rompe,
    seguro alguien cambió `<=` por `<` en el start."""
    assert is_night(datetime(2026, 4, 25, 22, 0)) is True


def test_is_night_false_at_8h_exact_exclusive():
    """Borde exclusive: 08:00 exacto YA NO es noche — el agent puede
    pushear el daily brief de las 08:00."""
    assert is_night(datetime(2026, 4, 25, 8, 0)) is False


def test_is_night_custom_window_via_env(monkeypatch):
    """Override via env var: ventana 13:00 → 14:00 (intra-día, no wrap).
    Verificamos que el branch `start < end` también funciona."""
    monkeypatch.setenv("RAG_ANTICIPATE_QUIET_NIGHT_START", "13:00")
    monkeypatch.setenv("RAG_ANTICIPATE_QUIET_NIGHT_END", "14:00")
    assert is_night(datetime(2026, 4, 25, 13, 30)) is True
    assert is_night(datetime(2026, 4, 25, 12, 59)) is False
    assert is_night(datetime(2026, 4, 25, 14, 0)) is False  # exclusive


def test_is_night_invalid_env_falls_back_to_default(monkeypatch):
    """Env var malformada → default. No queremos que un typo en .zshrc
    rompa el agent — degradación silenciosa al default 22→08."""
    monkeypatch.setenv("RAG_ANTICIPATE_QUIET_NIGHT_START", "not-a-time")
    monkeypatch.setenv("RAG_ANTICIPATE_QUIET_NIGHT_END", "also-bad")
    assert is_night(datetime(2026, 4, 25, 23, 0)) is True   # default 22:00
    assert is_night(datetime(2026, 4, 25, 10, 0)) is False  # default 08:00


# ── is_in_meeting ────────────────────────────────────────────────────────────

def test_is_in_meeting_false_without_events(monkeypatch):
    """Sin events en el calendario — el caso default (icalBuddy vacío
    o sin eventos hoy)."""
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: [])
    assert is_in_meeting(datetime(2026, 4, 25, 14, 0)) is False


def test_is_in_meeting_true_with_event_in_progress(monkeypatch):
    """Evento `start ≤ now < end` → True. Caso clásico: meeting
    14:00-15:00, ahora son las 14:30."""
    events = [
        {"title": "Standup", "start": "14:00", "end": "15:00"},
    ]
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: events)
    assert is_in_meeting(datetime(2026, 4, 25, 14, 30)) is True


def test_is_in_meeting_false_with_future_event(monkeypatch):
    """Evento que arranca en el futuro — todavía no estamos en él.
    El gate solo cierra cuando estás EN reunión, no cuando se aproxima."""
    events = [
        {"title": "1:1 con manager", "start": "16:00", "end": "16:30"},
    ]
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: events)
    assert is_in_meeting(datetime(2026, 4, 25, 14, 0)) is False


def test_is_in_meeting_false_with_past_event(monkeypatch):
    """Evento ya terminado — tampoco cierra el gate."""
    events = [
        {"title": "Morning sync", "start": "09:00", "end": "09:30"},
    ]
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: events)
    assert is_in_meeting(datetime(2026, 4, 25, 14, 0)) is False


def test_is_in_meeting_false_when_fetch_raises(monkeypatch):
    """Si `_fetch_calendar_today` explota (icalBuddy timeout, OS error)
    el gate degrada a False — preferimos un push extra que tumbar el
    agent."""
    def _boom(max_events=30):
        raise RuntimeError("icalBuddy timeout")
    monkeypatch.setattr(rag, "_fetch_calendar_today", _boom)
    assert is_in_meeting(datetime(2026, 4, 25, 14, 0)) is False


def test_is_in_meeting_skips_malformed_events(monkeypatch):
    """Eventos sin `start` o `end` parseable se skipean — pero un
    evento válido en el mismo batch sí dispara True."""
    events = [
        {"title": "Sin tiempos"},                              # malformed
        {"title": "Inválido", "start": "garbage", "end": "x"},  # malformed
        {"title": "All-day", "start": "", "end": ""},           # malformed
        {"title": "Real meeting", "start": "14:00", "end": "15:00"},  # OK
    ]
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: events)
    assert is_in_meeting(datetime(2026, 4, 25, 14, 30)) is True


def test_is_in_meeting_handles_ampm_format(monkeypatch):
    """`H:MM AM/PM` también parsea — algunos calendarios devuelven 12h
    en lugar de 24h."""
    events = [
        {"title": "Lunch", "start": "1:00 PM", "end": "2:00 PM"},
    ]
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: events)
    assert is_in_meeting(datetime(2026, 4, 25, 13, 30)) is True


# ── is_focus_state ───────────────────────────────────────────────────────────

def test_is_focus_state_false_without_state(monkeypatch):
    """Sin `_read_state` (atributo ausente) → False. La impl usa
    `hasattr` para no asumir el getter específico."""
    # Limpiar por si algún test previo lo agregó
    if hasattr(rag, "_read_state"):
        monkeypatch.delattr(rag, "_read_state")
    assert is_focus_state() is False


def test_is_focus_state_false_when_state_returns_none(monkeypatch):
    """`_read_state()` retorna None (TTL expiró, archivo no existe) →
    False. Branch `isinstance(s, dict)` cierra."""
    monkeypatch.setattr(rag, "_read_state", lambda: None, raising=False)
    assert is_focus_state() is False


def test_is_focus_state_true_with_focus_code(monkeypatch):
    """state text = 'focus-code' → matchea 'focus' substring → True."""
    monkeypatch.setattr(
        rag, "_read_state",
        lambda: {"text": "focus-code"},
        raising=False,
    )
    assert is_focus_state() is True


def test_is_focus_state_true_with_deep_work(monkeypatch):
    """`deep-work` también es focus-like."""
    monkeypatch.setattr(
        rag, "_read_state",
        lambda: {"text": "deep-work session"},
        raising=False,
    )
    assert is_focus_state() is True


def test_is_focus_state_true_with_spanish_concentrado(monkeypatch):
    """Keyword en español — el user puede escribir el state en ES."""
    monkeypatch.setattr(
        rag, "_read_state",
        lambda: {"text": "concentrado escribiendo el plan"},
        raising=False,
    )
    assert is_focus_state() is True


def test_is_focus_state_false_with_non_focus_text(monkeypatch):
    """state text que no contiene ninguna keyword → False. Ej.
    'cansado', 'inspirado' — son válidos pero NO son quiet hours."""
    monkeypatch.setattr(
        rag, "_read_state",
        lambda: {"text": "cansado"},
        raising=False,
    )
    assert is_focus_state() is False


def test_is_focus_state_silent_fail_on_exception(monkeypatch):
    """Si `_read_state()` explota → False (silent-fail). El gate prefiere
    pushear de más antes que tumbar el agent."""
    def _boom():
        raise RuntimeError("state file corrupt")
    monkeypatch.setattr(rag, "_read_state", _boom, raising=False)
    assert is_focus_state() is False


# ── is_quiet_now ─────────────────────────────────────────────────────────────

def test_is_quiet_now_combines_signals(monkeypatch):
    """Smoke test del orquestador: con night activo (23:00) y todo lo
    demás silenciado, retorna (True, 'night hours')."""
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: [])
    if hasattr(rag, "_read_state"):
        monkeypatch.delattr(rag, "_read_state")
    quiet, reason = is_quiet_now(datetime(2026, 4, 25, 23, 0))
    assert quiet is True
    assert reason == "night hours"

    # Daytime + sin meeting + sin focus → not quiet
    quiet, reason = is_quiet_now(datetime(2026, 4, 25, 14, 0))
    assert quiet is False
    assert reason == ""


def test_is_quiet_now_focus_state_takes_precedence_over_meeting(monkeypatch):
    """Cuando focus + meeting están ambos activos, el orden contractual
    es: night → focus → meeting. Verificamos que focus gana sobre
    meeting (porque el código chequea focus antes)."""
    events = [
        {"title": "Meeting", "start": "14:00", "end": "15:00"},
    ]
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: events)
    monkeypatch.setattr(
        rag, "_read_state",
        lambda: {"text": "focus-code"},
        raising=False,
    )
    quiet, reason = is_quiet_now(datetime(2026, 4, 25, 14, 30))
    assert quiet is True
    assert reason == "user in focus state"


def test_is_quiet_now_meeting_only(monkeypatch):
    """Meeting activo, sin night ni focus → reason='in meeting'."""
    events = [
        {"title": "Demo review", "start": "14:00", "end": "15:00"},
    ]
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: events)
    if hasattr(rag, "_read_state"):
        monkeypatch.delattr(rag, "_read_state")
    quiet, reason = is_quiet_now(datetime(2026, 4, 25, 14, 30))
    assert quiet is True
    assert reason == "in meeting"


def test_is_quiet_now_bypass_env_overrides_everything(monkeypatch):
    """`RAG_ANTICIPATE_BYPASS_QUIET=1` fuerza (False, '') incluso con
    night + meeting + focus todos activos. Es el escape hatch para
    debug del agent en horario nocturno."""
    monkeypatch.setenv("RAG_ANTICIPATE_BYPASS_QUIET", "1")
    events = [{"title": "Meeting", "start": "23:00", "end": "23:59"}]
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: events)
    monkeypatch.setattr(
        rag, "_read_state",
        lambda: {"text": "focus-code"},
        raising=False,
    )
    quiet, reason = is_quiet_now(datetime(2026, 4, 25, 23, 30))
    assert quiet is False
    assert reason == ""


def test_is_quiet_now_bypass_accepts_true_string(monkeypatch):
    """`RAG_ANTICIPATE_BYPASS_QUIET=true` también funciona — variante
    case-sensitive documentada en el docstring."""
    monkeypatch.setenv("RAG_ANTICIPATE_BYPASS_QUIET", "true")
    quiet, reason = is_quiet_now(datetime(2026, 4, 25, 23, 30))
    assert quiet is False
    assert reason == ""


def test_is_quiet_now_reason_strings_are_stable(monkeypatch):
    """Los reason strings son contrato — el daily brief / dashboard los
    formatea como `quiet=True (reason: ...)`. Si cambian, romper este
    test obliga al dev a actualizar consumers downstream."""
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: [])
    if hasattr(rag, "_read_state"):
        monkeypatch.delattr(rag, "_read_state")

    # Night
    _, reason_night = is_quiet_now(datetime(2026, 4, 25, 23, 0))
    assert reason_night == "night hours"

    # Focus
    monkeypatch.setattr(
        rag, "_read_state",
        lambda: {"text": "deep-work"},
        raising=False,
    )
    _, reason_focus = is_quiet_now(datetime(2026, 4, 25, 14, 0))
    assert reason_focus == "user in focus state"

    # Meeting
    if hasattr(rag, "_read_state"):
        monkeypatch.delattr(rag, "_read_state")
    monkeypatch.setattr(
        rag, "_fetch_calendar_today",
        lambda max_events=30: [{"title": "x", "start": "14:00", "end": "15:00"}],
    )
    _, reason_meeting = is_quiet_now(datetime(2026, 4, 25, 14, 30))
    assert reason_meeting == "in meeting"

    # No quiet
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: [])
    _, reason_clear = is_quiet_now(datetime(2026, 4, 25, 14, 0))
    assert reason_clear == ""
