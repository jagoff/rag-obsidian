"""Tests para `_parse_natural_datetime` y `_parse_natural_recurrence`.

Ambas helpers alimentan los tools `propose_reminder` / `propose_calendar_event`:
el LLM ya entrega `title` + `when` + opcional `recurrence_text`; nuestra tarea
es convertir los strings a `datetime` / RRULE dict deterministas.

Estrategia:
- dateparser (es+en, prefer_dates_from=future) resuelve el 80% de los casos;
  esos tests corren sin LLM.
- Para fechas que dateparser no agarra (ej. "el jueves que viene a las 3"),
  cae a helper LLM (mockeado).
- Recurrencia: regex hand-rolled sobre patrones comunes ES/EN; no LLM.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import rag


# ── _parse_natural_datetime: dateparser path ─────────────────────────────────


def test_parse_manana_10am():
    now = datetime(2026, 4, 20, 15, 0, 0)
    dt = rag._parse_natural_datetime("mañana a las 10am", now=now)
    assert dt is not None
    assert dt.date().isoformat() == "2026-04-21"
    assert dt.hour == 10
    assert dt.minute == 0


def test_parse_en_2_horas():
    now = datetime(2026, 4, 20, 15, 0, 0)
    dt = rag._parse_natural_datetime("en 2 horas", now=now)
    assert dt is not None
    # Tolerancia de ±1 minuto — dateparser computa desde "now" real.
    delta = abs((dt - datetime(2026, 4, 20, 17, 0, 0)).total_seconds())
    assert delta < 120


def test_parse_iso_string():
    now = datetime(2026, 4, 20, 15, 0, 0)
    dt = rag._parse_natural_datetime("2026-04-25 14:30", now=now)
    assert dt is not None
    assert dt.isoformat(timespec="minutes") == "2026-04-25T14:30"


def test_parse_english_monday_3pm():
    """'next monday 3pm' confunde a dateparser; 'monday 3pm' con prefer_future
    rinde el próximo lunes. El caso 'next X' cae al LLM fallback (ver más abajo).
    """
    now = datetime(2026, 4, 20, 15, 0, 0)  # Monday
    dt = rag._parse_natural_datetime("monday 3pm", now=now)
    assert dt is not None
    # Next monday = 27, 15:00
    assert dt.weekday() == 0
    assert dt.hour == 15


def test_parse_pasado_manana_falls_back_to_llm(monkeypatch):
    """'pasado mañana' dateparser no lo entiende en español — LLM fallback."""
    fake_client = MagicMock()
    fake_client.chat.return_value = MagicMock(
        message=MagicMock(content='{"iso": "2026-04-22T09:30"}')
    )
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    now = datetime(2026, 4, 20, 15, 0, 0)
    dt = rag._parse_natural_datetime("pasado mañana 9:30", now=now)
    assert dt == datetime(2026, 4, 22, 9, 30, 0)


def test_parse_prefers_future_for_bare_weekday():
    """'el jueves' sin referencia semanal → PRÓXIMO jueves, no el pasado."""
    # Lunes 20-abr-2026 → jueves futuro es 23-abr.
    now = datetime(2026, 4, 20, 15, 0, 0)
    dt = rag._parse_natural_datetime("jueves 14:00", now=now)
    assert dt is not None
    assert dt >= now


def test_parse_empty_returns_none():
    assert rag._parse_natural_datetime("", now=datetime(2026, 4, 20)) is None
    assert rag._parse_natural_datetime("   ", now=datetime(2026, 4, 20)) is None


def test_parse_garbage_returns_none(monkeypatch):
    """Texto no-parseable + LLM fallback que también devuelve null → None."""
    # Forzar que el fallback LLM también devuelva null.
    fake_client = MagicMock()
    fake_client.chat.return_value = MagicMock(
        message=MagicMock(content='{"iso": null}')
    )
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    dt = rag._parse_natural_datetime("asdf xyz qwerty", now=datetime(2026, 4, 20))
    assert dt is None


# ── _parse_natural_datetime: LLM fallback ───────────────────────────────────


def test_parse_llm_fallback_invoked_when_dateparser_fails(monkeypatch):
    now = datetime(2026, 4, 20, 15, 0, 0)
    fake_client = MagicMock()
    fake_client.chat.return_value = MagicMock(
        message=MagicMock(content='{"iso": "2026-04-30T11:00"}')
    )
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    # Input que dateparser no resuelve bien (ni con es ni en).
    dt = rag._parse_natural_datetime("la próxima reunión del comité", now=now)
    assert dt == datetime(2026, 4, 30, 11, 0, 0)
    fake_client.chat.assert_called_once()


def test_parse_llm_fallback_malformed_json_returns_none(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.return_value = MagicMock(
        message=MagicMock(content="not json at all")
    )
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    dt = rag._parse_natural_datetime(
        "algo super raro que no es fecha", now=datetime(2026, 4, 20)
    )
    assert dt is None


def test_parse_llm_fallback_error_returns_none(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.side_effect = RuntimeError("ollama caído")
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    dt = rag._parse_natural_datetime(
        "algo raro que no parsea", now=datetime(2026, 4, 20)
    )
    assert dt is None


def test_parse_llm_fallback_skipped_when_dateparser_succeeds(monkeypatch):
    """Performance guard: dateparser primero, LLM no se toca si ya resolvió."""
    fake_client = MagicMock()
    monkeypatch.setattr(rag, "_helper_client", lambda: fake_client)
    dt = rag._parse_natural_datetime("mañana a las 10", now=datetime(2026, 4, 20))
    assert dt is not None
    fake_client.chat.assert_not_called()


# ── _parse_natural_recurrence ───────────────────────────────────────────────


def test_recurrence_todos_los_dias():
    r = rag._parse_natural_recurrence("todos los días")
    assert r == {"freq": "DAILY", "interval": 1}


def test_recurrence_diariamente():
    assert rag._parse_natural_recurrence("diariamente") == {"freq": "DAILY", "interval": 1}


def test_recurrence_todos_los_lunes():
    r = rag._parse_natural_recurrence("todos los lunes")
    assert r == {"freq": "WEEKLY", "interval": 1, "byday": ["MO"]}


def test_recurrence_cada_martes():
    r = rag._parse_natural_recurrence("cada martes")
    assert r == {"freq": "WEEKLY", "interval": 1, "byday": ["TU"]}


def test_recurrence_semanalmente():
    assert rag._parse_natural_recurrence("semanalmente") == {"freq": "WEEKLY", "interval": 1}


def test_recurrence_cada_2_semanas():
    r = rag._parse_natural_recurrence("cada 2 semanas")
    assert r == {"freq": "WEEKLY", "interval": 2}


def test_recurrence_mensualmente():
    assert rag._parse_natural_recurrence("mensualmente") == {"freq": "MONTHLY", "interval": 1}


def test_recurrence_todos_los_meses():
    assert rag._parse_natural_recurrence("todos los meses") == {"freq": "MONTHLY", "interval": 1}


def test_recurrence_anualmente():
    assert rag._parse_natural_recurrence("anualmente") == {"freq": "YEARLY", "interval": 1}


def test_recurrence_every_day_english():
    assert rag._parse_natural_recurrence("every day") == {"freq": "DAILY", "interval": 1}


def test_recurrence_weekly_english():
    assert rag._parse_natural_recurrence("weekly") == {"freq": "WEEKLY", "interval": 1}


def test_recurrence_every_monday_english():
    r = rag._parse_natural_recurrence("every monday")
    assert r == {"freq": "WEEKLY", "interval": 1, "byday": ["MO"]}


def test_recurrence_none_for_non_recurring():
    assert rag._parse_natural_recurrence("mañana a las 10") is None
    assert rag._parse_natural_recurrence("") is None
    assert rag._parse_natural_recurrence(None) is None
