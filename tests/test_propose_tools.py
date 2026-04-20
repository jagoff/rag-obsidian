"""Tests para los tools de propuesta del web chat (Fase 2.1).

`propose_reminder` y `propose_calendar_event` NO crean nada — sólo parsean
el input NL a campos estructurados y devuelven JSON con un `proposal_id`
estable. El servidor web emite ese payload como evento SSE `proposal` y la
UI lo renderiza como tarjeta con botones ✓/✗.

Mockeamos `_parse_natural_datetime` / `_parse_natural_recurrence` cuando
necesitamos fechas deterministas (los tests pasan `now` a esos helpers —
acá, como llamamos al tool directo, monkeypatcheamos el resultado).
"""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from web import tools as web_tools


# ── propose_reminder ────────────────────────────────────────────────────────


def test_propose_reminder_basic_shape(monkeypatch):
    import rag
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda txt, now=None, **kw: datetime(2026, 4, 21, 10, 0, 0),
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)

    out = web_tools.propose_reminder(title="llamar a Juan", when="mañana a las 10")
    data = json.loads(out)
    assert data["kind"] == "reminder"
    assert data["proposal_id"].startswith("prop-")
    assert data["fields"]["title"] == "llamar a Juan"
    assert data["fields"]["due_iso"] == "2026-04-21T10:00:00"
    assert data["fields"]["due_text"] == "mañana a las 10"
    # Campos opcionales ausentes → None, no se omiten.
    assert data["fields"]["list"] is None
    assert data["fields"]["priority"] is None


def test_propose_reminder_with_all_fields(monkeypatch):
    import rag
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda txt, now=None, **kw: datetime(2026, 4, 25, 14, 0, 0),
    )
    monkeypatch.setattr(
        rag, "_parse_natural_recurrence",
        lambda txt: {"freq": "WEEKLY", "interval": 1, "byday": ["MO"]},
    )

    out = web_tools.propose_reminder(
        title="pagar alquiler",
        when="lunes a las 14",
        list="Trabajo",
        priority=1,
        notes="transferencia bancaria",
        recurrence_text="todos los lunes",
    )
    data = json.loads(out)
    assert data["fields"]["list"] == "Trabajo"
    assert data["fields"]["priority"] == 1
    assert data["fields"]["notes"] == "transferencia bancaria"
    assert data["fields"]["recurrence"] == {
        "freq": "WEEKLY", "interval": 1, "byday": ["MO"],
    }


def test_propose_reminder_unparseable_date_returns_error_shape(monkeypatch):
    """Si la fecha NL no parsea (dateparser + LLM fallback ambos None),
    el tool debe señalar el error de forma estructurada — no crashear y
    no inventar una fecha. La UI puede mostrar "clarificá la fecha".
    """
    import rag
    monkeypatch.setattr(rag, "_parse_natural_datetime", lambda *a, **kw: None)

    out = web_tools.propose_reminder(title="x", when="un día de estos")
    data = json.loads(out)
    assert data["kind"] == "reminder"
    assert data["fields"]["due_iso"] is None
    # Flag explícito para que el LLM sepa que hay que pedir aclaración.
    assert data.get("needs_clarification") is True


def test_propose_reminder_no_due(monkeypatch):
    """Reminder sin fecha (caso válido — sólo título)."""
    out = web_tools.propose_reminder(title="comprar pan", when="")
    data = json.loads(out)
    assert data["fields"]["title"] == "comprar pan"
    assert data["fields"]["due_iso"] is None
    # Sin fecha NO es error — es válido crear reminder undated.
    assert data.get("needs_clarification") is not True


def test_propose_reminder_proposal_id_stable_across_calls():
    """Cada call genera un proposal_id distinto (UUID)."""
    out1 = web_tools.propose_reminder(title="x", when="")
    out2 = web_tools.propose_reminder(title="x", when="")
    id1 = json.loads(out1)["proposal_id"]
    id2 = json.loads(out2)["proposal_id"]
    assert id1 != id2


# ── propose_calendar_event ──────────────────────────────────────────────────


def test_propose_event_basic_shape(monkeypatch):
    import rag
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda txt, now=None, **kw: datetime(2026, 4, 23, 16, 0, 0),
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)

    out = web_tools.propose_calendar_event(
        title="reunión con equipo", start="jueves 4pm",
    )
    data = json.loads(out)
    assert data["kind"] == "event"
    assert data["proposal_id"].startswith("prop-")
    assert data["fields"]["title"] == "reunión con equipo"
    assert data["fields"]["start_iso"] == "2026-04-23T16:00:00"
    # Sin end explícito → start + 1h.
    assert data["fields"]["end_iso"] == "2026-04-23T17:00:00"


def test_propose_event_explicit_end(monkeypatch):
    import rag
    calls = []

    def fake_parse(txt, now=None, **kw):
        calls.append(txt)
        if "4pm" in txt:
            return datetime(2026, 4, 23, 16, 0, 0)
        if "5:30" in txt:
            return datetime(2026, 4, 23, 17, 30, 0)
        return None

    monkeypatch.setattr(rag, "_parse_natural_datetime", fake_parse)
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)

    out = web_tools.propose_calendar_event(
        title="x", start="jueves 4pm", end="jueves 5:30pm",
    )
    data = json.loads(out)
    assert data["fields"]["end_iso"] == "2026-04-23T17:30:00"


def test_propose_event_all_day(monkeypatch):
    import rag
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda txt, now=None, **kw: datetime(2026, 7, 10, 0, 0, 0),
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)

    out = web_tools.propose_calendar_event(
        title="vacaciones", start="10 de julio", all_day=True,
    )
    data = json.loads(out)
    assert data["fields"]["all_day"] is True


def test_propose_event_unparseable_start(monkeypatch):
    import rag
    monkeypatch.setattr(rag, "_parse_natural_datetime", lambda *a, **kw: None)
    out = web_tools.propose_calendar_event(title="x", start="cualquier momento")
    data = json.loads(out)
    assert data["fields"]["start_iso"] is None
    assert data.get("needs_clarification") is True


def test_propose_event_with_location_and_notes(monkeypatch):
    import rag
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda *a, **kw: datetime(2026, 4, 25, 14, 0),
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)

    out = web_tools.propose_calendar_event(
        title="x", start="viernes",
        location="Palermo Soho", notes="llevar laptop",
    )
    data = json.loads(out)
    assert data["fields"]["location"] == "Palermo Soho"
    assert data["fields"]["notes"] == "llevar laptop"


def test_propose_event_with_recurrence(monkeypatch):
    import rag
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda *a, **kw: datetime(2026, 4, 27, 10, 0),
    )
    monkeypatch.setattr(
        rag, "_parse_natural_recurrence",
        lambda txt: {"freq": "WEEKLY", "interval": 1, "byday": ["MO"]},
    )
    out = web_tools.propose_calendar_event(
        title="standup", start="lunes 10am", recurrence_text="todos los lunes",
    )
    data = json.loads(out)
    assert data["fields"]["recurrence"]["freq"] == "WEEKLY"
    assert data["fields"]["recurrence"]["byday"] == ["MO"]


# ── registry ────────────────────────────────────────────────────────────────


def test_propose_tools_registered_in_chat_tools():
    """Ambas tools viven en CHAT_TOOLS + TOOL_FNS para que el loop de
    ollama las vea."""
    names = {fn.__name__ for fn in web_tools.CHAT_TOOLS}
    assert "propose_reminder" in names
    assert "propose_calendar_event" in names
    assert "propose_reminder" in web_tools.TOOL_FNS
    assert "propose_calendar_event" in web_tools.TOOL_FNS


def test_propose_tools_are_parallel_safe():
    """Sólo parsing puro, seguro para thread pool."""
    assert "propose_reminder" in web_tools.PARALLEL_SAFE
    assert "propose_calendar_event" in web_tools.PARALLEL_SAFE


def test_propose_tool_names_exported():
    """Set compartido con web/server.py para routing SSE."""
    assert "propose_reminder" in web_tools.PROPOSAL_TOOL_NAMES
    assert "propose_calendar_event" in web_tools.PROPOSAL_TOOL_NAMES
