"""Tests para los tools de creación del web chat.

Post-auto-create: los tools YA NO son no-op. Si la fecha parsea clara →
crean via _create_reminder / _create_calendar_event y devuelven
`{created: true, reminder_id/event_uid, fields}`. Si la fecha es
ambigua → devuelven `{needs_clarification: true, proposal_id, fields}`
como antes. Si el create falla → `{created: false, error, fields}`.

El servidor inspecciona el payload y emite evento SSE `created` (toast
+ undo en UI) o `proposal` (card en UI) según corresponda.

Mockeamos:
  - `rag._parse_natural_datetime` / `rag._parse_natural_recurrence` para
    controlar el outcome del parsing
  - `rag._create_reminder` / `rag._create_calendar_event` para no tocar
    osascript real
"""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from web import tools as web_tools


# ── propose_reminder: AUTO-CREATE happy path ─────────────────────────────────


def test_propose_reminder_auto_creates_when_date_is_clear(monkeypatch):
    import rag
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda txt, now=None, **kw: datetime(2026, 4, 21, 10, 0, 0),
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)
    create_mock = MagicMock(return_value=(True, "x-apple-reminderkit://ABC-123"))
    monkeypatch.setattr(rag, "_create_reminder", create_mock)

    out = web_tools.propose_reminder(title="llamar a Juan", when="mañana a las 10")
    data = json.loads(out)
    assert data["kind"] == "reminder"
    assert data["created"] is True
    assert data["reminder_id"] == "x-apple-reminderkit://ABC-123"
    assert data["fields"]["title"] == "llamar a Juan"
    assert data["fields"]["due_iso"] == "2026-04-21T10:00:00"
    # No debe venir proposal_id cuando se creó.
    assert "proposal_id" not in data
    # _create_reminder fue llamado con los campos correctos.
    assert create_mock.called
    args, kwargs = create_mock.call_args
    assert args[0] == "llamar a Juan"
    assert kwargs["due_dt"] == datetime(2026, 4, 21, 10, 0, 0)


def test_propose_reminder_auto_creates_without_date(monkeypatch):
    """Un recordatorio sin fecha también es auto-create (undated reminder
    es un caso válido en Reminders.app)."""
    import rag
    create_mock = MagicMock(return_value=(True, "id-undated"))
    monkeypatch.setattr(rag, "_create_reminder", create_mock)
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)

    out = web_tools.propose_reminder(title="comprar pan", when="")
    data = json.loads(out)
    assert data["created"] is True
    assert data["fields"]["due_iso"] is None


def test_propose_reminder_passes_all_fields_to_create(monkeypatch):
    import rag
    create_mock = MagicMock(return_value=(True, "id"))
    monkeypatch.setattr(rag, "_create_reminder", create_mock)
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda *a, **kw: datetime(2026, 4, 25, 14, 0),
    )
    monkeypatch.setattr(
        rag, "_parse_natural_recurrence",
        lambda txt: {"freq": "WEEKLY", "interval": 1, "byday": ["MO"]},
    )

    web_tools.propose_reminder(
        title="pagar alquiler", when="lunes 14",
        list="Trabajo", priority=1, notes="transferencia",
        recurrence_text="todos los lunes",
    )
    _, kwargs = create_mock.call_args
    assert kwargs["list_name"] == "Trabajo"
    assert kwargs["priority"] == 1
    assert kwargs["notes"] == "transferencia"
    assert kwargs["recurrence"]["byday"] == ["MO"]


# ── propose_reminder: PROPOSAL fallback path ─────────────────────────────────


def test_propose_reminder_falls_back_to_proposal_when_ambiguous(monkeypatch):
    """Si el when fue provisto pero no parsea → proposal card."""
    import rag
    monkeypatch.setattr(rag, "_parse_natural_datetime", lambda *a, **kw: None)
    # _create_reminder debe NO ser llamado — fallamos al proposal antes.
    create_mock = MagicMock()
    monkeypatch.setattr(rag, "_create_reminder", create_mock)

    out = web_tools.propose_reminder(title="x", when="un día de estos")
    data = json.loads(out)
    assert data["kind"] == "reminder"
    assert data["needs_clarification"] is True
    assert data["proposal_id"].startswith("prop-")
    assert data.get("created") is not True
    assert not create_mock.called


# ── propose_reminder: ERROR path ─────────────────────────────────────────────


def test_propose_reminder_surface_create_error(monkeypatch):
    """Si _create_reminder falla (Apple disabled, permission denied) →
    created:false con el mensaje de error."""
    import rag
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda *a, **kw: datetime(2026, 4, 21, 10, 0),
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)
    monkeypatch.setattr(
        rag, "_create_reminder",
        lambda *a, **kw: (False, "Apple integration deshabilitada"),
    )

    out = web_tools.propose_reminder(title="x", when="mañana 10am")
    data = json.loads(out)
    assert data["created"] is False
    assert "deshabilitada" in data["error"]


# ── propose_calendar_event: AUTO-CREATE happy path ───────────────────────────


def test_propose_event_auto_creates_when_start_is_clear(monkeypatch):
    import rag
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda *a, **kw: datetime(2026, 4, 23, 16, 0, 0),
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)
    create_mock = MagicMock(return_value=(True, "UID-ABC-123"))
    monkeypatch.setattr(rag, "_create_calendar_event", create_mock)

    out = web_tools.propose_calendar_event(
        title="reunión con equipo", start="jueves 4pm",
    )
    data = json.loads(out)
    assert data["kind"] == "event"
    assert data["created"] is True
    assert data["event_uid"] == "UID-ABC-123"
    assert data["fields"]["start_iso"] == "2026-04-23T16:00:00"
    # Default end = start + 1h
    assert data["fields"]["end_iso"] == "2026-04-23T17:00:00"
    # _create_calendar_event debe recibir start_dt y end_dt correctos.
    args, kwargs = create_mock.call_args
    assert args[0] == "reunión con equipo"
    assert args[1] == datetime(2026, 4, 23, 16, 0, 0)
    assert args[2] == datetime(2026, 4, 23, 17, 0, 0)


def test_propose_event_explicit_end_passed_to_create(monkeypatch):
    import rag

    def fake_parse(txt, now=None, **kw):
        if "4pm" in txt:
            return datetime(2026, 4, 23, 16, 0, 0)
        if "5:30" in txt:
            return datetime(2026, 4, 23, 17, 30, 0)
        return None

    monkeypatch.setattr(rag, "_parse_natural_datetime", fake_parse)
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)
    create_mock = MagicMock(return_value=(True, "UID"))
    monkeypatch.setattr(rag, "_create_calendar_event", create_mock)

    web_tools.propose_calendar_event(
        title="x", start="jueves 4pm", end="jueves 5:30pm",
    )
    args, _ = create_mock.call_args
    assert args[2] == datetime(2026, 4, 23, 17, 30, 0)


# ── propose_calendar_event: PROPOSAL fallback ────────────────────────────────


def test_propose_event_falls_back_when_start_unparseable(monkeypatch):
    import rag
    monkeypatch.setattr(rag, "_parse_natural_datetime", lambda *a, **kw: None)
    create_mock = MagicMock()
    monkeypatch.setattr(rag, "_create_calendar_event", create_mock)

    out = web_tools.propose_calendar_event(title="x", start="cualquier momento")
    data = json.loads(out)
    assert data["needs_clarification"] is True
    assert data.get("created") is not True
    assert not create_mock.called


def test_propose_event_surfaces_create_error(monkeypatch):
    import rag
    monkeypatch.setattr(
        rag, "_parse_natural_datetime",
        lambda *a, **kw: datetime(2026, 4, 23, 16, 0),
    )
    monkeypatch.setattr(rag, "_parse_natural_recurrence", lambda txt: None)
    monkeypatch.setattr(
        rag, "_create_calendar_event",
        lambda *a, **kw: (False, "calendar not found"),
    )

    out = web_tools.propose_calendar_event(title="x", start="jueves 4pm")
    data = json.loads(out)
    assert data["created"] is False
    assert "calendar not found" in data["error"]


# ── registry ────────────────────────────────────────────────────────────────


def test_propose_tools_registered_in_chat_tools():
    names = {fn.__name__ for fn in web_tools.CHAT_TOOLS}
    assert "propose_reminder" in names
    assert "propose_calendar_event" in names
    assert "propose_reminder" in web_tools.TOOL_FNS
    assert "propose_calendar_event" in web_tools.TOOL_FNS


def test_propose_tools_are_parallel_safe():
    """Parsing + osascript — el segundo es IO-bound pero no comparte
    estado, safe para thread pool."""
    assert "propose_reminder" in web_tools.PARALLEL_SAFE
    assert "propose_calendar_event" in web_tools.PARALLEL_SAFE


def test_propose_tool_names_exported():
    assert "propose_reminder" in web_tools.PROPOSAL_TOOL_NAMES
    assert "propose_calendar_event" in web_tools.PROPOSAL_TOOL_NAMES
