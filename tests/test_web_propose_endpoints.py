"""Tests del web server para los endpoints y la detección de intent de
crear. Cubre:
  - POST /api/reminders/create con campos nuevos (due_iso, priority,
    notes, recurrence) — y backward compat con due="tomorrow".
  - POST /api/calendar/create shape + errores.
  - _detect_propose_intent: patrones de creación reminder/event.
  - _maybe_emit_proposal: SSE frame válido o None.
  - CHAT_TOOLS y PROPOSAL_TOOL_NAMES expuestos correctamente.

No levanta ollama — los creators están mockeados vía monkeypatch sobre
`rag._create_reminder` / `rag._create_calendar_event`.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import rag
from web import server as web_server


@pytest.fixture
def client(monkeypatch):
    # Desactivar warmup / prewarm threads que no corresponden en tests.
    monkeypatch.setattr(web_server, "_warmup", lambda: None)
    return TestClient(web_server.app)


# ── /api/reminders/create backward compat ───────────────────────────────────


def test_reminders_create_legacy_due_tomorrow(client, monkeypatch):
    captured = {}

    def fake_create(name, due_token=None, list_name=None, **kwargs):
        captured["name"] = name
        captured["due_token"] = due_token
        captured["due_dt"] = kwargs.get("due_dt")
        captured["list_name"] = list_name
        return True, "x-apple-reminderkit://ABC"

    monkeypatch.setattr(rag, "_create_reminder", fake_create)
    res = client.post("/api/reminders/create", json={"text": "pan", "due": "tomorrow"})
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["id"].startswith("x-apple-reminderkit://")
    assert captured["name"] == "pan"
    assert captured["due_token"] == "tomorrow"
    assert captured["due_dt"] is None


def test_reminders_create_with_due_iso(client, monkeypatch):
    captured = {}

    def fake_create(name, due_token=None, list_name=None, **kwargs):
        captured["due_token"] = due_token
        captured["due_dt"] = kwargs.get("due_dt")
        captured["priority"] = kwargs.get("priority")
        captured["notes"] = kwargs.get("notes")
        captured["recurrence"] = kwargs.get("recurrence")
        return True, "id-123"

    monkeypatch.setattr(rag, "_create_reminder", fake_create)
    res = client.post("/api/reminders/create", json={
        "text": "llamar a Juan",
        "due_iso": "2026-04-25T14:30:00",
        "priority": 1,
        "notes": "sobre el proyecto",
        "recurrence": {"freq": "WEEKLY", "interval": 1, "byday": ["MO"]},
    })
    assert res.status_code == 200
    # due_iso wins → due_token should NOT be passed.
    assert captured["due_token"] is None
    assert captured["due_dt"] == datetime(2026, 4, 25, 14, 30, 0)
    assert captured["priority"] == 1
    assert captured["notes"] == "sobre el proyecto"
    assert captured["recurrence"]["freq"] == "WEEKLY"


def test_reminders_create_malformed_iso(client):
    res = client.post("/api/reminders/create", json={
        "text": "x", "due_iso": "not-a-date",
    })
    assert res.status_code == 400
    assert "due_iso" in res.json()["detail"]


def test_reminders_create_creator_fails(client, monkeypatch):
    monkeypatch.setattr(
        rag, "_create_reminder",
        lambda *a, **kw: (False, "Apple integration deshabilitada"),
    )
    res = client.post("/api/reminders/create", json={"text": "x"})
    assert res.status_code == 400
    assert "deshabilitada" in res.json()["detail"]


# ── /api/calendar/create ────────────────────────────────────────────────────


def test_calendar_create_happy(client, monkeypatch):
    captured = {}

    def fake_create(title, start, end=None, **kwargs):
        captured["title"] = title
        captured["start"] = start
        captured["end"] = end
        captured["calendar"] = kwargs.get("calendar")
        captured["location"] = kwargs.get("location")
        captured["notes"] = kwargs.get("notes")
        captured["all_day"] = kwargs.get("all_day")
        captured["recurrence"] = kwargs.get("recurrence")
        return True, "UID-XYZ"

    monkeypatch.setattr(rag, "_create_calendar_event", fake_create)
    res = client.post("/api/calendar/create", json={
        "title": "reunión",
        "start_iso": "2026-04-25T14:00:00",
        "end_iso": "2026-04-25T15:30:00",
        "calendar": "Trabajo",
        "location": "Palermo",
        "notes": "llevar laptop",
        "recurrence": {"freq": "WEEKLY", "interval": 1},
    })
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["uid"] == "UID-XYZ"
    assert captured["title"] == "reunión"
    assert captured["start"] == datetime(2026, 4, 25, 14, 0)
    assert captured["end"] == datetime(2026, 4, 25, 15, 30)
    assert captured["calendar"] == "Trabajo"
    assert captured["location"] == "Palermo"
    assert captured["notes"] == "llevar laptop"
    assert captured["recurrence"]["freq"] == "WEEKLY"


def test_calendar_create_default_end(client, monkeypatch):
    captured = {}

    def fake_create(title, start, end=None, **kwargs):
        captured["end"] = end
        return True, "UID"

    monkeypatch.setattr(rag, "_create_calendar_event", fake_create)
    res = client.post("/api/calendar/create", json={
        "title": "x",
        "start_iso": "2026-04-25T14:00:00",
    })
    assert res.status_code == 200
    # Server passes end=None when end_iso omitted; rag._create_calendar_event
    # then computes +1h internally.
    assert captured["end"] is None


def test_calendar_create_malformed_start(client):
    res = client.post("/api/calendar/create", json={
        "title": "x", "start_iso": "not-iso",
    })
    assert res.status_code == 400
    assert "start_iso" in res.json()["detail"]


def test_calendar_create_creator_fails(client, monkeypatch):
    monkeypatch.setattr(
        rag, "_create_calendar_event",
        lambda *a, **kw: (False, "calendar not found"),
    )
    res = client.post("/api/calendar/create", json={
        "title": "x", "start_iso": "2026-04-25T14:00:00",
        "calendar": "Nonexistent",
    })
    assert res.status_code == 400
    assert "calendar not found" in res.json()["detail"]


# ── _detect_propose_intent ──────────────────────────────────────────────────


@pytest.mark.parametrize("q", [
    "recordame llamar a Juan mañana a las 10",
    "recordáme pagar el alquiler",
    "acordate de comprar pan",
    "agendame una llamada con el banco",
    "ponete un recordatorio para el lunes",
    "agregá un recordatorio para el viernes",
    "creá un evento para la reunión del jueves",
    "agendá una reunión con el equipo",
    "bloqueá un turno para el dentista",
    "poné una cita con el médico",
    # Implicit statement form — "tengo/hay/tenemos X" + temporal.
    "mañana tengo una daily meeting a las 10am",
    "tengo reunión con el equipo el jueves 4pm",
    "hay standup mañana 10:30",
    "la semana que viene tengo turno con el dentista",
    "me citaron para una entrevista el viernes a las 3",
    "el lunes que viene tengo call con el cliente",
    "mañana hay demo a las 16hs",
    # NEW: explicit calendar-action verbs (no event noun required).
    "el miercoles viene gracia a casa, calendarizalo",
    "calendarizame una reunión mañana",
    "agendalo el jueves",
    "agendala para el viernes",
    "anotá en el calendario el turno del jueves",
    "poné en la agenda la reunión del viernes 10am",
    # NEW: visit/arrival pattern (viene/pasa/llega X + temporal).
    "mañana pasa el plomero a casa",
    "el jueves llega mi vieja",
    "mañana viene mi hermana a casa",
    # NEW 2026-04-21: declaration with explicit clock time. Regression
    # after the Fer F. web-chat report: "el viernes 20hs tengo que ir de
    # Seba" slipped through every branch and got a generic RAG response
    # instead of a propose card. Branch 4 now catches any
    # (temporal anchor + explicit time) even without an event noun /
    # visit verb. "el jueves a las 4" moved here from negative — it used
    # to be flagged ambiguous, but the preview + confirm UX downstream
    # lets the user cancel if it wasn't what they meant.
    "el viernes 20hs tengo que ir de Seba",
    "el lunes a las 10am dentista",
    "hoy 18hs parte el vuelo",
    "el jueves a las 4",
    "el miércoles a las 9 voy al médico",
    "mañana a las 15:30",
    # NEW 2026-04-21 Playwright probe: reminder verbs that weren't
    # covered previously. Without these, "anotame llamar al plomero el
    # viernes" fell through to a plain retrieval path and the LLM
    # fabricated "Se ha registrado tu recordatorio…" without calling
    # any tool (silent hallucination — user thinks something was saved,
    # nothing was).
    "anotame llamar al plomero el viernes",
    "anotá comprar yerba el lunes",
    "apuntame revisar el PR mañana",
    "no te olvides de pagar el alquiler",
    "no te olvides de llamar al médico mañana",
    # NEW 2026-04-21 Fer F. second probe: absolute date forms + "cumple"
    # apócope. "el 26 de Mayo es el cumple de Astor" fell through because:
    #   1. `_TEMPORAL_ANCHOR_RE` didn't match "<day> de <month>" as an
    #      absolute anchor (only bare weekdays / "el <weekday>").
    #   2. `_EVENT_NOUN_RE` had "cumpleaños" but not the AR apócope
    #      "cumple".
    # Both regexes extended; now the declaration + event_noun path fires.
    "el 26 de Mayo es el cumple de Astor",
    "el 5 de enero viaje a Bariloche",
    "26 de mayo cumple Astor",
    "el 12/05 reunión con el abogado",
    "15/12/2026 aniversario de casamiento",
    "el cumple de mamá es el 3 de abril",
])
def test_detect_propose_intent_positive(q):
    assert web_server._detect_propose_intent(q) is True


@pytest.mark.parametrize("q", [
    "qué reuniones tengo esta semana",
    "qué pendientes tengo",
    "cuánto gasté en abril",
    "cómo llego a Palermo",
    "qué eventos tengo mañana",
    "mostrame los recordatorios",
    "cuándo es la próxima reunión",
    "dónde es la reunión",
    # NEW: question-word forms that look like create-intent but aren't.
    "quién viene mañana",
    "quiénes vienen el jueves",
    "cuándo viene el plomero",
    # NEW: `¿`-prefixed question form (rioplatense); _QUESTION_START_RE
    # didn't accept the opening mark pre-fix so these used to slip in.
    "¿qué hago el viernes 20hs?",
    "¿cuándo pasa Grecia?",
    # Bare event noun without temporal → not enough signal
    "tengo reunión",
    # Temporal without event noun / visit pattern / explicit time → still
    # ambiguous. "el viernes compromiso" names something but gives no
    # clock, so branch 4 declines.
    "el viernes compromiso",
    "el viernes tengo algo",
    "",
    None,
])
def test_detect_propose_intent_negative(q):
    assert web_server._detect_propose_intent(q or "") is False


# ── _maybe_emit_proposal ────────────────────────────────────────────────────


def test_maybe_emit_proposal_reminder_ok():
    import json
    payload = {
        "kind": "reminder",
        "proposal_id": "prop-abc",
        "fields": {"title": "x"},
    }
    out = web_server._maybe_emit_proposal("propose_reminder", json.dumps(payload))
    assert out is not None
    assert out.startswith("event: proposal\n")
    assert "prop-abc" in out


def test_maybe_emit_proposal_event_ok():
    import json
    payload = {
        "kind": "event",
        "proposal_id": "prop-xyz",
        "fields": {"title": "x"},
    }
    out = web_server._maybe_emit_proposal(
        "propose_calendar_event", json.dumps(payload),
    )
    assert out is not None
    assert "event: proposal\n" in out


def test_maybe_emit_proposal_wrong_tool_name():
    """Non-proposal tools return None even if output happens to be JSON."""
    out = web_server._maybe_emit_proposal("search_vault", '{"foo": "bar"}')
    assert out is None


def test_maybe_emit_proposal_malformed_json():
    """Parse error → None, caller falls through to plain tool_done."""
    out = web_server._maybe_emit_proposal("propose_reminder", "not json")
    assert out is None


def test_maybe_emit_proposal_missing_proposal_id():
    """Valid JSON but missing proposal_id → not a real proposal, None."""
    out = web_server._maybe_emit_proposal("propose_reminder", '{"kind":"reminder"}')
    assert out is None
