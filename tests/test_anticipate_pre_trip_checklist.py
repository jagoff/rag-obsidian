"""Tests for the 'pre_trip_checklist' Anticipatory Agent signal.

Mockea `_fetch_calendar_ahead` — evita osascript / icalBuddy.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from rag.integrations import calendar as calendar_pkg
from rag_anticipate.signals.pre_trip_checklist import (
    _score_from_label,
    _slug,
    pre_trip_checklist_signal,
)


_NOW = datetime(2026, 5, 10, 12, 0, 0)


def _make_event(title: str = "Viaje a Madrid", date_label: str = "tomorrow") -> dict:
    return {"title": title, "date_label": date_label}


@pytest.fixture
def mock_calendar(monkeypatch):
    state: dict = {"events": []}

    def _fake_fetch(days_ahead: int = 3, max_events: int = 40):
        return list(state["events"])

    monkeypatch.setattr(calendar_pkg, "_fetch_calendar_ahead", _fake_fetch)
    return state


# ── _slug ────────────────────────────────────────────────────────────────────


def test_slug_basic():
    assert _slug("Viaje a Madrid") == "viaje-a-madrid"


def test_slug_strips_special():
    assert _slug("Vuelo MIA-EZE!") == "vuelo-mia-eze"


# ── _score_from_label ───────────────────────────────────────────────────────


def test_score_today():
    assert _score_from_label("today") == (1.0, "hoy")
    assert _score_from_label("hoy") == (1.0, "hoy")


def test_score_tomorrow():
    assert _score_from_label("tomorrow") == (1.0, "mañana")
    assert _score_from_label("mañana") == (1.0, "mañana")


def test_score_in_2_days():
    assert _score_from_label("in 2 days") == (0.8, "en 2 días")
    assert _score_from_label("en 2 días") == (0.8, "en 2 días")


def test_score_in_3_days():
    assert _score_from_label("in 3 days") == (0.6, "en 3 días")


def test_score_too_far():
    assert _score_from_label("in 7 days") is None
    assert _score_from_label("next week") is None


def test_score_empty():
    assert _score_from_label("") is None
    assert _score_from_label(None) is None


# ── Empty / silent-fail ──────────────────────────────────────────────────────


def test_no_events_returns_empty(mock_calendar):
    mock_calendar["events"] = []
    assert pre_trip_checklist_signal(_NOW) == []


def test_calendar_fetch_returns_none(monkeypatch):
    monkeypatch.setattr(calendar_pkg, "_fetch_calendar_ahead",
                        lambda days_ahead=3, max_events=40: None)
    assert pre_trip_checklist_signal(_NOW) == []


def test_calendar_raises(monkeypatch):
    def _broken(**kw):
        raise RuntimeError("ical down")
    monkeypatch.setattr(calendar_pkg, "_fetch_calendar_ahead", _broken)
    assert pre_trip_checklist_signal(_NOW) == []


# ── Pattern matching ─────────────────────────────────────────────────────────


def test_viaje_matches(mock_calendar):
    mock_calendar["events"] = [_make_event(title="Viaje a Madrid")]
    result = pre_trip_checklist_signal(_NOW)
    assert len(result) == 1
    assert "Madrid" in result[0].message


def test_vuelo_matches(mock_calendar):
    mock_calendar["events"] = [_make_event(title="Vuelo MIA-EZE LH 2345")]
    assert len(pre_trip_checklist_signal(_NOW)) == 1


def test_trip_matches(mock_calendar):
    mock_calendar["events"] = [_make_event(title="Trip to NYC")]
    assert len(pre_trip_checklist_signal(_NOW)) == 1


def test_flight_matches(mock_calendar):
    mock_calendar["events"] = [_make_event(title="Flight to Madrid")]
    assert len(pre_trip_checklist_signal(_NOW)) == 1


def test_vacaciones_matches(mock_calendar):
    mock_calendar["events"] = [_make_event(title="Vacaciones familia")]
    assert len(pre_trip_checklist_signal(_NOW)) == 1


def test_unrelated_event_skipped(mock_calendar):
    """`Reunión X` sin word `viaje/vuelo/etc` → no match."""
    mock_calendar["events"] = [_make_event(title="Reunión cliente")]
    assert pre_trip_checklist_signal(_NOW) == []


def test_word_boundary_no_false_positive(mock_calendar):
    """`Reunión preparar viaje` SÍ matchea (palabra completa "viaje")."""
    mock_calendar["events"] = [_make_event(title="Reunión preparar viaje")]
    # Es ambiguo — el regex es \bviaje\b, "preparar viaje" tiene la
    # palabra completa, así que matchea (false positive aceptable
    # según el docstring del signal).
    assert len(pre_trip_checklist_signal(_NOW)) == 1


def test_substring_no_match(mock_calendar):
    """`viajero` (substring de viaje) NO matchea por word boundary."""
    mock_calendar["events"] = [_make_event(title="Reunión con viajero")]
    # "viajero" tiene boundary en "v" pero no después de "viaje" — el
    # \b al final de "viaje" requiere non-word char y "r" es word. NO match.
    assert pre_trip_checklist_signal(_NOW) == []


# ── Score by proximity ───────────────────────────────────────────────────────


def test_today_emits_score_1_0(mock_calendar):
    mock_calendar["events"] = [_make_event(date_label="today")]
    result = pre_trip_checklist_signal(_NOW)
    assert result[0].score == 1.0
    assert "hoy" in result[0].message


def test_tomorrow_emits_score_1_0(mock_calendar):
    mock_calendar["events"] = [_make_event(date_label="tomorrow")]
    result = pre_trip_checklist_signal(_NOW)
    assert result[0].score == 1.0


def test_in_2_days_score_0_8(mock_calendar):
    mock_calendar["events"] = [_make_event(date_label="in 2 days")]
    result = pre_trip_checklist_signal(_NOW)
    assert result[0].score == 0.8


def test_in_3_days_score_0_6(mock_calendar):
    mock_calendar["events"] = [_make_event(date_label="in 3 days")]
    result = pre_trip_checklist_signal(_NOW)
    assert result[0].score == 0.6


def test_far_future_skipped(mock_calendar):
    mock_calendar["events"] = [_make_event(date_label="in 7 days")]
    assert pre_trip_checklist_signal(_NOW) == []


# ── Empty title skipped ──────────────────────────────────────────────────────


def test_empty_title_skipped(mock_calendar):
    mock_calendar["events"] = [_make_event(title="")]
    assert pre_trip_checklist_signal(_NOW) == []


# ── Cap ──────────────────────────────────────────────────────────────────────


def test_max_2_candidates(mock_calendar):
    mock_calendar["events"] = [
        _make_event(title="Viaje 1"),
        _make_event(title="Viaje 2"),
        _make_event(title="Viaje 3"),
    ]
    result = pre_trip_checklist_signal(_NOW)
    assert len(result) == 2


# ── Dedup key ────────────────────────────────────────────────────────────────


def test_dedup_key_format(mock_calendar):
    mock_calendar["events"] = [_make_event(title="Viaje a Madrid", date_label="tomorrow")]
    result = pre_trip_checklist_signal(_NOW)
    assert result[0].dedup_key == "trip:2026-05-10:viaje-a-madrid"


# ── Registry ─────────────────────────────────────────────────────────────────


def test_signal_registered():
    import rag_anticipate.signals  # noqa: F401
    from rag_anticipate.signals.base import SIGNALS
    names = [n for n, _ in SIGNALS]
    assert "pre_trip_checklist" in names
