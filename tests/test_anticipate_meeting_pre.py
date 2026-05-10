"""Tests for the 'meeting_pre' Anticipatory Agent signal.

Mockea `rag.integrations.calendar._fetch_calendar_today` — evita
osascript / icalBuddy.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from rag.integrations import calendar as calendar_pkg
from rag_anticipate.signals.meeting_pre import (
    _parse_hhmm,
    _slug,
    meeting_pre_signal,
)


# 2026-05-10 10:00 — fixed reference. Eventos a las 10:05, 10:10, 10:15
# están en window. Eventos pre-09:59 ya empezaron, post-10:16 fuera.
_NOW = datetime(2026, 5, 10, 10, 0, 0)


def _make_event(
    title: str = "Standup",
    start: str = "10:10",
    end: str = "10:30",
) -> dict:
    return {"title": title, "start": start, "end": end}


@pytest.fixture
def mock_calendar(monkeypatch):
    state: dict = {"events": []}

    def _fake_fetch(max_events: int = 20):
        return list(state["events"])

    monkeypatch.setattr(calendar_pkg, "_fetch_calendar_today", _fake_fetch)
    return state


# ── _slug ────────────────────────────────────────────────────────────────────


def test_slug_lowercase_alphanum():
    assert _slug("Stand-Up Meeting") == "stand-up-meeting"


def test_slug_strips_special_chars():
    assert _slug("1:1 con Juan!") == "11-con-juan"


def test_slug_collapses_spaces():
    assert _slug("a   b   c") == "a-b-c"


def test_slug_caps_at_60():
    long = "a" * 100
    assert len(_slug(long)) <= 60


# ── _parse_hhmm ──────────────────────────────────────────────────────────────


def test_parse_hhmm_basic():
    assert _parse_hhmm("10:30") == (10, 30)


def test_parse_hhmm_single_digit_hour():
    assert _parse_hhmm("9:05") == (9, 5)


def test_parse_hhmm_invalid_minute():
    assert _parse_hhmm("10:99") is None


def test_parse_hhmm_invalid_hour():
    assert _parse_hhmm("25:00") is None


def test_parse_hhmm_empty():
    assert _parse_hhmm("") is None
    assert _parse_hhmm(None) is None


def test_parse_hhmm_garbage():
    assert _parse_hhmm("hola") is None
    assert _parse_hhmm("10-30") is None


# ── Empty / silent-fail ──────────────────────────────────────────────────────


def test_no_events_returns_empty(mock_calendar):
    mock_calendar["events"] = []
    assert meeting_pre_signal(_NOW) == []


def test_calendar_fetch_returns_none(monkeypatch):
    monkeypatch.setattr(calendar_pkg, "_fetch_calendar_today",
                        lambda max_events=20: None)
    assert meeting_pre_signal(_NOW) == []


def test_calendar_raises(monkeypatch):
    def _broken(max_events=20):
        raise RuntimeError("ical down")
    monkeypatch.setattr(calendar_pkg, "_fetch_calendar_today", _broken)
    assert meeting_pre_signal(_NOW) == []


# ── Window filtering ─────────────────────────────────────────────────────────


def test_event_already_started_skipped(mock_calendar):
    """Evento que arrancó hace 5min → skip."""
    mock_calendar["events"] = [_make_event(start="09:55", end="10:30")]
    assert meeting_pre_signal(_NOW) == []


def test_event_too_far_skipped(mock_calendar):
    """Evento a 30min → fuera de window 15."""
    mock_calendar["events"] = [_make_event(start="10:30", end="11:00")]
    assert meeting_pre_signal(_NOW) == []


def test_event_starting_now_emits(mock_calendar):
    """Lead = 0 → emit con score 1.0."""
    mock_calendar["events"] = [_make_event(start="10:00", end="10:30")]
    result = meeting_pre_signal(_NOW)
    assert len(result) == 1
    assert result[0].score == 1.0
    assert "ahora" in result[0].message


# ── Score buckets ────────────────────────────────────────────────────────────


def test_lead_3min_score_1_0(mock_calendar):
    mock_calendar["events"] = [_make_event(start="10:03", end="10:30")]
    result = meeting_pre_signal(_NOW)
    assert result[0].score == 1.0


def test_lead_8min_score_0_9(mock_calendar):
    mock_calendar["events"] = [_make_event(start="10:08", end="10:30")]
    result = meeting_pre_signal(_NOW)
    assert result[0].score == 0.9


def test_lead_13min_score_0_7(mock_calendar):
    mock_calendar["events"] = [_make_event(start="10:13", end="10:30")]
    result = meeting_pre_signal(_NOW)
    assert result[0].score == 0.7


# ── Anti-noise ───────────────────────────────────────────────────────────────


def test_busy_title_skipped(mock_calendar):
    mock_calendar["events"] = [_make_event(title="Busy", start="10:10")]
    assert meeting_pre_signal(_NOW) == []


def test_tentative_title_skipped(mock_calendar):
    mock_calendar["events"] = [_make_event(title="Tentative", start="10:10")]
    assert meeting_pre_signal(_NOW) == []


def test_empty_title_skipped(mock_calendar):
    mock_calendar["events"] = [_make_event(title="", start="10:10")]
    assert meeting_pre_signal(_NOW) == []


def test_all_day_event_skipped(mock_calendar):
    """all-day events typically tienen `start=""` o no-HH:MM."""
    mock_calendar["events"] = [_make_event(title="All-day", start="")]
    assert meeting_pre_signal(_NOW) == []


# ── Sort + max ───────────────────────────────────────────────────────────────


def test_max_3_candidates(mock_calendar):
    """5 reuniones back-to-back → cap a 3."""
    mock_calendar["events"] = [
        _make_event(title=f"M{i}", start=f"10:{i:02d}", end=f"10:{i+10:02d}")
        for i in range(1, 16, 3)  # 5 events: 10:01, 10:04, ...
    ]
    result = meeting_pre_signal(_NOW)
    assert len(result) <= 3


# ── Message format ───────────────────────────────────────────────────────────


def test_message_contains_emoji_and_time_range(mock_calendar):
    mock_calendar["events"] = [_make_event(
        title="Daily Standup", start="10:05", end="10:30",
    )]
    result = meeting_pre_signal(_NOW)
    msg = result[0].message
    assert "🗓" in msg
    assert "Daily Standup" in msg
    assert "10:05-10:30" in msg


def test_message_lead_1_minute(mock_calendar):
    mock_calendar["events"] = [_make_event(start="10:01", end="10:30")]
    result = meeting_pre_signal(_NOW)
    assert "en 1min" in result[0].message


def test_message_lead_5_min(mock_calendar):
    mock_calendar["events"] = [_make_event(start="10:05", end="10:30")]
    result = meeting_pre_signal(_NOW)
    assert "en 5min" in result[0].message


# ── Dedup key ────────────────────────────────────────────────────────────────


def test_dedup_key_format(mock_calendar):
    mock_calendar["events"] = [_make_event(
        title="My Meeting", start="10:05", end="10:30",
    )]
    result = meeting_pre_signal(_NOW)
    # `meeting:YYYY-MM-DD:slug:HH:MM`
    assert result[0].dedup_key == "meeting:2026-05-10:my-meeting:10:05"


def test_dedup_key_changes_if_event_moves(mock_calendar):
    mock_calendar["events"] = [_make_event(start="10:05")]
    r1 = meeting_pre_signal(_NOW)
    mock_calendar["events"] = [_make_event(start="10:10")]
    r2 = meeting_pre_signal(_NOW)
    assert r1[0].dedup_key != r2[0].dedup_key


# ── Registry ─────────────────────────────────────────────────────────────────


def test_signal_registered():
    import rag_anticipate.signals  # noqa: F401
    from rag_anticipate.signals.base import SIGNALS
    names = [n for n, _ in SIGNALS]
    assert "meeting_pre" in names
