"""Tests for `_parse_agenda_window` — the temporal-anchor → (ts_start,
ts_end) parser used by the agenda intent to narrow results from the
full corpus down to a concrete time window.

All tests pin `now` explicitly so weekday math / week boundaries are
deterministic across the machine this suite runs on. Anchor chosen:
2026-04-22 (Wednesday) at 10:30 local — puts us mid-week to exercise
week-boundary calculations (weekday=2).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import rag


# Fixed anchor for every test. Wednesday 2026-04-22 10:30.
ANCHOR = datetime(2026, 4, 22, 10, 30, 0)


def _day_range(start_day: datetime) -> tuple[datetime, datetime]:
    s = start_day.replace(hour=0, minute=0, second=0, microsecond=0)
    return (s, s + timedelta(days=1))


# ── Day anchors ──────────────────────────────────────────────────────

def test_hoy_window():
    start, end = rag._parse_agenda_window("qué tengo hoy", now=ANCHOR)
    # Hoy = ANCHOR day (2026-04-22). Half-open [00:00, next 00:00).
    exp_s, exp_e = _day_range(ANCHOR)
    assert datetime.fromtimestamp(start) == exp_s
    assert datetime.fromtimestamp(end) == exp_e


def test_manana_window():
    start, end = rag._parse_agenda_window("qué tengo mañana", now=ANCHOR)
    exp_s, exp_e = _day_range(ANCHOR + timedelta(days=1))
    assert datetime.fromtimestamp(start) == exp_s
    assert datetime.fromtimestamp(end) == exp_e


def test_pasado_manana_window():
    start, end = rag._parse_agenda_window(
        "qué tengo pasado mañana", now=ANCHOR,
    )
    exp_s, exp_e = _day_range(ANCHOR + timedelta(days=2))
    assert datetime.fromtimestamp(start) == exp_s
    assert datetime.fromtimestamp(end) == exp_e


def test_pasado_manana_before_manana_precedence():
    """Regex ordering must match 'pasado mañana' BEFORE 'mañana' — if
    the order flips, 'pasado mañana' falls to the mañana branch."""
    # Anchor Wed 22. Pasado mañana = Fri 24. Mañana = Thu 23.
    start, _ = rag._parse_agenda_window("pasado mañana", now=ANCHOR)
    assert datetime.fromtimestamp(start) == datetime(2026, 4, 24, 0, 0)


def test_accent_variants_normalized():
    """'mañana' and 'manana' should both fire — users type both."""
    for text in ("mañana", "manana"):
        start, _ = rag._parse_agenda_window(f"qué tengo {text}", now=ANCHOR)
        assert datetime.fromtimestamp(start) == datetime(2026, 4, 23, 0, 0)


# ── Weekday anchors (via dateparser) ─────────────────────────────────

def test_el_viernes_this_week():
    """Anchor is Wed 22; 'el viernes' should resolve to Fri 24 (2d ahead)."""
    start, end = rag._parse_agenda_window("qué tengo el viernes", now=ANCHOR)
    # Fri 2026-04-24 [00:00, next 00:00)
    assert datetime.fromtimestamp(start) == datetime(2026, 4, 24, 0, 0)
    assert datetime.fromtimestamp(end) == datetime(2026, 4, 25, 0, 0)


def test_el_proximo_lunes():
    """Anchor is Wed 22; 'el próximo lunes' = next Mon (2026-04-27 or
    -30, dateparser resolves). Accept either but assert ≥ +4 days."""
    start, _ = rag._parse_agenda_window(
        "qué tengo el próximo lunes", now=ANCHOR,
    )
    dt = datetime.fromtimestamp(start)
    # It's a Monday on or after +5 days.
    assert dt.weekday() == 0, f"expected Monday, got {dt.strftime('%A')}"
    assert (dt - ANCHOR.replace(hour=0, minute=0, second=0, microsecond=0)).days >= 5


def test_este_sabado():
    """'Este sábado' = next upcoming Saturday (anchor Wed 22 → Sat 25)."""
    start, _ = rag._parse_agenda_window("qué tengo este sábado", now=ANCHOR)
    dt = datetime.fromtimestamp(start)
    assert dt.weekday() == 5  # Saturday
    assert dt.date() == datetime(2026, 4, 25).date()


# ── Weekend anchor ───────────────────────────────────────────────────

def test_el_finde_window():
    """'el finde' = próximo sábado + domingo como ventana de 2 días."""
    start, end = rag._parse_agenda_window("qué tengo el finde", now=ANCHOR)
    # Anchor Wed 22; el finde = Sat 25 → Mon 27 (exclusive).
    assert datetime.fromtimestamp(start) == datetime(2026, 4, 25, 0, 0)
    assert datetime.fromtimestamp(end) == datetime(2026, 4, 27, 0, 0)


# ── Week anchors ─────────────────────────────────────────────────────

def test_esta_semana_window():
    """Anchor Wed 2026-04-22 → week is Mon 20 → Mon 27 (ISO)."""
    start, end = rag._parse_agenda_window("qué tengo esta semana", now=ANCHOR)
    assert datetime.fromtimestamp(start) == datetime(2026, 4, 20, 0, 0)
    assert datetime.fromtimestamp(end) == datetime(2026, 4, 27, 0, 0)


def test_la_semana_que_viene():
    """Anchor Wed 22 → next week Mon 27 → Mon 4 May."""
    start, end = rag._parse_agenda_window(
        "qué tengo la semana que viene", now=ANCHOR,
    )
    assert datetime.fromtimestamp(start) == datetime(2026, 4, 27, 0, 0)
    assert datetime.fromtimestamp(end) == datetime(2026, 5, 4, 0, 0)


def test_la_proxima_semana():
    start, _ = rag._parse_agenda_window(
        "qué tengo la próxima semana", now=ANCHOR,
    )
    assert datetime.fromtimestamp(start) == datetime(2026, 4, 27, 0, 0)


# ── Month anchors ────────────────────────────────────────────────────

def test_este_mes_window():
    """April 2026: [Apr 1 00:00, May 1 00:00)."""
    start, end = rag._parse_agenda_window("qué tengo este mes", now=ANCHOR)
    assert datetime.fromtimestamp(start) == datetime(2026, 4, 1, 0, 0)
    assert datetime.fromtimestamp(end) == datetime(2026, 5, 1, 0, 0)


def test_el_proximo_mes():
    """From April → May [May 1, Jun 1)."""
    start, end = rag._parse_agenda_window(
        "qué tengo el próximo mes", now=ANCHOR,
    )
    assert datetime.fromtimestamp(start) == datetime(2026, 5, 1, 0, 0)
    assert datetime.fromtimestamp(end) == datetime(2026, 6, 1, 0, 0)


def test_el_mes_que_viene():
    start, _ = rag._parse_agenda_window(
        "qué tengo el mes que viene", now=ANCHOR,
    )
    assert datetime.fromtimestamp(start) == datetime(2026, 5, 1, 0, 0)


def test_mes_wraparound_december():
    """December + 'el próximo mes' must wrap to Jan of next year."""
    dec_anchor = datetime(2026, 12, 15, 10, 0)
    start, end = rag._parse_agenda_window(
        "qué tengo el próximo mes", now=dec_anchor,
    )
    assert datetime.fromtimestamp(start) == datetime(2027, 1, 1, 0, 0)
    assert datetime.fromtimestamp(end) == datetime(2027, 2, 1, 0, 0)


# ── Year anchors ─────────────────────────────────────────────────────

def test_este_ano_window():
    start, end = rag._parse_agenda_window("qué tengo este año", now=ANCHOR)
    assert datetime.fromtimestamp(start) == datetime(2026, 1, 1, 0, 0)
    assert datetime.fromtimestamp(end) == datetime(2027, 1, 1, 0, 0)


def test_el_proximo_ano():
    start, end = rag._parse_agenda_window(
        "qué tengo el próximo año", now=ANCHOR,
    )
    assert datetime.fromtimestamp(start) == datetime(2027, 1, 1, 0, 0)
    assert datetime.fromtimestamp(end) == datetime(2028, 1, 1, 0, 0)


# ── No anchor → None ─────────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "mi agenda",              # Possessive without temporal.
    "mis eventos",            # Same.
    "qué reuniones tengo",    # Event noun without temporal.
    "qué es ikigai",          # Plain semantic.
])
def test_no_anchor_returns_none(q):
    assert rag._parse_agenda_window(q, now=ANCHOR) is None


# ── Defensive: half-open interval (end is exclusive) ─────────────────

def test_window_half_open():
    """[start, end) must not include end — if a calendar event starts
    exactly at end_ts, it belongs to the NEXT window."""
    start, end = rag._parse_agenda_window("hoy", now=ANCHOR)
    # Day window: start = 2026-04-22 00:00, end = 2026-04-23 00:00.
    # end is 86400s after start, not 86400-1.
    assert end - start == pytest.approx(86400.0)


# ── Ordering guard: narrower anchors before broader ─────────────────

def test_esta_semana_not_captured_by_hoy():
    """'qué tengo esta semana' contains 'esta' but not 'hoy'. If the
    regex overmatches 'hoy' in some tokenization quirk, the test would
    fail on window size."""
    start, end = rag._parse_agenda_window("qué tengo esta semana", now=ANCHOR)
    assert end - start == pytest.approx(7 * 86400.0)  # 7 days, not 1


def test_este_mes_beats_esta_semana_when_both_tokens_present():
    """Edge case: 'qué tengo esta semana o este mes' — first match wins
    by dispatch order. Semana (week) is checked before mes (month) in
    my design because "semana" is narrower. Lock the order."""
    start, end = rag._parse_agenda_window(
        "qué tengo esta semana o este mes", now=ANCHOR,
    )
    # Should be week window (7d), not month (~30d).
    assert end - start == pytest.approx(7 * 86400.0)
