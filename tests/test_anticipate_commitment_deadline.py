"""Tests for the 'commitment_deadline' Anticipatory Agent signal.

La signal lee `rag_promises` SQL. Tests usan un DB temporal y patch
`_ragvec_state_conn` para apuntar a él.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest

import rag
from rag_anticipate.signals.commitment_deadline import (
    _format_message,
    _parse_due_ts,
    _score_for_proximity,
    commitment_deadline_signal,
)


_REF_NOW = datetime(2026, 5, 9, 12, 0, 0)


# ── DDL helper ───────────────────────────────────────────────────────────────


_RAG_PROMISES_DDL = """
CREATE TABLE IF NOT EXISTS rag_promises (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  contact_jid TEXT NOT NULL,
  contact_name TEXT,
  promise_text TEXT NOT NULL,
  direction TEXT NOT NULL,
  due_ts TEXT,
  due_confidence REAL,
  source_msg_id TEXT,
  source_chat_jid TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  reminder_sent_ts TEXT,
  closed_ts TEXT,
  closed_reason TEXT,
  extra_json TEXT
)
"""


@pytest.fixture
def mock_db(tmp_path, monkeypatch):
    """DB temporal con `rag_promises` DDL + patch `_ragvec_state_conn`."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(_RAG_PROMISES_DDL)
    conn.commit()
    conn.close()

    @contextmanager
    def _fake_conn():
        c = sqlite3.connect(str(db_path))
        try:
            yield c
            c.commit()
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_conn)
    return db_path


def _insert_promise(db_path, **kwargs):
    """Insert helper. Defaults: pending outgoing user→Juan due hoy."""
    defaults = {
        "ts": _REF_NOW.isoformat(),
        "contact_jid": "5491100000000@c.us",
        "contact_name": "Juan",
        "promise_text": "te paso el archivo mañana",
        "direction": "outgoing",
        "due_ts": _REF_NOW.date().isoformat(),
        "status": "pending",
    }
    defaults.update(kwargs)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO rag_promises (ts, contact_jid, contact_name, "
        "promise_text, direction, due_ts, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (defaults["ts"], defaults["contact_jid"], defaults["contact_name"],
         defaults["promise_text"], defaults["direction"], defaults["due_ts"],
         defaults["status"]),
    )
    conn.commit()
    conn.close()


# ── Helper unit tests ────────────────────────────────────────────────────────


def test_parse_due_ts_iso_date():
    assert _parse_due_ts("2026-05-09") == datetime(2026, 5, 9).date()


def test_parse_due_ts_iso_datetime():
    assert _parse_due_ts("2026-05-09T15:30:00") == datetime(2026, 5, 9).date()


def test_parse_due_ts_iso_with_offset():
    assert _parse_due_ts("2026-05-09T15:30:00Z") == datetime(2026, 5, 9).date()


def test_parse_due_ts_empty_returns_none():
    assert _parse_due_ts("") is None
    assert _parse_due_ts(None) is None
    assert _parse_due_ts("   ") is None


def test_parse_due_ts_malformed():
    assert _parse_due_ts("ayer a la mañana") is None
    assert _parse_due_ts("2026-13-99") is None


def test_score_for_proximity():
    assert _score_for_proximity(-7) == 1.0  # overdue
    assert _score_for_proximity(-1) == 1.0
    assert _score_for_proximity(0) == 1.0   # hoy
    assert _score_for_proximity(1) == 0.75
    assert _score_for_proximity(2) == 0.5
    assert _score_for_proximity(3) == 0.25


def test_format_message_outgoing_today():
    msg = _format_message(
        "te paso el doc", "Juan", "outgoing", days_until=0,
    )
    assert "🤝" in msg
    assert "Juan" in msg
    assert "vence hoy" in msg
    assert "te paso el doc" in msg


def test_format_message_incoming_overdue():
    msg = _format_message(
        "te llamo cuando vuelva", "Maria", "incoming", days_until=-3,
    )
    assert "📌" in msg
    assert "Maria" in msg
    assert "overdue (3d)" in msg


def test_format_message_truncates_long_promise():
    long_text = "a" * 200
    msg = _format_message(long_text, "X", "outgoing", days_until=0)
    assert "…" in msg
    # 120 cap + ellipsis
    assert "a" * 200 not in msg


def test_format_message_unknown_contact():
    msg = _format_message("X", "", "outgoing", days_until=1)
    assert "alguien" in msg


# ── Empty / edge ─────────────────────────────────────────────────────────────


def test_empty_table_returns_empty(mock_db):
    assert commitment_deadline_signal(_REF_NOW) == []


def test_no_due_ts_skipped(mock_db):
    _insert_promise(mock_db, due_ts="")
    assert commitment_deadline_signal(_REF_NOW) == []


def test_status_closed_skipped(mock_db):
    _insert_promise(mock_db, status="closed")
    assert commitment_deadline_signal(_REF_NOW) == []


def test_empty_promise_text_skipped(mock_db):
    _insert_promise(mock_db, promise_text="")
    assert commitment_deadline_signal(_REF_NOW) == []


def test_invalid_direction_skipped(mock_db):
    _insert_promise(mock_db, direction="weird")
    assert commitment_deadline_signal(_REF_NOW) == []


# ── Window filtering ─────────────────────────────────────────────────────────


def test_too_old_overdue_skipped(mock_db):
    """Overdue > 7 días → ruido, skip."""
    old_due = (_REF_NOW - timedelta(days=15)).date().isoformat()
    _insert_promise(mock_db, due_ts=old_due)
    assert commitment_deadline_signal(_REF_NOW) == []


def test_too_far_upcoming_skipped(mock_db):
    """Upcoming > +3 días → todavía no es señal."""
    far_due = (_REF_NOW + timedelta(days=10)).date().isoformat()
    _insert_promise(mock_db, due_ts=far_due)
    assert commitment_deadline_signal(_REF_NOW) == []


# ── Happy path ───────────────────────────────────────────────────────────────


def test_due_today_outgoing(mock_db):
    _insert_promise(mock_db,
                    due_ts=_REF_NOW.date().isoformat(),
                    direction="outgoing",
                    contact_name="Juan")
    result = commitment_deadline_signal(_REF_NOW)
    assert len(result) == 1
    c = result[0]
    assert c.kind == "anticipate-commitment_deadline"
    assert c.score == 1.0
    assert "🤝" in c.message  # outgoing emoji
    assert "Juan" in c.message
    assert "vence hoy" in c.message


def test_due_today_incoming(mock_db):
    _insert_promise(mock_db,
                    due_ts=_REF_NOW.date().isoformat(),
                    direction="incoming",
                    contact_name="Maria")
    result = commitment_deadline_signal(_REF_NOW)
    assert len(result) == 1
    assert "📌" in result[0].message  # incoming emoji


def test_due_tomorrow_score_0_75(mock_db):
    tomorrow = (_REF_NOW + timedelta(days=1)).date().isoformat()
    _insert_promise(mock_db, due_ts=tomorrow)
    result = commitment_deadline_signal(_REF_NOW)
    assert result[0].score == 0.75
    assert "vence mañana" in result[0].message


def test_overdue_score_1_0(mock_db):
    overdue = (_REF_NOW - timedelta(days=3)).date().isoformat()
    _insert_promise(mock_db, due_ts=overdue)
    result = commitment_deadline_signal(_REF_NOW)
    assert result[0].score == 1.0
    assert "overdue (3d)" in result[0].message


# ── Multi-row + sorting ──────────────────────────────────────────────────────


def test_max_2_emit(mock_db):
    """3 promesas pending → emit solo 2."""
    today = _REF_NOW.date().isoformat()
    for i in range(3):
        _insert_promise(mock_db, contact_name=f"P{i}", due_ts=today)
    result = commitment_deadline_signal(_REF_NOW)
    assert len(result) == 2


def test_overdue_first_then_upcoming(mock_db):
    """Sort: más overdue primero, después closest upcoming."""
    overdue = (_REF_NOW - timedelta(days=2)).date().isoformat()
    today = _REF_NOW.date().isoformat()
    upcoming = (_REF_NOW + timedelta(days=2)).date().isoformat()

    _insert_promise(mock_db, contact_name="UpcomingPerson", due_ts=upcoming)
    _insert_promise(mock_db, contact_name="OverduePerson", due_ts=overdue)
    _insert_promise(mock_db, contact_name="TodayPerson", due_ts=today)

    result = commitment_deadline_signal(_REF_NOW)
    assert len(result) == 2
    # Most overdue first
    assert "OverduePerson" in result[0].message
    # Then today
    assert "TodayPerson" in result[1].message


# ── Dedup key ────────────────────────────────────────────────────────────────


def test_dedup_key_format(mock_db):
    _insert_promise(mock_db, due_ts=_REF_NOW.date().isoformat())
    result = commitment_deadline_signal(_REF_NOW)
    today_iso = _REF_NOW.date().isoformat()
    assert result[0].dedup_key.startswith("commitment:")
    assert today_iso in result[0].dedup_key


# ── Registry ─────────────────────────────────────────────────────────────────


def test_signal_registered():
    import rag_anticipate.signals  # noqa: F401
    from rag_anticipate.signals.base import SIGNALS
    names = [n for n, _ in SIGNALS]
    assert "commitment_deadline" in names


# ── Silent fail ──────────────────────────────────────────────────────────────


def test_table_missing_returns_empty(tmp_path, monkeypatch):
    """DB sin la tabla `rag_promises` (instalación nueva sin DDL aplicada)."""
    db_path = tmp_path / "no_table.db"
    sqlite3.connect(str(db_path)).close()  # crea vacío

    @contextmanager
    def _fake_conn():
        c = sqlite3.connect(str(db_path))
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_conn)
    assert commitment_deadline_signal(_REF_NOW) == []


def test_conn_error_returns_empty(monkeypatch):
    """Si `_ragvec_state_conn` raisea (DB lockada, path inválido) → []."""
    def _broken():
        raise RuntimeError("DB lock")
    monkeypatch.setattr(rag, "_ragvec_state_conn", _broken)
    assert commitment_deadline_signal(_REF_NOW) == []
