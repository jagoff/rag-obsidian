"""Tests for readers still-tail-reading JSONL after T10 SQL cutover.

Context: T10 (2026-04-19) moved all `log_*_event` writers from JSONL to
SQL. `rag log` was fixed in session 2026-04-21 evening; this suite pins
the behaviour for 3 more readers that had the same drift:

- `_scan_queries_log(days)` — consumed by `rag emergent` + `rag dashboard`
- `feedback_counts()` — consumed by `rag insights`
- `_feedback_augmented_cases()` — consumed by `rag tune` to mine
  corrective_path feedback; without this, tune can't learn from
  "here's the correct path" signal at all

All three now read from rag_queries / rag_feedback SQL.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import rag


def _open_db(tmp_path: Path) -> sqlite3.Connection:
    db = tmp_path / "ragvec.db"
    conn = sqlite3.connect(str(db), isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_schema_version ("
        " table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
    )
    rag._ensure_telemetry_tables(conn)
    return conn


@pytest.fixture
def sql_env(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    conn = _open_db(tmp_path)
    yield tmp_path, conn
    conn.close()


def _insert_feedback(conn: sqlite3.Connection, **kwargs) -> None:
    row = rag._map_feedback_row(kwargs)
    rag._sql_append_event(conn, "rag_feedback", row)


# ── _scan_queries_log ─────────────────────────────────────────────────

def test_scan_queries_log_returns_events_within_window(sql_env):
    from datetime import datetime, timedelta
    _, _ = sql_env
    # 3 events: one in-window (today), one at window edge (boundary), one
    # out of window (20 days ago).
    now = datetime.now()
    rag.log_query_event({
        "q": "in-window",
        "ts": now.isoformat(timespec="seconds"),
        "top_score": 0.5, "cmd": "query",
    })
    rag.log_query_event({
        "q": "old",
        "ts": (now - timedelta(days=20)).isoformat(timespec="seconds"),
        "top_score": 0.3, "cmd": "query",
    })
    rag.log_query_event({
        "q": "fresh",
        "ts": (now - timedelta(days=1)).isoformat(timespec="seconds"),
        "top_score": 0.7, "cmd": "chat",
    })
    events = rag._scan_queries_log(days=14)
    qs = {e["q"] for e in events}
    assert "in-window" in qs
    assert "fresh" in qs
    assert "old" not in qs


def test_scan_queries_log_empty_table(sql_env):
    assert rag._scan_queries_log(days=14) == []


def test_scan_queries_log_hoists_extra_json_keys(sql_env):
    """Callers (emergent, dashboard) read q_reformulated / answered /
    gated_low_confidence from the event dict. These live inside
    extra_json in SQL — they must be hoisted to top-level keys."""
    rag.log_query_event({
        "q": "sample",
        "q_reformulated": "rewritten",
        "top_score": 0.5,
        "answered": True,
        "gated_low_confidence": False,
        "cmd": "query",
    })
    events = rag._scan_queries_log(days=14)
    assert len(events) == 1
    e = events[0]
    assert e["q_reformulated"] == "rewritten"
    assert e["answered"] is True
    assert e["gated_low_confidence"] is False


def test_scan_queries_log_chronological_order(sql_env):
    """Order ASC so clustering / dashboard iteration sees events in the
    order they happened."""
    from datetime import datetime, timedelta
    now = datetime.now()
    for i in range(3):
        rag.log_query_event({
            "q": f"q{i}",
            "ts": (now - timedelta(minutes=30 - i * 10)).isoformat(
                timespec="seconds"
            ),
            "top_score": 0.5,
        })
    events = rag._scan_queries_log(days=1)
    assert [e["q"] for e in events] == ["q0", "q1", "q2"]


def test_scan_queries_log_does_not_include_admin_events(sql_env):
    """Admin events (cmd='followup', cmd='read', q='') shouldn't pollute
    emergent / dashboard. Same rule as `_read_queries_for_log`."""
    # No explicit filter on `q` in `_scan_queries_log` is fine IF the
    # callers filter themselves (emergent does: `len(q) >= 6`). But
    # dashboard counts everything — we prefer to exclude admin rows
    # here so both callers get clean data.
    rag.log_query_event({"q": "real query", "top_score": 0.5, "cmd": "query"})
    rag.log_query_event({"cmd": "followup", "top_score": None})  # q=""
    events = rag._scan_queries_log(days=14)
    qs = [e["q"] for e in events]
    assert "real query" in qs
    # Admin row excluded.
    assert "" not in qs


# ── feedback_counts ───────────────────────────────────────────────────

def test_feedback_counts_empty(sql_env):
    """No rows → (0, 0)."""
    assert rag.feedback_counts() == (0, 0)


def test_feedback_counts_positives_negatives(sql_env):
    _, conn = sql_env
    _insert_feedback(conn, turn_id="t1", rating=1, q="a")
    _insert_feedback(conn, turn_id="t2", rating=1, q="b")
    _insert_feedback(conn, turn_id="t3", rating=-1, q="c")
    _insert_feedback(conn, turn_id="t4", rating=1, q="d")
    _insert_feedback(conn, turn_id="t5", rating=-1, q="e")
    pos, neg = rag.feedback_counts()
    assert pos == 3
    assert neg == 2


def test_feedback_counts_ignores_zero_rating(sql_env):
    """Historical data can have rating=0 (synthetic / failed inserts).
    `feedback_counts` must count only strict +1 / -1 (not 0)."""
    _, conn = sql_env
    _insert_feedback(conn, turn_id="t1", rating=1, q="a")
    _insert_feedback(conn, turn_id="t2", rating=0, q="b")  # neither
    pos, neg = rag.feedback_counts()
    assert (pos, neg) == (1, 0)


# ── _feedback_augmented_cases ─────────────────────────────────────────

def test_feedback_augmented_cases_mines_corrective_path(sql_env):
    """The only path into the feedback_augmented training set is feedback
    rows with `corrective_path` set (user said: this was the correct
    answer). Must surface those as (q, [corrective_path]) cases."""
    _, conn = sql_env
    _insert_feedback(
        conn, turn_id="t1", rating=-1, q="qué es ikigai",
        corrective_path="03-Resources/Coaching/ikigai.md",
    )
    _insert_feedback(
        conn, turn_id="t2", rating=-1, q="letra de muros",
        corrective_path="04-Archive/Musica/muros.md",
    )
    cases = rag._feedback_augmented_cases(min_len=4)
    assert len(cases) == 2
    paths_per_q = {c["question"]: c["expected"] for c in cases}
    assert paths_per_q["qué es ikigai"] == ["03-Resources/Coaching/ikigai.md"]
    assert paths_per_q["letra de muros"] == ["04-Archive/Musica/muros.md"]


def test_feedback_augmented_cases_skips_session_scope(sql_env):
    """Session-scope feedback isn't tied to a specific query — can't be a
    training sample. Filter it out."""
    _, conn = sql_env
    _insert_feedback(
        conn, turn_id="t1", rating=-1, q="q",
        corrective_path="path.md",
        scope="session",
    )
    _insert_feedback(
        conn, turn_id="t2", rating=-1, q="real q",
        corrective_path="real.md",
        scope="turn",
    )
    cases = rag._feedback_augmented_cases(min_len=4)
    assert len(cases) == 1
    assert cases[0]["question"] == "real q"


def test_feedback_augmented_cases_dedupes_by_normalized_query(sql_env):
    """If the same query got corrective feedback multiple times, only
    one case surfaces. Case normalisation = lowercased + whitespace-
    collapsed."""
    _, conn = sql_env
    _insert_feedback(
        conn, turn_id="t1", rating=-1, q="Qué es  Ikigai",
        corrective_path="a.md",
    )
    _insert_feedback(
        conn, turn_id="t2", rating=-1, q="qué es ikigai",
        corrective_path="b.md",
    )
    cases = rag._feedback_augmented_cases(min_len=4)
    assert len(cases) == 1


def test_feedback_augmented_cases_respects_min_len(sql_env):
    """Short queries (e.g. "q" or single-word typos) aren't useful
    training samples."""
    _, conn = sql_env
    _insert_feedback(
        conn, turn_id="t1", rating=-1, q="ab",  # too short
        corrective_path="a.md",
    )
    _insert_feedback(
        conn, turn_id="t2", rating=-1, q="long enough",
        corrective_path="b.md",
    )
    cases = rag._feedback_augmented_cases(min_len=4)
    assert len(cases) == 1
    assert cases[0]["question"] == "long enough"


def test_feedback_augmented_cases_ignores_missing_corrective_path(sql_env):
    """Feedback without corrective_path (plain thumbs-down) isn't
    training data. Only corrective rows are."""
    _, conn = sql_env
    _insert_feedback(conn, turn_id="t1", rating=-1, q="thumbs down only")
    _insert_feedback(
        conn, turn_id="t2", rating=-1, q="with path",
        corrective_path="right.md",
    )
    cases = rag._feedback_augmented_cases(min_len=4)
    assert len(cases) == 1
    assert cases[0]["question"] == "with path"
