"""Tests for the `rag log` post-T10 SQL reader.

Pre-2026-04-21-evening the command read `~/.local/share/obsidian-rag/
queries.jsonl` — a file whose *query-event* writers were removed during
the T10 cutover (2026-04-19). Query events now live in `rag_queries` SQL.
The CLI silently rendered empty rows because the JSONL file was being
filled by a different log stream (`_persist_conversation_turn`
observability events) that doesn't share the event schema.

This suite pins the new behaviour:
- `_read_queries_for_log` reads from `rag_queries` SQL, returns
  chronologically-ordered event-shaped dicts.
- `_read_feedback_map_for_log` reads from `rag_feedback`, returns
  {turn_id: +1|-1}.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import rag


def _open_db(tmp_path: Path) -> sqlite3.Connection:
    db = tmp_path / rag._TELEMETRY_DB_FILENAME
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
    """Redirect `DB_PATH` so `_ragvec_state_conn()` points at a clean tmp db."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    conn = _open_db(tmp_path)
    yield tmp_path, conn
    conn.close()


# ── _read_queries_for_log ─────────────────────────────────────────────

def test_read_queries_empty(sql_env):
    """No rows → empty list, not an exception."""
    tmp, _ = sql_env
    assert rag._read_queries_for_log(20) == []


def test_read_queries_returns_newest_first_then_flipped(sql_env):
    """SQL query is ORDER BY ts DESC; the helper reverses to chronological
    for the renderer (newest last = scrolls down in terminal like tail -f).
    """
    tmp, _ = sql_env
    # Three events, clearly-ordered ts.
    rag.log_query_event({"q": "first", "ts": "2026-04-21T10:00:00",
                          "top_score": 0.5, "t_retrieve": 1.0, "cmd": "query"})
    rag.log_query_event({"q": "second", "ts": "2026-04-21T11:00:00",
                          "top_score": 0.6, "t_retrieve": 1.0, "cmd": "query"})
    rag.log_query_event({"q": "third", "ts": "2026-04-21T12:00:00",
                          "top_score": 0.7, "t_retrieve": 1.0, "cmd": "query"})
    events = rag._read_queries_for_log(20)
    assert [e["q"] for e in events] == ["first", "second", "third"]


def test_read_queries_respects_n_limit(sql_env):
    """LIMIT N clamps the SELECT — return only the N newest (post-flip
    still chronological)."""
    tmp, _ = sql_env
    for i in range(5):
        rag.log_query_event({
            "q": f"q{i}",
            "ts": f"2026-04-21T10:{i:02d}:00",
            "top_score": 0.5,
        })
    events = rag._read_queries_for_log(3)
    assert len(events) == 3
    # Newest 3 = q2, q3, q4.
    assert [e["q"] for e in events] == ["q2", "q3", "q4"]


def test_read_queries_low_confidence_filter(sql_env):
    """`low_confidence=True` → only rows with top_score < CONFIDENCE_RERANK_MIN."""
    tmp, _ = sql_env
    rag.log_query_event({"q": "high", "top_score": 0.8})
    rag.log_query_event({"q": "low", "top_score": 0.01})
    rag.log_query_event({"q": "refuse", "top_score": -0.5})
    rag.log_query_event({"q": "no_score", "top_score": None})
    events = rag._read_queries_for_log(20, low_confidence=True)
    # 0.8 is high; 0.01 < CONFIDENCE_RERANK_MIN (0.015) passes;
    # -0.5 < threshold passes; None → NULL → filtered out by `IS NOT NULL`.
    q_list = [e["q"] for e in events]
    assert "high" not in q_list
    assert "no_score" not in q_list
    assert "low" in q_list
    assert "refuse" in q_list


def test_read_queries_extracts_turn_id_from_extra_json(sql_env):
    """`turn_id` lives inside `extra_json` (dict), not as a first-class
    column. The reader hoists it so the feedback join can match."""
    tmp, _ = sql_env
    rag.log_query_event({
        "q": "sample",
        "turn_id": "abc-123",
        "top_score": 0.5,
    })
    events = rag._read_queries_for_log(20)
    assert len(events) == 1
    assert events[0]["turn_id"] == "abc-123"


def test_read_queries_tolerates_missing_extra_json(sql_env):
    """Rows without `extra_json` (legacy or minimal events) must not raise."""
    tmp, _ = sql_env
    rag.log_query_event({"q": "minimal", "top_score": 0.5})
    events = rag._read_queries_for_log(20)
    assert len(events) == 1
    assert "turn_id" not in events[0]  # no synthetic key when absent


def test_read_queries_carries_ts_q_score_timings_mode(sql_env):
    """Smoke: every field the renderer uses comes through intact."""
    tmp, _ = sql_env
    rag.log_query_event({
        "q": "full",
        "ts": "2026-04-21T14:00:00",
        "top_score": 0.42,
        "t_retrieve": 1.5,
        "t_gen": 3.2,
        "mode": "semantic",
        "cmd": "query",
    })
    events = rag._read_queries_for_log(20)
    assert len(events) == 1
    e = events[0]
    assert e["ts"] == "2026-04-21T14:00:00"
    assert e["q"] == "full"
    assert e["top_score"] == 0.42
    assert e["t_retrieve"] == 1.5
    assert e["t_gen"] == 3.2
    assert e["mode"] == "semantic"


def test_read_queries_n_zero_returns_empty(sql_env):
    """Defensive: N <= 0 short-circuits without hitting SQL."""
    tmp, _ = sql_env
    rag.log_query_event({"q": "exists", "top_score": 0.5})
    assert rag._read_queries_for_log(0) == []
    assert rag._read_queries_for_log(-5) == []


def test_read_queries_excludes_rows_with_empty_q(sql_env):
    """Admin-style events (cmd='followup' / 'read' / etc) log to
    rag_queries with `q=""` via `_map_queries_row`'s setdefault fallback.
    Those shouldn't show up as "last queries" — the user expects actual
    search queries there. Filter at the SQL level."""
    tmp, _ = sql_env
    # Real queries (q non-empty).
    rag.log_query_event({"q": "real query 1", "top_score": 0.5, "cmd": "query"})
    rag.log_query_event({"q": "real query 2", "top_score": 0.6, "cmd": "chat"})
    # Admin-style events (q omitted → setdefault to "").
    rag.log_query_event({"cmd": "followup", "top_score": None})
    rag.log_query_event({"cmd": "read", "top_score": None})
    rag.log_query_event({"q": "", "cmd": "web.tasks"})

    events = rag._read_queries_for_log(20)
    q_list = [e["q"] for e in events]
    assert "real query 1" in q_list
    assert "real query 2" in q_list
    # No empty queries surfaced.
    assert "" not in q_list
    assert len(events) == 2


# ── _read_feedback_map_for_log ────────────────────────────────────────

def test_read_feedback_map_empty(sql_env):
    assert rag._read_feedback_map_for_log() == {}


def _insert_feedback(conn: sqlite3.Connection, **kwargs) -> None:
    """Direct INSERT into rag_feedback. Avoids the `record_feedback`
    side-effect (golden cache clear) so tests stay isolated to the
    reader logic."""
    row = rag._map_feedback_row(kwargs)
    rag._sql_append_event(conn, "rag_feedback", row)


def test_read_feedback_map_normalizes_ratings(sql_env):
    """Raw ratings can be any int (e.g. -2, +3 rarely); the map collapses
    to +1/-1 for the renderer's thumb emoji logic."""
    _, conn = sql_env
    _insert_feedback(conn, turn_id="t1", rating=1, q="a")
    _insert_feedback(conn, turn_id="t2", rating=-1, q="a")
    _insert_feedback(conn, turn_id="t3", rating=3, q="a")
    _insert_feedback(conn, turn_id="t4", rating=-2, q="a")
    fb = rag._read_feedback_map_for_log()
    assert fb == {"t1": 1, "t2": -1, "t3": 1, "t4": -1}


def test_read_feedback_map_skips_null_turn_id(sql_env):
    """Some legacy feedback rows have no turn_id (scope='global') — they
    can't attach to a specific query in the renderer, so skip."""
    _, conn = sql_env
    _insert_feedback(conn, turn_id="t1", rating=1, q="a")
    _insert_feedback(conn, turn_id=None, rating=-1, scope="global", q="b")
    fb = rag._read_feedback_map_for_log()
    assert fb == {"t1": 1}


def test_read_feedback_map_latest_rating_wins(sql_env):
    """If the same turn_id has multiple ratings (user flipped their
    opinion), the latest ts wins. Anchor via explicit ts per row."""
    _, conn = sql_env
    _insert_feedback(conn, turn_id="tx", rating=1, q="a",
                      ts="2026-04-21T10:00:00")
    _insert_feedback(conn, turn_id="tx", rating=-1, q="a",
                      ts="2026-04-21T14:00:00")
    fb = rag._read_feedback_map_for_log()
    assert fb == {"tx": -1}
