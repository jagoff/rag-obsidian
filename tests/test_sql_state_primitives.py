"""Unit tests for the T1 SQL state-store foundation.

Feature flag RAG_STATE_SQL gates the writer path in later tasks. Here the
helpers are exercised directly against a fresh sqlite3 connection — the
flag is irrelevant for the primitives themselves; callers in T3 will
short-circuit upstream when the flag is off.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sqlite3
from pathlib import Path

import pytest

import rag


def _fresh_conn(path: Path) -> sqlite3.Connection:
    """Open a conn with the same pragmas SqliteVecClient would apply, then
    call _ensure_telemetry_tables. Also creates rag_schema_version up-front
    so the schema_version registration branch runs.
    """
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_schema_version ("
        " table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
    )
    rag._ensure_telemetry_tables(conn)
    return conn


@pytest.fixture
def conn(tmp_path):
    c = _fresh_conn(tmp_path / rag._TELEMETRY_DB_FILENAME)
    yield c
    c.close()


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rag_%' OR name LIKE 'system_memory_%'"
        ).fetchall()
    }


def test_ensure_telemetry_tables_idempotent(tmp_path):
    db = tmp_path / rag._TELEMETRY_DB_FILENAME
    c = _fresh_conn(db)
    before = _table_names(c)
    rag._ensure_telemetry_tables(c)
    rag._ensure_telemetry_tables(c)
    after = _table_names(c)
    assert before == after
    # All telemetry tables present (rag_schema_version already created).
    # 12 from PM plan (queries, behavior, feedback, feedback_golden,
    # feedback_golden_meta, tune, contradictions, ambient, ambient_state,
    # brief_written, brief_state, conversations_index) + 9 extras inferred
    # from live JSONL shapes (wa_tasks, archive_log, filing_log, eval_runs,
    # surface_log, proactive_log, cpu_metrics, memory_metrics,
    # system_memory_metrics) + 1 OCR cache (rag_ocr_cache, 2026-04-21) +
    # 2 entity tables (rag_entities, rag_entity_mentions, Improvement #2 Fase A) +
    # 1 response cache (rag_response_cache, GC#1 2026-04-22) +
    # 1 audio transcript cache (rag_audio_transcripts, STT MVP 2026-04-22) +
    # 1 score calibration (rag_score_calibration, Feature #2 2026-04-23) =
    # 27 tables total.
    expected = {name for name, _ in rag._TELEMETRY_DDL}
    assert expected.issubset(after)
    assert len(expected) == 27
    c.close()


def test_schema_version_registered(conn):
    rows = conn.execute(
        "SELECT table_name, version FROM rag_schema_version ORDER BY table_name"
    ).fetchall()
    registered = {name: ver for name, ver in rows}
    for name, _ in rag._TELEMETRY_DDL:
        assert registered.get(name) == 1, f"{name} missing or wrong version"


def test_schema_version_skipped_when_table_absent(tmp_path):
    """If rag_schema_version doesn't exist, _ensure_telemetry_tables must not
    fail — it just skips the registration branch. This mirrors the scenario
    in T9 rehearsal where a bare DB is used for migration dry-runs."""
    db = tmp_path / "bare.db"
    c = sqlite3.connect(str(db), isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")
    # Intentionally DO NOT create rag_schema_version.
    rag._ensure_telemetry_tables(c)
    # Tables got created even without schema_version.
    names = _table_names(c)
    assert "rag_queries" in names
    # But schema_version table still doesn't exist.
    found = c.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rag_schema_version'"
    ).fetchone()
    assert found is None
    c.close()


def test_sql_append_event_roundtrip(conn):
    row_q = {
        "ts": "2026-04-19T12:00:00",
        "cmd": "query",
        "q": "qué dice el vault sobre X",
        "session": "wa:5493425153999@s.whatsapp.net",
        "mode": "strict",
        "top_score": 0.42,
        "t_retrieve": 0.15,
        "t_gen": 1.8,
        "answer_len": 420,
        "citation_repaired": 0,
        "critique_fired": 0,
        "critique_changed": 0,
        "variants_json": ["v1", "v2", "v3"],
        "paths_json": ["00-Inbox/a.md", "01-Projects/b.md"],
        "scores_json": [0.42, 0.31],
        "filters_json": {"folder": "01-Projects"},
        "bad_citations_json": [],
        "extra_json": {"loose": False},
    }
    rowid = rag._sql_append_event(conn, "rag_queries", row_q)
    assert rowid >= 1

    fetched = conn.execute(
        "SELECT q, session, top_score, variants_json, paths_json, filters_json "
        "FROM rag_queries WHERE id = ?",
        (rowid,),
    ).fetchone()
    assert fetched[0] == row_q["q"]
    assert fetched[1] == row_q["session"]
    assert abs(fetched[2] - 0.42) < 1e-9
    assert json.loads(fetched[3]) == ["v1", "v2", "v3"]
    assert json.loads(fetched[4]) == ["00-Inbox/a.md", "01-Projects/b.md"]
    assert json.loads(fetched[5]) == {"folder": "01-Projects"}

    row_b = {
        "ts": "2026-04-19T12:01:00",
        "source": "cli",
        "event": "open",
        "path": "01-Projects/b.md",
        "query": "qué dice el vault sobre X",
        "rank": 2,
        "dwell_s": 15.3,
        "extra_json": {"ui": "terminal"},
    }
    bid = rag._sql_append_event(conn, "rag_behavior", row_b)
    fb = conn.execute(
        "SELECT source, event, path, rank, dwell_s, extra_json FROM rag_behavior WHERE id = ?",
        (bid,),
    ).fetchone()
    assert fb[0] == "cli"
    assert fb[1] == "open"
    assert fb[2] == "01-Projects/b.md"
    assert fb[3] == 2
    assert abs(fb[4] - 15.3) < 1e-9
    assert json.loads(fb[5]) == {"ui": "terminal"}


def test_sql_upsert_replace(conn):
    row1 = {
        "session_id": "wa:5493425153999",
        "relative_path": "00-Inbox/conversations/2026-04-19-1100-aaa.md",
        "updated_at": "2026-04-19T11:00:00",
    }
    row2 = {
        "session_id": "wa:5493425153999",  # same PK
        "relative_path": "00-Inbox/conversations/2026-04-19-1200-bbb.md",
        "updated_at": "2026-04-19T12:00:00",
    }
    rag._sql_upsert(conn, "rag_conversations_index", row1, ("session_id",))
    rag._sql_upsert(conn, "rag_conversations_index", row2, ("session_id",))

    rows = conn.execute(
        "SELECT session_id, relative_path, updated_at FROM rag_conversations_index"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][1] == row2["relative_path"]
    assert rows[0][2] == row2["updated_at"]


def test_sql_query_window_time_filter(conn):
    for i, ts in enumerate([
        "2026-04-10T00:00:00",
        "2026-04-12T00:00:00",
        "2026-04-15T00:00:00",  # inside window
        "2026-04-16T00:00:00",  # inside window
        "2026-04-17T00:00:00",  # inside window (ts < until is exclusive)
    ]):
        rag._sql_append_event(conn, "rag_queries", {
            "ts": ts,
            "q": f"q{i}",
        })

    rows = rag._sql_query_window(
        conn,
        "rag_queries",
        since_ts="2026-04-15T00:00:00",
        until_ts="2026-04-18T00:00:00",
    )
    qs = [r["q"] for r in rows]
    assert qs == ["q2", "q3", "q4"]

    # No until_ts → everything from since
    rows_all = rag._sql_query_window(conn, "rag_queries", "2026-04-12T00:00:00")
    assert len(rows_all) == 4

    # With extra where clause
    filtered = rag._sql_query_window(
        conn,
        "rag_queries",
        "2026-04-01T00:00:00",
        where="q = ?",
        params=("q2",),
    )
    assert len(filtered) == 1
    assert filtered[0]["q"] == "q2"


def test_sql_count_since(conn):
    for ts in ["2026-04-10T00:00:00", "2026-04-15T00:00:00", "2026-04-19T00:00:00"]:
        rag._sql_append_event(conn, "rag_queries", {"ts": ts, "q": "x"})
    assert rag._sql_count_since(conn, "rag_queries", "2026-04-14T00:00:00") == 2
    assert rag._sql_count_since(conn, "rag_queries", "2026-04-01T00:00:00") == 3
    assert rag._sql_count_since(conn, "rag_queries", "2026-04-20T00:00:00") == 0


def test_sql_max_ts_feedback_golden(conn):
    # empty table → None
    assert rag._sql_max_ts(conn, "rag_feedback") is None
    rag._sql_append_event(conn, "rag_feedback", {
        "ts": "2026-04-10T00:00:00", "rating": 1,
    })
    rag._sql_append_event(conn, "rag_feedback", {
        "ts": "2026-04-19T00:00:00", "rating": 1,
    })
    rag._sql_append_event(conn, "rag_feedback", {
        "ts": "2026-04-15T00:00:00", "rating": -1,
    })
    assert rag._sql_max_ts(conn, "rag_feedback") == "2026-04-19T00:00:00"


def test_synchronous_normal_applied(conn):
    # synchronous=NORMAL is PRAGMA value 1
    val = conn.execute("PRAGMA synchronous").fetchone()[0]
    assert val == 1, f"expected synchronous=NORMAL (1), got {val}"


def _writer_worker(args):
    """Module-level so it's picklable by multiprocessing.Pool."""
    db_path, n_inserts, worker_id = args
    import sqlite3 as _sq
    import rag as _rag
    conn = _sq.connect(db_path, isolation_level=None, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        for i in range(n_inserts):
            _rag._sql_append_event(conn, "rag_behavior", {
                "ts": f"2026-04-19T12:{worker_id:02d}:{i:02d}",
                "source": "test",
                "event": "open",
                "path": f"w{worker_id}/n{i}.md",
                "rank": i,
            })
    finally:
        conn.close()
    return worker_id


def test_concurrent_writers_no_starvation(tmp_path):
    db = tmp_path / rag._TELEMETRY_DB_FILENAME
    c = _fresh_conn(db)
    c.close()

    n_workers = 10
    n_each = 50
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        results = pool.map(
            _writer_worker,
            [(str(db), n_each, wid) for wid in range(n_workers)],
        )
    assert sorted(results) == list(range(n_workers))

    verify = sqlite3.connect(str(db))
    count = verify.execute("SELECT COUNT(*) FROM rag_behavior").fetchone()[0]
    verify.close()
    assert count == n_workers * n_each
