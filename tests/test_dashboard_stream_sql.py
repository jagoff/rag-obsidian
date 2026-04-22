"""Tests for the SQL-polled /api/dashboard/stream (post-T10 cutover).

Covers:
  - Row-to-event mappers preserve the legacy JSONL shape `_stream_payload`
    consumes (downstream dashboard JS untouched).
  - `_stream_max_id` returns 0 on empty/missing table.
  - `_stream_fetch_since` respects the row cap + ordering.
  - A deterministic end-to-end via the `gen()` coroutine: seed rows →
    iterate a few SSE events → assert the expected kinds + payload fields.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

import rag
from web import server as web_server


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
def db_env(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG",
                         tmp_path / "sql_state_errors.jsonl")
    # Force the state conn to resolve to our tmp DB — the helper already
    # reads DB_PATH so this is a no-op redundancy safeguard.
    _open_db(tmp_path).close()
    return tmp_path


# ── mappers ──────────────────────────────────────────────────────────────────

def test_row_to_query_ev_unpacks_json_fields_and_extras():
    row = {
        "id": 1,
        "ts": "2026-04-20T10:00:00",
        "q": "hola",
        "cmd": "query",
        "session": "web:abc",
        "top_score": 0.42,
        "t_retrieve": 0.12,
        "t_gen": 0.55,
        "paths_json": json.dumps(["a.md", "b.md"]),
        "bad_citations_json": json.dumps(["bad.md"]),
        "extra_json": json.dumps(
            {"gated_low_confidence": True, "error": "oops"}
        ),
    }
    ev = web_server._row_to_query_ev(row)
    assert ev["q"] == "hola"
    assert ev["cmd"] == "query"
    assert ev["paths"] == ["a.md", "b.md"]
    assert ev["bad_citations"] == ["bad.md"]
    assert ev["gated_low_confidence"] is True
    assert ev["error"] == "oops"
    # Payload shape after _stream_payload should be what the JS expects.
    payload = web_server._stream_payload("query", ev)
    assert payload["q"] == "hola"
    assert payload["source"] == "web"  # sid starts with web:
    assert payload["bad_citations"] == 1
    assert payload["n_paths"] == 2
    assert payload["gated"] is True


def test_row_to_query_ev_handles_missing_and_corrupt_json():
    row = {
        "id": 2, "ts": "2026-04-20T10:00:00",
        "q": "x", "cmd": "query", "session": "cli:z",
        "top_score": None, "t_retrieve": None, "t_gen": None,
        "paths_json": "{bad json",
        "bad_citations_json": None,
        "extra_json": "",
    }
    ev = web_server._row_to_query_ev(row)
    assert ev["paths"] == []
    assert ev["bad_citations"] == []
    assert ev["gated_low_confidence"] is None
    assert ev["error"] is None


def test_row_to_feedback_ev_unpacks_reason_from_extra():
    row = {
        "id": 1, "ts": "2026-04-20T10:00:00",
        "rating": 1, "q": "hola", "scope": "web",
        "paths_json": json.dumps(["a.md"]),
        "extra_json": json.dumps(
            {"reason": "buena", "corrective_path": "better.md"}
        ),
    }
    ev = web_server._row_to_feedback_ev(row)
    assert ev["rating"] == 1
    assert ev["paths"] == ["a.md"]
    assert ev["reason"] == "buena"
    assert ev["corrective_path"] == "better.md"


def test_row_to_ambient_ev_pulls_wikilinks_from_payload():
    row = {
        "id": 1, "ts": "2026-04-20T10:00:00",
        "cmd": "note.write", "path": "00-Inbox/x.md", "hash": "abc",
        "payload_json": json.dumps({"wikilinks_applied": 3}),
    }
    ev = web_server._row_to_ambient_ev(row)
    assert ev["path"] == "00-Inbox/x.md"
    assert ev["wikilinks_applied"] == 3


def test_row_to_ambient_ev_missing_payload_yields_zero():
    row = {
        "id": 1, "ts": "2026-04-20T10:00:00",
        "cmd": "note.write", "path": "00-Inbox/x.md",
        "hash": "abc", "payload_json": None,
    }
    ev = web_server._row_to_ambient_ev(row)
    assert ev["wikilinks_applied"] == 0


def test_row_to_contradiction_ev_path_is_subject_path():
    row = {
        "id": 1, "ts": "2026-04-20T10:00:00",
        "subject_path": "02-Areas/Foo.md",
        "contradicts_json": json.dumps(["03-Resources/Bar.md"]),
        "helper_raw": "...",
        "skipped": None,
    }
    ev = web_server._row_to_contradiction_ev(row)
    assert ev["path"] == "02-Areas/Foo.md"
    assert ev["contradicts"] == ["03-Resources/Bar.md"]
    assert ev["skipped"] is False


def test_row_to_contradiction_ev_skipped_flag():
    row = {
        "id": 1, "ts": "2026-04-20T10:00:00",
        "subject_path": "02-Areas/Foo.md",
        "contradicts_json": "[]",
        "helper_raw": "",
        "skipped": "too_short",
    }
    ev = web_server._row_to_contradiction_ev(row)
    assert ev["skipped"] is True


# ── cursor helpers ───────────────────────────────────────────────────────────

def test_stream_max_id_zero_on_empty(db_env):
    with rag._ragvec_state_conn() as conn:
        assert web_server._stream_max_id(conn, "rag_queries") == 0


def test_stream_fetch_since_respects_order_and_cap(db_env, monkeypatch):
    # Drop cap temporarily to test tight behavior with a small count.
    monkeypatch.setattr(web_server, "_STREAM_ROW_CAP", 3)
    with rag._ragvec_state_conn() as conn:
        for i in range(5):
            rag._sql_append_event(conn, "rag_queries", {
                "ts": f"2026-04-20T10:00:0{i}",
                "q": f"q{i}", "cmd": "query", "session": "cli:x",
            })
        rows = web_server._stream_fetch_since(conn, "rag_queries", 0)
        assert len(rows) == 3  # capped
        assert [r["q"] for r in rows] == ["q0", "q1", "q2"]
        rows2 = web_server._stream_fetch_since(conn, "rag_queries", rows[-1]["id"])
        assert [r["q"] for r in rows2] == ["q3", "q4"]


def test_stream_fetch_since_missing_table_returns_empty(db_env, monkeypatch):
    # Point at a non-existent table.
    with rag._ragvec_state_conn() as conn:
        assert web_server._stream_fetch_since(conn, "rag_nonexistent", 0) == []


# ── end-to-end SSE gen ───────────────────────────────────────────────────────

def _collect_events(gen_coro, max_events: int = 20, max_iters: int = 200):
    """Drive `gen_coro` until we've seen `max_events` non-hello/heartbeat
    events or `max_iters` iterations pass — whichever first."""
    events: list[tuple[str, dict]] = []
    agen = gen_coro
    loop = asyncio.new_event_loop()
    try:
        for _ in range(max_iters):
            try:
                raw = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
            # SSE frame: `event: X\ndata: {...}\n\n`
            lines = raw.splitlines()
            kind = None
            data = None
            for ln in lines:
                if ln.startswith("event: "):
                    kind = ln[len("event: "):]
                elif ln.startswith("data: "):
                    try:
                        data = json.loads(ln[len("data: "):])
                    except Exception:
                        data = None
            if kind and kind not in ("hello", "heartbeat"):
                events.append((kind, data))
                if len(events) >= max_events:
                    break
    finally:
        loop.run_until_complete(agen.aclose())
        loop.close()
    return events


def test_dashboard_stream_emits_sql_rows(db_env, monkeypatch):
    """Seed rag_queries + rag_feedback, drive the SSE generator, confirm
    both kinds show up with the expected payloads. Patch asyncio.sleep to
    a no-op so the poll loop iterates at full speed inside the event loop.
    """
    # First call (cursor init) skips seed rows; new rows inserted after the
    # init must be emitted. We simulate this by seeding AFTER the generator
    # initialises its cursors — easiest via patching _stream_max_id once,
    # then inserting rows before the next poll. In practice the test below
    # pre-seeds + uses max_id=0 by monkey-patching _stream_max_id.
    with rag._ragvec_state_conn() as conn:
        rag._sql_append_event(conn, "rag_queries", {
            "ts": "2026-04-20T10:00:00", "q": "hola", "cmd": "query",
            "session": "web:abc",
            "top_score": 0.5, "t_retrieve": 0.1, "t_gen": 0.3,
            "paths_json": ["a.md"],
        })
        rag._sql_append_event(conn, "rag_feedback", {
            "ts": "2026-04-20T10:00:01", "turn_id": "t1",
            "rating": 1, "q": "hola", "scope": "web",
            "paths_json": ["a.md"],
        })

    # Force cursors to 0 at start so pre-seeded rows get emitted.
    monkeypatch.setattr(web_server, "_stream_max_id", lambda conn, table: 0)
    # Speed up the poll loop — no real sleep in tests.
    async def _no_sleep(_s):
        return None
    monkeypatch.setattr(web_server.asyncio, "sleep", _no_sleep)

    resp = asyncio.new_event_loop().run_until_complete(
        web_server.dashboard_stream()
    )
    gen = resp.body_iterator
    events = _collect_events(gen, max_events=2)

    kinds = [e[0] for e in events]
    assert "query" in kinds
    assert "feedback" in kinds
    # Query payload shape checks (via _stream_payload pipeline).
    q_ev = next(p for k, p in events if k == "query")
    assert q_ev["q"] == "hola"
    assert q_ev["source"] == "web"
    f_ev = next(p for k, p in events if k == "feedback")
    assert f_ev["rating"] == 1
    assert f_ev["q"] == "hola"
