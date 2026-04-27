"""T3: writer-swap tests.

Each log/state writer in rag.py (plus web/server.py samplers) must:
  - route to the matching rag_* table when RAG_STATE_SQL=1
  - route to the legacy JSONL path when the flag is off
  - fall through to JSONL if the SQL write raises (cutover fail-safe)
  - NOT dual-write on the happy path

Feature-flag + DB path are patched per test via monkeypatch. `_ragvec_state_conn`
reads `rag.DB_PATH` and constructs `DB_PATH / rag._TELEMETRY_DB_FILENAME`, so pointing
DB_PATH at tmp_path is sufficient to isolate each test.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import rag
from web import server as web_server


def _count(conn, table: str, where: str = "", params: tuple = ()) -> int:
    sql = f"SELECT COUNT(*) FROM {table}"
    if where:
        sql += f" WHERE {where}"
    return int(conn.execute(sql, params).fetchone()[0])


def _open_db(tmp_path: Path) -> sqlite3.Connection:
    """Open the on-disk telemetry.db the writers created + apply T1 DDL so the
    test can SELECT against the same schema that was populated."""
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
    """Flag ON + DB_PATH redirected. Writers will use tmp_path/telemetry.db."""
    monkeypatch.setattr(rag, "RAG_STATE_SQL", True)
    monkeypatch.setattr(web_server, "RAG_STATE_SQL", True)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    # Redirect all JSONL paths too so any accidental JSONL write is detectable
    # and doesn't land in the real ~/.local/share/obsidian-rag/.
    _redirect_jsonl_paths(monkeypatch, tmp_path / "jsonl")
    yield tmp_path


@pytest.fixture
def jsonl_env(tmp_path, monkeypatch):
    """Flag OFF — JSONL paths redirected + any SQL write would go to a DB that
    we can still inspect to assert nothing was written."""
    monkeypatch.setattr(rag, "RAG_STATE_SQL", False)
    monkeypatch.setattr(web_server, "RAG_STATE_SQL", False)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    _redirect_jsonl_paths(monkeypatch, tmp_path / "jsonl")
    yield tmp_path


def _redirect_jsonl_paths(monkeypatch, base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "LOG_PATH", base / "queries.jsonl")
    monkeypatch.setattr(rag, "BEHAVIOR_LOG_PATH", base / "behavior.jsonl")
    monkeypatch.setattr(rag, "FEEDBACK_PATH", base / "feedback.jsonl")
    monkeypatch.setattr(rag, "FEEDBACK_GOLDEN_PATH", base / "feedback_golden.json")
    monkeypatch.setattr(rag, "BRIEF_WRITTEN_PATH", base / "brief_written.jsonl")
    monkeypatch.setattr(rag, "BRIEF_STATE_PATH", base / "brief_state.jsonl")
    monkeypatch.setattr(rag, "TUNE_LOG_PATH", base / "tune.jsonl")
    monkeypatch.setattr(rag, "CONTRADICTION_LOG_PATH", base / "contradictions.jsonl")
    monkeypatch.setattr(rag, "AMBIENT_LOG_PATH", base / "ambient.jsonl")
    monkeypatch.setattr(rag, "AMBIENT_STATE_PATH", base / "ambient_state.jsonl")
    monkeypatch.setattr(rag, "PROACTIVE_LOG_PATH", base / "proactive.jsonl")
    monkeypatch.setattr(rag, "SURFACE_LOG_PATH", base / "surface.jsonl")
    monkeypatch.setattr(rag, "FILING_LOG_PATH", base / "filing.jsonl")
    monkeypatch.setattr(rag, "ARCHIVE_LOG_PATH", base / "archive.jsonl")
    monkeypatch.setattr(rag, "WA_TASKS_LOG_PATH", base / "wa_tasks.jsonl")
    monkeypatch.setattr(rag, "EVAL_LOG_PATH", base / "eval.jsonl")
    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG", base / "sql_state_errors.jsonl")
    # web/server.py samplers
    monkeypatch.setattr(web_server, "_MEMORY_STATE_PATH", base / "rag_memory.jsonl")
    monkeypatch.setattr(web_server, "_CPU_STATE_PATH", base / "rag_cpu.jsonl")


def _flush_log_queue() -> None:
    """Drain the background writer before asserting on JSONL contents."""
    rag._LOG_QUEUE.join()


# ── Flag ON: writes land in SQL ──────────────────────────────────────────────

def test_log_query_writes_to_sql_when_flag_on(sql_env):
    rag.log_query_event({
        "cmd": "query", "q": "sql-ping", "top_score": 0.5,
        "t_retrieve": 0.1, "t_gen": 0.5,
    })
    conn = _open_db(sql_env)
    try:
        assert _count(conn, "rag_queries", "q = ?", ("sql-ping",)) == 1
        row = conn.execute(
            "SELECT cmd, q, top_score, t_retrieve, t_gen FROM rag_queries WHERE q = ?",
            ("sql-ping",),
        ).fetchone()
        assert row == ("query", "sql-ping", 0.5, 0.1, 0.5)
    finally:
        conn.close()


def test_log_query_writes_to_sql_even_with_flag_off(jsonl_env):
    """Post-T10: the RAG_STATE_SQL flag is inert — writers always target SQL.
    Flag-OFF callers still end up in rag_queries, not in a JSONL file."""
    rag.log_query_event({"cmd": "query", "q": "jsonl-ping", "top_score": 0.2})
    assert not rag.LOG_PATH.exists(), "JSONL written post-T10 (should be SQL)"
    conn = _open_db(jsonl_env)
    try:
        row = conn.execute(
            "SELECT q FROM rag_queries WHERE q = ?", ("jsonl-ping",),
        ).fetchone()
        assert row is not None
    finally:
        conn.close()


def test_behavior_writer_flag_on(sql_env):
    rag.log_behavior_event({"source": "cli", "event": "open", "path": "a.md",
                              "rank": 1, "dwell_s": 4.2})
    conn = _open_db(sql_env)
    try:
        row = conn.execute(
            "SELECT source, event, path, rank, dwell_s FROM rag_behavior"
        ).fetchone()
        assert row == ("cli", "open", "a.md", 1, 4.2)
    finally:
        conn.close()


def test_feedback_writer_flag_on(sql_env):
    rag.record_feedback("turn-1", 1, "why", ["a.md", "b.md"], reason="good")
    conn = _open_db(sql_env)
    try:
        row = conn.execute(
            "SELECT turn_id, rating, q, scope, paths_json, extra_json FROM rag_feedback"
        ).fetchone()
        assert row[0] == "turn-1"
        assert row[1] == 1
        assert row[2] == "why"
        assert row[3] == "turn"
        assert json.loads(row[4]) == ["a.md", "b.md"]
        extra = json.loads(row[5])
        assert extra.get("reason") == "good"
    finally:
        conn.close()


def test_tune_writer_flag_on(sql_env):
    rag._log_tune_event({
        "cmd": "tune", "samples": 100, "seed": 42, "n_cases": 30,
        "baseline": {"hit5": 0.8}, "best": {"hit5": 0.85}, "delta": 0.05,
    })
    conn = _open_db(sql_env)
    try:
        row = conn.execute(
            "SELECT cmd, samples, seed, n_cases, delta, baseline_json, best_json FROM rag_tune"
        ).fetchone()
        assert row[0] == "tune"
        assert row[1] == 100
        assert row[2] == 42
        assert row[3] == 30
        assert row[4] == 0.05
        assert json.loads(row[5]) == {"hit5": 0.8}
        assert json.loads(row[6]) == {"hit5": 0.85}
    finally:
        conn.close()


def test_contradiction_writer_flag_on(sql_env):
    rag._log_contradictions(
        "01-Projects/a.md",
        contrad=[{"path": "02-Areas/b.md", "note": "b", "why": "conflict"}],
        helper_raw="raw-json",
        skipped=None,
    )
    conn = _open_db(sql_env)
    try:
        row = conn.execute(
            "SELECT subject_path, contradicts_json, helper_raw, skipped FROM rag_contradictions"
        ).fetchone()
        assert row[0] == "01-Projects/a.md"
        assert json.loads(row[1])[0]["path"] == "02-Areas/b.md"
        assert row[2] == "raw-json"
        assert row[3] is None
    finally:
        conn.close()


def test_ambient_writer_flag_on(sql_env):
    rag._ambient_log_event({"cmd": "ambient_send", "path": "00-Inbox/x.md",
                              "hash": "abc123"})
    conn = _open_db(sql_env)
    try:
        row = conn.execute(
            "SELECT cmd, path, hash FROM rag_ambient"
        ).fetchone()
        assert row == ("ambient_send", "00-Inbox/x.md", "abc123")
    finally:
        conn.close()


def test_ambient_state_upsert_replaces(sql_env):
    rag._ambient_state_record("p.md", "h1", {"kind": "note"})
    rag._ambient_state_record("p.md", "h2", {"kind": "note-updated"})
    conn = _open_db(sql_env)
    try:
        rows = conn.execute(
            "SELECT path, hash, payload_json FROM rag_ambient_state"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "p.md"
        assert rows[0][1] == "h2"  # second write wins
        payload = json.loads(rows[0][2])
        assert payload["kind"] == "note-updated"
    finally:
        conn.close()


def test_brief_written_writer_flag_on(sql_env):
    rag.record_brief_written(
        "morning", Path("04-Archive/99-obsidian-system/99-AI/reviews/2026-04-19.md"),
        ["01-Projects/a.md", "02-Areas/b.md"],
        {"Prioridades": ["01-Projects/a.md"]},
    )
    conn = _open_db(sql_env)
    try:
        row = conn.execute(
            "SELECT brief_type, brief_path, paths_cited_json, citations_by_section_json FROM rag_brief_written"
        ).fetchone()
        assert row[0] == "morning"
        assert row[1] == "04-Archive/99-obsidian-system/99-AI/reviews/2026-04-19.md"
        assert json.loads(row[2]) == ["01-Projects/a.md", "02-Areas/b.md"]
        assert json.loads(row[3]) == {"Prioridades": ["01-Projects/a.md"]}
    finally:
        conn.close()


def test_sql_write_failure_logs_and_drops(sql_env, monkeypatch):
    """Post-T10: if the SQL path raises, the error is logged to
    sql_state_errors.jsonl and the event is silently dropped (no JSONL
    fallback write)."""
    orig = rag._sql_append_event

    def boom(*a, **kw):
        raise sqlite3.OperationalError("simulated-failure")
    monkeypatch.setattr(rag, "_sql_append_event", boom)
    try:
        rag.log_query_event({"cmd": "query", "q": "boom-test"})
    finally:
        monkeypatch.setattr(rag, "_sql_append_event", orig)
    # No JSONL fallback anymore.
    assert not rag.LOG_PATH.exists(), (
        "JSONL fallback fired despite T10 removing that path"
    )
    # Error log line must exist.
    err_text = rag._SQL_STATE_ERROR_LOG.read_text(encoding="utf-8")
    assert "queries_sql_write_failed" in err_text


def test_no_double_write_when_sql_succeeds(sql_env):
    """Happy path: SQL succeeds → JSONL file must NOT be touched."""
    rag.log_query_event({"cmd": "query", "q": "only-sql"})
    _flush_log_queue()
    assert not rag.LOG_PATH.exists(), "JSONL written despite SQL success"
    conn = _open_db(sql_env)
    try:
        assert _count(conn, "rag_queries", "q = ?", ("only-sql",)) == 1
    finally:
        conn.close()


# ── Parameterised: 9 metrics/tracking writers ────────────────────────────────
# Each tuple: (label, call_lambda, table, where_clause_to_find_row)
# Row-count assertion is 1 after a single call.

def _call_wa_tasks_log(base: Path) -> None:
    """wa_tasks writer is inline inside `run_wa_tasks` — test it via the
    same helpers (the row-mapper + _sql_append_event) it reaches for."""
    ev = {
        "ts": "2026-04-19T10:00:00",
        "since": "2026-04-19T09:00:00",
        "chats": 3, "items": 5,
        "path": "00-Inbox/WA-2026-04-19.md",
    }
    with rag._ragvec_state_conn() as conn:
        rag._sql_append_event(conn, "rag_wa_tasks", rag._map_wa_tasks_row(ev))


def _call_archive(base: Path) -> None:
    rag._log_archive_event({
        "cmd": "archive", "min_age_days": 365, "query_window_days": 30,
        "folder": "", "dry_run": True, "force": False, "gate": 20,
        "n_candidates": 7, "n_plan": 5, "n_applied": 0, "n_skipped": 2,
        "gated": False, "batch_path": "filing_batches/archive-x.jsonl",
    })


def _call_filing(base: Path) -> None:
    rag._filing_log_proposal({
        "path": "00-Inbox/x.md", "note": "x", "folder": "01-Projects/p/",
        "confidence": 0.8, "upward_title": "Parent", "upward_kind": "moc",
        "neighbors": [{"path": "01-Projects/p/y.md", "sim": 0.5}],
    })


def _call_eval(base: Path) -> None:
    entry = {
        "ts": "2026-04-19T10:00:00",
        "singles": {"hit5": 0.88, "mrr": 0.77, "n": 42},
        "chains": {"hit5": 0.79, "mrr": 0.63, "chain_success": 0.5,
                   "turns": 33, "chains": 12},
    }
    with rag._ragvec_state_conn() as conn:
        rag._sql_append_event(conn, "rag_eval_runs", rag._map_eval_row(entry))


def _call_surface(base: Path) -> None:
    rag._surface_log_run(
        {"n_pairs": 3, "sim_threshold": 0.7, "min_hops": 2, "top": 5,
         "skip_young_days": 14, "llm": True, "duration_ms": 123.4},
        [{"a": "01-Projects/a.md", "b": "02-Areas/b.md", "sim": 0.72}],
    )


def _call_proactive(base: Path) -> None:
    rag._proactive_log({"kind": "emergent", "sent": True, "reason": "ok"})


def _call_cpu(base: Path) -> None:
    sample = {
        "ts": "2026-04-19T10:00:00",
        "total_pct": 45.2, "ncores": 10, "interval_s": 60.0,
        "by_category": {"rag": 12.0, "ollama": 30.0, "sqlite-vec": 1.0,
                        "whatsapp": 2.2},
        "top": [{"name": "ollama runner", "pct": 25.0, "cat": "ollama"}],
    }
    web_server._cpu_persist(sample)


def _call_memory(base: Path) -> None:
    sample = {
        "ts": "2026-04-19T10:00:00",
        "total_mb": 1234.5,
        "by_category": {"rag": 300.0, "ollama": 800.0, "sqlite-vec": 20.0,
                        "whatsapp": 50.0},
        "top": [{"name": "ollama runner", "mb": 700.0, "cat": "ollama"}],
        "vm": {"free_mb": 4000, "wired_mb": 2000},
    }
    web_server._memory_persist(sample)


def _call_system_memory(base: Path) -> None:
    """system_memory_metrics has no live JSONL writer in rag.py/web/server.py.
    Covered via direct _sql_append_event to ensure the table is reachable
    from the runtime helpers — migration script (T2) is its only current
    user, but T6 will wire a writer. Kept in this parametrized battery as
    smoke for the table plumbing so a rename would fail-fast here."""
    entry = {
        "ts": "2026-04-19T10:00:00",
        "total_mb": 9000.0,
        "by_category": {"python": 1000.0, "browser": 2000.0,
                        "ollama": 3000.0, "node": 500.0,
                        "claude": 1500.0, "other": 1000.0},
        "top": [{"name": "Chrome", "mb": 2000.0, "cat": "browser"}],
        "vm": {"free_mb": 5000.0},
    }
    with rag._ragvec_state_conn() as conn:
        rag._sql_append_event(conn, "system_memory_metrics",
                               rag._map_memory_row(entry))


@pytest.mark.parametrize("label,call,table", [
    ("wa_tasks",       _call_wa_tasks_log, "rag_wa_tasks"),
    ("archive",        _call_archive,      "rag_archive_log"),
    ("filing",         _call_filing,       "rag_filing_log"),
    ("eval",           _call_eval,         "rag_eval_runs"),
    ("surface",        _call_surface,      "rag_surface_log"),
    ("proactive",      _call_proactive,    "rag_proactive_log"),
    ("cpu",            _call_cpu,          "rag_cpu_metrics"),
    ("memory",         _call_memory,       "rag_memory_metrics"),
    ("system_memory",  _call_system_memory, "system_memory_metrics"),
])
def test_metrics_writers_flag_on(sql_env, label, call, table):
    call(sql_env)
    conn = _open_db(sql_env)
    try:
        count = _count(conn, table)
        # surface writes 1 run row + 1 pair row (2 total); all others = 1
        expected = 2 if label == "surface" else 1
        assert count == expected, f"{label}: expected {expected} row(s) in {table}, got {count}"
    finally:
        conn.close()
