"""T4: reader-swap tests.

Each flag-gated reader in rag.py must:
  - read from the matching rag_* table when RAG_STATE_SQL=1 AND SQL has data
  - fall through to JSONL when the flag is on but SQL is empty (cutover bridge)
  - run the pure JSONL path when the flag is off (unchanged behavior)
  - fall through to JSONL if the SQL read raises (cutover fail-safe)
  - invalidate cache when MAX(ts) advances

Feature-flag + DB path are patched per test via monkeypatch. `_ragvec_state_conn`
reads `rag.DB_PATH` and constructs `DB_PATH / rag._TELEMETRY_DB_FILENAME`, so pointing
DB_PATH at tmp_path is sufficient to isolate each test.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

import rag


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _open_db(tmp_path: Path) -> sqlite3.Connection:
    """Open the on-disk telemetry.db with T1 DDL applied."""
    db = tmp_path / rag._TELEMETRY_DB_FILENAME
    conn = sqlite3.connect(str(db), isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_schema_version ("
        " table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
    )
    rag._ensure_telemetry_tables(conn)
    return conn


def _redirect_jsonl_paths(monkeypatch, base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "BEHAVIOR_LOG_PATH", base / "behavior.jsonl")
    monkeypatch.setattr(rag, "FEEDBACK_PATH", base / "feedback.jsonl")
    monkeypatch.setattr(rag, "FEEDBACK_GOLDEN_PATH",
                         base / "feedback_golden.json")
    monkeypatch.setattr(rag, "BRIEF_STATE_PATH", base / "brief_state.jsonl")
    monkeypatch.setattr(rag, "BRIEF_WRITTEN_PATH", base / "brief_written.jsonl")
    monkeypatch.setattr(rag, "AMBIENT_STATE_PATH", base / "ambient_state.jsonl")
    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG",
                         base / "sql_state_errors.jsonl")


def _reset_caches(monkeypatch) -> None:
    """Clear module-level memo state so each test starts fresh."""
    monkeypatch.setattr(rag, "_behavior_priors_cache", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key_sql", None)
    monkeypatch.setattr(rag, "_feedback_golden_memo", None)
    monkeypatch.setattr(rag, "_feedback_golden_source_ts_sql", None)


@pytest.fixture
def sql_env(tmp_path, monkeypatch):
    """Flag ON + DB_PATH + JSONL paths redirected."""
    monkeypatch.setattr(rag, "RAG_STATE_SQL", True)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    _redirect_jsonl_paths(monkeypatch, tmp_path / "jsonl")
    _reset_caches(monkeypatch)
    yield tmp_path


@pytest.fixture
def jsonl_env(tmp_path, monkeypatch):
    """Legacy flag-OFF fixture — kept for test-signature compatibility.
    Post-T10 the flag is inert, so this is effectively an alias for sql_env
    pointing at a tmp DB. Tests that relied on JSONL-only behavior have been
    rewritten to assert the new SQL-only semantics."""
    monkeypatch.setattr(rag, "RAG_STATE_SQL", True)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    _redirect_jsonl_paths(monkeypatch, tmp_path / "jsonl")
    _reset_caches(monkeypatch)
    yield tmp_path


# ── Helpers to seed SQL data ─────────────────────────────────────────────────

def _seed_rag_behavior(tmp_path: Path, events: list[dict]) -> None:
    conn = _open_db(tmp_path)
    try:
        for ev in events:
            rag._sql_append_event(conn, "rag_behavior",
                                    rag._map_behavior_row(ev))
    finally:
        conn.close()


def _seed_rag_feedback(tmp_path: Path, events: list[dict]) -> None:
    conn = _open_db(tmp_path)
    try:
        for ev in events:
            rag._sql_append_event(conn, "rag_feedback",
                                    rag._map_feedback_row(ev))
    finally:
        conn.close()


def _seed_rag_feedback_golden(tmp_path: Path, positives: list[dict],
                                negatives: list[dict], source_ts: str) -> None:
    """Seed the SQL golden table directly (bypassing rebuild)."""
    conn = _open_db(tmp_path)
    try:
        rag._write_feedback_golden_sql(
            conn, {"positives": positives, "negatives": negatives}, source_ts,
        )
    finally:
        conn.close()


# ── behavior priors ──────────────────────────────────────────────────────────

def test_behavior_priors_reads_sql_when_flag_on(sql_env):
    _seed_rag_behavior(sql_env, [
        {"ts": "2026-04-19T10:00:00", "source": "cli", "event": "open",
         "path": "01-Projects/a.md"},
        {"ts": "2026-04-19T11:00:00", "source": "cli", "event": "open",
         "path": "01-Projects/a.md"},
        {"ts": "2026-04-19T12:00:00", "source": "cli",
         "event": "negative_implicit", "path": "02-Areas/b.md"},
    ])
    priors = rag._load_behavior_priors()
    # 2 positive events on a.md → 3/12 CTR (Laplace)
    assert priors["click_prior"]["01-Projects/a.md"] == pytest.approx(3 / 12)
    # 1 negative on b.md → 1/11
    assert priors["click_prior"]["02-Areas/b.md"] == pytest.approx(1 / 11)
    # Folder-level: 01-Projects has 2 positive clicks over 2 impressions
    assert priors["click_prior_folder"]["01-Projects"] == pytest.approx(3 / 12)
    assert priors["n_events"] == 3
    assert priors["hash"].startswith("sql:")


def test_behavior_priors_returns_empty_snapshot_when_sql_empty(sql_env):
    """Post-T10: SQL-only. Empty rag_behavior → empty priors snapshot.
    JSONL on disk is ignored (no fallback path)."""
    rag.BEHAVIOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rag.BEHAVIOR_LOG_PATH.write_text(
        json.dumps({"ts": "2026-04-19T10:00:00", "source": "cli",
                     "event": "open", "path": "01-Projects/legacy.md"}) + "\n",
        encoding="utf-8",
    )
    priors = rag._load_behavior_priors()
    assert priors["click_prior"] == {}
    assert priors["n_events"] == 0
    assert priors["hash"] == "sql:empty"


def test_feedback_golden_rebuilt_when_source_ts_advances(sql_env):
    """Seed old feedback + stale meta → newer rag_feedback → rebuild triggered."""
    # Seed a feedback row (becomes source_ts).
    _seed_rag_feedback(sql_env, [
        {"ts": "2026-04-01T10:00:00", "turn_id": "t1", "rating": 1,
         "q": "old query", "paths": ["01-Projects/a.md"]},
    ])
    # Seed golden with old meta source_ts (simulate stale build).
    _seed_rag_feedback_golden(
        sql_env,
        positives=[{"q": "old query", "emb": [0.1, 0.2, 0.3],
                     "paths": ["01-Projects/a.md"]}],
        negatives=[],
        source_ts="2026-03-01T00:00:00",  # older than max feedback ts
    )
    # Patch embed to return deterministic shape.
    with patch.object(rag, "embed", return_value=[[0.9, 0.1, 0.0]]):
        golden = rag.load_feedback_golden()
    assert len(golden["positives"]) == 1
    # Meta should now match newer source_ts.
    conn = _open_db(sql_env)
    try:
        meta_ts = conn.execute(
            "SELECT v FROM rag_feedback_golden_meta WHERE k='last_built_source_ts'"
        ).fetchone()[0]
        assert meta_ts == "2026-04-01T10:00:00"
        # Built_at moved forward.
        built_row = conn.execute(
            "SELECT built_at FROM rag_feedback_golden WHERE path=?",
            ("01-Projects/a.md",),
        ).fetchone()
        assert built_row is not None
    finally:
        conn.close()


def test_feedback_golden_empty_when_no_feedback_rows(sql_env):
    """Post-T10: SQL-only. With no rag_feedback rows + no golden rows, even if
    a legacy feedback_golden.json sits on disk, the loader returns an empty
    snapshot (no JSON bridge)."""
    rag.FEEDBACK_GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    rag.FEEDBACK_GOLDEN_PATH.write_text(json.dumps({
        "positives": [
            {"q": "how to rag", "emb": [0.2, 0.3, 0.5],
             "paths": ["01-Projects/bridged.md"]},
        ],
        "negatives": [],
    }), encoding="utf-8")
    golden = rag.load_feedback_golden()
    assert golden == {"positives": [], "negatives": []}


def test_behavior_augmented_cases_14d_window(sql_env):
    """Events older than 14 days must be excluded (ts window filter)."""
    import datetime as _dt
    now = _dt.datetime.now()
    fresh = (now - _dt.timedelta(days=5)).isoformat(timespec="seconds")
    stale = (now - _dt.timedelta(days=30)).isoformat(timespec="seconds")
    _seed_rag_behavior(sql_env, [
        {"ts": fresh, "source": "cli", "event": "open",
         "path": "fresh.md", "query": "recent search"},
        {"ts": stale, "source": "cli", "event": "open",
         "path": "stale.md", "query": "old search"},
    ])
    cases = rag._behavior_augmented_cases(days=14)
    questions = [c["question"] for c in cases]
    assert "recent search" in questions
    assert "old search" not in questions


def test_brief_state_dedup_sql_hit(sql_env):
    """Seed a pair in rag_brief_state → seen returns True."""
    rag._brief_state_record("04-Archive/99-obsidian-system/99-Claude/reviews/2026-04-19.md", "01-Projects/a.md")
    assert rag._brief_state_seen("04-Archive/99-obsidian-system/99-Claude/reviews/2026-04-19.md",
                                   "01-Projects/a.md") is True
    assert rag._brief_state_seen("04-Archive/99-obsidian-system/99-Claude/reviews/2026-04-19.md",
                                   "01-Projects/NEW.md") is False


def test_brief_state_dedup_sql_only(sql_env):
    """Post-T10: only SQL is consulted. A legacy brief_state.jsonl with the
    pair but no SQL row → seen returns False (no JSONL fallback)."""
    rag.BRIEF_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = "04-Archive/99-obsidian-system/99-Claude/reviews/2026-04-18.md\x0001-Projects/legacy.md"
    rag.BRIEF_STATE_PATH.write_text(
        json.dumps({"ts": "2026-04-18T07:00:00", "key": key}) + "\n",
        encoding="utf-8",
    )
    assert rag._brief_state_seen("04-Archive/99-obsidian-system/99-Claude/reviews/2026-04-18.md",
                                   "01-Projects/legacy.md") is False


def test_ambient_state_lookup_sql_returns_analyzed_at(sql_env):
    """Upsert a row → _ambient_should_skip True inside dedup window."""
    conn = _open_db(sql_env)
    try:
        import time as _time
        rag._sql_upsert(conn, "rag_ambient_state",
                          rag._map_ambient_state_row(
                              "00-Inbox/a.md", "abc123", _time.time(),
                              {"summary": "test"}),
                          ("path",))
    finally:
        conn.close()
    assert rag._ambient_should_skip("00-Inbox/a.md", "abc123") is True
    # Different hash → not a hit.
    assert rag._ambient_should_skip("00-Inbox/a.md", "different") is False


def test_reader_cache_invalidates_on_new_max_ts(sql_env):
    """Load priors (cached) → insert newer row → next load returns updated."""
    _seed_rag_behavior(sql_env, [
        {"ts": "2026-04-19T08:00:00", "source": "cli", "event": "open",
         "path": "first.md"},
    ])
    first = rag._load_behavior_priors()
    assert "first.md" in first["click_prior"]
    first_key = rag._behavior_priors_cache_key_sql
    # Insert newer event.
    _seed_rag_behavior(sql_env, [
        {"ts": "2026-04-19T09:00:00", "source": "cli", "event": "open",
         "path": "second.md"},
    ])
    second = rag._load_behavior_priors()
    assert "second.md" in second["click_prior"]
    assert rag._behavior_priors_cache_key_sql != first_key


def test_reader_sql_exception_returns_empty(sql_env):
    """Post-T10: when _sql_query_window raises, _behavior_augmented_cases
    logs the error and returns an empty list (no JSONL fallback)."""
    import datetime as _dt
    fresh = (_dt.datetime.now() - _dt.timedelta(days=3)).isoformat(
        timespec="seconds")
    rag.BEHAVIOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rag.BEHAVIOR_LOG_PATH.write_text(
        json.dumps({"ts": fresh, "event": "open", "path": "jsonl.md",
                     "query": "should not be read"}) + "\n",
        encoding="utf-8",
    )

    with patch.object(rag, "_sql_query_window",
                       side_effect=RuntimeError("boom")):
        cases = rag._behavior_augmented_cases(days=14)
    assert cases == []
    # Error log should have a record.
    log_txt = rag._SQL_STATE_ERROR_LOG.read_text(encoding="utf-8")
    assert "behavior_augmented_cases_sql_read_failed" in log_txt


# ── Extra coverage: contradictions + ambient fallback + flag-off ambient ─────

def test_contradictions_sql_read(sql_env):
    import datetime as _dt
    now = _dt.datetime.now()
    fresh = (now - _dt.timedelta(days=2)).isoformat(timespec="seconds")
    conn = _open_db(sql_env)
    try:
        rag._sql_append_event(conn, "rag_contradictions",
                                rag._map_contradiction_row({
                                    "ts": fresh,
                                    "subject_path": "01-Projects/s.md",
                                    "contradicts": [{"path": "02-Areas/x.md",
                                                      "note": "n", "why": "w"}],
                                    "helper_raw": "r",
                                }))
    finally:
        conn.close()
    out = rag._pendientes_recent_contradictions(
        rag.CONTRADICTION_LOG_PATH, now, days=14, max_items=5)
    assert len(out) == 1
    assert out[0]["subject_path"] == "01-Projects/s.md"


def test_ambient_should_skip_sql_only(sql_env):
    """Post-T10: SQL-only. A legacy ambient_state.jsonl with the entry but no
    rag_ambient_state row → _ambient_should_skip returns False."""
    import time as _time
    rag.AMBIENT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rag.AMBIENT_STATE_PATH.write_text(
        json.dumps({"path": "00-Inbox/legacy.md", "hash": "xyz",
                     "analyzed_at": _time.time()}) + "\n",
        encoding="utf-8",
    )
    assert rag._ambient_should_skip("00-Inbox/legacy.md", "xyz") is False
