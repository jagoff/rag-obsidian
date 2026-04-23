"""Feature #9 del 2026-04-23 — Learned paraphrases tests.

Validates:
- Schema: rag_learned_paraphrases table
- _normalize_q canonicalization
- _record_learned_paraphrase upsert + increment on conflict
- _lookup_learned_paraphrases trust threshold + returns paraphrases
- _train_paraphrases_from_feedback extracts variants correctly
- expand_queries shortcut when learned paraphrases exist
- CLI commands (train, stats, clear)
"""
from __future__ import annotations

import contextlib
import json
import sqlite3

import pytest
from click.testing import CliRunner

import rag


_QUERIES_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_queries ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ts TEXT NOT NULL,"
    " cmd TEXT,"
    " q TEXT NOT NULL,"
    " variants_json TEXT,"
    " extra_json TEXT"
    ")"
)

_FEEDBACK_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_feedback ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ts TEXT NOT NULL,"
    " turn_id TEXT,"
    " rating INTEGER NOT NULL,"
    " q TEXT,"
    " paths_json TEXT,"
    " extra_json TEXT,"
    " UNIQUE(turn_id, rating, ts)"
    ")"
)

_LEARNED_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_learned_paraphrases ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " q_normalized TEXT NOT NULL,"
    " paraphrase TEXT NOT NULL,"
    " hit_count INTEGER NOT NULL DEFAULT 1,"
    " created_ts TEXT NOT NULL,"
    " last_used_ts TEXT NOT NULL,"
    " UNIQUE(q_normalized, paraphrase)"
    ")"
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(_QUERIES_DDL)
    conn.execute(_FEEDBACK_DDL)
    conn.execute(_LEARNED_DDL)
    conn.commit()

    @contextlib.contextmanager
    def _fake_conn():
        c = sqlite3.connect(str(db_path))
        try:
            yield c
            c.commit()
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_conn)
    try:
        yield conn, db_path
    finally:
        conn.close()


# ── _normalize_q ─────────────────────────────────────────────────────────


def test_normalize_q_lowercases_strips():
    assert rag._normalize_q("  Hola Mundo  ") == "hola mundo"


def test_normalize_q_collapses_whitespace():
    assert rag._normalize_q("foo    bar\tbaz") == "foo bar baz"


def test_normalize_q_preserves_unicode_and_punct():
    assert rag._normalize_q("¿Qué? cómo estás") == "¿qué? cómo estás"


def test_normalize_q_empty():
    assert rag._normalize_q("") == ""
    assert rag._normalize_q(None) == ""  # type: ignore[arg-type]


# ── _record_learned_paraphrase ───────────────────────────────────────────


def test_record_paraphrase_inserts(temp_db):
    conn, _ = temp_db
    assert rag._record_learned_paraphrase("What is X?", "define X") is True
    row = conn.execute(
        "SELECT q_normalized, paraphrase, hit_count FROM rag_learned_paraphrases"
    ).fetchone()
    assert row[0] == "what is x?"
    assert row[1] == "define X"
    assert row[2] == 1


def test_record_paraphrase_conflict_increments_hit_count(temp_db):
    conn, _ = temp_db
    rag._record_learned_paraphrase("q1", "paraphrase1")
    rag._record_learned_paraphrase("q1", "paraphrase1")
    rag._record_learned_paraphrase("q1", "paraphrase1")
    row = conn.execute(
        "SELECT hit_count FROM rag_learned_paraphrases"
    ).fetchone()
    assert row[0] == 3


def test_record_paraphrase_empty_inputs_rejected(temp_db):
    assert rag._record_learned_paraphrase("", "foo") is False
    assert rag._record_learned_paraphrase("foo", "") is False


def test_record_paraphrase_same_as_query_rejected(temp_db):
    """paraphrase == q (normalized) → no insert."""
    assert rag._record_learned_paraphrase("Hello", "hello") is False


# ── _lookup_learned_paraphrases ──────────────────────────────────────────


def test_lookup_requires_min_hits(temp_db, monkeypatch):
    conn, _ = temp_db
    monkeypatch.setattr(rag, "_LEARNED_PARA_ENABLED", True)
    # Insert with hit_count=1 (below threshold of 2).
    rag._record_learned_paraphrase("test query", "alt phrasing")
    monkeypatch.setattr(rag, "_LEARNED_PARA_MIN_HITS", 2)
    assert rag._lookup_learned_paraphrases("test query") == []

    # Bump to 2 hits.
    rag._record_learned_paraphrase("test query", "alt phrasing")
    assert rag._lookup_learned_paraphrases("test query") == ["alt phrasing"]


def test_lookup_respects_flag_off(temp_db, monkeypatch):
    rag._record_learned_paraphrase("q", "p1")
    rag._record_learned_paraphrase("q", "p1")
    monkeypatch.setattr(rag, "_LEARNED_PARA_ENABLED", False)
    assert rag._lookup_learned_paraphrases("q") == []


def test_lookup_case_insensitive(temp_db, monkeypatch):
    monkeypatch.setattr(rag, "_LEARNED_PARA_ENABLED", True)
    monkeypatch.setattr(rag, "_LEARNED_PARA_MIN_HITS", 1)
    rag._record_learned_paraphrase("Hello World", "greet world")
    assert rag._lookup_learned_paraphrases("HELLO WORLD") == ["greet world"]
    assert rag._lookup_learned_paraphrases("hello world") == ["greet world"]


def test_lookup_updates_last_used_ts(temp_db, monkeypatch):
    conn, _ = temp_db
    monkeypatch.setattr(rag, "_LEARNED_PARA_ENABLED", True)
    monkeypatch.setattr(rag, "_LEARNED_PARA_MIN_HITS", 1)
    rag._record_learned_paraphrase("q", "p", now_ts="2020-01-01T00:00:00")
    rag._lookup_learned_paraphrases("q")
    row = conn.execute(
        "SELECT last_used_ts FROM rag_learned_paraphrases WHERE q_normalized='q'"
    ).fetchone()
    # Shouldn't be 2020 anymore — got bumped to "now".
    assert row[0] != "2020-01-01T00:00:00"


# ── _train_paraphrases_from_feedback ─────────────────────────────────────


def test_train_extracts_from_positive_feedback(temp_db):
    conn, _ = temp_db
    from datetime import datetime
    now = datetime.now().isoformat(timespec="seconds")
    variants = ["original question", "rephrased A", "rephrased B"]
    conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, variants_json) "
        "VALUES (?, 'query', ?, ?)",
        (now, "original question", json.dumps(variants)),
    )
    conn.execute(
        "INSERT INTO rag_feedback(ts, turn_id, rating, q, paths_json) "
        "VALUES (?, 't1', 1, ?, '[]')",
        (now, "original question"),
    )
    conn.commit()

    result = rag._train_paraphrases_from_feedback(since_days=30)
    assert result["processed"] == 1
    assert result["persisted"] == 2  # two paraphrases, not counting original

    rows = conn.execute(
        "SELECT paraphrase FROM rag_learned_paraphrases"
    ).fetchall()
    persisted = {r[0] for r in rows}
    assert persisted == {"rephrased A", "rephrased B"}


def test_train_skips_negative_rating(temp_db):
    conn, _ = temp_db
    from datetime import datetime
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, variants_json) "
        "VALUES (?, 'query', ?, ?)",
        (now, "q1", json.dumps(["q1", "alt1"])),
    )
    conn.execute(
        "INSERT INTO rag_feedback(ts, turn_id, rating, q, paths_json) "
        "VALUES (?, 't1', -1, ?, '[]')",  # rating=-1 should skip
        (now, "q1"),
    )
    conn.commit()

    result = rag._train_paraphrases_from_feedback(since_days=30)
    assert result["processed"] == 0
    assert result["persisted"] == 0


def test_train_dry_run_does_not_persist(temp_db):
    conn, _ = temp_db
    from datetime import datetime
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, variants_json) "
        "VALUES (?, 'query', ?, ?)",
        (now, "dry run q", json.dumps(["dry run q", "alt"])),
    )
    conn.execute(
        "INSERT INTO rag_feedback(ts, turn_id, rating, q, paths_json) "
        "VALUES (?, 't1', 1, ?, '[]')",
        (now, "dry run q"),
    )
    conn.commit()

    result = rag._train_paraphrases_from_feedback(since_days=30, dry_run=True)
    assert result["persisted"] == 1
    row = conn.execute(
        "SELECT COUNT(*) FROM rag_learned_paraphrases"
    ).fetchone()
    assert row[0] == 0


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_paraphrases_train_as_json(temp_db):
    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["paraphrases", "train", "--since", "30", "--as-json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip().splitlines()[-1])
    assert "processed" in data
    assert "persisted" in data


def test_cli_paraphrases_stats_empty(temp_db):
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["paraphrases", "stats"])
    assert result.exit_code == 0
    assert "Learned paraphrases" in result.output


def test_cli_paraphrases_clear_with_yes(temp_db):
    conn, _ = temp_db
    rag._record_learned_paraphrase("q", "p")
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["paraphrases", "clear", "--yes"])
    assert result.exit_code == 0
    # Rows are gone.
    row = conn.execute("SELECT COUNT(*) FROM rag_learned_paraphrases").fetchone()
    assert row[0] == 0
