"""Unit tests for `rag feedback {status,backfill,harvest}` subgroup.

All tests patch `rag._ragvec_state_conn` to point at a tmp DB so they never
touch the user's real telemetry.db. No real retrieve/ollama/network.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

import rag


_FEEDBACK_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_feedback ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ts TEXT NOT NULL,"
    " turn_id TEXT,"
    " rating INTEGER NOT NULL,"
    " q TEXT,"
    " scope TEXT,"
    " paths_json TEXT,"
    " extra_json TEXT,"
    " UNIQUE(turn_id, rating, ts)"
    ")"
)

_QUERIES_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_queries ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ts TEXT NOT NULL,"
    " cmd TEXT,"
    " q TEXT NOT NULL,"
    " session TEXT,"
    " paths_json TEXT,"
    " scores_json TEXT,"
    " top_score REAL,"
    " extra_json TEXT"
    ")"
)

# Tables the golden cache invalidation paths touch — create so DELETE
# doesn't blow up in a temp DB.
_GOLDEN_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_feedback_golden ("
    " q TEXT, path TEXT, weight REAL)",
    "CREATE TABLE IF NOT EXISTS rag_feedback_golden_meta ("
    " k TEXT PRIMARY KEY, v TEXT)",
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Temp telemetry.db + patched `_ragvec_state_conn`.

    Yields (conn, db_path). Conn stays open across helper calls because the
    patched context-manager reopens on each invocation — the `conn` is for
    the test's own inserts only.
    """
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(_FEEDBACK_DDL)
    conn.execute(_QUERIES_DDL)
    for ddl in _GOLDEN_DDL:
        conn.execute(ddl)
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
    # Reset in-process caches so invalidation paths don't blow up.
    monkeypatch.setattr(rag, "_feedback_golden_memo", None, raising=False)
    monkeypatch.setattr(rag, "_feedback_golden_source_ts_sql", None, raising=False)
    try:
        yield conn, db_path
    finally:
        conn.close()


def _insert_fb(conn, *, ts, rating, q, paths=None, extra=None, turn_id="t"):
    conn.execute(
        "INSERT INTO rag_feedback(ts, turn_id, rating, q, paths_json, extra_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, turn_id, rating, q,
         json.dumps(paths or []),
         json.dumps(extra) if extra is not None else None),
    )
    conn.commit()


def _insert_q(conn, *, ts, q, top_score, paths=None, scores=None, cmd="query"):
    conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, paths_json, scores_json, top_score) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, cmd, q,
         json.dumps(paths or []),
         json.dumps(scores or [0.0] * len(paths or [])),
         top_score),
    )
    conn.commit()


# ── _feedback_stats ──────────────────────────────────────────────────────

def test_stats_empty_db_returns_zeros(temp_db):
    s = rag._feedback_stats()
    assert s == {
        "total": 0, "pos": 0, "neg": 0,
        "with_cp": 0, "pos_no_cp": 0, "neg_no_cp": 0,
    }


def test_stats_counts_mixed_correctly(temp_db):
    conn, _ = temp_db
    # 3 positives: 1 with cp, 2 without
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="q1",
               extra={"corrective_path": "a.md"}, turn_id="t1")
    _insert_fb(conn, ts="2026-04-22T10:01", rating=1, q="q2", turn_id="t2")
    _insert_fb(conn, ts="2026-04-22T10:02", rating=1, q="q3", turn_id="t3")
    # 2 negatives: 0 with cp (negatives typically don't have cp)
    _insert_fb(conn, ts="2026-04-22T10:03", rating=-1, q="q4", turn_id="t4")
    _insert_fb(conn, ts="2026-04-22T10:04", rating=-1, q="q5",
               extra={"reason": "bad"}, turn_id="t5")
    # Empty-string cp (treated as no-cp)
    _insert_fb(conn, ts="2026-04-22T10:05", rating=1, q="q6",
               extra={"corrective_path": ""}, turn_id="t6")

    s = rag._feedback_stats()
    assert s["total"] == 6
    assert s["pos"] == 4
    assert s["neg"] == 2
    assert s["with_cp"] == 1
    assert s["pos_no_cp"] == 3
    assert s["neg_no_cp"] == 2


# ── _set_feedback_corrective_path ────────────────────────────────────────

def test_set_cp_writes_and_preserves_other_extras(temp_db):
    conn, _ = temp_db
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="q1",
               extra={"reason": "ya-tenia-reason"}, turn_id="t1")

    ok = rag._set_feedback_corrective_path(1, "golden/path.md")
    assert ok is True

    row = conn.execute(
        "SELECT extra_json FROM rag_feedback WHERE id=1"
    ).fetchone()
    extra = json.loads(row[0])
    # Both original + new field preserved
    assert extra["corrective_path"] == "golden/path.md"
    assert extra["reason"] == "ya-tenia-reason"


def test_set_cp_creates_extra_json_when_null(temp_db):
    conn, _ = temp_db
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="q1",
               extra=None, turn_id="t1")

    ok = rag._set_feedback_corrective_path(1, "nuevo.md")
    assert ok is True

    row = conn.execute(
        "SELECT extra_json FROM rag_feedback WHERE id=1"
    ).fetchone()
    extra = json.loads(row[0])
    assert extra == {"corrective_path": "nuevo.md"}


def test_set_cp_overwrites_existing_cp(temp_db):
    conn, _ = temp_db
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="q1",
               extra={"corrective_path": "viejo.md"}, turn_id="t1")

    ok = rag._set_feedback_corrective_path(1, "nuevo.md")
    assert ok is True

    row = conn.execute(
        "SELECT extra_json FROM rag_feedback WHERE id=1"
    ).fetchone()
    extra = json.loads(row[0])
    assert extra["corrective_path"] == "nuevo.md"


# ── _feedback_rows_without_cp ────────────────────────────────────────────

def test_rows_without_cp_filters_by_rating(temp_db):
    conn, _ = temp_db
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="qp1",
               paths=["a.md"], turn_id="tp1")
    _insert_fb(conn, ts="2026-04-22T10:01", rating=-1, q="qn1",
               paths=["b.md"], turn_id="tn1")

    pos_only = rag._feedback_rows_without_cp("pos", 365, 10)
    assert len(pos_only) == 1
    assert pos_only[0]["q"] == "qp1"

    neg_only = rag._feedback_rows_without_cp("neg", 365, 10)
    assert len(neg_only) == 1
    assert neg_only[0]["q"] == "qn1"

    both = rag._feedback_rows_without_cp("both", 365, 10)
    assert len(both) == 2


def test_rows_without_cp_excludes_rows_with_cp(temp_db):
    conn, _ = temp_db
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="q_skip",
               paths=["a.md"], extra={"corrective_path": "a.md"}, turn_id="t_skip")
    _insert_fb(conn, ts="2026-04-22T10:01", rating=1, q="q_keep",
               paths=["b.md"], turn_id="t_keep")

    rows = rag._feedback_rows_without_cp("both", 365, 10)
    assert len(rows) == 1
    assert rows[0]["q"] == "q_keep"


def test_rows_without_cp_orders_by_ts_desc(temp_db):
    conn, _ = temp_db
    _insert_fb(conn, ts="2026-04-20T10:00", rating=1, q="old",
               paths=["a.md"], turn_id="t_old")
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="new",
               paths=["b.md"], turn_id="t_new")

    rows = rag._feedback_rows_without_cp("both", 365, 10)
    assert [r["q"] for r in rows] == ["new", "old"]


# ── _harvest_candidates ──────────────────────────────────────────────────

def test_harvest_candidates_returns_low_confidence_unrated(temp_db):
    conn, _ = temp_db
    # Recent, low confidence, never rated
    _insert_q(conn, ts="2026-04-22T10:00", q="pick me",
              top_score=0.05, paths=["a.md", "b.md"])
    # High confidence — should be excluded
    _insert_q(conn, ts="2026-04-22T10:01", q="skip me high conf",
              top_score=0.9, paths=["c.md"])
    # Low confidence but already rated — should be excluded
    _insert_q(conn, ts="2026-04-22T10:02", q="already rated",
              top_score=0.05, paths=["d.md"])
    _insert_fb(conn, ts="2026-04-22T10:02:01", rating=1,
               q="already rated", paths=["d.md"], turn_id="t_rated")
    # Query too short — excluded by length filter
    _insert_q(conn, ts="2026-04-22T10:03", q="x",
              top_score=0.01, paths=["e.md"])
    # Blacklisted smoke query
    _insert_q(conn, ts="2026-04-22T10:04", q="test",
              top_score=0.01, paths=["f.md"])

    cands = rag._harvest_candidates(since_days=30, confidence_below=0.2, limit=10)
    assert [c["q"] for c in cands] == ["pick me"]
    assert cands[0]["paths"] == ["a.md", "b.md"]


def test_harvest_candidates_empty_when_no_queries(temp_db):
    cands = rag._harvest_candidates(since_days=30, confidence_below=0.2, limit=10)
    assert cands == []


# ── _feedback_insert_harvested ───────────────────────────────────────────

def test_insert_harvested_flattens_extra_json_fields(temp_db):
    conn, _ = temp_db
    ok = rag._feedback_insert_harvested(
        q="harvested q", rating=1,
        paths=["a.md"], original_query_id=42,
        corrective_path="a.md",
    )
    assert ok is True

    row = conn.execute(
        "SELECT rating, q, paths_json, extra_json FROM rag_feedback"
    ).fetchone()
    assert row[0] == 1
    assert row[1] == "harvested q"
    assert json.loads(row[2]) == ["a.md"]
    extra = json.loads(row[3])
    # Flat, not nested under 'extra_json_extras'
    assert extra["source"] == "harvester"
    assert extra["original_query_id"] == 42
    assert extra["corrective_path"] == "a.md"
    assert extra["reason"] == "corrective"


def test_insert_harvested_without_corrective_omits_reason(temp_db):
    conn, _ = temp_db
    ok = rag._feedback_insert_harvested(
        q="plain neg", rating=-1,
        paths=["bad1.md", "bad2.md"], original_query_id=7,
    )
    assert ok is True

    row = conn.execute(
        "SELECT extra_json FROM rag_feedback"
    ).fetchone()
    extra = json.loads(row[0])
    assert "reason" not in extra
    assert "corrective_path" not in extra
    assert extra["source"] == "harvester"


# ── feedback status CLI ─────────────────────────────────────────────────

def test_cli_status_renders_breakdown(temp_db):
    conn, _ = temp_db
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="q1",
               extra={"corrective_path": "a.md"}, turn_id="t1")
    _insert_fb(conn, ts="2026-04-22T10:01", rating=1, q="q2", turn_id="t2")
    _insert_fb(conn, ts="2026-04-22T10:02", rating=-1, q="q3", turn_id="t3")

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["feedback", "status"])
    assert result.exit_code == 0, result.output
    assert "Total:" in result.output
    # Strip Rich markup chars — they may survive in Click's no-TTY renderer
    assert "+1: 2" in result.output
    assert "−1: 1" in result.output
    assert "1" in result.output and "/ 20" in result.output


# ── feedback backfill CLI ───────────────────────────────────────────────

def test_cli_backfill_applies_numeric_choice(temp_db):
    conn, _ = temp_db
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="backfill me",
               paths=["a.md", "b.md", "c.md"], turn_id="tb1")

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["feedback", "backfill", "--limit", "5",
                  "--rating", "pos", "--since", "365"],
        input="2\nq\n",  # choose path #2, then quit
    )
    assert result.exit_code == 0, result.output
    row = conn.execute(
        "SELECT extra_json FROM rag_feedback"
    ).fetchone()
    extra = json.loads(row[0])
    assert extra["corrective_path"] == "b.md"


def test_cli_backfill_accepts_free_text_path(temp_db):
    conn, _ = temp_db
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="free text",
               paths=["a.md"], turn_id="tf1")

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["feedback", "backfill", "--limit", "5",
                  "--rating", "pos", "--since", "365"],
        input="custom/path.md\n",
    )
    assert result.exit_code == 0, result.output
    row = conn.execute(
        "SELECT extra_json FROM rag_feedback"
    ).fetchone()
    extra = json.loads(row[0])
    assert extra["corrective_path"] == "custom/path.md"


def test_cli_backfill_skip_does_not_write(temp_db):
    conn, _ = temp_db
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="skip me",
               paths=["a.md"], turn_id="ts1")

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["feedback", "backfill", "--limit", "5",
                  "--rating", "pos", "--since", "365"],
        input="s\n",
    )
    assert result.exit_code == 0, result.output
    row = conn.execute(
        "SELECT extra_json FROM rag_feedback WHERE id=1"
    ).fetchone()
    # Still null (or empty dict)
    assert row[0] is None or json.loads(row[0]).get("corrective_path", "") == ""


def test_cli_backfill_quit_stops_immediately(temp_db):
    conn, _ = temp_db
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="first",
               paths=["a.md"], turn_id="tq1")
    _insert_fb(conn, ts="2026-04-22T10:01", rating=1, q="second",
               paths=["b.md"], turn_id="tq2")

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["feedback", "backfill", "--limit", "5",
                  "--rating", "pos", "--since", "365"],
        input="q\n",
    )
    assert result.exit_code == 0, result.output
    # Neither turn got a cp — quit was immediate
    rows = conn.execute(
        "SELECT extra_json FROM rag_feedback ORDER BY id"
    ).fetchall()
    for r in rows:
        if r[0]:
            assert "corrective_path" not in json.loads(r[0])


def test_cli_backfill_empty_is_noop(temp_db):
    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["feedback", "backfill", "--limit", "5",
                  "--rating", "pos", "--since", "365"],
    )
    assert result.exit_code == 0
    assert "Sin rows" in result.output


# ── feedback harvest CLI ────────────────────────────────────────────────

def test_cli_harvest_positive_choice_inserts_narrow_positive(temp_db):
    conn, _ = temp_db
    _insert_q(conn, ts="2026-04-22T10:00", q="harvest me",
              top_score=0.05, paths=["golden.md", "other.md", "x.md"])

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["feedback", "harvest", "--since", "365",
                  "--confidence-below", "0.2", "--limit", "5"],
        input="1\n",
    )
    assert result.exit_code == 0, result.output
    rows = conn.execute(
        "SELECT rating, q, paths_json, extra_json FROM rag_feedback"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][1] == "harvest me"
    assert json.loads(rows[0][2]) == ["golden.md"]
    extra = json.loads(rows[0][3])
    assert extra["source"] == "harvester"


def test_cli_harvest_negative_inserts_all_paths_as_negatives(temp_db):
    conn, _ = temp_db
    _insert_q(conn, ts="2026-04-22T10:00", q="all bad",
              top_score=0.05, paths=["bad1.md", "bad2.md"])

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["feedback", "harvest", "--since", "365",
                  "--confidence-below", "0.2", "--limit", "5"],
        input="-\n",
    )
    assert result.exit_code == 0, result.output
    row = conn.execute(
        "SELECT rating, paths_json FROM rag_feedback"
    ).fetchone()
    assert row[0] == -1
    assert json.loads(row[1]) == ["bad1.md", "bad2.md"]


def test_cli_harvest_correction_inserts_both_rows(temp_db):
    conn, _ = temp_db
    _insert_q(conn, ts="2026-04-22T10:00", q="needs correction",
              top_score=0.05, paths=["bad.md"])

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["feedback", "harvest", "--since", "365",
                  "--confidence-below", "0.2", "--limit", "5"],
        input="c\ngolden.md\n",
    )
    assert result.exit_code == 0, result.output
    rows = conn.execute(
        "SELECT rating, paths_json, extra_json FROM rag_feedback ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    # First row: −1 with original bad paths
    assert rows[0][0] == -1
    assert json.loads(rows[0][1]) == ["bad.md"]
    # Second row: +1 with the golden, corrective_path set
    assert rows[1][0] == 1
    assert json.loads(rows[1][1]) == ["golden.md"]
    extra = json.loads(rows[1][2])
    assert extra["corrective_path"] == "golden.md"
    assert extra["reason"] == "corrective"


def test_cli_harvest_empty_is_noop(temp_db):
    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["feedback", "harvest", "--since", "365",
                  "--confidence-below", "0.2", "--limit", "5"],
    )
    assert result.exit_code == 0
    assert "Sin candidatos" in result.output
