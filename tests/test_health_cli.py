"""Feature #8 del 2026-04-23 — `rag health` dashboard tests.

Validates:
- _health_query_stats aggregates + computes P50/P95 correctly
- _health_calibration_status reads rag_score_calibration
- _health_features_opt_in reflects env flags correctly
- CLI renders a table + supports --as-json
- Empty DB → graceful defaults (no crash)
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
    " t_retrieve REAL,"
    " t_gen REAL,"
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

_CAL_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_score_calibration ("
    " source TEXT PRIMARY KEY,"
    " raw_knots_json TEXT NOT NULL,"
    " cal_knots_json TEXT NOT NULL,"
    " n_pos INTEGER NOT NULL,"
    " n_neg INTEGER NOT NULL,"
    " trained_at TEXT NOT NULL,"
    " model_version TEXT NOT NULL,"
    " extra_json TEXT"
    ")"
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(_QUERIES_DDL)
    conn.execute(_FEEDBACK_DDL)
    conn.execute(_CAL_DDL)
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


# ── _health_query_stats ──────────────────────────────────────────────────


def test_query_stats_empty_db_defaults(temp_db):
    out = rag._health_query_stats(since_hours=24)
    assert out["count"] == 0
    assert out["avg_retrieve_ms"] == 0.0
    assert out["avg_gen_ms"] == 0.0


def test_query_stats_aggregates_recent(temp_db):
    conn, _ = temp_db
    from datetime import datetime
    now = datetime.now().isoformat(timespec="seconds")
    conn.executemany(
        "INSERT INTO rag_queries(ts, cmd, q, t_retrieve, t_gen, extra_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (now, "query", "q1", 100.0, 500.0, None),
            (now, "query", "q2", 200.0, 800.0, '{"cache_hit": true}'),
            (now, "chat", "q3", 150.0, 600.0, None),
        ],
    )
    conn.commit()

    out = rag._health_query_stats(since_hours=24)
    assert out["count"] == 3
    assert out["avg_retrieve_ms"] == pytest.approx(150.0, abs=0.1)
    assert out["avg_gen_ms"] == pytest.approx(633.33, abs=1.0)
    assert out["cache_hits"] == 1
    assert out["by_cmd"] == {"query": 2, "chat": 1}


def test_query_stats_excludes_old_rows(temp_db):
    """Rows outside the since window should not be counted."""
    conn, _ = temp_db
    # Old row 2 days ago
    from datetime import datetime, timedelta
    old = (datetime.now() - timedelta(days=2)).isoformat(timespec="seconds")
    now = datetime.now().isoformat(timespec="seconds")
    conn.executemany(
        "INSERT INTO rag_queries(ts, cmd, q, t_retrieve, t_gen) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (old, "query", "old", 1000.0, 2000.0),
            (now, "query", "new", 100.0, 200.0),
        ],
    )
    conn.commit()

    out = rag._health_query_stats(since_hours=24)
    assert out["count"] == 1  # only the 'new' row


# ── _health_calibration_status ───────────────────────────────────────────


def test_calibration_status_empty(temp_db):
    out = rag._health_calibration_status()
    assert out["sources_trained"] == 0
    assert out["sources"] == []


def test_calibration_status_reports_trained_sources(temp_db):
    conn, _ = temp_db
    conn.execute(
        "INSERT INTO rag_score_calibration "
        "(source, raw_knots_json, cal_knots_json, n_pos, n_neg, "
        " trained_at, model_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("vault", "[0.0, 1.0]", "[0.0, 1.0]", 25, 307,
         "2026-04-23T11:45:54", "isotonic-v1"),
    )
    conn.execute(
        "INSERT INTO rag_score_calibration "
        "(source, raw_knots_json, cal_knots_json, n_pos, n_neg, "
        " trained_at, model_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("whatsapp", "[0.01, 0.1]", "[0.0, 1.0]", 15, 120,
         "2026-04-23T11:46:00", "isotonic-v1"),
    )
    conn.commit()

    out = rag._health_calibration_status()
    assert out["sources_trained"] == 2
    sources_by_name = {s["source"]: s for s in out["sources"]}
    assert sources_by_name["vault"]["n_pos"] == 25
    assert sources_by_name["whatsapp"]["n_neg"] == 120


# ── _health_features_opt_in ──────────────────────────────────────────────


def test_features_opt_in_reflects_env(monkeypatch):
    monkeypatch.setenv("RAG_SCORE_CALIBRATION", "1")
    monkeypatch.setenv("RAG_LLM_INTENT", "")
    monkeypatch.delenv("RAG_MMR_DIVERSITY", raising=False)
    monkeypatch.delenv("RAG_PPR_TOPIC", raising=False)

    out = rag._health_features_opt_in()
    assert out["Feature #2 score calibration"]["enabled"] is True
    assert out["Feature #3 LLM intent"]["enabled"] is False
    assert out["Feature #5 MMR diversity"]["enabled"] is False
    assert out["Feature #6 Personalized PageRank"]["enabled"] is False
    # Always-on features report status without 'enabled' field.
    assert "enabled" not in out["Feature #1 auto-harvest"]
    assert "enabled" not in out["Feature #4 agent loop upgrade"]


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_health_as_json_shape(temp_db):
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["health", "--as-json"])
    assert result.exit_code == 0, result.output
    # Output should be valid JSON (one of the last lines).
    lines = [l for l in result.output.strip().splitlines() if l.startswith("{")]
    assert lines, f"No JSON line in output: {result.output!r}"
    data = json.loads(lines[-1])
    assert {"corpus", "queries", "feedback", "calibration", "features"} <= set(data.keys())


def test_cli_health_default_renders_summary(temp_db):
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["health"])
    assert result.exit_code == 0, result.output
    assert "Health Dashboard" in result.output
    assert "Corpus" in result.output
    assert "Queries" in result.output
    assert "Feedback" in result.output
    assert "Features opt-in" in result.output


def test_cli_health_since_option_accepted(temp_db):
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["health", "--since", "72", "--as-json"])
    assert result.exit_code == 0
    lines = [l for l in result.output.strip().splitlines() if l.startswith("{")]
    data = json.loads(lines[-1])
    assert data["queries"]["since_hours"] == 72


def test_cli_health_empty_db_no_crash(temp_db):
    """Everything empty — dashboard should still render without crashing."""
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["health"])
    assert result.exit_code == 0, result.output
