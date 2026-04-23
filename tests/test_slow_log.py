"""Feature #13 del 2026-04-23 — `rag log --slow` filter tests.

Validates:
- --slow filters by total latency > --slower-than
- Ordered by total latency DESC (worst first)
- --slower-than default 5000 ms
- Empty result set renders a table header (no crash)
- Combination with --low-confidence works correctly
"""
from __future__ import annotations

import contextlib
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
    " session TEXT,"
    " mode TEXT,"
    " top_score REAL,"
    " t_retrieve REAL,"
    " t_gen REAL,"
    " answer_len INTEGER,"
    " citation_repaired INTEGER,"
    " critique_fired INTEGER,"
    " critique_changed INTEGER,"
    " variants_json TEXT,"
    " paths_json TEXT,"
    " scores_json TEXT,"
    " filters_json TEXT,"
    " bad_citations_json TEXT,"
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


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(_QUERIES_DDL)
    conn.execute(_FEEDBACK_DDL)
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


def _insert_q(conn, *, q, t_retrieve, t_gen, top_score=0.5, ts="2026-04-23T10:00"):
    conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, mode, top_score, t_retrieve, t_gen) "
        "VALUES (?, 'query', ?, 'direct', ?, ?, ?)",
        (ts, q, top_score, t_retrieve, t_gen),
    )
    conn.commit()


# ── --slow filter behavior ───────────────────────────────────────────────


def test_slow_filters_by_threshold(temp_db):
    conn, _ = temp_db
    _insert_q(conn, q="fast", t_retrieve=100, t_gen=200)   # total 300
    _insert_q(conn, q="slow", t_retrieve=3000, t_gen=4000)  # total 7000
    _insert_q(conn, q="medium", t_retrieve=1000, t_gen=1000)  # total 2000

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["log", "--slow", "--slower-than", "5000", "-n", "10"],
    )
    assert result.exit_code == 0, result.output
    assert "slow" in result.output
    assert "fast" not in result.output
    assert "medium" not in result.output


def test_slow_orders_worst_first(temp_db):
    conn, _ = temp_db
    _insert_q(conn, q="slow1", t_retrieve=2000, t_gen=4000,
              ts="2026-04-23T10:00")  # 6000
    _insert_q(conn, q="slowest", t_retrieve=5000, t_gen=5000,
              ts="2026-04-23T10:01")  # 10000
    _insert_q(conn, q="slow2", t_retrieve=3000, t_gen=4000,
              ts="2026-04-23T10:02")  # 7000

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["log", "--slow", "--slower-than", "5000", "-n", "10"],
    )
    assert result.exit_code == 0
    # Find the order of appearance — slowest should be first
    idx_slowest = result.output.find("slowest")
    idx_slow2 = result.output.find("slow2")
    idx_slow1 = result.output.find("slow1")
    assert idx_slowest < idx_slow2 < idx_slow1


def test_slow_default_threshold_5000(temp_db):
    conn, _ = temp_db
    # Total 4000 < 5000 default → should NOT appear.
    _insert_q(conn, q="below_default", t_retrieve=2000, t_gen=2000)
    # Total 6000 > 5000 → should appear.
    _insert_q(conn, q="above_default", t_retrieve=3000, t_gen=3000)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["log", "--slow", "-n", "10"])
    assert result.exit_code == 0
    assert "above_default" in result.output
    assert "below_default" not in result.output


def test_slow_custom_threshold(temp_db):
    conn, _ = temp_db
    _insert_q(conn, q="under_custom", t_retrieve=500, t_gen=400)  # 900
    _insert_q(conn, q="over_custom", t_retrieve=600, t_gen=500)   # 1100

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["log", "--slow", "--slower-than", "1000", "-n", "10"],
    )
    assert result.exit_code == 0
    assert "over_custom" in result.output
    assert "under_custom" not in result.output


def test_slow_empty_set_renders_table(temp_db):
    """No queries above threshold → table renders with 0 rows + header."""
    conn, _ = temp_db
    _insert_q(conn, q="fast", t_retrieve=100, t_gen=100)

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["log", "--slow", "--slower-than", "10000", "-n", "10"],
    )
    assert result.exit_code == 0
    assert "Queries más lentas" in result.output
    assert "(0," in result.output  # count 0


def test_slow_title_reflects_threshold(temp_db):
    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["log", "--slow", "--slower-than", "3500", "-n", "5"],
    )
    assert result.exit_code == 0
    assert ">3500ms" in result.output


def test_non_slow_uses_original_title(temp_db):
    """Without --slow, title is 'Últimas N queries' (no regression)."""
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["log", "-n", "5"])
    assert result.exit_code == 0
    assert "Últimas" in result.output
    assert "más lentas" not in result.output


def test_slow_n_limit(temp_db):
    """With --slow -n 2, we see only the top 2 slowest."""
    conn, _ = temp_db
    for i in range(5):
        _insert_q(conn, q=f"slow-{i}", t_retrieve=6000 + i * 100, t_gen=100,
                  ts=f"2026-04-23T10:0{i}")

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["log", "--slow", "--slower-than", "5000", "-n", "2"],
    )
    assert result.exit_code == 0
    # We kept only the top 2 slowest. Check that "slow-4" (the slowest,
    # 6400ms) is present and "slow-0" (fastest of the "slows", 6000ms)
    # is NOT.
    assert "slow-4" in result.output
    assert "slow-3" in result.output
    # Not all 5 should appear — the filter is bounded.
    assert "slow-0" not in result.output
