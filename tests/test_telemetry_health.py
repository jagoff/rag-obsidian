"""Tests for `telemetry_health()` + `rag insights telemetry-health` CLI.

Covers GC#2.A (gamechangers-plan 2026-04-22): intent coverage on rag_queries,
CTR-by-source on rag_behavior, and feedback gaps on rag_feedback. The helper
must degrade gracefully on empty/malformed data and filter by is_user_query.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

import rag


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


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


def _ts(offset_hours: float = 0.0) -> str:
    return (datetime.now() - timedelta(hours=offset_hours)).isoformat(timespec="seconds")


def _insert_query(conn: sqlite3.Connection, *, cmd: str, q: str,
                  extra: dict | str | None = None, ts: str | None = None) -> None:
    if isinstance(extra, dict):
        extra_json = json.dumps(extra, ensure_ascii=False)
    else:
        extra_json = extra
    conn.execute(
        "INSERT INTO rag_queries (ts, cmd, q, extra_json) VALUES (?, ?, ?, ?)",
        (ts or _ts(), cmd, q, extra_json),
    )


def _insert_behavior(conn: sqlite3.Connection, *, source: str | None, event: str,
                     ts: str | None = None) -> None:
    conn.execute(
        "INSERT INTO rag_behavior (ts, source, event) VALUES (?, ?, ?)",
        (ts or _ts(), source, event),
    )


def _insert_feedback(conn: sqlite3.Connection, *, turn_id: str, rating: int = 1,
                     ts: str | None = None) -> None:
    conn.execute(
        "INSERT INTO rag_feedback (ts, turn_id, rating) VALUES (?, ?, ?)",
        (ts or _ts(), turn_id, rating),
    )


@pytest.fixture
def db(tmp_path):
    conn = _open_db(tmp_path)
    db_path = tmp_path / rag._TELEMETRY_DB_FILENAME
    yield db_path, conn
    conn.close()


# ── Intent coverage ──────────────────────────────────────────────────────────


def test_intent_populated_pct_all_null(db):
    db_path, conn = db
    for i in range(10):
        _insert_query(conn, cmd="query", q=f"pregunta {i}", extra={"foo": "bar"})
    out = rag.telemetry_health(db_path, days=7)
    assert out["total_user_queries"] == 10
    assert out["intent_populated"] == 0
    assert out["intent_populated_pct"] == 0.0


def test_intent_populated_pct_mixed(db):
    db_path, conn = db
    for i in range(7):
        _insert_query(conn, cmd="query", q=f"pregunta {i}",
                      extra={"intent": "semantic"})
    for i in range(3):
        _insert_query(conn, cmd="query", q=f"otra {i}", extra={"intent": ""})
    out = rag.telemetry_health(db_path, days=7)
    assert out["total_user_queries"] == 10
    assert out["intent_populated"] == 7
    assert out["intent_populated_pct"] == 70.0


def test_intent_filter_respects_is_user_query(db):
    db_path, conn = db
    _insert_query(conn, cmd="query", q="real", extra={"intent": "semantic"})
    _insert_query(conn, cmd="read", q="job1", extra={"intent": "semantic"})
    _insert_query(conn, cmd="followup", q="job2", extra={"intent": "semantic"})
    _insert_query(conn, cmd="insights", q="", extra={"intent": "semantic"})
    out = rag.telemetry_health(db_path, days=7)
    assert out["total_user_queries"] == 1
    assert out["intent_populated"] == 1
    assert out["intent_populated_pct"] == 100.0


def test_intent_malformed_extra_json(db):
    db_path, conn = db
    _insert_query(conn, cmd="query", q="a", extra="{not valid json")
    _insert_query(conn, cmd="query", q="b", extra="null")
    _insert_query(conn, cmd="query", q="c", extra=None)
    _insert_query(conn, cmd="query", q="d", extra={"intent": "semantic"})
    out = rag.telemetry_health(db_path, days=7)
    assert out["total_user_queries"] == 4
    assert out["intent_populated"] == 1
    assert out["intent_populated_pct"] == 25.0


def test_intent_out_of_window_excluded(db):
    db_path, conn = db
    old = (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds")
    _insert_query(conn, cmd="query", q="viejo", extra={"intent": "semantic"}, ts=old)
    _insert_query(conn, cmd="query", q="nuevo", extra={"intent": "semantic"})
    out = rag.telemetry_health(db_path, days=7)
    assert out["total_user_queries"] == 1


# ── CTR by source ────────────────────────────────────────────────────────────


def test_ctr_by_source_laplace_smoothing(db):
    db_path, conn = db
    _insert_behavior(conn, source="cli", event="open")
    for _ in range(4):
        _insert_behavior(conn, source="cli", event="impression")
    out = rag.telemetry_health(db_path, days=7)
    stats = out["ctr_by_source"]["cli"]
    assert stats["clicks"] == 1
    assert stats["impressions"] == 5
    assert stats["ctr"] == pytest.approx((1 + 1) / (5 + 10))


def test_ctr_source_null_bucketed_as_unknown(db):
    db_path, conn = db
    _insert_behavior(conn, source="", event="impression")
    _insert_behavior(conn, source="", event="open")
    _insert_behavior(conn, source="   ", event="impression")
    out = rag.telemetry_health(db_path, days=7)
    assert "unknown" in out["ctr_by_source"]
    stats = out["ctr_by_source"]["unknown"]
    assert stats["clicks"] == 1
    assert stats["impressions"] == 3


def test_ctr_multiple_click_events_counted(db):
    db_path, conn = db
    for ev in ("open", "positive_implicit", "save", "kept"):
        _insert_behavior(conn, source="web", event=ev)
    _insert_behavior(conn, source="web", event="impression")
    _insert_behavior(conn, source="web", event="negative_implicit")
    out = rag.telemetry_health(db_path, days=7)
    stats = out["ctr_by_source"]["web"]
    assert stats["clicks"] == 4
    assert stats["impressions"] == 6


# ── Feedback gaps ────────────────────────────────────────────────────────────


def test_feedback_gaps_excludes_rated_queries(db):
    db_path, conn = db
    _insert_query(conn, cmd="query", q="rateada", extra={"turn_id": "t1"})
    _insert_query(conn, cmd="query", q="rateada", extra={"turn_id": "t2"})
    _insert_query(conn, cmd="query", q="huérfana", extra={"turn_id": "t3"})
    _insert_query(conn, cmd="query", q="huérfana", extra={"turn_id": "t4"})
    _insert_feedback(conn, turn_id="t1", rating=1)
    out = rag.telemetry_health(db_path, days=7)
    gap_qs = [g["q"] for g in out["feedback_gaps"]]
    assert "rateada" not in gap_qs
    assert "huérfana" in gap_qs


def test_feedback_gaps_top_n_sorted_by_count(db):
    db_path, conn = db
    for i, (q, n) in enumerate([("alta", 5), ("media", 3), ("baja", 2)]):
        for j in range(n):
            _insert_query(conn, cmd="query", q=q, extra={"turn_id": f"t{i}-{j}"})
    out = rag.telemetry_health(db_path, days=7, top_n=10)
    qs = [g["q"] for g in out["feedback_gaps"]]
    assert qs == ["alta", "media", "baja"]
    assert out["feedback_gaps"][0]["count"] == 5


def test_feedback_gaps_excludes_singletons(db):
    db_path, conn = db
    _insert_query(conn, cmd="query", q="una sola", extra={"turn_id": "x1"})
    _insert_query(conn, cmd="query", q="repe", extra={"turn_id": "x2"})
    _insert_query(conn, cmd="query", q="repe", extra={"turn_id": "x3"})
    out = rag.telemetry_health(db_path, days=7)
    qs = [g["q"] for g in out["feedback_gaps"]]
    assert "una sola" not in qs
    assert "repe" in qs


def test_feedback_gaps_groups_by_normalized_query(db):
    db_path, conn = db
    _insert_query(conn, cmd="query", q="¿Qué es RAG?", extra={"turn_id": "a"})
    _insert_query(conn, cmd="query", q="que es rag", extra={"turn_id": "b"})
    _insert_query(conn, cmd="query", q="QUE ES RAG", extra={"turn_id": "c"})
    out = rag.telemetry_health(db_path, days=7)
    gaps = out["feedback_gaps"]
    assert len(gaps) == 1
    assert gaps[0]["count"] == 3


# ── Graceful degradation ─────────────────────────────────────────────────────


def test_missing_db_returns_zero_structure(tmp_path):
    out = rag.telemetry_health(tmp_path / "does-not-exist.db", days=7)
    assert out["total_user_queries"] == 0
    assert out["intent_populated_pct"] == 0.0
    assert out["ctr_by_source"] == {}
    assert out["feedback_gaps"] == []


def test_empty_tables_do_not_crash(db):
    db_path, _ = db
    out = rag.telemetry_health(db_path, days=7)
    assert out["window_days"] == 7
    assert out["intent_populated_pct"] == 0.0
    assert out["ctr_by_source"] == {}
    assert out["feedback_gaps"] == []


# ── CLI surface ──────────────────────────────────────────────────────────────


def test_cli_insights_legacy_still_works(monkeypatch, tmp_path):
    """`rag insights --days 30` keeps the pre-refactor behaviour (no subcommand)."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(rag, "LOG_PATH", tmp_path / "queries.jsonl")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path / "vault")
    (tmp_path / "vault").mkdir(parents=True, exist_ok=True)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["insights", "--days", "30", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert set(data.keys()) >= {"window_days", "gaps", "hot_queries", "orphan_notes"}


def test_cli_telemetry_health_json(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    conn = _open_db(tmp_path)
    try:
        _insert_query(conn, cmd="query", q="hola", extra={"intent": "semantic"})
        _insert_behavior(conn, source="cli", event="open")
    finally:
        conn.close()
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["insights", "telemetry-health", "--days", "7", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert set(data.keys()) >= {
        "window_days", "total_user_queries", "intent_populated_pct",
        "ctr_by_source", "feedback_gaps",
    }
    assert data["intent_populated_pct"] == 100.0
    assert "cli" in data["ctr_by_source"]


def test_cli_telemetry_health_plain_no_ansi(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    _open_db(tmp_path).close()
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["insights", "telemetry-health", "--days", "7", "--plain"])
    assert result.exit_code == 0, result.output
    assert ANSI_RE.search(result.output) is None
    assert "Telemetry health" in result.output
