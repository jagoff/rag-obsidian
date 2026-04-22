"""T6: dashboard SQL-path parity tests.

`_dashboard_compute` must produce the same shape (every top-level key the
dashboard JS reads) whether data is sourced from rag_* SQL tables or JSONL.
Numerical parity within ±epsilon where floats are involved, exact for counts.

Each test isolates DB + JSONL paths via monkeypatch (same pattern as
`test_rag_readers_sql.py`) so the real `~/.local/share/obsidian-rag/` never
gets touched.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

import rag
from web import server as web_server


# ── Fixtures ────────────────────────────────────────────────────────────────

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


def _redirect_paths(monkeypatch, base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG",
                         base / "sql_state_errors.jsonl")


@pytest.fixture
def sql_env(tmp_path, monkeypatch):
    """Flag ON + DB_PATH redirected; data_dir (JSONL) pointed at tmp."""
    monkeypatch.setattr(rag, "RAG_STATE_SQL", True)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    # server.py has its own module-level RAG_STATE_SQL import — patch both.
    monkeypatch.setattr(web_server, "RAG_STATE_SQL", True)
    _redirect_paths(monkeypatch, tmp_path)
    # Redirect Path.home so `data_dir = Path.home() / ".local/share/obsidian-rag"`
    # resolves to an isolated tmp location — no reliance on real host JSONL.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    data_dir = tmp_path / ".local/share/obsidian-rag"
    data_dir.mkdir(parents=True, exist_ok=True)
    yield {"tmp": tmp_path, "data_dir": data_dir}


@pytest.fixture
def jsonl_env(tmp_path, monkeypatch):
    """Flag OFF + Path.home redirected."""
    monkeypatch.setattr(rag, "RAG_STATE_SQL", False)
    monkeypatch.setattr(web_server, "RAG_STATE_SQL", False)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    _redirect_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    data_dir = tmp_path / ".local/share/obsidian-rag"
    data_dir.mkdir(parents=True, exist_ok=True)
    yield {"tmp": tmp_path, "data_dir": data_dir}


# ── Seeding helpers ─────────────────────────────────────────────────────────

def _seed_sql(tmp_path: Path, table: str, events: list[dict], mapper) -> None:
    conn = _open_db(tmp_path)
    try:
        for ev in events:
            rag._sql_append_event(conn, table, mapper(ev))
    finally:
        conn.close()


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(e, default=str) for e in events) + "\n",
        encoding="utf-8",
    )


def _now_iso(offset_days: float = 0.0, hour: int = 10) -> str:
    return (datetime.now() - timedelta(days=offset_days)).replace(
        hour=hour, minute=0, second=0, microsecond=0
    ).isoformat(timespec="seconds")


# ── Shape parity ─────────────────────────────────────────────────────────────

def test_dashboard_sql_path_matches_jsonl_shape(sql_env):
    """Seed identical events to SQL + JSONL, compare aggregate shape."""
    data_dir = sql_env["data_dir"]
    now = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)

    queries = []
    for i in range(20):
        ts = (now - timedelta(hours=i)).isoformat(timespec="seconds")
        queries.append({
            "ts": ts,
            "cmd": "query" if i % 2 == 0 else "chat",
            "q": f"test question {i % 3}",
            "session": "web:abc" if i % 4 == 0 else "cli",
            "top_score": 0.3 + (i % 5) * 0.1,
            "t_retrieve": 0.5,
            "t_gen": 1.2,
            "paths": [f"01-Projects/a{i % 3}.md"],
            "answer_len": 100 + i * 10,
        })

    behavior = [
        {"ts": _now_iso(i), "source": "cli", "event": "open",
         "path": f"note{i}.md"}
        for i in range(10)
    ]

    feedback = [
        {"ts": _now_iso(i), "turn_id": f"t{i}",
         "rating": 1 if i % 2 == 0 else -1,
         "q": f"q {i}", "paths": [f"fb{i}.md"]}
        for i in range(5)
    ]

    # Seed SQL
    _seed_sql(sql_env["tmp"], "rag_queries", queries, rag._map_queries_row)
    _seed_sql(sql_env["tmp"], "rag_behavior", behavior, rag._map_behavior_row)
    _seed_sql(sql_env["tmp"], "rag_feedback", feedback, rag._map_feedback_row)

    # Seed matching JSONL (same data)
    _write_jsonl(data_dir / "queries.jsonl", queries)
    _write_jsonl(data_dir / "feedback.jsonl", feedback)
    _write_jsonl(data_dir / "behavior.jsonl", behavior)

    sql_out = web_server._dashboard_compute_sql(days=7)

    # Post-T10: SQL path is the only path. Assert core shape + KPIs against
    # the seed counts rather than against a JSONL reference (which is gone).
    assert sql_out["kpis"]["total_queries"] == 20
    assert sql_out["kpis"]["total_queries_all_time"] == 20
    assert sql_out["kpis"]["feedback_positive"] == 3  # i=0,2,4 rated +1
    assert sql_out["kpis"]["feedback_negative"] == 2  # i=1,3 rated -1
    # Hours heatmap has all 24 buckets.
    assert len(sql_out["hours"]) == 24
    # Score p50 sits near the mean of 0.3..0.7 cycled across 20 samples.
    assert 0.3 <= sql_out["score_stats"]["p50"] <= 0.8


def test_dashboard_sql_empty_returns_zero_shape(sql_env):
    """Empty SQL + empty JSONL → all numeric aggregates are 0/empty, no KeyError."""
    d = web_server._dashboard_compute_sql(days=7)
    assert d["kpis"]["total_queries"] == 0
    assert d["kpis"]["total_queries_all_time"] == 0
    assert d["kpis"]["feedback_positive"] == 0
    assert d["kpis"]["feedback_negative"] == 0
    assert d["queries_per_day"] == {}
    assert d["latency_per_day"] == {}
    assert d["sources"] == {}
    assert d["cmds"] == {}
    # Every bucket zero.
    assert all(v == 0 for v in d["score_distribution"])
    # Hours dict has all 24 keys, values all 0.
    assert len(d["hours"]) == 24
    assert all(v == 0 for v in d["hours"].values())
    assert d["hot_topics"] == []
    assert d["chat_keywords"] == []


def test_dashboard_sql_cutoff_filters_properly(sql_env):
    """5 rows within 7d + 5 outside → only 5 counted in window (windowed KPI).

    total_queries_all_time still counts all 10 rows (unwindowed).
    """
    now = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
    fresh = [{
        "ts": (now - timedelta(days=i)).isoformat(timespec="seconds"),
        "cmd": "query",
        "q": f"fresh {i}",
        "top_score": 0.5,
    } for i in range(5)]
    stale = [{
        "ts": (now - timedelta(days=30 + i)).isoformat(timespec="seconds"),
        "cmd": "query",
        "q": f"stale {i}",
        "top_score": 0.5,
    } for i in range(5)]
    _seed_sql(sql_env["tmp"], "rag_queries", fresh + stale, rag._map_queries_row)

    d = web_server._dashboard_compute_sql(days=7)
    assert d["kpis"]["total_queries"] == 5
    assert d["kpis"]["total_queries_all_time"] == 10


def test_dashboard_flag_off_still_uses_sql(jsonl_env):
    """Post-T10: RAG_STATE_SQL flag is inert. `_dashboard_compute` always
    reads from SQL. JSONL on disk is ignored."""
    data_dir = jsonl_env["data_dir"]
    # JSONL on disk, SQL empty.
    _write_jsonl(data_dir / "queries.jsonl", [{
        "ts": _now_iso(0), "cmd": "query", "q": "from jsonl",
        "top_score": 0.42,
    }])
    # Also seed SQL so we can prove SQL is the source.
    _seed_sql(jsonl_env["tmp"], "rag_queries", [{
        "ts": _now_iso(0), "cmd": "query", "q": "from sql",
        "top_score": 0.88,
    }], rag._map_queries_row)

    d = web_server._dashboard_compute(days=7)
    assert d["kpis"]["total_queries"] == 1
    assert d["score_stats"]["p50"] == pytest.approx(0.88, abs=1e-3)


def test_dashboard_sql_exception_returns_empty_shape(sql_env):
    """Post-T10: when the SQL compute raises, the dispatcher logs the error
    and returns an empty-shape payload (no JSONL fallback)."""
    data_dir = sql_env["data_dir"]
    _write_jsonl(data_dir / "queries.jsonl", [{
        "ts": _now_iso(0), "cmd": "query", "q": "ignored jsonl",
        "top_score": 0.33,
    }])

    with patch.object(rag, "_sql_query_window",
                       side_effect=RuntimeError("boom")):
        d = web_server._dashboard_compute(days=7)

    # Empty-shape payload, not JSONL-backed.
    assert d["kpis"]["total_queries"] == 0
    assert d["kpis"]["total_queries_all_time"] == 0
    # Error log gained a record with the expected event type.
    log_txt = rag._SQL_STATE_ERROR_LOG.read_text(encoding="utf-8")
    assert "dashboard_sql_compute_failed" in log_txt


def test_dashboard_sql_hot_topics_top15(sql_env):
    """Known-frequency topic distribution → top-15 sorted descending by count."""
    now = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
    # 20 distinct first-3-word topics; high-freq topic repeated 5x.
    events: list[dict] = []
    for i in range(20):
        for _ in range(5 if i == 0 else 1):
            events.append({
                "ts": (now - timedelta(minutes=len(events))).isoformat(
                    timespec="seconds"),
                "cmd": "query",
                "q": f"topic{i} alpha beta",
                "top_score": 0.5,
            })
    _seed_sql(sql_env["tmp"], "rag_queries", events, rag._map_queries_row)

    d = web_server._dashboard_compute_sql(days=7)
    # Top-15 sorted descending; top entry is the 5x repeat.
    assert len(d["hot_topics"]) == 15
    counts = [t["count"] for t in d["hot_topics"]]
    assert counts == sorted(counts, reverse=True)
    assert d["hot_topics"][0]["count"] == 5


def test_dashboard_sql_hourly_heatmap_24_entries(sql_env):
    """Seed queries at known hours → hours dict has correct counts."""
    today = datetime.now().replace(minute=0, second=0, microsecond=0)
    events = []
    for hour, n in ((3, 2), (9, 5), (20, 1)):
        for i in range(n):
            events.append({
                "ts": today.replace(hour=hour, minute=i).isoformat(
                    timespec="seconds"),
                "cmd": "query",
                "q": f"at hour {hour}",
                "top_score": 0.5,
            })
    _seed_sql(sql_env["tmp"], "rag_queries", events, rag._map_queries_row)

    d = web_server._dashboard_compute_sql(days=7)
    assert len(d["hours"]) == 24
    assert d["hours"]["3"] == 2
    assert d["hours"]["9"] == 5
    assert d["hours"]["20"] == 1
    # Everything else is zero.
    total = sum(d["hours"].values())
    assert total == 8


def test_dashboard_sql_sources_breakdown(sql_env):
    """wa:/web:/cli:/tg: session + serve-cmd rule → sources counts match."""
    now = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
    events = [
        {"ts": now.isoformat(timespec="seconds"), "cmd": "query", "q": "q",
         "session": "wa:120363426178035051@g.us", "top_score": 0.5},
        {"ts": (now - timedelta(minutes=1)).isoformat(timespec="seconds"),
         "cmd": "serve.chat", "q": "q", "session": "other:x", "top_score": 0.5},
        {"ts": (now - timedelta(minutes=2)).isoformat(timespec="seconds"),
         "cmd": "query", "q": "q", "session": "web:abc", "top_score": 0.5},
        {"ts": (now - timedelta(minutes=3)).isoformat(timespec="seconds"),
         "cmd": "query", "q": "q", "session": "tg:456", "top_score": 0.5},
        {"ts": (now - timedelta(minutes=4)).isoformat(timespec="seconds"),
         "cmd": "query", "q": "q", "session": None, "top_score": 0.5},
    ]
    _seed_sql(sql_env["tmp"], "rag_queries", events, rag._map_queries_row)

    d = web_server._dashboard_compute_sql(days=7)
    assert d["sources"].get("whatsapp") == 2  # wa: session + serve cmd
    assert d["sources"].get("web") == 1
    # Post-2026-04 rename: `tg:` sessions count as "legacy" (bot was
    # deprecated in favor of WhatsApp listener; the bucket stays as a
    # grouping for any old `tg:*` session_ids still in rag_queries).
    assert d["sources"].get("legacy") == 1
    assert d["sources"].get("cli") == 1
