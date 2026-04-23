"""Feature #18 del 2026-04-23 — `rag trends` dashboard tests.

Validates:
- _extract_trends groups by folder, tag, keyword, source correctly.
- Stopwords filtered from keywords.
- Cross-source paths contribute to source breakdown.
- CLI output + --as-json.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime, timedelta

import pytest
from click.testing import CliRunner

import rag


_QUERIES_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_queries ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ts TEXT NOT NULL,"
    " cmd TEXT,"
    " q TEXT NOT NULL,"
    " paths_json TEXT,"
    " extra_json TEXT"
    ")"
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(_QUERIES_DDL)
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


def _insert_q(conn, *, q, paths, ts=None, extra=None):
    ts = ts or datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, paths_json, extra_json) "
        "VALUES (?, 'query', ?, ?, ?)",
        (ts, q, json.dumps(paths),
         json.dumps(extra) if extra else None),
    )
    conn.commit()


# ── _extract_trends ──────────────────────────────────────────────────────


def test_empty_db_returns_zeros(temp_db):
    r = rag._extract_trends(days=7)
    assert r["n_queries"] == 0
    assert r["folders"] == []
    assert r["tags"] == []
    assert r["keywords"] == []


def test_folders_aggregated_from_paths(temp_db):
    conn, _ = temp_db
    _insert_q(conn, q="test query",
              paths=["01-Projects/a.md", "01-Projects/b.md", "02-Areas/c.md"])
    r = rag._extract_trends(days=7)
    folders = dict(r["folders"])
    assert folders["01-Projects"] == 2
    assert folders["02-Areas"] == 1


def test_cross_source_paths_go_to_sources_not_folders(temp_db):
    conn, _ = temp_db
    _insert_q(conn, q="q",
              paths=["whatsapp://jid/m1", "01-Projects/a.md"])
    r = rag._extract_trends(days=7)
    sources = dict(r["sources"])
    assert sources["whatsapp"] == 1
    assert sources["vault"] == 1
    folders = dict(r["folders"])
    assert folders.get("whatsapp:") is None  # cross-source not counted as folder


def test_keywords_filters_stopwords(temp_db):
    conn, _ = temp_db
    _insert_q(conn, q="¿qué es el proyecto de coaching que tengo?",
              paths=["01-Projects/a.md"])
    r = rag._extract_trends(days=7)
    kw = dict(r["keywords"])
    # Stopwords should NOT appear.
    for stop in ("que", "el", "es", "de"):
        assert stop not in kw
    # Substantive words should.
    assert "proyecto" in kw
    assert "coaching" in kw


def test_keywords_ignores_short_tokens(temp_db):
    conn, _ = temp_db
    _insert_q(conn, q="a b cd efgh", paths=[])
    r = rag._extract_trends(days=7)
    kw = dict(r["keywords"])
    # Tokens with <3 chars filtered.
    assert "a" not in kw
    assert "cd" not in kw
    assert "efgh" in kw


def test_tags_hit_from_extra_json(temp_db):
    conn, _ = temp_db
    _insert_q(conn, q="q1", paths=["01-P/a.md"],
              extra={"tags_hit": ["proyecto", "urgente"]})
    _insert_q(conn, q="q2", paths=["01-P/b.md"],
              extra={"tags_hit": ["proyecto"]})
    r = rag._extract_trends(days=7)
    tags = dict(r["tags"])
    assert tags["proyecto"] == 2
    assert tags["urgente"] == 1


def test_excludes_old_rows(temp_db):
    conn, _ = temp_db
    old_ts = (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds")
    _insert_q(conn, q="old", paths=["01-P/old.md"], ts=old_ts)
    _insert_q(conn, q="new", paths=["01-P/new.md"])
    r = rag._extract_trends(days=7)
    assert r["n_queries"] == 1
    kw = dict(r["keywords"])
    assert "new" in kw
    assert "old" not in kw


def test_respects_top_n(temp_db):
    conn, _ = temp_db
    # 15 different folders
    for i in range(15):
        _insert_q(conn, q=f"q{i}", paths=[f"folder{i:02d}/a.md"])
    r = rag._extract_trends(days=7, top_n=5)
    assert len(r["folders"]) == 5


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_trends_renders(temp_db):
    conn, _ = temp_db
    _insert_q(conn, q="coaching proyecto nuevo",
              paths=["01-Projects/a.md"])
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["trends"])
    assert result.exit_code == 0
    assert "Trends" in result.output
    assert "Folders" in result.output


def test_cli_trends_as_json(temp_db):
    conn, _ = temp_db
    _insert_q(conn, q="test", paths=["01-P/a.md"])
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["trends", "--as-json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip().splitlines()[-1])
    assert "n_queries" in data
    assert isinstance(data["folders"], list)


def test_cli_trends_empty_graceful(temp_db):
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["trends", "--days", "7"])
    assert result.exit_code == 0
    assert "0 queries" in result.output or "sin data" in result.output.lower()


def test_cli_trends_days_option(temp_db):
    conn, _ = temp_db
    _insert_q(conn, q="test", paths=["01-P/a.md"])
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["trends", "--days", "30", "--as-json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip().splitlines()[-1])
    assert data["days"] == 30
