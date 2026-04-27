import json
import os
import sqlite3
from datetime import datetime

import click
import pytest
from click.testing import CliRunner

import rag


@pytest.fixture
def digest_env(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    qlog = tmp_path / "queries.jsonl"
    clog = tmp_path / "contradictions.jsonl"
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "LOG_PATH", qlog)
    monkeypatch.setattr(rag, "CONTRADICTION_LOG_PATH", clog)
    return vault, qlog, clog


def _open_telemetry_db(tmp_path):
    """Open a fresh telemetry.db in tmp_path and ensure all tables exist."""
    db = tmp_path / rag._TELEMETRY_DB_FILENAME
    conn = sqlite3.connect(str(db), isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_schema_version "
        "(table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
    )
    rag._ensure_telemetry_tables(conn)
    return conn


@pytest.fixture
def digest_sql_env(tmp_path, monkeypatch):
    """digest_env + DB_PATH isolation so _ragvec_state_conn() uses tmp DB.

    Uses snap+restore (not monkeypatch.setattr) to avoid the teardown-order
    bug where _stabilize_rag_state sees DB_PATH still pointing to tmp_path.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    qlog = tmp_path / "queries.jsonl"
    clog = tmp_path / "contradictions.jsonl"
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "LOG_PATH", qlog)
    monkeypatch.setattr(rag, "CONTRADICTION_LOG_PATH", clog)
    snap_db = rag.DB_PATH
    rag.DB_PATH = tmp_path
    conn = _open_telemetry_db(tmp_path)
    try:
        yield vault, qlog, clog, conn
    finally:
        conn.close()
        rag.DB_PATH = snap_db


def _write_note(vault, rel_path, body, mtime=None):
    p = vault / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    if mtime is not None:
        ts = mtime.timestamp()
        os.utime(p, (ts, ts))
    return p


def _append_jsonl(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def test_iso_week_label_known_date():
    assert rag._iso_week_label(datetime(2026, 4, 9)) == "2026-W15"
    assert rag._iso_week_label(datetime(2026, 4, 13)) == "2026-W16"


def test_parse_iso_week_returns_monday_to_monday():
    start, end = rag._parse_iso_week("2026-W15")
    assert start == datetime(2026, 4, 6)
    assert end == datetime(2026, 4, 13)


def test_parse_iso_week_rejects_bad_format():
    with pytest.raises(click.BadParameter):
        rag._parse_iso_week("no-such-week")


def test_collect_evidence_empty_sources(digest_env):
    vault, qlog, clog = digest_env
    ev = rag._collect_week_evidence(
        datetime(2026, 4, 6), datetime(2026, 4, 13), vault, qlog, clog,
    )
    assert ev == {
        "recent_notes": [],
        "fm_contradictions": [],
        "index_contradictions": [],
        "query_contradictions": [],
        "low_conf_queries": [],
    }


def test_collect_evidence_recent_notes_respects_window(digest_env):
    vault, qlog, clog = digest_env
    in_window = datetime(2026, 4, 9, 12, 0, 0)
    out_window = datetime(2026, 3, 1, 12, 0, 0)
    _write_note(vault, "foo.md", "---\ntags: []\n---\n# foo\nbody in window.",
                mtime=in_window)
    _write_note(vault, "bar.md", "---\ntags: []\n---\n# bar\nout of window.",
                mtime=out_window)
    ev = rag._collect_week_evidence(
        datetime(2026, 4, 6), datetime(2026, 4, 13), vault, qlog, clog,
    )
    paths = {n["path"] for n in ev["recent_notes"]}
    assert paths == {"foo.md"}
    assert ev["recent_notes"][0]["title"] == "foo"
    assert "body in window" in ev["recent_notes"][0]["snippet"]


def test_collect_evidence_reads_frontmatter_contradicts(digest_env):
    vault, qlog, clog = digest_env
    _write_note(
        vault, "alpha.md",
        "---\ncontradicts:\n- beta.md\n- gamma.md\n---\nbody",
        mtime=datetime(2026, 4, 9, 10, 0, 0),
    )
    _write_note(vault, "no-fm.md", "# plain note",
                mtime=datetime(2026, 4, 9, 10, 0, 0))
    ev = rag._collect_week_evidence(
        datetime(2026, 4, 6), datetime(2026, 4, 13), vault, qlog, clog,
    )
    assert len(ev["fm_contradictions"]) == 1
    fc = ev["fm_contradictions"][0]
    assert fc["path"] == "alpha.md"
    assert fc["targets"] == ["beta.md", "gamma.md"]


def test_collect_evidence_windows_contradiction_log(digest_sql_env):
    """_collect_week_evidence reads rag_contradictions SQL, not JSONL."""
    vault, qlog, clog, conn = digest_sql_env
    # Insert three rows: one in-window with targets, one out-of-window, one in-window with empty targets.
    conn.execute(
        "INSERT INTO rag_contradictions (ts, subject_path, contradicts_json) VALUES (?, ?, ?)",
        ("2026-04-09T10:00:00", "new.md", json.dumps([{"path": "old.md", "why": "tensión"}])),
    )
    conn.execute(
        "INSERT INTO rag_contradictions (ts, subject_path, contradicts_json) VALUES (?, ?, ?)",
        ("2026-03-01T10:00:00", "old-note.md", json.dumps([{"path": "other.md", "why": "viejo"}])),
    )
    conn.execute(
        "INSERT INTO rag_contradictions (ts, subject_path, contradicts_json) VALUES (?, ?, ?)",
        ("2026-04-10T10:00:00", "empty.md", json.dumps([])),
    )
    ev = rag._collect_week_evidence(
        datetime(2026, 4, 6), datetime(2026, 4, 13), vault, qlog, clog,
    )
    assert len(ev["index_contradictions"]) == 1
    ic = ev["index_contradictions"][0]
    assert ic["subject_path"] == "new.md"
    assert ic["targets"] == [{"path": "old.md", "why": "tensión"}]


def test_collect_evidence_reads_query_log(digest_sql_env):
    """_collect_week_evidence reads rag_queries SQL for query_contradictions and low_conf."""
    vault, qlog, clog, conn = digest_sql_env
    # In-window: has contradictions in extra_json
    conn.execute(
        "INSERT INTO rag_queries (ts, cmd, q, top_score, extra_json) VALUES (?, ?, ?, ?, ?)",
        (
            "2026-04-09T10:00:00", "query", "¿qué es X?", 0.35,
            json.dumps({"contradictions": [{"path": "a.md", "why": "X vs Y"}]}),
        ),
    )
    # In-window: low confidence, no contradictions
    conn.execute(
        "INSERT INTO rag_queries (ts, cmd, q, top_score, extra_json) VALUES (?, ?, ?, ?, ?)",
        ("2026-04-10T10:00:00", "query", "algo oscuro", 0.005, None),
    )
    # Out-of-window: should not appear
    conn.execute(
        "INSERT INTO rag_queries (ts, cmd, q, top_score, extra_json) VALUES (?, ?, ?, ?, ?)",
        ("2026-03-01T10:00:00", "query", "fuera de rango", 0.001, None),
    )
    ev = rag._collect_week_evidence(
        datetime(2026, 4, 6), datetime(2026, 4, 13), vault, qlog, clog,
    )
    assert len(ev["query_contradictions"]) == 1
    assert ev["query_contradictions"][0]["path"] == "a.md"
    low_qs = {lq["q"] for lq in ev["low_conf_queries"]}
    assert low_qs == {"algo oscuro"}


def test_digest_dry_run_with_no_evidence(digest_env, monkeypatch):
    monkeypatch.setattr(
        rag, "_generate_digest_narrative", lambda p: "should-not-be-called",
    )
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["digest", "--week", "2026-W15", "--dry-run"])
    assert result.exit_code == 0
    assert "Sin evidencia" in result.output


def test_digest_dry_run_with_evidence(digest_env, monkeypatch):
    vault, qlog, clog = digest_env
    _write_note(
        vault, "tema.md",
        "---\ncontradicts:\n- viejo.md\n---\nhay un cambio de opinión.",
        mtime=datetime(2026, 4, 9, 10, 0, 0),
    )
    monkeypatch.setattr(
        rag, "_generate_digest_narrative",
        lambda p: "Esta semana [[tema]] reorganizó mi vista previa.",
    )
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["digest", "--week", "2026-W15", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Review 2026-W15" in result.output
    assert "[[tema]]" in result.output
    # dry-run must NOT write
    assert not (vault / "04-Archive/99-obsidian-system/99-AI/reviews" / "2026-W15.md").exists()


def test_digest_writes_and_indexes(digest_env, monkeypatch):
    vault, qlog, clog = digest_env
    _write_note(
        vault, "tema.md",
        "---\ntags: [reflexion]\n---\nbody útil.",
        mtime=datetime(2026, 4, 9, 10, 0, 0),
    )
    monkeypatch.setattr(
        rag, "_generate_digest_narrative",
        lambda p: "Prosa narrativa mock. [[tema]] apareció.",
    )
    indexed: list = []
    monkeypatch.setattr(rag, "get_db", lambda: "FAKE_COL")
    monkeypatch.setattr(
        rag, "_index_single_file", lambda col, path: indexed.append((col, path))
    )
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["digest", "--week", "2026-W15"])
    assert result.exit_code == 0, result.output
    out_path = vault / "04-Archive/99-obsidian-system/99-AI/reviews" / "2026-W15.md"
    assert out_path.is_file()
    written = out_path.read_text(encoding="utf-8")
    assert "week: '2026-W15'" in written
    assert "Prosa narrativa mock" in written
    assert indexed == [("FAKE_COL", out_path)]
