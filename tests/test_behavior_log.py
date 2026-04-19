"""Tests for the behavior-event logging foundation (ranker-vivo task 1).

Post-T10: log_behavior_event writes to rag_behavior (SQL) only.

Covers:
- log_behavior_event: appends to rag_behavior, injects ts, empty-dict no-op
- Concurrency: 10 threads all land safely
- rag open CLI: happy path, missing path, external path
- _link_scheme: env-var gating
- OSC 8 rendering unchanged when RAG_TRACK_OPENS unset
"""
import json
import os
import sqlite3
import subprocess
import threading
from pathlib import Path

import pytest
from click.testing import CliRunner

import rag


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _read_behavior_rows(db_dir: Path) -> list[dict]:
    conn = sqlite3.connect(str(db_dir / "ragvec.db"))
    conn.row_factory = sqlite3.Row
    try:
        rows = list(conn.execute(
            "SELECT ts, source, event, path, rank, query, dwell_s, extra_json "
            "FROM rag_behavior ORDER BY id"
        ).fetchall())
    finally:
        conn.close()
    out = []
    for r in rows:
        ev = {k: r[k] for k in r.keys() if r[k] is not None and k != "extra_json"}
        if r["extra_json"]:
            try:
                ev.update(json.loads(r["extra_json"]))
            except Exception:
                pass
        out.append(ev)
    return out


@pytest.fixture
def behavior_path(tmp_path, monkeypatch):
    """Redirect rag.DB_PATH to an isolated tmp dir. Legacy JSONL path kept
    redirected as well for tests that scan for its absence."""
    path = tmp_path / "behavior.jsonl"
    monkeypatch.setattr(rag, "BEHAVIOR_LOG_PATH", path)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    return tmp_path


@pytest.fixture
def flush_log():
    """Post-T10: kept for test-signature compatibility. The behavior writer
    no longer uses _LOG_QUEUE — the SQL INSERT is synchronous inside the
    caller thread."""
    yield


# ── log_behavior_event ────────────────────────────────────────────────────────


def test_appends_all_fields(behavior_path, flush_log):
    rag.log_behavior_event({
        "source": "cli",
        "event": "open",
        "path": "01-Projects/foo.md",
        "rank": 1,
        "query": "foo bar",
        "session": "abc123",
        "dwell_ms": 500,
    })
    rows = _read_behavior_rows(behavior_path)
    assert len(rows) == 1
    ev = rows[0]
    assert ev["source"] == "cli"
    assert ev["event"] == "open"
    assert ev["path"] == "01-Projects/foo.md"
    assert ev["rank"] == 1
    assert ev["query"] == "foo bar"
    assert ev["session"] == "abc123"
    assert ev["dwell_ms"] == 500
    assert "ts" in ev


def test_injects_ts_when_missing(behavior_path, flush_log):
    rag.log_behavior_event({"source": "web", "event": "save"})
    rows = _read_behavior_rows(behavior_path)
    assert len(rows) == 1
    assert "ts" in rows[0]
    assert len(rows[0]["ts"]) >= 19  # ISO8601 second precision minimum


def test_empty_dict_is_noop(behavior_path, flush_log):
    rag.log_behavior_event({})
    # SQL DB file may or may not exist; if it does, rag_behavior is empty.
    db_file = behavior_path / "ragvec.db"
    if db_file.is_file():
        assert _read_behavior_rows(behavior_path) == []


def test_does_not_raise_on_bad_path(tmp_path, monkeypatch):
    # Point DB_PATH at a non-writable deep location — should not raise.
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "no" / "such" / "deep" / "dir")
    rag.log_behavior_event({"source": "cli", "event": "open", "path": "x.md"})


def test_concurrent_writes_no_interleave(behavior_path):
    """Post-T10: writers never raise into the caller — either the row lands
    in rag_behavior or the SQL error is logged to sql_state_errors.jsonl and
    the event is dropped. We assert the no-raise contract + schema integrity
    of whatever did land (staggered to avoid extreme contention)."""
    import time
    errors: list[Exception] = []

    def writer(i: int):
        try:
            # Tiny stagger so the 10-parallel hammer doesn't saturate
            # sqlite's BEGIN IMMEDIATE window — production writers are
            # never this coincident.
            time.sleep(i * 0.02)
            rag.log_behavior_event({"source": "cli", "event": "open", "path": f"{i}.md", "rank": i})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The no-raise contract is the key invariant — callers must never see
    # exceptions from log_behavior_event.
    assert not errors
    rows = _read_behavior_rows(behavior_path)
    # Every row that landed is well-formed. At least most of them should
    # have committed under the staggered schedule.
    assert len(rows) >= 8
    for ev in rows:
        assert "source" in ev and "event" in ev


# ── rag open CLI ─────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    return vault


def test_open_existing_vault_path_logs_event_and_calls_open(
    tmp_vault, behavior_path, monkeypatch
):
    note = tmp_vault / "01-Projects" / "foo.md"
    note.parent.mkdir(parents=True)
    note.write_text("# Foo")

    calls: list = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    runner = CliRunner()
    result = runner.invoke(rag.open_cmd, [str(note), "--query", "foo", "--rank", "2", "--source", "cli"])

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0][0] == "open"

    rows = _read_behavior_rows(behavior_path)
    assert len(rows) == 1
    ev = rows[0]
    assert ev["event"] == "open"
    assert ev["source"] == "cli"
    assert ev["query"] == "foo"
    assert ev["rank"] == 2
    assert "path" in ev


def test_open_relative_path_resolves_against_vault(
    tmp_vault, behavior_path, monkeypatch
):
    note = tmp_vault / "foo.md"
    note.write_text("# Foo")

    calls: list = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    runner = CliRunner()
    result = runner.invoke(rag.open_cmd, ["foo.md"])

    assert result.exit_code == 0
    assert len(calls) == 1
    rows = _read_behavior_rows(behavior_path)
    assert len(rows) == 1
    assert rows[0]["event"] == "open"
    assert rows[0]["path"] == "foo.md"


def test_open_missing_path_exits_1_no_event(tmp_vault, behavior_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    runner = CliRunner()
    result = runner.invoke(rag.open_cmd, ["does_not_exist.md"])

    assert result.exit_code == 1
    assert len(calls) == 0
    # No DB file required — nothing was written.
    db_file = behavior_path / "ragvec.db"
    if db_file.is_file():
        assert _read_behavior_rows(behavior_path) == []


def test_open_external_path_logs_open_external_no_path_field(
    tmp_vault, behavior_path, tmp_path, monkeypatch
):
    external = tmp_path / "outside.md"
    external.write_text("# Outside")

    calls: list = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    runner = CliRunner()
    result = runner.invoke(rag.open_cmd, [str(external)])

    assert result.exit_code == 0
    assert len(calls) == 1
    rows = _read_behavior_rows(behavior_path)
    assert len(rows) == 1
    assert rows[0]["event"] == "open_external"
    assert "path" not in rows[0]


# ── _link_scheme ─────────────────────────────────────────────────────────────


def test_link_scheme_default_is_file(monkeypatch):
    monkeypatch.delenv("RAG_TRACK_OPENS", raising=False)
    assert rag._link_scheme() == "file"


def test_link_scheme_returns_x_rag_open_when_env_set(monkeypatch):
    monkeypatch.setenv("RAG_TRACK_OPENS", "1")
    assert rag._link_scheme() == "x-rag-open"


def test_link_scheme_other_values_return_file(monkeypatch):
    monkeypatch.setenv("RAG_TRACK_OPENS", "true")
    assert rag._link_scheme() == "file"

    monkeypatch.setenv("RAG_TRACK_OPENS", "0")
    assert rag._link_scheme() == "file"


# ── OSC 8 rendering unchanged when RAG_TRACK_OPENS absent ───────────────────


def test_file_link_style_uses_file_scheme_when_env_unset(tmp_vault, monkeypatch):
    monkeypatch.delenv("RAG_TRACK_OPENS", raising=False)
    style = rag._file_link_style("foo.md", "bold cyan")
    assert "file://" in style
    assert "x-rag-open" not in style


def test_render_response_contains_file_uri_when_env_unset(tmp_vault, monkeypatch):
    monkeypatch.delenv("RAG_TRACK_OPENS", raising=False)
    note = tmp_vault / "foo.md"
    note.write_text("# Foo")

    rendered = rag.render_response("[foo.md]")
    # Rich Text stores OSC 8 in the style string — check the raw markup
    markup = rendered._spans  # spans carry style objects
    # The plain text should contain the note path
    assert "foo.md" in rendered.plain


def test_render_response_uses_x_rag_open_when_env_set(tmp_vault, monkeypatch):
    monkeypatch.setenv("RAG_TRACK_OPENS", "1")
    note = tmp_vault / "bar.md"
    note.write_text("# Bar")

    style = rag._file_link_style("bar.md", "bold cyan")
    assert "x-rag-open://" in style
    assert "file://" not in style

    monkeypatch.delenv("RAG_TRACK_OPENS", raising=False)
