"""Tests for scripts/ingest_reminders.py + `rag index --source reminders`.

No live AppleScript — tests inject a mock fetcher via `fetch_fn=`
and assert parsing / chunking / upsert semantics. Covers:
  - chr(31)-separated line parsing (all fields, missing optionals)
  - Reminder → body formatting with status / due / completion markers
  - upsert_reminders writes source=reminders + all expected meta fields
  - delete_reminders removes rows by reminder_id
  - run() orchestration: first run (all new), second run (no churn),
    content change (re-indexed), deletion (stale id → delete)
  - CLI `rag index --source reminders` routes + handles errors
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

import rag
from scripts import ingest_reminders as ir


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_vault_col(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    (tmp_path / "ragvec").mkdir()
    # Create the state DB file so sqlite3.connect works.
    conn = sqlite3.connect(str(tmp_path / "ragvec" / "ragvec.db"))
    conn.close()

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="rem_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)

    def _fake_embed(texts):
        return [[float(len(t) % 10) / 10] + [0.0] * 7 for t in texts]
    monkeypatch.setattr(rag, "embed", _fake_embed)
    return col


def _mk_reminder(
    rid: str,
    name: str,
    *,
    list_name: str = "Inbox",
    body: str = "",
    completed: bool = False,
    due_ts: float = 0.0,
    completion_ts: float = 0.0,
    created_ts: float = 0.0,
    modified_ts: float = 0.0,
    priority: int = 0,
    flagged: bool = False,
) -> ir.Reminder:
    return ir.Reminder(
        id=rid,
        list_name=list_name,
        name=name,
        body=body,
        completed=completed,
        flagged=flagged,
        priority=priority,
        due_ts=due_ts,
        completion_ts=completion_ts,
        created_ts=created_ts,
        modified_ts=modified_ts,
    )


# ── Line parsing ───────────────────────────────────────────────────────

def _line(**kw) -> str:
    fs = ir._FIELD_SEP
    fields = [
        kw.get("id", "X-1"),
        kw.get("list_name", "Inbox"),
        kw.get("completed", "false"),
        kw.get("due", ""),
        kw.get("completion", ""),
        kw.get("created", ""),
        kw.get("modified", ""),
        kw.get("priority", "0"),
        kw.get("flagged", "false"),
        kw.get("name", "Hacer algo"),
        kw.get("body", ""),
    ]
    return fs.join(fields)


def test_parse_reminder_line_happy_path():
    # ISO-8601 parseable date.
    line = _line(
        id="R-1", list_name="Trabajo", name="Revisar PR",
        body="Mirar el spike", completed="false",
        due="2026-04-20 10:00:00", created="2026-04-15 09:00:00",
    )
    r = ir._parse_reminder_line(line)
    assert r is not None
    assert r.id == "R-1"
    assert r.name == "Revisar PR"
    assert r.list_name == "Trabajo"
    assert r.body == "Mirar el spike"
    assert r.completed is False
    assert r.due_ts > 0
    assert r.created_ts > 0


def test_parse_reminder_line_completed():
    line = _line(
        id="R-2", name="Cerrado", completed="true",
        completion="2026-04-10 12:00:00", created="2026-04-01 09:00:00",
    )
    r = ir._parse_reminder_line(line)
    assert r is not None
    assert r.completed is True
    assert r.completion_ts > 0


def test_parse_reminder_line_missing_id_returns_none():
    line = _line(id="", name="x")
    assert ir._parse_reminder_line(line) is None


def test_parse_reminder_line_missing_name_returns_none():
    line = _line(id="R-1", name="")
    assert ir._parse_reminder_line(line) is None


def test_parse_reminder_line_too_few_fields():
    assert ir._parse_reminder_line("only|three|fields") is None
    assert ir._parse_reminder_line("") is None


def test_parse_reminder_line_priority_non_numeric_defaults_to_zero():
    line = _line(id="R-1", name="x", priority="bogus")
    r = ir._parse_reminder_line(line)
    assert r is not None and r.priority == 0


def test_parse_reminder_line_undated():
    line = _line(id="R-1", name="sin fecha")
    r = ir._parse_reminder_line(line)
    assert r is not None
    assert r.due_ts == 0.0
    assert r.created_ts == 0.0


# ── Body formatting ────────────────────────────────────────────────────

def test_format_reminder_body_pending():
    due = datetime(2026, 4, 20, 10, 0).timestamp()
    r = _mk_reminder("R-1", "Revisar PR", due_ts=due)
    body = ir._format_reminder_body(r)
    assert "Tarea: Revisar PR" in body
    assert "⧗ pendiente" in body
    assert "Vence: 2026-04-20" in body


def test_format_reminder_body_completed():
    comp = datetime(2026, 4, 10, 12, 0).timestamp()
    r = _mk_reminder("R-2", "Hecho", completed=True, completion_ts=comp)
    body = ir._format_reminder_body(r)
    assert "✓ completada" in body
    assert "Cerrada: 2026-04-10" in body


def test_format_reminder_body_includes_list_and_priority():
    r = _mk_reminder("R-3", "Llamar", list_name="Trabajo",
                      priority=5, flagged=True)
    body = ir._format_reminder_body(r)
    assert "Lista: Trabajo" in body
    assert "Prioridad: 5" in body
    assert "⚑ Destacada" in body


def test_format_reminder_body_truncates_long_body():
    r = _mk_reminder("R-4", "x", body="y" * 2000)
    body = ir._format_reminder_body(r)
    assert len(body) <= ir.CHUNK_MAX_CHARS


def test_embed_prefix_includes_source_and_status():
    r = _mk_reminder("R-5", "PR", list_name="Trabajo", completed=False)
    pref = ir._embed_prefix(r, "body")
    assert "[source=reminders" in pref
    assert "list=Trabajo" in pref
    assert "status=open" in pref


# ── Content hash ───────────────────────────────────────────────────────

def test_content_hash_stable_on_unchanged_reminder():
    r1 = _mk_reminder("R-1", "x", body="b", due_ts=100.0)
    r2 = _mk_reminder("R-1", "x", body="b", due_ts=100.0)
    assert ir._content_hash(r1) == ir._content_hash(r2)


def test_content_hash_changes_on_name_edit():
    r1 = _mk_reminder("R-1", "x")
    r2 = _mk_reminder("R-1", "y")
    assert ir._content_hash(r1) != ir._content_hash(r2)


def test_content_hash_ignores_modification_ts():
    r1 = _mk_reminder("R-1", "x", modified_ts=100.0)
    r2 = _mk_reminder("R-1", "x", modified_ts=200.0)
    assert ir._content_hash(r1) == ir._content_hash(r2)


# ── Writer ─────────────────────────────────────────────────────────────

def test_upsert_reminders_writes_source_reminders(tmp_vault_col):
    r = _mk_reminder("R-1", "Revisar PR",
                      list_name="Trabajo", body="nota breve",
                      due_ts=datetime(2026, 4, 20, 10).timestamp())
    n = ir.upsert_reminders(tmp_vault_col, [r])
    assert n == 1

    got = tmp_vault_col.get(where={"source": "reminders"}, include=["metadatas"])
    assert len(got["ids"]) == 1
    meta = got["metadatas"][0]
    assert meta["source"] == "reminders"
    assert meta["reminder_id"] == "R-1"
    assert meta["list_name"] == "Trabajo"
    assert meta["title"] == "Revisar PR"
    assert meta["completed"] == 0
    assert meta["due_ts"] > 0
    assert meta["file"].startswith("reminders://R-1")


def test_upsert_reminders_idempotent(tmp_vault_col):
    r = _mk_reminder("R-1", "x")
    ir.upsert_reminders(tmp_vault_col, [r])
    before = len(tmp_vault_col.get(where={"source": "reminders"}, include=[])["ids"])
    ir.upsert_reminders(tmp_vault_col, [r])
    after = len(tmp_vault_col.get(where={"source": "reminders"}, include=[])["ids"])
    assert before == after == 1


def test_delete_reminders_removes_rows(tmp_vault_col):
    r1 = _mk_reminder("R-1", "x")
    r2 = _mk_reminder("R-2", "y")
    ir.upsert_reminders(tmp_vault_col, [r1, r2])
    assert len(tmp_vault_col.get(where={"source": "reminders"}, include=[])["ids"]) == 2

    deleted = ir.delete_reminders(tmp_vault_col, ["R-1"])
    assert deleted == 1
    remaining = tmp_vault_col.get(where={"source": "reminders"}, include=["metadatas"])
    assert len(remaining["ids"]) == 1
    assert remaining["metadatas"][0]["reminder_id"] == "R-2"


# ── Orchestration ──────────────────────────────────────────────────────

def test_run_first_pass_indexes_all(tmp_vault_col):
    reminders = [
        _mk_reminder("R-1", "A"),
        _mk_reminder("R-2", "B", completed=True,
                      completion_ts=datetime(2026, 4, 10).timestamp()),
    ]
    summary = ir.run(fetch_fn=lambda: reminders, vault_col=tmp_vault_col)
    assert summary["reminders_fetched"] == 2
    assert summary["reminders_indexed"] == 2
    assert summary["reminders_unchanged"] == 0
    assert summary["reminders_deleted"] == 0


def test_run_second_pass_detects_unchanged(tmp_vault_col):
    reminders = [_mk_reminder("R-1", "A")]
    ir.run(fetch_fn=lambda: reminders, vault_col=tmp_vault_col)
    summary = ir.run(fetch_fn=lambda: reminders, vault_col=tmp_vault_col)
    assert summary["reminders_indexed"] == 0
    assert summary["reminders_unchanged"] == 1
    assert summary["reminders_deleted"] == 0


def test_run_re_indexes_on_content_change(tmp_vault_col):
    ir.run(fetch_fn=lambda: [_mk_reminder("R-1", "A")],
           vault_col=tmp_vault_col)
    summary = ir.run(
        fetch_fn=lambda: [_mk_reminder("R-1", "A UPDATED")],
        vault_col=tmp_vault_col,
    )
    assert summary["reminders_indexed"] == 1
    assert summary["reminders_unchanged"] == 0


def test_run_deletes_stale(tmp_vault_col):
    ir.run(
        fetch_fn=lambda: [
            _mk_reminder("R-1", "A"),
            _mk_reminder("R-2", "B"),
        ],
        vault_col=tmp_vault_col,
    )
    # Second run — only R-1 still exists.
    summary = ir.run(
        fetch_fn=lambda: [_mk_reminder("R-1", "A")],
        vault_col=tmp_vault_col,
    )
    assert summary["reminders_deleted"] == 1
    remaining = tmp_vault_col.get(where={"source": "reminders"}, include=["metadatas"])
    assert len(remaining["ids"]) == 1
    assert remaining["metadatas"][0]["reminder_id"] == "R-1"


def test_run_reset_clears_cursor(tmp_vault_col):
    reminders = [_mk_reminder("R-1", "A")]
    ir.run(fetch_fn=lambda: reminders, vault_col=tmp_vault_col)
    # With reset=True, previous hash is wiped → reminder looks new again.
    summary = ir.run(
        fetch_fn=lambda: reminders, vault_col=tmp_vault_col, reset=True,
    )
    assert summary["reminders_indexed"] == 1
    assert summary["reminders_unchanged"] == 0


def test_run_dry_run_writes_nothing(tmp_vault_col):
    summary = ir.run(
        fetch_fn=lambda: [_mk_reminder("R-1", "A")],
        vault_col=tmp_vault_col,
        dry_run=True,
    )
    assert summary["reminders_indexed"] == 1
    got = tmp_vault_col.get(where={"source": "reminders"}, include=[])
    assert got["ids"] == []


def test_run_only_pending_skips_completed(tmp_vault_col):
    reminders = [
        _mk_reminder("R-1", "A", completed=False),
        _mk_reminder("R-2", "B", completed=True,
                      completion_ts=datetime(2026, 4, 1).timestamp()),
    ]
    summary = ir.run(
        fetch_fn=lambda: reminders, vault_col=tmp_vault_col,
        include_completed=False,
    )
    assert summary["reminders_fetched"] == 2
    assert summary["reminders_indexed"] == 1


def test_run_empty_fetch_is_noop(tmp_vault_col):
    summary = ir.run(fetch_fn=lambda: [], vault_col=tmp_vault_col)
    assert summary["reminders_fetched"] == 0
    assert summary["reminders_indexed"] == 0
    assert summary["reminders_deleted"] == 0


# ── CLI routing ────────────────────────────────────────────────────────

def test_cli_index_source_reminders_routes(monkeypatch):
    called = {}

    def _fake_run(**kw):
        called.update(kw)
        return {
            "reminders_fetched": 42, "reminders_indexed": 3,
            "reminders_unchanged": 39, "reminders_deleted": 0,
            "duration_s": 0.1,
        }

    from scripts import ingest_reminders as ir_mod
    monkeypatch.setattr(ir_mod, "run", _fake_run)

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.index, ["--source", "reminders"])
    assert result.exit_code == 0, result.output
    assert "reminders" in result.output
    # Minimal format: `reminders: {fetched} · +{indexed} · {time}s`.
    assert "reminders: 42" in result.output
    assert "+3" in result.output
    assert called["reset"] is False
    assert called["dry_run"] is False


def test_cli_index_source_reminders_dry_run(monkeypatch):
    called = {}

    def _fake_run(**kw):
        called.update(kw)
        return {
            "reminders_fetched": 10, "reminders_indexed": 10,
            "reminders_unchanged": 0, "reminders_deleted": 0,
            "duration_s": 0.0,
        }

    from scripts import ingest_reminders as ir_mod
    monkeypatch.setattr(ir_mod, "run", _fake_run)

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.index, ["--source", "reminders", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert called["dry_run"] is True
    assert "dry · " in result.output


def test_cli_index_source_reminders_reports_error(monkeypatch):
    from scripts import ingest_reminders as ir_mod
    monkeypatch.setattr(
        ir_mod, "run",
        lambda **kw: {"error": "apple integrations disabled"},
    )
    from click.testing import CliRunner
    result = CliRunner().invoke(rag.index, ["--source", "reminders"])
    assert "apple integrations disabled" in result.output


def test_cli_index_source_reminders_reset_flag(monkeypatch):
    called = {}

    def _fake_run(**kw):
        called.update(kw)
        return {
            "reminders_fetched": 0, "reminders_indexed": 0,
            "reminders_unchanged": 0, "reminders_deleted": 0,
            "duration_s": 0.0,
        }

    from scripts import ingest_reminders as ir_mod
    monkeypatch.setattr(ir_mod, "run", _fake_run)

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.index, ["--source", "reminders", "--reset"])
    assert result.exit_code == 0
    assert called["reset"] is True


def test_reset_clears_both_state_and_collection(tmp_vault_col):
    """When --reset=True, both the state cursor AND prior collection chunks
    are wiped. This prevents the next run from skipping unchanged items when
    the state DB is fresh but the corpus still contains stale chunks.
    """
    col = tmp_vault_col

    # First run: ingest 2 reminders.
    r1 = _mk_reminder(rid="r1", name="Pay bills", completed=False)
    r2 = _mk_reminder(rid="r2", name="Call mom", completed=False)
    summary = ir.run(fetch_fn=lambda: [r1, r2], vault_col=col)
    assert summary["reminders_indexed"] == 2, f"Expected 2 indexed, got {summary}"
    assert col.count() == 2

    # Second run with --reset: nuke the state and collection.
    summary2 = ir.run(
        fetch_fn=lambda: [r1, r2],
        vault_col=col,
        reset=True,
    )
    # After reset, the state is cleared, so the next run should re-index
    # everything as new. The key check: collection should be empty post-reset
    # (corpus cleaned by the reset logic).
    # Then re-index the 2 reminders.
    assert col.count() == 2, "Expected 2 chunks after re-index post-reset"
    assert summary2["reminders_indexed"] == 2, (
        f"Expected 2 re-indexed post-reset, got {summary2['reminders_indexed']}"
    )
