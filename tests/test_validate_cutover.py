"""Tests for `rag maintenance --validate-cutover`.

Read-only audit comparing SQL row counts vs the source JSONL line
count of the most recent .bak.<ts>. Purpose: gate before T10 strips
the JSONL fallback. SQL >= JSONL = safe; SQL < JSONL = lost rows.

Note: RAG_STATE_SQL was removed 2026-05-04 (post-T10 SQL is the only
path). Tests that exercised the flag-off branch were deleted; the
remaining tests exercise the SQL path directly.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import rag


# ── _count_jsonl_lines ────────────────────────────────────────────────────────


def test_count_jsonl_lines_basic(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text('{"a":1}\n{"b":2}\n{"c":3}\n', encoding="utf-8")
    assert rag._count_jsonl_lines(p) == 3


def test_count_jsonl_lines_ignores_blanks(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text('{"a":1}\n\n{"b":2}\n\n\n', encoding="utf-8")
    assert rag._count_jsonl_lines(p) == 2


def test_count_jsonl_lines_missing_file_returns_zero(tmp_path):
    assert rag._count_jsonl_lines(tmp_path / "never") == 0


def test_count_jsonl_lines_empty_file(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text("", encoding="utf-8")
    assert rag._count_jsonl_lines(p) == 0


# ── _validate_cutover_state ──────────────────────────────────────────────────


def _seed_bak(state_dir: Path, jsonl: str, lines: int, ts: int = 1700000000) -> Path:
    bak = state_dir / f"{jsonl}.bak.{ts}"
    bak.write_text("\n".join(f'{{"i":{i}}}' for i in range(lines)) + "\n", encoding="utf-8")
    return bak


def test_validate_cutover_missing_bak_reports_no_bak(tmp_path, monkeypatch):
    """A source with no .bak.<ts> file → status='no_bak' (migration
    hasn't happened yet, or the source never had a JSONL)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".local/share/obsidian-rag").mkdir(parents=True)

    results = rag._validate_cutover_state()
    # Every seeded source should be no_bak since we didn't create any.
    assert all(r["status"] == "no_bak" for r in results), (
        f"expected all no_bak, got statuses: {[(r['source'], r['status']) for r in results]}"
    )


def test_validate_cutover_sql_greater_than_jsonl_is_ok(tmp_path, monkeypatch):
    """Post-cutover activity accumulates → SQL rows grow past the
    snapshot .bak. That's the happy path."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    state_dir = tmp_path / ".local/share/obsidian-rag"
    state_dir.mkdir(parents=True)
    _seed_bak(state_dir, "queries.jsonl", 100)

    import contextlib

    class FakeCursor:
        def __init__(self, count): self._count = count
        def fetchone(self): return (self._count,)

    class FakeConn:
        def __init__(self, count): self._count = count
        def execute(self, sql):
            return FakeCursor(self._count)
        def close(self): pass

    @contextlib.contextmanager
    def fake_conn_ctx():
        yield FakeConn(count=150)

    with patch.object(rag, "_ragvec_state_conn", fake_conn_ctx):
        results = rag._validate_cutover_state()

    q = [r for r in results if r["source"] == "queries.jsonl"][0]
    assert q["status"] == "ok"
    assert q["bak_lines"] == 100
    assert q["sql_rows"] == 150
    assert q["delta_pct"] == 50.0


def test_validate_cutover_sql_slightly_less_is_warn(tmp_path, monkeypatch):
    """SQL within 1% below JSONL is tolerated (migration drops a
    handful of malformed pre-existing rows — this is documented)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    state_dir = tmp_path / ".local/share/obsidian-rag"
    state_dir.mkdir(parents=True)
    _seed_bak(state_dir, "queries.jsonl", 1000)

    import contextlib

    @contextlib.contextmanager
    def fake_conn_ctx():
        class C:
            def execute(self, sql):
                class R: pass
                r = R()
                r.fetchone = lambda: (995,)  # -0.5% delta → warn
                return r
            def close(self): pass
        yield C()

    with patch.object(rag, "_ragvec_state_conn", fake_conn_ctx):
        results = rag._validate_cutover_state()

    q = [r for r in results if r["source"] == "queries.jsonl"][0]
    assert q["status"] == "warn"
    assert q["delta_pct"] == -0.5


def test_validate_cutover_sql_way_less_is_fail(tmp_path, monkeypatch):
    """SQL significantly below JSONL → status='fail'. The maintenance
    command returns exit 1 in this case so automated cron can notice."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    state_dir = tmp_path / ".local/share/obsidian-rag"
    state_dir.mkdir(parents=True)
    _seed_bak(state_dir, "queries.jsonl", 1000)

    import contextlib

    @contextlib.contextmanager
    def fake_conn_ctx():
        class C:
            def execute(self, sql):
                class R: pass
                r = R()
                r.fetchone = lambda: (500,)  # -50% delta → fail
                return r
            def close(self): pass
        yield C()

    with patch.object(rag, "_ragvec_state_conn", fake_conn_ctx):
        results = rag._validate_cutover_state()

    q = [r for r in results if r["source"] == "queries.jsonl"][0]
    assert q["status"] == "fail"
    assert q["delta_pct"] == -50.0


def test_validate_cutover_uses_newest_bak(tmp_path, monkeypatch):
    """If multiple .bak.<ts> exist (e.g. migration re-run), pick the
    newest by timestamp suffix. sorted() handles this as long as all
    timestamps are numeric + same width."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    state_dir = tmp_path / ".local/share/obsidian-rag"
    state_dir.mkdir(parents=True)
    _seed_bak(state_dir, "queries.jsonl", 100, ts=1700000000)
    _seed_bak(state_dir, "queries.jsonl", 200, ts=1700000500)  # newer

    import contextlib

    @contextlib.contextmanager
    def fake_conn_ctx():
        class C:
            def execute(self, sql):
                class R: pass
                r = R()
                r.fetchone = lambda: (200,)
                return r
            def close(self): pass
        yield C()

    with patch.object(rag, "_ragvec_state_conn", fake_conn_ctx):
        results = rag._validate_cutover_state()

    q = [r for r in results if r["source"] == "queries.jsonl"][0]
    assert q["bak_lines"] == 200, "must pick the newest .bak by ts suffix"
    assert q["bak_file"].endswith(".bak.1700000500")
