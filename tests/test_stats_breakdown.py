"""Tests for `_corpus_breakdown_by_source` + `rag stats` per-source table.

The helper introspects the real sqlite-vec DB and state tables — tests
build minimal fixtures to exercise happy path + edge cases (missing
state tables, empty corpus, pre-schema vault-only data).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import rag


def _seed_db(db_path: Path, *, rows: list[tuple[str, str | None]]):
    """Build a minimal sqlite file with `meta_<COLLECTION_NAME>` populated.

    rows: list of (chunk_id, source). `source=None` exercises the pre-schema
    legacy path where meta lacked the source column.
    """
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    col = rag.COLLECTION_NAME
    cur.execute(
        f'CREATE TABLE "meta_{col}" '
        f'(rowid INTEGER PRIMARY KEY, chunk_id TEXT, source TEXT)'
    )
    for i, (cid, src) in enumerate(rows):
        cur.execute(
            f'INSERT INTO "meta_{col}" (rowid, chunk_id, source) VALUES (?, ?, ?)',
            (i + 1, cid, src),
        )
    con.commit()
    con.close()


def _seed_state_table(db_path: Path, table: str, col: str, ts: str):
    con = sqlite3.connect(str(db_path))
    con.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ("{col}" TEXT)')
    con.execute(f'INSERT INTO "{table}" ("{col}") VALUES (?)', (ts,))
    con.commit()
    con.close()


@pytest.fixture
def tmp_ragvec_db(tmp_path, monkeypatch) -> Path:
    """Redirect rag.DB_PATH to a tmp dir so tests don't touch the real DB."""
    tmp_dir = tmp_path / "ragvec"
    tmp_dir.mkdir()
    monkeypatch.setattr(rag, "DB_PATH", tmp_dir)
    return tmp_dir / "ragvec.db"


def test_breakdown_vault_only_emits_single_row(tmp_ragvec_db):
    """Corpus 100% vault (legacy source=None) → 1 row labeled 'vault'."""
    _seed_db(tmp_ragvec_db, rows=[
        ("chunk1", None), ("chunk2", None), ("chunk3", None),
    ])
    out = rag._corpus_breakdown_by_source()
    assert len(out) == 1
    src, cnt, last_ts = out[0]
    # COALESCE(source, "vault") → legacy nulls get labeled "vault"
    assert src == "vault"
    assert cnt == 3
    assert last_ts == "—", "no state table configured → '—'"


def test_breakdown_mixed_sources_sorted_by_count(tmp_ragvec_db):
    """Multi-source corpus returns rows sorted DESC by count."""
    rows = (
        [("v" + str(i), "vault") for i in range(5)]
        + [("w" + str(i), "whatsapp") for i in range(10)]
        + [("g" + str(i), "gmail") for i in range(2)]
    )
    _seed_db(tmp_ragvec_db, rows=rows)
    out = rag._corpus_breakdown_by_source()
    sources_in_order = [src for src, _, _ in out]
    counts_in_order = [cnt for _, cnt, _ in out]
    assert sources_in_order == ["whatsapp", "vault", "gmail"]
    assert counts_in_order == [10, 5, 2]


def test_breakdown_pulls_last_indexed_from_state_table(tmp_ragvec_db):
    """last_indexed comes from source-specific state tables."""
    _seed_db(tmp_ragvec_db, rows=[("w1", "whatsapp"), ("g1", "gmail")])
    _seed_state_table(
        tmp_ragvec_db, "rag_whatsapp_state", "updated_at", "2026-04-21T12:50:00",
    )
    _seed_state_table(
        tmp_ragvec_db, "rag_gmail_state", "updated_at", "2026-04-21T11:04:00",
    )
    out = {src: last_ts for src, _, last_ts in rag._corpus_breakdown_by_source()}
    assert out["whatsapp"].startswith("2026-04-21T12:50")
    assert out["gmail"].startswith("2026-04-21T11:04")


def test_breakdown_no_meta_table_returns_empty(tmp_ragvec_db):
    """No meta_<COLLECTION_NAME> at all → empty result, no crash."""
    # Create DB file but no meta table
    con = sqlite3.connect(str(tmp_ragvec_db))
    con.execute("CREATE TABLE unrelated (x INTEGER)")
    con.commit()
    con.close()
    out = rag._corpus_breakdown_by_source()
    assert out == []


def test_breakdown_missing_state_table_shows_dash(tmp_ragvec_db):
    """Source without a state table (vault) → last_indexed = '—' (no crash
    even though the query fails on the missing table)."""
    _seed_db(tmp_ragvec_db, rows=[("v1", "vault")])
    # Deliberately NO rag_*_state tables created
    out = rag._corpus_breakdown_by_source()
    assert out == [("vault", 1, "—")]


def test_breakdown_silent_fail_on_unreadable_db(tmp_path, monkeypatch):
    """DB path pointing at nothing → returns [] without raising."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "no-such-dir")
    # DB_PATH / ragvec.db doesn't exist; sqlite3 can't open in ro mode
    out = rag._corpus_breakdown_by_source()
    assert out == []


# ── CLI: rag stats ─────────────────────────────────────────────────────────

def test_stats_renders_breakdown_when_multiple_sources(tmp_ragvec_db, monkeypatch):
    """`rag stats` must include the 'Por source' table when ≥2 sources."""
    _seed_db(tmp_ragvec_db, rows=[
        ("v1", "vault"), ("v2", "vault"),
        ("w1", "whatsapp"), ("w2", "whatsapp"), ("w3", "whatsapp"),
    ])
    # Stub everything else stats touches (ollama, embeddings, etc.)
    monkeypatch.setattr(rag, "get_db", lambda: type("C", (), {"count": lambda self: 5})())
    monkeypatch.setattr(rag, "get_urls_db", lambda: type("C", (), {"count": lambda self: 0})())
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "qwen2.5:7b")
    monkeypatch.setattr(rag, "feedback_counts", lambda: (0, 0))

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.stats, [])
    assert result.exit_code == 0, result.output
    # Section header + both sources + counts present
    assert "Por source" in result.output
    assert "whatsapp" in result.output
    assert "vault" in result.output
    assert "3" in result.output  # whatsapp count
    assert "2" in result.output  # vault count


def test_stats_skips_breakdown_when_vault_only(tmp_ragvec_db, monkeypatch):
    """Single-source corpus → no breakdown table (noise avoidance)."""
    _seed_db(tmp_ragvec_db, rows=[("v1", "vault"), ("v2", "vault")])
    monkeypatch.setattr(rag, "get_db", lambda: type("C", (), {"count": lambda self: 2})())
    monkeypatch.setattr(rag, "get_urls_db", lambda: type("C", (), {"count": lambda self: 0})())
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "qwen2.5:7b")
    monkeypatch.setattr(rag, "feedback_counts", lambda: (0, 0))

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.stats, [])
    assert result.exit_code == 0, result.output
    assert "Por source" not in result.output
