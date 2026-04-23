"""Feature #15 del 2026-04-23 — `rag hygiene` vault dashboard tests.

Validates:
- _hygiene_scan deduplicates by file (chunks of same note count once)
- Categories detect correctly: sin_tags, sin_outlinks, huerfanas,
  vacias, stale, con_wip
- Cross-source pseudo-paths (whatsapp://, gmail://) skipped
- Sample size respected
- CLI renders table + --as-json works
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import rag


class _FakeCol:
    """Mimics SqliteVecCollection.count() + .get()."""
    def __init__(self, metas: list[dict], docs: list[str]):
        self._metas = metas
        self._docs = docs
        self.id = "test-uuid"

    def count(self) -> int:
        return len(self._docs)

    def get(self, include=None):
        return {
            "documents": self._docs,
            "ids": [f"id_{i}" for i in range(len(self._docs))],
            "metadatas": self._metas,
        }


def _patch_corpus(monkeypatch, metas: list[dict], docs: list[str]):
    """Bypass _load_corpus and feed known test data."""
    def fake_load_corpus(col):
        return {
            "docs": docs,
            "metas": metas,
            "count": len(docs),
            "collection_id": "test",
        }
    monkeypatch.setattr(rag, "_load_corpus", fake_load_corpus)
    monkeypatch.setattr(rag, "get_db", lambda: _FakeCol(metas, docs))


# ── _hygiene_scan categories ─────────────────────────────────────────────


def test_empty_corpus_gracefully_returns_zeros(monkeypatch):
    _patch_corpus(monkeypatch, [], [])
    r = rag._hygiene_scan(rag.get_db())
    assert r["total_notes"] == 0
    assert all(v == 0 for v in r["counts"].values())


def test_sin_tags_detected(monkeypatch):
    metas = [
        {"file": "a.md", "tags": "", "outlinks": "b.md",
         "modified": "2026-04-23T10:00:00"},
        {"file": "b.md", "tags": "proyecto", "outlinks": "a.md",
         "modified": "2026-04-23T10:00:00"},
    ]
    docs = ["a content here" * 10, "b content " * 10]
    _patch_corpus(monkeypatch, metas, docs)
    r = rag._hygiene_scan(rag.get_db())
    assert r["counts"]["sin_tags"] == 1
    assert "a.md" in r["samples"]["sin_tags"]


def test_sin_outlinks_detected(monkeypatch):
    metas = [
        {"file": "a.md", "tags": "x", "outlinks": "",
         "modified": "2026-04-23T10:00:00"},
        {"file": "b.md", "tags": "x", "outlinks": "a",
         "modified": "2026-04-23T10:00:00"},
    ]
    docs = ["aaaa" * 50, "bbbb" * 50]
    _patch_corpus(monkeypatch, metas, docs)
    r = rag._hygiene_scan(rag.get_db())
    assert r["counts"]["sin_outlinks"] == 1
    assert "a.md" in r["samples"]["sin_outlinks"]


def test_huerfanas_detected(monkeypatch):
    # a.md linked by b. c.md linked by nobody → huérfana.
    metas = [
        {"file": "a.md", "tags": "x", "outlinks": "",
         "modified": "2026-04-23T10:00:00"},
        {"file": "b.md", "tags": "x", "outlinks": "a",
         "modified": "2026-04-23T10:00:00"},
        {"file": "c.md", "tags": "x", "outlinks": "a",
         "modified": "2026-04-23T10:00:00"},
    ]
    docs = ["aa" * 100] * 3
    _patch_corpus(monkeypatch, metas, docs)
    r = rag._hygiene_scan(rag.get_db())
    # b.md NOT linked (nobody links to it), c.md NOT linked.
    # a.md linked by b+c.
    orphans = r["samples"]["huerfanas"]
    assert "b.md" in orphans
    assert "c.md" in orphans
    assert "a.md" not in orphans


def test_vacias_below_threshold(monkeypatch):
    metas = [
        {"file": "tiny.md", "tags": "x", "outlinks": "",
         "modified": "2026-04-23T10:00:00"},
        {"file": "big.md", "tags": "x", "outlinks": "",
         "modified": "2026-04-23T10:00:00"},
    ]
    docs = ["small", "x" * 200]
    _patch_corpus(monkeypatch, metas, docs)
    r = rag._hygiene_scan(rag.get_db(), empty_threshold=50)
    assert "tiny.md" in r["samples"]["vacias"]
    assert "big.md" not in r["samples"]["vacias"]


def test_stale_detected(monkeypatch):
    old_date = (datetime.now() - timedelta(days=200)).isoformat(timespec="seconds")
    new_date = datetime.now().isoformat(timespec="seconds")
    metas = [
        {"file": "old.md", "tags": "x", "outlinks": "new",
         "modified": old_date},
        {"file": "new.md", "tags": "x", "outlinks": "old",
         "modified": new_date},
    ]
    docs = ["x" * 100, "x" * 100]
    _patch_corpus(monkeypatch, metas, docs)
    r = rag._hygiene_scan(rag.get_db(), stale_days=180)
    assert "old.md" in r["samples"]["stale"]
    assert "new.md" not in r["samples"]["stale"]


def test_wip_markers_detected(monkeypatch):
    metas = [
        {"file": "wip.md", "tags": "x", "outlinks": "",
         "modified": "2026-04-23T10:00:00"},
        {"file": "clean.md", "tags": "x", "outlinks": "",
         "modified": "2026-04-23T10:00:00"},
    ]
    docs = [
        "body con TODO: completar esto" * 5,
        "body sin marcadores de work" * 5,
    ]
    _patch_corpus(monkeypatch, metas, docs)
    r = rag._hygiene_scan(rag.get_db())
    assert "wip.md" in r["samples"]["con_wip"]
    assert "clean.md" not in r["samples"]["con_wip"]


def test_cross_source_paths_skipped(monkeypatch):
    """whatsapp://, gmail://, calendar:// paths not scanned for hygiene."""
    metas = [
        {"file": "whatsapp://jid/msg1", "tags": "", "outlinks": "",
         "modified": "2026-04-23T10:00:00"},
        {"file": "real-note.md", "tags": "", "outlinks": "",
         "modified": "2026-04-23T10:00:00"},
    ]
    docs = ["x" * 100, "y" * 100]
    _patch_corpus(monkeypatch, metas, docs)
    r = rag._hygiene_scan(rag.get_db())
    assert r["total_notes"] == 1  # only real-note.md
    assert "real-note.md" in r["samples"]["sin_tags"]


def test_dedup_multiple_chunks_one_note(monkeypatch):
    """A note with 3 chunks still counts as 1 note."""
    metas = [
        {"file": "multi.md", "tags": "x", "outlinks": "",
         "modified": "2026-04-23T10:00:00"},
        {"file": "multi.md", "tags": "x", "outlinks": "",
         "modified": "2026-04-23T10:00:00"},
        {"file": "multi.md", "tags": "x", "outlinks": "",
         "modified": "2026-04-23T10:00:00"},
    ]
    docs = ["chunk1" * 30, "chunk2" * 30, "chunk3" * 30]
    _patch_corpus(monkeypatch, metas, docs)
    r = rag._hygiene_scan(rag.get_db())
    assert r["total_notes"] == 1


def test_sample_size_respected(monkeypatch):
    """--sample 2 → max 2 paths per category."""
    metas = [
        {"file": f"note{i}.md", "tags": "", "outlinks": "",
         "modified": "2026-04-23T10:00:00"}
        for i in range(10)
    ]
    docs = ["x" * 100] * 10
    _patch_corpus(monkeypatch, metas, docs)
    r = rag._hygiene_scan(rag.get_db(), sample_size=2)
    assert r["counts"]["sin_tags"] == 10
    assert len(r["samples"]["sin_tags"]) == 2


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_hygiene_as_json(monkeypatch):
    metas = [
        {"file": "a.md", "tags": "", "outlinks": "",
         "modified": "2026-04-23T10:00:00"},
    ]
    docs = ["x" * 100]
    _patch_corpus(monkeypatch, metas, docs)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["hygiene", "--as-json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip().splitlines()[-1])
    assert data["total_notes"] == 1
    assert data["counts"]["sin_tags"] == 1


def test_cli_hygiene_renders_summary(monkeypatch):
    metas = [{"file": "a.md", "tags": "", "outlinks": "",
              "modified": "2026-04-23T10:00:00"}]
    docs = ["x" * 100]
    _patch_corpus(monkeypatch, metas, docs)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["hygiene", "--sample", "3"])
    assert result.exit_code == 0, result.output
    assert "Vault hygiene" in result.output
    assert "Sin tags" in result.output


def test_cli_hygiene_empty_vault(monkeypatch):
    _patch_corpus(monkeypatch, [], [])
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["hygiene"])
    assert result.exit_code == 0
    assert "0 notas indexadas" in result.output or "Vault hygiene" in result.output
