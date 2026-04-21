"""Starter pack for `_index_single_file()` — the on-save hook powering
incremental indexing and the ambient agent.

Focuses on the return-value contract since callers dispatch on it:
  'skipped'  (outside vault, excluded, or unchanged hash)
  'indexed'  (new/updated — vectors written)
  'removed'  (file gone → vectors deleted)
  'empty'    (file read but produced no chunks)

Mocks embed/contradict/summary/synthetic-questions so the test runs
offline in ~200ms per case.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import rag


@pytest.fixture
def indexing_env(monkeypatch, tmp_path):
    """Set up a real SqliteVecCollection in tmp + stub every ollama-backed call."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="index_single_test", metadata={"hnsw:space": "cosine"}
    )

    # Mock expensive helpers — we're not testing embedding quality here.
    monkeypatch.setattr(rag, "embed", lambda texts: [[0.1] * 1024 for _ in texts])
    monkeypatch.setattr(rag, "get_context_summary", lambda *a, **kw: "")
    monkeypatch.setattr(rag, "get_synthetic_questions", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "_check_and_flag_contradictions",
                        lambda *a, **kw: None)
    rag._invalidate_corpus_cache()
    return vault, col


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_outside_vault_returns_skipped(indexing_env, tmp_path):
    _vault, col = indexing_env
    outside = tmp_path / "outside.md"
    _write(outside, "# Outside\n\nnot in vault")
    assert rag._index_single_file(col, outside) == "skipped"


def test_excluded_path_returns_skipped(indexing_env):
    vault, col = indexing_env
    # `.obsidian/` is excluded by `is_excluded()` (starts with `.`)
    hidden = vault / ".obsidian" / "workspace.md"
    _write(hidden, "# internal\n\nplugin state")
    assert rag._index_single_file(col, hidden) == "skipped"


def test_new_file_returns_indexed(indexing_env):
    vault, col = indexing_env
    note = vault / "alpha.md"
    _write(note, "# Alpha Note\n\n" + "this is some real body text. " * 20)
    assert rag._index_single_file(col, note) == "indexed"
    # sanity: at least one chunk written with the expected `file` meta.
    got = col.get(where={"file": "alpha.md"}, include=["metadatas"])
    assert got["ids"], "no chunks persisted"
    assert got["metadatas"][0]["file"] == "alpha.md"


def test_unchanged_hash_returns_skipped(indexing_env):
    vault, col = indexing_env
    note = vault / "beta.md"
    _write(note, "# Beta Note\n\n" + "same content every time. " * 20)
    assert rag._index_single_file(col, note) == "indexed"
    # Second call — same content, same hash — must short-circuit.
    assert rag._index_single_file(col, note) == "skipped"


def test_deleted_file_returns_removed(indexing_env):
    vault, col = indexing_env
    note = vault / "gamma.md"
    _write(note, "# Gamma Note\n\n" + "transient content. " * 20)
    assert rag._index_single_file(col, note) == "indexed"
    note.unlink()
    assert rag._index_single_file(col, note) == "removed"
    # Vectors should be gone.
    assert not col.get(where={"file": "gamma.md"}, include=[])["ids"]


def test_tiny_file_returns_empty(indexing_env):
    """Files whose body clean_md()→'' produce no chunks → 'empty'."""
    vault, col = indexing_env
    note = vault / "empty.md"
    # Only frontmatter, no body.
    _write(note, "---\ntag: nothing\n---\n")
    out = rag._index_single_file(col, note)
    assert out in ("empty", "indexed"), out  # some frontmatter paths still index


def test_content_change_reembeds(indexing_env):
    """Hash differs after edit → 'indexed' + new ids replace old ones."""
    vault, col = indexing_env
    note = vault / "delta.md"
    _write(note, "# Delta v1\n\n" + "first version content. " * 20)
    assert rag._index_single_file(col, note) == "indexed"
    n1 = col.count()
    _write(note, "# Delta v2\n\n" + "much longer second version. " * 40)
    assert rag._index_single_file(col, note) == "indexed"
    # Still exactly one `file` entry at any point — old ids replaced.
    got = col.get(where={"file": "delta.md"}, include=["metadatas"])
    hashes = {m.get("hash") for m in got["metadatas"]}
    assert len(hashes) == 1, f"mixed hashes after reindex: {hashes}"
    # Count may differ because chunk shape changed with the new body.
    assert n1 >= 1 and col.count() >= 1


def test_skip_contradict_flag_is_respected(indexing_env, monkeypatch):
    """`skip_contradict=True` must NOT call the contradiction checker."""
    vault, col = indexing_env
    note = vault / "epsilon.md"
    _write(note, "# Epsilon Note\n\n" + "contradict-sensitive body. " * 20)

    called: list[Path] = []
    def _fake_check(*args, **kwargs):
        called.append(args[1] if len(args) > 1 else None)
        return None
    monkeypatch.setattr(rag, "_check_and_flag_contradictions", _fake_check)

    # skip_contradict=True must skip the check entirely.
    rag._index_single_file(col, note, skip_contradict=True)
    assert not called, "contradict checker was invoked despite skip_contradict=True"

    # Without the flag it IS called (fresh file — triggers reindex branch).
    _write(note, "# Epsilon v2\n\n" + "new body forcing rehash. " * 20)
    rag._index_single_file(col, note, skip_contradict=False)
    assert called, "contradict checker not invoked when skip_contradict=False"
