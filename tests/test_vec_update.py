"""Tests for SqliteVecCollection.update() — metadata-only update path used
by `_maybe_backfill_created_ts` and future chromadb-era callers.

Regression: pre-fix, `col.update(ids=[...], metadatas=[...])` raised
`AttributeError: 'SqliteVecCollection' object has no attribute 'update'`,
breaking the lazy backfill of `created_ts` on vaults indexed before the
temporal-retrieval feature landed (silently — only surfaced when users
ran date-filtered queries).
"""
from __future__ import annotations

import pytest

import rag


@pytest.fixture
def col(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    client = rag.SqliteVecClient(path=str(tmp_path))
    collection = client.get_or_create_collection("test_update")
    # Seed three chunks so we can exercise partial updates + bulk batches.
    collection.add(
        ids=["c1", "c2", "c3"],
        embeddings=[[0.1] * 8, [0.2] * 8, [0.3] * 8],
        documents=["doc one", "doc two", "doc three"],
        metadatas=[
            {"file": "a.md", "folder": "01-Projects", "tags": "alpha"},
            {"file": "b.md", "folder": "02-Areas", "tags": "beta"},
            {"file": "c.md", "folder": "03-Resources", "tags": "gamma"},
        ],
    )
    return collection


def _fetch_meta(collection, cid: str) -> dict:
    """Helper: pull back a chunk's metadata via the public get() API."""
    res = collection.get(ids=[cid], include=["metadatas"])
    metas = res.get("metadatas") or []
    assert metas, f"chunk {cid!r} not found"
    return metas[0]


def test_update_sets_first_class_column(col):
    """created_ts is a first-class column — update should write it."""
    col.update(ids=["c1"], metadatas=[{"file": "a.md", "created_ts": 1234567890.0}])
    meta = _fetch_meta(col, "c1")
    assert meta["created_ts"] == 1234567890.0
    # Untouched columns persist.
    assert meta["folder"] == "01-Projects"
    assert meta["tags"] == "alpha"


def test_update_ignores_empty_ids(col):
    col.update(ids=[], metadatas=[])
    # Should be a no-op, no crash.
    meta = _fetch_meta(col, "c1")
    assert meta["file"] == "a.md"


def test_update_preserves_embedding_and_document(col):
    """Metadata-only update must NOT touch the vec0 row nor the document."""
    before = col.get(ids=["c2"], include=["documents"])
    col.update(ids=["c2"], metadatas=[{"file": "b.md", "folder": "99-Archive"}])
    after = col.get(ids=["c2"], include=["documents"])
    assert after["documents"][0] == before["documents"][0] == "doc two"
    # And the vec row is still queryable with the original embedding vector.
    hits = col.query(query_embeddings=[[0.2] * 8], n_results=1)
    assert hits["ids"][0] == ["c2"]


def test_update_coerces_created_ts_string_to_float(col):
    col.update(ids=["c3"], metadatas=[{"file": "c.md", "created_ts": "1234567890"}])
    meta = _fetch_meta(col, "c3")
    assert isinstance(meta["created_ts"], float)
    assert meta["created_ts"] == 1234567890.0


def test_update_noop_when_meta_has_no_known_cols(col):
    """Passing only extras (no known cols) still succeeds + rewrites extra_json."""
    col.update(ids=["c1"], metadatas=[{"custom_key": "hello"}])
    meta = _fetch_meta(col, "c1")
    assert meta.get("custom_key") == "hello"
    # Known cols are preserved.
    assert meta["file"] == "a.md"


def test_update_skips_none_meta(col):
    """`None` meta entries are skipped silently; others still apply."""
    col.update(
        ids=["c1", "c2"],
        metadatas=[None, {"file": "b.md", "folder": "new-folder"}],
    )
    assert _fetch_meta(col, "c1")["folder"] == "01-Projects"
    assert _fetch_meta(col, "c2")["folder"] == "new-folder"


def test_maybe_backfill_created_ts_no_longer_raises(tmp_path, monkeypatch):
    """Integration: _maybe_backfill_created_ts ran a chromadb-era
    `col.update(ids=..., metadatas=...)`; pre-fix it swallowed an
    AttributeError via the caller's broad except (`[yellow]Backfill
    created_ts falló: 'SqliteVecCollection' object has no attribute
    'update'[/yellow]`), so the backfill never actually ran. With the
    new method in place, the call succeeds and populates `created_ts`
    on chunks that were missing it.
    """
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    vault = tmp_path / "vault"
    (vault / "02-Areas").mkdir(parents=True)
    note = vault / "02-Areas" / "old.md"
    note.write_text("# old note\n\nalgun cuerpo para indexar.\n", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", vault)

    client = rag.SqliteVecClient(path=str(tmp_path))
    collection = client.get_or_create_collection(rag.COLLECTION_NAME)
    # Index the chunk deliberately WITHOUT created_ts — this is the shape
    # old vaults have in ragvec.db before the temporal feature landed.
    collection.add(
        ids=["old#0"],
        embeddings=[[0.05] * 8],
        documents=["algun cuerpo para indexar."],
        metadatas=[{"file": "02-Areas/old.md", "folder": "02-Areas"}],
    )
    monkeypatch.setattr(rag, "_CREATED_TS_BACKFILL_DONE", False)
    # Must not raise and must not log "falló" — the chunk should have
    # a created_ts after backfill.
    rag._maybe_backfill_created_ts()
    meta = _fetch_meta(collection, "old#0")
    assert meta.get("created_ts") is not None
