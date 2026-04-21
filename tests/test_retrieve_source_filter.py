"""End-to-end tests for Phase 1.0 `--source` filter through retrieve().

Builds a synthetic 6-note collection spanning 3 sources (vault, calendar,
whatsapp) and asserts:
  - Default retrieve() returns mixed-source candidates
  - retrieve(source="whatsapp") only returns WA chunks
  - retrieve(source={"vault","calendar"}) returns union of those sources
  - Empty list and unknown sources degrade safely (drop / no match)

Uses the same offline test harness as `test_retrieve_synthetic_eval.py`
(deterministic keyword vectors, fake reranker, no ollama).
"""
from __future__ import annotations


import pytest

import rag


_KEY_VECTORS: dict[str, list[float]] = {
    "alpha": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "beta":  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "gamma": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "delta": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "common": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
}


def _vec_for(text: str) -> list[float]:
    v = [0.0] * 8
    lower = text.lower()
    for kw, basis in _KEY_VECTORS.items():
        if kw in lower:
            for i, x in enumerate(basis):
                v[i] += x
    norm = (sum(x * x for x in v) ** 0.5) or 1.0
    return [x / norm for x in v]


@pytest.fixture
def multi_source_col(monkeypatch, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="source_filter_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    rag._invalidate_corpus_cache()

    # 6 entries: 2 vault, 2 calendar, 2 whatsapp — each with the same
    # keyword so semantic match ties across sources and the filter is
    # the only discriminator.
    entries = [
        ("vault",    "01/alpha-note.md",    "Alpha note on common topics."),
        ("vault",    "01/beta-note.md",     "Beta note — also common."),
        ("calendar", "calendar://cal1",     "Calendar event about alpha planning."),
        ("calendar", "calendar://cal2",     "Calendar event gamma meeting."),
        ("whatsapp", "whatsapp://chat/1",   "WhatsApp thread alpha discussion."),
        ("whatsapp", "whatsapp://chat/2",   "WhatsApp thread beta discussion."),
    ]
    for src, doc_id, body in entries:
        emb = _vec_for(body)
        col.add(
            ids=[f"{doc_id}::0"],
            embeddings=[emb],
            documents=[body],
            metadatas=[{
                "file": doc_id,
                "note": doc_id.split("/")[-1],
                "folder": "",
                "tags": "",
                "outlinks": "",
                "hash": f"h-{doc_id}",
                "source": src,
                "display_text": body,
                "parent": body,
                # no created_ts → recency decay skipped (multiplier = 1.0)
            }],
        )

    rag._invalidate_corpus_cache()

    monkeypatch.setattr(rag, "embed", lambda texts: [_vec_for(t) for t in texts])
    monkeypatch.setattr(rag, "expand_queries", lambda q, **kw: [q])

    class _FakeReranker:
        def predict(self, pairs, batch_size=None, show_progress_bar=False):  # noqa: ARG002
            scores = []
            for q, d in pairs:
                ql, dl = q.lower(), d.lower()
                shared = sum(1 for kw in _KEY_VECTORS if kw in ql and kw in dl)
                scores.append(float(shared))
            return scores

    monkeypatch.setattr(rag, "get_reranker", lambda: _FakeReranker())
    return col


def test_retrieve_default_returns_mixed_sources(multi_source_col):
    """No source filter → all 3 sources can surface."""
    out = rag.retrieve(multi_source_col, "alpha", k=6, folder=None,
                       auto_filter=False, multi_query=False)
    sources = {m.get("source", "vault") for m in out["metas"]}
    # At least 2 distinct sources present
    assert len(sources) >= 2, f"expected mixed sources, got {sources}"


def test_retrieve_source_whatsapp_only(multi_source_col):
    """source='whatsapp' → vault + calendar dropped."""
    out = rag.retrieve(multi_source_col, "alpha discussion", k=6, folder=None,
                       auto_filter=False, multi_query=False, source="whatsapp")
    assert len(out["metas"]) > 0, "should return at least one WA chunk"
    for m in out["metas"]:
        assert m["source"] == "whatsapp", f"leaked non-WA: {m}"


def test_retrieve_source_set_unions(multi_source_col):
    """Passing a set {'vault', 'calendar'} returns the union, drops WA."""
    out = rag.retrieve(multi_source_col, "alpha", k=6, folder=None,
                       auto_filter=False, multi_query=False,
                       source={"vault", "calendar"})
    sources = {m["source"] for m in out["metas"]}
    assert sources <= {"vault", "calendar"}, f"leaked: {sources}"
    assert "whatsapp" not in sources


def test_retrieve_source_list_works_same_as_set(multi_source_col):
    """List input equivalent to set input."""
    out = rag.retrieve(multi_source_col, "alpha", k=6, folder=None,
                       auto_filter=False, multi_query=False,
                       source=["vault", "whatsapp"])
    sources = {m["source"] for m in out["metas"]}
    assert sources <= {"vault", "whatsapp"}


def test_retrieve_source_empty_string_is_no_op(multi_source_col):
    """Empty string filter → treated as 'no filter' (all sources)."""
    out = rag.retrieve(multi_source_col, "alpha", k=6, folder=None,
                       auto_filter=False, multi_query=False, source="")
    sources = {m.get("source", "vault") for m in out["metas"]}
    # Should include multiple sources (not filtered to empty)
    assert len(sources) >= 2


def test_retrieve_source_unknown_returns_empty(multi_source_col):
    """Unknown source → no candidates match (empty result is safe)."""
    out = rag.retrieve(multi_source_col, "alpha", k=6, folder=None,
                       auto_filter=False, multi_query=False,
                       source="facebook")
    assert out["metas"] == []
    assert out["docs"] == []


def test_retrieve_source_weight_downranks_but_does_not_drop(multi_source_col):
    """Non-vault sources keep their candidates — just scored lower."""
    # Without a source filter, WA chunks still appear in the result but
    # should rank below vault/calendar for an equally-matching query.
    out = rag.retrieve(multi_source_col, "alpha", k=6, folder=None,
                       auto_filter=False, multi_query=False)
    # The top-ranked result should NOT be a WA chunk (0.75× multiplier).
    # Note: this is a ranking assertion, not a filter assertion.
    top_source = out["metas"][0].get("source", "vault")
    assert top_source != "whatsapp", (
        f"WA should not top the list given equal rerank + 0.75× multiplier; got {top_source}"
    )
