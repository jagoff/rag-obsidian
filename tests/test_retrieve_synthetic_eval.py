"""Synthetic retrieve() regression — CI-safe alternative to `rag eval`.

`rag eval` requires ollama + the real vault + ~2 min wall. This test builds
a 4-note synthetic vault, monkey-patches `embed()`, `rerank()`, and
`expand_queries()` to be deterministic, and asserts the full `retrieve()`
pipeline produces the expected top-1 path for a handful of canned queries.

Runs in <1s, no external dependencies, so it joins the regular test suite
and catches ranking-logic regressions (RRF merge, graph expansion hook,
post-rerank scoring formula) without the heavy eval machinery.

Intentionally narrow: we care about the pipeline wiring + scoring-formula
contract, not the hit@k numbers on the real vault. Those live in
`queries.yaml` and are exercised by `rag eval`/`make eval`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import rag


# Deterministic, orthogonal keyword vectors so semantic similarity is
# an inner-product of keyword overlap. Dim = 8 (tiny, fast).
_KEY_VECTORS: dict[str, list[float]] = {
    "alpha":  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "beta":   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "gamma":  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "delta":  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "common": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
}


def _vec_for(text: str) -> list[float]:
    """Sum of orthogonal keyword vectors present in `text`, normalised."""
    v = [0.0] * 8
    lower = text.lower()
    for kw, basis in _KEY_VECTORS.items():
        if kw in lower:
            for i, x in enumerate(basis):
                v[i] += x
    # Normalise so cosine == inner product.
    norm = (sum(x * x for x in v) ** 0.5) or 1.0
    return [x / norm for x in v]


@pytest.fixture
def synthetic_col(monkeypatch, tmp_path):
    """Build a 4-note synthetic vault keyed by distinct orthogonal vectors."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="synth_retrieve_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    rag._invalidate_corpus_cache()

    # 4 notes, each dominated by one keyword + a common token. The reranker
    # will tie-break on which keyword appears in the query.
    notes = [
        ("01/alpha.md",  "Alpha Note",  "alpha common concept explained"),
        ("01/beta.md",   "Beta Note",   "beta common example walkthrough"),
        ("02/gamma.md",  "Gamma Note",  "gamma common reference material"),
        ("02/delta.md",  "Delta Note",  "delta common deep-dive"),
    ]
    for rel, title, body in notes:
        p = vault / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
        emb = _vec_for(f"{title} {body}")
        col.add(
            ids=[f"{rel}::0"],
            embeddings=[emb],
            documents=[body],
            metadatas=[{
                "file": rel, "note": title, "folder": str(Path(rel).parent),
                "tags": "", "outlinks": "", "hash": f"h-{rel}",
                "display_text": body, "parent": body,
            }],
        )

    rag._invalidate_corpus_cache()

    # Mock embed/expand/rerank so retrieval is deterministic and offline.
    monkeypatch.setattr(rag, "embed", lambda texts: [_vec_for(t) for t in texts])
    monkeypatch.setattr(rag, "expand_queries", lambda q, **kw: [q])

    # Reranker: score each (query, doc) pair by keyword-overlap count.
    class _FakeReranker:
        def predict(self, pairs, batch_size=None, show_progress_bar=False):  # noqa: ARG002
            scores = []
            for q, d in pairs:
                ql = q.lower()
                dl = d.lower()
                shared = sum(1 for kw in _KEY_VECTORS if kw in ql and kw in dl)
                scores.append(float(shared))
            return scores

    monkeypatch.setattr(rag, "get_reranker", lambda: _FakeReranker())

    return col


def test_retrieve_top1_alpha(synthetic_col):
    out = rag.retrieve(synthetic_col, "tell me about alpha", k=3, folder=None,
                       auto_filter=False, multi_query=False)
    top = out["metas"][0]["file"]
    assert top == "01/alpha.md", f"expected alpha, got {top}; all={[m['file'] for m in out['metas']]}"


def test_retrieve_top1_gamma(synthetic_col):
    out = rag.retrieve(synthetic_col, "what does gamma mean", k=3, folder=None,
                       auto_filter=False, multi_query=False)
    top = out["metas"][0]["file"]
    assert top == "02/gamma.md", f"expected gamma, got {top}"


def test_retrieve_folder_filter(synthetic_col):
    """Folder filter must isolate candidate pool to `02/` only."""
    out = rag.retrieve(synthetic_col, "common reference", k=3, folder="02",
                       auto_filter=False, multi_query=False)
    for m in out["metas"]:
        assert m["file"].startswith("02/"), f"leaked outside 02/: {m['file']}"


def test_retrieve_returns_confidence(synthetic_col):
    out = rag.retrieve(synthetic_col, "alpha", k=3, folder=None,
                       auto_filter=False, multi_query=False)
    assert "confidence" in out
    assert isinstance(out["confidence"], (int, float))


def test_retrieve_empty_query_does_not_crash(synthetic_col):
    """Defensive: empty query after stopword strip shouldn't raise."""
    out = rag.retrieve(synthetic_col, "the", k=3, folder=None,
                       auto_filter=False, multi_query=False)
    # May return anything, but shouldn't raise + keys must be present.
    assert set(out.keys()) >= {"docs", "metas", "scores", "confidence"}
