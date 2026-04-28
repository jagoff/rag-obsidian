"""Tests for the HyDE / `precise` decoupling (added 2026-04-25).

Background — `retrieve()` historically activated HyDE only when
`precise=True`. That bundle made the `--hyde` CLI flag a synonym for
"precise mode" and meant `rag eval` could not measure HyDE's actual
impact independent of history-aware reformulation.

These tests pin the new contract:

  * `retrieve(precise=True, hyde=None)`  -> HyDE active   (legacy bundle)
  * `retrieve(precise=False, hyde=None)` -> HyDE inactive (legacy bundle)
  * `retrieve(precise=True, hyde=False)` -> HyDE inactive (override OFF)
  * `retrieve(precise=False, hyde=True)` -> HyDE active   (override ON)

We instrument both `hyde_embed` (HyDE path) and `embed` (non-HyDE batch
path) to count which branch ran, so we don't depend on real Ollama or
on retrieval quality -- pure wiring assertion.

The synthetic vault fixture is a slim copy of the one in
`tests/test_retrieve_synthetic_eval.py`. We don't import it because that
fixture monkeypatches `embed` itself; here we want to wrap our own
counter on top of `embed`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import rag


# Tiny orthogonal keyword space so cosine == keyword overlap.
_KEY_VECTORS: dict[str, list[float]] = {
    "alpha":  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "beta":   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "gamma":  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "common": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
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
def hyde_col(monkeypatch, tmp_path):
    """Synthetic 3-note vault + counters for HyDE vs batch embed.

    Returns a tuple `(col, counters)` where counters is a dict like
    `{"hyde": int, "embed": int}` updated as `retrieve()` runs.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="hyde_decouple_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    rag._invalidate_corpus_cache()

    notes = [
        ("01/alpha.md", "Alpha Note", "alpha common explanation"),
        ("01/beta.md",  "Beta Note",  "beta common walkthrough"),
        ("02/gamma.md", "Gamma Note", "gamma common reference"),
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

    counters = {"hyde": 0, "embed": 0}

    def fake_embed(texts):
        counters["embed"] += 1
        return [_vec_for(t) for t in texts]

    def fake_hyde_embed(question):
        counters["hyde"] += 1
        return _vec_for(question)

    monkeypatch.setattr(rag, "embed", fake_embed)
    monkeypatch.setattr(rag, "hyde_embed", fake_hyde_embed)
    monkeypatch.setattr(rag, "expand_queries", lambda q, **kw: [q])
    # Keep the local-embed fast path off so `embed()` is the only batch
    # path observed in the non-HyDE branch.
    monkeypatch.setattr(rag, "_local_embed_enabled", lambda: False)

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

    return col, counters


def test_hyde_explicit_true_with_precise_false(hyde_col):
    """`hyde=True` overrides the legacy bundle: HyDE runs even with precise=False."""
    col, counters = hyde_col
    rag.retrieve(
        col, "alpha common", k=3, folder=None,
        auto_filter=False, multi_query=False,
        precise=False, hyde=True,
    )
    assert counters["hyde"] >= 1, "expected hyde_embed to be called when hyde=True"
    assert counters["embed"] == 0, (
        f"expected the non-HyDE batch embed path to NOT run; got embed={counters['embed']}"
    )


def test_hyde_explicit_false_with_precise_true(hyde_col):
    """`hyde=False` overrides the legacy bundle: no HyDE even with precise=True."""
    col, counters = hyde_col
    rag.retrieve(
        col, "alpha common", k=3, folder=None,
        auto_filter=False, multi_query=False,
        precise=True, hyde=False,
    )
    assert counters["hyde"] == 0, (
        f"expected hyde_embed NOT to run when hyde=False; got hyde={counters['hyde']}"
    )
    assert counters["embed"] >= 1, "expected the non-HyDE batch embed path to run"


def test_hyde_none_defaults_to_precise_true(hyde_col):
    """Backward compat: hyde=None + precise=True -> HyDE runs (legacy bundle)."""
    col, counters = hyde_col
    rag.retrieve(
        col, "alpha common", k=3, folder=None,
        auto_filter=False, multi_query=False,
        precise=True, hyde=None,
    )
    assert counters["hyde"] >= 1, (
        "legacy bundle: precise=True with hyde=None must still trigger HyDE"
    )
    assert counters["embed"] == 0


def test_hyde_none_defaults_to_precise_false(hyde_col):
    """Backward compat: hyde=None + precise=False -> no HyDE (legacy bundle)."""
    col, counters = hyde_col
    rag.retrieve(
        col, "alpha common", k=3, folder=None,
        auto_filter=False, multi_query=False,
        precise=False, hyde=None,
    )
    assert counters["hyde"] == 0, (
        "legacy bundle: precise=False with hyde=None must NOT trigger HyDE"
    )
    assert counters["embed"] >= 1
