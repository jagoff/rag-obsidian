"""First real end-to-end `retrieve()` test — no `embed()`/`expand_queries()`
monkeypatching, no stub reranker. Builds a 3-note vault, indexes via the
real `_index_single_file` (hits ollama bge-m3), runs `retrieve()` (hits
reranker + expand_queries), asserts the top-1 result is the Rust note and
the score is alive (> 0.3 — way above CONFIDENCE_RERANK_MIN=0.015).

Env-gated behind `RAG_RUN_E2E=1` because:
  - Requires ollama running with `bge-m3` + `qwen2.5:3b` pulled.
  - Requires `BAAI/bge-reranker-v2-m3` cached in ~/.cache/huggingface.
  - Cold-load first query: ~5-10s (reranker + embed + expand). Not suitable
    for the default fast suite.

Run it:

    RAG_RUN_E2E=1 .venv/bin/python -m pytest tests/test_retrieve_e2e.py -q

If the gate is off (default), the whole module skips. We prefer an env flag
over a `try-import ollama`-style auto-detect because the latter would
silently run in dev envs that have ollama up but no bge-m3 pulled, and
fail with confusing `ollama._types.ResponseError`. Explicit opt-in keeps
the skip signal clean.
"""
from __future__ import annotations

import os

import pytest

# Memory watchdog silent (would log during slow test otherwise).
os.environ.setdefault("RAG_MEMORY_PRESSURE_DISABLE", "1")

pytestmark = pytest.mark.skipif(
    not os.environ.get("RAG_RUN_E2E"),
    reason=(
        "End-to-end retrieve test — requires ollama + bge-m3 + "
        "bge-reranker-v2-m3 + qwen2.5:3b. Opt in with RAG_RUN_E2E=1."
    ),
)


@pytest.fixture
def e2e_vault(tmp_path, monkeypatch):
    """Real ollama-backed indexing into a tmp vault + sqlite-vec collection.

    Intentionally does NOT monkeypatch `embed`, `expand_queries`, or
    `get_reranker` — the whole point is to exercise the real stack.
    We DO monkeypatch VAULT_PATH / DB_PATH / get_db so the test doesn't
    touch the production vault or the production sqlite-vec store.
    """
    import rag

    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="retrieve_e2e_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    # Skip contextual summary + synthetic questions — they require qwen2.5:3b
    # and add ~1s/note of latency without affecting the retrieve() invariant
    # we're measuring here. The main retrieval stack (bge-m3 embed + sqlite-vec
    # + BM25 + bge-reranker) is still fully real.
    monkeypatch.setattr(rag, "get_context_summary", lambda *a, **kw: "")
    monkeypatch.setattr(rag, "get_synthetic_questions", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "_check_and_flag_contradictions",
                        lambda *a, **kw: None)
    rag._invalidate_corpus_cache()

    notes = [
        ("a.md", "# Python\nPython es un lenguaje de programación dinámico."),
        ("b.md", "# JavaScript\nJavaScript corre en navegadores web."),
        ("c.md", "# Rust\nRust es un lenguaje seguro y rápido, pensado para sistemas."),
    ]
    for rel, body in notes:
        (vault / rel).write_text(body, encoding="utf-8")
        status = rag._index_single_file(col, vault / rel, skip_contradict=True)
        assert status == "indexed", f"expected 'indexed' for {rel}, got {status}"

    rag._invalidate_corpus_cache()
    assert col.count() >= 3, f"expected >=3 chunks in collection, got {col.count()}"
    return vault, col


def test_retrieve_e2e_picks_rust_for_safe_and_fast_query(e2e_vault):
    """Full-stack retrieve: query in Spanish, expect Rust (c.md) top-1, with
    a confidence score >0.3 (sane reranker territory — collapsed fp16 path
    would show <0.01, total garbage would show <0.1)."""
    import rag
    _vault, col = e2e_vault

    result = rag.retrieve(
        col, "¿qué lenguaje es seguro y rápido?", k=3,
        folder=None, auto_filter=False,
    )
    assert result["metas"], f"retrieve returned no metas: {result!r}"

    top_file = result["metas"][0].get("file", "")
    assert top_file.endswith("c.md"), (
        f"top-1 should be Rust (c.md), got {top_file!r}. "
        f"Full ranking: {[m.get('file') for m in result['metas']]} "
        f"scores: {result['scores']}"
    )

    top_score = float(result["scores"][0])
    assert top_score > 0.3, (
        f"top-1 score {top_score:.4f} looks collapsed — expected >0.3 for "
        f"a clean keyword-match reranker pass. Check bge-reranker-v2-m3 "
        f"isn't running in fp16/CPU fallback."
    )


def test_retrieve_e2e_confidence_above_gate(e2e_vault):
    """Sanity: the top score clears CONFIDENCE_RERANK_MIN (0.015), so
    rag query wouldn't refuse. Cheap extra assertion that catches the
    collapsed-score failure mode even more aggressively than >0.3."""
    import rag
    _vault, col = e2e_vault

    result = rag.retrieve(
        col, "qué lenguaje es seguro y rápido", k=3,
        folder=None, auto_filter=False,
    )
    assert result["scores"], "no scores returned"
    top = float(result["scores"][0])
    assert top > rag.CONFIDENCE_RERANK_MIN, (
        f"top-score {top:.4f} did not clear CONFIDENCE_RERANK_MIN="
        f"{rag.CONFIDENCE_RERANK_MIN}; rag query would refuse without --force."
    )
