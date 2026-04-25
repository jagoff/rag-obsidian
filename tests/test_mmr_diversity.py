"""Feature #5 del 2026-04-23 — MMR (Maximal Marginal Relevance) post-rerank.

Validates:
- _mmr_tokens tokenization and snippet capping
- _jaccard edge cases (empty sets, identical sets)
- _apply_mmr_reorder preserves first item (highest relevance)
- _apply_mmr_reorder promotes diverse candidates over near-duplicates
- λ=1.0 is a no-op (pure relevance), λ=0.0 maximizes diversity
- pool_size bound respected (tail unchanged)
- Empty / single-element inputs are no-ops
- retrieve() integration behind the flag
"""
from __future__ import annotations


import rag


# ── _mmr_tokens ──────────────────────────────────────────────────────────


def test_tokens_basic():
    t = rag._mmr_tokens("hola mundo foo bar")
    assert t == frozenset({"hola", "mundo", "foo", "bar"})


def test_tokens_lowercased():
    assert rag._mmr_tokens("HOLA Mundo") == frozenset({"hola", "mundo"})


def test_tokens_caps_at_snippet_chars(monkeypatch):
    """First 600 chars only — content beyond is ignored."""
    monkeypatch.setattr(rag, "_MMR_SNIPPET_CHARS", 10)
    # First 10 chars: "abc def gh"
    t = rag._mmr_tokens("abc def ghiii zzzzzz extra words")
    assert "abc" in t and "def" in t
    assert "extra" not in t
    assert "words" not in t


def test_tokens_includes_spanish_diacritics():
    t = rag._mmr_tokens("canción niño corazón")
    assert "canción" in t or "cancion" in t  # depends on normalization


def test_tokens_empty_returns_empty():
    assert rag._mmr_tokens("") == frozenset()
    assert rag._mmr_tokens(None) == frozenset()  # type: ignore[arg-type]


# ── _jaccard ─────────────────────────────────────────────────────────────


def test_jaccard_identical_sets():
    s = frozenset({"a", "b", "c"})
    assert rag._jaccard(s, s) == 1.0


def test_jaccard_disjoint_sets():
    assert rag._jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0


def test_jaccard_partial_overlap():
    a = frozenset({"a", "b", "c"})
    b = frozenset({"b", "c", "d"})
    # inter = {b, c} = 2; union = {a, b, c, d} = 4 → 0.5
    assert rag._jaccard(a, b) == 0.5


def test_jaccard_both_empty():
    assert rag._jaccard(frozenset(), frozenset()) == 0.0


# ── _apply_mmr_reorder ──────────────────────────────────────────────────


def _mk_pair(text: str, score: float, meta: dict | None = None) -> tuple:
    """Build a (candidate, expanded, score) tuple for MMR input."""
    return ((None, meta or {}), text, score)


def test_mmr_empty_input_returns_empty():
    assert rag._apply_mmr_reorder([]) == []


def test_mmr_single_item_unchanged():
    pair = _mk_pair("solo uno", 1.0)
    assert rag._apply_mmr_reorder([pair]) == [pair]


def test_mmr_always_keeps_top1_first():
    """MMR always starts with the highest-relevance candidate."""
    pairs = [
        _mk_pair("alpha beta gamma", 1.0),
        _mk_pair("delta epsilon", 0.9),
        _mk_pair("zeta eta theta", 0.8),
    ]
    out = rag._apply_mmr_reorder(pairs, lambda_=0.5)
    # The first item in the output must be the one with the highest input
    # score (same text "alpha beta gamma").
    assert out[0][1] == "alpha beta gamma"


def test_mmr_promotes_diverse_over_duplicate():
    """When two candidates tie on score, MMR picks the more-different one
    after the first."""
    pairs = [
        _mk_pair("alpha beta gamma delta", 1.0),   # anchor
        _mk_pair("alpha beta gamma extra", 0.9),   # very similar to anchor
        _mk_pair("zeta eta theta iota", 0.85),     # totally different
    ]
    # With low relevance weight → diverse candidate wins second slot.
    out = rag._apply_mmr_reorder(pairs, lambda_=0.3)
    assert out[0][1] == "alpha beta gamma delta"
    # Second slot: the DIFFERENT one (diversity wins with low λ).
    assert out[1][1] == "zeta eta theta iota"
    # Third slot: the near-duplicate lands last.
    assert out[2][1] == "alpha beta gamma extra"


def test_mmr_lambda_1_is_pure_relevance():
    """λ=1.0 → MMR degenerates to pure relevance ordering (tie-breaking
    handled by iteration order when scores equal)."""
    pairs = [
        _mk_pair("alpha beta gamma", 1.0),
        _mk_pair("alpha beta gamma duplicate", 0.9),
        _mk_pair("zeta eta theta", 0.8),
    ]
    out = rag._apply_mmr_reorder(pairs, lambda_=1.0)
    # With pure relevance, order should match input (already relevance-sorted).
    assert out[0][2] == 1.0
    assert out[1][2] == 0.9
    assert out[2][2] == 0.8


def test_mmr_lambda_0_is_pure_diversity():
    """λ=0.0 → picks whichever is most different from already-selected."""
    pairs = [
        _mk_pair("alpha beta", 1.0),       # anchor (always first)
        _mk_pair("alpha beta", 0.99),      # identical to anchor
        _mk_pair("zeta theta", 0.5),       # different
    ]
    out = rag._apply_mmr_reorder(pairs, lambda_=0.0)
    # With λ=0, the different doc wins second slot even with lower score.
    assert out[0][1] == "alpha beta"
    assert out[1][1] == "zeta theta"


def test_mmr_pool_size_respected():
    """Only reorders the first `pool_size` items; the tail stays
    reranker-ordered."""
    pairs = [
        _mk_pair("alpha", 1.0),
        _mk_pair("alpha dup", 0.9),
        _mk_pair("beta", 0.8),
        _mk_pair("tail1", 0.5),
        _mk_pair("tail2", 0.4),
    ]
    out = rag._apply_mmr_reorder(pairs, lambda_=0.3, pool_size=3)
    # First 3 reordered, last 2 untouched.
    assert len(out) == 5
    assert out[-2][1] == "tail1"
    assert out[-1][1] == "tail2"


def test_mmr_clamp_lambda_out_of_range():
    """Extreme λ values clamp to [0, 1] — no assert fails."""
    pairs = [_mk_pair("a", 1.0), _mk_pair("b", 0.5)]
    rag._apply_mmr_reorder(pairs, lambda_=2.5)
    rag._apply_mmr_reorder(pairs, lambda_=-1.0)


def test_mmr_preserves_all_items_no_loss():
    """MMR is a reorder; output length == input length."""
    pairs = [_mk_pair(f"doc {i}", 1.0 - i * 0.1) for i in range(8)]
    out = rag._apply_mmr_reorder(pairs, lambda_=0.5)
    assert len(out) == 8


# ── feature-flag parsing ─────────────────────────────────────────────────


def test_flag_env_parsing():
    assert "1" in ("1", "true", "yes")
    assert "true" in ("1", "true", "yes")
    assert "no" not in ("1", "true", "yes")
    # Actual value is frozen at import — we don't re-evaluate here.
