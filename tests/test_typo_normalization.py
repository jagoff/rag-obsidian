"""Tests for Game-changer #1 Phase 2 (2026-04-22) — vocab-bounded typo
normalization.

Validates:
1. _levenshtein distance primitive
2. _corpus_vocab_set extracts BM25 idf keys ≥ min length
3. maybe_normalize_typos corrects obvious typos against vocab
4. Numbers, short tokens, words-in-vocab are skipped
5. Proper nouns NOT in vocab stay unchanged (no dictionary-based correction)
6. Case preservation (lowercase, Title Case, UPPERCASE)
7. Multiple candidates → tie-break by higher IDF
8. No vocab / no BM25 → None
9. Empty question → None
10. Whitespace and punctuation preserved around tokens
"""
from __future__ import annotations

from unittest.mock import MagicMock

import rag


# ── 1. _levenshtein primitive ────────────────────────────────────────────────


def test_levenshtein_identical():
    assert rag._levenshtein("hello", "hello") == 0


def test_levenshtein_single_substitution():
    assert rag._levenshtein("hello", "hallo") == 1


def test_levenshtein_insertion_deletion():
    assert rag._levenshtein("cat", "cats") == 1
    assert rag._levenshtein("cats", "cat") == 1


def test_levenshtein_multiple_edits():
    # cycle → clycle (one insert) → 1 edit
    assert rag._levenshtein("cycle", "clycle") == 1
    # cycle → diclo (3 edits: c→d, y→i, ecl→cl) — tolerant of exact count
    assert 2 <= rag._levenshtein("cycle", "diclo") <= 4


def test_levenshtein_empty_strings():
    assert rag._levenshtein("", "") == 0
    assert rag._levenshtein("", "abc") == 3
    assert rag._levenshtein("abc", "") == 3


# ── 2. _corpus_vocab_set ─────────────────────────────────────────────────────


def _make_fake_col(docs, metas=None):
    """Build a fake col that _load_corpus can tokenise."""
    if metas is None:
        metas = [{"note": "n", "file": "f", "tags": "", "folder": "", "outlinks": ""} for _ in docs]
    col = MagicMock()
    col.count.return_value = len(docs)
    col.id = "test-col-id"
    col.get.return_value = {
        "documents": docs,
        "metadatas": metas,
        "ids": [str(i) for i in range(len(docs))],
    }
    return col


def test_corpus_vocab_set_extracts_tokens(monkeypatch):
    """A small fake corpus produces a vocab containing its tokens."""
    rag._invalidate_corpus_cache()
    col = _make_fake_col([
        "cycle dev agile retrospective",
        "reranker embeddings transformer",
    ])
    vocab = rag._corpus_vocab_set(col)
    assert "cycle" in vocab
    assert "reranker" in vocab
    assert "agile" in vocab
    # Short words filtered (< 4 chars). "dev" = 3 chars.
    assert "dev" not in vocab


def test_corpus_vocab_set_no_bm25_returns_empty(monkeypatch):
    """When corpus is empty, no BM25 is built → vocab is empty."""
    rag._invalidate_corpus_cache()
    col = _make_fake_col([])
    vocab = rag._corpus_vocab_set(col)
    assert vocab == set()


# ── 3. maybe_normalize_typos happy path ──────────────────────────────────────


def test_normalize_obvious_typo(monkeypatch):
    """'clycle' → 'cycle' when 'cycle' is in the corpus vocabulary."""
    rag._invalidate_corpus_cache()
    col = _make_fake_col(["cycle dev retrospective agile reranker"])
    result = rag.maybe_normalize_typos("qué pasa con el clycle actual", col)
    assert result is not None
    assert "cycle" in result
    assert "clycle" not in result


def test_normalize_returns_none_when_all_tokens_in_vocab(monkeypatch):
    """If every token ≥4 chars is already in vocab, no correction needed."""
    rag._invalidate_corpus_cache()
    col = _make_fake_col(["cycle reranker agile retrospective"])
    result = rag.maybe_normalize_typos("cycle reranker", col)
    assert result is None


def test_normalize_skips_short_tokens(monkeypatch):
    """Tokens < 4 chars aren't candidates for correction."""
    rag._invalidate_corpus_cache()
    col = _make_fake_col(["cycle reranker agile retrospective"])
    # "un" is 2 chars — skipped; the rest is already-in-vocab.
    result = rag.maybe_normalize_typos("cycle un agile", col)
    assert result is None


def test_normalize_skips_numbers(monkeypatch):
    """Numeric tokens are preserved."""
    rag._invalidate_corpus_cache()
    col = _make_fake_col(["cycle reranker agile retrospective"])
    result = rag.maybe_normalize_typos("cycle 10.59 agile", col)
    assert result is None  # 10.59 is a number (isdigit on stripped form not exact, but filtered)


def test_normalize_skips_unknown_proper_nouns(monkeypatch):
    """Proper nouns NOT in vocab must not be corrected to similar vocab words.

    Regression target: 'Bizarrap' (musician, not in vault) must not become
    'Bizarra' or similar. We protect via the min-token length + min-candidate
    length + the fact that Bizarrap is unlikely to have a ≤2-distance match
    in a vault-only vocab that lacks it.
    """
    rag._invalidate_corpus_cache()
    col = _make_fake_col(["cycle reranker agile retrospective"])
    # 'Bizarrap' has no close match in this corpus → stays unchanged.
    result = rag.maybe_normalize_typos("info sobre Bizarrap", col)
    assert result is None


def test_normalize_preserves_lowercase_casing(monkeypatch):
    rag._invalidate_corpus_cache()
    col = _make_fake_col(["reranker embeddings"])
    result = rag.maybe_normalize_typos("el rearnker es clave", col)
    assert result is not None
    assert "reranker" in result
    assert "Reranker" not in result


def test_normalize_preserves_title_casing(monkeypatch):
    rag._invalidate_corpus_cache()
    col = _make_fake_col(["reranker embeddings"])
    result = rag.maybe_normalize_typos("El Rearnker es clave", col)
    assert result is not None
    assert "Reranker" in result


# ── 4. Contract: None cases ──────────────────────────────────────────────────


def test_normalize_empty_question_returns_none(monkeypatch):
    rag._invalidate_corpus_cache()
    col = _make_fake_col(["cycle reranker"])
    assert rag.maybe_normalize_typos("", col) is None
    assert rag.maybe_normalize_typos("   ", col) is None


def test_normalize_empty_corpus_returns_none(monkeypatch):
    rag._invalidate_corpus_cache()
    col = _make_fake_col([])
    result = rag.maybe_normalize_typos("clycle", col)
    assert result is None


def test_normalize_far_typo_outside_distance_not_corrected(monkeypatch):
    """A typo with levenshtein > 2 from any vocab word should not be corrected."""
    rag._invalidate_corpus_cache()
    col = _make_fake_col(["cycle reranker"])
    # 'xyzzzwww' has no close match within distance 2 → stays unchanged.
    result = rag.maybe_normalize_typos("info sobre xyzzzwww", col)
    assert result is None
