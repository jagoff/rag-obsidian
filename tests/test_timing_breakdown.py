"""Tests for the always-on retrieval timing breakdown in queries.jsonl.

Pre-2026-04-20 the per-stage timing breakdown (embed_ms, sem_ms, bm25_ms,
rerank_ms, graph_expand_ms, total_ms) was only emitted to stderr when
RAG_RETRIEVE_TIMING=1. Post-audit it's always returned from retrieve()
and always logged (when non-empty) to queries.jsonl, so latency
regressions after `rag tune --apply` or a model swap are diagnosable
from the log alone.
"""
from __future__ import annotations


import rag


# ── _round_timing_ms helper ────────────────────────────────────────────────


def test_round_timing_rounds_to_int_ms():
    """Floats → ints. Sub-ms precision from perf_counter is noise for
    log analysis and inflates the jsonl size by 30-40%."""
    got = rag._round_timing_ms({
        "embed_ms": 123.456,
        "rerank_ms": 789.01,
        "total_ms": 912.467,
    })
    assert got == {"embed_ms": 123, "rerank_ms": 789, "total_ms": 912}


def test_round_timing_handles_none_and_empty():
    """None input → None output, so callers can pass result.get('timing')
    without a defensive `if timing:`."""
    assert rag._round_timing_ms(None) is None
    assert rag._round_timing_ms({}) is None


def test_round_timing_drops_non_numeric_values():
    """Pure-safety guard — if retrieve() ever grows a non-numeric timing
    field (e.g. a phase label), we skip it instead of crashing."""
    got = rag._round_timing_ms({"embed_ms": 42.0, "phase_label": "warm"})
    assert got == {"embed_ms": 42}


def test_round_timing_half_up():
    """round() uses banker's rounding — that's fine for log timings,
    just documenting. 0.5 → 0 (to even), 1.5 → 2 (to even)."""
    got = rag._round_timing_ms({"a_ms": 0.5, "b_ms": 1.5, "c_ms": 2.5})
    # Python round() is banker's: 0.5→0, 1.5→2, 2.5→2.
    assert got == {"a_ms": 0, "b_ms": 2, "c_ms": 2}


# ── retrieve() exposes timing on empty-corpus early exit ──────────────────────


def test_retrieve_returns_timing_on_empty_corpus(monkeypatch):
    """Even the early-exit `col.count() == 0` path returns a timing
    dict (empty) so callers can unconditionally call _round_timing_ms."""

    class EmptyCol:
        def count(self): return 0
        def query(self, **kw): return {"ids": [[]]}

    # retrieve() builds its own result dict from an early `_col_count == 0`
    # branch when col.count() == 0 — no other machinery touched.
    result = rag.retrieve(EmptyCol(), "anything", k=5, folder=None)
    assert "timing" in result
    assert isinstance(result["timing"], dict)


# ── retrieve() populates timing keys on the happy path ───────────────────────


def test_retrieve_timing_keys_expected(monkeypatch):
    """On a (mocked) full retrieve pass, the returned timing must
    include the stages the stderr printout lists — so downstream log
    analysis has consistent keys."""

    # Mock out every heavy dep so we can drive retrieve() end-to-end
    # without a real sqlite-vec/BM25/reranker/ollama.
    class FakeCol:
        def __init__(self):
            self.id = "fakecol-1"
        def count(self): return 10

        def query(self, **kw):
            return {"ids": [["id1"]]}

        def get(self, **kw):
            ids = kw.get("ids") or ["id1"]
            return {
                "ids": ids,
                "documents": ["chunk body"] * len(ids),
                "metadatas": [{"file": "foo.md", "tags": ""}] * len(ids),
            }

    monkeypatch.setattr(rag, "embed", lambda xs: [[0.1] * 1024 for _ in xs])
    monkeypatch.setattr(rag, "expand_queries", lambda q: [q])
    monkeypatch.setattr(rag, "classify_intent", lambda q, t, f: ("semantic", {}))
    monkeypatch.setattr(rag, "infer_filters", lambda q, t, f: (None, None))
    monkeypatch.setattr(rag, "get_vocabulary", lambda c: ([], []))
    monkeypatch.setattr(rag, "bm25_search", lambda c, q, k, f, t, dr=None: ["id1"])
    monkeypatch.setattr(rag, "_load_corpus", lambda c: {
        "files": ["foo.md"],
        "tags": set(),
        "title_to_paths": {},
        "adj": {},
    })
    monkeypatch.setattr(rag, "detect_temporal_intent", lambda q: (None, q))

    class FlatReranker:
        def predict(self, pairs, show_progress_bar=False, **_):
            return [0.5 for _ in pairs]

    monkeypatch.setattr(rag, "get_reranker", lambda: FlatReranker())
    monkeypatch.setattr(rag, "feedback_signals_for_query", lambda e: ({}, {}))
    monkeypatch.setattr(rag, "get_pagerank", lambda c: {})
    monkeypatch.setattr(rag, "_build_graph_adj", lambda corpus: {})
    monkeypatch.setattr(rag, "load_ignored_paths", lambda: set())

    result = rag.retrieve(FakeCol(), "test query", k=5, folder=None)

    assert "timing" in result
    timing = result["timing"]
    # The full-pass timing dict always has these keys populated.
    # Rounding/type is checked separately in _round_timing_ms tests.
    expected_keys = {"embed_ms", "sem_ms", "bm25_ms", "rrf_ms", "rerank_ms", "total_ms"}
    missing = expected_keys - set(timing.keys())
    assert not missing, f"expected timing keys missing: {missing}"
    # All values should be positive floats (perf_counter deltas).
    for k, v in timing.items():
        assert v >= 0, f"timing[{k}] = {v} should be non-negative"
