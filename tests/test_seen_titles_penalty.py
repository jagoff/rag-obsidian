"""Test that retrieve()'s seen_titles penalty demotes already-surfaced
notes without removing them from the top-k.

The pre-2026-04-20 attempt was to inject `seen_titles` as an LLM
instruction inside reformulate_query() — that regressed chains hit@5
by −16pp because the helper interpreted the list as "avoid these" and
drifted off-topic. The post-rerank shift, validated empirically on
queries.yaml, lifted chains hit@5 from 83.33% to 90.00%.
"""
from __future__ import annotations

import pytest

import rag


class FakeCol:
    """Minimal col stub to drive retrieve() end-to-end without sqlite-vec."""
    def __init__(self, docs_meta: list[dict]):
        # docs_meta: [{"id": str, "doc": str, "file": str, "note": str}, ...]
        self._items = docs_meta
        self.id = "seen-titles-test-col"

    def count(self):
        return len(self._items)

    def query(self, **kw):
        # Return every id in insertion order — the test controls rerank.
        return {"ids": [[it["id"] for it in self._items]]}

    def get(self, **kw):
        ids = kw.get("ids")
        if ids is None:
            filt = kw.get("where") or {}
            file_filter = (filt.get("file") or {}).get("$in")
            items = [it for it in self._items if not file_filter or it["file"] in file_filter]
        else:
            items = [it for it in self._items if it["id"] in ids]
        return {
            "ids": [it["id"] for it in items],
            "documents": [it["doc"] for it in items],
            "metadatas": [{"file": it["file"], "note": it["note"], "tags": ""}
                          for it in items],
        }


@pytest.fixture
def mocked_retrieve(monkeypatch):
    """Strip every heavy dep from retrieve() except the scoring loop.
    Returns nothing; caller provides the FakeCol + expected rerank scores.
    """
    monkeypatch.setattr(rag, "embed", lambda xs: [[0.1] * 1024 for _ in xs])
    monkeypatch.setattr(rag, "expand_queries", lambda q: [q])
    monkeypatch.setattr(rag, "classify_intent", lambda q, t, f: ("semantic", {}))
    monkeypatch.setattr(rag, "infer_filters", lambda q, t, f: (None, None))
    monkeypatch.setattr(rag, "get_vocabulary", lambda c: ([], []))
    monkeypatch.setattr(rag, "bm25_search", lambda c, q, k, f, t, dr=None: [])
    monkeypatch.setattr(rag, "_load_corpus", lambda c: {
        "files": [], "tags": set(), "title_to_paths": {}, "adj": {},
    })
    monkeypatch.setattr(rag, "detect_temporal_intent", lambda q: (None, q))
    monkeypatch.setattr(rag, "feedback_signals_for_query", lambda e: ({}, {}))
    monkeypatch.setattr(rag, "get_pagerank", lambda c: {})
    monkeypatch.setattr(rag, "_build_graph_adj", lambda corpus: {})
    monkeypatch.setattr(rag, "load_ignored_paths", lambda: set())


def test_retrieve_demotes_seen_title_candidate(mocked_retrieve, monkeypatch):
    """Two candidates with near-equal rerank scores: one title previously
    seen, one fresh. Without seen_titles the reranker order wins; with
    seen_titles the fresh one should leapfrog thanks to the penalty."""
    col = FakeCol([
        {"id": "a", "doc": "chunk A", "file": "notes/A.md", "note": "Alpha"},
        {"id": "b", "doc": "chunk B", "file": "notes/B.md", "note": "Beta"},
    ])

    class Reranker:
        def predict(self, pairs, show_progress_bar=False, **_):
            # A slightly higher than B by 0.05 — inside SEEN_TITLE_PENALTY range (0.1).
            return [0.55 if "A" in p[1] else 0.50 for p in pairs]

    monkeypatch.setattr(rag, "get_reranker", lambda: Reranker())

    # Baseline: no seen_titles → Alpha stays on top (55 > 50).
    r_base = rag.retrieve(col, "q", k=2, folder=None)
    assert r_base["metas"][0]["note"] == "Alpha"
    assert r_base["metas"][1]["note"] == "Beta"

    # With seen_titles=["Alpha"] → Alpha gets -0.1 penalty → Beta wins.
    r_seen = rag.retrieve(col, "q", k=2, folder=None, seen_titles=["Alpha"])
    assert r_seen["metas"][0]["note"] == "Beta", (
        f"Alpha should have been demoted — got top={r_seen['metas'][0]}"
    )
    assert r_seen["metas"][1]["note"] == "Alpha"


def test_retrieve_seen_title_penalty_does_not_drop_from_top_k(mocked_retrieve, monkeypatch):
    """Soft penalty, not a filter: a seen title with a big rerank lead
    should still appear in the top-k (just demoted by a slot or two).
    Prevents the chain-collapse fix from accidentally hiding a genuine
    re-mention the user asked about explicitly."""
    col = FakeCol([
        {"id": "a", "doc": "chunk A", "file": "notes/A.md", "note": "Alpha"},
        {"id": "b", "doc": "chunk B", "file": "notes/B.md", "note": "Beta"},
    ])

    class Reranker:
        def predict(self, pairs, show_progress_bar=False, **_):
            # Alpha dominates by 2.0 — dwarfs any 0.1 penalty.
            return [3.0 if "A" in p[1] else 1.0 for p in pairs]

    monkeypatch.setattr(rag, "get_reranker", lambda: Reranker())

    r = rag.retrieve(col, "q", k=2, folder=None, seen_titles=["Alpha"])
    notes = [m["note"] for m in r["metas"]]
    # Alpha keeps the top slot despite the penalty because 3.0 - 0.1 > 1.0.
    assert notes == ["Alpha", "Beta"]


def test_retrieve_seen_title_match_is_case_insensitive(mocked_retrieve, monkeypatch):
    """Users store titles with arbitrary casing ('Alpha' vs 'alpha').
    The penalty should match regardless so the diversity nudge is robust."""
    col = FakeCol([
        {"id": "a", "doc": "chunk A", "file": "notes/A.md", "note": "Alpha"},
        {"id": "b", "doc": "chunk B", "file": "notes/B.md", "note": "Beta"},
    ])

    class Reranker:
        def predict(self, pairs, show_progress_bar=False, **_):
            return [0.55 if "A" in p[1] else 0.50 for p in pairs]

    monkeypatch.setattr(rag, "get_reranker", lambda: Reranker())

    # Lowercase seen — should still match meta.note="Alpha".
    r = rag.retrieve(col, "q", k=2, folder=None, seen_titles=["alpha"])
    assert r["metas"][0]["note"] == "Beta", "case-insensitive match failed"


def test_retrieve_seen_title_penalty_noop_on_empty_list(mocked_retrieve, monkeypatch):
    """seen_titles=[] or None must be a pure no-op — callers can pass
    their accumulated list unconditionally without worrying about edge cases."""
    col = FakeCol([
        {"id": "a", "doc": "chunk A", "file": "notes/A.md", "note": "Alpha"},
    ])

    class Reranker:
        def predict(self, pairs, show_progress_bar=False, **_):
            return [0.55 for _ in pairs]

    monkeypatch.setattr(rag, "get_reranker", lambda: Reranker())

    r1 = rag.retrieve(col, "q", k=1, folder=None)
    r2 = rag.retrieve(col, "q", k=1, folder=None, seen_titles=[])
    r3 = rag.retrieve(col, "q", k=1, folder=None, seen_titles=None)

    # All three should produce identical scores — pass-through default.
    assert r1["scores"][0] == r2["scores"][0] == r3["scores"][0]
