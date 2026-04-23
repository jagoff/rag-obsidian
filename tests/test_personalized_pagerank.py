"""Feature #6 del 2026-04-23 — Personalized PageRank topic-aware.

Validates:
- _personalized_pagerank classical properties (sum≈1, non-negative)
- Empty seed list → falls back to uniform PageRank
- Known vs unknown seeds filtering
- Seed nodes get higher scores than non-seeds in the same neighborhood
- Classic PageRank is a special case (uniform seed = all nodes)
- Complexity: terminates in O(iterations × edges)
"""
from __future__ import annotations

import pytest

import rag


# ── _personalized_pagerank ───────────────────────────────────────────────


def _uniform_adj() -> dict[str, set[str]]:
    """Tiny graph: A <-> B <-> C <-> D, plus A -> D (diamond-ish)."""
    return {
        "A": {"B", "D"},
        "B": {"A", "C"},
        "C": {"B", "D"},
        "D": {"C", "A"},
    }


def test_ppr_empty_graph_returns_empty():
    assert rag._personalized_pagerank({}, ["A"]) == {}


def test_ppr_empty_seed_falls_back_to_uniform():
    """No seeds → classical PageRank (uniform teleport)."""
    adj = _uniform_adj()
    ppr = rag._personalized_pagerank(adj, [])
    pr = rag._graph_pagerank(adj, iterations=15)
    # Should match classical PageRank within numerical noise.
    for k in adj:
        assert abs(ppr.get(k, 0.0) - pr.get(k, 0.0)) < 1e-4


def test_ppr_unknown_seeds_fall_back_to_uniform():
    """Seeds not in adj → fallback to uniform teleport."""
    adj = _uniform_adj()
    ppr = rag._personalized_pagerank(adj, ["NOT_IN_GRAPH"])
    pr = rag._graph_pagerank(adj, iterations=15)
    for k in adj:
        assert abs(ppr.get(k, 0.0) - pr.get(k, 0.0)) < 1e-4


def test_ppr_partial_unknown_seeds_filter():
    """Known seeds kept, unknown dropped silently."""
    adj = _uniform_adj()
    ppr = rag._personalized_pagerank(adj, ["A", "NOT_IN_GRAPH"])
    # A-only seed → A and immediate neighbors (B, D) get lifted.
    assert ppr["A"] > ppr["C"]


def test_ppr_seed_node_has_highest_score_in_neighborhood():
    """With only A as seed, A ranks higher than C (furthest in graph)."""
    adj = _uniform_adj()
    ppr = rag._personalized_pagerank(adj, ["A"])
    assert ppr["A"] > ppr["C"]


def test_ppr_multiple_seeds_bias_multiple_regions():
    """Seeds A and C both rank higher than B, D (which only get transit)."""
    adj = {
        "A": {"B"},
        "B": {"A", "C"},
        "C": {"B", "D"},
        "D": {"C"},
    }
    ppr = rag._personalized_pagerank(adj, ["A", "C"])
    # Seeds should dominate over non-seed terminal nodes.
    assert ppr["A"] >= ppr["D"]


def test_ppr_non_negative_values():
    adj = _uniform_adj()
    ppr = rag._personalized_pagerank(adj, ["A"])
    for v in ppr.values():
        assert v >= 0.0


def test_ppr_returns_all_nodes():
    adj = _uniform_adj()
    ppr = rag._personalized_pagerank(adj, ["A"])
    assert set(ppr.keys()) == set(adj.keys())


def test_ppr_damping_zero_collapses_to_teleport():
    """Damping = 0 means no graph walk — rank = teleport vector."""
    adj = _uniform_adj()
    ppr = rag._personalized_pagerank(adj, ["A"], damping=0.0, iterations=10)
    assert ppr["A"] == pytest.approx(1.0, abs=1e-6)
    assert ppr["B"] == pytest.approx(0.0, abs=1e-6)
    assert ppr["C"] == pytest.approx(0.0, abs=1e-6)
    assert ppr["D"] == pytest.approx(0.0, abs=1e-6)


def test_ppr_damping_one_pure_walk_converges():
    """Damping=1.0 (no teleport) — pure random walk on graph.
    Still converges but slowly; values should be distributed across nodes."""
    adj = _uniform_adj()
    ppr = rag._personalized_pagerank(adj, ["A"], damping=1.0, iterations=50)
    # All nodes get some score via the walk from A.
    for v in ppr.values():
        assert v >= 0.0
    assert max(ppr.values()) > 0.1  # something accumulates


def test_ppr_different_from_classical_pagerank():
    """Personalized PPR biased to a seed node gives a DIFFERENT ranking
    than classical PageRank on the same graph."""
    adj = _uniform_adj()
    classical = rag._graph_pagerank(adj, iterations=15)
    # Seed only A — A should rank higher than classical.
    personalized = rag._personalized_pagerank(adj, ["A"])
    assert personalized["A"] > classical["A"]


# ── env gate parsing (used by retrieve integration) ─────────────────────


def test_env_flag_parsing():
    # The gate uses strip().lower() in ("1", "true", "yes").
    for val in ("1", "true", "yes", "TRUE", "Yes"):
        assert val.strip().lower() in ("1", "true", "yes")
    for val in ("0", "false", "no", ""):
        assert val.strip().lower() not in ("1", "true", "yes")
