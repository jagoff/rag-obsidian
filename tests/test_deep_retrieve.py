"""Tests for deep_retrieve() — the iterative sub-query retrieval that
auto-triggers when the first pass's top rerank score is below
CONFIDENCE_DEEP_THRESHOLD. Previously had zero coverage.

What this covers:
  - _judge_sufficiency parses "SUFICIENTE" vs. sub-query response
  - _judge_sufficiency returns (True, "") on LLM exception (fail-safe)
  - deep_retrieve stops after at most _DEEP_MAX_ITERS passes even if
    the LLM keeps saying the evidence is insufficient
  - deep_retrieve merges new chunks and dedups by (path, first 50 chars)
  - deep_retrieve stops early when sub-query surfaces no new chunks
  - deep_retrieve early-returns when first pass is empty
  - graph neighbours from sub-passes are merged and deduped by path

The real retrieve() is mocked — we care about the orchestration logic,
not the downstream pipeline.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import rag


# ── _judge_sufficiency ───────────────────────────────────────────────────────


def _make_helper_response(content: str):
    resp = MagicMock()
    resp.message.content = content
    return resp


def test_judge_sufficiency_recognises_suficiente(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_helper_response("SUFICIENTE")
    monkeypatch.setattr(rag, "_helper_client", lambda: mock_client)
    ok, sub = rag._judge_sufficiency("q", ["doc"], [{"note": "N"}])
    assert ok is True
    assert sub == ""


def test_judge_sufficiency_returns_sub_query_when_insufficient(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.return_value = _make_helper_response("cómo se llama el modelo helper?")
    monkeypatch.setattr(rag, "_helper_client", lambda: mock_client)
    ok, sub = rag._judge_sufficiency("q", ["doc"], [{"note": "N"}])
    assert ok is False
    assert sub == "cómo se llama el modelo helper?"


def test_judge_sufficiency_fails_safe_on_ollama_error(monkeypatch):
    """If the helper raises, deep_retrieve must not loop forever — the
    judge returns (True, '') as a kill-switch."""
    mock_client = MagicMock()
    mock_client.chat.side_effect = RuntimeError("ollama wedged")
    monkeypatch.setattr(rag, "_helper_client", lambda: mock_client)
    ok, sub = rag._judge_sufficiency("q", ["doc"], [{"note": "N"}])
    assert ok is True
    assert sub == ""


# ── deep_retrieve orchestration ──────────────────────────────────────────────


def _res(paths: list[str], scores: list[float], *, graph_paths: list[str] | None = None):
    """Build a retrieve()-shaped result dict for mocking."""
    n = len(paths)
    assert len(scores) == n
    metas = [{"file": p, "note": p.rsplit("/", 1)[-1]} for p in paths]
    docs = [f"content of {p}" for p in paths]
    gm = [{"file": gp, "note": gp} for gp in (graph_paths or [])]
    gd = [f"graph content {gp}" for gp in (graph_paths or [])]
    return {
        "docs": docs,
        "metas": metas,
        "scores": scores,
        "confidence": scores[0] if scores else float("-inf"),
        "graph_docs": gd,
        "graph_metas": gm,
    }


def test_deep_retrieve_returns_first_pass_when_no_docs(monkeypatch):
    """Empty first pass short-circuits — no judge call, no sub-queries."""
    first = {"docs": [], "metas": [], "scores": [], "confidence": float("-inf"),
             "graph_docs": [], "graph_metas": []}
    retrieve_mock = MagicMock(return_value=first)
    judge_mock = MagicMock(return_value=(False, "some sub"))
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)
    monkeypatch.setattr(rag, "_judge_sufficiency", judge_mock)
    out = rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)
    assert out is first
    judge_mock.assert_not_called()


def test_deep_retrieve_merges_new_chunks(monkeypatch):
    """Sub-query finds a path the first pass missed → appears in merged result."""
    first = _res(["a.md", "b.md"], [0.05, 0.03])
    second = _res(["b.md", "c.md"], [0.08, 0.04])  # b.md dup, c.md new
    retrieve_mock = MagicMock(side_effect=[first, second])
    judge_mock = MagicMock(return_value=(False, "sub query"))
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)
    monkeypatch.setattr(rag, "_judge_sufficiency", judge_mock)
    # After the first sub-pass, judge again → say sufficient to end the loop.
    judge_mock.side_effect = [(False, "sub"), (True, "")]

    out = rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)
    paths = [m["file"] for m in out["metas"]]
    assert "a.md" in paths
    assert "b.md" in paths
    assert "c.md" in paths
    # b.md should appear exactly once even though both passes returned it.
    assert paths.count("b.md") == 1


def test_deep_retrieve_stops_at_max_iters(monkeypatch):
    """Even if the judge perpetually says insufficient, deep_retrieve must
    stop at _DEEP_MAX_ITERS. We count retrieve() calls: 1 first pass +
    (_DEEP_MAX_ITERS-1) sub-passes max."""
    first = _res(["a.md"], [0.05])
    second = _res(["b.md"], [0.06])
    third = _res(["c.md"], [0.07])
    retrieve_mock = MagicMock(side_effect=[first, second, third, _res(["z.md"], [0.01])])
    judge_mock = MagicMock(return_value=(False, "need more"))
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)
    monkeypatch.setattr(rag, "_judge_sufficiency", judge_mock)

    rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)
    # _DEEP_MAX_ITERS=3 → 1 first + up to 2 sub-queries = 3 calls total.
    assert retrieve_mock.call_count == rag._DEEP_MAX_ITERS


def test_deep_retrieve_stops_early_when_sub_query_adds_nothing(monkeypatch):
    """Sub-query returns only duplicates → break out of the loop
    (prevents wasted iterations when the LLM keeps rephrasing around the
    same cluster)."""
    first = _res(["a.md", "b.md"], [0.05, 0.03])
    duplicate_only = _res(["a.md", "b.md"], [0.08, 0.03])  # both seen
    retrieve_mock = MagicMock(side_effect=[first, duplicate_only])
    judge_mock = MagicMock(return_value=(False, "same-area sub"))
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)
    monkeypatch.setattr(rag, "_judge_sufficiency", judge_mock)

    rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)
    # Exactly 2 calls — first + one sub-query that added nothing.
    assert retrieve_mock.call_count == 2


def test_deep_retrieve_deduplicates_graph_neighbours(monkeypatch):
    """Graph neighbours from the sub-pass merge into the first-pass's
    graph context, deduped by file path."""
    first = _res(["a.md"], [0.05], graph_paths=["g1.md", "g2.md"])
    second = _res(["b.md"], [0.06], graph_paths=["g2.md", "g3.md"])  # g2 dup
    retrieve_mock = MagicMock(side_effect=[first, second])
    judge_mock = MagicMock(side_effect=[(False, "sub"), (True, "")])
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)
    monkeypatch.setattr(rag, "_judge_sufficiency", judge_mock)

    out = rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)
    graph_paths = [gm["file"] for gm in out["graph_metas"]]
    assert set(graph_paths) == {"g1.md", "g2.md", "g3.md"}
    assert graph_paths.count("g2.md") == 1


def test_deep_retrieve_sets_confidence_to_new_top(monkeypatch):
    """After merging + re-sorting by score, confidence must reflect the
    best score seen across all passes."""
    first = _res(["a.md"], [0.05])
    second = _res(["b.md"], [0.18])  # higher score than first
    retrieve_mock = MagicMock(side_effect=[first, second])
    judge_mock = MagicMock(side_effect=[(False, "sub"), (True, "")])
    monkeypatch.setattr(rag, "retrieve", retrieve_mock)
    monkeypatch.setattr(rag, "_judge_sufficiency", judge_mock)

    out = rag.deep_retrieve(col=MagicMock(), question="q", k=5, folder=None)
    assert out["confidence"] == pytest.approx(0.18)
    assert out["metas"][0]["file"] == "b.md"


# ── Confidence threshold invariant ───────────────────────────────────────────


def test_confidence_deep_threshold_within_expected_range():
    """Guard against someone accidentally dropping the threshold to a
    value that would either never trigger or always trigger."""
    assert 0.01 < rag.CONFIDENCE_DEEP_THRESHOLD < 0.5
    # Must be above CONFIDENCE_RERANK_MIN (the refuse gate) — otherwise
    # deep_retrieve would fire on queries we already refused.
    assert rag.CONFIDENCE_DEEP_THRESHOLD > rag.CONFIDENCE_RERANK_MIN
