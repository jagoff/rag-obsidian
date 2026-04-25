"""Tests for fast_path propagation from multi_retrieve (2026-04-22).

Pre-fix, `multi_retrieve` returned `RetrieveResult` with `fast_path=False`
always because the constructor call omitted the field entirely (dataclass
default kicked in). That meant:

  1. The web endpoint (/api/chat → multi_retrieve) couldn't even SEE the
     marker, let alone honour it.
  2. Analytics (rag_queries.extra_json.fast_path) had 0 rows with
     fast_path=True for cmd='web', matching the last-7d measurement.
  3. Fast_path worked in CLI single-vault (9/24 rows → 2.75× speedup
     measured) but was invisible on the 80%-of-traffic surface.

Post-fix, multi_retrieve computes the same gate as single-vault
retrieve() and sets fast_path on the result. The web endpoint still
doesn't USE it to switch models — that's a separate change — but the
marker is now persisted to rag_queries for measurement.

Gate semantics (shared with retrieve()):
  fast_path = (
    _adaptive_routing()
    AND effective_intent == "semantic"
    AND top_score > _LOOKUP_THRESHOLD (default 0.6)
  )
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


# ── Fixture: mock retrieve + col so we can script scenarios ─────────────────


@pytest.fixture
def two_vaults(tmp_path):
    """Two dummy vault paths + a fake get_db_for that returns a non-empty
    col. We monkey-patch `retrieve` so we control its return shape."""
    v1 = tmp_path / "vault_a"
    v2 = tmp_path / "vault_b"
    v1.mkdir()
    v2.mkdir()
    return [("a", v1), ("b", v2)]


def _fake_col(count: int = 5):
    c = MagicMock()
    c.count.return_value = count
    return c


def _fake_retrieve_result(top_score: float, n_docs: int = 3) -> rag.RetrieveResult:
    """Build a RetrieveResult like retrieve() would return — scores
    decreasing from `top_score`."""
    scores = [top_score - 0.1 * i for i in range(n_docs)]
    return rag.RetrieveResult(
        docs=[f"doc_{i}" for i in range(n_docs)],
        metas=[{"file": f"p{i}.md"} for i in range(n_docs)],
        scores=scores,
        confidence=top_score,
        search_query="q",
        filters_applied={},
        query_variants=["q"],
        # Single-vault call sets fast_path here — multi_retrieve recomputes
        # for the merged top, so this value gets superseded in the multi
        # branch. In the single-vault shortcut the value is preserved.
        fast_path=(top_score > 0.6),
    )


# ── Single-vault shortcut: fast_path preserved ──────────────────────────────


def test_single_vault_preserves_fast_path_from_retrieve(two_vaults, monkeypatch):
    """When only one vault is in scope, multi_retrieve takes the shortcut
    and returns the underlying retrieve() result almost verbatim. The
    fast_path value must carry through."""
    monkeypatch.setattr(rag, "get_db_for", lambda p: _fake_col())
    monkeypatch.setattr(rag, "retrieve",
                        lambda *a, **kw: _fake_retrieve_result(0.8))

    r = rag.multi_retrieve(
        [two_vaults[0]], "q", k=3, folder=None,
        multi_query=False, auto_filter=False,
    )
    assert r.fast_path is True
    assert r["fast_path"] is True  # dict-access parity


def test_single_vault_preserves_fast_path_false(two_vaults, monkeypatch):
    """Low-score → single-vault retrieve returns fast_path=False → multi_retrieve
    shortcut preserves it."""
    monkeypatch.setattr(rag, "get_db_for", lambda p: _fake_col())
    monkeypatch.setattr(rag, "retrieve",
                        lambda *a, **kw: _fake_retrieve_result(0.3))

    r = rag.multi_retrieve(
        [two_vaults[0]], "q", k=3, folder=None,
        multi_query=False, auto_filter=False,
    )
    assert r.fast_path is False


# ── Cross-vault merge: fast_path computed on the merged top ─────────────────


def test_cross_vault_sets_fast_path_when_merged_top_is_high(two_vaults, monkeypatch):
    """Both vaults return mid-scores individually. Merged top keeps the
    best one. If that best score > 0.6, fast_path should be True on the
    merged result — regardless of individual per-vault values."""
    monkeypatch.setattr(rag, "get_db_for", lambda p: _fake_col())
    # Each vault's retrieve returns top_score=0.85 (both would fire
    # individually). Merged top is still 0.85.
    monkeypatch.setattr(rag, "retrieve",
                        lambda *a, **kw: _fake_retrieve_result(0.85))
    # Make sure adaptive routing is enabled for this test (default is
    # ON but an env override could break it).
    monkeypatch.delenv("RAG_ADAPTIVE_ROUTING", raising=False)
    monkeypatch.delenv("RAG_FORCE_FULL_PIPELINE", raising=False)

    r = rag.multi_retrieve(
        two_vaults, "q", k=3, folder=None,
        multi_query=False, auto_filter=False,
    )
    assert r.fast_path is True, f"expected True, got r.fast_path={r.fast_path}"
    assert r.confidence > 0.6


def test_cross_vault_sets_fast_path_false_when_merged_top_is_low(two_vaults, monkeypatch):
    """Low-score merged top → fast_path=False."""
    monkeypatch.setattr(rag, "get_db_for", lambda p: _fake_col())
    monkeypatch.setattr(rag, "retrieve",
                        lambda *a, **kw: _fake_retrieve_result(0.2))
    monkeypatch.delenv("RAG_ADAPTIVE_ROUTING", raising=False)

    r = rag.multi_retrieve(
        two_vaults, "q", k=3, folder=None,
        multi_query=False, auto_filter=False,
    )
    assert r.fast_path is False


def test_cross_vault_fast_path_honours_intent_gate(two_vaults, monkeypatch):
    """fast_path is False for non-semantic intents (count, list, etc.)
    even when the score is high. Mirrors retrieve() gate."""
    monkeypatch.setattr(rag, "get_db_for", lambda p: _fake_col())
    monkeypatch.setattr(rag, "retrieve",
                        lambda *a, **kw: _fake_retrieve_result(0.9))

    r = rag.multi_retrieve(
        two_vaults, "q", k=3, folder=None,
        multi_query=False, auto_filter=False,
        intent="count",  # non-semantic → fast_path must stay False
    )
    assert r.fast_path is False


def test_cross_vault_fast_path_off_when_adaptive_routing_disabled(
    two_vaults, monkeypatch,
):
    """RAG_ADAPTIVE_ROUTING=0 → fast_path=False regardless of score."""
    monkeypatch.setattr(rag, "get_db_for", lambda p: _fake_col())
    monkeypatch.setattr(rag, "retrieve",
                        lambda *a, **kw: _fake_retrieve_result(0.9))
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "0")

    r = rag.multi_retrieve(
        two_vaults, "q", k=3, folder=None,
        multi_query=False, auto_filter=False,
    )
    assert r.fast_path is False


# ── Empty vault list: fast_path=False by default ────────────────────────────


def test_empty_vaults_returns_fast_path_false():
    r = rag.multi_retrieve([], "q", k=3, folder=None)
    assert r.fast_path is False
    assert r.docs == []


# ── Web endpoint contract: log_query_event carries fast_path key ────────────


def test_web_log_includes_fast_path():
    """Grep-level contract: web/server.py's main log_query_event payload
    must include `fast_path` so the dashboard can count realised vs
    potential fast-path hits."""
    src = (Path(__file__).resolve().parent.parent
           / "web" / "server.py").read_text(encoding="utf-8")
    # Find the main `cmd: "web"` log call and confirm the key is there.
    # "cmd": "web" appears in a few places (low_conf_bypass also logs).
    # The main chat path is the last one before `yield _sse("done"…)`.
    # Find that specific block by anchoring on the full-pipeline timing
    # keys we persisted earlier.
    idx = src.find('"llm_decode_ms": int(_t_llm_decode_ms)')
    assert idx >= 0, "llm_decode_ms anchor missing — log_query_event changed?"
    # Window around the anchor — both directions.
    window = src[max(0, idx - 200) : idx + 1500]
    assert '"fast_path"' in window, (
        "fast_path key missing from /api/chat main log_query_event payload"
    )
