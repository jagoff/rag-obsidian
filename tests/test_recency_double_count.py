"""Audit 2026-04-30 — Fix 1: recency_cue + recency_always no se sumaban
double-count. Ambos pesos activos con cue temporal usan max() en vez de
+ para cap-ear la contribución total al mayor de los dos efectivos.

Aplica simétricamente a `retrieve()` (in-pipeline) y a `apply_weighted_scores`
(eval/tune) — los dos comparten el mismo invariante post-fix.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rag  # noqa: E402


def _feat(rerank: float, recency_raw: float, has_cue: bool, source: str = "vault") -> dict:
    """Minimal feat dict matching apply_weighted_scores()'s consumer shape."""
    return {
        "path": "n.md",
        "note": "n",
        "rerank": rerank,
        "recency_raw": recency_raw,
        "has_recency_cue": has_cue,
        "tag_hits": 0,
        "fb_pos_cos": 0.0,
        "fb_neg_cos": 0.0,
        "ignored": False,
        "graph_pagerank": 0.0,
        "click_prior": 0.0,
        "click_prior_folder": 0.0,
        "click_prior_hour": 0.0,
        "dwell_score": 0.0,
        "meta": {"file": "n.md", "source": source, "created_ts": None},
    }


def test_only_cue_active_contributes_cue_weight():
    """Solo recency_cue activa + has_cue=True → final += cue * raw."""
    w = rag.RankerWeights.defaults()
    w.recency_cue = 0.4
    w.recency_always = 0.0
    feats = [_feat(rerank=0.5, recency_raw=1.0, has_cue=True)]
    top = rag.apply_weighted_scores(feats, w, k=1)
    # 0.5 + 0.4 * 1.0 = 0.90 (source_weight vault=1.0 → no-op)
    assert top[0]["score"] == pytest.approx(0.90, abs=1e-3)


def test_only_always_active_contributes_always_weight():
    """Solo recency_always activa (sin cue) → final += always * raw."""
    w = rag.RankerWeights.defaults()
    w.recency_cue = 0.0
    w.recency_always = 0.3
    feats = [_feat(rerank=0.5, recency_raw=1.0, has_cue=False)]
    top = rag.apply_weighted_scores(feats, w, k=1)
    # 0.5 + 0.3 * 1.0 = 0.80
    assert top[0]["score"] == pytest.approx(0.80, abs=1e-3)


def test_both_active_with_cue_uses_max_not_sum():
    """Ambos pesos > 0 + has_cue=True → usa max(cue, always), NO suma."""
    w = rag.RankerWeights.defaults()
    w.recency_cue = 0.4
    w.recency_always = 0.3
    feats = [_feat(rerank=0.5, recency_raw=1.0, has_cue=True)]
    top = rag.apply_weighted_scores(feats, w, k=1)
    # Pre-fix: 0.5 + (0.4 + 0.3) * 1.0 = 1.20 (BUG: double-count)
    # Post-fix: 0.5 + max(0.4, 0.3) * 1.0 = 0.90
    assert top[0]["score"] == pytest.approx(0.90, abs=1e-3)
    # Y NO 1.20
    assert top[0]["score"] != pytest.approx(1.20, abs=1e-3)


def test_both_active_without_cue_uses_always():
    """Ambos pesos > 0 + has_cue=False → recency_always solo."""
    w = rag.RankerWeights.defaults()
    w.recency_cue = 0.4
    w.recency_always = 0.3
    feats = [_feat(rerank=0.5, recency_raw=1.0, has_cue=False)]
    top = rag.apply_weighted_scores(feats, w, k=1)
    # 0.5 + 0.3 * 1.0 = 0.80 (cue inactivo → solo always)
    assert top[0]["score"] == pytest.approx(0.80, abs=1e-3)


def test_zero_recency_raw_short_circuits():
    """recency_raw=0 → no aporta nada aunque haya pesos activos."""
    w = rag.RankerWeights.defaults()
    w.recency_cue = 0.4
    w.recency_always = 0.3
    feats = [_feat(rerank=0.5, recency_raw=0.0, has_cue=True)]
    top = rag.apply_weighted_scores(feats, w, k=1)
    assert top[0]["score"] == pytest.approx(0.50, abs=1e-3)


def test_always_higher_than_cue_uses_always():
    """recency_always > recency_cue + cue activa → max() devuelve always."""
    w = rag.RankerWeights.defaults()
    w.recency_cue = 0.2
    w.recency_always = 0.5
    feats = [_feat(rerank=0.5, recency_raw=1.0, has_cue=True)]
    top = rag.apply_weighted_scores(feats, w, k=1)
    # max(0.2, 0.5) = 0.5 → 0.5 + 0.5 * 1.0 = 1.00
    assert top[0]["score"] == pytest.approx(1.00, abs=1e-3)
