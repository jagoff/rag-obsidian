"""Tests for la integración de contradiction_penalty en retrieve scoring.

El penalty se aplica vía `apply_weighted_scores` (camino que comparte
math con el scorer del retrieve real). Cubre los 3 casos del spec:

  1. weights.contradiction_penalty=0 → score idéntico al baseline
     (feature está cargada en el dict pero no se aplica).
  2. weights.contradiction_penalty>0 con priors → downward shift
     proporcional al contradiction_count del path.
  3. Path ausente del dict de priors → penalty=0 (no rompe).

Diseño: usamos `apply_weighted_scores` directamente con feature dicts
sintéticos en lugar de invocar `retrieve()` end-to-end. Tests E2E del
ranker viven en `tests/test_retrieve_e2e.py`; acá nos interesa la
parsimonia matemática del penalty, no el roundtrip completo.
"""
from __future__ import annotations

import math

import rag


def _baseline_feats() -> list[dict]:
    """3 candidatos con scores rerank conocidos. El path B tiene
    contradiction_count > 0 — los otros 0.0."""
    return [
        {
            "path": "A.md", "note": "A", "rerank": 0.5,
            "recency_raw": 0.0, "tag_hits": 0, "title_match": 0.0,
            "fb_pos_cos": 0.0, "fb_neg_cos": 0.0, "ignored": False,
            "has_recency_cue": False, "graph_pagerank": 0.0,
            "click_prior": 0.0, "click_prior_folder": 0.0,
            "click_prior_hour": 0.0, "click_prior_dayofweek": 0.0,
            "dwell_score": 0.0, "contradiction_count": 0.0,
            "meta": {"file": "A.md", "source": "vault"},
        },
        {
            "path": "B.md", "note": "B", "rerank": 0.7,
            "recency_raw": 0.0, "tag_hits": 0, "title_match": 0.0,
            "fb_pos_cos": 0.0, "fb_neg_cos": 0.0, "ignored": False,
            "has_recency_cue": False, "graph_pagerank": 0.0,
            "click_prior": 0.0, "click_prior_folder": 0.0,
            "click_prior_hour": 0.0, "click_prior_dayofweek": 0.0,
            "dwell_score": 0.0, "contradiction_count": math.log1p(3),
            "meta": {"file": "B.md", "source": "vault"},
        },
        {
            "path": "C.md", "note": "C", "rerank": 0.3,
            "recency_raw": 0.0, "tag_hits": 0, "title_match": 0.0,
            "fb_pos_cos": 0.0, "fb_neg_cos": 0.0, "ignored": False,
            "has_recency_cue": False, "graph_pagerank": 0.0,
            "click_prior": 0.0, "click_prior_folder": 0.0,
            "click_prior_hour": 0.0, "click_prior_dayofweek": 0.0,
            "dwell_score": 0.0, "contradiction_count": 0.0,
            "meta": {"file": "C.md", "source": "vault"},
        },
    ]


def test_zero_weight_score_identical_to_baseline():
    """Default OFF: contradiction_penalty=0 → no afecta el score, mismo
    orden que sin la feature. Garantiza backward compat."""
    feats = _baseline_feats()
    weights = rag.RankerWeights(contradiction_penalty=0.0)
    out = rag.apply_weighted_scores(feats, weights, k=3)
    # Orden por rerank desc: B (0.7), A (0.5), C (0.3).
    assert [f["path"] for f in out] == ["B.md", "A.md", "C.md"]
    # B mantiene rerank intacto: contradiction_count > 0 pero weight=0.
    assert out[0]["score"] == 0.7


def test_positive_weight_applies_downward_shift():
    """Con weight > 0 y priors, el path con contradiction_count alto
    pierde score proporcional a `weight * contradiction_count`."""
    feats = _baseline_feats()
    w_value = 0.5
    weights = rag.RankerWeights(contradiction_penalty=w_value)
    out = rag.apply_weighted_scores(feats, weights, k=3)

    by_path = {f["path"]: f for f in out}
    # B perdió `0.5 * log1p(3)` ≈ 0.693. Score esperado: 0.7 - 0.693 ≈ 0.007.
    expected_b = 0.7 - w_value * math.log1p(3)
    assert abs(by_path["B.md"]["score"] - expected_b) < 1e-9
    # A y C no tienen contradiction_count → scores intactos.
    assert by_path["A.md"]["score"] == 0.5
    assert by_path["C.md"]["score"] == 0.3
    # Después del penalty B cae al último: 0.007 < 0.3 < 0.5.
    assert [f["path"] for f in out] == ["A.md", "C.md", "B.md"]


def test_path_missing_from_priors_no_op():
    """Path con contradiction_count ausente o 0.0 no recibe penalty.
    Verifica que el dict.get() no rompe con KeyError."""
    feats = _baseline_feats()
    # Quitamos la key de uno de los candidatos para simular el caso
    # donde collect_ranker_features no populó la feature (silent-fail).
    feats[0].pop("contradiction_count")
    weights = rag.RankerWeights(contradiction_penalty=0.5)
    out = rag.apply_weighted_scores(feats, weights, k=3)
    by_path = {f["path"]: f for f in out}
    # A.md sin la key → score sigue siendo el rerank crudo (no penalty).
    assert by_path["A.md"]["score"] == 0.5


def test_negative_score_still_decreased_by_penalty():
    """Si el rerank del path es negativo (e.g. logit), el penalty
    igual lo empuja MÁS abajo (no hacia cero). Mirror de la lógica
    pre-feedback (penalty siempre resta, nunca multiplica)."""
    feats = [
        {
            "path": "Bad.md", "note": "Bad", "rerank": -0.2,
            "recency_raw": 0.0, "tag_hits": 0, "title_match": 0.0,
            "fb_pos_cos": 0.0, "fb_neg_cos": 0.0, "ignored": False,
            "has_recency_cue": False, "graph_pagerank": 0.0,
            "click_prior": 0.0, "click_prior_folder": 0.0,
            "click_prior_hour": 0.0, "click_prior_dayofweek": 0.0,
            "dwell_score": 0.0, "contradiction_count": math.log1p(2),
            "meta": {"file": "Bad.md", "source": "vault"},
        },
    ]
    weights = rag.RankerWeights(contradiction_penalty=0.3)
    out = rag.apply_weighted_scores(feats, weights, k=1)
    expected = -0.2 - 0.3 * math.log1p(2)
    assert abs(out[0]["score"] - expected) < 1e-9


def test_weights_default_is_zero():
    """Por contrato: la feature arranca OFF. Si alguien crea
    RankerWeights() sin args, contradiction_penalty == 0.0."""
    w = rag.RankerWeights()
    assert w.contradiction_penalty == 0.0


def test_weights_serializes_contradiction_penalty():
    """as_dict + from_dict roundtrip preserva el field nuevo (no se
    pierde al guardar/cargar ranker.json)."""
    w1 = rag.RankerWeights(contradiction_penalty=0.42)
    d = w1.as_dict()
    assert "contradiction_penalty" in d
    w2 = rag.RankerWeights.from_dict(d)
    assert w2.contradiction_penalty == 0.42


def test_old_ranker_json_without_field_loads_default():
    """Backward compat: ranker.json viejo sin la key carga con default 0.0."""
    legacy = {"recency_cue": 0.1, "feedback_pos": 0.03}  # sin contradiction_penalty
    w = rag.RankerWeights.from_dict(legacy)
    assert w.contradiction_penalty == 0.0
