"""Tests for `contradiction_count` como 14ta feature del LightGBM ranker.

Cubre:
  - FEATURE_NAMES contiene "contradiction_count" en la última posición.
  - `_candidate_to_feature_vector` emite el valor pasado en el dict.
  - Default 0.0 cuando la key falta del candidate (silent-fail downstream).
  - Vector tiene exactamente len(FEATURE_NAMES) entries.
  - El feature aparece en el dict que produce `collect_ranker_features`
    (smoke test del wiring sin levantar la pipeline E2E).
"""
from __future__ import annotations

import math

from rag_ranker_lgbm.features import (
    FEATURE_NAMES,
    _candidate_to_feature_vector,
)


def test_feature_names_includes_contradiction_count():
    """El nombre nuevo está al final (orden importa para LGBM model dim)."""
    assert "contradiction_count" in FEATURE_NAMES
    # Tiene que ser la última posición — si alguien lo mete en el medio
    # invalida modelos previos sin retraining.
    assert FEATURE_NAMES[-1] == "contradiction_count"
    # Total: 13 features previas + 1 nueva.
    assert len(FEATURE_NAMES) == 14


def test_candidate_to_feature_vector_emits_contradiction_count():
    """Si el dict trae `contradiction_count`, el vector lo refleja
    en la posición correspondiente."""
    cand = {
        "rerank": 0.5,
        "recency_raw": 0.0,
        "tag_hits": 0,
        "fb_pos_cos": 0.0,
        "fb_neg_cos": 0.0,
        "graph_pagerank": 0.0,
        "click_prior": 0.0,
        "click_prior_folder": 0.0,
        "click_prior_hour": 0.0,
        "click_prior_dayofweek": 0.0,
        "dwell_score": 0.0,
        "contradiction_count": math.log1p(4),  # 4 detecciones distintas
    }
    vec = _candidate_to_feature_vector(cand, has_recency_cue=False)
    assert len(vec) == len(FEATURE_NAMES)
    idx = FEATURE_NAMES.index("contradiction_count")
    assert abs(vec[idx] - math.log1p(4)) < 1e-9


def test_candidate_to_feature_vector_defaults_to_zero():
    """Si el dict NO tiene la key (e.g. _load_contradiction_priors devolvió
    {} por error transient), el vector pone 0.0 — no rompe el LGBM scorer."""
    cand = {"rerank": 0.5}  # missing all behavior + contradiction keys
    vec = _candidate_to_feature_vector(cand, has_recency_cue=False)
    idx = FEATURE_NAMES.index("contradiction_count")
    assert vec[idx] == 0.0


def test_candidate_to_feature_vector_coerces_to_float():
    """Valores int también se aceptan (defensive coercion)."""
    cand = {"contradiction_count": 2}
    vec = _candidate_to_feature_vector(cand, has_recency_cue=False)
    idx = FEATURE_NAMES.index("contradiction_count")
    assert vec[idx] == 2.0
    assert isinstance(vec[idx], float)


def test_other_features_remain_in_correct_positions():
    """Verifica que el bump 13 → 14 no rompió el orden de las features
    pre-existentes — `feedback_match_floor` sigue en posición 6,
    `dwell_score` en 12, etc."""
    expected_order = [
        "rerank",
        "recency_cue",
        "recency_always",
        "tag_literal",
        "feedback_pos",
        "feedback_neg",
        "feedback_match_floor",
        "graph_pagerank",
        "click_prior",
        "click_prior_folder",
        "click_prior_hour",
        "click_prior_dayofweek",
        "dwell_score",
        "contradiction_count",
    ]
    assert FEATURE_NAMES == expected_order


def test_contradiction_count_separable_from_dwell():
    """Smoke: con dwell_score=0.0 y contradiction_count distinto de cero,
    el vector tiene exactamente UNA columna distinta de cero entre las
    dos últimas (anti-regresión por shift accidental de índice)."""
    cand = {"contradiction_count": 1.5, "dwell_score": 0.0}
    vec = _candidate_to_feature_vector(cand, has_recency_cue=False)
    dwell_idx = FEATURE_NAMES.index("dwell_score")
    contr_idx = FEATURE_NAMES.index("contradiction_count")
    assert vec[dwell_idx] == 0.0
    assert vec[contr_idx] == 1.5
