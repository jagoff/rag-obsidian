"""Tests para el módulo `rag_ranker_lgbm`.

Cubre:
- Feature extraction: feedback_to_training_data() con replay fake.
- Label assignment: _label_for_candidate() para todas las combinaciones.
- Train: training pipeline produce un modelo válido y serializable.
- Inference: LambdaRankerScorer carga modelo + scorea candidatos.
- Eval A/B: lambdarank vs linear sobre fixture de queries.

Tests pesan poco — usan datasets sintéticos chicos (5-10 queries) y
modelos con num_boost_round bajo (3-10) para que cada test corra <1s.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from rag_ranker_lgbm.features import (
    FEATURE_NAMES,
    _candidate_to_feature_vector,
    _label_for_candidate,
    feedback_to_training_data,
)
from rag_ranker_lgbm.inference import LambdaRankerScorer
from rag_ranker_lgbm.train import train_lambdarank


# ── _label_for_candidate ────────────────────────────────────────────────────

class TestLabelForCandidate:
    """Lógica pura — todas las combinaciones de paths/rating/corrective."""

    def test_corrective_path_is_label_2(self):
        assert _label_for_candidate(
            "right.md",
            rating=-1,
            paths=["wrong.md", "right.md"],
            corrective_path="right.md",
        ) == 2

    def test_corrective_works_even_when_outside_paths(self):
        """User puede marcar como correcto un path que NO estaba en top-k."""
        assert _label_for_candidate(
            "external.md",
            rating=-1,
            paths=["w1.md", "w2.md"],
            corrective_path="external.md",
        ) == 2

    def test_positive_rating_path_in_top_k_is_label_1(self):
        assert _label_for_candidate(
            "good.md",
            rating=1,
            paths=["good.md", "other.md"],
            corrective_path=None,
        ) == 1

    def test_negative_rating_path_in_top_k_is_label_0(self):
        assert _label_for_candidate(
            "bad.md",
            rating=-1,
            paths=["bad.md", "also-bad.md"],
            corrective_path=None,
        ) == 0

    def test_path_not_in_top_k_is_label_0(self):
        assert _label_for_candidate(
            "not-shown.md",
            rating=1,
            paths=["shown1.md", "shown2.md"],
            corrective_path=None,
        ) == 0

    def test_zero_rating_no_corrective_returns_none(self):
        """Sin signal, no labeleamos."""
        assert _label_for_candidate(
            "shown.md",
            rating=0,
            paths=["shown.md"],
            corrective_path=None,
        ) is None


# ── _candidate_to_feature_vector ────────────────────────────────────────────

def test_feature_vector_has_correct_length():
    cand = {
        "rerank": 0.7, "recency_raw": 0.5, "tag_hits": 1,
        "fb_pos_cos": 0.3, "fb_neg_cos": 0.0,
        "graph_pagerank": 0.05,
        "click_prior": 0.2, "click_prior_folder": 0.15, "click_prior_hour": 0.1,
        "dwell_score": 0.4,
    }
    vec = _candidate_to_feature_vector(cand, has_recency_cue=True)
    assert len(vec) == len(FEATURE_NAMES) == 12


def test_feature_vector_recency_cue_gating():
    """`recency_cue` (col 1) es 0 si has_recency_cue=False."""
    cand = {"recency_raw": 0.8}
    vec_with_cue = _candidate_to_feature_vector(cand, has_recency_cue=True)
    vec_without_cue = _candidate_to_feature_vector(cand, has_recency_cue=False)
    assert vec_with_cue[1] == 0.8
    assert vec_without_cue[1] == 0.0
    # `recency_always` (col 2) es siempre 0.8 indep del flag.
    assert vec_with_cue[2] == 0.8
    assert vec_without_cue[2] == 0.8


def test_feature_match_floor_is_one_when_any_feedback_signal():
    cand_with_pos = {"fb_pos_cos": 0.5}
    vec = _candidate_to_feature_vector(cand_with_pos, has_recency_cue=False)
    # feedback_match_floor (col 6).
    assert vec[6] == 1.0

    cand_no_signal = {}
    vec = _candidate_to_feature_vector(cand_no_signal, has_recency_cue=False)
    assert vec[6] == 0.0


# ── feedback_to_training_data ───────────────────────────────────────────────

@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.execute(
        """
        CREATE TABLE rag_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            turn_id TEXT,
            rating INTEGER NOT NULL,
            q TEXT,
            scope TEXT,
            paths_json TEXT,
            extra_json TEXT
        )
        """
    )
    yield c
    c.close()


def _insert_feedback(
    conn,
    *,
    ts="2026-04-25T18:00:00",
    rating=-1,
    q="qué tengo",
    paths=None,
    corrective_path=None,
):
    extra = {}
    if corrective_path:
        extra["corrective_path"] = corrective_path
    cur = conn.execute(
        "INSERT INTO rag_feedback (ts, rating, q, paths_json, extra_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            ts,
            rating,
            q,
            json.dumps(paths or []),
            json.dumps(extra) if extra else None,
        ),
    )
    return cur.lastrowid


def _fake_replay(query: str) -> list[dict]:
    """Devuelve 3 candidatos sintéticos con features simples."""
    return [
        {
            "path": "right.md",
            "rerank": 0.9,
            "recency_raw": 0.5,
            "tag_hits": 1,
            "fb_pos_cos": 0.0,
            "fb_neg_cos": 0.0,
            "graph_pagerank": 0.1,
            "click_prior": 0.3,
            "click_prior_folder": 0.2,
            "click_prior_hour": 0.1,
            "dwell_score": 0.4,
            "has_recency_cue": False,
        },
        {
            "path": "wrong.md",
            "rerank": 0.95,
            "recency_raw": 0.1,
            "tag_hits": 0,
            "fb_pos_cos": 0.0,
            "fb_neg_cos": 0.0,
            "graph_pagerank": 0.01,
            "click_prior": 0.0,
            "click_prior_folder": 0.0,
            "click_prior_hour": 0.0,
            "dwell_score": 0.0,
            "has_recency_cue": False,
        },
        {
            "path": "other.md",
            "rerank": 0.5,
            "recency_raw": 0.3,
            "tag_hits": 0,
            "fb_pos_cos": 0.0,
            "fb_neg_cos": 0.0,
            "graph_pagerank": 0.05,
            "click_prior": 0.05,
            "click_prior_folder": 0.05,
            "click_prior_hour": 0.05,
            "dwell_score": 0.1,
            "has_recency_cue": False,
        },
    ]


def test_feedback_to_training_data_with_corrective(conn):
    _insert_feedback(
        conn,
        rating=-1,
        q="cuánto debe Alex",
        paths=["wrong.md", "right.md", "other.md"],
        corrective_path="right.md",
    )

    data = feedback_to_training_data(conn, replay_features_fn=_fake_replay)

    assert data["n_queries"] == 1
    assert data["n_candidates"] == 3
    assert sum(data["group"]) == 3
    # right.md → label 2, wrong.md → label 0, other.md → label 0 (rating=-1, en paths).
    assert sorted(data["y"]) == [0, 0, 2]


def test_feedback_to_training_data_with_positive_rating(conn):
    _insert_feedback(
        conn,
        rating=1,
        q="info de Grecia",
        paths=["right.md", "wrong.md"],
    )

    data = feedback_to_training_data(conn, replay_features_fn=_fake_replay)
    # rating=1, paths=[right, wrong] → label 1 para esos dos en paths,
    # other.md → label 0 (no en paths).
    assert sorted(data["y"]) == [0, 1, 1]


def test_feedback_to_training_data_skips_no_signal(conn):
    """Feedback sin paths_json se skipea silenciosamente."""
    conn.execute(
        "INSERT INTO rag_feedback (ts, rating, q, paths_json) VALUES (?, ?, ?, ?)",
        ("2026-04-25T18:00:00", -1, "test", None),
    )
    data = feedback_to_training_data(conn, replay_features_fn=_fake_replay)
    assert data["n_queries"] == 0


def test_feedback_dedups_by_query_text(conn):
    """Si hay 2 feedbacks distintos sobre la MISMA query (el user re-rateó),
    procesamos una sola vez para no inflar el group counts."""
    _insert_feedback(
        conn, q="dup query", rating=-1, paths=["a.md"], corrective_path="b.md"
    )
    _insert_feedback(
        conn,
        ts="2026-04-25T19:00:00",
        q="dup query",
        rating=1,
        paths=["a.md"],
    )
    data = feedback_to_training_data(conn, replay_features_fn=_fake_replay)
    assert data["n_queries"] == 1


# ── train_lambdarank ────────────────────────────────────────────────────────

def test_train_lambdarank_minimal_e2e(tmp_path: Path):
    """Test end-to-end con datos sintéticos: el modelo entrena, persiste y carga."""
    # 5 queries, 3 candidatos cada una. Un signal claro: feature 0 = label.
    X: list[list[float]] = []
    y: list[int] = []
    group: list[int] = []
    for q_idx in range(5):
        for cand_idx in range(3):
            # Feature 0 alto = label alto. Los otros features ruido.
            label = cand_idx  # 0, 1, 2
            row = [float(label)] + [0.0] * 11
            X.append(row)
            y.append(label)
        group.append(3)

    output_path = tmp_path / "test_ranker.lgbm"
    result = train_lambdarank(
        X, y, group,
        feature_names=FEATURE_NAMES,
        output_path=output_path,
        num_boost_round=10,
        val_fraction=0.0,  # too few queries for val split
    )

    assert result["model_path"] == str(output_path)
    assert output_path.exists()
    assert output_path.with_suffix(".meta.json").exists()
    metadata = result["metadata"]
    assert metadata["n_features"] == 12
    assert metadata["n_train_queries"] == 5
    assert "feature_importance_gain" in metadata


def test_train_lambdarank_validates_input(tmp_path: Path):
    """Errores claros en inputs malformados."""
    with pytest.raises(ValueError, match="Training data está vacío"):
        train_lambdarank(
            [], [], [], feature_names=FEATURE_NAMES,
            output_path=tmp_path / "test.lgbm",
        )

    with pytest.raises(ValueError, match="X y y tienen len distintos"):
        train_lambdarank(
            [[0.0] * 12], [1, 2], [1],
            feature_names=FEATURE_NAMES,
            output_path=tmp_path / "test.lgbm",
        )

    with pytest.raises(ValueError, match="sum.group"):
        train_lambdarank(
            [[0.0] * 12, [0.0] * 12], [1, 2], [3],  # group sum=3, X len=2
            feature_names=FEATURE_NAMES,
            output_path=tmp_path / "test.lgbm",
        )


# ── LambdaRankerScorer ──────────────────────────────────────────────────────

@pytest.fixture
def trained_model_path(tmp_path: Path) -> Path:
    """Crea un modelo entrenado mínimo para los tests de inference."""
    X: list[list[float]] = []
    y: list[int] = []
    group: list[int] = []
    for _ in range(5):
        for cand_idx in range(3):
            X.append([float(cand_idx)] + [0.1] * 11)
            y.append(cand_idx)
        group.append(3)

    output_path = tmp_path / "test_inference.lgbm"
    train_lambdarank(
        X, y, group,
        feature_names=FEATURE_NAMES,
        output_path=output_path,
        num_boost_round=10,
        val_fraction=0.0,
    )
    return output_path


def test_scorer_load_and_predict(trained_model_path: Path):
    scorer = LambdaRankerScorer.load(trained_model_path)
    candidates = _fake_replay("test query")
    scores = scorer.predict(candidates)
    assert len(scores) == len(candidates)
    assert all(isinstance(s, float) for s in scores)


def test_scorer_predict_empty_returns_empty(trained_model_path: Path):
    scorer = LambdaRankerScorer.load(trained_model_path)
    assert scorer.predict([]) == []


def test_scorer_load_default_returns_none_if_missing(tmp_path: Path):
    """Si el path default no existe, load_default retorna None (no error)."""
    LambdaRankerScorer.clear_cache()
    fake_default = tmp_path / "definitely-does-not-exist.lgbm"
    result = LambdaRankerScorer.load_default(model_path=fake_default)
    assert result is None


def test_scorer_load_default_caches(trained_model_path: Path):
    """load_default reusa el booster para multiple llamadas."""
    LambdaRankerScorer.clear_cache()
    a = LambdaRankerScorer.load_default(model_path=trained_model_path)
    b = LambdaRankerScorer.load_default(model_path=trained_model_path)
    assert a is b  # mismo objeto cacheado
