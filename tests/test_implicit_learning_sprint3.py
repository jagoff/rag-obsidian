"""Tests para los componentes de Sprint 3:
- llm_judge_ensemble: ensemble vote + self_consistency_check
- auto_rollback: cluster_query, stratified_eval, should_rollback

Tests con LLM mockeado — no llaman a ollama real.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_implicit_learning.auto_rollback import (
    DEFAULT_REGRESSION_THRESHOLD_PP,
    cluster_queries_by_type,
    cluster_query,
    should_rollback,
    stratified_eval,
)
from rag_implicit_learning.llm_judge_ensemble import (
    DEFAULT_VOTE_CONF_FLOOR,
    judge_with_ensemble,
    self_consistency_check,
)


# ── Ensemble LLM-judge ──────────────────────────────────────────────────────

class TestJudgeEnsemble:
    """Tests con judge_fn mockeado."""

    def test_unanimous_vote_high_confidence(self):
        """3 jueces votan el mismo path → confidence alta."""
        candidates = [("a.md", "snippet a"), ("b.md", "snippet b")]

        def fake_judge(q, c, *, model):
            return {"verdict": "a.md", "confidence": 0.9, "reason": "match"}

        result = judge_with_ensemble(
            "test", candidates,
            models=["m1", "m2", "m3"], judge_fn=fake_judge,
        )
        assert result is not None
        assert result["verdict"] == "a.md"
        assert result["agreement"] == 1.0
        assert result["n_judges_voted"] == 3
        assert result["confidence"] == pytest.approx(0.9, abs=0.001)

    def test_split_vote_lower_confidence(self):
        """2 jueces votan a.md, 1 vota b.md → mayoría a.md, conf más baja."""
        candidates = [("a.md", ""), ("b.md", "")]

        votes = ["a.md", "a.md", "b.md"]

        def fake_judge(q, c, *, model):
            v = votes.pop(0)
            return {"verdict": v, "confidence": 0.8, "reason": ""}

        result = judge_with_ensemble(
            "test", candidates,
            models=["m1", "m2", "m3"], judge_fn=fake_judge,
        )
        assert result["verdict"] == "a.md"
        assert result["agreement"] == pytest.approx(2 / 3, abs=0.01)
        # confidence = 0.667 * 0.8 = 0.533
        assert 0.5 < result["confidence"] < 0.6

    def test_low_confidence_judge_abstains(self):
        """Voto con confidence < floor cuenta como abstain."""
        candidates = [("a.md", "")]

        confs = [0.9, 0.4]  # segundo es debajo del floor (0.5)

        def fake_judge(q, c, *, model):
            return {"verdict": "a.md", "confidence": confs.pop(0), "reason": ""}

        result = judge_with_ensemble(
            "test", candidates,
            models=["m1", "m2"], judge_fn=fake_judge,
        )
        assert result["n_judges_total"] == 2
        assert result["n_judges_voted"] == 1
        assert result["per_judge"][1]["abstained"] is True
        assert result["per_judge"][1]["abstain_reason"] == "below_confidence_floor"

    def test_invalid_path_in_verdict_abstains(self):
        """Juez inventa un path no en candidates → cuenta como abstain."""
        candidates = [("real.md", "")]

        def fake_judge(q, c, *, model):
            return {"verdict": "imaginario.md", "confidence": 0.95, "reason": ""}

        result = judge_with_ensemble(
            "test", candidates,
            models=["m1"], judge_fn=fake_judge,
        )
        assert result is None  # nadie votó válido
        # (judge_with_ensemble retorna None si no hay valid_votes)

    def test_all_judges_fail_returns_none(self):
        """Si todos los jueces lanzan / retornan None, output es None."""
        def fake_judge(q, c, *, model):
            return None

        result = judge_with_ensemble(
            "test", [("a.md", "")],
            models=["m1", "m2"], judge_fn=fake_judge,
        )
        assert result is None

    def test_empty_candidates_returns_none(self):
        """Sin candidates no hay nada para juzgar."""
        result = judge_with_ensemble(
            "test", [], models=["m1"], judge_fn=lambda q, c, *, model: None,
        )
        assert result is None

    def test_default_vote_floor_is_05(self):
        assert DEFAULT_VOTE_CONF_FLOOR == 0.5


# ── Self-consistency check ──────────────────────────────────────────────────

class TestSelfConsistency:
    """Tests con paraphrase + retrieve mockeados."""

    def test_consistent_when_all_paraphrases_converge(self):
        """5 paráfrasis, todas dan top-1 = right.md → consistente."""
        def fake_paraphrase(q, n):
            return [f"paráfrasis {i} de '{q}'" for i in range(n)]

        def fake_retrieve(q):
            return [{"path": "right.md"}, {"path": "other.md"}]

        result = self_consistency_check(
            "test query",
            paraphrase_fn=fake_paraphrase,
            retrieve_fn=fake_retrieve,
        )
        assert result["is_consistent"] is True
        assert result["consistency_score"] == 1.0
        assert result["winner_path"] == "right.md"

    def test_inconsistent_when_paraphrases_diverge(self):
        """Cada paráfrasis da un top-1 distinto → inconsistente."""
        def fake_paraphrase(q, n):
            return [f"para {i}" for i in range(n)]

        # 5 paráfrasis (incluye original) → 5 paths distintos.
        paths = ["a.md", "b.md", "c.md", "d.md", "e.md"]

        def fake_retrieve(q):
            return [{"path": paths.pop(0)}]

        result = self_consistency_check(
            "original",
            n_paraphrases=5,
            paraphrase_fn=fake_paraphrase,
            retrieve_fn=fake_retrieve,
        )
        assert result["is_consistent"] is False
        assert result["consistency_score"] == 0.2  # 1/5

    def test_partial_consistency(self):
        """3 de 5 paráfrasis dan el mismo top → consistencia 0.6."""
        def fake_paraphrase(q, n):
            return [f"p{i}" for i in range(n)]

        paths = ["a.md", "a.md", "a.md", "b.md", "c.md"]

        def fake_retrieve(q):
            return [{"path": paths.pop(0)}]

        result = self_consistency_check(
            "x",
            n_paraphrases=5,
            paraphrase_fn=fake_paraphrase,
            retrieve_fn=fake_retrieve,
            consistency_threshold=0.6,
        )
        assert result["consistency_score"] == 0.6
        # Threshold = 0.6, score = 0.6 → exactamente en el borde, cuenta como consistente.
        assert result["is_consistent"] is True


# ── Cluster query ───────────────────────────────────────────────────────────

class TestClusterQuery:
    """Heurística pura."""

    def test_temporal_keywords(self):
        assert cluster_query("¿cuándo fue eso?") == "temporal"
        assert cluster_query("¿qué tengo para hoy?") == "temporal"
        assert cluster_query("the meeting yesterday") == "temporal"

    def test_procedural_keywords(self):
        assert cluster_query("cómo configuro el ranker") == "procedural"
        assert cluster_query("how to install obsidian-rag") == "procedural"

    def test_definition_keywords(self):
        assert cluster_query("qué es ikigai") == "definition"
        assert cluster_query("what is RAG") == "definition"

    def test_listing_keywords(self):
        assert cluster_query("qué tengo en proyectos") == "listing"
        assert cluster_query("listame las notas de Grecia") == "listing"

    def test_comparison_keywords(self):
        assert cluster_query("diferencia entre BM25 y semantic") == "comparison"

    def test_entity_lookup_keywords(self):
        assert cluster_query("info sobre Alex") == "entity_lookup"
        assert cluster_query("qué tengo de Grecia") == "listing"  # listing wins

    def test_default_general(self):
        assert cluster_query("foo bar baz") == "general"
        assert cluster_query("") == "general"


def test_cluster_queries_by_type_groups():
    cases = [
        {"question": "qué es RAG"},
        {"question": "cómo configuro RAG"},
        {"question": "info sobre Grecia"},
    ]
    grouped = cluster_queries_by_type(cases)
    assert "definition" in grouped
    assert "procedural" in grouped
    assert "entity_lookup" in grouped


# ── stratified_eval + should_rollback ───────────────────────────────────────

class TestStratifiedEval:
    def test_eval_per_cluster(self):
        cases = [
            {"question": "qué es vault"},
            {"question": "qué es RAG"},
            {"question": "qué es bge-m3"},
            {"question": "cómo configuro X"},
            {"question": "cómo deploy"},
        ]

        def fake_eval(cluster_cases):
            # Mock: hit5 = 0.8 si >= 3 cases, sino 0.5.
            return {
                "hit5": 0.8 if len(cluster_cases) >= 3 else 0.5,
                "mrr": 0.7,
            }

        result = stratified_eval(cases, eval_fn=fake_eval, min_cluster_size=3)
        assert "definition" in result
        assert result["definition"]["n_cases"] == 3
        assert result["definition"]["included_in_gate"] is True
        assert result["procedural"]["n_cases"] == 2
        assert result["procedural"]["included_in_gate"] is False


class TestShouldRollback:
    """Decision logic — no LLM, pura."""

    def test_no_rollback_when_all_clusters_improve(self):
        baseline = {
            "definition": {"hit5": 0.6, "n_cases": 5, "included_in_gate": True},
            "procedural": {"hit5": 0.7, "n_cases": 5, "included_in_gate": True},
        }
        candidate = {
            "definition": {"hit5": 0.8, "n_cases": 5, "included_in_gate": True},
            "procedural": {"hit5": 0.75, "n_cases": 5, "included_in_gate": True},
        }
        decision = should_rollback(baseline, candidate)
        assert decision["rollback"] is False
        assert len(decision["improved_clusters"]) == 2

    def test_rollback_when_one_cluster_regresses(self):
        baseline = {
            "definition": {"hit5": 0.8, "n_cases": 5, "included_in_gate": True},
            "procedural": {"hit5": 0.7, "n_cases": 5, "included_in_gate": True},
        }
        # Definition regressed by 10pp (well above 5pp threshold).
        candidate = {
            "definition": {"hit5": 0.7, "n_cases": 5, "included_in_gate": True},
            "procedural": {"hit5": 0.75, "n_cases": 5, "included_in_gate": True},
        }
        decision = should_rollback(baseline, candidate)
        assert decision["rollback"] is True
        assert len(decision["regressed_clusters"]) == 1
        assert "definition" in decision["reason"]

    def test_small_clusters_not_in_gate(self):
        """Clusters con n_cases < min se reportan pero no gatekepean."""
        baseline = {
            "definition": {"hit5": 0.8, "n_cases": 5, "included_in_gate": True},
            "tiny": {"hit5": 0.9, "n_cases": 1, "included_in_gate": False},
        }
        candidate = {
            "definition": {"hit5": 0.85, "n_cases": 5, "included_in_gate": True},
            # Tiny cluster regresses 50pp pero no gatekep.
            "tiny": {"hit5": 0.4, "n_cases": 1, "included_in_gate": False},
        }
        decision = should_rollback(baseline, candidate)
        assert decision["rollback"] is False  # tiny no gatekep

    def test_rollback_threshold_just_below_does_not_trigger(self):
        """Regresión exactamente igual al threshold NO triggers rollback."""
        baseline = {
            "definition": {"hit5": 0.8, "n_cases": 5, "included_in_gate": True},
        }
        candidate = {
            # Regresión = 5pp exactos (igual al threshold default).
            "definition": {"hit5": 0.75, "n_cases": 5, "included_in_gate": True},
        }
        decision = should_rollback(baseline, candidate)
        assert decision["rollback"] is False  # < threshold, no >

    def test_default_threshold(self):
        assert DEFAULT_REGRESSION_THRESHOLD_PP == 5.0
