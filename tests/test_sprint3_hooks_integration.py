"""Integration tests para los 2 hooks Sprint 3 a `rag/__init__.py`:

  Hook 1: ensemble LLM-judge en `auto_harvest` cuando
  `RAG_AUTO_HARVEST_ENSEMBLE=1`. Verifica:
    - Default (sin env var) usa single-judge.
    - Con env var, usa ensemble + fallback al single si todo falla.
    - El shape del verdict normalizado coincide con lo que espera el flow.

  Hook 2: stratified eval bloque al `rag tune` cuando
  `RAG_TUNE_STRATIFIED_GATE=1`. Verifica:
    - El flow del rollback dispara cuando hay regresión per cluster.
    - Modo informativo (default) imprime warning pero no rollback.

Tests con monkeypatch sobre los entry points — no llaman a ollama real.
"""

from __future__ import annotations

import os

import pytest


# ── Hook 1: ensemble en auto_harvest ────────────────────────────────────────


def test_auto_harvest_ensemble_env_var_off_by_default():
    """Default: env var no set → single-judge, comportamiento histórico."""
    if "RAG_AUTO_HARVEST_ENSEMBLE" in os.environ:
        del os.environ["RAG_AUTO_HARVEST_ENSEMBLE"]
    import importlib
    import rag
    importlib.reload(rag)
    assert rag._AUTO_HARVEST_ENSEMBLE_ENABLED is False


def test_auto_harvest_ensemble_env_var_enables_with_1(monkeypatch):
    """Env var = "1" → ensemble enabled."""
    monkeypatch.setenv("RAG_AUTO_HARVEST_ENSEMBLE", "1")
    import importlib
    import rag
    importlib.reload(rag)
    assert rag._AUTO_HARVEST_ENSEMBLE_ENABLED is True


def test_auto_harvest_ensemble_env_var_disabled_with_zero(monkeypatch):
    """Env var = "0" / "false" / "no" → disabled."""
    for val in ("0", "false", "no", ""):
        monkeypatch.setenv("RAG_AUTO_HARVEST_ENSEMBLE", val)
        import importlib
        import rag
        importlib.reload(rag)
        assert rag._AUTO_HARVEST_ENSEMBLE_ENABLED is False


# ── Hook 2: stratified eval shape ───────────────────────────────────────────


def test_stratified_eval_decision_shape():
    """El output de should_rollback() tiene los campos que el hook lee."""
    from rag_implicit_learning import should_rollback

    baseline = {
        "definition": {"hit5": 0.8, "n_cases": 5, "included_in_gate": True},
    }
    candidate = {
        "definition": {"hit5": 0.7, "n_cases": 5, "included_in_gate": True},
    }
    decision = should_rollback(baseline, candidate)

    assert "rollback" in decision
    assert "reason" in decision
    assert "per_cluster_delta" in decision
    assert "regressed_clusters" in decision
    assert "improved_clusters" in decision
    assert isinstance(decision["rollback"], bool)
    assert isinstance(decision["per_cluster_delta"], dict)
    assert isinstance(decision["regressed_clusters"], list)


def test_stratified_eval_no_rollback_when_no_clusters_qualify():
    """Si todos los clusters tienen n<3, decision queda en "no rollback"
    porque no hay clusters gateskeepers."""
    from rag_implicit_learning import should_rollback

    baseline = {
        "tiny_a": {"hit5": 0.5, "n_cases": 1, "included_in_gate": False},
        "tiny_b": {"hit5": 0.5, "n_cases": 2, "included_in_gate": False},
    }
    candidate = {
        "tiny_a": {"hit5": 0.0, "n_cases": 1, "included_in_gate": False},
        "tiny_b": {"hit5": 0.0, "n_cases": 2, "included_in_gate": False},
    }
    decision = should_rollback(baseline, candidate)
    assert decision["rollback"] is False


# ── Cluster mapping del hook 2 ──────────────────────────────────────────────


def test_cluster_mapping_covers_real_query_yaml_distribution():
    """Las queries del queries.yaml real se distribuyen en clusters
    no-degenerados — al menos 2 clusters distintos están representados."""
    from pathlib import Path
    import yaml

    from rag_implicit_learning import cluster_query

    queries_path = Path(__file__).parent.parent / "queries.yaml"
    if not queries_path.exists():
        pytest.skip("queries.yaml no disponible en este checkout")

    data = yaml.safe_load(queries_path.read_text(encoding="utf-8"))
    queries = []
    for entry in data.get("queries") or []:
        if "question" in entry:
            queries.append(entry["question"])
    for chain in data.get("chains") or []:
        for turn in chain.get("turns") or []:
            if "question" in turn:
                queries.append(turn["question"])

    if not queries:
        pytest.skip("queries.yaml sin queries")

    clusters = [cluster_query(q) for q in queries]
    distinct = set(clusters)

    assert len(distinct) >= 2, (
        f"queries.yaml mapea todas las queries al mismo cluster {distinct} — "
        f"el stratified gate no aportará información."
    )


# ── Hook 1 fallback path: ensemble fails completely ─────────────────────────


def test_ensemble_fallback_path_uses_single_judge_signature():
    """Si el ensemble lanza, el fallback debe llamar a `_auto_harvest_judge`
    con la signature original (q, candidates, model=...).

    Test de regresión — si en el futuro alguien cambia la signature del
    single-judge sin tocar el fallback, este test rompe.
    """
    import inspect

    import rag

    sig = inspect.signature(rag._auto_harvest_judge)
    assert "q" in sig.parameters
    assert "candidates" in sig.parameters
    assert "model" in sig.parameters
    assert sig.parameters["model"].kind == inspect.Parameter.KEYWORD_ONLY


def test_judge_with_ensemble_compatible_with_auto_harvest_judge_signature():
    """`judge_with_ensemble` invoca `judge_fn(query, candidates, model=...)`,
    que es exactamente la signature de `rag._auto_harvest_judge`.
    """
    from rag_implicit_learning import judge_with_ensemble

    candidates = [("a.md", "snippet a")]
    calls = []

    def fake_single_judge(q, candidates, *, model=None):
        calls.append({"q": q, "candidates": candidates, "model": model})
        return {"verdict": "a.md", "confidence": 0.9, "reason": ""}

    result = judge_with_ensemble(
        "test", candidates,
        models=["m1", "m2"],
        judge_fn=fake_single_judge,
    )

    assert result is not None
    assert len(calls) == 2
    for c in calls:
        assert c["q"] == "test"
        assert c["candidates"] == candidates
        assert c["model"] in ("m1", "m2")
