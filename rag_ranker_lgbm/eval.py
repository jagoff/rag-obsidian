"""A/B eval: compara el LightGBM lambdarank vs el ranker linear actual
sobre `queries.yaml`.

Reusa la infra de `rag eval` pero usa `LambdaRankerScorer.predict()` en
lugar de `apply_weighted_scores()` para el sort. Reporta:

  - hit@5 / MRR / recall@5 con linear (baseline desde ranker.json)
  - hit@5 / MRR / recall@5 con LightGBM
  - delta absoluto + relativo

NO toca `rag.py` — el reranking se hace fuera del path de retrieve, así
que es safe para correr sin afectar producción.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _expected_hits_for_paths(
    candidates: list[dict[str, Any]], expected: set[str], k: int
) -> tuple[bool, float, float]:
    """Para un ranking, calcula hit@k, MRR(top hit), recall@k.

    Returns: (hit, reciprocal_rank, recall)
    """
    paths_top_k = [c["path"] for c in candidates[:k]]
    hits = [p for p in paths_top_k if p in expected]
    hit = bool(hits)
    if hit:
        first_hit_rank = paths_top_k.index(hits[0]) + 1
        rr = 1.0 / first_hit_rank
    else:
        rr = 0.0
    recall = len(set(paths_top_k) & expected) / max(1, len(expected))
    return hit, rr, recall


def eval_lambdarank_vs_linear(
    eval_cases: list[dict[str, Any]],
    *,
    k: int = 5,
    replay_features_fn=None,
    apply_weighted_scores_fn=None,
    lambdarank_scorer=None,
) -> dict[str, Any]:
    """Corre el eval A/B.

    Args:
        eval_cases: list[{"question": str, "expected": list[str]}].
            Mismo formato que queries.yaml.
        k: top-k para las métricas. Default 5.
        replay_features_fn: callable query → list[candidates]. Si None,
            usa el default (rag.collect_ranker_features sobre la prod
            collection).
        apply_weighted_scores_fn: callable que ordena candidates con el
            ranker linear. Si None, usa rag.apply_weighted_scores.
        lambdarank_scorer: LambdaRankerScorer ya cargado. Si None, intenta
            load_default() — si el modelo no está, eval falla con error
            descriptivo.

    Returns:
        dict con métricas + per-case breakdown.
    """
    from rag_ranker_lgbm.features import _default_replay_features
    from rag_ranker_lgbm.inference import LambdaRankerScorer

    if replay_features_fn is None:
        replay_features_fn = _default_replay_features

    if apply_weighted_scores_fn is None:
        import rag

        def _wrap_apply_linear(candidates):
            # `apply_weighted_scores` requires (feats, weights, k). k=10 es
            # suficientemente grande para no truncar los candidatos del top-15
            # del replay. `get_ranker_weights()` carga ranker.json (cached).
            return rag.apply_weighted_scores(
                candidates, rag.get_ranker_weights(), k=10,
            )

        apply_weighted_scores_fn = _wrap_apply_linear

    if lambdarank_scorer is None:
        lambdarank_scorer = LambdaRankerScorer.load_default()
        if lambdarank_scorer is None:
            raise RuntimeError(
                "No hay modelo lambdarank entrenado en "
                "~/.local/share/obsidian-rag/ranker.lgbm. Corré "
                "`rag tune-lambdarank --apply` primero."
            )

    metrics_linear = {"hit": 0, "rr": 0.0, "recall": 0.0}
    metrics_lgbm = {"hit": 0, "rr": 0.0, "recall": 0.0}
    per_case: list[dict[str, Any]] = []
    n = 0

    for case in eval_cases:
        question = case.get("question") or case.get("q")
        expected_paths = case.get("expected") or []
        if not question or not expected_paths:
            continue
        n += 1
        expected_set = set(expected_paths)

        try:
            candidates = replay_features_fn(question)
        except Exception as exc:
            logger.warning("retrieval failed for %r: %s", question[:80], exc)
            continue
        if not candidates:
            continue

        # Linear ranking
        try:
            ranked_linear = apply_weighted_scores_fn(candidates)
            # Si retorna list of (score, candidate), unpack.
            if ranked_linear and isinstance(ranked_linear[0], tuple):
                ranked_linear = [c for _, c in ranked_linear]
        except Exception as exc:
            logger.warning("linear scoring failed: %s", exc)
            ranked_linear = candidates

        hit_l, rr_l, rec_l = _expected_hits_for_paths(
            ranked_linear, expected_set, k
        )
        metrics_linear["hit"] += int(hit_l)
        metrics_linear["rr"] += rr_l
        metrics_linear["recall"] += rec_l

        # LightGBM ranking
        scores_lgbm = lambdarank_scorer.predict(candidates)
        ranked_lgbm = [
            c for _, c in sorted(
                zip(scores_lgbm, candidates), key=lambda x: -x[0]
            )
        ]
        hit_g, rr_g, rec_g = _expected_hits_for_paths(
            ranked_lgbm, expected_set, k
        )
        metrics_lgbm["hit"] += int(hit_g)
        metrics_lgbm["rr"] += rr_g
        metrics_lgbm["recall"] += rec_g

        per_case.append({
            "question": question,
            "expected": expected_paths,
            "linear_hit": hit_l,
            "linear_rr": rr_l,
            "linear_recall": rec_l,
            "lgbm_hit": hit_g,
            "lgbm_rr": rr_g,
            "lgbm_recall": rec_g,
        })

    if n == 0:
        return {
            "n_cases": 0,
            "linear": {"hit5": 0.0, "mrr": 0.0, "recall5": 0.0},
            "lgbm": {"hit5": 0.0, "mrr": 0.0, "recall5": 0.0},
            "delta": {"hit5_pp": 0.0, "mrr_pct": 0.0, "recall5_pp": 0.0},
            "per_case": [],
        }

    linear_summary = {
        "hit5": metrics_linear["hit"] / n,
        "mrr": metrics_linear["rr"] / n,
        "recall5": metrics_linear["recall"] / n,
    }
    lgbm_summary = {
        "hit5": metrics_lgbm["hit"] / n,
        "mrr": metrics_lgbm["rr"] / n,
        "recall5": metrics_lgbm["recall"] / n,
    }
    delta = {
        "hit5_pp": (lgbm_summary["hit5"] - linear_summary["hit5"]) * 100,
        "mrr_pct": (
            (lgbm_summary["mrr"] - linear_summary["mrr"]) / linear_summary["mrr"] * 100
            if linear_summary["mrr"] > 0 else 0.0
        ),
        "recall5_pp": (lgbm_summary["recall5"] - linear_summary["recall5"]) * 100,
    }

    return {
        "n_cases": n,
        "k": k,
        "linear": linear_summary,
        "lgbm": lgbm_summary,
        "delta": delta,
        "per_case": per_case,
    }
