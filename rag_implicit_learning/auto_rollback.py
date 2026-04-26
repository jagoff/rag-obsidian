"""Auto-rollback con stratified eval — protege producción contra regresiones
silenciosas tras un retrain del ranker.

El problema: hoy `online-tune` (3:30am) actualiza `ranker.json` cuando el
sweep encuentra weights con `objective > baseline + threshold`. Pero esa
métrica es agregada — un nuevo modelo puede ser **mejor en queries fáciles
y peor en críticas**, y la suma "pasa" mientras tu día-a-día empeora.

Solución: stratified eval. Antes de promover un nuevo ranker:

  1. Replay el golden set (queries.yaml).
  2. Cluster queries por tipo (factual / exploratory / conversational /
     multi-hop / vault-specific).
  3. Calcular hit@5 / MRR / recall@5 PER cluster con baseline_v1 y new_v2.
  4. Si **cualquier cluster** regresiona más que `regression_threshold`
     (default 5pp), abortar el promote → mantener el modelo viejo.
  5. Si todos los clusters mejoran o se mantienen, promote OK.

Este módulo es ortogonal — toma dos modelos abstractos (callables que
devuelven hit@5 por cluster) y decide promote/rollback. Lo conecta a
`online-tune` o a `rag tune-lambdarank --apply` un follow-up commit.

Diseño:
- `cluster_queries_by_type()`: heurística simple basada en keywords
  ("qué es" → factual, "cómo hago" → procedural, etc.).
- `stratified_eval()`: corre eval por cluster.
- `should_rollback()`: gate decision con thresholds configurables.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Default threshold de regresión per cluster: si el modelo nuevo es peor
# que el viejo en un cluster por más de 5pp, rollback. Conservador —
# preferimos rechazar mejoras pequeñas a aceptar regresiones grandes.
DEFAULT_REGRESSION_THRESHOLD_PP = 5.0

# Tamaño mínimo de cluster para que un test sea significativo. <3 queries
# en un cluster = no decisión confiable, se ignora ese cluster.
DEFAULT_MIN_CLUSTER_SIZE = 3


# Heurísticas de clustering — keyword patterns por tipo. Order matters
# (primer match gana).
_CLUSTER_PATTERNS = [
    ("temporal", re.compile(
        r"\b(cu[aá]ndo|when|hoy|ma[ñn]ana|ayer|esta\s+semana|"
        r"el\s+\d+|antes\s+de|despu[eé]s\s+de|reciente|"
        r"today|tomorrow|yesterday|last\s+(?:week|month|year)|"
        r"recent(?:ly)?|the\s+(?:meeting|note)\s+(?:from\s+)?(?:yesterday|today))\b",
        re.IGNORECASE,
    )),
    ("procedural", re.compile(
        r"\b(c[oó]mo|how\s+to|how\s+do|paso\s+a\s+paso|gu[ií]a|"
        r"pasos|tutorial)\b",
        re.IGNORECASE,
    )),
    ("comparison", re.compile(
        r"\b(diferencia\s+entre|vs|versus|comparar|"
        r"qu[eé]\s+es\s+mejor|differences?\s+between)\b",
        re.IGNORECASE,
    )),
    ("definition", re.compile(
        r"\b(qu[eé]\s+es|qu[eé]\s+significa|definici[oó]n|"
        r"explicame|what\s+is|define)\b",
        re.IGNORECASE,
    )),
    ("listing", re.compile(
        r"\b(qu[eé]\s+(?:tengo|hay|notas|listas?)|listame|enum[eé]ra|"
        r"todas?\s+las?\s+|all\s+(?:my\s+)?(?:notes?|files?))\b",
        re.IGNORECASE,
    )),
    ("entity_lookup", re.compile(
        r"\b(grecia|alex|maria|moka|rag|finops|ikigai|"
        r"sobre\s+\w+|info\s+(?:de|sobre)\s+\w+|"
        r"about\s+\w+|tell\s+me\s+about)\b",
        re.IGNORECASE,
    )),
]


def cluster_query(query: str) -> str:
    """Asigna una query a uno de los clusters predefinidos.

    Heurística por keywords. Order de match importa (primer match gana).
    Default: 'general' si no matchea ninguno.
    """
    if not query:
        return "general"
    for name, pattern in _CLUSTER_PATTERNS:
        if pattern.search(query):
            return name
    return "general"


def cluster_queries_by_type(
    cases: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Agrupa cases por cluster de query type."""
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        q = case.get("question") or case.get("q") or ""
        cluster = cluster_query(q)
        by_cluster.setdefault(cluster, []).append(case)
    return by_cluster


def stratified_eval(
    cases: list[dict[str, Any]],
    *,
    eval_fn: Callable[[list[dict[str, Any]]], dict[str, float]],
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> dict[str, dict[str, Any]]:
    """Eval per cluster.

    Args:
        cases: list[{"question": str, "expected": list[str]}].
        eval_fn: callable list[case] → dict con métricas (al menos 'hit5').
        min_cluster_size: clusters más chicos que esto se reportan pero
            su métrica no entra al gate de rollback.

    Returns:
        dict por cluster: {hit5, mrr, recall5, n_cases, included_in_gate}.
    """
    by_cluster = cluster_queries_by_type(cases)
    out: dict[str, dict[str, Any]] = {}

    for cluster, cluster_cases in by_cluster.items():
        if not cluster_cases:
            continue
        try:
            metrics = eval_fn(cluster_cases)
        except Exception as exc:
            logger.warning("eval failed for cluster %s: %s", cluster, exc)
            continue
        out[cluster] = {
            **metrics,
            "n_cases": len(cluster_cases),
            "included_in_gate": len(cluster_cases) >= min_cluster_size,
        }
    return out


def should_rollback(
    baseline_per_cluster: dict[str, dict[str, Any]],
    candidate_per_cluster: dict[str, dict[str, Any]],
    *,
    primary_metric: str = "hit5",
    regression_threshold_pp: float = DEFAULT_REGRESSION_THRESHOLD_PP,
) -> dict[str, Any]:
    """Decide si hacer rollback comparing baseline vs candidate per cluster.

    Args:
        baseline_per_cluster: output de stratified_eval con el modelo viejo.
        candidate_per_cluster: output de stratified_eval con el modelo nuevo.
        primary_metric: cuál métrica gatekeep. Default 'hit5'.
        regression_threshold_pp: si candidate es peor que baseline en un
            cluster por más de esto en pp, rollback. Default 5.0.

    Returns:
        dict con:
          - rollback: bool (True = NO promote).
          - reason: explicación.
          - per_cluster_delta: dict cluster → delta_pp.
          - regressed_clusters: list de clusters que regresionaron.
          - improved_clusters: list de clusters que mejoraron.
    """
    deltas: dict[str, float] = {}
    regressed: list[dict[str, Any]] = []
    improved: list[dict[str, Any]] = []

    all_clusters = set(baseline_per_cluster) | set(candidate_per_cluster)

    for cluster in sorted(all_clusters):
        b = baseline_per_cluster.get(cluster, {})
        c = candidate_per_cluster.get(cluster, {})

        if not b.get("included_in_gate") or not c.get("included_in_gate"):
            continue

        b_score = float(b.get(primary_metric, 0.0))
        c_score = float(c.get(primary_metric, 0.0))
        # Round antes de comparar — float precision puede generar
        # `(0.75 - 0.8) * 100 = -5.0000000000000004`, que dispara rollback
        # cuando el delta REAL es exactamente -5.0 (en el threshold).
        # 2 decimales = 0.01pp resolution, suficiente para gateskeep.
        delta_pp = round((c_score - b_score) * 100, 2)
        deltas[cluster] = delta_pp

        record = {
            "cluster": cluster,
            "n_cases": c.get("n_cases", 0),
            "baseline": round(b_score, 4),
            "candidate": round(c_score, 4),
            "delta_pp": round(delta_pp, 2),
        }
        if delta_pp < -regression_threshold_pp:
            regressed.append(record)
        elif delta_pp > 0:
            improved.append(record)

    if regressed:
        return {
            "rollback": True,
            "reason": (
                f"Regression detected in {len(regressed)} cluster(s): "
                + ", ".join(
                    f"{r['cluster']} ({r['delta_pp']}pp)" for r in regressed
                )
            ),
            "per_cluster_delta": deltas,
            "regressed_clusters": regressed,
            "improved_clusters": improved,
        }

    return {
        "rollback": False,
        "reason": (
            f"No regressions ≥ {regression_threshold_pp}pp. "
            f"{len(improved)} cluster(s) improved."
        ),
        "per_cluster_delta": deltas,
        "regressed_clusters": [],
        "improved_clusters": improved,
    }
