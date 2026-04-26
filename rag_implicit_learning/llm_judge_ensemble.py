"""Ensemble LLM-judge — múltiples jueces votan + self-consistency check.

Sprint 3 del cierre del loop de auto-aprendizaje. Mejora sobre
`_auto_harvest_judge` (single judge, qwen2.5:7b) en dos ejes:

1. **Ensemble**: 2-3 jueces independientes (con backbones distintos
   o el mismo modelo con prompts deliberadamente distintos) votan sobre
   cada query → majority vote final + agreement-based confidence.
   Cancela biases idiosincráticos de cada modelo individual.

2. **Self-consistency check**: para una query Q, generar 3-5 paráfrasis,
   replay retrieval sobre cada una, comparar si los top chunks convergen.
   Si convergen (≥60% de paráfrasis comparten el mismo top-1) → label de
   alta confianza. Si divergen → ranking inestable, baja confianza.

Diseño:
- Función `judge_with_ensemble()` ortogonal al ranker — toma una query
  + candidates (path, snippet) y devuelve verdict + confidence.
- Cada juez es opcional; el ensemble se adapta a 1, 2 ó 3 jueces
  disponibles sin error. Si solo hay 1 modelo, equivalente a single.
- Implementación REUSA `_auto_harvest_judge` del rag.py para no duplicar
  la lógica de prompting + JSON parsing. Solo agrega el voting.
- self_consistency_check() es independiente — toma query + retrieval_fn
  callable y reporta consistencia.

Tests con LLMs mockeados (no llaman a ollama). Validación end-to-end
queda como follow-up porque requiere infra running.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Default ensemble: 3 configs distintas. La diversidad puede venir de:
# - Modelos backbone distintos (qwen vs command-r vs gpt-oss).
# - Mismo modelo, prompts distintos ("strict relevance" vs "broad fit"
#   vs "conservative refusal").
#
# Nombres ollama estándar — si alguno no está cargado, se omite del
# ensemble silenciosamente.
DEFAULT_ENSEMBLE_MODELS: list[str] = [
    "qwen2.5:7b",
    "qwen2.5:3b",  # backbone más chico → ortogonalidad por capacity, no
                   # por arquitectura, pero sirve como segundo voto.
    "command-r",   # backbone realmente distinto si está disponible.
]

# Confidence floor para que un voto cuente. Vetar votos con conf < 0.5
# evita que un juez "tibio" arrastre el ensemble.
DEFAULT_VOTE_CONF_FLOOR = 0.5


def judge_with_ensemble(
    query: str,
    candidates: list[tuple[str, str]],
    *,
    models: list[str] | None = None,
    judge_fn: Callable[..., dict | None] | None = None,
    vote_confidence_floor: float = DEFAULT_VOTE_CONF_FLOOR,
) -> dict[str, Any] | None:
    """Ejecutar N jueces independientes y agregar via majority vote.

    Args:
        query: pregunta del user.
        candidates: list[(path, snippet)] — output de auto_harvest_snippets.
        models: jueces a usar. None → DEFAULT_ENSEMBLE_MODELS.
        judge_fn: callable (query, candidates, *, model=...) → dict|None.
            Default: import lazy de rag._auto_harvest_judge para reusar
            su lógica. Inyectable en tests.
        vote_confidence_floor: ignora votos del juez con conf debajo de
            esto (se cuentan como abstain).

    Returns:
        dict con campos:
          - verdict: el path con majority vote (o None si no hay).
          - confidence: agreement entre jueces (n_voting_in_majority / n_voting).
            En [0, 1]. 1.0 = todos coinciden.
          - n_judges_total: cuántos jueces se llamaron.
          - n_judges_voted: cuántos retornaron votos válidos (no abstain).
          - per_judge: list[dict] con detalle de cada juez.
        None si todos los jueces abstuvieron / fallaron.
    """
    if not candidates:
        return None

    if judge_fn is None:
        # Lazy import — evita pull rag.py en tests que mockean.
        import rag
        judge_fn = rag._auto_harvest_judge

    models = models or DEFAULT_ENSEMBLE_MODELS

    per_judge: list[dict[str, Any]] = []
    valid_votes: list[str] = []  # paths normalizados de los votos válidos

    for model in models:
        try:
            verdict_dict = judge_fn(query, candidates, model=model)
        except Exception as exc:
            logger.warning("judge %s failed: %s", model, exc)
            per_judge.append({
                "model": model, "verdict": None,
                "confidence": 0.0, "abstained": True, "error": str(exc),
            })
            continue

        if verdict_dict is None:
            per_judge.append({
                "model": model, "verdict": None,
                "confidence": 0.0, "abstained": True,
            })
            continue

        verdict = verdict_dict.get("verdict")
        conf = float(verdict_dict.get("confidence", 0.0))

        # Vote validation: must be a path that exists in candidates,
        # AND confidence above the floor.
        candidate_paths = {p for p, _ in candidates}
        if verdict and verdict in candidate_paths and conf >= vote_confidence_floor:
            valid_votes.append(verdict)
            per_judge.append({
                "model": model, "verdict": verdict, "confidence": conf,
                "abstained": False,
            })
        else:
            per_judge.append({
                "model": model, "verdict": verdict,
                "confidence": conf, "abstained": True,
                # Reason for abstain hint.
                "abstain_reason": (
                    "below_confidence_floor" if (verdict and verdict in candidate_paths)
                    else "verdict_not_in_candidates" if verdict
                    else "none_verdict"
                ),
            })

    if not valid_votes:
        return None

    # Majority vote.
    counts = Counter(valid_votes)
    top_verdict, top_count = counts.most_common(1)[0]
    n_voted = len(valid_votes)
    agreement = top_count / n_voted

    # Confidence = agreement * mean confidence of voters that voted for
    # the winning verdict. Penaliza splits incluso si la mayoría es alta.
    voters_for_winner = [
        j["confidence"] for j in per_judge
        if not j["abstained"] and j["verdict"] == top_verdict
    ]
    mean_winner_conf = (
        statistics.mean(voters_for_winner) if voters_for_winner else 0.0
    )
    final_confidence = agreement * mean_winner_conf

    return {
        "verdict": top_verdict,
        "confidence": round(final_confidence, 3),
        "agreement": round(agreement, 3),
        "n_judges_total": len(models),
        "n_judges_voted": n_voted,
        "per_judge": per_judge,
        "vote_counts": dict(counts),
    }


# ── Self-consistency check ──────────────────────────────────────────────────


def self_consistency_check(
    query: str,
    *,
    n_paraphrases: int = 5,
    paraphrase_fn: Callable[[str, int], list[str]] | None = None,
    retrieve_fn: Callable[[str], list[dict[str, Any]]] | None = None,
    consistency_threshold: float = 0.6,
) -> dict[str, Any]:
    """Generar N paráfrasis + ver si las retrievals convergen al mismo top.

    Si las paráfrasis convergen → ranking estable → label "alta confianza".
    Si divergen → ranking inestable → label "baja confianza, no usar para
    training".

    Args:
        query: query original.
        n_paraphrases: cuántas paráfrasis generar.
        paraphrase_fn: callable (query, n) → list[str]. Default: usa
            `expand_queries` de rag.py si está disponible.
        retrieve_fn: callable (query) → list[candidate dict]. Default:
            `_default_replay_features` del módulo lgbm.
        consistency_threshold: fracción mínima de paráfrasis que comparten
            el mismo top-1 para ser "consistente". 0.6 = al menos 3 de 5.

    Returns:
        dict con:
          - is_consistent: bool.
          - consistency_score: float [0, 1] — fracción de paráfrasis
            que comparten el mismo top-1.
          - top_1_paths: list de los top-1 paths (uno por paráfrasis).
          - winner_path: path que más se repitió.
          - n_paraphrases_used: cuántas se procesaron exitosamente.
    """
    if paraphrase_fn is None:
        try:
            import rag
            paraphrase_fn = lambda q, n: (rag.expand_queries(q) or [q])[:n]  # noqa: E731
        except Exception:
            return {
                "is_consistent": False,
                "consistency_score": 0.0,
                "top_1_paths": [],
                "winner_path": None,
                "n_paraphrases_used": 0,
                "error": "paraphrase_fn unavailable",
            }

    if retrieve_fn is None:
        from rag_ranker_lgbm.features import _default_replay_features
        retrieve_fn = _default_replay_features

    paraphrases = [query]
    try:
        extras = paraphrase_fn(query, n_paraphrases - 1)
        for p in extras:
            if p and p != query and p not in paraphrases:
                paraphrases.append(p)
    except Exception as exc:
        logger.warning("paraphrase generation failed: %s", exc)

    paraphrases = paraphrases[:n_paraphrases]

    top_1_paths: list[str] = []
    for p in paraphrases:
        try:
            candidates = retrieve_fn(p)
            if candidates:
                top_1_paths.append(candidates[0]["path"])
        except Exception as exc:
            logger.warning("retrieve failed for paraphrase %r: %s", p[:60], exc)

    if not top_1_paths:
        return {
            "is_consistent": False,
            "consistency_score": 0.0,
            "top_1_paths": [],
            "winner_path": None,
            "n_paraphrases_used": 0,
        }

    counts = Counter(top_1_paths)
    winner_path, winner_count = counts.most_common(1)[0]
    consistency_score = winner_count / len(top_1_paths)

    return {
        "is_consistent": consistency_score >= consistency_threshold,
        "consistency_score": round(consistency_score, 3),
        "top_1_paths": top_1_paths,
        "winner_path": winner_path,
        "n_paraphrases_used": len(top_1_paths),
        "vote_counts": dict(counts),
    }
