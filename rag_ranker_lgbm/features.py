"""Feature extraction para LightGBM lambdarank training data.

Estructura de los datos de training:
  - X: matriz (N_candidates, N_features) con los feature vectors
  - y: array (N_candidates,) con relevance labels {0, 1, 2}
  - group: array (N_queries,) con cuántos candidates por query

Cada query tiene `group[i]` candidates consecutivos en X — lambdarank los
ranquea entre sí (no entre queries).

Labels:
  - 2 = highly relevant: el path es el `corrective_path` que el user marcó
    como "esta era la nota correcta" después de un 👎.
  - 1 = relevant: el path está en `paths_json` Y el feedback es +1 (el
    user dio 👍 a una respuesta que citó este path).
  - 0 = not relevant: el path está en `paths_json` pero el feedback es -1
    O fue otro path el corrective.

Esta lógica se enfoca en **feedback rico**: queries con corrective_path
explícito o implícito (Sprint 1) son las más informativas. Queries con
solo rating sin paths o sin corrective las skipeamos — no aportan signal
útil al ranker (no podemos asignar labels per-candidato).

Features extraídas: las 11 que ya usa `collect_ranker_features` +
`apply_weighted_scores` en rag/__init__.py. Mismo orden, misma normalización.
Si en el futuro `collect_ranker_features` agrega features nuevos, este
módulo necesita seguir los cambios para que el feature vector matchee.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

# Features alineadas con `collect_ranker_features` + `apply_weighted_scores`
# en rag/__init__.py. ORDEN IMPORTA — tiene que matchear el feature_name del
# modelo. Si esto cambia, retrain desde cero (el inference path detecta el
# mismatch via `expected_dim != len(FEATURE_NAMES)` y degrada a 0.0).
FEATURE_NAMES: list[str] = [
    "rerank",                # cross-encoder score, 0-1
    "recency_cue",           # has_recency_cue * recency_raw
    "recency_always",        # recency_raw siempre
    "tag_literal",           # tag_hits cap 1
    "feedback_pos",          # fb_pos_cos cap 1
    "feedback_neg",          # fb_neg_cos cap 1
    "feedback_match_floor",  # 1 si match en feedback corpus, else 0
    "graph_pagerank",        # PR del path en el graph del vault
    "click_prior",           # CTR per-path desde behavior
    "click_prior_folder",    # CTR per-folder
    "click_prior_hour",      # CTR para hora del día
    "click_prior_dayofweek", # CTR para día de la semana (lun=0..dom=6)
    "dwell_score",           # log1p(mean_dwell_s)
    "contradiction_count",   # log1p(count_distinct_ts) en ventana 90d
]


def _candidate_to_feature_vector(
    candidate: dict[str, Any], has_recency_cue: bool
) -> list[float]:
    """Convierte un dict de `collect_ranker_features` en un feature vector.

    Mismo mapping que `apply_weighted_scores` en rag/__init__.py. Si esa
    función cambia el orden o la transformación, este código necesita
    seguirla.
    """
    rerank = float(candidate.get("rerank", 0.0))
    recency_raw = float(candidate.get("recency_raw", 0.0))
    tag_hits = int(candidate.get("tag_hits", 0))
    fb_pos = float(candidate.get("fb_pos_cos", 0.0))
    fb_neg = float(candidate.get("fb_neg_cos", 0.0))
    pagerank = float(candidate.get("graph_pagerank", 0.0))
    click_prior = float(candidate.get("click_prior", 0.0))
    click_prior_folder = float(candidate.get("click_prior_folder", 0.0))
    click_prior_hour = float(candidate.get("click_prior_hour", 0.0))
    click_prior_dayofweek = float(candidate.get("click_prior_dayofweek", 0.0))
    dwell_score = float(candidate.get("dwell_score", 0.0))
    # log1p(count_distinct_ts) en ventana 90d. Default 0.0 si el path no
    # aparece en rag_contradictions o si collect_ranker_features no lo
    # populó (silent-fail del SQL read).
    contradiction_count = float(candidate.get("contradiction_count", 0.0))

    # `feedback_match_floor`: 1 si el chunk tuvo match en feedback corpus
    # (signal "esta nota tuvo feedback antes"). El ranker linear lo usa
    # como floor para boostear matches conocidos.
    feedback_match_floor = 1.0 if (fb_pos > 0.0 or fb_neg > 0.0) else 0.0

    return [
        rerank,
        recency_raw if has_recency_cue else 0.0,
        recency_raw,  # always
        min(1.0, float(tag_hits)),
        min(1.0, fb_pos),
        min(1.0, fb_neg),
        feedback_match_floor,
        pagerank,
        click_prior,
        click_prior_folder,
        click_prior_hour,
        click_prior_dayofweek,
        dwell_score,
        contradiction_count,
    ]


def _label_for_candidate(
    candidate_path: str,
    *,
    rating: int,
    paths: list[str],
    corrective_path: str | None,
) -> int | None:
    """Asigna label {0, 1, 2} a un candidato dado el feedback de su query.

    Devuelve None si no se puede labelear con confianza (lo skipeamos).

    Reglas:
      - candidate == corrective_path (de cualquier rating) → label 2.
        El corrective marca explícitamente "esta era la correcta" — máxima
        certeza.
      - rating == +1 Y candidate ∈ paths Y corrective is None → label 1.
        El user dio 👍 a una respuesta que citó este chunk; signal positivo
        pero menos fuerte que un corrective.
      - rating == -1 Y candidate ∈ paths Y candidate != corrective → label 0.
        El user dio 👎 a una respuesta que citó este chunk → no relevant.
      - candidate ∉ paths → label 0.
        No fue mostrado al user, no hay confirmación; lo tratamos como
        no relevant (negative implícito por exclusión).
      - rating == +1 con corrective_path seteado → label 1 para los paths
        ≠ corrective (ambiguo: el user dijo 👍 pero también marcó otro como
        más correcto).
    """
    if corrective_path and candidate_path == corrective_path:
        return 2

    if candidate_path in paths:
        if rating == 1:
            return 1
        if rating == -1:
            return 0
        return None

    # Candidate fuera del top-k mostrado al user.
    return 0


def feedback_to_training_data(
    conn: sqlite3.Connection,
    *,
    min_corrective_per_query: int = 1,
    skip_queries_without_corrective: bool = False,
    replay_features_fn=None,
) -> dict[str, Any]:
    """Construir training data desde rag_feedback + collect_ranker_features.

    Flow:
      1. Lee rag_feedback con paths_json no nulo.
      2. Para cada feedback, deriva (rating, paths, corrective_path) de
         las cols + extra_json.
      3. Re-ejecuta collect_ranker_features() sobre la misma query para
         reconstruir el feature vector de cada candidato.
         (Esto es caro — ~1-2s por query con embed + reranker — pero one-shot
         antes del training, no en hot path.)
      4. Asigna labels via `_label_for_candidate()`.
      5. Acumula en X, y, group.

    Args:
        conn: connection a telemetry.db.
        min_corrective_per_query: skip queries con menos de N
            corrective_paths totales (default 1 = al menos uno).
        skip_queries_without_corrective: si True, ignora queries con
            rating != ±1 sin corrective. Útil para training puro sobre
            signal explícita / inferida.
        replay_features_fn: callable que toma `query: str` y retorna
            list[dict] de candidatos. Si None, importa lazy desde rag.

    Returns:
        dict con X, y, group, feature_names, n_queries, n_candidates,
        n_skipped_no_signal, n_skipped_no_features.
    """
    rows = conn.execute(
        """
        SELECT id, ts, turn_id, rating, q, paths_json, extra_json
        FROM rag_feedback
        WHERE paths_json IS NOT NULL AND paths_json != ''
        ORDER BY datetime(ts) ASC
        """
    ).fetchall()

    X: list[list[float]] = []
    y: list[int] = []
    group: list[int] = []
    n_skipped_no_signal = 0
    n_skipped_no_features = 0

    if replay_features_fn is None:
        replay_features_fn = _default_replay_features

    seen_queries: set[str] = set()

    for fb_id, ts, turn_id, rating, q, paths_json, extra_json in rows:
        if not q:
            continue
        # Avoid re-replaying the same query twice (different feedbacks
        # for the same q text).
        if q in seen_queries:
            continue
        seen_queries.add(q)

        try:
            paths = json.loads(paths_json)
            extra = json.loads(extra_json or "{}")
        except (json.JSONDecodeError, TypeError):
            n_skipped_no_signal += 1
            continue

        if not paths:
            n_skipped_no_signal += 1
            continue

        corrective_path = extra.get("corrective_path")

        # Skip queries con cero corrective si así se pidió.
        if skip_queries_without_corrective and not corrective_path:
            n_skipped_no_signal += 1
            continue

        # Skip queries sin signal útil — solo rating 0 sin paths útiles.
        if rating not in (-1, 1) and not corrective_path:
            n_skipped_no_signal += 1
            continue

        # Re-extract features. Si falla (modelo no cargado, error ollama),
        # skipeamos esa query.
        try:
            candidates = replay_features_fn(q)
        except Exception as exc:
            logger.warning(
                "replay_features_fn failed for q=%r: %s", q[:80], exc
            )
            n_skipped_no_features += 1
            continue

        if not candidates:
            n_skipped_no_features += 1
            continue

        has_recency_cue = bool(candidates[0].get("has_recency_cue", False))

        group_size = 0
        for cand in candidates:
            label = _label_for_candidate(
                cand["path"],
                rating=rating,
                paths=paths,
                corrective_path=corrective_path,
            )
            if label is None:
                continue
            X.append(_candidate_to_feature_vector(cand, has_recency_cue))
            y.append(label)
            group_size += 1

        if group_size == 0:
            n_skipped_no_signal += 1
            continue

        group.append(group_size)

    return {
        "X": X,
        "y": y,
        "group": group,
        "feature_names": FEATURE_NAMES,
        "n_queries": len(group),
        "n_candidates": len(X),
        "n_skipped_no_signal": n_skipped_no_signal,
        "n_skipped_no_features": n_skipped_no_features,
    }


def _default_replay_features(query: str) -> list[dict[str, Any]]:
    """Default replay: importa lazy desde rag y reusa el path de
    `collect_ranker_features` con la collection production.

    Aislado en una función para que tests puedan inyectar un fake.
    """
    import rag
    col = rag.get_db()
    return rag.collect_ranker_features(
        col, query, k_pool=15, multi_query=False, auto_filter=True
    )


def synthetic_to_training_data(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    replay_features_fn=None,
) -> dict[str, Any]:
    """Construir training data desde rag_synthetic_queries +
    rag_synthetic_negatives + collect_ranker_features.

    A diferencia de feedback_to_training_data() (que reads rag_feedback),
    este lee de las 2 tablas synthetic:
      - rag_synthetic_queries: positives con label 2.
      - rag_synthetic_negatives: hard negatives con label 0.

    Replay collect_ranker_features() para cada synth query, y label cada
    candidato según:
      - candidate == positive_path → label 2
      - candidate ∈ hard_negatives_for_this_query → label 0
      - otherwise → label 0 (in-batch negative implícito)

    Args:
        conn: connection a telemetry.db.
        limit: máximo synthetic queries a procesar.
        replay_features_fn: callable query → candidates. Default = bge-m3.

    Returns:
        dict con X, y, group + métricas.
    """
    if replay_features_fn is None:
        replay_features_fn = _default_replay_features

    sql = """
        SELECT sq.id, sq.query, sq.note_path
        FROM rag_synthetic_queries sq
        ORDER BY sq.id ASC
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    queries = conn.execute(sql).fetchall()

    X: list[list[float]] = []
    y: list[int] = []
    group: list[int] = []
    n_skipped_no_features = 0

    for synth_id, query, positive_path in queries:
        # Get hard negatives for this query (set lookup en RAM).
        neg_rows = conn.execute(
            "SELECT neg_path FROM rag_synthetic_negatives "
            "WHERE synthetic_query_id = ?",
            (synth_id,),
        ).fetchall()
        hard_negatives = {row[0] for row in neg_rows}

        # Replay features for this query.
        try:
            candidates = replay_features_fn(query)
        except Exception:
            n_skipped_no_features += 1
            continue
        if not candidates:
            n_skipped_no_features += 1
            continue

        has_recency_cue = bool(candidates[0].get("has_recency_cue", False))

        group_size = 0
        for cand in candidates:
            cand_path = cand["path"]
            if cand_path == positive_path:
                label = 2
            elif cand_path in hard_negatives:
                label = 0
            else:
                # In-batch implicit negative: el candidate apareció en el
                # top-k del retrieval pero NO es ni el positive ni un
                # hard negative explícito → likely irrelevant.
                label = 0
            X.append(_candidate_to_feature_vector(cand, has_recency_cue))
            y.append(label)
            group_size += 1

        if group_size == 0:
            n_skipped_no_features += 1
            continue
        group.append(group_size)

    return {
        "X": X,
        "y": y,
        "group": group,
        "feature_names": FEATURE_NAMES,
        "n_queries": len(group),
        "n_candidates": len(X),
        "n_skipped_no_features": n_skipped_no_features,
    }


def combined_training_data(
    conn: sqlite3.Connection,
    *,
    use_feedback: bool = True,
    use_synthetic: bool = True,
    synthetic_limit: int | None = None,
    feedback_kwargs: dict | None = None,
    replay_features_fn=None,
) -> dict[str, Any]:
    """Combina feedback real + synthetic data en un solo training set.

    Concatena los X/y/group de ambas fuentes. La feature dimension es la
    misma (FEATURE_NAMES) entonces concat es directo.

    Args:
        conn: connection.
        use_feedback: si True, incluye datos de rag_feedback.
        use_synthetic: si True, incluye datos de rag_synthetic_queries +
            rag_synthetic_negatives.
        synthetic_limit: máximo synthetic queries.
        feedback_kwargs: kwargs extra para feedback_to_training_data.
        replay_features_fn: callable query → candidates. Default real.

    Returns:
        dict combinado con X, y, group + breakdown de fuentes.
    """
    feedback_kwargs = feedback_kwargs or {}

    X_combined: list[list[float]] = []
    y_combined: list[int] = []
    group_combined: list[int] = []
    sources: dict[str, int] = {"feedback": 0, "synthetic": 0}

    if use_feedback:
        fb_data = feedback_to_training_data(
            conn, replay_features_fn=replay_features_fn, **feedback_kwargs
        )
        X_combined.extend(fb_data["X"])
        y_combined.extend(fb_data["y"])
        group_combined.extend(fb_data["group"])
        sources["feedback"] = fb_data["n_queries"]

    if use_synthetic:
        synth_data = synthetic_to_training_data(
            conn, limit=synthetic_limit,
            replay_features_fn=replay_features_fn,
        )
        X_combined.extend(synth_data["X"])
        y_combined.extend(synth_data["y"])
        group_combined.extend(synth_data["group"])
        sources["synthetic"] = synth_data["n_queries"]

    return {
        "X": X_combined,
        "y": y_combined,
        "group": group_combined,
        "feature_names": FEATURE_NAMES,
        "n_queries": len(group_combined),
        "n_candidates": len(X_combined),
        "sources": sources,
    }
