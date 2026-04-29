"""Aplicar el outcome de una session como signal a los turns que la componen.

Concepto: si una session terminó en `win` (el user dijo "perfecto"
después de 4 preguntas), TODOS los turns de esa session contribuyeron
al éxito → cada uno recibe un peso positivo implícito. Si terminó en
`loss`, los turns reciben peso negativo. Es **reward shaping** clásico
de RL: backpropagar el outcome final a las acciones intermedias.

Diseño:
- Lee analyses de `session_outcome.classify_recent_sessions`.
- Para cada session con outcome `win` o `loss` y confidence ≥ threshold:
  - Para cada turn de esa session que NO tenga ya un feedback explícito:
    - Insertar un feedback implícito `rating=+1` (win) o `-1` (loss).
    - `extra_json.implicit_loss_source = 'session_outcome_<outcome>'`.
    - `extra_json.session_confidence = <confidence>`.
- Skipea turns con feedback explícito previo (rating no NULL del user)
  para no overlappear señales contradictorias.
- Skipea sessions con outcome `partial` — son ambiguas, no podemos inferir
  reward sin riesgo de meterle ruido al ranker.
- Sessions con outcome `abandon`: ramificación 2026-04-29.
  - Si `top_score` del primer turn < `WEAK_NEGATIVE_TOP_SCORE_THRESHOLD`
    (0.4): la query no matcheó nada decente y el user se fue. Persistimos
    `rating=-1` con `implicit_loss_source='session_outcome_weak_negative'`
    para que el training del ranker (rag_ranker_lgbm.features.feedback_to_training_data)
    aplique `weight=0.3` en vez de 1.0, atenuando el costo si la inferencia
    es errada.
  - Si `top_score >= 0.4` o no hay top_score registrado: ambigüedad real
    (encontró algo razonable pero no actuó), seguimos skipeando.
  - El confidence gate NO aplica al branch weak_negative: el punto del
    feature es absorber datos débiles (confidence ~0.4-0.5).

Idempotencia: cada feedback implícito tiene `implicit_loss_source` que
empieza con `session_outcome_` — re-runs check that mismo turn no
tenga feedback con ese source.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from rag_implicit_learning.session_outcome import (
    SessionAnalysis,
    classify_recent_sessions,
)

# Confidence mínima para shapeear reward. Heurísticas con conf <0.7 son
# inferencias débiles ("silencio implica win") — no las usamos como
# signal propagada porque una falsa positiva en un win contamina todo
# el ranker.
DEFAULT_MIN_CONFIDENCE = 0.7

# 2026-04-29 — weak negative threshold for abandon sessions.
# Si el top_score del primer turn de la session estaba <0.4 (weak match),
# tratamos el abandon como negativo débil — la query no encontró nada
# satisfactorio y el user se fue. Un top_score >=0.4 con abandon indica
# ambigüedad real (encontró algo razonable pero no actuó), seguimos
# skipeando como antes.
#
# Pre-fix había 542 abandon sessions skipped vs 18 loss confirmados →
# asimetría 30:1 en signal negativa. Post-fix esperamos absorber ~50-100
# negativos débiles por semana sin contaminar los positivos (rating
# queda -1 igual que un loss "fuerte", pero el extra_json marca el
# source para que el training los re-pese con weight=0.3 vs 1.0 de
# un loss real).
WEAK_NEGATIVE_TOP_SCORE_THRESHOLD: float = 0.4
WEAK_NEGATIVE_SOURCE: str = "session_outcome_weak_negative"


def _ensure_feedback_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_feedback (
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


def _existing_feedback_for_turn(
    conn: sqlite3.Connection, turn_id: str
) -> dict[str, Any]:
    """Devuelve info sobre feedback ya existente para este turn_id.

    El schema real tiene UNIQUE(turn_id, rating, ts) — si ya hay un row
    con el mismo (turn_id, rating, ts), un INSERT explota.

    Para reward shaping nos importa:
    - Si ya hay feedback EXPLÍCITO (sin implicit_loss_source) → skip,
      no overlappear señales contradictorias.
    - Si ya hay un feedback implícito de ANY source para este turn →
      skip, queremos UN solo signal implícito por turn (el más fuerte
      ya está).
    - El UNIQUE constraint lo respetamos by skipping siempre que haya
      cualquier row con turn_id existente.
    """
    sources: set[str] = set()
    has_explicit = False
    rows = conn.execute(
        "SELECT extra_json, rating FROM rag_feedback WHERE turn_id = ?",
        (turn_id,),
    ).fetchall()
    for extra_json_str, rating in rows:
        try:
            extra = json.loads(extra_json_str or "{}")
        except (json.JSONDecodeError, TypeError):
            extra = {}
        src = extra.get("implicit_loss_source")
        if src:
            sources.add(src)
        else:
            has_explicit = True
    return {
        "implicit_sources": sources,
        "has_explicit": has_explicit,
        "any": bool(rows),
    }


def _turns_in_session(
    conn: sqlite3.Connection, session_id: str
) -> list[tuple[int, str, str, str | None]]:
    """Lista de (id, ts, q, paths_json) de los turns de esta session.

    Filtramos slash-commands.
    """
    rows = conn.execute(
        """
        SELECT id, ts, q, paths_json
        FROM rag_queries
        WHERE session = ? AND q IS NOT NULL AND q != ''
        ORDER BY datetime(ts) ASC
        """,
        (session_id,),
    ).fetchall()
    return [
        (rid, ts, q, paths)
        for rid, ts, q, paths in rows
        if not (q or "").lstrip().startswith("/")
    ]


def apply_reward_from_session_outcomes(
    conn: sqlite3.Connection,
    *,
    days: int = 7,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    dry_run: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Backpropaga win/loss session outcomes a feedback implícito por turn.

    Args:
        conn: connection a `telemetry.db`.
        days: ventana de sessions a clasificar.
        min_confidence: ignora analyses con confidence < esto.
        dry_run: si True, reporta sin escribir.
        now: timestamp para tests.

    Returns:
        dict con métricas + lista de updates.
    """
    _ensure_feedback_table(conn)
    now = now or datetime.now()
    now_iso = now.isoformat(timespec="seconds")

    analyses = classify_recent_sessions(conn, days=days, now=now)

    metrics: dict[str, int] = {
        "n_sessions_analyzed": len(analyses),
        "n_sessions_used_for_reward": 0,
        "n_skip_low_confidence": 0,
        "n_skip_ambiguous_outcome": 0,
        "n_turns_total": 0,
        "n_turns_skip_explicit": 0,
        "n_turns_skip_already_shaped": 0,
        "n_turns_inserted_pos": 0,
        "n_turns_inserted_neg": 0,
        # Sub-counter de n_turns_inserted_neg: cuántos de esos negativos
        # vinieron de la rama "weak negative" (abandon + top_score bajo).
        # Sirve para diagnostics — `rag feedback classify-sessions --json`
        # reporta el split.
        "n_turns_inserted_weak_negative": 0,
    }
    updates: list[dict[str, Any]] = []

    for analysis in analyses:
        # Resolución del outcome a (rating, source_tag, effective_confidence).
        # `effective_confidence` = la confidence persistida en extra_json,
        # que el branch weak_negative capa para señalizar "menos seguro
        # que un loss/win clásico" sin tocar `analysis.confidence` (el
        # rest del pipeline lo lee como inferido por session_outcome).
        rating: int
        source_tag: str
        effective_confidence: float

        if analysis.outcome == "partial":
            metrics["n_skip_ambiguous_outcome"] += 1
            continue

        if analysis.outcome == "abandon":
            # Abandon sin top_score útil = sin signal: skip ambiguo (legacy).
            # Abandon con top_score < threshold = weak negative: la query
            # no matcheó nada decente y el user se fue. Lo absorbemos como
            # rating=-1 PERO con source distintivo para que el ranker lo
            # re-pese en training (weight=0.3 vs 1.0 de un loss real).
            #
            # OJO: el confidence gate (min_confidence default 0.7) se SALTA
            # para weak_negative — el punto del feature es absorber data
            # débil que session_outcome marca con confidence ~0.4-0.5.
            turns_for_score = _turns_in_session(conn, analysis.session_id)
            if not turns_for_score:
                metrics["n_skip_ambiguous_outcome"] += 1
                continue
            first_turn_id_int = turns_for_score[0][0]
            top_score_row = conn.execute(
                "SELECT top_score FROM rag_queries WHERE id = ?",
                (first_turn_id_int,),
            ).fetchone()
            top_score: float | None = None
            if top_score_row is not None and top_score_row[0] is not None:
                try:
                    top_score = float(top_score_row[0])
                except (TypeError, ValueError):
                    top_score = None
            if (
                top_score is None
                or top_score >= WEAK_NEGATIVE_TOP_SCORE_THRESHOLD
            ):
                metrics["n_skip_ambiguous_outcome"] += 1
                continue
            rating = -1
            source_tag = WEAK_NEGATIVE_SOURCE
            # Capear effective_confidence para que el writer persista una
            # señal "menor" que la de un win/loss clásico aunque
            # session_outcome haya devuelto algo más alto. min(0.5, x)
            # es deliberado: queremos que el training pueda filtrar
            # weak_negative por confidence si así lo desea, sin perder
            # la firma textual del source.
            effective_confidence = min(analysis.confidence, 0.5)
        else:
            # win | loss — gate clásico de confidence.
            if analysis.confidence < min_confidence:
                metrics["n_skip_low_confidence"] += 1
                continue
            rating = +1 if analysis.outcome == "win" else -1
            source_tag = f"session_outcome_{analysis.outcome}"
            effective_confidence = analysis.confidence

        turns = _turns_in_session(conn, analysis.session_id)
        metrics["n_turns_total"] += len(turns)
        if not turns:
            continue

        metrics["n_sessions_used_for_reward"] += 1

        for turn_id_int, ts, q, paths_json in turns:
            turn_id = f"{analysis.session_id}:{turn_id_int}"
            existing = _existing_feedback_for_turn(conn, turn_id)
            if existing["has_explicit"]:
                metrics["n_turns_skip_explicit"] += 1
                continue
            if source_tag in existing["implicit_sources"]:
                metrics["n_turns_skip_already_shaped"] += 1
                continue
            # UNIQUE(turn_id, rating, ts) en el schema: si ya hay
            # cualquier feedback (incluido el implicit de re-query
            # detection que insertamos en otra pasada del pipeline),
            # respetamos el constraint y no creamos otro row para el
            # mismo turn. Una signal implícita ya es señal suficiente —
            # acumular múltiples implicit signals para el mismo turn
            # sería redundante (el ranker ya tiene la info).
            if existing["any"]:
                metrics["n_turns_skip_already_shaped"] += 1
                continue

            update = {
                "turn_id": turn_id,
                "session": analysis.session_id,
                "outcome": analysis.outcome,
                "rating": rating,
                "confidence": effective_confidence,
                "implicit_loss_source": source_tag,
                "ts": ts,
            }
            updates.append(update)

            # Counters miden "señal que se produjo" — funcionan igual en
            # dry-run (lo que se HABRÍA insertado) que en apply (lo que
            # SE insertó). El SQL insert solo se ejecuta si !dry_run.
            if rating > 0:
                metrics["n_turns_inserted_pos"] += 1
            else:
                metrics["n_turns_inserted_neg"] += 1
                if source_tag == WEAK_NEGATIVE_SOURCE:
                    metrics["n_turns_inserted_weak_negative"] += 1

            if not dry_run:
                extra = {
                    "session_id": analysis.session_id,
                    "implicit_loss_source": source_tag,
                    "implicit_loss_inferred_at": now_iso,
                    "session_confidence": effective_confidence,
                    "session_outcome_evidence": analysis.evidence,
                }
                conn.execute(
                    """
                    INSERT INTO rag_feedback
                      (ts, turn_id, rating, q, paths_json, extra_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now_iso,
                        turn_id,
                        rating,
                        q,
                        paths_json,
                        json.dumps(extra),
                    ),
                )

    return {
        **metrics,
        "updates": updates,
        "dry_run": dry_run,
        "min_confidence": min_confidence,
        "days": days,
    }
