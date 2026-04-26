"""Clasificar sessions completas como win / loss / abandon / partial.

Cada session de chat tiene un outcome implícito. Si la podemos clasificar
con confianza, ese outcome es signal de calidad GLOBAL — no por turn,
por la session entera. Eso permite reward shaping: backpropagar el
outcome a los ranking features de cada turn que la compone.

Heurísticas (ordenadas por prioridad):

  1. **Win por keyword positivo final**: el último turn del user contiene
     una expresión de gratitud / confirmación clara ("gracias", "perfecto",
     "exacto", "listo", "anda", "thanks"). El user "cerró" la session
     con una afirmación → todos los turns previos contribuyeron al win.

  2. **Loss por keyword negativo final**: el último turn dice "no", "mal",
     "incorrecto", "estás equivocado", etc. Cierre negativo explícito.

  3. **Loss por re-query inmediato**: dentro de la session hay al menos
     una re-query <30s tras una respuesta (lo detecta `requery_detection`).
     Hay momentos de loss puntual aunque no haya cierre negativo.

  4. **Win implícito por silencio**: la session cerró (5+ minutos sin
     turns nuevos) y NO hubo re-queries internos → asumimos que el user
     consiguió lo que necesitaba y pasó a otra cosa.

  5. **Abandon**: sólo 1 turn y nunca se volvió a la session. Sin signal.

  6. **Partial**: 3+ turns en el mismo tema (chain), no clear win ni
     loss. Engagement positivo pero ambiguo.

Ouput por session: `(outcome, confidence, evidence)`.
- `outcome ∈ {win, loss, abandon, partial}`
- `confidence ∈ [0, 1]` — 1 si keyword explícito, 0.5 si silencio inferido
- `evidence` es un dict con qué triggereó la decisión

Uso aguas arriba: el `reward_shaping` toma `outcome` y `confidence`
y lo aplica como peso multiplicador a `feedback_pos` / `feedback_neg`
de cada turn de la session — turns de una session "win" reciben +1
implícito ponderado, turns de "loss" reciben -1. La idea es **sumar
densidad de signal** sin esperar que el user toque botones.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

# Ventana para considerar la session "cerrada" sin re-query: si pasaron
# >= esto desde el último turn, asumimos que el user pasó a otra cosa.
DEFAULT_SESSION_CLOSE_AFTER_SECONDS = 300

# Para considerar re-query interna (signal local de loss). Reusa el
# threshold del detector standalone.
DEFAULT_REQUERY_WINDOW_SECONDS = 30

# Min turns para clasificar como "partial" (engagement chain). Menos de
# eso queda como abandon o se resuelve con keywords.
DEFAULT_MIN_PARTIAL_TURNS = 3


_POSITIVE_KW_PATTERN = re.compile(
    r"\b("
    r"gracias|perfecto|exacto|excelente|genial|listo|copado|"
    r"thanks?|perfect|exactly|got it|nice|great|"
    r"funciona|funciono|funcionó|anda|andó|"
    r"correcto|sí[\s,!.]?\s*es\s+as[ií]"
    r")\b",
    re.IGNORECASE,
)

_NEGATIVE_KW_PATTERN = re.compile(
    r"\b("
    r"mal|incorrecto|equivocad[oa]s?|err[oó]neo|"
    r"no\s+es\s+(?:as[ií]|eso|correcto)|"
    r"wrong|incorrect|nope|not\s+(?:correct|right|true)|"
    r"se\s+confunde|te\s+equivocas|error"
    r")\b",
    re.IGNORECASE,
)


Outcome = Literal["win", "loss", "abandon", "partial"]


@dataclass
class SessionAnalysis:
    """Resultado de clasificar una session."""

    session_id: str
    outcome: Outcome
    confidence: float  # 0.0 - 1.0
    n_turns: int
    first_ts: str
    last_ts: str
    evidence: dict[str, Any] = field(default_factory=dict)


def _detect_keyword_signal(
    text: str,
) -> tuple[Literal["positive", "negative"] | None, str | None]:
    """¿Hay keyword explícito de win o loss en este texto?

    Returns: (signal_type, matched_keyword) o (None, None).
    Negativo gana sobre positivo si ambos están — más informativo.
    """
    if not text:
        return None, None
    neg = _NEGATIVE_KW_PATTERN.search(text)
    if neg:
        return "negative", neg.group(0)
    pos = _POSITIVE_KW_PATTERN.search(text)
    if pos:
        return "positive", pos.group(0)
    return None, None


def _has_internal_requery(
    turns: list[tuple[str, str]],
    *,
    window_seconds: int,
    similarity_check: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """¿Hay un re-query interno en esta session?

    Si hay un par consecutivo con delta <window AND queries similares,
    reportamos True.

    Args:
        turns: lista de (ts, query_text), ordenada por ts asc.
        window_seconds: gap máximo entre turns.
        similarity_check: si True, requiere paráfrasis (no solo proximidad).

    Returns: (has_requery, evidence_dict)
    """
    # Lazy import para evitar circular en testing setups.
    from rag_implicit_learning.requery_detection import is_paraphrase

    for i in range(1, len(turns)):
        prev_ts, prev_q = turns[i - 1]
        curr_ts, curr_q = turns[i]
        # SQL-style julianday delta — usamos datetime parsing simple.
        try:
            prev_dt = datetime.fromisoformat(prev_ts.replace("Z", "+00:00"))
            curr_dt = datetime.fromisoformat(curr_ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        delta = (curr_dt - prev_dt).total_seconds()
        if delta > window_seconds:
            continue
        if not similarity_check or is_paraphrase(prev_q, curr_q):
            return True, {
                "prev_query": prev_q,
                "next_query": curr_q,
                "delta_seconds": delta,
            }
    return False, {}


def classify_session(
    session_id: str,
    turns: list[tuple[str, str]],  # [(ts, query), ...]
    *,
    now: datetime | None = None,
    session_close_after_seconds: int = DEFAULT_SESSION_CLOSE_AFTER_SECONDS,
    requery_window_seconds: int = DEFAULT_REQUERY_WINDOW_SECONDS,
    min_partial_turns: int = DEFAULT_MIN_PARTIAL_TURNS,
) -> SessionAnalysis:
    """Clasificar una session a partir de sus turns.

    `turns` es lista de tuplas (ts ISO, query texto), ordenada por ts asc.
    """
    now = now or datetime.now()

    if not turns:
        return SessionAnalysis(
            session_id=session_id,
            outcome="abandon",
            confidence=1.0,
            n_turns=0,
            first_ts="",
            last_ts="",
            evidence={"reason": "empty_session"},
        )

    n = len(turns)
    first_ts = turns[0][0]
    last_ts = turns[-1][0]

    # Heurísticas 1 + 2: keyword en el último turn.
    last_query = turns[-1][1]
    signal, matched = _detect_keyword_signal(last_query)
    if signal == "negative":
        return SessionAnalysis(
            session_id=session_id,
            outcome="loss",
            confidence=0.95,
            n_turns=n,
            first_ts=first_ts,
            last_ts=last_ts,
            evidence={
                "rule": "negative_keyword_in_last_turn",
                "matched": matched,
                "last_query": last_query,
            },
        )
    if signal == "positive":
        return SessionAnalysis(
            session_id=session_id,
            outcome="win",
            confidence=0.95,
            n_turns=n,
            first_ts=first_ts,
            last_ts=last_ts,
            evidence={
                "rule": "positive_keyword_in_last_turn",
                "matched": matched,
                "last_query": last_query,
            },
        )

    # Heurística 3: re-query interno → al menos un loss puntual.
    has_requery, requery_ev = _has_internal_requery(
        turns, window_seconds=requery_window_seconds
    )
    if has_requery:
        return SessionAnalysis(
            session_id=session_id,
            outcome="loss",
            confidence=0.7,
            n_turns=n,
            first_ts=first_ts,
            last_ts=last_ts,
            evidence={"rule": "internal_requery", **requery_ev},
        )

    # Heurística 4: silencio largo después del último turn → win implícito.
    try:
        last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        time_since_last = (now - last_dt).total_seconds()
    except (ValueError, AttributeError):
        time_since_last = 0.0

    if time_since_last >= session_close_after_seconds:
        if n == 1:
            # Una sola query y silencio: hard to say. Le damos abandon
            # con confidence baja — quizás el user encontró la respuesta
            # rápido, quizás se fue sin nada.
            return SessionAnalysis(
                session_id=session_id,
                outcome="abandon",
                confidence=0.4,
                n_turns=n,
                first_ts=first_ts,
                last_ts=last_ts,
                evidence={
                    "rule": "single_turn_then_silence",
                    "silence_seconds": int(time_since_last),
                },
            )
        # 2+ turns sin re-query y silencio: win implícito.
        return SessionAnalysis(
            session_id=session_id,
            outcome="win",
            confidence=0.5,
            n_turns=n,
            first_ts=first_ts,
            last_ts=last_ts,
            evidence={
                "rule": "silence_after_chain_no_requery",
                "silence_seconds": int(time_since_last),
            },
        )

    # Heurística 5: un solo turn y todavía abierta → abandon temprano.
    if n == 1:
        return SessionAnalysis(
            session_id=session_id,
            outcome="abandon",
            confidence=0.5,
            n_turns=n,
            first_ts=first_ts,
            last_ts=last_ts,
            evidence={"rule": "single_turn_session_still_open"},
        )

    # Heurística 6: chain engagement, no clear outcome → partial.
    if n >= min_partial_turns:
        return SessionAnalysis(
            session_id=session_id,
            outcome="partial",
            confidence=0.5,
            n_turns=n,
            first_ts=first_ts,
            last_ts=last_ts,
            evidence={"rule": "chain_engagement_no_keyword"},
        )

    # 2 turns sin re-query, sin keyword, todavía abierta: abandon
    # temprano de chain.
    return SessionAnalysis(
        session_id=session_id,
        outcome="abandon",
        confidence=0.4,
        n_turns=n,
        first_ts=first_ts,
        last_ts=last_ts,
        evidence={"rule": "two_turn_session_no_clear_signal"},
    )


def classify_recent_sessions(
    conn: sqlite3.Connection,
    *,
    days: int = 7,
    now: datetime | None = None,
) -> list[SessionAnalysis]:
    """Clasificar todas las sessions con queries en los últimos `days` días.

    Lee de `rag_queries` agrupando por `session`. Skipea sessions vacías
    o sin nombre. Slash-commands (`/q`, `/clear`, etc.) NO cuentan como
    turns reales — los filtramos.
    """
    now = now or datetime.now()
    rows = conn.execute(
        """
        SELECT session, ts, q
        FROM rag_queries
        WHERE session IS NOT NULL
          AND session != ''
          AND q IS NOT NULL
          AND q != ''
          AND datetime(ts) >= datetime('now', '-' || ? || ' days')
        ORDER BY session ASC, datetime(ts) ASC
        """,
        (int(days),),
    ).fetchall()

    by_session: dict[str, list[tuple[str, str]]] = {}
    for session, ts, q in rows:
        if (q or "").lstrip().startswith("/"):
            continue
        by_session.setdefault(session, []).append((ts, q))

    return [
        classify_session(sid, turns, now=now)
        for sid, turns in by_session.items()
    ]


def session_outcome_summary(
    analyses: list[SessionAnalysis],
) -> dict[str, Any]:
    """Resumen agregado de un batch de SessionAnalysis."""
    by_outcome: dict[str, int] = {
        "win": 0, "loss": 0, "abandon": 0, "partial": 0,
    }
    confidence_sum = 0.0
    for a in analyses:
        by_outcome[a.outcome] += 1
        confidence_sum += a.confidence
    return {
        "n_sessions": len(analyses),
        "by_outcome": by_outcome,
        "avg_confidence": (
            round(confidence_sum / len(analyses), 3) if analyses else 0.0
        ),
    }
