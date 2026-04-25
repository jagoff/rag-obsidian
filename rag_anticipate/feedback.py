"""Feedback loop del Anticipatory Agent.

Cuando el usuario responde por WhatsApp a un push proactivo (👍/👎/🔇),
este módulo persiste la respuesta en la tabla `rag_anticipate_feedback`
para analítica posterior y tuning de scores / thresholds / snoozes.

Diseño:
- Tabla propia (no reusa `rag_feedback` porque esa está ligada a queries
  con `turn_id`/`rating ∈ {-1, 0, 1}`; nuestro dominio es
  `{positive, negative, mute}` y la llave es `dedup_key` del candidate).
- Silent-fail en todos los writes/reads si el DB está inaccesible (la UX
  del agent nunca debe romperse por un error de telemetría).
- `_ensure_feedback_table()` idempotente — invocado por cada entry point
  público, así no hace falta un bootstrap global.
- `parse_wa_reply` es independiente del DB (parser puro de texto) — por
  eso no toca `_ensure_feedback_table`.

Llamadores previstos:
- Webhook del bridge de WhatsApp al recibir un reply a un push con
  `anticipate:<dedup_key>` en el context/metadata → `record_feedback`.
- Dashboard / `rag anticipate stats` → `feedback_stats` / `recent_feedback`.

Contract:
- `record_feedback(dedup_key, rating, *, reason='', source='wa') -> bool`
- `parse_wa_reply(reply_text) -> 'positive'|'negative'|'mute'|None`
- `feedback_stats(kind=None, days=30) -> dict`
- `recent_feedback(limit=20) -> list[dict]`

Todo retorna valores neutros en failure (False / {} / []).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import rag


# ── DDL ──────────────────────────────────────────────────────────────────────

_VALID_RATINGS: frozenset[str] = frozenset({"positive", "negative", "mute"})

_FEEDBACK_DDL: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS rag_anticipate_feedback ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ts TEXT NOT NULL,"
    " dedup_key TEXT NOT NULL,"
    " rating TEXT NOT NULL CHECK(rating IN ('positive', 'negative', 'mute')),"
    " source TEXT DEFAULT 'wa',"
    " reason TEXT"
    ")",
    "CREATE INDEX IF NOT EXISTS ix_rag_anticipate_feedback_ts "
    "ON rag_anticipate_feedback(ts)",
    "CREATE INDEX IF NOT EXISTS ix_rag_anticipate_feedback_dedup_key "
    "ON rag_anticipate_feedback(dedup_key)",
)


def _ensure_feedback_table(conn) -> None:
    """Crea tabla `rag_anticipate_feedback` + índices si no existen.

    Idempotente — se puede invocar tantas veces como haga falta, cada
    statement es CREATE IF NOT EXISTS. Usa la conn ya abierta por el
    caller (que la obtuvo de `rag._ragvec_state_conn()`).
    """
    for stmt in _FEEDBACK_DDL:
        conn.execute(stmt)


# ── Parser ───────────────────────────────────────────────────────────────────

# Emojis y tokens reconocidos. Agrupados por categoría. Los tokens de
# texto se matchean case-insensitively como substrings, así que 'SI' y
# 'si' y 'Si' dan el mismo resultado. Emojis van por `in` directo (los
# unicode chars no tienen casing).
_POSITIVE_EMOJIS: tuple[str, ...] = ("👍", "👌", "✅")
_NEGATIVE_EMOJIS: tuple[str, ...] = ("👎", "🚫", "❌")
_MUTE_EMOJIS: tuple[str, ...] = ("🔇", "🙅")

# Tokens textuales (ya lowercased en el match). :thumbsup: / :thumbsdown: /
# :mute: son los shortcodes de WhatsApp / Slack-style que algunos bridges
# expanden cuando el reply viene de un cliente que no soporta emojis.
_POSITIVE_TOKENS: tuple[str, ...] = (":thumbsup:", "util", "si", "ok")
_NEGATIVE_TOKENS: tuple[str, ...] = (":thumbsdown:", "noutil", "no")
_MUTE_TOKENS: tuple[str, ...] = (":mute:", "silenciar", "basta")


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    for needle in needles:
        if needle and needle in text:
            return True
    return False


def parse_wa_reply(reply_text: str) -> str | None:
    """Parsea el body de un reply de WhatsApp.

    Devuelve 'positive', 'negative', 'mute', o None si no hay match.

    Reglas:
    - 👍 / 👌 / ✅ / :thumbsup: / 'util' / 'si' / 'ok' → 'positive'
    - 👎 / 🚫 / ❌ / :thumbsdown: / 'no' / 'noutil' → 'negative'
    - 🔇 / 🙅 / :mute: / 'silenciar' / 'basta' → 'mute'
    - Cualquier otro → None

    Precedencia cuando hay múltiples señales en el mismo texto
    (conservador: respetar la más fuerte):
      mute > negative > positive

    Ejemplos:
      '👍' → 'positive'
      '👎 no sirvió' → 'negative'
      '🔇 basta' → 'mute'
      '👍 pero 👎' → 'negative' (conservador, respetar el negativo)
      'SI' → 'positive'
      '' / None / 'random text' → None
    """
    if not reply_text:
        return None
    normalized = reply_text.strip().lower()
    if not normalized:
        return None

    has_mute = (
        _contains_any(normalized, _MUTE_EMOJIS)
        or _contains_any(normalized, _MUTE_TOKENS)
    )
    has_negative = (
        _contains_any(normalized, _NEGATIVE_EMOJIS)
        or _contains_any(normalized, _NEGATIVE_TOKENS)
    )
    has_positive = (
        _contains_any(normalized, _POSITIVE_EMOJIS)
        or _contains_any(normalized, _POSITIVE_TOKENS)
    )

    # Precedencia conservadora: mute > negative > positive.
    # Si el user manda "👍 pero 👎" interpretamos como descontento —
    # preferimos suprimir un push innecesario a reforzar un patrón que
    # el humano marcó como ambiguo.
    if has_mute:
        return "mute"
    if has_negative:
        return "negative"
    if has_positive:
        return "positive"
    return None


# ── Writer ───────────────────────────────────────────────────────────────────


def record_feedback(
    dedup_key: str,
    rating: str,
    *,
    reason: str = "",
    source: str = "wa",
) -> bool:
    """Persiste un feedback event en `rag_anticipate_feedback`.

    Args:
        dedup_key: Identificador del candidate al que apunta el feedback
            (el mismo que se logueó en `rag_anticipate_candidates`).
        rating: 'positive' | 'negative' | 'mute'. Cualquier otro valor
            retorna False sin escribir nada.
        reason: Texto libre opcional (ej. el body completo del reply para
            debugging posterior).
        source: Origen del feedback (default 'wa' — WhatsApp). Forward-
            compatible con otros canales (web UI, CLI, etc.).

    Returns:
        True si el insert succeeded, False en cualquier error (rating
        inválido, DB inaccesible, etc.). Silent-fail — NO raise.
    """
    if not dedup_key:
        return False
    if rating not in _VALID_RATINGS:
        return False

    ts = datetime.now().isoformat(timespec="seconds")
    try:
        with rag._ragvec_state_conn() as conn:
            _ensure_feedback_table(conn)
            conn.execute(
                "INSERT INTO rag_anticipate_feedback "
                "(ts, dedup_key, rating, source, reason) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, dedup_key, rating, source, reason),
            )
            conn.commit()
        return True
    except Exception:
        return False


# ── Readers ──────────────────────────────────────────────────────────────────


def feedback_stats(kind: str | None = None, days: int = 30) -> dict:
    """Agregados de feedback en la ventana [now - days, now].

    Args:
        kind: Si se pasa un prefijo (ej. 'cal:' para candidates del signal
            calendar, 'anniv:' para anniversary), filtra por `dedup_key
            LIKE '<kind>%'`. None → sin filtro.
        days: Ventana hacia atrás en días. Default 30.

    Returns:
        Dict con contadores por rating + total + rate, del tipo:
        {
            'positive': N,
            'negative': N,
            'mute': N,
            'total': N,
            'rate': float,   # positives / total (0.0 si total == 0)
        }

        En error retorna un dict con ceros (mismo shape) para que el
        caller no tenga que chequear excepciones.
    """
    empty: dict = {
        "positive": 0,
        "negative": 0,
        "mute": 0,
        "total": 0,
        "rate": 0.0,
    }
    if days <= 0:
        return empty
    cutoff_ts = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    try:
        with rag._ragvec_state_conn() as conn:
            _ensure_feedback_table(conn)
            sql = (
                "SELECT rating, COUNT(*) FROM rag_anticipate_feedback "
                "WHERE ts >= ?"
            )
            params: list = [cutoff_ts]
            if kind:
                sql += " AND dedup_key LIKE ?"
                params.append(f"{kind}%")
            sql += " GROUP BY rating"
            rows = conn.execute(sql, tuple(params)).fetchall()
    except Exception:
        return empty

    out = dict(empty)
    for rating, count in rows:
        if rating in _VALID_RATINGS:
            out[rating] = int(count)
    out["total"] = out["positive"] + out["negative"] + out["mute"]
    out["rate"] = (out["positive"] / out["total"]) if out["total"] else 0.0
    return out


def recent_feedback(limit: int = 20) -> list[dict]:
    """Últimas N rows ordenadas desc por ts.

    Útil para un dashboard / `rag anticipate feedback tail` / debugging.
    Cada row es un dict con las columnas de la tabla.

    Returns:
        Lista de dicts. Lista vacía en error o si la tabla no tiene rows.
    """
    if limit <= 0:
        return []
    try:
        with rag._ragvec_state_conn() as conn:
            _ensure_feedback_table(conn)
            cursor = conn.execute(
                "SELECT id, ts, dedup_key, rating, source, reason "
                "FROM rag_anticipate_feedback "
                "ORDER BY ts DESC, id DESC LIMIT ?",
                (int(limit),),
            )
            cols = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
    except Exception:
        return []
    return [dict(zip(cols, row)) for row in rows]


__all__ = [
    "parse_wa_reply",
    "record_feedback",
    "feedback_stats",
    "recent_feedback",
]
