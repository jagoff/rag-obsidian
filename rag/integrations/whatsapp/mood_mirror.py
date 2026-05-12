"""Mood Mirror — heads-up banner en el thread header cuando combinan
mood signals + contexto del chat sugieren cuidar la respuesta.

Wow moment objetivo (user): el header te dice "no es buen día para
esa pelea" antes de que respondas mal a algo.

Inputs:
- Mood score de hoy (`rag_mood_score_daily`).
- Hora local actual.
- Último msg inbound del peer (regex tense detection).

Outputs: `{kind, message, severity}` o None.

`kind`:
- `mood_low`: score < -0.15, msg inbound parece tense → cuidado al
  responder.
- `mood_high`: score > 0.25, sugerencia para retomar/avanzar.
- `late_night`: 23:00-05:00 + msg con tono urgente → "programá para
  mañana, no respondas ahora".
- `mood_neutral`: score muy cerca de 0, no aporta — devuelve None.

`severity`: `low | medium | high`. UI decide qué tan prominente
renderear.

Silent-fail end-to-end: si no hay mood data del día, si no hay msgs
inbound recientes, si el SQL falla → None (no banner).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime
from typing import Any

logger = logging.getLogger("rag.wa.mood_mirror")

# Thresholds calibrados desde el sample real (last 30d en
# rag_mood_score_daily). El score es ~normalmente distributed en
# [-0.4, +0.4], con std ~0.15.
_LOW_THRESHOLD = -0.15
_HIGH_THRESHOLD = 0.25
_LATE_NIGHT_HOURS = (23, 24, 0, 1, 2, 3, 4)
_RECENT_MSGS = 5  # cuántos msgs inbound looks for tense markers

# Tense detection — heurística simple. Regex que matchean tonos
# conflictuales. No es perfect; el goal es ALTA precisión (pocos
# false-positives) — sugerir intervención solo cuando realmente
# vale la pena.
_TENSE_PATTERNS = [
    re.compile(r"\b(no me hablés|no me hables|estás (loco|enojad|mal|tarad)|que (carajo|mierda)|me cag(a|aste)|en serio[?!])", re.I),
    re.compile(r"\b(necesito (que|ya)|urgente|ahora mismo|de una vez|basta|todavía no)", re.I),
    re.compile(r"\b(siempre (lo mismo|haces|me)|nunca (me|te)|cada vez que)", re.I),
    re.compile(r"\?{2,}|!{2,}"),  # "??" o "!!" = énfasis emocional
]


def _today_mood_score() -> float | None:
    """Lee el score de hoy desde rag_mood_score_daily. None si no hay
    row o si el daemon mood-poll aún no corrió este día.
    """
    try:
        from rag.integrations.whatsapp import _db_local  # noqa: PLC0415
        from rag import _ragvec_state_conn  # noqa: PLC0415

        today = datetime.now().date().isoformat()
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT score, n_signals FROM rag_mood_score_daily "
                "WHERE date = ?", (today,),
            ).fetchone()
            if not row or (row[1] or 0) == 0:
                return None
            return float(row[0])
    except Exception:
        return None


def _last_inbound_text(jid: str, limit: int = _RECENT_MSGS) -> str:
    """Concatena el text de los últimos N msgs inbound del chat.
    Vacío si no hay msgs recent ni el bridge no está disponible.
    """
    try:
        import rag as _rag  # noqa: PLC0415
        db_path = _rag.WHATSAPP_DB_PATH
        if not db_path.is_file():
            return ""
        con = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=5.0,
        )
        try:
            rows = con.execute(
                """
                SELECT content FROM messages
                WHERE chat_jid = ? AND is_from_me = 0
                  AND content IS NOT NULL AND length(trim(content)) > 0
                ORDER BY timestamp DESC LIMIT ?
                """,
                (jid, int(limit)),
            ).fetchall()
            return " | ".join((r[0] or "")[:200] for r in rows)
        finally:
            con.close()
    except Exception:
        return ""


def _matches_tense(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _TENSE_PATTERNS)


def get_hint(jid: str) -> dict[str, Any] | None:
    """Calcula el mood hint para un thread. None si no aplica.

    Logic priority order:
      1. Late-night + tense msg → 'late_night' high severity.
      2. Mood bajo + tense msg → 'mood_low' high severity.
      3. Mood bajo (sin tense) → 'mood_low' medium (general
         heads-up sin context-specific trigger).
      4. Mood alto → 'mood_high' low severity (sugerencia opcional).
    """
    if not jid or "@" not in jid:
        return None

    score = _today_mood_score()
    now = datetime.now()
    hour = now.hour
    inbound = _last_inbound_text(jid)
    tense = _matches_tense(inbound)

    is_late = hour in _LATE_NIGHT_HOURS
    is_low = score is not None and score < _LOW_THRESHOLD
    is_high = score is not None and score > _HIGH_THRESHOLD

    # (1) Late night + tense → don't respond now.
    if is_late and tense:
        return {
            "kind": "late_night",
            "severity": "high",
            "message": "Tarde y mensaje tenso. Mejor programá tu respuesta para mañana — vas a contestar distinto con luz del día.",
            "icon": "🌙",
        }

    # (2) Mood bajo + tense → cuidado.
    if is_low and tense:
        return {
            "kind": "mood_low",
            "severity": "high",
            "message": "Bajón hoy + mensaje tenso. No es buen día para esa pelea — respirá antes de mandar.",
            "icon": "⚠️",
        }

    # (3) Mood bajo solo → heads-up general.
    if is_low:
        return {
            "kind": "mood_low",
            "severity": "medium",
            "message": "Bajón hoy. Tomá nota: lo que escribas puede sonar más duro de lo que sentís.",
            "icon": "💭",
        }

    # (4) Mood alto → sugerencia.
    if is_high:
        return {
            "kind": "mood_high",
            "severity": "low",
            "message": "Buen día de mood. Buen momento para retomar pendientes con esta persona.",
            "icon": "✨",
        }

    # Late night sin tense → sugerencia suave de programar.
    if is_late and inbound:
        return {
            "kind": "late_night",
            "severity": "low",
            "message": "Hora tardía. Si no es urgente, considerá programar la respuesta para la mañana.",
            "icon": "🌙",
        }

    return None


__all__ = ["get_hint"]
