"""Signal — Pregunta WA inbound sin responder hace ≥48h.

Subset de `stale_contact` pero con filtro adicional: el mensaje DEBE ser
una pregunta. Una pregunta deserves answer más que un statement (`ok`,
`ya llegué`), así que vale la pena un push aunque el mensaje "se enfríe"
después de 48h cuando `stale_contact` ya cerró su ventana (72h max).

Diferenciadores vs `stale_contact`:

- **Window**: 48h-30d (vs stale_contact 3-72h). Las preguntas valen pena
  recordar incluso semanas después; los statements no.
- **Filtro question-ness**: content debe contener `?` O empezar con
  interrogativos rioplatenses (`qué`, `cómo`, `cuándo`, `dónde`,
  `por qué`, `cuál`, `cuánt`, `quién`, `de dónde`, `para qué`).
- **Max 1 candidate**: preguntas son higher-signal, 1 push focalizado.
- **Snooze 48h**: re-emit cada 2 días si sigue sin responder, no diario.
- **Anti-greetings**: skip preguntas conocidas-trivia (`¿cómo estás?`,
  `¿todo bien?`, `¿qué hacés?`) que son saludos no question reales.

Score:
    >7d   → 1.0 (urgente — la persona ya se olvidó que preguntó pero
             vos sí podés rescatarlo)
    4-7d  → 0.8
    48h-4d → 0.6
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from rag_anticipate.signals.base import register_signal


# Window: pregunta sin responder hace ≥48h. Cap arriba a 30d para no
# rebuscar mensajes que ya son arqueología — si la persona preguntó
# hace 2 meses y no le contestaste, el push de hoy es weird.
_QUESTION_HOURS_MIN = 48
_QUESTION_HOURS_MAX = 24 * 30  # 30 días

# Solo 1 candidate por run — preguntas son higher signal que stale_contact.
_MAX_CANDIDATES_PER_RUN = 1

# Score buckets — más vieja la pregunta, más alto el score (es claro que
# se quedó atrás, no es ruido reciente).
_SCORE_BUCKETS = [
    (24 * 7, 1.0),   # >7 días
    (24 * 4, 0.8),   # 4-7 días
    (48, 0.6),       # 48h-4 días
]

# Interrogativos rioplatenses que abren preguntas legítimas.
_INTERROGATIVES = (
    "qué", "que ", "cómo", "como ", "cuándo", "cuando ", "dónde", "donde ",
    "por qué", "porque ", "cuál", "cual ", "cuánto", "cuanto ", "cuánta",
    "cuanta ", "quién", "quien ",
)

# Greetings comunes que SON preguntas técnicamente pero no merecen push.
# Match cuando es lo único / casi único del mensaje (≤30 chars).
_GREETING_QUESTION_PHRASES = (
    "como estas", "cómo estás", "como andas", "cómo andás",
    "todo bien", "todo ok", "que tal", "qué tal",
    "que hacés", "qué hacés", "que haces", "qué haces",
    "estás", "estas",
)


def _is_question(content: str) -> bool:
    """True si content contiene `?` o empieza con interrogativo."""
    s = content.strip().lower()
    if not s:
        return False
    if "?" in s or "¿" in s:
        return True
    # Empieza con interrogativo (sin signo de pregunta — WA users
    # tipean rápido y omiten signos).
    for q in _INTERROGATIVES:
        if s.startswith(q):
            return True
    return False


def _is_greeting_question(content: str) -> bool:
    """True si content es un greeting estándar (no question real)."""
    s = content.strip().lower()
    # Sacar puntuación + emoji básico para comparar
    cleaned = "".join(ch for ch in s if ch.isalnum() or ch.isspace()).strip()
    if not cleaned:
        return False
    # Greetings son cortos por naturaleza
    if len(cleaned) > 30:
        return False
    return any(g in cleaned for g in _GREETING_QUESTION_PHRASES)


@register_signal(name="open_loop_question", snooze_hours=48)
def open_loop_question_signal(now: datetime) -> list:
    """Detecta preguntas WA sin responder hace ≥48h."""
    try:
        from rag.anticipatory import AnticipatoryCandidate  # noqa: PLC0415
        from rag.integrations.whatsapp import (  # noqa: PLC0415
            WHATSAPP_DB_PATH, WHATSAPP_BOT_JID,
        )
    except Exception:
        return []
    if not WHATSAPP_DB_PATH.is_file():
        return []

    cutoff_max = now - timedelta(hours=_QUESTION_HOURS_MAX)
    cutoff_min = now - timedelta(hours=_QUESTION_HOURS_MIN)

    def _fmt_bound(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S-03:00")

    try:
        con = sqlite3.connect(
            f"file:{WHATSAPP_DB_PATH}?mode=ro", uri=True, timeout=5.0,
        )
        con.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []

    out: list = []
    try:
        try:
            rows = con.execute(
                """
                SELECT
                  m.id AS msg_id,
                  m.chat_jid AS jid,
                  m.content AS content,
                  m.timestamp AS ts,
                  c.name AS chat_name
                FROM messages m
                LEFT JOIN chats c ON c.jid = m.chat_jid
                WHERE m.is_from_me = 0
                  AND m.chat_jid LIKE '%@s.whatsapp.net'
                  AND m.chat_jid != ?
                  AND m.timestamp >= ?
                  AND m.timestamp <= ?
                  AND m.content IS NOT NULL AND m.content != ''
                ORDER BY m.timestamp DESC
                """,
                (WHATSAPP_BOT_JID, _fmt_bound(cutoff_max), _fmt_bound(cutoff_min)),
            ).fetchall()
        except sqlite3.Error:
            return []

        # Group por chat — solo última pregunta unanswered por chat.
        seen_chats: set = set()
        for row in rows or ():
            jid = row["jid"]
            if jid in seen_chats:
                continue

            content = (row["content"] or "").strip()
            if not content or content.startswith("​"):  # U+200B marker
                seen_chats.add(jid)
                continue

            if not _is_question(content):
                # NO marcamos seen_chats — un mensaje no-pregunta podría
                # estar followed-by una pregunta más vieja. Pero el sort DESC
                # ya nos da la más reciente primero — si la más reciente NO
                # es pregunta, el chat queda sin candidate.
                seen_chats.add(jid)
                continue

            if _is_greeting_question(content):
                seen_chats.add(jid)
                continue

            # Verificar que NO respondió después de este mensaje.
            try:
                later = con.execute(
                    "SELECT 1 FROM messages "
                    "WHERE chat_jid = ? AND is_from_me = 1 "
                    "AND timestamp > ? LIMIT 1",
                    (jid, row["ts"]),
                ).fetchone()
            except sqlite3.Error:
                seen_chats.add(jid)
                continue
            if later:
                seen_chats.add(jid)
                continue

            # Skip senders sin nombre / con flag bot.
            chat_name = (row["chat_name"] or "").strip()
            if not chat_name or not any(ch.isalpha() for ch in chat_name):
                seen_chats.add(jid)
                continue
            lower = chat_name.lower()
            if any(b in lower for b in ("bot", "noreply", "no-reply", "alert", "notif")):
                seen_chats.add(jid)
                continue

            try:
                msg_ts = datetime.strptime(row["ts"][:19], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                seen_chats.add(jid)
                continue

            hours_ago = (now - msg_ts).total_seconds() / 3600.0
            score = 0.0
            for thresh, s in _SCORE_BUCKETS:
                if hours_ago >= thresh:
                    score = s
                    break
            if score < 0.5:
                seen_chats.add(jid)
                continue

            snippet = content.replace("\n", " ")
            if len(snippet) > 140:
                snippet = snippet[:137] + "…"

            if hours_ago < 24:
                relative = f"hace {int(hours_ago)}hs"
            elif hours_ago < 24 * 7:
                relative = f"hace {int(hours_ago / 24)} días"
            else:
                relative = f"hace {int(hours_ago / 24)}d"

            message = (
                f"❓ *{chat_name}* te preguntó {relative} y no respondiste:\n"
                f'"{snippet}"\n'
                f"  ¿Te toca contestar o lo cerrás como 'visto y listo'?"
            )

            out.append(AnticipatoryCandidate(
                kind="anticipate-open_loop_question",
                score=score,
                message=message,
                dedup_key=f"open_loop_question:{jid}:{row['msg_id']}",
                snooze_hours=48,
                reason=(
                    f"chat={chat_name} hours_ago={hours_ago:.1f} "
                    f"msg_id={row['msg_id']}"
                ),
            ))
            seen_chats.add(jid)

            if len(out) >= _MAX_CANDIDATES_PER_RUN:
                break

    except sqlite3.Error:
        pass
    finally:
        try:
            con.close()
        except Exception:
            pass

    return out
