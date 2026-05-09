"""Signal — Contacto que te escribió hace >Nh sin respuesta del user.

Detecta inbound 1:1 messages a los que el user NO respondió en las últimas
N horas. Push proactivo: "respondele a X que te escribió hace 3hs: '...'"

## Diseño

- Lee el bridge SQLite (`messages.db`) — fuente autoritativa de mensajes.
- Filtros:
  - `chat_jid LIKE '%@s.whatsapp.net'` (1:1, no grupos — el user no
    responde a grupos genéricos por default).
  - `is_from_me=0` (inbound).
  - Último msg inbound en chat > _STALE_HOURS_MIN AND < _STALE_HOURS_MAX.
  - Skip si hubo un msg outbound DEL USER después del último inbound
    (= ya respondió).
  - Skip chats con flag `WA_NOTIFY_STALE_SKIP_<jid>` (whitelist user).
- Cap: máximo `_MAX_CANDIDATES_PER_RUN` (default 3) por tick — evitar
  spam si hay backlog grande post-vacaciones.
- Score: 1.0 si hace >12hs, 0.7 si hace 6-12hs, 0.4 si hace 3-6hs.
- dedup_key: `stale-contact:<jid>:<msg_id>` — solo 1 push por mensaje
  específico unanswered.
- snooze_hours: 24 — re-push al día siguiente si sigue stale.

## Anti-noise

- Skip si el chat_jid es del bot mismo (RagNet self).
- Skip si el msg empieza con U+200B (anti-loop, son outputs nuestros).
- Skip si content es vacío (media-only sin caption).
- Skip si el sender es un bot conocido (heurística: nombre contiene
  "bot", "noreply", numeros sin alpha).

## Why deferred imports

`WHATSAPP_DB_PATH`, `_ragvec_state_conn` viven en rag/__init__.py.
Imports adentro del cuerpo evitan ciclos al load del package.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from rag_anticipate.signals.base import register_signal


# Window: solo emitir si el msg llegó hace [3h, 72h]. <3h es muy temprano
# (user puede estar en mitad de algo); >72h asumimos el user lo dejó pasar
# a propósito.
_STALE_HOURS_MIN = 3
_STALE_HOURS_MAX = 72

# Cap defensivo por tick — si volvés post-vacaciones con 20 mensajes
# sin responder, no querés 20 pushes en 1 minuto.
_MAX_CANDIDATES_PER_RUN = 3

# Score buckets — más antiguo = más urgente.
_SCORE_BUCKETS = [
    (12, 1.0),   # >12hs → urgente
    (6, 0.7),    # 6-12hs → importante
    (3, 0.4),    # 3-6hs → reminder suave
]


@register_signal(name="stale_contact", snooze_hours=24)
def _signal_stale_contact(now: datetime) -> list:
    """Detecta inbound 1:1 messages sin respuesta del user."""
    from rag.anticipatory import AnticipatoryCandidate
    from rag.integrations.whatsapp import WHATSAPP_DB_PATH, WHATSAPP_BOT_JID
    if not WHATSAPP_DB_PATH.is_file():
        return []

    cutoff_max = now - timedelta(hours=_STALE_HOURS_MAX)
    cutoff_min = now - timedelta(hours=_STALE_HOURS_MIN)

    # Format del bridge: 'YYYY-MM-DD HH:MM:SS-03:00'
    def _fmt_bound(dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S-03:00")

    out: list = []
    try:
        con = sqlite3.connect(f"file:{WHATSAPP_DB_PATH}?mode=ro", uri=True, timeout=5.0)
        con.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []

    try:
        # Pull inbound msgs en window. Pre-aggregate por chat para tener
        # el último msg sin contestar (más eficiente que full scan + filter).
        candidates = con.execute(
            """
            SELECT
              m.id AS msg_id,
              m.chat_jid AS jid,
              m.sender AS sender,
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

        # Group by chat — keep solo el LATEST inbound msg per chat (que el user
        # vió y no respondió).
        latest_per_chat: dict[str, sqlite3.Row] = {}
        for r in candidates:
            jid = r["jid"]
            if jid not in latest_per_chat:
                latest_per_chat[jid] = r

        # Para cada candidate, verificar si hubo un OUTBOUND del user después
        # del inbound (= respondió). Skip si sí.
        for jid, msg_row in latest_per_chat.items():
            try:
                later_out = con.execute(
                    "SELECT 1 FROM messages "
                    "WHERE chat_jid = ? AND is_from_me = 1 "
                    "AND timestamp > ? LIMIT 1",
                    (jid, msg_row["ts"]),
                ).fetchone()
            except sqlite3.Error:
                continue
            if later_out:
                continue  # ya respondió

            # Skip senders sospechosos (bots, sin nombre).
            chat_name = (msg_row["chat_name"] or "").strip()
            if not chat_name or not any(ch.isalpha() for ch in chat_name):
                continue
            lower_name = chat_name.lower()
            if any(b in lower_name for b in ("bot", "noreply", "no-reply", "alert", "notif")):
                continue

            # Skip si content es nuestro own marker U+200B (defensivo).
            content = (msg_row["content"] or "").strip()
            if content.startswith("​") or not content:
                continue

            # Calcular score por bucket de antigüedad.
            try:
                msg_ts = datetime.strptime(msg_row["ts"][:19], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                continue
            hours_ago = (now - msg_ts).total_seconds() / 3600.0
            score = 0.0
            for threshold, s in _SCORE_BUCKETS:
                if hours_ago >= threshold:
                    score = s
                    break
            if score < 0.3:
                continue

            # Snippet preview cap a 120 chars.
            snippet = content.replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:117] + "…"

            relative = (
                f"hace {int(hours_ago)}hs" if hours_ago < 24
                else f"hace {int(hours_ago / 24)}d"
            )

            message = (
                f"💬 *{chat_name}* te escribió {relative} y no respondiste:\n"
                f'"{snippet}"\n\n'
                f"_anticipate:stale-contact:{jid}:{msg_row['msg_id']}_"
            )

            out.append(AnticipatoryCandidate(
                kind="anticipate-stale_contact",
                score=score,
                message=message,
                dedup_key=f"stale-contact:{jid}:{msg_row['msg_id']}",
                snooze_hours=24,
                reason=f"chat={chat_name} hours_ago={hours_ago:.1f} latest_msg_id={msg_row['msg_id']}",
            ))

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
