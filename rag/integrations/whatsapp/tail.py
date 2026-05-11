"""Async poller del bridge SQLite para alimentar SSE inbound a `/wa`.

Diseño
------
Una sola task corriendo en el event loop del web server. Cada N segundos
(default 1.0) abre conn read-only al `messages.db` del bridge y queryea
incrementalmente las 4 tablas que cambian:

- ``messages``  → emite ``new_message`` + ``chat_update``.
- ``reactions`` → emite ``reaction_changed`` (incluye remove cuando emoji='').
- ``revokes``   → emite ``message_revoked``.
- ``presence``  → emite ``presence`` (typing/recording).

High-water-marks (HWM) persistidos en ``~/.local/share/obsidian-rag/wa-hwm.json``
para sobrevivir restart sin reemitir todo el historial. Dedup ring de 2000
``(table, id)`` para evitar duplicar dentro de un mismo tick.

Fanout a multiple SSE clients via lista de ``asyncio.Queue`` registradas
con ``subscribe()``. El loop arranca lazy con el primer subscriber y
queda vivo (idle cheap) — si todos se desconectan, el loop sigue pero
sin trabajo broadcast (cada client maneja su propio disconnect).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger("rag.wa.tail")

# Intervalo de poll. Override via env para tests / debug.
_POLL_INTERVAL = float(os.environ.get("OBSIDIAN_RAG_WA_POLL_INTERVAL", "1.0"))
_DEDUP_MAX = 2000

_HWM_PATH = Path.home() / ".local/share/obsidian-rag/wa-hwm.json"

# Estado de módulo (un solo poller por proceso).
_subscribers: list[asyncio.Queue] = []
_loop_task: asyncio.Task | None = None
_dedup: dict[str, set[str]] = {"msg": set(), "react": set(), "revoke": set(), "presence": set()}
_dedup_order: dict[str, list[str]] = {"msg": [], "react": [], "revoke": [], "presence": []}


def _load_hwm() -> dict[str, str]:
    """Lee HWM persistido. Si no existe / corrupto, arranca desde NOW
    (no replay completo del historial — el cliente arrancado fresh
    espera ver mensajes nuevos desde que abrió la página).
    """
    try:
        if _HWM_PATH.is_file():
            data = json.loads(_HWM_PATH.read_text())
            if isinstance(data, dict):
                return {
                    "messages": str(data.get("messages") or _now_bridge_ts()),
                    "reactions": str(data.get("reactions") or _now_bridge_ts()),
                    "revokes": str(data.get("revokes") or _now_bridge_ts()),
                    "presence": str(data.get("presence") or _now_bridge_ts()),
                }
    except Exception as e:
        logger.warning("HWM load failed (%s); arrancando desde now", e)
    now = _now_bridge_ts()
    return {"messages": now, "reactions": now, "revokes": now, "presence": now}


def _save_hwm(hwm: dict[str, str]) -> None:
    try:
        _HWM_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HWM_PATH.write_text(json.dumps(hwm, ensure_ascii=False))
    except Exception as e:
        logger.warning("HWM save failed: %s", e)


def _now_bridge_ts() -> str:
    """Formatea ahora en el mismo formato que el bridge persiste —
    ``YYYY-MM-DD HH:MM:SS-03:00``. Permite string-lex compare.
    """
    from datetime import datetime as _dt

    from .fetch import _BRIDGE_TS_FMT, _BRIDGE_TZ_OFFSET

    return _dt.now().strftime(_BRIDGE_TS_FMT) + _BRIDGE_TZ_OFFSET


def _ring_check_add(kind: str, key: str) -> bool:
    """Returns True si la key NO estaba (nueva). False si era duplicada.
    Acota el ring a `_DEDUP_MAX` por kind (FIFO).
    """
    ring = _dedup[kind]
    if key in ring:
        return False
    ring.add(key)
    order = _dedup_order[kind]
    order.append(key)
    if len(order) > _DEDUP_MAX:
        old = order.pop(0)
        ring.discard(old)
    return True


def _open_bridge_conn() -> sqlite3.Connection | None:
    import rag as _rag

    db_path = _rag.WHATSAPP_DB_PATH
    if not db_path.is_file():
        return None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        con.row_factory = sqlite3.Row
        return con
    except sqlite3.Error as e:
        logger.warning("bridge conn failed: %s", e)
        return None


async def _broadcast(event: dict[str, Any]) -> None:
    """Push event a todos los subscribers (queues con .put_nowait).

    Si un queue está lleno (cliente lento), tira el event para ese
    cliente (los SSE clients deben tolerar drops — el broker no es
    durable).
    """
    dead: list[asyncio.Queue] = []
    for q in list(_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            logger.debug("subscriber queue full, dropping event")
        except Exception:
            dead.append(q)
    for d in dead:
        try:
            _subscribers.remove(d)
        except ValueError:
            pass


def _normalize_ts(ts: str) -> str:
    """Convierte ts del bridge a ISO 8601 (replace space por T)."""
    if not ts:
        return ""
    return ts.replace(" ", "T", 1) if " " in ts else ts


def _poll_once(hwm: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Ejecuta una pasada de polling. Devuelve (events, new_hwm).

    Llamada desde el loop async vía run_in_executor — el SQLite driver
    bloquea, no querés bloquear el event loop con I/O DB.
    """
    import rag as _rag

    bot_jid = _rag.WHATSAPP_BOT_JID
    con = _open_bridge_conn()
    if con is None:
        return [], hwm
    new_hwm = dict(hwm)
    events: list[dict[str, Any]] = []
    try:
        # ── messages
        rows = con.execute(
            """
            SELECT id, chat_jid, sender, content, timestamp, is_from_me,
                   media_type, filename, quoted_message_id, quoted_text
            FROM messages
            WHERE timestamp > ?
              AND chat_jid != ?
              AND chat_jid NOT LIKE '%status@broadcast'
            ORDER BY timestamp ASC
            LIMIT 200
            """,
            (hwm["messages"], bot_jid),
        ).fetchall()
        for r in rows:
            msg_id = r["id"] or ""
            jid = r["chat_jid"] or ""
            key = f"{jid}|{msg_id}"
            if not _ring_check_add("msg", key):
                continue
            ts = r["timestamp"] or ""
            if ts > new_hwm["messages"]:
                new_hwm["messages"] = ts
            is_from_me = bool(r["is_from_me"])
            sender = (r["sender"] or "").strip()
            content = (r["content"] or "").strip().replace("\n", " ")
            media = (r["media_type"] or "").strip()
            filename = (r["filename"] or "").strip()
            quoted_id = (r["quoted_message_id"] or "").strip()
            quoted_text = (r["quoted_text"] or "").strip()
            msg_payload = {
                "id": msg_id,
                "chat_jid": jid,
                "ts": _normalize_ts(ts),
                "sender": sender,
                "sender_label": "yo" if is_from_me else (sender.split("@")[0] or jid),
                "content": content,
                "is_from_me": is_from_me,
                "media_type": media or None,
                "filename": filename or None,
                "quoted": {"id": quoted_id, "text": quoted_text} if quoted_id else None,
                "reactions": [],
                "revoked": False,
            }
            events.append({"type": "new_message", "data": {"jid": jid, "message": msg_payload}})
            # chat_update — preview + delta unread (frontend lo aplica).
            preview = content or (f"[{media}]" if media else "")
            if len(preview) > 120:
                preview = preview[:117] + "…"
            events.append({
                "type": "chat_update",
                "data": {
                    "jid": jid,
                    "last_ts": _normalize_ts(ts),
                    "last_preview": preview,
                    "last_from_me": is_from_me,
                    "unread_delta": 0 if is_from_me else 1,
                },
            })

        # ── reactions
        rrows = con.execute(
            """
            SELECT message_id, chat_jid, sender_jid, emoji, ts
            FROM reactions
            WHERE ts > ?
            ORDER BY ts ASC
            LIMIT 100
            """,
            (hwm["reactions"],),
        ).fetchall()
        for r in rrows:
            key = f"{r['message_id']}|{r['sender_jid']}|{r['ts']}"
            if not _ring_check_add("react", key):
                continue
            ts = r["ts"] or ""
            if ts > new_hwm["reactions"]:
                new_hwm["reactions"] = ts
            events.append({
                "type": "reaction_changed",
                "data": {
                    "jid": r["chat_jid"],
                    "message_id": r["message_id"],
                    "sender_jid": r["sender_jid"],
                    "emoji": r["emoji"] or "",
                    "removed": (r["emoji"] or "") == "",
                    "ts": _normalize_ts(ts),
                },
            })

        # ── revokes
        rev_rows = con.execute(
            """
            SELECT message_id, chat_jid, revoked_by, ts
            FROM revokes
            WHERE ts > ?
            ORDER BY ts ASC
            LIMIT 100
            """,
            (hwm["revokes"],),
        ).fetchall()
        for r in rev_rows:
            key = r["message_id"] or ""
            if not _ring_check_add("revoke", key):
                continue
            ts = r["ts"] or ""
            if ts > new_hwm["revokes"]:
                new_hwm["revokes"] = ts
            events.append({
                "type": "message_revoked",
                "data": {
                    "jid": r["chat_jid"],
                    "message_id": r["message_id"],
                    "revoked_by": r["revoked_by"],
                    "ts": _normalize_ts(ts),
                },
            })

        # ── presence (typing)
        pres_rows = con.execute(
            """
            SELECT chat_jid, sender_jid, state, media, ts
            FROM presence
            WHERE ts > ?
            ORDER BY ts ASC
            LIMIT 50
            """,
            (hwm["presence"],),
        ).fetchall()
        for r in pres_rows:
            key = f"{r['chat_jid']}|{r['sender_jid']}|{r['ts']}"
            if not _ring_check_add("presence", key):
                continue
            ts = r["ts"] or ""
            if ts > new_hwm["presence"]:
                new_hwm["presence"] = ts
            events.append({
                "type": "presence",
                "data": {
                    "chat_jid": r["chat_jid"],
                    "sender_jid": r["sender_jid"],
                    "state": r["state"] or "",
                    "media": r["media"] or "",
                    "ts": _normalize_ts(ts),
                },
            })
    except sqlite3.Error as e:
        logger.warning("poll query failed: %s", e)
    finally:
        try:
            con.close()
        except Exception:
            pass
    return events, new_hwm


async def _tail_loop() -> None:
    """Forever loop: poll → broadcast → sleep."""
    logger.info("wa-tail loop started (poll=%.1fs)", _POLL_INTERVAL)
    hwm = _load_hwm()
    loop = asyncio.get_running_loop()
    save_counter = 0
    try:
        while True:
            try:
                events, new_hwm = await loop.run_in_executor(None, _poll_once, hwm)
                if events:
                    for ev in events:
                        await _broadcast(ev)
                hwm = new_hwm
                save_counter += 1
                if save_counter >= 30:  # persistir cada ~30s
                    _save_hwm(hwm)
                    save_counter = 0
            except Exception as e:
                logger.exception("wa-tail tick error: %s", e)
            await asyncio.sleep(_POLL_INTERVAL)
    except asyncio.CancelledError:
        logger.info("wa-tail loop cancelled")
        _save_hwm(hwm)
        raise


async def _ensure_loop_running() -> None:
    global _loop_task
    if _loop_task is None or _loop_task.done():
        _loop_task = asyncio.create_task(_tail_loop(), name="wa-tail")


async def subscribe(maxsize: int = 200) -> asyncio.Queue:
    """Registra un subscriber para recibir events. El caller debe llamar
    `unsubscribe(queue)` cuando se desconecte.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    _subscribers.append(q)
    await _ensure_loop_running()
    return q


def unsubscribe(queue: asyncio.Queue) -> None:
    try:
        _subscribers.remove(queue)
    except ValueError:
        pass


def subscriber_count() -> int:
    return len(_subscribers)
