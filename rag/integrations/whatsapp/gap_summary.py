"""Auto-resume al reabrir un chat después de un gap largo.

Cuando el user vuelve a un chat tras 24h+ sin mirar, mostrar 3 líneas
resumiendo qué pasó. LLM helper (`qwen2.5:3b` MLX in-process), salida
JSON estricta `{summary: [str], urgent: bool}`.

Flujo:

1. `summarize_gap(jid, threshold_unread=5, threshold_hours=24)` decide
   si vale la pena llamar al LLM.
2. Si no hay gap (poca actividad nueva o reciente) → devuelve None
   sin tocar el LLM.
3. Cache process-level (LRU 256) keyed por (jid, last_msg_id) — si el
   estado del chat no cambió desde el último resume, no re-corremos.
4. LLM call con timeout 8s; falla → None silent.

El endpoint `/api/wa/thread/{jid}/gap-summary` consume esto, separado
del fetch principal del thread para no bloquear el render con 2-5s
de LLM.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import OrderedDict
from datetime import datetime, timedelta
from threading import Lock
from typing import Any

logger = logging.getLogger("rag.wa.gap_summary")

_CACHE: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
_CACHE_LOCK = Lock()
_CACHE_MAX = 256

# Mensaje mínimo de unread + tiempo mínimo desde el último msg leído.
# Ambas condiciones deben cumplirse para gatillar el resume.
_DEFAULT_UNREAD = 5
_DEFAULT_HOURS = 24
_MAX_MSGS_IN_PROMPT = 30


def _cache_get(key: tuple[str, str]) -> dict | None:
    with _CACHE_LOCK:
        v = _CACHE.get(key)
        if v is not None:
            _CACHE.move_to_end(key)
        return v


def _cache_put(key: tuple[str, str], value: dict) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)


def _open_bridge_ro():
    import rag as _rag

    db_path = _rag.WHATSAPP_DB_PATH
    if not db_path.is_file():
        return None
    try:
        con = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=5.0,
        )
        con.row_factory = sqlite3.Row
        return con
    except sqlite3.Error:
        return None


def _fetch_unread_window(
    jid: str, last_seen_ts: str, max_msgs: int,
) -> list[dict[str, Any]]:
    """Trae msgs inbound con `ts > last_seen_ts`, hasta `max_msgs` cap.

    Solo inbound (`is_from_me=0`) — el resume es de lo que el peer dijo
    mientras el user no estaba. Si el user respondió mid-gap, sus
    propios msgs son contexto pero no necesitan resumirse.
    """
    con = _open_bridge_ro()
    if con is None:
        return []
    try:
        rows = con.execute(
            """
            SELECT id, sender, content, timestamp AS ts
            FROM messages
            WHERE chat_jid = ?
              AND is_from_me = 0
              AND timestamp > ?
              AND content IS NOT NULL
              AND length(trim(content)) > 0
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (jid, last_seen_ts or "1970-01-01T00:00:00", int(max_msgs)),
        ).fetchall()
        return [
            {"id": r["id"], "ts": r["ts"], "content": r["content"]}
            for r in rows
        ]
    except sqlite3.Error:
        return []
    finally:
        con.close()


def _hours_since(iso: str, now: datetime) -> float:
    """Horas entre `iso` y `now`. Defensivo ante tz-aware mixto: si el
    iso trae offset (`+00:00`, `-03:00`), lo strippeamos para comparar
    naive contra naive — el bridge guarda local time AR, `datetime.now()`
    también local, así que dropear el offset es seguro y evita el
    TypeError de mixing aware/naive."""
    if not iso:
        return float("inf")
    s = iso.replace(" ", "T", 1)
    # Strip tz offset (`+00:00`, `-03:00`, `Z`).
    for sep in ("+", "Z"):
        if sep in s and s.index(sep) > 10:
            s = s.split(sep, 1)[0]
            break
    if "-" in s[10:]:  # offset negativo después de la fecha
        s = s[:10] + s[10:].split("-", 1)[0]
    try:
        ts = datetime.fromisoformat(s)
    except ValueError:
        return float("inf")
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return max(0.0, (now - ts).total_seconds() / 3600.0)


def _build_prompt(label: str, hours_ago: float, msgs: list[dict]) -> str:
    convo_lines = []
    for m in msgs[-_MAX_MSGS_IN_PROMPT:]:
        ts = (m.get("ts") or "")[:16].replace("T", " ")
        text = (m.get("content") or "").strip()
        if not text:
            continue
        convo_lines.append(f"[{ts}] {text}")
    convo = "\n".join(convo_lines)
    if len(convo) > 4000:
        convo = convo[-4000:]
    h_round = int(hours_ago) if hours_ago < 100 else 100
    return (
        "IDIOMA: español rioplatense (voseo argentino). Nunca portugués "
        "ni tuteo peninsular.\n\n"
        f"Te perdiste {h_round}h de conversación con {label}. Esto es "
        "lo que dijo el peer mientras no estabas:\n\n"
        f"{convo}\n\n"
        "Resumí en 3 líneas máximo LO MÁS IMPORTANTE: qué pidió, qué "
        "decisión espera respuesta tuya, si hay deadline. Cada línea "
        "<80 chars, en voseo. Ignorá saludos, memes, audios sueltos.\n\n"
        "Detectá urgencia: hay urgencia si menciona deadline ('hoy', "
        "'urgente', 'antes de las X'), queja explícita, o pregunta "
        "repetida sin respuesta.\n\n"
        "Output JSON estricto:\n"
        '{"summary": ["...", "...", "..."], "urgent": true|false}'
    )


def _llm_summarize(prompt: str) -> dict | None:
    try:
        from rag import (  # noqa: PLC0415
            HELPER_MODEL, HELPER_OPTIONS, LLM_KEEP_ALIVE, _summary_client,
        )
    except Exception as exc:
        logger.warning("imports failed: %s", exc)
        return None
    try:
        resp = _summary_client().chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={**HELPER_OPTIONS, "num_predict": 320, "num_ctx": 3072},
            keep_alive=LLM_KEEP_ALIVE,
            format="json",
        )
        raw = (resp.message.content or "").strip()
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("llm failed: %s", exc)
        return None
    if not isinstance(data, dict):
        return None
    summary = data.get("summary") or []
    if not isinstance(summary, list):
        return None
    cleaned = [
        str(s).strip() for s in summary[:5]
        if isinstance(s, str) and s.strip()
    ]
    if not cleaned:
        return None
    return {
        "summary": cleaned[:3],
        "urgent": bool(data.get("urgent")),
    }


def summarize_gap(
    jid: str,
    *,
    threshold_unread: int = _DEFAULT_UNREAD,
    threshold_hours: int = _DEFAULT_HOURS,
    label: str | None = None,
) -> dict | None:
    """Devuelve `{summary: [str], urgent: bool, msgs_count, hours_ago,
    last_msg_id}` si vale la pena resumir, o None.

    Gate: necesita >= threshold_unread msgs inbound desde el último
    `last_seen_ts` registrado Y un gap >= threshold_hours desde ese
    último seen. Si no cumple, devuelve None sin LLM call.

    Cache: (jid, latest_msg_id) — si el último msg inbound no cambió,
    devolvemos el resume cacheado.
    """
    if not jid or "@" not in jid:
        return None
    from . import _db_local  # noqa: PLC0415

    last_seen_ts = _db_local.get_last_seen(jid) or ""
    now = datetime.now()
    hours_ago = _hours_since(last_seen_ts, now) if last_seen_ts else 999.0

    # Si el user vio el chat hace menos de threshold_hours, no hay gap.
    if hours_ago < threshold_hours:
        return None

    # Trae los inbound posteriores al last_seen_ts. Cap a 50 para no
    # saturar el prompt.
    msgs = _fetch_unread_window(jid, last_seen_ts, max_msgs=50)
    if len(msgs) < threshold_unread:
        return None

    latest_msg_id = msgs[-1].get("id") or ""
    cache_key = (jid, latest_msg_id)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    label_clean = (label or jid.split("@")[0]).strip()
    prompt = _build_prompt(label_clean, hours_ago, msgs)
    llm_out = _llm_summarize(prompt)
    if not llm_out:
        return None

    result = {
        "summary": llm_out["summary"],
        "urgent": llm_out["urgent"],
        "msgs_count": len(msgs),
        "hours_ago": round(hours_ago, 1),
        "last_msg_id": latest_msg_id,
    }
    _cache_put(cache_key, result)
    return result


__all__ = ["summarize_gap"]
