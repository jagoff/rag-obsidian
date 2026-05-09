"""F4.4 — IPC handler ``compute_mood`` para mood on-demand.

Complementa el cron ``mood_poll`` (30min) con un path on-demand: el web
UI puede llamar al supervisor vía IPC para recomputar el daily score
en <500ms en lugar de esperar al próximo tick.

Endpoint IPC: ``{"action": "compute_mood", "date": "YYYY-MM-DD"|null}``
- Si ``date`` es null → compute para hoy.
- Cache TTL 30min (matchea cadencia del cron) — invalida al recibir un
  POST /api/mood self-report (lo hace el web side; acá solo respondemos).

NO reemplaza el cron — lo complementa. El cron garantiza que el daily
score se compute incluso si el user nunca abre el web.

Si el módulo ``rag.mood`` no es importable (caso edge: gate
RAG_MOOD_ENABLED off), el handler retorna error y el cliente decide
fallback.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from rag.runtime import ipc

logger = logging.getLogger(__name__)


# Cache simple — single-key TTL ya que solo cacheamos "today".
_CACHE_LOCK = threading.Lock()
_CACHE_TTL_S = 1800  # 30 min — matchea la cadencia del cron mood_poll
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _cache_get(key: str) -> dict[str, Any] | None:
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > _CACHE_TTL_S:
            _CACHE.pop(key, None)
            return None
        return value


def _cache_set(key: str, value: dict[str, Any]) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), value)


def cache_invalidate() -> None:
    """Limpia todo el cache. Llamado en F4.4-followup desde el web cuando
    el user hace POST /api/mood self-report (la signal cambia y el
    cache cacheado quedaría stale)."""
    with _CACHE_LOCK:
        _CACHE.clear()


@ipc.handler("compute_mood")
def compute_mood_handler(payload: dict[str, Any]) -> dict[str, Any]:
    """IPC handler: recompute (o read del cache) del daily mood score.

    Payload:
    - ``date``: ``"YYYY-MM-DD"`` opcional. None → hoy.
    - ``force``: ``true`` para bypass del cache (default false).
    """
    date = payload.get("date")
    force = bool(payload.get("force", False))
    cache_key = date or "_today_"

    if not force:
        cached = _cache_get(cache_key)
        if cached is not None:
            # Copia shallow — sino mutar cached["cache_hit"]=True afecta
            # responses anteriores que ya retornaron al caller (alias del
            # mismo dict en _CACHE_set).
            response = dict(cached)
            response["cache_hit"] = True
            return response

    try:
        from rag.mood import compute_daily_score  # noqa: PLC0415
    except ImportError as exc:
        logger.warning("compute_mood: rag.mood no importable: %s", exc)
        return {"ok": False, "error": f"mood module not available: {exc}"}

    try:
        result = compute_daily_score(date)
    except Exception as exc:  # noqa: BLE001
        logger.exception("compute_mood: compute_daily_score raised")
        return {"ok": False, "error": str(exc)}

    response = {
        "ok": True,
        "date": result.get("date") if isinstance(result, dict) else None,
        "score": result.get("value") if isinstance(result, dict) else None,
        "n_signals": result.get("n_signals") if isinstance(result, dict) else 0,
        "sources_used": result.get("sources_used", []) if isinstance(result, dict) else [],
        "cache_hit": False,
        "computed_at": time.time(),
    }
    _cache_set(cache_key, response)
    return response


@ipc.handler("invalidate_mood_cache")
def invalidate_handler(_payload: dict[str, Any]) -> dict[str, Any]:
    """IPC: limpia el cache de mood. Llamado por el web cuando se
    inserta un manual self-report."""
    cache_invalidate()
    return {"ok": True, "invalidated": True}
