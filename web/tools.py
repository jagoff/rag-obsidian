"""Chat-scoped tool registry for `POST /api/chat` ollama-native tool-calling.

Exports `CHAT_TOOLS` (ordered list of 7 callables), `TOOL_FNS` (name→callable
dispatch map), `PARALLEL_SAFE` (names safe to run concurrently in a thread
pool), `CHAT_TOOL_OPTIONS` (ollama options for tool-deciding call), and
`_WEB_TOOL_ADDENDUM` (constant system-prompt suffix — kept byte-identical
for ollama prefix caching).

Glue only — all real logic lives in `rag.py` / `web.server`. Each tool has a
Google-style docstring; ollama derives the JSON schema from signature +
docstring at call time.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import (  # noqa: E402
    _agent_tool_read_note,
    _agent_tool_search,
    _agent_tool_weather,
    _fetch_gmail_evidence,
    _fetch_reminders_due,
    _gmail_service,
    _gmail_thread_last_meta,
)


_WEB_TOOL_ADDENDUM: str = """Tenés 7 tools: search_vault, read_note, reminders_due, gmail_recent, finance_summary, calendar_ahead, weather. Usalas SOLO si la pregunta las necesita; si el CONTEXTO ya alcanza, respondé sin tools.
Regla de citas: cita SOLO paths del vault devueltos por search_vault/read_note. NUNCA cites thread_id, event_id, category, ni ningún identificador de tools externos (gmail/finance/calendar/reminders/weather) — esos datos van en prosa, sin `[...](...)`.
Paralelismo: podés llamar varias tools en el mismo turno si son independientes. Máximo 3 rondas.
"""


CHAT_TOOL_OPTIONS: dict = {
    "num_ctx": 4096,
    "num_predict": 512,
    "temperature": 0.0,
    "seed": 42,
}


def search_vault(query: str, k: int = 5) -> str:
    """Buscar chunks relevantes en el vault (semantic + BM25 + rerank).

    Args:
        query: Pregunta o tema a buscar, en lenguaje natural.
        k: Cantidad de chunks a devolver (1–10, default 5).

    Returns:
        JSON con lista de {note, path, score, content} por score descendente.
    """
    k = max(1, min(10, int(k)))
    return _agent_tool_search(query, k)


def read_note(path: str) -> str:
    """Leer contenido completo de una nota del vault. path debe terminar en .md.

    Args:
        path: Ruta relativa al vault (ej. "02-Areas/Coaching/Ikigai.md").

    Returns:
        Markdown completo de la nota, o mensaje de error.
    """
    return _agent_tool_read_note(path)


def reminders_due(days_ahead: int = 7) -> str:
    """Apple Reminders con fecha pendiente o sin fecha, próximos N días.

    Args:
        days_ahead: Horizonte en días (1–30, default 7).

    Returns:
        JSON `{dated: [...], undated: [...]}`. Errores → lista vacía + "error".
    """
    try:
        horizon = max(1, min(30, int(days_ahead)))
        items = _fetch_reminders_due(datetime.now(), horizon_days=horizon, max_items=40)
        dated = [r for r in items if r.get("bucket") != "undated"]
        undated = [r for r in items if r.get("bucket") == "undated"]
        return json.dumps({"dated": dated, "undated": undated}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"dated": [], "undated": [], "error": str(exc)}, ensure_ascii=False)


def gmail_recent() -> str:
    """Gmail reciente: no leídos, starred, awaiting-reply (≤8 threads).

    Returns:
        JSON `{unread_count: int, threads: [...]}`. Error → ambos vacíos.
    """
    # TODO: _fetch_gmail_evidence does not emit thread_id/received_at; we
    # re-query the Gmail service directly to enrich with thread_id + ISO
    # received_at. If creds/deps/network fail, return degraded shape.
    try:
        now = datetime.now()
        ev = _fetch_gmail_evidence(now) or {}
        unread_count = int(ev.get("unread_count") or 0)
        threads: list[dict] = []

        svc = _gmail_service()

        def _mk_thread(kind: str, item: dict) -> dict:
            base = {
                "kind": kind,
                "from": item.get("from", ""),
                "subject": item.get("subject", ""),
                "snippet": item.get("snippet", ""),
                "days_old": item.get("days_old"),
                "thread_id": "",
                "received_at": "",
            }
            return base

        awaiting = list(ev.get("awaiting_reply") or [])
        starred = list(ev.get("starred") or [])

        # awaiting first (more actionable), then starred; cap 8.
        for item in awaiting:
            if len(threads) >= 8:
                break
            threads.append(_mk_thread("awaiting_reply", item))
        for item in starred:
            if len(threads) >= 8:
                break
            threads.append(_mk_thread("starred", item))

        if svc is not None and threads:
            # Enrich with thread_id + received_at by re-querying the same
            # filters and matching by subject+from. Keeps degraded shape on
            # any single-thread failure.
            try:
                enrich_items: list[dict] = []
                try:
                    q = (
                        "in:inbox newer_than:14d older_than:3d "
                        "-category:promotions -category:social "
                        "-category:updates -category:forums"
                    )
                    r = svc.users().threads().list(userId="me", q=q, maxResults=15).execute()
                    for th in r.get("threads", []) or []:
                        tid = th.get("id") or ""
                        meta = _gmail_thread_last_meta(svc, tid)
                        if meta:
                            enrich_items.append({"thread_id": tid, "meta": meta})
                except Exception:
                    pass
                try:
                    r2 = svc.users().threads().list(
                        userId="me", q="is:starred in:inbox newer_than:7d", maxResults=3,
                    ).execute()
                    for th in r2.get("threads", []) or []:
                        tid = th.get("id") or ""
                        meta = _gmail_thread_last_meta(svc, tid)
                        if meta:
                            enrich_items.append({"thread_id": tid, "meta": meta})
                except Exception:
                    pass

                by_key = {
                    (e["meta"].get("subject", ""), e["meta"].get("from", "")): e
                    for e in enrich_items
                }
                for t in threads:
                    key = (t["subject"], t["from"])
                    e = by_key.get(key)
                    if e:
                        t["thread_id"] = e["thread_id"]
                        ms = int(e["meta"].get("internal_date_ms") or 0)
                        if ms:
                            t["received_at"] = datetime.fromtimestamp(ms / 1000.0).isoformat(timespec="minutes")
            except Exception:
                pass

        return json.dumps({"unread_count": unread_count, "threads": threads}, ensure_ascii=False)
    except Exception:
        return json.dumps({"unread_count": 0, "threads": []}, ensure_ascii=False)


def finance_summary(month: str | None = None) -> str:
    """Resumen de gastos del mes (YYYY-MM) o mes actual desde MOZE.

    Args:
        month: Mes objetivo en formato `YYYY-MM`. Default: mes actual.

    Returns:
        JSON con el summary, o `"{}"` si no hay datos / error.
    """
    try:
        from web.server import _fetch_finance  # lazy: web.server importa este módulo.

        import re as _re
        anchor = datetime.now()
        if month:
            m = _re.match(r"^(\d{4})-(\d{2})$", month.strip())
            if m:
                try:
                    anchor = datetime(int(m.group(1)), int(m.group(2)), 1)
                except ValueError:
                    anchor = datetime.now()
        result = _fetch_finance(anchor)
        return json.dumps(result or {}, ensure_ascii=False)
    except Exception:
        return "{}"


def calendar_ahead(days: int = 3) -> str:
    """Próximos eventos de calendario en los próximos N días.

    Args:
        days: Horizonte en días (1–14, default 3).

    Returns:
        JSON con lista de eventos; `"[]"` si no hay datos / error.
    """
    try:
        from web.server import _fetch_calendar_ahead  # lazy to avoid cycle.
        n = max(1, min(14, int(days)))
        result = _fetch_calendar_ahead(n, max_events=40)
        return json.dumps(result or [], ensure_ascii=False)
    except Exception:
        return "[]"


def weather(location: str | None = None) -> str:
    """Pronóstico: hoy + 2 días.

    Args:
        location: Ciudad a consultar. Default: ubicación configurada.

    Returns:
        JSON con condición actual + 3 días, o mensaje de error.
    """
    return _agent_tool_weather(location)


CHAT_TOOLS: list[Callable] = [
    search_vault,
    read_note,
    reminders_due,
    gmail_recent,
    finance_summary,
    calendar_ahead,
    weather,
]

TOOL_FNS: dict[str, Callable] = {fn.__name__: fn for fn in CHAT_TOOLS}

PARALLEL_SAFE: set[str] = {
    "weather",
    "finance_summary",
    "calendar_ahead",
    "reminders_due",
    "gmail_recent",
}
