"""Apple Reminders integration — leaf ETL extracted from `rag/__init__.py` (Phase 1b).

Source: macOS Reminders.app via `osascript`. Two scripts iterate every list:

- `_REMINDERS_SCRIPT` — INCOMPLETE reminders (`completed is false`), emits
  `id|name|due|list` per line. Powers `rag morning` and `rag remind` views.
- `_COMPLETED_REMINDERS_SCRIPT` — COMPLETED reminders (`completed is true`),
  emits `name|completion_date|list` per line. Powers `rag followup` so we
  can cross-resolve open vault checkboxes against tasks the user already
  closed in Reminders.app.

## Invariants
- Silent-fail: `OBSIDIAN_RAG_NO_APPLE=1`, osascript timeout (45s for due,
  60s for completed — Reminders.app gets slow with thousands of completed
  items), parse error → return `[]`. Never raise.
- Pipe-separated text format MUST stay 1-line-per-record. The `_REMINDERS_SCRIPT`
  legacy path (3-field shape `name|due|list` instead of new 4-field
  `id|name|due|list`) is intentionally kept so a script revert doesn't
  silently drop reminders for the user mid-rollback.
- Output sort order:
  - due: bucketed (overdue → today → upcoming → undated), then by due date asc.
  - completed: most recent first.

## Why deferred imports
`_apple_enabled`, `_osascript`, and `_parse_applescript_date` live in
`rag.__init__`. Module-level imports here would deadlock the package
load. Function-body imports run after `rag.__init__` finishes loading and
also resolve test-time `monkeypatch.setattr(rag, "_X", ...)` correctly,
because each call re-resolves the attribute on the `rag` module.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta


# 2026-05-01: TTL cache para `_fetch_reminders_due`. El osascript itera
# todas las listas de Reminders + todos los items con `completed is false`
# — observado 2.1s p50 en producción ([chat-timing] tool_ms=2153 dominado
# por reminders, otros tools del bucket paralelo en 240-300ms). Como el
# pre-router web dispara reminders + calendar + whatsapp_pending en
# paralelo, reminders es el "long pole" del wall-clock del chat web.
#
# Impacto: el user que hace varias preguntas seguidas ("qué tengo hoy"
# → "y mañana?" → "y la semana que viene?") en <30s reusa el resultado
# cacheado en vez de re-ejecutar osascript. Latencia del bucket paralelo
# baja de ~2s a ~300ms (calendario via icalBuddy) en hits subsiguientes.
#
# TTL 30s es seguro: si el user crea un reminder nuevo en Reminders.app
# durante el chat, lo verá en máx 30s. Conservador comparado con la
# semántica del agent (las queries del chat se procesan en horizonte de
# segundos, no de tiempo real).
#
# Override via env: `RAG_REMINDERS_CACHE_TTL=N` (segundos). 0 desactiva.
# Tests usan 0 para evitar cross-test pollution.
_REMINDERS_CACHE_LOCK = threading.Lock()
# Tuple: (timestamp_mono, raw_osascript_output). None when not yet populated.
_REMINDERS_CACHE: dict[str, tuple[float, str]] = {}


def _reminders_cache_ttl() -> float:
    try:
        return float(os.environ.get("RAG_REMINDERS_CACHE_TTL", "30"))
    except (ValueError, TypeError):
        return 30.0


def _reminders_cache_clear() -> None:
    """Test helper — wipes the cache between cases."""
    with _REMINDERS_CACHE_LOCK:
        _REMINDERS_CACHE.clear()


_REMINDERS_SCRIPT = '''
set _out to ""
tell application "Reminders"
  repeat with _list in lists
    try
      set _pending to (reminders of _list whose completed is false)
      repeat with _r in _pending
        try
          set _rid to id of _r
          set _due to due date of _r
          if _due is not missing value then
            set _out to _out & _rid & "|" & (name of _r) & "|" & (_due as string) & "|" & (name of _list) & linefeed
          else
            set _out to _out & _rid & "|" & (name of _r) & "||" & (name of _list) & linefeed
          end if
        end try
      end repeat
    end try
  end repeat
end tell
return _out
'''


# Completed reminders — para cruzar con open loops del vault (rag followup).
# Mismo shape que `_REMINDERS_SCRIPT` (name|date|list) pero filtra por
# `completed is true` y emite `completion date` en vez de `due date`.
_COMPLETED_REMINDERS_SCRIPT = '''
set _out to ""
tell application "Reminders"
  repeat with _list in lists
    try
      set _done to (reminders of _list whose completed is true)
      repeat with _r in _done
        try
          set _comp to completion date of _r
          if _comp is not missing value then
            set _out to _out & (name of _r) & "|" & (_comp as string) & "|" & (name of _list) & linefeed
          end if
        end try
      end repeat
    end try
  end repeat
end tell
return _out
'''


def _fetch_reminders_due(now: datetime, horizon_days: int = 1, max_items: int = 20) -> list[dict]:
    """Incomplete reminders with due date ≤ today + horizon_days, plus
    reminders without any due date. Splits into buckets: ``overdue`` / ``today``
    / ``upcoming`` (dated) and ``undated`` (no due). Undated reminders land at
    the bottom of the sort order — still actionable but not time-sensitive.

    The raw osascript output is cached with TTL `RAG_REMINDERS_CACHE_TTL`
    (default 30s) — see module-level docstring. Filtering by `now` /
    `horizon_days` / `max_items` always runs fresh against the cached
    output, so changing those args produces correct results within the
    TTL window.
    """
    from rag import _apple_enabled, _osascript, _parse_applescript_date
    if not _apple_enabled():
        return []
    # TTL cache lookup (key independiente de args — el script siempre
    # devuelve TODOS los reminders incompletos; el filter por horizon/
    # now/max corre Python-side abajo).
    _ttl = _reminders_cache_ttl()
    out: str = ""
    _now_mono = time.monotonic()
    if _ttl > 0:
        with _REMINDERS_CACHE_LOCK:
            _entry = _REMINDERS_CACHE.get("raw")
            if _entry is not None and (_now_mono - _entry[0]) < _ttl:
                out = _entry[1]
    if not out:
        out = _osascript(_REMINDERS_SCRIPT, timeout=45.0)
        if _ttl > 0 and out:
            with _REMINDERS_CACHE_LOCK:
                # Guardamos el raw script output (no la lista filtrada) —
                # así callers con horizon_days distintos comparten el cache.
                _REMINDERS_CACHE["raw"] = (_now_mono, out)
    if not out:
        return []
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    horizon = today + timedelta(days=horizon_days + 1)
    items: list[dict] = []
    for line in out.splitlines():
        parts = line.split("|", 3)
        # New shape: id|name|due|list. Legacy fallback (name|due|list) kept
        # so a script that reverts doesn't silently drop all reminders.
        if len(parts) == 4:
            rid, name, due_raw, list_name = (p.strip() for p in parts)
        elif len(parts) == 3:
            rid = ""
            name, due_raw, list_name = (p.strip() for p in parts)
        elif len(parts) == 2:
            rid = ""
            name, due_raw = (p.strip() for p in parts)
            list_name = ""
        else:
            continue
        if not name:
            continue
        if not due_raw:
            items.append({
                "id": rid,
                "name": name,
                "due": "",
                "list": list_name,
                "bucket": "undated",
            })
            continue
        due_dt = _parse_applescript_date(due_raw)
        if due_dt is None:
            continue
        if due_dt >= horizon:
            continue
        if due_dt < today:
            bucket = "overdue"
        elif due_dt < today + timedelta(days=1):
            bucket = "today"
        else:
            bucket = "upcoming"
        items.append({
            "id": rid,
            "name": name,
            "due": due_dt.isoformat(timespec="minutes"),
            "list": list_name,
            "bucket": bucket,
        })
    order = {"overdue": 0, "today": 1, "upcoming": 2, "undated": 3}
    items.sort(key=lambda r: (order.get(r["bucket"], 9), r["due"]))
    return items[:max_items]


def _fetch_completed_reminders(
    now: datetime, days: int = 30, max_items: int = 200,
) -> list[dict]:
    """Completed Apple Reminders in the last `days`. Used by `rag followup`
    to cross-resolve open loops in the vault against tasks already checked
    off in Reminders.app — if you closed "comprar pan" there, the vault's
    checkbox "comprar pan" is implicitly resolved.

    Silent-fail: `_apple_enabled()=False` or osascript empty → []. Same
    contract as `_fetch_reminders_due`. Shape: `[{name, completed_date,
    list}]` sorted newest-first.
    """
    from rag import _apple_enabled, _osascript, _parse_applescript_date
    if not _apple_enabled():
        return []
    out = _osascript(_COMPLETED_REMINDERS_SCRIPT, timeout=60.0)
    if not out:
        return []
    cutoff = now - timedelta(days=days)
    items: list[dict] = []
    for line in out.splitlines():
        parts = line.split("|", 2)
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        comp_raw = parts[1].strip()
        list_name = parts[2].strip() if len(parts) > 2 else ""
        if not name:
            continue
        comp_dt = _parse_applescript_date(comp_raw)
        if comp_dt is None:
            continue
        if comp_dt < cutoff:
            continue
        items.append({
            "name": name,
            "completed_date": comp_dt.isoformat(timespec="minutes"),
            "list": list_name,
        })
    items.sort(key=lambda r: r["completed_date"], reverse=True)
    return items[:max_items]
