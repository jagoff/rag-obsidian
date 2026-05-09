"""WAL checkpointer watchdog — daemon thread que mantiene el WAL chico.

Phase 2a de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer el WAL checkpointer desde `rag/__init__.py`.

## Motivación (audit 2026-04-24)

402 `semantic_cache_store_failed` + 316 `queries_sql_write_failed` +
177 `behavior_priors_sql_read_failed` en 7 días, dominados por
`OperationalError('database is locked')` y `disk I/O error`. El WAL
de ragvec.db acumulaba hasta 22 MB sin checkpoint (visible pre-fix
con `ls -lh .../ragvec.db-wal`). Causa: SQLite `wal_autocheckpoint=
1000 páginas` es oportunista — se saltea cuando hay un reader
activo tocando las páginas pendientes. Bajo carga (web + watch +
listener + sampler-per-min) siempre hay un reader → el WAL crece
sin pausa hasta que un writer intenta ampliarlo y se queda contra
el busy_timeout de 30s.

## Fix

Daemon thread que corre `PRAGMA wal_checkpoint(PASSIVE)` cada 30s
sobre ambas DBs (`ragvec.db` + `telemetry.db`). PASSIVE es
no-bloqueante vs writers activos (no toma exclusive lock) y
procesa las páginas libres. Basta con una recurrencia frecuente
para que el WAL se mantenga cerca de 0.

## Rollback / tuning

- `RAG_WAL_CHECKPOINT_DISABLE=1` → opt-out completo
- `RAG_WAL_CHECKPOINT_INTERVAL=N` → segundos entre checkpoints
  (default 30; 60 si vale tolerar más WAL growth con menos CPU)

## Lazy imports

Este módulo depende de `DB_PATH`, `_TELEMETRY_DB_FILENAME`,
`_silent_log` y `_daemon_shutdown_event` — todos definidos en
`rag/__init__.py`. Lazy imports adentro de las funciones evitan
circular import.

## Re-export

`rag/__init__.py` hace `from rag._wal_watchdog import *  # noqa`.
Preserva 100% compat con call sites históricos
(`rag.start_wal_checkpointer()`).
"""

from __future__ import annotations

import os
import threading

__all__ = [
    "_wal_checkpointer_started",
    "_wal_checkpointer_lock",
    "_wal_checkpoint_once",
    "_wal_checkpointer_loop",
    "start_wal_checkpointer",
]


_wal_checkpointer_started = False
_wal_checkpointer_lock = threading.Lock()


def _wal_checkpoint_once(path: str) -> tuple[bool, int]:
    """Ejecuta `PRAGMA wal_checkpoint(PASSIVE)` sobre una DB.

    Returns `(ok, pages_checkpointed)`. `ok=False` con `pages=0` indica
    que la DB no existe (o el archivo es pre-WAL) — silencioso.
    Cualquier otra excepción se swallowea + logea para no tumbar el
    loop del watchdog.
    """
    import sqlite3 as _sqlite3  # noqa: PLC0415

    from rag import _silent_log  # noqa: PLC0415

    if not os.path.exists(path):
        return (False, 0)
    try:
        conn = _sqlite3.connect(path, isolation_level=None,
                                check_same_thread=False, timeout=10.0)
        try:
            conn.execute("PRAGMA busy_timeout=10000")
            cur = conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            row = cur.fetchone()
            # row = (busy, log_pages, checkpointed_pages)
            pages = int(row[2]) if row and row[2] is not None else 0
            return (True, pages)
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as exc:
        _silent_log(f"wal_checkpoint:{os.path.basename(path)}", exc)
        return (False, 0)


def _wal_checkpointer_loop(interval: int) -> None:
    """Loop del thread daemon. PASSIVE checkpoint cada `interval` segundos."""
    # Import tardío: DB_PATH se settea al importar el módulo, pero
    # resolvemos los paths acá para tolerar reconfigs futuras.
    from rag import (  # noqa: PLC0415
        DB_PATH,
        _TELEMETRY_DB_FILENAME,
        _daemon_shutdown_event,
        _silent_log,
    )

    while True:
        try:
            # `wait()` retorna True si el event fue set (shutdown solicitado),
            # False si timeout — equivalente a sleep pero interruptible.
            if _daemon_shutdown_event.wait(timeout=interval):
                return
            for name in ("ragvec.db", _TELEMETRY_DB_FILENAME):
                _wal_checkpoint_once(str(DB_PATH / name))
        except Exception as exc:
            _silent_log("wal_checkpointer_loop", exc)


def start_wal_checkpointer() -> bool:
    """Arrancar el WAL checkpointer como daemon thread. Idempotente.

    Llamar desde long-running processes (`rag serve`, web startup). El
    CLI one-shot NO lo necesita — el proceso termina antes del primer
    tick y el close final corre su propio checkpoint implícito.

    Retorna True si arrancó (o ya estaba), False si se skippeó por
    `RAG_WAL_CHECKPOINT_DISABLE=1`.
    """
    global _wal_checkpointer_started
    with _wal_checkpointer_lock:
        if _wal_checkpointer_started:
            return True
        if os.environ.get("RAG_WAL_CHECKPOINT_DISABLE") == "1":
            return False
        try:
            interval = int(os.environ.get("RAG_WAL_CHECKPOINT_INTERVAL", "30"))
        except ValueError:
            interval = 30
        if interval < 5:
            interval = 5  # clamp — <5s no tiene sentido, solo quema CPU
        t = threading.Thread(
            target=_wal_checkpointer_loop,
            args=(interval,),
            name="rag-wal-checkpointer",
            daemon=True,
        )
        t.start()
        _wal_checkpointer_started = True
        print(
            f"[wal-checkpointer] started interval={interval}s",
            flush=True,
        )
        return True
