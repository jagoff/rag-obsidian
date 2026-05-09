"""Writer dedicado para ``rag_supervisor_jobs``.

Schema:

```sql
CREATE TABLE IF NOT EXISTS rag_supervisor_jobs (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_start     TEXT NOT NULL,         -- ISO 8601 UTC
  ts_end       TEXT NOT NULL,
  job_label    TEXT NOT NULL,         -- ej. "auto_harvest", "anticipate"
  duration_s   REAL NOT NULL,
  exit_code    INTEGER NOT NULL,      -- 0 = ok, 1 = exception, 2 = skipped
  trigger      TEXT,                  -- "cron" | "interval" | "ipc" | "event"
  signals      TEXT,                  -- JSON: {"n_emitted": N, "by_kind": {...}}
  error        TEXT,                  -- str(exc) si exit_code != 0
  result       TEXT                   -- JSON serialization del result handler
);
CREATE INDEX IF NOT EXISTS ix_rag_supervisor_jobs_label_ts
  ON rag_supervisor_jobs(job_label, ts_start);
```

Por qué writer separado de ``rag/_sql_state_io.py``:

- Mantiene runtime/ aislado del paquete principal hasta F2.
- Self-contained: la primera escritura crea la tabla si no existe.
- Cuando F2 mergee al spec central de telemetría, se mueve esta entry a
  ``_TELEMETRY_TABLES`` en ``rag/__init__.py`` y se borra el ensure local.
- Retention 90d se manejará agregando ``("rag_supervisor_jobs", 90)`` al
  ``_LOG_TABLE_TTLS`` en F2.

Silent-fail: ningún error de SQL bloquea el job. Se loggea a
``~/.local/share/obsidian-rag/runtime-telemetry.error.log`` para auditoría.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_table",
    "insert_supervisor_job_run",
    "supervisor_jobs_db_path",
]

_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_supervisor_jobs ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ts_start TEXT NOT NULL,"
    " ts_end TEXT NOT NULL,"
    " job_label TEXT NOT NULL,"
    " duration_s REAL NOT NULL,"
    " exit_code INTEGER NOT NULL,"
    " trigger TEXT,"
    " signals TEXT,"
    " error TEXT,"
    " result TEXT"
    ")"
)
_DDL_INDEX = (
    "CREATE INDEX IF NOT EXISTS ix_rag_supervisor_jobs_label_ts"
    " ON rag_supervisor_jobs(job_label, ts_start)"
)


_ENSURED = False
_LOCK = threading.Lock()


def supervisor_jobs_db_path() -> Path:
    """Path a telemetry.db. Reuso del helper del paquete principal cuando
    está disponible; fallback al default para tests/standalone runs.
    """
    override = os.environ.get("OBSIDIAN_RAG_DB_PATH")
    if override:
        base = Path(override)
    else:
        base = Path.home() / ".local/share/obsidian-rag/ragvec"
    return base / "telemetry.db"


def _open_conn() -> sqlite3.Connection:
    db_path = supervisor_jobs_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def ensure_table() -> None:
    """Crea la tabla si no existe. Idempotente + cacheado per-process."""
    global _ENSURED
    if _ENSURED:
        return
    with _LOCK:
        if _ENSURED:
            return
        try:
            conn = _open_conn()
            try:
                conn.execute(_DDL)
                conn.execute(_DDL_INDEX)
            finally:
                conn.close()
            _ENSURED = True
        except Exception as exc:  # noqa: BLE001 — never block job
            logger.warning("supervisor telemetry ensure_table failed: %s", exc)


def _safe_json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        try:
            return json.dumps(str(value), ensure_ascii=False)
        except Exception:
            return None


def insert_supervisor_job_run(
    *,
    label: str,
    ts_start: float,
    duration_s: float,
    exit_code: int,
    error: str | None = None,
    result: Any = None,
    trigger: str = "cron",
    signals: dict[str, Any] | None = None,
) -> None:
    """Inserta un row a ``rag_supervisor_jobs``. Silent-fail.

    ``ts_start`` es Unix timestamp (epoch float). Se convierte a ISO UTC
    para almacenar consistente con resto de tablas de telemetría.
    """
    if os.environ.get("RAG_RUNTIME_TELEMETRY_DISABLE", "0") == "1":
        return
    try:
        ensure_table()
        ts_start_iso = (
            datetime.fromtimestamp(ts_start, tz=timezone.utc)
            .isoformat(timespec="seconds")
        )
        ts_end_iso = (
            datetime.fromtimestamp(ts_start + duration_s, tz=timezone.utc)
            .isoformat(timespec="seconds")
        )
        conn = _open_conn()
        try:
            conn.execute(
                "INSERT INTO rag_supervisor_jobs "
                "(ts_start, ts_end, job_label, duration_s, exit_code,"
                " trigger, signals, error, result)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts_start_iso,
                    ts_end_iso,
                    label,
                    float(duration_s),
                    int(exit_code),
                    trigger,
                    _safe_json(signals),
                    error,
                    _safe_json(result),
                ),
            )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 — never block job
        logger.warning(
            "supervisor telemetry insert failed for %s: %s", label, exc,
        )


def reset_for_tests() -> None:
    """Resetea el cache ensure-once. Solo para tests."""
    global _ENSURED
    with _LOCK:
        _ENSURED = False
