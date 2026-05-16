"""SQLite telemetry-state primitives for the RAG runtime."""
from __future__ import annotations

import contextlib
import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

__all__ = [
    "_JSON_COL_SUFFIX",
    "_sql_serialise_row",
    "_sql_append_event",
    "_sql_upsert",
    "_sql_query_window",
    "_sql_count_since",
    "_sql_max_ts",
    "_SQL_STATE_ERROR_LOG",
    "_log_sql_state_error",
    "_TRANSIENT_SQL_ERROR_TOKENS",
    "_is_transient_sql_error",
    "_sql_write_with_retry",
    "_sql_read_with_retry",
    "_telemetry_tables_ensured",
    "_telemetry_tables_ensured_mtime",
    "_telemetry_tables_ensured_lock",
    "_ragvec_state_conn",
]


@dataclass(frozen=True)
class _SqlStateDeps:
    sql_state_error_log_path: Callable[[], Path]
    db_path: Callable[[], Path]
    telemetry_db_filename: Callable[[], str]
    ensure_telemetry_tables: Callable[[sqlite3.Connection], None]
    bump_silent_log_counter: Callable[[], None]
    log_sql_state_error: Callable[..., None]
    is_transient_sql_error: Callable[[Exception], bool]


_DEPS: _SqlStateDeps | None = None


def configure_sql_state(
    *,
    sql_state_error_log_path: Callable[[], Path],
    db_path: Callable[[], Path],
    telemetry_db_filename: Callable[[], str],
    ensure_telemetry_tables: Callable[[sqlite3.Connection], None],
    bump_silent_log_counter: Callable[[], None],
    log_sql_state_error: Callable[..., None],
    is_transient_sql_error: Callable[[Exception], bool],
) -> None:
    """Wire runtime dependencies from the live `rag` facade."""
    global _DEPS, _SQL_STATE_ERROR_LOG
    _DEPS = _SqlStateDeps(
        sql_state_error_log_path=sql_state_error_log_path,
        db_path=db_path,
        telemetry_db_filename=telemetry_db_filename,
        ensure_telemetry_tables=ensure_telemetry_tables,
        bump_silent_log_counter=bump_silent_log_counter,
        log_sql_state_error=log_sql_state_error,
        is_transient_sql_error=is_transient_sql_error,
    )
    _SQL_STATE_ERROR_LOG = sql_state_error_log_path()


_SQL_STATE_ERROR_LOG = Path.home() / ".local/share/obsidian-rag/sql_state_errors.jsonl"
_JSON_COL_SUFFIX = "_json"


def _sql_serialise_row(row: dict) -> dict:
    """Return a shallow copy with dict/list values JSON-encoded for *_json columns."""
    out: dict = {}
    for k, v in row.items():
        if v is None or isinstance(v, (int, float, str, bytes)):
            out[k] = v
            continue
        if k.endswith(_JSON_COL_SUFFIX):
            out[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
        else:
            out[k] = v
    return out


def _sql_append_event(conn, table: str, row: dict) -> int:
    """INSERT row into table and return lastrowid."""
    serialised = _sql_serialise_row(row)
    cols = list(serialised.keys())
    placeholders = ",".join("?" for _ in cols)
    col_sql = ",".join(cols)
    sql = f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})"
    cur = conn.execute(sql, [serialised[c] for c in cols])
    conn.commit()
    return cur.lastrowid


def _sql_upsert(conn, table: str, row: dict, pk_cols: tuple) -> None:
    """INSERT OR REPLACE row into table."""
    del pk_cols  # Kept for the historical public signature.
    serialised = _sql_serialise_row(row)
    cols = list(serialised.keys())
    placeholders = ",".join("?" for _ in cols)
    col_sql = ",".join(cols)
    sql = f"INSERT OR REPLACE INTO {table} ({col_sql}) VALUES ({placeholders})"
    conn.execute(sql, [serialised[c] for c in cols])
    conn.commit()


def _sql_query_window(
    conn,
    table: str,
    since_ts: str,
    until_ts: str | None = None,
    where: str | None = None,
    params: tuple = (),
    max_rows: int | None = None,
) -> list:
    """SELECT rows from table in a timestamp window."""
    clauses = ["ts >= ?"]
    args: list = [since_ts]
    if until_ts is not None:
        clauses.append("ts < ?")
        args.append(until_ts)
    if where:
        clauses.append(f"({where})")
        args.extend(params)
    sql = f"SELECT * FROM {table} WHERE {' AND '.join(clauses)} ORDER BY ts"
    if max_rows is not None and max_rows > 0:
        sql += f" LIMIT {max_rows}"
    prev_row_factory = conn.row_factory
    try:
        conn.row_factory = sqlite3.Row
        return list(conn.execute(sql, args).fetchall())
    finally:
        conn.row_factory = prev_row_factory


def _sql_count_since(conn, table: str, since_ts: str) -> int:
    """Fast COUNT(*) using the table timestamp index."""
    row = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE ts >= ?", (since_ts,)
    ).fetchone()
    return int(row[0] if row else 0)


def _sql_max_ts(conn, table: str) -> str | None:
    """Return MAX(ts) for table, or None when there are no rows."""
    row = conn.execute(f"SELECT MAX(ts) FROM {table}").fetchone()
    return row[0] if row and row[0] is not None else None


def _log_sql_state_error(event_type: str, **fields) -> None:
    """Append an error record to sql_state_errors.jsonl. Never raises."""
    try:
        path = (
            _DEPS.sql_state_error_log_path()
            if _DEPS is not None
            else _SQL_STATE_ERROR_LOG
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event_type,
            **fields,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except (json.JSONDecodeError, TypeError, OSError, RuntimeError):
        pass
    try:
        if _DEPS is not None:
            _DEPS.bump_silent_log_counter()
    except Exception:
        pass


_TRANSIENT_SQL_ERROR_TOKENS: tuple[str, ...] = (
    "locked",
    "disk i/o error",
)


def _is_transient_sql_error(exc: Exception) -> bool:
    """True when an OperationalError is worth retrying."""
    msg = str(exc).lower()
    return any(tok in msg for tok in _TRANSIENT_SQL_ERROR_TOKENS)


def _emit_sql_state_error(error_tag: str, **fields) -> None:
    if _DEPS is not None:
        _DEPS.log_sql_state_error(error_tag, **fields)
    else:
        _log_sql_state_error(error_tag, **fields)


def _sql_error_is_transient(exc: Exception) -> bool:
    if _DEPS is not None:
        return bool(_DEPS.is_transient_sql_error(exc))
    return _is_transient_sql_error(exc)


def _sql_write_with_retry(write_fn, error_tag: str, *, attempts: int = 8) -> None:
    """Run a SQL-write closure with transient-error retry and silent logging."""
    for attempt in range(attempts):
        try:
            write_fn()
            return
        except sqlite3.OperationalError as exc:
            if not _sql_error_is_transient(exc) or attempt == attempts - 1:
                _emit_sql_state_error(error_tag, err=repr(exc))
                return
            time.sleep(0.15 + random.random() * 0.45)
        except Exception as exc:
            _emit_sql_state_error(error_tag, err=repr(exc))
            return


def _sql_read_with_retry(
    read_fn, error_tag: str, *, default=None, attempts: int = 5,
):
    """Run a SQL-read closure with transient-error retry and fallback default."""
    for attempt in range(attempts):
        try:
            return read_fn()
        except sqlite3.OperationalError as exc:
            if not _sql_error_is_transient(exc) or attempt == attempts - 1:
                _emit_sql_state_error(error_tag, err=repr(exc))
                return default
            time.sleep(0.15 + random.random() * 0.35)
        except Exception as exc:
            _emit_sql_state_error(error_tag, err=repr(exc))
            return default
    return default


_telemetry_tables_ensured: bool = False
_telemetry_tables_ensured_mtime: float = 0.0
_telemetry_tables_ensured_path: str | None = None
_telemetry_tables_ensured_lock = threading.Lock()


@contextlib.contextmanager
def _ragvec_state_conn():
    """Open a short-lived SQLite connection to telemetry.db."""
    global _telemetry_tables_ensured, _telemetry_tables_ensured_mtime
    global _telemetry_tables_ensured_path
    if _DEPS is None:
        raise RuntimeError("sql state dependencies not configured")
    db_path = _DEPS.db_path()
    db_path.mkdir(parents=True, exist_ok=True)
    telemetry_db_path = str(db_path / _DEPS.telemetry_db_filename())
    conn = sqlite3.connect(
        telemetry_db_path,
        isolation_level=None,
        check_same_thread=False,
        timeout=60.0,
    )
    try:
        conn.execute("PRAGMA busy_timeout=60000")
        cur = conn.execute("PRAGMA journal_mode")
        row = cur.fetchone()
        if (row[0] if row else "").lower() != "wal":
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-65536")
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rag_schema_version ("
            " table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
        )
        try:
            db_mtime = os.path.getmtime(telemetry_db_path)
        except OSError:
            db_mtime = 0.0
        with _telemetry_tables_ensured_lock:
            needs_ensure = (
                not _telemetry_tables_ensured
                or _telemetry_tables_ensured_path != telemetry_db_path
            )
        if needs_ensure:
            _DEPS.ensure_telemetry_tables(conn)
            with _telemetry_tables_ensured_lock:
                _telemetry_tables_ensured = True
                _telemetry_tables_ensured_mtime = db_mtime
                _telemetry_tables_ensured_path = telemetry_db_path
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass
