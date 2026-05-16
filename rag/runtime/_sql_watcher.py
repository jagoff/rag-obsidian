"""SQLite change watcher para event-reactive jobs (F4.1-F4.3).

Implementación: polling con cursor sobre `MAX(rowid)`. SQLite WAL no
soporta inotify portable; el polling pequeño cada 30-60s es la
solución estándar y barata (un single-row SELECT, ~100µs).

Pattern de uso:

```python
from rag.runtime._sql_watcher import SqlChangeWatcher
from rag.runtime.events import bus

watcher = SqlChangeWatcher(
    db_path=Path("~/.local/share/obsidian-rag/ragvec/telemetry.db"),
    table="rag_feedback",
    poll_interval_s=30,
    event_name="sql.feedback.inserted",
)
watcher.start()  # thread daemon
# Subscribers reciben payload {"new_rows": int, "min_rowid": ..., "max_rowid": ...}.
```

Recovery semantics:
- Al startup, `last_seen_rowid = MAX(rowid) ahora` para NO re-disparar
  todos los rows históricos. Solo INSERTs **post-supervisor-start**
  emiten events.
- Si la DB no existe / locked / corrupted, el watcher loggea + sigue
  poleando. NUNCA crashea.
- Si el supervisor muere, los rows insertados durante el downtime se
  pierden — pero el cron fallback (que NO se bootea aún en F4) los
  procesa en su próxima ventana. Defense in depth.

Coexistencia con cron:
- F4.1-F4.3 NO eliminan los cron jobs todavía. Ambos paths corren en
  paralelo. UNIQUE constraints o idempotencia en handlers de-dupean.
- Después de F2.3+F3.5 (bootout), los cron quedan solo en el
  supervisor scheduler (no plists launchd). En esa fase decidimos si
  el cron sigue como backup o se borra completo.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


__all__ = ["SqlChangeWatcher"]


@dataclass
class _WatcherState:
    last_seen_rowid: int = 0
    polls_count: int = 0
    inserts_seen_total: int = 0
    last_poll_ts: float = 0.0
    last_error: str | None = None
    errors_count: int = 0


class SqlChangeWatcher:
    """Polling watcher para una tabla SQLite. Thread-safe, idempotent.

    El watcher mantiene un cursor `last_seen_rowid` y al cada poll:
    1. Abre conexión read-only de corta duración.
    2. ``SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM <table>
        WHERE rowid > last_seen_rowid``.
    3. Si COUNT > 0, emite ``event_name`` con payload + actualiza
       ``last_seen_rowid = MAX``.
    4. Sleep poll_interval_s.

    El emit es vía un callback (típicamente ``bus.publish``). Tests
    pueden inyectar otro callback para verify dispatching.
    """

    def __init__(
        self,
        *,
        db_path: Path,
        table: str,
        event_name: str,
        poll_interval_s: int = 30,
        emit_fn: Callable[[str, dict[str, Any]], int] | None = None,
        anchor_at_start: bool = True,
    ):
        self.db_path = Path(db_path)
        self.table = table
        self.event_name = event_name
        self.poll_interval_s = poll_interval_s
        self._emit_fn = emit_fn
        self._anchor_at_start = anchor_at_start
        self.state = _WatcherState()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _emit(self, payload: dict[str, Any]) -> None:
        if self._emit_fn is not None:
            self._emit_fn(self.event_name, payload)
            return
        # Default: bus.publish del event bus singleton.
        try:
            from rag.runtime.events import bus  # noqa: PLC0415
            bus.publish(self.event_name, payload)
        except Exception:
            logger.exception("sql_watcher: bus.publish raised for %s", self.event_name)

    def _open_conn(self) -> sqlite3.Connection | None:
        try:
            if not self.db_path.exists():
                return None
            uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=2.0)
            conn.execute("PRAGMA busy_timeout=2000")
            return conn
        except Exception as exc:  # noqa: BLE001
            self.state.last_error = f"open_conn: {exc}"
            self.state.errors_count += 1
            return None

    def _max_rowid(self, conn: sqlite3.Connection) -> int | None:
        """Returns the current MAX(rowid) or None on error.

        Callers MUST treat None as "unknown" and NOT update last_seen_rowid —
        returning 0 on error would cause all historical rows to fire on the
        next successful poll (WHERE rowid > 0 matches everything).
        """
        try:
            cur = conn.execute(f"SELECT IFNULL(MAX(rowid), 0) FROM {self.table}")
            row = cur.fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error as exc:
            self.state.last_error = f"max_rowid: {exc}"
            self.state.errors_count += 1
            return None

    def _poll_once(self) -> None:
        conn = self._open_conn()
        if conn is None:
            return
        try:
            self.state.polls_count += 1
            self.state.last_poll_ts = time.time()
            cur = conn.execute(
                f"SELECT IFNULL(MIN(rowid), 0), IFNULL(MAX(rowid), 0), COUNT(*) "
                f"FROM {self.table} WHERE rowid > ?",
                (self.state.last_seen_rowid,),
            )
            row = cur.fetchone()
            if row is None:
                return
            min_rowid, max_rowid, count = int(row[0]), int(row[1]), int(row[2])
            if count == 0:
                return
            self.state.inserts_seen_total += count
            self.state.last_seen_rowid = max_rowid
            self._emit({
                "table": self.table,
                "new_rows": count,
                "min_rowid": min_rowid,
                "max_rowid": max_rowid,
            })
        except sqlite3.Error as exc:
            self.state.last_error = f"poll: {exc}"
            self.state.errors_count += 1
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _loop(self) -> None:
        # Anchor inicial — no re-disparar histórico al startup.
        # Si el anchor falla (DB locked/corrupt), reintentamos hasta 3 veces
        # con 1s de pausa antes de desistir.  Si todos los intentos fallan,
        # last_seen_rowid queda en 0 y el primer poll emitirá todos los rows
        # históricos — peor caso tolerable vs. silenciar events reales.
        if self._anchor_at_start:
            for _attempt in range(3):
                conn = self._open_conn()
                if conn is not None:
                    try:
                        max_rid = self._max_rowid(conn)
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass
                    if max_rid is not None:
                        self.state.last_seen_rowid = max_rid
                        break
                    # _max_rowid returned None (DB error) — retry after brief pause.
                if _attempt < 2:
                    time.sleep(1.0)
            else:
                logger.warning(
                    "sql_watcher: anchor failed for %s after 3 attempts — "
                    "first poll will emit all historical rows",
                    self.table,
                )
        logger.info(
            "sql_watcher: started for %s (anchor rowid=%d, interval=%ds)",
            self.table, self.state.last_seen_rowid, self.poll_interval_s,
        )
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.exception("sql_watcher: unexpected exception in loop")
            # Sleep interruptible.
            self._stop.wait(timeout=self.poll_interval_s)

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"rag-sql-watch-{self.table}",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def trigger_poll_for_test(self) -> None:
        """Dispara un poll sincrónico — solo para tests, evita esperar
        al thread."""
        self._poll_once()
