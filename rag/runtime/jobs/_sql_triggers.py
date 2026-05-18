"""F4.1+F4.2+F4.3 — SQL triggers + bridge hook (event-reactive).

Tres watchers que polean tablas SQLite y emiten events cuando hay nuevos
INSERTs. Los handlers reaccionan in-process disparando los jobs
correspondientes — reemplaza el polling cron por captura inmediata
post-write.

| Watcher                | DB                          | Tabla            | Event                   |
|------------------------|-----------------------------|------------------|-------------------------|
| F4.1 routing-rules     | telemetry.db                | rag_feedback     | sql.feedback.inserted   |
| F4.2 drift-watcher     | telemetry.db                | rag_eval_runs    | sql.eval_run.completed  |
| F4.3 wa-tasks          | bridge messages.db (WA)     | messages         | wa.message.inbound      |

Coexistencia con cron:
- En F4.1+F4.2+F4.3, los cron jobs (`routing_rules`, `drift_watcher`,
  `wa_tasks` en jobs/frequent.py + jobs/drift_watcher.py) SIGUEN
  registrados. Eso es intencional como **fallback** — si el watcher
  thread crashea o el supervisor restartea, el cron asegura ejecución.
- Idempotencia de los handlers + UNIQUE constraints en las DB destino
  hacen que el solapamiento sea seguro.
- F4-followup post-A/B: si los watchers son confiables 1+ semana,
  podemos bajar la cadencia del cron de 5min/6h a 1h/24h como safety
  net pasivo.

Auto-start al import. Threading: cada watcher tiene su propio thread
daemon. Stops cuando supervisor recibe SIGTERM (daemons mueren con el
proceso).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
from typing import Any

from rag.runtime._sql_watcher import SqlChangeWatcher
from rag.runtime.events import bus

logger = logging.getLogger(__name__)


_TELEMETRY_DB = (
    Path(os.environ.get("OBSIDIAN_RAG_DB_PATH")
         or str(Path.home() / ".local/share/obsidian-rag/ragvec"))
    / "telemetry.db"
)
_BRIDGE_DB = Path.home() / "repos/whatsapp-mcp/whatsapp-bridge/store/messages.db"


# ── F4.1 — routing-rules trigger ────────────────────────────────────────────


_ROUTING_WATCHER = SqlChangeWatcher(
    db_path=_TELEMETRY_DB,
    table="rag_feedback",
    event_name="sql.feedback.inserted",
    poll_interval_s=30,
)


@bus.subscribe("sql.feedback.inserted", async_dispatch=True)
def _on_feedback_inserted(payload: dict[str, Any]) -> None:
    """Handler F4.1: dispara `rag routing extract-rules --auto-promote`
    en respuesta a INSERTs en rag_feedback. Async dispatch para no
    bloquear al watcher (extract-rules puede tomar varios segundos).
    """
    from rag.runtime.jobs.frequent import routing_rules_job  # noqa: PLC0415

    new_rows = payload.get("new_rows", 0)
    if new_rows < 5:
        # Threshold mínimo: rules necesita 5+ samples del mismo patrón
        # antes de promote. Bypassamos llamadas con baja yield.
        logger.info(
            "f4.1: skipping routing-rules trigger (only %d new feedback rows)",
            new_rows,
        )
        return
    logger.info(
        "f4.1: feedback insert detected (%d new rows), triggering routing-rules",
        new_rows,
    )
    try:
        result = routing_rules_job()
        logger.info("f4.1: routing-rules done — exit_code=%s",
                    result.get("exit_code"))
    except Exception:
        logger.exception("f4.1: routing-rules handler crashed")


# ── F4.2 — drift-watcher trigger ────────────────────────────────────────────


_DRIFT_WATCHER = SqlChangeWatcher(
    db_path=_TELEMETRY_DB,
    table="rag_eval_runs",
    event_name="sql.eval_run.completed",
    poll_interval_s=60,  # eval runs son raros (nightly), polling más laxo
)


@bus.subscribe("sql.eval_run.completed", async_dispatch=True)
def _on_eval_run_completed(payload: dict[str, Any]) -> None:
    """Handler F4.2: dispara drift_watcher_job al detectar nuevo
    rag_eval_runs row.

    Latencia: nuevo eval run → drift check en <60s (vs hasta 6h con
    cron). Importante porque el bot WA push de drift alert quiere ser
    inmediato post-regression, no esperar al próximo tick cron.
    """
    from rag.runtime.jobs.drift_watcher import drift_watcher_job  # noqa: PLC0415

    new_rows = payload.get("new_rows", 0)
    logger.info(
        "f4.2: eval_run insert detected (%d new), triggering drift check",
        new_rows,
    )
    try:
        result = drift_watcher_job()
        alerts = result.get("alerts", 0)
        if alerts > 0:
            logger.warning(
                "f4.2: drift detected — %d alerts (kinds=%s)",
                alerts, result.get("kinds", []),
            )
        else:
            logger.info("f4.2: no drift")
    except Exception:
        logger.exception("f4.2: drift-watcher handler crashed")


# ── F4.3 — wa-tasks bridge hook ─────────────────────────────────────────────


_WA_WATCHER = SqlChangeWatcher(
    db_path=_BRIDGE_DB,
    table="messages",
    event_name="wa.message.inbound",
    poll_interval_s=30,
)
_WA_TASKS_TRIGGER_LOCK = threading.Lock()


@bus.subscribe("wa.message.inbound", async_dispatch=True)
def _on_wa_message_inbound(payload: dict[str, Any]) -> None:
    """Handler F4.3: dispara wa_tasks_job al detectar nuevos mensajes
    inbound en el bridge SQLite.

    NO necesariamente hay action items en cada mensaje — el job interno
    cap-ea a 12 chats con LLM call. Pero post-event garantiza que action
    items ≤30s post-mensaje en lugar de esperar hasta 30min con cron.

    Threshold: 3+ mensajes nuevos. Mensaje suelto raramente vale el
    LLM call (cap diario daily_cap se gasta rápido).
    """
    from rag.runtime.jobs.frequent import wa_tasks_job  # noqa: PLC0415

    new_rows = payload.get("new_rows", 0)
    if new_rows < 3:
        logger.info(
            "f4.3: skipping wa-tasks (only %d new messages — threshold 3)",
            new_rows,
        )
        return
    logger.info(
        "f4.3: %d wa messages detected, triggering wa-tasks extractor",
        new_rows,
    )
    if not _WA_TASKS_TRIGGER_LOCK.acquire(blocking=False):
        logger.info(
            "f4.3: skipping wa-tasks trigger — previous extractor still running"
        )
        return
    try:
        result = wa_tasks_job()
        logger.info("f4.3: wa-tasks done — exit_code=%s",
                    result.get("exit_code"))
    except Exception:
        logger.exception("f4.3: wa-tasks handler crashed")
    finally:
        _WA_TASKS_TRIGGER_LOCK.release()


# ── Lifecycle ───────────────────────────────────────────────────────────────


def start_all_watchers() -> dict[str, bool]:
    """Arranca los 3 watchers. Idempotent. Retorna dict por watcher.

    Opt-out global: ``RAG_SQL_WATCHERS_DISABLED=1``. Para granular
    individuales:
    - ``RAG_F41_DISABLED=1`` — routing-rules trigger
    - ``RAG_F42_DISABLED=1`` — drift-watcher trigger
    - ``RAG_F43_DISABLED=1`` — wa-tasks bridge hook
    """
    if os.environ.get("RAG_SQL_WATCHERS_DISABLED") == "1":
        logger.info("sql watchers: disabled via env var")
        return {"routing": False, "drift": False, "wa": False}

    return {
        "routing": (
            False if os.environ.get("RAG_F41_DISABLED") == "1"
            else _ROUTING_WATCHER.start()
        ),
        "drift": (
            False if os.environ.get("RAG_F42_DISABLED") == "1"
            else _DRIFT_WATCHER.start()
        ),
        "wa": (
            False if os.environ.get("RAG_F43_DISABLED") == "1"
            else _WA_WATCHER.start()
        ),
    }


def watchers_status() -> dict[str, Any]:
    """Stats del trio de watchers — IPC handler-friendly."""
    return {
        "routing": {
            "thread_alive": (
                _ROUTING_WATCHER._thread is not None
                and _ROUTING_WATCHER._thread.is_alive()
            ),
            **vars(_ROUTING_WATCHER.state),
        },
        "drift": {
            "thread_alive": (
                _DRIFT_WATCHER._thread is not None
                and _DRIFT_WATCHER._thread.is_alive()
            ),
            **vars(_DRIFT_WATCHER.state),
        },
        "wa": {
            "thread_alive": (
                _WA_WATCHER._thread is not None
                and _WA_WATCHER._thread.is_alive()
            ),
            **vars(_WA_WATCHER.state),
        },
    }


# IPC handler para inspección.
from rag.runtime import ipc  # noqa: E402


@ipc.handler("status_sql_watchers")
def status_sql_watchers_handler(_payload: dict[str, Any]) -> dict[str, Any]:
    return watchers_status()


# Auto-start al import (cuando supervisor descubre el módulo).
start_all_watchers()
