"""Entrypoint long-running del supervisor.

Levanta:
1. Logging estructurado a ``~/.local/share/obsidian-rag/supervisor.log``.
2. Imports de ``rag.runtime.jobs.*`` — cada módulo registra sus jobs vía
   decorators (auto-registration).
3. IPC handlers built-in (``status``, ``ping``, ``run``, ``jobs``).
4. APScheduler arranca con todos los jobs registrados.
5. IPC server thread daemon.
6. Espera SIGTERM/SIGINT → graceful shutdown.

Invocado desde:
- ``rag supervisor run`` (CLI)
- ``com.fer.obsidian-rag-supervisor`` plist (launchd KeepAlive=true)

Crash semantics:
- Errores en un job NO matan el supervisor — el scheduler los captura.
- Errores en IPC handler NO matan el supervisor — captura per-handler.
- Errores en startup (import jobs/registrar IPC) → exit 1, launchd
  KeepAlive lo restartea.
- SIGTERM → 20s graceful (matchea ``ExitTimeOut`` del plist).
"""
from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "Supervisor",
    "main",
]

_LOG_DIR = Path.home() / ".local/share/obsidian-rag"
_LOG_FILE = _LOG_DIR / "supervisor.log"


def _setup_logging() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    # Silenciar APScheduler verbose (cada misfire loggeado por default).
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def _import_jobs() -> int:
    """Import all submodules of ``rag.runtime.jobs`` so their decorators
    register handlers in the global Scheduler.

    Retorna count de jobs registrados después del import. Errores de
    import per-job se loggean pero no escalan (si el módulo X falla
    importar, los demás siguen).
    """
    from rag.runtime.scheduler import Scheduler  # noqa: PLC0415
    n_before = len(Scheduler.global_instance().jobs())

    try:
        import rag.runtime.jobs as jobs_pkg  # noqa: PLC0415, F401
    except ImportError as exc:
        logger.error("supervisor: jobs package not importable: %s", exc)
        return 0

    import importlib  # noqa: PLC0415
    import pkgutil  # noqa: PLC0415

    if not hasattr(jobs_pkg, "__path__"):
        return 0
    for finder, name, ispkg in pkgutil.iter_modules(jobs_pkg.__path__):
        full = f"rag.runtime.jobs.{name}"
        try:
            importlib.import_module(full)
        except Exception as exc:  # noqa: BLE001
            logger.exception("supervisor: failed to import %s: %s", full, exc)

    n_after = len(Scheduler.global_instance().jobs())
    return n_after - n_before


def _register_builtin_ipc() -> None:
    from rag.runtime import ipc  # noqa: PLC0415
    from rag.runtime.scheduler import Scheduler  # noqa: PLC0415

    @ipc.handler("ping")
    def _ping(_payload: dict[str, Any]) -> dict[str, Any]:
        return {"pong": True, "ts": time.time()}

    @ipc.handler("status")
    def _status(_payload: dict[str, Any]) -> dict[str, Any]:
        sched = Scheduler.global_instance()
        jobs_info = []
        for label, job in sched.jobs().items():
            jobs_info.append({
                "label": label,
                "trigger_kind": job.trigger_kind,
                "trigger_args": job.trigger_args,
                "runs_count": job.runs_count,
                "fails_count": job.fails_count,
                "last_run_ts": job.last_run_ts,
                "last_exit_code": job.last_exit_code,
                "last_duration_s": job.last_duration_s,
                "last_error": job.last_error,
            })
        return {
            "n_jobs": len(jobs_info),
            "jobs": jobs_info,
            "uptime_s": time.time() - _STARTED_AT,
        }

    @ipc.handler("jobs")
    def _jobs(_payload: dict[str, Any]) -> dict[str, Any]:
        sched = Scheduler.global_instance()
        return {"labels": sorted(sched.jobs().keys())}

    @ipc.handler("run")
    def _run(payload: dict[str, Any]) -> dict[str, Any]:
        label = payload.get("job")
        if not label:
            return {"ok": False, "error": "missing 'job' field"}
        sched = Scheduler.global_instance()
        result = sched.run_now(label, trigger="ipc")
        return result


_STARTED_AT = time.time()
_SHUTDOWN_REQUESTED = threading.Event()


# ── F3.4 — daemon-watchdog + wake-hook reemplazados por supervisor internals ─
#
# Los plists viejos:
# - ``com.fer.obsidian-rag-daemon-watchdog`` (5min reconcile + retry de
#   exit≠0) — su rol era retry-ear plists que crashearan. APScheduler
#   maneja retry vía ``misfire_grace_time=60`` + ``max_instances=1``
#   built-in. Si un job raisea, scheduler lo loggea y la próxima
#   ventana sigue normal. Si excede ``misfire_grace_time``, reintenta.
#
# - ``com.fer.obsidian-rag-wake-hook`` (KeepAlive pmset poller →
#   kickstart-overdue) — su rol era catchup post-Mac-wake porque
#   launchd ``StartCalendarInterval`` NO dispara retroactivamente.
#   APScheduler usa ``coalesce=True`` (ya seteado en ``Scheduler.__init__``)
#   que junta todos los misfires en 1 sola corrida cuando el supervisor
#   recobra control post-wake. Cubre el caso sin necesidad de
#   pmset listener propio.
#
# F3.5 hará bootout de ambos plists. Hasta entonces siguen vivos y el
# supervisor convive con ellos sin conflict.


def _trigger_mlx_warmup() -> None:
    """Pre-load MLX models al startup del supervisor (F2.2).

    Reusa ``rag.warmup_async()`` — la misma función que el CLI usa cuando
    ``RAG_NO_WARMUP=0``. Carga 5 targets en threads paralelos (~5s pico):
    reranker MPS, in-process embedder, corpus BM25 + pagerank, chat
    qwen2.5:7b warmup, helper qwen2.5:3b.

    Beneficio post-F2: cualquier job in-process (drift_watcher hoy, los
    F3 jobs después) que necesite LLM/embed/rerank lo tiene caché en
    process. Hot-path queries del web NO se benefician (proceso aparte).

    Opt-out: ``RAG_SUPERVISOR_MLX_WARMUP=0`` en el plist para arranque
    rápido sin warmup (útil si el supervisor solo corre subprocess
    jobs, F2.1 actual). Default ON desde F2.2.
    """
    import os  # noqa: PLC0415

    if os.environ.get("RAG_SUPERVISOR_MLX_WARMUP", "1") != "1":
        logger.info("supervisor: MLX warmup OFF (RAG_SUPERVISOR_MLX_WARMUP=0)")
        return

    try:
        from rag import warmup_async  # noqa: PLC0415
    except ImportError as exc:
        logger.warning("supervisor: warmup_async no importable: %s", exc)
        return

    try:
        warmup_async()
        logger.info("supervisor: MLX warmup dispatched (paralelo, ~5s pico)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("supervisor: warmup_async raised: %s", exc)


class Supervisor:
    """Orquestador principal. Tests pueden instanciar y controlar el
    lifecycle manualmente."""

    def __init__(self):
        self._ipc_server = None
        self._ipc_thread: threading.Thread | None = None

    def start(self) -> int:
        """Arranca scheduler + IPC. Retorna count de jobs registrados.

        No bloquea — el caller (típicamente ``main()``) hace el wait.
        """
        from rag.runtime import ipc  # noqa: PLC0415
        from rag.runtime._telemetry import ensure_table  # noqa: PLC0415
        from rag.runtime.scheduler import Scheduler  # noqa: PLC0415

        ensure_table()
        n_jobs = _import_jobs()
        _register_builtin_ipc()

        self._ipc_server = ipc.IPCServer()
        self._ipc_thread = threading.Thread(
            target=self._ipc_server.serve_forever,
            name="rag-ipc",
            daemon=True,
        )
        self._ipc_thread.start()

        Scheduler.global_instance().start()

        # MLX warmup async (F2.2) — paraleliza con el resto del startup
        # para no bloquear. Si fallan los modelos, el supervisor sigue
        # funcionando para subprocess jobs (F2.1 path).
        _trigger_mlx_warmup()

        logger.info(
            "supervisor: started — %d jobs, ipc=%s",
            n_jobs,
            ipc.DEFAULT_SOCKET_PATH,
        )
        return n_jobs

    def shutdown(self, *, timeout: float = 20.0) -> None:
        from rag.runtime.scheduler import Scheduler  # noqa: PLC0415

        logger.info("supervisor: shutdown requested")
        try:
            Scheduler.global_instance().shutdown(wait=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("supervisor: scheduler shutdown raised: %s", exc)
        if self._ipc_server is not None:
            try:
                self._ipc_server.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("supervisor: ipc shutdown raised: %s", exc)
        if self._ipc_thread is not None:
            self._ipc_thread.join(timeout=timeout)
        logger.info("supervisor: stopped cleanly")


def _install_signal_handlers(supervisor: Supervisor) -> None:
    def _handle(signum: int, _frame) -> None:
        logger.info("supervisor: received signal %d", signum)
        _SHUTDOWN_REQUESTED.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def main() -> int:
    """Entrypoint para ``rag supervisor run`` y para el plist launchd."""
    _setup_logging()
    sup = Supervisor()
    _install_signal_handlers(sup)
    try:
        sup.start()
    except Exception:
        logger.exception("supervisor: fatal startup error")
        return 1

    # Bloquea hasta SIGTERM. Tick periódico para que el thread principal
    # pueda manejar señales correctamente en macOS.
    while not _SHUTDOWN_REQUESTED.is_set():
        _SHUTDOWN_REQUESTED.wait(timeout=1.0)

    sup.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
