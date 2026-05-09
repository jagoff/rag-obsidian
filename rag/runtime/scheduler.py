"""APScheduler wrapper con decorators ``@cron``, ``@interval``, ``@at``.

Wraps APScheduler ``BackgroundScheduler`` con una API declarativa más
acotada al uso del proyecto:

- ``@cron(hour=3, minute=0)`` — equivalente a ``StartCalendarInterval``
  ``Hour=3`` ``Minute=0``.
- ``@interval(seconds=900)`` — equivalente a ``StartInterval=900``.
- ``@at(weekday=0, hour=22, minute=0)`` — calendar interval por día de
  semana (lunes=0, mismo formato launchd offset por +1 que mapeamos acá).
- ``@on_signal("system.wake.detected")`` — hookea al event bus.

Cada decorator registra el handler en ``Scheduler.global_instance()`` que
es la única instancia del proceso supervisor. Para tests, ``Scheduler()``
se puede instanciar manualmente sin tocar la global.

Job registration es eager pero start es lazy: importar un módulo de
``rag.runtime.jobs.*`` registra los jobs, el scheduler arranca recién en
``Scheduler.start()`` (llamado desde ``supervisor.py``).

Telemetría: cada ejecución envía un row a ``rag_supervisor_jobs`` (ver
F0.5). Wrapping transparente — el handler decorated no se entera.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# APScheduler import lazy — para que ``import rag.runtime.scheduler`` en
# tests no crashee si el venv no tiene apscheduler instalado (gateado).
try:
    from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore
    from apscheduler.triggers.cron import CronTrigger  # type: ignore
    from apscheduler.triggers.interval import IntervalTrigger  # type: ignore
    _APSCHEDULER_AVAILABLE = True
except ImportError:  # pragma: no cover — apscheduler always pinned in pyproject post-F1
    BackgroundScheduler = None  # type: ignore
    CronTrigger = None  # type: ignore
    IntervalTrigger = None  # type: ignore
    _APSCHEDULER_AVAILABLE = False


__all__ = [
    "Job",
    "Scheduler",
    "at",
    "cron",
    "interval",
]


@dataclass
class Job:
    """Representación de un job registrado.

    Atributos persistidos en ``rag_supervisor_jobs`` cada vez que corre.
    """
    label: str
    handler: Callable[[], Any]
    trigger_kind: str  # "cron" | "interval" | "calendar"
    trigger_args: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    last_run_ts: float | None = None
    last_exit_code: int | None = None
    last_duration_s: float | None = None
    last_error: str | None = None
    runs_count: int = 0
    fails_count: int = 0


class Scheduler:
    """In-process job scheduler para el supervisor.

    Uso típico (singleton via ``Scheduler.global_instance()``):

    ```python
    from rag.runtime.scheduler import cron, interval, Scheduler

    @cron(hour=3, minute=0, label="auto_harvest")
    def my_nightly_job():
        ...

    @interval(seconds=900, label="anticipate")
    def my_proactive_job():
        ...

    # En supervisor.py:
    Scheduler.global_instance().start()
    ```

    Para tests:

    ```python
    sched = Scheduler(headless=True)  # no APScheduler real
    sched.register_job(...)
    sched.run_now("auto_harvest")  # dispatch inmediato sync
    ```
    """

    _global: "Scheduler | None" = None
    _global_lock = threading.Lock()

    def __init__(self, *, headless: bool = False):
        self.headless = headless or not _APSCHEDULER_AVAILABLE
        self._jobs: dict[str, Job] = {}
        self._aps: BackgroundScheduler | None = None
        self._started = False
        if not self.headless:
            self._aps = BackgroundScheduler(
                timezone=os.environ.get("RAG_TIMEZONE", "America/Argentina/Buenos_Aires"),
                job_defaults={
                    "coalesce": True,        # juntar misfires en uno
                    "max_instances": 1,      # no overlapping per job
                    "misfire_grace_time": 60,
                },
            )

    @classmethod
    def global_instance(cls) -> "Scheduler":
        """Singleton lazy. Thread-safe."""
        if cls._global is None:
            with cls._global_lock:
                if cls._global is None:
                    cls._global = Scheduler()
        return cls._global

    @classmethod
    def reset_global(cls) -> None:
        """Solo para tests — limpia el singleton entre runs."""
        with cls._global_lock:
            if cls._global is not None and cls._global._aps is not None:
                try:
                    cls._global._aps.shutdown(wait=False)
                except Exception:
                    pass
            cls._global = None

    def register_job(self, job: Job) -> None:
        """Registra un job. Idempotente: re-register sobreescribe."""
        if job.label in self._jobs:
            logger.info("scheduler: re-registering %s (overwrite)", job.label)
        self._jobs[job.label] = job

    def jobs(self) -> dict[str, Job]:
        """Snapshot de los jobs registrados."""
        return dict(self._jobs)

    def get_job(self, label: str) -> Job | None:
        return self._jobs.get(label)

    def start(self) -> None:
        """Arranca el scheduler real (no-op en headless)."""
        if self._started:
            return
        if self.headless:
            self._started = True
            return
        if self._aps is None:
            raise RuntimeError("APScheduler no disponible")
        for job in self._jobs.values():
            trigger = self._build_trigger(job)
            self._aps.add_job(
                self._make_runner(job),
                trigger=trigger,
                id=job.label,
                replace_existing=True,
            )
        self._aps.start()
        self._started = True
        logger.info("scheduler: started, %d jobs", len(self._jobs))

    def shutdown(self, *, wait: bool = True) -> None:
        if not self._started:
            return
        if self._aps is not None:
            try:
                self._aps.shutdown(wait=wait)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("scheduler shutdown raised: %s", exc)
        self._started = False

    def run_now(self, label: str) -> dict[str, Any]:
        """Dispara el handler de ``label`` sincrónicamente. Útil para
        tests, IPC ``/run/<job>`` y CLI ``rag supervisor trigger``.

        Retorna ``{"ok": bool, "duration_s": float, "error": str | None,
        "result": Any}``.
        """
        job = self._jobs.get(label)
        if job is None:
            return {"ok": False, "error": f"unknown job: {label}", "result": None}
        return self._make_runner(job)()

    def _build_trigger(self, job: Job):
        if job.trigger_kind == "cron":
            return CronTrigger(**job.trigger_args)
        if job.trigger_kind == "interval":
            return IntervalTrigger(**job.trigger_args)
        if job.trigger_kind == "calendar":
            # calendar = cron con day_of_week + hour + minute
            return CronTrigger(**job.trigger_args)
        raise ValueError(f"trigger_kind desconocido: {job.trigger_kind}")

    def _make_runner(self, job: Job) -> Callable[[], dict[str, Any]]:
        """Wrap el handler con telemetría + error handling."""
        def _run() -> dict[str, Any]:
            t0 = time.time()
            error: str | None = None
            result: Any = None
            try:
                result = job.handler()
                exit_code = 0
            except Exception as exc:
                logger.exception("scheduler: job %s failed", job.label)
                error = str(exc)
                exit_code = 1
            duration = time.time() - t0
            job.last_run_ts = t0
            job.last_exit_code = exit_code
            job.last_duration_s = duration
            job.last_error = error
            job.runs_count += 1
            if exit_code != 0:
                job.fails_count += 1
            try:
                _persist_run(job, t0, duration, exit_code, error, result)
            except Exception as persist_exc:  # noqa: BLE001 — telemetría nunca debe romper job
                logger.warning("scheduler: persist failed for %s: %s",
                               job.label, persist_exc)
            return {
                "ok": exit_code == 0,
                "duration_s": duration,
                "error": error,
                "result": result,
            }
        return _run


# ── Decorators ────────────────────────────────────────────────────────────


def cron(
    *,
    hour: int | str = 0,
    minute: int | str = 0,
    day_of_week: int | str | None = None,
    day: int | str | None = None,
    label: str | None = None,
    description: str = "",
) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    """Schedule cron-style. Wrappea el handler y registra en el scheduler
    global. Equivalente a ``StartCalendarInterval`` de launchd.

    Ejemplos:
        @cron(hour=3, minute=0, label="auto_harvest")
        @cron(hour=22, minute=0, day_of_week=0, label="digest")  # domingo 22:00
    """
    def _decorator(fn: Callable[[], Any]) -> Callable[[], Any]:
        actual_label = label or fn.__name__
        args: dict[str, Any] = {"hour": hour, "minute": minute}
        if day_of_week is not None:
            args["day_of_week"] = day_of_week
        if day is not None:
            args["day"] = day
        Scheduler.global_instance().register_job(Job(
            label=actual_label,
            handler=fn,
            trigger_kind="cron",
            trigger_args=args,
            description=description or fn.__doc__ or "",
        ))
        return fn
    return _decorator


def interval(
    *,
    seconds: int | None = None,
    minutes: int | None = None,
    hours: int | None = None,
    label: str | None = None,
    description: str = "",
) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    """Schedule interval-style. Equivalente a ``StartInterval`` de launchd.

    Pasar **una sola** unidad (seconds/minutes/hours).
    """
    def _decorator(fn: Callable[[], Any]) -> Callable[[], Any]:
        actual_label = label or fn.__name__
        args: dict[str, Any] = {}
        if seconds is not None:
            args["seconds"] = seconds
        if minutes is not None:
            args["minutes"] = minutes
        if hours is not None:
            args["hours"] = hours
        if not args:
            raise ValueError("interval: especificar seconds/minutes/hours")
        Scheduler.global_instance().register_job(Job(
            label=actual_label,
            handler=fn,
            trigger_kind="interval",
            trigger_args=args,
            description=description or fn.__doc__ or "",
        ))
        return fn
    return _decorator


def at(
    *,
    weekday: int,
    hour: int,
    minute: int = 0,
    label: str | None = None,
    description: str = "",
) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    """Schedule weekday-specific (Mon=0 .. Sun=6). Mapea a APScheduler
    ``day_of_week`` que usa el mismo offset (Mon=0).

    Ejemplos:
        @at(weekday=0, hour=22, minute=0, label="digest")  # lunes 22:00
    """
    return cron(
        day_of_week=weekday,
        hour=hour,
        minute=minute,
        label=label,
        description=description,
    )


# ── Telemetría persistence ────────────────────────────────────────────────


def _persist_run(
    job: Job,
    ts_start: float,
    duration_s: float,
    exit_code: int,
    error: str | None,
    result: Any,
) -> None:
    """Persiste el run a ``rag_supervisor_jobs`` (telemetry.db).

    Llamado dentro del runner wrapping. Falla silenciosa — telemetry no
    debe ser blocker del job. Schema definido en F0.5 (rag/runtime/
    _telemetry.py).
    """
    # Late import — evita ciclo + permite que el módulo sea importable
    # standalone para tests sin telemetry setup.
    try:
        from rag.runtime._telemetry import insert_supervisor_job_run
    except ImportError:
        return
    insert_supervisor_job_run(
        label=job.label,
        ts_start=ts_start,
        duration_s=duration_s,
        exit_code=exit_code,
        error=error,
        result=result,
    )
