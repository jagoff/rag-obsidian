"""Supervisor runtime package — single long-running daemon que reemplaza
los 30+ plists launchd individuales con scheduling in-process.

Modules:

- ``scheduler``: APScheduler wrapper con decorators ``@cron``, ``@interval``,
  ``@at``. Cada job decorated se registra automáticamente al
  ``Scheduler.global_instance()`` cuando se importa el módulo job.
- ``ipc``: Unix domain socket server en ``~/.local/share/obsidian-rag/
  supervisor.sock``. Protocolo JSON line-delimited. Handlers register via
  ``@ipc.handler("<route>")``.
- ``events``: pub/sub in-process. ``bus.publish(event_name, payload)`` +
  ``@bus.subscribe(event_name)``. Eventos efímeros — si nadie escucha
  cuando se publica, se pierde.
- ``supervisor``: entrypoint long-running. Levanta scheduler, IPC y carga
  los módulos en ``rag.runtime.jobs.*``.

Diseño general:

- **No imports circulares con ``rag.__init__``**: el runtime es plumbing
  puro. Los jobs importan ``rag`` cuando hace falta, pero el scheduler/
  ipc/events viven aislados.
- **Async-friendly pero no async-only**: scheduler corre handlers en
  thread pool (APScheduler default). IPC server usa ``asyncio`` para
  manejar múltiples conexiones simultáneas. Events son sync (publish
  bloquea hasta que todos los subscribers procesan).
- **Test-friendly**: cada componente tiene un modo "headless" (no socket,
  no thread, schedule manual) para tests unitarios. El supervisor real
  usa los modos "live".

Ver ADR: ``99-obsidian/99-AI/system/daemon-refactor-2026-05-09/
supervisor-refactor-adr.md``.
"""
from __future__ import annotations

__all__ = [
    "events",
    "ipc",
    "scheduler",
]
