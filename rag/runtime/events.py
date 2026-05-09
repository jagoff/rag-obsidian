"""In-process event bus para handlers reactivos del supervisor.

Patrón pub/sub mínimo. Eventos efímeros — si nadie escucha cuando se
publica, se pierde. Para casos durables (ej. "user pidió un brief, hacelo
en la próxima ventana"), usar IPC + scheduler explícito.

Uso típico:

```python
from rag.runtime.events import bus

@bus.subscribe("vault.note.changed")
def reindex_handler(payload):
    path = payload["path"]
    ...

# En otro lado (ej. file watcher):
bus.publish("vault.note.changed", {"path": "/foo/bar.md"})
```

Threading model:

- ``publish()`` es sync. Bloquea hasta que todos los subscribers
  procesaron. Si un subscriber raisea, se loggea y se sigue con los
  demás.
- ``subscribe()`` se puede llamar en cualquier momento. No hay garantía
  de orden entre subscribers de un mismo evento.
- Para handlers heavy (que no quieren bloquear al publisher), usar
  ``@bus.subscribe(event, async_dispatch=True)`` que mete la invocación
  a un thread pool propio (size 4 default).

Naming convention de eventos: ``<source>.<entity>.<action>`` en lower.
Ejemplos:
- ``vault.note.changed``
- ``sql.feedback.inserted``
- ``sql.eval_run.completed``
- ``system.wake.detected``
- ``wa.message.inbound``
- ``spotify.track.changed``
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "EventBus",
    "Subscription",
    "bus",
]


@dataclass
class Subscription:
    """Handle retornado por ``subscribe()``. Permite ``unsubscribe()``."""
    event: str
    handler: Callable[[dict[str, Any]], Any]
    async_dispatch: bool = False


class EventBus:
    """Pub/sub in-process thread-safe.

    Para tests usar instancia local en vez del singleton:

    ```python
    test_bus = EventBus()
    test_bus.subscribe("foo")(handler)
    test_bus.publish("foo", {"x": 1})
    ```
    """

    def __init__(self, *, async_pool_size: int = 4):
        self._subs: dict[str, list[Subscription]] = defaultdict(list)
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._async_pool_size = async_pool_size

    def _get_executor(self) -> ThreadPoolExecutor:
        # Lazy init — solo si hay async subscribers.
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=self._async_pool_size,
                        thread_name_prefix="rag-events-",
                    )
        return self._executor

    def subscribe(
        self,
        event: str,
        *,
        async_dispatch: bool = False,
    ) -> Callable[[Callable[[dict[str, Any]], Any]],
                   Callable[[dict[str, Any]], Any]]:
        """Decorator: ``@bus.subscribe("event.name")``.

        El handler recibe el payload (dict) que el publisher pasó.

        Si ``async_dispatch=True``, el handler corre en thread pool —
        ``publish()`` no espera. Útil para handlers heavy (LLM calls,
        sqlite writes que pueden tardar).
        """
        def _decorator(fn: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
            sub = Subscription(event=event, handler=fn, async_dispatch=async_dispatch)
            with self._lock:
                self._subs[event].append(sub)
            return fn
        return _decorator

    def publish(self, event: str, payload: dict[str, Any] | None = None) -> int:
        """Publica ``event`` con ``payload``. Retorna cantidad de subscribers
        que recibieron el evento. Si ningún sub registrado, retorna 0
        (no error — evento se pierde).

        Sync subscribers se invocan en orden de registro. Async se
        despachan a thread pool y publish retorna sin esperarlos.
        """
        payload = payload or {}
        with self._lock:
            subs = list(self._subs.get(event, []))
        if not subs:
            return 0
        for sub in subs:
            if sub.async_dispatch:
                executor = self._get_executor()
                executor.submit(self._invoke_safe, sub, payload)
            else:
                self._invoke_safe(sub, payload)
        return len(subs)

    def _invoke_safe(self, sub: Subscription, payload: dict[str, Any]) -> None:
        try:
            sub.handler(payload)
        except Exception:
            logger.exception(
                "events: handler for %s raised; continuing", sub.event,
            )

    def unsubscribe_all(self, event: str | None = None) -> int:
        """Quita todos los subs de ``event`` (o todos si ``event=None``).
        Retorna cantidad removida. Solo para tests.
        """
        with self._lock:
            if event is None:
                count = sum(len(v) for v in self._subs.values())
                self._subs.clear()
                return count
            count = len(self._subs.get(event, []))
            self._subs.pop(event, None)
            return count

    def shutdown(self) -> None:
        """Cierra el thread pool si fue creado."""
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None


# Singleton bus del proceso supervisor.
bus = EventBus()
