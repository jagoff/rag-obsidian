"""Signal framework — protocol + decorator + registry.

Diseñado para que 20+ señales puedan coexistir cada una en su archivo sin
tocar estado compartido (excepto la lista `SIGNALS`, que se append-only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # forward-only, evitar import circular


# Lista global de señales registradas. Append-only via `register_signal`.
# El tuple final se construye en rag.py leyendo esta lista.
SIGNALS: list[tuple[str, Callable[[datetime], list]]] = []


class SignalFn(Protocol):
    """Tipo de una signal function."""

    def __call__(self, now: datetime) -> list: ...


@dataclass
class SignalContext:
    """Contexto opcional que las señales pueden construir a demanda.

    Deliberadamente mínimo — la mayoría de las señales no necesitan esto
    porque importan `rag` directamente para acceder a `retrieve`, `get_db`,
    etc. Existe para casos donde una señal quiera recibir un contexto
    pre-construido por el orchestrator (future-proofing).
    """

    now: datetime
    vault_path: object | None = None    # Path, pero evitamos el import en typing
    extras: dict = field(default_factory=dict)


def register_signal(
    *,
    name: str,
    snooze_hours: int = 72,
) -> Callable[[SignalFn], SignalFn]:
    """Decorator — registra una signal function en `SIGNALS`.

    Args:
        name: identificador corto, se usa en logs y `_ANTICIPATE_SIGNALS`
            tuple. El `kind` efectivo del push es `anticipate-<name>`.
        snooze_hours: hint DEFAULT para la señal. La función igual puede
            override en sus AnticipatoryCandidate individuales. Informativo
            mostly — el valor real que se usa es el del candidate.

    El decorator es idempotente por `name`: si una signal con el mismo
    name se registra 2 veces (ej. reload de tests), la segunda reemplaza
    a la primera en lugar de duplicar.
    """

    def _decorator(fn: SignalFn) -> SignalFn:
        # Dedup por name: eliminar entradas previas con el mismo name.
        global SIGNALS
        SIGNALS[:] = [(n, f) for (n, f) in SIGNALS if n != name]
        SIGNALS.append((name, fn))
        # Anotar la función con su metadata para debugging.
        fn.__anticipate_name__ = name  # type: ignore[attr-defined]
        fn.__anticipate_default_snooze__ = snooze_hours  # type: ignore[attr-defined]
        return fn

    return _decorator


def clear_signals_for_test() -> None:
    """Test-only helper — limpia el registry. No usar en runtime."""
    SIGNALS.clear()
