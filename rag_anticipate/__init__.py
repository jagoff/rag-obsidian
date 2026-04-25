"""Anticipatory Agent — extension package.

Home de señales NUEVAS (post-MVP 2026-04-24). Las 3 señales originales
(calendar / echo / commitment) viven en `rag.py` bajo `# ── ANTICIPATORY
AGENT ──` por histórico. Cualquier señal ADICIONAL va acá.

## Cómo agregar una señal nueva

1. Crear `rag_anticipate/signals/<kind>.py`:

```python
from datetime import datetime
from rag_anticipate.signals.base import register_signal, SignalContext

@register_signal(name="anniversary", snooze_hours=720)
def anniversary_signal(now: datetime) -> list:
    from rag import AnticipatoryCandidate
    # ... tu lógica ...
    return [AnticipatoryCandidate(
        kind="anticipate-anniversary",
        score=0.7,
        message="...",
        dedup_key="anniv:...",
        snooze_hours=720,
        reason="...",
    )]
```

2. El decorator `@register_signal` auto-registra en `SIGNALS`. El auto-import
   al importar el package (`import rag_anticipate`) descubre todos los
   módulos bajo `signals/` y dispara los decorators.

3. `rag.py` lee `rag_anticipate.SIGNALS` y lo suma al tuple
   `_ANTICIPATE_SIGNALS` que consume `anticipate_run_impl()`.

4. Tests en `tests/test_anticipate_<kind>.py`. Mock todo input externo
   (filesystem, retrieve, calendar, etc.). Ver
   `tests/test_anticipate_agent.py` para el patrón.

## Contract que cada signal DEBE cumplir

- Signature: `fn(now: datetime) -> list[AnticipatoryCandidate]`
- Silent-fail: cualquier excepción interna debe ser capturada. El
  orchestrator tiene un outer try/except como safety net, pero cada signal
  debe degradar grácilmente (retornar `[]`) si su input no está disponible
  (icalBuddy down, vault no accesible, tabla vacía, etc.).
- Determinismo: dado el mismo `now` + mismo estado, debe devolver el mismo
  output. Los mocks en tests deben pasar con `datetime` fijo.
- dedup_key estable: dos runs del mismo momento deben producir la misma
  `dedup_key` para el mismo "item" detectado (event_uid, source_path, loop
  hash, etc.). Sin estabilidad → push indefinido cada 10 min.
- snooze_hours razonable: calendar 2h (evento único del día), resonancias
  ~72h (no repetir la misma observación por 3 días), accountability ~168h
  (1 semana).
- score calibrado: [0, 1]. Compite contra las otras signals por el slot
  top-1 post-threshold (`RAG_ANTICIPATE_MIN_SCORE`, default 0.35).
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings

from rag_anticipate.signals.base import (
    SIGNALS,
    SignalContext,
    register_signal,
)


def _autodiscover_signals() -> None:
    """Importa todos los módulos bajo `rag_anticipate/signals/` para disparar
    los `@register_signal` decorators. Idempotente (importlib cachea).

    Silent-fail per module: si una señal falla al importar (syntax error,
    missing dep, etc.), loguea un warning pero no tumba el agent. Las
    señales OK siguen funcionando.
    """
    from rag_anticipate import signals as _signals_pkg

    for _finder, name, _ispkg in pkgutil.iter_modules(_signals_pkg.__path__):
        if name == "base":
            continue  # no es una signal real, es el framework
        mod_name = f"rag_anticipate.signals.{name}"
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            warnings.warn(
                f"rag_anticipate: failed to load signal {mod_name!r}: {exc!r}",
                stacklevel=2,
            )


_autodiscover_signals()


__all__ = ["SIGNALS", "SignalContext", "register_signal"]
