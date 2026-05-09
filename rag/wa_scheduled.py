"""Compat shim — el módulo se mudó a `rag.integrations.whatsapp.scheduled`
como parte del split modular del 2026-05-08.

Trick: en vez de re-exportar nombres uno a uno, hacemos un **module alias**
en `sys.modules` para que `rag.wa_scheduled` sea LITERALMENTE el mismo
objeto que `rag.integrations.whatsapp.scheduled`. Así:

- `from rag.wa_scheduled import schedule` → schedule del módulo real.
- `from rag import wa_scheduled` → módulo real.
- `monkeypatch.setattr(wa_scheduled, "_log_ambient", mock)` → setea el
  attr en el módulo real, así `run_due_worker` (definido en el real)
  ve el patch en su lookup local. Esto soluciona el split-monkeypatch
  propagation problem que rompía los tests cuando hacíamos un
  re-export por nombre.

El owner real es ``rag/integrations/whatsapp/scheduled.py``; nuevo código
debería importar desde ahí o desde el package re-export
``rag.integrations.whatsapp``.
"""

from __future__ import annotations

import sys

from rag.integrations.whatsapp import scheduled as _real

# Alias en sys.modules — `rag.wa_scheduled is rag.integrations.whatsapp.scheduled`
# después de este punto. Python termina haciendo
# `rag.wa_scheduled = sys.modules['rag.wa_scheduled']` cuando el import
# de este file termina, así el atributo del package `rag` también queda
# apuntando al módulo real.
sys.modules[__name__] = _real
