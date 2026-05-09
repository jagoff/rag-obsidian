"""Compat shim — el comando `wa-tasks` se mudó a
`rag.integrations.whatsapp.cli` como parte del split modular del 2026-05-08.

Mismo trick que `rag.wa_scheduled`: alias del módulo en `sys.modules` para
que `rag.wa_tasks` sea LITERALMENTE el mismo objeto que
`rag.integrations.whatsapp.cli`. Tests con `monkeypatch.setattr(wa_tasks, ...)`
patchearían el módulo real (no usado actualmente, pero consistencia).
"""

from __future__ import annotations

import sys

from rag.integrations.whatsapp import cli as _real

sys.modules[__name__] = _real
