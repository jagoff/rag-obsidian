"""Smoke-test signal — prueba que el framework de auto-discovery y registro
funciona. Solo se activa si RAG_ANTICIPATE_SMOKE=1 está seteado.

Este archivo se puede borrar después de validar que señales reales cargan
via `@register_signal`.
"""

from __future__ import annotations

import os
from datetime import datetime

from rag_anticipate.signals.base import register_signal


@register_signal(name="_smoke", snooze_hours=9999)
def smoke_signal(now: datetime) -> list:
    """Emite un candidate de score 0.0 sólo si la env var está. Silencioso
    en uso normal."""
    if os.environ.get("RAG_ANTICIPATE_SMOKE", "").strip() not in ("1", "true"):
        return []
    # Emit un candidate trivial
    from rag import AnticipatoryCandidate
    return [AnticipatoryCandidate(
        kind="anticipate-_smoke",
        score=0.01,
        message="[smoke test]",
        dedup_key="smoke:always",
        snooze_hours=9999,
        reason="framework smoke test",
    )]
