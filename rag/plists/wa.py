"""WhatsApp time-sensitive worker: wa-fast.

Worker 5min unificado que corre `remind-wa` + `wa-scheduled-send` en
serie (consolidación 2026-05-04). NO incluye `wa-tasks` — ese vive
en `rag.integrations.whatsapp.plist` con cadencia 30min separada.

Migrado de rag/plists/_legacy.py en Phase 3 commit 3 (2026-05-09).
"""
from __future__ import annotations

from rag.plists._render import _logs, _render_plist

__all__ = [
    "_wa_fast_plist",
]


def _wa_fast_plist(rag_bin: str) -> str:
    """Worker unificado WhatsApp time-sensitive — every 5 minutes.

    Consolidación 2026-05-04: antes eran 2 plists separados con cadencia
    idéntica (`reminder-wa-push` + `wa-scheduled-send`, ambos 5 min).
    Se unificaron en un solo worker (`rag wa-fast`) que corre los 2
    sub-jobs en serie. Ahorra 1 cold-start (~3-4s de `import rag`) cada
    5 min = ~10+ min/día de CPU evitada. Ambos jobs son idempotentes
    (tablas `rag_reminder_wa_pushed` y `rag_whatsapp_scheduled`
    respectivamente), así que si un run se salta por Mac dormida /
    launchd backoff, el siguiente recupera los pendings.

    Sub-jobs:
      1. `remind-wa`         — Apple Reminders próximos a vencer → WA
      2. `wa-scheduled-send` — mensajes programados del user que vencieron

    NO incluye `wa-tasks` (cadencia 30min, LLM-heavy por chat). Fusionar
    ese acá saturaría Ollama.

    Silent-fail end-to-end: cada sub-job corre en try/except — si uno
    crashea, el otro corre igual. El worker siempre exit 0.
    """
    out, err = _logs("wa-fast")
    return _render_plist({
        "label": "com.fer.obsidian-rag-wa-fast",
        "program_arguments": [rag_bin, "wa-fast"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "schedule": {"interval_s": 300},
        "run_at_load": False,
        "throttle_s": 30,
        "process_type": "Background",
        "stdout_path": out,
        "stderr_path": err,
    })
