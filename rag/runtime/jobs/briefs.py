"""Brief output jobs migrados al supervisor (F3.3).

Re-implementación in-supervisor de los 3 daemons launchd de briefs:

| Schedule        | Job     | Plist viejo                       |
|-----------------|---------|-----------------------------------|
| Mon-Fri 07:00   | morning | com.fer.obsidian-rag-morning      |
| Mon-Fri 22:00   | today   | com.fer.obsidian-rag-today        |
| Dom 22:00       | digest  | com.fer.obsidian-rag-digest       |

**Override schedule via ``rag_brief_schedule_prefs`** (auto-tune feature
2026-04-29): el plist viejo regenera el XML antes del bootstrap leyendo
los prefs. En el supervisor lo manejamos diferente — F3.3 usa los
defaults históricos (mismo que cuando NO hay override). El
``brief_auto_tune_job`` puede cambiar los prefs pero requiere restart
del supervisor para que tomen efecto. F3-followup: agregar IPC handler
``/reload-schedules`` que re-register los 3 briefs sin restartear todo.

Schedule mapping (APScheduler day_of_week, Mon=0..Sun=6):
- ``morning``: ``day_of_week='mon-fri', hour=7, minute=0``
- ``today``: ``day_of_week='mon-fri', hour=22, minute=0``
- ``digest``: ``day_of_week='sun', hour=22, minute=0``
"""
from __future__ import annotations

from typing import Any

from rag.runtime.jobs.nightly import _RAG_BIN, _run_subprocess
from rag.runtime.scheduler import cron


@cron(
    day_of_week="mon-fri", hour=7, minute=0,
    label="morning",
    description="Morning brief Mon-Fri 07:00 (default histórico).",
)
def morning_job() -> dict[str, Any]:
    """``rag morning`` — Mon-Fri 07:00. Voice brief opt-in via
    ``RAG_MORNING_VOICE=1`` en env (default text-only)."""
    return _run_subprocess(
        [_RAG_BIN, "morning"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_EXPLORE": "1",
            "RAG_MORNING_VOICE": "",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=600,
    )


@cron(
    day_of_week="mon-fri", hour=22, minute=0,
    label="today",
    description="Today brief Mon-Fri 22:00 (default histórico).",
)
def today_job() -> dict[str, Any]:
    """``rag today`` — Mon-Fri 22:00."""
    return _run_subprocess(
        [_RAG_BIN, "today"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_EXPLORE": "1",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=600,
    )


@cron(
    day_of_week="sun", hour=22, minute=0,
    label="digest",
    description="Weekly digest Sunday 22:00 (default histórico).",
)
def digest_job() -> dict[str, Any]:
    """``rag digest`` — Domingo 22:00."""
    return _run_subprocess(
        [_RAG_BIN, "digest"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=900,
    )
