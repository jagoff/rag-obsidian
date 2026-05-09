"""Proactive weekly/monthly jobs migrados al supervisor (F3.2).

Re-implementación in-supervisor de los 7 daemons launchd weekly/monthly:

| Schedule         | Job                            | Plist viejo                                 |
|------------------|--------------------------------|---------------------------------------------|
| Vie 10:00        | emergent                       | com.fer.obsidian-rag-emergent               |
| Dom 20:00        | patterns                       | com.fer.obsidian-rag-patterns               |
| Día 1 23:00      | archive                        | com.fer.obsidian-rag-archive                |
| Dom 22:30        | distill                        | com.fer.obsidian-rag-distill                |
| Lun 10:00        | active_learning_nudge          | com.fer.obsidian-rag-active-learning-nudge  |
| Lun 11:00        | active_learning_suggest_goldens| com.fer.obsidian-rag-active-learning-suggest-goldens |
| Dom 03:00        | brief_auto_tune                | com.fer.obsidian-rag-brief-auto-tune        |

SHADOW MODE — subprocess wrappers con paridad estricta. Schedules
preservados (APScheduler ``day_of_week`` Mon=0..Sun=6 igual que launchd
``Weekday`` con offset por +1 que mapea acá).
"""
from __future__ import annotations

from typing import Any

from rag.runtime.jobs.nightly import _RAG_BIN, _run_subprocess
from rag.runtime.scheduler import at, cron


@at(
    weekday=4, hour=10, minute=0,
    label="emergent",
    description="Emergent theme detector — viernes 10am.",
)
def emergent_job() -> dict[str, Any]:
    """``rag emergent`` — Viernes 10:00. APScheduler day_of_week=4 (Mon=0)."""
    return _run_subprocess(
        [_RAG_BIN, "emergent"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=900,
    )


@at(
    weekday=6, hour=20, minute=0,
    label="patterns",
    description="Feedback pattern alert — domingo 20:00.",
)
def patterns_job() -> dict[str, Any]:
    """``rag feedback-patterns`` — Domingo 20:00. NOT ``rag patterns``
    (ese es otro grupo Click — shadowing histórico explicado en plist
    viejo)."""
    return _run_subprocess(
        [_RAG_BIN, "feedback-patterns"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=900,
    )


@cron(
    day=1, hour=23, minute=0,
    label="archive",
    description="Auto-archiver mensual — día 1 a las 23:00.",
)
def archive_job() -> dict[str, Any]:
    """``rag archive --apply --notify --report`` — día 1 de cada mes
    23:00. APScheduler usa ``day=1`` para day-of-month."""
    return _run_subprocess(
        [_RAG_BIN, "archive", "--apply", "--notify", "--report"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=1800,
    )


@at(
    weekday=6, hour=22, minute=30,
    label="distill",
    description="Conversation distiller — domingo 22:30.",
)
def distill_job() -> dict[str, Any]:
    """``rag distill-conversations --apply`` — Domingo 22:30. Defense
    in-depth con archive (rescata bot answers antes de archive mensual)."""
    return _run_subprocess(
        [_RAG_BIN, "distill-conversations", "--apply"],
        extra_env={"NO_COLOR": "1", "TERM": "dumb"},
        timeout=900,
    )


@at(
    weekday=0, hour=10, minute=0,
    label="active_learning_nudge",
    description="Nudge labeling queries low-conf — lunes 10am.",
)
def active_learning_nudge_job() -> dict[str, Any]:
    """``rag active-learning nudge --json`` — Lunes 10:00."""
    return _run_subprocess(
        [_RAG_BIN, "active-learning", "nudge", "--json"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=300,
    )


@at(
    weekday=0, hour=11, minute=0,
    label="active_learning_suggest_goldens",
    description="Suggest goldens — lunes 11am.",
)
def active_learning_suggest_goldens_job() -> dict[str, Any]:
    """``rag active-learning suggest-goldens --days 7 --limit 10
    --threshold 3 --json`` — Lunes 11:00."""
    return _run_subprocess(
        [_RAG_BIN, "active-learning", "suggest-goldens",
         "--days", "7", "--limit", "10", "--threshold", "3", "--json"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=300,
    )


@at(
    weekday=6, hour=3, minute=0,
    label="brief_auto_tune",
    description="Brief schedule auto-tune — domingo 03:00.",
)
def brief_auto_tune_job() -> dict[str, Any]:
    """``rag brief schedule auto-tune --apply`` — Domingo 03:00. Lee
    rag_brief_feedback last 30d, posiblemente shifts schedule de
    morning/today/digest dentro de safe band."""
    return _run_subprocess(
        [_RAG_BIN, "brief", "schedule", "auto-tune", "--apply"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=300,
    )
