"""Housekeeping daemons migrados al supervisor (F3.3).

| Schedule         | Job             | Plist viejo                          |
|------------------|-----------------|--------------------------------------|
| Daily 02:00      | vault_cleanup   | com.fer.obsidian-rag-vault-cleanup   |
| Daily 04:00      | wake_up         | com.fer.obsidian-rag-wake-up         |
| Lun 06:00        | consolidate     | com.fer.obsidian-rag-consolidate     |

``maintenance`` (daily 04:00) ya está en ``nightly.py`` — no se duplica.
``wake_up`` corre 04:00 mismo que ``maintenance`` pero ambos son
idempotentes y orquestan cosas distintas (wake-up llama a ``rag
wake-up`` que reindex + maintenance + patterns + emergent + morning +
warmup; maintenance solo hace WAL checkpoint + log rotation).
"""
from __future__ import annotations

from typing import Any

from rag.runtime.jobs.nightly import _RAG_BIN, _run_subprocess
from rag.runtime.scheduler import cron


@cron(
    hour=2, minute=0,
    label="vault_cleanup",
    description="Vault transient folder cleanup — daily 02:00.",
)
def vault_cleanup_job() -> dict[str, Any]:
    """``rag vault-cleanup`` — daily 02:00. Mueve archivos viejos en
    99-obsidian/99-AI/{tmp,conversations,sessions,plans,system,reviews}/
    al .trash/. Solo FS, no toca ragvec.db."""
    return _run_subprocess(
        [_RAG_BIN, "vault-cleanup"],
        extra_env={"NO_COLOR": "1", "TERM": "dumb"},
        timeout=600,
    )


@cron(
    hour=4, minute=0,
    label="wake_up",
    description="Wake-up orchestrator — daily 04:00.",
)
def wake_up_job() -> dict[str, Any]:
    """``rag wake-up`` — daily 04:00. Orquestador del bundle nightly:
    rag index + maintenance + patterns + emergent + morning + warmup.
    ~15-20min end-to-end. Si Mac estaba en sleep a su horario, launchd
    dispara el job al wake — supervisor también, ya que su scheduler
    APScheduler maneja misfires con coalesce=True."""
    return _run_subprocess(
        [_RAG_BIN, "wake-up"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=2700,  # 45min — wake-up es heavy
    )


@cron(
    day_of_week="mon", hour=6, minute=0,
    label="consolidate",
    description="Weekly memory consolidation — lunes 06:00.",
)
def consolidate_job() -> dict[str, Any]:
    """``rag consolidate --apply`` — Lunes 06:00. Promotes recurring
    conversation clusters de 99-obsidian/99-AI/conversations/ a PARA y
    archives los originales (Phase 2 del plan episodic-memory)."""
    return _run_subprocess(
        [_RAG_BIN, "consolidate", "--apply"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=1800,
    )
