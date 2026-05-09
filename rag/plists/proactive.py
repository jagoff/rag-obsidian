"""Proactive push factories: emergent / patterns / archive / distill /
anticipate / active-learning-nudge / brief-auto-tune.

Comparten pattern de "RAG empuja al user" (push proactivo) y comparten
budget global `daily_cap=3` vía `proactive_push`. Silenciables per-kind.

Migrado de rag/plists/_legacy.py en Phase 3 commit 2 (2026-05-09).
"""
from __future__ import annotations

from rag.plists._render import _logs, _render_plist

__all__ = [
    "_active_learning_nudge_plist",
    "_active_learning_suggest_goldens_plist",
    "_anticipate_plist",
    "_archive_plist",
    "_brief_auto_tune_plist",
    "_distill_plist",
    "_emergent_plist",
    "_patterns_plist",
]


def _emergent_plist(rag_bin: str) -> str:
    """Proactive #2 — emergent theme detector, viernes 10am."""
    out, err = _logs("emergent")
    return _render_plist({
        "label": "com.fer.obsidian-rag-emergent",
        "program_arguments": [rag_bin, "emergent"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "schedule": {
            "calendar": {"Weekday": 5, "Hour": 10, "Minute": 0},
        },
        "process_type": "Background",
        "low_priority_io": True,
        "stdout_path": out,
        "stderr_path": err,
    })


def _patterns_plist(rag_bin: str) -> str:
    """Proactive #4 — feedback pattern alert, domingo 20:00.

    Nota 2026-05-01: invoca `rag feedback-patterns` (no `rag patterns`)
    porque el comando original `patterns` quedó shadowed por el grupo
    Click `patterns` agregado en commit 887ece3 (cross-source Pearson).
    Antes del rename, este plist exiteaba con código 2 (Click muestra
    el help del grupo).
    """
    out, err = _logs("patterns")
    return _render_plist({
        "label": "com.fer.obsidian-rag-patterns",
        "program_arguments": [rag_bin, "feedback-patterns"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "schedule": {
            "calendar": {"Weekday": 0, "Hour": 20, "Minute": 0},
        },
        "process_type": "Background",
        "low_priority_io": True,
        "stdout_path": out,
        "stderr_path": err,
    })


def _archive_plist(rag_bin: str) -> str:
    """Proactive archiver — day 1 of each month at 23:00. Runs with --apply;
    the gate (>20 plan entries) short-circuits to a dry-run + notification
    so un-supervised drift can't accidentally move half the vault.
    """
    out, err = _logs("archive")
    return _render_plist({
        "label": "com.fer.obsidian-rag-archive",
        "program_arguments": [rag_bin, "archive", "--apply", "--notify", "--report"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "schedule": {
            "calendar": {"Day": 1, "Hour": 23, "Minute": 0},
        },
        "process_type": "Background",
        "low_priority_io": True,
        "stdout_path": out,
        "stderr_path": err,
    })


def _distill_plist(rag_bin: str) -> str:
    """Weekly conversation distiller — domingos 22:30. Rescata bot answers
    de conversations cuyas sources se evaporaron, escribiéndolos como
    runbook indexable bajo ``03-Resources/runbooks/from-conversations/``.

    Idempotente vía stamp ``distilled_to:`` en el frontmatter del original;
    re-corridas saltean lo ya destilado. Slot domingo 22:30 elegido para:
    correr DESPUÉS del ``digest`` semanal (Dom 22:00) y ANTES del primer
    ``archive`` mensual del lunes 1, así si una conversation cita una
    nota que está por archivarse, el runbook destilado queda indexado
    antes de que el original desaparezca (defense-in-depth con la regla
    promote-on-cite del archive).
    """
    out, err = _logs("distill")
    return _render_plist({
        "label": "com.fer.obsidian-rag-distill",
        "program_arguments": [rag_bin, "distill-conversations", "--apply"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
        },
        "schedule": {
            "calendar": {"Weekday": 0, "Hour": 22, "Minute": 30},
        },
        "process_type": "Background",
        "low_priority_io": True,
        "stdout_path": out,
        "stderr_path": err,
    })


def _anticipate_plist(rag_bin: str) -> str:
    """Anticipatory agent — every 10 min. Evalúa señales y empuja top-1 a WA.

    Cadencia 15min (ex 10min, audit 2026-05-09): el daily_cap=3 hace que
    pollear más seguido no compre coverage adicional — el evento más
    sensible es calendar proximity con threshold 15min, así un tick
    cada 15min lo captura igual con −33% LLM calls/día (96 vs 144).

    Game-changer 2026-04-24: el RAG deja de ser puramente "pull" y pasa a
    "push" cuando tiene algo timely para decirte. 3 señales activas:
      - calendar proximity (eventos próximos 15-90 min)
      - temporal echo (nota de hoy resuena con una vieja >60d)
      - stale commitment (open loop ≥7d, push 1×/sem por loop)

    Comparte daily_cap=3 con `emergent` y `patterns` vía `proactive_push`,
    así que el budget global de pushes por día NO se infla. Silenciable
    per-kind: `rag silence anticipate-calendar` etc. Kill switch global:
    `RAG_ANTICIPATE_DISABLED=1`.
    """
    out, err = _logs("anticipate")
    return _render_plist({
        "label": "com.fer.obsidian-rag-anticipate",
        "program_arguments": [rag_bin, "anticipate", "run"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "schedule": {"interval_s": 900},
        "run_at_load": False,
        "process_type": "Background",
        "stdout_path": out,
        "stderr_path": err,
    })


def _active_learning_nudge_plist(rag_bin: str) -> str:
    """Lunes 10am — recordatorio de labelear queries low-confidence.

    Reemplaza el plist con bash inline historico (que disparaba osascript
    notification de macOS y quedaba sepultado en el Notification Center)
    por una invocacion al command Python `rag active-learning nudge`,
    que prefiere mandar push WA al grupo RagNet con link directo a la
    UI de /learning + fallback a osascript si el bridge esta caido.

    Threshold default 20 candidates ultimos 7 dias. Override por flags
    del CLI si se necesita re-tunear (no env vars hoy — el plist es la
    fuente unica del schedule + parametros).
    """
    out, err = _logs("active-learning-nudge")
    return _render_plist({
        "label": "com.fer.obsidian-rag-active-learning-nudge",
        "program_arguments": [
            rag_bin, "active-learning", "nudge", "--json",
        ],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "schedule": {
            "calendar": {"Weekday": 1, "Hour": 10, "Minute": 0},
        },
        "run_at_load": False,
        "keep_alive": False,
        "process_type": "Background",
        "low_priority_io": True,
        "stdout_path": out,
        "stderr_path": err,
    })


def _active_learning_suggest_goldens_plist(rag_bin: str) -> str:
    """Lunes 11:00 — sugiere entries para queries.yaml desde feedback +1.

    Complementa el `active-learning-nudge` (Lunes 10:00, queries low-conf
    sin label). Este corre 1h después y empuja la otra dirección del loop:
    cuando el user marcó thumbs-up sobre query+top_path, esa señal es oro
    para el golden set y debería terminar en queries.yaml — pero hoy hay
    que recordar a mano correr el script. El plist cierra el loop.

    Threshold default 3 candidates accionables — debajo de eso no vale
    interrumpir al user con un push (los candidates ya estaban en el
    golden set, o el path no existe en el vault, etc).
    """
    out, err = _logs("active-learning-suggest-goldens")
    return _render_plist({
        "label": "com.fer.obsidian-rag-active-learning-suggest-goldens",
        "program_arguments": [
            rag_bin, "active-learning", "suggest-goldens",
            "--days", "7", "--limit", "10", "--threshold", "3",
            "--json",
        ],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "schedule": {
            "calendar": {"Weekday": 1, "Hour": 11, "Minute": 0},
        },
        "run_at_load": False,
        "keep_alive": False,
        "process_type": "Background",
        "low_priority_io": True,
        "stdout_path": out,
        "stderr_path": err,
    })


def _brief_auto_tune_plist(rag_bin: str) -> str:
    """Sunday 03:00 weekly auto-tune of brief schedules (2026-04-29).

    Reads `rag_brief_feedback`, decides whether to shift any of the
    morning/today/digest plists' StartCalendarInterval forward, and
    applies the override via `rag_brief_schedule_prefs`. Sunday 03:00
    is chosen so:

      - It runs AFTER online-tune (03:30 daily) on the only day it
        matters in the same window — actually online-tune is at 03:30,
        so 03:00 sneaks in BEFORE it. That's deliberate: the auto-tune
        write only touches `rag_brief_schedule_prefs` (a single-row PK
        upsert), zero contention with the heavy SQL of online-tune.
      - It's well before `rag digest` (Sunday 22:00 by default, or its
        override) so any shift takes effect on the same Sunday's digest.
      - The user is asleep — no UX surprise from a plist re-bootstrap.

    `--apply` writes the override AND re-bootstraps only the affected
    kind via `launchctl`. `RunAtLoad=false` so `rag setup` doesn't
    fire it on install (no point — there's nothing to tune yet).
    """
    out, err = _logs("brief-auto-tune")
    return _render_plist({
        "label": "com.fer.obsidian-rag-brief-auto-tune",
        "program_arguments": [
            rag_bin, "brief", "schedule", "auto-tune", "--apply",
        ],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "schedule": {
            "calendar": {"Weekday": 0, "Hour": 3, "Minute": 0},
        },
        "run_at_load": False,
        "keep_alive": False,
        "process_type": "Background",
        "low_priority_io": True,
        "stdout_path": out,
        "stderr_path": err,
    })
