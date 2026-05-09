"""Nightly batch jobs migrados al supervisor (F2.1 shadow mode).

Re-implementación in-supervisor de los 6 daemons launchd nightly:

| Schedule | Job              | Plist viejo                          |
|----------|------------------|--------------------------------------|
| 03:00    | auto_harvest     | com.fer.obsidian-rag-auto-harvest    |
| 03:15    | whisper_vocab    | com.fer.obsidian-rag-whisper-vocab   |
| 03:25    | implicit_feedback| com.fer.obsidian-rag-implicit-feedback|
| 03:30    | online_tune      | com.fer.obsidian-rag-online-tune     |
| 04:00    | maintenance      | com.fer.obsidian-rag-maintenance     |
| 05:00    | calibrate        | com.fer.obsidian-rag-calibrate       |

**SHADOW MODE F2.1**: cada job invoca el binario ``rag`` via subprocess
con los mismos argumentos del plist viejo. Garantiza paridad funcional
exacta — si el plist viejo y el supervisor están ambos cargados, los
ejecutables son idénticos modulo el cron timing.

El ahorro de cold-start NO se materializa hasta F2.2 (refactor a
in-process con MLX shared warmup). En F2.1, el supervisor solo
**consolidan el scheduling** — los 6 cron de launchd → 6 cron in-process
con telemetría unificada en ``rag_supervisor_jobs``.

Telemetría per-job:

- ``signals.exit_code`` — return code del subprocess.
- ``signals.stdout_lines`` — count de líneas de stdout.
- ``signals.stderr_lines`` — count.
- ``signals.last_stderr`` — últimos 200 chars (truncados) si exit_code≠0.

Cada job aborta silencioso si el subprocess falla: el scheduler captura
la exception y registra en ``rag_supervisor_jobs.exit_code=1``. Los
jobs siguientes corren igual (no encadenados).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from rag.runtime.scheduler import cron

logger = logging.getLogger(__name__)


_RAG_BIN = str(Path.home() / ".local/bin/rag")
_SUBPROCESS_TIMEOUT_S = 1800  # 30 min — generous, online-tune tarda 24min warm


def _run_subprocess(
    args: list[str],
    *,
    timeout: int = _SUBPROCESS_TIMEOUT_S,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Ejecuta ``args`` y captura stdout/stderr para telemetría.

    Retorna un dict apto para ``rag_supervisor_jobs.signals``:
    - ``exit_code``: int
    - ``stdout_lines``: int (count de líneas, no el contenido)
    - ``stderr_lines``: int
    - ``last_stderr``: str | None — primeros 200 chars del último stderr
      si exit_code != 0 (para diagnóstico rápido).
    """
    import os  # noqa: PLC0415
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("nightly job timeout: %s", " ".join(args))
        return {
            "exit_code": -1,
            "stdout_lines": 0,
            "stderr_lines": 0,
            "last_stderr": f"timeout after {timeout}s",
        }

    stdout_lines = result.stdout.count("\n") if result.stdout else 0
    stderr_lines = result.stderr.count("\n") if result.stderr else 0
    last_err = (result.stderr or "")[-200:] if result.returncode != 0 else None

    if result.returncode != 0:
        logger.warning(
            "nightly job exit=%d: %s — stderr tail: %s",
            result.returncode, " ".join(args), last_err,
        )

    return {
        "exit_code": result.returncode,
        "stdout_lines": stdout_lines,
        "stderr_lines": stderr_lines,
        "last_stderr": last_err,
    }


# ── Jobs ────────────────────────────────────────────────────────────────────


@cron(
    hour=3, minute=0,
    label="auto_harvest",
    description="Auto-labelear queries low-confidence con LLM-as-judge.",
)
def auto_harvest_job() -> dict[str, Any]:
    """Equivalente a ``rag feedback auto-harvest --since 1 --limit 50 --json``
    del plist viejo. Schedule: 03:00 daily."""
    return _run_subprocess(
        [_RAG_BIN, "feedback", "auto-harvest",
         "--since", "1", "--limit", "50", "--json"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )


@cron(
    hour=3, minute=15,
    label="whisper_vocab",
    description="Refresh vocab WA para Whisper transcription.",
)
def whisper_vocab_job() -> dict[str, Any]:
    """Equivalente a ``rag whisper vocab refresh`` del plist viejo.
    Schedule: 03:15 daily."""
    return _run_subprocess(
        [_RAG_BIN, "whisper", "vocab", "refresh"],
        extra_env={"NO_COLOR": "1", "TERM": "dumb"},
    )


@cron(
    hour=3, minute=25,
    label="implicit_feedback",
    description="Inferrer pipeline: corrective_path + requery + outcome.",
)
def implicit_feedback_job() -> dict[str, Any]:
    """Equivalente al pipeline shell del plist viejo:
    ``rag feedback infer-implicit && detect-requery && classify-sessions``.

    Tres sub-comandos en serie. Si el primero falla, los siguientes
    siguen (idempotentes). Schedule: 03:25 daily."""
    results: list[dict[str, Any]] = []
    for sub in ("infer-implicit", "detect-requery", "classify-sessions"):
        r = _run_subprocess(
            [_RAG_BIN, "feedback", sub, "--json"],
            extra_env={"NO_COLOR": "1", "TERM": "dumb"},
        )
        r["sub"] = sub
        results.append(r)
    worst = max((r["exit_code"] for r in results), default=0)
    return {
        "exit_code": worst,
        "sub_results": results,
        "n_subs_ok": sum(1 for r in results if r["exit_code"] == 0),
    }


@cron(
    hour=3, minute=30,
    label="online_tune",
    description="Online tune del ranker — re-aprendiza weights del último día.",
)
def online_tune_job() -> dict[str, Any]:
    """Equivalente a ``rag tune --online --days 14 --apply --yes``.
    Schedule: 03:30 daily. Tarda ~24min warm en M-chip."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return _run_subprocess(
        [_RAG_BIN, "tune", "--online", "--days", "14", "--apply", "--yes"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_EVAL_GATE_TIMEOUT_S": "2400",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=2700,  # 45min — online-tune con eval gate puede tardar ~30min
    )


@cron(
    hour=4, minute=0,
    label="maintenance",
    description="Daily housekeeping: WAL checkpoint + log rotation + VACUUM gateado.",
)
def maintenance_job() -> dict[str, Any]:
    """Equivalente a ``rag maintenance`` del plist viejo.
    Schedule: 04:00 daily."""
    return _run_subprocess(
        [_RAG_BIN, "maintenance"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )


@cron(
    hour=5, minute=0,
    label="calibrate",
    description="Score calibration isotonic — train per-source desde feedback 90d.",
)
def calibrate_job() -> dict[str, Any]:
    """Equivalente a ``rag calibrate --since 90 --as-json``.
    Schedule: 05:00 daily (stagger post online-tune que dura 24min).
    Audit 2026-05-09 movió de 04:30 → 05:00."""
    return _run_subprocess(
        [_RAG_BIN, "calibrate", "--since", "90", "--as-json"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_SCORE_CALIBRATION": "1",
        },
    )
