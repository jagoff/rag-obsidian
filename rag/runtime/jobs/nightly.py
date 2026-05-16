"""Nightly batch jobs migrados al supervisor (F2.1 shadow mode).

Re-implementación in-supervisor de los daemons launchd nightly:

| Schedule         | Job               | Plist viejo                            |
|------------------|-------------------|----------------------------------------|
| 03:00 daily      | auto_harvest      | com.fer.obsidian-rag-auto-harvest      |
| 03:15 daily      | whisper_vocab     | com.fer.obsidian-rag-whisper-vocab     |
| 03:25 daily      | implicit_feedback | com.fer.obsidian-rag-implicit-feedback |
| 03:30 daily      | online_tune       | com.fer.obsidian-rag-online-tune       |
| 04:00 daily      | maintenance       | com.fer.obsidian-rag-maintenance       |
| 04:00 weekly Sun | drafts_finetune   | (nuevo 2026-05-11, cierra draft loop)  |
| 04:30 weekly Sun | reranker_finetune | (nuevo 2026-05-10, ex feature dormida) |
| 05:00 daily      | calibrate         | com.fer.obsidian-rag-calibrate         |

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
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_VENV_PY = _REPO_ROOT / ".venv" / "bin" / "python"
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
    hour=2, minute=30,
    label="identity_fingerprint_refresh",
    description="Recompute style fingerprint del user (G2 — Identity Layer 2).",
)
def identity_fingerprint_refresh_job() -> dict[str, Any]:
    """Equivalente a `python -c 'from rag.identity_layer import compute_fingerprint; compute_fingerprint(90)'`.

    Schedule: 02:30 daily, antes de auto_harvest (03:00). El fingerprint
    se persiste a `~/.local/share/obsidian-rag/identity_fingerprint.json`
    y se lee on-demand por `system_prompt_for_intent` para inyectar al
    prompt del chat/draft loop.

    Out: stdout con metrics del último compute (n_samples, avg_words).
    """
    return _run_subprocess(
        [str(_VENV_PY), "-c",
         "from rag.identity_layer import compute_fingerprint; "
         "fp = compute_fingerprint(days=90); "
         "print(f\"n={fp.get('n_samples')} avg={fp.get('avg_words_per_msg')}\")"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "PYTHONPATH": str(_REPO_ROOT),
        },
        timeout=120,
    )


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
    ``rag feedback detect-requery && infer-implicit && classify-sessions``.

    Tres sub-comandos en serie. Si el primero falla, los siguientes
    siguen (idempotentes). Schedule: 03:25 daily."""
    results: list[dict[str, Any]] = []
    for sub in ("detect-requery", "infer-implicit", "classify-sessions"):
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
    hour=4, minute=30, day_of_week=6,
    label="reranker_finetune",
    description="Weekly LoRA fine-tune del bge-reranker si hay signal suficiente.",
)
def reranker_finetune_job() -> dict[str, Any]:
    """LoRA fine-tune semanal del reranker bge-reranker-v2-m3.

    Schedule: domingo 04:30 (después de online_tune + calibrate del sábado).
    Feature activada 2026-05-10 (audit flujos aprendizaje) — antes era
    dormida, el script existía pero ningún plist/job lo invocaba.

    Pipeline:
      1. Exporta pairs vía `scripts/export_training_pairs.py` a tmp JSONL.
      2. Cuenta líneas. Si < `RAG_FINETUNE_MIN_PAIRS` (default 20) → exit 0
         silent (signal insuficiente, próxima semana retry).
      3. Si >= 20 → corre `finetune_reranker.py --mode lora --pairs-from`
         que entrena LoRA r=8 a `~/.local/share/obsidian-rag/reranker_ft/`.
      4. El adapter se carga al próximo cold-load del reranker (web server
         restart o idle-unload), gated por `RAG_RERANKER_FT=1`.

    Silent-fail policy: cualquier exit_code≠0 queda en
    `rag_supervisor_jobs.signals.exit_code` para diagnóstico vía
    `rag daemons status`; no interrumpe otros jobs.
    """
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    py_bin = str(repo_root / ".venv" / "bin" / "python")
    export_script = str(repo_root / "scripts" / "export_training_pairs.py")
    finetune_script = str(repo_root / "scripts" / "finetune_reranker.py")
    min_pairs = int(os.environ.get("RAG_FINETUNE_MIN_PAIRS", "20"))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir="/tmp",
    ) as tmp:
        tmp_path = tmp.name

    try:
        export = _run_subprocess(
            [py_bin, export_script, "--days", "60", "--output", tmp_path,
             "--min-negatives", "1"],
            extra_env={"NO_COLOR": "1", "TERM": "dumb"},
            timeout=300,
        )
        if export["exit_code"] != 0:
            return {
                "exit_code": export["exit_code"],
                "phase": "export",
                "last_stderr": export.get("last_stderr"),
            }

        n_pairs = 0
        try:
            with open(tmp_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n_pairs += 1
        except OSError as exc:
            logger.warning("reranker_finetune: failed reading pairs jsonl: %s", exc)
            return {"exit_code": 1, "phase": "read_pairs", "n_pairs": 0}

        if n_pairs < min_pairs:
            logger.info(
                "reranker_finetune: skip — n_pairs=%d < min_pairs=%d (semana próxima retry)",
                n_pairs, min_pairs,
            )
            return {
                "exit_code": 0,
                "phase": "skip_insufficient_signal",
                "n_pairs": n_pairs,
                "min_pairs": min_pairs,
            }

        train = _run_subprocess(
            [py_bin, finetune_script, "--mode", "lora",
             "--pairs-from", tmp_path],
            extra_env={
                "NO_COLOR": "1",
                "TERM": "dumb",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            timeout=2700,
        )
        return {
            "exit_code": train["exit_code"],
            "phase": "trained" if train["exit_code"] == 0 else "train_failed",
            "n_pairs": n_pairs,
            "last_stderr": train.get("last_stderr"),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@cron(
    hour=4, minute=0, day_of_week=6,
    label="drafts_finetune",
    description="Weekly DPO+LoRA fine-tune del modelo de drafts WhatsApp si hay signal suficiente.",
)
def drafts_finetune_job() -> dict[str, Any]:
    """DPO+LoRA fine-tune semanal del modelo de drafts (Qwen2.5-7B base).

    Schedule: domingo 04:00 (antes del reranker 04:30 → eviten chocar
    GPU/CPU). Cierra el loop del bot WA: ``rag_draft_decisions`` ya
    captura ``/si /no /editar`` reales del user; este job aprende sobre
    esos pares de preferencia y deja un adapter LoRA que el endpoint
    ``/api/draft/preview`` carga cuando ``RAG_DRAFTS_FT=1``.

    Pipeline:
      1. Augmenta dataset via ``scripts/augment_drafts_dataset.py`` —
         inserta pares synthetic `(corporate_template_draft, sent_text)`
         desde outgoing reales del bridge SQLite. Solo agrega nuevos
         (idempotente vía `draft_id=synthetic-{wa_msg_id}` UNIQUE),
         existing rows skip. Si el bridge no está disponible → noop.
      2. Cuenta pares con preference completo (``approved_editar`` con
         ``sent_text`` + synthetic). Si < ``RAG_DRAFTS_FINETUNE_MIN_PAIRS``
         (default 100, según el script real) → exit 0 silent.
      3. Si ≥ threshold → corre ``scripts/finetune_drafts.py --epochs 1``
         con DPO. Adapter cae en ``~/.local/share/obsidian-rag/drafts_ft/``.
      4. Activación opt-in via env ``RAG_DRAFTS_FT=1`` (el endpoint
         ``/api/draft/preview`` lo lee). Default OFF — el user habilita
         a mano post-validation A/B.

    Gate por env ``RAG_AUTO_FINETUNE_DRAFTS`` (default ``1``, opt-out
    via ``=0``). Razón: si el job falla por dependencia faltante
    (``peft`` / ``trl``) o disco, NO queremos spammear daemon
    failures cada domingo — el operador puede silenciar con un env
    var en el plist del supervisor sin tocar código.

    Silent-fail policy igual al resto: exit_code≠0 queda en
    ``rag_supervisor_jobs.signals`` para ``rag daemons status``;
    no interrumpe otros jobs.
    """
    import os  # noqa: PLC0415

    if os.environ.get("RAG_AUTO_FINETUNE_DRAFTS", "1") != "1":
        return {"exit_code": 0, "phase": "gated_off"}

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    py_bin = str(repo_root / ".venv" / "bin" / "python")
    augment_script = str(repo_root / "scripts" / "augment_drafts_dataset.py")
    finetune_script = str(repo_root / "scripts" / "finetune_drafts.py")
    min_pairs = int(os.environ.get("RAG_DRAFTS_FINETUNE_MIN_PAIRS", "100"))

    # ── 1. Augment (best-effort) ─────────────────────────────────────
    if Path(augment_script).is_file():
        augment = _run_subprocess(
            [py_bin, augment_script],
            extra_env={"NO_COLOR": "1", "TERM": "dumb"},
            timeout=300,
        )
        augment_exit = augment["exit_code"]
        augment_stderr = augment.get("last_stderr")
    else:
        augment_exit = -1  # script missing → siguen los demás pasos
        augment_stderr = "augment_script_missing"

    # ── 2. Count preference pairs ────────────────────────────────────
    n_pairs = 0
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
        with _ragvec_state_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM rag_draft_decisions WHERE "
                "decision IN ('approved_editar') AND sent_text IS NOT NULL "
                "AND length(sent_text) >= 10"
            ).fetchone()
            n_pairs = int(row[0] or 0) if row else 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("drafts_finetune: count failed: %s", exc)
        return {
            "exit_code": 1,
            "phase": "count_failed",
            "last_stderr": str(exc)[:200],
            "augment_exit": augment_exit,
            "augment_stderr": augment_stderr,
        }

    if n_pairs < min_pairs:
        logger.info(
            "drafts_finetune: skip — n_pairs=%d < min_pairs=%d (próximo domingo retry)",
            n_pairs, min_pairs,
        )
        return {
            "exit_code": 0,
            "phase": "skip_insufficient_signal",
            "n_pairs": n_pairs,
            "min_pairs": min_pairs,
            "augment_exit": augment_exit,
        }

    # ── 3. Fine-tune ─────────────────────────────────────────────────
    if not Path(finetune_script).is_file():
        return {
            "exit_code": 1, "phase": "finetune_script_missing",
            "n_pairs": n_pairs,
        }
    train = _run_subprocess(
        [py_bin, finetune_script, "--epochs", "1"],
        extra_env={
            "NO_COLOR": "1",
            "TERM": "dumb",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        timeout=2700,
    )
    return {
        "exit_code": train["exit_code"],
        "phase": "trained" if train["exit_code"] == 0 else "train_failed",
        "n_pairs": n_pairs,
        "augment_exit": augment_exit,
        "last_stderr": train.get("last_stderr"),
    }


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
