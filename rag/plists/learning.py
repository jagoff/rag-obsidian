"""Learning loop nocturno — auto-aprendizaje del ranker + reranker.

Cadena nightly: auto-harvest 03:00 → whisper-vocab 03:15 →
implicit-feedback 03:25 → online-tune 03:30 → calibrate 04:30.
`routing-rules` corre cada 5min recolectando patterns que el online-tune
del día siguiente consume.

Migrado de rag/plists/_legacy.py en Phase 3 commit 2 (2026-05-09).
"""
from __future__ import annotations

from pathlib import Path

from rag.plists._render import _logs, _render_plist, _repo_root

__all__ = [
    "_auto_harvest_plist",
    "_calibration_plist",
    "_drift_watcher_plist",
    "_implicit_feedback_plist",
    "_online_tune_plist",
    "_routing_rules_plist",
    "_whisper_vocab_plist",
]


def _auto_harvest_plist(rag_bin: str) -> str:
    """Nightly auto-harvest — every day at 03:00, before online-tune (03:30).

    Corre `rag feedback auto-harvest` sobre queries low-confidence de las
    últimas 24h sin feedback explícito. Un LLM-as-judge decide qué chunk
    responde mejor cada query y sólo inserta rows cuando la confianza
    del juez es ≥ 0.8. Los rows tienen source='auto-harvester' en
    extra_json para poder auditarlos por separado del harvester manual.

    Programado a las 03:00 para que el online-tune de 03:30 ya vea la
    señal fresca que generó el auto-harvest. El ollama está idle a esa
    hora (después del day-use, antes de los daemons que ingestan).

    RunAtLoad=false — no conviene blockear rag setup con un run completo.
    """
    out, err = _logs("auto-harvest")
    return _render_plist({
        "label": "com.fer.obsidian-rag-auto-harvest",
        "program_arguments": [
            rag_bin, "feedback", "auto-harvest",
            "--since", "1", "--limit", "50", "--json",
        ],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_LLM_BACKEND": "mlx",
        },
        "schedule": {
            "calendar": {"Hour": 3, "Minute": 0},
        },
        "run_at_load": False,
        "keep_alive": False,
        "stdout_path": out,
        "stderr_path": err,
    })


def _online_tune_plist(rag_bin: str) -> str:
    """Nightly online-tune — every day at 03:30, after Ollama is idle.

    Bug 2026-04-20 → fix 2026-04-25: el plist no especificaba
    ``WorkingDirectory`` y launchd lanzaba el comando desde ``/``. ``rag tune
    --online`` defaultea ``--file queries.yaml`` (path relativo); resolvía a
    ``/queries.yaml`` (inexistente) y la función retornaba silenciosa con un
    "No existe /queries.yaml" en el log → 5 noches sin tune efectivo
    (``ranker.json saved_at=2026-04-20T19:19:12``). Fix: anclar el cwd al
    repo (donde vive ``queries.yaml``) usando el path del package.

    Bug 2026-04-25 → fix 2026-04-27: el CI gate timeoutea a 1200s (20 min)
    pero el ``rag eval`` real tarda 24 min en mac M-chip warm. Resultado:
    auto-rollback en TODA corrida nightly desde el 25, marcando el plist
    como crashed (``status=1``) y disparando el panel rojo "Algo no está
    bien" en /learning. Fix: setear ``RAG_EVAL_GATE_TIMEOUT_S=2400`` (40 min)
    explícito en el plist para no depender del default del código.
    """
    working_dir = _repo_root()
    out, err = _logs("online-tune")
    return _render_plist({
        "label": "com.fer.obsidian-rag-online-tune",
        "program_arguments": [
            rag_bin, "tune", "--online", "--days", "14", "--apply", "--yes",
        ],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_EVAL_GATE_TIMEOUT_S": "2400",
            "RAG_LLM_BACKEND": "mlx",
        },
        "schedule": {
            "calendar": {"Hour": 3, "Minute": 30},
        },
        "run_at_load": False,
        "keep_alive": False,
        "working_dir": str(working_dir),
        "stdout_path": out,
        "stderr_path": err,
    })


def _calibration_plist(rag_bin: str) -> str:
    """Nightly score calibration — 04:30, after auto-harvest (03:00) and
    online-tune (03:30). The --since 90 window covers the last 3 months
    of feedback for training isotonic per source; re-runs are cheap
    (<1s typical) because everything's in-process.

    `RAG_SCORE_CALIBRATION=1` (rolleado 2026-04-30): el daemon corría
    con `=0` heredado de la fase de validación, pero `calibrate_score()`
    bailea con el flag apagado y entonces el entrenamiento generaba un
    isotonic que nunca se aplicaba (telemetría 30d: 0 calibrated_score
    rows en `rag_queries.extra_json` aunque el job corría todas las
    noches). Con `=1` el `calibrate` command lee feedback real (que ya
    pasa por raw-score retrieval) y entrena el isotonic; la lectura
    misma del telemetry es en raw porque ya quedó persistida sin
    calibrar — el flag solo afecta NUEVAS queries del web/serve plists.
    Detalle del rollout en commit `4f7e41f`.
    """
    out, err = _logs("calibrate")
    return _render_plist({
        "label": "com.fer.obsidian-rag-calibrate",
        "program_arguments": [
            rag_bin, "calibrate", "--since", "90", "--as-json",
        ],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_SCORE_CALIBRATION": "1",
        },
        "schedule": {
            "calendar": {"Hour": 4, "Minute": 30},
        },
        "run_at_load": False,
        "keep_alive": False,
        "stdout_path": out,
        "stderr_path": err,
    })


def _implicit_feedback_plist(rag_bin: str) -> str:
    """Nightly implicit feedback pipeline — corre 03:25, 5 min antes del
    online-tune.

    Ejecuta 3 pasos en cadena via shell (cada uno persiste señal a
    `rag_feedback`, idempotentes):

      1. `rag feedback infer-implicit --json` — corrective_path desde behavior
         post-👎 (ver `rag_implicit_learning.corrective_paths`).
      2. `rag feedback detect-requery --json` — paráfrasis <30s = loss
         implícito (ver `rag_implicit_learning.requery_detection`).
      3. `rag feedback classify-sessions --json` — outcome win/loss/abandon
         con reward shaping a los turns (ver `rag_implicit_learning.session_outcome`
         + `reward_shaping`).

    Schedule a las 03:25 deliberado: el `online-tune` corre 03:30, y este
    pipeline lo precede para que la signal nueva entre a la primera corrida
    del tune. 5 minutos es suficiente — los 3 inferrers son ~50ms cada
    uno, dominados por SQL setup.

    Salida JSON al log (3 líneas por corrida, una por step) para que
    `tail -f implicit-feedback.log` muestre métricas estructuradas sin
    parseo. RunAtLoad=false — solo tiene sentido nightly tras acumular
    signal del día.

    Sprint 1 del cierre del loop de auto-aprendizaje (2026-04-26).
    """
    cmd = (
        f'{rag_bin} feedback infer-implicit --json && '
        f'{rag_bin} feedback detect-requery --json && '
        f'{rag_bin} feedback classify-sessions --json'
    )
    cmd_xml = cmd.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    out, err = _logs("implicit-feedback")
    return _render_plist({
        "label": "com.fer.obsidian-rag-implicit-feedback",
        "program_arguments": ["/bin/bash", "-c", cmd_xml],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
        },
        "schedule": {
            "calendar": {"Hour": 3, "Minute": 25},
        },
        "run_at_load": False,
        "keep_alive": False,
        "stdout_path": out,
        "stderr_path": err,
    })


def _routing_rules_plist(rag_bin: str) -> str:
    """Detector de patrones de ruteo — cada 5 minutos, analiza
    comportamiento y promueve nuevas rutas de queries automáticamente.

    Fix 2026-05-01: agregamos `--auto-promote` para que el cron cierre
    el loop end-to-end. Antes el daemon SOLO listaba candidatos
    (`extract-rules` sin flag = listing puro) → `rag_routing_rules`
    quedaba con 0 rows aunque hubiera patrones consistentes. Ahora,
    cuando un patrón cumple `min_count=5` y `min_ratio=0.90`, se
    upsertea directo a `rag_routing_rules(active=1)` y el listener
    WhatsApp lo aplica en el próximo dispatch. Sin esto, el loop
    quedaba half-closed (collector OK, trainer OK, apply ✗).
    """
    out, err = _logs("routing-rules")
    return _render_plist({
        "label": "com.fer.obsidian-rag-routing-rules",
        "program_arguments": [
            rag_bin, "routing", "extract-rules", "--auto-promote",
        ],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
        },
        "schedule": {"interval_s": 300},
        "run_at_load": False,
        "stdout_path": out,
        "stderr_path": err,
    })


def _drift_watcher_plist(rag_bin: str) -> str:
    """Drift watcher — every 6h, alerta si singles_hit5/chains_hit5 cae
    entre runs de `rag eval`.

    Motivación (audit ronda 2 2026-05-09): el gate de auto-rollback solo
    corre nightly (`com.fer.obsidian-rag-online-tune` 03:30). Una regresión
    a las 14:00 se detecta recién 13h después. Para captura rápida de
    drift silencioso, este daemon polea `rag_eval_runs` cada 6h y dispara:
      1. JSONL append a `~/.local/share/obsidian-rag/drift_alerts.jsonl`
         (source of truth offline).
      2. WhatsApp push best-effort vía bridge local — si el bridge está
         caído, JSONL queda como rastro.

    Thresholds:
      - `delta_singles < -0.05` (-5pp) — n grande, señal real.
      - `delta_chains < -0.07` (-7pp) — n más chico, threshold relajado.

    Idempotente: dedup window 12h por (kind, current_run_ts) leyendo las
    últimas 5 líneas del JSONL antes de escribir.

    Schedule cada 6h: balance entre detección rápida (<6h lag) y costo
    nulo (script termina en <1s lecturas SQL puras). RunAtLoad=true para
    que el primer tick post-bootstrap dé un baseline check.

    Working dir: el script usa Path.home() para todos los reads (DB +
    JSONL + ambient.json), así que cwd no importa. Usamos el repo root
    de todos modos para consistencia con online-tune.

    Lógica completa en `scripts/drift_watcher.py`. Para auditar manual:
    `make drift-watcher` en el repo.
    """
    repo = _repo_root()
    venv_python = repo / ".venv" / "bin" / "python"
    script = repo / "scripts" / "drift_watcher.py"
    out, err = _logs("drift-watcher")
    return _render_plist({
        "label": "com.fer.obsidian-rag-drift-watcher",
        "program_arguments": [str(venv_python), str(script)],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
        },
        "schedule": {"interval_s": 21600},  # 6h
        "run_at_load": True,
        "keep_alive": False,
        "working_dir": str(repo),
        "stdout_path": out,
        "stderr_path": err,
    })


def _whisper_vocab_plist(rag_bin: str) -> str:
    """Extractor nightly de vocabulario de transcripción WhatsApp — 03:15,
    popula rag_whisper_vocab para mejorar el reconocimiento de Whisper.

    Fix 2026-05-01: el comando real es `rag whisper vocab refresh` (3
    niveles: grupo `whisper` → subgrupo `vocab` → comando `refresh`).
    Antes el plist decía `whisper-vocab refresh` (con guión) que no
    existía como comando — el daemon fallaba silenciosamente cada noche
    desde el 2026-04-25, dejando `rag_whisper_vocab` con vocab estático
    (400 rows congeladas). Resultado: la transcripción de WhatsApp no
    aprendía términos nuevos del corpus reciente. Ver memoria
    `whisper-vocab-plist-fix-2026-05-01` en mem-vault para el detalle.
    """
    out, err = _logs("whisper-vocab")
    return _render_plist({
        "label": "com.fer.obsidian-rag-whisper-vocab",
        "program_arguments": [rag_bin, "whisper", "vocab", "refresh"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
        },
        "schedule": {
            "calendar": {"Hour": 3, "Minute": 15},
        },
        "stdout_path": out,
        "stderr_path": err,
    })
