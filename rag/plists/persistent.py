"""Daemons launchd persistentes (KeepAlive=true, sin schedule).

Hot path — long-running, sin cron. `_watch_plist` observa cambios en
todas las vaults registradas; `_web_plist` corre la FastAPI UI + SSE
chat en :8765.
"""
from __future__ import annotations

import os
from pathlib import Path

from rag.plists._render import _logs, _render_plist, _repo_root

__all__ = ["_supervisor_plist", "_watch_plist", "_web_plist"]


def _supervisor_plist(rag_bin: str) -> str:
    """Persistent supervisor — único daemon que orquesta scheduling
    in-process para reemplazar 32 plists individuales (refactor F1+ 2026-
    05-09). Ver ADR `99-obsidian/99-AI/system/daemon-refactor-2026-05-09/
    supervisor-refactor-adr.md`.

    Características:
    - ``KeepAlive=true`` + ``RunAtLoad=true`` — supervisor es persistent.
    - ``ProcessType=Background`` — audit 2026-05-10 bumpó de Adaptive →
      Background. Adaptive era el default genérico pero permitía que el
      supervisor fuera killed por jetsam macOS bajo memory pressure
      (gaps 216-218min en ``rag_supervisor_jobs.routing_rules`` coinciden
      con Mac sleep + posible jetsam durante wake transitorio). Background
      da prioridad persistente similar a Apple system daemons. NO ``Standard``
      porque bloquea suspend del Mac.
    - ``RAG_SUPERVISOR_MLX_WARMUP=0`` — NO carga 5 modelos en paralelo al
      startup (fix 2026-05-10). El warmup eager metía 7 GB residentes solo
      en el supervisor, sumado al web (~5 GB) → swap a 2.8 GB / OOM /
      reinicio. Jobs in-process lazy-loadean cuando los necesitan; idle-TTL
      los evicta. Override para volver al comportamiento previo: ``=1``.
    - ``RAG_MLX_IDLE_TTL=1800`` — 30min idle-evict. Antes era 7200 (2h),
      pero supervisor solo corre jobs batch sparse (drift_watcher, briefs,
      housekeeping) — keepar modelos 2h cargados sin uso era waste.
    - ``RAG_MEMORY_PRESSURE_*`` — watchdog activo (75%, 4 GB swap, 30s).
      Antes solo el web reaccionaba; el supervisor mantenía modelos
      pinneados aunque la Mac estuviera swappeando.
    - **Removido** ``RAG_RERANKER_NEVER_UNLOAD=1`` — pineaba reranker en
      MPS (~2-3 GB) para jobs batch que NO son hot-path. Sin pin, idle-TTL
      lo libera. La var sigue activa en el web plist (queries del user sí
      son hot-path).
    - ``ExitTimeOut=20`` — 20s graceful shutdown (matchea el SIGTERM
      handler del supervisor).
    - Entrypoint via ``venv_python -m rag.runtime.supervisor`` (no via
      ``rag supervisor run``) para garantizar que las env vars del plist
      se setean ANTES de cualquier import de Python — análogo al pattern
      del web plist donde HF_HUB_OFFLINE debe estar set antes de
      sentence-transformers.

    Working dir = repo root. Logs a ``~/.local/share/obsidian-rag/
    supervisor.log`` (stdout) + ``supervisor.error.log`` (stderr).
    """
    repo_root = _repo_root()
    venv_python = repo_root / ".venv" / "bin" / "python"
    out, err = _logs("supervisor")
    return _render_plist({
        "label": "com.fer.obsidian-rag-supervisor",
        "program_arguments": [
            str(venv_python),
            "-m", "rag.runtime.supervisor",
        ],
        "env": {
            "PYTHONUNBUFFERED": "1",
            "RAG_LLM_BACKEND": "mlx",
            "RAG_LOCAL_EMBED": "1",
            "RAG_SUPERVISOR_MLX_WARMUP": "0",
            "RAG_MLX_IDLE_TTL": "1800",
            "RAG_MEMORY_PRESSURE_DISABLE": "0",
            "RAG_MEMORY_PRESSURE_THRESHOLD": "75",
            "RAG_MEMORY_PRESSURE_SWAP_GB": "4.0",
            "RAG_MEMORY_PRESSURE_INTERVAL": "30",
            "RAG_STATE_SQL": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "run_at_load": True,
        "keep_alive": True,
        "throttle_s": 30,
        "exit_timeout_s": 20,
        "process_type": "Background",
        "working_dir": str(repo_root),
        "stdout_path": out,
        "stderr_path": err,
    })


def _watch_plist(rag_bin: str) -> str:
    """Persistent watchdog observing ALL registered vaults in a single process.

    `--all-vaults` (2026-04-22): prior plist invoked `rag watch` bare, which
    defaults to the active vault only — a 2-vault setup (e.g. home + work)
    left the non-active vault silently un-watched. New notes in `work`
    required a manual `rag index --vault work`. With `--all-vaults` a single
    watchdog observer monitors every registered vault in one process
    (sqlite-vec + sentence-transformers imported once, not per vault), so
    ~3-4 GB of RAM savings vs. running a second watch service.

    Memory shaping (fix 2026-05-10): el embedder MLX in-process pesaba
    ~2.7 GB pinneados forever. Sin idle-TTL ni memory pressure watchdog
    el watch contribuía al OOM/restart de la Mac. Ahora:
    - ``RAG_MLX_IDLE_TTL=600`` — 10min sin file changes evicta el embedder.
      Vault edits son sparse; primer save tras evict re-carga en ~1-2s
      (imperceptible).
    - ``RAG_MEMORY_PRESSURE_*`` — watchdog reacciona si el sistema entra
      en pressure por otro proceso (ej. web cargado).
    """
    out, err = _logs("watch")
    return _render_plist({
        "label": "com.fer.obsidian-rag-watch",
        "program_arguments": [rag_bin, "watch", "--all-vaults"],
        "env": {
            "RAG_INDEX_LOCAL_EMBED": "1",
            "RAG_MLX_IDLE_TTL": "600",
            "RAG_MEMORY_PRESSURE_DISABLE": "0",
            "RAG_MEMORY_PRESSURE_THRESHOLD": "75",
            "RAG_MEMORY_PRESSURE_SWAP_GB": "4.0",
            "RAG_MEMORY_PRESSURE_INTERVAL": "30",
            # Defer contradiction check (2026-05-10): cada nota indexada
            # disparaba el helper LLM (qwen2.5:3b, ~1.5 GB VRAM) en un
            # daemon thread del MISMO proceso watch — peak observado
            # 13.3 GB durante bursts de 4-5 memos seguidos. Con esto, watch
            # spillea el check a `~/.local/share/obsidian-rag/contradiction
            # _pending.jsonl` y el próximo `rag index` (manual o cron) lo
            # drena con el LLM ya warm para batch. Trade-off: contradicciones
            # se flaggean con delay (minutos vs segundos) — aceptable
            # porque el feedback loop principal del user es el chat, no
            # la frontmatter `contradicts:` realtime.
            "RAG_INDEX_DEFER_CONTRADICTIONS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "run_at_load": True,
        "keep_alive": True,
        "throttle_s": 30,
        "stdout_path": out,
        "stderr_path": err,
    })


def _web_plist(rag_bin: str) -> str:
    """Persistent FastAPI web UI + SSE chat endpoint on port 8765.

    Historical gap (2026-04-22): this plist was installed manually once (Apr
    19 2026) and therefore had no single source of truth for its env vars.
    The manual version was missing HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1,
    so `sentence_transformers` hit huggingface.co on every cold load; the
    2026-04-22 `web.error.log` had **64 `[local-embed] unavailable
    (couldn't connect to huggingface.co)` entries** that silently dropped
    the daemon back to backend embed (~140ms vs ~10-30ms in-process).

    It also lacked RAG_MEMORY_PRESSURE_INTERVAL, which the default of 60s
    left too coarse — `web.error.log` recorded **23 MPS Metal command-
    buffer OOMs** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) that
    happened inside the 60s sampling window. 20s gives the watchdog 3
    chances per minute to unload the chat model / reranker before the
    next MPS encoding starves.

    Note: the entry point is `python web/server.py` directly (not `rag
    web`), so env vars set via `os.environ.setdefault` at the top of rag.py
    run only after FastAPI's module graph is already partly loaded —
    anything that transitively pulls huggingface_hub before
    `from rag import ...` misses the setdefault. Setting the offline flags
    in the plist dict eliminates that race.
    """
    # Repo root computado via `_repo_root()` helper (en `_render.py`) para
    # evitar el bug histórico (2026-04-26 → exit 78) cuando un `.parent`
    # de menos generaba `…/obsidian-rag/rag/.venv/bin/python` y
    # `…/obsidian-rag/rag/web/server.py` (paths inexistentes). El venv y
    # `web/` viven en el repo root, NO dentro del package `rag/`.
    repo_root = _repo_root()
    venv_python = repo_root / ".venv" / "bin" / "python"
    web_server = repo_root / "web" / "server.py"
    working_dir = repo_root
    # Youtube key is optional — include only if present in the env to avoid
    # hard-coding a secret at setup time. Most installs leave it blank.
    youtube_key = os.environ.get("YOUTUBE_API_KEY", "")
    yt_xml = (
        f"    <key>YOUTUBE_API_KEY</key><string>{youtube_key}</string>\n"
        if youtube_key else ""
    )
    chat_model = os.environ.get("OBSIDIAN_RAG_WEB_CHAT_MODEL", "qwen2.5:7b")
    # 2026-04-28 (eval Playwright MEDIUM #12 — server crashes mid-conversation):
    # `launchctl print` mostraba `runs = 46` en 2h con un mix de SIGKILL by
    # launchd[1] + SIGTERM-then-SIGKILL escalado. Diagnóstico:
    #   1. RAG_MEMORY_PRESSURE_THRESHOLD default=85% reaccionaba TARDE — el
    #      jetsam macOS pegaba antes que el watchdog descargara modelos.
    #      Bajamos a 80% para desalojar chat model + reranker proactivamente.
    #   2. ExitTimeOut implícito de 5s (visible en `launchctl print`) cortaba
    #      el graceful shutdown de uvicorn cuando había una SSE en vuelo:
    #      launchd → SIGTERM → 5s → SIGKILL aunque la respuesta estuviera
    #      por terminar. 20s da margen al pipeline de chat (typical 8-15s
    #      end-to-end) para drenar antes que escale a SIGKILL.
    #   3. ProcessType=Interactive le indica a launchd que es un servicio
    #      foreground (web UI activamente en uso) → menos agresivo bajo
    #      memory pressure. Sin esto launchd lo trataba como Background,
    #      jetsam priority alta, primero en la cola para morir.
    # Síntoma user-facing pre-fix: "tu › qué decías sobre LangChain? error:
    # network error" + "tu › a ver si tengo otra referencia error: Failed to
    # fetch" — 2 turnos perdidos por turno de respawn. Ver
    # docs/eval-2026-04-28/playwright-conversations-bug-log.md MEDIUM #12.
    out, err = _logs("web")
    return _render_plist({
        "label": "com.fer.obsidian-rag-web",
        "program_arguments": [str(venv_python), str(web_server)],
        "env": {
            "PYTHONUNBUFFERED": "1",
            "OBSIDIAN_RAG_WEB_CHAT_MODEL": chat_model,
            "RAG_LOCAL_EMBED": "1",
            # Removido 2026-05-10 (memory game-changer): RAG_RERANKER_NEVER_UNLOAD=1
            # pineaba el bge-reranker (~2.5 GB MPS fp32) en VRAM forever, ignorando
            # el idle-TTL global. Ahora el reranker se evicta tras
            # RAG_RERANKER_IDLE_TTL=900 (15min idle) — primer query post-idle paga
            # ~1-2s extra de reload, pero idle baseline cae ~2.5 GB en uso típico
            # (user no chatea continuo > 15min). El memory_pressure_watchdog ya
            # podía forzar unload bajo presión, así que NEVER_UNLOAD funcionaba
            # solo en operación normal — donde es exactamente cuando NO se quiere
            # pinear (RAM ociosa = mejor disponible para el resto del sistema).
            "RAG_STATE_SQL": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "FASTEMBED_CACHE_PATH": f"{Path.home()}/.cache/fastembed",
            "RAG_MEMORY_PRESSURE_INTERVAL": "20",
            # 80 → 75 (memory game-changer 2026-05-10): match supervisor+watch
            # threshold post-fix 1eae161, evicción proactiva 5pp antes del jetsam.
            "RAG_MEMORY_PRESSURE_THRESHOLD": "75",
            "RAG_MEMORY_PRESSURE_SWAP_GB": "8.0",
            "RAG_AUTO_FIX_WORKER": "1",
            "RAG_AUTO_FIX_HOURLY_CAP": "12",
            "RAG_LLM_BACKEND": "mlx",
            # Removidos 2026-05-10: 4 prototypes (NLI_GROUNDING, MMR,
            # LLM_JUDGE, QUERY_DECOMPOSE) que el eval del 2026-05-09
            # rechazó (REGRESS o NO-OP, ver CLAUDE.md "Eval baselines").
            # Estaban activos en plist pese al reject → 28 restarts de
            # web por Metal GPU OOM (kIOGPUCommandBufferCallbackError-
            # OutOfMemory) por saturar unified memory con 4 modelos
            # extra simultáneos. Rollback opt-in via env override
            # explícito si alguien quiere re-evaluar uno por uno.
        },
        "extra_env_xml": yt_xml,
        "run_at_load": True,
        "keep_alive": True,
        "throttle_s": 30,
        "exit_timeout_s": 20,
        "process_type": "Interactive",
        "working_dir": str(working_dir),
        "stdout_path": out,
        "stderr_path": err,
    })
