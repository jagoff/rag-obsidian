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
    - ``ProcessType=Adaptive`` — NO Background. Es supervisor de Background
      workers. Adaptive le da prioridad estándar de scheduling pero no se
      lo trata como ``Interactive`` (no es UI foreground del user).
    - ``RAG_RERANKER_NEVER_UNLOAD=1`` — el reranker queda pinneado para que
      jobs proactive lo reusen entre invocaciones (anti pattern del web).
    - ``RAG_MLX_IDLE_TTL=7200`` — 2h. Más alto que el default (1800s) porque
      supervisor vive eternamente y los modelos cargados se amortizan en
      muchos jobs intra-día. El watchdog memory-pressure los unlodea si
      hace falta.
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
            "RAG_RERANKER_NEVER_UNLOAD": "1",
            "RAG_MLX_IDLE_TTL": "7200",
            "RAG_STATE_SQL": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "run_at_load": True,
        "keep_alive": True,
        "throttle_s": 30,
        "exit_timeout_s": 20,
        "process_type": "Adaptive",
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
    """
    out, err = _logs("watch")
    return _render_plist({
        "label": "com.fer.obsidian-rag-watch",
        "program_arguments": [rag_bin, "watch", "--all-vaults"],
        "env": {
            "RAG_INDEX_LOCAL_EMBED": "1",
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
            "RAG_RERANKER_NEVER_UNLOAD": "1",
            "RAG_STATE_SQL": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "FASTEMBED_CACHE_PATH": f"{Path.home()}/.cache/fastembed",
            "RAG_MEMORY_PRESSURE_INTERVAL": "20",
            "RAG_MEMORY_PRESSURE_THRESHOLD": "80",
            "RAG_MEMORY_PRESSURE_SWAP_GB": "8.0",
            "RAG_AUTO_FIX_WORKER": "1",
            "RAG_AUTO_FIX_HOURLY_CAP": "12",
            "RAG_LLM_BACKEND": "mlx",
            # Tier S quality flags (2026-05-09): activan features que
            # estaban en código pero default OFF. NLI grounding verifica
            # claims post-citation (menos hallucinations). MMR diversifica
            # top-k. LLM_JUDGE rescata queries borderline (top<0.5).
            # QUERY_DECOMPOSE rompe queries multi-aspect en sub-retrieves
            # con RRF merge — específicamente sube MRR de chains.
            # Rollback: setear var a "0" o quitarla del plist.
            "RAG_NLI_GROUNDING": "1",
            "RAG_MMR": "1",
            "RAG_LLM_JUDGE": "1",
            "RAG_QUERY_DECOMPOSE": "1",
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
