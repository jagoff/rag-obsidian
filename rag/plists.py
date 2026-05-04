"""Launchd plist factories — extracted from rag/__init__.py 2026-05-04.

All public symbols are re-exported into the ``rag`` namespace via::

    from rag.plists import *  # in rag/__init__.py

so callers that do ``import rag; rag._watch_plist(...)`` keep working
without any change.  Callers may also import directly::

    from rag.plists import _watch_plist
"""
from __future__ import annotations

import os
from pathlib import Path

from rag._constants import _GOOGLE_TOKEN_PATH

__all__ = [
    "_LAUNCH_AGENTS_DIR", "_RAG_LOG_DIR", "_GOOGLE_TOKEN_PATH",
    "_rag_binary", "_watch_plist", "_serve_plist", "_web_plist",
    "_digest_plist", "_morning_plist", "_today_plist", "_wa_fast_plist",
    "_emergent_plist", "_patterns_plist", "_archive_plist",
    "_distill_plist",
    "_consolidate_plist", "_vault_cleanup_plist", "_anticipate_plist",
    "_maintenance_plist", "_calibration_plist", "_auto_harvest_plist",
    "_active_learning_nudge_plist", "_online_tune_plist",
    "_implicit_feedback_plist", "_ingest_whatsapp_plist",
    "_ingest_cross_source_plist", "_ingest_gmail_plist",
    "_ingest_calendar_plist", "_ingest_reminders_plist",
    "_ingest_calls_plist", "_ingest_safari_plist", "_ingest_drive_plist",
    "_ingest_pillow_plist", "_mood_poll_plist", "_routing_rules_plist",
    "_whisper_vocab_plist", "_wake_up_plist", "_serve_watchdog_plist",
    "_brief_auto_tune_plist", "_daemon_watchdog_plist", "_wake_hook_plist",
    "_services_spec", "_google_token_exists", "_calendar_creds_exist",
    "_mood_daemon_opted_in", "_DEPRECATED_LABELS", "_INSTALL_GATES",
    "_services_spec_manual",
]

_LAUNCH_AGENTS_DIR = Path.home() / "Library/LaunchAgents"
_RAG_LOG_DIR = Path.home() / ".local/share/obsidian-rag"


def _rag_binary() -> str:
    """Best-effort path to the installed `rag` binary. Default uv tool path
    first; fall back to PATH lookup. The launchd service runs without our
    interactive PATH so we resolve it once at install time.
    """
    candidates = [
        Path.home() / ".local/bin/rag",
        Path("/usr/local/bin/rag"),
        Path("/opt/homebrew/bin/rag"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    import shutil
    found = shutil.which("rag")
    return found or str(candidates[0])


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
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-watch</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>watch</string>
    <string>--all-vaults</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/watch.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/watch.error.log</string>
</dict>
</plist>
"""


def _serve_plist(rag_bin: str) -> str:
    """[DEPRECATED 2026-05-01] Persistent `rag serve` HTTP query server on port 7832.

    NO instalado por `rag setup` desde 2026-05-01 — ver doc-block en
    `_services_spec` para el rationale completo (split-brain con
    `com.fer.obsidian-rag-web` + crash-loop bajo memory pressure). La
    función queda en el módulo para retrocompat de tests
    (`test_plist_web_serve.py`, `test_setup_online_tune.py`) y para que
    el user pueda re-instalar manualmente si tiene una razón fuerte.

    This is the hot path for the WhatsApp listener (and any other bot
    integration): it keeps the reranker, bge-m3 embedder, BM25 corpus, and
    chat model warm in memory so each request skips the ~5-10s subprocess
    cold-start that listener.ts's fallback pays per message.

    Env vars mirror the web plist: OLLAMA_KEEP_ALIVE=-1 (models pinned in
    VRAM forever), RAG_RERANKER_NEVER_UNLOAD=1 (cross-encoder stays resident,
    no 9s reload after idle eviction), RAG_LOCAL_EMBED=1 (in-process
    SentenceTransformer for query embedding, ~10-30ms vs ~140ms via ollama
    HTTP), RAG_STATE_SQL=1 (deployment symmetry — no-op post-T10),
    HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 (close the race where HEAD
    probes to huggingface.co fire BEFORE rag.py's module-init setdefault —
    see test_plist_web_serve.py for rationale; was causing 64× [local-embed]
    unavailable + fallback-to-ollama in web.error.log pre-2026-04-22),
    RAG_MEMORY_PRESSURE_INTERVAL=20 (the default 60s missed the MPS-OOM
    window measured in web.error.log; 20s gives the watchdog 3 samples per
    minute to catch memory pressure before Metal returns
    `kIOGPUCommandBufferCallbackErrorOutOfMemory`).

    KeepAlive + RunAtLoad mean launchd will resurrect it if it crashes or
    the host reboots. ThrottleInterval=30 prevents crash loops from burning
    CPU.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-serve</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>serve</string>
    <string>--port</string>
    <string>7832</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>PYTHONUNBUFFERED</key><string>1</string>
    <key>OLLAMA_KEEP_ALIVE</key><string>20m</string>
    <key>OLLAMA_MAX_LOADED_MODELS</key><string>2</string>
    <key>RAG_RERANKER_NEVER_UNLOAD</key><string>1</string>
    <key>RAG_LOCAL_EMBED</key><string>1</string>
    <key>RAG_STATE_SQL</key><string>1</string>
    <key>HF_HUB_OFFLINE</key><string>1</string>
    <key>TRANSFORMERS_OFFLINE</key><string>1</string>
    <key>FASTEMBED_CACHE_PATH</key><string>{Path.home()}/.cache/fastembed</string>
    <key>RAG_MEMORY_PRESSURE_INTERVAL</key><string>20</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/serve.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/serve.error.log</string>
</dict>
</plist>
"""


def _web_plist(rag_bin: str) -> str:
    """Persistent FastAPI web UI + SSE chat endpoint on port 8765.

    Historical gap (2026-04-22): this plist was installed manually once (Apr
    19 2026) and therefore had no single source of truth for its env vars.
    The manual version was missing HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1,
    so `sentence_transformers` hit huggingface.co on every cold load; the
    2026-04-22 `web.error.log` had **64 `[local-embed] unavailable
    (couldn't connect to huggingface.co)` entries** that silently dropped
    the daemon back to ollama embed (~140ms vs ~10-30ms in-process).

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
    # Repo root is two levels up from this file (rag/plists.py); the venv
    # and web/ subtree both live at the repo root, NOT inside the `rag/`
    # package dir. Using `.parent` only (a single level) yielded
    # `…/obsidian-rag/rag/.venv/bin/python` and `…/obsidian-rag/rag/web/server.py`
    # — paths that don't exist — and left the daemon crashing with exit 78
    # until the plist was patched by hand. Bug found 2026-04-26.
    repo_root = Path(__file__).resolve().parent.parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    web_server = repo_root / "web" / "server.py"
    working_dir = repo_root
    # Youtube key is optional — include only if present in the env to avoid
    # hard-coding a secret at setup time. Most installs leave it blank.
    youtube_key = os.environ.get("YOUTUBE_API_KEY", "")
    yt_line = (
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
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-web</string>
  <key>ProgramArguments</key>
  <array>
    <string>{venv_python}</string>
    <string>{web_server}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>PYTHONUNBUFFERED</key><string>1</string>
    <key>OBSIDIAN_RAG_WEB_CHAT_MODEL</key><string>{chat_model}</string>
    <key>OLLAMA_KEEP_ALIVE</key><string>20m</string>
    <key>OLLAMA_MAX_LOADED_MODELS</key><string>2</string>
    <key>RAG_LOCAL_EMBED</key><string>1</string>
    <key>RAG_RERANKER_NEVER_UNLOAD</key><string>1</string>
    <key>RAG_STATE_SQL</key><string>1</string>
    <key>HF_HUB_OFFLINE</key><string>1</string>
    <key>TRANSFORMERS_OFFLINE</key><string>1</string>
    <key>FASTEMBED_CACHE_PATH</key><string>{Path.home()}/.cache/fastembed</string>
    <key>RAG_MEMORY_PRESSURE_INTERVAL</key><string>20</string>
    <key>RAG_MEMORY_PRESSURE_THRESHOLD</key><string>80</string>
    <key>RAG_AUTO_FIX_WORKER</key><string>1</string>
    <key>RAG_AUTO_FIX_HOURLY_CAP</key><string>12</string>
{yt_line}  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>ExitTimeOut</key><integer>20</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>WorkingDirectory</key><string>{working_dir}</string>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/web.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/web.error.log</string>
</dict>
</plist>
"""


def _digest_plist(rag_bin: str, hour: int = 22, minute: int = 0) -> str:
    """Weekly digest plist — Sunday `hour:minute` (default 22:00).

    `hour`/`minute` honor a `rag_brief_schedule_prefs` override when
    `_services_spec()` reads one (auto-tune feature, 2026-04-29).
    Defaults preserve the historical schedule when no pref exists; the
    auto-tune writer enforces the safe band before persisting, so a
    pref-driven override never lands outside `[21:00, 23:30]`.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-digest</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>digest</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>0</integer>
    <key>Hour</key><integer>{int(hour)}</integer>
    <key>Minute</key><integer>{int(minute)}</integer>
  </dict>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/digest.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/digest.error.log</string>
</dict>
</plist>
"""


def _morning_plist(rag_bin: str, hour: int = 7, minute: int = 0) -> str:
    """Morning brief plist — Mon-Fri `hour:minute` (default 07:00).

    `hour`/`minute` honor a `rag_brief_schedule_prefs` override when
    `_services_spec()` reads one (auto-tune feature, 2026-04-29).
    Defaults preserve the historical schedule when no pref exists; the
    auto-tune writer enforces the safe band before persisting, so a
    pref-driven override never lands outside `[06:30, 09:00]`.

    Voice brief opt-in (Anticipatory Phase 2.C, 2026-04-29):
    ``RAG_MORNING_VOICE`` se inyecta vacío por default — text-only
    behavior. Para activar el audio matinal: el user edita el plist
    seteando el value a ``1`` (o ``true``/``yes``) y hace
    ``launchctl bootout`` + ``bootstrap`` (o re-instala con
    ``rag setup``). Voz de la lectura: env var ``TTS_VOICE`` (default
    ``Mónica``). Cap de audio: 5MB; si excede, fallback a text-only
    silencioso. Audios viejos (>30d) limpiados por ``rag maintenance``.
    """
    h = int(hour)
    m = int(minute)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-morning</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>morning</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>RAG_EXPLORE</key><string>1</string>
    <key>RAG_MORNING_VOICE</key><string></string>
  </dict>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>{h}</integer><key>Minute</key><integer>{m}</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>{h}</integer><key>Minute</key><integer>{m}</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>{h}</integer><key>Minute</key><integer>{m}</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>{h}</integer><key>Minute</key><integer>{m}</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>{h}</integer><key>Minute</key><integer>{m}</integer></dict>
  </array>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/morning.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/morning.error.log</string>
</dict>
</plist>
"""


def _today_plist(rag_bin: str, hour: int = 22, minute: int = 0) -> str:
    """Evening "today" brief plist — Mon-Fri `hour:minute` (default 22:00).

    `hour`/`minute` honor a `rag_brief_schedule_prefs` override when
    `_services_spec()` reads one (auto-tune feature, 2026-04-29). The
    safe band for `today` is `[18:00, 21:00]`; the historical default
    (22:00) lives outside that band by design — it's the user's
    explicit baseline, only auto-tune-driven shifts have to stay inside.
    """
    h = int(hour)
    m = int(minute)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-today</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>today</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>RAG_EXPLORE</key><string>1</string>
  </dict>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>{h}</integer><key>Minute</key><integer>{m}</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>{h}</integer><key>Minute</key><integer>{m}</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>{h}</integer><key>Minute</key><integer>{m}</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>{h}</integer><key>Minute</key><integer>{m}</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>{h}</integer><key>Minute</key><integer>{m}</integer></dict>
  </array>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/today.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/today.error.log</string>
</dict>
</plist>
"""


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
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-wa-fast</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>wa-fast</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/wa-fast.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/wa-fast.error.log</string>
</dict>
</plist>
"""


# `_wa_tasks_plist` moved to `rag.integrations.whatsapp` (Phase 1b,
# 2026-04-25). Re-exported at the bottom of this file.


def _emergent_plist(rag_bin: str) -> str:
    """Proactive #2 — emergent theme detector, viernes 10am."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-emergent</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>emergent</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>5</integer>
    <key>Hour</key><integer>10</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/emergent.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/emergent.error.log</string>
</dict>
</plist>
"""


def _patterns_plist(rag_bin: str) -> str:
    """Proactive #4 — feedback pattern alert, domingo 20:00.

    Nota 2026-05-01: invoca `rag feedback-patterns` (no `rag patterns`)
    porque el comando original `patterns` quedó shadowed por el grupo
    Click `patterns` agregado en commit 887ece3 (cross-source Pearson).
    Antes del rename, este plist exiteaba con código 2 (Click muestra
    el help del grupo).
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-patterns</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>feedback-patterns</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>0</integer>
    <key>Hour</key><integer>20</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/patterns.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/patterns.error.log</string>
</dict>
</plist>
"""


def _archive_plist(rag_bin: str) -> str:
    """Proactive archiver — day 1 of each month at 23:00. Runs with --apply;
    the gate (>20 plan entries) short-circuits to a dry-run + notification
    so un-supervised drift can't accidentally move half the vault.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-archive</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>archive</string>
    <string>--apply</string>
    <string>--notify</string>
    <string>--report</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Day</key><integer>1</integer>
    <key>Hour</key><integer>23</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/archive.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/archive.error.log</string>
</dict>
</plist>
"""


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
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-distill</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>distill-conversations</string>
    <string>--apply</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>0</integer>
    <key>Hour</key><integer>22</integer>
    <key>Minute</key><integer>30</integer>
  </dict>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/distill.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/distill.error.log</string>
</dict>
</plist>
"""


def _consolidate_plist(rag_bin: str) -> str:
    """Weekly episodic-memory consolidation — Mondays 06:00 local. Promotes
    recurring conversation clusters from
    04-Archive/99-obsidian-system/99-AI/conversations/ to PARA and
    archives the originals (see plans/episodic-memory.md Phase 2)."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-consolidate</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>consolidate</string>
    <string>--apply</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>OLLAMA_KEEP_ALIVE</key><string>20m</string>
    <key>RAG_STATE_SQL</key><string>1</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>1</integer>
    <key>Hour</key><integer>6</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>RunAtLoad</key><false/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/consolidate.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/consolidate.error.log</string>
</dict>
</plist>
"""


def _vault_cleanup_plist(rag_bin: str) -> str:
    """Daily vault transient-folder cleanup — every day at 02:00.

    Mueve archivos viejos en `04-Archive/99-obsidian-system/99-AI/{{tmp,
    conversations, sessions, plans, system, reviews}}/` y wipe completo
    de `Wiki/` al `.trash/` del vault. `memory/` y `skills/` están
    explícitamente protegidos. Reversible: los archivos quedan en
    `<vault>/.trash/` hasta que el user vacíe la papelera de Obsidian.

    Schedule a las 02:00 — antes del ciclo de housekeeping del SQL
    (auto-harvest 03:00 → implicit 03:25 → online-tune 03:30 →
    maintenance 04:00 → calibrate 04:30) para no competir por I/O en
    iCloud durante esa ventana. Solo toca FS del vault, no abre
    ragvec.db, así que no hay race con el ciclo SQL.

    `RunAtLoad=false` para que `rag setup` no dispare un cleanup
    inmediato — la primera corrida es a la próxima 02:00 AM, dándole
    al user tiempo de auditar el plist + revertir si quiere.

    Lógica completa en `scripts/cleanup_vault_transient.py` — TTLs y
    políticas por carpeta documentados ahí. Para auditar qué se va a
    borrar sin tocar nada: `rag vault-cleanup --dry-run --json`.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-vault-cleanup</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>vault-cleanup</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>2</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>RunAtLoad</key><false/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/vault-cleanup.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/vault-cleanup.error.log</string>
</dict>
</plist>
"""


def _anticipate_plist(rag_bin: str) -> str:
    """Anticipatory agent — every 10 min. Evalúa señales y empuja top-1 a WA.

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
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-anticipate</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>anticipate</string>
    <string>run</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartInterval</key><integer>600</integer>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/anticipate.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/anticipate.error.log</string>
</dict>
</plist>
"""


def _maintenance_plist(rag_bin: str) -> str:
    """Daily housekeeping — every day at 04:00, after online-tune.

    Background (2026-04-21 hardening pass): with 15 services writing to
    ragvec.db concurrently (watch, serve, 4 ingesters, morning/today,
    etc.), the WAL grows unbounded between manual invocations. Observed
    in production: 126 MB WAL against a 206 MB main DB, none of the
    rotatable tables (rag_queries, rag_behavior, rag_contradictions)
    trimmed, auto_vacuum=0. Reads degrade as sqlite scans the WAL on
    each query; external backup tools that only copy `ragvec.db` miss
    126 MB of data.

    `rag maintenance` does: (1) WAL checkpoint(TRUNCATE) — compacts the
    -wal file back to KBs; (2) log-table rotation — deletes rows older
    than the configured TTL from the 6 rotatable telemetry tables;
    (3) conditional VACUUM — only if page_count*page_size exceeds
    last_vacuum_size by >500 MB (VACUUM takes an exclusive lock, so
    we gate it). See `_vec_wal_checkpoint` + `_rotate_telemetry_logs`
    in rag.py for the implementation.

    Scheduled at 04:00 specifically so online-tune (03:30) has fully
    released its SQL connections before VACUUM can acquire the
    exclusive lock. `RunAtLoad=false` — first run happens at the next
    04:00, so `rag setup` doesn't block on a potentially-long VACUUM.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-maintenance</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>maintenance</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>RAG_STATE_SQL</key><string>1</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>4</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>RunAtLoad</key><false/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/maintenance.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/maintenance.error.log</string>
</dict>
</plist>
"""


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
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-calibrate</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>calibrate</string>
    <string>--since</string>
    <string>90</string>
    <string>--as-json</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>RAG_STATE_SQL</key><string>1</string>
    <key>RAG_SCORE_CALIBRATION</key><string>1</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>4</integer>
    <key>Minute</key><integer>30</integer>
  </dict>
  <key>RunAtLoad</key><false/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/calibrate.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/calibrate.error.log</string>
</dict>
</plist>
"""


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
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-auto-harvest</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>feedback</string>
    <string>auto-harvest</string>
    <string>--since</string>
    <string>1</string>
    <string>--limit</string>
    <string>50</string>
    <string>--json</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>RAG_STATE_SQL</key><string>1</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>3</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>RunAtLoad</key><false/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/auto-harvest.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/auto-harvest.error.log</string>
</dict>
</plist>
"""


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
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-active-learning-nudge</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>active-learning</string>
    <string>nudge</string>
    <string>--json</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>RAG_STATE_SQL</key><string>1</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>1</integer>
    <key>Hour</key><integer>10</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>RunAtLoad</key><false/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/active-learning-nudge.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/active-learning-nudge.error.log</string>
</dict>
</plist>
"""


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
    working_dir = Path(__file__).resolve().parent.parent
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-online-tune</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>tune</string>
    <string>--online</string>
    <string>--days</string>
    <string>14</string>
    <string>--apply</string>
    <string>--yes</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>RAG_EVAL_GATE_TIMEOUT_S</key><string>2400</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>3</integer>
    <key>Minute</key><integer>30</integer>
  </dict>
  <key>RunAtLoad</key><false/>
  <key>KeepAlive</key><false/>
  <key>WorkingDirectory</key><string>{working_dir}</string>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/online-tune.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/online-tune.error.log</string>
</dict>
</plist>
"""


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
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-implicit-feedback</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-c</string>
    <string>{cmd_xml}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>3</integer>
    <key>Minute</key><integer>25</integer>
  </dict>
  <key>RunAtLoad</key><false/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/implicit-feedback.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/implicit-feedback.error.log</string>
</dict>
</plist>
"""


def _ingest_whatsapp_plist(rag_bin: str) -> str:
    """Cross-source: WhatsApp ingester, cada 15min.

    Incremental por design — lee `messages` con `timestamp > cursor` desde la
    bridge SQLite, chunka, upsertea. En steady state con 0 mensajes nuevos el
    run termina en <1s (solo overhead de abrir la DB). Primer run full scan
    tarda ~1min por 4000 chunks (medido 2026-04-21: 12984 msgs / 65s / 4070
    chunks). Interval 900s es lo suficientemente freq para que queries tipo
    "último mensaje de X" no se sientan stale, y lo suficientemente spaced
    para no competir con watch/serve en CPU.

    `OLLAMA_KEEP_ALIVE=-1` pin's bge-m3 en VRAM para que el embed batch no
    pague cold-load en cada run. RAG_LOCAL_EMBED NO se setea (el ingester es
    bulk-embed path → ollama HTTP es el único batching-capable backend).

    `RunAtLoad=true` (2026-04-22): garantiza run inmediato al instalar o
    post-reboot; sin esto el primer refresh se demoraba hasta 15min tras
    cargar el servicio — suficiente para que "último mensaje de X" post-
    arranque del Mac devuelva data stale. El incremental cost es chico (<1s
    cuando no hay nuevos).
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-ingest-whatsapp</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>index</string>
    <string>--source</string>
    <string>whatsapp</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>OLLAMA_KEEP_ALIVE</key><string>20m</string>
  </dict>
  <key>StartInterval</key><integer>900</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/ingest-whatsapp.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/ingest-whatsapp.error.log</string>
</dict>
</plist>
"""


def _ingest_cross_source_plist(rag_bin: str) -> str:
    """Worker unificado cross-source — every 1h. Consolidación 2026-05-04.

    Reemplaza 7 plists individuales (ingest-{gmail,calendar,reminders,
    calls,safari,drive,pillow}). El comando `rag ingest-cross-source`
    itera cada sub-ingester y lo dispara si la edad desde el último run
    supera su TTL (`_INGEST_TTL_SECONDS`):

      - gmail / calendar / reminders : 1h (hourly)
      - calls / safari / drive       : 6h
      - pillow                       : 24h (+ gate 9am)

    Cursors per-source en `~/.local/share/obsidian-rag/ingest_cursors.json`
    (JSON flat, atomic write). Gates por creds se aplican acá (no en
    el install) — skip silencioso + cursor update si el token no existe.

    WhatsApp NO está acá (cadencia 15 min, plist propio).

    RunAtLoad=true: corre inmediato al instalar / post-reboot para que
    un user nuevo vea data cross-source en la primera hora (en vez de
    esperar la próxima hora redonda).
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-ingest-cross-source</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>ingest-cross-source</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>OLLAMA_KEEP_ALIVE</key><string>20m</string>
  </dict>
  <key>StartInterval</key><integer>3600</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/ingest-cross-source.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/ingest-cross-source.error.log</string>
</dict>
</plist>
"""


def _ingest_gmail_plist(rag_bin: str) -> str:
    """Cross-source: Gmail ingester, cada 1h.

    Incremental via `historyId` cursor almacenado en `rag_gmail_state`. Llama
    a la API de Google Gmail así que respeta rate limits (default quota
    suficiente para 1 run/h en corpus típico ~50k emails). Cold run (bootstrap
    365d retention) puede tardar minutos; subsecuent runs son típicamente
    <30s con cero emails nuevos.

    Interval 3600s (1h) es conservative — Gmail API es cloud-hosted
    (user override §10.6 rompe local-first pero el tradeoff está documentado)
    y cada HTTP call cuesta quota. Si querés ingest más frecuente, bajar el
    interval manual y monitorear `rag log` para ver si golpeaste quota.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-ingest-gmail</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>index</string>
    <string>--source</string>
    <string>gmail</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>OLLAMA_KEEP_ALIVE</key><string>20m</string>
  </dict>
  <key>StartInterval</key><integer>3600</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/ingest-gmail.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/ingest-gmail.error.log</string>
</dict>
</plist>
"""


def _ingest_calendar_plist(rag_bin: str) -> str:
    """Cross-source: Google Calendar ingester, cada 1h.

    Incremental via `syncToken` cursor por calendar en `rag_calendar_state`.
    Google Calendar API (cloud-hosted — user override §10.6 rompe local-first).
    Bootstrap pulls 2y history + 180d future (§2.6 del design doc), subsequent
    runs son típicamente <10s (singleEvents=True expand RRULEs per instance, pero
    el delta típico es chico).

    Interval 3600s (1h) alineado con Gmail — ambos son Google OAuth cloud y
    los eventos de Calendar no cambian tan frecuentemente que valga la pena
    bajarlo. Requiere `~/.calendar-mcp/gcp-oauth.keys.json` + `credentials.json`
    (correr el OAuth flow manual antes del primer run); sin esos archivos el
    ingester silent-drops (loader retorna None).
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-ingest-calendar</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>index</string>
    <string>--source</string>
    <string>calendar</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>OLLAMA_KEEP_ALIVE</key><string>20m</string>
  </dict>
  <key>StartInterval</key><integer>3600</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/ingest-calendar.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/ingest-calendar.error.log</string>
</dict>
</plist>
"""


def _ingest_reminders_plist(rag_bin: str) -> str:
    """Cross-source: Apple Reminders ingester, cada 1h.

    AppleScript full-scan (~7-100s dependiendo de cuántos reminders). Incremental
    via content-hash diff post-fetch — solo re-embedea los cambiados. En steady
    state con 0 cambios el run termina en ~7s (solo el scan sin embedding).

    Interval bajado de 6h → 1h (2026-04-22): empíricamente "37 fetched · 0
    indexados · 0 borrados · 7s" — el costo es negligible y reminders es la
    fuente MÁS dinámica en el día a día del usuario (marcar como done, crear
    nuevos). Alineado con gmail/calendar (1h) así las queries tipo "qué tengo
    esta semana" + "qué reminders pendientes" devuelven data fresh.

    `RunAtLoad=true`: corre inmediatamente al instalar / post-reboot, no hay
    que esperar 1h para ver el primer refresh.

    Local-only (EventKit via osascript); no OAuth quota. Si el AppleScript
    falla (Full Disk Access denegado, Reminders.app not running), el
    ingester silent-drops y la próxima corrida lo reintenta.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-ingest-reminders</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>index</string>
    <string>--source</string>
    <string>reminders</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>OLLAMA_KEEP_ALIVE</key><string>20m</string>
  </dict>
  <key>StartInterval</key><integer>3600</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/ingest-reminders.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/ingest-reminders.error.log</string>
</dict>
</plist>
"""


def _ingest_calls_plist(rag_bin: str) -> str:
    """Ingester de Apple Calls — cada 6 horas, mantine la tabla rag_calls
    actualizada con llamadas perdidas/entrantes/salientes del CallHistory."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-ingest-calls</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>index</string>
    <string>--source</string>
    <string>calls</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <array>
    <dict>
      <key>Hour</key><integer>0</integer>
      <key>Minute</key><integer>0</integer>
    </dict>
    <dict>
      <key>Hour</key><integer>6</integer>
      <key>Minute</key><integer>0</integer>
    </dict>
    <dict>
      <key>Hour</key><integer>12</integer>
      <key>Minute</key><integer>0</integer>
    </dict>
    <dict>
      <key>Hour</key><integer>18</integer>
      <key>Minute</key><integer>0</integer>
    </dict>
  </array>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/ingest-calls.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/ingest-calls.error.log</string>
</dict>
</plist>
"""


def _ingest_safari_plist(rag_bin: str) -> str:
    """Ingester de Safari — cada 6 horas, mantine la tabla rag_safari
    actualizada con history + bookmarks."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-ingest-safari</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>index</string>
    <string>--source</string>
    <string>safari</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <array>
    <dict>
      <key>Hour</key><integer>0</integer>
      <key>Minute</key><integer>15</integer>
    </dict>
    <dict>
      <key>Hour</key><integer>6</integer>
      <key>Minute</key><integer>15</integer>
    </dict>
    <dict>
      <key>Hour</key><integer>12</integer>
      <key>Minute</key><integer>15</integer>
    </dict>
    <dict>
      <key>Hour</key><integer>18</integer>
      <key>Minute</key><integer>15</integer>
    </dict>
  </array>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/ingest-safari.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/ingest-safari.error.log</string>
</dict>
</plist>
"""


def _ingest_drive_plist(rag_bin: str) -> str:
    """Ingester de Google Drive — cada 6 horas, mantine la tabla rag_drive
    actualizada con DAO + documentos compartidos."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-ingest-drive</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>index</string>
    <string>--source</string>
    <string>drive</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <array>
    <dict>
      <key>Hour</key><integer>1</integer>
      <key>Minute</key><integer>0</integer>
    </dict>
    <dict>
      <key>Hour</key><integer>7</integer>
      <key>Minute</key><integer>0</integer>
    </dict>
    <dict>
      <key>Hour</key><integer>13</integer>
      <key>Minute</key><integer>0</integer>
    </dict>
    <dict>
      <key>Hour</key><integer>19</integer>
      <key>Minute</key><integer>0</integer>
    </dict>
  </array>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/ingest-drive.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/ingest-drive.error.log</string>
</dict>
</plist>
"""


def _ingest_pillow_plist(rag_bin: str) -> str:
    """Ingester de Pillow (iOS sleep tracker) — corre 1×/día a las 09:30
    (post wake-up típico) para cargar la noche anterior. El export vive en
    `~/Library/Mobile Documents/com~apple~CloudDocs/Sueño/PillowData.txt`,
    sincronizado por Pillow Pro vía iCloud Drive. Silent-fail si el archivo
    no existe.

    Schedule único en lugar de cada-N-horas porque el archivo solo cambia
    1×/día post wake-up; correr más veces sería desperdicio de wake en el
    Mac. Si el sync de iCloud demora, el run del día siguiente lo recoge."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-ingest-pillow</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>index</string>
    <string>--source</string>
    <string>pillow</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>9</integer>
    <key>Minute</key><integer>30</integer>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/ingest-pillow.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/ingest-pillow.error.log</string>
</dict>
</plist>
"""


def _mood_poll_plist(rag_bin: str) -> str:
    """Mood signal poller — corre `scripts/mood_poll.py` cada 30 min para
    juntar señales de Spotify + journal + WA outbound + queries +
    calendar y recomputar el score diario.

    Behind opt-in explícito: el plist tiene `RAG_MOOD_ENABLED=1` pero
    el script sale silently si el state file `~/.local/share/obsidian-
    rag/mood_enabled` no existe. Toggle:

      rag mood enable    # crea state file → daemon activo
      rag mood disable   # borra state file → daemon dormant

    Mientras el daemon está disabled, los ticks de 30min son un
    `os.path.exists` + return — costo despreciable. Si el user nunca
    opt-in, la presencia del plist no contamina nada.

    Costo en producción (daemon enabled): ~200ms por ciclo en un día
    típico (5 scorers + 1 LLM call si hay journal nota nueva), ~6
    minutos de CPU/día acumulado. El LLM call más caro (qwen2.5:3b)
    está cacheado por (path, mtime) así que solo se dispara la primera
    vez que ve una nota nueva.

    `RunAtLoad=true` para que un `rag mood enable` recién hecho dispare
    un cycle inmediato sin esperar 30min.
    """
    repo_root = Path(__file__).resolve().parent.parent
    poll_script = repo_root / "scripts" / "mood_poll.py"
    # Reuso del mismo Python del uv tool venv que usa spotify-poll.
    uv_python = Path.home() / ".local/share/uv/tools/obsidian-rag/bin/python3"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-mood-poll</string>
  <key>ProgramArguments</key>
  <array>
    <string>{uv_python}</string>
    <string>{poll_script}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>RAG_MOOD_ENABLED</key><string>1</string>
    <key>RAG_STATE_SQL</key><string>1</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartInterval</key><integer>1800</integer>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/mood-poll.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/mood-poll.error.log</string>
  <key>ThrottleInterval</key><integer>60</integer>
</dict>
</plist>
"""


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
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-routing-rules</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>routing</string>
    <string>extract-rules</string>
    <string>--auto-promote</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/routing-rules.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/routing-rules.error.log</string>
</dict>
</plist>
"""


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
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-whisper-vocab</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>whisper</string>
    <string>vocab</string>
    <string>refresh</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>3</integer>
    <key>Minute</key><integer>15</integer>
  </dict>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/whisper-vocab.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/whisper-vocab.error.log</string>
</dict>
</plist>
"""


def _wake_up_plist(rag_bin: str) -> str:
    """Wake-up pack — 04:00 diario.

    Orquesta `rag index` + `maintenance` + `patterns` + `emergent` +
    `morning` + ollama warmup en ese orden. ~15-20min end-to-end.

    Corre a las 04:00 (no 06:00) para darle tiempo a completar todo
    antes de que el user se despierte. Asume que la Mac está prendida
    overnight (plugged-in o no en sleep agresivo). Si la Mac estaba en
    sleep, launchd dispara el job al wake y puede solaparse con el
    `morning` plist de las 07:00 — rag maneja esto porque cada paso es
    idempotente (hash-skip en ETLs, reindex incremental, etc.).

    No reemplaza los plists individuales de morning/maintenance/
    patterns/emergent — los amortigua (si alguno no corrió porque la
    Mac estaba en sleep a su horario, wake-up lo re-ejecuta).
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-wake-up</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>wake-up</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>OLLAMA_KEEP_ALIVE</key><string>20m</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>4</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/wake-up.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/wake-up.error.log</string>
</dict>
</plist>
"""


def _serve_watchdog_plist(rag_bin: str) -> str:  # noqa: ARG001 (rag_bin no usado)
    """[DEPRECATED 2026-05-01] Watchdog del `rag serve` (port 7832).

    NO instalado por `rag setup` desde 2026-05-01 — ver doc-block en
    `_services_spec` para el rationale (deprecación de `rag serve`).
    La función queda para retrocompat; el script `rag-serve-watchdog.sh`
    sigue en el repo por la misma razón.

    Watchdog del web server — corre cada 60s, healthcheck HTTP +
    catchup de plists nightly que se saltearon su window por Mac dormida.

    Originalmente vivía en `scripts/com.fer.obsidian-rag-serve-watchdog.plist`
    con paths hardcoded al repo (`~/repositories/obsidian-rag/scripts/...`).
    Migrado a generación dinámica el 2026-04-27 (después del audit
    subagent 2297bb6e que detectó el riesgo: si Fer mueve el repo, el
    plist viejo apunta a un path que ya no existe).

    El script bash `rag-serve-watchdog.sh` queda en el repo (es lógica
    no-Python que se mantiene mejor como bash standalone). El plist
    apunta a su path absoluto vía `Path(__file__).resolve().parent.parent
    / "scripts" / "rag-serve-watchdog.sh"` — si el repo se mueve, el
    siguiente `rag setup` regenera el plist con el path nuevo.

    Schedule: `StartInterval=60` + `RunAtLoad=true` (corre al boot).
    Lo de catchup de nightly plists se documenta dentro del script.
    """
    repo_root = Path(__file__).resolve().parent.parent
    watchdog_script = repo_root / "scripts" / "rag-serve-watchdog.sh"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-serve-watchdog</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>{watchdog_script}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>StartInterval</key><integer>60</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/serve-watchdog.stdout.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/serve-watchdog.stderr.log</string>
</dict>
</plist>
"""


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
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-brief-auto-tune</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>brief</string>
    <string>schedule</string>
    <string>auto-tune</string>
    <string>--apply</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>RAG_STATE_SQL</key><string>1</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>0</integer>
    <key>Hour</key><integer>3</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>RunAtLoad</key><false/>
  <key>KeepAlive</key><false/>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/brief-auto-tune.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/brief-auto-tune.error.log</string>
</dict>
</plist>
"""


def _daemon_watchdog_plist(rag_bin: str) -> str:
    """Self-healing loop para el control plane de daemons launchd.

    Corre `rag daemons reconcile --apply --gentle` cada 5 minutos para
    retry-ear daemons con exit ≠ 0 + kickstart-ear overdues (turnos que
    no dispararon en su StartInterval esperado por Mac asleep / launchd
    backoff / external restarts).

    `--gentle` NO regenera plists ni bootea huérfanos — solo re-intenta
    los ya registrados. Para cambios de infraestructura profundos, el user
    corre `rag setup` de forma interactiva.

    `RunAtLoad=true` + `StartInterval=300` produce un primer tick inmediato
    cuando el plist se bootstrappa (útil post-reboot del Mac para hacer
    catchup de lo que se perdió durante shutdown), después cada 5 min.

    `Throttle=60` evita ráfagas si el comando termina muy rápido —
    mínimo 60s entre runs.

    Reemplaza el catchup post-sleep que tenía el difunto `serve-watchdog`
    (deprecado 2026-05-01), pero ahora para TODO el stack de daemons
    en lugar de solo `serve`.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-daemon-watchdog</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>daemons</string>
    <string>reconcile</string>
    <string>--apply</string>
    <string>--gentle</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>RAG_STATE_SQL</key><string>1</string>
  </dict>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><true/>
  <key>Throttle</key><integer>60</integer>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/daemon-watchdog.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/daemon-watchdog.error.log</string>
</dict>
</plist>
"""


def _wake_hook_plist(rag_bin: str) -> str:
    """Wake hook — kickstart-overdue post-Mac-wake.

    launchd `StartCalendarInterval` no dispara retroactivamente cuando el
    Mac estuvo dormido a la hora del slot. El watchdog cada 5min mitiga,
    pero hay hasta 5min de lag post-wake. Este script corre como daemon
    long-running (`KeepAlive=true`) en un loop sleep 60s, detecta wakes
    user-visible vía `pmset -g log | grep "Display is turned on"` y
    dispara `rag daemons kickstart-overdue` cuando hay un wake nuevo
    desde el último check.

    State persistido en `~/.local/share/obsidian-rag/wake-hook-state.json`
    (`{last_wake, last_check_iso, last_kickstart_ok}`). El primer tick
    post-install hace bootstrap (anchorea al último wake actual sin
    disparar) — evita kickstart spurious al instalar.

    KeepAlive=true + el sleep interno de 60s es lo más cerca de un
    "power event hook" nativo que se puede hacer sin pyobjc / IOKit.
    Costo: ~1 proceso Python idle (~10 MB RSS) + un fork de pmset cada
    60s (~50ms).

    `_rag_bin` es el path absoluto al CLI (mismo que ProgramArguments
    de los otros daemons).
    """
    script_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "wake_hook.py"
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-wake-hook</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/env</string>
    <string>python3</string>
    <string>{script_path}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>RAG_WAKE_HOOK_RAG_BIN</key><string>{rag_bin}</string>
    <key>RAG_WAKE_HOOK_POLL_SECONDS</key><string>60</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/wake-hook.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/wake-hook.error.log</string>
</dict>
</plist>
"""


def _services_spec(rag_bin: str) -> list[tuple[str, str, str]]:
    """Return [(label, plist_filename, plist_xml), ...].

    Brief schedule overrides (2026-04-29): morning/today/digest plists
    consult `rag_brief_schedule_prefs` via `rag.brief_schedule.
    get_brief_schedule_pref()` BEFORE generating their XML. If a row
    exists, the override `(hour, minute)` is substituted; otherwise the
    historical defaults from the plist functions kick in. The lookup
    is silent-fail on any SQL error (fresh DB, locked file, etc.) so
    `rag setup` on a brand-new install never blocks on telemetry.
    """
    from rag.integrations.whatsapp import _wa_tasks_plist  # noqa: PLC0415
    # Lazy import keeps module-load lightweight + avoids a circular
    # import (brief_schedule lazy-imports back into rag for SQL conn).
    try:
        from rag.brief_schedule import get_brief_schedule_pref
    except Exception:
        def get_brief_schedule_pref(_kind):  # type: ignore
            return None

    def _override(kind: str, default_hour: int, default_minute: int) -> tuple[int, int]:
        try:
            pref = get_brief_schedule_pref(kind)
        except Exception:
            pref = None
        if pref is None:
            return (default_hour, default_minute)
        return (int(pref.get("hour", default_hour)), int(pref.get("minute", default_minute)))

    morning_h, morning_m = _override("morning", 7, 0)
    today_h, today_m = _override("today", 22, 0)
    digest_h, digest_m = _override("digest", 22, 0)

    return [
        ("com.fer.obsidian-rag-watch", "com.fer.obsidian-rag-watch.plist",
         _watch_plist(rag_bin)),
        # ── DEPRECATED 2026-05-01: `com.fer.obsidian-rag-serve` removido ─────
        # Histórico: `rag serve --port 7832` corría un BaseHTTPServer simple
        # como hot-path para el WhatsApp listener. Coexistía con
        # `com.fer.obsidian-rag-web` (FastAPI port 8765) — los dos plists
        # peleaban por VRAM (cada uno carga qwen2.5:7b chat + qwen2.5:3b
        # lookup + bge-m3 embedder + bge-reranker-v2-m3) bajo memory
        # pressure. Resultado observado el 2026-05-01: el `rag serve` se
        # colgaba en el `embed(["warmup"])` con `httpx.RemoteProtocolError`
        # mientras el FastAPI consumía el slot de ollama, crash-loopeaba via
        # `KeepAlive=true`, y degradaba al `BaseHTTPServer` que sólo
        # responde `/health` (404 a `/api/*`).
        #
        # Decisión: el FastAPI (`com.fer.obsidian-rag-web`) cubre todos los
        # endpoints reales (chat web + frontend + cloudflare tunnel). El
        # WhatsApp listener tiene fallback a subprocess (`rag query`,
        # ~5-10s cold start por mensaje) cuando :7832 no responde — flow
        # confirmado en `whatsapp-listener/listener.ts:1652` ("Server down
        # — fall through to subprocess"). El cost extra es aceptable
        # mientras eliminamos el split-brain de servers + el loop crash.
        #
        # Re-activación: si en el futuro hay razón fuerte para tener un
        # endpoint sync JSON dedicado (ej. otro bot con SLA <2s/query), la
        # opción más limpia es agregar un `POST /api/query` al FastAPI con
        # el mismo wire-format que el legacy `/query` del rag serve, y
        # apuntar el listener a `:8765` con `RAG_SERVE_URL=http://127.0.0.1:8765`.
        # Mientras tanto, `_serve_plist` + `_serve_watchdog_plist` siguen
        # como funciones (los tests `test_plist_web_serve.py` validan su
        # shape) pero NO se instalan por `rag setup`.
        # ────────────────────────────────────────────────────────────────────
        # Web UI daemon — previously installed manually outside `rag setup`
        # (Apr 2026), which left HF_HUB_OFFLINE + RAG_MEMORY_PRESSURE_INTERVAL
        # missing in the actual plist and produced 64× [local-embed] falls +
        # 23× MPS Metal OOMs. Now generated from source.
        ("com.fer.obsidian-rag-web", "com.fer.obsidian-rag-web.plist",
         _web_plist(rag_bin)),
        ("com.fer.obsidian-rag-digest", "com.fer.obsidian-rag-digest.plist",
         _digest_plist(rag_bin, hour=digest_h, minute=digest_m)),
        ("com.fer.obsidian-rag-morning", "com.fer.obsidian-rag-morning.plist",
         _morning_plist(rag_bin, hour=morning_h, minute=morning_m)),
        ("com.fer.obsidian-rag-today", "com.fer.obsidian-rag-today.plist",
         _today_plist(rag_bin, hour=today_h, minute=today_m)),
        ("com.fer.obsidian-rag-wake-up", "com.fer.obsidian-rag-wake-up.plist",
         _wake_up_plist(rag_bin)),
        ("com.fer.obsidian-rag-emergent", "com.fer.obsidian-rag-emergent.plist",
         _emergent_plist(rag_bin)),
        ("com.fer.obsidian-rag-patterns", "com.fer.obsidian-rag-patterns.plist",
         _patterns_plist(rag_bin)),
        ("com.fer.obsidian-rag-archive", "com.fer.obsidian-rag-archive.plist",
         _archive_plist(rag_bin)),
        # Weekly conversation distiller — rescata bot answers de
        # conversations con sources missing antes de que el conocimiento
        # quede sólo en logs no-indexados. Defense-in-depth con
        # promote-on-cite del archive (commit `e89c42f`, 2026-05-04).
        ("com.fer.obsidian-rag-distill", "com.fer.obsidian-rag-distill.plist",
         _distill_plist(rag_bin)),
        ("com.fer.obsidian-rag-wa-tasks", "com.fer.obsidian-rag-wa-tasks.plist",
         _wa_tasks_plist(rag_bin)),
        # WA workers time-sensitive (5 min) — worker unificado que corre
        # `remind-wa` + `wa-scheduled-send` en serie. Consolidación
        # 2026-05-04: antes eran 2 plists separados (mismo cron, mismo
        # budget), ahora 1. Ver `_wa_fast_plist` docstring para rationale.
        ("com.fer.obsidian-rag-wa-fast",
         "com.fer.obsidian-rag-wa-fast.plist",
         _wa_fast_plist(rag_bin)),
        ("com.fer.obsidian-rag-auto-harvest", "com.fer.obsidian-rag-auto-harvest.plist",
         _auto_harvest_plist(rag_bin)),
        # Active-learning nudge (C.6, 2026-04-29) — Lunes 10am: cuenta
        # queries low-conf ultimos 7d sin labels y manda push al RagNet
        # con link a /learning si supera 20 candidates. Reemplazo del
        # bash inline que disparaba osascript notification.
        ("com.fer.obsidian-rag-active-learning-nudge",
         "com.fer.obsidian-rag-active-learning-nudge.plist",
         _active_learning_nudge_plist(rag_bin)),
        # Implicit corrective_path inference — corre 03:25, 5 min antes
        # del online-tune. Pre-popula corrective_paths desde behavior
        # post-👎 sin pedir input al user. Sprint 1 del cierre del loop
        # de auto-aprendizaje (2026-04-26).
        ("com.fer.obsidian-rag-implicit-feedback",
         "com.fer.obsidian-rag-implicit-feedback.plist",
         _implicit_feedback_plist(rag_bin)),
        ("com.fer.obsidian-rag-online-tune", "com.fer.obsidian-rag-online-tune.plist",
         _online_tune_plist(rag_bin)),
        ("com.fer.obsidian-rag-calibrate", "com.fer.obsidian-rag-calibrate.plist",
         _calibration_plist(rag_bin)),
        ("com.fer.obsidian-rag-maintenance", "com.fer.obsidian-rag-maintenance.plist",
         _maintenance_plist(rag_bin)),
        ("com.fer.obsidian-rag-consolidate", "com.fer.obsidian-rag-consolidate.plist",
         _consolidate_plist(rag_bin)),
        # Daily vault transient cleanup (2026-04-27) — barre carpetas de
        # "sistema" bajo 04-Archive/99-obsidian-system/99-AI/ con TTLs por
        # carpeta y mueve archivos viejos a `.trash/` del vault. Whitelist:
        # `memory/` + `skills/`. Lógica en scripts/cleanup_vault_transient.py.
        ("com.fer.obsidian-rag-vault-cleanup",
         "com.fer.obsidian-rag-vault-cleanup.plist",
         _vault_cleanup_plist(rag_bin)),
        # Anticipatory agent (2026-04-24) — game-changer push proactivo cada
        # 10 min. Comparte daily_cap=3 con emergent/patterns. Silenciable
        # per-kind: `rag silence anticipate-calendar`. Kill global:
        # RAG_ANTICIPATE_DISABLED=1.
        ("com.fer.obsidian-rag-anticipate", "com.fer.obsidian-rag-anticipate.plist",
         _anticipate_plist(rag_bin)),
        # Cross-source ingesters (consolidación 2026-05-04) — antes eran 8
        # plists separados; ahora 2:
        #   - `ingest-whatsapp` queda aparte (cadencia 15 min, hot path).
        #   - `ingest-cross-source` (1h) wrappea gmail/calendar/reminders/
        #     calls/safari/drive/pillow con TTL per-source + gates por
        #     credencial. Los sub-ingesters siguen invocables standalone
        #     via `rag index --source X` para debug / manual refresh.
        ("com.fer.obsidian-rag-ingest-whatsapp",
         "com.fer.obsidian-rag-ingest-whatsapp.plist",
         _ingest_whatsapp_plist(rag_bin)),
        ("com.fer.obsidian-rag-ingest-cross-source",
         "com.fer.obsidian-rag-ingest-cross-source.plist",
         _ingest_cross_source_plist(rag_bin)),
        # Mood signal poller (2026-04-30) — cada 30min junta señales de
        # Spotify + journal + WA outbound + queries + calendar y
        # recomputa el score diario. Behind opt-in: el plist se carga
        # siempre pero el script exit-early si el state file
        # ~/.local/share/obsidian-rag/mood_enabled no existe.
        # Toggle con `rag mood enable` / `rag mood disable`.
        ("com.fer.obsidian-rag-mood-poll",
         "com.fer.obsidian-rag-mood-poll.plist",
         _mood_poll_plist(rag_bin)),
        # Routing rules detector (2026-04-30) — daemon long-running
        # que scanea patrones de comportamiento y sugiere new routes.
        ("com.fer.obsidian-rag-routing-rules",
         "com.fer.obsidian-rag-routing-rules.plist",
         _routing_rules_plist(rag_bin)),
        # Whisper vocabulary refresh (2026-04-30) — nightly extractor
        # para mejorar transcripción de audios WhatsApp.
        ("com.fer.obsidian-rag-whisper-vocab",
         "com.fer.obsidian-rag-whisper-vocab.plist",
         _whisper_vocab_plist(rag_bin)),
        # ── DEPRECATED 2026-05-01: `com.fer.obsidian-rag-serve-watchdog` ──
        # Servía para healthcheck del `rag serve` (port 7832), removido
        # cuando bajamos `com.fer.obsidian-rag-serve`. Ver doc-block
        # extenso arriba donde se quita el entry de `serve` para el
        # rationale completo. Si re-activás el `serve`, también re-agregá
        # este watchdog.
        # ────────────────────────────────────────────────────────────────
        # Brief schedule auto-tune (2026-04-29) — Sunday 03:00 reads
        # rag_brief_feedback last 30d, decides whether to shift any of
        # morning/today/digest plists' StartCalendarInterval forward
        # (within safe bands), and re-bootstraps only the affected kind.
        # Silent-fail end-to-end — no UX disruption from a stuck cron.
        ("com.fer.obsidian-rag-brief-auto-tune",
         "com.fer.obsidian-rag-brief-auto-tune.plist",
         _brief_auto_tune_plist(rag_bin)),
        # Daemon watchdog — T4 del control plane (2026-05-01). Self-healing
        # loop que corre `rag daemons reconcile --apply --gentle` cada 5min
        # para retry-ear daemons fallidos + kickstart-ear overdues sin tocar
        # el registro de plists. RunAtLoad=true + StartInterval=300 → primer
        # tick inmediato post-bootstrap, después cada 5min. Reemplaza el
        # catchup del difunto serve-watchdog, ahora para TODO el stack.
        ("com.fer.obsidian-rag-daemon-watchdog",
         "com.fer.obsidian-rag-daemon-watchdog.plist",
         _daemon_watchdog_plist(rag_bin)),
        # 2026-05-01: wake hook — sidecar Python loop con KeepAlive=true
        # que polea pmset cada 60s y dispara `rag daemons kickstart-overdue`
        # cuando detecta un wake user-visible nuevo. Resuelve el lag post-
        # Mac-wake del watchdog StartInterval=300 (era hasta 5min). Costo:
        # ~10 MB RSS idle + un fork de pmset cada 60s (~50ms).
        ("com.fer.obsidian-rag-wake-hook",
         "com.fer.obsidian-rag-wake-hook.plist",
         _wake_hook_plist(rag_bin)),
    ]


# ── Install gates: pre-requisitos por label ─────────────────────────────────
#
# Cada entry es `(check_fn, hint)`. Si `check_fn()` devuelve False, `rag setup`
# NO instala el plist y muestra `hint` para que el user sepa cómo activar.
# Re-correr `rag setup` después del pre-req instala el plist.
#
# Por qué este gate existe: sin él, un plist se carga y falla cada cadencia
# cuando falta la credencial / opt-in. Polución de logs + `daemons status`
# con rows en rojo que el user no mira hasta que rompe algo más visible.
# Aprendido el 2026-05-04 del daemon `ingest-gmail` en exit-loop hace 4 días
# sin alerta, + `mood-poll` cargado sin opt-in consumiendo nothing visible.
def _google_token_exists() -> bool:
    """Gmail + Drive comparten token — un OAuth flow cubre ambos."""
    return _GOOGLE_TOKEN_PATH.is_file()


def _calendar_creds_exist() -> bool:
    return (Path.home() / ".calendar-mcp" / "credentials.json").is_file()


def _mood_daemon_opted_in() -> bool:
    """True si el user hizo `rag mood enable` (crea el state file)."""
    return (Path.home() / ".local/share/obsidian-rag/mood_enabled").is_file()


# ── Labels deprecated ──────────────────────────────────────────────────────
#
# Plists que EXISTIERON en `_services_spec` y fueron consolidados/removidos.
# `rag setup` los bootouts + borra de disco para cerrar el gap de migración
# (sino el plist viejo sigue cargado y corriendo en paralelo con su reemplazo,
# doble notificación / doble envío / race conditions).
#
# Una vez que todos los usuarios hayan corrido `rag setup` al menos una vez
# post-consolidación, el entry correspondiente se puede remover de acá —
# pero mantenerlo no cuesta nada (es un set chico) y protege contra rollback
# parcial (user revertió a versión vieja y después volvió a la nueva).
_DEPRECATED_LABELS: frozenset[str] = frozenset({
    # Consolidados en `wa-fast` el 2026-05-04:
    "com.fer.obsidian-rag-reminder-wa-push",
    "com.fer.obsidian-rag-wa-scheduled-send",
    # Consolidados en `ingest-cross-source` el 2026-05-04:
    "com.fer.obsidian-rag-ingest-gmail",
    "com.fer.obsidian-rag-ingest-calendar",
    "com.fer.obsidian-rag-ingest-reminders",
    "com.fer.obsidian-rag-ingest-calls",
    "com.fer.obsidian-rag-ingest-safari",
    "com.fer.obsidian-rag-ingest-drive",
    "com.fer.obsidian-rag-ingest-pillow",
})


# Post-consolidación 2026-05-04: los gates de gmail/calendar/drive se
# movieron DENTRO del comando `ingest-cross-source` (skip silencioso +
# cursor update por-source si la creds no existe). Este dict queda sólo
# para labels que aún viven como plist individual en `_services_spec`.
_INSTALL_GATES: dict[str, tuple] = {
    "com.fer.obsidian-rag-mood-poll": (
        _mood_daemon_opted_in,
        "mood-poll es opt-in — activar con [cyan]rag mood enable[/cyan] "
        "+ [cyan]rag setup[/cyan]",
    ),
}


def _services_spec_manual() -> list[dict]:
    """Daemons launchd que existen en disco pero NO tienen factory en código.
    Instalados a mano por el usuario. `rag setup` no los toca; el control
    plane (`rag daemons status / reconcile`) los monitorea pero no los
    regenera. Si alguno se rompe, el fix es manual (re-copiar plist desde
    backup o regenerarlo en su repo origen).

    Limpieza 2026-05-04: se removieron 4 entries que llevaban meses como
    "fantasmas" — sin plist en disco, sin log, con tick `-` en cada
    `daemons status`:

      - `cloudflare-tunnel` + `cloudflare-tunnel-watcher`: se instalan
        aparte cuando el user decide exponer web vía `cloudflared`. Si
        están corriendo, aparecen en `daemons status` por launchctl;
        no hace falta el registry para eso.
      - `lgbm-train`, `paraphrases-train`: jobs de fine-tuning que se
        corren a mano (`rag tune …`), no tienen plist automatizado.
        Dejarlos en el registry les asignaba un slot en el dashboard
        que solo decía "missing".

    Los 3 restantes SÍ tienen histórico de ejecución (logs en
    `~/.local/share/obsidian-rag/{log-rotate,spotify-poll,synth-refresh}.log`
    de 2026-04/05) — se mantienen mientras el user decida si los usa
    o los archiva.
    """
    return [
        {"label": "com.fer.obsidian-rag-synth-refresh", "category": "manual_keep"},
        {"label": "com.fer.obsidian-rag-spotify-poll", "category": "manual_keep"},
        {"label": "com.fer.obsidian-rag-log-rotate", "category": "manual_keep"},
    ]


