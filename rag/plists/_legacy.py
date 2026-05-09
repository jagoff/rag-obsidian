"""TEMP home para factories aún no migradas a su sub-módulo de dominio.

Estado post-commit-2 (2026-05-09): 11 factories viven acá esperando
los commits 3-4 que las van a partir en sub-módulos por dominio
(`maintenance.py`, `ingest.py`, `wa.py`, `poll.py`, `control.py`).

Una vez que TODAS las factories estén en su sub-módulo final,
este archivo se borra (commit 4 del plan).

Mapping pendiente:
  maintenance.py  → _maintenance, _vault_cleanup, _consolidate
  ingest.py       → _ingest_whatsapp, _ingest_cross_source
  wa.py           → _wa_fast
  poll.py         → _mood_poll, _spotify_poll
  control.py      → _wake_up, _daemon_watchdog, _wake_hook
"""
from __future__ import annotations

from rag.plists._render import _logs, _render_plist, _repo_root

__all__ = [
    "_consolidate_plist",
    "_daemon_watchdog_plist",
    "_ingest_cross_source_plist",
    "_ingest_whatsapp_plist",
    "_maintenance_plist",
    "_mood_poll_plist",
    "_spotify_poll_plist",
    "_vault_cleanup_plist",
    "_wa_fast_plist",
    "_wake_hook_plist",
    "_wake_up_plist",
]


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
    out, err = _logs("wa-fast")
    return _render_plist({
        "label": "com.fer.obsidian-rag-wa-fast",
        "program_arguments": [rag_bin, "wa-fast"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
        },
        "schedule": {"interval_s": 300},
        "run_at_load": False,
        "stdout_path": out,
        "stderr_path": err,
    })


def _consolidate_plist(rag_bin: str) -> str:
    """Weekly episodic-memory consolidation — Mondays 06:00 local. Promotes
    recurring conversation clusters from
    99-obsidian/99-AI/conversations/ to PARA and
    archives the originals (see plans/episodic-memory.md Phase 2)."""
    out, err = _logs("consolidate")
    return _render_plist({
        "label": "com.fer.obsidian-rag-consolidate",
        "program_arguments": [rag_bin, "consolidate", "--apply"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_LLM_BACKEND": "mlx",
        },
        "schedule": {
            "calendar": {"Weekday": 1, "Hour": 6, "Minute": 0},
        },
        "run_at_load": False,
        "keep_alive": False,
        "stdout_path": out,
        "stderr_path": err,
    })


def _vault_cleanup_plist(rag_bin: str) -> str:
    """Daily vault transient-folder cleanup — every day at 02:00.

    Mueve archivos viejos en `99-obsidian/99-AI/{{tmp,
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
    out, err = _logs("vault-cleanup")
    return _render_plist({
        "label": "com.fer.obsidian-rag-vault-cleanup",
        "program_arguments": [rag_bin, "vault-cleanup"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
        },
        "schedule": {
            "calendar": {"Hour": 2, "Minute": 0},
        },
        "run_at_load": False,
        "keep_alive": False,
        "stdout_path": out,
        "stderr_path": err,
    })


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
    out, err = _logs("maintenance")
    return _render_plist({
        "label": "com.fer.obsidian-rag-maintenance",
        "program_arguments": [rag_bin, "maintenance"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
        },
        "schedule": {
            "calendar": {"Hour": 4, "Minute": 0},
        },
        "run_at_load": False,
        "keep_alive": False,
        "stdout_path": out,
        "stderr_path": err,
    })


def _ingest_whatsapp_plist(rag_bin: str) -> str:
    """Cross-source: WhatsApp ingester, cada 15min.

    Incremental por design — lee `messages` con `timestamp > cursor` desde la
    bridge SQLite, chunka, upsertea. En steady state con 0 mensajes nuevos el
    run termina en <1s (solo overhead de abrir la DB). Primer run full scan
    tarda ~1min por 4000 chunks (medido 2026-04-21: 12984 msgs / 65s / 4070
    chunks). Interval 900s es lo suficientemente freq para que queries tipo
    "último mensaje de X" no se sientan stale, y lo suficientemente spaced
    para no competir con watch/serve en CPU.

    `RAG_INDEX_LOCAL_EMBED=1` (Ola 6, 2026-05-06 cero-Ollama): embedder
    in-process siempre activo. Reemplaza el bulk-embed path via Ollama HTTP
    que requería `LLM_KEEP_ALIVE=-1` para pin bge-m3 en VRAM.

    `RunAtLoad=true` (2026-04-22): garantiza run inmediato al instalar o
    post-reboot; sin esto el primer refresh se demoraba hasta 15min tras
    cargar el servicio — suficiente para que "último mensaje de X" post-
    arranque del Mac devuelva data stale. El incremental cost es chico (<1s
    cuando no hay nuevos).
    """
    out, err = _logs("ingest-whatsapp")
    return _render_plist({
        "label": "com.fer.obsidian-rag-ingest-whatsapp",
        "program_arguments": [rag_bin, "index", "--source", "whatsapp"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_INDEX_LOCAL_EMBED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "schedule": {"interval_s": 900},
        "run_at_load": True,
        "stdout_path": out,
        "stderr_path": err,
    })


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
    out, err = _logs("ingest-cross-source")
    return _render_plist({
        "label": "com.fer.obsidian-rag-ingest-cross-source",
        "program_arguments": [rag_bin, "ingest-cross-source"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_INDEX_LOCAL_EMBED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "schedule": {"interval_s": 3600},
        "run_at_load": True,
        "stdout_path": out,
        "stderr_path": err,
    })


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
    from pathlib import Path
    repo_root = _repo_root()
    poll_script = repo_root / "scripts" / "mood_poll.py"
    # Reuso del mismo Python del uv tool venv que usa spotify-poll.
    uv_python = Path.home() / ".local/share/uv/tools/obsidian-rag/bin/python3"
    out, err = _logs("mood-poll")
    return _render_plist({
        "label": "com.fer.obsidian-rag-mood-poll",
        "program_arguments": [str(uv_python), str(poll_script)],
        "env": {
            "RAG_MOOD_ENABLED": "1",
            "RAG_STATE_SQL": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
        },
        "schedule": {"interval_s": 1800},
        "run_at_load": True,
        "keep_alive": False,
        "throttle_s": 60,
        "throttle_after_logs": True,  # único caso con ThrottleInterval después de logs
        "stdout_path": out,
        "stderr_path": err,
    })


def _spotify_poll_plist(rag_bin: str) -> str:
    """Spotify poller — corre `scripts/spotify_poll.py` cada 60s para
    grabar el track actualmente en reproducción en `rag_spotify_log`.

    Lógica: script llama `record_now_playing()` desde rag.integrations.
    Comportamiento esperado:
      - Si Spotify está cerrado o paused → sale silently (exit 0, no log)
      - Si hay un track en reproducción → graba a DB + stdout JSON

    No hay opt-in — siempre activo si el plist está cargado. Los datos
    se usan para context en briefs ("escuchabas X ayer") y futuro mood
    scoring.

    `RunAtLoad=true` para que bootstrap lance inmediatamente sin esperar
    60s al primer tick.
    """
    from pathlib import Path
    repo_root = _repo_root()
    poll_script = repo_root / "scripts" / "spotify_poll.py"
    uv_python = Path.home() / ".local/share/uv/tools/obsidian-rag/bin/python3"
    out, err = _logs("spotify-poll")
    return _render_plist({
        "label": "com.fer.obsidian-rag-spotify-poll",
        "program_arguments": [str(uv_python), str(poll_script)],
        "env": {
            "RAG_STATE_SQL": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
        },
        "schedule": {"interval_s": 60},
        "run_at_load": True,
        "stdout_path": out,
        "stderr_path": err,
    })


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
    out, err = _logs("wake-up")
    return _render_plist({
        "label": "com.fer.obsidian-rag-wake-up",
        "program_arguments": [rag_bin, "wake-up"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_LLM_BACKEND": "mlx",
        },
        "schedule": {
            "calendar": {"Hour": 4, "Minute": 0},
        },
        "stdout_path": out,
        "stderr_path": err,
    })


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
    out, err = _logs("daemon-watchdog")
    return _render_plist({
        "label": "com.fer.obsidian-rag-daemon-watchdog",
        "program_arguments": [
            rag_bin, "daemons", "reconcile", "--apply", "--gentle",
        ],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
        },
        "schedule": {"interval_s": 300},
        "run_at_load": True,
        "throttle_s": 60,
        "throttle_key": "Throttle",  # único caso que usa la key corta
        "stdout_path": out,
        "stderr_path": err,
    })


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
    script_path = _repo_root() / "scripts" / "wake_hook.py"
    out, err = _logs("wake-hook")
    return _render_plist({
        "label": "com.fer.obsidian-rag-wake-hook",
        "program_arguments": ["/usr/bin/env", "python3", str(script_path)],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_WAKE_HOOK_RAG_BIN": rag_bin,
            "RAG_WAKE_HOOK_POLL_SECONDS": "60",
        },
        "run_at_load": True,
        "keep_alive": True,
        "throttle_s": 30,
        "stdout_path": out,
        "stderr_path": err,
    })
