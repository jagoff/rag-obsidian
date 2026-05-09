"""TEMP home para factories aún no migradas a su sub-módulo de dominio.

Estado post-commit-3 (2026-05-09): 5 factories viven acá esperando
el commit 4 que las va a partir en `poll.py` + `control.py`.

Una vez que TODAS las factories estén en su sub-módulo final,
este archivo se borra (commit 4 del plan).

Mapping pendiente:
  poll.py         → _mood_poll, _spotify_poll
  control.py      → _wake_up, _daemon_watchdog, _wake_hook
"""
from __future__ import annotations

from rag.plists._render import _logs, _render_plist, _repo_root

__all__ = [
    "_daemon_watchdog_plist",
    "_mood_poll_plist",
    "_spotify_poll_plist",
    "_wake_hook_plist",
    "_wake_up_plist",
]


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
