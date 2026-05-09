"""External signal poller factories: mood-poll + spotify-poll.

Pollers de signal externa que NO usan el `rag` binary — corren con el
Python del uv-tool venv directamente sobre `scripts/{mood,spotify}_poll.py`.
mood-poll cada 30min (opt-in via `rag mood enable`); spotify-poll cada
60s (siempre activo).

Migrado de rag/plists/_legacy.py en Phase 3 commit 4 (2026-05-09).
"""
from __future__ import annotations

from pathlib import Path

from rag.plists._render import _logs, _render_plist, _repo_root

__all__ = [
    "_mood_poll_plist",
    "_spotify_poll_plist",
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
        "process_type": "Background",
        "stdout_path": out,
        "stderr_path": err,
    })


def _spotify_poll_plist(rag_bin: str) -> str:
    """Spotify poller — corre `scripts/spotify_poll.py` cada 5min para
    grabar el track actualmente en reproducción en `rag_spotify_log`.

    Lógica: script llama `record_now_playing()` desde rag.integrations.
    Comportamiento esperado:
      - Si Spotify está cerrado o paused → sale silently (exit 0, no log)
      - Si hay un track en reproducción → graba a DB + stdout JSON

    No hay opt-in — siempre activo si el plist está cargado. Los datos
    se usan para context en briefs ("escuchabas X ayer") y futuro mood
    scoring.

    Cadencia 5min (ex 60s, audit 2026-05-09): el track resolution
    granular no aporta a los briefs (que muestran top-N del día) y el
    mood scoring promedia por horas. Bajamos −83% spawn overhead
    (1440→288 ticks/día) sin pérdida de signal útil.

    `RunAtLoad=true` para que bootstrap lance inmediatamente sin esperar
    al primer tick.
    """
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
        "schedule": {"interval_s": 300},
        "run_at_load": True,
        "throttle_s": 60,
        "process_type": "Background",
        "stdout_path": out,
        "stderr_path": err,
    })
