"""Schema-driven plist renderer + helpers compartidos.

Esta es la infra que TODAS las factories de `rag.plists.*` consumen.
Vive en `_render.py` (underscore prefix) porque NO es código de dominio
sino plumbing — los call sites externos al paquete deberían usar las
factories de los sub-módulos por dominio (`persistent.py`, `briefs.py`,
etc.) o el re-export del `__init__.py`.

Spec dict keys (todas opcionales salvo `label` + `program_arguments`):

  label: str               Label launchd full (e.g. "com.fer.obsidian-rag-watch").
  program_arguments: list  Argv del proceso (rag_bin + args, o /bin/bash + script).
  env: dict[str, str]      Vars adicionales para EnvironmentVariables. HOME y
                           PATH se inyectan SIEMPRE primero (en ese orden);
                           el caller no necesita pasarlas.
  schedule: dict | None    Forma del schedule. Una sola key:
                             {"interval_s": int}     → StartInterval
                             {"calendar": dict}      → StartCalendarInterval (single dict)
                             {"calendar_list": list} → StartCalendarInterval (array of dicts)
                           None → no schedule (útil para KeepAlive=true puro).
  run_at_load: bool|None   <true/> o <false/>. None → key omitida.
  keep_alive: bool|None    <true/> o <false/>. None → key omitida.
  throttle_s: int|None     ThrottleInterval en segundos.
  throttle_key: str        "ThrottleInterval" (default) o "Throttle"
                           (daemon-watchdog usa la key corta).
  throttle_after_logs: bool  Si True, ThrottleInterval va DESPUÉS de los
                           log paths (mood-poll). Default False.
  exit_timeout_s: int|None ExitTimeOut.
  process_type: str|None   "Interactive" / "Background" / "Adaptive" / "Standard".
  low_priority_io: bool|None  LowPriorityIO — bajar prioridad I/O del proceso
                           bajo el scheduler de macOS (usa flag IOPOL_THROTTLE).
                           Pensado para batch nocturno (auto-harvest, online-tune,
                           maintenance, etc.) que NO debe pisar el chat/web del
                           user si éste vuelve a la app a las 3 AM.
  nice: int|None           Nice. Range 0..20 (mayor = menos prioridad CPU).
                           Default macOS para Background es ya algo +5; setear
                           explícito 5-10 sólo para batch realmente pesado.
  working_dir: str|None    WorkingDirectory.
  stdout_path: str         Path absoluto del log stdout.
  stderr_path: str         Path absoluto del log stderr.
  extra_env_xml: str       Líneas extra a appendear al final del bloque
                           EnvironmentVariables (escape hatch para el
                           YOUTUBE_API_KEY condicional del web plist).
"""
from __future__ import annotations

from pathlib import Path

from rag._constants import _GOOGLE_TOKEN_PATH

__all__ = [
    "_DEFAULT_PATH",
    "_GOOGLE_TOKEN_PATH",
    "_LAUNCH_AGENTS_DIR",
    "_PLIST_HEADER",
    "_RAG_LOG_DIR",
    "_logs",
    "_rag_binary",
    "_render_plist",
    "_repo_root",
]

_LAUNCH_AGENTS_DIR = Path.home() / "Library/LaunchAgents"
_RAG_LOG_DIR = Path.home() / ".local/share/obsidian-rag"

_DEFAULT_PATH = (
    f"/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin"
)


def _repo_root() -> Path:
    """Repo root absoluto.

    Pre-split (rag/plists.py): `Path(__file__).resolve().parent.parent`
    apuntaba a `<repo>/` (2 niveles arriba: rag/ → repo/).

    Post-split (rag/plists/_render.py): este archivo está un nivel más
    profundo, así que necesitamos `.parent.parent.parent` (3 niveles:
    _render.py → plists/ → rag/ → repo/).

    Centralizado acá para que las factories que necesitan el repo root
    (`_web_plist`, `_online_tune_plist`, `_mood_poll_plist`,
    `_spotify_poll_plist`, `_wake_hook_plist`) consuman este helper en
    vez de computar el path inline. Si el layout cambia de nuevo, hay
    UN solo lugar para fixear.

    Bug histórico (2026-04-26 → exit 78 en `_web_plist`): un `.parent`
    de menos en el path generaba `/repo/rag/.venv/bin/python` y
    `/repo/rag/web/server.py` (paths inexistentes). El daemon crasheaba
    silencioso. Centralizar acá previene regression similar post-split.
    """
    return Path(__file__).resolve().parent.parent.parent


def _rag_binary() -> str:
    """Best-effort path to the installed `rag` binary. Default uv tool path
    first; fall back to PATH lookup. The launchd service runs without our
    interactive PATH so we resolve it once at install time.

    Prefiere el wrapper del venv del proyecto (.venv/bin/rag) cuando existe:
    garantiza que los plists usen el Python correcto del proyecto (3.13)
    en lugar del global (potencialmente 3.14 de Homebrew), que puede tener
    paquetes incompatibles (ej. torch built para 3.13 instalado en env 3.14).
    """
    candidates = [
        _repo_root() / ".venv/bin/rag",   # dev venv — Python correcto del proyecto
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


_PLIST_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
    '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
    '<plist version="1.0">\n'
)


def _render_plist(spec: dict) -> str:
    """Render a launchd plist XML string from a structured spec dict.

    See module-level doc-block for the full list of supported spec keys.
    The output preserves the historical compact one-line format
    (`<key>X</key><value/>` per line) so existing tests that grep for
    specific tokens (e.g. ``"<key>RunAtLoad</key><false/>"``) keep
    passing without any change.
    """
    label = spec["label"]
    program_args = spec["program_arguments"]
    env = spec.get("env") or {}
    extra_env_xml = spec.get("extra_env_xml", "")
    schedule = spec.get("schedule")
    run_at_load = spec.get("run_at_load")
    keep_alive = spec.get("keep_alive")
    throttle_s = spec.get("throttle_s")
    throttle_key = spec.get("throttle_key", "ThrottleInterval")
    throttle_after_logs = spec.get("throttle_after_logs", False)
    exit_timeout_s = spec.get("exit_timeout_s")
    process_type = spec.get("process_type")
    low_priority_io = spec.get("low_priority_io")
    nice = spec.get("nice")
    working_dir = spec.get("working_dir")
    stdout_path = spec["stdout_path"]
    stderr_path = spec["stderr_path"]

    lines: list[str] = []
    lines.append(_PLIST_HEADER)
    lines.append("<dict>\n")
    lines.append(f"  <key>Label</key><string>{label}</string>\n")

    # ProgramArguments
    lines.append("  <key>ProgramArguments</key>\n")
    lines.append("  <array>\n")
    for arg in program_args:
        lines.append(f"    <string>{arg}</string>\n")
    lines.append("  </array>\n")

    # EnvironmentVariables — HOME y PATH siempre primero, luego en orden
    # de inserción del caller. Caller puede overridear PATH pasándolo en `env`
    # (raro; el `_serve_watchdog_plist` histórico lo hacía pre-borrado 2026-05-09).
    lines.append("  <key>EnvironmentVariables</key>\n")
    lines.append("  <dict>\n")
    lines.append(f"    <key>HOME</key><string>{Path.home()}</string>\n")
    custom_path = env.get("PATH")
    lines.append(
        f"    <key>PATH</key><string>{custom_path or _DEFAULT_PATH}</string>\n"
    )
    for k, v in env.items():
        if k in ("HOME", "PATH"):
            continue
        lines.append(f"    <key>{k}</key><string>{v}</string>\n")
    if extra_env_xml:
        lines.append(extra_env_xml)
    lines.append("  </dict>\n")

    # Schedule
    if schedule is not None:
        if "interval_s" in schedule:
            lines.append(
                f"  <key>StartInterval</key><integer>{int(schedule['interval_s'])}</integer>\n"
            )
        elif "calendar" in schedule:
            cal = schedule["calendar"]
            lines.append("  <key>StartCalendarInterval</key>\n")
            lines.append("  <dict>\n")
            for k in ("Weekday", "Day", "Hour", "Minute"):
                if k in cal:
                    lines.append(
                        f"    <key>{k}</key><integer>{int(cal[k])}</integer>\n"
                    )
            lines.append("  </dict>\n")
        elif "calendar_list" in schedule:
            lines.append("  <key>StartCalendarInterval</key>\n")
            lines.append("  <array>\n")
            for cal in schedule["calendar_list"]:
                inner = "".join(
                    f"<key>{k}</key><integer>{int(cal[k])}</integer>"
                    for k in ("Weekday", "Day", "Hour", "Minute")
                    if k in cal
                )
                lines.append(f"    <dict>{inner}</dict>\n")
            lines.append("  </array>\n")
        else:
            raise ValueError(f"Unknown schedule shape: {schedule!r}")

    # RunAtLoad / KeepAlive
    if run_at_load is True:
        lines.append("  <key>RunAtLoad</key><true/>\n")
    elif run_at_load is False:
        lines.append("  <key>RunAtLoad</key><false/>\n")
    if keep_alive is True:
        lines.append("  <key>KeepAlive</key><true/>\n")
    elif keep_alive is False:
        lines.append("  <key>KeepAlive</key><false/>\n")

    # Throttle (default: pre-logs)
    if throttle_s is not None and not throttle_after_logs:
        lines.append(
            f"  <key>{throttle_key}</key><integer>{int(throttle_s)}</integer>\n"
        )

    # ExitTimeOut / ProcessType / WorkingDirectory
    if exit_timeout_s is not None:
        lines.append(
            f"  <key>ExitTimeOut</key><integer>{int(exit_timeout_s)}</integer>\n"
        )
    if process_type is not None:
        lines.append(f"  <key>ProcessType</key><string>{process_type}</string>\n")
    if low_priority_io is True:
        lines.append("  <key>LowPriorityIO</key><true/>\n")
    elif low_priority_io is False:
        lines.append("  <key>LowPriorityIO</key><false/>\n")
    if nice is not None:
        lines.append(f"  <key>Nice</key><integer>{int(nice)}</integer>\n")
    if working_dir is not None:
        lines.append(f"  <key>WorkingDirectory</key><string>{working_dir}</string>\n")

    # Standard logs
    lines.append(f"  <key>StandardOutPath</key><string>{stdout_path}</string>\n")
    lines.append(f"  <key>StandardErrorPath</key><string>{stderr_path}</string>\n")

    # Throttle (post-logs, only mood-poll uses this ordering)
    if throttle_s is not None and throttle_after_logs:
        lines.append(
            f"  <key>{throttle_key}</key><integer>{int(throttle_s)}</integer>\n"
        )

    lines.append("</dict>\n")
    lines.append("</plist>\n")
    return "".join(lines)


def _logs(slug: str) -> tuple[str, str]:
    """Helper: standard `<_RAG_LOG_DIR>/<slug>.log` + `.error.log` paths."""
    return (
        f"{_RAG_LOG_DIR}/{slug}.log",
        f"{_RAG_LOG_DIR}/{slug}.error.log",
    )
