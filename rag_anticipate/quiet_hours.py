"""Quiet hours — gate global del Anticipatory Agent.

## Dos APIs coexistentes

### Legacy: `is_quiet_now(now)` (Phase 1)

Retorna `(True, reason)` cuando el agent NO debe pushear. Usa env vars
`RAG_ANTICIPATE_QUIET_NIGHT_START` / `_END` (default 22:00 → 08:00).
Reasons: `"night hours"` / `"in meeting"` / `"user in focus state"`.

### Phase 2.B: `is_in_quiet_hours(now=None)` — nuevo

Misma idea pero con shape de retorno `(bool, str | None)` y nuevas env
vars (más declarativas, banda única tipo `23-7`):

- `RAG_QUIET_HOURS_NIGHTTIME=23-7` (default `23-7`, hard rule)
- `RAG_QUIET_HOURS_MEETINGS=1` (default ON)
- `RAG_QUIET_HOURS_FOCUS_CODE=0` (default OFF — heurística experimental
  basada en `pgrep` de procesos IDE recientes)

Reasons (snake_case, machine-readable): `"nighttime"`, `"in_meeting"`,
`"focus_code"`.

Bypass total: `RAG_ANTICIPATE_BYPASS_QUIET=1` ignora todas las señales.

Silent-fail global: cualquier excepción interna degrada a "no quiet"
(prefiere un push ocasional sobre tumbar el agent entero).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime, time as dtime


def _get_night_window() -> tuple[dtime, dtime]:
    """Ventana nocturna. Default 22:00 → 08:00. Override via env:

      RAG_ANTICIPATE_QUIET_NIGHT_START="22:00"
      RAG_ANTICIPATE_QUIET_NIGHT_END="08:00"

    Si la env var es inválida (no parsea como `HH:MM`), cae al default
    silenciosamente — no queremos que un typo en el shell rc tumbe el
    agent.
    """
    def _parse(s: str, default: tuple[int, int]) -> dtime:
        try:
            h, m = s.split(":")
            return dtime(int(h), int(m))
        except Exception:
            return dtime(*default)

    start = _parse(
        os.environ.get("RAG_ANTICIPATE_QUIET_NIGHT_START", "22:00"),
        (22, 0),
    )
    end = _parse(
        os.environ.get("RAG_ANTICIPATE_QUIET_NIGHT_END", "08:00"),
        (8, 0),
    )
    return start, end


def is_night(now: datetime) -> bool:
    """True si `now` está dentro de la ventana nocturna.

    Maneja wrap-around (default 22:00 → 08:00 cruza medianoche). Start es
    inclusive, end es exclusive — `is_night(22:00) == True`,
    `is_night(08:00) == False`.
    """
    start, end = _get_night_window()
    t = now.time()
    if start < end:
        # Ventana intra-día (ej. 13:00 → 14:00)
        return start <= t < end
    # Wrap (ej. 22:00 → 08:00 crosses midnight)
    return t >= start or t < end


def _parse_hhmm(raw: str, now: datetime) -> datetime | None:
    """Parsear `HH:MM` (24h) o `H:MM AM/PM` a datetime en la fecha de `now`.

    Retorna `None` si el input está vacío o no parsea — caller decide
    qué hacer (skip evento). No tira excepciones.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    m = re.match(r"(\d{1,2}):(\d{2})\s*([AaPp][Mm])?", raw)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    ampm = (m.group(3) or "").upper()
    if ampm == "PM" and hh != 12:
        hh += 12
    elif ampm == "AM" and hh == 12:
        hh = 0
    if hh > 23 or mm > 59:
        return None
    return now.replace(hour=hh, minute=mm, second=0, microsecond=0)


def is_in_meeting(now: datetime) -> bool:
    """True si hay un evento de calendar AHORA (now ∈ [start, end]).

    Usa `rag._fetch_calendar_today()` + parseo de `start`/`end` HH:MM.
    Eventos sin `start`/`end` parseables se skipean. Silent-fail global
    (calendar source down, icalBuddy missing, etc.) → False.
    """
    try:
        from rag import _fetch_calendar_today
        events = _fetch_calendar_today(max_events=30)
    except Exception:
        return False
    for ev in events:
        try:
            start = _parse_hhmm(ev.get("start", ""), now)
            end = _parse_hhmm(ev.get("end", ""), now)
            if start and end and start <= now < end:
                return True
        except Exception:
            continue
    return False


def is_focus_state() -> bool:
    """True si el user state actual es focus-like.

    Keywords matcheadas (substring, case-insensitive): `focus`,
    `deep-work`, `concentrado`, `no molestar`. Usa `rag._read_state` si
    existe — no asumimos un getter específico para mantenerlo flexible
    al refactor del state subsystem. Silent-fail → False.
    """
    try:
        import rag
        if hasattr(rag, "_read_state"):
            s = rag._read_state()
            if isinstance(s, dict):
                state_text = (s.get("text") or "").lower()
                return any(
                    k in state_text
                    for k in ("focus", "deep-work", "concentrado", "no molestar")
                )
        return False
    except Exception:
        return False


def is_quiet_now(now: datetime) -> tuple[bool, str]:
    """Check master: combina night + meeting + focus.

    Returns `(quiet: bool, reason: str)`. `reason` es `""` si no es
    quiet. Orden de precedencia: night → focus → meeting (queda fijado
    en este orden para que las razones sean determinísticas en tests
    cuando dos señales activan simultáneamente).

    Bypass total: `RAG_ANTICIPATE_BYPASS_QUIET=1` (o `"true"`) fuerza
    `(False, "")` — útil para debug del agent en horario nocturno sin
    tener que reescribir el reloj.
    """
    if os.environ.get("RAG_ANTICIPATE_BYPASS_QUIET", "").strip() in ("1", "true"):
        return (False, "")
    if is_night(now):
        return (True, "night hours")
    if is_focus_state():
        return (True, "user in focus state")
    if is_in_meeting(now):
        return (True, "in meeting")
    return (False, "")


# ── Phase 2.B API ───────────────────────────────────────────────────────────
# Nueva API con shape `(bool, str | None)` y reasons machine-readable
# (`nighttime` / `in_meeting` / `focus_code`). Coexiste con `is_quiet_now`
# legacy para no romper callers existentes ni tests.

# Patrones de procesos IDE para la heurística focus-code. Match contra
# el `comm`/`args` que devuelve `pgrep -lf`. Lista conservadora —
# solo IDEs que típicamente implican concentración activa.
_FOCUS_CODE_PROC_PATTERN = "Code Helper|Cursor|cursor|nvim|neovim|vim|MacVim|jetbrains|pycharm|webstorm|intellij"

# Si el proceso IDE arrancó hace más de N segundos, ya no consideramos
# "focus reciente" (la idea es: si recién abriste el editor, estás
# por concentrarte; si lleva 4h abierto en background, no necesariamente).
_FOCUS_CODE_RECENT_SECONDS = 120


def _parse_nighttime_band(raw: str) -> tuple[int, int] | None:
    """Parsea `RAG_QUIET_HOURS_NIGHTTIME` en formato `H-H` (24h, hora entera).

    Ejemplos válidos:
      `"23-7"` → (23, 7)
      `"22-8"` → (22, 8)
      `"0-6"` → (0, 6)

    Retorna `None` si no parsea (caller usa default).
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    m = re.match(r"^(\d{1,2})\s*-\s*(\d{1,2})$", raw)
    if not m:
        return None
    start = int(m.group(1))
    end = int(m.group(2))
    if not (0 <= start <= 23 and 0 <= end <= 23):
        return None
    return (start, end)


def _is_nighttime(now: datetime) -> bool:
    """True si `now` cae dentro de la ventana `RAG_QUIET_HOURS_NIGHTTIME`.

    Default `23-7` (23:00 → 07:00, wrap around midnight). Start es
    inclusive, end es exclusive: `_is_nighttime(23:00)` → True,
    `_is_nighttime(07:00)` → False.
    """
    band = _parse_nighttime_band(
        os.environ.get("RAG_QUIET_HOURS_NIGHTTIME", "23-7"),
    )
    if band is None:
        band = (23, 7)
    start_h, end_h = band
    h = now.hour
    if start_h < end_h:
        # Ventana intra-día (ej. 13-14): start ≤ h < end.
        return start_h <= h < end_h
    if start_h == end_h:
        # Banda degenerada — interpretamos como "siempre quiet" sólo si
        # alguien pone `0-0`; por defecto preferimos `False` para no
        # bloquear al agent indefinidamente.
        return False
    # Wrap (ej. 23-7): h ≥ start O h < end.
    return h >= start_h or h < end_h


def _is_meeting_active(now: datetime) -> bool:
    """True si hay un evento de calendar en curso ahora.

    Reusa la heurística de la API legacy (`is_in_meeting`). Controlado
    por `RAG_QUIET_HOURS_MEETINGS` (default ON).
    """
    raw = os.environ.get("RAG_QUIET_HOURS_MEETINGS", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    try:
        return is_in_meeting(now)
    except Exception:
        return False


def _is_focus_code_active() -> bool:
    """True si hay procesos IDE activos y arrancaron hace <2 min.

    Heurística experimental — default OFF. Activar con
    `RAG_QUIET_HOURS_FOCUS_CODE=1`. Usa `pgrep -lf` (BSD/macOS) o
    fallback a `ps -eo` cuando `pgrep` no soporta los flags.

    Silent-fail: cualquier excepción del subprocess → False (no
    interrumpir el flow del agent por un parse error de `ps`).
    """
    raw = os.environ.get("RAG_QUIET_HOURS_FOCUS_CODE", "0").strip().lower()
    if raw not in ("1", "true", "yes", "on"):
        return False
    if shutil.which("pgrep") is None:
        return False
    try:
        # `-l` lista PID + comando, `-f` matchea contra args completos.
        # macOS pgrep soporta ambos. Timeout estricto (1s) para no
        # bloquear el daemon ante un sistema cargado.
        proc = subprocess.run(
            ["pgrep", "-lf", _FOCUS_CODE_PROC_PATTERN],
            capture_output=True, text=True, timeout=1.0,
        )
    except Exception:
        return False
    if proc.returncode != 0 or not proc.stdout.strip():
        return False
    pids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # `pgrep -l` formato: `<pid> <command...>`
        parts = line.split(None, 1)
        if not parts:
            continue
        try:
            pids.append(int(parts[0]))
        except ValueError:
            continue
    if not pids:
        return False
    # Chequear etime de cada PID. `ps -p <pid> -o etimes=` devuelve
    # segundos desde el start del proceso (BSD ps en macOS soporta
    # `etimes` desde Sierra). Si CUALQUIERA de los PIDs es reciente
    # (<2 min), consideramos focus activo.
    for pid in pids:
        try:
            ps = subprocess.run(
                ["ps", "-p", str(pid), "-o", "etime="],
                capture_output=True, text=True, timeout=1.0,
            )
        except Exception:
            continue
        etime_raw = ps.stdout.strip()
        if not etime_raw:
            continue
        secs = _parse_etime(etime_raw)
        if secs is None:
            continue
        if secs < _FOCUS_CODE_RECENT_SECONDS:
            return True
    return False


def _parse_etime(raw: str) -> int | None:
    """Parsea el output de `ps -o etime=` a segundos.

    Formatos posibles (BSD ps):
      `"      05"` → 5 s
      `"   01:23"` → 1 m 23 s
      `"01:23:45"` → 1 h 23 m 45 s
      `"1-02:03:04"` → 1 día 2 h 3 m 4 s
    """
    raw = raw.strip()
    if not raw:
        return None
    days = 0
    if "-" in raw:
        d_str, raw = raw.split("-", 1)
        try:
            days = int(d_str)
        except ValueError:
            return None
    parts = raw.split(":")
    try:
        if len(parts) == 1:
            secs = int(parts[0])
            return days * 86400 + secs
        if len(parts) == 2:
            mins, secs = int(parts[0]), int(parts[1])
            return days * 86400 + mins * 60 + secs
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return days * 86400 + h * 3600 + m * 60 + s
    except ValueError:
        return None
    return None


def is_in_quiet_hours(now: datetime | None = None) -> tuple[bool, str | None]:
    """Phase 2.B gate. Retorna `(True, reason)` o `(False, None)`.

    Args:
        now: Fecha/hora a evaluar. Default `datetime.now()`.

    Returns:
        `(quiet, reason)`. `reason` es uno de:
          - `"nighttime"` — dentro de `RAG_QUIET_HOURS_NIGHTTIME` (def
            `23-7`, hard rule).
          - `"in_meeting"` — evento en curso ahora (si `RAG_QUIET_HOURS_
            MEETINGS=1`, default ON).
          - `"focus_code"` — proceso IDE reciente (si
            `RAG_QUIET_HOURS_FOCUS_CODE=1`, default OFF).
        `None` cuando `quiet=False`.

    Bypass total: `RAG_ANTICIPATE_BYPASS_QUIET=1` → siempre `(False, None)`.
    """
    if os.environ.get("RAG_ANTICIPATE_BYPASS_QUIET", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        return (False, None)
    now = now or datetime.now()
    try:
        if _is_nighttime(now):
            return (True, "nighttime")
    except Exception:
        pass
    try:
        if _is_meeting_active(now):
            return (True, "in_meeting")
    except Exception:
        pass
    try:
        if _is_focus_code_active():
            return (True, "focus_code")
    except Exception:
        pass
    return (False, None)


__all__ = [
    "is_night",
    "is_in_meeting",
    "is_focus_state",
    "is_quiet_now",
    "is_in_quiet_hours",
]
