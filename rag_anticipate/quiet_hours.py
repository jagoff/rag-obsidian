"""Quiet hours — gate global del Anticipatory Agent.

`is_quiet_now(now)` retorna `(True, reason)` cuando el agent NO debe pushear:

  1. **Night hours** — `now` cae dentro de la ventana nocturna (default
     22:00 → 08:00, wrap-around-aware). Override con
     `RAG_ANTICIPATE_QUIET_NIGHT_START` / `RAG_ANTICIPATE_QUIET_NIGHT_END`
     en formato `HH:MM` 24h.
  2. **In meeting** — hay un evento de calendar en curso ahora (now ∈
     [start, end]). Lee `rag._fetch_calendar_today()` y parsea
     `start`/`end` como `HH:MM` (24h) o `H:MM AM/PM`.
  3. **Focus state** — el user state actual (vía `rag._read_state`)
     contiene una keyword tipo `focus`, `deep-work`, `concentrado` o
     `no molestar`.

Bypass total para debug: `RAG_ANTICIPATE_BYPASS_QUIET=1` fuerza retorno
`(False, "")` ignorando todas las señales.

Cualquier excepción interna en los probes (calendar fetch falla, state
file corrupto, parse error) se silencia → False (degradación grácil:
mejor un push ocasional fuera de quiet que tumbar el agent entero).
"""

from __future__ import annotations

import os
import re
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


__all__ = [
    "is_night",
    "is_in_meeting",
    "is_focus_state",
    "is_quiet_now",
]
