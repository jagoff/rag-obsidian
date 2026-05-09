"""Signal — Reunión próxima (pre-meeting reminder).

Lee el Calendar del user vía `_fetch_calendar_today()` (icalBuddy) y si
hay un evento que arranca en los próximos `_LEAD_MIN_MIN`-`_LEAD_MIN_MAX`
minutos, emite un push:

    🗓 *reunión '<title>' en 12min* (10:30-11:00)

## Diseño

- Solo events de HOY que tienen hora de start (skip all-day events).
- Window: 0 < lead_min ≤ _LEAD_MIN_MAX. <0 ya empezó, >MAX es muy
  temprano (apenas notificable).
- dedup_key: `meeting:<YYYY-MM-DD>:<title-slug>:<start-time>` —
  estable cross-runs del mismo día. Si el evento se mueve de hora,
  el dedup_key cambia y re-pushea (correcto).
- snooze_hours=1: si el user dismisses, no re-aparece dentro de 1h
  (ya entró a la meeting o decidió ignorar).
- Score lineal por proximidad: 1.0 si <5min, 0.9 si <10min, 0.7 si
  <15min. El kind framework usa el score para ordenamiento si hay
  varias signals compitiendo.

## Anti-noise

- Cap `_MAX_CANDIDATES=3`: día con 5 reuniones back-to-back no spamea
  3 pushes en 5min — el primero alcanza para alertar.
- Skip events cuyo título es ruido (`Busy`, `Tentative`, vacíos).
- Silent-fail: icalBuddy no instalado o `OBSIDIAN_RAG_NO_APPLE=1` → [].
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from rag_anticipate.signals.base import register_signal


# Window de notificación pre-meeting (minutos antes del start).
_LEAD_MIN_MIN = 1   # < 1 min ya empezó (0 = arranca AHORA, también sirve)
_LEAD_MIN_MAX = 15  # más allá de 15min es prematuro

# Cap defensivo: día con muchas reuniones contiguas.
_MAX_CANDIDATES = 3

# Títulos genéricos que no merecen push (Outlook auto-blocks, Tentative).
_NOISE_TITLES = frozenset({
    "busy", "tentative", "free", "out of office",
    "ocupado", "ausente", "fuera de oficina",
})


def _slug(s: str) -> str:
    """Normaliza un título para usarlo en dedup_key (alfanumérico + dash)."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:60]  # cap defensivo


def _parse_hhmm(s: str) -> tuple[int, int] | None:
    """Parsea 'HH:MM' a (hour, minute). Devuelve None si no parsea."""
    if not s:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})$", s.strip())
    if not m:
        return None
    try:
        hour = int(m.group(1))
        minute = int(m.group(2))
    except (ValueError, TypeError):
        return None
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    return (hour, minute)


@register_signal(name="meeting_pre", snooze_hours=1)
def meeting_pre_signal(now: datetime) -> list:
    """Emite hasta `_MAX_CANDIDATES` candidates: reuniones que arrancan
    en los próximos 15 min. Silent-fail completo.
    """
    try:
        from rag import AnticipatoryCandidate
        from rag.integrations.calendar import _fetch_calendar_today

        events = _fetch_calendar_today(max_events=20)
        if not events:
            return []

        out: list = []
        for ev in events:
            if len(out) >= _MAX_CANDIDATES:
                break
            try:
                title = (ev.get("title") or "").strip()
                if not title:
                    continue
                if title.lower() in _NOISE_TITLES:
                    continue
                start_str = (ev.get("start") or "").strip()
                hm = _parse_hhmm(start_str)
                if hm is None:
                    continue  # all-day o malformed
                hour, minute = hm
                # Construir el datetime de start con la fecha de `now`.
                start_dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                lead_seconds = (start_dt - now).total_seconds()
                lead_min = lead_seconds / 60.0
                if lead_min < _LEAD_MIN_MIN - 1:  # ya empezó (>1min ago)
                    continue
                if lead_min > _LEAD_MIN_MAX:
                    continue

                # Score por proximidad: cuanto más cerca, más urgente.
                if lead_min <= 5:
                    score = 1.0
                elif lead_min <= 10:
                    score = 0.9
                else:
                    score = 0.7

                # Lead minutes formateado para el message (entero, redondea down).
                lead_int = max(0, int(lead_min))
                end_str = (ev.get("end") or "").strip()
                time_range = f"{start_str}-{end_str}" if end_str else start_str

                if lead_int == 0:
                    when = "ahora"
                elif lead_int == 1:
                    when = "en 1min"
                else:
                    when = f"en {lead_int}min"

                message = f"🗓 *reunión '{title}' {when}* ({time_range})"

                day_str = now.strftime("%Y-%m-%d")
                out.append(AnticipatoryCandidate(
                    kind="anticipate-meeting_pre",
                    score=score,
                    message=message,
                    dedup_key=f"meeting:{day_str}:{_slug(title)}:{start_str}",
                    snooze_hours=1,
                    reason=f"title={title!r} start={start_str} lead={lead_min:.1f}min",
                ))
            except Exception:
                continue

        return out
    except Exception:
        return []
