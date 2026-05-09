"""Signal — Viaje próximo (pre-trip checklist nudge).

Lee el Calendar próximas 72hs vía `_fetch_calendar_ahead(days_ahead=3)`
y si encuentra un evento cuyo título matchea patrón de viaje
("viaje", "vuelo", "flight", "trip", "vacaciones"), emite push:

    ✈️ *viaje '<title>' en <when>* — ¿armamos checklist de pendientes?

## Diseño

- Solo events de las próximas 72hs (window útil para preparar viajes).
- Score lineal por proximidad usando `date_label` (icalBuddy lo
  devuelve como "today"/"tomorrow"/"in N days"):
    - hoy/mañana → 1.0
    - in 2 days → 0.8
    - in 3 days → 0.6
- dedup_key incluye fecha del trip + slug(title) — si se reagenda,
  re-pushea (correcto).
- snooze_hours=24: si dismisses, vuelve mañana (importante para no
  perder un viaje urgente que el user posterga el push).
- Cap `_MAX_CANDIDATES=2`: rara vez 2 viajes en 72h, mucho menos 3+.

## Pattern detection

Match case-insensitive contra `_TRIP_PATTERN`. Conservador:
- ✓ "Viaje a Buenos Aires"
- ✓ "Vuelo MIA-EZE LH 2345"
- ✓ "Trip to NYC"
- ✓ "Flight to Madrid"
- ✓ "Vacaciones familia"
- ✗ "Reunión preparar viaje" (false positive, requiere whole word)

## Anti-noise

- Silent-fail: icalBuddy down → []. No mola pushear todo lo del calendar
  como "viaje".
"""

from __future__ import annotations

import re
from datetime import datetime

from rag_anticipate.signals.base import register_signal


# Word-boundary patterns que disparan el match. Conservador para evitar
# false positives ("reunión preparar viaje" no matchea — requiere word).
_TRIP_PATTERN = re.compile(
    r"\b(viaje|vuelo|flight|trip|vacaciones|holidays?|getaway)\b",
    re.IGNORECASE,
)

# Cap defensivo (no spam si user tiene viaje + reunión-pre-viaje agendada).
_MAX_CANDIDATES = 2

# Mapping date_label icalBuddy → score por proximidad.
# Orden importa: probamos del más urgente al menos urgente.
_DATE_LABEL_SCORES: tuple[tuple[str, float, str], ...] = (
    ("today", 1.0, "hoy"),
    ("hoy", 1.0, "hoy"),
    ("tomorrow", 1.0, "mañana"),
    ("mañana", 1.0, "mañana"),
    ("in 2 days", 0.8, "en 2 días"),
    ("en 2 días", 0.8, "en 2 días"),
    ("in 3 days", 0.6, "en 3 días"),
    ("en 3 días", 0.6, "en 3 días"),
)


def _slug(s: str) -> str:
    """Normaliza título para dedup_key (alfanumérico + dash)."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:60]


def _score_from_label(date_label: str) -> tuple[float, str] | None:
    """Devuelve (score, when_label_es) si el label es válido para nuestro
    window. None si está fuera del window de 72hs."""
    if not date_label:
        return None
    lower = date_label.lower().strip()
    for needle, score, when in _DATE_LABEL_SCORES:
        if needle in lower:
            return (score, when)
    # Fallback: si dice "in N days" con N>3 o no parsea, fuera del window.
    return None


@register_signal(name="pre_trip_checklist", snooze_hours=24)
def pre_trip_checklist_signal(now: datetime) -> list:
    """Emite hasta `_MAX_CANDIDATES` candidates: viajes en próximas 72hs.
    Silent-fail completo.
    """
    try:
        from rag import AnticipatoryCandidate
        from rag.integrations.calendar import _fetch_calendar_ahead

        events = _fetch_calendar_ahead(days_ahead=3, max_events=40)
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
                if not _TRIP_PATTERN.search(title):
                    continue
                date_label = (ev.get("date_label") or "").strip()
                scored = _score_from_label(date_label)
                if scored is None:
                    continue
                score, when = scored

                message = (
                    f"✈️ *viaje '{title}' {when}* — "
                    f"¿armamos checklist de pendientes?"
                )

                # dedup_key con fecha de hoy + slug — si el trip se reagenda
                # de "tomorrow" a "in 3 days", el day_str cambia con el
                # tiempo y re-pushea correctamente.
                day_str = now.strftime("%Y-%m-%d")
                out.append(AnticipatoryCandidate(
                    kind="anticipate-pre_trip_checklist",
                    score=score,
                    message=message,
                    dedup_key=f"trip:{day_str}:{_slug(title)}",
                    snooze_hours=24,
                    reason=f"title={title!r} when={when} date_label={date_label!r}",
                ))
            except Exception:
                continue

        return out
    except Exception:
        return []
