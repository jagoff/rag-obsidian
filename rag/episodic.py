"""Episodic memory — narrative retrieval cuando la query es reflexiva.

Game-Changer G4 (2026-05-11). Las queries tipo "qué pasó la semana
pasada con X", "cuándo decidí Y", "cómo arrancó la conversación con
Cliente Z en abril" hoy las trata el path semantic común — retorna
chunks rankeados, no una narrativa temporal. Este módulo:

1. **Detecta** intent episodic en `classify_intent` via regex que
   combina ancla temporal + verbo reflexivo o pregunta narrativa.
2. **Parsea** el ancla temporal a `(start_ts, end_ts)` epoch usando
   dateutil + heurísticas rioplatense ("ayer", "la semana pasada",
   "en abril", "hace 2 meses", "el martes").
3. **Construye prompt narrativo** que se appendea al system prompt
   cuando intent=episodic: instruye al LLM a renderar la respuesta
   como cronología "Lunes: X · Martes: Y · ...", citando por evento.

El retrieve pipeline normal ya soporta `date_range` — solo le pasamos
la ventana parseada. El cambio fundamental es el RENDER del LLM, no
el retrieval per se. Trade-off intencional: shipping rápido + medible,
sin re-arquitectura del pipeline.

Fase 2 (no en este commit): cross-source merge (calendar + WA + mood
+ commits) ordenados cronológicamente como input al LLM en lugar de
los chunks tradicionales.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional


# Regex episodic intent: ancla temporal + verbo reflexivo o pregunta narrativa.
# Caso 1: "qué pasó/dije/hicimos/decidí + <ancla>"
# Caso 2: "cuándo + <verbo> + ..."
# Caso 3: "cómo arrancó/empezó/terminó + <X>"
_TEMPORAL_ANCHOR_RE = re.compile(
    r"\b("
    r"hoy|ayer|anteayer|"
    r"esta\s+semana|la\s+semana\s+pasada|"
    r"este\s+mes|el\s+mes\s+pasado|"
    r"este\s+año|este\s+a[ñn]o|el\s+a[ñn]o\s+pasado|"
    r"hace\s+\d+\s+(d[ií]as?|semanas?|meses?|a[ñn]os?)|"
    r"en\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    r"septiembre|octubre|noviembre|diciembre)|"
    r"el\s+(lunes|martes|mi[ée]rcoles|jueves|viernes|s[áa]bado|domingo)"
    r")\b",
    re.IGNORECASE,
)

_REFLECTIVE_VERB_RE = re.compile(
    r"\b("
    r"qu[ée]\s+pas[oó]|qu[ée]\s+dije|qu[ée]\s+hicimos|qu[ée]\s+decid[íi]|"
    r"qu[ée]\s+hablamos|qu[ée]\s+acord(?:amos|[ée])|"
    r"cu[áa]ndo\s+(?:fue|pas[oó]|decid[íi]|empez[oó]|termin[oó])|"
    r"c[oó]mo\s+(?:arranc[oó]|empez[oó]|termin[oó]|fue)"
    r")\b",
    re.IGNORECASE,
)

# Diccionario meses → número.
_MONTHS_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11,
    "diciembre": 12,
}


def is_episodic(question: str) -> bool:
    """True si la query califica como episodic (ancla + reflexive verb)."""
    if not question or len(question) < 10:
        return False
    return bool(
        _TEMPORAL_ANCHOR_RE.search(question)
        and _REFLECTIVE_VERB_RE.search(question)
    )


def parse_temporal_anchor(question: str, *, now: datetime | None = None) -> Optional[tuple[float, float]]:
    """Parsea el ancla temporal de la query a `(start_epoch, end_epoch)`.

    Devuelve None si no encuentra ancla o el parsing falla — caller
    debería caer al path semantic normal.
    """
    if now is None:
        now = datetime.now()
    q = question.lower()

    # "hoy"
    if re.search(r"\bhoy\b", q):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return (start.timestamp(), end.timestamp())

    # "ayer"
    if re.search(r"\bayer\b", q):
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return (start.timestamp(), end.timestamp())

    # "anteayer"
    if re.search(r"\banteayer\b", q):
        start = (now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return (start.timestamp(), end.timestamp())

    # "esta semana" — desde lunes 00:00 hasta ahora.
    if re.search(r"\besta\s+semana\b", q):
        start = now - timedelta(days=now.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        return (start.timestamp(), now.timestamp())

    # "la semana pasada"
    if re.search(r"\bla\s+semana\s+pasada\b", q):
        this_monday = now - timedelta(days=now.weekday())
        last_monday = this_monday - timedelta(days=7)
        start = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = this_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        return (start.timestamp(), end.timestamp())

    # "este mes"
    if re.search(r"\beste\s+mes\b", q):
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return (start.timestamp(), now.timestamp())

    # "el mes pasado"
    if re.search(r"\bel\s+mes\s+pasado\b", q):
        first_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Restar 1 día para ir al último día del mes anterior, luego día 1.
        last_day_prev = first_this - timedelta(days=1)
        first_prev = last_day_prev.replace(day=1)
        return (first_prev.timestamp(), first_this.timestamp())

    # "hace N (días|semanas|meses)"
    m = re.search(r"\bhace\s+(\d+)\s+(d[ií]as?|semanas?|meses?)\b", q)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("d"):
            delta_d = n
        elif unit.startswith("semana"):
            delta_d = n * 7
        else:  # meses
            delta_d = n * 30
        center = now - timedelta(days=delta_d)
        start = center.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return (start.timestamp(), end.timestamp())

    # "en <mes>" — ventana del mes en el año actual (o pasado si ya pasó).
    m = re.search(
        r"\ben\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
        r"septiembre|octubre|noviembre|diciembre)\b",
        q,
    )
    if m:
        month = _MONTHS_ES.get(m.group(1).lower())
        if month:
            year = now.year
            # Si el mes mencionado es futuro este año → asumimos año pasado.
            if month > now.month:
                year -= 1
            start = datetime(year, month, 1)
            # Último día del mes:
            if month == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month + 1, 1)
            return (start.timestamp(), end.timestamp())

    return None


_EPISODIC_INSTRUCTION = """

REGLA EPISODIC (esta query es reflexiva-temporal):

El user te está pidiendo una NARRATIVA cronológica de eventos en una
ventana temporal específica, no una respuesta factual ranqueada.
Renderá la respuesta así:

- Agrupá los eventos por día (o por semana si la ventana es ≥14 días).
- Ordená cronológicamente ascendente (lo más viejo primero).
- Por cada evento: una línea concisa "Lunes 5 — <qué pasó>" con cita.
- Cerrá con 1-2 líneas de síntesis: "Hilo principal de la ventana: X. Quedaron pendientes: Y."

Si la ventana NO tiene eventos relevantes, decílo: "No encontré
actividad sobre eso en <ventana>." No improvises.
"""


def episodic_system_prompt_suffix() -> str:
    """Devuelve el bloque de instrucción a appendear al system prompt
    cuando intent=episodic. Caller (system_prompt_for_intent) lo concatena
    al final.
    """
    return _EPISODIC_INSTRUCTION


__all__ = ["is_episodic", "parse_temporal_anchor", "episodic_system_prompt_suffix"]
