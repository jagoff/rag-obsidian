"""Regresiones de `_detect_propose_intent` que tratan CONSULTAS como
propose-intent (create) cuando claramente el user está pidiendo info.

Bug reportado 2026-04-24 por Fer F.:
- Query: "decime que tengo para la semana que viene?"
- Esperado: flow de lectura → pre-router dispara `calendar_ahead` +
  `reminders_due` y el LLM responde con los eventos de la semana.
- Observado: el agente escupió literal `tool_call(eventos_calendario:
  list_events_start_date=start_of_next_week,end_of_next_week)` — texto
  alucinado de tool-call, sin ejecución real.

Root cause: `_detect_propose_intent` retornaba `True` porque
`_VISIT_PATTERN_RE` matcheaba la palabra "viene" dentro del idiom
temporal "la semana que viene". El flow se saltaba el pre-router
determinístico (que tiene `_PLANNING_PAT` para "semana"), entregando
la query al LLM en modo "create" con las tools propose_* habilitadas.
El modelo, sin tools reales para listar eventos, alucinó.

Fix (rag.py):
1. `_IMPERATIVE_QUERY_RE` nuevo — descalifica imperativos de consulta
   al inicio ("decime/contame/dime/mostrame/tirame/listame").
2. `_TEMPORAL_VIENE_IDIOM_RE` nuevo — strip del texto "la/el/este/esta
   (semana|mes|año|finde|día|tarde|mañana|noche) que viene[n]" antes
   del check de `_VISIT_PATTERN_RE` para evitar que "viene" idiomático
   cuente como verbo de visita.
"""
from __future__ import annotations

import pytest

from rag import _detect_propose_intent


# ── El bug reportado — "decime que tengo para la semana que viene" ─────────
# Combinación de los dos problemas que el fix resuelve:
#   · Imperativo de consulta ("decime") al inicio.
#   · Idiom "la semana que viene" con "viene" literal.
@pytest.mark.parametrize("query", [
    "decime que tengo para la semana que viene?",
    "decime que tengo la semana que viene",
    "decime que tengo esta semana",
    "dime qué tengo la semana que viene",
    "contame que hago mañana",
    "mostrame mis eventos de la semana",
    "listame los pendientes de hoy",
])
def test_read_queries_no_son_propose(query: str) -> None:
    """Imperativos de consulta = pedido de LEER info, no CREAR."""
    assert _detect_propose_intent(query) is False, (
        f"{query!r} debería tratarse como read-intent, no propose"
    )


# ── Idioms temporales con "viene/vienen" que no son visitas ────────────────
@pytest.mark.parametrize("query", [
    "hoy no quiero nada la semana que viene",
    "a la tarde que viene no tengo nada",
    "el mes que viene arranca el proyecto de la casa",
    "el año que viene cumplo 40",
    "próxima semana que viene nos mudamos",
])
def test_idiom_viene_no_dispara_visit(query: str) -> None:
    """'Que viene' dentro de un temporal anchor no es un verbo de visita.

    Nota: estos queries no tienen ningún verbo de propose (recordame,
    agendá, etc.), ningún event noun (reunión, cita…) y ningún imperativo
    de consulta. Son declaraciones narrativas — deberían ser False.
    """
    assert _detect_propose_intent(query) is False, (
        f"{query!r}: 'viene' es idiom temporal, no visita"
    )


# ── Regresión: visits reales siguen funcionando ────────────────────────────
# Después del fix, los patterns legítimos de visita que rag.py
# _VISIT_PATTERN_RE reconoce NO deben romperse. Nota: el regex de
# rag.py cubre viene/vienen/pasa/pasan/llega/llegan/visita/trae/traen/
# busco/buscamos/buscan pero NO vuelve/vuelven — ése vive en el
# listener.ts con otro alcance.
@pytest.mark.parametrize("query", [
    "Juan viene el viernes",
    "Grecia viene el miercoles",
    "mañana pasa Juan por casa",
    "el jueves llega mamá",
    "visita el dentista el lunes",
    "traen la heladera mañana",
])
def test_visits_reales_siguen_propose(query: str) -> None:
    """Visits genuinas no deben caerse por el fix del idiom."""
    assert _detect_propose_intent(query) is True, (
        f"{query!r}: visita genuina, debería seguir siendo propose"
    )


# ── Idiom + visit real combinados ─────────────────────────────────────────
def test_idiom_mas_visit_real_gana_la_visita() -> None:
    """Si el texto tiene AMBOS — idiom temporal + visita real — gana la
    visita (después del strip del idiom, VISIT_PATTERN todavía matchea)."""
    q = "la semana que viene viene Juan a casa"
    assert _detect_propose_intent(q) is True


def test_idiom_mas_event_noun_gana_el_evento() -> None:
    """Idiom temporal + event noun explícito → propose (create del evento)."""
    q = "la semana que viene tengo reunión con Pepe"
    assert _detect_propose_intent(q) is True


# ── Los triggers explícitos siguen siendo propose ──────────────────────────
@pytest.mark.parametrize("query", [
    "recordame comprar pan mañana",
    "agendá reunión con Juan el lunes 10am",
    "no te olvides de llamar al dentista",
    "anotame revisar el PR",
    "calendarizá el partido del domingo",
    "poné en el calendario el vencimiento del 25",
])
def test_triggers_explicitos_sin_regresion(query: str) -> None:
    """El regex principal (_PROPOSE_INTENT_RE) matchea estos — el fix
    no debería alterar ese path."""
    assert _detect_propose_intent(query) is True


# ── Question-word start sigue descartando (original behavior) ──────────────
@pytest.mark.parametrize("query", [
    "qué tengo hoy",
    "¿qué tengo la semana que viene?",
    "cuándo viene Grecia",
    "dónde tengo que ir mañana",
    "cómo viene la semana",
])
def test_question_start_descarta(query: str) -> None:
    """_QUESTION_START_RE sigue como gate — no debería haberse roto."""
    assert _detect_propose_intent(query) is False
