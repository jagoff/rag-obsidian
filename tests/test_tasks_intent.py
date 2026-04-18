"""Tests for `_is_tasks_query` — the regex classifier that decides whether
to bypass vault RAG and fetch from the services layer (calendar, reminders,
mail) instead.

The classifier MUST NOT fire on topic queries like "qué hay sobre X" or
"qué tengo en notas sobre Y" — those are RAG queries about vault content,
not tasks/agenda lookups. Mis-routing them returns the agenda in ~90s
instead of the correct retrieve+LLM response.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# web/server.py is a script, not a package — load it via importlib.
_SERVER_PATH = Path(__file__).resolve().parent.parent / "web" / "server.py"
_spec = importlib.util.spec_from_file_location("web_server", _SERVER_PATH)
assert _spec and _spec.loader, "could not load web/server.py"
web_server = importlib.util.module_from_spec(_spec)
sys.modules["web_server"] = web_server
_spec.loader.exec_module(web_server)

_is_tasks_query = web_server._is_tasks_query


# ---------------------------------------------------------------------------
# POSITIVE — these MUST be classified as tasks intent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "q",
    [
        # Spanish — explicit task/time qualifier
        "qué tengo para hoy",
        "qué tengo para mañana",
        "qué tengo que hacer",
        "qué debo hacer hoy",
        "qué hay para mañana",
        "qué hay para hoy",
        "qué me falta hacer hoy",
        "qué tengo pendiente",
        # Token-only matches (TOKENS regex)
        "mis pendientes de la semana",
        "muestrame la agenda",
        "qué reuniones tengo",
        "recordatorios para mañana",
        "tareas urgentes",
        "compromisos pendientes",
        # Phrase variants — "tengo/hay algo X"
        "tengo algo pendiente",
        "hay eventos hoy",
        "tengo citas mañana",
        # "que hay para hoy/mañana" alt
        "que hay para hoy",
        "que hay para esta semana",
        # Organize patterns
        "organizame el día",
        "organizar la semana",
        # English
        "what do i need to do",
        "what do i have to complete",
    ],
)
def test_tasks_intent_fires(q: str) -> None:
    assert _is_tasks_query(q), f"should fire on tasks intent: {q!r}"


# ---------------------------------------------------------------------------
# NEGATIVE — these must NOT fire (they're RAG/topic queries)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "q",
    [
        # Q5 from the bench — the bug that motivated this fix
        "qué hay sobre comunicación no violenta",
        # Variants of "qué hay" + topic
        "qué hay sobre ikigai",
        "qué hay nuevo en mi vault",
        "qué hay de música nueva",
        # "qué tengo" + non-task qualifier
        "qué tengo en notas sobre productividad",
        "qué tengo escrito sobre liderazgo",
        # Topic queries with verb "hacer" (used to false-positive on bare "qué hacer")
        "qué hacer con la nota X",
        "qué dice mi nota sobre coaching",
        # Generic topic
        "información sobre guitarras",
        "letra de muros fractales",
        "comandos CLI de claude code",
        "qué es ikigai",
        "mis referentes en coaching",
        # Empty / edge
        "",
        "  ",
        "hola",
        "gracias",
        "ok",
    ],
)
def test_tasks_intent_does_not_fire(q: str) -> None:
    assert not _is_tasks_query(q), f"should NOT fire on topic query: {q!r}"
