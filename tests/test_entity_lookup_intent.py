"""Tests for _INTENT_ENTITY_LOOKUP_RE + classify_intent branch (Improvement #2 Fase B)."""
from __future__ import annotations

import pytest

import rag


TAGS: set[str] = set()
FOLDERS: set[str] = set()


@pytest.mark.parametrize("q", [
    "con quién hablé de ops",
    "con quien hable del proyecto",
    "a quién le mandé el documento",
    "a quién le mande la presentación",
    "a quién le dije que no",
    "qué dice max sobre ops",
    "qué me dijo fernando sobre el rag",
    "qué dijo juan de la reunión",
    "todo lo de juan",
    "todo lo que tengo de max",
    "todos los mensajes de max",
    "todos los mails de erica",
    "todas las notas de seba",
    "todos los chats de fernando",
    "mensajes de juan",
    "mails de fernando",
    "notas de erica",
    "conversaciones con max",
    "correos de seba",
    # 2026-04-22: patrones canónicos de "tell me about person" que faltaban.
    # Son frases que dominan las queries naturales sobre personas; sin estas
    # reglas el classifier los mandaba a `semantic`, gastando 7-8s en
    # retrieve+LLM cuando el handler directo (SQL + metas sort) resuelve en
    # ~200ms. Handler self-gated: si la entidad no existe en rag_entities,
    # returns [] y el caller cae a semantic igual — sin regresión funcional
    # para queries sobre temas abstractos ("qué sabés de React").
    "que sabes de Astor",
    "que sabés de Fernando",
    "qué sabes de max?",
    "qué sabés de juli?",
    "contame de fernando",
    "contame sobre seba",
    "contame acerca de juan",
    "hablame de max",
    "hablame sobre fernando",
    "decime de juan",
    "decime sobre max",
    "información de fernando",
    "informacion sobre max",
])
def test_entity_lookup_fires(q):
    intent, _ = rag.classify_intent(q, TAGS, FOLDERS)
    assert intent == "entity_lookup", f"expected entity_lookup for {q!r}, got {intent}"


@pytest.mark.parametrize("q", [
    "qué es una entidad",
    "qué significa RAG",
    "qué tengo esta semana",
    "mi agenda",
    "notas recientes de juan",
    "últimas notas modificadas",
    "juan vs fernando",
    "diferencia entre max y seba",
])
def test_entity_lookup_does_not_fire(q):
    intent, _ = rag.classify_intent(q, TAGS, FOLDERS)
    assert intent != "entity_lookup", f"entity_lookup should NOT fire for {q!r}, got {intent}"


def test_precedence_recent_wins_over_entity():
    """'notas recientes de juan' → recent (recientes token wins)."""
    intent, _ = rag.classify_intent("notas recientes de juan", TAGS, FOLDERS)
    assert intent == "recent"


def test_precedence_agenda_wins_over_entity():
    """'qué tengo esta semana' → agenda (temporal possessive wins)."""
    intent, _ = rag.classify_intent("qué tengo esta semana", TAGS, FOLDERS)
    assert intent == "agenda"


def test_precedence_comparison_wins_over_entity():
    """'juan vs fernando' → comparison (vs token wins)."""
    intent, _ = rag.classify_intent("juan vs fernando", TAGS, FOLDERS)
    assert intent == "comparison"
