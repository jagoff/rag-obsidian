"""Tests for `classify_intent` + `system_prompt_for_intent`.

Focus is on the comparison/synthesis intents wired 2026-04-21 (previously
extension-point prompts emitted by nobody). Also anchors existing intents
so future edits don't regress count/list/recent detection.
"""
from __future__ import annotations

import pytest

import rag


TAGS: set[str] = set()
FOLDERS: set[str] = set()


# ── Count / list / recent (unchanged baselines) ────────────────────────

@pytest.mark.parametrize("q", [
    "cuántas notas tengo sobre coaching?",
    "cuantas reuniones hubo esta semana",
    "how many projects do I have",
])
def test_count_intent(q):
    intent, _ = rag.classify_intent(q, TAGS, FOLDERS)
    assert intent == "count", f"expected count for {q!r}, got {intent}"


@pytest.mark.parametrize("q", [
    "listame todas las notas del area personal",
    "mostrame notas de coaching",
    "qué notas tengo sobre X",
    "dame las notas de esta semana",
])
def test_list_intent(q):
    intent, _ = rag.classify_intent(q, TAGS, FOLDERS)
    assert intent == "list"


@pytest.mark.parametrize("q", [
    "notas recientes de esta semana",
    "qué modificé hoy",
    "últimas notas sobre rag",
])
def test_recent_intent(q):
    intent, _ = rag.classify_intent(q, TAGS, FOLDERS)
    assert intent == "recent"


# ── Agenda (new intent, 2026-04-21 evening) ───────────────────────────
#
# Distingue "qué tengo esta semana" (browsing de agenda temporal) de
# "últimas notas modificadas" (recent). Antes caían ambos en `recent`
# porque `_INTENT_RECENT_RE` captura "esta semana" / "hoy" / "este mes"
# indistintamente. Post-calendar-ingest ese overlap se vuelve un bug:
# el handler `handle_recent` itera notas del vault ordenadas por modified
# desc y se pierde completamente los calendar events + reminders.

@pytest.mark.parametrize("q", [
    # Posesivos: "mi agenda", "mis eventos", "mis turnos"
    "mi agenda",
    "mis eventos",
    "mi calendario",
    "mis reuniones",
    "mis turnos",
    "mis citas",
    # "qué tengo X" (temporal)
    "qué tengo hoy",
    "qué tengo mañana",
    "qué tengo esta semana",
    "qué tengo este mes",
    "qué tengo el viernes",
    "qué tengo el próximo lunes",
    # "qué hay X" (temporal / programado)
    "qué hay hoy",
    "qué hay programado para hoy",
    "qué tenemos mañana",
    # Event noun + tengo/hay
    "qué eventos tengo mañana",
    "qué reuniones tengo esta semana",
    "qué turnos tengo",
    "qué citas hay el viernes",
    # Agenda de/para X
    "agenda de hoy",
    "agenda del viernes",
    "agenda para mañana",
    "agenda de esta semana",
])
def test_agenda_intent_fires(q):
    intent, _ = rag.classify_intent(q, TAGS, FOLDERS)
    assert intent == "agenda", f"expected agenda for {q!r}, got {intent}"


@pytest.mark.parametrize("q", [
    # Lookups específicos de eventos: son semantic, no agenda browsing.
    # El retrieve() source-weighted + reranker resuelve estos bien sin
    # necesitar el intent dedicado.
    "cuándo es el workshop de AI Engineer",
    "turno con martín segal psicólogo",
    "workshop de AI Engineer sobre sistemas RAG",
    "despedida de jardín de astor",
    # Notas modificadas (vault-specific): stay recent.
    "notas recientes de esta semana",
    "qué modifiqué hoy",
    "notas modificadas esta semana",
    "últimas notas sobre rag",
    # Plain semantic — sin signal de browsing temporal.
    "qué es ikigai",
    "comandos CLI de claude code",
])
def test_agenda_does_not_fire_on_specific_or_vault_queries(q):
    """False-positive guard: agenda intent is narrow — only browsing
    patterns ('qué tengo X', 'mi agenda', etc). Specific lookups like
    'cuándo es el workshop' stay semantic; 'notas modificadas' stays
    recent."""
    intent, _ = rag.classify_intent(q, TAGS, FOLDERS)
    assert intent != "agenda", f"{q!r} should not fire agenda, got {intent}"


def test_agenda_precedence_over_recent():
    """'qué tengo esta semana' used to match `_INTENT_RECENT_RE` via the
    'esta semana' token. With agenda firing first, the query now routes
    to the calendar/reminders handler instead of listing vault notes
    sorted by modified desc — the precise bug flagged in the 2026-04-21
    evening session."""
    intent, _ = rag.classify_intent("qué tengo esta semana", TAGS, FOLDERS)
    assert intent == "agenda"


# ── Comparison ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "diferencia entre RAG y fine-tuning",
    "diferencias entre qwen y command-r",
    "compará bge-m3 con nomic-embed",
    "comparar claude y gpt-4",
    "comparame sqlite-vec vs qdrant",
    "qwen3 vs qwen2.5",
    "qwen3 versus qwen2.5",
    "compare LoRA and QLoRA",
    "en qué se diferencian RAG y cache-augmented generation",
    "qué distingue a BM25 de embeddings",
    "contraste entre rerank y retrieval puro",
])
def test_comparison_intent_fires(q):
    intent, _ = rag.classify_intent(q, TAGS, FOLDERS)
    assert intent == "comparison", f"expected comparison for {q!r}, got {intent}"


# ── Synthesis ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "resumí todo lo que hay sobre coaching",
    "resumime lo que sé sobre obsidian-rag",
    "resumen de lo que tengo sobre ollama",
    "síntesis de mis notas sobre RAG",
    "sintetizame lo que hay sobre metaprogramación",
    "integrame todo lo que tengo sobre ranker-vivo",
    "qué dice el vault sobre episódico",
    "qué hay en el vault sobre coaching narrativo",
    "todo lo que hay sobre bge-m3",
    "todo lo que tengo sobre ETL",
    "summary of my notes on RAG",
    "synthesis of the project notes",
    "synthesize everything about ollama",
])
def test_synthesis_intent_fires(q):
    intent, _ = rag.classify_intent(q, TAGS, FOLDERS)
    assert intent == "synthesis", f"expected synthesis for {q!r}, got {intent}"


# ── Semantic fallback (the anti-false-positive anchor) ─────────────────

@pytest.mark.parametrize("q", [
    "qué es RAG",                              # plain definition — stays semantic
    "cómo funciona el reranker",
    "donde quedó la nota del último standup",
    "porqué command-r falla con JSON",
    "aspectos clave del coaching narrativo",   # no aggregation cue
    "la última decisión sobre ollama",         # "última" alone isn't "recent"
])
def test_plain_questions_stay_semantic(q):
    intent, _ = rag.classify_intent(q, TAGS, FOLDERS)
    assert intent == "semantic", f"{q!r} should stay semantic, got {intent}"


# ── Comparison beats synthesis when both could match ──────────────────

def test_comparison_precedence_over_synthesis():
    # "compará X y Y" includes neither "resumen" nor "síntesis" —
    # comparison cleanly wins. Harder case: "comparar los resúmenes".
    intent, _ = rag.classify_intent(
        "comparar los resúmenes de qwen y command-r", TAGS, FOLDERS,
    )
    assert intent == "comparison"


# ── system_prompt_for_intent routing ───────────────────────────────────

def test_system_prompt_routing_semantic():
    assert rag.system_prompt_for_intent("semantic", loose=False) == rag.SYSTEM_RULES_STRICT


def test_system_prompt_routing_lookup_family():
    # agenda joins count/list/recent in the lookup family — all four are
    # "list-style" intents that bypass the prose LLM path and render a
    # file list directly; SYSTEM_RULES_LOOKUP is only used when the LLM
    # IS consulted (e.g. --loose override). Keeping agenda in this tuple
    # means a loose-fallback render routes through the terse lookup
    # prompt, not the strict semantic one.
    for i in ("count", "list", "recent", "agenda"):
        assert rag.system_prompt_for_intent(i, loose=False) == rag.SYSTEM_RULES_LOOKUP


def test_system_prompt_routing_synthesis():
    assert rag.system_prompt_for_intent("synthesis", loose=False) == rag.SYSTEM_RULES_SYNTHESIS


def test_system_prompt_routing_comparison():
    assert rag.system_prompt_for_intent("comparison", loose=False) == rag.SYSTEM_RULES_COMPARISON


def test_system_prompt_loose_overrides_all_intents():
    # --loose flips every intent to SYSTEM_RULES (ext marker allowance).
    for i in ("semantic", "count", "list", "recent", "agenda", "synthesis", "comparison"):
        assert rag.system_prompt_for_intent(i, loose=True) == rag.SYSTEM_RULES


def test_system_prompt_unknown_intent_defaults_to_strict():
    # Defensive: if a future intent sneaks in without prompt routing, fall
    # back to the most restrictive default.
    assert rag.system_prompt_for_intent("quantum-magic", loose=False) == rag.SYSTEM_RULES_STRICT


# ── Narrow false-positive guards ──────────────────────────────────────

def test_comparison_does_not_fire_on_plural_diferencia():
    # "las diferencias" alone without "entre" shouldn't fire.
    intent, _ = rag.classify_intent(
        "qué son las diferencias en un diff de git", TAGS, FOLDERS,
    )
    assert intent == "semantic"


def test_synthesis_does_not_fire_on_plain_resume_verb():
    # "resumiendo" mid-sentence isn't the trigger — it needs
    # "resumí de/sobre/acerca de" to fire. This guards against
    # aggressive verb matching that would turn every "resumen:"
    # in a query into synthesis.
    intent, _ = rag.classify_intent(
        "cuál fue el resumen ejecutivo que escribí ayer", TAGS, FOLDERS,
    )
    assert intent == "semantic"
