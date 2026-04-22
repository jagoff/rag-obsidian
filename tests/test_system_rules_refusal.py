"""Tests para el refusal explícito en SYSTEM_RULES_SYNTHESIS / COMPARISON.

Auditoría 2026-04-22 sobre `rag.py:11012-11039`:

  SYSTEM_RULES_SYNTHESIS:
    Regla 3: "Cuando ≥2 fuentes se solapan, citalas explícitamente..."
    ← NO dice qué hacer si hay <2 fuentes.
    → El modelo puede inventar síntesis con 1 fuente o agregar prosa
      externa. Output pasa verify_citations() clean porque no hay paths
      inventados, pero la respuesta es hallucinated.

  SYSTEM_RULES_COMPARISON:
    Regla 3: "Si solo hay una fuente relevante, respondé directamente
     sin forzar la estructura."
    ← EXPLICITAMENTE permite respuesta con 1 fuente → hallucination gate.
    Ejemplo patológico: query "diferencia entre stoicism y epicureanism",
    vault tiene notas de stoicism y 0 de epicureanism → modelo responde
    "Stoicism propone X, epicureanism propone Y, la diferencia es...",
    inventando Y desde conocimiento general sin citar.

  SYSTEM_RULES_LOOKUP (baseline a imitar):
    Regla 1: "Si no está cubierta: responder exacto 'No encontré esto
      en el vault.' y cortar."
    ← Refusal explícito, el modelo obedece.

Gap: synthesis/comparison no tienen refusal equivalente. El 22.5% de las
queries en la zona gris `0.015 < top_score < 0.5` (202/1056 en 7 días)
pueden terminar con hallucinations silenciosas.

Cambio 2026-04-22: agregar refusal explícito en ambos prompts. Tests
gatean la presencia de la frase exacta que el LLM debe emitir — si
alguien reescribe el prompt y la remueve, los tests rompen.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


# ── SYNTHESIS refusal ────────────────────────────────────────────────────────


def test_synthesis_prompt_contains_refusal_for_few_sources():
    """SYSTEM_RULES_SYNTHESIS debe decirle al modelo cómo rechazar cuando
    hay <2 fuentes relevantes, en vez de permitirle inventar una síntesis."""
    assert "No hay suficientes fuentes en el vault para sintetizar" \
           in rag.SYSTEM_RULES_SYNTHESIS, \
        "SYNTHESIS prompt must include the explicit refusal phrase for <2 sources"


def test_synthesis_prompt_still_requires_2plus_sources():
    """El prompt debe seguir exigiendo ≥2 fuentes cuando hay síntesis real.
    La regla nueva no anula la vieja, la complementa."""
    assert "≥2 fuentes" in rag.SYSTEM_RULES_SYNTHESIS or \
           "2 o más fuentes" in rag.SYSTEM_RULES_SYNTHESIS, \
        "SYNTHESIS prompt must still require 2+ sources for valid synthesis"


def test_synthesis_prompt_forbids_external_knowledge():
    """Rule 1 no debe debilitarse — sin conocimiento externo."""
    assert "Sin parafraseos, intros ni conocimiento externo" in rag.SYSTEM_RULES_SYNTHESIS


# ── COMPARISON refusal ───────────────────────────────────────────────────────


def test_comparison_prompt_contains_refusal_for_single_source():
    """SYSTEM_RULES_COMPARISON debe incluir refusal explícito cuando hay
    <2 fuentes distintas (antes: permitía 'respondé directamente sin forzar
    la estructura', que era una puerta abierta a hallucinations)."""
    assert "No hay suficientes fuentes en el vault para comparar" \
           in rag.SYSTEM_RULES_COMPARISON, \
        "COMPARISON prompt must include the explicit refusal phrase for <2 sources"


def test_comparison_prompt_no_longer_permits_single_source_response():
    """Regression guard: la frase vieja 'Si solo hay una fuente relevante,
    respondé directamente sin forzar la estructura' era la raíz del
    problema. Verificamos que YA NO está en el prompt."""
    forbidden = "respondé directamente sin forzar la estructura"
    assert forbidden not in rag.SYSTEM_RULES_COMPARISON, \
        f"COMPARISON prompt still contains the hallucination-enabling phrase: {forbidden!r}"


def test_comparison_prompt_keeps_structure_directive():
    """Cuando hay ≥2 fuentes, el prompt debe seguir pidiendo el formato
    '[Fuente A] dice X / [Fuente B] dice Y / Diferencia clave: …'."""
    assert "Diferencia clave" in rag.SYSTEM_RULES_COMPARISON


def test_comparison_prompt_forbids_external_knowledge():
    assert "Sin parafraseos, intros ni conocimiento externo" in rag.SYSTEM_RULES_COMPARISON


# ── Cross-check: LOOKUP baseline not regressed ───────────────────────────────


def test_lookup_prompt_still_has_refusal():
    """LOOKUP ya tenía refusal — este test garantiza que mis cambios no
    lo tocaron."""
    assert "No encontré esto en el vault" in rag.SYSTEM_RULES_LOOKUP


# ── system_prompt_for_intent dispatch sigue funcionando ──────────────────────


def test_system_prompt_for_intent_dispatches_synthesis():
    assert rag.system_prompt_for_intent("synthesis", loose=False) \
        is rag.SYSTEM_RULES_SYNTHESIS


def test_system_prompt_for_intent_dispatches_comparison():
    assert rag.system_prompt_for_intent("comparison", loose=False) \
        is rag.SYSTEM_RULES_COMPARISON


def test_system_prompt_for_intent_loose_always_uses_system_rules():
    """`--loose` override debe seguir funcionando para todos los intents."""
    for intent in ("synthesis", "comparison", "count", "list", "semantic"):
        assert rag.system_prompt_for_intent(intent, loose=True) is rag.SYSTEM_RULES
