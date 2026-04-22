"""Tests for the name-preservation guardrail — _NAME_PRESERVATION_RULE in
every SYSTEM_RULES* + exposed via system_prompt_for_intent().

Motivation (2026-04-21): user asked about 'Bizarrap' (Argentine music
producer), vault had no musical info, LLM answered refusing about 'Bizarra'
— silently swapping the proper noun to a more common dictionary word. The
fix is a prompt-level rule prepended to every variant, right after REGLA 0
(chunks-as-data) and before the variant-specific rules. These tests are
the regression harness: constant exists, is present in every variant
exactly once, appears AFTER _CHUNK_AS_DATA_RULE, and routes through
system_prompt_for_intent() for every known intent.

No live ollama/retrieval — pure unit tests.
"""
from __future__ import annotations

import pytest

import rag


# ── The constant exists and has the expected shape ─────────────────────────

def test_name_preservation_rule_exists() -> None:
    assert hasattr(rag, "_NAME_PRESERVATION_RULE"), \
        "rag._NAME_PRESERVATION_RULE must exist"
    assert isinstance(rag._NAME_PRESERVATION_RULE, str)
    assert rag._NAME_PRESERVATION_RULE.strip(), "rule must be non-empty"


def test_name_preservation_rule_has_signature_phrasing() -> None:
    rule = rag._NAME_PRESERVATION_RULE
    # Signature tokens the tests guard against regression. If future edits
    # rephrase, update this list — but the intent (Bizarrap anti-correction)
    # must survive.
    assert "REGLA DE NOMBRES PROPIOS" in rule
    assert "TEXTUAL" in rule
    assert "Bizarrap" in rule, \
        "canonical example must stay — it's the regression we're blocking"
    assert "corrij" in rule.lower() or "corrig" in rule.lower(), \
        "rule must talk about NOT correcting"


def test_name_preservation_rule_ends_with_blank_line() -> None:
    # Matches _CHUNK_AS_DATA_RULE convention so concatenation yields a
    # visible separation before the variant-specific body.
    assert rag._NAME_PRESERVATION_RULE.endswith("\n\n")


# ── Present in every SYSTEM_RULES* variant ─────────────────────────────────

_VARIANTS = [
    "SYSTEM_RULES",
    "SYSTEM_RULES_STRICT",
    "SYSTEM_RULES_CHAT",
    "SYSTEM_RULES_WEB",
    "SYSTEM_RULES_LOOKUP",
    "SYSTEM_RULES_SYNTHESIS",
    "SYSTEM_RULES_COMPARISON",
]


@pytest.mark.parametrize("prompt_name", _VARIANTS)
def test_every_variant_contains_name_preservation_rule(prompt_name: str) -> None:
    prompt = getattr(rag, prompt_name)
    assert rag._NAME_PRESERVATION_RULE in prompt, \
        f"{prompt_name} missing _NAME_PRESERVATION_RULE verbatim"
    # Also assert the signature phrase, in case someone paraphrases inline
    # without updating the constant.
    assert "REGLA DE NOMBRES PROPIOS" in prompt, \
        f"{prompt_name} missing 'REGLA DE NOMBRES PROPIOS' marker"
    assert "Bizarrap" in prompt, \
        f"{prompt_name} missing canonical example"


@pytest.mark.parametrize("prompt_name", _VARIANTS)
def test_name_preservation_rule_not_duplicated(prompt_name: str) -> None:
    prompt = getattr(rag, prompt_name)
    # The full constant must appear exactly once — duplicate concatenation
    # would bloat the prompt and confuse the model.
    assert prompt.count(rag._NAME_PRESERVATION_RULE) == 1, \
        f"{prompt_name} has the rule {prompt.count(rag._NAME_PRESERVATION_RULE)} times, expected 1"
    assert prompt.count("REGLA DE NOMBRES PROPIOS") == 1, \
        f"{prompt_name} has the marker phrase more than once"


# ── Ordering invariant: chunks-as-data first, names second, body last ──────

@pytest.mark.parametrize("prompt_name", _VARIANTS)
def test_name_preservation_rule_appears_after_chunk_as_data(prompt_name: str) -> None:
    prompt = getattr(rag, prompt_name)
    idx_chunks = prompt.find("REGLA 0")
    idx_names = prompt.find("REGLA DE NOMBRES PROPIOS")
    assert idx_chunks >= 0, f"{prompt_name} missing REGLA 0"
    assert idx_names >= 0, f"{prompt_name} missing REGLA DE NOMBRES PROPIOS"
    assert idx_chunks < idx_names, \
        f"{prompt_name}: REGLA 0 must precede REGLA DE NOMBRES PROPIOS"


@pytest.mark.parametrize("prompt_name", _VARIANTS)
def test_name_preservation_rule_precedes_variant_body(prompt_name: str) -> None:
    # The name-preservation rule must sit between REGLA 0 and the first
    # variant-specific rule so the body text stays enforceable.
    prompt = getattr(rag, prompt_name)
    idx_names = prompt.find("REGLA DE NOMBRES PROPIOS")
    idx_regla_1 = prompt.find("REGLA 1")
    idx_reglas_block = prompt.find("REGLAS:")  # STRICT/LOOKUP/SYNTHESIS/COMPARISON style
    # At least one of those body markers should exist and come after the rule.
    body_starts = [i for i in (idx_regla_1, idx_reglas_block) if i >= 0]
    if body_starts:
        assert idx_names < min(body_starts), \
            f"{prompt_name}: name rule must precede the variant body"


# ── Routed through system_prompt_for_intent() for every known intent ──────

@pytest.mark.parametrize("intent", [
    "semantic",
    "count",
    "list",
    "recent",
    "agenda",
    "synthesis",
    "comparison",
    # Unknown intents fall back to SYSTEM_RULES_STRICT — still must carry the rule.
    "unknown_fallback",
])
def test_system_prompt_for_intent_carries_both_rules(intent: str) -> None:
    prompt = rag.system_prompt_for_intent(intent, loose=False)
    assert isinstance(prompt, str)
    assert "REGLA 0" in prompt, f"intent={intent!r} lost REGLA 0"
    assert "REGLA DE NOMBRES PROPIOS" in prompt, \
        f"intent={intent!r} lost REGLA DE NOMBRES PROPIOS"
    assert rag._NAME_PRESERVATION_RULE in prompt, \
        f"intent={intent!r} dropped the full name-preservation constant"


@pytest.mark.parametrize("intent", [
    "semantic", "count", "list", "recent", "agenda", "synthesis", "comparison",
])
def test_system_prompt_for_intent_loose_path_carries_both_rules(intent: str) -> None:
    # loose=True collapses to SYSTEM_RULES for every intent — still must
    # carry both rules.
    prompt = rag.system_prompt_for_intent(intent, loose=True)
    assert prompt == rag.SYSTEM_RULES
    assert "REGLA 0" in prompt
    assert "REGLA DE NOMBRES PROPIOS" in prompt
    assert rag._NAME_PRESERVATION_RULE in prompt
