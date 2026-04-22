"""Tests for Fase B: skip reformulate en metadata-only intents (Improvement #3).

Scope: verificar el helper _should_skip_reformulate respeta el flag +
_METADATA_ONLY_INTENTS. Tests a nivel unit (no integration con click commands).
"""
from __future__ import annotations

import pytest

import rag


@pytest.mark.parametrize("intent", ["count", "list", "recent", "agenda", "entity_lookup"])
def test_skip_when_adaptive_on_and_metadata_intent(monkeypatch, intent):
    """Adaptive ON + metadata intent → skip."""
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "1")
    monkeypatch.delenv("RAG_FORCE_FULL_PIPELINE", raising=False)
    assert rag._should_skip_reformulate(intent) is True


@pytest.mark.parametrize("intent", ["count", "list", "recent", "agenda", "entity_lookup", "semantic"])
def test_no_skip_when_adaptive_off(monkeypatch, intent):
    """Adaptive explícitamente OFF (rollback vía `RAG_ADAPTIVE_ROUTING=0`)
    → pipeline legacy bit-idéntico: nunca skippea, cualquier intent.

    Pre 2026-04-22 el default era OFF y este test validaba el happy path.
    Post-flip, para mantener la semántica del test (legacy behaviour),
    seteamos el env explícito en 0. El default ON se cubre en
    `test_adaptive_routing_default.py` y en el test parametrizado de
    arriba (`test_skip_when_adaptive_on_and_metadata_intent`)."""
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "0")
    monkeypatch.delenv("RAG_FORCE_FULL_PIPELINE", raising=False)
    assert rag._should_skip_reformulate(intent) is False


@pytest.mark.parametrize("intent", ["semantic", "comparison", "synthesis"])
def test_no_skip_for_non_metadata_intent(monkeypatch, intent):
    """Adaptive ON + non-metadata intent → no skip."""
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "1")
    monkeypatch.delenv("RAG_FORCE_FULL_PIPELINE", raising=False)
    assert rag._should_skip_reformulate(intent) is False


def test_force_full_pipeline_overrides(monkeypatch):
    """RAG_FORCE_FULL_PIPELINE=1 → _adaptive_routing() False → no skip."""
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "1")
    monkeypatch.setenv("RAG_FORCE_FULL_PIPELINE", "1")
    assert rag._should_skip_reformulate("count") is False


def test_metadata_only_intents_frozen_set():
    """Set inmutable."""
    assert isinstance(rag._METADATA_ONLY_INTENTS, frozenset)
    assert "count" in rag._METADATA_ONLY_INTENTS
    assert "list" in rag._METADATA_ONLY_INTENTS
    assert "recent" in rag._METADATA_ONLY_INTENTS
    assert "agenda" in rag._METADATA_ONLY_INTENTS
    assert "entity_lookup" in rag._METADATA_ONLY_INTENTS
    assert "semantic" not in rag._METADATA_ONLY_INTENTS
    assert "comparison" not in rag._METADATA_ONLY_INTENTS
    assert "synthesis" not in rag._METADATA_ONLY_INTENTS
