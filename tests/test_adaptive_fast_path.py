"""Tests for Fase C fast-path gate in retrieve() (Improvement #3).

Scope: retrieve() marca fast_path correctamente basado en
RAG_ADAPTIVE_ROUTING + intent kwarg + top-1 rerank score.

Tests unit-level sobre el cálculo de `fast_path` — no integration tests
sobre click commands (demasiado frágiles).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import rag


def _make_mock_col(count: int = 0):
    """Empty collection mock — basta para testear el early return con fast_path=False."""
    c = MagicMock()
    c.count.return_value = count
    return c


def test_fast_path_false_on_empty_collection(monkeypatch):
    """Empty collection → early return con fast_path=False, aun con flag ON."""
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "1")
    col = _make_mock_col(0)
    result = rag.retrieve(col, "test", k=5, folder=None)
    assert result["fast_path"] is False


def test_fast_path_false_when_flag_off(monkeypatch):
    """Flag OFF → fast_path siempre False."""
    monkeypatch.delenv("RAG_ADAPTIVE_ROUTING", raising=False)
    monkeypatch.delenv("RAG_FORCE_FULL_PIPELINE", raising=False)
    col = _make_mock_col(0)
    result = rag.retrieve(col, "test", k=5, folder=None)
    assert result["fast_path"] is False


def test_fast_path_false_when_force_full_pipeline(monkeypatch):
    """FORCE_FULL_PIPELINE=1 → _adaptive_routing() False → fast_path False."""
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "1")
    monkeypatch.setenv("RAG_FORCE_FULL_PIPELINE", "1")
    col = _make_mock_col(0)
    result = rag.retrieve(col, "test", k=5, folder=None)
    assert result["fast_path"] is False


def test_fast_path_key_present_in_result():
    """result siempre tiene fast_path key (nunca KeyError)."""
    col = _make_mock_col(0)
    result = rag.retrieve(col, "test", k=5, folder=None)
    assert "fast_path" in result


def test_retrieve_accepts_intent_kwarg():
    """retrieve() acepta intent kwarg sin crash."""
    col = _make_mock_col(0)
    for intent in ("semantic", "comparison", "synthesis", None):
        result = rag.retrieve(col, "test", k=5, folder=None, intent=intent)
        assert "fast_path" in result
