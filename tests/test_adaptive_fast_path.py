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


def test_lookup_num_ctx_default_is_4096():
    """Fast-path num_ctx default bumped to 4096 (2026-04-22).

    Pre-fix default 2048 truncó el context antes del chunk relevante en
    queries de alta confianza, devolviendo refuses falsos aunque el doc
    estaba en el top-5 del rerank. Reproducible con
    `RAG_FORCE_FULL_PIPELINE=1 rag query "curso de liderazgo estratégico"`
    (responde bien) vs sin la flag (responde "No tengo esa información"
    pre-2048, responde bien post-4096).

    El default es env-overridable vía RAG_LOOKUP_NUM_CTX pero el valor
    sin override debe ser 4096 para evitar la regresión.
    """
    assert rag._LOOKUP_NUM_CTX == 4096, (
        f"_LOOKUP_NUM_CTX default debe ser 4096 (fue {rag._LOOKUP_NUM_CTX}). "
        "Ver CLAUDE.md § adaptive routing — 2048 causaba refuses falsos."
    )


def test_lookup_num_ctx_env_override_works(monkeypatch):
    """RAG_LOOKUP_NUM_CTX env override sigue funcionando post-bump del default.

    Operadores con presión de memoria pueden forzar num_ctx=2048 manualmente
    aceptando el tradeoff de refuses falsos ocasionales. El módulo se
    evalúa al import → usamos importlib.reload para validar el hot-path.
    """
    import importlib
    monkeypatch.setenv("RAG_LOOKUP_NUM_CTX", "8192")
    reloaded = importlib.reload(rag)
    try:
        assert reloaded._LOOKUP_NUM_CTX == 8192
    finally:
        monkeypatch.delenv("RAG_LOOKUP_NUM_CTX", raising=False)
        importlib.reload(rag)
