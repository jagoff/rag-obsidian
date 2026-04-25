"""Tests for adaptive routing feature flags and constants (Improvement #3 Fase A).

Scope: verificar que los env vars se leen correctamente y que _adaptive_routing()
respeta la semántica ON-default (post 2026-04-22 flip) + rollback via
RAG_ADAPTIVE_ROUTING=0.  NO testea side-effects en retrieve()/expand_queries()
— esos tests viven en test_adaptive_fast_path.py / test_adaptive_metadata_skip.py.
"""
from __future__ import annotations

import importlib


def test_adaptive_routing_default_on(monkeypatch):
    """Post 2026-04-22 el default pasa a ON.  Sin env explícito, True.
    Pre-flip este test era `default_off`.  Ver
    tests/test_adaptive_routing_default.py para la justificación."""
    monkeypatch.delenv("RAG_ADAPTIVE_ROUTING", raising=False)
    import rag
    assert rag._adaptive_routing() is True


def test_adaptive_routing_enabled_by_env(monkeypatch):
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "1")
    import rag
    assert rag._adaptive_routing() is True


def test_adaptive_routing_disabled_values(monkeypatch):
    """Rollback explícito vía valores falsy estrictos."""
    for val in ("0", "false", "no"):
        monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", val)
        import rag
        assert rag._adaptive_routing() is False, f"RAG_ADAPTIVE_ROUTING={val!r} should be OFF"


def test_adaptive_routing_empty_string_is_on(monkeypatch):
    """String vacío (launchd plists sin el env) se interpreta como "no seteado".
    Post 2026-04-22 eso implica default ON — pre-flip era OFF."""
    monkeypatch.setenv("RAG_ADAPTIVE_ROUTING", "")
    import rag
    assert rag._adaptive_routing() is True


def test_lookup_threshold_default():
    import rag
    assert rag._LOOKUP_THRESHOLD == 0.6


def test_lookup_threshold_env_override(monkeypatch):
    monkeypatch.setenv("RAG_LOOKUP_THRESHOLD", "0.75")
    import rag
    importlib.reload(rag)
    assert rag._LOOKUP_THRESHOLD == 0.75
    # Restore for other tests
    monkeypatch.delenv("RAG_LOOKUP_THRESHOLD")
    importlib.reload(rag)


def test_lookup_model_default():
    import rag
    assert rag._LOOKUP_MODEL == "qwen2.5:3b"


def test_rerank_pool_by_intent_synthesis():
    import rag
    assert rag._RERANK_POOL_BY_INTENT["synthesis"] == 30


def test_rerank_pool_by_intent_comparison():
    import rag
    assert rag._RERANK_POOL_BY_INTENT["comparison"] == 30


def test_rerank_pool_by_intent_no_other_intents():
    """Solo synthesis + comparison en el dict; semantic/count/list/etc usan default."""
    import rag
    assert set(rag._RERANK_POOL_BY_INTENT.keys()) == {"comparison", "synthesis"}


def test_expand_skip_intents_contains_synthesis_comparison():
    import rag
    assert "synthesis" in rag._EXPAND_SKIP_INTENTS
    assert "comparison" in rag._EXPAND_SKIP_INTENTS


def test_expand_skip_intents_frozenset():
    import rag
    assert isinstance(rag._EXPAND_SKIP_INTENTS, frozenset)


def test_expand_skip_intents_excludes_semantic():
    """Semantic default NO skipea expand — ese es el path rico."""
    import rag
    assert "semantic" not in rag._EXPAND_SKIP_INTENTS


def test_graph_always_intents_synthesis_comparison():
    import rag
    assert rag._GRAPH_ALWAYS_INTENTS == frozenset({"synthesis", "comparison"})


def test_adaptive_helper_is_callable():
    import rag
    assert callable(rag._adaptive_routing)
