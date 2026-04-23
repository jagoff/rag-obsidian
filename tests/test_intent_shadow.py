"""Tests del LLM intent shadow mode (Opción C Fase 0, 2026-04-23).

Shadow mode corre el LLM classifier en paralelo al regex sin cambiar
routing, logueando ambas predicciones a `extra_json.intent_shadow` para
medir agreement antes de decidir si Opción C (router LLM full) vale.

Cobertura:
  • compute_intent_shadow: flag on/off, empty question, invalid regex,
    LLM hit/miss/timeout, agree/disagree flag, latency_ms capture.
  • ChatTurnResult.intent_shadow field + to_log_event emite intent_shadow.
  • run_chat_turn propaga _intent_shadow al result en todos los 3 paths
    (empty retrieve, cache hit, normal end).
"""
from __future__ import annotations

import pytest

import rag


# ── compute_intent_shadow: core helper ─────────────────────────────────────


def test_shadow_disabled_returns_none(monkeypatch):
    """Sin RAG_LLM_INTENT_SHADOW=1 → None, no LLM call fired."""
    monkeypatch.delenv("RAG_LLM_INTENT_SHADOW", raising=False)
    called = {"n": 0}

    def _spy(q, model=None):
        called["n"] += 1
        return "semantic"

    monkeypatch.setattr(rag, "_classify_intent_llm", _spy)
    assert rag.compute_intent_shadow("cualquier cosa", "semantic") is None
    assert called["n"] == 0, "LLM must not fire when shadow disabled"


def test_shadow_enabled_agree(monkeypatch):
    monkeypatch.setenv("RAG_LLM_INTENT_SHADOW", "1")
    monkeypatch.setattr(rag, "_classify_intent_llm",
                        lambda q, model=None: "count")
    out = rag.compute_intent_shadow("cuántas notas tengo", "count")
    assert out is not None
    assert out["llm"] == "count"
    assert out["regex"] == "count"
    assert out["agree"] is True
    assert out["llm_timed_out"] is False
    assert isinstance(out["latency_ms"], int)


def test_shadow_enabled_disagree(monkeypatch):
    """LLM upgrade a intent más específica — disagree flag + llm_timed_out=False."""
    monkeypatch.setenv("RAG_LLM_INTENT_SHADOW", "1")
    monkeypatch.setattr(rag, "_classify_intent_llm",
                        lambda q, model=None: "entity_lookup")
    out = rag.compute_intent_shadow("qué sé de María", "semantic")
    assert out["llm"] == "entity_lookup"
    assert out["regex"] == "semantic"
    assert out["agree"] is False
    assert out["llm_timed_out"] is False


def test_shadow_llm_timeout(monkeypatch):
    """LLM devuelve None (timeout/JSON error) → llm_timed_out=True, agree=False."""
    monkeypatch.setenv("RAG_LLM_INTENT_SHADOW", "1")
    monkeypatch.setattr(rag, "_classify_intent_llm",
                        lambda q, model=None: None)
    out = rag.compute_intent_shadow("q", "count")
    assert out["llm"] is None
    assert out["agree"] is False
    assert out["llm_timed_out"] is True
    assert out["regex"] == "count"


def test_shadow_llm_raises_caught(monkeypatch):
    """Si el LLM helper raisea (ollama caído), shadow degrada a timeout."""
    monkeypatch.setenv("RAG_LLM_INTENT_SHADOW", "1")

    def _boom(q, model=None):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(rag, "_classify_intent_llm", _boom)
    out = rag.compute_intent_shadow("q", "semantic")
    assert out is not None, "exception must NOT propagate"
    assert out["llm"] is None
    assert out["llm_timed_out"] is True


def test_shadow_empty_question_returns_none(monkeypatch):
    monkeypatch.setenv("RAG_LLM_INTENT_SHADOW", "1")
    assert rag.compute_intent_shadow("", "semantic") is None
    assert rag.compute_intent_shadow("   ", "semantic") is None


def test_shadow_invalid_regex_intent_returns_none(monkeypatch):
    """Caller pasa regex_intent vacío → None (guarda contra misuse)."""
    monkeypatch.setenv("RAG_LLM_INTENT_SHADOW", "1")
    assert rag.compute_intent_shadow("q", "") is None
    assert rag.compute_intent_shadow("q", None) is None  # type: ignore[arg-type]


def test_shadow_flag_accepts_variants(monkeypatch):
    """RAG_LLM_INTENT_SHADOW acepta 1/true/yes (case-insensitive)."""
    monkeypatch.setattr(rag, "_classify_intent_llm",
                        lambda q, model=None: "semantic")
    for value in ("1", "true", "TRUE", "yes", "YES"):
        monkeypatch.setenv("RAG_LLM_INTENT_SHADOW", value)
        assert rag.compute_intent_shadow("q", "semantic") is not None, (
            f"value={value!r} should enable shadow"
        )
    for value in ("", "0", "false", "no", "off"):
        monkeypatch.setenv("RAG_LLM_INTENT_SHADOW", value)
        assert rag.compute_intent_shadow("q", "semantic") is None, (
            f"value={value!r} should keep shadow off"
        )


# ── ChatTurnResult integration ─────────────────────────────────────────────


def test_chat_turn_result_default_intent_shadow_none():
    rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=-float("inf"),
        search_query="q", query_variants=["q"],
    )
    r = rag.ChatTurnResult(answer="a", retrieve_result=rr, question="q")
    assert r.intent_shadow is None


def test_chat_turn_result_to_log_event_emits_intent_shadow():
    rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=-float("inf"),
        search_query="q", query_variants=["q"],
    )
    shadow = {
        "llm": "entity_lookup", "regex": "semantic",
        "agree": False, "latency_ms": 123, "llm_timed_out": False,
    }
    r = rag.ChatTurnResult(
        answer="a", retrieve_result=rr, question="q",
        intent_shadow=shadow,
    )
    ev = r.to_log_event(cmd="chat", session_id="test")
    assert ev["intent_shadow"] == shadow


# ── run_chat_turn integration: 3 paths (empty vaults / empty retrieve /
#    cache hit / normal end). Las 3 deben propagar intent_shadow cuando el
#    flag está ON, sin afectar routing.


def test_run_chat_turn_empty_retrieve_propagates_shadow(monkeypatch):
    monkeypatch.setenv("RAG_LLM_INTENT_SHADOW", "1")
    monkeypatch.setattr(rag, "_classify_intent_llm",
                        lambda q, model=None: "entity_lookup")
    monkeypatch.setattr(rag, "_semantic_cache_enabled", lambda: False)

    # Empty retrieve → short-circuit antes de ollama.chat.
    empty_rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=-float("inf"),
        search_query="q", query_variants=["q"],
    )
    monkeypatch.setattr(rag, "multi_retrieve", lambda *a, **kw: empty_rr)

    from pathlib import Path
    req = rag.ChatTurnRequest(
        question="qué sé de María",
        vaults=[("test", Path("/tmp/fake"))],
    )
    result = rag.run_chat_turn(req)
    assert result.answer == "Sin resultados relevantes."
    assert result.intent_shadow is not None
    assert result.intent_shadow["llm"] == "entity_lookup"
    assert result.intent_shadow["regex"] == "semantic"
    assert result.intent_shadow["agree"] is False


def test_run_chat_turn_shadow_none_when_disabled(monkeypatch):
    monkeypatch.delenv("RAG_LLM_INTENT_SHADOW", raising=False)
    empty_rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=-float("inf"),
        search_query="q", query_variants=["q"],
    )
    monkeypatch.setattr(rag, "multi_retrieve", lambda *a, **kw: empty_rr)
    monkeypatch.setattr(rag, "_semantic_cache_enabled", lambda: False)

    from pathlib import Path
    req = rag.ChatTurnRequest(
        question="q", vaults=[("t", Path("/tmp/fake"))],
    )
    result = rag.run_chat_turn(req)
    assert result.intent_shadow is None


def test_run_chat_turn_shadow_never_changes_intent(monkeypatch):
    """CRITICAL invariant: shadow mode NUNCA altera el `intent` final.
    Ese campo sigue el regex result — solo intent_shadow refleja la
    disagreement. Sin esto, shadow seria un hidden routing change."""
    monkeypatch.setenv("RAG_LLM_INTENT_SHADOW", "1")
    # LLM clasifica distinto al regex ("semantic").
    monkeypatch.setattr(rag, "_classify_intent_llm",
                        lambda q, model=None: "count")
    empty_rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=-float("inf"),
        search_query="q", query_variants=["q"],
    )
    monkeypatch.setattr(rag, "multi_retrieve", lambda *a, **kw: empty_rr)
    monkeypatch.setattr(rag, "_semantic_cache_enabled", lambda: False)

    from pathlib import Path
    req = rag.ChatTurnRequest(
        question="cualquier cosa",
        vaults=[("t", Path("/tmp/fake"))],
    )
    result = rag.run_chat_turn(req)
    # intent sigue siendo lo que devolvió el regex (algo ≠ 'count').
    assert result.intent != "count", (
        "shadow MUST NOT change routing — intent debe venir del regex"
    )
    # Pero el shadow SÍ refleja el disagreement.
    assert result.intent_shadow["llm"] == "count"
    assert result.intent_shadow["agree"] is False
