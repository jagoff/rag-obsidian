"""Tests para el wiring del semantic cache en `run_chat_turn`
(helper compartido — cubre `chat()` CLI turn-1, web turn-1, etc.).

Post 2026-04-23: antes, solo `query()` CLI estaba wireado (150 de 2,335
queries). El helper unificado ahora hace el lookup antes de retrieve y
el store después de post-process, gate-ado por elegibilidad (no history,
single vault, sin source/folder/tag/filter, sin critique/counter).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import rag


@pytest.fixture
def clean_cache_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_CACHE_ENABLED", "1")
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_COSINE", 0.95)
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_DEFAULT_TTL", 86400)
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_MAX_ROWS", 100)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    rag.semantic_cache_clear()
    yield tmp_path
    rag.semantic_cache_clear()


def _emb_vec(*floats: float, dim: int = 1024):
    base = np.zeros(dim, dtype="float32")
    for i, v in enumerate(floats):
        base[i] = v
    if np.linalg.norm(base) == 0:
        base[0] = 1.0
    return base / np.linalg.norm(base)


def _mk_req(**overrides):
    """Build a minimal ChatTurnRequest with a fake single vault."""
    defaults = dict(
        question="qué es ikigai",
        vaults=[("test", Path("/tmp/test-vault"))],
    )
    defaults.update(overrides)
    return rag.ChatTurnRequest(**defaults)


def test_empty_vaults_returns_empty_result_with_cache_fields():
    """req.vaults vacío → ChatTurnResult short-circuit con cache_probe=None."""
    req = rag.ChatTurnRequest(question="q", vaults=[])
    result = rag.run_chat_turn(req)
    assert result.empty_vaults is True
    # Cache fields default a False/None cuando no se llegó a la lógica.
    assert result.cache_hit is False
    assert result.cache_stored is False


def test_cache_hit_short_circuits_llm(clean_cache_env, monkeypatch):
    """Hit del cache debe devolver ChatTurnResult con cache_hit=True SIN
    invocar retrieve() ni multi_retrieve() ni ollama.chat.
    """
    vault = clean_cache_env

    # Seed el cache con una entry matcheable.
    emb = _emb_vec(1.0)
    rag.semantic_cache_store(
        emb, question="q", response="cached response",
        paths=["note.md"], scores=[0.9], top_score=1.2,
        intent="semantic",
        corpus_hash=rag._compute_corpus_hash(_FakeCol(42)),
    )
    # Monkeypatch para que `embed()` devuelva nuestro vector + `get_db_for`
    # devuelva un FakeCol con count=42 (mismo corpus_hash).
    monkeypatch.setattr(rag, "embed", lambda texts: [emb.tolist()])
    monkeypatch.setattr(rag, "get_db_for", lambda _p: _FakeCol(42))

    # Monkeypatch retrieve pipeline para detectar si se llama por error.
    retrieve_called = {"n": 0}
    def _spy_mr(*a, **kw):
        retrieve_called["n"] += 1
        return rag.RetrieveResult(
            docs=[], metas=[], scores=[], confidence=-float("inf"),
            search_query="", query_variants=[],
        )
    monkeypatch.setattr(rag, "multi_retrieve", _spy_mr)

    req = _mk_req(vaults=[("test", vault)])
    result = rag.run_chat_turn(req)

    assert result.cache_hit is True, "hit should short-circuit"
    assert result.answer == "cached response"
    assert result.cache_probe is not None
    assert result.cache_probe["result"] == "hit"
    assert retrieve_called["n"] == 0, "multi_retrieve must NOT be called on hit"


def test_cache_miss_runs_pipeline_normally(clean_cache_env, monkeypatch):
    """Miss → pipeline completo. El probe queda populado con reason='miss'
    para telemetría downstream.
    """
    vault = clean_cache_env
    emb = _emb_vec(1.0)
    monkeypatch.setattr(rag, "embed", lambda texts: [emb.tolist()])
    monkeypatch.setattr(rag, "get_db_for", lambda _p: _FakeCol(42))

    # Devolver retrieve vacío así no disparamos LLM.
    empty_rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=-float("inf"),
        search_query="qué es ikigai", query_variants=["qué es ikigai"],
    )
    monkeypatch.setattr(rag, "multi_retrieve", lambda *a, **kw: empty_rr)

    req = _mk_req(vaults=[("test", vault)])
    result = rag.run_chat_turn(req)

    assert result.cache_hit is False
    assert result.cache_probe is not None
    # Cache vacío para este corpus_hash → reason=corpus_mismatch
    assert result.cache_probe["result"] in ("miss", "hit")
    assert result.cache_probe["reason"] in ("corpus_mismatch", "below_threshold")


def test_cache_skipped_when_history_present(clean_cache_env, monkeypatch):
    """Con history[] poblado (turn 2+), el cache se skippea con reason=flags_skip.

    Rationale: la retrieval de turn 2+ reformulate con el prior turn, así
    que un hit del cache serviría una respuesta que NO usó el contexto
    conversacional — cross-turn staleness clásica.
    """
    vault = clean_cache_env
    empty_rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=-float("inf"),
        search_query="", query_variants=[],
    )
    monkeypatch.setattr(rag, "multi_retrieve", lambda *a, **kw: empty_rr)
    monkeypatch.setattr(rag, "get_db_for", lambda _p: _FakeCol(42))

    req = _mk_req(
        vaults=[("test", vault)],
        history=[{"q": "turn 1", "a": "answer 1", "paths": []}],
    )
    result = rag.run_chat_turn(req)

    assert result.cache_hit is False
    assert result.cache_probe is not None
    assert result.cache_probe["result"] == "skipped"
    assert result.cache_probe["reason"] == "flags_skip"


def test_cache_skipped_when_source_filter(clean_cache_env, monkeypatch):
    """`source='whatsapp'` cambia el candidate pool → cache skippea."""
    vault = clean_cache_env
    empty_rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=-float("inf"),
        search_query="", query_variants=[],
    )
    monkeypatch.setattr(rag, "multi_retrieve", lambda *a, **kw: empty_rr)
    monkeypatch.setattr(rag, "get_db_for", lambda _p: _FakeCol(42))

    req = _mk_req(vaults=[("test", vault)], source="whatsapp")
    result = rag.run_chat_turn(req)

    assert result.cache_hit is False
    assert result.cache_probe["reason"] == "flags_skip"


def test_cache_skipped_when_critique(clean_cache_env, monkeypatch):
    """`critique=True` re-ejecuta el LLM para auto-review; cacheado no aplica."""
    vault = clean_cache_env
    empty_rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=-float("inf"),
        search_query="", query_variants=[],
    )
    monkeypatch.setattr(rag, "multi_retrieve", lambda *a, **kw: empty_rr)
    monkeypatch.setattr(rag, "get_db_for", lambda _p: _FakeCol(42))

    req = _mk_req(vaults=[("test", vault)], critique=True)
    result = rag.run_chat_turn(req)

    assert result.cache_hit is False
    assert result.cache_probe["reason"] == "flags_skip"


def test_cache_skipped_multi_vault(clean_cache_env, monkeypatch):
    """Multi-vault scope (`req.vaults` con ≥2 entries) → skip (single-vault
    only per design — corpus_hash es por col)."""
    vault = clean_cache_env
    empty_rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=-float("inf"),
        search_query="", query_variants=[],
    )
    monkeypatch.setattr(rag, "multi_retrieve", lambda *a, **kw: empty_rr)
    monkeypatch.setattr(rag, "get_db_for", lambda _p: _FakeCol(42))

    req = _mk_req(
        vaults=[("a", vault), ("b", vault)],  # 2 vaults
    )
    result = rag.run_chat_turn(req)

    assert result.cache_hit is False
    assert result.cache_probe["reason"] == "flags_skip"


def test_cache_lookup_disabled_via_request_flag(clean_cache_env, monkeypatch):
    """Caller puede forzar `cache_lookup=False` en el request → probe
    reporta 'cache_lookup_disabled'."""
    vault = clean_cache_env
    empty_rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=-float("inf"),
        search_query="", query_variants=[],
    )
    monkeypatch.setattr(rag, "multi_retrieve", lambda *a, **kw: empty_rr)
    monkeypatch.setattr(rag, "get_db_for", lambda _p: _FakeCol(42))

    req = _mk_req(vaults=[("test", vault)], cache_lookup=False)
    result = rag.run_chat_turn(req)

    assert result.cache_hit is False
    assert result.cache_probe["reason"] == "cache_lookup_disabled"


def test_to_log_event_emits_cache_fields(clean_cache_env):
    """`ChatTurnResult.to_log_event()` debe exponer cache_hit/stored/probe
    en el log event para que caiga en `rag_queries.extra_json`."""
    rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=-float("inf"),
        search_query="q", query_variants=["q"],
    )
    result = rag.ChatTurnResult(
        answer="a", retrieve_result=rr, question="q",
        cache_hit=True,
        cache_probe={"result": "hit", "reason": "match",
                     "top_cosine": 0.99, "candidates": 1},
        cache_stored=False,
    )
    ev = result.to_log_event(cmd="web.chat", session_id="web:abc")
    assert ev["cache_hit"] is True
    assert ev["cache_stored"] is False
    assert ev["cache_probe"]["result"] == "hit"
    assert ev["cache_probe"]["top_cosine"] == 0.99


# ── Helpers ──────────────────────────────────────────────────────────────────


class _FakeCol:
    def __init__(self, count: int):
        self._count = count

    def count(self):
        return self._count
