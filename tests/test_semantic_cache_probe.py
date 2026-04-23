"""Tests para `cache_probe` — instrumentación introducida 2026-04-23.

`semantic_cache_lookup(..., return_probe=True)` devuelve una tupla
`(hit_or_None, probe_dict)` donde el `probe_dict` siempre está populado
con metadata para telemetría downstream. Eso permite responder preguntas
como "por qué no hice hit" vía SQL sobre `rag_queries.extra_json.cache_probe`.

Estos tests cubren cada valor posible de `probe["reason"]` + la
retrocompatibilidad con el API legacy (sin `return_probe`).
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import rag


@pytest.fixture
def clean_cache_env(monkeypatch):
    monkeypatch.setenv("RAG_CACHE_ENABLED", "1")
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_COSINE", 0.95)
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_DEFAULT_TTL", 86400)
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_RECENT_TTL", 600)
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_MAX_ROWS", 100)
    rag.semantic_cache_clear()
    yield
    rag.semantic_cache_clear()


def _emb(*floats: float, dim: int = 64):
    base = np.zeros(dim, dtype="float32")
    for i, v in enumerate(floats):
        base[i] = v
    if np.linalg.norm(base) == 0:
        base[0] = 1.0
    return base / np.linalg.norm(base)


def test_probe_legacy_api_unchanged(clean_cache_env):
    """`return_probe=False` (default) preserva el retorno None/hit_dict.

    Callers pre-existentes que llaman `semantic_cache_lookup(emb, hash)`
    sin pasar `return_probe` siguen recibiendo exactamente el mismo shape.
    """
    emb = _emb(1.0)
    # Miss — legacy devuelve None puro, no tupla.
    assert rag.semantic_cache_lookup(emb, "H") is None

    # Hit — legacy devuelve el dict de hit, no tupla.
    rag.semantic_cache_store(
        emb, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent="semantic", corpus_hash="H",
    )
    hit = rag.semantic_cache_lookup(emb, "H")
    assert hit is not None
    assert isinstance(hit, dict)
    assert hit["response"] == "a"
    # Debe ser dict, no tupla — nadie debería haberse roto por el nuevo param.
    assert not isinstance(hit, tuple)


def test_probe_hit_contract(clean_cache_env):
    """Hit: probe.result='hit', reason='match', top_cosine>=threshold,
    candidates==1."""
    emb = _emb(1.0)
    rag.semantic_cache_store(
        emb, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent="semantic", corpus_hash="H",
    )
    hit, probe = rag.semantic_cache_lookup(emb, "H", return_probe=True)
    assert hit is not None
    assert probe["result"] == "hit"
    assert probe["reason"] == "match"
    assert probe["top_cosine"] is not None
    assert probe["top_cosine"] > 0.99  # identical vector
    assert probe["candidates"] == 1


def test_probe_miss_below_threshold(clean_cache_env):
    """Miss porque el cosine máximo está por debajo del threshold.

    El probe debe reportar `below_threshold` y exponer el cosine real
    alcanzado (para que el operador decida si bajar el threshold).
    """
    emb_a = _emb(1.0, 0.0)
    emb_b = _emb(0.3, 1.0)  # ~0.28 cosine vs emb_a
    rag.semantic_cache_store(
        emb_a, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="H",
    )
    hit, probe = rag.semantic_cache_lookup(emb_b, "H", return_probe=True)
    assert hit is None
    assert probe["result"] == "miss"
    assert probe["reason"] == "below_threshold"
    assert probe["top_cosine"] is not None
    assert probe["top_cosine"] < 0.95  # below default threshold
    assert probe["candidates"] == 1


def test_probe_miss_corpus_mismatch(clean_cache_env):
    """Miss porque no hay filas con ese corpus_hash — i.e. cache vacío
    para el corpus actual."""
    emb = _emb(1.0)
    # Store under a DIFFERENT corpus_hash so the lookup under H2 finds
    # zero candidate rows.
    rag.semantic_cache_store(
        emb, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="H1",
    )
    hit, probe = rag.semantic_cache_lookup(emb, "H2", return_probe=True)
    assert hit is None
    assert probe["result"] == "miss"
    assert probe["reason"] == "corpus_mismatch"
    assert probe["candidates"] == 0


def test_probe_miss_ttl_expired(clean_cache_env, monkeypatch):
    """Miss porque todas las filas tienen TTL expirado. El probe debe
    reflejar `ttl_expired` + contar cuántas filas se skippearon por TTL.
    """
    emb = _emb(1.0)
    rag.semantic_cache_store(
        emb, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent="semantic", corpus_hash="H",
        ttl_seconds=60,  # short TTL
    )
    # Simular que estamos 2 min en el futuro.
    hit, probe = rag.semantic_cache_lookup(
        emb, "H", return_probe=True, now=time.time() + 120,
    )
    assert hit is None
    assert probe["result"] == "miss"
    assert probe["reason"] == "ttl_expired"
    assert probe["skipped_ttl"] == 1


def test_probe_disabled_cache(monkeypatch):
    """Probe con cache deshabilitado: result='disabled', reason='cache_disabled'."""
    monkeypatch.setenv("RAG_CACHE_ENABLED", "0")
    emb = _emb(1.0)
    hit, probe = rag.semantic_cache_lookup(emb, "H", return_probe=True)
    assert hit is None
    assert probe["result"] == "disabled"
    assert probe["reason"] == "cache_disabled"


def test_probe_empty_corpus_hash(clean_cache_env):
    """Probe con corpus_hash vacío (índice ausente / col.count() falló):
    result='empty_corpus_hash', reason='no_corpus_hash'."""
    emb = _emb(1.0)
    hit, probe = rag.semantic_cache_lookup(emb, "", return_probe=True)
    assert hit is None
    assert probe["result"] == "empty_corpus_hash"
    assert probe["reason"] == "no_corpus_hash"


def test_probe_zero_norm_query(clean_cache_env):
    """Probe con query de norma cero (todo en 0): no debe crashear;
    devuelve miss + reason='zero_norm_query'."""
    zero = np.zeros(64, dtype="float32")
    hit, probe = rag.semantic_cache_lookup(zero, "H", return_probe=True)
    assert hit is None
    assert probe["reason"] == "zero_norm_query"
