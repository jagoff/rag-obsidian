"""Tests para el reranker fast-path (audit 2026-04-25 finding R2-1 #2).

La lógica del fast-path se aísla en `rag._should_skip_rerank()`. Si esa
función devuelve True, el flujo principal de `retrieve()` salteará el
cross-encoder y asignará scores sintéticos basados en el orden RRF.

Estos tests cubren la decisión, no el flujo end-to-end del retrieve.
Validamos cada rama del decision tree:

- 3 señales convergentes (sem, BM25, RRF acuerdan) + cosine bajo → skip
- Cosine alto → no skip (rerank importa para borderline)
- Sem y BM25 disienten → no skip
- RRF top distinto a sem o BM25 → no skip (debería ser raro)
- Distance None → no skip (no podemos confiar)
- Env var = 0 → no skip (escape hatch)
"""
from __future__ import annotations

import pytest

import rag


def test_skip_when_all_three_agree_and_distance_low():
    """Caso happy: sem, BM25 y RRF acuerdan en top-1 con cosine 0.05."""
    assert rag._should_skip_rerank(
        rrf_top_id="A",
        sem_top_id="A",
        bm25_top_id="A",
        sem_top_distance=0.05,
    ) is True


def test_dont_skip_when_distance_above_threshold():
    """Cosine 0.50 > 0.10 default → rerank decide aunque haya consenso."""
    assert rag._should_skip_rerank(
        rrf_top_id="A",
        sem_top_id="A",
        bm25_top_id="A",
        sem_top_distance=0.50,
    ) is False


def test_dont_skip_when_distance_at_exact_threshold(monkeypatch):
    """Distance == threshold debe contar como skip (≤, no <)."""
    monkeypatch.setenv("RAG_RERANK_FASTPATH_DIST", "0.10")
    assert rag._should_skip_rerank(
        rrf_top_id="A",
        sem_top_id="A",
        bm25_top_id="A",
        sem_top_distance=0.10,
    ) is True


def test_dont_skip_when_sem_and_bm25_disagree():
    """Sin consenso entre sem y BM25 (sem dice A, BM25 dice B) → rerank
    decide cuál gana. Caso típico de queries con keyword vs semántico."""
    assert rag._should_skip_rerank(
        rrf_top_id="A",
        sem_top_id="A",
        bm25_top_id="B",
        sem_top_distance=0.05,
    ) is False


def test_dont_skip_when_rrf_disagrees_with_sem():
    """Si RRF eligió un top distinto a sem (puede pasar si BM25 era muy
    fuerte y dominó el RRF) → rerank, porque no tenemos las 3 señales
    convergentes."""
    assert rag._should_skip_rerank(
        rrf_top_id="C",  # RRF eligió un 3rd-party
        sem_top_id="A",
        bm25_top_id="B",
        sem_top_distance=0.05,
    ) is False


def test_dont_skip_when_distance_is_none():
    """Si chromadb no devolvió distances (caso degenerado), no podemos
    aplicar el fast-path — debemos rerank por seguridad."""
    assert rag._should_skip_rerank(
        rrf_top_id="A",
        sem_top_id="A",
        bm25_top_id="A",
        sem_top_distance=None,
    ) is False


def test_dont_skip_when_any_top_id_missing():
    """Si alguno de los retrievers vino vacío, el fast-path no aplica."""
    assert rag._should_skip_rerank(
        rrf_top_id=None, sem_top_id="A", bm25_top_id="A",
        sem_top_distance=0.05,
    ) is False
    assert rag._should_skip_rerank(
        rrf_top_id="A", sem_top_id=None, bm25_top_id="A",
        sem_top_distance=0.05,
    ) is False
    assert rag._should_skip_rerank(
        rrf_top_id="A", sem_top_id="A", bm25_top_id=None,
        sem_top_distance=0.05,
    ) is False


def test_env_var_zero_disables_fast_path(monkeypatch):
    """RAG_RERANK_FASTPATH_DIST=0 deshabilita totalmente — escape hatch
    para regression testing o A/B comparison."""
    monkeypatch.setenv("RAG_RERANK_FASTPATH_DIST", "0")
    # Caso happy normalmente, pero env var lo overridea.
    assert rag._should_skip_rerank(
        rrf_top_id="A",
        sem_top_id="A",
        bm25_top_id="A",
        sem_top_distance=0.01,  # extremely low cosine
    ) is False


def test_env_var_higher_threshold_allows_more_skips(monkeypatch):
    """Threshold custom 0.30 permite skipear cosine 0.20 (que con default
    0.10 NO skipearía). Útil para latency-sensitive surfaces que aceptan
    más false-positives en orden."""
    monkeypatch.setenv("RAG_RERANK_FASTPATH_DIST", "0.30")
    assert rag._should_skip_rerank(
        rrf_top_id="A",
        sem_top_id="A",
        bm25_top_id="A",
        sem_top_distance=0.20,
    ) is True


def test_default_threshold_is_010(monkeypatch):
    """Sin env var, el default es 0.10. Documenta la decisión por escrito
    para que cambios accidentales sean visibles."""
    monkeypatch.delenv("RAG_RERANK_FASTPATH_DIST", raising=False)
    # Cosine 0.11 NO debe skip con default 0.10
    assert rag._should_skip_rerank(
        rrf_top_id="A", sem_top_id="A", bm25_top_id="A",
        sem_top_distance=0.11,
    ) is False
    # Cosine 0.10 SÍ debe skip (≤)
    assert rag._should_skip_rerank(
        rrf_top_id="A", sem_top_id="A", bm25_top_id="A",
        sem_top_distance=0.10,
    ) is True
