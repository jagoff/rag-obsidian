"""Tests for cache hit/miss observability registry (2026-04-22).

Valida el contrato de record_cache_event + cache_stats_snapshot + el hook
en embed() + _load_corpus() + semantic_cache_lookup + load_feedback_golden.
"""
from __future__ import annotations

import threading
import pytest

import rag


@pytest.fixture(autouse=True)
def _reset_stats():
    """Cada test arranca con el registry vacío."""
    rag.cache_stats_reset()
    yield
    rag.cache_stats_reset()


# ── 1. record_cache_event primitive ──────────────────────────────────────────


def test_record_hits_only():
    rag.record_cache_event("embed", hits=3)
    snap = rag.cache_stats_snapshot()
    assert snap == {
        "embed": {"hits": 3, "misses": 0, "total": 3, "ratio": 1.0}
    }


def test_record_misses_only():
    rag.record_cache_event("corpus", misses=2)
    snap = rag.cache_stats_snapshot()
    assert snap["corpus"]["hits"] == 0
    assert snap["corpus"]["misses"] == 2
    assert snap["corpus"]["total"] == 2
    assert snap["corpus"]["ratio"] == 0.0


def test_record_accumulates():
    rag.record_cache_event("embed", hits=2, misses=1)
    rag.record_cache_event("embed", hits=3, misses=2)
    snap = rag.cache_stats_snapshot()
    assert snap["embed"] == {"hits": 5, "misses": 3, "total": 8, "ratio": 0.625}


def test_record_zero_no_op():
    """hits=0 miss=0 no crea entry en el registry."""
    rag.record_cache_event("embed", hits=0, misses=0)
    assert rag.cache_stats_snapshot() == {}


def test_snapshot_empty_registry():
    assert rag.cache_stats_snapshot() == {}


def test_snapshot_ratio_zero_total():
    """Si por algún camino hits+misses=0 (no debería pasar), ratio=0 sin div/0."""
    # Forzamos la entry manualmente (no pasa por record_cache_event normal)
    rag._cache_stats["weird"] = {"hits": 0, "misses": 0}
    snap = rag.cache_stats_snapshot()
    assert snap["weird"]["ratio"] == 0.0
    assert snap["weird"]["total"] == 0


def test_snapshot_returns_copy_not_reference():
    """Mutar el snapshot NO afecta el registry interno."""
    rag.record_cache_event("embed", hits=1)
    snap = rag.cache_stats_snapshot()
    snap["embed"]["hits"] = 9999
    snap2 = rag.cache_stats_snapshot()
    assert snap2["embed"]["hits"] == 1  # original intacto


def test_reset_clears():
    rag.record_cache_event("embed", hits=5)
    rag.record_cache_event("corpus", misses=3)
    assert rag.cache_stats_snapshot() != {}
    rag.cache_stats_reset()
    assert rag.cache_stats_snapshot() == {}


# ── 2. Thread-safety ─────────────────────────────────────────────────────────


def test_concurrent_records_no_loss():
    """100 threads × 100 increments = 10000 final. Si no hay locking, el
    count final sería menor por race conditions."""
    def worker():
        for _ in range(100):
            rag.record_cache_event("embed", hits=1)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = rag.cache_stats_snapshot()
    assert snap["embed"]["hits"] == 10000
    assert snap["embed"]["total"] == 10000


# ── 3. embed() instrumentation ───────────────────────────────────────────────


def test_embed_records_hits_and_misses(monkeypatch):
    """Primer embed("abc") → miss. Segundo embed("abc") → hit."""
    # Prime ollama stub
    class _FakeResp:
        embeddings = [[1.0, 2.0, 3.0]]
    monkeypatch.setattr(rag.ollama, "embed", lambda **kw: _FakeResp())

    # Clear the embed cache (autouse en conftest ya lo hace, pero por si acaso)
    rag._embed_cache.clear()

    # Call 1: miss
    out1 = rag.embed(["abc"])
    assert out1 == [[1.0, 2.0, 3.0]]

    # Call 2: hit (from cache)
    out2 = rag.embed(["abc"])
    assert out2 == [[1.0, 2.0, 3.0]]

    snap = rag.cache_stats_snapshot()
    assert snap["embed"]["hits"] == 1
    assert snap["embed"]["misses"] == 1


def test_embed_batch_mixed(monkeypatch):
    """embed(['a', 'b', 'c']) con 'a' en cache → 1 hit + 2 misses en una call."""
    class _FakeResp:
        def __init__(self, embs):
            self.embeddings = embs
    monkeypatch.setattr(rag.ollama, "embed",
                        lambda model, input, keep_alive: _FakeResp(
                            [[float(ord(t))] for t in input]
                        ))
    rag._embed_cache.clear()
    # Pre-populate 'a'
    rag.embed(["a"])
    rag.cache_stats_reset()

    # Batch con 1 hit (a) + 2 misses (b, c)
    rag.embed(["a", "b", "c"])
    snap = rag.cache_stats_snapshot()
    assert snap["embed"]["hits"] == 1
    assert snap["embed"]["misses"] == 2


def test_embed_empty_no_stats(monkeypatch):
    """embed([]) no registra nada."""
    rag.embed([])
    assert rag.cache_stats_snapshot() == {}
