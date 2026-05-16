from __future__ import annotations

import time

import pytest

import rag


@pytest.fixture(autouse=True)
def _restore_local_embedder_state():
    old_model = rag._local_embedder
    old_ready = rag._local_embedder_ready.is_set()
    old_last_used = rag._local_embedder_last_used
    old_cache_flag = rag._LOCAL_EMBED_ENABLED_CACHED
    yield
    rag._local_embedder = old_model
    rag._local_embedder_last_used = old_last_used
    rag._LOCAL_EMBED_ENABLED_CACHED = old_cache_flag
    rag._local_embedder_ready.clear()
    if old_ready:
        rag._local_embedder_ready.set()


def test_embed_sets_local_embed_ready_after_lazy_fallback(monkeypatch):
    class _FakeModel:
        def encode(self, texts, **kwargs):
            import numpy as np
            return np.zeros((len(texts), 1024), dtype="float32")

    monkeypatch.setenv("RAG_LOCAL_EMBED", "1")
    monkeypatch.setattr(rag, "_get_local_embedder", lambda: _FakeModel())
    rag._local_embedder_ready.clear()
    rag._embed_cache.clear()

    rag.embed(["lazy-ready-test"])

    assert rag._local_embedder_ready.is_set()


def test_maybe_unload_local_embedder_clears_singleton_after_idle(monkeypatch):
    monkeypatch.setenv("RAG_LOCAL_EMBED_IDLE_TTL", "1")
    rag._local_embedder = object()
    rag._local_embedder_ready.set()
    rag._local_embedder_last_used = time.monotonic() - 2

    assert rag.maybe_unload_local_embedder() is True
    assert rag._local_embedder is None
    assert not rag._local_embedder_ready.is_set()


def test_maybe_unload_local_embedder_respects_idle_ttl(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("RAG_LOCAL_EMBED_IDLE_TTL", "600")
    rag._local_embedder = sentinel
    rag._local_embedder_ready.set()
    rag._local_embedder_last_used = time.monotonic()

    assert rag.maybe_unload_local_embedder() is False
    assert rag._local_embedder is sentinel
    assert rag._local_embedder_ready.is_set()

    rag._local_embedder = None
    rag._local_embedder_ready.clear()
