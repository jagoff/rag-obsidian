"""Tests for MLXBackend idle-unload watchdog (RAG_MLX_IDLE_TTL enforcement).

All tests mock mlx_lm so no real model is loaded.
"""

from __future__ import annotations

import sys
import time
import types
import unittest.mock as mock

import pytest

from rag.llm_backend import MLXBackend, reset_backend, to_mlx


def _make_fake_mlx_lm() -> types.ModuleType:
    m = types.ModuleType("mlx_lm")
    m.load = mock.MagicMock(return_value=(mock.MagicMock(), mock.MagicMock()))
    m.generate = mock.MagicMock(return_value="response")
    m.stream_generate = mock.MagicMock(return_value=iter([]))
    return m


def _make_fake_mlx_core() -> types.ModuleType:
    core = types.ModuleType("mlx.core")
    core.clear_cache = mock.MagicMock()
    core.random = mock.MagicMock()
    return core


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_backend()
    yield
    reset_backend()


@pytest.fixture()
def fake_mlx_modules(monkeypatch):
    fake_lm = _make_fake_mlx_lm()
    fake_core = _make_fake_mlx_core()
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_lm)
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = mock.MagicMock(return_value=None)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    return fake_lm, fake_core


# 1. Watchdog thread starts when TTL > 0
def test_watchdog_thread_starts_when_ttl_positive(monkeypatch, fake_mlx_modules):
    monkeypatch.setenv("RAG_MLX_IDLE_TTL", "60")
    monkeypatch.delenv("RAG_MLX_IDLE_DISABLE", raising=False)

    backend = MLXBackend()
    try:
        assert backend._watchdog_thread is not None
        assert backend._watchdog_thread.is_alive()
        assert backend._watchdog_thread.daemon is True
    finally:
        backend.shutdown_watchdog()


# 2. Watchdog thread does NOT start when TTL = 0
def test_watchdog_thread_not_started_when_ttl_zero(monkeypatch, fake_mlx_modules):
    monkeypatch.setenv("RAG_MLX_IDLE_TTL", "0")

    backend = MLXBackend()
    assert backend._watchdog_thread is None


# 3. Watchdog thread does NOT start when RAG_MLX_IDLE_DISABLE=1
def test_watchdog_thread_not_started_when_disabled(monkeypatch, fake_mlx_modules):
    monkeypatch.setenv("RAG_MLX_IDLE_TTL", "60")
    monkeypatch.setenv("RAG_MLX_IDLE_DISABLE", "1")

    backend = MLXBackend()
    assert backend._watchdog_thread is None


# 4. _evict_idle removes stale models
def test_evict_idle_removes_stale_model(monkeypatch, fake_mlx_modules):
    monkeypatch.setenv("RAG_MLX_IDLE_TTL", "10")
    monkeypatch.setenv("RAG_MLX_IDLE_DISABLE", "1")

    backend = MLXBackend()
    canonical = to_mlx("qwen2.5:3b")

    with backend._loaded_lock:
        backend._loaded[canonical] = (mock.MagicMock(), mock.MagicMock())
        backend._last_used[canonical] = time.monotonic() - 20

    backend._evict_idle()

    assert canonical not in backend._loaded
    assert canonical not in backend._last_used


# 5. _evict_idle keeps fresh models
def test_evict_idle_keeps_fresh_model(monkeypatch, fake_mlx_modules):
    monkeypatch.setenv("RAG_MLX_IDLE_TTL", "10")
    monkeypatch.setenv("RAG_MLX_IDLE_DISABLE", "1")

    backend = MLXBackend()
    canonical = to_mlx("qwen2.5:3b")

    with backend._loaded_lock:
        backend._loaded[canonical] = (mock.MagicMock(), mock.MagicMock())
        backend._last_used[canonical] = time.monotonic() - 5

    backend._evict_idle()

    assert canonical in backend._loaded


# 6. Direct eviction after marking stale
def test_model_evicted_after_marking_stale(monkeypatch, fake_mlx_modules):
    monkeypatch.setenv("RAG_MLX_IDLE_TTL", "2")
    monkeypatch.setenv("RAG_MLX_IDLE_DISABLE", "1")

    backend = MLXBackend()
    canonical = to_mlx("qwen2.5:3b")

    with backend._loaded_lock:
        backend._loaded[canonical] = (mock.MagicMock(), mock.MagicMock())
        backend._last_used[canonical] = time.monotonic() - 3

    backend._evict_idle()

    assert canonical not in backend._loaded


# 7. shutdown_watchdog stops the thread
def test_shutdown_watchdog_stops_thread(monkeypatch, fake_mlx_modules):
    monkeypatch.setenv("RAG_MLX_IDLE_TTL", "60")
    monkeypatch.delenv("RAG_MLX_IDLE_DISABLE", raising=False)

    backend = MLXBackend()
    assert backend._watchdog_thread is not None
    t = backend._watchdog_thread

    backend.shutdown_watchdog()
    assert not t.is_alive()
    assert backend._watchdog_thread is None


# 8. reset_backend calls shutdown_watchdog
def test_reset_backend_shuts_down_watchdog(monkeypatch, fake_mlx_modules):
    from rag.llm_backend import get_backend

    monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")
    monkeypatch.setenv("RAG_MLX_IDLE_TTL", "60")
    monkeypatch.delenv("RAG_MLX_IDLE_DISABLE", raising=False)

    backend = get_backend()
    assert isinstance(backend, MLXBackend)
    t = backend._watchdog_thread

    reset_backend()
    assert t is None or not t.is_alive()


# 9. _last_used bumped on cache hit in _load
def test_last_used_updated_on_cache_hit(monkeypatch, fake_mlx_modules):
    monkeypatch.setenv("RAG_MLX_IDLE_TTL", "60")
    monkeypatch.setenv("RAG_MLX_IDLE_DISABLE", "1")

    backend = MLXBackend()
    canonical = to_mlx("qwen2.5:3b")

    fake_lm, _ = fake_mlx_modules
    fake_lm.load.return_value = (mock.MagicMock(), mock.MagicMock())

    backend._load("qwen2.5:3b")
    t1 = backend._last_used.get(canonical, 0)

    time.sleep(0.05)
    backend._load("qwen2.5:3b")  # cache hit
    t2 = backend._last_used.get(canonical, 0)

    assert t2 > t1


# 10. unload clears _last_used entry
def test_unload_clears_last_used(monkeypatch, fake_mlx_modules):
    monkeypatch.setenv("RAG_MLX_IDLE_TTL", "60")
    monkeypatch.setenv("RAG_MLX_IDLE_DISABLE", "1")

    backend = MLXBackend()
    canonical = to_mlx("qwen2.5:3b")

    with backend._loaded_lock:
        backend._loaded[canonical] = (mock.MagicMock(), mock.MagicMock())
        backend._last_used[canonical] = time.monotonic()

    backend.unload("qwen2.5:3b")

    assert canonical not in backend._loaded
    assert canonical not in backend._last_used
