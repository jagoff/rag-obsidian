"""Tests for the local embedder warmup path.

Motivation: `_maybe_auto_enable_local_embed` (rag.py:6970) turns on
`RAG_LOCAL_EMBED=1` for every query-like CLI subcommand, and every launchd
plist (web, serve) exports the same flag. But the `warmup_async()` and
`serve()` eager-warmup routines only touch the reranker + the ollama-path
embed — they never load the in-process `SentenceTransformer('BAAI/bge-m3')`
singleton. Result: the FIRST query in each fresh process pays the ~5s cold
load of the SentenceTransformer + weights on MPS on the critical path (as
confirmed by `serve` logs showing `embed_ms` of 3455–4898 ms on the first
few turns after a daemon restart, dropping to ~260–400 ms once warm).

This module locks in:

1. `_warmup_local_embedder()` is a no-op when the flag is falsy.
2. `_warmup_local_embedder()` loads the singleton + runs a dummy encode
   when the flag is truthy.
3. Exceptions from load/encode are swallowed (best-effort — must never
   crash the callers).
4. `warmup_async()` invokes the helper on its background thread when the
   flag is truthy.
5. `warmup_async()` skips the helper when the flag is falsy (consistent
   with the existing `_local_embed_enabled()` gate).
"""
import os
import threading
import types

import pytest

import rag


@pytest.fixture(autouse=True)
def _reset_warmup_started(monkeypatch):
    """`warmup_async()` is idempotent via the module-level `_warmup_started`
    flag. Tests that drive warmup must reset it to get a fresh thread spawn.
    """
    monkeypatch.setattr(rag, "_warmup_started", False, raising=False)
    yield


@pytest.fixture
def stub_local_embedder(monkeypatch):
    """Replace `_get_local_embedder()` with a stub that records calls and
    returns a fake model whose `encode()` also records its calls. Avoids
    loading the real 2 GB SentenceTransformer weights during tests.
    """
    calls = {"load": 0, "encode": []}

    class _FakeModel:
        def encode(self, texts, **kwargs):
            calls["encode"].append(list(texts))
            return [[0.0] * 1024 for _ in texts]

    fake = _FakeModel()

    def _stub():
        calls["load"] += 1
        return fake

    monkeypatch.setattr(rag, "_get_local_embedder", _stub)
    return calls


# ── helper contract ───────────────────────────────────────────────────────────


def test_warmup_helper_noop_when_flag_unset(monkeypatch, stub_local_embedder):
    """`RAG_LOCAL_EMBED` unset → helper must not even touch the embedder
    loader (avoids paying the MPS cold load for bulk paths like `rag index`
    that deliberately stay on ollama)."""
    monkeypatch.delenv("RAG_LOCAL_EMBED", raising=False)
    assert rag._warmup_local_embedder() is False
    assert stub_local_embedder["load"] == 0
    assert stub_local_embedder["encode"] == []


@pytest.mark.parametrize("falsy", ["", "0", "false", "no"])
def test_warmup_helper_noop_for_falsy_values(
    monkeypatch, stub_local_embedder, falsy
):
    """Explicit falsy values are respected (user opted out)."""
    monkeypatch.setenv("RAG_LOCAL_EMBED", falsy)
    assert rag._warmup_local_embedder() is False
    assert stub_local_embedder["load"] == 0


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "TRUE"])
def test_warmup_helper_loads_and_encodes_when_enabled(
    monkeypatch, stub_local_embedder, truthy
):
    """Truthy flag → load singleton + one dummy encode to trigger MPS JIT."""
    monkeypatch.setenv("RAG_LOCAL_EMBED", truthy)
    assert rag._warmup_local_embedder() is True
    assert stub_local_embedder["load"] == 1
    assert len(stub_local_embedder["encode"]) == 1


def test_warmup_helper_returns_false_when_loader_returns_none(monkeypatch):
    """`_get_local_embedder()` returns None when HF cache is missing — caller
    must treat that as "skip silently", not as an error."""
    monkeypatch.setenv("RAG_LOCAL_EMBED", "1")
    monkeypatch.setattr(rag, "_get_local_embedder", lambda: None)
    assert rag._warmup_local_embedder() is False


def test_warmup_helper_swallows_loader_exception(monkeypatch):
    """Loader raising must not escape — a failed warmup is just a missed
    opportunity; the retrieve path falls back to ollama embed automatically."""
    monkeypatch.setenv("RAG_LOCAL_EMBED", "1")

    def _raise():
        raise RuntimeError("HF cache missing")

    monkeypatch.setattr(rag, "_get_local_embedder", _raise)
    assert rag._warmup_local_embedder() is False


def test_warmup_helper_swallows_encode_exception(monkeypatch):
    """Encode raising (e.g. MPS OOM) must not escape either."""
    monkeypatch.setenv("RAG_LOCAL_EMBED", "1")

    class _BadModel:
        def encode(self, texts, **kwargs):
            raise RuntimeError("MPS out of memory")

    monkeypatch.setattr(rag, "_get_local_embedder", lambda: _BadModel())
    assert rag._warmup_local_embedder() is False


# ── warmup_async integration ─────────────────────────────────────────────────


def test_warmup_async_invokes_helper_when_enabled(monkeypatch):
    """`warmup_async` runs its body on a daemon thread; the helper must be
    called there when the flag is set. We stub every other heavy call to
    zero-cost no-ops and assert the helper ran within a short timeout."""
    monkeypatch.setenv("RAG_LOCAL_EMBED", "1")
    monkeypatch.delenv("RAG_NO_WARMUP", raising=False)
    # Stub the heavy dependencies so the thread exits fast.
    monkeypatch.setattr(rag, "get_reranker", lambda: None)
    monkeypatch.setattr(rag, "embed", lambda texts: [[0.0] * 1024 for _ in texts])
    monkeypatch.setattr(
        rag,
        "get_db",
        lambda: types.SimpleNamespace(count=lambda: 0),
    )
    monkeypatch.setattr(rag, "_load_corpus", lambda col: None)
    monkeypatch.setattr(rag, "get_pagerank", lambda col: {})

    fired = threading.Event()

    def _spy():
        fired.set()
        return True

    monkeypatch.setattr(rag, "_warmup_local_embedder", _spy)
    rag.warmup_async()
    # The daemon thread is tiny — 2s is generous even on a contended CI.
    assert fired.wait(timeout=2.0), "warmup_async never invoked the helper"


def test_warmup_async_respects_no_warmup_env(monkeypatch):
    """`RAG_NO_WARMUP=1` must short-circuit before spawning the thread —
    we assert the helper is NOT called. Uses a spy on threading.Thread to
    also confirm no thread was launched."""
    monkeypatch.setenv("RAG_LOCAL_EMBED", "1")
    monkeypatch.setenv("RAG_NO_WARMUP", "1")
    fired = threading.Event()
    monkeypatch.setattr(
        rag,
        "_warmup_local_embedder",
        lambda: (fired.set(), True)[1],
    )
    rag.warmup_async()
    # Give the (non-)thread 100 ms to misbehave.
    assert not fired.wait(timeout=0.1), (
        "warmup_async spawned work despite RAG_NO_WARMUP=1"
    )


def test_warmup_async_helper_called_even_when_flag_unset(monkeypatch):
    """When the flag is falsy the helper still runs — and returns False
    internally. The important invariant is that `warmup_async` doesn't
    gate the CALL (so enabling the flag later in the same process still
    benefits from lazy-load on first retrieve). This test pins the
    decision point to `_warmup_local_embedder` itself.
    """
    monkeypatch.delenv("RAG_LOCAL_EMBED", raising=False)
    monkeypatch.delenv("RAG_NO_WARMUP", raising=False)
    monkeypatch.setattr(rag, "get_reranker", lambda: None)
    monkeypatch.setattr(rag, "embed", lambda texts: [[0.0] * 1024 for _ in texts])
    monkeypatch.setattr(
        rag,
        "get_db",
        lambda: types.SimpleNamespace(count=lambda: 0),
    )
    monkeypatch.setattr(rag, "_load_corpus", lambda col: None)
    monkeypatch.setattr(rag, "get_pagerank", lambda col: {})

    called = threading.Event()
    monkeypatch.setattr(
        rag,
        "_warmup_local_embedder",
        lambda: (called.set(), False)[1],
    )
    rag.warmup_async()
    assert called.wait(timeout=2.0), (
        "warmup_async skipped the helper — gate belongs INSIDE the helper"
    )


def test_warmup_async_is_idempotent(monkeypatch):
    """Calling `warmup_async()` twice must not spawn a second thread. The
    helper should be invoked at most once per process."""
    monkeypatch.setenv("RAG_LOCAL_EMBED", "1")
    monkeypatch.delenv("RAG_NO_WARMUP", raising=False)
    monkeypatch.setattr(rag, "get_reranker", lambda: None)
    monkeypatch.setattr(rag, "embed", lambda texts: [[0.0] * 1024 for _ in texts])
    monkeypatch.setattr(
        rag,
        "get_db",
        lambda: types.SimpleNamespace(count=lambda: 0),
    )
    monkeypatch.setattr(rag, "_load_corpus", lambda col: None)
    monkeypatch.setattr(rag, "get_pagerank", lambda col: {})

    count = {"n": 0}
    done = threading.Event()

    def _spy():
        count["n"] += 1
        done.set()
        return True

    monkeypatch.setattr(rag, "_warmup_local_embedder", _spy)
    rag.warmup_async()
    assert done.wait(timeout=2.0)
    # Second call should no-op (idempotency guard).
    rag.warmup_async()
    # Give any stray thread 100 ms to race.
    import time as _t

    _t.sleep(0.1)
    assert count["n"] == 1
