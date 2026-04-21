"""Non-blocking readiness tests for `query_embed_local`.

Pre-2026-04-21-evening the function would enter `_get_local_embedder`'s
lock and block the main thread for 5-12s on a cold CLI invocation,
racing the background warmup thread. Measured in production
(rag_queries.extra_json: embed_ms up to 12014ms).

The fix: check a `threading.Event` set by `_warmup_local_embedder` after
the first successful encode. If not set, return None immediately so the
caller uses ollama embed (~150ms consistent).
"""
from __future__ import annotations

import threading
import time

import pytest

import rag


@pytest.fixture(autouse=True)
def _reset_local_embedder(monkeypatch):
    """Isolate the module-level singleton across tests. Also forces the
    RAG_LOCAL_EMBED flag ON so the codepath actually runs."""
    monkeypatch.setenv("RAG_LOCAL_EMBED", "1")
    # Reset the ready event (not a monkeypatch target since Event isn't
    # easily reassigned via setattr on a module; clear+set roundtrips
    # instead).
    rag._local_embedder_ready.clear()
    yield
    rag._local_embedder_ready.clear()


def test_query_embed_local_returns_none_when_not_ready():
    """Cold state: Event not set → should bail immediately, no lock enter."""
    assert not rag._local_embedder_ready.is_set()
    result = rag.query_embed_local(["hello"])
    assert result is None


def test_query_embed_local_does_not_block_while_loader_is_slow(monkeypatch):
    """Regression: with the Event-gate in place, `query_embed_local` must
    NOT enter `_get_local_embedder` — even if that function is wedged.
    Pre-fix it would block on the lock and pay the full load time.
    """
    calls: list[float] = []

    def slow_loader():
        calls.append(time.monotonic())
        time.sleep(2.0)  # simulated cold bge-m3 load
        return object()

    monkeypatch.setattr(rag, "_get_local_embedder", slow_loader)

    t0 = time.monotonic()
    result = rag.query_embed_local(["hello"])
    elapsed = time.monotonic() - t0

    assert result is None
    # With the gate, we return in well under 1ms; tolerate 100ms for
    # interpreter noise. Pre-fix this would have taken 2s+.
    assert elapsed < 0.1, (
        f"query_embed_local blocked for {elapsed:.3f}s — "
        "the non-blocking gate is broken"
    )
    assert calls == [], "loader must NOT be called when Event is clear"


def test_query_embed_local_proceeds_when_ready(monkeypatch):
    """Post-warmup: Event set → full code path runs. Stubbed encoder
    returns fake vectors so we can assert the return shape."""

    class FakeModel:
        def encode(self, texts, *, normalize_embeddings, batch_size,
                    show_progress_bar):
            # Return a NumPy-ish object with .tolist() per element.
            class FakeVec:
                def tolist(self):
                    return [0.1, 0.2, 0.3]
            return [FakeVec() for _ in texts]

    monkeypatch.setattr(rag, "_get_local_embedder", lambda: FakeModel())
    rag._local_embedder_ready.set()

    result = rag.query_embed_local(["a", "b", "c"])
    assert result == [[0.1, 0.2, 0.3]] * 3


def test_query_embed_local_returns_none_when_flag_disabled(monkeypatch):
    """`RAG_LOCAL_EMBED=0` short-circuits before the Event check — test
    that removing the flag bypasses even when the Event IS set (would
    be a bug if a stale Event survived a flag-flip)."""
    monkeypatch.setenv("RAG_LOCAL_EMBED", "0")
    rag._local_embedder_ready.set()
    assert rag.query_embed_local(["x"]) is None


def test_warmup_local_embedder_sets_event_on_success(monkeypatch):
    """On happy path (model loads + first encode works), the Event fires
    so subsequent `query_embed_local` calls take the fast path."""

    class FakeModel:
        def encode(self, *args, **kwargs):
            return [[0.0] * 10]

    monkeypatch.setattr(rag, "_get_local_embedder", lambda: FakeModel())
    rag._local_embedder_ready.clear()

    ok = rag._warmup_local_embedder()
    assert ok is True
    assert rag._local_embedder_ready.is_set()


def test_warmup_local_embedder_does_not_set_event_on_load_fail(monkeypatch):
    """Loader returning None (HF cache missing, MPS OOM) must NOT fire
    the Event — otherwise query_embed_local would spin on a None model."""
    monkeypatch.setattr(rag, "_get_local_embedder", lambda: None)
    rag._local_embedder_ready.clear()

    ok = rag._warmup_local_embedder()
    assert ok is False
    assert not rag._local_embedder_ready.is_set()


def test_warmup_local_embedder_does_not_set_event_on_encode_fail(monkeypatch):
    """First-encode failure (MPS runtime error, broken driver) must NOT
    set the Event — the model might be in an inconsistent state."""

    class BadModel:
        def encode(self, *args, **kwargs):
            raise RuntimeError("encode failed")

    monkeypatch.setattr(rag, "_get_local_embedder", lambda: BadModel())
    rag._local_embedder_ready.clear()

    ok = rag._warmup_local_embedder()
    assert ok is False
    assert not rag._local_embedder_ready.is_set()


def test_warmup_does_not_run_when_flag_disabled(monkeypatch):
    """Belt-and-suspenders: explicit disable short-circuits warmup
    before touching the loader. No Event fires."""
    monkeypatch.setenv("RAG_LOCAL_EMBED", "0")
    rag._local_embedder_ready.clear()

    # Loader must not be called — use a sentinel that would raise.
    def forbidden_loader():
        raise AssertionError("loader called despite RAG_LOCAL_EMBED=0")

    monkeypatch.setattr(rag, "_get_local_embedder", forbidden_loader)

    ok = rag._warmup_local_embedder()
    assert ok is False
    assert not rag._local_embedder_ready.is_set()


def test_warmup_and_concurrent_query_dont_race():
    """End-to-end: start a slow warmup in a thread; concurrent query
    embed calls during the warmup window should all return None without
    blocking on the lock. Once the warmup completes, the Event is set
    and subsequent calls would proceed (asserted only indirectly —
    `is_set()` flips to True)."""

    state = {"loaded": False}

    def slow_model_load():
        time.sleep(0.3)  # simulate ~300ms load
        state["loaded"] = True

        class M:
            def encode(self, *a, **k):
                return [[0.0]]
        return M()

    # Stash the original loader + event state, restore after.
    original_loader = rag._get_local_embedder
    original_event_was_set = rag._local_embedder_ready.is_set()
    rag._get_local_embedder = slow_model_load  # type: ignore[assignment]
    rag._local_embedder_ready.clear()

    try:
        t = threading.Thread(target=rag._warmup_local_embedder, daemon=True)
        t.start()

        # While warmup is running, query_embed_local should return None
        # fast without waiting for the loader.
        t0 = time.monotonic()
        nones = [rag.query_embed_local(["x"]) for _ in range(5)]
        elapsed = time.monotonic() - t0
        assert all(r is None for r in nones)
        # 5 calls should complete in well under 1ms total; cap at 100ms.
        assert elapsed < 0.1, (
            f"5 non-blocking calls took {elapsed:.3f}s — "
            "some call must have entered the lock"
        )

        t.join(timeout=2.0)
        assert state["loaded"] is True
        assert rag._local_embedder_ready.is_set()
    finally:
        rag._get_local_embedder = original_loader  # type: ignore[assignment]
        rag._local_embedder_ready.clear()
        if original_event_was_set:
            rag._local_embedder_ready.set()
