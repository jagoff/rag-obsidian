"""Regression tests for reranker idle-unload + TOCTOU race fix.

See rag.py:7263-7330 (get_reranker + maybe_unload_reranker). The timestamp
update in get_reranker used to live outside `_reranker_lock`, leaving a
TOCTOU window with the background sweeper where it could read stale
`_reranker_last_use`, pass the idle check, acquire the lock, and unload
a model that another thread had just touched.
"""
from __future__ import annotations

import threading
import time

import pytest

import rag


class _Sentinel:
    """Stand-in for the CrossEncoder object — just needs to not be None."""
    def predict(self, *_a, **_kw):
        return [0.0]


@pytest.fixture
def preloaded(monkeypatch):
    """Pre-install a sentinel reranker so get_reranker() never hits the
    sentence-transformers/torch import path. Restores state after.
    """
    prev_reranker = rag._reranker
    prev_last_use = rag._reranker_last_use
    rag._reranker = _Sentinel()
    rag._reranker_last_use = time.time()
    try:
        yield
    finally:
        rag._reranker = prev_reranker
        rag._reranker_last_use = prev_last_use


def test_get_reranker_refreshes_last_use(preloaded):
    """Every get_reranker() call must refresh _reranker_last_use — that's
    the signal the sweeper reads to decide whether to evict."""
    rag._reranker_last_use = 0.0
    rag.get_reranker()
    assert rag._reranker_last_use > 0.0
    # Second call re-refreshes (monotonic >= previous).
    snapshot = rag._reranker_last_use
    time.sleep(0.005)
    rag.get_reranker()
    assert rag._reranker_last_use >= snapshot


def test_maybe_unload_noop_when_recently_used(monkeypatch, preloaded):
    """If _reranker_last_use is fresh, the sweeper must NOT unload."""
    monkeypatch.setattr(rag, "_RERANKER_IDLE_TTL", 60.0)
    rag._reranker_last_use = time.time()
    assert rag.maybe_unload_reranker() is False
    assert rag._reranker is not None


def test_maybe_unload_evicts_when_idle(monkeypatch, preloaded):
    """If idle > TTL, the sweeper unloads and sets _reranker to None."""
    monkeypatch.setattr(rag, "_RERANKER_IDLE_TTL", 0.05)
    rag._reranker_last_use = time.time() - 10.0  # well past TTL
    assert rag.maybe_unload_reranker() is True
    assert rag._reranker is None


def test_maybe_unload_noop_when_already_unloaded(monkeypatch):
    """If _reranker is already None, maybe_unload returns False fast."""
    prev = rag._reranker
    rag._reranker = None
    try:
        assert rag.maybe_unload_reranker() is False
    finally:
        rag._reranker = prev


def test_get_reranker_holds_lock_while_updating_timestamp(monkeypatch, preloaded):
    """Regression guard for the pre-2026-04-20 race: the timestamp
    update lived OUTSIDE _reranker_lock, so the sweeper could read it
    stale. This test replaces _reranker_lock with an instrumented one
    and asserts that any timestamp write happens while the lock is held.
    """
    ts_writes_inside_lock: list[bool] = []

    class TrackingLock:
        def __init__(self):
            self._inner = threading.Lock()
            self._held = False

        def __enter__(self):
            self._inner.acquire()
            self._held = True
            return self

        def __exit__(self, *a):
            # Probe the ordering: at the moment we exit, any write to
            # _reranker_last_use during this critical section must have
            # been recorded as inside-lock. We snapshot right before
            # release to give the body a chance to mutate state.
            self._held = False
            self._inner.release()
            return False

        def acquire(self, *a, **kw):
            result = self._inner.acquire(*a, **kw)
            if result:
                self._held = True
            return result

        def release(self):
            self._held = False
            self._inner.release()

        @property
        def held(self) -> bool:
            return self._held

    tracking = TrackingLock()
    # Patch the module-level lock that get_reranker/maybe_unload_reranker use.
    monkeypatch.setattr(rag, "_reranker_lock", tracking)

    # Wrap time.time so we can detect when it's called AND verify the
    # lock is held at that moment (the timestamp write sits right after).
    real_time = time.time
    orig_last_use = rag._reranker_last_use

    def tracking_time():
        # If this time.time() call is the one feeding _reranker_last_use,
        # the lock must be held. We cooperate with the test by only
        # recording *during* get_reranker.
        ts_writes_inside_lock.append(tracking.held)
        return real_time()

    monkeypatch.setattr(rag.time, "time", tracking_time)

    try:
        rag.get_reranker()
    finally:
        rag._reranker_last_use = orig_last_use

    # At least one time.time() call inside get_reranker must have been
    # while holding the lock — that's the timestamp write.
    assert ts_writes_inside_lock, "expected time.time() to be called"
    assert any(ts_writes_inside_lock), (
        "get_reranker() called time.time() but never while holding "
        "_reranker_lock — regression of the pre-fix TOCTOU race"
    )


def test_concurrent_get_and_unload_does_not_evict_active_use(monkeypatch, preloaded):
    """Stress the race from the other direction: many threads calling
    get_reranker() in a tight loop must keep _reranker_last_use fresh
    enough that a concurrent sweeper never fires its idle check.

    With a TTL of 0.5s and threads poking get_reranker every ~1ms,
    maybe_unload_reranker() must always return False.
    """
    monkeypatch.setattr(rag, "_RERANKER_IDLE_TTL", 0.5)
    rag._reranker_last_use = time.time()

    stop = threading.Event()
    unload_fires: list[bool] = []

    def pinger():
        while not stop.is_set():
            rag.get_reranker()
            time.sleep(0.001)

    def sweeper():
        # Run 20 iterations at 25ms — ~500ms window. If the race were
        # real the sweeper would observe stale timestamps and return True.
        for _ in range(20):
            if stop.is_set():
                break
            unload_fires.append(rag.maybe_unload_reranker())
            time.sleep(0.025)

    threads = [threading.Thread(target=pinger) for _ in range(3)]
    sw = threading.Thread(target=sweeper)
    for t in threads:
        t.start()
    sw.start()
    sw.join()
    stop.set()
    for t in threads:
        t.join(timeout=2.0)

    assert not any(unload_fires), (
        "maybe_unload_reranker fired while get_reranker was actively "
        "refreshing the timestamp — TOCTOU race regression"
    )
    assert rag._reranker is not None, "reranker should not have been evicted"
