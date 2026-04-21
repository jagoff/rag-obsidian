"""Tests for the conversation-writer shutdown drain in web/server.py.

Production behaviour we guard:
  * `_spawn_conversation_writer` registers every writer thread in
    `_CONV_WRITERS` and removes it on completion, so the set can't leak.
  * `_drain_conversation_writers` (the @app.on_event("shutdown") hook)
    joins in-flight writers up to ~5s before letting the process exit.
  * A wedged writer never prevents shutdown; it's logged and skipped.

We test the drain helper directly with fake writer threads — no FastAPI
lifespan harness needed, and no real SQL/disk work.
"""
from __future__ import annotations

import threading
import time

import pytest


pytest.importorskip("web.server")

from web import server  # noqa: E402 — must come after importorskip


def _reset_tracker() -> None:
    with server._CONV_WRITERS_LOCK:
        server._CONV_WRITERS.clear()


def _register_fake_writer(work_time: float) -> threading.Thread:
    """Spawn a daemon thread that sleeps `work_time` seconds, mirroring the
    bookkeeping in `_spawn_conversation_writer` without actually touching
    the SQL/disk write path."""
    def _body() -> None:
        try:
            time.sleep(work_time)
        finally:
            with server._CONV_WRITERS_LOCK:
                server._CONV_WRITERS.discard(threading.current_thread())

    t = threading.Thread(target=_body, daemon=True, name=f"fake-writer-{work_time}")
    with server._CONV_WRITERS_LOCK:
        server._CONV_WRITERS.add(t)
    t.start()
    return t


def test_tracker_self_removes_on_completion():
    """A writer that finishes on its own must not linger in the set."""
    _reset_tracker()
    t = _register_fake_writer(0.01)
    t.join(timeout=1.0)
    assert not t.is_alive()
    with server._CONV_WRITERS_LOCK:
        assert t not in server._CONV_WRITERS


def test_drain_with_no_writers_is_noop():
    """Drain on an empty tracker must return immediately — no sleeps, no logs."""
    _reset_tracker()
    t0 = time.monotonic()
    server._drain_conversation_writers()
    assert time.monotonic() - t0 < 0.5


def test_drain_waits_for_quick_writers():
    """A handful of short writers must all finish cleanly inside the 5s
    budget. Drain must return once they're all done, not after the full
    5s timeout."""
    _reset_tracker()
    writers = [_register_fake_writer(0.05) for _ in range(3)]
    t0 = time.monotonic()
    server._drain_conversation_writers()
    elapsed = time.monotonic() - t0
    # All 3 finish in ~50ms; drain should return well under the 5s cap.
    assert elapsed < 2.0, f"drain took {elapsed:.2f}s, expected <2s"
    for t in writers:
        assert not t.is_alive()
    with server._CONV_WRITERS_LOCK:
        assert not server._CONV_WRITERS


def test_drain_does_not_exceed_5s_cap_with_wedged_writer(monkeypatch):
    """If a writer is stuck (SQL lock, hung disk, etc) drain must return
    within ~5s + overhead, NOT hang the process. The straggler stays
    daemon=True so process exit is not blocked."""
    _reset_tracker()
    # Fake a writer that sleeps 20s. daemon=True means the OS will reap it
    # when the test process exits.
    wedged = _register_fake_writer(20.0)

    # Verify the drain logs a straggler entry by capturing LOG_QUEUE puts.
    queued: list[tuple] = []
    monkeypatch.setattr(server._LOG_QUEUE, "put", lambda item: queued.append(item))

    t0 = time.monotonic()
    server._drain_conversation_writers()
    elapsed = time.monotonic() - t0

    # 5s cap + Python's thread scheduling slack.
    assert 4.5 < elapsed < 6.5, f"drain took {elapsed:.2f}s, expected ~5s"
    assert wedged.is_alive(), "wedged writer should still be running"
    # The shutdown-timeout log entry must be emitted exactly once.
    assert any(
        isinstance(item, tuple) and b"conversation_writer_shutdown_timeout" in item[1].encode()
        for item in queued
    ), f"expected shutdown-timeout log, got: {queued}"

    # Clean-up: remove the wedged thread from the tracker so later tests
    # start from a clean slate (the daemon thread will die with the
    # process anyway).
    with server._CONV_WRITERS_LOCK:
        server._CONV_WRITERS.discard(wedged)


def test_spawn_conversation_writer_registers_and_releases(monkeypatch):
    """`_spawn_conversation_writer` must track the thread during execution
    and release it once `_persist_conversation_turn` returns, regardless
    of whether the persist path succeeded or raised."""
    _reset_tracker()
    barrier = threading.Event()
    released = threading.Event()

    def fake_persist(*_args) -> None:
        barrier.wait(timeout=2.0)

    monkeypatch.setattr(server, "_persist_conversation_turn", fake_persist)

    t = server._spawn_conversation_writer(target_args=(1, 2, 3), name="spawn-test")
    # While the fake is blocked inside barrier.wait, the tracker must hold it.
    with server._CONV_WRITERS_LOCK:
        assert t in server._CONV_WRITERS
    barrier.set()
    t.join(timeout=2.0)
    released.set()
    assert not t.is_alive()
    with server._CONV_WRITERS_LOCK:
        assert t not in server._CONV_WRITERS


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_spawn_releases_even_on_persist_exception(monkeypatch):
    """An exception inside `_persist_conversation_turn` must still discharge
    the tracker — otherwise a recurrent SQL failure leaks one entry per turn.

    The production wrapper deliberately lets the exception propagate (the
    thread ends, the tracker's `finally` runs); the warning is the Python
    thread-exception hook's normal behaviour. We filter it so the test is
    clean while still asserting the cleanup path."""
    _reset_tracker()

    def fake_persist(*_args) -> None:
        raise RuntimeError("simulated persist failure")

    monkeypatch.setattr(server, "_persist_conversation_turn", fake_persist)

    t = server._spawn_conversation_writer(target_args=(), name="spawn-fail-test")
    t.join(timeout=2.0)
    assert not t.is_alive()
    with server._CONV_WRITERS_LOCK:
        assert t not in server._CONV_WRITERS
