"""Tests fundacionales de rag.runtime.events.

Cubren:
- subscribe + publish basic
- multiple subscribers same event
- handler raisea no rompe a los demás
- async_dispatch no bloquea publish
- unsubscribe_all helper
- payload pass-through
"""
from __future__ import annotations

import threading
import time

from rag.runtime.events import EventBus


def test_subscribe_publish_single_subscriber():
    bus = EventBus()
    received = []

    @bus.subscribe("test.event")
    def handler(payload):
        received.append(payload)

    n = bus.publish("test.event", {"value": 1})
    assert n == 1
    assert received == [{"value": 1}]


def test_publish_with_no_subscribers():
    bus = EventBus()
    n = bus.publish("nobody.listens")
    assert n == 0


def test_multiple_subscribers_all_called():
    bus = EventBus()
    calls = []

    @bus.subscribe("multi")
    def h1(p):
        calls.append(("h1", p))

    @bus.subscribe("multi")
    def h2(p):
        calls.append(("h2", p))

    @bus.subscribe("multi")
    def h3(p):
        calls.append(("h3", p))

    n = bus.publish("multi", {"x": 42})
    assert n == 3
    assert ("h1", {"x": 42}) in calls
    assert ("h2", {"x": 42}) in calls
    assert ("h3", {"x": 42}) in calls


def test_handler_exception_does_not_break_others():
    bus = EventBus()
    successful_calls = []

    @bus.subscribe("crash_test")
    def good(p):
        successful_calls.append(("good", p))

    @bus.subscribe("crash_test")
    def bad(p):
        raise RuntimeError("boom")

    @bus.subscribe("crash_test")
    def also_good(p):
        successful_calls.append(("also_good", p))

    n = bus.publish("crash_test", {"x": 1})
    assert n == 3  # 3 subs received the event regardless
    assert ("good", {"x": 1}) in successful_calls
    assert ("also_good", {"x": 1}) in successful_calls


def test_async_dispatch_does_not_block():
    bus = EventBus()
    started = threading.Event()
    can_proceed = threading.Event()

    @bus.subscribe("slow", async_dispatch=True)
    def slow_handler(p):
        started.set()
        # Bloquea hasta que el test te permita continuar.
        can_proceed.wait(timeout=2.0)

    t0 = time.time()
    n = bus.publish("slow")
    elapsed = time.time() - t0

    # Publish debería retornar inmediato (async).
    assert n == 1
    assert elapsed < 0.5, f"publish blocked for {elapsed}s — async_dispatch broken"

    # El handler arrancó.
    assert started.wait(timeout=2.0)
    can_proceed.set()
    bus.shutdown()


def test_async_publish_drops_cleanly_during_executor_shutdown(monkeypatch):
    """Interpreter shutdown can reject new futures; publish must stay quiet."""
    bus = EventBus()

    @bus.subscribe("late", async_dispatch=True)
    def late_handler(p):
        raise AssertionError("should not run")

    class _ShutdownExecutor:
        def submit(self, *args, **kwargs):
            raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(bus, "_get_executor", lambda: _ShutdownExecutor())

    assert bus.publish("late", {"x": 1}) == 1


def test_unsubscribe_all_specific_event():
    bus = EventBus()

    @bus.subscribe("event.a")
    def h_a(p):
        pass

    @bus.subscribe("event.b")
    def h_b(p):
        pass

    n = bus.unsubscribe_all("event.a")
    assert n == 1
    assert bus.publish("event.a") == 0
    assert bus.publish("event.b") == 1


def test_unsubscribe_all_global():
    bus = EventBus()

    @bus.subscribe("event.a")
    def h_a(p):
        pass

    @bus.subscribe("event.b")
    def h_b(p):
        pass

    @bus.subscribe("event.b")
    def h_b2(p):
        pass

    n = bus.unsubscribe_all()
    assert n == 3


def test_payload_default_empty_dict():
    bus = EventBus()
    received = []

    @bus.subscribe("nopayload")
    def handler(p):
        received.append(p)

    bus.publish("nopayload")
    assert received == [{}]


def test_singleton_bus_isolated_from_test_bus():
    """El singleton ``bus`` y un ``EventBus()`` local son independientes."""
    from rag.runtime.events import bus as singleton

    test_bus = EventBus()
    test_received = []

    @test_bus.subscribe("isolated")
    def th(p):
        test_received.append(p)

    # Publicar en singleton no afecta test_bus.
    singleton.publish("isolated", {"x": 1})
    assert test_received == []

    # Y viceversa.
    test_bus.publish("isolated", {"y": 2})
    assert test_received == [{"y": 2}]
