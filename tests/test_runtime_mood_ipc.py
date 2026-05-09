"""F4.4 tests — IPC handler ``compute_mood`` con TTL cache."""
from __future__ import annotations

import sys

import pytest

from rag.runtime import ipc


@pytest.fixture(autouse=True)
def _reset():
    """Reset cache + IPC handlers entre tests."""
    ipc._reset_handlers()
    # Forzar import del módulo + reset state ANTES de cada test.
    import rag.runtime.jobs._mood_ipc as mod
    mod._CACHE.clear()
    ipc.register_handler("compute_mood", mod.compute_mood_handler)
    ipc.register_handler("invalidate_mood_cache", mod.invalidate_handler)
    yield
    ipc._reset_handlers()
    mod._CACHE.clear()


def _import_handler():
    import rag.runtime.jobs._mood_ipc as mod
    # Reset cache state entre tests.
    mod._CACHE.clear()
    # Re-register handlers que el reset borró.
    ipc.register_handler("compute_mood", mod.compute_mood_handler)
    ipc.register_handler("invalidate_mood_cache", mod.invalidate_handler)
    return mod


def test_handler_registered():
    _import_handler()
    handlers = ipc._registered_handlers()
    assert "compute_mood" in handlers
    assert "invalidate_mood_cache" in handlers


def test_compute_mood_returns_score_when_compute_succeeds(monkeypatch):
    mod = _import_handler()

    fake_compute = lambda d: {  # noqa: E731
        "date": "2026-05-09",
        "value": 0.5,
        "n_signals": 3,
        "sources_used": ["queries", "manual"],
    }
    fake_mood = type("M", (), {"compute_daily_score": staticmethod(fake_compute)})
    monkeypatch.setitem(sys.modules, "rag.mood", fake_mood)

    result = mod.compute_mood_handler({})
    assert result["ok"] is True
    assert result["score"] == 0.5
    assert result["n_signals"] == 3
    assert "queries" in result["sources_used"]
    assert result["cache_hit"] is False


def test_compute_mood_uses_cache_on_second_call(monkeypatch):
    mod = _import_handler()
    n_calls = {"i": 0}

    def fake_compute(d):
        n_calls["i"] += 1
        return {"date": "2026-05-09", "value": 0.0, "n_signals": 0, "sources_used": []}

    fake_mood = type("M", (), {"compute_daily_score": staticmethod(fake_compute)})
    monkeypatch.setitem(sys.modules, "rag.mood", fake_mood)

    r1 = mod.compute_mood_handler({})
    r2 = mod.compute_mood_handler({})
    assert n_calls["i"] == 1, "compute_daily_score se llamó 2x — cache no funcionó"
    assert r1["cache_hit"] is False
    assert r2["cache_hit"] is True


def test_compute_mood_force_bypasses_cache(monkeypatch):
    mod = _import_handler()
    n_calls = {"i": 0}

    def fake_compute(d):
        n_calls["i"] += 1
        return {"date": "2026-05-09", "value": 0.0, "n_signals": 0, "sources_used": []}

    fake_mood = type("M", (), {"compute_daily_score": staticmethod(fake_compute)})
    monkeypatch.setitem(sys.modules, "rag.mood", fake_mood)

    mod.compute_mood_handler({})
    mod.compute_mood_handler({"force": True})
    assert n_calls["i"] == 2, "force=True no bypaseó cache"


def test_invalidate_clears_cache(monkeypatch):
    mod = _import_handler()
    n_calls = {"i": 0}

    def fake_compute(d):
        n_calls["i"] += 1
        return {"date": "2026-05-09", "value": 0.0, "n_signals": 0, "sources_used": []}

    fake_mood = type("M", (), {"compute_daily_score": staticmethod(fake_compute)})
    monkeypatch.setitem(sys.modules, "rag.mood", fake_mood)

    mod.compute_mood_handler({})  # populates cache
    inv = mod.invalidate_handler({})
    assert inv["invalidated"] is True
    mod.compute_mood_handler({})  # should re-compute
    assert n_calls["i"] == 2


def test_compute_mood_handles_import_failure(monkeypatch):
    mod = _import_handler()

    # Forzar ImportError simulando que rag.mood no es importable.
    monkeypatch.setitem(sys.modules, "rag.mood", None)
    result = mod.compute_mood_handler({})
    # Sin rag.mood disponible → respuesta debe indicar error.
    assert result.get("ok") is False
    assert "error" in result


def test_compute_mood_handles_compute_exception(monkeypatch):
    mod = _import_handler()

    def fake_compute(d):
        raise RuntimeError("DB locked")

    fake_mood = type("M", (), {"compute_daily_score": staticmethod(fake_compute)})
    monkeypatch.setitem(sys.modules, "rag.mood", fake_mood)

    result = mod.compute_mood_handler({})
    assert result["ok"] is False
    assert "DB locked" in (result.get("error") or "")


def test_cache_ttl_expires(monkeypatch):
    mod = _import_handler()
    n_calls = {"i": 0}

    def fake_compute(d):
        n_calls["i"] += 1
        return {"date": "2026-05-09", "value": 0.0, "n_signals": 0, "sources_used": []}

    fake_mood = type("M", (), {"compute_daily_score": staticmethod(fake_compute)})
    monkeypatch.setitem(sys.modules, "rag.mood", fake_mood)

    # Call once.
    mod.compute_mood_handler({})

    # Manualmente expirar la cache entry.
    with mod._CACHE_LOCK:
        ts, val = mod._CACHE["_today_"]
        mod._CACHE["_today_"] = (ts - mod._CACHE_TTL_S - 1, val)

    # Next call should recompute.
    mod.compute_mood_handler({})
    assert n_calls["i"] == 2
