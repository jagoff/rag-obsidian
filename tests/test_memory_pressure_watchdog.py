"""Tests for the memory-pressure watchdog.

The watchdog samples macOS vm_stat + sysctl hw.memsize every N seconds and,
when (wired + active + compressed) / total_memory exceeds a threshold,
proactively unloads the chat model (via ollama keep_alive=0) and optionally
force-unloads the cross-encoder reranker. Prevents the 2026-04-17 Mac-freeze
regression from re-appearing under concurrent-app VRAM pressure.

These tests cover the logic WITHOUT actually starting a thread (we test the
functions directly). The loop itself is a thin wrapper around _handle_memory_pressure
which has its own dedicated tests.
"""
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

import rag  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_watchdog_state():
    """Ensure clean slate per test — watchdog global flag + reranker cache."""
    rag._memory_watchdog_started = False
    rag._CHAT_MODEL_RESOLVED = None
    rag._reranker = None
    rag._reranker_last_use = 0
    yield
    rag._memory_watchdog_started = False


# ── _system_memory_used_pct ────────────────────────────────────────────────


def _fake_vmstat_output(wired_pages: int, active_pages: int, compressed_pages: int,
                        free_pages: int = 100000, inactive_pages: int = 200000,
                        page_size: int = 16384) -> str:
    """Build a realistic vm_stat stdout mimic.

    Uses 16384-byte pages (Apple Silicon default) unless overridden — catches
    the page-size-parse path. Apple Silicon uses 16 KB pages; Intel uses 4 KB.
    """
    return (
        f"Mach Virtual Memory Statistics: (page size of {page_size} bytes)\n"
        f"Pages free:                               {free_pages}.\n"
        f"Pages active:                             {active_pages}.\n"
        f"Pages inactive:                           {inactive_pages}.\n"
        f"Pages wired down:                         {wired_pages}.\n"
        f"Pages occupied by compressor:             {compressed_pages}.\n"
        f"Pages speculative:                        10000.\n"
    )


def test_memory_pct_linux_returns_none(monkeypatch):
    """Non-darwin hosts can't reliably get memory pressure — return None so
    the watchdog becomes a silent no-op."""
    monkeypatch.setattr(sys, "platform", "linux")
    assert rag._system_memory_used_pct() is None


def test_memory_pct_subprocess_failure_returns_none(monkeypatch):
    """vm_stat missing / sysctl missing → graceful None, don't raise."""
    monkeypatch.setattr(sys, "platform", "darwin")

    def _raise(*args, **kwargs):
        raise FileNotFoundError("vm_stat not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert rag._system_memory_used_pct() is None


def test_memory_pct_parses_apple_silicon_page_size(monkeypatch):
    """Apple Silicon uses 16 KB pages — parser must pick up the declared size,
    not default to 4 KB. A 16 KB page is 4× bigger, so using 4 KB would give
    a 4× underestimate of used memory (dangerous false-negative under pressure)."""
    monkeypatch.setattr(sys, "platform", "darwin")

    def _fake_run(cmd, **kwargs):
        result = MagicMock()
        if "vm_stat" in cmd:
            result.stdout = _fake_vmstat_output(
                wired_pages=500_000, active_pages=800_000, compressed_pages=200_000,
                page_size=16384,
            )
        elif "sysctl" in cmd:
            # 36 GB unified memory (user's Mac)
            result.stdout = str(36 * 1024 * 1024 * 1024) + "\n"
        return result

    monkeypatch.setattr(subprocess, "run", _fake_run)
    pct = rag._system_memory_used_pct()
    assert pct is not None
    # used = (500k + 800k + 200k) * 16384 B ≈ 22.89 GiB
    # total = 36 * 1024^3 B = 36 GiB
    # 22.89 / 36 ≈ 63.58%
    assert 63.0 <= pct <= 64.5


def test_memory_pct_zero_total_returns_none(monkeypatch):
    """Defensive: if sysctl returns 0 or garbage, don't divide by zero."""
    monkeypatch.setattr(sys, "platform", "darwin")

    def _fake_run(cmd, **kwargs):
        result = MagicMock()
        if "vm_stat" in cmd:
            result.stdout = _fake_vmstat_output(1, 1, 1)
        else:
            result.stdout = "0\n"
        return result

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert rag._system_memory_used_pct() is None


# ── _handle_memory_pressure ────────────────────────────────────────────────


def test_handle_pressure_unloads_chat_model(monkeypatch):
    """Paso 1: invoca ollama.chat con keep_alive=0 para evictar el chat model."""
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "qwen2.5:7b")
    monkeypatch.setattr(rag, "_system_memory_used_pct", lambda: 70.0)

    ollama_calls = []

    class _FakeOllama:
        @staticmethod
        def chat(**kwargs):
            ollama_calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "ollama", _FakeOllama)

    actions = rag._handle_memory_pressure(pct_before=90.0, threshold=85.0)

    assert actions["chat_unloaded"] is True
    assert actions["chat_model"] == "qwen2.5:7b"
    assert len(ollama_calls) == 1
    assert ollama_calls[0]["model"] == "qwen2.5:7b"
    assert ollama_calls[0]["keep_alive"] == 0
    assert ollama_calls[0]["options"]["num_predict"] == 1


def test_handle_pressure_no_chat_model_when_resolve_fails(monkeypatch):
    """Si resolve_chat_model lanza RuntimeError, no crasheamos — solo saltamos
    el step 1 y evaluamos si hace falta force-unload del reranker."""
    def _raise():
        raise RuntimeError("no chat model installed")

    monkeypatch.setattr(rag, "resolve_chat_model", _raise)
    monkeypatch.setattr(rag, "_system_memory_used_pct", lambda: 70.0)

    actions = rag._handle_memory_pressure(pct_before=90.0, threshold=85.0)

    assert actions["chat_unloaded"] is False
    assert actions["chat_model"] is None


def test_handle_pressure_skips_reranker_when_chat_unload_enough(monkeypatch):
    """Paso 2 solo fires si el pct post-chat-unload sigue ≥ threshold.
    Acá baja a 70% → reranker no se toca."""
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "qwen2.5:7b")
    monkeypatch.setattr(rag, "_system_memory_used_pct", lambda: 70.0)

    class _FakeOllama:
        @staticmethod
        def chat(**_kwargs):
            pass

    monkeypatch.setitem(sys.modules, "ollama", _FakeOllama)

    unload_calls = []

    def _fake_unload(force=False):
        unload_calls.append(force)
        return True

    monkeypatch.setattr(rag, "maybe_unload_reranker", _fake_unload)

    actions = rag._handle_memory_pressure(pct_before=90.0, threshold=85.0)

    assert actions["reranker_unloaded"] is False
    assert unload_calls == []


def test_handle_pressure_force_unloads_reranker_when_still_high(monkeypatch):
    """Si tras unload del chat sigue ≥ threshold, force-unload del reranker."""
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "qwen2.5:7b")
    # Simulamos que el unload no alivió la presión — sigue a 88%.
    monkeypatch.setattr(rag, "_system_memory_used_pct", lambda: 88.0)

    class _FakeOllama:
        @staticmethod
        def chat(**_kwargs):
            pass

    monkeypatch.setitem(sys.modules, "ollama", _FakeOllama)

    unload_calls = []

    def _fake_unload(force=False):
        unload_calls.append(force)
        return True

    monkeypatch.setattr(rag, "maybe_unload_reranker", _fake_unload)

    actions = rag._handle_memory_pressure(pct_before=92.0, threshold=85.0)

    assert actions["chat_unloaded"] is True
    assert actions["reranker_unloaded"] is True
    assert unload_calls == [True]  # force=True passed


def test_handle_pressure_ollama_exception_logged(monkeypatch):
    """Si ollama.chat explota, lo capturamos — la respuesta no puede crashear
    el watchdog thread."""
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "qwen2.5:7b")
    monkeypatch.setattr(rag, "_system_memory_used_pct", lambda: 70.0)

    class _FakeOllama:
        @staticmethod
        def chat(**_kwargs):
            raise RuntimeError("ollama daemon stuck")

    monkeypatch.setitem(sys.modules, "ollama", _FakeOllama)

    # Should NOT raise
    actions = rag._handle_memory_pressure(pct_before=90.0, threshold=85.0)
    assert actions["chat_unloaded"] is False


# ── start_memory_pressure_watchdog ─────────────────────────────────────────


def test_start_watchdog_disabled_by_env(monkeypatch):
    """RAG_MEMORY_PRESSURE_DISABLE=1 → no thread, return False."""
    monkeypatch.setenv("RAG_MEMORY_PRESSURE_DISABLE", "1")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert rag.start_memory_pressure_watchdog() is False
    assert rag._memory_watchdog_started is False


def test_start_watchdog_skips_non_darwin(monkeypatch):
    """Linux hosts can't read vm_stat — no point starting the thread."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("RAG_MEMORY_PRESSURE_DISABLE", raising=False)
    assert rag.start_memory_pressure_watchdog() is False
    assert rag._memory_watchdog_started is False


def test_start_watchdog_idempotent(monkeypatch):
    """Second call should NOT spawn new threads — returns True (already running).

    Post 2026-05-02: `start_memory_pressure_watchdog` arranca DOS daemon
    threads en la primera llamada (watchdog + MPS cache drop loop). Los
    calls subsiguientes son no-op porque `_memory_watchdog_started`
    queda en True. El test cuenta threads totales: 2 en la primera
    llamada, 0 nuevos en las siguientes.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("RAG_MEMORY_PRESSURE_DISABLE", raising=False)
    monkeypatch.delenv("RAG_MPS_CACHE_DROP_INTERVAL", raising=False)

    threads_started = []
    original_thread = threading.Thread

    def _tracked_thread(*args, **kwargs):
        t = original_thread(*args, **kwargs)
        threads_started.append(t)
        return t

    monkeypatch.setattr(threading, "Thread", _tracked_thread)

    assert rag.start_memory_pressure_watchdog() is True
    n_after_first = len(threads_started)
    assert rag.start_memory_pressure_watchdog() is True
    assert rag.start_memory_pressure_watchdog() is True
    # Watchdog (1) + MPS cache drop loop (1) en la primera llamada;
    # subsiguientes son no-op.
    assert n_after_first == 2
    assert len(threads_started) == n_after_first


def test_start_watchdog_respects_custom_threshold(monkeypatch):
    """Env var RAG_MEMORY_PRESSURE_THRESHOLD is propagated to the loop.

    Verifica que el watchdog principal recibe los args (threshold, interval)
    desde las env vars. Filtra el thread del MPS cache drop loop (su tupla
    de args es (mps_interval,) — un solo elemento, no dos).
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("RAG_MEMORY_PRESSURE_DISABLE", raising=False)
    monkeypatch.setenv("RAG_MEMORY_PRESSURE_THRESHOLD", "95")
    monkeypatch.setenv("RAG_MEMORY_PRESSURE_INTERVAL", "30")

    captured_args = []
    original_thread = threading.Thread

    def _capture_thread(*args, **kwargs):
        captured_args.append(kwargs.get("args", ()))
        t = original_thread(target=lambda: None, daemon=True)  # inert
        return t

    monkeypatch.setattr(threading, "Thread", _capture_thread)

    rag.start_memory_pressure_watchdog()
    # El watchdog principal usa args=(threshold, interval) — 2 elementos.
    # El MPS cache drop loop usa args=(mps_interval,) — 1 elemento.
    watchdog_args = [a for a in captured_args if len(a) == 2]
    assert watchdog_args == [(95.0, 30)]


def test_start_watchdog_invalid_threshold_falls_back(monkeypatch):
    """Garbage threshold string → default 85.0, no crash."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("RAG_MEMORY_PRESSURE_DISABLE", raising=False)
    monkeypatch.setenv("RAG_MEMORY_PRESSURE_THRESHOLD", "not-a-number")

    captured_args = []
    original_thread = threading.Thread

    def _capture_thread(*args, **kwargs):
        captured_args.append(kwargs.get("args", ()))
        return original_thread(target=lambda: None, daemon=True)

    monkeypatch.setattr(threading, "Thread", _capture_thread)

    rag.start_memory_pressure_watchdog()
    assert captured_args[0][0] == 85.0


# ── maybe_unload_reranker(force=True) ──────────────────────────────────────


def test_maybe_unload_reranker_default_respects_idle_ttl(monkeypatch):
    """Sin force, el check de idle se mantiene — reranker en uso reciente
    NO se descarga."""
    rag._reranker = "fake-reranker-obj"
    rag._reranker_last_use = time.time()  # just touched
    assert rag.maybe_unload_reranker() is False
    assert rag._reranker == "fake-reranker-obj"  # still loaded


def test_maybe_unload_reranker_force_bypasses_idle(monkeypatch):
    """force=True bypassea el TTL check y descarga incondicionalmente."""
    rag._reranker = "fake-reranker-obj"
    rag._reranker_last_use = time.time()  # just touched

    # Patch torch/gc so we don't actually try to free MPS memory in tests
    monkeypatch.setattr(rag, "_silent_log", lambda *a, **k: None)

    assert rag.maybe_unload_reranker(force=True) is True
    assert rag._reranker is None


def test_maybe_unload_reranker_none_loaded_returns_false(monkeypatch):
    """Si no hay reranker cargado, force o no, retorna False sin errores."""
    rag._reranker = None
    assert rag.maybe_unload_reranker() is False
    assert rag.maybe_unload_reranker(force=True) is False
