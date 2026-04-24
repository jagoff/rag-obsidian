"""Tests for `query_embed_local` blocking wait (2026-04-22).

Pre-fix, `query_embed_local` bailed immediately (returned None) if the
background warmup hadn't yet set `_local_embedder_ready`. This was a
conscious non-blocking choice to avoid the main thread stalling on the
model lock — but on CLI one-shot, where the warmup starts ~100-300ms
before retrieve and typically completes a few hundred ms later, it
meant we ALWAYS fell back to ollama embed (~150ms / variant × 3
variants ≈ 500ms+ extra per query).

Post-fix, callers pass `wait_ready_timeout` to block up to N seconds
on the Event. If it fires in time we proceed with local embed; if
not we fall back to ollama same as before.

Telemetry before the fix (rag_queries.t_retrieve, 30d window):
  web p50 = 2.8s (warmup runs in daemon → Event set before queries)
  cli p50 = 11.9s (one-shot → warmup races, always ollama fallback)
The 4× gap is dominated by this race.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_ready_event(monkeypatch):
    """Each test gets a fresh Event so we control readiness precisely."""
    fresh = threading.Event()
    monkeypatch.setattr(rag, "_local_embedder_ready", fresh)
    # Also ensure `_local_embed_enabled()` returns True so the early
    # bail doesn't mask the waiting behaviour.
    monkeypatch.setattr(rag, "_local_embed_enabled", lambda: True)
    # Stub the model so encode returns objects with `.tolist()` — the
    # real sentence-transformers returns numpy arrays, and the code
    # path in query_embed_local calls `v.tolist()` on each element.
    class _FakeVec:
        def __init__(self, v): self._v = v
        def tolist(self): return list(self._v)
    fake_model = MagicMock()
    # `encode` is called with a list, return one _FakeVec per input.
    fake_model.encode.side_effect = lambda texts, **kw: [
        _FakeVec([0.1 + 0.01 * i] * 1024) for i in range(len(texts))
    ]
    monkeypatch.setattr(rag, "_get_local_embedder", lambda: fake_model)
    yield fresh


# ── Legacy behaviour: no timeout → non-blocking bail ────────────────────────


def test_without_timeout_bails_immediately_when_not_ready():
    """Default call path (no timeout arg): if Event isn't set, return
    None without waiting. Matches the pre-2026-04-22 contract exactly."""
    # Event NOT set — legacy path returns None immediately.
    t0 = time.perf_counter()
    result = rag.query_embed_local(["hola", "chau"])
    elapsed = time.perf_counter() - t0
    assert result is None
    # Should be sub-millisecond. Give 100ms margin for scheduler jitter.
    assert elapsed < 0.1, f"non-blocking bail took {elapsed:.3f}s"


def test_event_already_set_returns_embeds(_reset_ready_event):
    """Happy path when the warmup already finished before we arrived.
    Applies to long-running daemons (rag serve, web server) and to any
    CLI that happens to race-win."""
    _reset_ready_event.set()
    result = rag.query_embed_local(["a", "b"])
    assert result is not None
    assert len(result) == 2
    assert len(result[0]) == 1024


# ── New behaviour: positive timeout blocks up to N seconds ──────────────────


def test_timeout_bails_when_event_never_fires(_reset_ready_event):
    """If the warmup thread doesn't signal within the timeout, we return
    None (ollama fallback). Event is intentionally left unset."""
    t0 = time.perf_counter()
    result = rag.query_embed_local(["q"], wait_ready_timeout=0.2)
    elapsed = time.perf_counter() - t0
    assert result is None
    # Should have waited approximately the timeout, not sub-millisecond
    # (proves we actually blocked) and not much longer (proves we
    # respect the timeout).
    assert 0.15 <= elapsed <= 0.5, (
        f"expected ~0.2s wait, got {elapsed:.3f}s"
    )


def test_timeout_proceeds_when_event_fires_in_time(_reset_ready_event):
    """If the background warmup sets the Event mid-wait, we should
    wake up and proceed with local embed — this is the CLI one-shot
    win case."""
    # Fire the event from a helper thread 50ms in. The main thread
    # should be waiting with timeout=500ms.
    def _delayed_set():
        time.sleep(0.05)
        _reset_ready_event.set()
    threading.Thread(target=_delayed_set, daemon=True).start()

    t0 = time.perf_counter()
    result = rag.query_embed_local(["q"], wait_ready_timeout=0.5)
    elapsed = time.perf_counter() - t0
    assert result is not None, "expected embeds since Event fired within timeout"
    # Should have woken up around the 50ms mark, not waited the full
    # 500ms timeout.
    assert elapsed < 0.25, (
        f"woke up late — expected <250ms, got {elapsed:.3f}s"
    )


def test_timeout_zero_is_legacy_non_blocking(_reset_ready_event):
    """wait_ready_timeout=0 (or explicit 0.0) keeps legacy semantics."""
    t0 = time.perf_counter()
    result = rag.query_embed_local(["q"], wait_ready_timeout=0)
    elapsed = time.perf_counter() - t0
    assert result is None
    assert elapsed < 0.05, (
        "timeout=0 should not block — treat as legacy non-blocking"
    )


def test_timeout_none_is_legacy_non_blocking(_reset_ready_event):
    """Explicit None also yields legacy behaviour."""
    result = rag.query_embed_local(["q"], wait_ready_timeout=None)
    assert result is None


# ── Env-driven tuning via retrieve() ────────────────────────────────────────


def test_env_var_overrides_default_wait(monkeypatch, tmp_path):
    """`retrieve()` reads `RAG_LOCAL_EMBED_WAIT_MS` from env and passes it
    through as seconds. Set to 0 → legacy bail, set to 3000 → 3s wait.

    2026-04-23: default bumped 4.0→6.0 tras observar en prod el patrón de
    `embed_ms=4005` exacto — el wait timeaba justo antes del Event fire.

    We test the env parsing directly on the pattern used in retrieve():
    """
    # Mirror the env-parsing block in retrieve() (rag.py)
    def _compute_wait_s() -> float:
        import os
        try:
            raw = os.environ.get("RAG_LOCAL_EMBED_WAIT_MS")
            return float(raw) / 1000.0 if raw is not None else 6.0
        except ValueError:
            return 6.0

    monkeypatch.delenv("RAG_LOCAL_EMBED_WAIT_MS", raising=False)
    assert _compute_wait_s() == 6.0  # default (post 2026-04-23)

    monkeypatch.setenv("RAG_LOCAL_EMBED_WAIT_MS", "0")
    assert _compute_wait_s() == 0.0  # legacy opt-out

    monkeypatch.setenv("RAG_LOCAL_EMBED_WAIT_MS", "3000")
    assert _compute_wait_s() == 3.0  # shorter wait (fast disk)

    monkeypatch.setenv("RAG_LOCAL_EMBED_WAIT_MS", "10000")
    assert _compute_wait_s() == 10.0  # longer wait (cold disk)

    # Parse error falls back to default.
    monkeypatch.setenv("RAG_LOCAL_EMBED_WAIT_MS", "notanumber")
    assert _compute_wait_s() == 6.0


# ── Idempotence: setting the Event twice is a no-op (regression guard) ─────


def test_multiple_waits_all_proceed_once_set(_reset_ready_event):
    """Once the Event is set, every subsequent call returns embeds
    without blocking — Event.wait() on a set Event returns True
    immediately."""
    _reset_ready_event.set()
    for _ in range(5):
        t0 = time.perf_counter()
        result = rag.query_embed_local(["q"], wait_ready_timeout=1.0)
        elapsed = time.perf_counter() - t0
        assert result is not None
        assert elapsed < 0.1, f"call {_}: {elapsed:.3f}s"
