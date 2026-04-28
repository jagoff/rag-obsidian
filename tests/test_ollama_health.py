"""Tests para el ollama health watchdog (eval 2026-04-28 fix)."""
from __future__ import annotations

import time
from unittest import mock

import pytest

from rag import _ollama_health as health


class TestPercentile:
    def test_p50_simple(self):
        assert health._percentile([1, 2, 3, 4, 5], 50) in (2.0, 3.0)

    def test_p95_simple(self):
        vals = list(range(1, 101))
        p95 = health._percentile(vals, 95)
        assert p95 in (94.0, 95.0)

    def test_empty(self):
        assert health._percentile([], 50) is None
        assert health._percentile(None, 50) is None  # type: ignore

    def test_all_none(self):
        assert health._percentile([None, None], 50) is None  # type: ignore


class TestLatencyDegradationCheck:
    def test_skip_on_insufficient_recent(self, monkeypatch):
        monkeypatch.setattr(
            health, "_read_recent_query_latencies",
            lambda window_minutes: ([100, 200], [100, 200] * 50),
        )
        result = health._latency_degradation_check(min_recent=5)
        assert result["action"] == "skip"
        assert "insufficient_recent" in result["reason"]

    def test_ok_when_under_threshold(self, monkeypatch):
        # Recent p95 ≈ 1000ms, baseline p95 ≈ 800ms → ratio 1.25 < 1.8
        recent = [800, 900, 1000, 1000, 1000, 1000]
        baseline = [600, 700, 800, 800, 800] * 20
        monkeypatch.setattr(
            health, "_read_recent_query_latencies",
            lambda window_minutes: (recent, baseline),
        )
        result = health._latency_degradation_check(threshold=1.8)
        assert result["action"] == "ok"
        assert result["ratio"] is not None
        assert result["ratio"] < 1.8

    def test_triggers_restart_when_over_threshold(self, monkeypatch):
        # Recent p95 ≈ 5000ms, baseline p95 ≈ 1000ms → ratio 5.0 >> 1.8
        recent = [4000, 5000, 5000, 5000, 5000, 5000]
        baseline = [800, 900, 1000, 1000, 1000] * 20
        monkeypatch.setattr(
            health, "_read_recent_query_latencies",
            lambda window_minutes: (recent, baseline),
        )
        # Mock the actual restart so we don't bounce ollama in tests.
        called = {"count": 0}
        def fake_restart():
            called["count"] += 1
            return True, "test-mock"
        monkeypatch.setattr(health, "_restart_ollama_daemon", fake_restart)
        # Reset cooldown to 0 for the test
        monkeypatch.setattr(health, "_last_restart_ts", 0.0)
        result = health._latency_degradation_check(threshold=1.8, cooldown_seconds=0)
        assert result["action"] == "restart_attempted"
        assert called["count"] == 1
        assert result["ratio"] >= 1.8

    def test_cooldown_prevents_repeat_restart(self, monkeypatch):
        recent = [4000, 5000, 5000, 5000, 5000, 5000]
        baseline = [800, 900, 1000, 1000, 1000] * 20
        monkeypatch.setattr(
            health, "_read_recent_query_latencies",
            lambda window_minutes: (recent, baseline),
        )
        called = {"count": 0}
        def fake_restart():
            called["count"] += 1
            return True, "test-mock"
        monkeypatch.setattr(health, "_restart_ollama_daemon", fake_restart)
        # Set last_restart to now → cooldown active
        monkeypatch.setattr(health, "_last_restart_ts", time.time())
        result = health._latency_degradation_check(
            threshold=1.8, cooldown_seconds=300,
        )
        assert result["action"] == "restart_skipped_cooldown"
        assert called["count"] == 0


class TestStartWatchdog:
    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("RAG_LATENCY_WATCHDOG_DISABLE", "1")
        # Reset module state
        monkeypatch.setattr(health, "_started", False)
        result = health.start_latency_degradation_watchdog()
        assert result is False

    def test_idempotent(self, monkeypatch):
        # Mock loop so the thread doesn't actually run forever
        monkeypatch.setattr(health, "_watchdog_loop", lambda *a, **kw: None)
        monkeypatch.setattr(health, "_started", False)
        monkeypatch.delenv("RAG_LATENCY_WATCHDOG_DISABLE", raising=False)
        first = health.start_latency_degradation_watchdog()
        second = health.start_latency_degradation_watchdog()
        assert first is True
        assert second is True  # Already started but doesn't error
