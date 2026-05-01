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


class TestInFlightGuard:
    """Watchdog NO debe restartear ollama mientras hay /api/chat streaming."""

    def test_skips_restart_when_chat_in_flight(self, monkeypatch):
        # Recent p95 mucho mayor que baseline → ratio dispara restart.
        recent = [200_000] * 10
        baseline = [50_000] * 50
        monkeypatch.setattr(
            health, "_read_recent_query_latencies",
            lambda window_minutes: (recent, baseline),
        )
        # Simular un chat en vuelo via el counter.
        monkeypatch.setattr(health, "_in_flight_count", 1)
        monkeypatch.setattr(health, "_last_restart_ts", 0.0)
        called = {"count": 0}
        monkeypatch.setattr(
            health, "_restart_ollama_daemon",
            lambda: (called.update({"count": called["count"] + 1}) or (True, "test")),
        )
        result = health._latency_degradation_check(threshold=1.8, cooldown_seconds=0)
        assert result["action"] == "restart_skipped_in_flight"
        assert "in_flight=1" in result["reason"]
        assert called["count"] == 0  # NO se llamó el restart

    def test_proceeds_when_no_chat_in_flight(self, monkeypatch):
        recent = [200_000] * 10
        baseline = [50_000] * 50
        monkeypatch.setattr(
            health, "_read_recent_query_latencies",
            lambda window_minutes: (recent, baseline),
        )
        monkeypatch.setattr(health, "_in_flight_count", 0)
        monkeypatch.setattr(health, "_last_restart_ts", 0.0)
        called = {"count": 0}
        monkeypatch.setattr(
            health, "_restart_ollama_daemon",
            lambda: (called.update({"count": called["count"] + 1}) or (True, "test")),
        )
        result = health._latency_degradation_check(threshold=1.8, cooldown_seconds=0)
        assert result["action"] == "restart_attempted"
        assert called["count"] == 1

    def test_begin_end_chat_counter(self):
        # Reset to 0 (other tests may have modified it).
        health._in_flight_count = 0
        assert health.in_flight_chats() == 0
        health.begin_chat()
        health.begin_chat()
        assert health.in_flight_chats() == 2
        health.end_chat()
        assert health.in_flight_chats() == 1
        health.end_chat()
        health.end_chat()  # should clamp at 0, not go negative
        assert health.in_flight_chats() == 0


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


class TestPersistedRestartTs:
    """`_last_restart_ts` debe sobrevivir restarts del web server.

    Sin esto, cada vez que el web server reinicia (manual, plist reload,
    crash) el watchdog ve `now - 0 = enorme >> cooldown` y restartea
    ollama el primer check post-arranque. Si las queries lentas que
    dispararon p95 alto siguen en la window de 10min, el ciclo
    autodestructivo continúa: ollama restart → cold-load → próxima
    query lenta → watchdog ve p95 alto otra vez → restart.

    El fix persiste el ts en `~/.local/share/obsidian-rag/ollama_health_state.json`.
    """

    def test_persist_round_trip(self, tmp_path, monkeypatch):
        """Un `_persist_restart_ts(ts)` seguido de `_load_persisted_restart_ts()`
        recupera el mismo timestamp."""
        state_file = tmp_path / "ollama_health_state.json"
        monkeypatch.setattr(health, "_STATE_FILE", state_file)

        ts = time.time() - 100.0  # 100s atrás
        health._persist_restart_ts(ts)
        loaded = health._load_persisted_restart_ts()
        assert abs(loaded - ts) < 0.001

    def test_load_returns_zero_when_no_file(self, tmp_path, monkeypatch):
        """Sin archivo previo, retorna 0.0 (mismo comportamiento que pre-fix)."""
        state_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr(health, "_STATE_FILE", state_file)
        assert health._load_persisted_restart_ts() == 0.0

    def test_load_returns_zero_on_corrupt_json(self, tmp_path, monkeypatch):
        """JSON corrupto → 0.0 silent-fail (no rompe el watchdog)."""
        state_file = tmp_path / "corrupt.json"
        state_file.write_text("not-json {{")
        monkeypatch.setattr(health, "_STATE_FILE", state_file)
        assert health._load_persisted_restart_ts() == 0.0

    def test_persist_after_restart_attempt(self, tmp_path, monkeypatch):
        """Cuando el check dispara `restart_attempted`, `_last_restart_ts`
        queda persistido en disco antes de que el daemon bouncing termine."""
        state_file = tmp_path / "ollama_health_state.json"
        monkeypatch.setattr(health, "_STATE_FILE", state_file)
        recent = [200_000] * 10
        baseline = [50_000] * 50
        monkeypatch.setattr(
            health, "_read_recent_query_latencies",
            lambda window_minutes: (recent, baseline),
        )
        monkeypatch.setattr(health, "_in_flight_count", 0)
        monkeypatch.setattr(health, "_last_restart_ts", 0.0)
        monkeypatch.setattr(
            health, "_restart_ollama_daemon",
            lambda: (True, "test-mock"),
        )

        before = time.time()
        result = health._latency_degradation_check(threshold=1.8, cooldown_seconds=0)
        after = time.time()
        assert result["action"] == "restart_attempted"

        # File debe existir y tener el ts dentro del rango.
        assert state_file.exists()
        loaded = health._load_persisted_restart_ts()
        assert before <= loaded <= after

    def test_start_watchdog_hydrates_from_disk(self, tmp_path, monkeypatch):
        """Al arrancar el watchdog, `_last_restart_ts` se hidrata del archivo
        en lugar de quedar en 0.0. Eso evita el restart autodestructivo
        post-boot del web server."""
        state_file = tmp_path / "ollama_health_state.json"
        monkeypatch.setattr(health, "_STATE_FILE", state_file)
        # Simulamos un restart hace 60s persistido en disco.
        ts_60s_ago = time.time() - 60.0
        health._persist_restart_ts(ts_60s_ago)

        # Reset module state como si el proceso hubiera reiniciado.
        monkeypatch.setattr(health, "_started", False)
        monkeypatch.setattr(health, "_last_restart_ts", 0.0)
        monkeypatch.setattr(health, "_watchdog_loop", lambda *a, **kw: None)
        monkeypatch.delenv("RAG_LATENCY_WATCHDOG_DISABLE", raising=False)

        ok = health.start_latency_degradation_watchdog()
        assert ok is True
        # El ts hidratado debe estar dentro de ~5s del valor original.
        assert abs(health._last_restart_ts - ts_60s_ago) < 5.0

    def test_start_watchdog_no_hydrate_when_file_absent(self, tmp_path, monkeypatch):
        """Sin archivo previo (primera ejecución del binario), `_last_restart_ts`
        queda en 0.0 — comportamiento legacy preservado."""
        state_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr(health, "_STATE_FILE", state_file)
        monkeypatch.setattr(health, "_started", False)
        monkeypatch.setattr(health, "_last_restart_ts", 0.0)
        monkeypatch.setattr(health, "_watchdog_loop", lambda *a, **kw: None)
        monkeypatch.delenv("RAG_LATENCY_WATCHDOG_DISABLE", raising=False)

        ok = health.start_latency_degradation_watchdog()
        assert ok is True
        assert health._last_restart_ts == 0.0
