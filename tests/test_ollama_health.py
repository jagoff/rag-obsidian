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


class TestEscalateKillRagIndex:
    """Watchdog escala a kill de `rag index` zombies cuando la degradación
    persiste sostenidamente Y los gates (in_flight / cooldown) bloquean
    el restart de Ollama. Esto es defensa en profundidad contra el bug
    del 2026-05-01 (sesión devin huérfana en otra TTY mantenía un index
    en `recvfrom` 1.5h, saturando Ollama; el chat del user quedó pegado
    porque el `in_flight_guard` impedía el restart correctamente — pero
    nadie mataba al consumer real)."""

    @pytest.fixture(autouse=True)
    def _reset_counter(self):
        """Aislar tests: reset el counter global antes de cada uno."""
        health._consecutive_degraded_skips = 0
        yield
        health._consecutive_degraded_skips = 0

    def _setup_degraded(self, monkeypatch):
        """Helper: simula degradación sostenida + in_flight chat para que
        el check siempre devuelva `restart_skipped_in_flight`."""
        monkeypatch.setattr(
            health, "_read_recent_query_latencies",
            lambda window_minutes: ([200_000] * 10, [50_000] * 50),
        )
        monkeypatch.setattr(health, "_in_flight_count", 1)

    def test_skips_increment_counter(self, monkeypatch):
        """Cada check degraded incrementa `_consecutive_degraded_skips`."""
        self._setup_degraded(monkeypatch)
        for expected in range(1, 5):
            result = health._latency_degradation_check(threshold=1.8)
            assert result["action"] == "restart_skipped_in_flight"
            assert health._consecutive_degraded_skips == expected
            assert f"degraded_skips={expected}/" in result["reason"]

    def test_ok_resets_counter(self, monkeypatch):
        """Cuando el ratio vuelve a < threshold, el counter se resetea
        para evitar disparar la escalation a partir de skips de hace
        horas que ya quedaron stale."""
        self._setup_degraded(monkeypatch)
        # 5 skips degraded
        for _ in range(5):
            health._latency_degradation_check(threshold=1.8)
        assert health._consecutive_degraded_skips == 5
        # Ahora simular recovery: ratio < threshold
        monkeypatch.setattr(
            health, "_read_recent_query_latencies",
            lambda window_minutes: ([60_000] * 10, [50_000] * 50),
        )
        result = health._latency_degradation_check(threshold=1.8)
        assert result["action"] == "ok"
        assert health._consecutive_degraded_skips == 0

    def test_escalates_after_threshold_skips(self, monkeypatch):
        """Tras `ESCALATE_KILL_AFTER_SKIPS` skips consecutivos, el watchdog
        escala a `_kill_stuck_rag_index_clients` y resetea el counter."""
        self._setup_degraded(monkeypatch)
        kill_calls: dict[str, int] = {"count": 0}

        def fake_kill():
            kill_calls["count"] += 1
            return 2, [11111, 22222]

        monkeypatch.setattr(health, "_kill_stuck_rag_index_clients", fake_kill)
        monkeypatch.setattr(health, "ESCALATE_KILL_AFTER_SKIPS", 5)

        # 4 skips: counter sube, NO escala
        for _ in range(4):
            r = health._latency_degradation_check(threshold=1.8)
            assert r["action"] == "restart_skipped_in_flight"
        assert kill_calls["count"] == 0

        # 5to skip: hits threshold → escala
        result = health._latency_degradation_check(threshold=1.8)
        assert result["action"] == "escalated_killed_rag_index"
        assert kill_calls["count"] == 1
        assert "killed=2" in result["reason"]
        assert "11111" in result["reason"]
        assert "prev_action=restart_skipped_in_flight" in result["reason"]
        # Counter reset post-escalation
        assert health._consecutive_degraded_skips == 0

    def test_restart_attempted_resets_counter(self, monkeypatch):
        """Si los gates eventualmente liberan y el restart se ejecuta,
        el counter se resetea — no queremos escalar a kill JUSTO después
        de un restart que apenas se firmó."""
        # Primero unos skips degraded
        self._setup_degraded(monkeypatch)
        for _ in range(3):
            health._latency_degradation_check(threshold=1.8)
        assert health._consecutive_degraded_skips == 3
        # Ahora liberar el gate (in_flight=0, cooldown=0) y mockear restart
        monkeypatch.setattr(health, "_in_flight_count", 0)
        monkeypatch.setattr(health, "_last_restart_ts", 0.0)
        monkeypatch.setattr(health, "_restart_ollama_daemon", lambda: (True, "mock"))
        result = health._latency_degradation_check(
            threshold=1.8, cooldown_seconds=0,
        )
        assert result["action"] == "restart_attempted"
        # Counter reset post-restart real
        assert health._consecutive_degraded_skips == 0


class TestListOllamaClients:
    """`_list_ollama_clients()` parsea `lsof -i :11434 -P -F pc` y devuelve
    los PIDs no-ollama hablando con el daemon."""

    def test_returns_empty_when_lsof_missing(self, monkeypatch):
        def fake_run(cmd, **kw):
            raise FileNotFoundError("lsof not installed")
        monkeypatch.setattr(health.subprocess, "run", fake_run)
        assert health._list_ollama_clients() == []

    def test_returns_empty_when_lsof_fails(self, monkeypatch):
        class FakeResult:
            returncode = 1
            stdout = ""
            stderr = "lsof: error"
        monkeypatch.setattr(
            health.subprocess, "run",
            lambda *a, **kw: FakeResult(),
        )
        assert health._list_ollama_clients() == []

    def test_excludes_ollama_self(self, monkeypatch):
        """El daemon ollama y los runners NO deben aparecer como clients."""
        # First call: lsof — return ollama daemon + a python rag index
        # Second call: ps -p <pid> for python (ollama gets skipped before ps)
        lsof_output = (
            "p1234\n"   # ollama daemon
            "collama\n"
            "p5678\n"   # ollama runner
            "collama-runner\n"
            "p9999\n"   # rag index
            "cpython3.1\n"  # truncated lsof field
        )
        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            calls.append(list(cmd))
            class R:
                returncode = 0
                stdout = lsof_output if "lsof" in cmd else "/path/python3 /bin/rag index --foo\n"
                stderr = ""
            return R()

        monkeypatch.setattr(health.subprocess, "run", fake_run)
        clients = health._list_ollama_clients()
        # Solo el PID 9999 sobrevive al filtro
        assert len(clients) == 1
        assert clients[0][0] == 9999
        assert "rag index" in clients[0][1]


class TestKillStuckRagIndexClients:
    """`_kill_stuck_rag_index_clients()` SOLO mata procesos cuyo command
    matchea `rag index` literal. Cualquier otro consumer de Ollama (web
    server, rag watch, rag chat, etc.) queda intacto — la heurística es
    conservadora a propósito para minimizar false positives."""

    def test_no_targets_means_no_kills(self, monkeypatch):
        monkeypatch.setattr(
            health, "_list_ollama_clients",
            lambda: [(1234, "ollama serve"),
                     (5678, "/python /bin/rag watch --all-vaults"),
                     (9999, "/python /web/server.py")],
        )
        kill_log: list[tuple[int, int]] = []
        monkeypatch.setattr(
            health.os, "kill",
            lambda pid, sig: kill_log.append((pid, sig)),
        )
        n, pids = health._kill_stuck_rag_index_clients()
        assert n == 0
        assert pids == []
        assert kill_log == []  # nada se mató

    def test_kills_rag_index_only(self, monkeypatch):
        """De 4 clients (rag watch + web server + rag query + rag index),
        solo `rag index` se va. Los otros quedan intactos."""
        monkeypatch.setattr(
            health, "_list_ollama_clients",
            lambda: [
                (1111, "/python /bin/rag watch --all-vaults"),
                (2222, "/python /web/server.py"),
                (3333, "/python /bin/rag query 'hola'"),
                (4444, "/python /bin/rag index --no-contradict"),
            ],
        )
        kill_log: list[tuple[int, int]] = []

        def fake_kill(pid, sig):
            kill_log.append((pid, sig))
            # Simulate signal 0 returning OSError (process dead) for our target
            if sig == 0:
                raise OSError("dead")

        monkeypatch.setattr(health.os, "kill", fake_kill)
        # Avoid the 5s sleep in the real function
        monkeypatch.setattr(health.time, "sleep", lambda s: None)
        n, pids = health._kill_stuck_rag_index_clients()
        assert n == 1
        assert pids == [4444]
        # Solo SIGTERM al target (signal 0 sí va para health-check)
        sigterm_targets = [p for p, s in kill_log if s == health.signal.SIGTERM]
        assert sigterm_targets == [4444]
        # Ningún SIGTERM al watch/web/query
        assert 1111 not in sigterm_targets
        assert 2222 not in sigterm_targets
        assert 3333 not in sigterm_targets

    def test_sigkill_survivors(self, monkeypatch):
        """Si el target sigue vivo después del SIGTERM + sleep, escalar a SIGKILL."""
        monkeypatch.setattr(
            health, "_list_ollama_clients",
            lambda: [(8888, "/python /bin/rag index")],
        )
        kill_log: list[tuple[int, int]] = []

        def fake_kill(pid, sig):
            kill_log.append((pid, sig))
            # Signal 0 NO raises → proceso sigue vivo después del SIGTERM

        monkeypatch.setattr(health.os, "kill", fake_kill)
        monkeypatch.setattr(health.time, "sleep", lambda s: None)
        n, pids = health._kill_stuck_rag_index_clients()
        assert n == 1
        assert pids == [8888]
        # Tanto SIGTERM como SIGKILL fueron mandados al survivor
        signals_sent = [s for p, s in kill_log if p == 8888]
        assert health.signal.SIGTERM in signals_sent
        assert health.signal.SIGKILL in signals_sent
