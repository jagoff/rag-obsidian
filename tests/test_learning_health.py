"""Tests for the system-health traffic-light banner.

Endpoint: ``/api/learning/health``
Backend: ``web.learning_queries.system_health()`` + private ``_health_*`` helpers.

The banner shows a green/yellow/red light at the top of the learning dashboard
so a non-technical user can tell in 2 seconds if everything is OK.

Strategy: most logic in this module is pure (worst-case-wins, age humanizer,
shape contract). The DB-touching helpers are tested via a temporary
SQLite DB with `_RAG_STATE_PATH` monkeypatched. Subprocess-touching helper
``_health_services`` is mocked via stubbing ``subprocess.run``.
"""
from __future__ import annotations

import contextlib
from datetime import datetime
import sqlite3
import subprocess

import pytest


@pytest.fixture
def lq():
    """Lazy import: keeps the module out of import time so a syntax error in
    learning_queries doesn't break unrelated tests."""
    import web.learning_queries as _lq
    return _lq


# ── Pure helpers ────────────────────────────────────────────────────────────

class TestLevelWorst:
    """`_level_worst` implements worst-case-wins: red > yellow > green."""

    def test_all_green(self, lq):
        assert lq._level_worst(["green", "green", "green"]) == "green"

    def test_one_yellow(self, lq):
        assert lq._level_worst(["green", "yellow", "green"]) == "yellow"

    def test_one_red(self, lq):
        assert lq._level_worst(["green", "yellow", "red"]) == "red"

    def test_red_dominates_yellow(self, lq):
        assert lq._level_worst(["yellow", "yellow", "red"]) == "red"

    def test_empty_list_defaults_green(self, lq):
        # Defensive: si no hay señales, asumimos OK. La UI maneja igual el
        # estado loading via la clase .is-loading que no es ninguno de estos.
        assert lq._level_worst([]) == "green"

    def test_unknown_levels_ignored(self, lq):
        # Si una señal devuelve un level desconocido (bug en _signal), el
        # default es green — peor caso es underreporting, mejor que crashear.
        assert lq._level_worst(["green", "purple"]) == "green"


class TestHumanizeAge:
    """`_humanize_age` produces user-readable strings in Spanish."""

    @pytest.mark.parametrize("seconds, expected", [
        (0, "0 segundos"),
        (1, "1 segundo"),
        (45, "45 segundos"),
        (60, "1 minuto"),
        (90, "1 minuto"),  # truncates to whole minutes
        (300, "5 minutos"),
        (3600, "1 hora"),
        (7200, "2 horas"),
        (3600 * 24, "24 horas"),  # under 48h boundary
        (3600 * 47, "47 horas"),
        (3600 * 48, "2 días"),  # boundary: 48h → days
        (3600 * 24 * 7, "7 días"),
    ])
    def test_age_formatting(self, lq, seconds, expected):
        assert lq._humanize_age(seconds) == expected

    def test_negative_clamps_to_zero(self, lq):
        # Si el reloj se desfasó hacia atrás (NTP, dock sleep), no queremos
        # mostrar "-3600 segundos" — lo capamos a 0.
        assert lq._humanize_age(-3600) == "0 segundos"


# ── Signal shape contract ───────────────────────────────────────────────────

class TestSignalShape:
    """Every signal returned by `_health_*` MUST conform to the same shape so
    the frontend renders without defensive checks."""

    REQUIRED_KEYS = {"key", "label", "level", "value_text", "value_raw",
                     "tooltip", "explanation"}

    def test_signal_helper_shape(self, lq):
        s = lq._signal(
            key="test", label="Test", level="green",
            value_text="all good", value_raw=1.0, tooltip="test",
            explanation="test",
        )
        assert set(s.keys()) == self.REQUIRED_KEYS
        assert s["level"] == "green"

    def test_system_health_shape(self, lq, monkeypatch):
        """End-to-end: stub launchctl + use the real DB. Verify the top-level
        contract that the frontend depends on."""
        # Mock launchctl so we don't depend on the test runner having any
        # rag daemons installed.
        def _fake_run(*args, **kwargs):
            class R:
                returncode = 0
                stdout = ""  # No daemons listed → all critical "missing"
            return R()
        monkeypatch.setattr(subprocess, "run", _fake_run)

        out = lq.system_health()
        assert "level" in out
        assert out["level"] in ("green", "yellow", "red")
        assert "headline" in out and isinstance(out["headline"], str)
        assert "summary" in out and isinstance(out["summary"], str)
        assert "checked_at" in out
        assert "signals" in out and isinstance(out["signals"], list)
        # Each signal must conform to the contract.
        for s in out["signals"]:
            assert set(s.keys()) >= self.REQUIRED_KEYS, \
                f"Signal {s.get('key')} missing keys: {self.REQUIRED_KEYS - set(s.keys())}"
            assert s["level"] in ("green", "yellow", "red"), s["level"]


class TestSystemHealthOverall:
    """El headline global debe distinguir gaps de cobertura vs fallas."""

    def _sig(self, lq, key, level, text, raw=1):
        return lq._signal(
            key=key, label=key, level=level,
            value_text=text, value_raw=raw, tooltip=key,
            explanation=key,
        )

    def test_insufficient_coverage_does_not_make_overall_yellow(self, lq, monkeypatch):
        def _retrieval(_column, *, key, **_kwargs):
            return self._sig(
                lq, key, "yellow", "Sin suficientes datos todavía", raw=None,
            )

        monkeypatch.setattr(lq, "_health_retrieval", _retrieval)
        monkeypatch.setattr(lq, "_health_services",
                            lambda: self._sig(lq, "services", "green", "OK"))
        monkeypatch.setattr(lq, "_health_vault_freshness",
                            lambda: self._sig(lq, "vault_freshness", "green", "OK"))
        monkeypatch.setattr(lq, "_health_errors_24h",
                            lambda: self._sig(lq, "errors_24h", "green", "OK"))
        monkeypatch.setattr(
            lq, "_health_response_speed",
            lambda: self._sig(
                lq, "response_speed", "yellow",
                "Sin suficientes consultas para medir (n=1)", raw=None,
            ),
        )

        out = lq.system_health()

        assert out["level"] == "green"
        assert out["headline"] == "Todo lo crítico funcionando"

    def test_actionable_yellow_still_makes_overall_yellow(self, lq, monkeypatch):
        def _retrieval(_column, *, key, **_kwargs):
            return self._sig(
                lq, key, "yellow", "Sin suficientes datos todavía", raw=None,
            )

        monkeypatch.setattr(lq, "_health_retrieval", _retrieval)
        monkeypatch.setattr(lq, "_health_services",
                            lambda: self._sig(lq, "services", "green", "OK"))
        monkeypatch.setattr(lq, "_health_vault_freshness",
                            lambda: self._sig(lq, "vault_freshness", "green", "OK"))
        monkeypatch.setattr(lq, "_health_errors_24h",
                            lambda: self._sig(lq, "errors_24h", "yellow", "1 degraded", raw=0.01))
        monkeypatch.setattr(
            lq, "_health_response_speed",
            lambda: self._sig(
                lq, "response_speed", "yellow",
                "Sin suficientes consultas para medir (n=1)", raw=None,
            ),
        )

        out = lq.system_health()

        assert out["level"] == "yellow"
        assert out["headline"] == "Hay algo para vigilar"

    def test_read_failure_is_not_treated_as_coverage_gap(self, lq):
        s = self._sig(
            lq, "retrieval_singles", "yellow", "No pude leer la base", raw=None,
        )
        assert not lq._is_coverage_gap_signal(s)


# ── Services helper (subprocess-mocked) ─────────────────────────────────────

class TestHealthServices:
    """Tests para `_health_services` — la única señal que invoca subprocess."""

    def _stub_launchctl(self, monkeypatch, stdout: str, returncode: int = 0):
        def _run(*args, **kwargs):
            class R:
                pass
            R.returncode = returncode
            R.stdout = stdout
            return R()
        monkeypatch.setattr(subprocess, "run", _run)

    def test_all_critical_running(self, lq, monkeypatch):
        """3 críticos con PID asignado → green."""
        self._stub_launchctl(monkeypatch,
            "12345\t0\tcom.fer.obsidian-rag-web\n"
            "12346\t0\tcom.fer.obsidian-rag-watch\n"
            "12347\t0\tcom.fer.obsidian-rag-serve\n"
        )
        s = lq._health_services()
        assert s["level"] == "green"
        assert "respondiendo bien" in s["value_text"].lower()

    def test_critical_missing_red(self, lq, monkeypatch):
        """`web` no aparece en la lista → red, marcado como (no instalado)."""
        self._stub_launchctl(monkeypatch,
            "12346\t0\tcom.fer.obsidian-rag-watch\n"
            "12347\t0\tcom.fer.obsidian-rag-serve\n"
        )
        s = lq._health_services()
        assert s["level"] == "red"
        assert "web" in s["value_text"]
        assert "no instalado" in s["value_text"]

    def test_critical_no_pid_red(self, lq, monkeypatch):
        """`web` listado pero con PID `-` → red (KeepAlive=true sin PID = caído)."""
        self._stub_launchctl(monkeypatch,
            "-\t0\tcom.fer.obsidian-rag-web\n"
            "12346\t0\tcom.fer.obsidian-rag-watch\n"
            "12347\t0\tcom.fer.obsidian-rag-serve\n"
        )
        s = lq._health_services()
        assert s["level"] == "red"

    def test_pid_with_negative_status_is_running(self, lq, monkeypatch):
        """Daemon con PID asignado AHORA es el estado actual; el `status`
        que reporta launchctl es del último exit ANTERIOR (típicamente -15
        SIGTERM post-restart). Si tiene PID, está vivo, regardless del
        status que aparezca. Regresión: el primer cut de la lógica marcaba
        cualquier status no-cero como crash, lo que daba falsos rojos
        después de cada `bootout && bootstrap`."""
        self._stub_launchctl(monkeypatch,
            "12345\t-15\tcom.fer.obsidian-rag-web\n"  # SIGTERM previous, ahora vivo
            "12346\t0\tcom.fer.obsidian-rag-watch\n"
            "12347\t0\tcom.fer.obsidian-rag-serve\n"
        )
        s = lq._health_services()
        assert s["level"] == "green", \
            f"PID asignado debe ganar al status anterior; got {s}"

    def test_secondary_crashed_is_yellow(self, lq, monkeypatch):
        """Un secundario sin PID con last-exit != 0 (y != -15) → yellow
        (cuando son 1-2; ≥3 escala a red — ver test_secondary_3_crashed_is_red).
        Refinado 2026-05-01 para coincidir con el docstring original que
        siempre dijo "1 secundario caído → yellow", aunque la implementación
        anterior lo marcaba red."""
        self._stub_launchctl(monkeypatch,
            "12345\t0\tcom.fer.obsidian-rag-web\n"
            "12346\t0\tcom.fer.obsidian-rag-watch\n"
            "-\t1\tcom.fer.obsidian-rag-digest\n"  # secundario crashed
        )
        s = lq._health_services()
        assert s["level"] == "yellow", f"1 secundario caído → yellow; got {s}"
        assert "digest" in s["value_text"]

    def test_secondary_3_crashed_is_red(self, lq, monkeypatch):
        """≥3 secundarios con last-exit != 0 → red (sistémico).
        Threshold conservador: si caen 3+ jobs distintos en sus últimas
        corridas hay algo sistémico (Ollama colgado, disco lleno, etc.)."""
        self._stub_launchctl(monkeypatch,
            "12345\t0\tcom.fer.obsidian-rag-web\n"
            "12346\t0\tcom.fer.obsidian-rag-watch\n"
            "-\t1\tcom.fer.obsidian-rag-digest\n"
            "-\t1\tcom.fer.obsidian-rag-emergent\n"
            "-\t1\tcom.fer.obsidian-rag-ingest-safari\n"
        )
        s = lq._health_services()
        assert s["level"] == "red", \
            f"3+ secundarios caídos → red (sistémico); got {s}"
        assert s["value_raw"] == 3

    def test_launchctl_unavailable_yellow(self, lq, monkeypatch):
        """No-macOS environment: launchctl falla → yellow, no red.
        Mejor degradar a "no pude verificar" que asustar con falso rojo."""
        def _run(*args, **kwargs):
            raise FileNotFoundError("launchctl: not found")
        monkeypatch.setattr(subprocess, "run", _run)
        s = lq._health_services()
        assert s["level"] == "yellow"


# ── Response speed helper ──────────────────────────────────────────────────

class TestHealthResponseSpeed:
    """Tests para `_health_response_speed` sobre DB aislada."""

    def _patch_temp_db(self, tmp_path, monkeypatch):
        import rag

        db_path = tmp_path / "telemetry.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "CREATE TABLE rag_queries ("
                "ts TEXT, t_retrieve REAL, t_gen REAL"
                ")"
            )

        @contextlib.contextmanager
        def _conn():
            conn = sqlite3.connect(db_path)
            try:
                yield conn
            finally:
                conn.close()

        monkeypatch.setattr(rag, "_ragvec_state_conn", _conn)
        monkeypatch.setattr(
            rag,
            "_sql_read_with_retry",
            lambda fn, *_args, **_kwargs: fn(),
        )
        return db_path

    def test_counts_total_samples_when_p95_bucket_would_be_empty(
        self, lq, tmp_path, monkeypatch,
    ):
        """Regresión: NTILE(100) con 1 row no crea bucket 95; el health
        decía n=0 aunque había consultas recientes."""
        db_path = self._patch_temp_db(tmp_path, monkeypatch)
        now = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO rag_queries(ts, t_retrieve, t_gen) VALUES (?, ?, ?)",
                (now, 5.0, 4.5),
            )

        s = lq._health_response_speed()

        assert s["level"] == "yellow"
        assert s["value_raw"] is None
        assert "(n=1)" in s["value_text"]
        assert "n=1 < min=" in s["tooltip"]

    def test_reports_nearest_rank_p95_when_enough_samples(
        self, lq, tmp_path, monkeypatch,
    ):
        db_path = self._patch_temp_db(tmp_path, monkeypatch)
        now = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                "INSERT INTO rag_queries(ts, t_retrieve, t_gen) VALUES (?, ?, ?)",
                [(now, float(i), 0.0) for i in range(1, 21)],
            )

        s = lq._health_response_speed()

        assert s["level"] == "green"
        assert s["value_raw"] == 19.0
        assert "n=20" in s["tooltip"]


# ── Threshold contract ──────────────────────────────────────────────────────

class TestHealthThresholds:
    """Smoke test: thresholds están definidos y son números sanos."""

    def test_retrieval_thresholds_ordered(self, lq):
        assert lq._HEALTH_RETRIEVAL_SINGLES_GREEN > lq._HEALTH_RETRIEVAL_SINGLES_RED
        assert lq._HEALTH_RETRIEVAL_CHAINS_GREEN > lq._HEALTH_RETRIEVAL_CHAINS_RED

    def test_speed_thresholds_ordered(self, lq):
        # Verde es más rápido que rojo.
        assert lq._HEALTH_SPEED_GREEN_S < lq._HEALTH_SPEED_RED_S

    def test_vault_freshness_ordered(self, lq):
        # Verde es "más reciente" (menos horas) que rojo.
        assert lq._HEALTH_VAULT_GREEN_HOURS < lq._HEALTH_VAULT_RED_HOURS

    def test_critical_services_set(self, lq):
        # No queremos que la lista crítica se accidentalmente vacíe.
        # Tras la deprecación de `serve` en commit 1326d85 (2026-05-01),
        # el set crítico bajó a 2: `web` (cubre /api/query post-merge) +
        # `watch` (file watcher). Si bajara a 1 o 0, algo se rompió.
        assert "com.fer.obsidian-rag-web" in lq._HEALTH_CRITICAL_SERVICES
        assert "com.fer.obsidian-rag-serve" not in lq._HEALTH_CRITICAL_SERVICES
        assert len(lq._HEALTH_CRITICAL_SERVICES) >= 2

    def test_secondary_red_threshold_sane(self, lq):
        # 3 = "varios jobs cayeron a la vez = sistémico". Si fuera 1
        # estaríamos como antes (red por un cron-job aislado, ruidoso).
        # Si fuera ≥10 nunca dispararía (=> alarma muerta).
        assert 2 <= lq._HEALTH_SECONDARY_RED_THRESHOLD <= 5
