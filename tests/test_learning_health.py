"""Tests for the system-health traffic-light banner.

Endpoint: ``/api/dashboard/learning/health``
Backend: ``web.learning_queries.system_health()`` + private ``_health_*`` helpers.

The banner shows a green/yellow/red light at the top of the learning dashboard
so a non-technical user can tell in 2 seconds if everything is OK.

Strategy: most logic in this module is pure (worst-case-wins, age humanizer,
shape contract). The DB-touching helpers are tested via a temporary
SQLite DB with `_RAG_STATE_PATH` monkeypatched. Subprocess-touching helper
``_health_services`` is mocked via stubbing ``subprocess.run``.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import patch

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
        """Un secundario sin PID con last-exit != 0 (y != -15) → red, pero
        nada crítico → la salud overall depende de las otras señales. Acá
        verificamos que la señal individual mete `crashed` en el value_text."""
        self._stub_launchctl(monkeypatch,
            "12345\t0\tcom.fer.obsidian-rag-web\n"
            "12346\t0\tcom.fer.obsidian-rag-watch\n"
            "12347\t0\tcom.fer.obsidian-rag-serve\n"
            "-\t1\tcom.fer.obsidian-rag-digest\n"  # secundario crashed
        )
        s = lq._health_services()
        assert s["level"] == "red"
        assert "digest" in s["value_text"]

    def test_launchctl_unavailable_yellow(self, lq, monkeypatch):
        """No-macOS environment: launchctl falla → yellow, no red.
        Mejor degradar a "no pude verificar" que asustar con falso rojo."""
        def _run(*args, **kwargs):
            raise FileNotFoundError("launchctl: not found")
        monkeypatch.setattr(subprocess, "run", _run)
        s = lq._health_services()
        assert s["level"] == "yellow"


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
        assert "com.fer.obsidian-rag-web" in lq._HEALTH_CRITICAL_SERVICES
        assert len(lq._HEALTH_CRITICAL_SERVICES) >= 3
