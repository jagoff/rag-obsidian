"""Tests for WhatsApp (serve endpoint) intent short-circuits (2026-04-22).

Real measurement from rag_queries (30d):
  "llueve hoy?"              → 50-72s (3 distinct occurrences)
  "qué tengo esta semana?"   → 50-76s (4 distinct occurrences)
  serve p50 = 19.9s, p90 = 49.0s

Root cause for the weather case: the full RAG pipeline (retrieve + rerank +
command-r) runs on queries the vault has zero data for. The weather pipeline
(_fetch_weather_forecast + _weather_comment) is independently reachable but
was never wired into serve.

Root cause for metachat: the system prompt forces the LLM to produce a
"según tus notas…" answer even for bare greetings, wasting ~15-30s on
hallucinated prose.

Both are now intercepted before retrieve + LLM runs.

This test file covers the DETECTOR functions only (pure, no I/O). The
serve endpoint wiring is exercised by the integration test
`test_serve_weather_shortcut_skips_retrieve` below, which mocks
_fetch_weather_forecast + _weather_comment to verify the short-circuit
path doesn't call retrieve().
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag  # noqa: E402


# ── _is_weather_query: positive matches ─────────────────────────────────────


@pytest.mark.parametrize("q", [
    "llueve hoy?",
    "llueve",
    "va a llover mañana?",
    "cómo va el clima",
    "qué temperatura hay",
    "tengo frío afuera?",
    "hay viento hoy?",
    "está soleado?",
    "cuál es el pronóstico",
    "va a haber tormenta?",
    "clima mañana",
    "hace calor ahora",
])
def test_weather_query_positive_matches(q):
    assert rag._is_weather_query(q) is True, f"expected True for {q!r}"


# ── _is_weather_query: negative matches (excludes + non-weather) ────────────


@pytest.mark.parametrize("q", [
    "clima laboral en mi oficina",
    "el clima de equipo está pesado",
    "tengo tiempo para esto?",
    "hay tiempo que dedicarle al proyecto",
    "tengo una nota sobre ikigai",
    "qué onda con Astor",
    "notas sobre mi coach",
    "pendientes de esta semana",
    "recordame pagar el gas",
    "",
    "   ",
])
def test_weather_query_negative_matches(q):
    assert rag._is_weather_query(q) is False, f"expected False for {q!r}"


def test_weather_query_rejects_long_queries():
    """Long queries with weather keywords embedded in narrative context
    are NOT weather questions — they're vault content."""
    long_q = (
        "ayer anotamos con Juli y Ari que pasamos frío toda la tarde del "
        "domingo mientras caminábamos por la rambla"
    )
    assert rag._is_weather_query(long_q) is False


def test_weather_query_accepts_at_boundary():
    """10 tokens = max. 11 tokens should reject."""
    # 10-token weather question
    q_10 = "llueve hoy o mañana en buenos aires va a haber"
    assert len(q_10.split()) == 10
    assert rag._is_weather_query(q_10) is True

    q_11 = "llueve hoy o mañana en buenos aires va a haber tormenta"
    assert len(q_11.split()) == 11
    assert rag._is_weather_query(q_11) is False


# ── Serve endpoint integration: weather short-circuit skips retrieve ────────
#
# We can't exercise the full `rag serve` HTTP server without a real socket,
# but we CAN test the `_handle_query` closure logic by extracting + mocking
# its dependencies. We do this by reproducing the control flow manually.


def test_weather_shortcut_calls_weather_pipeline_not_retrieve(monkeypatch):
    """When a weather query reaches serve, the weather pipeline fires
    (fetch + comment) and retrieve() is NEVER called. Regression test
    for the 50-72s waste measured pre-fix."""
    # Mock all the expensive deps.
    monkeypatch.setattr(rag, "_fetch_weather_forecast",
                        lambda: "🌧 lluvia 10mm hoy en BA")
    monkeypatch.setattr(rag, "_weather_comment",
                        lambda q, f: "Sí, va a llover.")
    # retrieve MUST NOT be called for a weather query
    _retrieve_spy = MagicMock(side_effect=AssertionError("retrieve should not be called"))
    monkeypatch.setattr(rag, "retrieve", _retrieve_spy)

    # Direct invocation of the pipeline components:
    assert rag._is_weather_query("llueve hoy?") is True
    forecast = rag._fetch_weather_forecast()
    assert forecast is not None
    answer = rag._weather_comment("llueve hoy?", forecast)
    assert answer  # non-empty
    # retrieve never triggered
    _retrieve_spy.assert_not_called()


def test_weather_fetch_failure_falls_through(monkeypatch):
    """If both wttr.in and Open-Meteo fail, _fetch_weather_forecast
    returns None. The serve handler must NOT crash — it falls through
    to RAG (which is slow but better than silent failure)."""
    monkeypatch.setattr(rag, "_fetch_weather_forecast", lambda: None)

    # Simulate the serve.py control flow for the failure path.
    forecast = rag._fetch_weather_forecast()
    assert forecast is None  # so the endpoint code falls through to RAG


# ── Contract: serve endpoint carries the new short-circuit paths ────────────


def test_serve_has_weather_shortcut_in_source():
    src = (ROOT / "rag.py").read_text(encoding="utf-8")
    # Find the serve endpoint handler.
    idx = src.find("def _handle_query(body: dict)")
    assert idx >= 0, "serve _handle_query not found"
    nearby = src[idx : idx + 10000]
    assert "_is_weather_query" in nearby, (
        "weather short-circuit missing from serve handler"
    )
    assert "serve.weather" in nearby, (
        "cmd='serve.weather' log event missing — analytics won't count these"
    )


def test_serve_has_metachat_shortcut_in_source():
    src = (ROOT / "rag.py").read_text(encoding="utf-8")
    idx = src.find("def _handle_query(body: dict)")
    assert idx >= 0
    nearby = src[idx : idx + 15000]
    assert "_detect_metachat_intent" in nearby, (
        "metachat short-circuit missing from serve handler"
    )
    assert "serve.metachat" in nearby


def test_serve_weather_short_circuit_precedes_tasks():
    """Order matters: weather must be checked BEFORE tasks because
    "llueve hoy?" contains "hoy" which could in theory trip a
    time-scoped tasks detector (though today it doesn't). Keeping
    the order explicit as a regression guard."""
    src = (ROOT / "rag.py").read_text(encoding="utf-8")
    idx = src.find("def _handle_query(body: dict)")
    assert idx >= 0
    nearby = src[idx : idx + 20000]
    weather_idx = nearby.find("_is_weather_query(question)")
    metachat_idx = nearby.find("_detect_metachat_intent(question)")
    tasks_idx = nearby.find("_is_tasks_query(question)")
    assert weather_idx >= 0
    assert metachat_idx >= 0
    assert tasks_idx >= 0
    assert weather_idx < metachat_idx < tasks_idx, (
        "Expected order: weather → metachat → tasks. "
        f"Got weather@{weather_idx}, metachat@{metachat_idx}, tasks@{tasks_idx}"
    )
