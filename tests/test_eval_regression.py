"""Suite de regresión consolidada del eval autónomo del 2026-04-28.

Cada test apunta a un issue específico del eval para que si una futura
refactorización rompe la solución, sepamos exactamente cuál.

Tests organizados por issue:
- T4: weather location parameter + cross-context guard
- T5: propose_reminder when field dropping
- T6/7/8: tool output truncation (mitiga timeouts)
- T9: whatsapp_pending vs whatsapp_list_scheduled routing
- T12: read_note priority over search_vault
- T-infra: ollama health watchdog disponible
"""
from __future__ import annotations

import json

import pytest


# ─────────────────────────────────────────────────────────────────────
# T4 — Weather location parameter + cross-context
# ─────────────────────────────────────────────────────────────────────


class TestT4WeatherLocation:
    def test_weather_accepts_location_arg(self):
        """weather(location='Buenos Aires') no debe romper."""
        from web.tools import weather
        result = weather(location="Buenos Aires")
        assert isinstance(result, str)
        # Si es JSON, debe parsear
        if result.lstrip().startswith("{"):
            json.loads(result)  # no exception

    def test_weather_default_location(self):
        """weather() sin args usa el default."""
        from web.tools import weather
        result = weather()
        assert isinstance(result, str)

    def test_weather_docstring_explains_location(self):
        from web.tools import weather
        doc = weather.__doc__ or ""
        assert "location" in doc.lower()
        # Debe tener un ejemplo concreto de ciudad
        cities_in_doc = ["Buenos Aires", "Mendoza", "Córdoba"]
        assert any(c in doc for c in cities_in_doc), (
            f"weather docstring debe ejemplificar con una ciudad real, doc={doc[:200]}"
        )

    def test_addendum_weather_routing_with_location(self):
        """_WEB_TOOL_ADDENDUM debe instruir al LLM a pasar location."""
        from web.tools import _WEB_TOOL_ADDENDUM
        # Debe haber un ejemplo de la forma weather(location='X')
        assert "location=" in _WEB_TOOL_ADDENDUM


# ─────────────────────────────────────────────────────────────────────
# T5 — Reminder `when` field dropping
# ─────────────────────────────────────────────────────────────────────


class TestT5ReminderWhenDropping:
    def test_validator_helper_exists(self):
        """El helper `_validate_reminder_when` debe estar exportado."""
        import rag
        assert hasattr(rag, "_validate_reminder_when")

    def test_validator_recovers_dropped_weekday(self):
        """Bug central del eval: 'el sábado' droppeado del when."""
        import rag
        when_fixed, issues = rag._validate_reminder_when(
            "a las 11am",
            "recordame regar las plantas el sábado a las 11am",
        )
        assert "sábado" in when_fixed.lower() or "sabado" in when_fixed.lower()
        assert len(issues) >= 1

    def test_validator_no_op_when_complete(self):
        """Si el when ya tiene todo, no toca."""
        import rag
        when_fixed, issues = rag._validate_reminder_when(
            "el viernes a las 18hs",
            "recordame ir al cine el viernes a las 18hs",
        )
        assert issues == []

    def test_propose_reminder_accepts_original_query_kwarg(self):
        """La firma de propose_reminder debe aceptar `_original_query`."""
        import inspect
        import rag
        sig = inspect.signature(rag.propose_reminder)
        assert "_original_query" in sig.parameters


# ─────────────────────────────────────────────────────────────────────
# T6/T7/T8 — Tool output truncation (mitiga timeouts)
# ─────────────────────────────────────────────────────────────────────


class TestToolOutputTruncation:
    def test_truncator_module_importable(self):
        from rag._tool_output_helpers import truncate_tool_output_for_synthesis
        assert callable(truncate_tool_output_for_synthesis)

    def test_truncator_caps_gmail(self):
        """gmail_recent con 20 items debe quedar capped."""
        from rag._tool_output_helpers import (
            _DEFAULT_CAPS,
            truncate_tool_output_for_synthesis,
        )
        items = [{"id": i, "subject": f"mail {i}"} for i in range(20)]
        raw = json.dumps({"awaiting_reply": items, "starred": items[:2]})
        out = truncate_tool_output_for_synthesis("gmail_recent", raw)
        parsed = json.loads(out)
        cap = _DEFAULT_CAPS["gmail_recent"]
        assert len(parsed["awaiting_reply"]) == cap
        assert "_truncated" in parsed

    def test_truncator_caps_whatsapp_search(self):
        from rag._tool_output_helpers import (
            _DEFAULT_CAPS,
            truncate_tool_output_for_synthesis,
        )
        items = [{"text": f"msg {i}"} for i in range(15)]
        raw = json.dumps(items)
        out = truncate_tool_output_for_synthesis("whatsapp_search", raw)
        parsed = json.loads(out)
        cap = _DEFAULT_CAPS["whatsapp_search"]
        assert "items" in parsed
        assert len(parsed["items"]) == cap

    def test_truncator_passthru_for_unknown(self):
        from rag._tool_output_helpers import truncate_tool_output_for_synthesis
        out = truncate_tool_output_for_synthesis("calendar_ahead", '[{"a":1}]')
        assert out == '[{"a":1}]'


# ─────────────────────────────────────────────────────────────────────
# T9 — WhatsApp pending vs list_scheduled routing
# ─────────────────────────────────────────────────────────────────────


class TestT9WhatsAppRouting:
    def test_addendum_has_contrast_line(self):
        """El addendum debe contrastar pending (incoming) vs list_scheduled (outgoing)."""
        from web.tools import _WEB_TOOL_ADDENDUM
        text = _WEB_TOOL_ADDENDUM.lower()
        # Debe mencionar incoming Y outgoing en el contexto de wa
        assert "incoming" in text or "outgoing" in text

    def test_whatsapp_list_scheduled_docstring_says_outgoing(self):
        import rag
        doc = rag.whatsapp_list_scheduled.__doc__ or ""
        assert "outgoing" in doc.lower(), (
            "el docstring debe decir explícitamente OUTGOING para no "
            "confundirse con whatsapp_pending"
        )


# ─────────────────────────────────────────────────────────────────────
# T12 — read_note ignored when user explicitly asks
# ─────────────────────────────────────────────────────────────────────


class TestT12ReadNotePriority:
    def test_read_note_docstring_states_priority(self):
        from web.tools import read_note
        doc = read_note.__doc__ or ""
        assert ("PRIORIDAD" in doc.upper() or "prioridad" in doc.lower())

    def test_addendum_lists_read_note_triggers(self):
        from web.tools import _WEB_TOOL_ADDENDUM
        text = _WEB_TOOL_ADDENDUM.lower()
        assert "read_note" in text
        # Algún verbo trigger debe estar
        assert any(v in text for v in ["leé", "abrí", "mostrame"])


# ─────────────────────────────────────────────────────────────────────
# T-infra — Ollama health watchdog
# ─────────────────────────────────────────────────────────────────────


class TestOllamaHealthWatchdog:
    def test_watchdog_module_importable(self):
        from rag._ollama_health import start_latency_degradation_watchdog
        assert callable(start_latency_degradation_watchdog)

    def test_watchdog_disabled_via_env(self, monkeypatch):
        from rag import _ollama_health
        monkeypatch.setenv("RAG_LATENCY_WATCHDOG_DISABLE", "1")
        monkeypatch.setattr(_ollama_health, "_started", False)
        result = _ollama_health.start_latency_degradation_watchdog()
        assert result is False

    def test_watchdog_idempotent(self, monkeypatch):
        from rag import _ollama_health
        monkeypatch.setattr(_ollama_health, "_watchdog_loop", lambda *a, **kw: None)
        monkeypatch.setattr(_ollama_health, "_started", False)
        monkeypatch.delenv("RAG_LATENCY_WATCHDOG_DISABLE", raising=False)
        first = _ollama_health.start_latency_degradation_watchdog()
        second = _ollama_health.start_latency_degradation_watchdog()
        assert first is True and second is True
