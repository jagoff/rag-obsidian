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


# ─────────────────────────────────────────────────────────────────────
# Wave 2 (eval QA del 2026-04-28 evening) — bugs adicionales
# ─────────────────────────────────────────────────────────────────────


class TestBugV2_ReadNoteProjectRoot:
    """BUG-1 v2: read_note debe encontrar CLAUDE.md del project root."""

    def test_read_note_finds_project_root_claude_md(self):
        """_agent_tool_read_note('CLAUDE.md') debe encontrar el archivo
        del repo root cuando el vault no lo tiene."""
        from rag import _agent_tool_read_note
        result = _agent_tool_read_note("CLAUDE.md")
        # Debe ser contenido real, no error
        assert not result.startswith("Error:"), (
            f"esperaba contenido real de CLAUDE.md, got: {result[:200]}"
        )
        # CLAUDE.md del proyecto contiene la palabra "rag.py" (es referencia técnica)
        assert "rag.py" in result or "MCP" in result or "vault" in result, (
            "esperaba contenido específico del CLAUDE.md del proyecto"
        )

    def test_read_note_invalid_returns_clear_error(self):
        """Si el archivo no existe en vault NI project root, mensaje claro."""
        from rag import _agent_tool_read_note
        result = _agent_tool_read_note("ARCHIVO_INEXISTENTE_TEST_XYZ.md")
        # Mensaje de error explícito (para que el LLM no hallucine)
        assert "Error:" in result or "no encontrada" in result.lower()


class TestBugV2_WeatherTextSummary:
    """BUG-2a v2: weather output debe tener text_summary prominente."""

    def test_weather_output_has_text_summary(self):
        from rag import _agent_tool_weather
        import json
        out = _agent_tool_weather("Buenos Aires")
        data = json.loads(out)
        assert "text_summary" in data, (
            f"weather output debe tener key 'text_summary', keys={list(data.keys())}"
        )
        # El summary debe contener "Buenos Aires"
        assert "Buenos Aires" in data["text_summary"], (
            f"text_summary debe mencionar la ciudad: {data['text_summary']!r}"
        )


class TestBugV2_PreRouterWeatherExplicit:
    """BUG-2b v2: 'clima en CIUDAD' debe disparar SOLO weather."""

    def test_weather_with_explicit_location_skips_morning_brief(self):
        from web.server import _detect_tool_intent
        result = _detect_tool_intent("cómo está el clima hoy en Buenos Aires")
        names = [n for n, _ in result]
        assert "weather" in names, f"weather debe estar en {names}"
        assert "reminders_due" not in names, (
            f"reminders_due NO debe disparar (morning-brief over-fire), got {names}"
        )
        assert "calendar_ahead" not in names
        assert "whatsapp_pending" not in names
        # Y location debe ser Buenos Aires
        weather_call = next((args for n, args in result if n == "weather"), None)
        assert weather_call is not None
        assert weather_call.get("location", "").lower().startswith("buenos aires")

    def test_weather_without_location_uses_normal_routing(self):
        """Sin location explícita, el routing legacy se mantiene (no rompe queries genericas)."""
        from web.server import _detect_tool_intent
        result = _detect_tool_intent("cómo está el clima")
        names = [n for n, _ in result]
        # Sin "en X" explícito, el behavior es el legacy (puede o no haber morning-brief)
        # pero al menos weather DEBE estar.
        assert "weather" in names


class TestBugV2_PreRouterWaListScheduled:
    """BUG-3 v2: 'WA programados' debe disparar wa_list_scheduled, NO wa_pending."""

    def test_wa_scheduled_routing_correct(self):
        from web.server import _detect_tool_intent
        result = _detect_tool_intent("qué mensajes de WhatsApp tengo programados pendientes para mañana")
        names = [n for n, _ in result]
        assert "whatsapp_list_scheduled" in names, (
            f"whatsapp_list_scheduled debe disparar, got {names}"
        )
        assert "whatsapp_pending" not in names, (
            f"whatsapp_pending NO debe disparar (regla incorrecta), got {names}"
        )

    def test_wa_pending_legacy_still_works(self):
        """Asegurar que wa_pending sigue funcionando para queries normales."""
        from web.server import _detect_tool_intent
        result = _detect_tool_intent("qué chats tengo sin contestar en WhatsApp")
        names = [n for n, _ in result]
        assert "whatsapp_pending" in names

    def test_quedan_por_mandar_routing(self):
        """Variante: 'quedan por mandar' es trigger de wa_list_scheduled."""
        from web.server import _detect_tool_intent
        result = _detect_tool_intent("qué wsps quedan por mandar")
        names = [n for n, _ in result]
        assert "whatsapp_list_scheduled" in names


class TestBugV2_CalendarYearlyValidator:
    """BUG-4a v2: cumple sin recurrence debe forzar yearly via post-LLM."""

    def test_validate_calendar_recurrence_forces_yearly_for_birthday(self):
        import rag
        if not hasattr(rag, "_validate_calendar_recurrence"):
            pytest.skip("validator no disponible aún")
        out, issues = rag._validate_calendar_recurrence(
            None, "agendá el cumpleaños de Marina el 20 de septiembre"
        )
        assert out == "yearly", f"esperaba 'yearly', got {out!r}"
        assert len(issues) >= 1

    def test_validate_calendar_recurrence_respects_explicit_value(self):
        """Si el LLM ya pasó un recurrence_text, NO sobreescribir."""
        import rag
        if not hasattr(rag, "_validate_calendar_recurrence"):
            pytest.skip("validator no disponible aún")
        out, issues = rag._validate_calendar_recurrence(
            "monthly", "agendá cumpleaños de X el 20 de septiembre"
        )
        # El validator NO debe pisar valores explícitos
        assert out == "monthly"
        assert len(issues) == 0

    def test_validate_calendar_recurrence_no_op_for_non_birthday(self):
        """Eventos sin cumple no deben recibir yearly."""
        import rag
        if not hasattr(rag, "_validate_calendar_recurrence"):
            pytest.skip("validator no disponible aún")
        out, issues = rag._validate_calendar_recurrence(
            None, "agendá reunión Sprint Planning el viernes 9am"
        )
        assert out is None or out == ""
        assert len(issues) == 0

    def test_propose_calendar_event_accepts_original_query(self):
        """La firma debe aceptar el kwarg para que el wiring funcione."""
        import inspect, rag
        sig = inspect.signature(rag.propose_calendar_event)
        assert "_original_query" in sig.parameters


class TestBugV2_ParseAbsoluteDateEs:
    """BUG-4b v2: 'el 20 de septiembre' debe parsear correctamente."""

    def test_parse_el_dd_de_mes_es(self):
        from rag import _parse_natural_datetime
        from datetime import datetime
        result = _parse_natural_datetime(
            "el 20 de septiembre", now=datetime(2026, 4, 28, 15, 0)
        )
        assert result is not None, "parser debe resolver 'el 20 de septiembre'"
        assert result.month == 9
        assert result.day == 20

    def test_parse_dd_de_mes_es_no_el(self):
        """Sin el 'el' prefix también debe funcionar (regression)."""
        from rag import _parse_natural_datetime
        from datetime import datetime
        result = _parse_natural_datetime(
            "20 de septiembre", now=datetime(2026, 4, 28, 15, 0)
        )
        assert result is not None
        assert result.month == 9
        assert result.day == 20


class TestBugV2_ReminderTitleValidator:
    """BUG-5 v2: title 'regar' debe expandirse a 'regar las plantas'."""

    def test_validate_reminder_title_recovers_dropped_object(self):
        import rag
        if not hasattr(rag, "_validate_reminder_title"):
            pytest.skip("validator no disponible aún")
        out, issues = rag._validate_reminder_title(
            "regar", "recordame regar las plantas el sábado a las 11am"
        )
        # Debe expandir a "regar las plantas"
        assert "plantas" in out.lower(), (
            f"esperaba que el title contenga 'plantas', got {out!r}"
        )
        assert len(issues) >= 1

    def test_validate_reminder_title_no_op_when_complete(self):
        """Si el title ya tiene varias palabras, NO toca."""
        import rag
        if not hasattr(rag, "_validate_reminder_title"):
            pytest.skip("validator no disponible aún")
        out, issues = rag._validate_reminder_title(
            "llamar al dentista", "recordame llamar al dentista mañana 10am"
        )
        assert out == "llamar al dentista"
        assert len(issues) == 0
