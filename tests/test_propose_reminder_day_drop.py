"""Tests de regresión para el drop de día en propose_reminder
detectado en eval autónomo del 2026-04-28.

Bug observado: query "recordame regar las plantas el sábado a las 11am"
→ LLM pasaba when="a las 11am" (sin "el sábado") → reminder se creaba
para mañana 11am en vez del sábado próximo.

Fix: `_validate_reminder_when()` post-check recupera el día del
query original cuando el LLM lo droppea.
"""
from __future__ import annotations

import json

import pytest

import rag


class TestValidateReminderWhen:
    def test_no_drop_when_complete(self):
        when_fixed, issues = rag._validate_reminder_when(
            "el sábado a las 11am",
            "recordame regar las plantas el sábado a las 11am",
        )
        assert issues == []
        assert "sábado" in when_fixed.lower()

    def test_recovers_dropped_weekday(self):
        when_fixed, issues = rag._validate_reminder_when(
            "a las 11am",
            "recordame regar las plantas el sábado a las 11am",
        )
        assert any("sábado" in i.lower() or "sabado" in i.lower() for i in issues)
        assert "sábado" in when_fixed.lower() or "sabado" in when_fixed.lower()
        assert "11am" in when_fixed.lower()

    def test_recovers_dropped_mañana(self):
        when_fixed, issues = rag._validate_reminder_when(
            "a las 3pm",
            "recordame llamar a mama mañana a las 3pm",
        )
        assert len(issues) >= 1
        assert "mañana" in when_fixed.lower() or "manana" in when_fixed.lower()

    def test_no_temporal_info_no_change(self):
        when_fixed, issues = rag._validate_reminder_when(
            "",
            "recordame comprar leche",
        )
        assert issues == []
        assert when_fixed == ""

    def test_empty_when_with_temporal_in_query(self):
        when_fixed, issues = rag._validate_reminder_when(
            "",
            "recordame pagar tarjeta el lunes",
        )
        assert len(issues) >= 1
        assert "lunes" in when_fixed.lower()

    def test_no_double_recovery_when_weekday_present(self):
        """Si el LLM ya pasó el weekday, no debe duplicarlo."""
        when_fixed, issues = rag._validate_reminder_when(
            "el viernes a las 18hs",
            "recordame ir al cine el viernes a las 18hs",
        )
        assert issues == []
        # No debería haber dos "viernes" en el resultado
        assert when_fixed.lower().count("viernes") == 1

    def test_handles_none_inputs(self):
        when_fixed, issues = rag._validate_reminder_when(None, None)
        assert when_fixed == ""
        assert issues == []


class TestProposeReminderIntegration:
    def test_propose_reminder_recovers_dropped_day(self, monkeypatch):
        """Si _original_query se pasa y el LLM droppeó el día, propose_reminder
        debe recuperarlo antes de parsear la fecha."""
        # Mock _create_reminder para no escribir realmente en Apple Reminders
        monkeypatch.setattr(
            rag, "_create_reminder",
            lambda *a, **kw: (True, "x-apple-reminder://test-mock"),
        )
        raw = rag.propose_reminder(
            title="regar las plantas",
            when="a las 11am",  # <-- LLM droppeó "el sábado"
            _original_query="recordame regar las plantas el sábado a las 11am",
        )
        result = json.loads(raw)
        # due_text debería contener "sábado" porque el post-check lo recuperó
        due_text = (result.get("fields", {}).get("due_text") or "").lower()
        assert "sábado" in due_text or "sabado" in due_text, (
            f"esperaba 'sábado' en due_text, got: {due_text!r}"
        )

    def test_propose_reminder_without_original_query_unchanged(self, monkeypatch):
        """Backwards compat: sin _original_query, comportamiento es identical."""
        monkeypatch.setattr(
            rag, "_create_reminder",
            lambda *a, **kw: (True, "x-apple-reminder://test-mock"),
        )
        raw = rag.propose_reminder(
            title="regar las plantas",
            when="a las 11am",
        )
        result = json.loads(raw)
        # Sin _original_query, when permanece igual (no fix)
        due_text = (result.get("fields", {}).get("due_text") or "").lower()
        assert due_text == "a las 11am"
