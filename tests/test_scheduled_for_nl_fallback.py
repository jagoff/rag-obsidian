"""Tests for `_validate_scheduled_for` NL fallback (bug del 2026-04-28).

El fix permite al LLM pasar `scheduled_for="a las 12:55"` en lugar de
exigirle ISO8601 calculado. La razón es que qwen2.5:7b consistentemente
alucina años random ("2028-11-29T12:55:00-03:00" para "a las 12:55"
de hoy). El validator ahora hace fallback a `_parse_natural_datetime`
con anchor=now, así no le exigimos al modelo hacer date math.

Cubre:
- ISO8601 directo: passthru sin tocar.
- NL ("a las HH:MM"): resuelve a hoy + esa hora si futuro, o mañana si
  ya pasó (vía dateparser PREFER_DATES_FROM=future).
- NL ambiguo / inválido: devuelve None (legacy: la card sale sin
  schedule, envío inmediato).
"""
from __future__ import annotations

from datetime import datetime
from unittest import mock

import pytest

import rag


def test_validate_iso_passthru():
    """ISO8601 valid string pasa-thru sin modificación."""
    iso = "2026-04-28T12:55:00-03:00"
    assert rag._validate_scheduled_for(iso) == iso


def test_validate_iso_naive_passthru():
    """ISO sin offset también pasa (Python 3.11+ acepta `fromisoformat`)."""
    iso = "2026-04-28T12:55:00"
    assert rag._validate_scheduled_for(iso) == iso


def test_validate_nl_a_las_hh_mm_resolves():
    """'a las HH:MM' resuelve a un ISO con esa hora exacta.

    No mockeamos datetime para no romper el `isinstance(parsed, datetime)`
    en el validator. El día resuelto puede variar según el wall-clock
    actual (hoy si futuro, mañana si pasó), pero la hora siempre es
    la que pidió el user, y el formato siempre es ISO con offset.
    """
    result = rag._validate_scheduled_for("a las 14:30")
    assert result is not None, "NL parsing failed"
    # Hour component is exact regardless of which day got picked.
    assert "T14:30:00" in result
    assert result.endswith("-03:00")
    # Result is a valid ISO that round-trips through fromisoformat.
    parsed = datetime.fromisoformat(result)
    assert parsed.hour == 14
    assert parsed.minute == 30


def test_validate_nl_invalid_returns_none():
    """Strings que no son ni ISO ni NL parseable devuelven None.

    El comportamiento esperado downstream: la card sale sin schedule
    (envío inmediato), comportamiento legacy.
    """
    assert rag._validate_scheduled_for("xyz garbage no se parsea") is None
    assert rag._validate_scheduled_for("") is None
    assert rag._validate_scheduled_for("   ") is None


def test_validate_non_string_returns_none():
    """Tipos que no son str (None, int, dict, etc.) devuelven None."""
    assert rag._validate_scheduled_for(None) is None
    assert rag._validate_scheduled_for(12345) is None
    assert rag._validate_scheduled_for({"foo": "bar"}) is None
    assert rag._validate_scheduled_for([]) is None


def test_validate_nl_relative_phrases():
    """'mañana 14:30', 'en 2 horas' resuelven con anchor=now.

    No fijamos el now porque dateparser lo lee del datetime.now() actual
    al resolver, y el anchor que pasamos a `_parse_natural_datetime` es
    el datetime real. Lo importante es que devuelva un ISO no-None.
    """
    for phrase in ["mañana 14:30", "en 2 horas", "esta tarde"]:
        result = rag._validate_scheduled_for(phrase)
        assert result is not None, f"NL phrase {phrase!r} no parsed"
        # Format check: ISO8601 with -03:00 offset.
        assert "T" in result
        assert result.endswith("-03:00")


def test_validate_iso_with_z_suffix_passes():
    """ISO con Z suffix (UTC) también es válido — `fromisoformat` 3.11+
    lo acepta.
    """
    # Python 3.11+ accepts the Z suffix natively; if running on older
    # the test will skip-fail with ValueError. Our `.python-version`
    # is 3.13.
    result = rag._validate_scheduled_for("2026-04-28T15:55:00Z")
    assert result == "2026-04-28T15:55:00Z"
