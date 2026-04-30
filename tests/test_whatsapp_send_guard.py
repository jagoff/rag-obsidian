"""Tests para el hard kill-switch de envío al WhatsApp real bajo testing.

Bug 2026-04-30: tests del CLI `rag today` posteaban al grupo RagNet del
user porque ningún test mockeaba `_ambient_whatsapp_send`. 4 evening
briefs duplicados con texto placeholder ("texto de recap hoy", "item
sin tags", "seed 1/2") aparecieron en el WhatsApp real entre 13:54 y
13:56 — el user los vio.

Fix de defensa-en-profundidad:
  - Capa 1 (acá tested): guard inline en `_whatsapp_send_to_jid` que
    devuelve `False` sin hacer el HTTP POST cuando hay env var
    `RAG_DISABLE_WHATSAPP_SEND=1` o `RAG_TESTING=1`.
  - Capa 2 (conftest.py): autouse fixture `_block_real_whatsapp_send`
    que setea `RAG_DISABLE_WHATSAPP_SEND=1` para toda la suite por
    default. Tests del wire format opt-out con `monkeypatch.delenv`.

Estos tests verifican que la CAPA 1 funciona — la conftest puede ser
borrada/modificada; el guard inline es la última línea de defensa.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from rag.integrations.whatsapp import _whatsapp_send_to_jid


def test_conftest_default_blocks_real_send():
    """Sanity: el conftest top-level setea RAG_DISABLE_WHATSAPP_SEND=1
    por default — el send retorna False sin postear, en cualquier test
    que no haga `monkeypatch.delenv` explícito."""
    # El conftest fixture `_block_real_whatsapp_send` ya seteó la env var.
    assert os.environ.get("RAG_DISABLE_WHATSAPP_SEND") == "1"
    result = _whatsapp_send_to_jid("test@s.whatsapp.net", "test msg")
    assert result is False


def test_rag_disable_whatsapp_send_blocks(monkeypatch):
    """Override explícito via env var. Test idempotente vs el conftest:
    si el conftest no estuviera, este sigue funcionando."""
    monkeypatch.setenv("RAG_DISABLE_WHATSAPP_SEND", "1")
    result = _whatsapp_send_to_jid("test@s.whatsapp.net", "test msg")
    assert result is False


def test_rag_testing_alias_blocks(monkeypatch):
    """`RAG_TESTING=1` es alias semántico de `RAG_DISABLE_WHATSAPP_SEND=1`.
    Lo usan algunos integration tests que ya tienen ese env var convention."""
    # Limpiamos el primario para asegurar que SOLO el alias bloquea.
    monkeypatch.delenv("RAG_DISABLE_WHATSAPP_SEND", raising=False)
    monkeypatch.setenv("RAG_TESTING", "1")
    result = _whatsapp_send_to_jid("test@s.whatsapp.net", "test msg")
    assert result is False


def test_guard_truthy_values():
    """El guard acepta varios valores truthy comunes."""
    for val in ("1", "true", "yes", "TRUE", "Yes"):
        with patch.dict(os.environ, {"RAG_DISABLE_WHATSAPP_SEND": val}, clear=False):
            assert _whatsapp_send_to_jid("x@x", "y") is False, f"falló con {val!r}"


def test_guard_falsy_values_does_not_block(monkeypatch):
    """Valores falsy NO activan el guard. Mockeamos urlopen para verificar
    que el send INTENTA el POST (no corto-circuit por guard)."""
    monkeypatch.delenv("RAG_DISABLE_WHATSAPP_SEND", raising=False)
    monkeypatch.delenv("RAG_TESTING", raising=False)
    for val in ("0", "false", "no", "off"):
        monkeypatch.setenv("RAG_DISABLE_WHATSAPP_SEND", val)
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.status = 200
            result = _whatsapp_send_to_jid("x@x", "y")
            assert mock_open.called, (
                f"con {val!r} esperaba que el send hiciera POST real, "
                f"pero el guard cortó antes"
            )
            assert result is True


@pytest.mark.parametrize("env_name", [
    "RAG_DISABLE_WHATSAPP_SEND",
    "RAG_TESTING",
])
def test_each_guard_var_independently_blocks(monkeypatch, env_name):
    """Cada env var bloquea independientemente — no hay AND implícito."""
    monkeypatch.delenv("RAG_DISABLE_WHATSAPP_SEND", raising=False)
    monkeypatch.delenv("RAG_TESTING", raising=False)
    monkeypatch.setenv(env_name, "1")
    assert _whatsapp_send_to_jid("x@x", "y") is False


def test_conftest_fixture_active_by_default():
    """Verifica que el autouse fixture `_block_real_whatsapp_send` del
    conftest top-level está activo en este archivo (sin opt-out)."""
    # Sin monkeypatch.delenv, RAG_DISABLE_WHATSAPP_SEND debe estar set
    # porque la fixture autouse del conftest top-level la setea.
    assert os.environ.get("RAG_DISABLE_WHATSAPP_SEND") == "1"
