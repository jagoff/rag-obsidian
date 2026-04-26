"""Tests para /api/whatsapp/context (audit 2026-04-25 R2-Tests #1).

Endpoint nuevo del sprint 2026-04-25 sin cobertura previa. El frontend
(WhatsApp proposal cards) depende de este endpoint para mostrar contexto
de la conversación reciente al lado del proposal.

El endpoint vive en `web/server.py:3135-3157` y delega en
`rag.integrations.whatsapp._fetch_whatsapp_recent_with_jid`. La import
es lazy (dentro del try), así que monkeypatchear el atributo del módulo
funciona — el código resuelve el símbolo en cada call.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import web.server as _server


_client = TestClient(_server.app)


def test_context_returns_messages_for_valid_jid(monkeypatch):
    """Con un jid válido el endpoint devuelve el payload del bridge tal
    cual (passthrough). Mockeamos `_fetch_whatsapp_recent_with_jid`
    para no depender del bridge SQLite real."""
    fake_payload = {
        "jid": "+5491112345678@s.whatsapp.net",
        "messages_count": 2,
        "last_contact_at": "2026-04-25T10:01:00-03:00",
        "messages": [
            {
                "id": "abc1",
                "ts": "2026-04-25T10:00:00-03:00",
                "who": "me",
                "text": "hola",
                "is_from_me": True,
            },
            {
                "id": "abc2",
                "ts": "2026-04-25T10:01:00-03:00",
                "who": "Juan",
                "text": "hola, todo bien?",
                "is_from_me": False,
            },
        ],
        "contact": "Juan",
    }
    monkeypatch.setattr(
        "rag.integrations.whatsapp._fetch_whatsapp_recent_with_jid",
        lambda jid, limit: fake_payload,
    )
    resp = _client.get(
        "/api/whatsapp/context"
        "?jid=%2B5491112345678%40s.whatsapp.net&limit=5"
    )
    assert resp.status_code == 200
    assert resp.json() == fake_payload


def test_context_rejects_jid_without_at_sign(monkeypatch):
    """Un jid sin `@` es inválido (jid completo de WhatsApp siempre
    incluye `@s.whatsapp.net` o `@g.us`). Endpoint debe rebotar 400
    con detail "jid inválido" antes de tocar el bridge."""
    called = []
    monkeypatch.setattr(
        "rag.integrations.whatsapp._fetch_whatsapp_recent_with_jid",
        lambda jid, limit: called.append((jid, limit)) or {},
    )
    resp = _client.get("/api/whatsapp/context?jid=5491112345678&limit=5")
    assert resp.status_code == 400
    assert resp.json()["detail"] == "jid inválido"
    # El bridge NO debe ser llamado si la validación falla temprano.
    assert called == []


def test_context_rejects_empty_jid(monkeypatch):
    """jid="" (vacío) → 400. La validación es `not j or "@" not in j`,
    así que el string vacío entra por la primera condición."""
    called = []
    monkeypatch.setattr(
        "rag.integrations.whatsapp._fetch_whatsapp_recent_with_jid",
        lambda jid, limit: called.append((jid, limit)) or {},
    )
    resp = _client.get("/api/whatsapp/context?jid=&limit=5")
    assert resp.status_code == 400
    assert resp.json()["detail"] == "jid inválido"
    assert called == []


def test_context_returns_500_when_bridge_unreachable(monkeypatch):
    """Si el bridge tira excepción (DB locked, archivo borrado, sqlite
    crashed, etc.), el endpoint debe devolver 500 con un detail legible
    que mencione "error leyendo bridge" — el frontend puede mostrar un
    mensaje de "no pude cargar contexto" en vez de quedar en blanco."""
    def _boom(jid, limit):
        raise RuntimeError("sqlite database is locked")

    monkeypatch.setattr(
        "rag.integrations.whatsapp._fetch_whatsapp_recent_with_jid",
        _boom,
    )
    resp = _client.get(
        "/api/whatsapp/context"
        "?jid=%2B5491112345678%40s.whatsapp.net&limit=5"
    )
    assert resp.status_code == 500
    assert "error leyendo bridge" in resp.json()["detail"]


def test_context_respects_limit_parameter(monkeypatch):
    """El query param `limit` debe ser propagado al call interno tal
    cual (después de `int()`). Mockeamos y verificamos que el bridge
    fue invocado con `limit=3` cuando el cliente pasa `limit=3`."""
    captured = {}

    def _capture(jid, limit):
        captured["jid"] = jid
        captured["limit"] = limit
        return {"jid": jid, "messages_count": 0, "messages": []}

    monkeypatch.setattr(
        "rag.integrations.whatsapp._fetch_whatsapp_recent_with_jid",
        _capture,
    )
    resp = _client.get(
        "/api/whatsapp/context"
        "?jid=%2B5491112345678%40s.whatsapp.net&limit=3"
    )
    assert resp.status_code == 200
    assert captured["limit"] == 3
    assert isinstance(captured["limit"], int)
    assert captured["jid"] == "+5491112345678@s.whatsapp.net"
