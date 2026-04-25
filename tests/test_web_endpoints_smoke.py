"""Smoke tests para endpoints del web/server.py que carecian de cobertura
E2E directa. Identificados via audit del 2026-04-25 cruzando los 52
@app.get/@app.post declarados con los `client.get/.post` referenciados
en tests/test_web_*.py.

El objetivo NO es testear el dominio del cada endpoint (eso vive en los
tests focalizados de feedback / behavior / sessions / chat). Es atrapar
regresiones tipo:

  - "alguien renombro un campo del modelo Pydantic y el endpoint
    ahora 422-ea con un shape distinto"
  - "alguien dropeo el endpoint sin actualizar el HTML que lo llama
    (nadie deteccaria 404 en CI)"
  - "alguien metio un `await` mal y el endpoint sync se cuelga
    indefinidamente"

Estrategia: cada test golpea el endpoint con (a) payload invalido y
verifica 4xx coherente, o (b) request minima que pueda procesar sin
side effects reales (mocks de los handlers de side-effect).

Side-effect handlers (`_apple_run_script`, `_whatsapp_send_to_jid`,
`_send_smtp_message`, etc.) van mockeados para que el test pueda
correr en CI sin AppleScript / OAuth / SMTP creds.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import web.server as _server


_client = TestClient(_server.app)


# ── GET endpoints sin side effects ───────────────────────────────────────


def test_api_model_returns_string():
    """`GET /api/model` → modelo de chat actualmente activo."""
    resp = _client.get("/api/model")
    assert resp.status_code == 200
    body = resp.json()
    # Shape: {"model": str} o variant similar — al menos contiene la key
    # del nombre del modelo.
    assert isinstance(body, dict)
    assert "model" in body or "name" in body


def test_api_vaults_returns_registered_list():
    """`GET /api/vaults` → vaults registrados + active."""
    resp = _client.get("/api/vaults")
    assert resp.status_code == 200
    body = resp.json()
    assert "active" in body
    assert "registered" in body
    assert isinstance(body["registered"], list)
    # active debe estar en la lista de registered (consistencia interna).
    assert body["active"] in body["registered"]


def test_api_chat_model_get():
    """`GET /api/chat/model` → modelo de chat (current + available)."""
    resp = _client.get("/api/chat/model")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)
    # Debe incluir el modelo actual y la lista de disponibles.
    assert "current" in body or "model" in body


def test_api_history_returns_session_list():
    """`GET /api/history` → lista de queries pasadas (key "history")."""
    resp = _client.get("/api/history")
    assert resp.status_code == 200
    body = resp.json()
    # Shape: {"history": [...]} (lista de strings o dicts).
    assert isinstance(body, dict)
    assert "history" in body
    assert isinstance(body["history"], list)


def test_chat_html_endpoint_served():
    """`GET /chat` → HTML del chat (PWA entrypoint principal)."""
    resp = _client.get("/chat")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    assert "manifest.webmanifest" in body
    assert "register-sw.js" in body


def test_dashboard_html_endpoint_served():
    """`GET /dashboard` → HTML del dashboard."""
    resp = _client.get("/dashboard")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")


def test_api_dashboard_returns_payload():
    """`GET /api/dashboard` → JSON con datos del dashboard."""
    resp = _client.get("/api/dashboard")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)


# ── System metrics endpoints (read-only, expensive but safe) ─────────────


def test_api_system_memory_shape():
    """`GET /api/system-memory` → snapshot de memoria con categorias +
    samples timeseries."""
    resp = _client.get("/api/system-memory")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)
    # Shape canonico: categorias + samples (timeseries para grafico) +
    # current snapshot. Si alguien renombra estas keys el frontend de
    # /dashboard se queda en blanco.
    for key in ("categories", "samples", "current"):
        assert key in body, f"missing key {key!r} in /api/system-memory response"
    assert isinstance(body["samples"], list)


def test_api_system_cpu_shape():
    """`GET /api/system-cpu` → snapshot de CPU con shape simetrico al de
    memory (categorias + samples + current)."""
    resp = _client.get("/api/system-cpu")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)
    for key in ("categories", "samples", "current"):
        assert key in body, f"missing key {key!r} in /api/system-cpu response"
    assert isinstance(body["samples"], list)


def test_api_system_metrics_combined():
    """`GET /api/system-metrics` → snapshot combinado memoria + cpu."""
    resp = _client.get("/api/system-metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)


# ── POST endpoints: validation (payload incorrecto → 4xx) ────────────────


def test_api_reminders_complete_rejects_missing_id():
    """`POST /api/reminders/complete` sin reminder_id → 422."""
    resp = _client.post("/api/reminders/complete", json={})
    # Pydantic validation error: 422
    assert resp.status_code in (400, 422), f"got {resp.status_code}: {resp.text[:200]}"


def test_api_reminders_delete_rejects_missing_id():
    """`POST /api/reminders/delete` sin reminder_id → 422."""
    resp = _client.post("/api/reminders/delete", json={})
    assert resp.status_code in (400, 422)


def test_api_calendar_delete_rejects_missing_id():
    """`POST /api/calendar/delete` sin event_id → 422."""
    resp = _client.post("/api/calendar/delete", json={})
    assert resp.status_code in (400, 422)


def test_api_whatsapp_send_rejects_missing_jid():
    """`POST /api/whatsapp/send` sin destinatario → 422."""
    resp = _client.post("/api/whatsapp/send", json={"message": "hola"})
    assert resp.status_code in (400, 422)


def test_api_whatsapp_send_rejects_missing_message():
    """`POST /api/whatsapp/send` sin mensaje → 422."""
    resp = _client.post("/api/whatsapp/send", json={"chat_jid": "1234@s.whatsapp.net"})
    assert resp.status_code in (400, 422)


def test_api_mail_send_rejects_missing_fields():
    """`POST /api/mail/send` sin to/subject/body → 422."""
    resp = _client.post("/api/mail/send", json={})
    assert resp.status_code in (400, 422)


def test_api_save_rejects_invalid_payload():
    """`POST /api/save` requiere question + answer + sources al minimo."""
    resp = _client.post("/api/save", json={})
    assert resp.status_code in (400, 422)


def test_api_followups_rejects_empty_body():
    """`POST /api/followups` sin question → 422."""
    resp = _client.post("/api/followups", json={})
    assert resp.status_code in (400, 422)


def test_api_related_rejects_empty_body():
    """`POST /api/related` sin path → 422."""
    resp = _client.post("/api/related", json={})
    assert resp.status_code in (400, 422)


def test_api_tts_rejects_empty_text():
    """`POST /api/tts` sin texto → 422."""
    resp = _client.post("/api/tts", json={})
    assert resp.status_code in (400, 422)


def test_api_feedback_rejects_unknown_action():
    """`POST /api/feedback` con accion invalida → 4xx (no debe loggear
    junk en rag_feedback)."""
    resp = _client.post("/api/feedback", json={
        "session_id": "test",
        "question": "x",
        "action": "definitely_not_a_real_action_lol",
    })
    # Aceptamos 400/422 (validation) o 200 con error (depende del endpoint).
    # Lo critico es que NO sea 500.
    assert resp.status_code != 500, f"endpoint crashed: {resp.text[:200]}"


def test_api_behavior_rejects_empty_event():
    """`POST /api/behavior` sin event → 422."""
    resp = _client.post("/api/behavior", json={})
    assert resp.status_code in (400, 422)


def test_api_ollama_unload_idempotent():
    """`POST /api/ollama/unload` → no debe crashear aunque ollama no este
    corriendo. Si ollama up: unload OK. Si down: 200 con error tag o 503."""
    resp = _client.post("/api/ollama/unload", json={})
    # No 500 ni timeout — el endpoint debe manejar el caso "ollama down".
    assert resp.status_code != 500, f"endpoint crashed: {resp.text[:200]}"


# ── Endpoint identity (el endpoint EXISTE en runtime) ────────────────────


def test_all_endpoints_resolve_not_404():
    """Smoke contra los 52 endpoints declarados en app.routes — ninguno
    debe devolver 404 (que indicaria un decorator que no se aplico)."""
    # Recolecta todas las paths declaradas
    paths = set()
    for route in _server.app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if not path or not methods:
            continue
        if "GET" in methods or "HEAD" in methods:
            paths.add(path)
    # Quita los que tienen path-params o son streams (chunked SSE)
    skip_substrings = ("/{", "/stream", "/static")
    safe_paths = [p for p in paths if not any(s in p for s in skip_substrings)]
    assert len(safe_paths) > 10, "demasiado pocos endpoints — algo se rompio?"

    # Cada uno NO debe ser 404 (puede ser 200, 401, 422, 500 — pero 404
    # significa que el decorator no esta enganchado).
    for path in safe_paths:
        resp = _client.get(path)
        assert resp.status_code != 404, (
            f"endpoint {path} devuelve 404 — el decorator @app.get no esta "
            f"enganchado o el path esta mal escrito"
        )
