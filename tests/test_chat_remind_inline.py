"""Feature K — "recordame X" inline en chat web.

Tests:
  1. `parse_remind_intent("recordame llamar a Juan mañana 9am")` → dict válido.
  2. `parse_remind_intent("recordame algo")` → None (sin tiempo).
  3. `parse_remind_intent("una pregunta normal")` → None.
  4. POST `/api/chat` con `"recordame X mañana"` → SSE `created`
     con kind="reminder" + flag remind_inline=True.
  5. El handler llama a `_create_reminder` (mocked) — equivalente a
     "el reminder aparece en la lista del user" en un test offline.

Ver `rag.parse_remind_intent` y la zona Feature K en `web/server.py`.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import rag
from web import server as server_mod
from web.server import app

# `_parse_sse` y `_OllamaMock` viven en conftest.py — ver
# tests/conftest.py "Shared test helpers".
from tests.conftest import _parse_sse  # noqa: F401


# ── Telemetry isolation ────────────────────────────────────────────────────
# Mismo patrón que test_web_chat_tools — evita pollute de
# telemetry.db real.
@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path):
    snap = rag.DB_PATH
    rag.DB_PATH = tmp_path / "ragvec"
    try:
        yield
    finally:
        rag.DB_PATH = snap


@pytest.fixture(autouse=True)
def _reset_chat_buckets():
    """Drop rate-limit buckets entre tests para evitar 429s."""
    server_mod._CHAT_BUCKETS.clear()
    yield


# ══ 1. parse_remind_intent: caso happy ═════════════════════════════════════


def test_parse_remind_intent_happy_path():
    out = rag.parse_remind_intent("recordame llamar a Juan mañana 9am")
    assert out is not None
    assert out["title"] == "llamar a Juan"
    # due_iso debe ser ISO con offset AR — el día exacto depende de "now"
    # del runner pero el patrón es estable.
    assert "T09:00:00" in out["due_iso"]
    assert out["due_iso"].endswith("-03:00")
    assert out["original_text"] == "recordame llamar a Juan mañana 9am"


# ══ 2. parse_remind_intent: ambiguo (sin tiempo claro) → None ══════════════


def test_parse_remind_intent_ambiguous_no_time():
    """Sin marker temporal en el rest → None (caller hace fallback al LLM)."""
    out = rag.parse_remind_intent("recordame algo")
    assert out is None


def test_parse_remind_intent_ambiguous_just_trigger():
    """Sólo trigger sin nada más → None."""
    out = rag.parse_remind_intent("recordame")
    assert out is None


def test_parse_remind_intent_no_clear_time_marker():
    """Trigger + texto sin marker temporal explícito → None."""
    out = rag.parse_remind_intent("recordame llamar a Juan")
    assert out is None


# ══ 3. parse_remind_intent: pregunta normal → None ═════════════════════════


def test_parse_remind_intent_no_trigger():
    out = rag.parse_remind_intent("una pregunta normal")
    assert out is None


def test_parse_remind_intent_question_with_temporal_word():
    """No hay trigger aunque la pregunta mencione 'mañana' — None."""
    out = rag.parse_remind_intent("qué tengo mañana?")
    assert out is None


def test_parse_remind_intent_empty_input():
    assert rag.parse_remind_intent("") is None
    assert rag.parse_remind_intent(None) is None
    assert rag.parse_remind_intent("   ") is None


# ══ 4. POST /api/chat con "recordame X mañana" → SSE created ═══════════════


def test_chat_remind_inline_emits_created_sse(monkeypatch):
    """End-to-end: POST /api/chat con un texto que matchea el intent
    debe terminar con SSE `created` event sin tocar LLM ni retrieval."""
    # Mock `_create_reminder` para que devuelva success sin tocar
    # AppleScript real (el conftest autouse `_isolate_apple_integrations`
    # ya setea OBSIDIAN_RAG_NO_APPLE=1, pero monkeypatch garantiza
    # control total del retorno).
    create_mock = MagicMock(return_value=(True, "x-apple-reminderkit://ABC-123"))
    monkeypatch.setattr(rag, "_create_reminder", create_mock)
    # Disable rate limit + ollama probe — defensivo.
    monkeypatch.setattr(server_mod, "_ollama_alive", lambda timeout=2.0: True)

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "question": "recordame llamar a Juan mañana 9am",
            "vault_scope": None,
        },
    )
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)
    event_names = [e[0] for e in events]

    # Debe estar el `created` event con kind="reminder" + remind_inline=True.
    created_evts = [e for e in events if e[0] == "created"]
    assert len(created_evts) == 1, (
        f"esperado 1 `created` SSE event, got events={event_names}"
    )
    payload = created_evts[0][1]
    assert payload.get("kind") == "reminder"
    assert payload.get("created") is True
    assert payload.get("remind_inline") is True
    assert payload.get("reminder_id") == "x-apple-reminderkit://ABC-123"
    assert payload.get("fields", {}).get("title") == "llamar a Juan"
    assert "T09:00:00" in payload.get("fields", {}).get("due_iso", "")

    # Debe haberse llamado a _create_reminder con el title parseado.
    assert create_mock.call_count == 1
    args, kwargs = create_mock.call_args
    assert args[0] == "llamar a Juan"
    # due_dt debe ser un datetime con hora 9.
    assert isinstance(kwargs.get("due_dt"), datetime)
    assert kwargs["due_dt"].hour == 9

    # Tampoco debe haber tokens del LLM (skipped el flow normal).
    # Sólo los tokens canned del confirmation msg "✓ Reminder creado: …".
    token_text = "".join(
        e[1].get("delta", "") for e in events if e[0] == "token"
    )
    assert "Reminder creado" in token_text


# ══ 5. _create_reminder fue llamado correctamente (= reminder en lista) ═══


def test_chat_remind_inline_calls_create_reminder_with_correct_args(monkeypatch):
    """Verifica que el wire-up llama `_create_reminder` con el title
    parseado + due_dt correcto. Equivalente a "el reminder aparece en
    la lista del user" en un test offline (no podemos consultar
    Reminders.app real porque OBSIDIAN_RAG_NO_APPLE=1 + AppleScript
    bloqueado en el test env)."""
    captured: list[tuple] = []

    def fake_create(title, **kwargs):
        captured.append((title, kwargs))
        return True, f"id-{len(captured)}"

    monkeypatch.setattr(rag, "_create_reminder", fake_create)
    monkeypatch.setattr(server_mod, "_ollama_alive", lambda timeout=2.0: True)

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "question": "recordame comprar pan en 2 horas",
            "vault_scope": None,
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(captured) == 1, "expected exactly one _create_reminder call"
    title, kwargs = captured[0]
    assert title == "comprar pan"
    assert isinstance(kwargs.get("due_dt"), datetime)


def test_chat_remind_inline_does_not_fire_for_normal_question(monkeypatch):
    """Una pregunta NO-recordatorio NO debe disparar `_create_reminder`."""
    create_mock = MagicMock()
    monkeypatch.setattr(rag, "_create_reminder", create_mock)
    monkeypatch.setattr(server_mod, "_ollama_alive", lambda timeout=2.0: True)
    # El flow normal va a llamar muchísimas cosas (retrieve, etc.) que
    # no queremos ejercer acá. Stub `multi_retrieve` para que devuelva
    # algo barato.
    monkeypatch.setattr(server_mod, "multi_retrieve", lambda *a, **kw: {
        "docs": [], "metas": [], "scores": [], "confidence": 0.0,
        "search_query": "", "filters_applied": {}, "query_variants": [],
    })
    monkeypatch.setattr(server_mod, "_persist_conversation_turn", lambda *a, **kw: None)
    monkeypatch.setattr(server_mod, "save_session", lambda sess: None)
    monkeypatch.setattr(server_mod, "log_query_event", lambda ev: None)
    # Bypass del LLM call también — ollama.chat puede no estar.
    monkeypatch.setattr(server_mod, "_ollama_chat_probe", lambda timeout_s=6.0: True)

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "question": "qué tal el clima en Buenos Aires?",
            "vault_scope": None,
        },
    )
    # No exigimos status_code específico — el flow normal puede fallar
    # si no hay vault index. Lo que importa: NO se llamó _create_reminder.
    assert create_mock.call_count == 0
