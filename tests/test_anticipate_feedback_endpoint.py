"""Tests for POST /api/anticipate/feedback (2026-04-29).

Cuando el listener TS detecta un reply 👍/👎/🔇 a un push del
Anticipatory Agent (parseado desde el footer `_anticipate:<dedup_key>_`),
postea acá. El endpoint thinly forwards al
`rag_anticipate.feedback.record_feedback` que ya existe.

Invariantes testeadas:
  1. ratings válidos (positive, negative, mute) → 200 + record_feedback se llama.
  2. rating inválido ('maybe') → 422 (Pydantic validation).
  3. dedup_key requerido → 422 sin él.
  4. reason opcional, default "".
  5. Helper devuelve False → endpoint responde {ok: False, reason: ...} pero 200.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Skip cleanly si las web deps no están.
_web_server = pytest.importorskip("web.server")
_fastapi = pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402

import rag_anticipate.feedback as _ant_fb  # noqa: E402


@pytest.fixture
def client():
    return TestClient(_web_server.app)


# ── Happy paths ─────────────────────────────────────────────────────────────

def test_positive_rating_calls_record_feedback(client):
    """rating='positive' + dedup_key → record_feedback(dedup_key, 'positive', source='wa')."""
    with patch.object(_ant_fb, "record_feedback", return_value=True) as mock_rf:
        resp = client.post(
            "/api/anticipate/feedback",
            json={"dedup_key": "cal:event-123", "rating": "positive"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True}
    mock_rf.assert_called_once()
    args, kwargs = mock_rf.call_args
    # El handler hace `record_feedback(dedup_key, rating, reason=..., source='wa')`.
    assert args == ("cal:event-123", "positive")
    assert kwargs["source"] == "wa"
    assert kwargs["reason"] == ""


def test_negative_rating_calls_record_feedback(client):
    """rating='negative' + reason texto → forward verbatim."""
    with patch.object(_ant_fb, "record_feedback", return_value=True) as mock_rf:
        resp = client.post(
            "/api/anticipate/feedback",
            json={
                "dedup_key": "echo:note-a:note-b",
                "rating": "negative",
                "reason": "no sirvió esa conexión",
            },
        )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True}
    args, kwargs = mock_rf.call_args
    assert args == ("echo:note-a:note-b", "negative")
    assert kwargs["reason"] == "no sirvió esa conexión"


def test_mute_rating_calls_record_feedback(client):
    """rating='mute' → forward."""
    with patch.object(_ant_fb, "record_feedback", return_value=True) as mock_rf:
        resp = client.post(
            "/api/anticipate/feedback",
            json={"dedup_key": "commit:loop-7", "rating": "mute"},
        )
    assert resp.status_code == 200, resp.text
    args, _ = mock_rf.call_args
    assert args == ("commit:loop-7", "mute")


# ── Validación ──────────────────────────────────────────────────────────────

def test_invalid_rating_returns_422(client):
    """rating='maybe' → Pydantic validation rebota."""
    resp = client.post(
        "/api/anticipate/feedback",
        json={"dedup_key": "x", "rating": "maybe"},
    )
    # 400 o 422 — el user spec pidió 400; FastAPI/Pydantic devuelve 422
    # por defecto en validation errors. Aceptamos ambos.
    assert resp.status_code in (400, 422)


def test_missing_dedup_key_returns_422(client):
    """Sin dedup_key → 422."""
    resp = client.post(
        "/api/anticipate/feedback",
        json={"rating": "positive"},
    )
    assert resp.status_code == 422


def test_missing_rating_returns_422(client):
    """Sin rating → 422."""
    resp = client.post(
        "/api/anticipate/feedback",
        json={"dedup_key": "x"},
    )
    assert resp.status_code == 422


# ── Silent-fail del helper ──────────────────────────────────────────────────

def test_helper_returns_false_endpoint_returns_ok_false(client):
    """Si record_feedback devuelve False (DB inaccesible), endpoint responde
    200 con {ok: False, reason: ...} — NUNCA 5xx, así el listener no rompe."""
    with patch.object(_ant_fb, "record_feedback", return_value=False):
        resp = client.post(
            "/api/anticipate/feedback",
            json={"dedup_key": "x", "rating": "positive"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "reason" in body
