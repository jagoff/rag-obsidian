"""Tests for POST /api/brief/feedback (2026-04-29).

Cuando el listener TS detecta una reaction 👍/👎/🔇 a un push del daemon
de briefs (parseado desde el footer `_brief:<vault_relpath>_`), postea
acá. El endpoint thinly forwards a `rag._record_brief_feedback`. Mismo
patrón / shape que el endpoint de anticipate feedback (mute > negative >
positive de precedencia, silent-fail si la DB falla).

Invariantes testeadas:
  1. ratings válidos (positive, negative, mute) → 200 + helper se llama.
  2. rating inválido ('maybe') → 422 (Pydantic validation).
  3. dedup_key requerido → 422 sin él.
  4. reason opcional, default "".
  5. Helper devuelve None → endpoint responde {ok: False, reason: ...} pero 200.
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

import rag as _rag  # noqa: E402


@pytest.fixture
def client():
    return TestClient(_web_server.app)


# ── Happy paths ─────────────────────────────────────────────────────────────

def test_positive_rating_calls_helper(client):
    """rating='positive' + dedup_key → _record_brief_feedback se llama
    con keyword args (dedup_key, rating, reason='', source='wa')."""
    with patch.object(_rag, "_record_brief_feedback", return_value=42) as mock_h:
        resp = client.post(
            "/api/brief/feedback",
            json={
                "dedup_key": "02-Areas/Briefs/2026-04-29-morning.md",
                "rating": "positive",
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["id"] == 42
    mock_h.assert_called_once()
    _, kwargs = mock_h.call_args
    assert kwargs["dedup_key"] == "02-Areas/Briefs/2026-04-29-morning.md"
    assert kwargs["rating"] == "positive"
    assert kwargs["reason"] == ""
    assert kwargs["source"] == "wa"


def test_negative_rating_calls_helper(client):
    """rating='negative' + reason texto → forward verbatim."""
    with patch.object(_rag, "_record_brief_feedback", return_value=11) as mock_h:
        resp = client.post(
            "/api/brief/feedback",
            json={
                "dedup_key": "02-Areas/Briefs/2026-04-29-evening.md",
                "rating": "negative",
                "reason": "hoy no agregó nada útil",
            },
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["ok"] is True
    _, kwargs = mock_h.call_args
    assert kwargs["dedup_key"] == "02-Areas/Briefs/2026-04-29-evening.md"
    assert kwargs["rating"] == "negative"
    assert kwargs["reason"] == "hoy no agregó nada útil"


def test_mute_rating_calls_helper(client):
    """rating='mute' → forward."""
    with patch.object(_rag, "_record_brief_feedback", return_value=7) as mock_h:
        resp = client.post(
            "/api/brief/feedback",
            json={
                "dedup_key": "02-Areas/Briefs/2026-04-week-17-digest.md",
                "rating": "mute",
            },
        )
    assert resp.status_code == 200, resp.text
    _, kwargs = mock_h.call_args
    assert kwargs["rating"] == "mute"


# ── Validación ──────────────────────────────────────────────────────────────

def test_invalid_rating_returns_422(client):
    """rating='maybe' → Pydantic validation rebota."""
    resp = client.post(
        "/api/brief/feedback",
        json={"dedup_key": "x", "rating": "maybe"},
    )
    # FastAPI/Pydantic devuelve 422 por defecto en validation errors.
    # Aceptamos 400 también para alinear con el endpoint de anticipate.
    assert resp.status_code in (400, 422)


def test_missing_dedup_key_returns_422(client):
    """Sin dedup_key → 422."""
    resp = client.post(
        "/api/brief/feedback",
        json={"rating": "positive"},
    )
    assert resp.status_code == 422


def test_missing_rating_returns_422(client):
    """Sin rating → 422."""
    resp = client.post(
        "/api/brief/feedback",
        json={"dedup_key": "x"},
    )
    assert resp.status_code == 422


def test_dedup_key_too_long_returns_422(client):
    """dedup_key con >400 chars → 422 (cap defensivo)."""
    resp = client.post(
        "/api/brief/feedback",
        json={"dedup_key": "x" * 401, "rating": "positive"},
    )
    assert resp.status_code == 422


# ── Silent-fail del helper ──────────────────────────────────────────────────

def test_helper_returns_none_endpoint_returns_ok_false(client):
    """Si _record_brief_feedback devuelve None (DB inaccesible o rating
    inválido cuando llega), endpoint responde 200 con {ok: False,
    reason: ...} — NUNCA 5xx, así el listener no rompe."""
    with patch.object(_rag, "_record_brief_feedback", return_value=None):
        resp = client.post(
            "/api/brief/feedback",
            json={"dedup_key": "x", "rating": "positive"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "reason" in body


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Aísla telemetry DB en tmp_path. `_ragvec_state_conn` crea las
    tablas on-demand al primer uso (vía `_ensure_telemetry_tables`).
    Mismo patrón que `tests/test_draft_decisions_endpoint.py::state_db`.
    """
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_rag, "DB_PATH", db_path)
    _rag.SqliteVecClient(path=str(db_path))
    with _rag._ragvec_state_conn() as _conn:
        pass
    return db_path


def test_persistence_end_to_end(state_db):
    """Smoke: la row se persiste a `rag_brief_feedback` y la podemos
    leer con un SELECT directo. NO mockeamos el helper — corremos el
    código real contra una telemetry.db temporal."""
    real_client = TestClient(_web_server.app)
    resp = real_client.post(
        "/api/brief/feedback",
        json={
            "dedup_key": "02-Areas/Briefs/2026-04-29-morning.md",
            "rating": "positive",
            "reason": "buen brief",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["id"] >= 1

    # Verificamos directo en la DB.
    import sqlite3
    telemetry_db = state_db / _rag._TELEMETRY_DB_FILENAME
    assert telemetry_db.exists(), f"no encontré telemetry.db en {state_db}"
    conn = sqlite3.connect(str(telemetry_db))
    rows = conn.execute(
        "SELECT dedup_key, rating, reason, source FROM rag_brief_feedback "
        "ORDER BY id DESC LIMIT 1"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "02-Areas/Briefs/2026-04-29-morning.md"
    assert rows[0][1] == "positive"
    assert rows[0][2] == "buen brief"
    assert rows[0][3] == "wa"
