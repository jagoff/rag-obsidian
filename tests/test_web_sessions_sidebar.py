"""Tests de los endpoints que alimentan la sidebar de /chat (2026-04-24).

Dos endpoints nuevos:
  - GET /api/sessions           → lista de sesiones web (sidebar "Recientes").
  - GET /api/session/{sid}/turns → turns completos para rehidratar la vista
                                   cuando el user hace click en una sesión.

Regresiones que atrapan:
  - /api/sessions devuelve sólo sesiones iniciadas en el web chat
    (id prefijado `web:`), filtrando CLI (`rag ask`, `rag do`) para que
    la sidebar no muestre ruido.
  - Sesiones vacías (0 turns + sin first_q) no se listan.
  - El `title` se deriva del primer q no vacío, trimmed a 80 chars.
  - /api/session/{sid}/turns devuelve q/a/paths/ts por turn.
  - Sesión inexistente → 404 (no 200 con body vacío).

No testeamos:
  - Ordenamiento por updated_at (lo hace `list_sessions` y ya tiene tests
    propios en test_sessions.py).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import rag
import web.server as _server


_client = TestClient(_server.app)


# ── Audit 2026-04-26 BUG #1 telemetry — DB_PATH isolation ────────────────
# Previene pollution de la prod telemetry.db cuando el TestClient ejercita
# endpoints que disparan log_query_event/semantic_cache_store/etc.
# Snap+restore manual (NO monkeypatch.setattr) — el conftest autouse
# `_stabilize_rag_state` corre teardown ANTES de monkeypatch y emite
# warning falso si está set. Mismo patrón que tests/test_rag_log_sql_read.py.
@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path):
    import rag as _rag_isolate
    _snap = _rag_isolate.DB_PATH
    _rag_isolate.DB_PATH = tmp_path / "ragvec"
    try:
        yield
    finally:
        _rag_isolate.DB_PATH = _snap

@pytest.fixture
def sessions_tmp(tmp_path, monkeypatch):
    """Isolated SESSIONS_DIR so tests don't pollute each other or the
    user's real sessions dir."""
    sdir = tmp_path / "sessions"
    lastf = tmp_path / "last_session"
    monkeypatch.setattr(rag, "SESSIONS_DIR", sdir)
    monkeypatch.setattr(rag, "LAST_SESSION_FILE", lastf)
    return sdir


def _mk_web_session(sid: str, q: str, a: str = "respuesta") -> dict:
    """Build + save a session with one turn, forcing a specific id.

    We bypass `ensure_session` because that mints a fresh id for invalid
    inputs, and we want `web:foo` literal.
    """
    sess = rag.ensure_session(sid, mode="chat")
    rag.append_turn(sess, {"q": q, "a": a, "paths": ["note.md"]})
    rag.save_session(sess)
    return sess


def test_sessions_filters_to_web_prefix(sessions_tmp):
    """Solo sesiones `web:*` aparecen; CLI (tg:, ask:, …) quedan fuera."""
    _mk_web_session("web:abc123", "hola mundo")
    _mk_web_session("tg:999", "cli-only")  # NO debe aparecer
    rag.ensure_session("ask:xxx", mode="ask")  # mint fresh id → NO web

    resp = _client.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.json()
    sids = [s["id"] for s in data["sessions"]]
    assert "web:abc123" in sids
    assert "tg:999" not in sids
    assert all(s.startswith("web:") for s in sids)


def test_sessions_title_from_first_q(sessions_tmp):
    """El `title` es el primer q del session, trimmed a 80 chars."""
    q_long = "q" * 200
    _mk_web_session("web:long1", q_long)
    _mk_web_session("web:short1", "pregunta corta")

    resp = _client.get("/api/sessions")
    sessions = {s["id"]: s for s in resp.json()["sessions"]}
    assert sessions["web:short1"]["title"] == "pregunta corta"
    # 80-char cap.
    assert len(sessions["web:long1"]["title"]) == 80
    assert sessions["web:long1"]["title"] == "q" * 80


def test_sessions_skips_empty_sessions(sessions_tmp):
    """Sesiones sin turns + sin first_q no se listan (serían noise)."""
    # Session with no turns at all — shouldn't appear.
    empty = rag.ensure_session("web:empty1", mode="chat")
    rag.save_session(empty)
    # Session with one real turn — should appear.
    _mk_web_session("web:real1", "hola")

    resp = _client.get("/api/sessions")
    sids = [s["id"] for s in resp.json()["sessions"]]
    assert "web:real1" in sids
    assert "web:empty1" not in sids


def test_sessions_respects_limit(sessions_tmp):
    """El query param `limit` acota el tamaño del resultset."""
    for i in range(5):
        _mk_web_session(f"web:s{i}", f"pregunta {i}")

    resp = _client.get("/api/sessions?limit=2")
    assert resp.status_code == 200
    assert len(resp.json()["sessions"]) == 2


def test_sessions_shape(sessions_tmp):
    """El shape del response coincide con lo que consume app.js."""
    _mk_web_session("web:shape1", "qshape")

    resp = _client.get("/api/sessions")
    row = resp.json()["sessions"][0]
    # Todos los campos que el JS lee — si alguno se borra, la UI rompe.
    for k in ("id", "title", "turns", "updated_at", "created_at"):
        assert k in row, f"missing field: {k}"


def test_session_turns_returns_full_conversation(sessions_tmp):
    """GET /api/session/{sid}/turns devuelve q/a/paths por turn."""
    sess = rag.ensure_session("web:conv1", mode="chat")
    rag.append_turn(sess, {"q": "hola", "a": "buenas", "paths": ["a.md"]})
    rag.append_turn(sess, {"q": "y?", "a": "todo bien", "paths": ["b.md", "c.md"]})
    rag.save_session(sess)

    resp = _client.get("/api/session/web:conv1/turns")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "web:conv1"
    assert len(data["turns"]) == 2
    assert data["turns"][0]["q"] == "hola"
    assert data["turns"][0]["a"] == "buenas"
    assert data["turns"][0]["paths"] == ["a.md"]
    assert data["turns"][1]["paths"] == ["b.md", "c.md"]


def test_session_turns_missing_returns_404(sessions_tmp):
    """Session inexistente → 404 (no 200 con body vacío — eso confundiría
    al frontend que loopea sobre `turns`)."""
    resp = _client.get("/api/session/web:doesnotexist/turns")
    assert resp.status_code == 404
