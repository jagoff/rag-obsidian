"""Smoke tests para los endpoints `/api/wa/*` del WhatsApp Web propio.

Cubre el surface read + mark_read + SSE smoke + validación de auth en
endpoints destructivos. Los tests corren sin bridge `messages.db`
disponible (entorno CI / dev sin whatsmeow corriendo): cada función
`*_for_ui` está diseñada para devolver `{chats: [], ...}` o
`{messages: [], ...}` cuando la bridge DB no existe, así que el surface
público no rompe.

Por qué importa: estos endpoints alimentan la UI nueva de `/wa` y un
500 silencioso en cualquiera rompe el shell completo (chat list o
thread). Snapshot mínimo + auth gates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_web_server = pytest.importorskip("web.server")
app = _web_server.app

from fastapi.testclient import TestClient  # noqa: E402


# ── DB isolation ──────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path, monkeypatch):
    """Aisla `DB_PATH` per-test + apunta `WHATSAPP_DB_PATH` a path inexistente.

    Cuando el bridge DB falta, `list_chats_for_ui` y `fetch_thread_for_ui`
    devuelven shapes vacías sin tirar excepción. Eso es exactamente lo
    que queremos validar acá.

    También resetea el flag `_MIGRATION_DONE` del módulo `_db_local` para
    que cada test ejerza la migración limpia sobre su propio
    `telemetry.db`.
    """
    import rag as _rag
    from rag.integrations.whatsapp import _db_local

    snap_db = _rag.DB_PATH
    snap_wa = _rag.WHATSAPP_DB_PATH
    _rag.DB_PATH = tmp_path / "ragvec"
    _rag.DB_PATH.mkdir(parents=True, exist_ok=True)
    _rag.WHATSAPP_DB_PATH = tmp_path / "bridge_does_not_exist" / "messages.db"
    _db_local._MIGRATION_DONE = False
    try:
        yield
    finally:
        _rag.DB_PATH = snap_db
        _rag.WHATSAPP_DB_PATH = snap_wa
        _db_local._MIGRATION_DONE = False


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def valid_token():
    return _web_server._ADMIN_TOKEN


# ── /api/wa/chats ─────────────────────────────────────────────────────────────
def test_chats_returns_shape_when_bridge_db_missing(client):
    """Sin bridge DB, debe devolver lista vacía + next_before_ts=None."""
    resp = client.get("/api/wa/chats?limit=10")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, dict)
    assert "chats" in body and isinstance(body["chats"], list)
    assert body["chats"] == []
    assert body.get("next_before_ts") is None


def test_chats_passes_query_params(client):
    """Los params se aceptan sin 422 (FastAPI validation)."""
    resp = client.get("/api/wa/chats?limit=5&q=fer&before_ts=2026-01-01T00:00:00-03:00")
    assert resp.status_code == 200, resp.text


def test_chats_caps_limit_safely(client):
    """Limit fuera de rango no debería tirar 500 — el helper internal
    clampea (`min(limit, 200)`) — devolver 200 vacío es OK acá."""
    resp = client.get("/api/wa/chats?limit=99999")
    assert resp.status_code == 200, resp.text


# ── /api/wa/thread/{jid} ──────────────────────────────────────────────────────
def test_thread_returns_shape_when_bridge_db_missing(client):
    """Sin bridge DB, devuelve dict con messages=[] sin tirar 500."""
    resp = client.get("/api/wa/thread/5491100000000@s.whatsapp.net?limit=10")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("messages") == []
    assert body.get("next_before_ts") is None
    assert body.get("jid") == "5491100000000@s.whatsapp.net"


def test_thread_bad_jid_returns_empty(client):
    """JID sin `@` → empty (no 500)."""
    resp = client.get("/api/wa/thread/no-at-sign?limit=5")
    assert resp.status_code == 200, resp.text
    assert resp.json().get("messages") == []


# ── /api/wa/mark_read ─────────────────────────────────────────────────────────
def test_mark_read_persists(client):
    """Mark read escribe en `rag_wa_read_state` y devuelve el ts efectivo."""
    jid = "5491100000000@s.whatsapp.net"
    resp = client.post(
        "/api/wa/mark_read",
        json={"jid": jid, "last_seen_ts": "2026-05-11T12:00:00-03:00"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["jid"] == jid
    assert body["last_seen_ts"] == "2026-05-11T12:00:00-03:00"

    # Confirmar que efectivamente quedó persistido.
    from rag.integrations.whatsapp import _db_local
    assert _db_local.get_last_seen(jid) == "2026-05-11T12:00:00-03:00"


def test_mark_read_uses_now_if_no_ts(client):
    """Sin `last_seen_ts`, el endpoint debe generar uno (timestamp actual)."""
    resp = client.post(
        "/api/wa/mark_read",
        json={"jid": "5491100000000@s.whatsapp.net"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["last_seen_ts"]  # non-empty


def test_mark_read_rejects_bad_jid(client):
    resp = client.post("/api/wa/mark_read", json={"jid": "bad"})
    assert resp.status_code == 400


# ── /api/wa/search ────────────────────────────────────────────────────────────
def test_search_returns_empty_without_mirror(client):
    """Sin filas en el mirror, search devuelve hits=[] sin tirar 500."""
    resp = client.get("/api/wa/search?q=cualquiercosa&limit=5")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "hits" in body and isinstance(body["hits"], list)
    assert body["hits"] == []


def test_search_requires_q(client):
    """Sin `q`, debe devolver 422 (FastAPI Query(...))."""
    resp = client.get("/api/wa/search")
    assert resp.status_code == 422


# ── Auth-gated endpoints (destructive) ────────────────────────────────────────
def test_revoke_requires_admin_token(client):
    resp = client.post(
        "/api/wa/revoke",
        json={"jid": "5491100000000@s.whatsapp.net", "message_id": "ABC"},
    )
    assert resp.status_code == 401


def test_search_backfill_requires_admin_token(client):
    resp = client.post("/api/wa/search/backfill")
    assert resp.status_code == 401


def test_revoke_with_admin_token_not_401(client, valid_token):
    """Con token correcto la auth no bloquea (sí puede fallar 500 por
    falta de bridge, lo importante es que pase el gate)."""
    resp = client.post(
        "/api/wa/revoke",
        json={"jid": "5491100000000@s.whatsapp.net", "message_id": "ABC"},
        headers={"Authorization": f"Bearer {valid_token}"},
    )
    assert resp.status_code != 401


# ── /wa shell HTML ────────────────────────────────────────────────────────────
def test_wa_shell_html_served(client):
    """`GET /wa` debe servir el HTML del shell."""
    resp = client.get("/wa")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    body = resp.text.lower()
    # Sanity check del shell — debe tener el title brand `rag·wa`.
    assert "wa-app" in body or "rag·wa" in body or "wa-sidebar" in body


# ── PWA manifest tiene el shortcut /wa ────────────────────────────────────────
def test_manifest_has_wa_shortcut():
    """Fase 11 — el manifest debe declarar `/wa` como PWA shortcut."""
    import json
    mpath = ROOT / "web" / "static" / "manifest.webmanifest"
    data = json.loads(mpath.read_text(encoding="utf-8"))
    shortcuts = data.get("shortcuts") or []
    urls = {s.get("url") for s in shortcuts}
    assert "/wa" in urls, f"shortcut /wa falta — shortcuts: {urls}"


# ── SSE smoke ─────────────────────────────────────────────────────────────────
def test_wa_stream_route_registered():
    """`/api/wa/stream` está registrada con method GET — smoke route-level.

    NO ejercemos el streaming end-to-end porque arranca un asyncio loop
    de fondo (`_tail_loop` en `tail.py`) que sobrevive al request en
    TestClient y hace hang el resto de la suite. El test real del flow
    completo vive en `scripts/test_wa_sse.sh` (manual / smoke local).
    """
    pytest.importorskip("rag.integrations.whatsapp.tail")
    routes = [r for r in app.routes if getattr(r, "path", "") == "/api/wa/stream"]
    assert routes, "/api/wa/stream no registrada en app.routes"
    methods = routes[0].methods or set()
    assert "GET" in methods
