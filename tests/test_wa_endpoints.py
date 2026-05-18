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
    assert body["last_seen_ts"] == "2026-05-11 12:00:00-03:00"

    # Confirmar que efectivamente quedó persistido.
    from rag.integrations.whatsapp import _db_local
    assert _db_local.get_last_seen(jid) == "2026-05-11 12:00:00-03:00"


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


def _seed_bridge_db(path: Path):
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE chats (jid TEXT PRIMARY KEY, name TEXT)")
    con.execute(
        "CREATE TABLE messages ("
        "id TEXT PRIMARY KEY, chat_jid TEXT, sender TEXT, content TEXT, "
        "timestamp TEXT, is_from_me INTEGER, media_type TEXT, filename TEXT, "
        "quoted_message_id TEXT, quoted_text TEXT)"
    )
    return con


def test_chats_pagination_accepts_iso_before_ts(tmp_path, monkeypatch):
    """`before_ts` viene del frontend en ISO; el helper debe compararlo
    contra timestamps bridge sin caer en HAVING inválido ni orden lex roto.
    """
    import rag as _rag

    bridge = tmp_path / "bridge" / "messages.db"
    con = _seed_bridge_db(bridge)
    con.executemany(
        "INSERT INTO chats (jid, name) VALUES (?, ?)",
        [
            ("5491111111111@s.whatsapp.net", "Maria"),
            ("5491222222222@s.whatsapp.net", "Grecia"),
        ],
    )
    con.executemany(
        "INSERT INTO messages "
        "(id, chat_jid, sender, content, timestamp, is_from_me, media_type, filename, "
        "quoted_message_id, quoted_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("a1", "5491111111111@s.whatsapp.net", "5491111111111@s.whatsapp.net",
             "más nuevo", "2026-05-11 12:00:00-03:00", 0, "", "", "", ""),
            ("b1", "5491222222222@s.whatsapp.net", "5491222222222@s.whatsapp.net",
             "más viejo", "2026-05-11 11:00:00-03:00", 0, "", "", "", ""),
        ],
    )
    con.commit()
    con.close()
    monkeypatch.setattr(_rag, "WHATSAPP_DB_PATH", bridge)

    from rag.integrations.whatsapp import fetch as _wa_fetch

    page1 = _wa_fetch.list_chats_for_ui(limit=1)
    assert [c["label"] for c in page1] == ["Maria"]
    page2 = _wa_fetch.list_chats_for_ui(limit=1, before_ts=page1[0]["last_ts"])
    assert [c["label"] for c in page2] == ["Grecia"]


def test_chats_includes_ragnet_bot_chat_for_wzp_ui(tmp_path, monkeypatch):
    """La UI `/wzp` debe mostrar el grupo Ra/RagNet aunque las ingestas
    lo filtren para evitar loops de contexto.
    """
    import rag as _rag

    bridge = tmp_path / "bridge" / "messages.db"
    jid = "120363426178035051@g.us"
    con = _seed_bridge_db(bridge)
    con.execute("INSERT INTO chats (jid, name) VALUES (?, ?)", (jid, "Ra"))
    con.execute(
        "INSERT INTO messages "
        "(id, chat_jid, sender, content, timestamp, is_from_me, media_type, filename, "
        "quoted_message_id, quoted_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "ra1", jid, "5491111111111@s.whatsapp.net", "hola desde Ra",
            "2026-05-18 12:00:00-03:00", 0, "", "", "", "",
        ),
    )
    con.commit()
    con.close()
    monkeypatch.setattr(_rag, "WHATSAPP_DB_PATH", bridge)
    monkeypatch.setattr(_rag, "WHATSAPP_BOT_JID", jid)

    from rag.integrations.whatsapp import fetch as _wa_fetch

    chats = _wa_fetch.list_chats_for_ui(limit=10, q="Ra")
    assert [c["jid"] for c in chats] == [jid]
    assert chats[0]["label"] == "Ra"


def test_thread_includes_ragnet_bot_chat_messages_for_wzp_ui(tmp_path, monkeypatch):
    """Abrir el canal Ra en `/wzp` debe devolver su historial real."""
    import rag as _rag

    bridge = tmp_path / "bridge" / "messages.db"
    jid = "120363426178035051@g.us"
    con = _seed_bridge_db(bridge)
    con.execute("INSERT INTO chats (jid, name) VALUES (?, ?)", (jid, "Ra"))
    con.execute(
        "INSERT INTO messages "
        "(id, chat_jid, sender, content, timestamp, is_from_me, media_type, filename, "
        "quoted_message_id, quoted_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "ra1", jid, "5491111111111@s.whatsapp.net", "draft listo",
            "2026-05-18 12:00:00-03:00", 1, "", "", "", "",
        ),
    )
    con.commit()
    con.close()
    monkeypatch.setattr(_rag, "WHATSAPP_DB_PATH", bridge)
    monkeypatch.setattr(_rag, "WHATSAPP_BOT_JID", jid)

    from rag.integrations.whatsapp import fetch as _wa_fetch

    thread = _wa_fetch.fetch_thread_for_ui(jid, limit=10)
    assert thread["label"] == "Ra"
    assert [m["content"] for m in thread["messages"]] == ["draft listo"]


def test_mark_read_same_day_unread_comparison_uses_bridge_format(tmp_path, monkeypatch):
    """Unread usa comparación string contra bridge timestamps. Guardar `T`
    hacía que mensajes nuevos del mismo día no contaran como unread.
    """
    import rag as _rag

    bridge = tmp_path / "bridge" / "messages.db"
    jid = "5491111111111@s.whatsapp.net"
    con = _seed_bridge_db(bridge)
    con.execute("INSERT INTO chats (jid, name) VALUES (?, ?)", (jid, "Maria"))
    con.executemany(
        "INSERT INTO messages "
        "(id, chat_jid, sender, content, timestamp, is_from_me, media_type, filename, "
        "quoted_message_id, quoted_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("m1", jid, jid, "leído", "2026-05-11 12:00:00-03:00", 0, "", "", "", ""),
            ("m2", jid, jid, "nuevo", "2026-05-11 12:30:00-03:00", 0, "", "", "", ""),
        ],
    )
    con.commit()
    con.close()
    monkeypatch.setattr(_rag, "WHATSAPP_DB_PATH", bridge)

    from rag.integrations.whatsapp import fetch as _wa_fetch

    _wa_fetch.mark_read_for_ui(jid, "2026-05-11T12:00:00-03:00")
    chats = _wa_fetch.list_chats_for_ui(limit=10)
    assert chats[0]["unread_count"] == 1


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
def test_revoke_requires_admin_token(client, monkeypatch):
    monkeypatch.delenv("OBSIDIAN_RAG_TEST_ADMIN_BYPASS", raising=False)
    resp = client.post(
        "/api/wa/revoke",
        json={"jid": "5491100000000@s.whatsapp.net", "message_id": "ABC"},
    )
    assert resp.status_code == 401


def test_search_backfill_requires_admin_token(client, monkeypatch):
    monkeypatch.delenv("OBSIDIAN_RAG_TEST_ADMIN_BYPASS", raising=False)
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
