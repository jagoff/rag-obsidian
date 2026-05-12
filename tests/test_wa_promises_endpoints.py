"""Tests para los endpoints HTTP del Promise Tracker (no del extractor LLM).

Cubre:
- `_detect_outbound_promise()` — hook que /api/wa/send dispara post-send.
- `GET /api/wa/promises` — list filtrado por jid + status.
- `POST /api/wa/promises/<id>/resolve` — marca done.
- `POST /api/wa/promises/<id>/cancel` — marca cancelled.

Isolate DB_PATH per test usando tmp dir + monkey-patch a `rag.DB_PATH`
para no contaminar la telemetry.db real.
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


@pytest.fixture()
def isolated_db(tmp_path):
    """Snap+restore DB_PATH para que el test write a tmp_path."""
    import rag as _rag
    orig = _rag.DB_PATH
    try:
        _rag.DB_PATH = Path(tmp_path)
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rag_promises ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " ts TEXT NOT NULL,"
                " contact_jid TEXT NOT NULL,"
                " contact_name TEXT,"
                " promise_text TEXT NOT NULL,"
                " direction TEXT NOT NULL,"
                " due_ts TEXT,"
                " due_confidence REAL,"
                " source_msg_id TEXT,"
                " source_chat_jid TEXT,"
                " status TEXT NOT NULL DEFAULT 'pending',"
                " reminder_sent_ts TEXT,"
                " closed_ts TEXT,"
                " closed_reason TEXT,"
                " extra_json TEXT)"
            )
            conn.commit()
        yield tmp_path
    finally:
        _rag.DB_PATH = orig


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _insert_promise(jid="x@y", text="te paso el deck mañana",
                     direction="outbound", status="pending"):
    """Helper: inserta una row directa (sin pasar por send)."""
    from datetime import datetime, timezone
    from rag import _ragvec_state_conn
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _ragvec_state_conn() as conn:
        cur = conn.execute(
            "INSERT INTO rag_promises("
            " ts, contact_jid, contact_name, promise_text, direction,"
            " due_ts, due_confidence, status"
            ") VALUES (?, ?, NULL, ?, ?, NULL, 0.3, ?)",
            (now, jid, text, direction, status),
        )
        conn.commit()
        return int(cur.lastrowid)


def test_detect_outbound_promise_with_hint(isolated_db):
    """Texto que matchea `_has_promise_hint` → row insertada."""
    rowid = _web_server._detect_outbound_promise("5491155556666@s.whatsapp.net",
                                                  "Te paso el deck mañana")
    assert rowid is not None
    assert rowid > 0


def test_detect_outbound_promise_without_hint(isolated_db):
    """Texto sin hint → None, no row."""
    rowid = _web_server._detect_outbound_promise("x@y", "hola qué tal")
    assert rowid is None


def test_detect_outbound_promise_empty_text(isolated_db):
    """Texto vacío → None."""
    assert _web_server._detect_outbound_promise("x@y", "") is None
    assert _web_server._detect_outbound_promise("x@y", "   ") is None


def test_list_filters_by_jid(client, isolated_db):
    """GET /api/wa/promises?jid=X solo devuelve rows con contact_jid=X."""
    _insert_promise(jid="a@y", text="te paso el deck mañana")
    _insert_promise(jid="b@y", text="ya te aviso")
    r = client.get("/api/wa/promises?jid=a@y")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert len(body["promises"]) == 1
    assert body["promises"][0]["contact_jid"] == "a@y"


def test_list_filters_by_status(client, isolated_db):
    """status='done' filtra solo cerradas."""
    _insert_promise(jid="x@y", status="pending")
    _insert_promise(jid="x@y", status="done")
    r = client.get("/api/wa/promises?jid=x@y&status=done")
    body = r.json()
    assert all(p["status"] == "done" for p in body["promises"])
    assert len(body["promises"]) == 1


def test_list_status_all(client, isolated_db):
    """status=all devuelve todo sin filtro de status."""
    _insert_promise(jid="x@y", status="pending")
    _insert_promise(jid="x@y", status="done")
    _insert_promise(jid="x@y", status="cancelled")
    r = client.get("/api/wa/promises?jid=x@y&status=all")
    assert len(r.json()["promises"]) == 3


def test_resolve_marks_done(client, isolated_db):
    """POST /resolve → status=done + closed_ts populated."""
    pid = _insert_promise()
    r = client.post(f"/api/wa/promises/{pid}/resolve")
    assert r.status_code == 200
    body = r.json()
    assert body["promise"]["status"] == "done"
    assert body["promise"]["closed_reason"] == "manual_resolve"
    assert body["promise"]["closed_ts"] is not None


def test_cancel_marks_cancelled(client, isolated_db):
    """POST /cancel → status=cancelled."""
    pid = _insert_promise()
    r = client.post(f"/api/wa/promises/{pid}/cancel")
    assert r.status_code == 200
    body = r.json()
    assert body["promise"]["status"] == "cancelled"
    assert body["promise"]["closed_reason"] == "manual_cancel"


def test_resolve_404_for_unknown_id(client, isolated_db):
    """POST /resolve con id inexistente → 404."""
    r = client.post("/api/wa/promises/99999/resolve")
    assert r.status_code == 404
