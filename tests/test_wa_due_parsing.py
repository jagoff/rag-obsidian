"""Tests para `try_parse_due_from_text` y el filter de due en endpoints."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_web_server = pytest.importorskip("web.server")
app = _web_server.app

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def isolated_db(tmp_path):
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


def test_due_parse_manana():
    """'te paso mañana el deck' → due_ts en el futuro."""
    from rag.integrations.whatsapp.tail import try_parse_due_from_text
    out = try_parse_due_from_text("te paso mañana el deck")
    assert out is not None
    # Debe ser parseable como ISO future.
    dt = datetime.fromisoformat(out.replace("Z", "+00:00"))
    assert dt > datetime.now(timezone.utc)


def test_due_parse_en_2hs():
    from rag.integrations.whatsapp.tail import try_parse_due_from_text
    out = try_parse_due_from_text("ya te aviso en 2 hs")
    assert out is not None


def test_due_parse_no_match_returns_none():
    from rag.integrations.whatsapp.tail import try_parse_due_from_text
    assert try_parse_due_from_text("te paso el deck") is None
    assert try_parse_due_from_text("") is None
    assert try_parse_due_from_text(None) is None


def test_endpoint_filter_overdue(client, isolated_db):
    """Promises con due_ts < now aparecen con overdue_only=true."""
    from rag import _ragvec_state_conn
    past = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(timespec="seconds")
    future = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat(timespec="seconds")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _ragvec_state_conn() as conn:
        conn.execute(
            "INSERT INTO rag_promises(ts, contact_jid, promise_text, direction, due_ts, due_confidence, status)"
            " VALUES (?, 'x@y', 'overdue one', 'outbound', ?, 0.7, 'pending')",
            (now, past),
        )
        conn.execute(
            "INSERT INTO rag_promises(ts, contact_jid, promise_text, direction, due_ts, due_confidence, status)"
            " VALUES (?, 'x@y', 'future one', 'outbound', ?, 0.7, 'pending')",
            (now, future),
        )
        conn.commit()

    r = client.get("/api/wa/promises?overdue_only=true")
    body = r.json()
    assert len(body["promises"]) == 1
    assert body["promises"][0]["promise_text"] == "overdue one"


def test_endpoint_filter_due_within_hours(client, isolated_db):
    """Promises con due_ts <= now+N aparecen con due_within_hours=N."""
    from rag import _ragvec_state_conn
    soon = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(timespec="seconds")
    far = (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat(timespec="seconds")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _ragvec_state_conn() as conn:
        conn.execute(
            "INSERT INTO rag_promises(ts, contact_jid, promise_text, direction, due_ts, due_confidence, status)"
            " VALUES (?, 'x@y', 'soon one', 'outbound', ?, 0.7, 'pending')",
            (now, soon),
        )
        conn.execute(
            "INSERT INTO rag_promises(ts, contact_jid, promise_text, direction, due_ts, due_confidence, status)"
            " VALUES (?, 'x@y', 'far one', 'outbound', ?, 0.7, 'pending')",
            (now, far),
        )
        conn.commit()

    r = client.get("/api/wa/promises?due_within_hours=6")
    body = r.json()
    assert len(body["promises"]) == 1
    assert body["promises"][0]["promise_text"] == "soon one"


def test_voice_list_endpoint_ok(client, monkeypatch):
    """`say -v ?` parseado correctamente."""
    import subprocess
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/say")

    class FakeProc:
        returncode = 0
        stdout = (
            "Mónica              es_MX    # Hola, soy Mónica.\n"
            "Paulina             es_MX    # Hola, soy Paulina.\n"
            "Diego               es_AR    # Hola, soy Diego.\n"
            "Daniel              en_GB    # Hello, I am Daniel.\n"
        )

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeProc())
    r = client.get("/api/wa/voice/list")
    body = r.json()
    assert body["ok"] is True
    names = [v["name"] for v in body["voices"]]
    assert "Mónica" in names
    assert "Diego" in names


def test_voice_list_say_missing(client, monkeypatch):
    """Sin `say` instalado → ok=False."""
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    r = client.get("/api/wa/voice/list")
    body = r.json()
    assert body["ok"] is False
    assert body["voices"] == []
