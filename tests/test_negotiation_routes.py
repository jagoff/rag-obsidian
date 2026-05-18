from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag
from web.negotiation_routes import register_negotiation_routes


@pytest.fixture()
def state_db(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    rag._TELEMETRY_DDL_ENSURED_PATHS.clear()
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)
    return db_path


@pytest.fixture()
def client(state_db):
    app = FastAPI()
    register_negotiation_routes(app, require_admin_token=lambda: None)
    return TestClient(app, raise_server_exceptions=False)


def _create_negotiation(
    client: TestClient,
    *,
    target_jid: str = "jid@s.whatsapp.net",
) -> int:
    resp = client.post(
        "/api/negotiations",
        json={
            "user_intent": "coordinar cafe",
            "target_jid": target_jid,
            "target_name": "Test Contact",
            "perimeter": {"topic": "scheduling"},
            "confidence_threshold": 0.9,
            "max_messages": 4,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ok"] is True
    return int(data["id"])


def test_create_and_list_negotiations(client):
    neg_id = _create_negotiation(client)

    resp = client.get("/api/negotiations", params={"limit": 50})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert [item["id"] for item in data["items"]] == [neg_id]
    assert data["items"][0]["status"] == "draft"
    assert data["items"][0]["perimeter"] == {"topic": "scheduling"}


def test_list_negotiations_filters_by_status(client):
    draft_id = _create_negotiation(client, target_jid="draft@s.whatsapp.net")
    launched_id = _create_negotiation(client, target_jid="launched@s.whatsapp.net")
    resp = client.post(
        f"/api/negotiations/{launched_id}/transition",
        json={"transition": "launch"},
    )
    assert resp.status_code == 200

    resp = client.get("/api/negotiations", params={"status": "draft", "limit": 10})
    assert resp.status_code == 200
    data = resp.json()
    assert [item["id"] for item in data["items"]] == [draft_id]


def test_legal_transition_updates_status(client):
    neg_id = _create_negotiation(client)

    resp = client.post(
        f"/api/negotiations/{neg_id}/transition",
        json={"transition": "launch"},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "id": neg_id,
        "from_status": "draft",
        "to_status": "launched",
    }

    from rag_negotiations.crud import get_negotiation

    row = get_negotiation(neg_id)
    assert row is not None
    assert row["status"] == "launched"


def test_illegal_transition_returns_400_and_leaves_status(client):
    neg_id = _create_negotiation(client)

    resp = client.post(
        f"/api/negotiations/{neg_id}/transition",
        json={"transition": "first_msg_ack"},
    )
    assert resp.status_code == 400
    assert "first_msg_ack" in resp.json()["detail"]
    assert "draft" in resp.json()["detail"]

    from rag_negotiations.crud import get_negotiation

    row = get_negotiation(neg_id)
    assert row is not None
    assert row["status"] == "draft"


def test_transition_missing_negotiation_returns_404(client):
    resp = client.post(
        "/api/negotiations/99999/transition",
        json={"transition": "launch"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "negotiation not found"


def test_queue_send_and_list_pending_sends(client):
    neg_id = _create_negotiation(client)

    resp = client.post(
        f"/api/negotiations/{neg_id}/queue-send",
        json={"content": "hola, coordinamos?", "delay_seconds": 60},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["negotiation_id"] == neg_id
    assert data["status"] == "pending"

    resp = client.get(f"/api/negotiations/{neg_id}/pending-sends")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == data["id"]
    assert items[0]["content"] == "hola, coordinamos?"
    assert items[0]["status"] == "pending"


def test_queue_send_rejects_terminal_negotiation(client):
    neg_id = _create_negotiation(client)
    resp = client.post(
        f"/api/negotiations/{neg_id}/transition",
        json={"transition": "user_cancels"},
    )
    assert resp.status_code == 200

    resp = client.post(
        f"/api/negotiations/{neg_id}/queue-send",
        json={"content": "no deberia salir"},
    )
    assert resp.status_code == 400
    assert "terminal" in resp.json()["detail"]


def test_queue_send_missing_negotiation_returns_404(client):
    resp = client.post(
        "/api/negotiations/99999/queue-send",
        json={"content": "hola"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "negotiation not found"


def test_process_due_sends_dry_run_does_not_mark_sent(client):
    neg_id = _create_negotiation(client)
    queued = client.post(
        f"/api/negotiations/{neg_id}/queue-send",
        json={"content": "sale ahora"},
    )
    assert queued.status_code == 200
    send_id = queued.json()["id"]

    resp = client.post(
        "/api/negotiations/process-due-sends",
        json={"dry_run": True, "limit": 10},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["dry_run"] is True
    assert data["items"] == [{
        "id": send_id,
        "negotiation_id": neg_id,
        "target_jid": "jid@s.whatsapp.net",
        "status": "due",
        "would_send": True,
    }]

    pending = client.get(f"/api/negotiations/{neg_id}/pending-sends").json()["items"]
    assert pending[0]["status"] == "pending"


def test_process_due_sends_sends_and_records_turn(client, monkeypatch):
    from rag.integrations.whatsapp import send as wa_send
    from rag_negotiations.crud import get_negotiation, list_turns

    calls = []

    def fake_send(jid, text, *, anti_loop=True, reply_to=None):
        calls.append({
            "jid": jid,
            "text": text,
            "anti_loop": anti_loop,
            "reply_to": reply_to,
        })
        return True, "sent"

    monkeypatch.setattr(wa_send, "_whatsapp_send_to_jid_detailed", fake_send)
    neg_id = _create_negotiation(client)
    queued = client.post(
        f"/api/negotiations/{neg_id}/queue-send",
        json={"content": "sale ahora", "typing_simulation_ms": 1200},
    )
    assert queued.status_code == 200
    send_id = queued.json()["id"]

    resp = client.post(
        "/api/negotiations/process-due-sends",
        json={"dry_run": False, "limit": 10},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["dry_run"] is False
    assert data["items"][0]["id"] == send_id
    assert data["items"][0]["status"] == "sent"
    assert calls == [{
        "jid": "jid@s.whatsapp.net",
        "text": "sale ahora",
        "anti_loop": True,
        "reply_to": None,
    }]

    pending = client.get(f"/api/negotiations/{neg_id}/pending-sends").json()["items"]
    assert pending[0]["status"] == "sent"
    turns = list_turns(neg_id)
    assert len(turns) == 1
    assert turns[0]["direction"] == "out"
    assert turns[0]["content"] == "sale ahora"
    assert turns[0]["pause_simulated_ms"] == 1200
    row = get_negotiation(neg_id)
    assert row is not None
    assert row["messages_sent"] == 1
