from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag
from web.anticipate_routes import register_anticipate_routes


@pytest.fixture()
def state_db(tmp_path, monkeypatch):
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    monkeypatch.setattr(rag, "PROACTIVE_STATE_PATH", tmp_path / "proactive.json")
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)
    return db_path


@pytest.fixture()
def client(state_db):
    app = FastAPI()
    register_anticipate_routes(app)
    return TestClient(app, raise_server_exceptions=False)


def _insert_candidate(
    *,
    kind: str = "anticipate-calendar",
    score: float = 0.7,
    dedup_key: str = "k1",
    selected: int = 0,
    sent: int = 0,
    reason: str = "test reason",
    message_preview: str = "test preview",
    message_full: str | None = None,
    minutes_ago: int = 1,
) -> int:
    ts = (datetime.now() - timedelta(minutes=minutes_ago)).isoformat(timespec="seconds")
    with rag._ragvec_state_conn() as conn:
        cur = conn.execute(
            "INSERT INTO rag_anticipate_candidates "
            "(ts, kind, score, dedup_key, selected, sent, reason, message_preview, message_full) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts,
                kind,
                score,
                dedup_key,
                selected,
                sent,
                reason,
                message_preview,
                message_full,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def test_inbox_returns_required_shape_and_summary(client, monkeypatch):
    monkeypatch.setattr(rag, "_ambient_config", lambda: {"jid": "test@s.whatsapp.net"})
    _insert_candidate(
        kind="anticipate-sent",
        dedup_key="sent",
        selected=1,
        sent=1,
        minutes_ago=3,
    )
    _insert_candidate(
        kind="anticipate-blocked",
        dedup_key="blocked",
        selected=1,
        sent=0,
        minutes_ago=2,
    )
    _insert_candidate(
        kind="anticipate-candidate",
        dedup_key="candidate",
        selected=0,
        sent=0,
        minutes_ago=1,
    )

    resp = client.get("/api/anticipate/inbox?limit=50&days=7")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["summary"]["total"] == 3
    assert body["summary"]["returned"] == 3

    required = {
        "id",
        "ts",
        "kind",
        "score",
        "dedup_key",
        "selected",
        "sent",
        "reason",
        "message_preview",
        "message_full",
        "message_len",
        "status",
        "skip_reason",
        "actionable",
    }
    assert required <= set(body["items"][0].keys())
    by_kind = {item["kind"]: item for item in body["items"]}
    assert by_kind["anticipate-sent"]["status"] == "sent"
    assert by_kind["anticipate-sent"]["actionable"] is False
    assert by_kind["anticipate-blocked"]["status"] == "blocked"
    assert by_kind["anticipate-blocked"]["skip_reason"] == "unknown_not_sent"
    assert by_kind["anticipate-candidate"]["status"] == "candidate"
    assert by_kind["anticipate-candidate"]["skip_reason"] == "not_selected"


def test_inbox_selected_unsent_reports_ambient_missing(client, monkeypatch):
    monkeypatch.setattr(rag, "_ambient_config", lambda: None)
    _insert_candidate(selected=1, sent=0, dedup_key="ambient-missing")

    resp = client.get("/api/anticipate/inbox")
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["status"] == "blocked"
    assert item["skip_reason"] == "ambient_missing"
    assert item["actionable"] is False


def test_inbox_only_actionable_filters_sent_silenced_snoozed_and_dedup_sent(
    client,
    monkeypatch,
):
    import rag.proactive as _proactive

    monkeypatch.setattr(rag, "_ambient_config", lambda: {"jid": "test@s.whatsapp.net"})
    today = datetime.now().strftime("%Y-%m-%d")
    _proactive._proactive_save_state({
        "date": today,
        "daily_count": 0,
        "silenced": ["anticipate-silenced"],
        "snooze": {
            "anticipate-snoozed": (
                datetime.now() + timedelta(hours=2)
            ).isoformat(timespec="seconds")
        },
    })
    _insert_candidate(kind="anticipate-open", dedup_key="open", selected=0, sent=0)
    _insert_candidate(kind="anticipate-silenced", dedup_key="silenced", selected=0, sent=0)
    _insert_candidate(kind="anticipate-snoozed", dedup_key="snoozed", selected=0, sent=0)
    _insert_candidate(kind="anticipate-dup", dedup_key="dup", selected=0, sent=0)
    _insert_candidate(kind="anticipate-dup", dedup_key="dup", selected=1, sent=1)
    _insert_candidate(kind="anticipate-sent", dedup_key="already-sent", selected=1, sent=1)

    resp = client.get("/api/anticipate/inbox?only_actionable=true")
    assert resp.status_code == 200
    body = resp.json()
    assert [item["dedup_key"] for item in body["items"]] == ["open"]
    assert body["items"][0]["actionable"] is True
    assert body["summary"]["hidden_by_only_actionable"] == 5


def test_send_calls_proactive_push_with_candidate_full_message(client, monkeypatch):
    candidate_id = _insert_candidate(
        kind="anticipate-send",
        dedup_key="send-key",
        selected=0,
        sent=0,
        reason="fallback reason",
        message_preview="stored preview",
        message_full="stored preview with the complete body and non-truncated next steps",
    )
    calls = []

    def fake_push(kind, message, *, snooze_hours=None, dedup_key=None):
        calls.append({
            "kind": kind,
            "message": message,
            "snooze_hours": snooze_hours,
            "dedup_key": dedup_key,
        })
        return True, None

    monkeypatch.setattr(rag, "proactive_push", fake_push)

    resp = client.post(f"/api/anticipate/{candidate_id}/send")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "sent": True, "skip_reason": None}
    assert calls == [{
        "kind": "anticipate-send",
        "message": "stored preview with the complete body and non-truncated next steps",
        "snooze_hours": None,
        "dedup_key": "send-key",
    }]
    with rag._ragvec_state_conn() as conn:
        sent = conn.execute(
            "SELECT sent FROM rag_anticipate_candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()[0]
    assert sent == 1


def test_send_falls_back_to_preview_for_legacy_candidates(client, monkeypatch):
    candidate_id = _insert_candidate(
        kind="anticipate-legacy",
        dedup_key="legacy-key",
        selected=0,
        sent=0,
        reason="fallback reason",
        message_preview="legacy preview",
        message_full=None,
    )
    calls = []

    def fake_push(kind, message, *, snooze_hours=None, dedup_key=None):
        del snooze_hours
        calls.append((kind, message, dedup_key))
        return True, None

    monkeypatch.setattr(rag, "proactive_push", fake_push)

    resp = client.post(f"/api/anticipate/{candidate_id}/send")

    assert resp.status_code == 200
    assert resp.json()["sent"] is True
    assert calls == [("anticipate-legacy", "legacy preview", "legacy-key")]


def test_snooze_updates_proactive_state(client):
    import rag.proactive as _proactive

    candidate_id = _insert_candidate(kind="anticipate-snooze", dedup_key="snooze-key")
    before = datetime.now()

    resp = client.post(f"/api/anticipate/{candidate_id}/snooze", json={"hours": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["kind"] == "anticipate-snooze"
    until = datetime.fromisoformat(body["snoozed_until"])
    assert before + timedelta(hours=1, minutes=50) < until
    assert until < datetime.now() + timedelta(hours=2, minutes=10)

    state = _proactive._proactive_load_state()
    assert state["snooze"]["anticipate-snooze"] == body["snoozed_until"]


def test_silence_adds_and_removes_kind_from_proactive_state(client):
    import rag.proactive as _proactive

    candidate_id = _insert_candidate(kind="anticipate-silence", dedup_key="silence-key")

    resp = client.post(f"/api/anticipate/{candidate_id}/silence", json={})
    assert resp.status_code == 200
    assert resp.json()["enabled"] is True
    assert "anticipate-silence" in _proactive._proactive_load_state()["silenced"]

    resp = client.post(
        f"/api/anticipate/{candidate_id}/silence",
        json={"enabled": False},
    )
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False
    assert "anticipate-silence" not in _proactive._proactive_load_state()["silenced"]
