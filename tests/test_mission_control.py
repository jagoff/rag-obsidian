from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag
from web import mission_control as mc


@pytest.fixture()
def isolated_state(tmp_path: Path, monkeypatch):
    db_dir = tmp_path / "ragvec"
    db_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_dir)
    monkeypatch.setattr(mc._audit, "SQL_ERRORS_LOG", tmp_path / "sql_state_errors.jsonl")
    monkeypatch.setattr(mc._audit, "SILENT_ERRORS_LOG", tmp_path / "silent_errors.jsonl")

    db_path = db_dir / "telemetry.db"
    with sqlite3.connect(str(db_path)) as conn:
        rag._ensure_telemetry_tables(conn)
        conn.commit()
    return db_path


@pytest.fixture()
def client(isolated_state):
    app = FastAPI()
    registered = mc.register_mission_control_routes(app)
    assert "api_mission_control_health" in registered
    return TestClient(app, raise_server_exceptions=False)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _with_conn(db_path: Path):
    return sqlite3.connect(str(db_path))


def _seed_query(
    conn: sqlite3.Connection,
    *,
    cmd: str = "web",
    top_score: float | None = 0.7,
    t_retrieve: float | None = 0.2,
    t_gen: float | None = 0.5,
    critique_fired: int | None = 1,
    extra_json: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO rag_queries
          (ts, cmd, q, top_score, t_retrieve, t_gen, critique_fired, extra_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (_now(), cmd, "test query", top_score, t_retrieve, t_gen, critique_fired, extra_json),
    )


def _seed_anticipate(
    conn: sqlite3.Connection,
    *,
    count: int = 1,
    sent: int = 1,
    kind: str = "anticipate-calendar",
    score: float = 0.9,
) -> None:
    for idx in range(count):
        conn.execute(
            """
            INSERT INTO rag_anticipate_candidates
              (ts, kind, score, dedup_key, selected, sent, reason, message_preview)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (_now(), kind, score, f"{kind}:{idx}", 1, sent, "test", "preview"),
        )


def _seed_corrective_gate_open(conn: sqlite3.Connection) -> None:
    for idx in range(20):
        conn.execute(
            "INSERT INTO rag_feedback (ts, rating, q, extra_json) VALUES (?, ?, ?, ?)",
            (
                _now(),
                -1,
                f"bad answer {idx}",
                json.dumps({"corrective_path": f"notes/golden_{idx}.md"}),
            ),
        )


def _seed_all_ok(db_path: Path) -> None:
    with _with_conn(db_path) as conn:
        _seed_anticipate(conn)
        _seed_query(conn, extra_json=json.dumps({"cache_probe": {"result": "hit"}}))
        _seed_corrective_gate_open(conn)
        conn.commit()


def _subsystems(payload: dict) -> dict[str, dict]:
    return {item["id"]: item for item in payload["subsystems"]}


def _action_ids(payload: dict) -> set[str]:
    return set(payload["actions"])


def test_mission_control_payload_shape_and_ok_status(client, isolated_state):
    _seed_all_ok(isolated_state)

    resp = client.get("/api/health/mission-control?days=7")

    assert resp.status_code == 200
    payload = resp.json()
    assert set(payload) == {
        "ok",
        "generated_at",
        "window_days",
        "overall",
        "subsystems",
        "actions",
    }
    assert payload["ok"] is True
    assert payload["window_days"] == 7
    assert payload["overall"] == "ok"

    subsystems = _subsystems(payload)
    assert {
        "anticipate",
        "retrieval",
        "chat",
        "telemetry_errors",
        "learning_gates",
    }.issubset(subsystems)
    for item in subsystems.values():
        assert {"id", "label", "status", "issues", "details"}.issubset(item)

    rag_queries = subsystems["telemetry_errors"]["details"]["db_stats"]["tables"]["rag_queries"]
    assert rag_queries["rows"] == 1
    assert payload["actions"] == []


def test_mission_control_missing_tables_is_graceful(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    monkeypatch.setattr(mc._audit, "SQL_ERRORS_LOG", tmp_path / "sql_state_errors.jsonl")
    monkeypatch.setattr(mc._audit, "SILENT_ERRORS_LOG", tmp_path / "silent_errors.jsonl")
    empty_db = tmp_path / "empty.db"

    @contextmanager
    def _empty_conn():
        conn = sqlite3.connect(str(empty_db))
        try:
            yield conn
        finally:
            conn.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _empty_conn)

    app = FastAPI()
    mc.register_mission_control_routes(app)
    resp = TestClient(app, raise_server_exceptions=False).get("/api/health/mission-control")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["overall"] == "degraded"
    subsystems = _subsystems(payload)
    assert subsystems["retrieval"]["status"] == "degraded"
    assert any("rag_queries" in issue for issue in subsystems["retrieval"]["issues"])
    assert subsystems["learning_gates"]["status"] == "degraded"


def test_mission_control_overall_uses_worst_subsystem_status(client):
    resp = client.get("/api/health/mission-control?days=7")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["overall"] == "down"
    subsystems = _subsystems(payload)
    assert subsystems["retrieval"]["status"] == "down"
    assert subsystems["chat"]["status"] == "down"


def test_mission_control_actions_for_anticipate_and_corrective_gap(client, isolated_state):
    with _with_conn(isolated_state) as conn:
        _seed_anticipate(conn, count=30, sent=0, score=0.2)
        _seed_query(conn)
        conn.commit()

    resp = client.get("/api/health/mission-control?days=7")

    assert resp.status_code == 200
    payload = resp.json()
    subsystems = _subsystems(payload)
    assert subsystems["anticipate"]["status"] == "degraded"
    assert subsystems["learning_gates"]["status"] == "degraded"
    assert {"open_anticipate_inbox", "run_feedback_harvest"}.issubset(_action_ids(payload))


def test_mission_control_action_for_many_silent_errors(client, isolated_state, tmp_path: Path):
    log_path = tmp_path / "silent_errors.jsonl"
    now = _now()
    with log_path.open("w", encoding="utf-8") as fh:
        for idx in range(10):
            fh.write(json.dumps({"ts": now, "where": f"silent_{idx}"}) + "\n")
    mc._audit.SILENT_ERRORS_LOG = log_path
    _seed_all_ok(isolated_state)

    resp = client.get("/api/health/mission-control?days=7")

    assert resp.status_code == 200
    payload = resp.json()
    assert _subsystems(payload)["telemetry_errors"]["status"] == "degraded"
    assert "inspect_silent_errors" in _action_ids(payload)


def test_mission_control_action_inspects_silent_errors(client, isolated_state, tmp_path: Path):
    log_path = tmp_path / "silent_errors.jsonl"
    log_path.write_text(json.dumps({"ts": _now(), "where": "boom"}) + "\n", encoding="utf-8")
    mc._audit.SILENT_ERRORS_LOG = log_path
    _seed_all_ok(isolated_state)

    resp = client.post(
        "/api/health/mission-control/action",
        json={"action_id": "inspect_silent_errors", "days": 7},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["executed"] is False
    assert payload["result"]["total_errors"] == 1


def test_mission_control_action_feedback_harvest_dry_run(client, isolated_state):
    with _with_conn(isolated_state) as conn:
        _seed_query(conn, top_score=0.1)
        conn.commit()

    resp = client.post(
        "/api/health/mission-control/action",
        json={"action_id": "run_feedback_harvest", "dry_run": True, "days": 7},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["executed"] is False
    assert payload["result"]["count"] >= 1
    assert "rag feedback harvest" in payload["result"]["harvest_command"]


def test_mission_control_action_check_telemetry_db(client):
    resp = client.post(
        "/api/health/mission-control/action",
        json={"action_id": "check_telemetry_db"},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["executed"] is True
    assert payload["result"]["accessible"] is True


def test_mission_control_unknown_action_returns_400(client):
    resp = client.post(
        "/api/health/mission-control/action",
        json={"action_id": "does_not_exist"},
    )

    assert resp.status_code == 400
    assert "does_not_exist" in resp.json()["detail"]
