"""Tests for roadmap-facing active-learning API aliases."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path: Path):
    import rag

    snap = rag.DB_PATH
    db_dir = tmp_path / "ragvec"
    db_dir.mkdir(parents=True, exist_ok=True)
    rag.DB_PATH = db_dir
    with rag._TELEMETRY_DDL_LOCK:
        snapped_paths = set(rag._TELEMETRY_DDL_ENSURED_PATHS)
        rag._TELEMETRY_DDL_ENSURED_PATHS.clear()
    try:
        with rag._ragvec_state_conn() as conn:
            rag._ensure_telemetry_tables(conn)
        yield db_dir
    finally:
        rag.DB_PATH = snap
        with rag._TELEMETRY_DDL_LOCK:
            rag._TELEMETRY_DDL_ENSURED_PATHS.clear()
            rag._TELEMETRY_DDL_ENSURED_PATHS.update(snapped_paths)


@pytest.fixture()
def client():
    from web.server import app

    return TestClient(app, raise_server_exceptions=False)


def test_learning_queue_alias_wraps_fine_tunning_queue(client, _isolate_db_path):
    import rag

    with rag._ragvec_state_conn() as conn:
        conn.execute(
            """
            INSERT INTO rag_queries
                (ts, q, top_score, answer_len, paths_json, cmd)
            VALUES
                (datetime('now', '-1 hour'),
                 'learning queue alias candidate', 0.2, 512,
                 '["candidate.md"]', 'web')
            """
        )
        conn.commit()

    resp = client.get("/api/learning/queue?limit=20&days=30")

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "fine_tunning"
    assert body["count"] >= 1
    assert any(
        item["label"] == "learning queue alias candidate"
        for item in body["items"]
    )


def test_learning_corrective_path_writes_feedback(client, _isolate_db_path):
    import rag

    resp = client.post(
        "/api/learning/corrective-path",
        json={
            "turn_id": "turn-learning-alias",
            "q": "pregunta mal respondida",
            "paths": ["wrong.md"],
            "corrective_path": "right.md",
            "session_id": "web:test-learning-alias",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    with sqlite3.connect(rag.DB_PATH / "telemetry.db") as conn:
        row = conn.execute(
            "SELECT rating, q, paths_json, extra_json FROM rag_feedback "
            "WHERE turn_id = ?",
            ("turn-learning-alias",),
        ).fetchone()
    assert row is not None
    assert row[0] == -1
    assert row[1] == "pregunta mal respondida"
    assert json.loads(row[2]) == ["wrong.md"]
    assert json.loads(row[3])["corrective_path"] == "right.md"


def test_learning_corrective_path_rejects_uris(client):
    resp = client.post(
        "/api/learning/corrective-path",
        json={
            "turn_id": "turn-learning-alias",
            "corrective_path": "https://example.com/right.md",
        },
    )

    assert resp.status_code == 422
