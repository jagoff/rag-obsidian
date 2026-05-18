from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag
from rag.conversation_distiller import RUNBOOKS_DIR
from web.memory_routes import register_memory_routes


@pytest.fixture
def memory_env(tmp_path, monkeypatch):
    db_dir = tmp_path / "rag-state"
    db_dir.mkdir(parents=True)
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "DB_PATH", db_dir)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    with rag._TELEMETRY_DDL_LOCK:
        ensured_paths = set(rag._TELEMETRY_DDL_ENSURED_PATHS)
        rag._TELEMETRY_DDL_ENSURED_PATHS.clear()
    try:
        with sqlite3.connect(db_dir / "telemetry.db") as conn:
            rag._ensure_telemetry_tables(conn)
            conn.commit()
        yield {"db_dir": db_dir, "vault": vault}
    finally:
        with rag._TELEMETRY_DDL_LOCK:
            rag._TELEMETRY_DDL_ENSURED_PATHS.clear()
            rag._TELEMETRY_DDL_ENSURED_PATHS.update(ensured_paths)


@pytest.fixture
def client(memory_env):
    app = FastAPI()
    register_memory_routes(app)
    return TestClient(app)


def _seed_conversations(db_dir: Path, rows: list[tuple[str, str, str]]) -> None:
    with sqlite3.connect(db_dir / "telemetry.db") as conn:
        conn.executemany(
            """
            INSERT INTO rag_conversations_index(session_id, relative_path, updated_at)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def _write_runbook(vault: Path, name: str, mtime: float) -> Path:
    path = vault / RUNBOOKS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {name}\n", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def _write_conversation(vault: Path, name: str = "ops.md") -> Path:
    path = vault / "99-obsidian/99-AI/conversations" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """---
confidence_avg: 0.900
sources:
  - 00-Inbox/missing-source.md
---

## Turn 1

> como reinicio el daemon web?

Usar launchctl kickstart para reiniciar el servicio web y luego verificar /api/status.

**Sources**: 00-Inbox/missing-source.md
""",
        encoding="utf-8",
    )
    return path


def test_unified_payload_includes_memo_conversations_and_runbooks(
    client,
    memory_env,
    monkeypatch,
):
    snapshot_calls: list[int] = []

    def fake_snapshot(limit=20, type_filter=None):
        del type_filter
        snapshot_calls.append(limit)
        return {
            "ok": True,
            "totals": {"all": 3},
            "recent": [{"id": "memo-new"}, {"id": "memo-old"}],
        }

    monkeypatch.setattr("web.memo_dashboard.snapshot", fake_snapshot)
    _seed_conversations(
        memory_env["db_dir"],
        [
            (
                "web:old",
                "99-obsidian/99-AI/conversations/old.md",
                "2026-05-01T10:00:00Z",
            ),
            (
                "web:new",
                "99-obsidian/99-AI/conversations/new.md",
                "2026-05-02T10:00:00Z",
            ),
        ],
    )
    _write_runbook(memory_env["vault"], "old.md", 1_700_000_000)
    newest_runbook = _write_runbook(memory_env["vault"], "new.md", 1_800_000_000)

    response = client.get("/api/memory/unified?limit=1")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert isinstance(body["generated_at"], str)
    assert set(body["sections"]) == {"memo", "conversations", "runbooks"}
    assert snapshot_calls == [1]
    assert body["summary"]["status"] == "ok"
    assert body["summary"]["counts"] == {
        "memo": 3,
        "conversations": 2,
        "runbooks": 2,
    }
    assert body["summary"]["memo_count"] == 3
    assert body["summary"]["conversations_count"] == 2
    assert body["summary"]["runbooks_count"] == 2

    memo = body["sections"]["memo"]
    assert memo["ok"] is True
    assert memo["count"] == 3
    assert memo["latest"] == [{"id": "memo-new"}]
    assert memo["snapshot"]["totals"]["all"] == 3

    conversations = body["sections"]["conversations"]
    assert conversations["ok"] is True
    assert conversations["count"] == 2
    assert conversations["latest"] == [
        {
            "session_id": "web:new",
            "relative_path": "99-obsidian/99-AI/conversations/new.md",
            "updated_at": "2026-05-02T10:00:00Z",
        },
    ]

    runbooks = body["sections"]["runbooks"]
    assert runbooks["ok"] is True
    assert runbooks["count"] == 2
    assert runbooks["latest"][0]["relative_path"] == str(
        newest_runbook.relative_to(memory_env["vault"]),
    )


def test_unified_payload_gracefully_handles_memo_failure(
    client,
    memory_env,
    monkeypatch,
):
    def broken_snapshot(limit=20, type_filter=None):
        del limit, type_filter
        raise RuntimeError("memo down")

    monkeypatch.setattr("web.memo_dashboard.snapshot", broken_snapshot)
    _seed_conversations(
        memory_env["db_dir"],
        [
            (
                "web:one",
                "99-obsidian/99-AI/conversations/one.md",
                "2026-05-03T10:00:00Z",
            ),
        ],
    )

    response = client.get("/api/memory/unified")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["summary"]["status"] == "partial"
    assert body["summary"]["counts"]["memo"] == 0
    assert body["summary"]["counts"]["conversations"] == 1
    assert body["summary"]["errors"] == ["memo"]
    assert body["sections"]["memo"]["ok"] is False
    assert "RuntimeError: memo down" in body["sections"]["memo"]["error"]
    assert body["sections"]["conversations"]["ok"] is True
    assert body["sections"]["runbooks"]["ok"] is True


def test_unified_payload_is_empty_when_sources_have_no_data(
    client,
    memory_env,
    monkeypatch,
):
    def empty_snapshot(limit=20, type_filter=None):
        del limit, type_filter
        return {"ok": True, "totals": {"all": 0}, "recent": []}

    monkeypatch.setattr("web.memo_dashboard.snapshot", empty_snapshot)
    monkeypatch.setattr(rag, "VAULT_PATH", memory_env["vault"] / "missing-vault")

    response = client.get("/api/memory/unified")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["status"] == "empty"
    assert body["summary"]["counts"] == {
        "memo": 0,
        "conversations": 0,
        "runbooks": 0,
    }
    assert body["sections"]["runbooks"]["ok"] is True
    assert body["sections"]["runbooks"]["latest"] == []


def test_distill_conversations_dry_run_returns_runbook_plan(client, memory_env):
    _write_conversation(memory_env["vault"])

    response = client.post(
        "/api/memory/distill-conversations",
        json={"apply": False, "limit": 5, "min_confidence": 0.5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["result"]["apply"] is False
    assert body["result"]["candidates"] == 1
    assert body["result"]["distilled"][0]["runbook"].startswith(RUNBOOKS_DIR)
    assert not list((memory_env["vault"] / RUNBOOKS_DIR).glob("*.md"))


def test_distill_conversations_apply_writes_runbook(client, memory_env):
    conv_path = _write_conversation(memory_env["vault"])

    response = client.post(
        "/api/memory/distill-conversations",
        json={"apply": True, "limit": 5, "min_confidence": 0.5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["apply"] is True
    assert body["result"]["candidates"] == 1
    runbook_rel = body["result"]["distilled"][0]["runbook"]
    runbook = memory_env["vault"] / runbook_rel
    assert runbook.is_file()
    assert "launchctl kickstart" in runbook.read_text(encoding="utf-8")
    assert f"distilled_to: {runbook_rel}" in conv_path.read_text(encoding="utf-8")
