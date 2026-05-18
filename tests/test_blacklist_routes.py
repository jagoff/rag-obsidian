from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_web_server = pytest.importorskip("web.server")
app = _web_server.app


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _isolate_blacklist(monkeypatch, tmp_path: Path):
    import rag.exclusions as _exclusions

    monkeypatch.delenv("OBSIDIAN_RAG_TEST_ADMIN_BYPASS", raising=False)
    monkeypatch.setattr(_exclusions, "_DB_PATH", tmp_path / "blacklist.db")
    monkeypatch.setattr(_exclusions, "_CONFIG_PATH", tmp_path / "blacklist.json")
    monkeypatch.setattr(_exclusions, "_LEGACY_IGNORED_PATH", tmp_path / "ignored_notes.json")
    monkeypatch.setattr(_exclusions, "_CACHE", None)
    monkeypatch.setattr(_exclusions, "_LEGACY_CACHE", None)
    yield


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {_web_server._ADMIN_TOKEN}"}


def test_blacklist_page_served(client):
    resp = client.get("/blacklist")
    assert resp.status_code == 200
    assert "blacklist" in resp.text


def test_blacklist_api_requires_admin(client):
    resp = client.get("/api/blacklist")
    assert resp.status_code == 401


def test_blacklist_api_adds_and_removes_items(client):
    resp = client.get("/api/blacklist", headers=_auth())
    assert resp.status_code == 200
    assert "Cloud Services" in resp.json()["config"]["chats"]

    resp = client.post(
        "/api/blacklist",
        headers=_auth(),
        json={"kind": "palabra_parecida", "value": "japon"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["changed"] is True
    assert "japon" in data["config"]["fuzzy_words"]

    resp = client.post(
        "/api/blacklist/delete",
        headers=_auth(),
        json={"kind": "fuzzy_words", "value": "japon"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["changed"] is True
    assert "japon" not in data["config"]["fuzzy_words"]
