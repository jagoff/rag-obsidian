from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_web_server = pytest.importorskip("web.server")
app = _web_server.app


@pytest.fixture()
def client(monkeypatch, tmp_path: Path):
    import rag.ui_layout_state as _state

    monkeypatch.setattr(_state, "_DB_PATH", tmp_path / "ui-layout.db")
    return TestClient(app, raise_server_exceptions=False)


def test_ui_layout_api_persists_items(client):
    resp = client.get("/api/ui-layout/home.v2")
    assert resp.status_code == 200
    assert resp.json()["state"] == {}
    assert resp.json()["updated_at"] == {}

    resp = client.post(
        "/api/ui-layout/home.v2",
        json={
            "key": "home.v2.panels.order.v1",
            "value": {"home-cmdbar": ["kpi-loops", "kpi-inbox"]},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["changed"] is True

    resp = client.get("/api/ui-layout/home.v2")
    assert resp.status_code == 200
    assert resp.json()["state"]["home.v2.panels.order.v1"] == {
        "home-cmdbar": ["kpi-loops", "kpi-inbox"],
    }
    assert isinstance(resp.json()["updated_at"]["home.v2.panels.order.v1"], str)
    assert resp.json()["updated_at"]["home.v2.panels.order.v1"]


def test_ui_layout_api_deletes_and_clears(client):
    client.post(
        "/api/ui-layout/home.v2",
        json={"key": "home.v2.hero.collapsed.v1", "value": "1"},
    )
    resp = client.post(
        "/api/ui-layout/home.v2",
        json={"key": "home.v2.hero.collapsed.v1", "value": None},
    )
    assert resp.status_code == 200
    assert resp.json()["changed"] is True
    assert "home.v2.hero.collapsed.v1" not in resp.json()["state"]

    client.post(
        "/api/ui-layout/home.v2/snapshot",
        json={"state": {"home.v2.panel-sizes.v1": {"p-inbox": {"w": "full", "h": "xl"}}}},
    )
    resp = client.delete("/api/ui-layout/home.v2")
    assert resp.status_code == 200
    assert resp.json()["changed"] is True
    assert resp.json()["state"] == {}
    assert resp.json()["updated_at"] == {}
