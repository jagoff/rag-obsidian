"""Tests del panel Daemons launchd en el dashboard web.

Cubre:
1. GET /api/daemons/status → 200 + shape correcta.
2. Cada item tiene los 8 keys del dict de _gather_daemon_status.
3. unhealthy_count se computa correctamente.
4. Rate limit: el 121° request en 60s devuelve 429.
5. Cache 30s: dos requests consecutivos no llaman _gather_daemon_status dos veces.
6. GET /dashboard devuelve HTML con "Daemons launchd" (smoke del wiring).
7. dashboard.js referencia /api/daemons/status.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import web.server as _server
import rag as _rag


# ── DB isolation (patrón canónico del repo) ───────────────────────────────
@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path):
    snap = _rag.DB_PATH
    _rag.DB_PATH = tmp_path / "ragvec"
    try:
        yield
    finally:
        _rag.DB_PATH = snap


# ── Helpers ───────────────────────────────────────────────────────────────

_DAEMON_KEYS = {
    "label", "category", "state", "runs",
    "last_exit", "last_tick_iso", "overdue", "expected_cadence_s",
}

_FAKE_MANAGED = [
    {"label": "com.fer.obsidian-rag-web",      "category": "managed",      "state": "running",     "runs": 12,   "last_exit": 0,    "last_tick_iso": "2026-05-01T08:00:00", "overdue": False, "expected_cadence_s": None},
    {"label": "com.fer.obsidian-rag-morning",  "category": "managed",      "state": "not running", "runs": 3,    "last_exit": 0,    "last_tick_iso": "2026-05-01T07:00:00", "overdue": False, "expected_cadence_s": 86400},
    {"label": "com.fer.obsidian-rag-missing",  "category": "managed",      "state": "missing",     "runs": None, "last_exit": None, "last_tick_iso": None,                  "overdue": False, "expected_cadence_s": None},
]
_FAKE_MANUAL = [
    {"label": "com.fer.obsidian-rag-cloudflare-tunnel", "category": "manual_keep", "state": "running", "runs": 5, "last_exit": 1, "last_tick_iso": "2026-04-30T12:00:00", "overdue": True, "expected_cadence_s": 3600},
]
_ALL_FAKE = _FAKE_MANAGED + _FAKE_MANUAL


def _make_fake_gather(items=None):
    """Devuelve un fake para _gather_daemon_status que retorna items de _ALL_FAKE en orden."""
    if items is None:
        items = _ALL_FAKE
    call_counter = {"n": 0}
    label_map = {it["label"]: it for it in items}

    def _fake(label, category):
        call_counter["n"] += 1
        return label_map.get(label, {
            "label": label, "category": category, "state": "missing",
            "runs": None, "last_exit": None, "last_tick_iso": None,
            "overdue": False, "expected_cadence_s": None,
        })

    return _fake, call_counter


def _make_fake_all_labels(items=None):
    if items is None:
        items = _ALL_FAKE
    return lambda: [(it["label"], it["category"]) for it in items]


# ── Fixture: client con cache reseteada y mocks ───────────────────────────

@pytest.fixture()
def client_with_mocks(monkeypatch):
    """TestClient con _gather_daemon_status y _all_daemon_labels parchados.

    También resetea el cache interno del endpoint para que cada test empiece
    desde cero (sin TTL residual de un test anterior).
    """
    # Reset cache
    _server._daemons_cache_ts = 0.0
    _server._daemons_cache_data = None

    fake_gather, counter = _make_fake_gather()
    monkeypatch.setattr("web.server._gather_daemon_status", fake_gather)
    monkeypatch.setattr("web.server._all_daemon_labels", _make_fake_all_labels())

    client = TestClient(_server.app, raise_server_exceptions=True)
    return client, counter


# ── Tests ─────────────────────────────────────────────────────────────────

def test_daemons_status_200_and_shape(client_with_mocks):
    """GET /api/daemons/status → 200 + shape esperada."""
    client, _ = client_with_mocks
    resp = client.get("/api/daemons/status")
    assert resp.status_code == 200
    data = resp.json()

    # Keys de nivel superior
    assert "items" in data
    assert "ts" in data
    assert "managed_count" in data
    assert "manual_count" in data
    assert "unhealthy_count" in data

    assert isinstance(data["items"], list)
    assert len(data["items"]) == len(_ALL_FAKE)

    # Conteos de categoría
    assert data["managed_count"] == sum(1 for it in _ALL_FAKE if it["category"] == "managed")
    assert data["manual_count"] == sum(1 for it in _ALL_FAKE if it["category"] != "managed")


def test_each_item_has_required_keys(client_with_mocks):
    """Cada item del response tiene los 8 keys del dict de _gather_daemon_status."""
    client, _ = client_with_mocks
    data = client.get("/api/daemons/status").json()
    for item in data["items"]:
        missing = _DAEMON_KEYS - item.keys()
        assert missing == set(), f"Item {item.get('label')} falta keys: {missing}"


def test_unhealthy_count_computation(client_with_mocks):
    """unhealthy_count cuenta state=missing|unknown | int last_exit!=0 | overdue=True.
    "not running" y last_exit como string ("(never exited)") NO cuentan como unhealthy.
    """
    client, _ = client_with_mocks
    data = client.get("/api/daemons/status").json()

    expected_unhealthy = 0
    for it in _ALL_FAKE:
        if (
            it["state"] in ("missing", "unknown")
            or (isinstance(it["last_exit"], int) and it["last_exit"] != 0)
            or it["overdue"]
        ):
            expected_unhealthy += 1

    assert data["unhealthy_count"] == expected_unhealthy


def test_rate_limit_429(monkeypatch):
    """121 llamadas en 60s → la 121ª devuelve 429."""
    # Reset cache y buckets
    _server._daemons_cache_ts = 0.0
    _server._daemons_cache_data = None
    _server._BEHAVIOR_BUCKETS.clear()

    monkeypatch.setattr("web.server._gather_daemon_status", _make_fake_gather()[0])
    monkeypatch.setattr("web.server._all_daemon_labels", _make_fake_all_labels())

    client = TestClient(_server.app, raise_server_exceptions=False)
    limit = _server._BEHAVIOR_RATE_LIMIT  # 120

    statuses = []
    for _ in range(limit + 1):
        r = client.get("/api/daemons/status")
        statuses.append(r.status_code)

    assert statuses[-1] == 429, f"Último status debería ser 429, fue {statuses[-1]}"
    assert all(s == 200 for s in statuses[:-1]), "Los primeros {limit} deben ser 200"

    # Limpiar bucket para no contaminar otros tests
    _server._BEHAVIOR_BUCKETS.clear()


def test_cache_30s_prevents_double_call(monkeypatch):
    """Dos requests consecutivos usan el cache → _gather_daemon_status se llama solo una vez."""
    # Reset cache
    _server._daemons_cache_ts = 0.0
    _server._daemons_cache_data = None

    fake_gather, counter = _make_fake_gather()
    monkeypatch.setattr("web.server._gather_daemon_status", fake_gather)
    monkeypatch.setattr("web.server._all_daemon_labels", _make_fake_all_labels())

    client = TestClient(_server.app)

    # Primer request — puebla el cache
    r1 = client.get("/api/daemons/status")
    assert r1.status_code == 200
    calls_after_first = counter["n"]
    assert calls_after_first > 0

    # Segundo request inmediato — debe leer del cache (TTL 30s)
    r2 = client.get("/api/daemons/status")
    assert r2.status_code == 200
    calls_after_second = counter["n"]
    assert calls_after_second == calls_after_first, (
        f"El segundo request llamó _gather_daemon_status otra vez "
        f"({calls_after_second} calls vs {calls_after_first} esperadas)"
    )


def test_dashboard_html_contains_daemons_string():
    """GET /dashboard → HTML incluye 'Daemons launchd' (smoke del wiring del panel)."""
    client = TestClient(_server.app)
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "Daemons launchd" in resp.text


def test_dashboard_js_references_api_daemons():
    """dashboard.js referencia la URL /api/daemons/status."""
    js_path = Path(_server.STATIC_DIR) / "dashboard.js"
    assert js_path.exists(), f"dashboard.js no encontrado en {js_path}"
    content = js_path.read_text()
    assert "/api/daemons/status" in content, (
        "dashboard.js no referencia /api/daemons/status — revisar daemonsRefresh()"
    )
