"""Tests para GET /api/admin/token + integración del frontend.

Cobertura:
1. localhost (127.0.0.1 / ::1 / localhost) → 200 con el token.
2. LAN remoto (192.168.x.x) → 403.
3. tunnel cloudflare → 403.
4. Sin client (request.client=None) → 403 (defensive).
5. Token devuelto matchea _ADMIN_TOKEN.
6. Los 3 HTML cargan admin-auth.js antes que app.js / dashboard.js / home.js.
7. admin-auth.js está en /static/ y es servible.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path):
    import rag as _rag
    snap = _rag.DB_PATH
    _rag.DB_PATH = tmp_path / "ragvec"
    try:
        yield
    finally:
        _rag.DB_PATH = snap


def _client():
    from fastapi.testclient import TestClient
    from web.server import app
    return TestClient(app)


# ── 1. localhost happy paths ──────────────────────────────────────────


def test_admin_token_endpoint_localhost_127_returns_token():
    from fastapi.testclient import TestClient
    from web.server import app, _ADMIN_TOKEN
    client = TestClient(app)
    # TestClient envía request.client.host = "testclient" by default;
    # forzamos via headers no funciona — patcheamos _is_localhost_request.
    with patch("web.server._is_localhost_request", return_value=True):
        resp = client.get("/api/admin/token")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "token" in body
    assert body["token"] == _ADMIN_TOKEN
    assert len(body["token"]) > 20  # secrets.token_urlsafe(32) → ~43 chars


# ── 2. LAN / remote NEGATIVE ───────────────────────────────────────────


def test_admin_token_endpoint_lan_remote_returns_403():
    client = _client()
    with patch("web.server._is_localhost_request", return_value=False):
        resp = client.get("/api/admin/token")
    assert resp.status_code == 403
    assert "localhost" in resp.json().get("detail", "").lower()


# ── 3. _is_localhost_request unit ──────────────────────────────────────


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True),
    ("::1", True),
    ("localhost", True),
    ("LOCALHOST", True),  # case-insensitive
    ("192.168.1.50", False),
    ("10.0.0.5", False),
    ("172.16.0.1", False),
    ("evil.trycloudflare.com", False),
    ("", False),
    ("unknown", False),
])
def test_is_localhost_request_classifier(host, expected):
    from web.server import _is_localhost_request
    fake_request = type("R", (), {"client": type("C", (), {"host": host})()})()
    assert _is_localhost_request(fake_request) is expected


def test_is_localhost_request_no_client_returns_false():
    from web.server import _is_localhost_request
    fake_request = type("R", (), {"client": None})()
    assert _is_localhost_request(fake_request) is False


# ── 4. Frontend wiring — admin-auth.js está cargado en los 3 HTML ─────


@pytest.mark.parametrize("html_path", [
    "web/static/index.html",
    "web/static/dashboard.html",
    "web/static/home.html",
])
def test_html_loads_admin_auth_js(html_path):
    repo = Path(__file__).resolve().parent.parent
    content = (repo / html_path).read_text(encoding="utf-8")
    assert "/static/admin-auth.js" in content, (
        f"{html_path} no incluye admin-auth.js — los endpoints admin "
        f"van a fallar con 401 desde la UI."
    )


def test_admin_auth_js_exists_and_has_monkey_patch():
    repo = Path(__file__).resolve().parent.parent
    js = (repo / "web/static/admin-auth.js").read_text(encoding="utf-8")
    # Sanity checks: monkey-patch del fetch global + lista de admin paths +
    # endpoint /api/admin/token.
    assert "window.fetch" in js
    assert "/api/admin/token" in js
    assert "/api/auto-fix-devin" in js
    assert "/api/reindex" in js
    assert "Authorization" in js
    assert "Bearer" in js


# ── 5. admin-auth.js NO se carga con `defer` ──────────────────────────


@pytest.mark.parametrize("html_path", [
    "web/static/index.html",
    "web/static/dashboard.html",
    "web/static/home.html",
])
def test_admin_auth_js_not_deferred(html_path):
    """Si admin-auth.js carga con defer, app.js puede ejecutar antes y
    hacer fetch admin sin token. Tiene que ser sync (sin defer) para
    garantizar que el monkey-patch de fetch corre primero.
    """
    repo = Path(__file__).resolve().parent.parent
    content = (repo / html_path).read_text(encoding="utf-8")
    # Buscar la línea exacta del script admin-auth.js
    for line in content.splitlines():
        if "admin-auth.js" in line:
            assert "defer" not in line, (
                f"{html_path}: admin-auth.js no debe tener defer "
                f"(monkey-patch de fetch debe correr antes que el resto)"
            )
            return
    pytest.fail(f"{html_path}: admin-auth.js script tag no encontrado")
