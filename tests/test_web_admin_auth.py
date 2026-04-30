"""Tests para el admin token en endpoints de pánico.

Valida que los 8 endpoints protegidos devuelven 401 sin token,
401 con token incorrecto, y 2xx/proceden con el token correcto.
También valida la generación del token file al primer boot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_web_server = pytest.importorskip("web.server")
app = _web_server.app

from fastapi.testclient import TestClient  # noqa: E402


# ── DB isolation ──────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path):
    import rag as _rag
    snap = _rag.DB_PATH
    _rag.DB_PATH = tmp_path / "ragvec"
    try:
        yield
    finally:
        _rag.DB_PATH = snap


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def valid_token():
    """Devuelve el token admin que el server cargó al startup."""
    return _web_server._ADMIN_TOKEN


# ── Endpoints a proteger ──────────────────────────────────────────────────────
_ADMIN_ENDPOINTS = [
    "/api/ollama/restart",
    "/api/ollama/unload",
    "/api/reindex",
    "/api/status/action",
    "/api/diagnose-error/execute",
    "/api/auto-fix",
    "/api/auto-fix-devin",
    "/api/chat/model",
]


@pytest.mark.parametrize("endpoint", _ADMIN_ENDPOINTS)
def test_admin_endpoint_401_without_token(client, endpoint):
    """Sin header Authorization → 401."""
    resp = client.post(endpoint, json={})
    assert resp.status_code == 401, (
        f"{endpoint} retornó {resp.status_code}, esperaba 401 sin token"
    )


@pytest.mark.parametrize("endpoint", _ADMIN_ENDPOINTS)
def test_admin_endpoint_401_wrong_token(client, endpoint):
    """Token incorrecto → 401."""
    resp = client.post(
        endpoint,
        json={},
        headers={"Authorization": "Bearer wrong-token-abc123"},
    )
    assert resp.status_code == 401, (
        f"{endpoint} retornó {resp.status_code}, esperaba 401 con token incorrecto"
    )


@pytest.mark.parametrize("endpoint", _ADMIN_ENDPOINTS)
def test_admin_endpoint_not_401_with_correct_token(client, valid_token, endpoint):
    """Con token correcto, el endpoint no debe devolver 401.
    (Puede devolver 422 por payload vacío, 500 por falta de ollama, etc. — lo
    importante es que la capa de auth no bloquea.)"""
    resp = client.post(
        endpoint,
        json={},
        headers={"Authorization": f"Bearer {valid_token}"},
    )
    assert resp.status_code != 401, (
        f"{endpoint} devolvió 401 con token correcto — el Depends no está funcionando"
    )


def test_admin_token_file_created_on_boot(tmp_path, monkeypatch):
    """_load_or_create_admin_token genera el archivo si no existe."""
    token_path = tmp_path / ".config" / "obsidian-rag" / "admin_token.txt"
    monkeypatch.setattr(_web_server, "_ADMIN_TOKEN_PATH", token_path)

    token = _web_server._load_or_create_admin_token()

    assert token_path.exists(), "El archivo de token no fue creado"
    assert token_path.read_text(encoding="utf-8").strip() == token
    assert len(token) >= 32, "Token demasiado corto"
    # Verificar permisos 0o600
    mode = oct(token_path.stat().st_mode)[-3:]
    assert mode == "600", f"Permisos incorrectos: {mode} (esperaba 600)"


def test_admin_token_reuses_existing_file(tmp_path, monkeypatch):
    """Si el archivo ya existe, _load_or_create_admin_token lo reuza."""
    token_path = tmp_path / ".config" / "obsidian-rag" / "admin_token.txt"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    existing_token = "my-existing-token-abc123xyz"
    token_path.write_text(existing_token + "\n")
    monkeypatch.setattr(_web_server, "_ADMIN_TOKEN_PATH", token_path)

    token = _web_server._load_or_create_admin_token()

    assert token == existing_token


def test_admin_token_bearer_scheme_required(client, valid_token):
    """Esquema Basic u otro → 401 (solo Bearer es válido)."""
    resp = client.post(
        "/api/reindex",
        json={},
        headers={"Authorization": f"Basic {valid_token}"},
    )
    assert resp.status_code == 401
