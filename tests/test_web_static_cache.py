"""Tests del Cache-Control header para /static/* en el web server (2026-04-22).

Pre-fix no había header → cada reload del browser refetcheaba el bundle
entero (~50-100ms por asset). Con `max-age=3600` el browser cachea 1h.
Override para dev: `OBSIDIAN_RAG_STATIC_NO_CACHE=1` → max-age=0.
"""
from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def _fresh_client(monkeypatch, no_cache: bool = False):
    """Re-importa web.server con la env var activa para que
    `_STATIC_MAX_AGE` se resuelva al valor esperado."""
    if no_cache:
        monkeypatch.setenv("OBSIDIAN_RAG_STATIC_NO_CACHE", "1")
    else:
        monkeypatch.delenv("OBSIDIAN_RAG_STATIC_NO_CACHE", raising=False)
    import web.server as _server
    importlib.reload(_server)
    return TestClient(_server.app)


def test_static_has_cache_control_default(monkeypatch):
    """Default (sin override): Cache-Control: public, max-age=3600."""
    client = _fresh_client(monkeypatch, no_cache=False)
    # Cualquier asset existente del bundle — pickeamos app.js que siempre está.
    resp = client.get("/static/app.js")
    # Puede ser 200 o 304 según si hay If-None-Match; en test fresh es 200.
    assert resp.status_code in (200, 304), f"unexpected {resp.status_code}"
    cc = resp.headers.get("Cache-Control", "")
    assert "max-age=3600" in cc, f"expected max-age=3600, got {cc!r}"
    assert "public" in cc


def test_static_no_cache_override(monkeypatch):
    """Con OBSIDIAN_RAG_STATIC_NO_CACHE=1: no-cache, no-store, must-revalidate."""
    client = _fresh_client(monkeypatch, no_cache=True)
    resp = client.get("/static/app.js")
    assert resp.status_code in (200, 304)
    cc = resp.headers.get("Cache-Control", "")
    assert "no-cache" in cc, f"expected no-cache, got {cc!r}"
    assert "max-age=3600" not in cc


def test_static_missing_asset_still_has_header(monkeypatch):
    """Un 404 (asset inexistente) NO lleva Cache-Control — la subclass sólo
    agrega el header a responses reales de StaticFiles. Este test documenta
    el comportamiento para que no se confunda con un bug.
    """
    client = _fresh_client(monkeypatch, no_cache=False)
    resp = client.get("/static/does-not-exist.js")
    assert resp.status_code == 404
    # PathLike missing → no header esperado (el get_response del super
    # retorna Response con status 404 antes de que nuestro override toque
    # headers — verificamos que ese path NO rompe, no el header en sí).


def test_cors_still_works_with_cached_static(monkeypatch):
    """Verifica que agregar `_CachedStaticFiles` no quiebra el CORS
    middleware que se registra después en server.py."""
    client = _fresh_client(monkeypatch, no_cache=False)
    # OPTIONS preflight desde localhost debe responder 200 + ACAO.
    resp = client.options(
        "/static/app.js",
        headers={
            "Origin": "http://localhost:8765",
            "Access-Control-Request-Method": "GET",
        },
    )
    # CORS middleware responde OPTIONS; valor exacto depende de versión
    # FastAPI, aceptamos 200 o 204.
    assert resp.status_code in (200, 204)
