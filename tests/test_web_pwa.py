"""Tests del wiring PWA del web server (2026-04-23).

Regresiones que atrapan:
  - Manifest accesible en root con MIME correcto → sin esto iOS/Chrome
    ignoran la PWA silenciosamente (no prompt de install, no splash).
  - SW accesible en root con `Service-Worker-Allowed: /` y `no-cache`
    → sin scope root el SW solo controla /static/** (inútil).
  - Los 3 HTML inyectan `<link rel="manifest">` y `register-sw.js` →
    si alguien edita un HTML y se olvida, la PWA queda parcial.
  - Los icons + splash generados existen en disco → un regen roto
    dejaría 404s que rompen el splash screen iOS.

No testeamos:
  - El JS del SW corre bien (eso es Playwright territory, ya verificado
    manualmente en dev).
  - Pixel-perfect match de los PNGs generados (irrelevante, el
    re-render es determinístico pero tiny diffs son OK).
"""
from __future__ import annotations

import pytest
import re
from pathlib import Path

from fastapi.testclient import TestClient

import web.server as _server


_STATIC_DIR = Path(_server.STATIC_DIR)
_client = TestClient(_server.app)


# ── Audit 2026-04-26 BUG #1 telemetry — DB_PATH isolation ────────────────
# Previene pollution de la prod telemetry.db cuando el TestClient ejercita
# endpoints que disparan log_query_event/semantic_cache_store/etc.
# Snap+restore manual (NO monkeypatch.setattr) — el conftest autouse
# `_stabilize_rag_state` corre teardown ANTES de monkeypatch y emite
# warning falso si está set. Mismo patrón que tests/test_rag_log_sql_read.py.
@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path):
    import rag as _rag_isolate
    _snap = _rag_isolate.DB_PATH
    _rag_isolate.DB_PATH = tmp_path / "ragvec"
    try:
        yield
    finally:
        _rag_isolate.DB_PATH = _snap

def test_manifest_endpoint_mime_and_body():
    """GET /manifest.webmanifest → 200 application/manifest+json con
    un manifest válido (name, start_url, display, icons)."""
    resp = _client.get("/manifest.webmanifest")
    assert resp.status_code == 200
    # El content-type DEBE ser application/manifest+json (no application/json)
    # para que Chrome/Safari lo traten como web manifest.
    assert resp.headers["content-type"].startswith("application/manifest+json")
    data = resp.json()
    assert data["name"] == "rag · obsidian-rag"
    assert data["short_name"] == "rag"
    assert data["start_url"] == "/chat"
    assert data["display"] == "standalone"
    assert data["scope"] == "/"
    assert data["theme_color"] == "#1a1a1f"
    # Al menos 2 icons para el manifest + 2 maskable (Android 12+).
    icons = data["icons"]
    assert len(icons) >= 4
    assert any(i.get("purpose") == "maskable" for i in icons)
    assert any(i.get("sizes") == "512x512" for i in icons)


def test_manifest_has_cache_control():
    """El manifest cacheado 1 día — cambia raro, no queremos que cada
    navegación vaya a la red."""
    resp = _client.get("/manifest.webmanifest")
    cc = resp.headers.get("cache-control", "")
    assert "max-age=86400" in cc


def test_service_worker_endpoint_headers():
    """GET /sw.js → 200 application/javascript con headers específicos
    de Service Worker (no-cache + Service-Worker-Allowed root scope)."""
    resp = _client.get("/sw.js")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/javascript")
    # Cache-Control no-cache fuerza revalidación ETag por navigation →
    # updates del SW se propagan rápido.
    assert "no-cache" in resp.headers.get("cache-control", "")
    # Service-Worker-Allowed: / permite al SW controlar todo el site,
    # no sólo su directorio de origen.
    assert resp.headers.get("service-worker-allowed") == "/"


def test_service_worker_body_declares_version_and_skip_waiting():
    """El SW tiene la version tag (para ver en DevTools) y skipWaiting
    (para que el update se active al toque)."""
    resp = _client.get("/sw.js")
    body = resp.text
    # La CACHE_VERSION es un identificador humano; cambiarla fuerza
    # el wipe de caches viejos en activate.
    assert re.search(r'const\s+CACHE_VERSION\s*=\s*"rag-pwa-', body)
    assert "skipWaiting" in body
    # El SW NO debe intentar cachear /api/* — eso rompería SSE.
    assert 'isApi' in body


def test_manifest_icons_files_exist():
    """Todos los icons referenciados en el manifest existen en disco."""
    resp = _client.get("/manifest.webmanifest")
    data = resp.json()
    for icon in data["icons"]:
        src = icon["src"]  # ej: /static/pwa/icon-192.png
        assert src.startswith("/static/")
        path = _STATIC_DIR / src[len("/static/"):]
        assert path.exists(), f"icon missing on disk: {path}"
        assert path.stat().st_size > 100, f"icon too small: {path}"


def test_apple_splash_files_exist():
    """Los 10 splash screens iPhone deben existir después de haber
    corrido `scripts/gen_pwa_assets.py`."""
    expected = [
        "splash-iphone-x.png",
        "splash-iphone-xr.png",
        "splash-iphone-xs-max.png",
        "splash-iphone-12-mini.png",
        "splash-iphone-12.png",
        "splash-iphone-12-pro-max.png",
        "splash-iphone-14-pro.png",
        "splash-iphone-14-pro-max.png",
        "splash-iphone-16-pro.png",
        "splash-iphone-16-pro-max.png",
    ]
    pwa_dir = _STATIC_DIR / "pwa"
    for name in expected:
        path = pwa_dir / name
        assert path.exists(), f"splash missing on disk: {path}"
        # Los PNGs son ~18-33KB; < 1KB sería un placeholder/error.
        assert path.stat().st_size > 1024, f"splash too small: {path}"


def test_all_three_html_pages_reference_manifest():
    """home.html, index.html (chat) y dashboard.html DEBEN linkear el
    manifest, apple-touch-icon, y el register-sw script. Si alguien
    edita uno de los HTML y se olvida, queda con PWA parcial."""
    pages = {
        "/": "home.html",
        "/chat": "index.html",
        "/dashboard": "dashboard.html",
    }
    for route, _ in pages.items():
        resp = _client.get(route)
        assert resp.status_code == 200, f"{route} returned {resp.status_code}"
        html = resp.text
        assert '<link rel="manifest"' in html, f"{route}: no manifest link"
        assert "apple-touch-icon" in html, f"{route}: no apple-touch-icon"
        assert "register-sw.js" in html, f"{route}: no SW register script"
        # viewport-fit=cover es critical para safe-area-inset-*.
        assert "viewport-fit=cover" in html, f"{route}: missing viewport-fit"
        assert "apple-mobile-web-app-capable" in html, (
            f"{route}: missing iOS standalone meta"
        )
        # Contamos los 10 splash screens.
        splash_count = html.count("apple-touch-startup-image")
        assert splash_count >= 10, (
            f"{route}: only {splash_count}/10 splash screens"
        )


def test_register_sw_js_served_via_static():
    """El register-sw.js vive en /static/pwa/ (cachado 1h) — no es un
    SW así que no necesita no-cache."""
    resp = _client.get("/static/pwa/register-sw.js")
    assert resp.status_code == 200
    assert "serviceWorker" in resp.text
    assert "register" in resp.text


def test_manifest_shortcuts_point_to_valid_routes():
    """Los `shortcuts` del manifest (ej. long-press en el icon en Android)
    apuntan a rutas reales que el server sirve."""
    resp = _client.get("/manifest.webmanifest")
    data = resp.json()
    shortcuts = data.get("shortcuts", [])
    assert len(shortcuts) >= 1
    valid_routes = {"/", "/chat", "/dashboard", "/learning"}
    for sc in shortcuts:
        assert sc["url"] in valid_routes, f"shortcut url {sc['url']} not routable"


def test_api_chat_get_redirects_to_chat_ui():
    """Regresión 2026-04-24 (Fer F.): el user tenía bookmark a
    `/api/chat` (la ruta del endpoint, no del UI). FastAPI devolvía
    `{"detail":"Method Not Allowed"}` porque el endpoint solo acepta POST.
    ~256 405s observadas en `web.log` antes del fix. Ahora GET devuelve
    un redirect 307 a `/chat` para que el browser muestre el UI.
    """
    # `follow_redirects=False` para que podamos assertir el 307 literal.
    resp = _client.get("/api/chat", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/chat"


def test_api_chat_still_rejects_other_methods():
    """Sanity: PUT/DELETE siguen cayendo con 405 (no queremos que el
    redirect los ataje también — si un cliente programático hace un
    verb raro es bug del cliente y queremos que se rompa visible)."""
    resp = _client.put("/api/chat", json={})
    assert resp.status_code == 405
    resp = _client.delete("/api/chat")
    assert resp.status_code == 405
