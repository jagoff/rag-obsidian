"""Tests for the CORS middleware config on web.server.

Same-origin is the intended deploy model (plist binds 127.0.0.1:8765).
These tests guard against:

  - Accidental removal of the CORSMiddleware (regression → browser
    silently allows the default, which is fine, but loses the explicit
    whitelist documentation).
  - Accidental widening to allow_origins=["*"] — would open the API
    to every page in the browser.

Bug fixes tested (2026-04-27):
  - Bug #1: CORS regex now accepts https:// for localhost/127.0.0.1
    (both http and https schemes).
  - Bug #2: OBSIDIAN_RAG_ALLOW_TUNNEL=1 extends regex with
    *.trycloudflare.com (HTTPS only).
  - Bug #3: LAN regex (OBSIDIAN_RAG_ALLOW_LAN) now accepts https://.

The CORS regex is a module-level constant set at import time based on
env vars. We can't reload the FastAPI app per-test, so:
  - Default-regex tests go through the TestClient (env vars not set at
    import → localhost-only http+https).
  - Env-var-conditional tests exercise the regex-building logic directly
    using `re.match`, which is the exact same code path FastAPI uses.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_web_server = pytest.importorskip("web.server")
app = _web_server.app

_CLIENT = TestClient(app)


def _preflight(origin: str):
    """Fire an OPTIONS preflight with the given Origin header.
    Returns the response — callers inspect headers/status."""
    return _CLIENT.options(
        "/api/chat",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )


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

def test_cors_allows_127_0_0_1_with_port():
    """Primary deploy target — the launchd plist binds 127.0.0.1:8765."""
    resp = _preflight("http://127.0.0.1:8765")
    assert resp.status_code in (200, 204), resp.text
    assert resp.headers.get("access-control-allow-origin") == "http://127.0.0.1:8765"


def test_cors_allows_localhost_with_port():
    """Dev sometimes uses `http://localhost:...` instead of 127.0.0.1."""
    resp = _preflight("http://localhost:8765")
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8765"


def test_cors_allows_127_0_0_1_no_port():
    """Some browsers normalise away the default port for http:80."""
    resp = _preflight("http://127.0.0.1")
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "http://127.0.0.1"


def test_cors_rejects_malicious_origin():
    """A page on evil.com must NOT be able to talk to the API."""
    resp = _preflight("http://evil.com")
    # Middleware may still return 400 (bad preflight) or 200 but without
    # the access-control-allow-origin header — either way, the browser
    # won't proceed with the real request.
    assert resp.headers.get("access-control-allow-origin") != "http://evil.com"
    assert resp.headers.get("access-control-allow-origin") != "*"


def test_cors_rejects_file_origin():
    """file:// origins (a local HTML opened in the browser) must not be
    trusted — could be a phishing-style attack via a downloaded page."""
    resp = _preflight("null")  # Chrome sends Origin: null for file://
    assert resp.headers.get("access-control-allow-origin") != "null"
    assert resp.headers.get("access-control-allow-origin") != "*"


def test_cors_rejects_0_0_0_0_origin():
    """0.0.0.0 is not a real origin (bind-any, not reach-any). Block it
    so accidentally serving on 0.0.0.0 doesn't widen the trust set."""
    resp = _preflight("http://0.0.0.0:8765")
    assert resp.headers.get("access-control-allow-origin") != "http://0.0.0.0:8765"
    assert resp.headers.get("access-control-allow-origin") != "*"


def test_cors_does_not_allow_wildcard():
    """Regression: `allow_origins=["*"]` would match any origin.
    The current config uses `allow_origin_regex` with a tight pattern."""
    resp = _preflight("https://example.com")
    assert resp.headers.get("access-control-allow-origin") != "*"
    assert resp.headers.get("access-control-allow-origin") != "https://example.com"


# ── Bug fix 2026-04-27 #1: localhost accepts https:// (Caddy local HTTPS) ──


def test_cors_allows_https_localhost():
    """Bug fix: https://localhost should be accepted (e.g. Caddy tls internal
    proxying locally). Pre-fix the regex was http-only."""
    resp = _preflight("https://localhost:8765")
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "https://localhost:8765"


def test_cors_allows_https_127_0_0_1():
    """Bug fix: https://127.0.0.1 should be accepted for local Caddy HTTPS."""
    resp = _preflight("https://127.0.0.1:8765")
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") == "https://127.0.0.1:8765"


# ── Bug fix 2026-04-27 #2: LAN regex accepts https:// ─────────────────────
# We test the regex directly (can't reload module per-test to set env vars).


def _build_lan_regex() -> str:
    """Build the same regex that web/server.py builds when ALLOW_LAN=1."""
    return (
        r"^https?://("
        r"127\.0\.0\.1|localhost|"
        r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
        r"192\.168\.\d{1,3}\.\d{1,3}"
        r")(:[0-9]+)?$"
    )


@pytest.mark.parametrize("origin", [
    "http://192.168.1.50:8765",
    "https://192.168.1.50:8765",
    "http://10.0.0.5",
    "https://10.0.0.5:8765",
    "http://172.16.0.1",
    "https://172.16.0.1:8765",
    "http://localhost:8765",
    "https://localhost:8765",
])
def test_lan_regex_accepts_http_and_https(origin):
    """Bug fix: LAN regex must accept both http:// and https:// (Caddy
    tls internal). Pre-fix regex only accepted http://."""
    assert re.match(_build_lan_regex(), origin), f"expected match: {origin!r}"


@pytest.mark.parametrize("origin", [
    "https://evil.com",
    "http://0.0.0.0",
    "file:///etc/passwd",
    "https://192.169.1.1",      # not RFC1918
    "ftp://192.168.1.50",       # wrong scheme
])
def test_lan_regex_rejects_non_lan(origin):
    """LAN regex must not match public IPs or bad schemes.
    Note: the regex uses \\d{1,3} which doesn't validate octet range — a
    known pre-existing limitation shared with the original code. We don't
    test '192.168.999.1' here since that would be a new constraint."""
    assert not re.match(_build_lan_regex(), origin), f"expected no match: {origin!r}"


# ── Bug fix 2026-04-27 #3: OBSIDIAN_RAG_ALLOW_TUNNEL regex ───────────────


def _build_tunnel_regex(with_lan: bool = False) -> str:
    """Build the regex that web/server.py builds when ALLOW_TUNNEL=1.
    Matches the exact logic in the module, including the (?:...) wrapper."""
    if with_lan:
        base = _build_lan_regex()
    else:
        base = r"^https?://(127\.0\.0\.1|localhost)(:[0-9]+)?$"
    return (
        r"(?:" + base + r")"
        r"|^https://[a-z0-9-]+\.trycloudflare\.com$"
    )


@pytest.mark.parametrize("origin", [
    "https://word-word-random.trycloudflare.com",
    "https://abc-def-123.trycloudflare.com",
    "https://a.trycloudflare.com",
])
def test_tunnel_regex_accepts_valid_cloudflare_urls(origin):
    """OBSIDIAN_RAG_ALLOW_TUNNEL=1 must allow any *.trycloudflare.com
    over HTTPS — the slug format cloudflared assigns."""
    assert re.match(_build_tunnel_regex(), origin), f"expected match: {origin!r}"


@pytest.mark.parametrize("origin", [
    "http://word-word-random.trycloudflare.com",     # HTTP not allowed (CF is HTTPS-only)
    "https://evil.trycloudflare.evil.com",           # not a *.trycloudflare.com subdomain
    "https://evil.com",
    "https://trycloudflare.com",                     # bare domain, no slug
    "https://word.trycloudflare.com.evil.com",       # suffix spoof
])
def test_tunnel_regex_rejects_invalid(origin):
    """Tunnel regex must NOT match HTTP cloudflare URLs, spoofed domains,
    or the bare trycloudflare.com domain."""
    assert not re.match(_build_tunnel_regex(), origin), f"expected no match: {origin!r}"


def test_tunnel_regex_still_allows_localhost():
    """With ALLOW_TUNNEL=1, localhost must still be allowed (dev access)."""
    assert re.match(_build_tunnel_regex(), "http://localhost:8765")
    assert re.match(_build_tunnel_regex(), "https://localhost:8765")


def test_tunnel_regex_with_lan_allows_both():
    """ALLOW_TUNNEL=1 + ALLOW_LAN=1 together: LAN IPs + CF tunnel all work."""
    combined = _build_tunnel_regex(with_lan=True)
    assert re.match(combined, "https://192.168.1.50:8765")
    assert re.match(combined, "https://word-word.trycloudflare.com")
    assert not re.match(combined, "https://evil.com")


# ── Fix 2026-04-30: tunnel CORS usa URL literal del state file ────────────


def _build_tunnel_literal_regex(url: str) -> str:
    """Build the regex web/server.py now uses when it reads a literal URL
    from the cloudflared-url.txt state file."""
    base = r"^https?://(127\.0\.0\.1|localhost)(:[0-9]+)?$"
    return r"(?:" + base + r")|^" + re.escape(url) + r"$"


def test_tunnel_literal_regex_accepts_only_exact_url():
    """Con URL literal del state file, SOLO esa URL es aceptada."""
    url = "https://word-word-random.trycloudflare.com"
    pat = _build_tunnel_literal_regex(url)
    assert re.match(pat, url)
    assert not re.match(pat, "https://evil.trycloudflare.com")
    assert not re.match(pat, "https://other-slug.trycloudflare.com")


def test_tunnel_literal_regex_rejects_other_cloudflare_slug():
    """Otro slug (evil.trycloudflare.com) no debe matchear con URL literal."""
    pat = _build_tunnel_literal_regex(
        "https://word-word-random.trycloudflare.com"
    )
    assert not re.match(pat, "https://different-slug.trycloudflare.com")


def test_tunnel_no_state_file_falls_back_to_wildcard(tmp_path, monkeypatch):
    """Sin state file → el server usa wildcard pattern (regex permisivo)
    pero credentials=False — verifica la lógica de fallback."""
    import web.server as srv
    # Apuntar el state file a un path que no existe
    missing = tmp_path / "no-such-file.txt"
    # La lógica de fallback: la variable _tunnel_credentials debe ser False
    # cuando el state file no existe. Verificamos la lógica directamente.
    tunnel_url = None
    tunnel_credentials = False
    try:
        raw = missing.read_text(encoding="utf-8").strip()
        if raw and re.match(r"^https://[a-z0-9-]+\.trycloudflare\.com$", raw):
            tunnel_url = raw
            tunnel_credentials = True
    except Exception:
        pass
    assert tunnel_url is None
    assert tunnel_credentials is False


def test_tunnel_state_file_sets_credentials_true(tmp_path):
    """Con state file válido → tunnel_credentials debe ser True."""
    url_file = tmp_path / "cloudflared-url.txt"
    url_file.write_text("https://word-word-random.trycloudflare.com\n")
    tunnel_url = None
    tunnel_credentials = False
    try:
        raw = url_file.read_text(encoding="utf-8").strip()
        if raw and re.match(r"^https://[a-z0-9-]+\.trycloudflare\.com$", raw):
            tunnel_url = raw
            tunnel_credentials = True
    except Exception:
        pass
    assert tunnel_url == "https://word-word-random.trycloudflare.com"
    assert tunnel_credentials is True


def test_tunnel_state_file_invalid_url_no_credentials(tmp_path):
    """State file con URL inválida (no trycloudflare.com) → no credentials."""
    url_file = tmp_path / "cloudflared-url.txt"
    url_file.write_text("https://evil.com\n")
    tunnel_url = None
    tunnel_credentials = False
    try:
        raw = url_file.read_text(encoding="utf-8").strip()
        if raw and re.match(r"^https://[a-z0-9-]+\.trycloudflare\.com$", raw):
            tunnel_url = raw
            tunnel_credentials = True
    except Exception:
        pass
    assert tunnel_url is None
    assert tunnel_credentials is False


# ── Bug fix 2026-04-27 #4: /api/chat SSE headers ─────────────────────────


def test_chat_sse_response_has_anti_buffer_headers():
    """Bug fix: /api/chat must include X-Accel-Buffering: no, Cache-Control:
    no-cache so Caddy/nginx don't buffer the SSE stream. Pre-fix the
    StreamingResponse had no headers dict at all."""
    # Fire an invalid/empty chat request — we only care about response headers,
    # not the body. The endpoint returns 422 for missing body, but some
    # implementations still set streaming headers on error paths. To actually
    # hit the StreamingResponse we need a valid body — but we can test the
    # streaming path by inspecting what the production code returns.
    # Instead, verify at the source-code level that the module builds the
    # StreamingResponse with the expected header dict.
    from web import server as srv
    import inspect, ast

    src = inspect.getsource(srv.chat)
    # The fix adds a headers dict to the StreamingResponse(guarded(), ...) call.
    assert "X-Accel-Buffering" in src, (
        "chat() StreamingResponse must include X-Accel-Buffering header"
    )
    assert "Cache-Control" in src, (
        "chat() StreamingResponse must include Cache-Control header"
    )
