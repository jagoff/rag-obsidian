"""Tests for the CORS middleware config on web.server.

Same-origin is the intended deploy model (plist binds 127.0.0.1:8765).
These tests guard against:

  - Accidental removal of the CORSMiddleware (regression → browser
    silently allows the default, which is fine, but loses the explicit
    whitelist documentation).
  - Accidental widening to allow_origins=["*"] — would open the API
    to every page in the browser.
"""
from __future__ import annotations

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
