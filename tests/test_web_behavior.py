"""Tests for POST /api/behavior endpoint (ranker-vivo task 4).

Strategy: build a minimal FastAPI test app that mirrors the /api/behavior
handler logic exactly, with rag symbols substituted by test stubs. This avoids
importing the full server.py (which pulls in chromadb/ollama/sentence-trans).
"""
from __future__ import annotations

import collections
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel


# ── Test-scoped behavior log writer ──────────────────────────────────────────

_BEHAVIOR_LOCK = threading.Lock()
_BEHAVIOR_PATH: Path | None = None  # set per-test via fixture


def _log_behavior_event_stub(event: dict) -> None:
    if not event or _BEHAVIOR_PATH is None:
        return
    ev = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
    line = json.dumps(ev, ensure_ascii=False) + "\n"
    with _BEHAVIOR_LOCK:
        _BEHAVIOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_BEHAVIOR_PATH, "a", encoding="utf-8") as f:
            f.write(line)


# ── Minimal FastAPI app mirroring the /api/behavior route ────────────────────

_BEHAVIOR_BUCKETS: dict[str, list[float]] = collections.defaultdict(list)
_RATE_LIMIT = 120
_RATE_WINDOW = 60.0
_KNOWN_EVENTS = frozenset({
    "open", "open_external", "positive_implicit",
    "negative_implicit", "kept", "deleted", "save",
})
_SESSION_RE = re.compile(r"^[A-Za-z0-9_.@:-]{1,80}$")


class BehaviorRequest(BaseModel):
    source: str
    event: str
    query: str | None = None
    path: str | None = None
    rank: int | None = None
    dwell_ms: int | None = None
    session: str | None = None


test_app = FastAPI()


@test_app.post("/api/behavior")
def submit_behavior(req: BehaviorRequest, request: Request) -> dict:
    """Replica of the production /api/behavior handler for test isolation."""
    if req.source != "web":
        raise HTTPException(status_code=400, detail="source must be 'web'")
    if req.event not in _KNOWN_EVENTS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown event '{req.event}'; valid: {sorted(_KNOWN_EVENTS)}",
        )
    if req.session is not None and not _SESSION_RE.match(req.session):
        raise HTTPException(status_code=400, detail="session id format invalid")
    if req.path is not None:
        p = req.path
        if p.startswith("/") or ".." in p.split("/"):
            raise HTTPException(status_code=400, detail="path must be vault-relative")
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _BEHAVIOR_BUCKETS[client_ip]
    cutoff = now - _RATE_WINDOW
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= _RATE_LIMIT:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    bucket.append(now)
    try:
        _log_behavior_event_stub({
            "source": req.source,
            "event": req.event,
            "query": req.query,
            "path": req.path,
            "rank": req.rank,
            "dwell_ms": req.dwell_ms,
            "session": req.session,
        })
    except Exception as exc:
        print(f"[behavior] write error: {exc}", flush=True)
        raise HTTPException(status_code=503, detail="event log unavailable")
    return {"ok": True}


_CLIENT = TestClient(test_app, raise_server_exceptions=True)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def behavior_path(tmp_path):
    global _BEHAVIOR_PATH
    p = tmp_path / "behavior.jsonl"
    _BEHAVIOR_PATH = p
    yield p
    _BEHAVIOR_PATH = None
    if p.exists():
        p.unlink()


@pytest.fixture(autouse=True)
def clear_buckets():
    _BEHAVIOR_BUCKETS.clear()
    yield
    _BEHAVIOR_BUCKETS.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def post_behavior(payload):
    return _CLIENT.post("/api/behavior", json=payload)


VALID = {
    "source": "web",
    "event": "open",
    "path": "01-Projects/foo.md",
    "rank": 1,
    "session": "web:aabbccdd1122",
}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_valid_payload_returns_200_and_logs(behavior_path):
    resp = post_behavior(VALID)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert behavior_path.exists()
    ev = json.loads(behavior_path.read_text().strip())
    assert ev["event"] == "open"
    assert ev["source"] == "web"
    assert ev["path"] == "01-Projects/foo.md"
    assert ev["rank"] == 1
    assert "ts" in ev


def test_source_not_web_returns_400():
    resp = post_behavior({**VALID, "source": "cli"})
    assert resp.status_code == 400
    assert "source" in resp.json()["detail"].lower()


def test_path_traversal_dotdot_returns_400():
    resp = post_behavior({**VALID, "path": "../../../etc/passwd"})
    assert resp.status_code == 400


def test_path_absolute_returns_400():
    resp = post_behavior({**VALID, "path": "/etc/passwd"})
    assert resp.status_code == 400


def test_unknown_event_returns_400():
    resp = post_behavior({**VALID, "event": "hack"})
    assert resp.status_code == 400
    assert "event" in resp.json()["detail"].lower()


def test_invalid_session_returns_400():
    resp = post_behavior({**VALID, "session": "bad session!with spaces"})
    assert resp.status_code == 400
    assert "session" in resp.json()["detail"].lower()


def test_optional_fields_absent_accepted(behavior_path):
    """path, session, rank, query, dwell_ms are all optional."""
    resp = post_behavior({"source": "web", "event": "save"})
    assert resp.status_code == 200
    ev = json.loads(behavior_path.read_text().strip())
    assert ev["event"] == "save"


def test_all_known_events_accepted(behavior_path):
    known = ["open", "open_external", "positive_implicit",
             "negative_implicit", "kept", "deleted", "save"]
    for ev_name in known:
        behavior_path.unlink(missing_ok=True)
        resp = post_behavior({"source": "web", "event": ev_name})
        assert resp.status_code == 200, f"event '{ev_name}' rejected"


def test_rate_limit_rejects_after_120_requests():
    """The 121st request from the same IP within 60s must get 429."""
    for i in range(120):
        resp = post_behavior({"source": "web", "event": "open"})
        assert resp.status_code == 200, f"request {i + 1} unexpectedly rejected"

    resp = post_behavior({"source": "web", "event": "open"})
    assert resp.status_code == 429


def test_rate_limit_window_resets():
    """Requests with all bucket entries older than 60s are accepted."""
    old_ts = time.time() - 61.0
    _BEHAVIOR_BUCKETS["testclient"] = [old_ts] * _RATE_LIMIT

    resp = post_behavior({"source": "web", "event": "open"})
    assert resp.status_code == 200
