"""Tests for the /api/home progress streaming.

`/api/home/stream` is the SSE companion to `/api/home`. It runs the same
14-fetcher fan-out (`_home_compute`) and emits one `stage` event per
fetcher (start + done|error|timeout) plus a final `done` event with the
full payload. The home page wires this into the manual-refresh button so
the user can see in real time which fetcher is the bottleneck.

Coverage:
  1. `_home_compute(progress=cb)` calls `cb(stage, "start", 0.0, None)`
     and `cb(stage, "done", elapsed_ms, None)` for every fetcher.
  2. `cb(stage, "error", elapsed_ms, msg)` fires when a fetcher raises.
  3. `cb(stage, "timeout", budget_ms, None)` fires when a fetcher
     exceeds its per-stage budget (worker keeps running, main thread
     bails — no false "done").
  4. `/api/home/stream` SSE frame ordering: `hello` first (with the
     full stage list so the UI can paint placeholders), then a stream
     of `stage` events, then exactly one `done` with the payload.
  5. `hello` event lists every stage name in `STAGE_NAMES` (frontend
     contract — JS uses this to seed the chip strip even before the
     first fetcher reports back).
  6. The endpoint sets the right SSE headers (`text/event-stream`,
     `Cache-Control: no-cache`, `X-Accel-Buffering: no`) so reverse
     proxies don't buffer events into a single chunk at the end.
  7. Final `done` event carries the same payload shape as `/api/home`
     (today/signals/tomorrow_calendar/weather_forecast top-level keys).
  8. When `_home_compute` raises HTTPException (today_evidence
     mandatory abort), the stream emits a single `error` event and
     closes — no `done` event.

All tests stub the 14 fetchers so the suite stays fast (<2s) and never
touches ollama / sqlite / iCalBuddy / Gmail / weather APIs.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient


# ── SSE parsing helpers ────────────────────────────────────────────────────


_EVENT_RE = re.compile(r"event: (?P<event>[^\n]+)\ndata: (?P<data>[^\n]*)\n\n")


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for m in _EVENT_RE.finditer(body):
        try:
            payload = json.loads(m.group("data"))
        except Exception:
            payload = {}
        out.append((m.group("event"), payload))
    return out


# Names of every fetcher the SSE `hello` event must announce so the UI
# can paint placeholder chips immediately. Matches the order in
# `_home_compute`'s fan-out.
_EXPECTED_STAGES = [
    "today", "signals", "tomorrow", "forecast",
    "pagerank", "chrome", "eval", "followup",
    "drive", "wa_unreplied", "bookmarks", "vaults",
    "finance", "youtube",
]


# ── Stubbing fixture: replace every fetcher with a fast no-op ──────────────


@pytest.fixture
def stub_fetchers(monkeypatch):
    """Replace every fetcher inside `_home_compute` with a fast stub so
    the suite never touches real services. Each stub returns a minimal
    valid value (empty list / dict / None) matching the production
    contract. Returns a `dict` of monkeypatched call counters in case a
    test wants to assert which fetchers ran.
    """
    from web import server as server_mod

    calls: dict[str, int] = {name: 0 for name in _EXPECTED_STAGES}

    def _bump(name: str):
        def _stub(*a, **kw):
            calls[name] += 1
            # Tiny sleep so elapsed_ms > 0 — lets us assert the field.
            time.sleep(0.005)
            if name == "today":
                return {"recent_notes": [], "inbox_today": [], "todos": [],
                        "new_contradictions": [], "low_conf_queries": []}
            if name == "signals":
                return {}
            if name == "vaults":
                return {}
            return [] if name in {"tomorrow", "pagerank", "chrome", "drive",
                                  "wa_unreplied", "bookmarks", "youtube"} else None
        return _stub

    monkeypatch.setattr(server_mod, "_collect_today_evidence", _bump("today"))
    monkeypatch.setattr(server_mod, "_pendientes_collect", _bump("signals"))
    monkeypatch.setattr(server_mod, "_fetch_calendar_ahead", _bump("tomorrow"))
    monkeypatch.setattr(server_mod, "_fetch_weather_forecast", _bump("forecast"))
    monkeypatch.setattr(server_mod, "_fetch_pagerank_top", _bump("pagerank"))
    monkeypatch.setattr(server_mod, "_fetch_chrome_top_week", _bump("chrome"))
    monkeypatch.setattr(server_mod, "_fetch_eval_trend", _bump("eval"))
    monkeypatch.setattr(server_mod, "_fetch_followup_aging", _bump("followup"))
    monkeypatch.setattr(server_mod, "_fetch_drive_recent", _bump("drive"))
    monkeypatch.setattr(server_mod, "_fetch_whatsapp_unreplied", _bump("wa_unreplied"))
    monkeypatch.setattr(server_mod, "_fetch_chrome_bookmarks_used", _bump("bookmarks"))
    monkeypatch.setattr(server_mod, "_fetch_vault_activity", _bump("vaults"))
    monkeypatch.setattr(server_mod, "_fetch_finance", _bump("finance"))
    monkeypatch.setattr(server_mod, "_fetch_youtube_watched", _bump("youtube"))

    # Skip pendientes-urgent / narrative — not fetchers, but they read
    # signals + may call ollama. Stub to keep _home_compute pure.
    monkeypatch.setattr(server_mod, "_pendientes_urgent", lambda *a, **kw: [])
    monkeypatch.setattr(server_mod, "_today_cached_narrative", lambda d: "")

    # `get_db()` returns a sqlite-vec col; pass a dummy — every fetcher
    # consuming it is stubbed above so it's never dereferenced.
    monkeypatch.setattr(server_mod, "get_db", lambda: None)

    # Disable disk cache persistence so tests never write to user state.
    monkeypatch.setattr(server_mod, "_persist_home_cache", lambda body, ts: None)
    return calls


# ── Direct unit tests on _home_compute(progress=cb) ────────────────────────


def test_progress_callback_fires_start_and_done_for_every_stage(stub_fetchers):
    from web.server import _home_compute

    events: list[tuple[str, str]] = []

    def cb(stage, status, elapsed_ms, err):
        events.append((stage, status))

    payload = _home_compute(regenerate=False, progress=cb)
    assert isinstance(payload, dict)

    for name in _EXPECTED_STAGES:
        assert (name, "start") in events, f"missing start event for {name}: {events}"
        assert (name, "done") in events, f"missing done event for {name}: {events}"


def test_progress_done_carries_positive_elapsed_ms(stub_fetchers):
    from web.server import _home_compute

    elapsed_by_stage: dict[str, float] = {}

    def cb(stage, status, elapsed_ms, err):
        if status == "done":
            elapsed_by_stage[stage] = elapsed_ms

    _home_compute(progress=cb)
    # Every stub sleeps 5ms; allow generous floor to avoid flakiness.
    for name in _EXPECTED_STAGES:
        assert elapsed_by_stage.get(name, 0) >= 1.0, (
            f"elapsed_ms for {name} was {elapsed_by_stage.get(name)}")


def test_progress_callback_fires_error_when_fetcher_raises(monkeypatch, stub_fetchers):
    from web import server as server_mod
    from web.server import _home_compute

    def _boom(*a, **kw):
        raise RuntimeError("forecast api 503")

    monkeypatch.setattr(server_mod, "_fetch_weather_forecast", _boom)

    errors: list[tuple[str, str | None]] = []

    def cb(stage, status, elapsed_ms, err):
        if status == "error":
            errors.append((stage, err))

    _home_compute(progress=cb)
    assert ("forecast", "forecast api 503") in errors


def test_progress_callback_fires_timeout_when_fetcher_blows_budget(
    monkeypatch, stub_fetchers
):
    """A wedged fetcher must surface as `timeout` (not silently as
    "done") so the UI can label it skipped. We patch
    `concurrent.futures.Future.result` to force the first 5s-budget
    `result(timeout=)` call to raise — `_home_compute` should translate
    that into a timeout progress event for whichever stage drew the
    short straw (chrome/eval/finance/youtube/bookmarks all share a 5s
    budget; ordering varies by GIL / dispatcher).
    """
    from web.server import _home_compute

    import concurrent.futures as _cf

    real_result = _cf.Future.result
    fired = {"once": False}

    def fake_result(self, timeout=None):
        if timeout == 5 and not fired["once"]:
            fired["once"] = True
            raise _cf.TimeoutError()
        return real_result(self, timeout=timeout)

    monkeypatch.setattr(_cf.Future, "result", fake_result)

    timeouts: list[tuple[str, float]] = []

    def cb(stage, status, elapsed_ms, err):
        if status == "timeout":
            timeouts.append((stage, elapsed_ms))

    _home_compute(progress=cb)

    # Exactly one timeout, on whichever 5s-budget fetcher landed first.
    assert len(timeouts) == 1, timeouts
    name, elapsed = timeouts[0]
    assert name in {"chrome", "eval", "finance", "youtube", "bookmarks"}, name
    # Elapsed reported = budget * 1000ms = 5000ms (timeout doesn't
    # measure real elapsed; it reports the budget that was blown).
    assert elapsed == 5000.0


# ── /api/home/stream endpoint integration ─────────────────────────────────


def test_stream_endpoint_emits_hello_then_stages_then_done(stub_fetchers):
    from web.server import app

    with TestClient(app) as client:
        with client.stream("GET", "/api/home/stream?regenerate=false") as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = b"".join(resp.iter_bytes()).decode()

    events = _parse_sse(body)
    assert events, "no SSE events parsed"
    # First event = hello
    assert events[0][0] == "hello"
    assert set(events[0][1].get("stages", [])) == set(_EXPECTED_STAGES)
    # At least one stage event between hello and done
    kinds = [e[0] for e in events]
    assert "stage" in kinds
    # Exactly one done event, and it's the last one
    done_idx = [i for i, e in enumerate(events) if e[0] == "done"]
    assert len(done_idx) == 1
    assert done_idx[0] == len(events) - 1
    # Done payload looks like /api/home (top-level keys)
    done_payload = events[done_idx[0]][1]
    for key in ("today", "signals", "tomorrow_calendar", "weather_forecast"):
        assert key in done_payload, f"missing {key} in done payload"


def test_stream_endpoint_stage_events_have_required_shape(stub_fetchers):
    from web.server import app

    with TestClient(app) as client:
        with client.stream("GET", "/api/home/stream?regenerate=false") as resp:
            body = b"".join(resp.iter_bytes()).decode()

    events = _parse_sse(body)
    stage_events = [(name, payload) for name, payload in events if name == "stage"]
    assert stage_events, "no stage events emitted"
    for _, payload in stage_events:
        assert "stage" in payload
        assert "status" in payload
        assert payload["status"] in {"start", "done", "error", "timeout"}
        assert "elapsed_ms" in payload
        assert isinstance(payload["elapsed_ms"], (int, float))
        if payload["status"] == "start":
            assert payload["elapsed_ms"] == 0.0


def test_stream_endpoint_sets_anti_buffering_headers(stub_fetchers):
    from web.server import app

    with TestClient(app) as client:
        with client.stream("GET", "/api/home/stream?regenerate=false") as resp:
            assert resp.status_code == 200
            cache_ctrl = resp.headers.get("cache-control", "")
            assert "no-cache" in cache_ctrl, cache_ctrl
            assert resp.headers.get("x-accel-buffering") == "no"
            # Drain the body so the connection closes cleanly before
            # the context manager exits — otherwise httpx leaves the
            # generator half-consumed and the test client warns.
            b"".join(resp.iter_bytes())


def test_stream_endpoint_emits_error_on_today_evidence_failure(monkeypatch, stub_fetchers):
    from web import server as server_mod
    from web.server import app

    def _boom(*a, **kw):
        raise RuntimeError("vault unreadable")

    monkeypatch.setattr(server_mod, "_collect_today_evidence", _boom)

    with TestClient(app) as client:
        with client.stream("GET", "/api/home/stream?regenerate=false") as resp:
            body = b"".join(resp.iter_bytes()).decode()

    events = _parse_sse(body)
    kinds = [e[0] for e in events]
    assert "error" in kinds, kinds
    assert "done" not in kinds, "should not emit done after a hard error"


def test_hello_event_advertises_signals_substages(stub_fetchers):
    """The `hello` event must declare every sub-fetcher of `signals`
    so the UI can paint the nested chip strip from t=0. The 9 names
    match `_pendientes_collect`'s task dict 1:1."""
    from web.server import app

    expected_subs = {
        "signals.mail_unread", "signals.reminders", "signals.calendar",
        "signals.whatsapp", "signals.weather", "signals.gmail",
        "signals.loops", "signals.contradictions", "signals.low_conf",
    }
    with TestClient(app) as client:
        with client.stream("GET", "/api/home/stream?regenerate=false") as resp:
            body = b"".join(resp.iter_bytes()).decode()

    events = _parse_sse(body)
    hello = next(p for ev, p in events if ev == "hello")
    assert "substages" in hello, hello
    subs = set(hello["substages"].get("signals", []))
    assert subs == expected_subs, f"missing or extra: {subs ^ expected_subs}"


def test_pendientes_progress_callback_emits_substages(monkeypatch):
    """`_pendientes_collect(progress=cb)` must call cb start+done for
    each of the 9 inner fetchers so the SSE stream can surface them
    as `signals.<name>` sub-stages."""
    import rag

    expected = {
        "mail_unread", "reminders", "calendar", "whatsapp",
        "weather", "gmail", "loops", "contradictions", "low_conf",
    }

    # Stub each underlying fetcher to a no-op so we don't hit real
    # services. _pendientes_collect uses these via lambdas inside the
    # task dict — patching the rag.* names is enough.
    monkeypatch.setattr(rag, "_fetch_mail_unread", lambda: [])
    monkeypatch.setattr(rag, "_fetch_reminders_due",
                        lambda now, horizon_days=1, max_items=30: [])
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda: [])
    monkeypatch.setattr(rag, "_fetch_whatsapp_unread",
                        lambda hours, max_chats: [])
    monkeypatch.setattr(rag, "_fetch_weather_rain", lambda: None)
    monkeypatch.setattr(rag, "_fetch_gmail_evidence", lambda now: {})
    monkeypatch.setattr(rag, "_pendientes_extract_loops_fast",
                        lambda vault, days=14, max_items=40: [])
    monkeypatch.setattr(rag, "_pendientes_recent_contradictions",
                        lambda log, now, days=14, max_items=5: [])
    monkeypatch.setattr(rag, "_pendientes_low_conf_queries",
                        lambda log, now, days=14: [])

    seen_starts: set[str] = set()
    seen_dones: set[str] = set()

    def cb(stage, status, elapsed_ms, err):
        if status == "start":
            seen_starts.add(stage)
        elif status == "done":
            seen_dones.add(stage)

    from datetime import datetime
    rag._pendientes_collect(None, datetime.now(), 14, progress=cb)
    assert seen_starts == expected, f"missing starts: {expected - seen_starts}"
    assert seen_dones == expected, f"missing dones: {expected - seen_dones}"


def test_stream_done_persists_into_home_state_cache(stub_fetchers):
    """After a successful stream, `/api/home` (non-stream) should serve
    the same payload from cache without re-running the fan-out — the
    `done` handler in the stream generator hydrates `_HOME_STATE`.
    """
    from web import server as server_mod
    from web.server import app

    # Reset state so the test doesn't read stale cache from another test.
    server_mod._HOME_STATE["body"] = None
    server_mod._HOME_STATE["payload"] = None
    server_mod._HOME_STATE["ts"] = 0.0

    with TestClient(app) as client:
        with client.stream("GET", "/api/home/stream?regenerate=false") as resp:
            b"".join(resp.iter_bytes())

        # Now /api/home should hit the cache (age < SOFT_TTL).
        r2 = client.get("/api/home")
        assert r2.status_code == 200
        data = r2.json()
        assert "today" in data
        # Not the placeholder "warming" payload.
        assert data.get("warming") is not True
