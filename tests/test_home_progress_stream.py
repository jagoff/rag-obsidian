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

import time

import pytest
from fastapi.testclient import TestClient

# SSE parsing helpers vienen de conftest (consolidado 2026-04-25 — antes estos
# mismos 10 LOC estaban duplicados en 4 archivos web_chat + este).
from tests.conftest import _parse_sse  # noqa: F401 — usado en test bodies


# Names of every fetcher the SSE `hello` event must announce so the UI
# can paint placeholder chips immediately. Matches the order in
# `_home_compute`'s fan-out.
_EXPECTED_STAGES = [
    "today", "signals", "tomorrow", "forecast",
    "pagerank", "chrome", "eval", "followup",
    "drive", "wa_unreplied", "bookmarks", "vaults",
    "finance", "cards", "youtube",
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
    monkeypatch.setattr(server_mod, "_fetch_credit_cards", _bump("cards"))
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


@pytest.mark.slow
def test_progress_callback_fires_timeout_when_fetcher_blows_budget(
    monkeypatch, stub_fetchers
):
    """A wedged fetcher must surface as `timeout` (not silently as
    "done") so the UI can label it skipped. We patch
    `concurrent.futures.Future.result` to force the first 5s-budget
    `result(timeout=)` call to raise — `_home_compute` should translate
    that into a timeout progress event for whichever stage drew the
    short straw (chrome/eval/finance/cards/youtube/bookmarks all share
    a 5s budget; ordering varies by GIL / dispatcher).
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
    assert name in {"chrome", "eval", "finance", "cards", "youtube", "bookmarks"}, name
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


@pytest.mark.slow
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


# ── HARD_CAP_S regression guard ────────────────────────────────────────────
#
# El cap del SSE stream limita cuánto tiempo el server espera por un
# `done` antes de bailar con `event: error`. Bajado de 90s → 30s en
# 2026-04-24 — el cap original sumaba budgets per-fetcher (today=30 +
# signals=45 + cushion=15) pero esos corren en paralelo. P95 real del
# `_home_compute` es ~15s, peor outlier en 7d fue 21.5s. 30s da ~40%
# margen sobre ese outlier.
#
# Este assertion fija el valor: un futuro bump tendría que actualizar
# el test deliberadamente (con justificación documentada). Sin este
# guard, alguien podría re-bumpear el cap a 60s/90s sin notar que
# recursos del server quedan tied up por minutos en wedges.


def test_hard_cap_s_is_30_seconds_or_less():
    """SSE stream cap stays ≤ 30s. Si necesitás más, abrir issue + medir
    P95 real de `_home_compute` antes de subir.

    Implementación: el cap es local al `_gen()` async generator, no una
    constante module-level. Lo extraemos via source-grep de la asignación
    `HARD_CAP_S = N.0` en `web/server.py` — match laxo (admite spaces o
    tipo float/int) pero específico al token name.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "web" / "server.py"
    body = src.read_text(encoding="utf-8")
    matches = re.findall(r"HARD_CAP_S\s*=\s*([\d.]+)", body)
    assert matches, "HARD_CAP_S assignment not found in web/server.py"
    # Take the FIRST match (the in-function definition; any other refs
    # are reads). All assignments must be ≤ 30.
    for raw in matches:
        val = float(raw)
        assert val <= 30.0, (
            f"HARD_CAP_S = {val}s exceeds the 30s ceiling. If you really "
            f"need more, document the regression in the test + commit."
        )


# ── Degraded event ─────────────────────────────────────────────────────────
#
# Probes a la unidad — lógica del threshold + diagnóstico — y al
# end-to-end (`event: degraded` aparece en el stream cuando elapsed
# pasa el umbral). El end-to-end usa un fixture que fuerza un fetcher
# lento + monkeypatchea el threshold floor.


def test_record_home_compute_total_drops_outliers():
    """Outliers obvios (≤0 o >600s) no pueden contaminar la rolling
    window — esos vendrían de bugs de medición, no de runs reales."""
    from web import server as server_mod

    # Reset window for isolation (other tests may have populated it).
    server_mod._HOME_COMPUTE_HISTORY.clear()
    server_mod._record_home_compute_total(0)
    server_mod._record_home_compute_total(-5)
    server_mod._record_home_compute_total(700)
    server_mod._record_home_compute_total(8.5)
    assert list(server_mod._HOME_COMPUTE_HISTORY) == [8.5]


def test_record_home_compute_total_caps_at_window_max():
    """La rolling window no crece sin límite — se queda en
    `_HOME_COMPUTE_HISTORY_MAX`, oldest-first eviction."""
    from web import server as server_mod

    server_mod._HOME_COMPUTE_HISTORY.clear()
    cap = server_mod._HOME_COMPUTE_HISTORY_MAX
    for i in range(cap + 5):
        server_mod._record_home_compute_total(float(i + 1))
    snap = list(server_mod._HOME_COMPUTE_HISTORY)
    assert len(snap) == cap
    # Should retain the LAST `cap` entries (newest values).
    assert snap[0] == float(6)  # first 5 evicted
    assert snap[-1] == float(cap + 5)


def test_degraded_threshold_floors_when_history_thin():
    """Con <3 samples, devolvemos el floor (8s) — no arriesgamos
    fire-on-startup cuando sólo tenemos 1-2 mediciones que pueden
    venir de un disk cache hit (~1.5s) que daría threshold absurdo."""
    from web import server as server_mod

    server_mod._HOME_COMPUTE_HISTORY.clear()
    assert server_mod._home_compute_degraded_threshold() == server_mod._HOME_COMPUTE_DEGRADED_FLOOR

    server_mod._record_home_compute_total(2.0)
    server_mod._record_home_compute_total(2.5)
    # Still <3 → floor.
    assert server_mod._home_compute_degraded_threshold() == server_mod._HOME_COMPUTE_DEGRADED_FLOOR


def test_degraded_threshold_uses_2x_median_when_window_warm():
    """Con suficiente data, threshold = max(floor, median × 2). Median
    > mean cuando hay un outlier alto — más robusto a un wedge único
    que no debería disparar el floor del threshold para todas las
    futuras corridas."""
    from web import server as server_mod

    server_mod._HOME_COMPUTE_HISTORY.clear()
    # 9 samples around 6s + 1 anomalous 60s outlier
    for v in [5, 5.5, 6, 6, 6.2, 6.5, 7, 7.5, 8, 60]:
        server_mod._record_home_compute_total(v)
    threshold = server_mod._home_compute_degraded_threshold()
    # Median = 6.5 (5th value when sorted). Threshold = max(8, 13) = 13.
    assert threshold == 13.0


def test_diagnose_home_slowdown_shape(monkeypatch):
    """`_diagnose_home_slowdown` returns a stable dict shape. Stub the
    HTTP probe so we don't actually hit ollama (CI-safe + fast)."""
    from web import server as server_mod
    import urllib.request

    class _StubResp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b'{"models":[]}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _StubResp())
    diag = server_mod._diagnose_home_slowdown()
    assert "cause" in diag
    assert "details" in diag
    assert isinstance(diag["details"], dict)
    assert "ollama_state" in diag["details"]
    assert "ollama_ms" in diag["details"]
    # All-OK probe → cause is "unknown" or "memory_pressure" depending
    # on the actual machine state, NEVER an ollama_* tag.
    assert not diag["cause"].startswith("ollama_")


def test_diagnose_home_slowdown_detects_unreachable_ollama(monkeypatch):
    """Cuando urllib tira URLError, `cause` debe ser `ollama_unreachable`."""
    from web import server as server_mod
    import urllib.error
    import urllib.request

    def _boom(*a, **kw):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    diag = server_mod._diagnose_home_slowdown()
    assert diag["cause"] == "ollama_unreachable"
    assert diag["details"]["ollama_state"] == "unreachable"


@pytest.mark.slow
def test_stream_emits_degraded_event_when_slow(monkeypatch, stub_fetchers):
    """End-to-end: con un fetcher artificialmente lento + threshold
    bajado a 0.1s, el stream debe emitir UN sólo `event: degraded`
    antes del `done`.

    El probe de diagnóstico se monkeypatchea para evitar el hit a
    ollama localhost (no determinista en CI).
    """
    import time as _time
    from web import server as server_mod
    from web.server import app

    # Force threshold to ~0 by clearing history and lowering the floor.
    server_mod._HOME_COMPUTE_HISTORY.clear()
    monkeypatch.setattr(server_mod, "_HOME_COMPUTE_DEGRADED_FLOOR", 0.05)

    # Make `today` a slow stub so total elapsed crosses 0.05s easily
    # (each chip starts immediately; the slowdown only fires after the
    # first iteration of the queue loop, ~1s due to DISCONNECT_PROBE_S).
    def _slow_today(*a, **kw):
        _time.sleep(0.2)
        return {"recent_notes": [], "inbox_today": [], "todos": [],
                "new_contradictions": [], "low_conf_queries": []}

    monkeypatch.setattr(server_mod, "_collect_today_evidence", _slow_today)
    monkeypatch.setattr(server_mod, "_diagnose_home_slowdown",
                        lambda: {"cause": "unknown", "details": {"stub": True}})

    with TestClient(app) as client:
        with client.stream("GET", "/api/home/stream?regenerate=false") as resp:
            body = b"".join(resp.iter_bytes()).decode()

    events = _parse_sse(body)
    kinds = [e[0] for e in events]
    assert kinds.count("degraded") == 1, kinds
    degraded_payload = next(p for ev, p in events if ev == "degraded")
    assert degraded_payload["cause"] == "unknown"
    assert "elapsed_ms" in degraded_payload
    assert "threshold_ms" in degraded_payload
    assert degraded_payload["details"]["stub"] is True


def test_stream_does_not_emit_degraded_when_fast(stub_fetchers):
    """Path feliz: si el stream termina antes del threshold, NO hay
    `event: degraded`. Los stubs por default son sub-50ms cada uno,
    bien debajo del floor de 8s."""
    from web.server import app

    with TestClient(app) as client:
        with client.stream("GET", "/api/home/stream?regenerate=false") as resp:
            body = b"".join(resp.iter_bytes()).decode()

    events = _parse_sse(body)
    kinds = [e[0] for e in events]
    assert "degraded" not in kinds


# ── Reminder push plist ─────────────────────────────────────────────────────
#
# Validá que el plist generador produce un XML cargable por launchd y
# que está registrado en `_services_spec` para que `rag setup` lo
# instale en una máquina nueva.


def test_reminder_wa_push_plist_shape():
    """El plist debe llevar el binario, el comando `remind-wa`, un
    StartInterval razonable, y los logs apuntando al directorio
    estándar de obsidian-rag."""
    import rag

    xml = rag._reminder_wa_push_plist("/usr/local/bin/rag")
    assert "<key>Label</key><string>com.fer.obsidian-rag-reminder-wa-push</string>" in xml
    assert "<string>remind-wa</string>" in xml
    assert "<string>/usr/local/bin/rag</string>" in xml
    # 5min cadence — tighter than the 30min wa-tasks cron.
    assert "<key>StartInterval</key><integer>300</integer>" in xml
    assert "reminder-wa-push.log" in xml
    assert "reminder-wa-push.error.log" in xml


def test_reminder_wa_push_plist_in_services_spec():
    """`_services_spec` debe listarlo así `rag setup` lo instala."""
    import rag

    spec = rag._services_spec("/usr/local/bin/rag")
    labels = [s[0] for s in spec]
    assert "com.fer.obsidian-rag-reminder-wa-push" in labels
    # Asociado al filename canónico — `rag setup` usa este path para
    # escribir en ~/Library/LaunchAgents/.
    entry = next(s for s in spec if s[0] == "com.fer.obsidian-rag-reminder-wa-push")
    assert entry[1] == "com.fer.obsidian-rag-reminder-wa-push.plist"
    # XML consistent con el generator.
    assert "remind-wa" in entry[2]


# ── SQL persistence + /api/status/home ─────────────────────────────────────


@pytest.fixture
def isolate_home_metrics_sql(tmp_path, monkeypatch):
    """Redirigir DB_PATH a tmp_path y limpiar la rolling window
    in-memory para que cada test arranque en estado conocido. Restaura
    paths en su propio finally para evitar el warning falso del
    conftest autouse (mismo patrón que tests/test_reminder_wa_push.py)."""
    import rag
    from web import server as server_mod

    snap_db = rag.DB_PATH
    rag.DB_PATH = tmp_path
    server_mod._HOME_COMPUTE_HISTORY.clear()
    try:
        yield tmp_path
    finally:
        rag.DB_PATH = snap_db
        server_mod._HOME_COMPUTE_HISTORY.clear()


def test_record_home_compute_persists_to_sql(isolate_home_metrics_sql):
    """`_record_home_compute_total` debe escribir un row en
    `rag_home_compute_metrics` para que la baseline sobreviva un
    restart del web service."""
    import sqlite3
    import time as _time

    from web import server as server_mod

    server_mod._record_home_compute_total(
        7.5, regenerate=False, degraded=False,
    )
    server_mod._record_home_compute_total(
        12.3, regenerate=True, degraded=True, degraded_cause="ollama_slow",
    )

    # Writer is a daemon thread — give it a moment to land.
    deadline = _time.time() + 2.0
    db = isolate_home_metrics_sql / "telemetry.db"
    rows: list[tuple] = []
    while _time.time() < deadline:
        if db.is_file():
            try:
                with sqlite3.connect(str(db)) as conn:
                    rows = list(conn.execute(
                        "SELECT elapsed_s, regenerate, degraded, degraded_cause "
                        "FROM rag_home_compute_metrics ORDER BY id"
                    ).fetchall())
            except sqlite3.OperationalError:
                rows = []
            if len(rows) >= 2:
                break
        _time.sleep(0.05)

    assert len(rows) == 2, rows
    assert rows[0] == (7.5, 0, 0, None)
    assert rows[1] == (12.3, 1, 1, "ollama_slow")


def test_hydrate_home_compute_history_from_sql(isolate_home_metrics_sql):
    """Al startup, la rolling window se hidrata con los últimos
    `_HOME_COMPUTE_HISTORY_MAX` samples desde SQL — así el threshold
    no falla a `floor` después de cada kickstart."""
    import sqlite3

    from web import server as server_mod

    # Sembrar SQL directamente (saltea el writer async) para eliminar
    # races de timing en CI.
    db = isolate_home_metrics_sql / "telemetry.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "CREATE TABLE rag_home_compute_metrics ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
            " elapsed_s REAL NOT NULL, regenerate INTEGER NOT NULL DEFAULT 0,"
            " degraded INTEGER NOT NULL DEFAULT 0, degraded_cause TEXT)"
        )
        for i, v in enumerate([3.0, 4.0, 5.0, 6.0, 7.0]):
            conn.execute(
                "INSERT INTO rag_home_compute_metrics(ts, elapsed_s) VALUES(?, ?)",
                (f"2026-04-25T10:0{i}:00", v),
            )
        conn.commit()

    # Window vacía → hidratar.
    server_mod._HOME_COMPUTE_HISTORY.clear()
    server_mod._hydrate_home_compute_history_from_sql()
    assert list(server_mod._HOME_COMPUTE_HISTORY) == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_status_home_endpoint_returns_threshold_and_samples(
    isolate_home_metrics_sql,
):
    """`/api/status/home` devuelve threshold actual + últimos N
    samples + breakdown 24h."""
    import sqlite3

    from datetime import datetime
    from fastapi.testclient import TestClient
    from web.server import app

    db = isolate_home_metrics_sql / "telemetry.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "CREATE TABLE rag_home_compute_metrics ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
            " elapsed_s REAL NOT NULL, regenerate INTEGER NOT NULL DEFAULT 0,"
            " degraded INTEGER NOT NULL DEFAULT 0, degraded_cause TEXT)"
        )
        recent_iso = datetime.now().isoformat(timespec="seconds")
        for v in [5.0, 6.0, 7.0, 20.0]:
            conn.execute(
                "INSERT INTO rag_home_compute_metrics"
                "(ts, elapsed_s, regenerate, degraded, degraded_cause)"
                " VALUES(?, ?, 0, ?, ?)",
                (recent_iso, v, 1 if v > 15 else 0,
                 "ollama_slow" if v > 15 else None),
            )
        conn.commit()

    with TestClient(app) as client:
        r = client.get("/api/status/home")
        assert r.status_code == 200
        data = r.json()

    assert "samples" in data
    assert "window_24h" in data
    assert "threshold_s" in data
    assert "floor_s" in data
    assert "cold" in data
    # Cold = True porque la rolling window in-memory está vacía
    # (el endpoint no la hidrata por sí solo).
    assert data["cold"] is True
    assert data["floor_s"] == 8.0
    assert data["window_24h"]["total"] == 4
    assert data["window_24h"]["degraded"] == 1
    assert data["window_24h"]["by_cause"] == [{"cause": "ollama_slow", "count": 1}]


def test_status_home_endpoint_shape_when_table_empty(isolate_home_metrics_sql):
    """Al instalar (tabla vacía o nonexistente), el endpoint devuelve
    estructura mínima sin 500. Importa para que el frontend no
    branchee entre "instalado" vs "vacío"."""
    from fastapi.testclient import TestClient
    from web.server import app

    with TestClient(app) as client:
        r = client.get("/api/status/home")
        assert r.status_code == 200
        data = r.json()
    assert data["samples"] == []
    assert data["window_24h"] == {
        "total": 0, "degraded": 0, "regenerate": 0, "by_cause": [],
    }
    assert data["cold"] is True
    assert data["threshold_s"] == 8.0


def test_record_home_compute_drops_outliers_before_sql_write(
    isolate_home_metrics_sql,
):
    """Outliers (≤0 o >600s) ni siquiera disparan el writer thread —
    no contaminan SQL ni la rolling window."""
    import sqlite3
    import time as _time

    from web import server as server_mod

    server_mod._record_home_compute_total(0)
    server_mod._record_home_compute_total(-1)
    server_mod._record_home_compute_total(700)
    _time.sleep(0.3)  # let any rogue daemon thread land if it's going to

    db = isolate_home_metrics_sql / "telemetry.db"
    if db.is_file():
        with sqlite3.connect(str(db)) as conn:
            rows = list(conn.execute(
                "SELECT COUNT(*) FROM rag_home_compute_metrics"
            ).fetchall())
        assert rows[0][0] == 0, "outliers leaked to SQL"
    # Si la tabla no existe → ningún writer corrió, mejor todavía.


# ── Bug fix 2026-04-27: SSE slot cap on /api/home/stream ──────────────────


def test_home_stream_slot_cap_returns_429_when_full(monkeypatch):
    """Bug fix: /api/home/stream must honour the per-IP SSE slot cap
    (same as dashboard_stream / system_memory_stream / system_cpu_stream).
    When a given IP already has `_SSE_MAX_PER_IP` slots taken, a new
    request must get HTTP 429 before launching any _home_compute worker."""
    from web import server as srv

    test_ip = "10.0.0.1"

    # Pre-fill the slot counter to the cap.
    with srv._SSE_CONNECTIONS_LOCK:
        srv._SSE_CONNECTIONS_PER_IP[test_ip] = srv._SSE_MAX_PER_IP

    try:
        from fastapi import Request
        from unittest.mock import MagicMock
        import asyncio
        from fastapi import HTTPException

        # Build a minimal fake Request with the test IP.
        mock_request = MagicMock(spec=Request)
        mock_request.client = MagicMock()
        mock_request.client.host = test_ip

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(srv.home_stream(mock_request))

        assert exc_info.value.status_code == 429
        detail = exc_info.value.detail
        assert "too many concurrent streams" in detail
        assert str(srv._SSE_MAX_PER_IP) in detail
    finally:
        # Clean up so other tests see a clean counter.
        with srv._SSE_CONNECTIONS_LOCK:
            srv._SSE_CONNECTIONS_PER_IP.pop(test_ip, None)


def test_home_stream_slot_released_after_stream(monkeypatch, stub_fetchers):
    """Bug fix: slot must be released when the SSE stream ends so a
    second connection from the same IP can succeed (no permanent leak)."""
    from web import server as srv
    from fastapi.testclient import TestClient

    test_ip = "127.0.0.1"

    # Snapshot the counter before the request.
    with srv._SSE_CONNECTIONS_LOCK:
        count_before = srv._SSE_CONNECTIONS_PER_IP.get(test_ip, 0)

    client = TestClient(srv.app)
    resp = client.get(
        "/api/home/stream",
        headers={"Accept": "text/event-stream"},
    )
    # The TestClient drains the full response.
    assert resp.status_code == 200

    with srv._SSE_CONNECTIONS_LOCK:
        count_after = srv._SSE_CONNECTIONS_PER_IP.get(test_ip, 0)

    assert count_after == count_before, (
        f"slot was not released: before={count_before} after={count_after}"
    )
