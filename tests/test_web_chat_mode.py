"""Tests for the `mode` field on POST /api/chat (2026-04-24).

The web UI now sends `{mode: "auto"|"fast"|"deep"}` in the chat body
so the user can override the adaptive `fast_path` routing produced by
`retrieve()`:

- ``auto`` (default) → honour ``result["fast_path"]`` as before.
- ``fast``           → force small-model path (qwen2.5:3b + _LOOKUP_NUM_CTX).
- ``deep``           → force full chat model + _WEB_CHAT_NUM_CTX.

Invalid / unknown values fall back to ``"auto"`` silently — we never
400, so old clients (curl scripts, MCP, pre-feature PWA builds) keep
working without surprise breakage.

Telemetry persists both the effective ``mode`` and ``mode_origin``
(``"request"`` if the client sent the field, ``"default"`` if absent)
in ``rag_queries.extra_json``.

The pre-router → fast-path downgrade stays orthogonal: even with
``mode="deep"`` already on the full model, if the pre-router fires tools
the ``fast_path_downgraded`` marker still fires (the downgrade is a
prefill safety rail, not a routing decision — see the companion file
``test_web_fast_path_downgrade.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_web_server = pytest.importorskip("web.server")
_fastapi_testclient = pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402

from web import server as server_mod  # noqa: E402
from web.server import app  # noqa: E402


# ── Helpers (mirror test_web_fast_path_downgrade.py) ─────────────────────


def _canned_retrieve(fast_path: bool, query: str = "explicame qué es ikigai") -> dict:
    """Canned retrieve result with configurable fast_path marker.

    Uses the same shape as `multi_retrieve` so the generator accepts it.
    """
    return {
        "docs": ["doc body 1"],
        "metas": [
            {"file": "01-Projects/a.md", "note": "a", "folder": "01-Projects"},
        ],
        "scores": [1.5],
        "confidence": 0.8,
        "search_query": query,
        "filters_applied": {},
        "query_variants": [query],
        "vault_scope": ["default"],
        "fast_path": fast_path,
        "intent": "recent",
        "timing": {"total_ms": 500.0, "embed_ms": 30.0},
    }


# _OllamaMock viene de conftest (consolidado 2026-04-25 — antes byte-idéntico
# al de test_web_chat_tools con solo el docstring en español).
from tests.conftest import _OllamaMock  # noqa: F401 — usado en test bodies


def _mk_stream(tokens: list[str]) -> list[SimpleNamespace]:
    return [SimpleNamespace(message=SimpleNamespace(content=t)) for t in tokens]


class _LogCapture:
    """Captures log_query_event payloads for inspection.

    server_mod.log_query_event is swapped with an instance of this class
    via monkeypatch; the endpoint calls it once per turn with the whole
    extra_json dict (plus top-level cmd/session/paths/etc.).
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, event: dict) -> None:
        # Copy so the test doesn't observe downstream mutations.
        self.events.append(dict(event))


@pytest.fixture
def chat_env(monkeypatch):
    """Shared stubs so `/api/chat` runs in-process without touching
    network / ollama / vault. Caller picks what retrieve's fast_path
    flag is by re-applying `multi_retrieve` per-test.
    """
    monkeypatch.setattr(server_mod, "_ollama_alive", lambda timeout=2.0: True)
    monkeypatch.setattr(server_mod, "_ollama_chat_probe", lambda timeout_s=6.0: True)
    monkeypatch.setattr(server_mod, "_fetch_whatsapp_unread", lambda *a, **kw: [])
    import rag as _rag
    monkeypatch.setattr(_rag, "build_person_context", lambda q: None)
    monkeypatch.setattr(server_mod, "save_session", lambda sess: None)
    monkeypatch.setattr(server_mod, "_chat_cache_get", lambda key: None)
    monkeypatch.setattr(server_mod, "_chat_cache_put", lambda key, val: None)
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_DISABLED", "1")
    monkeypatch.setattr(server_mod, "_spawn_conversation_writer", lambda *a, **kw: None)
    monkeypatch.setattr(server_mod, "_is_tasks_query", lambda q: False)
    # Reset the rate-limit bucket — TestClient uses a fixed IP, so a
    # run with many tests consecutively fills the 30-req/60s bucket
    # and subsequent tests flake with 429. Pattern copied from
    # tests/test_web_chat_semantic_cache.py.
    import collections
    monkeypatch.setattr(
        server_mod, "_CHAT_BUCKETS", collections.defaultdict(list),
    )
    return monkeypatch


def _consume(resp) -> None:
    """Drain the SSE stream so the generator runs to completion."""
    for _ in resp.iter_lines():
        pass


# ── Field: mode is accepted by the Pydantic model ──────────────────────


def test_mode_field_exists_on_chat_request():
    """The `mode` field must be declared on ChatRequest or the Pydantic
    validator will reject bodies that include it with 422 (extra fields
    forbidden in the default config)."""
    from web.server import ChatRequest
    # Instantiate directly to avoid HTTP roundtrip noise.
    req = ChatRequest(question="hola", mode="fast")
    assert req.mode == "fast"
    req = ChatRequest(question="hola")  # field is optional
    assert req.mode is None


# ── Case 1: mode="fast" forces fast_path=True even when retrieve=False ─


def test_mode_fast_forces_small_model(chat_env):
    """`mode="fast"` must route to `_LOOKUP_MODEL` (qwen2.5:3b) even
    when retrieve() judged the query "complex" (fast_path=False). The
    log should record `fast_path=True` (effective, post-mode).
    """
    chat_env.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve(fast_path=False),
    )
    log = _LogCapture()
    chat_env.setattr(server_mod, "log_query_event", log)

    mock = _OllamaMock([_mk_stream(["ok"])])
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "question": "explicame qué es ikigai",
            "vault_scope": None,
            "mode": "fast",
        },
    )
    assert resp.status_code == 200, resp.text
    _consume(resp)

    streaming_calls = [c for c in mock.calls if c.get("stream")]
    assert streaming_calls, f"no streaming call: {mock.calls}"
    final_call = streaming_calls[-1]
    assert final_call["model"] == server_mod._LOOKUP_MODEL, (
        f"mode='fast' should force {server_mod._LOOKUP_MODEL}, "
        f"got {final_call['model']!r}"
    )
    # Adaptive num_ctx (2026-04-25): el ceiling es _LOOKUP_NUM_CTX (4096)
    # pero el valor efectivo se computa runtime desde final_messages chars.
    # Verificamos el contrato: 1024 <= num_ctx <= cap.
    _num_ctx = final_call["options"]["num_ctx"]
    assert 1024 <= _num_ctx <= server_mod._LOOKUP_NUM_CTX, (
        f"num_ctx={_num_ctx} fuera del rango adaptativo "
        f"[1024, {server_mod._LOOKUP_NUM_CTX}]"
    )

    assert log.events, "log_query_event never called"
    ev = log.events[-1]
    assert ev["fast_path"] is True, f"expected fast_path=True, got {ev['fast_path']}"
    assert ev["chat_mode"] == "fast"
    assert ev["chat_mode_origin"] == "request"


# ── Case 2: mode="deep" forces fast_path=False when retrieve=True ──────


def test_mode_deep_forces_full_model(chat_env):
    """`mode="deep"` must route to the full chat model even when
    retrieve() judged the query a fast-path candidate."""
    chat_env.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve(fast_path=True),
    )
    log = _LogCapture()
    chat_env.setattr(server_mod, "log_query_event", log)

    mock = _OllamaMock([_mk_stream(["ok"])])
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "question": "explicame qué es ikigai",
            "vault_scope": None,
            "mode": "deep",
        },
    )
    assert resp.status_code == 200, resp.text
    _consume(resp)

    streaming_calls = [c for c in mock.calls if c.get("stream")]
    assert streaming_calls, f"no streaming call: {mock.calls}"
    final_call = streaming_calls[-1]
    full_model = server_mod._resolve_web_chat_model()
    assert final_call["model"] == full_model, (
        f"mode='deep' should force {full_model!r}, got {final_call['model']!r}"
    )
    # Adaptive num_ctx (2026-04-25): cap es _WEB_CHAT_NUM_CTX (4096),
    # valor efectivo se calcula runtime. Ver test_mode_fast para detalle.
    _num_ctx = final_call["options"]["num_ctx"]
    assert 1024 <= _num_ctx <= server_mod._WEB_CHAT_NUM_CTX, (
        f"num_ctx={_num_ctx} fuera del rango adaptativo "
        f"[1024, {server_mod._WEB_CHAT_NUM_CTX}]"
    )

    ev = log.events[-1]
    assert ev["fast_path"] is False, f"expected fast_path=False, got {ev['fast_path']}"
    assert ev["chat_mode"] == "deep"
    assert ev["chat_mode_origin"] == "request"


# ── Case 3: mode="auto" delegates to retrieve() ───────────────────────


def test_mode_auto_delegates_to_retrieve(chat_env):
    """Explicit `mode="auto"` (i.e. user actively picked auto) must
    behave identically to omitting the field: the retrieve() flag
    decides. Only difference vs. omission is `mode_origin="request"`."""
    chat_env.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve(fast_path=True),
    )
    log = _LogCapture()
    chat_env.setattr(server_mod, "log_query_event", log)

    mock = _OllamaMock([_mk_stream(["ok"])])
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "question": "explicame qué es ikigai",
            "vault_scope": None,
            "mode": "auto",
        },
    )
    assert resp.status_code == 200, resp.text
    _consume(resp)

    streaming_calls = [c for c in mock.calls if c.get("stream")]
    final_call = streaming_calls[-1]
    # retrieve() said fast_path=True → small model.
    assert final_call["model"] == server_mod._LOOKUP_MODEL
    # Adaptive num_ctx (2026-04-25): cap _LOOKUP_NUM_CTX (4096), runtime
    # effective value. Ver test_mode_fast para detalle.
    _num_ctx = final_call["options"]["num_ctx"]
    assert 1024 <= _num_ctx <= server_mod._LOOKUP_NUM_CTX, (
        f"num_ctx={_num_ctx} fuera del rango adaptativo "
        f"[1024, {server_mod._LOOKUP_NUM_CTX}]"
    )

    ev = log.events[-1]
    assert ev["fast_path"] is True
    assert ev["chat_mode"] == "auto"
    assert ev["chat_mode_origin"] == "request"


# ── Case 4: missing mode → "auto" + mode_origin="default" ─────────────


def test_missing_mode_defaults_to_auto(chat_env):
    """Bodies without a `mode` field (old curl scripts, MCP, pre-feature
    PWA) must be treated as `mode="auto"` with `mode_origin="default"`."""
    chat_env.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve(fast_path=True),
    )
    log = _LogCapture()
    chat_env.setattr(server_mod, "log_query_event", log)

    mock = _OllamaMock([_mk_stream(["ok"])])
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"question": "explicame qué es ikigai", "vault_scope": None},
    )
    assert resp.status_code == 200, resp.text
    _consume(resp)

    ev = log.events[-1]
    assert ev["chat_mode"] == "auto"
    assert ev["chat_mode_origin"] == "default", (
        f"missing mode must record mode_origin='default', got {ev['mode_origin']!r}"
    )
    # retrieve=True → fast-path kept.
    assert ev["fast_path"] is True


# ── Case 5: invalid mode values silently fall back to "auto" ──────────


@pytest.mark.parametrize("bad_mode", ["rápido", "thinking", "FULL", "   ", "¯\\_(ツ)_/¯"])
def test_invalid_mode_silent_fallback(chat_env, bad_mode):
    """Unknown / accented / garbage `mode` values must NOT 400. They
    silently fall back to "auto". `mode_origin` stays "request" because
    the client did include the field (analytics can grep for
    `mode="auto" AND mode_origin="request" AND <value that shipped>`
    to spot broken clients)."""
    chat_env.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve(fast_path=True),
    )
    log = _LogCapture()
    chat_env.setattr(server_mod, "log_query_event", log)

    mock = _OllamaMock([_mk_stream(["ok"])])
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "question": "explicame qué es ikigai",
            "vault_scope": None,
            "mode": bad_mode,
        },
    )
    assert resp.status_code == 200, (
        f"invalid mode={bad_mode!r} must not 400: got {resp.status_code} {resp.text}"
    )
    _consume(resp)

    ev = log.events[-1]
    assert ev["chat_mode"] == "auto", (
        f"invalid mode={bad_mode!r} should fall back to 'auto', "
        f"got {ev['mode']!r}"
    )
    assert ev["chat_mode_origin"] == "request", (
        f"client sent garbage but mode_origin should still be 'request' "
        f"(analytics signal), got {ev['mode_origin']!r}"
    )


# ── Case 5b: case-insensitive valid modes ("FAST" → "fast") ───────────


def test_mode_case_insensitive(chat_env):
    """`.lower().strip()` normalisation means "FAST" / "  Deep  " are
    accepted as valid. This is incidental (per spec "Lowercase + strip")
    — documented in a test so we don't accidentally tighten later."""
    chat_env.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve(fast_path=False),
    )
    log = _LogCapture()
    chat_env.setattr(server_mod, "log_query_event", log)

    mock = _OllamaMock([_mk_stream(["ok"])])
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "question": "explicame qué es ikigai",
            "vault_scope": None,
            "mode": "  FAST  ",
        },
    )
    assert resp.status_code == 200, resp.text
    _consume(resp)

    ev = log.events[-1]
    assert ev["chat_mode"] == "fast", (
        f"' FAST ' should normalise to 'fast', got {ev['mode']!r}"
    )
    assert ev["fast_path"] is True


# ── Case 6: mode="deep" + pre-router fires → fast_path_downgraded=True ─


def test_mode_deep_still_downgrades_when_preroute_fires(chat_env):
    """CRITICAL INVARIANT: even with `mode="deep"` (already on full
    model), if the pre-router fires tools the `fast_path_downgraded`
    marker must still be recorded in telemetry. The downgrade is a
    prefill safety rail, not a routing decision — it fires whenever
    the small-model path was plausible (retrieve=True OR mode="fast")
    regardless of the user-chosen mode.

    Same fixture as `test_web_fast_path_downgrade.py`: retrieve returns
    fast_path=True, query "qué pendientes tengo" matches the reminders_due
    pre-router regex.
    """
    chat_env.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve(
            fast_path=True, query="qué pendientes tengo"
        ),
    )
    log = _LogCapture()
    chat_env.setattr(server_mod, "log_query_event", log)

    # Stub the reminders_due tool so the pre-router has something
    # concrete to inject.
    from web import tools as tools_mod
    chat_env.setattr(
        tools_mod, "_fetch_reminders_due",
        lambda days=7: {
            "dated": [
                {"title": "comprar pan", "due": "2026-04-24", "list": "Tareas"},
            ],
            "undated": [],
        },
    )

    mock = _OllamaMock([_mk_stream(["ok"])])
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "question": "qué pendientes tengo",
            "vault_scope": None,
            "mode": "deep",
        },
    )
    assert resp.status_code == 200, resp.text
    _consume(resp)

    # Final streaming must use the full model (both mode="deep" AND
    # the downgrade force it to full).
    streaming_calls = [c for c in mock.calls if c.get("stream")]
    full_model = server_mod._resolve_web_chat_model()
    assert streaming_calls[-1]["model"] == full_model

    ev = log.events[-1]
    assert ev["chat_mode"] == "deep"
    assert ev["chat_mode_origin"] == "request"
    # Effective fast_path is False (deep forced it; pre-router also
    # would have forced it).
    assert ev["fast_path"] is False
    # CRITICAL: downgrade marker still fires — even though we were
    # already on the full model, the pre-router engaged and that's a
    # meaningful signal for analytics (query retrieve wanted to
    # fast-path, pre-router said no).
    assert ev["fast_path_downgraded"] is True, (
        f"mode='deep' + pre-router must still record "
        f"fast_path_downgraded=True; got {ev['fast_path_downgraded']}"
    )
