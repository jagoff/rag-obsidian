"""Resilience tests for `/api/chat` — guards against generator hangs and
unhandled exceptions that would leave the SSE stream open without a `done`
event (the symptom is "PWA spinner congelado para siempre").

Two specific paths covered:

  1. **NLI grounding hang** — `_emit_grounding` calls `ground_claims_nli`
     synchronously. Pre-fix, if the NLI helper hung (qwen2.5:3b stuck-load,
     MPS contention), the SSE generator blocked indefinitely. The fix wraps
     the call in `ThreadPoolExecutor + result(timeout=...) +
     shutdown(wait=False, cancel_futures=True)` — same pattern that
     `_emit_enrich` uses (audit 2026-04-24 "quedó colgado").

  2. **Pre-router exception** — the call to `_detect_tool_intent(question)`
     at the early "have_docs?" branch (line ~9065) was OUTSIDE any try
     block. If the regex matcher or its helpers raised, the exception
     propagated through `gen()` and Starlette closed the connection
     without emitting `error + done` SSE events. Fix wraps the call in a
     try/except that emits both events and `return`s cleanly.

Both bugs were identified by the audit on 2026-04-27. These tests are
the regression battery for the fixes.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from web import server as server_mod
from web.server import app
from tests.conftest import _parse_sse


# ── Shared fixtures (mirror the chat_env pattern from test_web_chat_tools)
# We don't import that fixture directly because (a) it's not a session-scoped
# fixture exposed via conftest, (b) we want a slightly different mock matrix
# (need `multi_retrieve` + `_detect_tool_intent` + `ground_claims_nli` mocks
# specifically). Keeping this self-contained avoids cross-file fragility.


def _canned_retrieve_result(query: str = "x") -> dict:
    return {
        "docs": ["doc body 1", "doc body 2"],
        "metas": [
            {"file": "01-Projects/a.md", "note": "a", "folder": "01-Projects",
             "chunk_id": "01-Projects/a.md::0"},
            {"file": "02-Areas/b.md", "note": "b", "folder": "02-Areas",
             "chunk_id": "02-Areas/b.md::0"},
        ],
        "scores": [1.5, 1.0],
        "confidence": 0.8,
        "search_query": query,
        "filters_applied": {},
        "query_variants": [query],
        "vault_scope": ["default"],
    }


@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path):
    """Same isolation pattern as test_web_chat_tools — avoid prod telemetry
    pollution from TestClient runs."""
    import rag as _rag_isolate
    _snap = _rag_isolate.DB_PATH
    _rag_isolate.DB_PATH = tmp_path / "ragvec"
    try:
        yield
    finally:
        _rag_isolate.DB_PATH = _snap


@pytest.fixture
def chat_env(monkeypatch):
    """Common `/api/chat` mocks. Specific tests override individual entries
    (e.g. `_detect_tool_intent` to raise, `ground_claims_nli` to hang)."""
    monkeypatch.setenv("RAG_WEB_TOOL_LLM_DECIDE", "0")  # skip the LLM-decide round
    server_mod._CHAT_BUCKETS.clear()
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve_result(a[1] if len(a) >= 2 else "x"),
    )
    monkeypatch.setattr(server_mod, "_ollama_alive", lambda timeout=2.0: True)
    monkeypatch.setattr(server_mod, "_ollama_chat_probe", lambda timeout_s=6.0: True)
    monkeypatch.setattr(server_mod, "_fetch_whatsapp_unread", lambda *a, **kw: [])
    import rag as _rag
    monkeypatch.setattr(_rag, "build_person_context", lambda q: None)
    monkeypatch.setattr(server_mod, "_persist_conversation_turn", lambda *a, **kw: None)
    monkeypatch.setattr(server_mod, "save_session", lambda sess: None)
    monkeypatch.setattr(server_mod, "log_query_event", lambda ev: None)
    monkeypatch.setattr(server_mod, "_chat_cache_get", lambda key: None)
    monkeypatch.setattr(server_mod, "_chat_cache_put", lambda key, val: None)
    monkeypatch.setattr(server_mod, "_is_tasks_query", lambda q: False)
    return monkeypatch


def _post_chat(question: str = "qué hay sobre docker en mis notas") -> tuple[
    list[tuple[str, dict]], str
]:
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"question": question, "vault_scope": None},
    )
    assert resp.status_code == 200, resp.text
    return _parse_sse(resp.text), resp.text


# ─────────────────────────────────────────────────────────────────────────────
# Bug #1 — NLI grounding hang
# ─────────────────────────────────────────────────────────────────────────────


def test_grounding_hang_does_not_block_sse_done(chat_env, monkeypatch):
    """If `ground_claims_nli` hangs, the SSE stream must still emit `done`
    within a reasonable wall-time budget (we allow up to 12s — the
    grounding timeout itself is 4s, plus other stages add overhead in
    TestClient's threaded model).

    Pre-fix: the chat endpoint hung indefinitely because `_emit_grounding`
    awaited `ground_claims_nli` synchronously with no timeout.
    Post-fix: `ThreadPoolExecutor + result(timeout=...) +
    shutdown(wait=False)` wrapper releases the SSE stream after 4s.
    """
    # Force grounding ON so the path is exercised.
    monkeypatch.setenv("RAG_NLI_GROUNDING", "1")

    # Make `ground_claims_nli` hang for 30s — way over any sensible budget.
    # The fix should bail at 4s, the test budget is 12s (with margin).
    import rag as _rag

    def _hanging_nli(*args, **kwargs):
        time.sleep(30.0)  # hang >>> grounding budget
        # never reached:
        from rag import GroundingResult  # type: ignore[attr-defined]
        return GroundingResult(claims=[], claims_total=0, claims_supported=0,
                               claims_contradicted=0, claims_neutral=0, nli_ms=0)

    monkeypatch.setattr(_rag, "ground_claims_nli", _hanging_nli)

    # Stub the LLM so we get a deterministic stream that triggers grounding.
    from types import SimpleNamespace

    def _mock_chat(*args, **kwargs):
        if kwargs.get("stream"):
            return iter([
                SimpleNamespace(message=SimpleNamespace(content="hola "))
                for _ in range(2)
            ] + [SimpleNamespace(message=SimpleNamespace(content="mundo"))])
        return SimpleNamespace(message=SimpleNamespace(content="", tool_calls=None))

    monkeypatch.setattr(server_mod.ollama, "chat", _mock_chat)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", _mock_chat)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", _mock_chat)

    t0 = time.perf_counter()
    events, _ = _post_chat()
    elapsed = time.perf_counter() - t0

    # Hard upper bound: 12s. With the hang, pre-fix this would be 30+s
    # (or worse — TestClient may keep retrying).
    assert elapsed < 12.0, (
        f"chat endpoint took {elapsed:.1f}s — grounding hang was not "
        f"contained by timeout. Events: {[e for e, _ in events]}"
    )

    # Stream must still close cleanly.
    names = [ev for ev, _ in events]
    assert names[0] == "session"
    assert "done" in names, f"missing 'done' event — stream closed without close: {names}"


# ─────────────────────────────────────────────────────────────────────────────
# Bug #2 — Pre-router exception not caught
# ─────────────────────────────────────────────────────────────────────────────


def test_pre_router_detect_exception_emits_error_and_done(chat_env, monkeypatch):
    """If `_detect_tool_intent` (called early at line ~9065 to set
    `_has_forced_tools` for the empty-docs gate) raises an exception, the
    SSE stream must still emit `error` + `done(error=True)` before closing.

    Pre-fix: that early call was outside any try block. An exception
    propagated through `gen()`, Starlette closed the connection, and the
    PWA showed a frozen spinner with no diagnostic message.
    Post-fix: the call is wrapped so the SSE contract holds.
    """
    # Force the early call site (line 9065) to fail. The pre-router has a
    # SECOND call at line 9654 that is already inside a try/except — that
    # one is unaffected and protects the LLM-decide tool round.
    def _boom(_q):
        raise RuntimeError("intent matcher exploded (synthetic)")

    monkeypatch.setattr(server_mod, "_detect_tool_intent", _boom)

    # Also stub the LLM so if the test ever falls through past the early
    # branch, we don't make a real network call.
    from types import SimpleNamespace
    monkeypatch.setattr(
        server_mod.ollama, "chat",
        lambda *a, **kw: SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=None)
        ) if not kw.get("stream") else iter([]),
    )
    monkeypatch.setattr(
        server_mod._OLLAMA_STREAM_CLIENT, "chat",
        lambda *a, **kw: iter([SimpleNamespace(
            message=SimpleNamespace(content="x")
        )]) if kw.get("stream") else SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=None)
        ),
    )
    monkeypatch.setattr(
        server_mod._OLLAMA_TOOL_CLIENT, "chat",
        lambda *a, **kw: SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=None)
        ),
    )

    events, body = _post_chat("qué tengo para hacer hoy")

    names = [ev for ev, _ in events]
    # The session event always fires first — a sanity check that the
    # generator at least started.
    assert names[0] == "session", (
        f"session event missing; SSE generator never even started? body={body[:300]}"
    )
    # The two assertions that prove the fix:
    assert "error" in names, (
        f"missing 'error' event — pre-router exception leaked silently. "
        f"events: {names}\nbody: {body[:500]}"
    )
    assert "done" in names, (
        f"missing 'done' event — SSE stream closed without close marker, "
        f"client would hang forever. events: {names}"
    )
    # The done event must carry an `error: True` flag so the frontend can
    # distinguish a clean close from an error close.
    done_evt = [data for ev, data in events if ev == "done"][-1]
    assert done_evt.get("error") is True, (
        f"done event missing 'error: True' flag: {done_evt}"
    )
