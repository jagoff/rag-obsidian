"""Offline tests for the `/api/chat` ollama-native tool-calling loop.

All tests monkeypatch `ollama.chat` + collector `_fetch_*` helpers so nothing
hits the network, the real Ollama daemon, or the real vault. The TestClient
consumes the SSE stream and we assert on event order + the `[chat-timing]`
log line emitted via `print(..., flush=True)` to stdout (captured via
`capsys`).

Covers:
  1. `web.tools` module exports + metadata.
  2. Each parallel-safe tool wrapper is silent-fail on collector exceptions.
  3. `/api/chat` path when LLM emits no tool_calls (baseline).
  4. `/api/chat` path when LLM calls one tool then answers.
  5. Three parallel-safe tools in one round — fan-out ordering.
  6. Serial bucket (search_vault) runs before parallel (weather).
  7. Tool raising inside the loop — stream still completes.
  8. Round cap (3) hit → nudge appended, final streaming call fires.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


# ── Import targets under test ──────────────────────────────────────────────

from web import tools as tools_mod
from web.tools import (
    CHAT_TOOL_OPTIONS,
    CHAT_TOOLS,
    PARALLEL_SAFE,
    TOOL_FNS,
    _WEB_TOOL_ADDENDUM,
    calendar_ahead,
    finance_summary,
    gmail_recent,
    reminders_due,
    weather,
)
from web import server as server_mod
from web.server import app


# ── SSE parsing helpers ────────────────────────────────────────────────────


_EVENT_RE = re.compile(r"event: (?P<event>[^\n]+)\ndata: (?P<data>[^\n]*)\n\n")


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE stream body into `[(event, data_dict), ...]`."""
    out: list[tuple[str, dict]] = []
    for m in _EVENT_RE.finditer(body):
        try:
            payload = json.loads(m.group("data"))
        except Exception:
            payload = {}
        out.append((m.group("event"), payload))
    return out


def _last_chat_timing(captured_text: str) -> str | None:
    """Return the most recent `[chat-timing] ...` line in captured output."""
    lines = [ln for ln in captured_text.splitlines() if ln.startswith("[chat-timing]")]
    return lines[-1] if lines else None


# ── Ollama mock ────────────────────────────────────────────────────────────


class _OllamaMock:
    """Scripted stand-in for `ollama.chat`.

    `responses` is a list of pre-built responses consumed in FIFO order. Each
    element is either:
      - a SimpleNamespace (non-streaming, returned directly), OR
      - a list of SimpleNamespace chunks (streaming, wrapped in `iter(...)`).

    The distinction is detected via the `stream` kwarg on each call.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("OllamaMock: ran out of scripted responses")
        resp = self.responses.pop(0)
        if kwargs.get("stream"):
            # Streaming: wrap the list in an iterator of chunks.
            return iter(resp)
        return resp


def _mk_msg(content: str = "", tool_calls=None) -> SimpleNamespace:
    """Build a non-streaming ollama response shape."""
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls),
    )


def _mk_tool_call(name: str, args: dict) -> SimpleNamespace:
    """Build a single tool_call with a `.model_dump()` method (server uses it)."""
    dump = {
        "function": {"name": name, "arguments": args},
    }
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=args),
        model_dump=lambda d=dump: d,
    )


def _mk_stream(tokens: list[str]) -> list[SimpleNamespace]:
    """Build a list of streaming chunks — `_OllamaMock` will wrap in iter()."""
    return [SimpleNamespace(message=SimpleNamespace(content=t)) for t in tokens]


# ── Canned retrieve result ─────────────────────────────────────────────────


def _canned_retrieve_result(query: str = "x") -> dict:
    return {
        "docs": ["doc 1 body", "doc 2 body"],
        "metas": [
            {"file": "01-Projects/a.md", "note": "a", "folder": "01-Projects"},
            {"file": "02-Areas/b.md", "note": "b", "folder": "02-Areas"},
        ],
        "scores": [1.5, 1.0],
        "confidence": 0.8,
        "search_query": query,
        "filters_applied": {},
        "query_variants": [query],
        "vault_scope": ["default"],
    }


# ── Common chat-endpoint fixture ───────────────────────────────────────────


@pytest.fixture
def chat_env(monkeypatch):
    """Shared monkeypatches for `/api/chat` invocations.

    - Stub `multi_retrieve` with canned output.
    - Bypass the ollama-alive probe.
    - Disable the WhatsApp fetch (no SQLite hit).
    - Skip person-mention enrichment.
    - Skip the episodic-conversation daemon thread.
    - No-op session persistence + log writer.
    - Force RAG_WEB_TOOL_LLM_DECIDE=1 so the LLM tool-deciding round runs
      even when the pre-router matches nothing (default prod behaviour is
      to skip it for latency; these tests assert that round's semantics).
    """
    monkeypatch.setenv("RAG_WEB_TOOL_LLM_DECIDE", "1")
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve_result(a[1] if len(a) >= 2 else "x"),
    )
    monkeypatch.setattr(server_mod, "_ollama_alive", lambda timeout=2.0: True)
    monkeypatch.setattr(server_mod, "_ollama_chat_probe", lambda timeout_s=6.0: True)
    monkeypatch.setattr(server_mod, "_fetch_whatsapp_unread", lambda *a, **kw: [])
    # build_person_context is a lazy `from rag import ...` inside the
    # endpoint, so we patch the source module.
    import rag as _rag
    monkeypatch.setattr(_rag, "build_person_context", lambda q: None)
    monkeypatch.setattr(server_mod, "_persist_conversation_turn", lambda *a, **kw: None)
    monkeypatch.setattr(server_mod, "save_session", lambda sess: None)
    monkeypatch.setattr(server_mod, "log_query_event", lambda ev: None)
    # Disable the response cache so every test hits the real tool loop.
    monkeypatch.setattr(server_mod, "_chat_cache_get", lambda key: None)
    monkeypatch.setattr(server_mod, "_chat_cache_put", lambda key, val: None)
    # Tasks intent detection off — we want the full chat path every time.
    monkeypatch.setattr(server_mod, "_is_tasks_query", lambda q: False)
    return monkeypatch


def _post_chat(question: str = "hola") -> tuple[list[tuple[str, dict]], str]:
    """Fire a POST to /api/chat, parse SSE, return `(events, raw_body)`."""
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"question": question, "vault_scope": None},
    )
    assert resp.status_code == 200, resp.text
    body = resp.text
    return _parse_sse(body), body


# ── 1. Module exports ──────────────────────────────────────────────────────


def test_tools_module_exports():
    assert len(CHAT_TOOLS) == 9
    assert len(TOOL_FNS) == 9
    assert PARALLEL_SAFE == {
        "weather", "finance_summary", "calendar_ahead",
        "reminders_due", "gmail_recent",
        "propose_reminder", "propose_calendar_event",
    }
    assert CHAT_TOOL_OPTIONS == {
        "num_ctx": 4096,
        "num_predict": 512,
        "temperature": 0.0,
        "seed": 42,
    }
    assert isinstance(_WEB_TOOL_ADDENDUM, str)
    assert len(_WEB_TOOL_ADDENDUM) > 300
    for fn in CHAT_TOOLS:
        assert callable(fn), f"{fn!r} not callable"
        assert fn.__doc__ and fn.__doc__.strip(), (
            f"{fn.__name__} has empty docstring — ollama derives the JSON "
            f"schema from it, so it cannot be blank"
        )


# ── 2. Silent-fail wrappers ────────────────────────────────────────────────


def test_tool_wrappers_silent_fail(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("boom")

    # reminders_due — catches exception from _fetch_reminders_due.
    monkeypatch.setattr(tools_mod, "_fetch_reminders_due", _boom)
    out = reminders_due(7)
    parsed = json.loads(out)
    assert parsed.get("dated") == []
    assert parsed.get("undated") == []

    # gmail_recent — _fetch_gmail_evidence raises → degraded shape.
    monkeypatch.setattr(tools_mod, "_fetch_gmail_evidence", _boom)
    out = gmail_recent()
    parsed = json.loads(out)
    assert parsed == {"unread_count": 0, "threads": []}

    # finance_summary — inner import + fetch raises → "{}".
    def _bad_fetch_finance(anchor):
        raise RuntimeError("no moze")
    monkeypatch.setattr(server_mod, "_fetch_finance", _bad_fetch_finance)
    out = finance_summary()
    assert json.loads(out) == {}

    # calendar_ahead — inner fetch raises → "[]".
    def _bad_fetch_cal(n, max_events=40):
        raise RuntimeError("no cal")
    monkeypatch.setattr(server_mod, "_fetch_calendar_ahead", _bad_fetch_cal)
    out = calendar_ahead(3)
    assert json.loads(out) == []

    # weather — _agent_tool_weather handles the error internally; should
    # still return a string + not raise.
    monkeypatch.setattr(tools_mod, "_agent_tool_weather", lambda loc=None: "Error: offline")
    out = weather("Santa Fe")
    assert isinstance(out, str)


def test_gmail_recent_consumes_enriched_evidence(monkeypatch):
    """TODO-2 regression: gmail_recent must pull thread_id + internal_date_ms
    from _fetch_gmail_evidence directly (no secondary _gmail_service re-query).
    Also verifies the internal_date_ms → received_at ISO conversion.
    """
    fake_ev = {
        "unread_count": 7,
        "awaiting_reply": [
            {
                "subject": "Proyecto X",
                "from": "juan@example.com",
                "snippet": "dale cuando puedas",
                "days_old": 4.2,
                "thread_id": "tid-aaa",
                "internal_date_ms": 1_713_200_000_000,  # 2024-04-15 ~14:53 UTC
            },
        ],
        "starred": [
            {
                "subject": "Importante",
                "from": "ana@example.com",
                "snippet": "ojo con esto",
                "thread_id": "tid-bbb",
                "internal_date_ms": 1_713_300_000_000,
            },
        ],
    }
    monkeypatch.setattr(tools_mod, "_fetch_gmail_evidence", lambda now: fake_ev)

    out = gmail_recent()
    parsed = json.loads(out)
    assert parsed["unread_count"] == 7
    assert len(parsed["threads"]) == 2
    # awaiting first, then starred (gmail_recent's documented ordering).
    t0, t1 = parsed["threads"]
    assert t0["kind"] == "awaiting_reply"
    assert t0["thread_id"] == "tid-aaa"
    assert t0["subject"] == "Proyecto X"
    assert t0["days_old"] == 4.2
    # received_at = local ISO 'YYYY-MM-DDTHH:MM' derived from internal_date_ms.
    assert t0["received_at"]
    assert "T" in t0["received_at"] and t0["received_at"].count(":") == 1
    assert t1["kind"] == "starred"
    assert t1["thread_id"] == "tid-bbb"
    assert t1["received_at"]


def test_gmail_recent_missing_internal_date_is_empty_received_at(monkeypatch):
    """If _fetch_gmail_evidence emits an item without internal_date_ms (fallback
    path on gmail_thread_last_meta failure), received_at falls back to ''."""
    fake_ev = {
        "unread_count": 1,
        "awaiting_reply": [
            {"subject": "s", "from": "x@y.com", "snippet": "", "days_old": 0.0,
             "thread_id": "tid-x", "internal_date_ms": 0},
        ],
        "starred": [],
    }
    monkeypatch.setattr(tools_mod, "_fetch_gmail_evidence", lambda now: fake_ev)
    parsed = json.loads(gmail_recent())
    assert parsed["threads"][0]["thread_id"] == "tid-x"
    assert parsed["threads"][0]["received_at"] == ""


# ── 3. No-tool-calls path ──────────────────────────────────────────────────


def test_chat_endpoint_no_tools_path(chat_env, capsys):
    responses = [
        # Tool-deciding round 1 — no tool_calls.
        _mk_msg(content="", tool_calls=None),
        # Final streaming call — 3 chunks.
        _mk_stream(["hola ", "mundo ", "listo"]),
    ]
    mock = _OllamaMock(responses)
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    # Input must NOT match `_detect_metachat_intent` (short-circuits the
    # retrieval path this test is asserting on). "que tal" was moved into
    # the metachat bucket 2026-04-21, so switched to a real retrieval-
    # flavoured query.
    events, _body = _post_chat("qué hay sobre docker en mis notas")

    names = [ev for ev, _ in events]
    # Order expectation.
    assert names[0] == "session"
    assert ("status", {"stage": "retrieving"}) in events
    assert "sources" in names
    assert ("status", {"stage": "generating"}) in events
    assert names[-1] == "done"
    # No tool-related status events.
    for _ev, data in events:
        assert data.get("stage") != "tool"
        assert data.get("stage") != "tool_done"
    # ≥1 token event.
    assert any(ev == "token" for ev, _ in events)

    out = capsys.readouterr().out
    line = _last_chat_timing(out)
    assert line is not None, out
    assert "tool_rounds=0" in line
    assert "tool_ms=0" in line
    assert "tool_names=" in line  # empty tail


# ── 4. Single-tool path ────────────────────────────────────────────────────


def test_chat_endpoint_tool_path(chat_env, capsys):
    # Stub the weather collector so it returns deterministic JSON.
    chat_env.setattr(
        tools_mod, "_agent_tool_weather",
        lambda loc=None: '{"today": "soleado", "tomorrow": "lluvia"}',
    )
    mock = _OllamaMock([
        # Round 1 — ask for weather.
        _mk_msg(tool_calls=[_mk_tool_call("weather", {"location": "Santa Fe"})]),
        # Round 2 — no more tools.
        _mk_msg(tool_calls=[]),
        # Final streaming — 2 chunks.
        _mk_stream(["tiempo: ", "soleado"]),
    ])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    events, _ = _post_chat("qué tiempo va a hacer")

    stages = [
        (ev, data.get("stage"), data.get("name"))
        for ev, data in events if ev == "status"
    ]
    assert ("status", "retrieving", None) in stages
    assert ("status", "tool", "weather") in stages
    assert ("status", "tool_done", "weather") in stages
    assert ("status", "generating", None) in stages

    # Tool event must come before tool_done.
    order = [(data.get("stage"), data.get("name"))
             for ev, data in events if ev == "status"]
    i_tool = order.index(("tool", "weather"))
    i_done = order.index(("tool_done", "weather"))
    assert i_tool < i_done

    # Timing log.
    line = _last_chat_timing(capsys.readouterr().out)
    assert line is not None
    # tool_rounds == 2: the keyword pre-router fires one deterministic
    # round before the LLM-decide loop (which adds a second round).
    # See pre-router commits ee8ae85..345f343 — both rounds are emitted
    # even when the LLM tool is the same as the pre-routed one.
    assert "tool_rounds=2" in line
    m = re.search(r"tool_ms=(\d+)", line)
    assert m and int(m.group(1)) >= 0
    assert "tool_names=weather" in line


# ── 5. Parallel tools in one round ─────────────────────────────────────────


def test_chat_endpoint_parallel_tools(chat_env, capsys):
    chat_env.setattr(tools_mod, "_agent_tool_weather", lambda loc=None: '{"w": 1}')
    chat_env.setattr(server_mod, "_fetch_finance", lambda anchor: {"spend": 100})
    chat_env.setattr(server_mod, "_fetch_calendar_ahead", lambda n, max_events=40: [])

    mock = _OllamaMock([
        _mk_msg(tool_calls=[
            _mk_tool_call("weather", {}),
            _mk_tool_call("finance_summary", {}),
            _mk_tool_call("calendar_ahead", {"days": 3}),
        ]),
        _mk_msg(tool_calls=[]),
        _mk_stream(["ok"]),
    ])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    events, _ = _post_chat("resumen del dia")

    status_events = [
        (data.get("stage"), data.get("name"))
        for ev, data in events if ev == "status"
    ]
    # Collect indices.
    tool_idxs = [i for i, s in enumerate(status_events) if s[0] == "tool"]
    done_idxs = [i for i, s in enumerate(status_events) if s[0] == "tool_done"]
    # Pre-router fires a deterministic round before the LLM-decide loop
    # (ee8ae85/5680f3b), so the stream contains pre-router tool/tool_done
    # pairs interleaved with the LLM fan-out. The parallel-fan-out invariant
    # only applies to the LLM bucket: its ≥3 `tool` events all emit before
    # any of its `tool_done`. Identify the LLM bucket by taking the last 3
    # tool_idxs and the last 3 done_idxs — pre-router events came earlier.
    assert len(tool_idxs) >= 3
    assert len(done_idxs) >= 3
    llm_tool_idxs = tool_idxs[-3:]
    llm_done_idxs = done_idxs[-3:]
    assert max(llm_tool_idxs) < min(llm_done_idxs), (
        f"LLM-bucket parallel fan-out violated: tool idxs {llm_tool_idxs}, "
        f"done idxs {llm_done_idxs}"
    )

    tool_names_in_events = {
        s[1] for s in status_events if s[0] == "tool"
    }
    # The 3 LLM-decided tools must all appear. Pre-router may inject extra
    # tools (e.g. reminders_due) when the prompt has triggering keywords —
    # that's a feature, not a failure. Assert the 3 LLM tools are a subset.
    assert {"weather", "finance_summary", "calendar_ahead"}.issubset(tool_names_in_events)

    # Log line.
    line = _last_chat_timing(capsys.readouterr().out)
    assert line is not None
    # tool_rounds == 2: pre-router round + LLM-decide round.
    assert "tool_rounds=2" in line
    # All 3 LLM names in tool_names= (order not asserted — parallel).
    m = re.search(r"tool_names=([^\s]+)", line)
    assert m
    names = set(m.group(1).split(","))
    assert {"weather", "finance_summary", "calendar_ahead"}.issubset(names)


# ── 6. Serial bucket precedes parallel ─────────────────────────────────────


def test_chat_endpoint_serial_then_parallel(chat_env, capsys):
    chat_env.setattr(tools_mod, "_agent_tool_search", lambda q, k=5: '[]')
    chat_env.setattr(tools_mod, "_agent_tool_weather", lambda loc=None: '{"w": 1}')

    mock = _OllamaMock([
        _mk_msg(tool_calls=[
            _mk_tool_call("search_vault", {"query": "foo"}),
            _mk_tool_call("weather", {}),
        ]),
        _mk_msg(tool_calls=[]),
        _mk_stream(["done"]),
    ])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    events, _ = _post_chat("buscá foo y contame el clima")

    status_events = [
        (data.get("stage"), data.get("name"))
        for ev, data in events if ev == "status"
    ]
    # search_vault's tool + tool_done must both precede weather's LLM-emitted
    # tool event (serial bucket runs fully before parallel fan-out starts).
    # Note: the pre-router fires weather BEFORE the LLM decides anything, so
    # the first ("tool", "weather") belongs to the pre-router. We want the
    # LAST ("tool", "weather") — the LLM's decision — as the reference point.
    idx_sv_tool = status_events.index(("tool", "search_vault"))
    idx_sv_done = status_events.index(("tool_done", "search_vault"))
    # Last index of ("tool", "weather").
    weather_tool_positions = [
        i for i, s in enumerate(status_events) if s == ("tool", "weather")
    ]
    idx_w_tool = weather_tool_positions[-1]
    assert idx_sv_tool < idx_sv_done < idx_w_tool


# ── 7. Tool exception is caught and stream completes ───────────────────────


def test_chat_endpoint_tool_exception_recovers(chat_env, capsys):
    def _blow_up(loc=None):
        raise RuntimeError("api down")
    chat_env.setattr(tools_mod, "_agent_tool_weather", _blow_up)

    mock = _OllamaMock([
        _mk_msg(tool_calls=[_mk_tool_call("weather", {})]),
        _mk_msg(tool_calls=[]),
        _mk_stream(["ok"]),
    ])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    events, _ = _post_chat("clima")

    names = [ev for ev, _ in events]
    assert "error" not in names, f"unexpected error event in {names}"
    assert names[-1] == "done"
    # tool_done still emitted despite the exception.
    status_events = [
        (data.get("stage"), data.get("name"))
        for ev, data in events if ev == "status"
    ]
    assert ("tool_done", "weather") in status_events

    # Timing log.
    line = _last_chat_timing(capsys.readouterr().out)
    assert line is not None
    # tool_rounds == 2: pre-router round + LLM-decide round (see test_chat_
    # endpoint_tool_path).
    assert "tool_rounds=2" in line
    assert "tool_names=weather" in line

    # Spy on the SECOND ollama.chat call — its messages kwarg should include
    # a role:"tool" message whose content starts with "Error:".
    second_call = mock.calls[1]
    msgs = second_call["messages"]
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert tool_msgs, "expected a role=tool message in the 2nd round"
    assert tool_msgs[0]["content"].startswith("Error:")


# ── 8. Round cap respected + nudge appended ────────────────────────────────


def test_chat_endpoint_round_cap_respected(chat_env, capsys):
    chat_env.setattr(tools_mod, "_agent_tool_weather", lambda loc=None: '{"w": 1}')

    mock = _OllamaMock([
        # Rounds 1, 2, 3 — model keeps asking for weather.
        _mk_msg(tool_calls=[_mk_tool_call("weather", {})]),
        _mk_msg(tool_calls=[_mk_tool_call("weather", {})]),
        _mk_msg(tool_calls=[_mk_tool_call("weather", {})]),
        # 4th call: final streaming answer.
        _mk_stream(["forzado"]),
    ])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    events, _ = _post_chat("dame clima 3 veces")

    names = [ev for ev, _ in events]
    assert names[-1] == "done"

    # Exactly 4 ollama.chat calls (3 tool-deciding rounds + 1 final stream).
    assert len(mock.calls) == 4, mock.calls

    # 4th call is the streaming one (stream=True) and its messages must
    # include the system nudge appended by the round-cap branch.
    final_call = mock.calls[3]
    assert final_call.get("stream") is True
    final_msgs = final_call["messages"]
    nudge = "Alcanzado cap de herramientas; respondé con lo que tenés."
    assert any(
        m.get("role") == "system" and m.get("content") == nudge
        for m in final_msgs
    ), f"nudge missing from final call messages: {final_msgs!r}"

    # Timing log.
    line = _last_chat_timing(capsys.readouterr().out)
    assert line is not None
    # tool_rounds == 4: pre-router round + 3 capped LLM rounds.
    # _TOOL_ROUND_CAP=3 is the LLM-loop cap; pre-router adds one extra.
    assert "tool_rounds=4" in line
    assert "tool_names=weather,weather,weather,weather" in line
