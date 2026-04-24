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
    # Reset per-IP rate limit bucket — el `_CHAT_BUCKETS` default dict
    # acumula timestamps entre tests y el límite de 30 req/60s se
    # excedía cuando la suite entera corría junta, tirando 429s en
    # tests que en isolation pasaban. Clear garantiza que cada test
    # arranque con el budget lleno. TestClient usa "testclient" como
    # client IP por defecto.
    server_mod._CHAT_BUCKETS.clear()
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
    assert len(CHAT_TOOLS) == 16
    assert len(TOOL_FNS) == 16
    assert PARALLEL_SAFE == {
        "weather", "finance_summary", "calendar_ahead",
        "reminders_due", "gmail_recent", "drive_search",
        "whatsapp_pending", "whatsapp_search", "whatsapp_thread",
        "propose_reminder", "propose_calendar_event",
        # `propose_whatsapp_send` intentionally NOT here — see web/tools.py
        # comment: destructive action + osascript-heavy lookup.
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


def test_gmail_recent_surfaces_inbox_bucket_when_awaiting_starred_empty(monkeypatch):
    """Iter 5 regression (2026-04-24, Fer F. user report): si el user tiene
    inbox-zero-ish (nada starred, nada awaiting), el tool original
    devolvía `threads: []` y el LLM respondía "no encontré mails
    recientes" — aunque el inbox del user tuviera perfectamente mails
    navegables. El bucket `recent` agregado en `_fetch_gmail_evidence`
    tapa ese gap: son los últimos N del inbox sin filtros de status.

    Este test verifica que cuando awaiting/starred vienen vacíos pero
    `recent` trae items, `gmail_recent` los surface con `kind="recent"`
    y el shape esperado (subject/from/received_at).
    """
    fake_ev = {
        "unread_count": 0,
        "awaiting_reply": [],
        "starred": [],
        "recent": [
            {
                "subject": "Aviso de débito automático",
                "from": "Santander <avisos@santander.com.ar>",
                "snippet": "Monto U$S9,99 Comercio APPLECOM BILL",
                "thread_id": "tid-r1",
                "internal_date_ms": 1_713_400_000_000,
            },
            {
                "subject": "CI passed: main (abc123)",
                "from": "GitHub <notifications@github.com>",
                "snippet": "All jobs succeeded",
                "thread_id": "tid-r2",
                "internal_date_ms": 1_713_300_000_000,
            },
        ],
    }
    monkeypatch.setattr(tools_mod, "_fetch_gmail_evidence", lambda now: fake_ev)

    parsed = json.loads(gmail_recent())
    # Unread count passthrough (0 acá, bien).
    assert parsed["unread_count"] == 0
    # Los 2 recents aparecen con kind="recent".
    assert len(parsed["threads"]) == 2
    t0, t1 = parsed["threads"]
    assert t0["kind"] == "recent"
    assert t0["subject"] == "Aviso de débito automático"
    assert t0["thread_id"] == "tid-r1"
    assert t0["received_at"]  # ISO string derivado de internal_date_ms
    assert t1["kind"] == "recent"
    assert t1["subject"] == "CI passed: main (abc123)"


def test_gmail_recent_priority_order_awaiting_starred_recent(monkeypatch):
    """Con los 3 buckets llenos, el orden en `threads` es:
    awaiting_reply → starred → recent. Prioridad por actionability:
    mails esperando respuesta del user son más urgentes que starred
    viejos, y ambos más útiles que el último de la lista general.
    """
    fake_ev = {
        "unread_count": 2,
        "awaiting_reply": [
            {"subject": "aw1", "from": "a@x.com", "snippet": "",
             "days_old": 5.0, "thread_id": "aw-1", "internal_date_ms": 1_000_000_000_000},
        ],
        "starred": [
            {"subject": "st1", "from": "b@x.com", "snippet": "",
             "thread_id": "st-1", "internal_date_ms": 1_100_000_000_000},
        ],
        "recent": [
            {"subject": "re1", "from": "c@x.com", "snippet": "",
             "thread_id": "re-1", "internal_date_ms": 1_200_000_000_000},
            {"subject": "re2", "from": "d@x.com", "snippet": "",
             "thread_id": "re-2", "internal_date_ms": 1_210_000_000_000},
        ],
    }
    monkeypatch.setattr(tools_mod, "_fetch_gmail_evidence", lambda now: fake_ev)

    parsed = json.loads(gmail_recent())
    kinds = [t["kind"] for t in parsed["threads"]]
    assert kinds == ["awaiting_reply", "starred", "recent", "recent"]
    # thread_ids también respetan el orden.
    tids = [t["thread_id"] for t in parsed["threads"]]
    assert tids == ["aw-1", "st-1", "re-1", "re-2"]


def test_gmail_recent_caps_at_12_total_threads(monkeypatch):
    """El cap de 12 threads evita que una invocación con todos los
    buckets a tope infle el CONTEXTO con >12 items."""
    # 5 awaiting + 5 starred + 8 recent = 18 → cap a 12.
    fake_ev = {
        "unread_count": 0,
        "awaiting_reply": [
            {"subject": f"aw{i}", "from": "a@x.com", "snippet": "",
             "days_old": 3.0, "thread_id": f"aw-{i}",
             "internal_date_ms": 1_000_000_000_000 + i}
            for i in range(5)
        ],
        "starred": [
            {"subject": f"st{i}", "from": "b@x.com", "snippet": "",
             "thread_id": f"st-{i}",
             "internal_date_ms": 1_100_000_000_000 + i}
            for i in range(5)
        ],
        "recent": [
            {"subject": f"re{i}", "from": "c@x.com", "snippet": "",
             "thread_id": f"re-{i}",
             "internal_date_ms": 1_200_000_000_000 + i}
            for i in range(8)
        ],
    }
    monkeypatch.setattr(tools_mod, "_fetch_gmail_evidence", lambda now: fake_ev)

    parsed = json.loads(gmail_recent())
    assert len(parsed["threads"]) == 12
    # Se cortó al llegar al cap — los primeros 5 awaiting + 5 starred +
    # 2 recent completan los 12.
    kinds = [t["kind"] for t in parsed["threads"]]
    assert kinds == ["awaiting_reply"] * 5 + ["starred"] * 5 + ["recent"] * 2


# ── 3. No-tool-calls path ──────────────────────────────────────────────────


@pytest.mark.requires_ollama
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
    chat_env.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    # Input must NOT match `_detect_metachat_intent` (short-circuits the
    # retrieval path this test is asserting on). "que tal" was moved into
    # the metachat bucket 2026-04-21, so switched to a real retrieval-
    # flavoured query.
    events, _body = _post_chat("qué hay sobre docker en mis notas")

    names = [ev for ev, _ in events]
    # Order expectation.
    assert names[0] == "session"
    # Status events now carry `intent` (added 2026-04-22 commit 73c3138).
    # The stage value is what matters here — match on that, not the whole dict.
    assert any(ev == "status" and data.get("stage") == "retrieving"
               for ev, data in events), f"missing retrieving status in {events}"
    assert "sources" in names
    assert any(ev == "status" and data.get("stage") == "generating"
               for ev, data in events), f"missing generating status in {events}"
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


@pytest.mark.requires_ollama
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
    chat_env.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

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


@pytest.mark.requires_ollama
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
    chat_env.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

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


@pytest.mark.requires_ollama
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
    chat_env.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

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


@pytest.mark.requires_ollama
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
    chat_env.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

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


@pytest.mark.requires_ollama
def test_chat_timing_log_sanitizes_infinity_confidence(chat_env, capsys, monkeypatch):
    """Regression 2026-04-21 (batch B.4): cuando `retrieve()` devuelve
    `float('-inf')` (corpus vacío / meta-chat path), el `[chat-timing]`
    loggeaba `confidence=-inf` crudo. Grep / dashboards levantaban eso
    como "error sin sanear". Tras el fix, `_sanitize_confidence` clampea
    a 0.000 para que el log sea shape-stable.
    """
    # Stub multi_retrieve para devolver confidence=-inf (caso corpus vacío).
    def _empty_retrieve(*a, **kw):
        r = _canned_retrieve_result("x")
        r["confidence"] = float("-inf")
        r["scores"] = [float("-inf")] * len(r["metas"])
        return r
    monkeypatch.setattr(server_mod, "multi_retrieve", _empty_retrieve)

    mock = _OllamaMock([
        _mk_msg(content="", tool_calls=None),
        _mk_stream(["resp"]),
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    events, _body = _post_chat("consulta con corpus vacío")
    assert events[-1][0] == "done"

    captured = capsys.readouterr()
    line = _last_chat_timing(captured.out)
    assert line is not None
    # Pre-fix: `confidence=-inf`. Post-fix: `confidence=0.000`.
    assert "confidence=-inf" not in line, f"regression, -inf en timing log: {line}"
    assert "confidence=0.000" in line, f"falta sanitize: {line}"

    # El SSE `done` event también debe reportar top_score finito.
    done_event = next(data for ev, data in events if ev == "done")
    top = done_event.get("top_score")
    assert top is not None
    import math
    assert not math.isinf(float(top)), f"done.top_score inf: {top}"
    assert not math.isnan(float(top)), f"done.top_score nan: {top}"


@pytest.mark.requires_ollama
def test_chat_endpoint_topic_shift_no_cache_key_unbound_error(chat_env, capsys, monkeypatch):
    """Regression 2026-04-21 (bug #3): cuando la /api/chat llega con
    `history` no vacío (skipping el bloque que computa `_cache_key`) y
    luego el topic-shift gate (~línea 3914) reasigna `history = []`, el
    PUT path entraba con `_cache_key` UnboundLocalError y loggeaba
    `[chat-cache] put failed: cannot access local variable '_cache_key'`.

    Este test reproduce el escenario exacto:
      1. `session_history` devuelve turns previos → rama `if not history:`
         que asigna `_cache_key` se SALTA.
      2. `detect_topic_shift` devuelve `(True, "person!")` → `history = []`.
      3. Stream completa normal; assert que NO aparece el UnboundLocalError
         en stdout y que el cache PUT stub NO se llamó (porque
         `_cache_key is None` bloquea el branch).
    """
    import rag
    # Un historia non-empty fuerza el path donde `_cache_key` NO se setea.
    monkeypatch.setattr(
        server_mod, "session_history",
        lambda s, window=6: [
            {"role": "user", "content": "pregunta anterior"},
            {"role": "assistant", "content": "respuesta anterior"},
        ],
    )
    # Fuerza topic-shift so el gate reasigna `history = []` post-check.
    monkeypatch.setattr(
        rag, "detect_topic_shift",
        lambda q, h, *, person_fired: (True, "person!"),
    )

    # Overrides el stub del fixture para capturar si se llama al PUT.
    put_calls: list = []
    monkeypatch.setattr(
        server_mod, "_chat_cache_put",
        lambda key, val: put_calls.append((key, val)),
    )

    # Flow típico sin tools: un decide-round vacío + streaming final.
    mock = _OllamaMock([
        _mk_msg(content="", tool_calls=None),
        _mk_stream(["nueva ", "respuesta"]),
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    events, _body = _post_chat("cuándo es el cumple de mi mamá")

    # 1. El stream tiene que completar OK.
    names = [ev for ev, _ in events]
    assert names[-1] == "done", f"stream no completó: {names}"

    # 2. El UnboundLocalError pre-fix se printeaba a stdout por el
    #    `print(f"[chat-cache] put failed: {...}", flush=True)`.
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "cannot access local variable '_cache_key'" not in combined, (
        "regression: UnboundLocalError en cache-put path reapareció"
    )
    assert "[chat-cache] put failed" not in combined, (
        f"cache-put falló por otra razón: {combined[-500:]}"
    )

    # 3. Con el guard `_cache_key is not None`, el PUT NO debe haberse
    #    ejecutado (el turno arrancó con history → no había key computada).
    assert put_calls == [], (
        f"PUT no debería haberse llamado (topic-shift sin key), got: {put_calls!r}"
    )

    # 4. El timing line confirma que el path de topic-shift se ejecutó.
    line = _last_chat_timing(captured.out)
    assert line is not None
    assert "topic_shift=person" in line, line


@pytest.mark.requires_ollama
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
    chat_env.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

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


# ── 9. Empty-retrieve bail must NOT swallow pre-router-covered queries ────


def _empty_retrieve_result(query: str = "x") -> dict:
    """`docs=[]` + `confidence=-inf` — shape de un retrieve vacío real."""
    return {
        "docs": [],
        "metas": [],
        "scores": [],
        "confidence": float("-inf"),
        "search_query": query,
        "filters_applied": {},
        "query_variants": [query],
        "vault_scope": ["default"],
    }


@pytest.mark.requires_ollama
def test_empty_retrieve_with_forced_tools_skips_bail(chat_env, monkeypatch):
    """Regression 2026-04-22 (Fer F.): "qué tengo para hacer esta semana?"
    matchea `reminders_due` + `calendar_ahead` via `_detect_tool_intent`,
    pero si el retrieve devuelve `docs=[]` (por cualquier razón transitoria
    o un corpus sin notas lexicalmente cercanas), el handler bailaba con
    `event: empty` + 'Sin resultados relevantes.' ANTES de llegar al pre-
    router de tools — el user veía "no hay nada" aunque los tools habrían
    listado los pendientes reales. Fix: si `docs=[]` pero el query matchea
    el pre-router, NO bailar — dejar pasar al tool loop.
    """
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _empty_retrieve_result(a[1] if len(a) >= 2 else "x"),
    )
    monkeypatch.setattr(
        server_mod, "_fetch_reminders_due", lambda anchor: {"dated": [], "undated": []},
    )
    monkeypatch.setattr(
        server_mod, "_fetch_calendar_ahead", lambda n, max_events=40: [],
    )

    mock = _OllamaMock([
        # Pre-router replaces CONTEXTO with tool output + the LLM tool-
        # deciding round runs (RAG_WEB_TOOL_LLM_DECIDE=1 en chat_env).
        _mk_msg(tool_calls=[]),
        _mk_stream(["listo"]),
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    events, _ = _post_chat("que tengo para hacer esta semana?")
    names = [ev for ev, _ in events]

    # CONTRATO: ningún `empty` event; el tool loop ejecutó.
    assert "empty" not in names, (
        f"empty bail disparó a pesar de tools forzados: {names}"
    )
    assert names[-1] == "done"

    # El pre-router disparó al menos reminders_due + calendar_ahead.
    tool_names_called = {
        data.get("name")
        for ev, data in events
        if ev == "status" and data.get("stage") == "tool"
    }
    assert "reminders_due" in tool_names_called
    assert "calendar_ahead" in tool_names_called


@pytest.mark.requires_ollama
def test_empty_retrieve_without_forced_tools_still_bails(chat_env, monkeypatch):
    """No-regression: cuando el retrieve viene vacío Y el query no matchea
    el pre-router, el bail `Sin resultados relevantes.` sigue firing. Solo
    queremos bypassear el bail cuando hay una vía clara de respuesta (tool).
    """
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _empty_retrieve_result(a[1] if len(a) >= 2 else "x"),
    )
    # Sin stub de ollama — el bail debe disparar antes de llegar al LLM.

    events, _ = _post_chat("pasame el output de mi script de ayer")
    names = [ev for ev, _ in events]
    assert "empty" in names
    # El mensaje del `empty` event es el canned.
    empty_payload = next(data for ev, data in events if ev == "empty")
    assert "Sin resultados" in empty_payload.get("message", "")


@pytest.mark.requires_ollama
def test_low_conf_bypass_with_forced_tools_skips_bypass(chat_env, monkeypatch):
    """Regression 2026-04-22 (mismo user report): si el retrieve devuelve
    docs pero confidence < CONFIDENCE_RERANK_MIN (0.015), el handler
    normalmente corta con 'No tengo info sobre "..." en tus notas.' +
    fallback cluster (Google/YouTube/Wiki). Cuando el query matchea el
    pre-router — "qué tengo para hacer esta semana?" → reminders_due +
    calendar_ahead — ese bypass esconde los datos reales que los tools
    habrían servido. Fix: si hay forced tools, NO bypasar, correr tools.
    """
    def _low_conf_retrieve(*a, **kw):
        r = _canned_retrieve_result(a[1] if len(a) >= 2 else "x")
        r["confidence"] = 0.005  # below CONFIDENCE_RERANK_MIN=0.015
        r["scores"] = [0.005, 0.003]
        return r
    monkeypatch.setattr(server_mod, "multi_retrieve", _low_conf_retrieve)
    monkeypatch.setattr(
        server_mod, "_fetch_reminders_due", lambda anchor: {"dated": [], "undated": []},
    )
    monkeypatch.setattr(
        server_mod, "_fetch_calendar_ahead", lambda n, max_events=40: [],
    )

    mock = _OllamaMock([
        _mk_msg(tool_calls=[]),
        _mk_stream(["ok"]),
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    events, _ = _post_chat("que tengo para hacer esta semana?")

    # CONTRATO: el `done` event NO lleva `low_conf_bypass: true` cuando hay
    # forced tools — el handler debe haber seguido al tool loop en lugar
    # de servir el template "No tengo info sobre '...' en tus notas.".
    done_event = next(data for ev, data in events if ev == "done")
    assert done_event.get("low_conf_bypass") is not True, (
        f"low_conf_bypass firing a pesar de forced tools: {done_event!r}"
    )

    # Verifica que los tools del pre-router ejecutaron.
    tool_names_called = {
        data.get("name")
        for ev, data in events
        if ev == "status" and data.get("stage") == "tool"
    }
    assert "reminders_due" in tool_names_called
    assert "calendar_ahead" in tool_names_called


@pytest.mark.requires_ollama
def test_low_conf_bypass_without_forced_tools_still_fires(chat_env, monkeypatch):
    """No-regression: cuando no hay forced tools, el bypass sigue firing
    (template canned, ahorro de 5-8s de LLM cold prefill sobre queries
    donde el vault claramente no tiene data)."""
    def _low_conf_retrieve(*a, **kw):
        r = _canned_retrieve_result(a[1] if len(a) >= 2 else "x")
        r["confidence"] = 0.003
        r["scores"] = [0.003, 0.002]
        return r
    monkeypatch.setattr(server_mod, "multi_retrieve", _low_conf_retrieve)

    events, _ = _post_chat("qué película ver esta noche")
    done_event = next(data for ev, data in events if ev == "done")
    assert done_event.get("low_conf_bypass") is True, (
        f"bypass debió firing sin forced tools: {done_event!r}"
    )


# ── 10. Source-specific intent hint — bug report 2026-04-24 ──────────────


@pytest.mark.requires_ollama
def test_source_intent_hint_injected_when_pre_router_fires_gmail(
    chat_env, monkeypatch,
):
    """Regression 2026-04-24 (Fer F. user report): "cuales son mis ultimos
    mails?" enganchó `gmail_recent` en el pre-router (tras fix de regex
    plurals) pero el LLM seguía respondiendo sobre WhatsApp/notas como si
    fueran la respuesta principal. Fix: inyectar un 3er system msg
    turn-scoped (`_build_source_intent_hint`) que le dice al LLM "anclá
    en '### Mails', si está vacío decilo ANTES de fallback".

    Este test verifica que cuando una query dispara `gmail_recent` vía
    pre-router, el `messages=` que recibe `ollama.chat` incluye un system
    message adicional con la frase 'INTENCIÓN EXPLÍCITA' y el texto de
    fallback esperado ('Busqué en tus mails/correos').
    """
    monkeypatch.setattr(
        tools_mod, "_fetch_gmail_evidence",
        lambda now: {"unread_count": 0, "awaiting_reply": [], "starred": []},
    )

    mock = _OllamaMock([
        _mk_msg(tool_calls=[]),         # tool-deciding round, no extra tools
        _mk_stream(["ok"]),             # final streaming
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    events, _ = _post_chat("cuales son mis ultimos mails?")
    assert events, "SSE stream vacío"

    # El pre-router debe haber ejecutado `gmail_recent`.
    tool_names_called = {
        data.get("name")
        for ev, data in events
        if ev == "status" and data.get("stage") == "tool"
    }
    assert "gmail_recent" in tool_names_called, (
        f"pre-router NO disparó gmail_recent (bug regex plurals vuelto?); "
        f"tools ejecutados: {tool_names_called}"
    )

    # Primer call a ollama.chat = tool-deciding round.
    first_call = mock.calls[0]
    messages = first_call.get("messages") or []
    system_msgs = [m for m in messages if m.get("role") == "system"]
    # Default = 2 system msgs (_WEB_SYSTEM_PROMPT + _WEB_TOOL_ADDENDUM).
    # Con source-specific intent agregamos el 3er: el hint.
    assert len(system_msgs) >= 3, (
        f"Esperaba ≥3 system messages (hint incluido), got {len(system_msgs)}"
    )
    hint_msg = system_msgs[-1]["content"]
    assert "INTENCIÓN EXPLÍCITA" in hint_msg, (
        f"El 3er system msg no parece el hint source-specific: "
        f"{hint_msg[:200]!r}"
    )
    assert "tus mails/correos" in hint_msg
    # El hint debe decirle al LLM que extraiga asuntos de las notas
    # `03-Resources/Gmail/` en vez de hablar de "fuentes" abstractas
    # (user feedback iter 3).
    assert "03-Resources/Gmail" in hint_msg
    assert "asunto" in hint_msg.lower()
    assert "### Mails" in hint_msg
    # Frase canned para empty-state en vez del vago "te dejo otras fuentes".
    assert "No encontré mails recientes en tu corpus" in hint_msg


@pytest.mark.requires_ollama
def test_source_intent_hint_omitted_when_no_source_specific_tools(
    chat_env, monkeypatch,
):
    """No-regression: queries que NO disparan tools source-specific no
    deben pagar la inyección del hint (prefix cache se mantiene warm).
    Query "hola" no matchea ningún regex del pre-router → solo 2 system
    msgs en el stack.
    """
    mock = _OllamaMock([
        _mk_msg(tool_calls=[]),
        _mk_stream(["hola!"]),
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    # "dame info sobre grecia" — query semántica de vault sin pre-router match.
    events, _ = _post_chat("dame info sobre grecia")
    if not mock.calls:
        # Metachat bypass o low-conf bypass pudo firing — en ese caso el
        # test no aplica (no hubo call a ollama). La ausencia de hint se
        # verifica vía ausencia del msg en tool_messages, que requiere
        # una call real.
        pytest.skip("sin llamada a ollama.chat (bypass path)")

    first_call = mock.calls[0]
    messages = first_call.get("messages") or []
    system_msgs = [m for m in messages if m.get("role") == "system"]
    # Esperamos exactamente 2 (prompt + addendum), NO 3.
    assert len(system_msgs) == 2, (
        f"hint se inyectó sin pre-router match (no debería): "
        f"{len(system_msgs)} system msgs"
    )
    assert not any("INTENCIÓN EXPLÍCITA" in m["content"] for m in system_msgs)


# ── 10.5. CONTEXTO preservation cuando todos los tools del pre-router vuelven vacíos ─


@pytest.mark.requires_ollama
def test_empty_tool_output_preserves_vault_context(chat_env, monkeypatch):
    """Regression 2026-04-24 iteración 2 (Fer F.): el user preguntó
    "cuales son mis ultimos mails?", el pre-router disparó `gmail_recent`
    (tras fix plurales), pero el tool vino vacío (no había starred ni
    awaiting-reply). La lógica original REEMPLAZABA el CONTEXTO entero
    con el tool output vacío → descartaba el retrieve del vault (que
    podía incluir `03-Resources/Gmail/YYYY-MM-DD.md` con mails indexados)
    → el LLM respondía "te dejo otras fuentes que podrían ayudarte" de
    forma genérica, sin material para resumir.

    Fix: cuando `_is_empty_tool_output` es True para TODOS los
    `_forced_results`, preservamos el CONTEXTO original del vault y
    agregamos el tool output como sección "CONSULTAS EN VIVO (todas
    vacías)". Así el LLM tiene las notas del vault disponibles como
    fallback concreto.

    Este test stubbea `gmail_recent` para que devuelva vacío, deja el
    `multi_retrieve` canned (que retorna 2 notas del vault), y verifica
    que el `user` msg enviado al LLM contiene BOTH el CONTEXTO del vault
    Y el marker de CONSULTAS EN VIVO.
    """
    # gmail_recent → empty.
    monkeypatch.setattr(
        tools_mod, "_fetch_gmail_evidence",
        lambda now: {"unread_count": 0, "awaiting_reply": [], "starred": []},
    )

    mock = _OllamaMock([
        _mk_msg(tool_calls=[]),   # tool-deciding round
        _mk_stream(["ok"]),       # streaming final
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    events, _ = _post_chat("cuales son mis ultimos mails?")
    assert mock.calls, "se esperaba al menos una call a ollama.chat"

    # En la PRIMERA call (tool-deciding round, pre-router ya inyectó
    # el user_content nuevo) el user msg debería incluir:
    #   1. Al menos una marca del CONTEXTO del vault (canned: "doc 1 body"
    #      o "doc 2 body" o "01-Projects/a.md").
    #   2. El marker "CONSULTAS EN VIVO" del tool empty.
    #   3. La sección rendered "### Mails" con "_Sin mails pendientes._".
    first_call = mock.calls[0]
    messages = first_call.get("messages") or []
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assert user_msgs, "no hay user msg en las messages del LLM"
    user_content = user_msgs[-1]["content"]

    # CONTEXTO original del vault preservado.
    assert "CONTEXTO:" in user_content, (
        f"CONTEXTO del vault no preservado cuando tool output es empty. "
        f"user_content[:300]={user_content[:300]!r}"
    )
    # Alguna evidencia de las notas canneadas por _canned_retrieve_result.
    # Los docs canned tienen path "01-Projects/a.md" + "02-Areas/b.md"
    # y bodies "doc 1 body" + "doc 2 body".
    assert ("doc 1 body" in user_content) or ("doc 2 body" in user_content) or (
        "01-Projects/a.md" in user_content or "02-Areas/b.md" in user_content
    ), f"contenido del vault no aparece en el user msg: {user_content[:500]!r}"

    # Marker de tools vacíos + header de gmail.
    assert "CONSULTAS EN VIVO" in user_content, (
        f"marker de 'consultas en vivo vacías' no aparece: "
        f"{user_content[:500]!r}"
    )
    assert "### Mails" in user_content
    assert "_Sin mails pendientes._" in user_content


@pytest.mark.requires_ollama
def test_nonempty_tool_output_replaces_vault_context(chat_env, monkeypatch):
    """No-regresión: cuando el pre-router dispara un tool con DATA (no
    empty), el comportamiento original se mantiene — el CONTEXTO se
    reemplaza con el tool output y el vault se descarta (era la decisión
    de diseño original: tool output es authoritative > vault retrieve).
    """
    # gmail_recent → con data (starred thread).
    monkeypatch.setattr(
        tools_mod, "_fetch_gmail_evidence",
        lambda now: {
            "unread_count": 3,
            "awaiting_reply": [],
            "starred": [{
                "subject": "Proyecto Alpha",
                "from": "juan@example.com",
                "snippet": "necesito tu feedback",
                "thread_id": "tid-1",
                "internal_date_ms": 1_730_000_000_000,
            }],
        },
    )

    mock = _OllamaMock([
        _mk_msg(tool_calls=[]),
        _mk_stream(["ok"]),
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    events, _ = _post_chat("revisar mails")
    assert mock.calls, "se esperaba al menos una call a ollama.chat"

    first_call = mock.calls[0]
    messages = first_call.get("messages") or []
    user_msgs = [m for m in messages if m.get("role") == "user"]
    user_content = user_msgs[-1]["content"]

    # Flag del comportamiento "authoritative": el prefijo hace explícita
    # la sustitución ("no del vault").
    assert "CONTEXTO (datos en vivo, no del vault)" in user_content
    # El marker de "CONSULTAS EN VIVO todas vacías" NO debe aparecer
    # porque el tool sí trajo data.
    assert "CONSULTAS EN VIVO" not in user_content
    # La retrieve canned no debe estar — el CONTEXTO fue reemplazado.
    assert "doc 1 body" not in user_content
    assert "doc 2 body" not in user_content
    # La data del tool sí aparece.
    assert "Proyecto Alpha" in user_content


# ── 10.7. `source_specific` flag en el done event ─────────────────────────


@pytest.mark.requires_ollama
def test_done_event_flags_source_specific_true_for_gmail_query(
    chat_env, monkeypatch,
):
    """UI contract (2026-04-24, Fer F. user report iter 4): el `event: done`
    debe llevar `source_specific: true` cuando el pre-router disparó un
    tool mapeado a una fuente concreta del user (gmail_recent /
    calendar_ahead / reminders_due). El frontend usa el flag para apagar
    el CTA "↗ buscar en internet" (Google no sabe qué mails pendientes
    tengo yo) + el fallback YouTube.
    """
    monkeypatch.setattr(
        tools_mod, "_fetch_gmail_evidence",
        lambda now: {"unread_count": 0, "awaiting_reply": [], "starred": []},
    )
    mock = _OllamaMock([
        _mk_msg(tool_calls=[]),
        _mk_stream(["ok"]),
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    events, _ = _post_chat("cuales son mis ultimos mails?")
    done_payload = next(data for ev, data in events if ev == "done")
    assert done_payload.get("source_specific") is True, (
        f"done event debe llevar source_specific=true cuando gmail_recent "
        f"fired: {done_payload!r}"
    )


@pytest.mark.requires_ollama
def test_done_event_flags_source_specific_false_for_generic_query(
    chat_env, monkeypatch,
):
    """No-regresión: queries que no matchean ningún tool del pre-router
    emiten `source_specific: false` (o ausencia). El CTA "buscar en
    internet" sigue activo para esos turns si la confianza es baja.
    """
    mock = _OllamaMock([
        _mk_msg(tool_calls=[]),
        _mk_stream(["respuesta"]),
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    events, _ = _post_chat("dame info sobre grecia")
    done_payload = next(
        (data for ev, data in events if ev == "done"),
        None,
    )
    if done_payload is None:
        # Bypass path → skippable, el flag no aplica.
        pytest.skip("bypass path: no done event")
    assert done_payload.get("source_specific") is not True, (
        f"source_specific=true en query que no matchea pre-router: "
        f"{done_payload!r}"
    )


@pytest.mark.requires_ollama
def test_done_event_flags_source_specific_false_for_weather_only(
    chat_env, monkeypatch,
):
    """El pre-router dispara `weather` para queries de clima, pero weather
    NO está en `_SOURCE_INTENT_META` (no tiene sentido "busqué el clima
    y no encontré" — weather es passthrough de data externa, no una
    "fuente" del user). Por lo tanto el `source_specific` flag queda en
    false aunque el pre-router haya fired.
    """
    monkeypatch.setattr(
        tools_mod, "_agent_tool_weather",
        lambda loc=None: "Santa Fe: 22°C, despejado",
    )
    mock = _OllamaMock([
        _mk_msg(tool_calls=[]),
        _mk_stream(["ok"]),
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    # "va a llover?" matchea solo el regex de weather (`llov`), no toca
    # `_PLANNING_PAT` (hoy/mañana/semana) que dispararía calendar_ahead
    # + reminders_due — y esos sí son source-specific. Query intencional-
    # mente sin tokens temporales para aislar weather.
    events, _ = _post_chat("va a llover?")
    done_payload = next(data for ev, data in events if ev == "done")
    # Confirmamos que el pre-router efectivamente fired weather (y nada
    # más), si no el test no verifica lo que prometió verificar.
    tool_names_called = [
        data.get("name")
        for ev, data in events
        if ev == "status" and data.get("stage") == "tool"
    ]
    assert tool_names_called == ["weather"], (
        f"query de weather aislada debería disparar SOLO weather, "
        f"pero el pre-router lanzó: {tool_names_called}"
    )
    assert done_payload.get("source_specific") is not True, (
        f"weather-only query NO debería flagear source_specific=true: "
        f"{done_payload!r}"
    )


# ── 10.8. LLM tool-decide safety-net fallback (iter 7) ────────────────────


@pytest.fixture
def chat_env_prod(monkeypatch):
    """Variante del fixture `chat_env` SIN `RAG_WEB_TOOL_LLM_DECIDE=1`,
    replicando el comportamiento de producción donde el LLM tool-decide
    está apagado por default. Así podemos testear el gate de fallback
    (iter 7): cuando el pre-router no matcheó Y el retrieve vino con
    conf<0.10, el LLM tool-decide debería prenderse solo como safety-net.
    """
    # No setear RAG_WEB_TOOL_LLM_DECIDE — dejamos el default (off).
    monkeypatch.delenv("RAG_WEB_TOOL_LLM_DECIDE", raising=False)
    # Reset rate-limit bucket (ver comentario en `chat_env` para detalle).
    server_mod._CHAT_BUCKETS.clear()
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


@pytest.mark.requires_ollama
def test_llm_tool_decide_fires_when_pre_router_misses_and_vault_weak(
    chat_env_prod, monkeypatch, capsys,
):
    """Iter 7 regression (user question 2026-04-24): "¿no debería el LLM
    intervenir acá? qué pasa si hay otra palabra que no matchea con el
    pre-router?" — sí, ahora interviene como safety-net cuando regex +
    vault ambos fallan.

    Este test usa una query ("qué correspondencia tengo?") que NO matchea
    ningún pattern del pre-router, stubbea multi_retrieve para devolver
    confidence baja (< CONFIDENCE_DEEP_THRESHOLD=0.10), y verifica que
    el LLM tool-decide round efectivamente corrió (mock recibió al menos
    la call no-streaming del tool-deciding phase).
    """
    # Retrieve con confidence baja (vault really failed: < 0.10).
    def _low_conf_retrieve(*a, **kw):
        r = _canned_retrieve_result(a[1] if len(a) >= 2 else "x")
        r["confidence"] = 0.05  # debajo del threshold
        r["scores"] = [0.05, 0.04]
        return r
    monkeypatch.setattr(server_mod, "multi_retrieve", _low_conf_retrieve)

    mock = _OllamaMock([
        _mk_msg(tool_calls=[]),   # tool-deciding call (no tool picked)
        _mk_stream(["fallback response"]),
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    # Query intencionalmente sin keywords del pre-router — "correspondencia"
    # no está en los regex de `_TOOL_INTENT_RULES`.
    events, _ = _post_chat("qué correspondencia nueva tengo?")

    # Evidencia: el mock recibió ≥2 calls (tool-deciding + streaming).
    # Sin el fallback sería solo 1 (solo streaming).
    assert len(mock.calls) >= 2, (
        f"LLM tool-decide round NO corrió a pesar del fallback "
        f"(pre-router miss + conf<0.10). Calls a ollama: {len(mock.calls)}"
    )
    # Confirmá que el tool-deciding call tiene `stream=False` y
    # `tools=` (es el shape del decide round, no el final stream).
    decide_call = mock.calls[0]
    assert decide_call.get("stream") is False
    assert decide_call.get("tools"), "tool-deciding call sin tools= kwarg"

    # Log explícito del fallback gate — lo emitimos para debug + tunear.
    stdout = capsys.readouterr().out
    assert "[chat-llm-fallback]" in stdout, (
        f"falta log [chat-llm-fallback] cuando el gate se activa: "
        f"stdout tail={stdout[-500:]!r}"
    )


@pytest.mark.requires_ollama
def test_llm_tool_decide_skipped_when_pre_router_matches(
    chat_env_prod, monkeypatch,
):
    """No-regresión: si el pre-router SI matcheó (p.ej. gmail_recent
    para "últimos mails"), el LLM tool-decide sigue skipped — no
    necesitamos pagar la latencia porque el regex ya dio data. Path
    rápido original preservado.
    """
    # Stub gmail_recent para evitar llamadas a la API real.
    monkeypatch.setattr(
        tools_mod, "_fetch_gmail_evidence",
        lambda now: {"unread_count": 2, "awaiting_reply": [], "starred": [],
                     "recent": [{"subject": "x", "from": "a@b.com",
                                 "snippet": "", "thread_id": "t1",
                                 "internal_date_ms": 1_700_000_000_000}]},
    )
    # Retrieve irrelevante — cualquier valor vale porque el pre-router
    # matchea primero. Conf alta o baja no cambia el gate cuando hay
    # forced tools.
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve_result(a[1] if len(a) >= 2 else "x"),
    )

    mock = _OllamaMock([
        _mk_stream(["respuesta basada en gmail_recent"]),
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    events, _ = _post_chat("mis últimos mails")

    # Con pre-router match + sin env var ni propose intent, el LLM
    # tool-decide NO corre → solo hay 1 call a ollama (el streaming).
    assert len(mock.calls) == 1, (
        f"LLM tool-decide round corrió cuando no debía (pre-router "
        f"matcheó → path rápido esperado). Calls: {len(mock.calls)}"
    )
    # Y esa call es el streaming, no el tool-decide.
    assert mock.calls[0].get("stream") is True


@pytest.mark.requires_ollama
def test_llm_tool_decide_skipped_when_vault_strong(
    chat_env_prod, monkeypatch,
):
    """No-regresión: si el vault devolvió confidence alta (≥ 0.10), el
    LLM tool-decide sigue skipped aunque el pre-router no matcheó — el
    vault respondió bien, no necesitamos un fallback.
    """
    def _strong_retrieve(*a, **kw):
        r = _canned_retrieve_result(a[1] if len(a) >= 2 else "x")
        r["confidence"] = 0.85
        return r
    monkeypatch.setattr(server_mod, "multi_retrieve", _strong_retrieve)

    mock = _OllamaMock([
        _mk_stream(["respuesta del vault"]),
    ])
    monkeypatch.setattr(server_mod.ollama, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)
    monkeypatch.setattr(server_mod._OLLAMA_TOOL_CLIENT, "chat", mock)

    # Query que no matchea pre-router pero el vault tiene info buena.
    events, _ = _post_chat("hablame sobre Grecia")

    # Sin pre-router + vault strong + sin env + sin propose → skip LLM
    # tool-decide. Solo 1 call: el streaming final.
    assert len(mock.calls) == 1, (
        f"LLM tool-decide corrió cuando vault strong (conf=0.85): "
        f"{len(mock.calls)} calls"
    )
    assert mock.calls[0].get("stream") is True


# ── 11. Metachat SSE: `metachat: true` flag in sources + done ─────────────


def test_metachat_sources_event_flags_metachat_true():
    """UI contract (2026-04-22): el `event: sources` de un reply de
    metachat ("Hola") debe llevar `metachat: true` para que el frontend
    skippee el link inline "↗ buscar en internet" (que se mostraba por
    `sources.length === 0 && confidence === null`). Sin la flag la UX
    era inconsistente — el canned reply era correcto pero aparecía un
    CTA de Google absurdo al costado.
    """
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"question": "Hola", "vault_scope": None, "session_id": None},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)

    sources_event = next(data for ev, data in events if ev == "sources")
    assert sources_event.get("metachat") is True, (
        f"sources event falta flag metachat=true: {sources_event!r}"
    )
    # El done event también lo marca (documentación defensiva).
    done_event = next(data for ev, data in events if ev == "done")
    assert done_event.get("metachat") is True


# ── WhatsApp deep-links: source citations + proposal recipient ────────────


def test_app_js_has_whatsapp_href_helper():
    """`waHref()` convierte un `whatsapp://<jid>/<msg_id>` o JID crudo a
    `https://wa.me/<phone>` para 1:1 chats. Universal link funciona en
    iOS/Android (abre la app de WhatsApp en el chat correspondiente)
    y en desktop redirige a web.whatsapp.com.

    Devuelve "" para grupos (`@g.us`) porque WhatsApp no expone
    deep-link a grupos."""
    from pathlib import Path
    js = (Path(__file__).resolve().parent.parent /
          "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "function waHref(uri)" in js, (
        "waHref helper missing — sin esto las citations whatsapp:// "
        "del chat no se pueden abrir en la app"
    )
    # Lista negativa: groups no deben tener wa.me link.
    assert "@g.us" in js
    # Smoke del prefijo wa.me.
    assert "https://wa.me/" in js


def test_app_js_renders_whatsapp_sources_as_wa_me_links():
    """En `appendSources()`, las fuentes con `s.file` empezando con
    `whatsapp://` deben generar un `<a href="https://wa.me/...">` para
    1:1 chats. Para grupos, span plano (no engañar al user con un link
    que abre la home de WhatsApp en lugar del grupo)."""
    from pathlib import Path
    js = (Path(__file__).resolve().parent.parent /
          "web" / "static" / "app.js").read_text(encoding="utf-8")
    # appendSources debe distinguir whatsapp:// de https:// y de paths
    # vault. Buscar el flag `isWA` o similar.
    assert "isWA" in js or "whatsapp://" in js
    # waHref se invoca dentro de appendSources.
    assert "waHref(s.file)" in js, (
        "appendSources no llama a waHref → las citations WA siguen "
        "rotas (apuntan a obsidian://whatsapp:// que no abre nada)"
    )


def test_app_js_makes_whatsapp_proposal_recipient_clickable():
    """En la tarjeta de propuesta de WhatsApp, el name del recipient
    ("Para: <name>") debe ser clickable y abrir el chat en la app de
    WhatsApp para que el user pueda verificar la conversación antes
    de mandar."""
    from pathlib import Path
    js = (Path(__file__).resolve().parent.parent /
          "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "waHref(fields.jid" in js, (
        "appendWhatsAppProposal no genera link en el recipient — el "
        "name de 'Para: <X>' debería ser clickable a wa.me/<phone>"
    )


# ── Contacts popover keyboard nav (verified manualmente con Playwright) ───


def test_app_js_wires_arrowdown_arrowup_in_contacts_popover():
    """↑/↓ deben mover el highlight entre contactos. Verificado a mano
    con Playwright — el handler vive en el global keydown listener
    cuando `contactsPopover.hidden === false`. Si alguien refactorea
    los handlers y quita esto, el user pierde la nav por teclado."""
    from pathlib import Path
    js = (Path(__file__).resolve().parent.parent /
          "web" / "static" / "app.js").read_text(encoding="utf-8")
    # El handler debe chequear contactsPopover.hidden y actualizar
    # contactsIndex con ArrowDown/Up.
    assert "contactsPopover.hidden" in js
    assert "ArrowDown" in js and "ArrowUp" in js
    # Búsqueda más estricta: el handler de ArrowDown debe incrementar
    # contactsIndex (cíclicamente vía modulo).
    assert "(contactsIndex + 1) % contactsItems.length" in js, (
        "ArrowDown handler en contacts popover roto/missing — el user "
        "no puede navegar la lista con teclado"
    )
    assert "contactsIndex - 1 + contactsItems.length" in js, (
        "ArrowUp handler en contacts popover roto/missing"
    )


def test_app_js_wires_tab_and_enter_to_pick_contact():
    """Tab y Enter deben seleccionar el contacto highlighted. Tab es la
    convención de autocomplete; Enter es la que más espera el user
    (símil a Gmail / Slack)."""
    from pathlib import Path
    js = (Path(__file__).resolve().parent.parent /
          "web" / "static" / "app.js").read_text(encoding="utf-8")
    # Buscar el bloque del handler que matchea Tab o Enter dentro del
    # contacts popover y llama a pickContact.
    # Heurística: ambas keys deben aparecer en el mismo `if`.
    assert 'e.key === "Tab" || (e.key === "Enter"' in js, (
        "Tab + Enter no pican el contacto highlighted — el user tiene "
        "que usar el mouse aunque ya esté con foco en el input"
    )
    assert "pickContact(pick)" in js


def test_app_js_wires_escape_to_close_contacts_popover():
    """Esc debe cerrar el popover sin perder lo tipeado en el input.
    Convención universal (Gmail / Slack / Notion / VSCode)."""
    from pathlib import Path
    js = (Path(__file__).resolve().parent.parent /
          "web" / "static" / "app.js").read_text(encoding="utf-8")
    # Buscar Escape dentro del contacts popover handler.
    # Heurística: contactsPopover y Escape en proximidad.
    # Buscamos la 2da ocurrencia (la 1ra está en updateContactsPopover,
    # la 2da en el keydown handler). Heurística más estricta: buscar
    # `contactsPopover.hidden && contactsItems.length` que es el guard
    # del keydown handler.
    cidx = js.find('!contactsPopover.hidden && contactsItems.length')
    assert cidx > 0, "keydown handler de contacts popover no encontrado"
    block = js[cidx:cidx + 1500]
    assert 'Escape' in block, (
        "Esc no cierra el contacts popover — el user queda atrapado "
        "con el popover abierto si cambia de idea"
    )
    assert 'hideContactsPopover()' in block
