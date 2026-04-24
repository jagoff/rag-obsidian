"""Tests del wiring del semantic cache en /api/chat (post 2026-04-23).

Segundo layer después del LRU exact-string (5min TTL in-memory):
`semantic_cache_lookup` contra `rag_response_cache` SQL con embedding +
corpus_hash. Hits replayean SSE con `cache_layer=semantic` y skip completo
del pipeline LLM. Misses corren pipeline normal y storean ambas layers.

Cubre:
1. LRU miss → semantic hit → replay SSE + no ollama.chat + done event
   con cache_layer="semantic".
2. LRU miss → semantic miss → pipeline corre + SEMANTIC PUT al final.
3. Semantic skipped con history (follow-up turns dependen del contexto).
4. Semantic skipped con is_propose_intent (create actions, no queries).
5. Semantic skipped con multi-vault (corpus_hash es per-col).
6. Semantic lookup exception → fallback gracefull al pipeline.
7. cache_probe payload en extra_json cuando hay miss semántico.
8. Sources sintetizados desde paths cuando hay hit semántico (shape).
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient


from web import server as server_mod
from web.server import app
import rag


# ── SSE helpers ──────────────────────────────────────────────────────────────

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


class _OllamaMock:
    """Tracks every invocation — a semantic hit replay must show `calls == []`."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError(
                "OllamaMock: unscripted call — semantic hit path must skip LLM."
            )
        resp = self.responses.pop(0)
        if kwargs.get("stream"):
            return iter(resp)
        return resp


def _canned_retrieve(confidence=0.5, n=2):
    return {
        "docs": [f"doc {i}" for i in range(n)],
        "metas": [
            {"file": f"02-Areas/note{i}.md", "note": f"note{i}",
             "folder": "02-Areas"}
            for i in range(n)
        ],
        "scores": [confidence] * n,
        "confidence": confidence,
        "search_query": "q",
        "filters_applied": {},
        "query_variants": ["q"],
        "vault_scope": ["default"],
    }


def _make_semantic_hit(response="cached response",
                       paths=("02-Areas/ikigai.md",),
                       scores=(0.88,),
                       cosine=0.96,
                       top_score=1.15,
                       age_seconds=120.0):
    return {
        "id": 42,
        "question": "q",
        "response": response,
        "paths": list(paths),
        "scores": list(scores),
        "top_score": top_score,
        "intent": "semantic",
        "cosine": cosine,
        "cached_ts": 1_700_000_000.0,
        "age_seconds": age_seconds,
    }


@pytest.fixture
def chat_env(monkeypatch):
    """Monkeypatches compartidos. Default: LRU miss + pipeline-friendly retrieve."""
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve(confidence=0.5),
    )
    monkeypatch.setattr(server_mod, "_ollama_alive", lambda timeout=2.0: True)
    monkeypatch.setattr(
        server_mod, "_ollama_chat_probe", lambda timeout_s=6.0: True,
    )
    monkeypatch.setattr(
        server_mod, "_fetch_whatsapp_unread", lambda *a, **kw: [],
    )
    monkeypatch.setattr(rag, "_match_mentions_in_query", lambda q: [])
    monkeypatch.setattr(rag, "build_person_context", lambda q: None)
    monkeypatch.setattr(
        server_mod, "_spawn_conversation_writer", lambda **kw: None,
    )
    monkeypatch.setattr(server_mod, "save_session", lambda sess: None)
    monkeypatch.setattr(server_mod, "append_turn", lambda s, t: None)
    _events: list[dict] = []
    monkeypatch.setattr(
        server_mod, "log_query_event", lambda ev: _events.append(ev),
    )
    # LRU miss por default — tests que quieren LRU hit lo re-monkeypatchean.
    monkeypatch.setattr(server_mod, "_chat_cache_get", lambda key: None)
    monkeypatch.setattr(server_mod, "_chat_cache_put", lambda key, val: None)
    monkeypatch.setattr(server_mod, "_is_tasks_query", lambda q: False)
    # Limpiar el bucket del rate limiter — TestClient usa IP fija, y
    # las N tests consecutivos llenan el bucket de 30/60s y a partir
    # del test 31 devuelven 429.
    monkeypatch.setattr(
        server_mod, "_CHAT_BUCKETS",
        __import__("collections").defaultdict(list),
    )
    # Evitar que short-strings caigan en el gate "degenerate" (`<2 alnum
    # chars → canned reply`) antes de llegar al semantic lookup.
    monkeypatch.setattr(server_mod, "_is_degenerate_query", lambda q: False)
    monkeypatch.setattr(server_mod, "_detect_metachat_intent", lambda q: False)
    monkeypatch.setattr(server_mod, "_detect_propose_intent", lambda q: False)
    # Por default NO hay hit semántico — tests lo overridean.
    monkeypatch.setattr(
        rag, "semantic_cache_lookup",
        lambda *a, **kw: (None, {"result": "miss", "reason": "corpus_mismatch",
                                  "top_cosine": None, "candidates": 0,
                                  "skipped_stale": 0, "skipped_ttl": 0})
        if kw.get("return_probe") else None,
    )
    # Stub embed + corpus_hash para evitar el ollama roundtrip + vault scan.
    _fake_emb = np.zeros(1024, dtype="float32")
    _fake_emb[0] = 1.0
    _fake_emb = (_fake_emb / np.linalg.norm(_fake_emb)).tolist()
    monkeypatch.setattr(rag, "embed", lambda texts: [_fake_emb])
    monkeypatch.setattr(rag, "_corpus_hash_cached", lambda col: "HWEB")
    monkeypatch.setattr(rag, "_semantic_cache_enabled", lambda: True)

    class _FakeCol:
        def count(self): return 42
    monkeypatch.setattr(rag, "get_db_for", lambda p: _FakeCol())
    # Track semantic store calls.
    _sem_stores: list[dict] = []
    def _spy_sem_store(*args, **kwargs):
        _sem_stores.append(kwargs)
        return True
    monkeypatch.setattr(rag, "semantic_cache_store", _spy_sem_store)
    monkeypatch.log_events = _events  # type: ignore[attr-defined]
    monkeypatch.sem_stores = _sem_stores  # type: ignore[attr-defined]
    # Ollama stream LLM returns a non-empty body so the PUT path fires.
    _ollama_mock = _OllamaMock(
        responses=[[SimpleNamespace(message=SimpleNamespace(content="respuesta generada"))]],
    )
    monkeypatch.setattr(server_mod.ollama, "chat", _ollama_mock)
    monkeypatch.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", _ollama_mock)
    monkeypatch.ollama_mock = _ollama_mock  # type: ignore[attr-defined]
    return monkeypatch


def _post_chat(question: str = "qué es ikigai", session_id=None,
               vault_scope=None) -> tuple[list[tuple[str, dict]], str]:
    client = TestClient(app)
    payload = {"question": question, "vault_scope": vault_scope}
    if session_id:
        payload["session_id"] = session_id
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200, resp.text
    return _parse_sse(resp.text), resp.text


# ── 1. Semantic hit → replay, no ollama.chat ────────────────────────────────


def test_semantic_hit_skips_llm_and_replays_sse(chat_env):
    """LRU miss + semantic hit → SSE replay completo + 0 llamadas a ollama.chat."""
    hit = _make_semantic_hit(
        response="Ikigai es tu razón de ser.",
        paths=["03-Resources/ikigai.md", "03-Resources/flow.md"],
        scores=[0.92, 0.78],
        cosine=0.98,
        top_score=1.25,
        age_seconds=600.0,
    )

    def _spy_lookup(emb, corpus_hash, *, return_probe=False, **kw):
        probe = {"result": "hit", "reason": "match", "top_cosine": 0.98,
                 "candidates": 1, "skipped_stale": 0, "skipped_ttl": 0}
        if return_probe:
            return (hit, probe)
        return hit
    chat_env.setattr(rag, "semantic_cache_lookup", _spy_lookup)

    events, _ = _post_chat("qué es ikigai")

    # 0 llamadas a ollama.chat — el semantic hit NO puede correr el LLM.
    assert chat_env.ollama_mock.calls == [], (
        f"semantic hit must skip ollama.chat, got {chat_env.ollama_mock.calls}"
    )
    # El SSE debe tener: session, status=cached, sources, tokens, done.
    event_types = [ev for ev, _ in events]
    assert "session" in event_types
    assert any(ev == "status" and data.get("stage") == "cached"
               for ev, data in events)
    assert any(ev == "sources" for ev, _ in events)
    token_deltas = [data.get("delta", "") for ev, data in events
                    if ev == "token"]
    assert "".join(token_deltas) == "Ikigai es tu razón de ser."
    done_events = [data for ev, data in events if ev == "done"]
    assert done_events, "missing `done` event"
    assert done_events[-1].get("cached") is True
    assert done_events[-1].get("cache_layer") == "semantic"
    assert done_events[-1].get("llm_ms") == 0
    assert done_events[-1].get("retrieve_ms") == 0


def test_semantic_hit_synthesizes_sources_from_paths(chat_env):
    """Sources items en SSE se sintetizan desde paths + scores del hit."""
    hit = _make_semantic_hit(
        paths=["01-Projects/a.md", "02-Areas/Coaching/b.md"],
        scores=[0.91, 0.76],
    )
    chat_env.setattr(
        rag, "semantic_cache_lookup",
        lambda *a, **kw: (hit, {"result": "hit"}) if kw.get("return_probe") else hit,
    )
    events, _ = _post_chat("q")
    sources_events = [data for ev, data in events if ev == "sources"]
    assert sources_events, "missing sources event"
    items = sources_events[0]["items"]
    assert len(items) == 2
    assert items[0]["file"] == "01-Projects/a.md"
    assert items[0]["note"] == "a"
    assert items[0]["folder"] == "01-Projects"
    assert items[0]["score"] == pytest.approx(0.91, rel=1e-2)
    assert "bar" in items[0]
    # Segundo item preserva el folder full.
    assert items[1]["folder"] == "02-Areas/Coaching"
    assert items[1]["note"] == "b"


def test_semantic_hit_logs_query_event_with_cache_fields(chat_env):
    """log_query_event post-hit incluye cache_hit=True + cache_probe + cache_layer."""
    hit = _make_semantic_hit(cosine=0.97, age_seconds=180.0)
    chat_env.setattr(
        rag, "semantic_cache_lookup",
        lambda *a, **kw: (hit, {"result": "hit", "reason": "match",
                                "top_cosine": 0.97, "candidates": 1,
                                "skipped_stale": 0, "skipped_ttl": 0})
        if kw.get("return_probe") else hit,
    )
    _post_chat("q")
    events = chat_env.log_events
    assert events, "log_query_event not called"
    ev = events[-1]
    assert ev["cmd"] == "web.chat.cached_semantic"
    assert ev["cache_hit"] is True
    assert ev["cache_layer"] == "semantic"
    assert ev["cache_cosine"] == pytest.approx(0.97, rel=1e-2)
    assert ev["cache_age_seconds"] == 180
    assert ev["cache_probe"]["result"] == "hit"


# ── 2. LRU miss + semantic miss → pipeline runs + stores both ─────────────


def test_semantic_miss_runs_pipeline_and_stores(chat_env):
    """Miss en ambas layers → pipeline corre + semantic_cache_store llamado al final."""
    events, _ = _post_chat("nueva pregunta sin cache")

    # ollama.chat SÍ se llama (pipeline completo).
    assert len(chat_env.ollama_mock.calls) >= 1

    # done event no tiene cache_layer=semantic.
    done = [data for ev, data in events if ev == "done"]
    assert done, "missing done"
    assert done[-1].get("cache_layer") != "semantic"

    # semantic_cache_store fue invocado con los args correctos.
    assert len(chat_env.sem_stores) == 1
    call = chat_env.sem_stores[0]
    assert call["question"] == "nueva pregunta sin cache"
    assert call["response"] == "respuesta generada"
    assert call["corpus_hash"] == "HWEB"
    assert call["background"] is True  # web server es long-running
    assert len(call["paths"]) == 2  # canned retrieve returns 2 metas
    assert all(p.endswith(".md") for p in call["paths"])


def test_semantic_miss_logs_cache_probe_in_event(chat_env):
    """Audit 2026-04-24 — pre-fix el `log_query_event` del miss path NO
    incluía `cache_probe`. Sólo el HIT path lo loggeaba (línea 5717+).
    Resultado: 998 web queries en 7 días sin probe data → telemetry
    inservible para tunear el cache. Post-fix, AMBOS paths loggean
    cache_probe.

    Este test asegura que un miss pasa el probe al log_query_event."""
    _post_chat("nueva pregunta sin cache")
    events = chat_env.log_events
    assert events, "log_query_event not called"
    ev = events[-1]
    # Contract: el miss event tiene cmd="web" + cache_probe poblado +
    # cache_hit=False + cache_layer=None.
    assert ev["cmd"] == "web", f"expected cmd=web, got {ev['cmd']}"
    assert ev["cache_hit"] is False, (
        f"miss event must have cache_hit=False, got {ev.get('cache_hit')}"
    )
    assert ev["cache_layer"] is None, (
        f"miss event must have cache_layer=None, got {ev.get('cache_layer')!r}"
    )
    assert "cache_probe" in ev, (
        "cache_probe missing from miss-path log event — regression del audit 2026-04-24"
    )
    # El probe puede ser un dict de result/reason (cuando lookup corrió)
    # o None (cuando lookup nunca alcanzó por error pre-conn). Ambos
    # válidos — solo asserteamos que el field está presente.


# ── 3. Gates: history / propose / multi-vault ────────────────────────────


def test_semantic_skipped_when_history_present(chat_env):
    """Con history[] poblado, semantic lookup NI store se ejecuta."""
    _lookup_calls: list[tuple] = []
    def _spy_lookup(*args, **kwargs):
        _lookup_calls.append(args)
        if kwargs.get("return_probe"):
            return (None, {"result": "miss"})
        return None
    chat_env.setattr(rag, "semantic_cache_lookup", _spy_lookup)

    # Inyectar history vía session_history — la fixture default devuelve [].
    chat_env.setattr(
        server_mod, "session_history",
        lambda sess, window=10: [{"q": "prior turn", "a": "prior ans"}],
    )
    _post_chat("follow-up")
    assert len(_lookup_calls) == 0, (
        f"semantic lookup must skip with history, got {len(_lookup_calls)} calls"
    )
    assert len(chat_env.sem_stores) == 0, "semantic store must skip with history"


def test_semantic_skipped_when_propose_intent(chat_env):
    """`is_propose_intent=True` (ej. "recordame X") bloquea semantic."""
    _lookup_calls: list[tuple] = []
    def _spy_lookup(*args, **kwargs):
        _lookup_calls.append(args)
        if kwargs.get("return_probe"):
            return (None, {"result": "miss"})
        return None
    chat_env.setattr(rag, "semantic_cache_lookup", _spy_lookup)
    chat_env.setattr(server_mod, "_detect_propose_intent", lambda q: True)
    chat_env.setattr(server_mod, "_detect_metachat_intent", lambda q: False)
    # Propose intent path en web no toca retrieve() normal — stubeamos
    # el branch de tool calling para que devuelva una respuesta mínima.
    # Si el semantic lookup corrió, el test falla independiente de todo.
    try:
        _post_chat("recordame llamar a mamá a las 18")
    except Exception:
        # El propose path tiene muchos side effects (Apple MCP calls etc)
        # que están stubeados parcialmente. Para este test solo nos
        # importa que el semantic NO haya corrido.
        pass
    assert _lookup_calls == [], (
        f"propose_intent must skip semantic lookup, got {_lookup_calls}"
    )


def test_semantic_skipped_multi_vault(chat_env):
    """Multi-vault scope → single-vault-only gate bloquea el lookup."""
    _lookup_calls: list[tuple] = []
    def _spy_lookup(*args, **kwargs):
        _lookup_calls.append(args)
        if kwargs.get("return_probe"):
            return (None, {"result": "miss"})
        return None
    chat_env.setattr(rag, "semantic_cache_lookup", _spy_lookup)
    # _resolve_scope devuelve ≥2 vaults.
    chat_env.setattr(
        server_mod, "_resolve_scope",
        lambda scope: [("a", object()), ("b", object())],
    )
    _post_chat("q")
    assert _lookup_calls == [], (
        f"multi-vault must skip semantic lookup, got {_lookup_calls}"
    )


# ── 4. Resilience: lookup exception → pipeline fallthrough ───────────────


def test_semantic_lookup_exception_degrades_to_pipeline(chat_env):
    """Si semantic_cache_lookup rompe, el pipeline sigue normal (no 500)."""
    def _raise(*args, **kwargs):
        raise RuntimeError("SQL boom")
    chat_env.setattr(rag, "semantic_cache_lookup", _raise)

    events, _ = _post_chat("q con cache roto")
    # Debe completar sin 500.
    event_types = [ev for ev, _ in events]
    assert "done" in event_types or "token" in event_types, (
        f"pipeline must fall through on cache exception, events={event_types}"
    )
    # Store NO debe llamarse si el emb/hash quedaron reseteados post-exc.
    assert len(chat_env.sem_stores) == 0


# ── 5. LRU hit short-circuit takes priority over semantic ────────────────


def test_lru_hit_beats_semantic(chat_env):
    """LRU hit retorna antes — ni siquiera se consulta semantic."""
    chat_env.setattr(
        server_mod, "_chat_cache_get",
        lambda key: {
            "ts": 1_700_000_000.0,
            "text": "LRU response",
            "sources_items": [],
            "top_score": 0.9,
        },
    )
    _lookup_calls: list[tuple] = []
    def _spy_lookup(*args, **kwargs):
        _lookup_calls.append(args)
        if kwargs.get("return_probe"):
            return (None, {"result": "miss"})
        return None
    chat_env.setattr(rag, "semantic_cache_lookup", _spy_lookup)

    events, _ = _post_chat("repetida exactamente")
    tokens = "".join(d.get("delta", "") for ev, d in events if ev == "token")
    assert tokens == "LRU response"
    assert _lookup_calls == [], (
        f"LRU hit must short-circuit before semantic, got {_lookup_calls}"
    )
