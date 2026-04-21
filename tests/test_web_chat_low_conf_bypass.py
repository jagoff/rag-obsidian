"""Offline tests for the low-confidence LLM bypass path in `/api/chat`.

Cuando el vault no tiene info relevante (`result["confidence"] <
CONFIDENCE_RERANK_MIN`) y la query no pega con un mention / no es un
propose intent, `chat()` saltea el `ollama.chat` streaming call y
devuelve un template fijo (`No tengo info sobre "{question}" en tus
notas.`). El SSE `done` event incluye `low_conf_bypass=True` para que
el frontend renderee el cluster de "¿querés que busque en internet?"
en vez del inline link del path weakAnswer.

Cobertura:
  1. Low-conf + no mention + no propose → bypass activo, 0 llamadas
     a ollama.chat, template con la question literal se streamea.
  2. `done` event tiene `low_conf_bypass=True` + `bypassed=True` +
     `llm_ms=0` + `retrieve_ms` real.
  3. Mention hits bloquean el bypass (mention prevalece aunque conf
     sea baja).
  4. `is_propose_intent=True` bloquea el bypass (tool loop corre
     normal).
  5. Conf >= CONFIDENCE_RERANK_MIN NO hace bypass (normal path).
  6. Question con caracteres especiales (comillas, UTF-8, emojis) se
     preserva textual en el template.
  7. El `[chat-bypass]` log line se emite con los campos esperados
     (telemetry para dashboards).
  8. Una question con comillas dobles se escapa `\"` para no romper
     el formato del template.

Todos los tests stubean `multi_retrieve` + `ollama.chat` + los writers
de conversación/log — 0 llamadas de red, 0 llamadas al daemon real.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


# ── Import targets under test ──────────────────────────────────────────────

from web import server as server_mod
from web.server import app


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


# ── Ollama mock: tracks call count so tests can assert bypass skipped LLM ──


class _OllamaMock:
    """Counts every invocation. Tests asserting bypass verify
    `mock.calls == []` post-stream."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError(
                "OllamaMock: test did not script any responses — bypass "
                "path should NOT be hitting ollama.chat at all."
            )
        resp = self.responses.pop(0)
        if kwargs.get("stream"):
            return iter(resp)
        return resp


def _mk_msg(content: str = "", tool_calls=None) -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls),
    )


def _mk_stream(tokens: list[str]) -> list[SimpleNamespace]:
    return [SimpleNamespace(message=SimpleNamespace(content=t)) for t in tokens]


# ── Canned retrieve result with a tunable confidence ───────────────────────


def _canned_retrieve_result(
    query: str = "x",
    *,
    confidence: float = 0.005,
    n_docs: int = 2,
) -> dict:
    """`confidence=0.005` is below CONFIDENCE_RERANK_MIN=0.015 — default
    triggers the bypass. Pass higher values to exercise the normal path.
    """
    return {
        "docs": [f"doc {i} body" for i in range(n_docs)],
        "metas": [
            {"file": f"01-Projects/a{i}.md", "note": f"a{i}",
             "folder": "01-Projects"}
            for i in range(n_docs)
        ],
        "scores": [confidence] * n_docs,
        "confidence": confidence,
        "search_query": query,
        "filters_applied": {},
        "query_variants": [query],
        "vault_scope": ["default"],
    }


# ── Shared env fixture ─────────────────────────────────────────────────────


@pytest.fixture
def chat_env(monkeypatch):
    """Shared monkeypatches. Defaults to a retrieve result con
    conf=0.005 (triggers bypass). Tests que quieren normal path
    re-monkeypatch `multi_retrieve` con conf más alta.
    """
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve_result(
            a[1] if len(a) >= 2 else "x"
        ),
    )
    monkeypatch.setattr(server_mod, "_ollama_alive", lambda timeout=2.0: True)
    monkeypatch.setattr(
        server_mod, "_ollama_chat_probe", lambda timeout_s=6.0: True,
    )
    monkeypatch.setattr(
        server_mod, "_fetch_whatsapp_unread", lambda *a, **kw: [],
    )
    import rag as _rag
    # _match_mentions_in_query default: no hits. Tests specific a
    # mentions override per-case.
    monkeypatch.setattr(_rag, "_match_mentions_in_query", lambda q: [])
    monkeypatch.setattr(_rag, "build_person_context", lambda q: None)
    # Never invoke the episodic writer — the daemon thread would try to
    # write to VAULT_PATH (isolated tmp via conftest fixture) and race
    # the test teardown.
    monkeypatch.setattr(
        server_mod, "_spawn_conversation_writer", lambda **kw: None,
    )
    monkeypatch.setattr(server_mod, "save_session", lambda sess: None)
    # Tests assert on log_query_event payloads — replace with a collector.
    _events: list[dict] = []
    monkeypatch.setattr(
        server_mod, "log_query_event", lambda ev: _events.append(ev),
    )
    monkeypatch.setattr(server_mod, "_chat_cache_get", lambda key: None)
    monkeypatch.setattr(server_mod, "_chat_cache_put", lambda key, val: None)
    monkeypatch.setattr(server_mod, "_is_tasks_query", lambda q: False)
    # Expose the events collector on the monkeypatch for test access.
    monkeypatch.log_events = _events  # type: ignore[attr-defined]
    return monkeypatch


def _post_chat(question: str = "hola") -> tuple[list[tuple[str, dict]], str]:
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"question": question, "vault_scope": None},
    )
    assert resp.status_code == 200, resp.text
    return _parse_sse(resp.text), resp.text


# ── 1. Bypass happy-path: low conf, no mention, no propose ─────────────────


def test_bypass_skips_ollama_chat(chat_env):
    """Conf < CONFIDENCE_RERANK_MIN + no mention + no propose →
    ollama.chat NO debería llamarse ni una vez.
    """
    mock = _OllamaMock(responses=[])  # zero scripted responses
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    events, _ = _post_chat("qué sabés de Bizarrap")

    # No LLM call at all. Any __call__ into ollama would have raised
    # (OllamaMock has an empty response queue).
    assert mock.calls == [], (
        f"expected 0 ollama.chat calls en el bypass path, got "
        f"{len(mock.calls)}: {mock.calls!r}"
    )

    # Stream completed con done.
    names = [ev for ev, _ in events]
    assert names[-1] == "done", names


def test_bypass_template_contains_question_verbatim(chat_env):
    """El template emite tokens que reconstruyen
    `No tengo info sobre "{question}" en tus notas.` — la question
    literal, sin modificar.
    """
    mock = _OllamaMock(responses=[])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    q = "qué sabés de Bizarrap"
    events, _ = _post_chat(q)

    # Reconstruir el texto full desde los `token` events.
    tokens = [data["delta"] for ev, data in events if ev == "token"]
    full = "".join(tokens)
    assert full == f'No tengo info sobre "{q}" en tus notas.', full
    # El nombre propio se preserva TEXTUAL — anti-regresión del caso
    # Bizarrap → Bizarra que motivó esta feature. No podemos usar
    # `"Bizarra" not in full` porque es prefijo de "Bizarrap"; la
    # igualdad estricta arriba ya garantiza preservación verbatim.
    assert "Bizarrap" in full


def test_bypass_done_event_payload(chat_env):
    """El `done` event debe cargar `low_conf_bypass=True`, `bypassed=True`,
    `llm_ms=0`, y campos de timing finitos.
    """
    mock = _OllamaMock(responses=[])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    events, _ = _post_chat("query sin match en vault")

    done = next(data for ev, data in events if ev == "done")
    assert done["low_conf_bypass"] is True, done
    assert done["bypassed"] is True, done
    assert done["llm_ms"] == 0, done
    assert isinstance(done.get("retrieve_ms"), int) and done["retrieve_ms"] >= 0
    assert isinstance(done.get("total_ms"), int) and done["total_ms"] >= 0
    assert "turn_id" in done and done["turn_id"]


# ── 2. Mention hits override the bypass ────────────────────────────────────


def test_mention_hit_blocks_bypass(chat_env):
    """Si `_match_mentions_in_query` devuelve paths, el bypass NO dispara
    aunque la confianza sea baja — la mention es authoritative y
    queremos que el LLM produzca una respuesta real con el preamble.
    """
    import rag as _rag
    chat_env.setattr(
        _rag, "_match_mentions_in_query",
        lambda q: ["99 Mentions @/bizarrap.md"],
    )
    # build_person_context no puede ser None cuando hay mention — stubeamos
    # para no romper el path downstream.
    chat_env.setattr(
        _rag, "build_person_context",
        lambda q: "## Bizarrap\nArtista argentino.",
    )

    # RAG_WEB_TOOL_LLM_DECIDE no está seteado + no hay propose intent →
    # el tool-decide loop se saltea (ver `_skip_llm_tool_round` en
    # web/server.py). Solo corre el final streaming call, así que el
    # mock scriptea una sola response.
    mock = _OllamaMock([
        _mk_stream(["respuesta ", "real"]),
    ])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    events, _ = _post_chat("qué sabés de Bizarrap")

    # Hubo al menos una llamada a ollama.chat (path normal).
    assert len(mock.calls) >= 1, (
        "bypass no debería haberse activado con mention hit; esperaba "
        ">=1 llamada a ollama.chat, got 0"
    )

    done = next(data for ev, data in events if ev == "done")
    assert done.get("low_conf_bypass") is not True, done


# ── 3. Propose intent overrides the bypass ─────────────────────────────────


def test_propose_intent_blocks_bypass(chat_env):
    """`_detect_propose_intent=True` → el tool loop (propose_reminder /
    propose_calendar_event) debe correr aunque la conf sea baja.
    """
    import rag as _rag
    chat_env.setattr(_rag, "_detect_propose_intent", lambda q: True)
    chat_env.setattr(server_mod, "_detect_propose_intent", lambda q: True)

    # Tool loop: un decide-round vacío + final stream.
    mock = _OllamaMock([
        _mk_msg(content="", tool_calls=None),
        _mk_stream(["ok"]),
    ])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    events, _ = _post_chat("recordame comprar pan mañana")

    assert len(mock.calls) >= 1, mock.calls
    done = next(data for ev, data in events if ev == "done")
    assert done.get("low_conf_bypass") is not True, done


# ── 4. High-confidence retrieve takes the normal path ──────────────────────


def test_high_confidence_no_bypass(chat_env):
    """Conf >= CONFIDENCE_RERANK_MIN → normal path (LLM runs)."""
    # Override retrieve con confidence alto.
    chat_env.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve_result(
            a[1] if len(a) >= 2 else "x",
            confidence=0.85,
        ),
    )
    # Tool-decide loop skippeado por default — solo corre el streaming.
    mock = _OllamaMock([
        _mk_stream(["hola ", "mundo"]),
    ])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    events, _ = _post_chat("qué hay sobre docker en mis notas")

    assert len(mock.calls) >= 1, (
        f"normal path con conf alta debería llamar a ollama.chat, "
        f"got {mock.calls!r}"
    )
    done = next(data for ev, data in events if ev == "done")
    assert done.get("low_conf_bypass") is not True, done


def test_confidence_exactly_at_threshold_no_bypass(chat_env):
    """Edge: conf == CONFIDENCE_RERANK_MIN (0.015) NO dispara bypass
    (strict `<`, no `<=`). Invariant documentado en rag.py.
    """
    from rag import CONFIDENCE_RERANK_MIN
    chat_env.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve_result(
            a[1] if len(a) >= 2 else "x",
            confidence=CONFIDENCE_RERANK_MIN,  # exact boundary
        ),
    )
    mock = _OllamaMock([
        _mk_stream(["resp"]),
    ])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    events, _ = _post_chat("query en el borde")

    assert len(mock.calls) >= 1, mock.calls
    done = next(data for ev, data in events if ev == "done")
    assert done.get("low_conf_bypass") is not True, done


# ── 5. Special characters in the question ──────────────────────────────────


@pytest.mark.parametrize("question, expected_snippet", [
    # UTF-8 básico + acentos.
    ("qué sabés de año nuevo",
     'No tengo info sobre "qué sabés de año nuevo" en tus notas.'),
    # Emojis.
    ("info sobre 🎸 guitarras",
     'No tengo info sobre "info sobre 🎸 guitarras" en tus notas.'),
    # Comillas dobles — deben escaparse con `\"` dentro del template
    # para no romper el formato (la assertion matchea el raw stream).
    ('he said "hello"',
     'No tengo info sobre "he said \\"hello\\"" en tus notas.'),
    # Caracteres especiales (barras, paréntesis).
    ("query con /slash y (parens)",
     'No tengo info sobre "query con /slash y (parens)" en tus notas.'),
])
def test_bypass_preserves_special_chars(chat_env, question, expected_snippet):
    """El template debe emitir la question textual (con escape mínimo
    de `"`) — UTF-8, emojis, símbolos pasan intactos.
    """
    mock = _OllamaMock(responses=[])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    events, _ = _post_chat(question)

    tokens = [data["delta"] for ev, data in events if ev == "token"]
    full = "".join(tokens)
    assert full == expected_snippet, (
        f"question={question!r}\nexpected={expected_snippet!r}\n"
        f"got={full!r}"
    )
    # Bypass path activo — 0 LLM calls.
    assert mock.calls == [], mock.calls


# ── 6. Telemetry: [chat-bypass] stdout line + log_query_event payload ──────


def test_bypass_emits_log_line(chat_env, capsys):
    """Verificar que el `[chat-bypass]` line aparece en stdout con los
    campos esperados (conf, reason=low_conf, retrieve_ms).
    """
    mock = _OllamaMock(responses=[])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    _post_chat("query sin match")

    captured = capsys.readouterr()
    bypass_lines = [
        ln for ln in captured.out.splitlines()
        if ln.startswith("[chat-bypass]")
    ]
    assert len(bypass_lines) == 1, (
        f"expected exactly one [chat-bypass] line, got "
        f"{len(bypass_lines)}:\n{captured.out}"
    )
    line = bypass_lines[0]
    assert "reason=low_conf" in line, line
    assert "conf=" in line, line
    assert "retrieve_ms=" in line, line


def test_bypass_log_query_event_payload(chat_env):
    """El `log_query_event` call en el bypass path debe incluir
    `cmd=web.chat.low_conf_bypass` + `low_conf_bypass=True`, para
    poder filtrar estos turns en el dashboard de analytics.
    """
    mock = _OllamaMock(responses=[])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    _post_chat("otra query sin match")

    events = chat_env.log_events  # type: ignore[attr-defined]
    assert len(events) == 1, events
    ev = events[0]
    assert ev.get("cmd") == "web.chat.low_conf_bypass", ev
    assert ev.get("low_conf_bypass") is True, ev
    assert ev.get("q") == "otra query sin match", ev
    assert ev.get("paths") == [], ev
    assert ev.get("t_gen") == 0.0, ev


# ── 7. Shape-stable [chat-timing] line for downstream parsers ──────────────


def test_bypass_emits_chat_timing_line(chat_env, capsys):
    """Dashboards grep `[chat-timing]` para extraer métricas por turn.
    El bypass path debe emitir uno shape-compatible (con `bypassed=1`
    como tag terminal), mismo approach que el cache-hit path.
    """
    mock = _OllamaMock(responses=[])
    chat_env.setattr(server_mod.ollama, "chat", mock)
    chat_env.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    _post_chat("una query cualquiera")

    captured = capsys.readouterr()
    timing_lines = [
        ln for ln in captured.out.splitlines()
        if ln.startswith("[chat-timing]")
    ]
    assert len(timing_lines) == 1, captured.out
    line = timing_lines[0]
    assert "bypassed=1" in line, line
    # Campos requeridos por el parser de dashboards.
    for kw in ("retrieve=", "reform=", "confidence=", "tool_rounds=0"):
        assert kw in line, f"missing {kw} en: {line}"
