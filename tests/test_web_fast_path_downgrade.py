"""Tests for the fast-path downgrade when the pre-router fires tools
(2026-04-24).

Pre-fix `/api/chat` respetaba `result["fast_path"]` como la sola señal
para switchear a `_LOOKUP_MODEL` (qwen2.5:3b) + `_LOOKUP_NUM_CTX` (4096)
en el streaming final. Pero cuando el pre-router (regex-based) matchea
tools deterministas (`reminders_due`, `calendar_ahead`, `finance_summary`,
`gmail_recent`, `weather`) el CONTEXTO completo se REEMPLAZA con la
salida formateada de esos tools — fácil 2-4K tokens de listas. qwen2.5:3b
en M3 Max prefillea a ~2.5ms/tok → 3K tokens = 7.5s de prefill, mientras
que qwen2.5:7b prefillea a ~0.5ms/tok (mejor FA throughput) = 1.5s.

Medido el 2026-04-23 en prod: query "qué pendientes tengo" con
fast_path=1 + tool_rounds=1 arrojó llm_prefill_ms=11595ms, total_ms=16278.
Post-fix el mismo flujo debería caer bajo 5s total.

Estos tests son source-level porque `/api/chat` es un generator con SSE
y mockearlo end-to-end es pesado. Chequean que el código tiene (a) la
inicialización de `_fast_path_downgraded`, (b) el gate con la env var
de rollback, (c) el log_query_event persiste el marker.
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

SERVER_SRC = (ROOT / "web" / "server.py").read_text(encoding="utf-8")


# ── Initialization: `_fast_path_downgraded` starts at False ────────────────


def test_fast_path_downgraded_initialized_false():
    """`_fast_path_downgraded` must be initialized to False before the
    try/pre-router block — si no se inicializa, el log_query_event al
    final del generator lee una variable undefined y crashea con
    NameError cuando NO hubo downgrade (la ruta más común)."""
    # Buscar bloque de inicialización cerca del setup de _fast_path
    idx = SERVER_SRC.find("_fast_path = bool(result.get(\"fast_path\", False))")
    assert idx >= 0, "_fast_path setup not found"
    # El marker debe aparecer dentro de las próximas 1000 líneas después
    nearby = SERVER_SRC[idx : idx + 2000]
    assert "_fast_path_downgraded = False" in nearby, (
        "_fast_path_downgraded init missing near _fast_path setup"
    )


# ── Gate: env var RAG_FAST_PATH_KEEP_WITH_TOOLS controls rollback ─────────


def test_fast_path_downgrade_gate_has_rollback_env():
    """El gate debe leer `RAG_FAST_PATH_KEEP_WITH_TOOLS` para operadores
    que quieran mantener el comportamiento pre-fix (qwen2.5:3b aunque
    el contexto esté inflado)."""
    assert "RAG_FAST_PATH_KEEP_WITH_TOOLS" in SERVER_SRC, (
        "rollback env var not referenced anywhere in web/server.py"
    )
    # Y debe ser un truthy/falsy check, no hard-coded
    idx = SERVER_SRC.find("RAG_FAST_PATH_KEEP_WITH_TOOLS")
    assert idx >= 0
    nearby = SERVER_SRC[idx : idx + 300]
    # Debe usar el patrón estándar del repo: `not in ("", "0", "false", "no")`
    # invertido → significa "default ON, rollback con valor truthy"
    assert "\"0\"" in nearby and "\"false\"" in nearby and "\"no\"" in nearby, (
        f"gate no usa el patrón estándar del repo para detectar "
        f"truthy/falsy: {nearby[:200]!r}"
    )


# ── Downgrade wiring: después de pre-router fire, rebuild _WEB_CHAT_OPTIONS ─


def test_fast_path_downgrade_rebuilds_web_chat_options():
    """Cuando dispara el downgrade, _WEB_CHAT_OPTIONS debe rebuildearse
    con `num_ctx=_WEB_CHAT_NUM_CTX` (no _LOOKUP_NUM_CTX). Sin este rebuild
    el streaming sigue pidiendo 4096 ctx aunque el modelo ya es 7b —
    inofensivo pero subóptimo (podríamos permitir más context).
    """
    idx = SERVER_SRC.find("_fast_path_downgraded = True")
    assert idx >= 0, "downgrade True assignment not found"
    # Dentro de ~500 chars siguientes debe aparecer el rebuild
    nearby = SERVER_SRC[idx : idx + 1000]
    assert "_web_model = _web_model_full" in nearby, (
        "missing _web_model reassignment to full model"
    )
    assert "_WEB_CHAT_OPTIONS = {" in nearby, (
        "missing _WEB_CHAT_OPTIONS rebuild"
    )
    # Crítico: num_ctx debe ser _WEB_CHAT_NUM_CTX (4096), NOT _LOOKUP_NUM_CTX
    assert '"num_ctx": _WEB_CHAT_NUM_CTX' in nearby, (
        "num_ctx must be _WEB_CHAT_NUM_CTX post-downgrade, "
        "not _LOOKUP_NUM_CTX"
    )


# ── Downgrade inside the pre-router `if _forced_tools:` block ──────────────


def test_fast_path_downgrade_scoped_to_preroute():
    """El downgrade NO debe dispararse cuando _forced_tools está vacío —
    queries semánticas puras se quedan con fast-path. Chequear que el
    `if _fast_path` gate vive DENTRO del `if _forced_tools:` block."""
    forced_idx = SERVER_SRC.find("if _forced_tools:")
    assert forced_idx >= 0
    # Buscar el downgrade — debe aparecer DESPUÉS del `if _forced_tools:`
    # y ANTES del `# Gate the LLM tool-deciding loop.` comment (que marca
    # el fin del pre-router block)
    downgrade_idx = SERVER_SRC.find("_fast_path_downgraded = True")
    gate_comment_idx = SERVER_SRC.find("# Gate the LLM tool-deciding loop.")
    assert downgrade_idx > forced_idx, (
        "downgrade must be AFTER the pre-router opens"
    )
    assert downgrade_idx < gate_comment_idx, (
        "downgrade must be BEFORE the LLM tool-decide gate (i.e., "
        "inside the pre-router block)"
    )


# ── Log: persistir fast_path_downgraded en rag_queries.extra_json ──────────


def test_log_query_event_persists_fast_path_downgraded():
    """El log del endpoint debe incluir `fast_path_downgraded` en el
    payload del log_query_event. Sin esto no hay forma de medir desde
    analytics cuántas queries se benefician del downgrade."""
    # Buscar el bloque principal log_query_event con cmd="web"
    web_idx = SERVER_SRC.find('"cmd": "web",')
    assert web_idx >= 0
    nearby = SERVER_SRC[web_idx : web_idx + 6000]
    assert "\"fast_path_downgraded\"" in nearby, (
        "fast_path_downgraded missing from main web log_query_event"
    )
    assert "_fast_path_downgraded" in nearby, (
        "the value must be wired from the local var"
    )


# ── Observability: print a visible chat-fast-path-downgrade line ──────────


def test_fast_path_downgrade_has_print_marker():
    """Cuando dispara el downgrade, imprimir una línea identificable para
    grep. Sin log es muy difícil debuggear desde tail -f web.log si
    el downgrade está funcionando o no."""
    assert "[chat-fast-path-downgrade]" in SERVER_SRC, (
        "debug print line missing — analytics users can't grep"
    )


# ── Functional: /api/chat actually uses full model when pre-router fired ──


def _canned_retrieve(fast_path: bool, query: str = "qué pendientes tengo") -> dict:
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


class _OllamaMock:
    """Scripted stand-in para `ollama.chat`.
    Collectea cada call con kwargs (model, options, stream, tools...) así
    los tests pueden assertear cuál modelo se usó en qué step.
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
            return iter(resp)
        return resp


def _mk_stream(tokens: list[str]) -> list[SimpleNamespace]:
    return [SimpleNamespace(message=SimpleNamespace(content=t)) for t in tokens]


@pytest.fixture
def chat_env_fastpath(monkeypatch):
    """Shared monkeypatches so `/api/chat` runs in-process sin tocar
    network/ollama/vault. Similar al fixture de test_web_chat_tools pero
    forzando `fast_path=True` en el retrieve result.
    """
    # Retrieve devuelve `fast_path=True` para forzar el path bajo test.
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: _canned_retrieve(fast_path=True),
    )
    monkeypatch.setattr(server_mod, "_ollama_alive", lambda timeout=2.0: True)
    monkeypatch.setattr(server_mod, "_ollama_chat_probe", lambda timeout_s=6.0: True)
    monkeypatch.setattr(server_mod, "_fetch_whatsapp_unread", lambda *a, **kw: [])
    import rag as _rag
    monkeypatch.setattr(_rag, "build_person_context", lambda q: None)
    monkeypatch.setattr(server_mod, "save_session", lambda sess: None)
    monkeypatch.setattr(server_mod, "log_query_event", lambda ev: None)
    monkeypatch.setattr(server_mod, "_chat_cache_get", lambda key: None)
    monkeypatch.setattr(server_mod, "_chat_cache_put", lambda key, val: None)
    # Semantic cache también off — queremos el path crudo. La función
    # es importada lazy dentro del endpoint, así que basta con disable
    # via el env var que el código mismo consulta.
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_DISABLED", "1")
    # Spawn writer: no-op (daemon thread).
    monkeypatch.setattr(server_mod, "_spawn_conversation_writer", lambda *a, **kw: None)
    # _emit_enrich / _emit_grounding son closures definidos dentro de
    # `chat()` — NO module attrs, no se pueden monkeypatchear. En el
    # TestClient corren pero sus deps ya están stubeadas (retrieve,
    # log_query_event) así que no hacen harm.
    # is_tasks_query off — queremos el chat regular.
    monkeypatch.setattr(server_mod, "_is_tasks_query", lambda q: False)
    # Stub reminders_due tool function para devolver algo predecible.
    # El pre-router regex-based matchea "pendientes" → reminders_due.
    from web import tools as tools_mod
    monkeypatch.setattr(
        tools_mod, "_fetch_reminders_due",
        lambda days=7: {
            "dated": [
                {"title": "comprar pan", "due": "2026-04-24", "list": "Tareas"}
            ],
            "undated": [],
        },
    )
    return monkeypatch


def test_functional_downgrade_fires_when_preroute_matches(chat_env_fastpath):
    """End-to-end: query "qué pendientes tengo" → pre-router matchea
    reminders_due + calendar_ahead → fast_path debe downgradear al
    modelo full. Verifica que el último ollama.chat (el streaming final)
    usa el modelo resuelto por `_resolve_web_chat_model()`, no
    `_LOOKUP_MODEL`.
    """
    mock = _OllamaMock([
        # Streaming final response
        _mk_stream(["ok", " tenés", " pan", " pendiente"]),
    ])
    chat_env_fastpath.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"question": "qué pendientes tengo", "vault_scope": None},
    )
    assert resp.status_code == 200, resp.text

    # El streaming final debe haber usado el full model (no _LOOKUP_MODEL).
    # Después del pre-router, fast_path queda downgraded → qwen2.5:7b.
    streaming_calls = [c for c in mock.calls if c.get("stream")]
    assert streaming_calls, f"no streaming call was made: {mock.calls}"
    final_call = streaming_calls[-1]
    full_model = server_mod._resolve_web_chat_model()
    assert final_call["model"] == full_model, (
        f"final streaming used {final_call['model']!r} but expected "
        f"full model {full_model!r} (downgrade should have fired "
        f"because pre-router matched reminders_due)"
    )
    # num_ctx también debe estar capeado por _WEB_CHAT_NUM_CTX (4096), no
    # _LOOKUP_NUM_CTX. Adaptive num_ctx (2026-04-25, developer-1) calcula
    # un valor runtime entre [1024, cap] basado en final_messages chars,
    # así que verificamos el contrato del cap (lo que importa para el
    # downgrade es que el ceiling sea el del modelo full, no que el valor
    # exacto sea 4096).
    _num_ctx = final_call["options"]["num_ctx"]
    assert 1024 <= _num_ctx <= server_mod._WEB_CHAT_NUM_CTX, (
        f"num_ctx={_num_ctx} fuera del rango adaptativo "
        f"[1024, {server_mod._WEB_CHAT_NUM_CTX}] post-downgrade"
    )


def test_functional_no_downgrade_when_no_preroute_match(chat_env_fastpath):
    """Pure semantic query (no pre-router match) mantiene fast-path. El
    streaming final debe usar `_LOOKUP_MODEL` como la calibración
    original decidió.
    """
    mock = _OllamaMock([
        _mk_stream(["resumen", " del", " vault"]),
    ])
    chat_env_fastpath.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    client = TestClient(app)
    # "explicame qué es ikigai" no tiene ningún keyword que el pre-router
    # catche — ni gastos, ni mail, ni calendari, ni pendientes, ni weather,
    # ni el _PLANNING_PAT (hoy/mañana/semana/etc.). Query puramente
    # conceptual sobre una nota del vault.
    resp = client.post(
        "/api/chat",
        json={"question": "explicame qué es ikigai", "vault_scope": None},
    )
    assert resp.status_code == 200, resp.text

    streaming_calls = [c for c in mock.calls if c.get("stream")]
    assert streaming_calls
    final_call = streaming_calls[-1]
    assert final_call["model"] == server_mod._LOOKUP_MODEL, (
        f"sin pre-router match, el fast-path debía sobrevivir — "
        f"got {final_call['model']!r}, expected {server_mod._LOOKUP_MODEL!r}"
    )
    # Adaptive num_ctx (2026-04-25): cap es _LOOKUP_NUM_CTX (4096), runtime
    # effective value depende de final_messages chars. Ver
    # test_functional_downgrade_fires_when_preroute_matches para detalle.
    _num_ctx = final_call["options"]["num_ctx"]
    assert 1024 <= _num_ctx <= server_mod._LOOKUP_NUM_CTX, (
        f"num_ctx={_num_ctx} fuera del rango adaptativo "
        f"[1024, {server_mod._LOOKUP_NUM_CTX}]"
    )


def test_functional_rollback_env_keeps_fast_path_with_tools(chat_env_fastpath, monkeypatch):
    """Setting `RAG_FAST_PATH_KEEP_WITH_TOOLS=1` restores pre-fix behaviour
    (qwen2.5:3b aunque el pre-router haya fired tools). Para operadores
    que detecten una regresión específica con el downgrade.
    """
    monkeypatch.setenv("RAG_FAST_PATH_KEEP_WITH_TOOLS", "1")

    mock = _OllamaMock([
        _mk_stream(["ok"]),
    ])
    chat_env_fastpath.setattr(server_mod._OLLAMA_STREAM_CLIENT, "chat", mock)

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"question": "qué pendientes tengo", "vault_scope": None},
    )
    assert resp.status_code == 200, resp.text

    streaming_calls = [c for c in mock.calls if c.get("stream")]
    assert streaming_calls
    final_call = streaming_calls[-1]
    # Con rollback env, el fast-path sobrevive aunque el pre-router fired.
    assert final_call["model"] == server_mod._LOOKUP_MODEL, (
        f"rollback env set pero downgrade fired: model={final_call['model']!r}"
    )
