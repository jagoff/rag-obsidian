"""Tests for the `whatsapp_pending` chat tool.

Origin: user report 2026-04-24 (Fer F.) — preguntó "qué tengo pendiente
esta semana" y el chat respondió con tareas + eventos + una sección
vaga de "conversaciones pendientes con wzp relacionadas a X" que en
realidad eran artefactos de Phase 2 radar (contradicciones del vault),
NO chats reales de WhatsApp. El LLM alucinaba que las contradicciones
de gastos eran conversaciones de WhatsApp porque no tenía ningún tool
que le devolviera chats reales pendientes.

Fix: nuevo tool `whatsapp_pending` wrapeando `_fetch_whatsapp_unreplied`
(que ya existía pero sólo se usaba en el dashboard). Integrado al
pre-router vía `_PLANNING_PAT` — cualquier planning query ("qué
tengo esta semana/hoy/mañana") dispara ahora reminders_due +
calendar_ahead + whatsapp_pending en paralelo. Keywords explícitos
(whatsapp/wzp/wsp/chats/pendiente de responder) también gatillan.

Source emission: cuando el tool devuelve chats, el servidor re-emite
el SSE `sources` con links a WhatsApp (`https://wa.me/<phone>` para
DMs, `https://web.whatsapp.com/` para grupos) prepended al listado
del vault, mismo patrón que drive_search.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from web.server import (
    _SOURCE_INTENT_META,
    _build_source_intent_hint,
    _detect_tool_intent,
    _format_forced_tool_output,
    _format_whatsapp_block,
    _is_empty_tool_output,
    app,
)
from web import server as server_mod
from web import tools as tools_mod


@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path):
    import rag as _rag
    snap = _rag.DB_PATH
    _rag.DB_PATH = tmp_path / "ragvec"
    try:
        yield
    finally:
        _rag.DB_PATH = snap


# ── 1. Tool wrapper ────────────────────────────────────────────────────────


def test_whatsapp_pending_tool_registered():
    assert "whatsapp_pending" in tools_mod.TOOL_FNS
    assert tools_mod.whatsapp_pending in tools_mod.CHAT_TOOLS
    assert "whatsapp_pending" in tools_mod.PARALLEL_SAFE
    assert tools_mod.whatsapp_pending.__doc__
    assert "WhatsApp" in tools_mod.whatsapp_pending.__doc__


def test_whatsapp_pending_addendum_mentions_tool():
    """`_WEB_TOOL_ADDENDUM` documenta el tool con keywords + la semántica
    "planning query → también fire whatsapp_pending"."""
    addendum = tools_mod._WEB_TOOL_ADDENDUM.lower()
    assert "whatsapp_pending" in addendum
    assert "wzp" in addendum or "whatsapp" in addendum


def test_whatsapp_pending_silent_fail(monkeypatch):
    """Si `_fetch_whatsapp_unreplied` crashea, el tool devuelve "[]" en
    vez de raisear — el chat loop tolera bien el shape vacío."""
    def _boom(**_kw):
        raise RuntimeError("db locked")
    monkeypatch.setattr(server_mod, "_fetch_whatsapp_unreplied", _boom)

    out = tools_mod.whatsapp_pending(hours=48)
    assert out == "[]"


def test_whatsapp_pending_happy_path(monkeypatch):
    """Call con data válida serializa la lista tal cual como JSON."""
    monkeypatch.setattr(
        server_mod, "_fetch_whatsapp_unreplied",
        lambda hours, max_chats: [
            {"jid": "5491111@s.whatsapp.net", "name": "Juan",
             "last_snippet": "te respondo mañana", "hours_waiting": 5.2},
            {"jid": "5492222@s.whatsapp.net", "name": "María",
             "last_snippet": "llegaste?", "hours_waiting": 1.3},
        ],
    )
    out = tools_mod.whatsapp_pending()
    parsed = json.loads(out)
    assert len(parsed) == 2
    assert parsed[0]["name"] == "Juan"
    assert parsed[1]["name"] == "María"


def test_whatsapp_pending_clamps_args(monkeypatch):
    """`hours` y `max_chats` se clampan al rango definido."""
    captured: dict = {}
    def _capture(hours, max_chats):
        captured["hours"] = hours
        captured["max_chats"] = max_chats
        return []
    monkeypatch.setattr(server_mod, "_fetch_whatsapp_unreplied", _capture)

    tools_mod.whatsapp_pending(hours=10000, max_chats=100)
    assert captured["hours"] == 168  # cap 168h (1 week)
    assert captured["max_chats"] == 20

    tools_mod.whatsapp_pending(hours=0, max_chats=0)
    assert captured["hours"] == 1
    assert captured["max_chats"] == 1


# ── 2. Pre-router regex ────────────────────────────────────────────────────


@pytest.mark.parametrize("query", [
    # Keywords explícitos.
    "qué tengo en whatsapp",
    "chats de wzp pendientes",
    "mensajes de wsp",
    "mis chats sin responder",
    "quién está pendiente de respuesta",
    "contactos pendientes de contestar",
    # Planning queries: user preguntó por "qué tengo esta semana" sin
    # mencionar WA, debe fire igual (WA chats son pendientes semánticos).
    "qué tengo esta semana",
    "qué tengo hoy",
    "qué tengo para mañana",
    "cómo viene la semana",
    "qué tengo pendiente",
])
def test_pre_router_fires_whatsapp_pending(query: str):
    matched = [n for n, _ in _detect_tool_intent(query)]
    assert "whatsapp_pending" in matched, (
        f"Query {query!r} debería haber enganchado whatsapp_pending, "
        f"vi {matched}"
    )


@pytest.mark.parametrize("query", [
    # Queries que NO tienen que ver con WA ni planning — no deben fire.
    "cómo está el clima",  # solo weather
    "los gastos de marzo",  # solo finance_summary
    "buscá en mi drive",  # solo drive_search
    "qué me dijo Juan sobre python",  # sin keyword WA ni planning
    "explicame cómo funciona FastAPI",  # pregunta técnica sin intent
])
def test_pre_router_does_not_fire_whatsapp_for_unrelated(query: str):
    matched = [n for n, _ in _detect_tool_intent(query)]
    assert "whatsapp_pending" not in matched, (
        f"Query {query!r} NO debería haber enganchado whatsapp_pending, "
        f"vi {matched}"
    )


def test_pre_router_planning_query_fires_all_three():
    """`qué tengo esta semana` debe fire los 3 tools pending: reminders,
    calendar, whatsapp — para que el LLM tenga visión completa del
    bucket 'pending'."""
    matched = [n for n, _ in _detect_tool_intent("qué tengo esta semana")]
    assert "reminders_due" in matched
    assert "calendar_ahead" in matched
    assert "whatsapp_pending" in matched


# ── 3. Format block ────────────────────────────────────────────────────────


def test_format_whatsapp_block_happy_path():
    raw = json.dumps([
        {"jid": "a@s.whatsapp.net", "name": "Juan",
         "last_snippet": "mañana te confirmo", "hours_waiting": 3.5},
        {"jid": "b@s.whatsapp.net", "name": "María",
         "last_snippet": "hola!", "hours_waiting": 72.0},
    ])
    out = _format_whatsapp_block(raw)
    assert out.startswith("### WhatsApp (2 chats esperando respuesta)")
    assert "**Juan** (hace 3h)" in out
    assert "mañana te confirmo" in out
    # >24h → "hace Xd"
    assert "**María** (hace 3d)" in out
    assert "hola!" in out


def test_format_whatsapp_block_recent():
    """hours_waiting < 1 → 'recién' (no "0h" que queda raro)."""
    raw = json.dumps([{
        "jid": "a@s.whatsapp.net", "name": "X",
        "last_snippet": "test", "hours_waiting": 0.4,
    }])
    out = _format_whatsapp_block(raw)
    assert "**X** (recién)" in out


def test_format_whatsapp_block_empty():
    assert "Sin chats esperando" in _format_whatsapp_block(json.dumps([]))


def test_format_whatsapp_block_malformed():
    """JSON roto → passthrough bajo el header (no crash)."""
    out = _format_whatsapp_block("not json")
    assert out.startswith("### WhatsApp\n")


def test_format_whatsapp_dispatcher_wired():
    raw = json.dumps([{
        "jid": "a@s.whatsapp.net", "name": "X",
        "last_snippet": "hi", "hours_waiting": 2.0,
    }])
    direct = _format_whatsapp_block(raw)
    dispatched = _format_forced_tool_output("whatsapp_pending", raw)
    assert direct == dispatched


# ── 4. Empty-state detection ───────────────────────────────────────────────


def test_is_empty_whatsapp_pending_empty_list():
    assert _is_empty_tool_output("whatsapp_pending", "[]") is True


def test_is_empty_whatsapp_pending_with_chats():
    raw = json.dumps([{"jid": "x", "name": "Y", "last_snippet": "", "hours_waiting": 0}])
    assert _is_empty_tool_output("whatsapp_pending", raw) is False


def test_is_empty_whatsapp_pending_malformed():
    assert _is_empty_tool_output("whatsapp_pending", "not json") is False


# ── 5. Source-intent meta ──────────────────────────────────────────────────


def test_whatsapp_source_intent_meta():
    meta = _SOURCE_INTENT_META["whatsapp_pending"]
    assert meta["label"] == "tus chats de WhatsApp esperando respuesta"
    assert meta["live_section"] == "### WhatsApp"
    assert "WhatsApp" in meta["empty_phrase"]


def test_whatsapp_source_intent_hint():
    hint = _build_source_intent_hint(["whatsapp_pending"])
    assert hint is not None
    assert "WhatsApp" in hint
    assert "### WhatsApp" in hint
    assert "No hay chats de WhatsApp" in hint


# ── 6. End-to-end SSE: sources emit con wa.me link ────────────────────────


_EVENT_RE = re.compile(r"event: (?P<event>[^\n]+)\ndata: (?P<data>[^\n]*)\n\n")


def test_whatsapp_sources_emitted_in_sse_with_wa_me_link(monkeypatch):
    """El stream SSE emite un segundo evento `sources` con links wa.me
    prepended cuando whatsapp_pending encuentra chats. Verifica el mismo
    patrón que drive_search pero para WA."""
    server_mod._CHAT_BUCKETS.clear()
    monkeypatch.setattr(server_mod, "_ollama_alive", lambda timeout=2.0: True)
    monkeypatch.setattr(server_mod, "_ollama_chat_probe", lambda timeout_s=6.0: True)
    monkeypatch.setattr(server_mod, "_fetch_whatsapp_unread", lambda *a, **kw: [])
    monkeypatch.setattr(server_mod, "_persist_conversation_turn", lambda *a, **kw: None)
    monkeypatch.setattr(server_mod, "save_session", lambda sess: None)
    monkeypatch.setattr(server_mod, "log_query_event", lambda ev: None)
    monkeypatch.setattr(server_mod, "_chat_cache_get", lambda key: None)
    monkeypatch.setattr(server_mod, "_chat_cache_put", lambda key, val: None)
    monkeypatch.setattr(server_mod, "_is_tasks_query", lambda q: False)

    import rag as _rag_mod
    monkeypatch.setattr(_rag_mod, "build_person_context", lambda q: None)
    monkeypatch.setattr(
        server_mod, "multi_retrieve",
        lambda *a, **kw: {
            "docs": ["vault doc"],
            "metas": [{"file": "03-Resources/test.md", "note": "test", "folder": "03-Resources"}],
            "scores": [0.4],
            "confidence": 0.4,
            "search_query": a[1] if len(a) >= 2 else "x",
            "filters_applied": {},
            "query_variants": [],
            "vault_scope": ["default"],
        },
    )

    # Mock whatsapp_pending to return 2 chats: 1 DM + 1 group.
    _wa_result = [
        {"jid": "5491122334455@s.whatsapp.net", "name": "Juan",
         "last_snippet": "hola", "hours_waiting": 3.0},
        {"jid": "120363999999999@g.us", "name": "Família",
         "last_snippet": "cuándo llegás?", "hours_waiting": 1.0},
    ]
    monkeypatch.setitem(
        server_mod.TOOL_FNS, "whatsapp_pending",
        lambda hours=48, max_chats=10: json.dumps(_wa_result),
    )
    # Stub other tools potentially also triggered by the planning pattern.
    monkeypatch.setitem(server_mod.TOOL_FNS, "reminders_due", lambda **kw: "[]")
    monkeypatch.setitem(server_mod.TOOL_FNS, "calendar_ahead", lambda **kw: "[]")

    # Mock ollama.chat: empty tool_calls (bypass tool-deciding) + one
    # streaming token for the final answer.
    class _Mock:
        def __call__(self, *a, **kw):
            if kw.get("stream"):
                return iter([SimpleNamespace(message=SimpleNamespace(content="ok"))])
            return SimpleNamespace(message=SimpleNamespace(content="", tool_calls=None))
    import ollama
    monkeypatch.setattr(ollama, "chat", _Mock())

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"question": "qué tengo pendiente esta semana", "vault_scope": None},
    )
    assert resp.status_code == 200, resp.text

    events = []
    for m in _EVENT_RE.finditer(resp.text):
        try:
            payload = json.loads(m.group("data"))
        except Exception:
            payload = {}
        events.append((m.group("event"), payload))

    sources_events = [p for ev, p in events if ev == "sources"]
    assert len(sources_events) >= 2, (
        f"esperaba ≥2 eventos sources (inicial + post-tool WA), "
        f"vi {len(sources_events)}; eventos: {[ev for ev,_ in events]}"
    )
    last_items = sources_events[-1].get("items", [])
    assert last_items, "último 'sources' vacío"

    # Los primeros items deben ser WA (folder="WhatsApp"); DM usa wa.me,
    # grupo cae a web.whatsapp.com root.
    wa_items = [it for it in last_items if it.get("folder") == "WhatsApp"]
    assert len(wa_items) == 2, f"esperaba 2 items WA, vi {len(wa_items)}: {last_items}"

    juan = [it for it in wa_items if it["note"] == "Juan"][0]
    assert juan["file"] == "https://wa.me/5491122334455"
    assert juan["score"] == 5.0
    assert juan["bar"] == "■■■■■"

    grupo = [it for it in wa_items if it["note"] == "Família"][0]
    # Grupos usan hash fragment con el JID para que el dedup del frontend
    # (keyed por `s.file`) no colapse grupos distintos al mismo URL.
    assert grupo["file"] == "https://web.whatsapp.com/#120363999999999@g.us"

    # Vault source sigue presente.
    vault_items = [it for it in last_items if it.get("folder") not in ("WhatsApp", "Google Drive")]
    assert any(it.get("file") == "03-Resources/test.md" for it in vault_items), (
        f"vault source se perdió: {last_items}"
    )
