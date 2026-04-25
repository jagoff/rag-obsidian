"""Tests for the WhatsApp content-search chat tool.

Origin: el RAG ya tenía `whatsapp_pending` (lista de chats sin
respuesta) pero no había forma de buscar DENTRO del contenido de los
mensajes. El user pedía cosas como "qué me dijo Juan sobre la deuda?"
o "cuándo quedamos con Grecia para el doctor?" y el chat sólo podía
responder "no sé, no tengo acceso a tus chats" — aún cuando el corpus
WA tenía 4500+ chunks indexados con `source="whatsapp"`.

Este archivo cubre el fix en 4 capas (bottom-up), igual estilo que
`test_drive_search_tool.py`:

1. `_agent_tool_whatsapp_search` — helper de rag con `retrieve()` y
   `_whatsapp_jid_from_contact` mockeados.
2. `whatsapp_search` en `web.tools` — wrapper silent-fail + cap en k.
3. Pre-router regex + `_format_whatsapp_search_block` +
   `_SOURCE_INTENT_META` — el glue layer que el user toca al chatear.
4. Module-level: que el tool esté en `CHAT_TOOLS`, `TOOL_FNS`,
   `PARALLEL_SAFE` (read-only es safe en paralelo) y NO en
   `PROPOSAL_TOOL_NAMES` (no es destructivo).

Todos los tests son puros — no tocan TestClient ni la red. `retrieve`
se monkeypatchea via `rag.retrieve` devolviendo dicts canned.
"""
from __future__ import annotations

import json

import pytest

import rag
from web import tools as tools_mod
from web.server import (
    _SOURCE_INTENT_META,
    _build_source_intent_hint,
    _detect_tool_intent,
    _format_forced_tool_output,
    _format_whatsapp_search_block,
    _is_empty_tool_output,
)


# ── Canned retrieve helpers ────────────────────────────────────────────────


def _wa_meta(
    *,
    chat_jid: str = "5491134567890@s.whatsapp.net",
    chat_name: str = "Juan Pérez",
    sender: str = "Juan Pérez",
    first_ts: float = 1745169790.0,  # 2025-04-20 ~15:23:10 UTC
    note: str | None = None,
) -> dict:
    return {
        "file": f"whatsapp://{chat_jid}/msg-1",
        "note": note or f"WA: {chat_name}",
        "folder": "",
        "tags": "",
        "source": "whatsapp",
        "created_ts": first_ts,
        "chat_jid": chat_jid,
        "chat_name": chat_name,
        "sender": sender,
        "first_msg_id": "msg-1",
        "last_msg_id": "msg-2",
        "first_ts": first_ts,
        "last_ts": first_ts + 60.0,
        "parent": "",
    }


def _make_retrieve_result(items: list[tuple[str, dict, float]]) -> dict:
    """Build a `retrieve()`-shaped dict from `(doc, meta, score)` tuples."""
    return {
        "docs": [it[0] for it in items],
        "metas": [it[1] for it in items],
        "scores": [it[2] for it in items],
        "confidence": items[0][2] if items else 0.0,
        "search_query": "x",
        "filters_applied": {},
        "query_variants": [],
        "vault_scope": ["default"],
    }


# ── 1. `_agent_tool_whatsapp_search` helper ────────────────────────────────


def test_whatsapp_search_empty_query_short_circuits(monkeypatch):
    """Empty / whitespace-only queries short-circuit con error explícito,
    sin pegarle a `retrieve()`."""
    def _boom(*_a, **_kw):
        pytest.fail("retrieve no debió ser llamado para query vacía")
    monkeypatch.setattr(rag, "retrieve", _boom)

    out = rag._agent_tool_whatsapp_search("")
    parsed = json.loads(out)
    assert parsed["messages"] == []
    assert parsed["error"] == "query vacía"
    assert parsed["query"] == ""
    assert parsed["contact_filter"] is None

    # Whitespace also.
    out2 = rag._agent_tool_whatsapp_search("   \t  ")
    assert json.loads(out2)["error"] == "query vacía"


def test_whatsapp_search_returns_top_k_unfiltered(monkeypatch):
    """Sin filtro de contacto, devuelve los top-k chunks de retrieve
    formateados como messages con who/ts/snippet/score."""
    items = [
        (
            "Juan Pérez: la deuda de la macbook quedó saldada\n"
            "yo: gracias",
            _wa_meta(chat_name="Juan Pérez", sender="Juan Pérez"),
            0.87,
        ),
        (
            "yo: te debo 5000 todavía\n"
            "María: no problem dejá",
            _wa_meta(
                chat_jid="5491198765432@s.whatsapp.net",
                chat_name="María",
                sender="yo",
                first_ts=1744000000.0,
            ),
            0.62,
        ),
    ]
    captured: dict = {}
    def _fake_retrieve(col, q, k, **kwargs):
        captured["q"] = q
        captured["k"] = k
        captured["kwargs"] = kwargs
        return _make_retrieve_result(items)
    monkeypatch.setattr(rag, "retrieve", _fake_retrieve)
    monkeypatch.setattr(rag, "get_db", lambda: object())

    out = rag._agent_tool_whatsapp_search("deuda macbook", k=5)
    parsed = json.loads(out)

    assert parsed["query"] == "deuda macbook"
    assert parsed["contact_filter"] is None
    assert "warning" not in parsed
    assert len(parsed["messages"]) == 2

    m0 = parsed["messages"][0]
    assert m0["jid"] == "5491134567890@s.whatsapp.net"
    assert m0["contact"] == "Juan Pérez"
    assert m0["who"] == "inbound"  # sender != "yo"
    assert m0["score"] == 0.87
    assert "deuda de la macbook" in m0["text"]
    # ISO UTC con sufijo "Z".
    assert m0["ts"].endswith("Z")
    assert m0["ts"].startswith("2025-04-20T")

    m1 = parsed["messages"][1]
    assert m1["who"] == "outbound"  # sender == "yo"
    assert m1["contact"] == "María"

    # retrieve fue llamado con source="whatsapp" y k=5 (no overfetch sin filtro).
    assert captured["k"] == 5
    assert captured["kwargs"].get("source") == "whatsapp"
    # multi_query=False: WA-only short-circuits paraphrase generation.
    assert captured["kwargs"].get("multi_query") is False


def test_whatsapp_search_caps_k_at_8(monkeypatch):
    """`k=20` se capa silenciosamente a 8 (max param documentado)."""
    captured: dict = {}
    def _fake_retrieve(col, q, k, **kwargs):
        captured["k"] = k
        return _make_retrieve_result([])
    monkeypatch.setattr(rag, "retrieve", _fake_retrieve)
    monkeypatch.setattr(rag, "get_db", lambda: object())

    rag._agent_tool_whatsapp_search("test", k=20)
    # Sin filtro, k pasa directo (capado a 8).
    assert captured["k"] == 8


def test_whatsapp_search_overfetches_when_filtering(monkeypatch):
    """Con filtro de contacto, retrieve recibe k×3 (cap 30) para
    compensar chunks de otros chats que vendrían por score primero."""
    captured: dict = {}
    def _fake_retrieve(col, q, k, **kwargs):
        captured["k"] = k
        return _make_retrieve_result([])
    monkeypatch.setattr(rag, "retrieve", _fake_retrieve)
    monkeypatch.setattr(rag, "get_db", lambda: object())
    # Mock contact lookup: returns valid jid → triggers filter path.
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact",
                        lambda name: {
                            "jid": "5491134567890@s.whatsapp.net",
                            "full_name": "Juan Pérez",
                            "phones": ["+54 9 11 3456-7890"],
                            "error": None,
                        })

    rag._agent_tool_whatsapp_search("deuda", contact="Juan", k=5)
    # k=5 × 3 = 15 (under cap 30).
    assert captured["k"] == 15

    # k=8 × 3 = 24, still under cap.
    rag._agent_tool_whatsapp_search("deuda", contact="Juan", k=8)
    assert captured["k"] == 24


def test_whatsapp_search_filters_by_contact_jid(monkeypatch):
    """Con `contact='Juan'`, sólo los chunks cuyo `chat_jid` matchee el
    suffix de 10 dígitos del teléfono de Juan deben pasar el filtro."""
    juan_jid = "5491134567890@s.whatsapp.net"
    other_jid = "5491198765432@s.whatsapp.net"
    items = [
        # Juan match — last 10 digits of jid (1134567890) match phone last 10.
        ("Juan: deuda saldada", _wa_meta(chat_jid=juan_jid, chat_name="Juan",
                                         sender="Juan"), 0.9),
        # Other chat — should be filtered out.
        ("María: hola", _wa_meta(chat_jid=other_jid, chat_name="María",
                                 sender="María"), 0.85),
        # Another Juan match.
        ("yo: te pago el viernes",
         _wa_meta(chat_jid=juan_jid, chat_name="Juan", sender="yo",
                  first_ts=1745255000.0),
         0.8),
    ]
    monkeypatch.setattr(rag, "retrieve",
                        lambda col, q, k, **kw: _make_retrieve_result(items))
    monkeypatch.setattr(rag, "get_db", lambda: object())
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact",
                        lambda name: {
                            "jid": juan_jid,
                            "full_name": "Juan Pérez",
                            "phones": ["+54 9 11 3456-7890"],
                            "error": None,
                        })

    out = rag._agent_tool_whatsapp_search("deuda", contact="Juan", k=5)
    parsed = json.loads(out)

    assert parsed["contact_filter"] == "Juan Pérez"
    assert "warning" not in parsed
    assert len(parsed["messages"]) == 2
    assert all(m["jid"] == juan_jid for m in parsed["messages"])
    assert {m["who"] for m in parsed["messages"]} == {"inbound", "outbound"}


def test_whatsapp_search_contact_not_resolved_fallback(monkeypatch):
    """Si el contacto no se resuelve, hacemos búsqueda sin filtro y
    agregamos un `warning` al output. NO devolvemos error — el search
    sigue siendo útil."""
    items = [
        ("Cualquiera: contenido", _wa_meta(), 0.7),
    ]
    monkeypatch.setattr(rag, "retrieve",
                        lambda col, q, k, **kw: _make_retrieve_result(items))
    monkeypatch.setattr(rag, "get_db", lambda: object())
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact",
                        lambda name: {
                            "jid": None, "full_name": None, "phones": [],
                            "error": "not_found",
                        })

    out = rag._agent_tool_whatsapp_search("hola", contact="NombreInexistente")
    parsed = json.loads(out)

    assert parsed["contact_filter"] is None
    assert "warning" in parsed
    assert "NombreInexistente" in parsed["warning"]
    assert "not_found" in parsed["warning"]
    # Seguimos devolviendo los messages (no había filtro JID).
    assert len(parsed["messages"]) == 1


def test_whatsapp_search_contact_lookup_raises_is_treated_as_unresolved(monkeypatch):
    """Si `_whatsapp_jid_from_contact` raisea (osascript hangea, bridge
    caído, etc), atrapamos y caemos al modo unfiltered + warning. NO
    debemos propagar la excepción."""
    monkeypatch.setattr(rag, "retrieve",
                        lambda col, q, k, **kw: _make_retrieve_result([]))
    monkeypatch.setattr(rag, "get_db", lambda: object())
    def _boom(*_a, **_kw):
        raise RuntimeError("osascript timeout")
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact", _boom)

    out = rag._agent_tool_whatsapp_search("hola", contact="Pepe")
    parsed = json.loads(out)
    assert parsed["contact_filter"] is None
    assert "warning" in parsed
    assert "lookup_failed" in parsed["warning"]


def test_whatsapp_search_empty_corpus_returns_empty_list(monkeypatch):
    """Cuando retrieve devuelve docs=[], el output es `messages: []` sin
    error — el corpus está vacío o no hay matches, ambos válidos."""
    monkeypatch.setattr(rag, "retrieve",
                        lambda col, q, k, **kw: _make_retrieve_result([]))
    monkeypatch.setattr(rag, "get_db", lambda: object())

    out = rag._agent_tool_whatsapp_search("foo bar baz")
    parsed = json.loads(out)
    assert parsed["messages"] == []
    assert "error" not in parsed
    assert "warning" not in parsed


def test_whatsapp_search_retrieve_error_silent_fail(monkeypatch):
    """Si `retrieve()` raisea, el helper devuelve un error JSON pero
    NUNCA propaga la excepción — el chat loop se rompe si lo hace."""
    def _boom(*_a, **_kw):
        raise RuntimeError("vec_db corrupted")
    monkeypatch.setattr(rag, "retrieve", _boom)
    monkeypatch.setattr(rag, "get_db", lambda: object())

    out = rag._agent_tool_whatsapp_search("hola")
    parsed = json.loads(out)
    assert parsed["messages"] == []
    assert parsed["error"].startswith("retrieve_failed")
    assert "vec_db corrupted" in parsed["error"]


def test_whatsapp_search_snippet_capped_at_400_chars(monkeypatch):
    """Cada snippet en `messages[].text` se capa a 400 chars — evita
    inflar el CONTEXTO con chunks de 2-3KB."""
    long_doc = "x" * 1000
    items = [(long_doc, _wa_meta(), 0.5)]
    monkeypatch.setattr(rag, "retrieve",
                        lambda col, q, k, **kw: _make_retrieve_result(items))
    monkeypatch.setattr(rag, "get_db", lambda: object())

    out = rag._agent_tool_whatsapp_search("test")
    parsed = json.loads(out)
    assert len(parsed["messages"][0]["text"]) == 400


def test_whatsapp_search_who_classification(monkeypatch):
    """`who` es "outbound" sólo cuando el sender del chunk es
    exactamente "yo" (case-insensitive). Cualquier otra cosa (nombre
    resuelto, "…3891" mask, "?" fallback) → "inbound"."""
    items = [
        ("yo: hola", _wa_meta(sender="yo"), 0.9),
        ("Yo: hola2", _wa_meta(sender="Yo"), 0.85),  # case-insensitive
        ("Juan: respuesta", _wa_meta(sender="Juan Pérez"), 0.8),
        ("…3891: ping", _wa_meta(sender="…3891"), 0.7),
        ("?: anon", _wa_meta(sender=""), 0.6),  # empty → "inbound"
    ]
    monkeypatch.setattr(rag, "retrieve",
                        lambda col, q, k, **kw: _make_retrieve_result(items))
    monkeypatch.setattr(rag, "get_db", lambda: object())

    parsed = json.loads(rag._agent_tool_whatsapp_search("test"))
    whos = [m["who"] for m in parsed["messages"]]
    assert whos == ["outbound", "outbound", "inbound", "inbound", "inbound"]


def test_whatsapp_search_truncates_to_k(monkeypatch):
    """Si retrieve devuelve más chunks que `k`, sólo devolvemos los
    primeros k después del filtro."""
    items = [(f"doc {i}", _wa_meta(first_ts=1745000000.0 + i), 0.9 - i * 0.05)
             for i in range(10)]
    monkeypatch.setattr(rag, "retrieve",
                        lambda col, q, k, **kw: _make_retrieve_result(items))
    monkeypatch.setattr(rag, "get_db", lambda: object())

    parsed = json.loads(rag._agent_tool_whatsapp_search("test", k=3))
    assert len(parsed["messages"]) == 3


# ── 2. `web.tools` wrapper ─────────────────────────────────────────────────


def test_web_tools_exposes_whatsapp_search():
    """Wrapper registrado en `CHAT_TOOLS`, `TOOL_FNS` y `PARALLEL_SAFE`
    (es read-only — safe en paralelo). NO debe estar en
    `PROPOSAL_TOOL_NAMES` (no es destructivo, no necesita confirmación)."""
    assert "whatsapp_search" in tools_mod.TOOL_FNS
    assert tools_mod.whatsapp_search in tools_mod.CHAT_TOOLS
    assert "whatsapp_search" in tools_mod.PARALLEL_SAFE
    assert "whatsapp_search" not in tools_mod.PROPOSAL_TOOL_NAMES
    assert (tools_mod.whatsapp_search.__doc__ and
            "WhatsApp" in tools_mod.whatsapp_search.__doc__)


def test_web_tool_addendum_mentions_whatsapp_search():
    """El addendum tiene una entrada explícita para `whatsapp_search`
    distinta de `whatsapp_pending` — el LLM tiene que poder distinguir
    entre "buscar contenido" y "listar chats abiertos"."""
    addendum = tools_mod._WEB_TOOL_ADDENDUM
    assert "whatsapp_search" in addendum
    assert "whatsapp_pending" in addendum
    # El routing hint clave: dijo / mandó / hablamos / charlamos.
    assert "qué me dijo" in addendum.lower() or "qué me mandó" in addendum.lower()
    # Indica que es DISTINTO de whatsapp_pending (búsqueda por contenido).
    assert "contenido" in addendum.lower()


def test_web_wrapper_caps_k(monkeypatch):
    """`whatsapp_search` wrapper capea k a 8 antes de delegar — no
    queremos que un user con LLM verboso meta k=999 al helper."""
    captured: dict = {}
    def _fake(query, contact=None, k=8):
        captured["k"] = k
        return '{"messages": []}'
    monkeypatch.setattr(tools_mod, "_agent_tool_whatsapp_search", _fake)

    tools_mod.whatsapp_search("test", k=99)
    assert captured["k"] == 8

    tools_mod.whatsapp_search("test", k=0)
    assert captured["k"] == 1  # piso k=1


def test_web_wrapper_passes_contact(monkeypatch):
    """El wrapper passthrough el `contact` al helper sin transformar."""
    captured: dict = {}
    def _fake(query, contact=None, k=5):
        captured["contact"] = contact
        return '{"messages": []}'
    monkeypatch.setattr(tools_mod, "_agent_tool_whatsapp_search", _fake)

    tools_mod.whatsapp_search("hola", contact="Grecia")
    assert captured["contact"] == "Grecia"

    tools_mod.whatsapp_search("hola")
    assert captured["contact"] is None


# ── 3. Pre-router regex ────────────────────────────────────────────────────


@pytest.mark.parametrize("query", [
    "qué me dijo Juan sobre la deuda",
    "qué me dijeron en el grupo de la familia",
    "qué me mandó Grecia ayer",
    "qué me escribió Federico la semana pasada",
    "qué me comentó María del proyecto",
    "qué me contó Pepe del partido",
    "cuándo quedamos con Juan",
    "cuándo hablamos de la mudanza",
    "cuándo charlamos de la propiedad",
    "el chat donde Grecia mencionó la cita",
])
def test_pre_router_fires_whatsapp_search(query: str):
    """Las frases de "comunicación pasada" disparan whatsapp_search."""
    matched = _detect_tool_intent(query)
    names = [n for n, _ in matched]
    assert "whatsapp_search" in names, (
        f"Query {query!r} debería haber matcheado whatsapp_search; matched={names}"
    )
    # Cuando matchea, el arg `query` es la pregunta cruda — el LLM /
    # tool resuelven `contact` ellos mismos.
    for n, args in matched:
        if n == "whatsapp_search":
            assert args == {"query": query}


@pytest.mark.parametrize("query", [
    # Estos disparan `whatsapp_pending`, NO `whatsapp_search`. Importante
    # que la regla nueva no se solape con la pre-existente.
    "qué tengo pendiente esta semana",
    "qué chats tengo sin responder",
    "tengo mensajes sin contestar?",
    "mostrame los chats de whatsapp",
    "wzp pendientes",
])
def test_pre_router_does_not_hijack_whatsapp_pending(query: str):
    """Estos triggers deben ir a `whatsapp_pending`, NO a `whatsapp_search`.
    Críticamente: si `whatsapp_search` se dispara también, el comportamiento
    cambia (forzamos un retrieve en vez de listar chats sin reply)."""
    names = [n for n, _ in _detect_tool_intent(query)]
    assert "whatsapp_pending" in names, (
        f"Query {query!r} debería haber matcheado whatsapp_pending; matched={names}"
    )
    assert "whatsapp_search" not in names, (
        f"Query {query!r} NO debería disparar whatsapp_search también; "
        f"matched={names} — la regla solapa con whatsapp_pending"
    )


@pytest.mark.parametrize("query", [
    # Queries que NO mencionan comunicación pasada NI chats.
    "cuál es la capital de Francia?",
    "buscá notas sobre coaching",
    "cómo está el clima hoy",
    "agendame un evento mañana 10am",
])
def test_pre_router_no_false_positives_on_neutral_queries(query: str):
    """Queries neutrales no deben disparar whatsapp_search."""
    names = [n for n, _ in _detect_tool_intent(query)]
    assert "whatsapp_search" not in names, (
        f"Query {query!r} NO debería haber matcheado whatsapp_search; matched={names}"
    )


# ── 4. `_format_whatsapp_search_block` renderer ─────────────────────────────


def test_format_block_happy_path():
    raw = json.dumps({
        "query": "deuda macbook",
        "contact_filter": "Juan Pérez",
        "messages": [
            {
                "jid": "5491134567890@s.whatsapp.net",
                "contact": "Juan Pérez",
                "ts": "2026-04-20T15:23:10Z",
                "who": "inbound",
                "text": "la deuda de la macbook quedó saldada el viernes",
                "score": 0.87,
            },
            {
                "jid": "5491134567890@s.whatsapp.net",
                "contact": "Juan Pérez",
                "ts": "2026-04-15T09:00:00Z",
                "who": "outbound",
                "text": "te paso la transferencia hoy",
                "score": 0.71,
            },
        ],
    })
    out = _format_whatsapp_search_block(raw)
    # Header con contact + count.
    assert out.startswith("### WhatsApp (")
    assert "Juan Pérez" in out
    assert "2 mensajes" in out
    # Cada bullet tiene shape `[contact · YYYY-MM-DD]`.
    assert "[Juan Pérez · 2026-04-20]" in out
    assert "deuda de la macbook" in out
    # Outbound prefija "yo →".
    assert "yo → te paso la transferencia hoy" in out


def test_format_block_empty_messages_with_contact():
    raw = json.dumps({
        "query": "deuda", "contact_filter": "Juan Pérez",
        "messages": [],
    })
    out = _format_whatsapp_search_block(raw)
    assert "Sin resultados" in out
    assert "Juan Pérez" in out


def test_format_block_empty_messages_no_contact():
    raw = json.dumps({"query": "x", "contact_filter": None, "messages": []})
    out = _format_whatsapp_search_block(raw)
    assert "Sin resultados" in out
    assert "corpus de WhatsApp" in out


def test_format_block_with_warning():
    """Si el contacto no se resolvió, mostramos un aviso explícito en
    el bloque para que el LLM se lo cuente al user."""
    raw = json.dumps({
        "query": "hola", "contact_filter": None,
        "messages": [{
            "jid": "x", "contact": "Anyone", "ts": "2026-04-20T15:23:10Z",
            "who": "inbound", "text": "hola", "score": 0.5,
        }],
        "warning": "contact_not_resolved: 'Pepe' no se encontró.",
    })
    out = _format_whatsapp_search_block(raw)
    assert "Aviso" in out
    assert "Pepe" in out


def test_format_block_error_state():
    raw = json.dumps({
        "query": "hola", "contact_filter": None, "messages": [],
        "error": "retrieve_failed: vec_db corrupted",
    })
    out = _format_whatsapp_search_block(raw)
    assert "Sin resultados" in out
    assert "falló el retrieval" in out


def test_format_block_malformed_json_falls_through():
    """JSON inválido cae a passthrough crudo bajo el header — el LLM
    nunca lo recibe en seco sin contexto."""
    out = _format_whatsapp_search_block("not json at all")
    assert out.startswith("### WhatsApp")
    assert "not json at all" in out


def test_format_block_wired_into_dispatcher():
    """`_format_forced_tool_output('whatsapp_search', ...)` despacha al
    renderer correcto."""
    raw = json.dumps({
        "query": "x", "contact_filter": None,
        "messages": [{
            "jid": "x", "contact": "Y", "ts": "", "who": "inbound",
            "text": "z", "score": 0.5,
        }],
    })
    dispatched = _format_forced_tool_output("whatsapp_search", raw)
    direct = _format_whatsapp_search_block(raw)
    assert dispatched == direct


# ── 5. Empty-state detection + source-intent hint ──────────────────────────


def test_is_empty_whatsapp_search_no_messages():
    raw = json.dumps({"messages": [], "query": "x", "contact_filter": None})
    assert _is_empty_tool_output("whatsapp_search", raw) is True


def test_is_empty_whatsapp_search_with_messages():
    raw = json.dumps({
        "messages": [{"jid": "x", "contact": "Y", "ts": "", "who": "inbound",
                      "text": "z", "score": 0.5}],
        "query": "x",
    })
    assert _is_empty_tool_output("whatsapp_search", raw) is False


def test_is_empty_whatsapp_search_malformed_returns_false():
    """JSON roto → conservador (False) — dejamos que el LLM decida con
    el string crudo si el shape no es lo esperado."""
    assert _is_empty_tool_output("whatsapp_search", "not json") is False


def test_source_intent_meta_has_whatsapp_search():
    meta = _SOURCE_INTENT_META["whatsapp_search"]
    assert "WhatsApp" in meta["label"]
    assert meta["live_section"] == "### WhatsApp"
    assert "WhatsApp" in meta["empty_phrase"]


def test_source_intent_hint_for_whatsapp_search():
    hint = _build_source_intent_hint(["whatsapp_search"])
    assert hint is not None
    assert "WhatsApp" in hint
    assert "### WhatsApp" in hint
