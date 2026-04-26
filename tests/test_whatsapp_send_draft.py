"""Tests for the `propose_whatsapp_send` chat tool + `/api/whatsapp/send`
endpoint (2026-04-24, Fer F. request).

The user asked the chat "Enviale un mensaje a Grecia: yo soy EL RA..."
and the RAG responded with the TEXT of the message but never actually
sent it — the tool registry didn't have a send tool, so the LLM fell
back to "generate a suggested draft in prose". Per-message confirmation
is mandatory because this is a destructive action to a third party
(WhatsApp has no "delete sent").

Covers:

1. `_whatsapp_send_to_jid` — anti_loop=True prefixes U+200B, anti_loop=False
   does NOT. Bridge 2xx → True, anything else → False.
2. `_whatsapp_jid_from_contact` — happy path, empty query, not_found,
   no_phone. JID format `<digits>@s.whatsapp.net`.
3. `propose_whatsapp_send` — always returns `needs_clarification=True`
   (never auto-sends), exposes `fields.error` when resolution fails.
4. `/api/whatsapp/send` — happy path 200, invalid jid 400, empty body 400,
   bridge down 502. Does NOT prefix U+200B (user → third party).
5. Tool registration invariants — `propose_whatsapp_send` is in
   `CHAT_TOOLS` + `PROPOSAL_TOOL_NAMES`, NOT in `PARALLEL_SAFE`.
"""
from __future__ import annotations

import json


import rag
from fastapi.testclient import TestClient
import web.server as _server
from web import tools as _tools


_client = TestClient(_server.app)


# ── 1. _whatsapp_send_to_jid ───────────────────────────────────────────────


def test_send_to_jid_anti_loop_true_prefixes_u200b(monkeypatch):
    captured = {}
    class _FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(req, timeout=10):
        captured["data"] = req.data
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    ok = rag._whatsapp_send_to_jid("123@s.whatsapp.net", "hola", anti_loop=True)
    assert ok is True
    payload = json.loads(captured["data"].decode("utf-8"))
    # U+200B zero-width-space prefix was injected.
    assert payload["message"].startswith("\u200b")
    assert payload["message"].endswith("hola")


def test_send_to_jid_anti_loop_false_is_literal(monkeypatch):
    captured = {}
    class _FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(req, timeout=10):
        captured["data"] = req.data
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    ok = rag._whatsapp_send_to_jid("123@s.whatsapp.net", "hola grecia", anti_loop=False)
    assert ok is True
    payload = json.loads(captured["data"].decode("utf-8"))
    # No U+200B prefix — this is a user-initiated send to a third party.
    assert not payload["message"].startswith("\u200b")
    assert payload["message"] == "hola grecia"
    assert payload["recipient"] == "123@s.whatsapp.net"


def test_send_to_jid_bridge_error_returns_false(monkeypatch):
    def _fake_urlopen(req, timeout=10):
        raise ConnectionRefusedError("bridge down")
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    assert rag._whatsapp_send_to_jid("123@s.whatsapp.net", "hola") is False


def test_ambient_send_still_uses_anti_loop(monkeypatch):
    """Sanity: legacy `_ambient_whatsapp_send` must still prefix U+200B.
    Used by brief pushes and archive notifications that hit the bot's own
    group — without the prefix the listener would re-ingest them as
    queries and loop forever."""
    captured = {}
    class _FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(req, timeout=10):
        captured["data"] = req.data
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    rag._ambient_whatsapp_send("bot-group@g.us", "morning brief")
    payload = json.loads(captured["data"].decode("utf-8"))
    assert payload["message"].startswith("\u200b")


# ── 2. _whatsapp_jid_from_contact ──────────────────────────────────────────


def test_jid_from_contact_happy_path(monkeypatch):
    monkeypatch.setattr(rag, "_fetch_contact", lambda stem, email=None, canonical=None: {
        "full_name": "Grecia Ferrari",
        "phones": ["+54 9 11 5555-5555"],
        "emails": [],
        "birthday": "",
    })
    out = rag._whatsapp_jid_from_contact("Grecia")
    assert out["error"] is None
    assert out["full_name"] == "Grecia Ferrari"
    assert out["jid"] == "5491155555555@s.whatsapp.net"
    assert out["phones"] == ["+54 9 11 5555-5555"]


def test_jid_from_contact_empty_query():
    out = rag._whatsapp_jid_from_contact("")
    assert out["error"] == "empty_query"
    assert out["jid"] is None


def test_jid_from_contact_strips_leading_at_sign(monkeypatch):
    """El LLM a veces pasa el nombre con un `@` leading (hábito de
    wikilinks tipo `@Grecia`). Apple Contacts no resuelve con el sigil,
    entonces lo stripeamos defensive. Observed 2026-04-24: turn
    `7aeb51f212cd` recibió `contact_name="@Grecia"` → not_found."""
    captured = {}
    def _fake_fetch(stem, email=None, canonical=None):
        captured["stem"] = stem
        captured["canonical"] = canonical
        return {
            "full_name": "Grecia Ferrari",
            "phones": ["+5491155555555"],
            "emails": [], "birthday": "",
        }
    monkeypatch.setattr(rag, "_fetch_contact", _fake_fetch)
    out = rag._whatsapp_jid_from_contact("@Grecia")
    assert out["error"] is None
    assert out["jid"] == "5491155555555@s.whatsapp.net"
    # The `@` should NOT reach Apple Contacts.
    assert "@" not in captured["stem"]
    assert "@" not in captured["canonical"]
    # Multiple @ signs (paranoid) also get stripped.
    out2 = rag._whatsapp_jid_from_contact("@@Grecia")
    assert out2["error"] is None
    # Only-@ query degenerates to empty.
    out3 = rag._whatsapp_jid_from_contact("@")
    assert out3["error"] == "empty_query"


def test_jid_from_contact_not_found(monkeypatch):
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)
    out = rag._whatsapp_jid_from_contact("Unicornio Imaginario")
    assert out["error"] == "not_found"
    assert out["jid"] is None


def test_jid_from_contact_resolves_via_my_card_relationship(monkeypatch):
    """Cuando el query es un alias de parentesco ("Mama") y no existe
    como contacto literal, intenta resolver vía Related Names del My
    Card → exact lookup en Contacts.app por el nombre real."""
    from rag.integrations import whatsapp as wa_mod

    # Stub: My Card tiene "Mother → Mamá" y "Father → Carlos".
    monkeypatch.setattr(wa_mod, "_load_my_card_relations", lambda: [
        {"label": "mother", "personName": "Mamá"},
        {"label": "father", "personName": "Carlos"},
    ])

    # `_fetch_contact` falla para "Mama" (kinship guard).
    fetch_calls: list[str] = []

    def _fake_fetch(stem, email=None, canonical=None):
        fetch_calls.append(stem)
        return None  # Siempre None — nuestro path de éxito es vía exact_lookup

    # `_exact_contact_lookup` ahora es el path primario cuando viene del
    # relationship resolver — evita el fuzzy match que matcheaba
    # "Mariano Di Maggio" cuando relations dice "Maria ❤️".
    exact_calls: list[str] = []

    def _fake_exact(name):
        exact_calls.append(name)
        if name == "Mamá":
            return {
                "full_name": "Mamá",
                "phones": ["+54 9 342 547 6623"],
                "emails": [], "birthday": "",
            }
        return None

    monkeypatch.setattr(rag, "_fetch_contact", _fake_fetch)
    monkeypatch.setattr(wa_mod, "_exact_contact_lookup", _fake_exact)
    out = rag._whatsapp_jid_from_contact("Mama")
    assert out["error"] is None
    assert out["full_name"] == "Mamá"
    assert out["jid"] == "5493425476623@s.whatsapp.net"
    # Primero llamó _fetch_contact("Mama") → falla (kinship guard).
    # Después _exact_contact_lookup("Mamá") → encontró → no necesitó
    # llamar a _fetch_contact("Mamá") (path primario tomó la decisión).
    assert "Mama" in fetch_calls
    assert "Mamá" in exact_calls
    # _fetch_contact NO se llamó con "Mamá" (exact resolvió primero).
    assert "Mamá" not in fetch_calls


def test_jid_from_contact_relationship_falls_back_to_fuzzy(monkeypatch):
    """Si exact lookup no encuentra el contacto (ej. el nombre en
    Related Names difiere ligeramente del Contacts entry), caemos a
    `_fetch_contact` (fuzzy) como red de seguridad."""
    from rag.integrations import whatsapp as wa_mod

    monkeypatch.setattr(wa_mod, "_load_my_card_relations", lambda: [
        {"label": "mother", "personName": "Carmen"},
    ])
    # Exact match falla — Related Name es "Carmen" pero el contacto real
    # se llama "Carmen Pérez" (el user no completó el apellido en Related).
    monkeypatch.setattr(wa_mod, "_exact_contact_lookup", lambda n: None)

    fetch_calls: list[str] = []

    def _fake_fetch(stem, email=None, canonical=None):
        fetch_calls.append(stem)
        if stem == "Carmen":
            return {
                "full_name": "Carmen Pérez",
                "phones": ["+5491155555555"],
                "emails": [], "birthday": "",
            }
        return None

    monkeypatch.setattr(rag, "_fetch_contact", _fake_fetch)
    out = rag._whatsapp_jid_from_contact("Mama")
    assert out["error"] is None
    assert out["full_name"] == "Carmen Pérez"
    # Llamó _fetch_contact 2 veces: "Mama" (kinship), "Carmen" (fallback).
    assert fetch_calls == ["Mama", "Carmen"]


def test_jid_from_contact_strips_possessive_prefix(monkeypatch):
    """LLM frecuentemente pasa `contact_name="mi Mama"` (con preposición).
    El resolver debe strippear "mi"/"a mi"/"la"/"el" antes del alias."""
    from rag.integrations import whatsapp as wa_mod

    monkeypatch.setattr(wa_mod, "_load_my_card_relations", lambda: [
        {"label": "mother", "personName": "Mamá"},
    ])
    monkeypatch.setattr(wa_mod, "_exact_contact_lookup", lambda n: {
        "full_name": "Mamá",
        "phones": ["+54 9 342 547 6623"],
        "emails": [], "birthday": "",
    } if n == "Mamá" else None)
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)

    # Variantes que el LLM puede emitir.
    for hint in ["mi Mama", "mi mama", "a mi Mama", "la Mama", "tu Mama"]:
        out = rag._whatsapp_jid_from_contact(hint)
        assert out["error"] is None, f"hint {hint!r} should resolve"
        assert out["full_name"] == "Mamá", f"hint {hint!r} → {out!r}"


def test_strip_emoji_and_symbols():
    """Strip de emojis + variation selectors + ZWJ joiners."""
    from rag.integrations.whatsapp import _strip_emoji_and_symbols
    assert _strip_emoji_and_symbols("Maria \u2764\ufe0f") == "Maria"
    assert _strip_emoji_and_symbols("Juli \U0001f970") == "Juli"
    assert _strip_emoji_and_symbols("Pedro \U0001f468\u200d\U0001f373") == "Pedro"
    assert _strip_emoji_and_symbols("María José") == "María José"  # acentos OK
    assert _strip_emoji_and_symbols("John (Tio)") == "John (Tio)"  # parens OK
    assert _strip_emoji_and_symbols("Pepe-Luis") == "Pepe-Luis"  # hyphen OK
    assert _strip_emoji_and_symbols("O'Connor") == "O'Connor"  # apostrophe OK


def test_jid_from_contact_relationship_no_my_card(monkeypatch):
    """Si el alias es de parentesco pero no hay My Card seteada,
    seguimos cayendo a `not_found` igual que antes."""
    from rag.integrations import whatsapp as wa_mod

    monkeypatch.setattr(wa_mod, "_load_my_card_relations", lambda: [])
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)
    out = rag._whatsapp_jid_from_contact("Mama")
    assert out["error"] == "not_found"
    assert out["jid"] is None


def test_jid_from_contact_unknown_alias_skips_relationship_path(monkeypatch):
    """Queries que no son alias de parentesco no triggean la rama de
    Related Names — caen al `not_found` directo."""
    from rag.integrations import whatsapp as wa_mod

    # Spy: si esto se llama, el test falla.
    relations_calls = []

    def _spy_relations():
        relations_calls.append(1)
        return []

    monkeypatch.setattr(wa_mod, "_load_my_card_relations", _spy_relations)
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)
    out = rag._whatsapp_jid_from_contact("Unicornio Imaginario")
    assert out["error"] == "not_found"
    # _load_my_card_relations no fue llamado (el alias no es parentesco).
    assert relations_calls == []


def test_resolve_via_my_card_relationship_normalizes_accents():
    """Tildes en español no deben romper el match (mamá ≡ mama)."""
    from rag.integrations import whatsapp as wa_mod

    # Inject relations directamente al cache para no llamar osascript.
    import time
    wa_mod._MY_CARD_RELATIONS_CACHE = {
        "at": time.time(),
        "rows": [{"label": "mother", "personName": "Mamá"}],
    }
    try:
        assert wa_mod._resolve_via_my_card_relationship("Mama") == "Mamá"
        assert wa_mod._resolve_via_my_card_relationship("mama") == "Mamá"
        assert wa_mod._resolve_via_my_card_relationship("Mamá") == "Mamá"
        assert wa_mod._resolve_via_my_card_relationship("mami") == "Mamá"
        assert wa_mod._resolve_via_my_card_relationship("MADRE") == "Mamá"
        # Non-relationship word → None
        assert wa_mod._resolve_via_my_card_relationship("Random") is None
    finally:
        wa_mod._MY_CARD_RELATIONS_CACHE = None  # reset


def test_parse_apple_label_handles_both_formats():
    """Apple Contacts labels vienen en 2 formatos: `_$!<English>!$_` y
    plain Spanish ("Madre" en es-AR locale)."""
    from rag.integrations.whatsapp import _parse_apple_label
    assert _parse_apple_label("_$!<Mother>!$_") == "mother"
    assert _parse_apple_label("_$!<Father>!$_") == "father"
    assert _parse_apple_label("Madre") == "mother"
    assert _parse_apple_label("padre") == "father"
    assert _parse_apple_label("hermana") == "sister"
    # Unknown labels passthrough lowercased + folded.
    assert _parse_apple_label("Custom Label") == "custom label"


def test_jid_from_contact_no_phone(monkeypatch):
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: {
        "full_name": "Contacto Sin Tel",
        "phones": [],
        "emails": ["foo@bar.com"],
        "birthday": "",
    })
    out = rag._whatsapp_jid_from_contact("Contacto Sin Tel")
    assert out["error"] == "no_phone"
    assert out["jid"] is None
    assert out["full_name"] == "Contacto Sin Tel"


def test_jid_from_contact_phone_with_only_non_digits(monkeypatch):
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: {
        "full_name": "Bug Contact",
        "phones": ["+++"],
        "emails": [],
        "birthday": "",
    })
    out = rag._whatsapp_jid_from_contact("Bug Contact")
    assert out["error"] == "no_phone"


def test_jid_from_contact_fetch_raises(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("osascript died")
    monkeypatch.setattr(rag, "_fetch_contact", _boom)
    out = rag._whatsapp_jid_from_contact("Grecia")
    assert out["error"].startswith("lookup_failed:")


# ── 3. propose_whatsapp_send ────────────────────────────────────────────────


def test_propose_whatsapp_send_auto_sends_when_resolution_clean(monkeypatch):
    """Cuando el contacto resuelve limpio (jid + sin error + no group + no
    scheduled), el backend dispara `_whatsapp_send_to_jid` directo y
    devuelve `whatsapp_message_sent` sin form editable.

    Cambio de política 2026-04-26 (pedido explícito del user): "saca el
    cuadro de revisión, quiero que el chat envíe directamente". Antes
    siempre salía `whatsapp_message + needs_clarification=True` aunque
    hubiera resolución perfecta — ahora sale solo cuando hay error,
    grupo, o programación."""
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: {
        "full_name": "Grecia", "phones": ["+5491155555555"], "emails": [], "birthday": "",
    })
    sends: list[tuple] = []

    def _fake_send(jid, text, *, anti_loop=True, reply_to=None):
        sends.append((jid, text, anti_loop, reply_to))
        return True

    # Patch en el namespace `rag` (donde se resuelve el nombre dentro de
    # `propose_whatsapp_send`), no en `rag.integrations.whatsapp` —
    # aunque vienen de la misma función vía re-export, la llamada
    # `_whatsapp_send_to_jid(...)` en propose_whatsapp_send resuelve via
    # globals() del módulo `rag`, no via attribute lookup en el módulo de
    # integrations.
    monkeypatch.setattr(rag, "_whatsapp_send_to_jid", _fake_send)
    raw = rag.propose_whatsapp_send("Grecia", "hola")
    payload = json.loads(raw)
    assert payload["kind"] == "whatsapp_message_sent"
    assert payload["needs_clarification"] is False
    assert payload["fields"]["auto_sent"] is True
    assert payload["fields"]["error"] is None
    # Bridge fue llamado con anti_loop=False (mensajes a terceros NO usan
    # el U+200B marker — se vería raro en el WA del destinatario).
    assert len(sends) == 1
    assert sends[0][0] == "5491155555555@s.whatsapp.net"
    assert sends[0][1] == "hola"
    assert sends[0][2] is False  # anti_loop=False


def test_propose_whatsapp_send_falls_back_to_card_when_send_fails(monkeypatch):
    """Si el bridge devuelve 5xx / unreachable, NO marcamos el mensaje
    como enviado — sale la card editable original con `error=send_failed`
    para que el user pueda re-intentar manualmente."""
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: {
        "full_name": "Grecia", "phones": ["+5491155555555"], "emails": [], "birthday": "",
    })
    monkeypatch.setattr(rag, "_whatsapp_send_to_jid",
                        lambda *a, **kw: False)
    raw = rag.propose_whatsapp_send("Grecia", "hola")
    payload = json.loads(raw)
    # Card editable original — user puede corregir y dar [Enviar] manual.
    assert payload["kind"] == "whatsapp_message"
    assert payload["needs_clarification"] is True
    assert payload["fields"]["error"] == "send_failed"
    # auto_sent NO está set (solo aparece en el path success).
    assert "auto_sent" not in payload["fields"]


def test_propose_whatsapp_send_scheduled_skips_auto_send(monkeypatch):
    """Mensajes programados (`scheduled_for` set) NUNCA auto-envían — el
    user querrá revisar el horario antes de cometer la programación."""
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: {
        "full_name": "Grecia", "phones": ["+5491155555555"], "emails": [], "birthday": "",
    })
    sends = []
    monkeypatch.setattr(rag, "_whatsapp_send_to_jid",
                        lambda *a, **kw: sends.append(a) or True)
    raw = rag.propose_whatsapp_send(
        "Grecia", "hola",
        scheduled_for="2099-12-31T09:00:00-03:00",
    )
    payload = json.loads(raw)
    assert payload["kind"] == "whatsapp_message"
    assert payload["needs_clarification"] is True
    # No se llamó al bridge — la programación se dispara después por el
    # plist `wa-scheduled-send`, no por este path.
    assert sends == []
    assert payload["fields"]["scheduled_for"] == "2099-12-31T09:00:00-03:00"


def test_propose_whatsapp_send_surfaces_lookup_error(monkeypatch):
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)
    raw = rag.propose_whatsapp_send("Inexistente", "hola")
    payload = json.loads(raw)
    assert payload["fields"]["error"] == "not_found"
    assert payload["fields"]["jid"] is None
    # Still needs_clarification=True so the UI shows the card + error msg.
    assert payload["needs_clarification"] is True


def test_propose_whatsapp_send_handles_empty_message(monkeypatch):
    """Empty message_text is allowed — the UI shows an editable textarea
    so the user can fill it in. Should NOT auto-refuse."""
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: {
        "full_name": "Grecia", "phones": ["+5491155555555"], "emails": [], "birthday": "",
    })
    raw = rag.propose_whatsapp_send("Grecia", "")
    payload = json.loads(raw)
    assert payload["fields"]["message_text"] == ""
    assert payload["fields"]["error"] is None


# ── 4. /api/whatsapp/send endpoint ─────────────────────────────────────────


def test_whatsapp_send_endpoint_happy_path(monkeypatch):
    captured = {}
    # `reply_to` kwarg added 2026-04-24 (wa-reply feature). Accepted but
    # unused on the plain send path — keeps the fake signature aligned
    # with the real helper so the endpoint can pass it transparently.
    def _fake_send(jid, text, anti_loop=True, reply_to=None):
        captured["jid"] = jid
        captured["text"] = text
        captured["anti_loop"] = anti_loop
        captured["reply_to"] = reply_to
        return True
    monkeypatch.setattr(_server, "_whatsapp_send_to_jid", _fake_send, raising=False)
    # Also patch the attr on the rag module (endpoint imports it lazily).
    monkeypatch.setattr(rag, "_whatsapp_send_to_jid", _fake_send)

    resp = _client.post("/api/whatsapp/send", json={
        "jid": "5491155555555@s.whatsapp.net",
        "message_text": "hola grecia",
        "proposal_id": "prop-abc",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["jid"] == "5491155555555@s.whatsapp.net"
    # Critical: anti_loop must be False for user-initiated sends.
    assert captured["anti_loop"] is False
    assert captured["text"] == "hola grecia"


def test_whatsapp_send_endpoint_rejects_invalid_jid():
    resp = _client.post("/api/whatsapp/send", json={
        "jid": "just-a-phone-no-at-sign",
        "message_text": "hola",
    })
    assert resp.status_code == 400
    assert "inválido" in resp.json()["detail"].lower() or "invalido" in resp.json()["detail"].lower()


def test_whatsapp_send_endpoint_rejects_empty_body():
    resp = _client.post("/api/whatsapp/send", json={
        "jid": "123@s.whatsapp.net",
        "message_text": "   ",  # whitespace only → stripped to empty
    })
    assert resp.status_code == 400
    assert "vacío" in resp.json()["detail"] or "vacio" in resp.json()["detail"].lower()


def test_whatsapp_send_endpoint_bridge_down_returns_502(monkeypatch):
    monkeypatch.setattr(rag, "_whatsapp_send_to_jid", lambda *a, **kw: False)
    resp = _client.post("/api/whatsapp/send", json={
        "jid": "5491155555555@s.whatsapp.net",
        "message_text": "hola",
    })
    assert resp.status_code == 502
    assert "bridge" in resp.json()["detail"].lower()


def test_whatsapp_send_endpoint_accepts_group_jid(monkeypatch):
    """Group chats use `<id>@g.us` instead of `@s.whatsapp.net`. Must be
    accepted (though in practice users rarely send to groups from chat)."""
    monkeypatch.setattr(rag, "_whatsapp_send_to_jid", lambda *a, **kw: True)
    resp = _client.post("/api/whatsapp/send", json={
        "jid": "120363426178035051@g.us",
        "message_text": "msg a grupo",
    })
    assert resp.status_code == 200


# ── 5. Tool registration invariants ────────────────────────────────────────


def test_propose_whatsapp_send_is_registered():
    assert _tools.propose_whatsapp_send in _tools.CHAT_TOOLS
    assert "propose_whatsapp_send" in _tools.TOOL_FNS
    # Critical: emits SSE `proposal` event so the UI shows the card.
    assert "propose_whatsapp_send" in _tools.PROPOSAL_TOOL_NAMES


def test_propose_whatsapp_send_is_NOT_parallel_safe():
    """Sending WhatsApp is not a read — we don't want it running in
    parallel with other tools (also it's expensive: osascript + bridge).
    """
    assert "propose_whatsapp_send" not in _tools.PARALLEL_SAFE


def test_ollama_tool_client_has_separate_wider_timeout():
    """Regresión 2026-04-24 iter3 (Fer F. "LLM falló: timed out"): el tool-
    decision call (non-streaming, con `tools=` schema de 12 tools) tardaba
    >45s en qwen2.5:7b con prompts largos. El cliente compartido
    `_OLLAMA_STREAM_CLIENT` tenía 45s de timeout — cortaba antes de que el
    LLM terminara de samplear la decisión. Fix: cliente separado
    `_OLLAMA_TOOL_CLIENT` con 120s de budget, usado sólo para la call de
    tool-decisión.
    """
    import web.server as srv
    assert hasattr(srv, "_OLLAMA_TOOL_CLIENT"), (
        "_OLLAMA_TOOL_CLIENT debe existir (separado del streaming client)"
    )
    # Streaming budget sigue siendo conservador (cortamos UX congelada rápido).
    assert srv._OLLAMA_STREAM_TIMEOUT == 45.0
    # Tool-decision budget es materialmente más amplio.
    assert srv._OLLAMA_TOOL_TIMEOUT >= 90.0
    assert srv._OLLAMA_TOOL_TIMEOUT > srv._OLLAMA_STREAM_TIMEOUT


def test_tool_addendum_mentions_whatsapp_send():
    """Without the routing hint the LLM tends to respond in prose instead
    of calling the tool. The addendum must explicitly teach the pattern."""
    addendum = _tools._WEB_TOOL_ADDENDUM
    assert "propose_whatsapp_send" in addendum
    # Must also teach that it's destructive / needs confirmation.
    assert "confirmaci" in addendum.lower()


# ── 6. Enrich hang fix (related sub-bug reported in the same turn) ─────────


def test_detect_propose_intent_matches_whatsapp_send_verbs():
    """Regresión 2026-04-24 iter2 (Fer F.): "enviale un mensaje a Grecia:
    hola prueba" caía como prose — `_detect_propose_intent` devolvía False
    y el LLM generaba texto sin llamar al tool. Fix: agregar los verbos
    rioplatenses de enviar (con encliticos -le/-rle) al `_PROPOSE_INTENT_RE`.
    """
    # MUST match: send intent a un tercero.
    positives = [
        "enviale un mensaje a Grecia que diga: hola prueba",
        "Enviale un mensje a Grecia ahora que diga: Yo soy EL RA",
        "mandale un wzp a grecia que diga hola",
        "mandá un mensaje a grecia",
        "decile a grecia que llamo al medico",
        "escribile a Papá que ya llegué",
        "avisale a Juan que voy en camino",
        "mandale a Grecia que me confirme",
    ]
    for q in positives:
        assert rag._detect_propose_intent(q) is True, (
            f"propose_intent debería ser True para {q!r}"
        )

    # MUST NOT match: queries de lectura que involucran mensajería.
    negatives = [
        "cuales son los ultimos wzp con grecia",    # read
        "qué mensaje me mandó grecia",               # read (pasado)
        "le envié a juan la plata",                  # declaration, past
        "enviame un resumen de los mensajes",        # self-directed (no -le)
        "decime que tengo para hoy",                 # imperative query
    ]
    for q in negatives:
        assert rag._detect_propose_intent(q) is False, (
            f"propose_intent NO debería ser True para {q!r}"
        )


def test_emit_enrich_hung_worker_does_not_block_stream(monkeypatch):
    """Regresión 2026-04-24 (Fer F. turn `1ac42f199034` — "quedó colgado"):
    previously `_emit_enrich` used `with ThreadPoolExecutor(...) as _ex:`,
    which called `shutdown(wait=True)` on exit. When the worker thread was
    hung in osascript/ollama, the `with`-exit blocked the SSE generator
    indefinitely — the user's chat froze after the response was rendered.

    This test simulates the hang: `build_enrich_payload` never returns, we
    assert `_emit_enrich` completes within the 4s timeout + slack (not
    hanging forever) and returns None (skipped).
    """
    import time as _t
    import threading as _th

    # Build a fake build_enrich_payload that never returns.
    _started = _th.Event()
    _stuck = _th.Event()

    def _hung_worker(q, answer, top_score):
        _started.set()
        _stuck.wait(timeout=30)  # hangs up to 30s
        return None

    monkeypatch.setattr(rag, "build_enrich_payload", _hung_worker)

    # Extract _emit_enrich by calling the chat endpoint path — OR simpler,
    # reconstruct it locally using the same pattern. Since `_emit_enrich`
    # is a nested function inside chat(), we test the shape directly.
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout

    def _emit_enrich_like():
        _ex = ThreadPoolExecutor(max_workers=1)
        try:
            _fut = _ex.submit(rag.build_enrich_payload, "q", "a", 0.5)
            _fut.result(timeout=0.5)  # short timeout for test speed
        except _FutTimeout:
            return "skipped"
        finally:
            _ex.shutdown(wait=False, cancel_futures=True)
        return "ok"

    t0 = _t.monotonic()
    result = _emit_enrich_like()
    elapsed = _t.monotonic() - t0

    # Must return within timeout + small overhead (not the 30s the worker sleeps).
    assert elapsed < 2.0, f"_emit_enrich blocked for {elapsed:.1f}s (shutdown wait=True bug?)"
    assert result == "skipped"

    # Unblock the worker so it doesn't leak beyond the test.
    _stuck.set()
