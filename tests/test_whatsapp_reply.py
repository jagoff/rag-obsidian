"""Tests for the `propose_whatsapp_reply` chat tool + reply support in
`/api/whatsapp/send` (2026-04-24 wa-reply feature).

Covers:

1. `_parse_when_hint` — open / latest / hh:mm / ayer / hoy / keyword
   bandas (almuerzo/cena/desayuno) / hace-N-horas / fallback-keyword.
2. `_whatsapp_resolve_reply_target` — happy hit, miss (no inbound in
   window), miss (contact not found), miss (db missing), keyword
   filtering, last-10-digit suffix matching.
3. `_whatsapp_send_to_jid(reply_to=...)` — bridge payload includes
   `reply_to.message_id` + original_text + sender_jid. Backward compat:
   without reply_to the body is unchanged (no `reply_to` key).
4. `propose_whatsapp_reply` — kind="whatsapp_reply", needs_clarification
   always True, reply_to populated on hit, warning surfaced on miss.
5. `/api/whatsapp/send` — happy path with reply_to → 200, audit log
   `cmd=whatsapp_user_reply`. Backward compat without reply_to → still
   works (existing send tests). Empty reply_to.message_id → 400.
6. Tool registration invariants — `propose_whatsapp_reply` in
   `CHAT_TOOLS`, in `PROPOSAL_TOOL_NAMES`, NOT in `PARALLEL_SAFE`.
7. Pre-router routing — `_detect_propose_intent` matches "respondele a
   X / contestale a X / responde el último de X" but NOT "qué me
   respondió X" (read intent).

Style mirrors `tests/test_whatsapp_send_draft.py`. The bridge today
does NOT support native quote (see `rag._whatsapp_send_to_jid`
docstring) so the field travels in the payload but is ignored upstream
— we test that the CLIENT does its part regardless.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import rag
from fastapi.testclient import TestClient
import web.server as _server
from web import tools as _tools


_client = TestClient(_server.app)


# ── Fixture: tiny in-memory bridge DB with a few inbound messages ──────────


@pytest.fixture
def fake_bridge_db(tmp_path: Path) -> Path:
    """Create a SQLite file matching the whatsapp-mcp `messages` schema
    with a handful of inbound + outbound rows, so resolve_reply_target
    can exercise filtering by chat_jid suffix + time window + content."""
    db = tmp_path / "messages.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE chats ("
        "  jid TEXT PRIMARY KEY, name TEXT, last_message_time TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE messages ("
        "  id TEXT, chat_jid TEXT, sender TEXT, content TEXT,"
        "  timestamp TIMESTAMP, is_from_me BOOLEAN,"
        "  media_type TEXT, filename TEXT, url TEXT,"
        "  media_key BLOB, file_sha256 BLOB, file_enc_sha256 BLOB,"
        "  file_length INTEGER,"
        "  PRIMARY KEY (id, chat_jid))"
    )
    # Note: the resolver uses last-10-digit suffix matching, so any phone
    # ending in 5555555555 will route to this chat.
    grecia_jid = "5491155555555@s.whatsapp.net"
    juan_jid = "5491166666666@s.whatsapp.net"
    conn.execute(
        "INSERT INTO chats VALUES (?, ?, ?)",
        (grecia_jid, "Grecia", "2026-04-24 18:00:00-03:00"),
    )
    conn.execute(
        "INSERT INTO chats VALUES (?, ?, ?)",
        (juan_jid, "Juan", "2026-04-24 14:00:00-03:00"),
    )
    # Most recent inbound from Grecia (latest)
    rows = [
        # Grecia inbound — latest, evening
        ("WA001", grecia_jid, "5491155555555", "dale, te llamo en 5",
         "2026-04-24 18:00:00-03:00", 0),
        # Grecia inbound — afternoon (post-lunch)
        ("WA002", grecia_jid, "5491155555555", "almorzamos hoy?",
         "2026-04-24 13:30:00-03:00", 0),
        # Grecia outbound (is_from_me=1) — must be ignored
        ("WA003", grecia_jid, "yo", "perfecto",
         "2026-04-24 13:35:00-03:00", 1),
        # Juan inbound — different contact (must be filtered out by JID suffix)
        ("WA004", juan_jid, "5491166666666", "hola",
         "2026-04-24 14:00:00-03:00", 0),
        # Grecia inbound — yesterday
        ("WA005", grecia_jid, "5491155555555", "buen día!",
         "2026-04-23 09:00:00-03:00", 0),
    ]
    conn.executemany(
        "INSERT INTO messages (id, chat_jid, sender, content, timestamp, "
        "is_from_me) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


def _grecia_lookup(*a, **kw):
    return {
        "full_name": "Grecia Ferrari",
        "phones": ["+54 9 11 5555-5555"],
        "emails": [],
        "birthday": "",
    }


# ── 1. _parse_when_hint ─────────────────────────────────────────────────────


def test_parse_when_hint_none_returns_latest():
    low, high, kind = rag._parse_when_hint(None)
    assert (low, high) == (None, None)
    assert kind == "latest"


def test_parse_when_hint_empty_returns_latest():
    low, high, kind = rag._parse_when_hint("")
    assert kind == "latest"
    low, high, kind = rag._parse_when_hint("   ")
    assert kind == "latest"


def test_parse_when_hint_el_ultimo():
    for h in ["el último", "el ultimo", "lo último", "el más reciente",
              "lo mas reciente"]:
        low, high, kind = rag._parse_when_hint(h)
        assert kind == "latest", f"hint {h!r}: expected 'latest', got {kind!r}"
        assert (low, high) == (None, None)


def test_parse_when_hint_hh_mm():
    low, high, kind = rag._parse_when_hint("del 14:30")
    assert kind == "14:30"
    assert low is not None and high is not None
    # Window is ±10 min ⇒ 1200s wide
    assert pytest.approx(high - low, abs=1) == 1200


def test_parse_when_hint_keywords_almuerzo():
    low, high, kind = rag._parse_when_hint("del almuerzo")
    assert kind == "keyword:almuerzo"
    # 12:00 .. 15:00 today ⇒ 3h window
    assert pytest.approx(high - low, abs=1) == 3 * 3600


def test_parse_when_hint_yesterday():
    low, high, kind = rag._parse_when_hint("ayer")
    assert kind == "yesterday"
    assert pytest.approx(high - low, abs=1) == 24 * 3600


def test_parse_when_hint_hace_n_horas():
    low, high, kind = rag._parse_when_hint("hace 2 horas")
    assert kind.startswith("hace_2")
    assert low is not None and high is not None
    assert high > low


def test_parse_when_hint_hoy():
    low, high, kind = rag._parse_when_hint("hoy")
    assert kind == "today"
    assert low is not None and high is not None


def test_parse_when_hint_unknown_falls_back_to_keyword():
    low, high, kind = rag._parse_when_hint("del cumple de astor")
    assert kind.startswith("keyword:")


# ── 2. _whatsapp_resolve_reply_target ──────────────────────────────────────


def test_resolve_reply_target_happy_latest(fake_bridge_db, monkeypatch):
    monkeypatch.setattr(rag, "_fetch_contact", _grecia_lookup)
    out = rag._whatsapp_resolve_reply_target(
        "Grecia", when_hint=None, db_path=fake_bridge_db,
    )
    assert out.get("error") is None, out
    assert out["message_id"] == "WA001"
    assert "te llamo en 5" in out["text"]
    assert out["chat_jid"] == "5491155555555@s.whatsapp.net"


def test_resolve_reply_target_keyword_almuerzo(fake_bridge_db, monkeypatch):
    """Keyword in when_hint maps to a time band — should pick the
    afternoon message, not the evening one (latest)."""
    monkeypatch.setattr(rag, "_fetch_contact", _grecia_lookup)
    # Note: el time-band del 'almuerzo' es 12:00..15:00 hoy, pero los
    # rows del fixture son 2026-04-24 — la ventana de "almuerzo" se
    # calcula desde "today" runtime. Para que el test sea reproducible,
    # forzamos `keyword=` (matching contenido) sin time band.
    out = rag._whatsapp_resolve_reply_target(
        "Grecia", when_hint=None, db_path=fake_bridge_db,
        keyword="almorzamos",
    )
    assert out.get("error") is None, out
    assert out["message_id"] == "WA002"
    assert "almorzamos" in out["text"].lower()


def test_resolve_reply_target_no_inbound_in_window(fake_bridge_db, monkeypatch):
    """Filter that matches no rows → no_match (NOT a hard error)."""
    monkeypatch.setattr(rag, "_fetch_contact", _grecia_lookup)
    out = rag._whatsapp_resolve_reply_target(
        "Grecia", when_hint=None, db_path=fake_bridge_db,
        keyword="esto no aparece en ningun mensaje xyz",
    )
    assert out.get("error") == "no_match"
    assert out.get("contact_full_name") == "Grecia Ferrari"


def test_resolve_reply_target_contact_not_found(fake_bridge_db, monkeypatch):
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)
    out = rag._whatsapp_resolve_reply_target(
        "Inexistente", when_hint=None, db_path=fake_bridge_db,
    )
    assert out["error"].startswith("contact_")


def test_resolve_reply_target_empty_contact():
    out = rag._whatsapp_resolve_reply_target("", when_hint=None)
    assert out["error"] == "empty_contact"


def test_resolve_reply_target_db_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "_fetch_contact", _grecia_lookup)
    missing = tmp_path / "does-not-exist.db"
    out = rag._whatsapp_resolve_reply_target(
        "Grecia", when_hint=None, db_path=missing,
    )
    assert out["error"].startswith("bridge_db_missing")


def test_resolve_reply_target_filters_outbound(fake_bridge_db, monkeypatch):
    """Outbound (is_from_me=1) messages must NEVER be returned — we're
    looking for messages the user wants to reply TO, not echos of his
    own sends. The fixture has WA003 = outbound 'perfecto' between two
    inbound rows; if the filter is wrong we'd see it as latest."""
    monkeypatch.setattr(rag, "_fetch_contact", _grecia_lookup)
    out = rag._whatsapp_resolve_reply_target(
        "Grecia", when_hint=None, db_path=fake_bridge_db,
    )
    assert out["message_id"] != "WA003"
    assert out.get("text") != "perfecto"


def test_resolve_reply_target_filters_other_contacts(fake_bridge_db, monkeypatch):
    """JID-suffix filter must reject Juan's chat when we asked for Grecia."""
    monkeypatch.setattr(rag, "_fetch_contact", _grecia_lookup)
    out = rag._whatsapp_resolve_reply_target(
        "Grecia", when_hint=None, db_path=fake_bridge_db,
    )
    # The filter MUST reject WA004 even though it's only a few minutes
    # older than WA001 — it's from Juan, not Grecia.
    assert out["chat_jid"] == "5491155555555@s.whatsapp.net"


# ── 3. _whatsapp_send_to_jid with reply_to ─────────────────────────────────


def test_send_to_jid_includes_reply_to_in_payload(monkeypatch):
    captured = {}
    class _FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(req, timeout=10):
        captured["data"] = req.data
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    ok = rag._whatsapp_send_to_jid(
        "5491155555555@s.whatsapp.net",
        "ahí te llamo",
        anti_loop=False,
        reply_to={
            "message_id": "WA001",
            "original_text": "dale, te llamo en 5",
            "sender_jid": "5491155555555@s.whatsapp.net",
        },
    )
    assert ok is True
    payload = json.loads(captured["data"].decode("utf-8"))
    assert payload["message"] == "ahí te llamo"
    # reply_to forwarded — bridge ignores today, future-compat tomorrow.
    assert payload["reply_to"]["message_id"] == "WA001"
    assert payload["reply_to"]["original_text"] == "dale, te llamo en 5"
    assert payload["reply_to"]["sender_jid"] == "5491155555555@s.whatsapp.net"


def test_send_to_jid_without_reply_to_omits_field(monkeypatch):
    """Backward compat: a normal send (no reply_to) must NOT include
    the field in the bridge body — keeps audit logs / bridge logs
    distinguishable."""
    captured = {}
    class _FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(req, timeout=10):
        captured["data"] = req.data
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    rag._whatsapp_send_to_jid(
        "5491155555555@s.whatsapp.net", "hola", anti_loop=False,
    )
    payload = json.loads(captured["data"].decode("utf-8"))
    assert "reply_to" not in payload


def test_send_to_jid_drops_reply_to_without_message_id(monkeypatch):
    """Caller passes a malformed reply_to (no message_id) → silently dropped
    so we don't ship garbage to the bridge."""
    captured = {}
    class _FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(req, timeout=10):
        captured["data"] = req.data
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    rag._whatsapp_send_to_jid(
        "5491155555555@s.whatsapp.net", "hola",
        anti_loop=False, reply_to={"original_text": "huh"},
    )
    payload = json.loads(captured["data"].decode("utf-8"))
    assert "reply_to" not in payload


# ── 4. propose_whatsapp_reply ──────────────────────────────────────────────


def test_propose_whatsapp_reply_happy_path(fake_bridge_db, monkeypatch):
    """Hit: contact resolves + message resolves → kind=whatsapp_reply,
    fields.reply_to populated, needs_clarification True."""
    monkeypatch.setattr(rag, "_fetch_contact", _grecia_lookup)
    monkeypatch.setattr(rag, "WHATSAPP_BRIDGE_DB_PATH", fake_bridge_db)
    raw = rag.propose_whatsapp_reply("Grecia", "ahí te llamo")
    payload = json.loads(raw)
    assert payload["kind"] == "whatsapp_reply"
    assert payload["needs_clarification"] is True
    assert payload["proposal_id"].startswith("prop-")
    f = payload["fields"]
    assert f["contact_name"] == "Grecia"
    assert f["message_text"] == "ahí te llamo"
    assert f["jid"] == "5491155555555@s.whatsapp.net"
    assert f["error"] is None
    assert f["reply_to_warning"] is None
    rt = f["reply_to"]
    assert rt is not None
    assert rt["message_id"] == "WA001"
    assert "te llamo en 5" in rt["original_text"]


def test_propose_whatsapp_reply_contact_not_found(monkeypatch):
    """Contact lookup miss → fields.error set + reply_to None. UI
    disables [Enviar]."""
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)
    raw = rag.propose_whatsapp_reply("Unicornio", "hola")
    payload = json.loads(raw)
    assert payload["kind"] == "whatsapp_reply"
    assert payload["needs_clarification"] is True
    assert payload["fields"]["error"] == "not_found"
    assert payload["fields"]["jid"] is None
    assert payload["fields"]["reply_to"] is None


def test_propose_whatsapp_reply_no_target_message(tmp_path, monkeypatch):
    """Contact resolves but no matching inbound (db missing) → reply_to
    None + warning surfaced. UI shows the warning + offers to send
    without quote anyway."""
    monkeypatch.setattr(rag, "_fetch_contact", _grecia_lookup)
    monkeypatch.setattr(
        rag, "WHATSAPP_BRIDGE_DB_PATH", tmp_path / "missing.db",
    )
    raw = rag.propose_whatsapp_reply("Grecia", "dale", when_hint="el almuerzo")
    payload = json.loads(raw)
    assert payload["fields"]["error"] is None  # contact OK
    assert payload["fields"]["reply_to"] is None
    assert payload["fields"]["reply_to_warning"]
    assert "podés mandar" in payload["fields"]["reply_to_warning"].lower() or \
           "podes mandar" in payload["fields"]["reply_to_warning"].lower()
    assert payload["fields"]["reply_to_hint"] == "el almuerzo"


def test_propose_whatsapp_reply_when_hint_propagated(fake_bridge_db, monkeypatch):
    monkeypatch.setattr(rag, "_fetch_contact", _grecia_lookup)
    monkeypatch.setattr(rag, "WHATSAPP_BRIDGE_DB_PATH", fake_bridge_db)
    raw = rag.propose_whatsapp_reply("Grecia", "ok", when_hint="el último")
    payload = json.loads(raw)
    assert payload["fields"]["reply_to_hint"] == "el último"


# ── 5. /api/whatsapp/send endpoint with reply_to ──────────────────────────


def test_endpoint_happy_path_with_reply_to(monkeypatch):
    captured = {}
    def _fake_send(jid, text, anti_loop=True, reply_to=None):
        captured["jid"] = jid
        captured["text"] = text
        captured["anti_loop"] = anti_loop
        captured["reply_to"] = reply_to
        return True

    monkeypatch.setattr(_server, "_whatsapp_send_to_jid", _fake_send, raising=False)
    monkeypatch.setattr(rag, "_whatsapp_send_to_jid", _fake_send)

    resp = _client.post("/api/whatsapp/send", json={
        "jid": "5491155555555@s.whatsapp.net",
        "message_text": "ahí te llamo",
        "proposal_id": "prop-xyz",
        "reply_to": {
            "message_id": "WA001",
            "original_text": "dale, te llamo en 5",
            "sender_jid": "5491155555555@s.whatsapp.net",
        },
    })
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert body["ok"] is True
    assert captured["anti_loop"] is False
    assert captured["reply_to"]["message_id"] == "WA001"
    assert captured["reply_to"]["original_text"] == "dale, te llamo en 5"


def test_endpoint_backward_compat_without_reply_to(monkeypatch):
    """Existing send flow (no reply_to) must keep working unchanged.
    `propose_whatsapp_send` still emits whatsapp_message and POSTs without
    the field — assert reply_to=None reaches the helper."""
    captured = {}
    def _fake_send(jid, text, anti_loop=True, reply_to=None):
        captured["reply_to"] = reply_to
        return True

    monkeypatch.setattr(rag, "_whatsapp_send_to_jid", _fake_send)
    resp = _client.post("/api/whatsapp/send", json={
        "jid": "5491155555555@s.whatsapp.net",
        "message_text": "hola",
    })
    assert resp.status_code == 200
    assert captured["reply_to"] is None


def test_endpoint_rejects_empty_reply_to_message_id(monkeypatch):
    monkeypatch.setattr(rag, "_whatsapp_send_to_jid",
                        lambda *a, **kw: True)
    resp = _client.post("/api/whatsapp/send", json={
        "jid": "5491155555555@s.whatsapp.net",
        "message_text": "ahí",
        "reply_to": {"message_id": "   ", "original_text": "x"},
    })
    assert resp.status_code == 400
    assert "reply_to" in resp.json()["detail"].lower()


def test_endpoint_bridge_down_with_reply_to_returns_502(monkeypatch):
    monkeypatch.setattr(rag, "_whatsapp_send_to_jid",
                        lambda *a, **kw: False)
    resp = _client.post("/api/whatsapp/send", json={
        "jid": "5491155555555@s.whatsapp.net",
        "message_text": "ok",
        "reply_to": {"message_id": "WA001", "original_text": "x"},
    })
    assert resp.status_code == 502


def test_endpoint_audit_log_distinguishes_reply(monkeypatch):
    """When reply_to is present, the audit event must use
    `cmd=whatsapp_user_reply` so logs / dashboards can split the two
    flows. Plain sends keep `cmd=whatsapp_user_send`."""
    monkeypatch.setattr(rag, "_whatsapp_send_to_jid",
                        lambda *a, **kw: True)
    captured = {}
    def _fake_log(event):
        captured.setdefault("events", []).append(event)
    monkeypatch.setattr(rag, "_ambient_log_event", _fake_log)

    # Plain send
    _client.post("/api/whatsapp/send", json={
        "jid": "5491155555555@s.whatsapp.net",
        "message_text": "hola",
    })
    # Reply
    _client.post("/api/whatsapp/send", json={
        "jid": "5491155555555@s.whatsapp.net",
        "message_text": "ok",
        "reply_to": {"message_id": "WA001"},
    })
    cmds = [e["cmd"] for e in captured.get("events", [])]
    assert "whatsapp_user_send" in cmds
    assert "whatsapp_user_reply" in cmds
    reply_event = next(
        e for e in captured["events"] if e["cmd"] == "whatsapp_user_reply"
    )
    assert reply_event["reply_to_id"] == "WA001"


# ── 6. Tool registration invariants ───────────────────────────────────────


def test_propose_whatsapp_reply_is_registered():
    assert _tools.propose_whatsapp_reply in _tools.CHAT_TOOLS
    assert "propose_whatsapp_reply" in _tools.TOOL_FNS
    assert "propose_whatsapp_reply" in _tools.PROPOSAL_TOOL_NAMES


def test_propose_whatsapp_reply_is_NOT_parallel_safe():
    """Like send: not a read, has osascript + bridge side-effects via
    _whatsapp_jid_from_contact + sqlite scan, must run isolated per turn."""
    assert "propose_whatsapp_reply" not in _tools.PARALLEL_SAFE


def test_tool_addendum_mentions_whatsapp_reply():
    addendum = _tools._WEB_TOOL_ADDENDUM
    assert "propose_whatsapp_reply" in addendum
    # Must teach the routing pattern (respondele / contestale).
    assert "respondele" in addendum.lower() or "contestale" in addendum.lower()


def test_tool_addendum_count_matches_actual_tool_list():
    """El número en el header del addendum (`Tenés N tools…`) debe
    matchear con el length de CHAT_TOOLS — sino el LLM lee un count
    desactualizado y a veces deja de llamar el último tool registrado.

    El número crece cuando se registran tools nuevas (15 con el
    propose_mail_send agregado en 2026-04-24). Este test es flexible
    al count exacto pero estricto en la consistencia entre el texto y
    la lista real."""
    addendum = _tools._WEB_TOOL_ADDENDUM
    actual = len(_tools.CHAT_TOOLS)
    assert f"{actual} tools" in addendum, (
        f"addendum dice un count distinto al actual ({actual} tools en CHAT_TOOLS)"
    )


# ── 7. Pre-router (`_detect_propose_intent`) routing ──────────────────────


def test_detect_propose_intent_matches_reply_verbs():
    positives = [
        "respondele a Grecia que voy en 5",
        "Respondele a Juan al mensaje del almuerzo: confirmo, voy yo",
        "contestale a Grecia el último: dale, ahí te llamo",
        "responde el mensaje de Pedro: ok",
        "responde el último de Mariana: gracias!",
        "replyle a Pedro: ok",
    ]
    for q in positives:
        assert rag._detect_propose_intent(q) is True, (
            f"propose_intent debería ser True para {q!r}"
        )


def test_detect_propose_intent_does_not_match_read_about_replies():
    """Queries de lectura / consulta sobre respuestas pasadas NO deben
    triggear el reply-create flow.

    Nota: las negativas son sólo las que dependen del PATTERN propio del
    reply. Combos como "le respondí a X esta mañana" caen positivos por
    el branch independiente `_has_explicit_time` (declaration+time);
    eso es comportamiento pre-existente del propose-intent y NO un bug
    introducido por la feature de reply — el chat de propose flow ya
    pide confirmación antes de cualquier acción, así que un falso
    positivo acá no manda nada al exterior."""
    negatives = [
        # Sin temporal anchor + question word → claramente read.
        "qué me respondió Juan",
        "qué contestó Pedro",
        "decime quien respondió primero",
        # Imperative query con palabra interrogativa.
        "decime qué me respondió Grecia",
    ]
    for q in negatives:
        assert rag._detect_propose_intent(q) is False, (
            f"propose_intent NO debería ser True para {q!r}"
        )
