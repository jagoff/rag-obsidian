"""Tests for the `whatsapp_thread` chat tool.

Origin: el sprint de WhatsApp agregó `whatsapp_search` (búsqueda por
contenido sobre el corpus indexado) y `whatsapp_pending` (chats sin
respuesta), pero faltaba la tool para "leeme el chat con X" / "qué
quedamos con Juan ayer". El uso killer es meeting-detection sin
detector custom: el user pide "fijate qué quedamos con Juan y
agendalo", el LLM llama `whatsapp_thread("Juan")`, lee el último
mensaje ("el jueves 4pm dale"), y encadena `propose_calendar_event`
en la misma ronda. Sin regex, sin parser custom — el LLM hace todo.

Este archivo cubre en 2 capas:

1. `_agent_tool_whatsapp_thread` — helper de rag con
   `_whatsapp_jid_from_contact` y el SQLite del bridge mockeados.
2. `whatsapp_thread` en `web.tools` — wrapper + registro en
   CHAT_TOOLS / PARALLEL_SAFE + NOT en PROPOSAL_TOOL_NAMES.

Todos los tests son puros — no tocan el bridge real ni Apple
Contacts. SQLite se mockea creando una DB temporal con el schema
real del bridge + rows canned.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag
from web import tools as tools_mod


# ── Helpers: build a fake bridge DB with the real schema ───────────────────


BRIDGE_SCHEMA = """
CREATE TABLE messages (
    id TEXT,
    chat_jid TEXT,
    sender TEXT,
    content TEXT,
    timestamp TIMESTAMP,
    is_from_me BOOLEAN,
    media_type TEXT,
    filename TEXT,
    url TEXT,
    media_key BLOB,
    file_sha256 BLOB,
    file_enc_sha256 BLOB,
    file_length INTEGER,
    PRIMARY KEY (id, chat_jid)
);
CREATE INDEX idx_messages_chat_ts ON messages(chat_jid, timestamp DESC);
"""


def _make_bridge_db(tmp_path: Path, rows: list[tuple]) -> Path:
    """Create a tmp bridge DB with the real schema + given rows.

    `rows` elements: (id, chat_jid, sender, content, timestamp,
    is_from_me). Timestamps can be strings (Go RFC3339-ish) or epoch
    floats — both supported by `_parse_bridge_timestamp`.
    """
    db_path = tmp_path / "messages.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(BRIDGE_SCHEMA)
        conn.executemany(
            "INSERT INTO messages "
            "(id, chat_jid, sender, content, timestamp, is_from_me) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _ts_iso(offset_hours: float = 0.0) -> str:
    """Return a bridge-style timestamp string offset from now."""
    t = datetime.now() + timedelta(hours=offset_hours)
    return t.strftime("%Y-%m-%d %H:%M:%S-03:00")


JUAN_JID = "5491134567890@s.whatsapp.net"
JUAN_LOOKUP = {
    "jid": JUAN_JID,
    "full_name": "Juan Pérez",
    "phones": ["+54 9 11 3456-7890"],
    "error": None,
}


# ── 1. `_agent_tool_whatsapp_thread` helper ────────────────────────────────


def test_thread_happy_path_returns_messages_in_chrono_order(
    tmp_path, monkeypatch,
):
    """Rows DESC en la DB → output ASC (más viejo primero) para que
    el LLM lea el thread top-to-bottom como una conversación normal."""
    rows = [
        # Newest first in the DB (DESC by ts).
        ("m3", JUAN_JID, "Juan Pérez", "el jueves 4pm dale",
         _ts_iso(-1), 0),
        ("m2", JUAN_JID, "yo", "cuándo podrías juntarte?",
         _ts_iso(-2), 1),
        ("m1", JUAN_JID, "Juan Pérez", "hola, tengo que verte",
         _ts_iso(-3), 0),
    ]
    db = _make_bridge_db(tmp_path, rows)
    monkeypatch.setattr(rag, "WHATSAPP_BRIDGE_DB_PATH", db)
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact",
                        lambda name: dict(JUAN_LOOKUP))

    out = rag._agent_tool_whatsapp_thread("Juan")
    parsed = json.loads(out)

    assert parsed["contact_name"] == "Juan Pérez"
    assert parsed["jid"] == JUAN_JID
    assert parsed["count"] == 3
    assert parsed["days_back"] == 7
    assert [m["text"] for m in parsed["messages"]] == [
        "hola, tengo que verte",
        "cuándo podrías juntarte?",
        "el jueves 4pm dale",
    ]
    # Who classification: is_from_me=0 → inbound, 1 → outbound.
    assert [m["who"] for m in parsed["messages"]] == [
        "inbound", "outbound", "inbound"
    ]
    # ts is ISO (from `datetime.fromtimestamp(...).isoformat(...)`).
    for m in parsed["messages"]:
        assert "T" in m["ts"]


def test_thread_empty_contact_returns_error(monkeypatch):
    """Empty/whitespace contact short-circuits with a clear error."""
    def _boom(*_a, **_kw):
        pytest.fail("no debería llegar al lookup con contacto vacío")
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact", _boom)

    out = rag._agent_tool_whatsapp_thread("")
    parsed = json.loads(out)
    assert parsed["messages"] == []
    assert parsed["error"] == "empty_contact"

    out2 = rag._agent_tool_whatsapp_thread("   \t ")
    assert json.loads(out2)["error"] == "empty_contact"


def test_thread_contact_not_found_returns_error(monkeypatch):
    """Contacto que no existe en Apple Contacts → error explícito,
    no crash. La UI/LLM puede avisarle al user."""
    monkeypatch.setattr(
        rag, "_whatsapp_jid_from_contact",
        lambda name: {
            "jid": None, "full_name": None, "phones": [],
            "error": "not_found",
        },
    )

    out = rag._agent_tool_whatsapp_thread("NombreQueNoExiste")
    parsed = json.loads(out)
    assert parsed["messages"] == []
    assert parsed["error"] == "contact_not_found"
    assert parsed["contact_name"] == "NombreQueNoExiste"
    assert parsed["lookup_error"] == "not_found"


def test_thread_contact_lookup_raises_is_silent_fail(monkeypatch):
    """Si `_whatsapp_jid_from_contact` tira excepción (osascript
    timeout, etc), atrapamos y devolvemos error JSON. NO propagamos."""
    def _boom(*_a, **_kw):
        raise RuntimeError("osascript timeout")
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact", _boom)

    out = rag._agent_tool_whatsapp_thread("Juan")
    parsed = json.loads(out)
    assert parsed["messages"] == []
    assert parsed["error"].startswith("contact_lookup_failed")
    assert "osascript timeout" in parsed["error"]


def test_thread_bridge_db_missing_returns_error(tmp_path, monkeypatch):
    """Si `messages.db` no existe → error `bridge_db_missing`, no
    crash. Pasa cuando el user nunca prendió el bridge o lo borró."""
    ghost = tmp_path / "no-such-dir" / "messages.db"
    monkeypatch.setattr(rag, "WHATSAPP_BRIDGE_DB_PATH", ghost)
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact",
                        lambda name: dict(JUAN_LOOKUP))

    out = rag._agent_tool_whatsapp_thread("Juan")
    parsed = json.loads(out)
    assert parsed["messages"] == []
    assert parsed["error"] == "bridge_db_missing"
    assert parsed["contact_name"] == "Juan Pérez"


def test_thread_max_messages_hard_cap_at_30(tmp_path, monkeypatch):
    """`max_messages=999` se capa a 30 — el LLM no puede inflar el
    CONTEXTO con 200 mensajes aunque pida."""
    # Seed 50 fresh inbound messages.
    rows = [
        (f"m{i}", JUAN_JID, "Juan Pérez", f"msg {i}",
         _ts_iso(-0.01 * (i + 1)), 0)
        for i in range(50)
    ]
    db = _make_bridge_db(tmp_path, rows)
    monkeypatch.setattr(rag, "WHATSAPP_BRIDGE_DB_PATH", db)
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact",
                        lambda name: dict(JUAN_LOOKUP))

    out = rag._agent_tool_whatsapp_thread("Juan", max_messages=999)
    parsed = json.loads(out)
    assert parsed["count"] == 30
    assert len(parsed["messages"]) == 30

    # max_messages=0 → piso 1.
    out2 = rag._agent_tool_whatsapp_thread("Juan", max_messages=0)
    assert json.loads(out2)["count"] == 1


def test_thread_days_filter_excludes_older_messages(
    tmp_path, monkeypatch,
):
    """Mensajes más viejos que `days` quedan afuera — ventana pulcra."""
    rows = [
        # Inside 7-day window (a few hours ago).
        ("new", JUAN_JID, "Juan Pérez", "reciente", _ts_iso(-2), 0),
        # 10 days ago — excluded from a days=7 window.
        ("old", JUAN_JID, "Juan Pérez", "viejísimo",
         _ts_iso(-10 * 24), 0),
    ]
    db = _make_bridge_db(tmp_path, rows)
    monkeypatch.setattr(rag, "WHATSAPP_BRIDGE_DB_PATH", db)
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact",
                        lambda name: dict(JUAN_LOOKUP))

    out = rag._agent_tool_whatsapp_thread("Juan", days=7)
    parsed = json.loads(out)
    assert parsed["count"] == 1
    assert parsed["messages"][0]["text"] == "reciente"

    # Con days=30 agarra los dos.
    out2 = rag._agent_tool_whatsapp_thread("Juan", days=30)
    parsed2 = json.loads(out2)
    assert parsed2["count"] == 2


def test_thread_days_hard_cap_at_30(tmp_path, monkeypatch):
    """`days=365` se capa a 30 — mismo motivo que max_messages."""
    rows = [
        # Freshly old row — 60 days ago, should fall outside 30-day cap.
        ("vintage", JUAN_JID, "Juan Pérez", "60 días atrás",
         _ts_iso(-60 * 24), 0),
        ("today", JUAN_JID, "Juan Pérez", "hoy", _ts_iso(-1), 0),
    ]
    db = _make_bridge_db(tmp_path, rows)
    monkeypatch.setattr(rag, "WHATSAPP_BRIDGE_DB_PATH", db)
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact",
                        lambda name: dict(JUAN_LOOKUP))

    out = rag._agent_tool_whatsapp_thread("Juan", days=365)
    parsed = json.loads(out)
    # `days_back` se reporta capado.
    assert parsed["days_back"] == 30
    # Sólo el de hoy cae dentro de 30 días.
    assert parsed["count"] == 1
    assert parsed["messages"][0]["text"] == "hoy"


def test_thread_jid_suffix_match_last_10_digits(tmp_path, monkeypatch):
    """El chat_jid en la DB puede tener country-code distinto que
    Apple Contacts ("+5491134567890" vs "491134567890@...") — tienen
    que matchear por los últimos 10 dígitos igual que Sub#2 hace."""
    # DB jid usa un country prefix distinto (missing '5') pero mismos
    # 10 últimos dígitos (1134567890).
    variant_jid = "491134567890@s.whatsapp.net"
    other_jid = "5491199999999@s.whatsapp.net"  # distinto contacto
    rows = [
        ("j1", variant_jid, "Juan", "hola Juan desde variant",
         _ts_iso(-1), 0),
        ("o1", other_jid, "Otro", "ruido de otro chat",
         _ts_iso(-2), 0),
    ]
    db = _make_bridge_db(tmp_path, rows)
    monkeypatch.setattr(rag, "WHATSAPP_BRIDGE_DB_PATH", db)
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact",
                        lambda name: dict(JUAN_LOOKUP))

    out = rag._agent_tool_whatsapp_thread("Juan")
    parsed = json.loads(out)
    assert parsed["count"] == 1
    # Matched JID es el variante (no `primary_jid` del lookup).
    assert parsed["jid"] == variant_jid
    assert parsed["messages"][0]["text"] == "hola Juan desde variant"


def test_thread_ignores_group_chats(tmp_path, monkeypatch):
    """Los chats `@g.us` no deben aparecer en el output de
    `whatsapp_thread` — solamente 1:1. Grupos: la UX de "qué dijo X
    en el grupo Y" es ambigua y no la cubrimos desde este tool."""
    group_jid = "120363012345@g.us"
    rows = [
        ("g1", group_jid, "Juan Pérez", "mensaje en grupo",
         _ts_iso(-1), 0),
        ("p1", JUAN_JID, "Juan Pérez", "mensaje en 1:1",
         _ts_iso(-2), 0),
    ]
    db = _make_bridge_db(tmp_path, rows)
    monkeypatch.setattr(rag, "WHATSAPP_BRIDGE_DB_PATH", db)
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact",
                        lambda name: dict(JUAN_LOOKUP))

    out = rag._agent_tool_whatsapp_thread("Juan")
    parsed = json.loads(out)
    assert parsed["count"] == 1
    assert parsed["messages"][0]["text"] == "mensaje en 1:1"


def test_thread_text_capped_at_400_chars(tmp_path, monkeypatch):
    """Cada `messages[].text` se capa a 400 chars — igual que
    whatsapp_search. Evita inflar el CONTEXTO con mensajes
    descomunales (walls of text, pastes)."""
    long_text = "x" * 1000
    rows = [
        ("m1", JUAN_JID, "Juan Pérez", long_text, _ts_iso(-1), 0),
    ]
    db = _make_bridge_db(tmp_path, rows)
    monkeypatch.setattr(rag, "WHATSAPP_BRIDGE_DB_PATH", db)
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact",
                        lambda name: dict(JUAN_LOOKUP))

    out = rag._agent_tool_whatsapp_thread("Juan")
    parsed = json.loads(out)
    assert len(parsed["messages"][0]["text"]) == 400


def test_thread_no_messages_in_window_returns_empty_list(
    tmp_path, monkeypatch,
):
    """Contacto resuelto, DB presente, pero 0 mensajes en los últimos
    N días → `count: 0`, sin error."""
    rows = [
        # 60 días atrás, fuera de la ventana por defecto.
        ("old", JUAN_JID, "Juan Pérez", "viejo", _ts_iso(-60 * 24), 0),
    ]
    db = _make_bridge_db(tmp_path, rows)
    monkeypatch.setattr(rag, "WHATSAPP_BRIDGE_DB_PATH", db)
    monkeypatch.setattr(rag, "_whatsapp_jid_from_contact",
                        lambda name: dict(JUAN_LOOKUP))

    out = rag._agent_tool_whatsapp_thread("Juan", days=7)
    parsed = json.loads(out)
    assert parsed["count"] == 0
    assert parsed["messages"] == []
    assert "error" not in parsed


# ── 2. `web.tools` wrapper registration ────────────────────────────────────


def test_web_tools_exposes_whatsapp_thread():
    """Registrado en `CHAT_TOOLS`, `TOOL_FNS`, `PARALLEL_SAFE` (read-
    only) y NO en `PROPOSAL_TOOL_NAMES` (no es destructivo)."""
    assert "whatsapp_thread" in tools_mod.TOOL_FNS
    assert tools_mod.whatsapp_thread in tools_mod.CHAT_TOOLS
    assert "whatsapp_thread" in tools_mod.PARALLEL_SAFE
    assert "whatsapp_thread" not in tools_mod.PROPOSAL_TOOL_NAMES
    assert (tools_mod.whatsapp_thread.__doc__ and
            "WhatsApp" in tools_mod.whatsapp_thread.__doc__)


def test_web_tool_addendum_mentions_whatsapp_thread():
    """El addendum tiene una entrada explícita para `whatsapp_thread`
    y menciona la composición con `propose_calendar_event` para que
    el LLM encadene meeting-detection sin pedir confirmación."""
    addendum = tools_mod._WEB_TOOL_ADDENDUM
    assert "whatsapp_thread" in addendum
    # Triggers.
    lower = addendum.lower()
    assert "qué hablamos" in lower or "qué quedamos" in lower
    # Composition with propose_calendar_event.
    assert "propose_calendar_event" in addendum
