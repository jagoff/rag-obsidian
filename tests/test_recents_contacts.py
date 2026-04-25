"""Tests para `_recent_contact_keys` + integración con `_fuzzy_filter_contacts`
cuando q="". Recents leen del ambient log (tabla rag_ambient en
telemetry.db) y prioritizan los contactos a los que el user mandó WA /
mail más recientemente sobre el orden alfabético."""
from __future__ import annotations
import json
import sqlite3

import pytest

import rag


@pytest.fixture
def isolated_telemetry_db(monkeypatch, tmp_path):
    """Aisla el ambient log a una telemetry.db tmp para que los tests
    no contaminen el real ni dependan del estado del user."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    # Crear la tabla rag_ambient con el schema esperado
    db = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE rag_ambient (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            cmd TEXT,
            path TEXT,
            hash TEXT,
            payload_json TEXT
        )
    """)
    conn.commit()
    return conn


def _insert_ambient(conn, ts, cmd, payload):
    """Helper: insert a row into rag_ambient via raw sqlite (no rag wrappers)."""
    conn.execute(
        "INSERT INTO rag_ambient (ts, cmd, payload_json) VALUES (?, ?, ?)",
        (ts, cmd, json.dumps(payload)),
    )
    conn.commit()


# ── _recent_contact_keys ─────────────────────────────────────────────────


def test_recent_phone_keys_extracts_jids_in_recency_order(isolated_telemetry_db):
    conn = isolated_telemetry_db
    # 3 sends, oldest first
    _insert_ambient(conn, "2026-04-20T10:00:00", "whatsapp_user_send",
                    {"jid": "5491111111111@s.whatsapp.net", "len": 5})
    _insert_ambient(conn, "2026-04-21T10:00:00", "whatsapp_user_send",
                    {"jid": "5492222222222@s.whatsapp.net", "len": 5})
    _insert_ambient(conn, "2026-04-22T10:00:00", "whatsapp_user_send",
                    {"jid": "5493333333333@s.whatsapp.net", "len": 5})
    out = rag._recent_contact_keys("phone", limit=10)
    # Recency desc: most recent first.
    assert out == ["5493333333333", "5492222222222", "5491111111111"]


def test_recent_phone_keys_dedupes_across_repeated_sends(isolated_telemetry_db):
    """Si el user mandó 5 mensajes a Grecia y 1 a Mario, recents debe
    listar [Grecia, Mario] no [Grecia, Grecia, Grecia, Grecia, Grecia, Mario]."""
    conn = isolated_telemetry_db
    grecia = "5491155555555@s.whatsapp.net"
    mario = "5491177777777@s.whatsapp.net"
    for i in range(5):
        _insert_ambient(conn, f"2026-04-21T10:0{i}:00", "whatsapp_user_send",
                        {"jid": grecia, "len": 5})
    _insert_ambient(conn, "2026-04-21T11:00:00", "whatsapp_user_send",
                    {"jid": mario, "len": 5})
    out = rag._recent_contact_keys("phone", limit=10)
    assert out == ["5491177777777", "5491155555555"]


def test_recent_phone_keys_includes_replies_too(isolated_telemetry_db):
    conn = isolated_telemetry_db
    _insert_ambient(conn, "2026-04-21T10:00:00", "whatsapp_user_send",
                    {"jid": "5491111111111@s.whatsapp.net", "len": 5})
    _insert_ambient(conn, "2026-04-21T11:00:00", "whatsapp_user_reply",
                    {"jid": "5492222222222@s.whatsapp.net", "len": 5,
                     "reply_to_id": "X"})
    out = rag._recent_contact_keys("phone", limit=10)
    assert out == ["5492222222222", "5491111111111"]


def test_recent_phone_keys_skips_invalid_jids(isolated_telemetry_db):
    """JIDs malformados (sin dígitos suficientes) se descartan."""
    conn = isolated_telemetry_db
    _insert_ambient(conn, "2026-04-21T10:00:00", "whatsapp_user_send",
                    {"jid": "abc@s.whatsapp.net", "len": 5})  # no digits
    _insert_ambient(conn, "2026-04-21T11:00:00", "whatsapp_user_send",
                    {"jid": "123@g.us", "len": 5})  # too short
    _insert_ambient(conn, "2026-04-21T12:00:00", "whatsapp_user_send",
                    {"jid": "5491155555555@s.whatsapp.net", "len": 5})  # OK
    out = rag._recent_contact_keys("phone", limit=10)
    assert out == ["5491155555555"]


def test_recent_email_keys_extracts_to_field(isolated_telemetry_db):
    conn = isolated_telemetry_db
    _insert_ambient(conn, "2026-04-20T10:00:00", "mail_user_send",
                    {"to": "alice@example.com", "len": 5})
    _insert_ambient(conn, "2026-04-21T10:00:00", "mail_user_send",
                    {"to": "BOB@example.com", "len": 5})  # uppercase
    out = rag._recent_contact_keys("email", limit=10)
    # Lowercased + recency desc
    assert out == ["bob@example.com", "alice@example.com"]


def test_recent_email_keys_skips_invalid_emails(isolated_telemetry_db):
    conn = isolated_telemetry_db
    _insert_ambient(conn, "2026-04-21T10:00:00", "mail_user_send",
                    {"to": "no-at-sign", "len": 5})
    _insert_ambient(conn, "2026-04-21T11:00:00", "mail_user_send",
                    {"to": "ok@x.com", "len": 5})
    out = rag._recent_contact_keys("email", limit=10)
    assert out == ["ok@x.com"]


def test_recent_keys_invalid_channel_returns_empty(isolated_telemetry_db):
    out = rag._recent_contact_keys("potato", limit=10)
    assert out == []


def test_recent_keys_zero_limit_returns_empty(isolated_telemetry_db):
    out = rag._recent_contact_keys("phone", limit=0)
    assert out == []


def test_recent_keys_silent_fail_when_db_missing(monkeypatch, tmp_path):
    """Si el ambient table no existe, devolvemos [] en lugar de raisear."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    # No table created.
    out = rag._recent_contact_keys("phone", limit=10)
    assert out == []


# ── Integración con _fuzzy_filter_contacts (q="") ────────────────────────


def test_fuzzy_filter_empty_query_uses_recents_first(monkeypatch, isolated_telemetry_db):
    """Cuando q="" y hay recents en el ambient log, el recent contact aparece
    PRIMERO en la lista — no orden alfabético."""
    conn = isolated_telemetry_db
    # Recent send to "Mario Pérez" (phone digits 5491177777777).
    _insert_ambient(conn, "2026-04-22T10:00:00", "whatsapp_user_send",
                    {"jid": "5491177777777@s.whatsapp.net"})

    # Install fake contacts cache. Mario tiene phone que matchea.
    contacts = [
        {"name": "Ágatha García",  "phones": [],                "emails": ["agatha@y.com"]},
        {"name": "Grecia Ferrari", "phones": ["+5491155555555"], "emails": []},
        {"name": "Mario Pérez",    "phones": ["+5491177777777"], "emails": []},
    ]
    rag._contacts_phone_index = {
        "ts": 9999999999,
        "schema": rag._CONTACTS_CACHE_SCHEMA,
        "index": {},
        "contacts": contacts,
    }

    out = rag._fuzzy_filter_contacts("", kind="phone", limit=5)
    names = [c["name"] for c in out]
    # Mario primero (recent), Grecia segundo (alfabético, sin recent)
    assert names[0] == "Mario Pérez", f"expected Mario primero por recents, got {names}"
    assert "Grecia Ferrari" in names
    # Ágatha excluida (sin teléfono, kind=phone filter)
    assert "Ágatha García" not in names


def test_fuzzy_filter_recent_then_alphabetical_no_dupes(monkeypatch, isolated_telemetry_db):
    """Un recent NO debe aparecer dos veces (una en recents y otra
    en alfabético). Verifica el dedupe por `used_names`."""
    conn = isolated_telemetry_db
    _insert_ambient(conn, "2026-04-22T10:00:00", "whatsapp_user_send",
                    {"jid": "5491155555555@s.whatsapp.net"})

    contacts = [
        {"name": "Grecia Ferrari", "phones": ["+5491155555555"], "emails": []},
        {"name": "Mario Pérez",    "phones": ["+5491177777777"], "emails": []},
    ]
    rag._contacts_phone_index = {
        "ts": 9999999999,
        "schema": rag._CONTACTS_CACHE_SCHEMA,
        "index": {},
        "contacts": contacts,
    }

    out = rag._fuzzy_filter_contacts("", kind="phone", limit=5)
    names = [c["name"] for c in out]
    # Cada contacto exactamente UNA vez.
    assert names.count("Grecia Ferrari") == 1
    assert names.count("Mario Pérez") == 1


def test_fuzzy_filter_recent_with_query_ignores_recents(monkeypatch, isolated_telemetry_db):
    """Con query no vacía, los recents NO se aplican — ranking es por
    score (exact > prefix > substring) tal como sin la feature de recents."""
    conn = isolated_telemetry_db
    # Mario es el recent.
    _insert_ambient(conn, "2026-04-22T10:00:00", "whatsapp_user_send",
                    {"jid": "5491177777777@s.whatsapp.net"})

    contacts = [
        {"name": "Grecia Ferrari", "phones": ["+5491155555555"], "emails": []},
        {"name": "Mario Pérez",    "phones": ["+5491177777777"], "emails": []},
    ]
    rag._contacts_phone_index = {
        "ts": 9999999999,
        "schema": rag._CONTACTS_CACHE_SCHEMA,
        "index": {},
        "contacts": contacts,
    }

    out = rag._fuzzy_filter_contacts("Grec", kind="phone", limit=5)
    names = [c["name"] for c in out]
    # "Grec" matchea Grecia (no Mario). El recent (Mario) no se inyecta.
    assert names == ["Grecia Ferrari"]


def test_fuzzy_filter_recent_falls_back_to_alphabetical_when_no_recents(
        monkeypatch, isolated_telemetry_db):
    """Sin recents en el log, el comportamiento previo (alfabético) sigue."""
    # No insertamos nada en el ambient log.
    contacts = [
        {"name": "Mario Pérez",    "phones": ["+5491177777777"], "emails": []},
        {"name": "Ágatha García",  "phones": ["+1"],             "emails": []},
        {"name": "Grecia Ferrari", "phones": ["+5491155555555"], "emails": []},
    ]
    rag._contacts_phone_index = {
        "ts": 9999999999,
        "schema": rag._CONTACTS_CACHE_SCHEMA,
        "index": {},
        "contacts": contacts,
    }

    out = rag._fuzzy_filter_contacts("", kind="phone", limit=5)
    names = [c["name"] for c in out]
    # Folded alfabético: agatha < grecia < mario (todas con phone).
    assert names[0] == "Ágatha García"
    assert names == ["Ágatha García", "Grecia Ferrari", "Mario Pérez"]
