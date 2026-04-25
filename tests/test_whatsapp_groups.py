"""Tests del resolver de grupos WhatsApp (`@g.us`) — fix del audit
2026-04-25 (TOP 1). Antes el `_whatsapp_jid_from_contact` solo
soportaba 1:1 vía Apple Contacts; ahora cae a fuzzy match en la
tabla `chats` del bridge cuando Contacts no encuentra match.

Mockeamos:
- ``rag._fetch_contact`` (Apple Contacts) → simula miss/hit a voluntad.
- ``rag.WHATSAPP_DB_PATH`` → SQLite tmpfile con schema del bridge.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import rag
from rag.integrations.whatsapp import (
    _whatsapp_jid_from_contact,
    _whatsapp_resolve_group_jid,
)


_BRIDGE_SCHEMA = """
CREATE TABLE chats (
    jid TEXT PRIMARY KEY,
    name TEXT,
    last_message_time TIMESTAMP
);
CREATE INDEX idx_chats_last_msg_time ON chats(last_message_time DESC);
"""


@pytest.fixture
def fake_bridge_db(tmp_path, monkeypatch):
    """SQLite tmpfile con schema del bridge + grupos de fixture."""
    db_path = tmp_path / "fake_messages.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_BRIDGE_SCHEMA)
        rows = [
            # Grupos `@g.us`
            ("120363001@g.us", "Random", "2026-04-25 18:00:00-03:00"),
            ("120363002@g.us", "RagNet", "2026-04-25 17:00:00-03:00"),
            ("120363003@g.us", "Cloud Services", "2026-04-25 16:00:00-03:00"),
            ("120363004@g.us", "PublicCloudInfrastructure", "2026-04-25 15:00:00-03:00"),
            ("120363005@g.us", "Familia", "2026-04-25 14:00:00-03:00"),
            ("120363006@g.us", "Avature Cloud Services", "2026-04-25 13:00:00-03:00"),
            # 1:1 chats — el resolver de grupo los IGNORA (filtro @g.us)
            ("5491100@s.whatsapp.net", "Cloud", "2026-04-25 12:00:00-03:00"),
            ("5491101@s.whatsapp.net", "Random", "2026-04-25 11:00:00-03:00"),
            # Status broadcast — debe ser ignorado por el filter
            ("status@broadcast", None, "2026-04-25 10:00:00-03:00"),
        ]
        conn.executemany(
            "INSERT INTO chats (jid, name, last_message_time) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(rag, "WHATSAPP_DB_PATH", Path(str(db_path)))
    return db_path


# ── _whatsapp_resolve_group_jid (helper directo) ─────────────────────────


def test_resolve_group_jid_exact_match(fake_bridge_db):
    """Match exacto case-insensitive gana siempre."""
    r = _whatsapp_resolve_group_jid("Random")
    assert r["jid"] == "120363001@g.us"
    assert r["full_name"] == "Random"
    assert r["is_group"] is True
    assert r["phones"] == []
    assert r["error"] is None


def test_resolve_group_jid_case_insensitive(fake_bridge_db):
    """`random` matchea `Random` (lowercase comparison)."""
    r = _whatsapp_resolve_group_jid("random")
    assert r["jid"] == "120363001@g.us"
    r = _whatsapp_resolve_group_jid("RANDOM")
    assert r["jid"] == "120363001@g.us"


def test_resolve_group_jid_substring_unique(fake_bridge_db):
    """Substring que matchea exactamente 1 grupo → resuelve."""
    r = _whatsapp_resolve_group_jid("RagN")
    assert r["jid"] == "120363002@g.us"
    assert r["full_name"] == "RagNet"


def test_resolve_group_jid_substring_ambiguous_returns_candidates(fake_bridge_db):
    """`cloud` matchea 3 grupos → ambiguous con candidates ordenados
    por last_message_time DESC (más reciente arriba)."""
    r = _whatsapp_resolve_group_jid("cloud")
    assert r["jid"] is None
    assert r["error"] == "ambiguous"
    cands = r.get("candidates") or []
    assert len(cands) == 3
    # Ordenado por last_message_time DESC: Cloud Services > Public > Avature
    assert cands[0]["name"] == "Cloud Services"
    assert cands[1]["name"] == "PublicCloudInfrastructure"
    assert cands[2]["name"] == "Avature Cloud Services"


def test_resolve_group_jid_not_found(fake_bridge_db):
    """Query que no matchea ningún grupo → not_found."""
    r = _whatsapp_resolve_group_jid("NoExisteEsteGrupo123")
    assert r["jid"] is None
    assert r["error"] == "not_found"


def test_resolve_group_jid_empty_query(fake_bridge_db):
    """Empty query → empty_query (no scan a la DB)."""
    r = _whatsapp_resolve_group_jid("")
    assert r["error"] == "empty_query"
    r = _whatsapp_resolve_group_jid("   ")
    assert r["error"] == "empty_query"


def test_resolve_group_jid_ignores_1to1_chats(fake_bridge_db):
    """El resolver SOLO mira `@g.us`. `Cloud` matchea exact a un 1:1
    pero también substring a 3 grupos — solo cuenta los grupos."""
    r = _whatsapp_resolve_group_jid("Cloud")
    # Match exacto en grupo "Cloud Services" NO porque "Cloud" != "Cloud Services".
    # Pero substring matchea 3 grupos → ambiguous.
    assert r["error"] == "ambiguous"
    cands = r.get("candidates") or []
    # Ninguno de los candidates es 1:1 (filtro @g.us garantiza esto)
    for c in cands:
        assert c["jid"].endswith("@g.us")


def test_resolve_group_jid_ignores_status_broadcast(fake_bridge_db):
    """`status@broadcast` no debe aparecer aunque el filtro `@g.us`
    podría confundirse — verificar que el WHERE lo excluye."""
    # status@broadcast no termina en @g.us, así que el filtro de jid
    # naturalmente lo descarta. Verificación defensiva: no aparece.
    r = _whatsapp_resolve_group_jid("Random")
    assert r["jid"] != "status@broadcast"


def test_resolve_group_jid_no_bridge_db(monkeypatch, tmp_path):
    """Sin bridge DB existente → bridge_db_unavailable."""
    monkeypatch.setattr(rag, "WHATSAPP_DB_PATH", tmp_path / "nonexistent.db")
    r = _whatsapp_resolve_group_jid("Random")
    assert r["error"] == "bridge_db_unavailable"


# ── _whatsapp_jid_from_contact con fallback a grupos ────────────────────


def test_contact_lookup_finds_apple_contact_first(fake_bridge_db, monkeypatch):
    """Si Apple Contacts encuentra el contacto, NO cae a grupos
    aunque exista un grupo con el mismo nombre. El user normalmente
    quiere 1:1 cuando no especifica `grupo X`."""
    monkeypatch.setattr(
        rag, "_fetch_contact",
        lambda name, email=None, canonical=None: {
            "full_name": "Random Person",
            "phones": ["+5491155556666"],
        },
    )
    r = _whatsapp_jid_from_contact("Random")
    assert r["jid"] == "5491155556666@s.whatsapp.net"
    assert r["is_group"] is False
    assert r["full_name"] == "Random Person"


def test_contact_lookup_falls_back_to_group_when_contact_miss(fake_bridge_db, monkeypatch):
    """Si Apple Contacts NO encuentra → cae a grupos. Ese es el caso
    típico cuando el user nombra un grupo sin el prefijo `grupo X`."""
    monkeypatch.setattr(
        rag, "_fetch_contact",
        lambda name, email=None, canonical=None: None,
    )
    r = _whatsapp_jid_from_contact("RagNet")
    assert r["jid"] == "120363002@g.us"
    assert r["is_group"] is True
    assert r["full_name"] == "RagNet"


def test_contact_lookup_force_group_with_prefix(fake_bridge_db, monkeypatch):
    """Prefix `grupo X` saltea Contacts directamente y va al group
    resolver. Útil cuando hay un contacto Y un grupo con el mismo
    nombre y el user quiere específicamente el grupo."""
    contact_calls = []

    def _fetch(name, email=None, canonical=None):
        contact_calls.append(name)
        return {"full_name": "Random Person", "phones": ["+5491100000000"]}

    monkeypatch.setattr(rag, "_fetch_contact", _fetch)
    r = _whatsapp_jid_from_contact("grupo Random")
    assert r["jid"] == "120363001@g.us"  # @g.us, no @s.whatsapp.net
    assert r["is_group"] is True
    # Contacts NO se consultó (forced group)
    assert contact_calls == []


def test_contact_lookup_force_group_with_colon(fake_bridge_db, monkeypatch):
    """Variante con dos puntos: `grupo: Random` también funciona."""
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)
    r = _whatsapp_jid_from_contact("grupo: Random")
    assert r["jid"] == "120363001@g.us"
    assert r["is_group"] is True


def test_contact_lookup_force_group_english_prefix(fake_bridge_db, monkeypatch):
    """`group X` (inglés) también es válido — el LLM a veces emite EN
    aunque el user pidió ES."""
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)
    r = _whatsapp_jid_from_contact("group Random")
    assert r["is_group"] is True
    assert r["jid"] == "120363001@g.us"


def test_contact_lookup_not_found_in_either(fake_bridge_db, monkeypatch):
    """Ni en Contacts ni en grupos → not_found definitivo."""
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)
    r = _whatsapp_jid_from_contact("NombreFantasma")
    assert r["jid"] is None
    assert r["is_group"] is False  # default cuando no resolvió a nada
    assert r["error"] == "not_found"


def test_contact_lookup_group_ambiguous_propagates(fake_bridge_db, monkeypatch):
    """Si grupos retorna ambiguous, el contact_lookup propaga el error
    + candidates. El frontend muestra los candidatos."""
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)
    r = _whatsapp_jid_from_contact("cloud")
    assert r["error"] == "ambiguous"
    assert r["is_group"] is True
    assert len(r.get("candidates") or []) == 3


def test_contact_lookup_strips_at_prefix(fake_bridge_db, monkeypatch):
    """`@RagNet` (prefijo Obsidian wikilink) se strip y matchea grupo."""
    monkeypatch.setattr(rag, "_fetch_contact", lambda *a, **kw: None)
    r = _whatsapp_jid_from_contact("@RagNet")
    assert r["jid"] == "120363002@g.us"
    assert r["is_group"] is True


def test_backward_compat_1to1_returns_is_group_false(fake_bridge_db, monkeypatch):
    """El campo `is_group` SIEMPRE viene populado (no missing). Para
    1:1 es False — los callers que esperan ese shape no se rompen."""
    monkeypatch.setattr(
        rag, "_fetch_contact",
        lambda name, email=None, canonical=None: {
            "full_name": "Grecia",
            "phones": ["+5491155556666"],
        },
    )
    r = _whatsapp_jid_from_contact("Grecia")
    assert "is_group" in r
    assert r["is_group"] is False
    assert "phones" in r
    assert r["phones"] == ["+5491155556666"]
