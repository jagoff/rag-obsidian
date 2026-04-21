"""Tests for `_build_wa_cross_ref` and its helpers (WA cross-reference for
person-entity queries).

Feature: when the WA listener routes a factual question ("qué sabés de
Grecia?") to `rag serve /query`, the handler detects the person via
`_match_mentions_in_query`, pulls the last 3 WhatsApp messages mentioning
them from the bridge SQLite, and attaches a `wa_cross_ref` block to the
JSON payload. If any of those messages carry a parseable commitment
("nos vemos mañana 19hs"), a `propose` sub-payload is included so the
listener can prime a `pendingEvents` entry for one-shot confirmation.

These tests isolate the function with mocked vault + bridge. The real
bridge path is also tested for the SQL shape via an in-memory sqlite DB
seeded with the same schema the whatsapp-bridge uses.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

import rag


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Minimal vault with a 99-Mentions folder containing two dossiers."""
    mentions = tmp_path / "04-Archive/99-obsidian-system/99-Mentions"
    mentions.mkdir(parents=True)
    (mentions / "Grecia.md").write_text(
        "[[04-Archive/99-obsidian-system/99-Mentions/Grecia|@Grecia]]\n"
        "- **Relación**: hija\n",
        encoding="utf-8",
    )
    (mentions / "Seba.md").write_text(
        "---\n"
        "type: mention\n"
        "aliases:\n"
        "  - Sebastián\n"
        "  - Seba Serra\n"
        "---\n"
        "- **Relación**: amigo\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def wa_db(tmp_path: Path) -> Path:
    """Seeded in-memory-like WA bridge SQLite (real file, same schema)."""
    db = tmp_path / "messages.db"
    con = sqlite3.connect(str(db))
    con.executescript(
        """
        CREATE TABLE chats (jid TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE messages (
            id TEXT, chat_jid TEXT, sender TEXT, content TEXT,
            timestamp TEXT, is_from_me INTEGER, media_type TEXT
        );
        INSERT INTO chats VALUES
            ('grecia-direct@s.whatsapp.net', 'Grecia'),
            ('grecia-group@g.us', "Grecia's group"),
            ('other-group@g.us', 'Club de natación'),
            ('seba@s.whatsapp.net', 'Seba');
        """
    )
    con.commit()
    con.close()
    return db


def _seed_messages(db: Path, rows: list[tuple]) -> None:
    con = sqlite3.connect(str(db))
    con.executemany(
        "INSERT INTO messages (id, chat_jid, sender, content, timestamp, "
        "is_from_me, media_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()


# ── _parse_mention_dossier ──────────────────────────────────────────────────

def test_parse_mention_dossier_without_frontmatter(vault: Path) -> None:
    d = rag._parse_mention_dossier(
        "04-Archive/99-obsidian-system/99-Mentions/Grecia.md",
        vault_root=vault,
    )
    # `phone_digits` added 2026-04-21 alongside `name` / `aliases` — the
    # Grecia fixture has no phone line so the field is empty.
    assert d == {"name": "Grecia", "aliases": [], "phone_digits": ""}


def test_parse_mention_dossier_with_aliases(vault: Path) -> None:
    d = rag._parse_mention_dossier(
        "04-Archive/99-obsidian-system/99-Mentions/Seba.md",
        vault_root=vault,
    )
    assert d["name"] == "Seba"
    assert "Sebastián" in d["aliases"]
    assert "Seba Serra" in d["aliases"]


def test_parse_mention_dossier_missing_file(vault: Path) -> None:
    # Unreadable path → empty aliases, name from stem. Caller treats as
    # "entity with no extra context".
    d = rag._parse_mention_dossier(
        "does/not/exist.md",
        vault_root=vault,
    )
    assert d["aliases"] == []


# ── _lookup_wa_mentions_for_entity ──────────────────────────────────────────

def test_lookup_prefers_chat_name_match(vault: Path, wa_db: Path) -> None:
    """A message in a chat named 'Grecia' outranks a content match in
    another chat, even if the content-match is newer."""
    _seed_messages(wa_db, [
        ("m1", "other-group@g.us", "+5491100@s", "nos vemos con Grecia mañana",
         "2026-04-20 10:00:00", 0, ""),
        ("m2", "grecia-direct@s.whatsapp.net", "+5491200@s", "hola pa",
         "2026-04-18 09:00:00", 0, ""),
    ])
    msgs = rag._lookup_wa_mentions_for_entity(
        "04-Archive/99-obsidian-system/99-Mentions/Grecia.md",
        vault_root=vault, db_path=wa_db,
    )
    assert len(msgs) == 2
    assert msgs[0]["match"] == "chat"
    assert msgs[0]["id"] == "m2"
    assert msgs[1]["match"] == "content"


def test_lookup_caps_at_limit(vault: Path, wa_db: Path) -> None:
    _seed_messages(wa_db, [
        (f"m{i}", "grecia-direct@s.whatsapp.net", "+5491200@s",
         f"msg {i}", f"2026-04-{10+i:02d} 12:00:00", 0, "")
        for i in range(10)
    ])
    msgs = rag._lookup_wa_mentions_for_entity(
        "04-Archive/99-obsidian-system/99-Mentions/Grecia.md",
        vault_root=vault, db_path=wa_db, limit=3,
    )
    assert len(msgs) == 3
    # Newest first within chat bucket
    assert msgs[0]["text"].endswith("9")


def test_lookup_alias_hits_grecias_group(vault: Path, wa_db: Path) -> None:
    """Chat 'Grecia's group' should match via substring, not only the
    dedicated chat. Real user groups frequently embed the entity name."""
    _seed_messages(wa_db, [
        ("g1", "grecia-group@g.us", "+5491300@s", "Ahora estoy re cansada",
         "2026-04-19 18:11:00", 0, ""),
    ])
    msgs = rag._lookup_wa_mentions_for_entity(
        "04-Archive/99-obsidian-system/99-Mentions/Grecia.md",
        vault_root=vault, db_path=wa_db,
    )
    assert len(msgs) == 1
    assert msgs[0]["match"] == "chat"
    assert msgs[0]["chat_name"] == "Grecia's group"


def test_lookup_skips_empty_and_status_broadcast(vault: Path, wa_db: Path) -> None:
    con = sqlite3.connect(str(wa_db))
    con.execute("INSERT INTO chats VALUES ('status@broadcast', 'Status')")
    con.commit()
    con.close()
    _seed_messages(wa_db, [
        ("s1", "status@broadcast", None, "estado con Grecia",
         "2026-04-20 09:00:00", 0, ""),
        ("e1", "grecia-direct@s.whatsapp.net", "+a", "",
         "2026-04-20 08:00:00", 0, ""),  # empty → skipped
        ("ok", "grecia-direct@s.whatsapp.net", "+a", "hola",
         "2026-04-20 07:00:00", 0, ""),
    ])
    msgs = rag._lookup_wa_mentions_for_entity(
        "04-Archive/99-obsidian-system/99-Mentions/Grecia.md",
        vault_root=vault, db_path=wa_db,
    )
    assert [m["id"] for m in msgs] == ["ok"]


def test_lookup_handles_alias_tokens(vault: Path, wa_db: Path) -> None:
    """Entity `Seba` has aliases {Sebastián, Seba Serra}. A message from
    the `Seba` chat hits via chat-name, but a mention of `Seba Serra` in
    a group should also surface via content-alias match."""
    _seed_messages(wa_db, [
        ("s1", "seba@s.whatsapp.net", "+x", "todo bien?",
         "2026-04-20 11:00:00", 0, ""),
        ("s2", "other-group@g.us", "+y", "le dije a Seba Serra de venir",
         "2026-04-19 16:00:00", 0, ""),
    ])
    msgs = rag._lookup_wa_mentions_for_entity(
        "04-Archive/99-obsidian-system/99-Mentions/Seba.md",
        vault_root=vault, db_path=wa_db,
    )
    ids = [m["id"] for m in msgs]
    assert "s1" in ids
    assert "s2" in ids


def test_lookup_returns_empty_when_db_missing(vault: Path, tmp_path: Path) -> None:
    missing = tmp_path / "no-db.sqlite"
    assert not missing.exists()
    msgs = rag._lookup_wa_mentions_for_entity(
        "04-Archive/99-obsidian-system/99-Mentions/Grecia.md",
        vault_root=vault, db_path=missing,
    )
    assert msgs == []


def test_lookup_returns_empty_for_stemless_mention(tmp_path: Path, wa_db: Path) -> None:
    """A mention dossier whose stem is shorter than the min-token length
    cannot be safely matched (would hit unrelated content). Returns []."""
    mentions = tmp_path / "04-Archive/99-obsidian-system/99-Mentions"
    mentions.mkdir(parents=True)
    (mentions / "Al.md").write_text("- short", encoding="utf-8")  # stem=Al, 2 chars
    _seed_messages(wa_db, [
        ("a1", "seba@s.whatsapp.net", "+a", "Al fin llegó",
         "2026-04-20 09:00:00", 0, ""),
    ])
    msgs = rag._lookup_wa_mentions_for_entity(
        "04-Archive/99-obsidian-system/99-Mentions/Al.md",
        vault_root=tmp_path, db_path=wa_db,
    )
    assert msgs == []


# ── _build_wa_cross_ref ─────────────────────────────────────────────────────

def test_cross_ref_none_when_no_entity(vault: Path, wa_db: Path) -> None:
    with patch.object(rag, "WHATSAPP_DB_PATH", wa_db), \
         patch.object(rag, "VAULT_PATH", vault):
        rag._mentions_cache = None  # force reload
        ref = rag._build_wa_cross_ref("qué sabés de docker")
    assert ref is None


def test_cross_ref_none_when_no_messages(vault: Path, wa_db: Path) -> None:
    # DB has chats but no Grecia messages — build_wa_cross_ref returns None
    # rather than an empty-messages shell.
    with patch.object(rag, "WHATSAPP_DB_PATH", wa_db), \
         patch.object(rag, "VAULT_PATH", vault):
        rag._mentions_cache = None
        ref = rag._build_wa_cross_ref("qué sabés de Grecia")
    assert ref is None


def test_cross_ref_without_propose(vault: Path, wa_db: Path) -> None:
    _seed_messages(wa_db, [
        ("m1", "grecia-direct@s.whatsapp.net", "+g", "hola pa",
         "2026-04-18 09:00:00", 0, ""),
    ])
    with patch.object(rag, "WHATSAPP_DB_PATH", wa_db), \
         patch.object(rag, "VAULT_PATH", vault):
        rag._mentions_cache = None
        ref = rag._build_wa_cross_ref("qué sabés de Grecia")
    assert ref is not None
    assert ref["entity"] == "Grecia"
    assert ref["propose"] is None
    assert len(ref["messages"]) == 1


def test_cross_ref_with_propose_calendar_event(vault: Path, wa_db: Path) -> None:
    """Message with parseable time ('mañana 19hs') → propose kind=calendar_event."""
    _seed_messages(wa_db, [
        ("m1", "grecia-direct@s.whatsapp.net", "+g",
         "nos vemos mañana 19hs dale?", "2026-04-21 10:00:00", 0, ""),
    ])
    anchor = datetime(2026, 4, 21, 12, 0, 0)
    with patch.object(rag, "WHATSAPP_DB_PATH", wa_db), \
         patch.object(rag, "VAULT_PATH", vault):
        rag._mentions_cache = None
        ref = rag._build_wa_cross_ref("qué sabés de Grecia", now=anchor)
    assert ref is not None
    assert ref["propose"] is not None
    assert ref["propose"]["kind"] == "calendar_event"
    assert "19:00" in ref["propose"]["when_iso"]


def test_cross_ref_with_propose_reminder_date_only(vault: Path, wa_db: Path) -> None:
    """Message with date but no explicit time → propose kind=reminder."""
    _seed_messages(wa_db, [
        ("m1", "grecia-direct@s.whatsapp.net", "+g",
         "el viernes vamos a la playa", "2026-04-19 15:00:00", 0, ""),
    ])
    anchor = datetime(2026, 4, 21, 12, 0, 0)  # monday
    with patch.object(rag, "WHATSAPP_DB_PATH", wa_db), \
         patch.object(rag, "VAULT_PATH", vault):
        rag._mentions_cache = None
        ref = rag._build_wa_cross_ref("qué sabés de Grecia", now=anchor)
    assert ref is not None
    # Date without explicit time → reminder (all-day downstream)
    assert ref["propose"] is not None
    assert ref["propose"]["kind"] == "reminder"


def test_cross_ref_skips_outbound_for_propose(vault: Path, wa_db: Path) -> None:
    """A commitment written by `yo` is an invitation we already know
    about — only inbound messages should raise a propose card. The
    outbound message still appears in messages[] though."""
    _seed_messages(wa_db, [
        ("m1", "grecia-direct@s.whatsapp.net", "+me",
         "nos vemos mañana 19hs dale?", "2026-04-21 10:00:00", 1, ""),
    ])
    anchor = datetime(2026, 4, 21, 12, 0, 0)
    with patch.object(rag, "WHATSAPP_DB_PATH", wa_db), \
         patch.object(rag, "VAULT_PATH", vault):
        rag._mentions_cache = None
        ref = rag._build_wa_cross_ref("qué sabés de Grecia", now=anchor)
    assert ref is not None
    assert len(ref["messages"]) == 1
    assert ref["messages"][0]["is_from_me"] is True
    assert ref["propose"] is None  # outbound commitment → no card


def test_cross_ref_skips_past_dates(vault: Path, wa_db: Path) -> None:
    """'te vi ayer' parses to yesterday — not actionable."""
    _seed_messages(wa_db, [
        ("m1", "grecia-direct@s.whatsapp.net", "+g",
         "te vi ayer en el super", "2026-04-21 10:00:00", 0, ""),
    ])
    anchor = datetime(2026, 4, 21, 12, 0, 0)
    with patch.object(rag, "WHATSAPP_DB_PATH", wa_db), \
         patch.object(rag, "VAULT_PATH", vault):
        rag._mentions_cache = None
        ref = rag._build_wa_cross_ref("qué sabés de Grecia", now=anchor)
    # The message surfaces but no propose (dt <= anchor).
    if ref is not None:
        assert ref["propose"] is None


# ── _load_user_nickname ─────────────────────────────────────────────────────

@pytest.fixture
def vault_with_yo(tmp_path: Path) -> Path:
    """Vault with a Yo.md dossier (body-only, no aliases)."""
    mentions = tmp_path / "04-Archive/99-obsidian-system/99-Mentions"
    mentions.mkdir(parents=True)
    (mentions / "Yo.md").write_text(
        "[[Yo|@Yo]]\n"
        "- **Relación**: \n"
        "- **Apellido / nombre completo**: Ferrari Fernando\n"
        "- **Cumpleaños**: 19/07/1981\n",
        encoding="utf-8",
    )
    return tmp_path


def test_nickname_from_body_last_token(vault_with_yo: Path) -> None:
    """AR convention: 'Apellido Nombre' → last token = nombre de pila."""
    rag._user_nickname_cache = None
    assert rag._load_user_nickname(vault_with_yo) == "Fernando"


def test_nickname_prefers_aliases_over_body(tmp_path: Path) -> None:
    """User-declared alias wins — the `Fer` in aliases beats `Fernando`
    extracted from the body."""
    mentions = tmp_path / "04-Archive/99-obsidian-system/99-Mentions"
    mentions.mkdir(parents=True)
    (mentions / "Yo.md").write_text(
        "---\n"
        "type: mention\n"
        "aliases:\n"
        "  - Fer\n"
        "  - Fernando\n"
        "---\n"
        "- **Apellido / nombre completo**: Ferrari Fernando\n",
        encoding="utf-8",
    )
    rag._user_nickname_cache = None
    assert rag._load_user_nickname(tmp_path) == "Fer"


def test_nickname_inline_aliases(tmp_path: Path) -> None:
    mentions = tmp_path / "04-Archive/99-obsidian-system/99-Mentions"
    mentions.mkdir(parents=True)
    (mentions / "Yo.md").write_text(
        "---\naliases: [Fede, Federico Serra]\n---\n"
        "- body content\n",
        encoding="utf-8",
    )
    rag._user_nickname_cache = None
    assert rag._load_user_nickname(tmp_path) == "Fede"


def test_nickname_none_when_yo_missing(tmp_path: Path) -> None:
    (tmp_path / "04-Archive/99-obsidian-system/99-Mentions").mkdir(parents=True)
    # No Yo.md written.
    rag._user_nickname_cache = None
    assert rag._load_user_nickname(tmp_path) is None


def test_nickname_none_when_body_field_empty(tmp_path: Path) -> None:
    """Empty 'Apellido / nombre completo' → no fallback name."""
    mentions = tmp_path / "04-Archive/99-obsidian-system/99-Mentions"
    mentions.mkdir(parents=True)
    (mentions / "Yo.md").write_text(
        "- **Apellido / nombre completo**:\n",
        encoding="utf-8",
    )
    rag._user_nickname_cache = None
    assert rag._load_user_nickname(tmp_path) is None


def test_nickname_cached_by_mtime(tmp_path: Path) -> None:
    """Subsequent calls with unchanged mtime return the cached value."""
    mentions = tmp_path / "04-Archive/99-obsidian-system/99-Mentions"
    mentions.mkdir(parents=True)
    yo = mentions / "Yo.md"
    yo.write_text(
        "- **Apellido / nombre completo**: García Lucía\n",
        encoding="utf-8",
    )
    rag._user_nickname_cache = None
    assert rag._load_user_nickname(tmp_path) == "Lucía"
    # Simulate no mtime change → must hit cache (even if we tamper with
    # the file but preserve mtime, the result is stable).
    cached_before = rag._user_nickname_cache
    assert cached_before is not None
    assert rag._load_user_nickname(tmp_path) == "Lucía"
    assert rag._user_nickname_cache is cached_before


# ── _load_phone_index + _resolve_sender_to_name ────────────────────────────


@pytest.fixture
def vault_with_phones(tmp_path: Path) -> Path:
    """Vault with Mentions dossiers carrying phones for sender resolution."""
    mentions = tmp_path / "04-Archive/99-obsidian-system/99-Mentions"
    mentions.mkdir(parents=True)
    (mentions / "Grecia.md").write_text(
        "[[Grecia|@Grecia]]\n"
        "- **Relación**: hija\n"
        "- **Teléfono**: +54 9 342 5153999\n",
        encoding="utf-8",
    )
    (mentions / "Seba.md").write_text(
        "---\naliases: [Sebastián]\n---\n"
        "- **Teléfono**: +54 9 11 8765 4321\n",
        encoding="utf-8",
    )
    return tmp_path


def test_phone_index_extracts_digits(vault_with_phones: Path) -> None:
    rag._phone_index_cache = None
    idx = rag._load_phone_index(vault_root=vault_with_phones)
    # Digits-only keys; both entries present.
    assert "5493425153999" in idx
    assert idx["5493425153999"] == "Grecia"
    assert "5491187654321" in idx
    assert idx["5491187654321"] == "Seba"


def test_resolve_sender_with_matching_phone(vault_with_phones: Path) -> None:
    rag._phone_index_cache = None
    # Full JID with suffix.
    assert rag._resolve_sender_to_name(
        "5493425153999@s.whatsapp.net", vault_root=vault_with_phones,
    ) == "Grecia"
    # Bare digits.
    assert rag._resolve_sender_to_name(
        "5493425153999", vault_root=vault_with_phones,
    ) == "Grecia"


def test_resolve_sender_unmapped_shows_last_4(vault_with_phones: Path) -> None:
    """Unknown senders render as `…8025` (last 4 digits), not the full JID.
    Otherwise groups with unknown members show like `34084894028025` which
    is unreadable noise."""
    rag._phone_index_cache = None
    assert rag._resolve_sender_to_name(
        "34084894028025@s.whatsapp.net", vault_root=vault_with_phones,
    ) == "…8025"


def test_resolve_sender_empty_uses_fallback(vault_with_phones: Path) -> None:
    rag._phone_index_cache = None
    assert rag._resolve_sender_to_name(
        "", fallback="Grecia's group", vault_root=vault_with_phones,
    ) == "Grecia's group"


def test_resolve_sender_trailing_digits_match(vault_with_phones: Path) -> None:
    """Country-code variations: a JID stored as `+54 9 342 5153999` in the
    dossier should still match if WA reports the sender as a longer/
    shorter variant. Implemented as suffix-overlap fallback."""
    rag._phone_index_cache = None
    # Incoming without country prefix, dossier stored with it.
    assert rag._resolve_sender_to_name(
        "3425153999", vault_root=vault_with_phones,
    ) == "Grecia"


def test_cross_ref_independent_of_nickname(vault: Path, wa_db: Path) -> None:
    """`_build_wa_cross_ref` must not require a nickname to function —
    nickname is a pure UI concern surfaced via a different payload field."""
    _seed_messages(wa_db, [
        ("m1", "grecia-direct@s.whatsapp.net", "+g", "hola",
         "2026-04-19 10:00:00", 0, ""),
    ])
    with patch.object(rag, "WHATSAPP_DB_PATH", wa_db), \
         patch.object(rag, "VAULT_PATH", vault):
        rag._mentions_cache = None
        rag._user_nickname_cache = None
        ref = rag._build_wa_cross_ref("qué sabés de Grecia")
    assert ref is not None
    # Cross-ref payload doesn't embed nickname — serve handler attaches it.
    assert "user_nickname" not in ref
