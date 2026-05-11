"""Tests for `detect_wa_silences` dedup by normalized name.

Same contact frequently has multiple JIDs in the WA bridge (linked-device
`@lid` + phone-based `@s.whatsapp.net`). The silence detector groups by
`chat_jid` at the SQL layer, so without dedup the same person surfaces
twice on the home dashboard. These tests lock in the post-SQL dedup that
collapses duplicates and keeps the most-recently-active JID.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from rag.today_patterns import detect_wa_silences


@pytest.fixture
def wa_db(tmp_path: Path) -> Path:
    db = tmp_path / "messages.db"
    con = sqlite3.connect(str(db))
    con.executescript(
        """
        CREATE TABLE chats (jid TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE messages (
            id TEXT, chat_jid TEXT, sender TEXT, content TEXT,
            timestamp TEXT, is_from_me INTEGER, media_type TEXT
        );
        """
    )
    con.commit()
    con.close()
    return db


def _seed_chat(db: Path, jid: str, name: str) -> None:
    con = sqlite3.connect(str(db))
    con.execute("INSERT INTO chats (jid, name) VALUES (?, ?)", (jid, name))
    con.commit()
    con.close()


def _seed_messages_for_chat(
    db: Path,
    chat_jid: str,
    *,
    inbound_count: int,
    last_ts: str,
) -> None:
    """Seed `inbound_count` inbound rows for `chat_jid`, all dated 2 days
    before `last_ts` except the final one which uses `last_ts`. Enough to
    satisfy `_SILENCE_MIN_7D=15` when called with `inbound_count >= 15`.
    """
    rows = []
    base_dt = datetime.fromisoformat(last_ts) if "T" in last_ts else (
        datetime.fromisoformat(last_ts.replace(" ", "T"))
    )
    older_ts = base_dt.isoformat(sep=" ")
    for i in range(inbound_count - 1):
        rows.append((f"m-{chat_jid}-{i}", chat_jid, chat_jid, "x", older_ts, 0, None))
    rows.append((f"m-{chat_jid}-last", chat_jid, chat_jid, "x", last_ts, 0, None))
    con = sqlite3.connect(str(db))
    con.executemany(
        "INSERT INTO messages (id, chat_jid, sender, content, timestamp, "
        "is_from_me, media_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()


def test_silences_dedups_same_name_across_lid_and_phone_jids(wa_db: Path) -> None:
    """Two JIDs (linked-device `@lid` + phone-based) both named "Maria"
    must collapse to one entry with summed `msgs_7d` and the most-recent
    JID as the link target.
    """
    # Yesterday relative to a fixed "now"
    now = datetime(2026, 5, 11, 9, 0, 0)
    yesterday = "2026-05-10 17:00:00"
    four_days_ago = "2026-05-07 12:00:00"
    _seed_chat(wa_db, "255804326297735@lid", "Maria")
    _seed_chat(wa_db, "5493424303891@s.whatsapp.net", "Maria")
    _seed_messages_for_chat(
        wa_db, "255804326297735@lid", inbound_count=120, last_ts=yesterday,
    )
    _seed_messages_for_chat(
        wa_db, "5493424303891@s.whatsapp.net",
        inbound_count=34, last_ts=four_days_ago,
    )
    out = detect_wa_silences(wa_db, bot_jid="bot@s.whatsapp.net", now=now)
    marias = [r for r in out if r["name"].casefold() == "maria"]
    assert len(marias) == 1, f"expected single Maria entry, got {marias!r}"
    entry = marias[0]
    assert entry["msgs_7d"] == 154, "msgs_7d should sum across the two JIDs"
    # Yesterday is the most-recent — keep that JID as the canonical link.
    assert entry["jid"] == "255804326297735@lid"


def test_silences_dedups_case_insensitive_and_trims_trailing_space(
    wa_db: Path,
) -> None:
    """The WA bridge sometimes stores names with trailing whitespace (e.g.
    `"Maria "`). Casefold + trim must collapse those too.
    """
    now = datetime(2026, 5, 11, 9, 0, 0)
    last = "2026-05-10 18:00:00"
    other_last = "2026-05-08 12:00:00"
    _seed_chat(wa_db, "1111@lid", "Maria ")  # trailing space
    _seed_chat(wa_db, "2222@s.whatsapp.net", "maria")  # lowercase variant
    _seed_messages_for_chat(wa_db, "1111@lid", inbound_count=20, last_ts=last)
    _seed_messages_for_chat(
        wa_db, "2222@s.whatsapp.net", inbound_count=18, last_ts=other_last,
    )
    out = detect_wa_silences(wa_db, bot_jid="bot@s.whatsapp.net", now=now)
    marias = [r for r in out if r["name"].strip().casefold() == "maria"]
    assert len(marias) == 1, f"expected single Maria after dedup, got {marias!r}"
    assert marias[0]["msgs_7d"] == 38


def test_silences_keeps_distinct_names_intact(wa_db: Path) -> None:
    """Sanity check — different names must NOT merge, even with similar
    activity patterns.
    """
    now = datetime(2026, 5, 11, 9, 0, 0)
    last = "2026-05-10 18:00:00"
    _seed_chat(wa_db, "111@lid", "Maria")
    _seed_chat(wa_db, "222@lid", "Erica")
    _seed_messages_for_chat(wa_db, "111@lid", inbound_count=20, last_ts=last)
    _seed_messages_for_chat(wa_db, "222@lid", inbound_count=18, last_ts=last)
    out = detect_wa_silences(wa_db, bot_jid="bot@s.whatsapp.net", now=now)
    names = sorted(r["name"] for r in out)
    assert names == ["Erica", "Maria"]
