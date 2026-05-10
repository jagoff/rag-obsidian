"""Tests for the 'stale_contact' Anticipatory Agent signal.

Mockea bridge messages.db con DDL temporal + patch
`WHATSAPP_DB_PATH` + `WHATSAPP_BOT_JID`.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rag.integrations import whatsapp as wa_pkg
from rag_anticipate.signals.stale_contact import _signal_stale_contact


_REF_NOW = datetime(2026, 5, 10, 12, 0, 0)


_BRIDGE_DDL = """
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    chat_jid TEXT NOT NULL,
    sender TEXT NOT NULL,
    content TEXT,
    timestamp TEXT NOT NULL,
    is_from_me INTEGER NOT NULL DEFAULT 0,
    media_type TEXT,
    filename TEXT
);
CREATE TABLE IF NOT EXISTS chats (
    jid TEXT PRIMARY KEY,
    name TEXT,
    last_message_time TEXT
);
"""


@pytest.fixture
def mock_bridge(tmp_path, monkeypatch):
    db_path = tmp_path / "messages.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_BRIDGE_DDL)
    conn.commit()
    conn.close()
    monkeypatch.setattr(wa_pkg, "WHATSAPP_DB_PATH", db_path)
    monkeypatch.setattr(wa_pkg, "WHATSAPP_BOT_JID", "bot@s.whatsapp.net")
    return db_path


def _insert_msg(
    db_path: Path,
    *,
    msg_id: str,
    chat_jid: str = "5491100000000@s.whatsapp.net",
    chat_name: str = "Juan",
    content: str = "hola, estás?",
    is_from_me: int = 0,
    hours_ago: float = 4.0,
):
    ts = _REF_NOW - timedelta(hours=hours_ago)
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S-03:00")
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT OR IGNORE INTO chats (jid, name, last_message_time) "
        "VALUES (?, ?, ?)",
        (chat_jid, chat_name, ts_str),
    )
    sender = "self" if is_from_me else chat_jid
    conn.execute(
        "INSERT INTO messages (id, chat_jid, sender, content, timestamp, is_from_me) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (msg_id, chat_jid, sender, content, ts_str, is_from_me),
    )
    conn.commit()
    conn.close()


# ── Empty / DB missing ───────────────────────────────────────────────────────


def test_no_db_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(wa_pkg, "WHATSAPP_DB_PATH", tmp_path / "nope.db")
    assert _signal_stale_contact(_REF_NOW) == []


def test_empty_db_returns_empty(mock_bridge):
    assert _signal_stale_contact(_REF_NOW) == []


# ── Window filtering ─────────────────────────────────────────────────────────


def test_too_recent_skipped(mock_bridge):
    """<3hs → no emit (user todavía en ventana de respuesta normal)."""
    _insert_msg(mock_bridge, msg_id="m1", hours_ago=1.0)
    assert _signal_stale_contact(_REF_NOW) == []


def test_too_old_skipped(mock_bridge):
    """>72hs → no emit (asumimos paso a propósito)."""
    _insert_msg(mock_bridge, msg_id="m1", hours_ago=80.0)
    assert _signal_stale_contact(_REF_NOW) == []


# ── Score buckets ────────────────────────────────────────────────────────────


def test_4hours_score_0_4(mock_bridge):
    _insert_msg(mock_bridge, msg_id="m1", hours_ago=4.0)
    result = _signal_stale_contact(_REF_NOW)
    assert len(result) == 1
    assert result[0].score == 0.4


def test_8hours_score_0_7(mock_bridge):
    _insert_msg(mock_bridge, msg_id="m1", hours_ago=8.0)
    result = _signal_stale_contact(_REF_NOW)
    assert result[0].score == 0.7


def test_24hours_score_1_0(mock_bridge):
    _insert_msg(mock_bridge, msg_id="m1", hours_ago=24.0)
    result = _signal_stale_contact(_REF_NOW)
    assert result[0].score == 1.0


# ── Already replied ──────────────────────────────────────────────────────────


def test_user_replied_skipped(mock_bridge):
    """User mandó msg después del inbound → no emit."""
    _insert_msg(mock_bridge, msg_id="m1", hours_ago=4.0)
    _insert_msg(mock_bridge, msg_id="m2", is_from_me=1, hours_ago=3.0)
    assert _signal_stale_contact(_REF_NOW) == []


# ── Group chats excluded ─────────────────────────────────────────────────────


def test_group_chat_skipped(mock_bridge):
    _insert_msg(
        mock_bridge, msg_id="m1",
        chat_jid="9999@g.us",
        chat_name="Grupo",
        hours_ago=4.0,
    )
    assert _signal_stale_contact(_REF_NOW) == []


# ── Bot self excluded ────────────────────────────────────────────────────────


def test_bot_jid_excluded(mock_bridge):
    """Mensaje desde el bot mismo → skip (RagNet self-loop)."""
    _insert_msg(
        mock_bridge, msg_id="m1",
        chat_jid="bot@s.whatsapp.net",
        chat_name="Bot",
        hours_ago=4.0,
    )
    assert _signal_stale_contact(_REF_NOW) == []


# ── Anti-noise ───────────────────────────────────────────────────────────────


def test_no_chat_name_skipped(mock_bridge):
    _insert_msg(mock_bridge, msg_id="m1", chat_name="", hours_ago=4.0)
    assert _signal_stale_contact(_REF_NOW) == []


def test_numeric_chat_name_skipped(mock_bridge):
    """Chat name sin alfa (solo dígitos) → señal pobre, skip."""
    _insert_msg(mock_bridge, msg_id="m1", chat_name="+5491100000000", hours_ago=4.0)
    assert _signal_stale_contact(_REF_NOW) == []


def test_bot_name_skipped(mock_bridge):
    _insert_msg(mock_bridge, msg_id="m1", chat_name="NotifBot", hours_ago=4.0)
    assert _signal_stale_contact(_REF_NOW) == []


def test_alert_name_skipped(mock_bridge):
    _insert_msg(mock_bridge, msg_id="m1", chat_name="Banco Alert", hours_ago=4.0)
    assert _signal_stale_contact(_REF_NOW) == []


def test_zero_width_marker_skipped(mock_bridge):
    """Content con U+200B (loop marker) → skip."""
    _insert_msg(mock_bridge, msg_id="m1", content="​hola", hours_ago=4.0)
    assert _signal_stale_contact(_REF_NOW) == []


def test_empty_content_skipped(mock_bridge):
    """Bridge sometimes inserta media-only sin caption → content vacío."""
    _insert_msg(mock_bridge, msg_id="m1", content="", hours_ago=4.0)
    assert _signal_stale_contact(_REF_NOW) == []


# ── Multi-chat dedup ─────────────────────────────────────────────────────────


def test_max_3_candidates(mock_bridge):
    """5 chats stale → cap 3."""
    for i in range(5):
        _insert_msg(
            mock_bridge, msg_id=f"m{i}",
            chat_jid=f"{1000 + i}@s.whatsapp.net",
            chat_name=f"Person{i}",
            hours_ago=4.0 + i,
        )
    result = _signal_stale_contact(_REF_NOW)
    assert len(result) <= 3


def test_only_latest_per_chat(mock_bridge):
    """Si Juan mandó 3 mensajes, solo el último cuenta (1 candidate)."""
    _insert_msg(mock_bridge, msg_id="m1", hours_ago=10.0)
    _insert_msg(mock_bridge, msg_id="m2", hours_ago=8.0)
    _insert_msg(mock_bridge, msg_id="m3", hours_ago=4.0)
    result = _signal_stale_contact(_REF_NOW)
    assert len(result) == 1


# ── Message format ───────────────────────────────────────────────────────────


def test_message_contains_emoji_and_name(mock_bridge):
    _insert_msg(
        mock_bridge, msg_id="m1",
        chat_name="Juan",
        content="¿venís el viernes?",
        hours_ago=4.0,
    )
    result = _signal_stale_contact(_REF_NOW)
    msg = result[0].message
    assert "💬" in msg
    assert "Juan" in msg
    assert "¿venís el viernes?" in msg


def test_message_long_content_truncated(mock_bridge):
    long = "x" * 200
    _insert_msg(mock_bridge, msg_id="m1", content=long, hours_ago=4.0)
    result = _signal_stale_contact(_REF_NOW)
    msg = result[0].message
    # content snippet capped at 120
    assert "x" * 200 not in msg


# ── Dedup key ────────────────────────────────────────────────────────────────


def test_dedup_key_format(mock_bridge):
    _insert_msg(
        mock_bridge, msg_id="abc",
        chat_jid="9999@s.whatsapp.net",
        hours_ago=4.0,
    )
    result = _signal_stale_contact(_REF_NOW)
    assert result[0].dedup_key == "stale-contact:9999@s.whatsapp.net:abc"


# ── Registry ─────────────────────────────────────────────────────────────────


def test_signal_registered():
    import rag_anticipate.signals  # noqa: F401
    from rag_anticipate.signals.base import SIGNALS
    names = [n for n, _ in SIGNALS]
    assert "stale_contact" in names
