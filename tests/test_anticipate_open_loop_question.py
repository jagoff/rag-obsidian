"""Tests for the 'open_loop_question' Anticipatory Agent signal.

DB pattern: usa el bridge `messages.db` schema (campo `is_from_me`,
`chat_jid`, `content`, `timestamp`, `sender`, `id`). Patch
`rag.integrations.whatsapp.WHATSAPP_DB_PATH` para apuntar al
fixture temporal.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rag.integrations import whatsapp as wa_pkg
from rag_anticipate.signals.open_loop_question import (
    _is_greeting_question,
    _is_question,
    open_loop_question_signal,
)


_REF_NOW = datetime(2026, 5, 9, 18, 0, 0)


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
    """Crea bridge DB temporal + patch WHATSAPP_DB_PATH."""
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
    content: str = "¿cuándo nos vemos?",
    is_from_me: int = 0,
    days_ago: float = 3.0,
):
    """Insert msg + chat row (idempotente para chat)."""
    ts = _REF_NOW - timedelta(days=days_ago)
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


# ── _is_question helpers ─────────────────────────────────────────────────────


def test_is_question_with_question_mark():
    assert _is_question("¿cuándo nos vemos?")
    assert _is_question("nos vemos hoy?")
    assert _is_question("Q?")


def test_is_question_with_interrogative_word():
    assert _is_question("qué hora hablaste con él")
    assert _is_question("cómo te fue ayer")
    assert _is_question("cuándo llegás")
    assert _is_question("dónde dejaste el tema")
    assert _is_question("por qué no contestaste")


def test_is_question_statement_returns_false():
    assert not _is_question("ya llegué")
    assert not _is_question("ok dale")
    assert not _is_question("te paso el archivo mañana")


def test_is_question_empty():
    assert not _is_question("")
    assert not _is_question("   ")


def test_is_greeting_question_detected():
    assert _is_greeting_question("¿cómo estás?")
    assert _is_greeting_question("todo bien?")
    assert _is_greeting_question("qué tal!")
    assert _is_greeting_question("qué hacés")


def test_is_greeting_question_real_question_returns_false():
    """Pregunta real que MENCIONA palabras de greeting pero es legítima."""
    assert not _is_greeting_question(
        "¿cómo estás de tiempo para revisar el contrato la semana que viene?"
    )
    # Demasiado largo para ser greeting (>30 chars cleaned)


# ── Empty / DB missing ───────────────────────────────────────────────────────


def test_no_db_returns_empty(monkeypatch, tmp_path):
    """WHATSAPP_DB_PATH no existe → []."""
    monkeypatch.setattr(
        wa_pkg, "WHATSAPP_DB_PATH", tmp_path / "no-existe.db",
    )
    assert open_loop_question_signal(_REF_NOW) == []


def test_empty_db_returns_empty(mock_bridge):
    assert open_loop_question_signal(_REF_NOW) == []


# ── Window filtering ─────────────────────────────────────────────────────────


def test_too_recent_skipped(mock_bridge):
    """<48h → no emit."""
    _insert_msg(mock_bridge, msg_id="m1", days_ago=1.0)
    assert open_loop_question_signal(_REF_NOW) == []


def test_too_old_skipped(mock_bridge):
    """>30d → no emit."""
    _insert_msg(mock_bridge, msg_id="m1", days_ago=45.0)
    assert open_loop_question_signal(_REF_NOW) == []


# ── Question filter ──────────────────────────────────────────────────────────


def test_statement_skipped(mock_bridge):
    """No es pregunta → skip aunque esté en window."""
    _insert_msg(mock_bridge, msg_id="m1", content="ya llegué", days_ago=3.0)
    assert open_loop_question_signal(_REF_NOW) == []


def test_greeting_question_skipped(mock_bridge):
    """¿cómo estás? = greeting, skip."""
    _insert_msg(mock_bridge, msg_id="m1", content="¿cómo estás?", days_ago=3.0)
    assert open_loop_question_signal(_REF_NOW) == []


def test_question_window_48h_emits(mock_bridge):
    """Pregunta hace ~3 días → emit score 0.6."""
    _insert_msg(
        mock_bridge, msg_id="m1",
        content="¿cuándo nos juntamos para el demo?",
        days_ago=3.0,
    )
    result = open_loop_question_signal(_REF_NOW)
    assert len(result) == 1
    c = result[0]
    assert c.kind == "anticipate-open_loop_question"
    assert c.score == 0.6
    assert "Juan" in c.message
    assert "demo" in c.message


def test_question_5_days_emits_score_0_8(mock_bridge):
    _insert_msg(
        mock_bridge, msg_id="m1",
        content="¿qué te pareció el documento?",
        days_ago=5.0,
    )
    result = open_loop_question_signal(_REF_NOW)
    assert len(result) == 1
    assert result[0].score == 0.8


def test_question_15_days_emits_score_1_0(mock_bridge):
    _insert_msg(
        mock_bridge, msg_id="m1",
        content="¿finalmente encontraste tiempo para eso?",
        days_ago=15.0,
    )
    result = open_loop_question_signal(_REF_NOW)
    assert len(result) == 1
    assert result[0].score == 1.0


# ── Already answered ─────────────────────────────────────────────────────────


def test_user_replied_skip(mock_bridge):
    """User mandó msg después de la pregunta → no emit."""
    _insert_msg(
        mock_bridge, msg_id="m1",
        content="¿cuándo te llamo?", days_ago=3.0,
    )
    _insert_msg(
        mock_bridge, msg_id="m2",
        content="te llamo a las 5",
        is_from_me=1, days_ago=2.5,
    )
    assert open_loop_question_signal(_REF_NOW) == []


# ── Multiple chats ───────────────────────────────────────────────────────────


def test_max_1_candidate(mock_bridge):
    """3 preguntas en 3 chats → emit solo 1 (la más reciente match-eable)."""
    _insert_msg(mock_bridge, msg_id="m1",
                chat_jid="111@s.whatsapp.net", chat_name="A",
                content="¿qué onda?", days_ago=10.0)
    _insert_msg(mock_bridge, msg_id="m2",
                chat_jid="222@s.whatsapp.net", chat_name="B",
                content="¿cuándo nos juntamos?", days_ago=3.0)
    _insert_msg(mock_bridge, msg_id="m3",
                chat_jid="333@s.whatsapp.net", chat_name="C",
                content="¿qué tal?", days_ago=5.0)
    result = open_loop_question_signal(_REF_NOW)
    assert len(result) == 1


# ── Anti-noise ───────────────────────────────────────────────────────────────


def test_bot_chat_name_skipped(mock_bridge):
    """Sender con nombre tipo 'NotifBot' → skip."""
    _insert_msg(
        mock_bridge, msg_id="m1",
        chat_name="NotifBot",
        content="¿necesitás soporte hoy?",
        days_ago=3.0,
    )
    assert open_loop_question_signal(_REF_NOW) == []


def test_no_chat_name_skipped(mock_bridge):
    """Chat sin name → skip (señal pobre)."""
    _insert_msg(
        mock_bridge, msg_id="m1",
        chat_name="",
        content="¿venís el viernes?",
        days_ago=3.0,
    )
    assert open_loop_question_signal(_REF_NOW) == []


def test_self_loop_marker_skipped(mock_bridge):
    """Content con U+200B (marker bot loops) → skip."""
    _insert_msg(
        mock_bridge, msg_id="m1",
        content="​¿llegó el archivo?",
        days_ago=3.0,
    )
    assert open_loop_question_signal(_REF_NOW) == []


# ── Dedup key ────────────────────────────────────────────────────────────────


def test_dedup_key_format(mock_bridge):
    _insert_msg(
        mock_bridge, msg_id="abc123",
        chat_jid="999@s.whatsapp.net",
        content="¿revisaste el doc?",
        days_ago=3.0,
    )
    result = open_loop_question_signal(_REF_NOW)
    assert result[0].dedup_key == "open_loop_question:999@s.whatsapp.net:abc123"


# ── Group chats excluded ─────────────────────────────────────────────────────


def test_group_chat_skipped(mock_bridge):
    """Grupos (@g.us suffix) → skip por design."""
    _insert_msg(
        mock_bridge, msg_id="m1",
        chat_jid="9999@g.us",  # group
        chat_name="GrupoAmigos",
        content="¿quién banca el asado?",
        days_ago=3.0,
    )
    assert open_loop_question_signal(_REF_NOW) == []


# ── Registry ─────────────────────────────────────────────────────────────────


def test_signal_registered():
    import rag_anticipate.signals  # noqa: F401
    from rag_anticipate.signals.base import SIGNALS
    names = [n for n, _ in SIGNALS]
    assert "open_loop_question" in names
