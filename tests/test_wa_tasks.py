"""WhatsApp action-item extractor — state, window fetch, note writing.

No LLM or ollama calls — `_wa_extract_actions` is tested via the public
shape (stubbed). The real work is in deterministic helpers: the sqlite
window query, the state ring, and the markdown writer.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag


def _build_fake_wa_db(path: Path) -> None:
    """Mirrors whatsapp-bridge schema: `chats(jid, name, last_message_time)` +
    `messages(id, chat_jid, sender, content, timestamp, is_from_me, media_type)`.
    """
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE chats (
            jid TEXT PRIMARY KEY,
            name TEXT,
            last_message_time TIMESTAMP
        )
        """
    )
    con.execute(
        """
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
        )
        """
    )
    con.commit()
    con.close()


def _insert_chat(path: Path, jid: str, name: str) -> None:
    con = sqlite3.connect(path)
    con.execute("INSERT INTO chats (jid, name) VALUES (?, ?)", (jid, name))
    con.commit()
    con.close()


def _insert_msg(
    path: Path, *, mid: str, jid: str, sender: str, content: str,
    ts: datetime, is_from_me: int = 0, media_type: str = "",
) -> None:
    con = sqlite3.connect(path)
    con.execute(
        "INSERT INTO messages (id, chat_jid, sender, content, timestamp, "
        "is_from_me, media_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (mid, jid, sender, content,
         ts.strftime("%Y-%m-%d %H:%M:%S"), is_from_me, media_type),
    )
    con.commit()
    con.close()


class TestChatLabel:
    def test_alpha_name_passes_through(self):
        assert rag._wa_chat_label("Grecia's group", "123@g.us") == "Grecia's group"

    def test_numeric_only_falls_back_to_tail(self):
        assert rag._wa_chat_label("5491128506670", "5491128506670@s.whatsapp.net") == "Contacto …6670"

    def test_empty_name_uses_jid_tail(self):
        assert rag._wa_chat_label("", "5491131175833@s.whatsapp.net") == "Contacto …5833"

    def test_empty_jid_fallback(self):
        assert rag._wa_chat_label("", "") == "Contacto"


class TestState:
    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rag, "WA_TASKS_STATE_PATH", tmp_path / "wa.json")
        state = rag._wa_tasks_load_state()
        assert state == {"last_run_ts": None, "processed_ids": []}

    def test_save_and_reload_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rag, "WA_TASKS_STATE_PATH", tmp_path / "wa.json")
        rag._wa_tasks_save_state({
            "last_run_ts": "2026-04-17T12:00:00",
            "processed_ids": ["a", "b", "c"],
        })
        state = rag._wa_tasks_load_state()
        assert state["last_run_ts"] == "2026-04-17T12:00:00"
        assert state["processed_ids"] == ["a", "b", "c"]

    def test_processed_ids_ring_caps_at_2000(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rag, "WA_TASKS_STATE_PATH", tmp_path / "wa.json")
        rag._wa_tasks_save_state({
            "last_run_ts": None,
            "processed_ids": [f"id{i}" for i in range(2500)],
        })
        state = rag._wa_tasks_load_state()
        assert len(state["processed_ids"]) == 2000
        # Oldest ids dropped, newest kept.
        assert state["processed_ids"][0] == "id500"
        assert state["processed_ids"][-1] == "id2499"

    def test_corrupt_json_returns_empty(self, tmp_path, monkeypatch):
        p = tmp_path / "wa.json"
        p.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(rag, "WA_TASKS_STATE_PATH", p)
        assert rag._wa_tasks_load_state() == {"last_run_ts": None, "processed_ids": []}


class TestWindowFetch:
    def test_empty_db_returns_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rag, "WHATSAPP_DB_PATH", tmp_path / "nope.db")
        now = datetime(2026, 4, 17, 15, 0)
        assert rag._fetch_whatsapp_window(now - timedelta(hours=24), now, set()) == []

    def test_groups_by_chat_and_filters_below_min_inbound(self, tmp_path, monkeypatch):
        db = tmp_path / "wa.db"
        _build_fake_wa_db(db)
        _insert_chat(db, "A@g.us", "Grupo A")
        _insert_chat(db, "B@s.whatsapp.net", "Beto")
        now = datetime(2026, 4, 17, 15, 0)
        # Grupo A: 3 inbound (passes min=2)
        for i in range(3):
            _insert_msg(db, mid=f"a{i}", jid="A@g.us", sender="x@s.whatsapp.net",
                        content=f"msg {i}", ts=now - timedelta(minutes=60 - i * 5))
        # Beto: 1 inbound (below min)
        _insert_msg(db, mid="b0", jid="B@s.whatsapp.net", sender="B@s.whatsapp.net",
                    content="hola", ts=now - timedelta(minutes=10))
        monkeypatch.setattr(rag, "WHATSAPP_DB_PATH", db)

        out = rag._fetch_whatsapp_window(now - timedelta(hours=2), now, set())
        assert len(out) == 1
        assert out[0]["jid"] == "A@g.us"
        assert out[0]["label"] == "Grupo A"
        assert out[0]["is_group"] is True
        assert out[0]["inbound"] == 3

    def test_skips_bot_group_and_status(self, tmp_path, monkeypatch):
        db = tmp_path / "wa.db"
        _build_fake_wa_db(db)
        _insert_chat(db, rag.WHATSAPP_BOT_JID, "RagNet")
        _insert_chat(db, "status@broadcast", "Status")
        _insert_chat(db, "Real@s.whatsapp.net", "Real")
        now = datetime(2026, 4, 17, 15, 0)
        for i in range(3):
            _insert_msg(db, mid=f"bot{i}", jid=rag.WHATSAPP_BOT_JID, sender="x",
                        content="noise", ts=now - timedelta(minutes=30 - i))
            _insert_msg(db, mid=f"st{i}", jid="status@broadcast", sender="x",
                        content="status", ts=now - timedelta(minutes=20 - i))
            _insert_msg(db, mid=f"r{i}", jid="Real@s.whatsapp.net", sender="Real@s.whatsapp.net",
                        content=f"real {i}", ts=now - timedelta(minutes=15 - i))
        monkeypatch.setattr(rag, "WHATSAPP_DB_PATH", db)
        out = rag._fetch_whatsapp_window(now - timedelta(hours=2), now, set())
        jids = {e["jid"] for e in out}
        assert jids == {"Real@s.whatsapp.net"}

    def test_skips_chats_with_only_processed_messages(self, tmp_path, monkeypatch):
        db = tmp_path / "wa.db"
        _build_fake_wa_db(db)
        _insert_chat(db, "C@s.whatsapp.net", "Charlie")
        now = datetime(2026, 4, 17, 15, 0)
        for i in range(3):
            _insert_msg(db, mid=f"c{i}", jid="C@s.whatsapp.net", sender="C@s.whatsapp.net",
                        content=f"old {i}", ts=now - timedelta(minutes=60 - i * 5))
        monkeypatch.setattr(rag, "WHATSAPP_DB_PATH", db)
        # All ids already processed → chat dropped (no new inbound).
        out = rag._fetch_whatsapp_window(
            now - timedelta(hours=2), now, {"c0", "c1", "c2"},
        )
        assert out == []

    def test_drops_unnamed_contacts(self, tmp_path, monkeypatch):
        db = tmp_path / "wa.db"
        _build_fake_wa_db(db)
        # `chats.name` is the raw phone number (real-world case: unresolved
        # push-name). Should be dropped to match morning brief behavior.
        _insert_chat(db, "5491199999999@s.whatsapp.net", "5491199999999")
        _insert_chat(db, "Named@s.whatsapp.net", "Luis")
        now = datetime(2026, 4, 17, 15, 0)
        for i in range(3):
            _insert_msg(db, mid=f"u{i}", jid="5491199999999@s.whatsapp.net",
                        sender="5491199999999@s.whatsapp.net", content=f"u{i}",
                        ts=now - timedelta(minutes=30 - i))
            _insert_msg(db, mid=f"n{i}", jid="Named@s.whatsapp.net",
                        sender="Named@s.whatsapp.net", content=f"n{i}",
                        ts=now - timedelta(minutes=20 - i))
        monkeypatch.setattr(rag, "WHATSAPP_DB_PATH", db)
        out = rag._fetch_whatsapp_window(now - timedelta(hours=2), now, set())
        assert {e["label"] for e in out} == {"Luis"}


class TestWriteNote:
    def test_empty_extractions_skips_write(self, tmp_path):
        run_ts = datetime(2026, 4, 17, 15, 30)
        by_chat = [{"jid": "A@g.us", "label": "Grupo A", "messages": []}]
        extractions = [{"tasks": [], "questions": [], "commitments": []}]
        path, created, n = rag._wa_tasks_write_note(
            tmp_path, run_ts, by_chat, extractions,
        )
        assert n == 0
        assert created is False
        assert not path.exists()

    def test_creates_file_on_first_run(self, tmp_path):
        (tmp_path / "00-Inbox").mkdir()
        run_ts = datetime(2026, 4, 17, 15, 30)
        by_chat = [{
            "jid": "A@g.us", "label": "Grupo A", "is_group": True,
            "messages": [
                {"ts": "2026-04-17 15:20:00", "who": "x", "text": "x",
                 "is_from_me": False, "new": True, "id": "a0"},
            ],
            "new_ids": ["a0"], "inbound": 1,
        }]
        extractions = [{
            "tasks": ["comprar pan"],
            "questions": ["¿vas a venir?"],
            "commitments": ["mañana te mando el link"],
        }]
        path, created, n = rag._wa_tasks_write_note(
            tmp_path, run_ts, by_chat, extractions,
        )
        assert created is True
        assert n == 3
        body = path.read_text(encoding="utf-8")
        assert "type: wa-tasks" in body
        assert "ambient: skip" in body
        assert "date: 2026-04-17" in body
        assert "## 15:30" in body
        assert "[[03-Resources/WhatsApp/Grupo A/2026-04|Grupo A]]" in body
        assert "- [ ] comprar pan" in body
        assert "- ❓ ¿vas a venir?" in body
        assert "- 📌 mañana te mando el link" in body

    def test_appends_new_section_on_same_day(self, tmp_path):
        (tmp_path / "00-Inbox").mkdir()
        t1 = datetime(2026, 4, 17, 10, 0)
        t2 = datetime(2026, 4, 17, 14, 30)
        chat = {
            "jid": "A@g.us", "label": "Grupo A", "is_group": True,
            "messages": [{"ts": "2026-04-17 09:55:00", "who": "x", "text": "x",
                          "is_from_me": False, "new": True, "id": "a0"}],
            "new_ids": ["a0"], "inbound": 1,
        }
        rag._wa_tasks_write_note(tmp_path, t1, [chat], [
            {"tasks": ["primera"], "questions": [], "commitments": []},
        ])
        path, created, n = rag._wa_tasks_write_note(tmp_path, t2, [chat], [
            {"tasks": ["segunda"], "questions": [], "commitments": []},
        ])
        assert created is False
        assert n == 1
        body = path.read_text(encoding="utf-8")
        assert body.count("## 10:00") == 1
        assert body.count("## 14:30") == 1
        assert "primera" in body
        assert "segunda" in body
        # Frontmatter not duplicated.
        assert body.count("type: wa-tasks") == 1


class TestMonthLink:
    def test_basic_alpha_label(self):
        link = rag._wa_chat_month_link(
            "A@g.us", "Grupo A", "2026-04-17 15:00:00",
        )
        assert link == "[[03-Resources/WhatsApp/Grupo A/2026-04|Grupo A]]"

    def test_falls_back_to_jid_prefix_when_label_numeric(self):
        link = rag._wa_chat_month_link(
            "5491199999999@s.whatsapp.net", "Contacto …9999",
            "2026-04-17 15:00:00",
        )
        # Slug strips the "…" — becomes "Contacto 9999".
        assert "Contacto 9999" in link
        assert "2026-04" in link

    def test_bad_timestamp_returns_bare_label(self):
        link = rag._wa_chat_month_link("A@g.us", "Grupo A", "bogus")
        assert link == "[[Grupo A]]"
