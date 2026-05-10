"""Tests para `_wa_promises_persist` — writer que persiste promesas
extraídas a `rag_promises`.

Cubrimos:

- INSERT correcto con direction normalize ("out"→"outgoing").
- Idempotencia: dos runs del mismo (msg_id, text) NO duplican.
- Skip de items malformados (texto vacío, direction inválido).
- Silent-fail si tabla no existe.
- Cuenta de retorno coincide con rows insertadas.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rag.integrations.whatsapp.tasks_writer import _wa_promises_persist


_DDL = """
CREATE TABLE IF NOT EXISTS rag_promises (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  contact_jid TEXT NOT NULL,
  contact_name TEXT,
  promise_text TEXT NOT NULL,
  direction TEXT NOT NULL,
  due_ts TEXT,
  due_confidence REAL,
  source_msg_id TEXT,
  source_chat_jid TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  reminder_sent_ts TEXT,
  closed_ts TEXT,
  closed_reason TEXT,
  extra_json TEXT
);
"""


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch):
    """DB temporal con `rag_promises` DDL aplicada + monkeypatch de
    `_ragvec_state_conn` para apuntar a este path."""
    db = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_DDL)
    conn.commit()
    conn.close()

    import contextlib

    @contextlib.contextmanager
    def _conn_factory():
        c = sqlite3.connect(str(db), timeout=5.0)
        try:
            yield c
            c.commit()
        finally:
            c.close()

    import rag

    monkeypatch.setattr(rag, "_ragvec_state_conn", _conn_factory)
    return db


def _row_count(db: Path, where_extra: str = "") -> int:
    conn = sqlite3.connect(str(db))
    try:
        clause = f" WHERE {where_extra}" if where_extra else ""
        return conn.execute(f"SELECT COUNT(*) FROM rag_promises{clause}").fetchone()[0]
    finally:
        conn.close()


def _by_chat_one(jid: str = "5491111@s.whatsapp.net", label: str = "Maru"):
    return [{"jid": jid, "label": label, "messages": [], "is_group": False, "inbound": 1, "new_ids": []}]


def test_persist_basic_outgoing_promise(tmp_db):
    """Caso happy-path: una promesa outgoing entra como 'outgoing'."""
    now = datetime(2026, 5, 10, 10, 0, 0)
    extractions = [{
        "tasks": [], "questions": [], "commitments": [],
        "promises": [{
            "text": "te paso el PDF",
            "when_text": "mañana",
            "direction": "out",  # extractor LLM emite "out"/"in"
            "msg_id": "m1",
            "msg_ts": "2026-05-10T09:55:00",
            "speaker": "yo",
        }],
    }]
    n = _wa_promises_persist(_by_chat_one(), extractions, now)
    assert n == 1, f"expected 1 row inserted, got {n}"

    conn = sqlite3.connect(str(tmp_db))
    try:
        row = conn.execute(
            "SELECT direction, contact_jid, contact_name, promise_text, "
            "source_msg_id, status, due_ts, due_confidence, extra_json "
            "FROM rag_promises"
        ).fetchone()
    finally:
        conn.close()
    direction, jid, name, text, msg_id, status, due, conf, extra = row
    assert direction == "outgoing"
    assert jid == "5491111@s.whatsapp.net"
    assert name == "Maru"
    assert text == "te paso el PDF"
    assert msg_id == "m1"
    assert status == "pending"
    assert due  # due parsed from "mañana"
    assert conf == pytest.approx(0.9, abs=0.01)
    parsed = json.loads(extra)
    assert parsed["when_text"] == "mañana"
    assert parsed["chat_label"] == "Maru"


def test_persist_normalizes_in_to_incoming(tmp_db):
    """`direction='in'` debe normalizar a `'incoming'`."""
    now = datetime(2026, 5, 10, 10, 0, 0)
    extractions = [{
        "tasks": [], "questions": [], "commitments": [],
        "promises": [{
            "text": "te llamo en un rato",
            "when_text": "en 1h",
            "direction": "in",
            "msg_id": "m2",
            "msg_ts": "2026-05-10T09:55:00",
            "speaker": "Maru",
        }],
    }]
    n = _wa_promises_persist(_by_chat_one(), extractions, now)
    assert n == 1
    conn = sqlite3.connect(str(tmp_db))
    try:
        direction = conn.execute("SELECT direction FROM rag_promises").fetchone()[0]
    finally:
        conn.close()
    assert direction == "incoming"


def test_idempotent_dedup_by_msg_and_text(tmp_db):
    """Dos calls con mismo (msg_id, promise_text) → no duplican."""
    now = datetime(2026, 5, 10, 10, 0, 0)
    promise = {
        "text": "te paso el deck",
        "when_text": "mañana",
        "direction": "out",
        "msg_id": "m3",
        "msg_ts": "2026-05-10T09:55:00",
        "speaker": "yo",
    }
    extractions = [{"tasks": [], "questions": [], "commitments": [], "promises": [promise]}]

    n1 = _wa_promises_persist(_by_chat_one(), extractions, now)
    n2 = _wa_promises_persist(_by_chat_one(), extractions, now + timedelta(hours=1))
    assert n1 == 1
    assert n2 == 0  # dedup
    assert _row_count(tmp_db) == 1


def test_dedup_only_for_pending_status(tmp_db):
    """Si la promesa anterior fue cerrada (status != pending), una nueva
    con mismo (msg_id, text) DEBE poderse insertar — el dedup solo aplica
    a pendientes para evitar re-tracking de algo ya resuelto."""
    now = datetime(2026, 5, 10, 10, 0, 0)
    promise = {
        "text": "te paso el link",
        "when_text": "ahora",
        "direction": "out",
        "msg_id": "m4",
        "msg_ts": "2026-05-10T09:55:00",
        "speaker": "yo",
    }
    extractions = [{"tasks": [], "questions": [], "commitments": [], "promises": [promise]}]

    n1 = _wa_promises_persist(_by_chat_one(), extractions, now)
    assert n1 == 1
    # cerrar la promesa
    conn = sqlite3.connect(str(tmp_db))
    try:
        conn.execute("UPDATE rag_promises SET status='closed', closed_reason='done'")
        conn.commit()
    finally:
        conn.close()
    # nueva run debería re-insertar (porque la previa ya no está pending)
    n2 = _wa_promises_persist(_by_chat_one(), extractions, now + timedelta(days=1))
    assert n2 == 1
    assert _row_count(tmp_db) == 2


def test_skip_invalid_direction(tmp_db):
    """`direction` que no sea 'out'/'in'/'outgoing'/'incoming' → skip."""
    now = datetime(2026, 5, 10, 10, 0, 0)
    extractions = [{
        "tasks": [], "questions": [], "commitments": [],
        "promises": [
            {"text": "valid", "direction": "out", "msg_id": "ok", "when_text": "", "msg_ts": ""},
            {"text": "bad", "direction": "lateral", "msg_id": "bad", "when_text": "", "msg_ts": ""},
            {"text": "empty", "direction": "", "msg_id": "empty", "when_text": "", "msg_ts": ""},
        ],
    }]
    n = _wa_promises_persist(_by_chat_one(), extractions, now)
    assert n == 1, "solo el item con direction='out' debería persistirse"


def test_skip_empty_text(tmp_db):
    """Promesa con text vacío → skip."""
    now = datetime(2026, 5, 10, 10, 0, 0)
    extractions = [{
        "tasks": [], "questions": [], "commitments": [],
        "promises": [
            {"text": "", "direction": "out", "msg_id": "x", "when_text": "", "msg_ts": ""},
            {"text": "   ", "direction": "out", "msg_id": "y", "when_text": "", "msg_ts": ""},
        ],
    }]
    n = _wa_promises_persist(_by_chat_one(), extractions, now)
    assert n == 0


def test_no_promises_returns_zero(tmp_db):
    """extractions sin promises[] no rompe."""
    now = datetime(2026, 5, 10, 10, 0, 0)
    extractions = [{"tasks": ["t1"], "questions": [], "commitments": [], "promises": []}]
    n = _wa_promises_persist(_by_chat_one(), extractions, now)
    assert n == 0


def test_silent_fail_on_missing_table(tmp_path, monkeypatch):
    """Si la tabla no existe, el writer NO crashea — devuelve 0."""
    db = tmp_path / "tel.db"
    sqlite3.connect(str(db)).close()  # DB existe pero sin schema

    import contextlib

    @contextlib.contextmanager
    def _conn_factory():
        c = sqlite3.connect(str(db), timeout=5.0)
        try:
            yield c
            c.commit()
        finally:
            c.close()

    import rag

    monkeypatch.setattr(rag, "_ragvec_state_conn", _conn_factory)
    extractions = [{
        "tasks": [], "questions": [], "commitments": [],
        "promises": [{
            "text": "X",
            "when_text": "",
            "direction": "out",
            "msg_id": "m1",
            "msg_ts": "",
            "speaker": "yo",
        }],
    }]
    n = _wa_promises_persist(_by_chat_one(), extractions, datetime(2026, 5, 10, 10, 0, 0))
    assert n == 0  # silent-fail
