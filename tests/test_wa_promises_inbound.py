"""Tests para `_persist_inbound_promise_if_match` (tail.py).

Cubre el hook que tail.py dispara cuando un msg inbound matchea el regex
hint rioplatense → INSERT row en rag_promises con direction='inbound'.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def isolated_db(tmp_path):
    import rag as _rag
    orig = _rag.DB_PATH
    try:
        _rag.DB_PATH = Path(tmp_path)
        from rag import _ragvec_state_conn
        with _ragvec_state_conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rag_promises ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " ts TEXT NOT NULL,"
                " contact_jid TEXT NOT NULL,"
                " contact_name TEXT,"
                " promise_text TEXT NOT NULL,"
                " direction TEXT NOT NULL,"
                " due_ts TEXT,"
                " due_confidence REAL,"
                " source_msg_id TEXT,"
                " source_chat_jid TEXT,"
                " status TEXT NOT NULL DEFAULT 'pending',"
                " reminder_sent_ts TEXT,"
                " closed_ts TEXT,"
                " closed_reason TEXT,"
                " extra_json TEXT)"
            )
            conn.commit()
        yield tmp_path
    finally:
        _rag.DB_PATH = orig


def _count_rows(direction=None, jid=None):
    from rag import _ragvec_state_conn
    with _ragvec_state_conn() as conn:
        where = []
        params = []
        if direction:
            where.append("direction = ?")
            params.append(direction)
        if jid:
            where.append("contact_jid = ?")
            params.append(jid)
        sql = "SELECT COUNT(*) FROM rag_promises"
        if where:
            sql += " WHERE " + " AND ".join(where)
        return conn.execute(sql, params).fetchone()[0]


def test_inbound_with_hint_inserts(isolated_db):
    from rag.integrations.whatsapp.tail import _persist_inbound_promise_if_match
    _persist_inbound_promise_if_match("x@y", "ya te aviso esta tarde", "msg-1")
    assert _count_rows(direction="inbound") == 1


def test_inbound_without_hint_skips(isolated_db):
    from rag.integrations.whatsapp.tail import _persist_inbound_promise_if_match
    _persist_inbound_promise_if_match("x@y", "hola qué tal", "msg-2")
    assert _count_rows(direction="inbound") == 0


def test_inbound_empty_content_skips(isolated_db):
    from rag.integrations.whatsapp.tail import _persist_inbound_promise_if_match
    _persist_inbound_promise_if_match("x@y", "", "msg-3")
    _persist_inbound_promise_if_match("x@y", None, "msg-4")
    assert _count_rows(direction="inbound") == 0


def test_inbound_dedupe_by_msg_id(isolated_db):
    """Mismo msg_id no se inserta dos veces."""
    from rag.integrations.whatsapp.tail import _persist_inbound_promise_if_match
    _persist_inbound_promise_if_match("x@y", "te paso el deck mañana", "msg-5")
    _persist_inbound_promise_if_match("x@y", "te paso el deck mañana", "msg-5")
    assert _count_rows(direction="inbound") == 1


def test_inbound_different_msg_id_allows_both(isolated_db):
    """msg_ids distintos → 2 rows aunque texto sea igual."""
    from rag.integrations.whatsapp.tail import _persist_inbound_promise_if_match
    _persist_inbound_promise_if_match("x@y", "te paso el deck mañana", "msg-A")
    _persist_inbound_promise_if_match("x@y", "te paso el deck mañana", "msg-B")
    assert _count_rows(direction="inbound") == 2


def test_inbound_no_msg_id_skips_dedupe(isolated_db):
    """Sin msg_id, ambos inserts pasan (no podemos dedupear sin key)."""
    from rag.integrations.whatsapp.tail import _persist_inbound_promise_if_match
    _persist_inbound_promise_if_match("x@y", "te paso el deck mañana", None)
    _persist_inbound_promise_if_match("x@y", "te paso el deck mañana", None)
    assert _count_rows(direction="inbound") == 2
