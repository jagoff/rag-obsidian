"""Tests para scripts/ingest_base.py (helpers del patrón HASH-based state).

Cubre las 5 primitives de state management + las 2 de bulk delete de
chunks de Chroma. Usa sqlite3 en memoria para aislar tests del
estado real del rag.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from scripts import ingest_base


# --- DDL generation ----------------------------------------------------------


def test_make_state_table_ddl_default_text_key():
    sql = ingest_base.make_state_table_ddl("reminders_state", "reminder_id")
    assert "CREATE TABLE IF NOT EXISTS reminders_state" in sql
    assert "reminder_id TEXT PRIMARY KEY" in sql
    assert "content_hash TEXT NOT NULL" in sql
    assert "last_seen_ts TEXT NOT NULL" in sql
    assert "updated_at TEXT NOT NULL" in sql


def test_make_state_table_ddl_integer_key():
    sql = ingest_base.make_state_table_ddl(
        "safari_history_state", "history_item_id", key_type="INTEGER"
    )
    assert "history_item_id INTEGER PRIMARY KEY" in sql


# --- ensure_state_table ------------------------------------------------------


def test_ensure_state_table_creates_then_noop():
    conn = sqlite3.connect(":memory:")
    # First call creates
    ingest_base.ensure_state_table(conn, "calls_state", "call_uid")
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='calls_state'"
    ).fetchall()
    assert len(rows) == 1

    # Second call is no-op (IF NOT EXISTS)
    ingest_base.ensure_state_table(conn, "calls_state", "call_uid")
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='calls_state'"
    ).fetchall()
    assert len(rows) == 1


# --- load_hashes -------------------------------------------------------------


def test_load_hashes_empty():
    conn = sqlite3.connect(":memory:")
    ingest_base.ensure_state_table(conn, "contacts_state", "contact_uid")
    assert ingest_base.load_hashes(conn, "contacts_state", "contact_uid") == {}


def test_load_hashes_with_rows():
    conn = sqlite3.connect(":memory:")
    ingest_base.ensure_state_table(conn, "contacts_state", "contact_uid")
    for uid, h in [("a1", "h1"), ("b2", "h2"), ("c3", "h3")]:
        ingest_base.upsert_hash(
            conn, "contacts_state", "contact_uid", uid, h, now_iso="2026-04-25T10:00:00"
        )
    hashes = ingest_base.load_hashes(conn, "contacts_state", "contact_uid")
    assert hashes == {"a1": "h1", "b2": "h2", "c3": "h3"}


def test_load_hashes_integer_key_coerced_to_str():
    """Safari usa PK INTEGER — load_hashes debe devolver keys como str."""
    conn = sqlite3.connect(":memory:")
    ingest_base.ensure_state_table(
        conn, "safari_hist_state", "history_item_id", key_type="INTEGER"
    )
    ingest_base.upsert_hash(
        conn, "safari_hist_state", "history_item_id", 42, "hX", now_iso="2026-04-25T10:00:00"
    )
    hashes = ingest_base.load_hashes(conn, "safari_hist_state", "history_item_id")
    assert hashes == {"42": "hX"}


# --- upsert_hash -------------------------------------------------------------


def test_upsert_hash_inserts_then_updates():
    conn = sqlite3.connect(":memory:")
    ingest_base.ensure_state_table(conn, "reminders_state", "reminder_id")

    ingest_base.upsert_hash(
        conn, "reminders_state", "reminder_id", "r1", "hash_v1",
        now_iso="2026-04-25T10:00:00",
    )
    ingest_base.upsert_hash(
        conn, "reminders_state", "reminder_id", "r1", "hash_v2",
        now_iso="2026-04-25T11:00:00",
    )

    rows = conn.execute(
        "SELECT reminder_id, content_hash, last_seen_ts, updated_at FROM reminders_state"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0] == ("r1", "hash_v2", "2026-04-25T11:00:00", "2026-04-25T11:00:00")


def test_upsert_hash_default_timestamp_used_when_none():
    """Sin now_iso, usa datetime.now() → last_seen_ts no está vacío."""
    conn = sqlite3.connect(":memory:")
    ingest_base.ensure_state_table(conn, "reminders_state", "reminder_id")
    ingest_base.upsert_hash(conn, "reminders_state", "reminder_id", "r1", "h1")
    row = conn.execute(
        "SELECT last_seen_ts, updated_at FROM reminders_state WHERE reminder_id='r1'"
    ).fetchone()
    assert row[0] and row[1]  # non-empty ISO strings
    assert row[0] == row[1]  # mismo timestamp para ambos


# --- delete_hash -------------------------------------------------------------


def test_delete_hash_removes_row():
    conn = sqlite3.connect(":memory:")
    ingest_base.ensure_state_table(conn, "reminders_state", "reminder_id")
    ingest_base.upsert_hash(
        conn, "reminders_state", "reminder_id", "r1", "h1", now_iso="2026-04-25T10:00:00"
    )
    ingest_base.upsert_hash(
        conn, "reminders_state", "reminder_id", "r2", "h2", now_iso="2026-04-25T10:00:00"
    )

    ingest_base.delete_hash(conn, "reminders_state", "reminder_id", "r1")
    remaining = ingest_base.load_hashes(conn, "reminders_state", "reminder_id")
    assert remaining == {"r2": "h2"}


def test_delete_hash_silent_when_missing():
    conn = sqlite3.connect(":memory:")
    ingest_base.ensure_state_table(conn, "reminders_state", "reminder_id")
    # No raise, no-op
    ingest_base.delete_hash(conn, "reminders_state", "reminder_id", "nonexistent")


# --- delete_chunks_by_file_key ----------------------------------------------


def test_delete_chunks_by_file_key_returns_zero_on_missing():
    col = MagicMock()
    col.get.return_value = {"ids": []}
    n = ingest_base.delete_chunks_by_file_key(col, "reminders://missing")
    assert n == 0
    col.delete.assert_not_called()


def test_delete_chunks_by_file_key_removes_and_counts():
    col = MagicMock()
    col.get.return_value = {"ids": ["chunk_a", "chunk_b", "chunk_c"]}
    n = ingest_base.delete_chunks_by_file_key(col, "reminders://r1")
    assert n == 3
    col.get.assert_called_once_with(where={"file": "reminders://r1"}, include=[])
    col.delete.assert_called_once_with(ids=["chunk_a", "chunk_b", "chunk_c"])


def test_delete_chunks_by_file_key_silent_on_exception():
    col = MagicMock()
    col.get.side_effect = RuntimeError("collection down")
    # No raise — returns 0 para no abortar el batch del caller
    assert ingest_base.delete_chunks_by_file_key(col, "reminders://r1") == 0


# --- delete_chunks_by_file_keys (bulk) --------------------------------------


def test_delete_chunks_by_file_keys_sums_across_keys():
    col = MagicMock()
    # Cada .get() devuelve un set distinto de ids
    col.get.side_effect = [
        {"ids": ["a1", "a2"]},
        {"ids": ["b1"]},
        {"ids": []},
    ]
    total = ingest_base.delete_chunks_by_file_keys(
        col, ["reminders://r1", "reminders://r2", "reminders://r3"]
    )
    assert total == 3
    assert col.get.call_count == 3
    assert col.delete.call_count == 2  # solo los que tenían ids


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-v"])
