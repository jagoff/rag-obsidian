"""Tests for entity graph SQL schema (Improvement #2, Fase A).

Verifica que rag_entities + rag_entity_mentions se crean correctamente en
_TELEMETRY_DDL con sus constraints + índices. No testea extracción aún —
esa es Fase B (helpers _extract_entities_batch, _upsert_entities).
"""
from __future__ import annotations

import json
import sqlite3
import time

import pytest

import rag


@pytest.fixture
def conn():
    """In-memory SQLite with telemetry DDL applied + FK enforcement on."""
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON")
    rag._ensure_telemetry_tables(c)
    yield c
    c.close()


def test_rag_entities_table_exists(conn):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='rag_entities'"
    ).fetchone()
    assert row is not None


def test_rag_entity_mentions_table_exists(conn):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='rag_entity_mentions'"
    ).fetchone()
    assert row is not None


def test_rag_entities_columns(conn):
    cols = {
        row[1]: row[2]
        for row in conn.execute("PRAGMA table_info(rag_entities)")
    }
    assert set(cols.keys()) == {
        "id", "canonical_name", "normalized", "entity_type", "aliases",
        "first_seen_ts", "last_seen_ts", "mention_count", "confidence", "extra_json",
    }
    assert cols["canonical_name"] == "TEXT"
    assert cols["normalized"] == "TEXT"
    assert cols["entity_type"] == "TEXT"


def test_rag_entity_mentions_columns(conn):
    cols = {
        row[1]: row[2]
        for row in conn.execute("PRAGMA table_info(rag_entity_mentions)")
    }
    assert set(cols.keys()) == {
        "id", "entity_id", "chunk_id", "source", "ts", "snippet", "confidence",
    }


def test_rag_entities_unique_normalized_type(conn):
    """UNIQUE(normalized, entity_type) prevents duplicates for same identity."""
    conn.execute(
        "INSERT INTO rag_entities (canonical_name, normalized, entity_type, confidence) "
        "VALUES (?, ?, ?, ?)",
        ("Juan Pérez", "juan perez", "person", 0.95),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO rag_entities (canonical_name, normalized, entity_type, confidence) "
            "VALUES (?, ?, ?, ?)",
            ("Juan P.", "juan perez", "person", 0.90),
        )


def test_rag_entities_different_types_allowed(conn):
    """Same normalized string, different entity_type → both allowed."""
    conn.execute(
        "INSERT INTO rag_entities (canonical_name, normalized, entity_type, confidence) "
        "VALUES (?, ?, ?, ?)",
        ("Juan", "juan", "person", 0.95),
    )
    conn.execute(
        "INSERT INTO rag_entities (canonical_name, normalized, entity_type, confidence) "
        "VALUES (?, ?, ?, ?)",
        ("Juan", "juan", "organization", 0.85),
    )
    count = conn.execute("SELECT COUNT(*) FROM rag_entities").fetchone()[0]
    assert count == 2


def test_rag_entity_mentions_fk_cascade_delete(conn):
    """DELETE entity → mentions borradas automáticamente."""
    entity_id = conn.execute(
        "INSERT INTO rag_entities (canonical_name, normalized, entity_type, confidence) "
        "VALUES (?, ?, ?, ?)",
        ("Juan", "juan", "person", 0.95),
    ).lastrowid
    conn.execute(
        "INSERT INTO rag_entity_mentions (entity_id, chunk_id, source, ts) "
        "VALUES (?, ?, ?, ?)",
        (entity_id, "vault://test.md::0", "vault", time.time()),
    )
    conn.commit()

    count_before = conn.execute("SELECT COUNT(*) FROM rag_entity_mentions").fetchone()[0]
    assert count_before == 1

    conn.execute("DELETE FROM rag_entities WHERE id=?", (entity_id,))
    conn.commit()

    count_after = conn.execute("SELECT COUNT(*) FROM rag_entity_mentions").fetchone()[0]
    assert count_after == 0


def test_rag_entity_mentions_unique_entity_chunk(conn):
    """UNIQUE(entity_id, chunk_id) prevents duplicate mentions in same chunk."""
    entity_id = conn.execute(
        "INSERT INTO rag_entities (canonical_name, normalized, entity_type, confidence) "
        "VALUES (?, ?, ?, ?)",
        ("Juan", "juan", "person", 0.95),
    ).lastrowid
    conn.execute(
        "INSERT INTO rag_entity_mentions (entity_id, chunk_id, source) "
        "VALUES (?, ?, ?)",
        (entity_id, "chunk-1", "vault"),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO rag_entity_mentions (entity_id, chunk_id, source) "
            "VALUES (?, ?, ?)",
            (entity_id, "chunk-1", "vault"),
        )


def test_rag_entity_mentions_different_chunks_allowed(conn):
    """Same entity, different chunks → both allowed."""
    entity_id = conn.execute(
        "INSERT INTO rag_entities (canonical_name, normalized, entity_type, confidence) "
        "VALUES (?, ?, ?, ?)",
        ("Juan", "juan", "person", 0.95),
    ).lastrowid
    for cid in ("chunk-1", "chunk-2", "chunk-3"):
        conn.execute(
            "INSERT INTO rag_entity_mentions (entity_id, chunk_id, source) "
            "VALUES (?, ?, ?)",
            (entity_id, cid, "vault"),
        )
    count = conn.execute("SELECT COUNT(*) FROM rag_entity_mentions").fetchone()[0]
    assert count == 3


def test_rag_entities_indices_exist(conn):
    indices = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='rag_entities'"
        )
    }
    assert "idx_entities_normalized" in indices
    assert "idx_entities_type" in indices
    assert "idx_entities_canonical" in indices


def test_rag_entity_mentions_indices_exist(conn):
    indices = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='rag_entity_mentions'"
        )
    }
    assert "idx_mentions_entity" in indices
    assert "idx_mentions_chunk" in indices
    assert "idx_mentions_source" in indices
    assert "idx_mentions_ts" in indices
    # Compound index critical for handle_entity_lookup() hot path
    assert "idx_mentions_entity_ts" in indices


def test_telemetry_ddl_registers_entities_tables():
    """rag_entities + rag_entity_mentions must appear in _TELEMETRY_DDL."""
    table_names = {name for name, _stmts in rag._TELEMETRY_DDL}
    assert "rag_entities" in table_names
    assert "rag_entity_mentions" in table_names


def test_aliases_stored_as_json(conn):
    """aliases column stores JSON — store + retrieve round-trip works."""
    aliases = ["Juan", "JP", "Juancito"]
    conn.execute(
        "INSERT INTO rag_entities (canonical_name, normalized, entity_type, aliases, confidence) "
        "VALUES (?, ?, ?, ?, ?)",
        ("Juan Pérez", "juan perez", "person", json.dumps(aliases), 0.92),
    )
    row = conn.execute(
        "SELECT aliases FROM rag_entities WHERE normalized=?",
        ("juan perez",),
    ).fetchone()
    assert json.loads(row[0]) == aliases
