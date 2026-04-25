"""Tests for handle_entity_lookup (Improvement #2 Fase B).

Covers: empty tables, matching chunks, temporal window filter, limit,
missing question.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager


import rag


def _make_conn_factory(populator):
    """Build a _ragvec_state_conn replacement that yields an in-memory DB
    after running `populator(conn)`. populator can be None for empty DB."""
    @contextmanager
    def factory():
        c = sqlite3.connect(":memory:")
        c.execute("PRAGMA foreign_keys = ON")
        rag._ensure_telemetry_tables(c)
        if populator is not None:
            populator(c)
            c.commit()
        try:
            yield c
        finally:
            c.close()
    return factory


def test_handle_no_question_returns_empty():
    result = rag.handle_entity_lookup(col=None, params={}, question=None)
    assert result == []


def test_handle_empty_question_returns_empty():
    result = rag.handle_entity_lookup(col=None, params={}, question="")
    assert result == []


def test_handle_empty_entities_table_returns_empty(monkeypatch):
    """Tables con 0 entities → fallback (empty list)."""
    factory = _make_conn_factory(None)  # no populator
    monkeypatch.setattr(rag, "_ragvec_state_conn", factory)
    # _load_corpus no debería llamarse
    monkeypatch.setattr(rag, "_load_corpus", lambda col: {"metas": []})

    result = rag.handle_entity_lookup(col=None, params={}, question="qué dice juan sobre ops")
    assert result == []


def test_handle_entity_not_resolvable_returns_empty(monkeypatch):
    """Entity resolve devuelve None → lista vacía."""
    def populate(c):
        c.execute(
            "INSERT INTO rag_entities (id, canonical_name, normalized, entity_type) "
            "VALUES (1, 'Juan', 'juan', 'person')"
        )
    factory = _make_conn_factory(populate)
    monkeypatch.setattr(rag, "_ragvec_state_conn", factory)
    monkeypatch.setattr(rag, "_load_corpus", lambda col: {"metas": []})

    # Query para entidad distinta → no match
    result = rag.handle_entity_lookup(col=None, params={}, question="qué dice xyznonexistent sobre ops")
    assert result == []


def test_handle_returns_matching_chunks_sorted_by_ts_desc(monkeypatch):
    """Entity resolvible + mentions + corpus → metas sorted por ts desc."""
    def populate(c):
        c.execute(
            "INSERT INTO rag_entities (id, canonical_name, normalized, entity_type) "
            "VALUES (1, 'Juan', 'juan', 'person')"
        )
        c.execute(
            "INSERT INTO rag_entity_mentions (entity_id, chunk_id, source, ts) "
            "VALUES (1, 'chunk-1', 'vault', 1700000000.0)"
        )
        c.execute(
            "INSERT INTO rag_entity_mentions (entity_id, chunk_id, source, ts) "
            "VALUES (1, 'chunk-2', 'vault', 1700100000.0)"
        )
        c.execute(
            "INSERT INTO rag_entity_mentions (entity_id, chunk_id, source, ts) "
            "VALUES (1, 'chunk-3', 'vault', 1700200000.0)"
        )
    factory = _make_conn_factory(populate)
    monkeypatch.setattr(rag, "_ragvec_state_conn", factory)

    metas_mock = [
        {"file": "note-1.md", "chunk_id": "chunk-1", "note": "Note1", "source": "vault",
         "tags": "", "folder": ""},
        {"file": "note-2.md", "chunk_id": "chunk-2", "note": "Note2", "source": "vault",
         "tags": "", "folder": ""},
        {"file": "note-3.md", "chunk_id": "chunk-3", "note": "Note3", "source": "vault",
         "tags": "", "folder": ""},
    ]
    monkeypatch.setattr(rag, "_load_corpus", lambda col: {"metas": metas_mock})
    monkeypatch.setattr(rag, "_parse_agenda_window", lambda q: None)

    result = rag.handle_entity_lookup(col=None, params={}, question="qué dice juan sobre ops")
    assert len(result) == 3
    assert result[0]["note"] == "Note3"
    assert result[1]["note"] == "Note2"
    assert result[2]["note"] == "Note1"


def test_handle_respects_limit(monkeypatch):
    """limit=2 → top 2 metas."""
    def populate(c):
        c.execute(
            "INSERT INTO rag_entities (id, canonical_name, normalized, entity_type) "
            "VALUES (1, 'Juan', 'juan', 'person')"
        )
        for i in range(5):
            c.execute(
                "INSERT INTO rag_entity_mentions (entity_id, chunk_id, source, ts) "
                "VALUES (1, ?, 'vault', ?)",
                (f"chunk-{i}", 1700000000.0 + i * 1000),
            )
    factory = _make_conn_factory(populate)
    monkeypatch.setattr(rag, "_ragvec_state_conn", factory)

    metas_mock = [
        {"file": f"note-{i}.md", "chunk_id": f"chunk-{i}", "note": f"N{i}",
         "source": "vault", "tags": "", "folder": ""}
        for i in range(5)
    ]
    monkeypatch.setattr(rag, "_load_corpus", lambda col: {"metas": metas_mock})
    monkeypatch.setattr(rag, "_parse_agenda_window", lambda q: None)

    result = rag.handle_entity_lookup(col=None, params={}, limit=2, question="qué dice juan sobre ops")
    assert len(result) == 2


def test_handle_sql_failure_returns_empty(monkeypatch):
    """SQL error → empty list (silent log)."""
    @contextmanager
    def broken_factory():
        class BrokenConn:
            def execute(self, *a, **kw):
                raise sqlite3.OperationalError("simulated failure")
        yield BrokenConn()
    monkeypatch.setattr(rag, "_ragvec_state_conn", broken_factory)
    result = rag.handle_entity_lookup(col=None, params={}, question="qué dice juan sobre ops")
    assert result == []
