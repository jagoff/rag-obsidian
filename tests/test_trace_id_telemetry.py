"""Tests para trace_id en telemetry (audit 2026-04-25 R2-Telemetry #1).

Sin trace_id, no se puede correlacionar "el chat tardó 30s ayer 4pm" con
los rows correspondientes en rag_behavior + silent_errors.jsonl. Con
trace_id, un grep cruza las 3 fuentes en O(1).

Cubrimos:
- generate_trace_id() devuelve formato esperado
- log_query_event acepta trace_id como kwarg y lo persiste
- log_behavior_event acepta trace_id y lo persiste
- Un mismo trace_id en query + behavior permite el JOIN para debug
"""
from __future__ import annotations

import re
import sqlite3

import pytest

import rag


def test_generate_trace_id_returns_8_hex_chars():
    """Trace IDs son 8 chars hex (32 bits, suficiente para 90 días retention)."""
    tid = rag.generate_trace_id()
    assert isinstance(tid, str)
    assert len(tid) == 8
    assert re.match(r"^[0-9a-f]{8}$", tid), f"esperaba hex, got {tid!r}"


def test_generate_trace_id_is_unique():
    """100 IDs consecutivos no deben colisionar (probabilidad astronómica)."""
    ids = {rag.generate_trace_id() for _ in range(100)}
    assert len(ids) == 100


def test_log_query_event_persists_trace_id(tmp_path, monkeypatch):
    """log_query_event con trace_id en el dict lo persiste a la columna."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "0")  # sync para test

    tid = rag.generate_trace_id()
    rag.log_query_event({
        "cmd": "query",
        "q": "test query con trace",
        "session": "test-sess",
        "trace_id": tid,
    })

    # Read back
    conn = sqlite3.connect(str(tmp_path / "telemetry.db"))
    try:
        rows = conn.execute(
            "SELECT trace_id, q FROM rag_queries WHERE trace_id = ?",
            (tid,),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0][0] == tid
    assert rows[0][1] == "test query con trace"


def test_log_behavior_event_persists_trace_id(tmp_path, monkeypatch):
    """log_behavior_event con trace_id lo persiste a la columna."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setenv("RAG_LOG_BEHAVIOR_ASYNC", "0")

    tid = rag.generate_trace_id()
    rag.log_behavior_event({
        "source": "web",
        "event": "open",
        "path": "test.md",
        "trace_id": tid,
    })

    conn = sqlite3.connect(str(tmp_path / "telemetry.db"))
    try:
        rows = conn.execute(
            "SELECT trace_id, path FROM rag_behavior WHERE trace_id = ?",
            (tid,),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0][0] == tid
    assert rows[0][1] == "test.md"


def test_trace_id_correlates_query_and_behavior(tmp_path, monkeypatch):
    """Caso de uso real: usar el mismo trace_id para query + behavior
    del mismo request permite JOIN para debug ('mostrame qué hizo el
    user en la query X')."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "0")
    monkeypatch.setenv("RAG_LOG_BEHAVIOR_ASYNC", "0")

    tid = rag.generate_trace_id()
    rag.log_query_event({
        "cmd": "chat", "q": "buscame X", "trace_id": tid,
    })
    rag.log_behavior_event({
        "source": "web", "event": "open", "path": "x.md", "trace_id": tid,
    })
    rag.log_behavior_event({
        "source": "web", "event": "kept", "path": "x.md", "trace_id": tid,
    })

    conn = sqlite3.connect(str(tmp_path / "telemetry.db"))
    try:
        # JOIN simulado: query + behaviors del mismo trace
        query_rows = conn.execute(
            "SELECT q FROM rag_queries WHERE trace_id = ?", (tid,),
        ).fetchall()
        behavior_rows = conn.execute(
            "SELECT event, path FROM rag_behavior WHERE trace_id = ? "
            "ORDER BY id",
            (tid,),
        ).fetchall()
    finally:
        conn.close()

    assert len(query_rows) == 1
    assert query_rows[0][0] == "buscame X"
    assert len(behavior_rows) == 2
    events = [r[0] for r in behavior_rows]
    assert "open" in events
    assert "kept" in events


def test_log_query_event_works_without_trace_id(tmp_path, monkeypatch):
    """Backward compat: callers pre-trace_id siguen funcionando, la
    columna queda NULL."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "0")

    rag.log_query_event({
        "cmd": "query",
        "q": "sin trace",
        # No trace_id
    })

    conn = sqlite3.connect(str(tmp_path / "telemetry.db"))
    try:
        rows = conn.execute(
            "SELECT trace_id, q FROM rag_queries WHERE q = ?",
            ("sin trace",),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0][0] is None  # trace_id NULL
    assert rows[0][1] == "sin trace"
