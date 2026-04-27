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


def test_migration_runs_on_pre_trace_id_db(tmp_path, monkeypatch):
    """Regression 2026-04-26: si el bootstrap encuentra una telemetry.db
    pre-2026-04-25 con `rag_queries` y `rag_behavior` SIN la columna
    ``trace_id``, debe migrarla idempotentemente al primer
    ``_ragvec_state_conn()``.

    Bug origen: el partial-index ``ix_rag_queries_trace_id ON
    rag_queries(trace_id) WHERE trace_id IS NOT NULL`` vivía dentro de
    ``_TELEMETRY_DDL`` y se ejecutaba ANTES del ALTER de
    ``_migrate_trace_id_columns``. Como ``CREATE TABLE IF NOT EXISTS``
    es no-op en tablas pre-existentes, la columna ``trace_id`` faltaba,
    el CREATE INDEX rompía el BEGIN/COMMIT entero, y el endpoint
    /transcripts y el persist de rag_status_samples explotaban con
    ``OperationalError: no such column: trace_id``.

    Fix: el partial-index vive ahora SOLO en
    ``_migrate_trace_id_columns``, que corre DESPUÉS del ALTER. Este test
    simula el escenario migración construyendo un schema "viejo" sin
    trace_id y verificando que ``_ragvec_state_conn()`` lo levante sin
    raisear, agregue la columna, y persista filas con trace_id.
    """
    # Crear DB con schema "viejo" (pre-trace_id): tablas con SOLO las
    # columnas que existían antes del commit 2b7c0c1 (audit ronda 2 batch).
    db_path = tmp_path / "telemetry.db"
    legacy = sqlite3.connect(str(db_path))
    try:
        legacy.executescript("""
            CREATE TABLE rag_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                cmd TEXT,
                q TEXT NOT NULL,
                session TEXT,
                mode TEXT,
                top_score REAL,
                t_retrieve REAL,
                t_gen REAL,
                answer_len INTEGER,
                citation_repaired INTEGER,
                critique_fired INTEGER,
                critique_changed INTEGER,
                variants_json TEXT,
                paths_json TEXT,
                scores_json TEXT,
                filters_json TEXT,
                bad_citations_json TEXT,
                extra_json TEXT
            );
            CREATE TABLE rag_behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                source TEXT NOT NULL,
                event TEXT NOT NULL,
                path TEXT,
                query TEXT,
                rank INTEGER,
                dwell_s REAL,
                extra_json TEXT
            );
        """)
        legacy.commit()
    finally:
        legacy.close()

    # Apuntar el módulo al tmp_path. Telemetry.db file ya está en su lugar
    # — el _ensure_telemetry_tables debería detectar las tablas viejas y
    # correr la migración idempotente.
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(rag, "_TELEMETRY_DB_FILENAME", "telemetry.db")
    # Limpiar el cache global que skipea el DDL on second open.
    rag._TELEMETRY_DDL_ENSURED_PATHS.clear()

    # Esto no debe raisear — pre-fix tiraba "no such column: trace_id"
    # al intentar el CREATE INDEX partial sobre la tabla legacy.
    with rag._ragvec_state_conn() as conn:
        cols_q = [r[1] for r in conn.execute("PRAGMA table_info(rag_queries)").fetchall()]
        cols_b = [r[1] for r in conn.execute("PRAGMA table_info(rag_behavior)").fetchall()]
        assert "trace_id" in cols_q, f"migración no agregó trace_id a rag_queries: {cols_q}"
        assert "trace_id" in cols_b, f"migración no agregó trace_id a rag_behavior: {cols_b}"

        # El partial-index también debe existir post-migración.
        idx_names = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name LIKE '%trace_id%'"
            ).fetchall()
        ]
        assert "ix_rag_queries_trace_id" in idx_names
        assert "ix_rag_behavior_trace_id" in idx_names

    # Y un INSERT con trace_id debe funcionar end-to-end.
    monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "0")
    tid = rag.generate_trace_id()
    rag.log_query_event({"cmd": "query", "q": "post-migración", "trace_id": tid})
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT trace_id, q FROM rag_queries WHERE trace_id = ?", (tid,),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0] == (tid, "post-migración")


def test_migration_adds_trace_id_to_rag_ambient(tmp_path, monkeypatch):
    """Regression 2026-04-27: rag_ambient faltaba en _migrate_trace_id_columns.
    29 errores ambient_sql_write_failed: no such column: trace_id registrados
    en sql_state_errors.jsonl del 2026-04-25 desde una DB pre-trace_id.

    Este test simula el escenario: DB con rag_ambient sin trace_id →
    _ragvec_state_conn() debe migrarla sin raisear, y _ambient_log_event
    con trace_id en el payload debe persistirlo como columna (no en payload_json).
    """
    db_path = tmp_path / "telemetry.db"
    legacy = sqlite3.connect(str(db_path))
    try:
        legacy.executescript("""
            CREATE TABLE rag_ambient (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                cmd TEXT,
                path TEXT,
                hash TEXT,
                payload_json TEXT
            );
        """)
        legacy.commit()
    finally:
        legacy.close()

    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(rag, "_TELEMETRY_DB_FILENAME", "telemetry.db")
    rag._TELEMETRY_DDL_ENSURED_PATHS.clear()

    with rag._ragvec_state_conn() as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(rag_ambient)").fetchall()]
        assert "trace_id" in cols, f"migración no agregó trace_id a rag_ambient: {cols}"

    monkeypatch.setenv("RAG_LOG_BEHAVIOR_ASYNC", "0")
    rag._ambient_log_event({
        "cmd": "test_ambient_event",
        "trace_id": "ab123456",
        "path": "test/note.md",
    })

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT trace_id, cmd, path FROM rag_ambient WHERE cmd = ?",
            ("test_ambient_event",),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "ab123456"
    assert rows[0][1] == "test_ambient_event"
    assert rows[0][2] == "test/note.md"
