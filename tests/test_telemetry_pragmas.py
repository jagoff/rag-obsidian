"""Tests para los PRAGMAs aplicados a telemetry.db cada vez que se abre
una connection (audit 2026-04-25 R2-7 #5).

`PRAGMA foreign_keys=ON` es per-connection en SQLite y por default está
en OFF. Sin él, los `REFERENCES ... ON DELETE CASCADE` declarados en el
schema NO se enforcan — borrar una entity dejaba rows huérfanas en
`rag_entity_mentions` indefinidamente (hasta el prune cada 30 días).

Tests cubren:
- PRAGMA foreign_keys=ON está activo en conns frescas
- PRAGMA foreign_keys=ON está activo cuando el DDL ya fue ensureado
  (caché en `_TELEMETRY_DDL_ENSURED_PATHS`)
- CASCADE delete realmente funciona (test integration)
"""
from __future__ import annotations

import sqlite3

import pytest

import rag


def _open_test_telemetry(tmp_path):
    """Helper que abre una conn fresca a una telemetry.db en tmp_path
    y le aplica las DDLs + PRAGMAs."""
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None,
                           check_same_thread=False, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    rag._ensure_telemetry_tables(conn)
    return conn, db_path


def test_foreign_keys_pragma_is_on_after_ensure(tmp_path):
    """Después de _ensure_telemetry_tables, foreign_keys debe estar ON."""
    conn, _ = _open_test_telemetry(tmp_path)
    try:
        cur = conn.execute("PRAGMA foreign_keys")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == 1, "foreign_keys debe estar ON (1), got 0 (OFF)"
    finally:
        conn.close()


def test_foreign_keys_pragma_applied_per_connection(tmp_path):
    """SQLite resetea foreign_keys=OFF cada vez que abrís conn nueva.
    El PRAGMA debe aplicarse en CADA llamada a _ensure_telemetry_tables,
    no solo la primera (cuando se ensurea el DDL inicial)."""
    db_path = tmp_path / "telemetry.db"

    # 1ra conn: ensurea DDL + PRAGMA
    conn1 = sqlite3.connect(str(db_path), isolation_level=None,
                            check_same_thread=False, timeout=5.0)
    conn1.execute("PRAGMA journal_mode=WAL")
    rag._ensure_telemetry_tables(conn1)
    assert conn1.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    conn1.close()

    # 2da conn: ya hay cache de DDL ensured, pero el PRAGMA tiene que
    # seguir aplicándose porque es per-conn.
    conn2 = sqlite3.connect(str(db_path), isolation_level=None,
                            check_same_thread=False, timeout=5.0)
    conn2.execute("PRAGMA journal_mode=WAL")
    rag._ensure_telemetry_tables(conn2)
    assert conn2.execute("PRAGMA foreign_keys").fetchone()[0] == 1, (
        "2da conn no aplicó foreign_keys=ON — el PRAGMA está dentro del "
        "if-cached-paths return, debe estar antes"
    )
    conn2.close()


def test_cascade_delete_actually_works(tmp_path):
    """Test integration: con foreign_keys=ON, borrar una row de
    rag_entities debe cascadear y borrar las rows asociadas en
    rag_entity_mentions. Pre-fix esto NO pasaba — el CASCADE se
    declaraba en el schema pero SQLite lo ignoraba."""
    conn, _ = _open_test_telemetry(tmp_path)
    try:
        # Insertar entity y mention. Columnas reales del schema:
        # rag_entities: canonical_name, normalized, entity_type, aliases
        # rag_entity_mentions: entity_id, chunk_id, source, ts, snippet
        conn.execute(
            "INSERT INTO rag_entities "
            "(canonical_name, normalized, entity_type, aliases) "
            "VALUES ('Grecia', 'grecia', 'person', '[]')"
        )
        entity_id = conn.execute(
            "SELECT id FROM rag_entities WHERE canonical_name = 'Grecia'"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO rag_entity_mentions "
            "(entity_id, chunk_id, source, ts, snippet) "
            "VALUES (?, ?, ?, ?, ?)",
            (entity_id, "chunk-abc", "vault", 1745596800.0, "test"),
        )

        # Verificar que la mention existe.
        n = conn.execute(
            "SELECT COUNT(*) FROM rag_entity_mentions WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()[0]
        assert n == 1

        # Borrar la entity → CASCADE debería borrar la mention.
        conn.execute("DELETE FROM rag_entities WHERE id = ?", (entity_id,))

        n = conn.execute(
            "SELECT COUNT(*) FROM rag_entity_mentions WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()[0]
        assert n == 0, (
            f"CASCADE no funcionó — la mention quedó huérfana después "
            f"de borrar la entity (n={n}, expected 0)"
        )
    finally:
        conn.close()
