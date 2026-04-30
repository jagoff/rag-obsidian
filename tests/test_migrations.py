"""Tests for rag.migrations — versioned schema migrations.

Covers:
  1. DB virgen aplica migrations → version máxima registrada.
  2. DB con migrations parciales: aplica solo las nuevas.
  3. Migration que falla → SAVEPOINT rollback, version no se registra.
  4. Bootstrap en DB con rag_queries.trace_id existente: registra 1-4
     como aplicadas SIN re-ALTER (no tira `duplicate column`).
  5. Idempotencia: 2 corridas seguidas no duplican.
  6. CLI `rag migrations status` corre sin DB → version=0.
  7. CLI `rag migrations apply --dry-run` no escribe.
  8. Hash drift: si modificás una migration ya aplicada, log warning
     (no error fatal).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from rag import migrations as m


@pytest.fixture
def tmp_conn(tmp_path):
    """Conn raw a una DB tmp. NO pasa por `_ensure_telemetry_tables`,
    así los tests pueden ejercer migrations contra DBs en estados
    arbitrarios (virgen, parcialmente migrada, con history pre-existente)."""
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file), isolation_level=None)
    yield conn
    conn.close()


# ─── 1. DB virgen ──────────────────────────────────────────────────────────


def test_apply_pending_on_virgin_db_registers_max_version(tmp_conn):
    """En una DB virgen, apply_pending_migrations debe registrar todas las
    migrations conocidas y current_version queda en max(known)."""
    known = sorted(v for v, _, _ in m.known_migrations())
    assert m.current_version(tmp_conn) == 0
    applied = m.apply_pending_migrations(tmp_conn)
    assert applied == known
    assert m.current_version(tmp_conn) == known[-1]
    assert m.applied_versions(tmp_conn) == set(known)


# ─── 2. DB parcialmente migrada ─────────────────────────────────────────────


def test_apply_pending_skips_already_applied(tmp_conn):
    """Si la DB ya tiene versions 1-2 aplicadas, apply solo aplica el
    resto (3 hasta max_known)."""
    m._ensure_migrations_table(tmp_conn)
    # Simulate manual prior application of versions 1 + 2.
    tmp_conn.execute(
        "INSERT INTO rag_schema_migrations(version, name, applied_at, hash) VALUES(?, ?, ?, ?)",
        (1, "add_trace_id_to_queries", "2026-04-29T00:00:00+00:00", "fakehash1"),
    )
    tmp_conn.execute(
        "INSERT INTO rag_schema_migrations(version, name, applied_at, hash) VALUES(?, ?, ?, ?)",
        (2, "add_trace_id_to_behavior", "2026-04-29T00:00:00+00:00", "fakehash2"),
    )
    assert m.current_version(tmp_conn) == 2
    known = sorted(v for v, _, _ in m.known_migrations())
    expected_pending = [v for v in known if v > 2]
    applied = m.apply_pending_migrations(tmp_conn)
    assert applied == expected_pending
    assert m.applied_versions(tmp_conn) == set(known)


# ─── 3. Migration que falla ─────────────────────────────────────────────────


def test_failing_migration_rolls_back_savepoint(tmp_conn):
    """Una migration que tira excepción debe rollbackear el SAVEPOINT y
    NO registrar la version. Las migrations posteriores no corren."""
    # Snapshot del state actual.
    before = m._REGISTRY.copy()
    try:
        # Pre-aplicar las migrations sanas ANTES de registrar la fallante.
        max_known = max(v for v, _, _ in m.known_migrations())
        m.apply_pending_migrations(tmp_conn)
        assert m.current_version(tmp_conn) == max_known

        # Registrar una migration que tira — ahora va a quedar como pending.
        @m.migration(version=999, name="fails_intentionally")
        def fails(conn):
            conn.execute("CREATE TABLE _temp_test_fails (id INTEGER)")
            raise RuntimeError("simulated migration failure")

        with pytest.raises(RuntimeError, match="simulated migration failure"):
            m.apply_pending_migrations(tmp_conn)

        # 999 no se registró.
        assert 999 not in m.applied_versions(tmp_conn)
        # La tabla _temp_test_fails NO existe (rollback).
        row = tmp_conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_temp_test_fails'"
        ).fetchone()
        assert row is None, "SAVEPOINT no rollbackeó el CREATE TABLE de la migration fallida"
    finally:
        # Restaurar registry.
        m._REGISTRY.clear()
        m._REGISTRY.update(before)


# ─── 4. Bootstrap en DB pre-existente ───────────────────────────────────────


def test_bootstrap_existing_db_with_trace_id_skips_alter(tmp_conn):
    """Si la DB ya tiene rag_queries.trace_id (migrations corrieron en boot
    previo), bootstrap_existing_db registra todas las versions conocidas
    SIN correr el body — crítico para no tirar `duplicate column` en DBs
    reales del usuario."""
    # Simulamos una DB preexistente: rag_queries con trace_id ya presente.
    tmp_conn.execute(
        "CREATE TABLE rag_queries ("
        " id INTEGER PRIMARY KEY,"
        " ts TEXT,"
        " q TEXT,"
        " trace_id TEXT"
        ")"
    )
    tmp_conn.execute("INSERT INTO rag_queries(ts, q, trace_id) VALUES(?, ?, ?)",
                     ("2026-04-29", "test query", "abcd1234"))

    known = sorted(v for v, _, _ in m.known_migrations())
    registered = m.bootstrap_existing_db(tmp_conn)
    assert registered == known
    assert m.current_version(tmp_conn) == known[-1]

    # Datos previos intactos (no se hizo DROP/recreate).
    row = tmp_conn.execute("SELECT q, trace_id FROM rag_queries").fetchone()
    assert row == ("test query", "abcd1234")


def test_bootstrap_skips_when_history_already_present(tmp_conn):
    """Si rag_schema_migrations ya tiene rows, bootstrap es no-op."""
    m._ensure_migrations_table(tmp_conn)
    tmp_conn.execute(
        "INSERT INTO rag_schema_migrations(version, name, applied_at, hash)"
        " VALUES(?, ?, ?, ?)",
        (1, "add_trace_id_to_queries", "2026-04-29T00:00:00+00:00", "x"),
    )
    # Simular DB con rag_queries.trace_id (heurística calzaría).
    tmp_conn.execute("CREATE TABLE rag_queries (id INTEGER, trace_id TEXT)")
    registered = m.bootstrap_existing_db(tmp_conn)
    assert registered == []  # no bootstrap porque cur > 0
    assert m.current_version(tmp_conn) == 1


def test_bootstrap_skips_on_virgin_db(tmp_conn):
    """En una DB virgen (sin rag_queries) la heurística no calza →
    bootstrap no registra nada. apply_pending_migrations es el path
    correcto en ese caso."""
    registered = m.bootstrap_existing_db(tmp_conn)
    assert registered == []
    assert m.current_version(tmp_conn) == 0


# ─── 5. Idempotencia ────────────────────────────────────────────────────────


def test_apply_pending_is_idempotent(tmp_conn):
    """Llamar apply_pending_migrations 2 veces seguidas: la 2da no aplica
    nada y no muta la DB."""
    known = sorted(v for v, _, _ in m.known_migrations())
    first = m.apply_pending_migrations(tmp_conn)
    assert first == known
    snap_before = sorted(m.applied_versions(tmp_conn))
    second = m.apply_pending_migrations(tmp_conn)
    assert second == []
    snap_after = sorted(m.applied_versions(tmp_conn))
    assert snap_before == snap_after


# ─── 6. CLI `rag migrations status` sin DB ──────────────────────────────────


def test_cli_status_on_virgin_db(tmp_path, monkeypatch):
    """`rag migrations status --plain` contra DB virgen (DB_PATH apuntando
    a tmp) debe imprimir current_version=0 sin errores."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    # Limpiar cache para que el wiring no salte.
    rag._TELEMETRY_DDL_ENSURED_PATHS.clear()

    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["migrations", "status", "--plain"])
    assert result.exit_code == 0, result.output
    assert "current_version=0" in result.output
    max_known = max(v for v, _, _ in m.known_migrations())
    assert f"max_known={max_known}" in result.output


# ─── 7. CLI `rag migrations apply --dry-run` no escribe ──────────────────


def test_cli_apply_dry_run_does_not_write(tmp_path, monkeypatch):
    """`rag migrations apply --dry-run --plain` lista qué se aplicaría
    pero NO inserta filas en rag_schema_migrations."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    rag._TELEMETRY_DDL_ENSURED_PATHS.clear()

    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["migrations", "apply", "--dry-run", "--plain"])
    assert result.exit_code == 0, result.output
    assert "would apply" in result.output

    # Verificar que la DB no tiene rows aplicadas.
    db_file = tmp_path / "telemetry.db"
    assert db_file.exists()
    conn = sqlite3.connect(str(db_file))
    try:
        rows = conn.execute("SELECT version FROM rag_schema_migrations").fetchall()
        assert rows == []
    finally:
        conn.close()


# ─── 8. Hash drift ─────────────────────────────────────────────────────────


def test_hash_drift_logs_warning(tmp_conn, caplog):
    """Si una migration ya aplicada tiene un hash distinto del registrado
    (alguien la modificó desde el último apply), log warning. NO debe
    abortar — la app sigue corriendo."""
    m.apply_pending_migrations(tmp_conn)
    # Mutar el hash registrado para simular drift.
    tmp_conn.execute(
        "UPDATE rag_schema_migrations SET hash='deadbeef' WHERE version=1"
    )
    with caplog.at_level(logging.WARNING, logger="rag.migrations"):
        applied = m.apply_pending_migrations(tmp_conn)
    assert applied == []  # no nuevas migrations, solo drift check
    drift_records = [r for r in caplog.records if "schema_migration_drift" in r.message]
    assert drift_records, f"esperaba al menos 1 warning de drift, got {[r.message for r in caplog.records]}"
    assert "version=1" in drift_records[0].message


# ─── Extra: bootstrap end-to-end ────────────────────────────────────────────


def test_apply_after_bootstrap_is_noop(tmp_conn):
    """Después de un bootstrap exitoso, apply_pending_migrations no
    encuentra nada para aplicar. Edge crítico: en boot real, el wiring
    llama bootstrap PRIMERO y apply DESPUÉS — si el bootstrap registró
    bien, apply no debe re-ejecutar los ALTERs."""
    # DB con trace_id ya presente (heurística calza).
    tmp_conn.execute("CREATE TABLE rag_queries (id INTEGER, trace_id TEXT)")
    m.bootstrap_existing_db(tmp_conn)
    max_known = max(v for v, _, _ in m.known_migrations())
    assert m.current_version(tmp_conn) == max_known
    applied = m.apply_pending_migrations(tmp_conn)
    assert applied == []


# ─── 9. Migration 5: mood tables ────────────────────────────────────────────


def test_migration_005_creates_mood_tables(tmp_conn):
    """Migration 5 crea `rag_mood_signals` + `rag_mood_score_daily` con
    sus indexes. En DB virgen se aplica via apply_pending_migrations
    (que corre 1-5 en orden). Verificamos que las dos tablas existen
    con las columnas esperadas."""
    m.apply_pending_migrations(tmp_conn)
    assert 5 in m.applied_versions(tmp_conn)

    # Tabla 1: rag_mood_signals
    cols = {r[1] for r in tmp_conn.execute("PRAGMA table_info(rag_mood_signals)").fetchall()}
    expected_signal_cols = {
        "id", "ts", "date", "source", "signal_kind",
        "value", "weight", "evidence",
    }
    assert expected_signal_cols.issubset(cols), (
        f"rag_mood_signals missing cols: {expected_signal_cols - cols}"
    )

    # Tabla 2: rag_mood_score_daily
    cols = {r[1] for r in tmp_conn.execute("PRAGMA table_info(rag_mood_score_daily)").fetchall()}
    expected_score_cols = {
        "date", "score", "n_signals", "sources_used",
        "top_evidence", "updated_at",
    }
    assert expected_score_cols.issubset(cols), (
        f"rag_mood_score_daily missing cols: {expected_score_cols - cols}"
    )

    # Indexes — verificamos los 4 que la migration crea.
    indexes = {r[0] for r in tmp_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND tbl_name IN ('rag_mood_signals', 'rag_mood_score_daily')"
    ).fetchall()}
    expected_indexes = {
        "ix_rag_mood_signals_date",
        "ix_rag_mood_signals_ts",
        "ix_rag_mood_signals_source_date",
        "ix_rag_mood_score_daily_updated",
    }
    assert expected_indexes.issubset(indexes), (
        f"missing mood indexes: {expected_indexes - indexes}"
    )


def test_migration_005_idempotent_on_preexisting_tables(tmp_conn):
    """Si las tablas mood ya existen (creadas por _TELEMETRY_DDL en boot),
    la migration 5 corre sin tirar `table already exists` — usa CREATE
    TABLE IF NOT EXISTS."""
    # Pre-crear las tablas como las crearía _TELEMETRY_DDL.
    tmp_conn.execute(
        "CREATE TABLE rag_mood_signals ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " ts REAL NOT NULL,"
        " date TEXT NOT NULL,"
        " source TEXT NOT NULL,"
        " signal_kind TEXT NOT NULL,"
        " value REAL NOT NULL,"
        " weight REAL NOT NULL DEFAULT 1.0,"
        " evidence TEXT"
        ")"
    )
    tmp_conn.execute(
        "CREATE TABLE rag_mood_score_daily ("
        " date TEXT PRIMARY KEY,"
        " score REAL NOT NULL,"
        " n_signals INTEGER NOT NULL,"
        " sources_used TEXT,"
        " top_evidence TEXT,"
        " updated_at REAL NOT NULL"
        ")"
    )
    # Insertar una row para verificar que NO se trunca.
    tmp_conn.execute(
        "INSERT INTO rag_mood_score_daily(date, score, n_signals, updated_at) "
        "VALUES(?, ?, ?, ?)",
        ("2026-04-30", -0.42, 5, 1714521600.0),
    )
    applied = m.apply_pending_migrations(tmp_conn)
    assert 5 in applied
    # Data preserved.
    row = tmp_conn.execute(
        "SELECT score, n_signals FROM rag_mood_score_daily WHERE date=?",
        ("2026-04-30",),
    ).fetchone()
    assert row == (-0.42, 5)
