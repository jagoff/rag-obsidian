"""Tests para la tabla rag_daemon_runs (telemetry del control plane).

Cubre el DDL + writer `_log_daemon_run_event` con DB_PATH isolation
manual (snap+restore). Patrón obligatorio per CLAUDE.md sección
"Test DB_PATH isolation per-file".
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

import rag


@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path: Path):
    """Redirige rag.DB_PATH a tmp_path antes del test, restaura después.

    NO usar monkeypatch.setattr — el teardown se interpone con el
    stabilizer del conftest. Snap+restore manual es el patrón canónico.
    """
    snap = rag.DB_PATH
    rag.DB_PATH = tmp_path / "ragvec"
    rag.DB_PATH.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        rag.DB_PATH = snap


def _read_db_path() -> Path:
    """Resolver el path del telemetry.db actual."""
    return rag.DB_PATH / "telemetry.db"


def test_ddl_creates_rag_daemon_runs_table():
    """El DDL crea la tabla con las 8 columnas + 2 índices esperados."""
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)
        cols = conn.execute(
            "PRAGMA table_info(rag_daemon_runs)"
        ).fetchall()
        col_names = {c[1] for c in cols}
        assert col_names == {
            "id", "ts", "label", "action",
            "prev_state", "new_state", "exit_code", "reason",
        }, f"Columnas: {col_names}"
        # Índices
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='rag_daemon_runs'"
        ).fetchall()
        idx_names = {i[0] for i in indexes}
        assert "ix_rag_daemon_runs_ts" in idx_names
        assert "ix_rag_daemon_runs_label" in idx_names


def test_log_daemon_run_event_inserts_row():
    """Happy path — `_log_daemon_run_event` inserta row con todos los campos."""
    rag._log_daemon_run_event(
        label="com.fer.obsidian-rag-test",
        action="status_check",
        prev_state="running",
        new_state="running",
        exit_code=0,
        reason="manual test",
    )
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)
        rows = conn.execute(
            "SELECT label, action, prev_state, new_state, exit_code, reason "
            "FROM rag_daemon_runs WHERE label = 'com.fer.obsidian-rag-test'"
        ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row[0] == "com.fer.obsidian-rag-test"
    assert row[1] == "status_check"
    assert row[2] == "running"
    assert row[3] == "running"
    assert row[4] == 0
    assert row[5] == "manual test"


def test_log_daemon_run_event_optional_fields_null():
    """Llamar con sólo label + action → resto de campos quedan NULL."""
    rag._log_daemon_run_event(
        label="com.fer.obsidian-rag-test-min",
        action="kickstart",
    )
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)
        row = conn.execute(
            "SELECT prev_state, new_state, exit_code, reason "
            "FROM rag_daemon_runs WHERE label = 'com.fer.obsidian-rag-test-min'"
        ).fetchone()
    assert row == (None, None, None, None)


def test_log_daemon_run_event_silent_fail_on_sql_error():
    """Si SQL falla → no raisea, y _silent_log o equivalente registra."""
    with patch("rag._sql_append_event",
               side_effect=sqlite3.OperationalError("disk I/O error")):
        # No debe raisear.
        rag._log_daemon_run_event(
            label="test-fail",
            action="status_check",
            reason="silent fail test",
        )
    # El test pasa si llegamos acá sin exception.
