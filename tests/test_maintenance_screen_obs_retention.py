"""Tests for the Peekaboo screen_observations retention step in run_maintenance.

Cubre:
  - tabla vacía → deleted=0, sin error.
  - rows mixtos (algunas >7d, algunas <7d) → solo viejas eliminadas.
  - dry_run=True reporta count sin borrar.
  - tabla inexistente → ensure-once la crea y deleted=0 (no crash).
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

import rag


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Redirige DB_PATH a tmp + monkeypatcha vault para evitar tocar nada real."""
    db_dir = tmp_path / "ragvec"
    db_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(rag, "DB_PATH", db_dir)
    return db_dir


def _seed_observations(db_path: Path, rows: list[tuple]) -> None:
    """Inserta rows en rag_screen_observations. Cada row: (ts, app, title, caption)."""
    con = sqlite3.connect(str(db_path / "telemetry.db"))
    rag._ensure_telemetry_tables(con)
    con.executemany(
        "INSERT INTO rag_screen_observations "
        "(ts, app_name, window_title, caption, caption_simhash, took_ms, capture_mode) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(ts, app, title, caption, 0, 100, "frontmost") for (ts, app, title, caption) in rows],
    )
    con.commit()
    con.close()


def _count_rows(db_path: Path) -> int:
    con = sqlite3.connect(str(db_path / "telemetry.db"))
    try:
        n = con.execute("SELECT COUNT(*) FROM rag_screen_observations").fetchone()[0]
    finally:
        con.close()
    return n


def test_empty_table_no_op(isolated_db):
    """Tabla vacía → deleted=0, key presente."""
    result = rag.run_maintenance(dry_run=False, skip_reindex=True, skip_logs=True)
    assert "screen_obs_deleted" in result
    assert result["screen_obs_deleted"] == 0


def test_old_rows_deleted(isolated_db):
    """Rows >7d desaparecen, rows recientes quedan."""
    now = int(time.time())
    _seed_observations(isolated_db, [
        (now - 10 * 86400, "OldApp", "old window", "caption vieja"),     # >7d
        (now - 9 * 86400,  "Old2",   "still old", "otra vieja"),         # >7d
        (now - 3 * 86400,  "Recent", "fresh window", "caption fresca"),  # <7d
        (now - 60,         "Now",    "current",     "ahora mismo"),      # ahora
    ])
    assert _count_rows(isolated_db) == 4

    result = rag.run_maintenance(dry_run=False, skip_reindex=True, skip_logs=True)
    assert result["screen_obs_deleted"] == 2
    assert _count_rows(isolated_db) == 2

    # Confirmá cuáles quedaron — las recientes.
    con = sqlite3.connect(str(isolated_db / "telemetry.db"))
    try:
        apps = sorted(r[0] for r in con.execute(
            "SELECT app_name FROM rag_screen_observations",
        ).fetchall())
    finally:
        con.close()
    assert apps == ["Now", "Recent"]


def test_dry_run_reports_without_deleting(isolated_db):
    """dry_run=True calcula el count pero NO borra."""
    now = int(time.time())
    _seed_observations(isolated_db, [
        (now - 14 * 86400, "Ancient", "t1", "c1"),
        (now - 8 * 86400,  "Older",   "t2", "c2"),
        (now - 1 * 86400,  "Recent",  "t3", "c3"),
    ])
    assert _count_rows(isolated_db) == 3

    result = rag.run_maintenance(dry_run=True, skip_reindex=True, skip_logs=True)
    val = result.get("screen_obs_deleted")
    assert isinstance(val, str) and val.startswith("dry-run: would delete 2"), val
    assert _count_rows(isolated_db) == 3, "dry_run NO debe borrar rows"


def test_table_missing_ensures_and_no_op(isolated_db):
    """Si la tabla no existe (DB fresca), el retention la ensure-ea y retorna 0."""
    # No seeding — la DB ni siquiera existe todavía.
    assert not (isolated_db / "telemetry.db").exists()

    result = rag.run_maintenance(dry_run=False, skip_reindex=True, skip_logs=True)
    assert result["screen_obs_deleted"] == 0
    # Y la tabla ahora SÍ existe.
    con = sqlite3.connect(str(isolated_db / "telemetry.db"))
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='rag_screen_observations'",
        ).fetchall()]
    finally:
        con.close()
    assert names == ["rag_screen_observations"]


def test_no_error_on_repeat_invocations(isolated_db):
    """Idempotencia: 3 invocaciones consecutivas no degradan."""
    now = int(time.time())
    _seed_observations(isolated_db, [
        (now - 10 * 86400, "Old", "t", "c"),
        (now - 1 * 86400,  "New", "t", "c"),
    ])
    for _ in range(3):
        result = rag.run_maintenance(dry_run=False, skip_reindex=True, skip_logs=True)
        assert "screen_obs_retention_error" not in result
    assert _count_rows(isolated_db) == 1  # solo el New sobrevive
