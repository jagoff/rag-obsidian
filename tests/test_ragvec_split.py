"""Contract tests for the ragvec.db → telemetry.db split (T1-T3, 2026-04-21).

Guards against accidental drift: if someone renames _TELEMETRY_DB_FILENAME,
moves ingester state tables, or re-routes _ragvec_state_conn() to ragvec.db,
these tests catch it at CI time rather than in production.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

import rag


# ── a) _TELEMETRY_DB_FILENAME constant ───────────────────────────────────────

def test_telemetry_db_filename_is_telemetry_db():
    """Guard against accidental drift of the constant that routes all
    telemetry writes. The value must stay 'telemetry.db' — anything else
    means a rename happened without updating the migration script and all
    callers that open the file by name."""
    assert rag._TELEMETRY_DB_FILENAME == "telemetry.db"


# ── b) _ragvec_state_conn opens telemetry.db, NOT ragvec.db ─────────────────

def test_ragvec_state_conn_targets_telemetry_db(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        row = conn.execute("PRAGMA database_list").fetchone()
        # row = (seq, name, file) — file is the absolute path
        assert str(row[2]).endswith("telemetry.db"), (
            f"_ragvec_state_conn opened {row[2]!r}, expected …/telemetry.db"
        )
    assert (tmp_path / "telemetry.db").is_file()
    assert not (tmp_path / "ragvec.db").is_file()


# ── c) WAL checkpoints target distinct files ─────────────────────────────────

def test_wal_checkpoint_split(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    for name in ("ragvec.db", rag._TELEMETRY_DB_FILENAME):
        sqlite3.connect(str(tmp_path / name)).close()
    r_vec = rag._vec_wal_checkpoint(dry_run=True)
    r_tel = rag._telemetry_wal_checkpoint(dry_run=True)
    assert r_vec["ok"] is True
    assert r_tel["ok"] is True
    # Sanity: both returned a dry_run marker
    assert r_vec.get("dry_run") is True
    assert r_tel.get("dry_run") is True


# ── d) _sql_rotate_log_tables uses telemetry.db ──────────────────────────────

def test_sql_rotate_log_tables_skips_when_no_telemetry_db(monkeypatch, tmp_path):
    """When telemetry.db is absent, _sql_rotate_log_tables must report
    'skipped' — it must NOT look in ragvec.db instead."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    # Only ragvec.db present — telemetry.db absent.
    sqlite3.connect(str(tmp_path / "ragvec.db")).close()
    out = rag._sql_rotate_log_tables(dry_run=True)
    assert "skipped" in out, (
        "_sql_rotate_log_tables did not return 'skipped' when telemetry.db "
        "is absent — it may be reading from ragvec.db instead"
    )


def test_sql_rotate_log_tables_reads_telemetry_db(monkeypatch, tmp_path):
    """When telemetry.db exists with seeded rows, rotation finds them."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    # Seed a 120-day-old rag_queries row in telemetry.db (via _ragvec_state_conn).
    import time
    from datetime import datetime, timedelta
    old_ts = (datetime.now() - timedelta(days=120)).isoformat(timespec="seconds")
    with rag._ragvec_state_conn() as conn:
        rag._sql_append_event(conn, "rag_queries", {"ts": old_ts, "q": "split-test"})
    now = time.time()
    out = rag._sql_rotate_log_tables(dry_run=False, now_ts=now)
    assert "skipped" not in out
    assert out["rows_deleted"].get("rag_queries", 0) == 1


# ── e) migrate_ragvec_split.py is importable without IO side-effects ─────────

def test_migrate_ragvec_split_importable():
    """The migration script must be importable with only in-process side
    effects (sys.path insert, constant initialisation). No filesystem IO,
    no subprocess calls, no DB connections at module level."""
    scripts_dir = str(Path(__file__).parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import importlib
    mod = importlib.import_module("migrate_ragvec_split")
    # Verify the key constant exists on the module object.
    assert hasattr(mod, "_INGESTER_STATE_STAY_IN_RAGVEC")
    assert hasattr(mod, "_TELEMETRY_TABLES")
    assert hasattr(mod, "_SKIP_COPY")


# ── f) Ingester state tables stay in ragvec.db ───────────────────────────────

def test_ingester_state_tables_stay_in_ragvec_db():
    """Guard against accidentally routing ingest_whatsapp.py to telemetry.db.

    The script opens its own sqlite3.connect(rag.DB_PATH / 'ragvec.db')
    directly — bypassing _ragvec_state_conn(). Moving rag_whatsapp_state to
    telemetry.db without updating the script would silently re-create empty
    tables on the next ingest run, losing the incremental cursor.
    """
    src = (Path(__file__).parent.parent / "scripts" / "ingest_whatsapp.py").read_text()
    assert 'rag.DB_PATH / "ragvec.db"' in src, (
        "ingest_whatsapp.py no longer opens ragvec.db directly — "
        "verify that rag_whatsapp_state is still in ragvec.db and update this guard."
    )


# ── g) migrate script excludes ingester state tables ─────────────────────────

def test_migrate_ragvec_split_skips_ingester_state():
    """_INGESTER_STATE_STAY_IN_RAGVEC must contain the four ingester cursors."""
    scripts_dir = str(Path(__file__).parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import importlib
    mod = importlib.import_module("migrate_ragvec_split")
    stay = mod._INGESTER_STATE_STAY_IN_RAGVEC
    for table in ("rag_whatsapp_state", "rag_gmail_state",
                   "rag_calendar_state", "rag_reminders_state"):
        assert table in stay, (
            f"{table} missing from _INGESTER_STATE_STAY_IN_RAGVEC — "
            "it would be moved to telemetry.db, breaking the ingest cursor."
        )


def test_migrate_ragvec_skip_copy_includes_ingester_state():
    """_SKIP_COPY must union _INGESTER_STATE_STAY_IN_RAGVEC so the copy loop
    never touches those tables."""
    scripts_dir = str(Path(__file__).parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import importlib
    mod = importlib.import_module("migrate_ragvec_split")
    for table in mod._INGESTER_STATE_STAY_IN_RAGVEC:
        assert table in mod._SKIP_COPY, (
            f"{table} is in _INGESTER_STATE_STAY_IN_RAGVEC but not in _SKIP_COPY"
        )
