"""Tests F4.1-F4.3 — SQL change watcher base + triggers integrados.

Cubren:
- ``SqlChangeWatcher`` anchor inicial NO emite events para histórico.
- INSERTs nuevos disparan event con payload correcto.
- DB inexistente → watcher skip silently sin crashear.
- DB locked / corrupted → errors_count se incrementa, watcher sigue.
- Thread daemon idempotente (start 2x no duplica).
- Stop limpio.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from rag.runtime._sql_watcher import SqlChangeWatcher


@pytest.fixture
def short_db(tmp_path):
    """SQLite path bajo /tmp para no bloquear file watcher de pytest."""
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    d = Path(tempfile.mkdtemp(dir="/tmp", prefix="rag-sqlw-"))
    db = d / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, payload TEXT)")
    # Histórico: 3 rows pre-watcher.
    conn.executemany(
        "INSERT INTO events(payload) VALUES (?)",
        [("hist_1",), ("hist_2",), ("hist_3",)],
    )
    conn.commit()
    conn.close()
    yield db
    shutil.rmtree(d, ignore_errors=True)


def test_anchor_at_start_skips_history(short_db):
    """Watcher arrancado con anchor_at_start=True NO emite por los rows
    históricos (que ya estaban antes del watcher)."""
    received = []
    watcher = SqlChangeWatcher(
        db_path=short_db,
        table="events",
        event_name="test.events.inserted",
        emit_fn=lambda name, payload: received.append((name, payload)),
        anchor_at_start=True,
    )
    watcher.start()
    time.sleep(0.5)  # darle tiempo al thread a hacer anchor + 1 poll
    watcher.stop()

    # El anchor + primer poll NO deberían emitir nada — los 3 rows
    # estaban antes del start.
    assert received == [], f"esperaba [], got {received}"
    # El anchor debería haber leído rowid=3.
    assert watcher.state.last_seen_rowid == 3


def test_insert_after_start_emits_event(short_db):
    received = []
    watcher = SqlChangeWatcher(
        db_path=short_db,
        table="events",
        event_name="test.events.inserted",
        poll_interval_s=1,
        emit_fn=lambda name, payload: received.append((name, payload)),
    )
    watcher.start()
    time.sleep(0.3)  # anchor

    # Insert post-anchor.
    conn = sqlite3.connect(str(short_db))
    conn.execute("INSERT INTO events(payload) VALUES ('new_1')")
    conn.execute("INSERT INTO events(payload) VALUES ('new_2')")
    conn.commit()
    conn.close()

    # Force poll para no esperar al timer.
    watcher.trigger_poll_for_test()

    watcher.stop()

    assert len(received) >= 1
    name, payload = received[-1]
    assert name == "test.events.inserted"
    assert payload["new_rows"] == 2
    assert payload["table"] == "events"
    assert payload["min_rowid"] == 4
    assert payload["max_rowid"] == 5


def test_db_missing_no_crash(tmp_path):
    """Watcher arrancado contra DB inexistente NO crashea."""
    received = []
    watcher = SqlChangeWatcher(
        db_path=tmp_path / "nonexistent.db",
        table="events",
        event_name="test.miss",
        poll_interval_s=1,
        emit_fn=lambda name, payload: received.append((name, payload)),
    )
    watcher.start()
    time.sleep(0.3)
    watcher.trigger_poll_for_test()
    watcher.stop()

    # Sin crashear, sin events.
    assert received == []


def test_idempotent_start(short_db):
    watcher = SqlChangeWatcher(
        db_path=short_db,
        table="events",
        event_name="test",
        emit_fn=lambda name, payload: None,
    )
    started1 = watcher.start()
    started2 = watcher.start()
    watcher.stop()
    assert started1 is True
    assert started2 is False


def test_stop_terminates_thread(short_db):
    watcher = SqlChangeWatcher(
        db_path=short_db,
        table="events",
        event_name="test",
        poll_interval_s=2,
        emit_fn=lambda name, payload: None,
    )
    watcher.start()
    time.sleep(0.2)
    assert watcher._thread is not None
    assert watcher._thread.is_alive()
    watcher.stop()
    # join con timeout=5s adentro del stop().
    assert not watcher._thread.is_alive()


def test_unknown_table_records_error(tmp_path):
    """Tabla inexistente — error capturado, watcher sigue."""
    db = tmp_path / "tab.db"
    sqlite3.connect(str(db)).close()  # crear DB vacía
    received = []
    watcher = SqlChangeWatcher(
        db_path=db,
        table="does_not_exist",
        event_name="test",
        poll_interval_s=10,
        emit_fn=lambda name, payload: received.append((name, payload)),
    )
    watcher.start()
    time.sleep(0.2)
    watcher.trigger_poll_for_test()
    watcher.stop()

    # Errores se contaron sin crashear.
    assert watcher.state.errors_count >= 1
    assert received == []
