"""Tests para la signal `stale_etl`.

Cubrimos:
- Job stale (último éxito > threshold) Y con baseline reciente → emit.
- Job fresh → no emit.
- Job nunca corrió en últimos 30d → no emit (feature inactiva).
- Tabla rag_supervisor_jobs no existe → silent-fail (no crash).
- Per-label threshold override via env.
- Disable completo via RAG_STALE_ETL_LABELS="".
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rag_anticipate.signals.stale_etl import stale_etl_signal


_SUPERVISOR_DDL = """
CREATE TABLE IF NOT EXISTS rag_supervisor_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_start TEXT NOT NULL,
  ts_end TEXT NOT NULL,
  job_label TEXT NOT NULL,
  duration_s REAL NOT NULL,
  exit_code INTEGER NOT NULL,
  trigger TEXT,
  signals TEXT,
  error TEXT,
  result TEXT
);
"""


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(_SUPERVISOR_DDL)
    conn.commit()
    return conn


def _insert_run(conn: sqlite3.Connection, label: str, ts: datetime, exit_code: int = 0):
    iso = ts.astimezone(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO rag_supervisor_jobs "
        "(ts_start, ts_end, job_label, duration_s, exit_code) "
        "VALUES (?, ?, ?, ?, ?)",
        (iso, iso, label, 1.0, exit_code),
    )
    conn.commit()


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """DB temporal con DDL + monkeypatch del path resolver."""
    db = tmp_path / "telemetry.db"
    conn = _make_db(db)
    yield db, conn
    conn.close()


@pytest.fixture
def patched_db_path(tmp_db, monkeypatch):
    """Patchea `_telemetry_db_path` en la signal para apuntar al tmp DB."""
    db, _conn = tmp_db
    import rag_anticipate.signals.stale_etl as mod

    monkeypatch.setattr(mod, "_telemetry_db_path", lambda: db)
    return db, _conn


def test_emit_when_job_stale_with_baseline(patched_db_path, monkeypatch):
    """Job que corrió OK hace 25d Y otra vez hace 50h pero NO en las últimas
    24h → stale (threshold default 48h). Emit candidate."""
    db, conn = patched_db_path
    now = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    _insert_run(conn, "anticipate", now - timedelta(days=25))  # baseline
    _insert_run(conn, "anticipate", now - timedelta(hours=50))  # último éxito

    monkeypatch.setenv("RAG_STALE_ETL_LABELS", "anticipate")
    monkeypatch.setenv("RAG_STALE_ETL_HOURS", "48")

    cands = stale_etl_signal(now)
    assert len(cands) == 1
    c = cands[0]
    assert c.kind == "anticipate-stale_etl"
    assert "anticipate" in c.message
    assert c.dedup_key.startswith("stale_etl:anticipate:")


def test_no_emit_when_job_fresh(patched_db_path, monkeypatch):
    """Último éxito hace 12h, threshold 48h → no emit."""
    db, conn = patched_db_path
    now = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    _insert_run(conn, "anticipate", now - timedelta(hours=12))

    monkeypatch.setenv("RAG_STALE_ETL_LABELS", "anticipate")
    monkeypatch.setenv("RAG_STALE_ETL_HOURS", "48")

    assert stale_etl_signal(now) == []


def test_no_emit_when_no_recent_baseline(patched_db_path, monkeypatch):
    """Job que corrió OK hace 60d (fuera de los 30d de baseline) Y nada más
    reciente → no emit (asumimos feature inactiva, no es 'stale')."""
    db, conn = patched_db_path
    now = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    _insert_run(conn, "ingest_old_thing", now - timedelta(days=60))

    monkeypatch.setenv("RAG_STALE_ETL_LABELS", "ingest_old_thing")
    monkeypatch.setenv("RAG_STALE_ETL_HOURS", "48")

    assert stale_etl_signal(now) == []


def test_no_emit_when_never_ran(patched_db_path, monkeypatch):
    """Label sin runs en absoluto → no emit."""
    db, _conn = patched_db_path
    now = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setenv("RAG_STALE_ETL_LABELS", "never_existed")

    assert stale_etl_signal(now) == []


def test_per_label_threshold_override(patched_db_path, monkeypatch):
    """`RAG_STALE_ETL_HOURS_INGEST_WHATSAPP` override per-label."""
    db, conn = patched_db_path
    now = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    # Baseline + último éxito hace 30h
    _insert_run(conn, "ingest_whatsapp", now - timedelta(days=20))
    _insert_run(conn, "ingest_whatsapp", now - timedelta(hours=30))

    monkeypatch.setenv("RAG_STALE_ETL_LABELS", "ingest_whatsapp")
    # Global default 48h → 30h sería fresh
    # Per-label override 24h → 30h es stale
    monkeypatch.setenv("RAG_STALE_ETL_HOURS_INGEST_WHATSAPP", "24")

    cands = stale_etl_signal(now)
    assert len(cands) == 1
    assert "ingest" in cands[0].message


def test_disable_via_empty_labels(patched_db_path, monkeypatch):
    """`RAG_STALE_ETL_LABELS=` (empty) → signal completamente desactivada."""
    db, conn = patched_db_path
    now = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    _insert_run(conn, "anticipate", now - timedelta(days=10))
    _insert_run(conn, "anticipate", now - timedelta(hours=72))  # bien stale

    monkeypatch.setenv("RAG_STALE_ETL_LABELS", "")

    assert stale_etl_signal(now) == []


def test_max_emit_cap(patched_db_path, monkeypatch):
    """Si 5 jobs están stale, emitimos máximo 2 (cap defensivo)."""
    db, conn = patched_db_path
    now = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    for i, label in enumerate(("a", "b", "c", "d", "e")):
        # baseline
        _insert_run(conn, label, now - timedelta(days=10))
        # último éxito: cada uno con distinta antigüedad
        _insert_run(conn, label, now - timedelta(hours=72 + i * 6))

    monkeypatch.setenv("RAG_STALE_ETL_LABELS", "a,b,c,d,e")
    monkeypatch.setenv("RAG_STALE_ETL_HOURS", "48")

    cands = stale_etl_signal(now)
    assert len(cands) == 2  # cap _MAX_EMIT


def test_silent_fail_on_missing_table(tmp_path, monkeypatch):
    """DB existe pero sin DDL → no crash, retorna []."""
    db = tmp_path / "tel.db"
    sqlite3.connect(str(db)).close()
    import rag_anticipate.signals.stale_etl as mod

    monkeypatch.setattr(mod, "_telemetry_db_path", lambda: db)
    monkeypatch.setenv("RAG_STALE_ETL_LABELS", "anticipate")

    now = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    assert stale_etl_signal(now) == []


def test_silent_fail_on_missing_db_file(tmp_path, monkeypatch):
    """Path no existe → []."""
    nonexistent = tmp_path / "missing.db"
    import rag_anticipate.signals.stale_etl as mod

    monkeypatch.setattr(mod, "_telemetry_db_path", lambda: nonexistent)
    monkeypatch.setenv("RAG_STALE_ETL_LABELS", "anticipate")

    assert stale_etl_signal(datetime.now()) == []


def test_failed_runs_dont_count(patched_db_path, monkeypatch):
    """Runs con `exit_code != 0` no cuentan como 'éxito' para el threshold.

    Si un job estuvo crasheando hace 10h y el último OK fue hace 70h →
    sigue siendo stale aunque haya entries recientes."""
    db, conn = patched_db_path
    now = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    _insert_run(conn, "anticipate", now - timedelta(days=10))  # baseline OK
    _insert_run(conn, "anticipate", now - timedelta(hours=70))  # último OK
    # 5 runs failed más recientes — no deben contar.
    for i in range(1, 6):
        _insert_run(conn, "anticipate", now - timedelta(hours=i), exit_code=1)

    monkeypatch.setenv("RAG_STALE_ETL_LABELS", "anticipate")
    monkeypatch.setenv("RAG_STALE_ETL_HOURS", "48")

    cands = stale_etl_signal(now)
    assert len(cands) == 1
    assert "70" in cands[0].message or "70h" in cands[0].message or "2d" in cands[0].message or "3d" in cands[0].message
