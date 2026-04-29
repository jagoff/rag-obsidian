"""Tests for `_load_contradiction_priors` (loop "contradicciones → ranker").

El loader lee `rag_contradictions` (tabla ya existente, populada por
`_log_contradictions` cuando `rag contradictions <path>` corre) y devuelve
un dict {subject_path: log1p(count_distinct_ts)} dentro de una ventana de
N días. El score se consume como penalty en retrieve gated por
`weights.contradiction_penalty` (default 0.0 = OFF).

Cubre:
  - tabla vacía → {}
  - 1 row → {path: log1p(1)}
  - 3 ts distintos para mismo path → {path: log1p(3)}
  - rows fuera de la ventana se ignoran
  - cap `max_paths` toma top-N por count desc
  - silent-fail si el read SQL revienta
"""
from __future__ import annotations

import datetime as _dt
import math
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

import rag


# ── Fixtures (mínimo viable; reusan el mismo pattern que test_rag_readers_sql) ──

def _open_db(tmp_path: Path) -> sqlite3.Connection:
    db = tmp_path / rag._TELEMETRY_DB_FILENAME
    conn = sqlite3.connect(str(db), isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_schema_version ("
        " table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
    )
    rag._ensure_telemetry_tables(conn)
    return conn


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(rag, "_SQL_STATE_ERROR_LOG",
                         tmp_path / "sql_state_errors.jsonl")
    # Pre-create tables so reader doesn't have to.
    conn = _open_db(tmp_path)
    conn.close()
    yield tmp_path


def _seed_contradiction(tmp_path: Path, ts: str, subject_path: str,
                         contradicts: list[dict] | None = None) -> None:
    conn = _open_db(tmp_path)
    try:
        rag._sql_append_event(
            conn, "rag_contradictions",
            rag._map_contradiction_row({
                "ts": ts,
                "subject_path": subject_path,
                "contradicts": contradicts or [],
                "helper_raw": "",
            }),
        )
    finally:
        conn.close()


# ── Tests ────────────────────────────────────────────────────────────────────

def test_empty_table_returns_empty_dict(db_env):
    """Sin rows → {} (no levanta, no rompe retrieve)."""
    out = rag._load_contradiction_priors()
    assert out == {}


def test_single_detection_returns_log1p_one(db_env):
    """1 detección para path X → {path: log1p(1)} ≈ 0.693."""
    now = _dt.datetime.now()
    fresh = (now - _dt.timedelta(days=2)).isoformat(timespec="seconds")
    _seed_contradiction(db_env, fresh, "01-Projects/Note.md",
                          [{"path": "02-Areas/x.md", "note": "n", "why": "w"}])
    out = rag._load_contradiction_priors()
    assert "01-Projects/Note.md" in out
    assert abs(out["01-Projects/Note.md"] - math.log1p(1)) < 1e-9


def test_multiple_distinct_ts_same_path_aggregates(db_env):
    """3 ts distintos para mismo path → log1p(3) ≈ 1.386."""
    now = _dt.datetime.now()
    for d in (1, 2, 3):
        ts = (now - _dt.timedelta(days=d)).isoformat(timespec="seconds")
        _seed_contradiction(db_env, ts, "01-Projects/Note.md",
                              [{"path": "02-Areas/x.md", "note": "n",
                                "why": "w"}])
    out = rag._load_contradiction_priors()
    assert abs(out["01-Projects/Note.md"] - math.log1p(3)) < 1e-9


def test_window_filters_old_rows(db_env):
    """Rows fuera del window (window_days=10, row de hace 30d) se ignoran."""
    now = _dt.datetime.now()
    fresh = (now - _dt.timedelta(days=2)).isoformat(timespec="seconds")
    old = (now - _dt.timedelta(days=30)).isoformat(timespec="seconds")
    _seed_contradiction(db_env, fresh, "fresh.md")
    _seed_contradiction(db_env, old, "old.md")
    out = rag._load_contradiction_priors(window_days=10)
    assert "fresh.md" in out
    assert "old.md" not in out


def test_window_includes_boundary_within_days(db_env):
    """Path con 2 ts dentro del window cuenta 2; 1 de ellos fuera, no."""
    now = _dt.datetime.now()
    inside_a = (now - _dt.timedelta(days=1)).isoformat(timespec="seconds")
    inside_b = (now - _dt.timedelta(days=5)).isoformat(timespec="seconds")
    outside = (now - _dt.timedelta(days=20)).isoformat(timespec="seconds")
    for ts in (inside_a, inside_b, outside):
        _seed_contradiction(db_env, ts, "P.md")
    out = rag._load_contradiction_priors(window_days=10)
    assert abs(out["P.md"] - math.log1p(2)) < 1e-9


def test_max_paths_cap_takes_top_by_count(db_env):
    """`max_paths=2` retiene los 2 paths con más detecciones, descarta el resto."""
    now = _dt.datetime.now()

    def _seed(path: str, n: int) -> None:
        for i in range(n):
            ts = (now - _dt.timedelta(days=1, seconds=i)).isoformat(timespec="seconds")
            _seed_contradiction(db_env, ts, path)

    _seed("hot.md", 5)
    _seed("medium.md", 3)
    _seed("cold.md", 1)

    out = rag._load_contradiction_priors(max_paths=2)
    assert set(out.keys()) == {"hot.md", "medium.md"}
    assert "cold.md" not in out


def test_silent_fail_returns_empty_dict_on_db_error(db_env, monkeypatch):
    """Si el SQL read revienta (no transient), helper devuelve {} silently."""
    # Forzamos un error no transient en el read.
    def _boom():
        raise RuntimeError("synthetic SQL failure")

    monkeypatch.setattr(rag, "_ragvec_state_conn",
                         lambda: (_ for _ in ()).throw(RuntimeError("no conn")))
    out = rag._load_contradiction_priors()
    assert out == {}


def test_empty_subject_path_filtered(db_env):
    """Rows con subject_path vacío o None no entran al dict."""
    now = _dt.datetime.now()
    fresh = (now - _dt.timedelta(days=1)).isoformat(timespec="seconds")
    # rag_contradictions schema tiene NOT NULL en subject_path; usamos
    # string vacío que sí compila pero el WHERE del reader filtra.
    conn = _open_db(db_env)
    try:
        conn.execute(
            "INSERT INTO rag_contradictions (ts, subject_path, contradicts_json)"
            " VALUES (?, ?, ?)",
            (fresh, "", "[]"),
        )
    finally:
        conn.close()
    out = rag._load_contradiction_priors()
    assert out == {}


def test_unique_constraint_prevents_double_count(db_env):
    """rag_contradictions tiene UNIQUE(ts, subject_path). Si el caller
    re-loggea con mismo (ts, path), la segunda inserción falla y solo
    cuenta 1. Asegura que no inflamos el penalty con duplicates.
    """
    now = _dt.datetime.now()
    ts = (now - _dt.timedelta(days=1)).isoformat(timespec="seconds")
    _seed_contradiction(db_env, ts, "P.md")
    # Segunda inserción con mismo (ts, path) — debería ser ignorada por
    # UNIQUE(ts, subject_path). Hacemos INSERT directo capturando
    # IntegrityError porque _sql_append_event no usa OR IGNORE.
    conn = _open_db(db_env)
    try:
        try:
            conn.execute(
                "INSERT INTO rag_contradictions (ts, subject_path, contradicts_json)"
                " VALUES (?, ?, ?)",
                (ts, "P.md", "[]"),
            )
        except sqlite3.IntegrityError:
            pass  # Esperado: UNIQUE violation.
    finally:
        conn.close()
    out = rag._load_contradiction_priors()
    assert abs(out["P.md"] - math.log1p(1)) < 1e-9
