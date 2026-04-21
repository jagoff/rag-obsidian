"""Tests for persistent `created_ts` backfill marker.

Root cause: pre-2026-04-21 the `_maybe_backfill_created_ts()` idempotent guard
was an in-process global `_CREATED_TS_BACKFILL_DONE`. Every time launchd
restarted the web daemon (149 restarts counted in web.log over ~3 days), the
flag cleared and the first query with `date_range` paid a ~1s scan cost over
3600 chunks + 580 file reads.

Fix: persist the "backfill complete" marker to SQL (`rag_schema_version` as a
key/value store). Survives process restarts → scan runs ONCE per vault,
forever. Re-runs gracefully if the marker write ever fails or the index
gets `--reset`ed (col.count() goes to zero, marker loses meaning, but the
scan is a fast no-op anyway).
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

import rag  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state(tmp_path, monkeypatch):
    """Isolated DB + reset in-process flag between tests.

    `_maybe_backfill_created_ts` talks to `get_db()._db` directly, so we
    need to redirect `DB_PATH` to a fresh tmp folder (follows the pattern
    from `tests/test_collection_safety.py` and `tests/test_dashboard_sql.py`).
    Also clear `_db_singleton` so the next `get_db()` picks up the new
    DB_PATH and builds a fresh `SqliteVecCollection`.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    (tmp_path / "ragvec").mkdir(parents=True, exist_ok=True)
    # Drop the cached singleton so the next get_db() rebuilds against DB_PATH.
    rag._db_singleton = None
    rag._CREATED_TS_BACKFILL_DONE = False
    yield
    rag._db_singleton = None
    rag._CREATED_TS_BACKFILL_DONE = False


def test_marker_absent_initially():
    """Fresh vault has no marker in rag_schema_version."""
    assert rag._created_ts_backfill_persisted() is False


def test_mark_and_check_round_trip():
    """_mark_created_ts_backfill_persisted sets the SQL row, _created_ts_backfill_persisted
    reads it back as True."""
    rag._mark_created_ts_backfill_persisted()
    assert rag._created_ts_backfill_persisted() is True


def test_mark_is_idempotent():
    """Llamar `_mark_` dos veces no rompe nada (ON CONFLICT DO UPDATE)."""
    rag._mark_created_ts_backfill_persisted()
    rag._mark_created_ts_backfill_persisted()
    assert rag._created_ts_backfill_persisted() is True


def test_backfill_skips_when_marker_persisted(monkeypatch):
    """Regression: si el marker ya está persistido, el backfill debe saltar
    el escaneo incluso si el flag in-memory fue reseteado (simula restart)."""
    # Pre-set the marker
    rag._mark_created_ts_backfill_persisted()

    # Simulate restart: in-memory flag reset
    rag._CREATED_TS_BACKFILL_DONE = False

    # Monkeypatch `col.get` to raise if called — si se llama, el test falla
    # (el fast-path post-marker NO debe tocar metadata).
    called = {"get": False}
    original_get_db = rag.get_db

    def _track_get(*a, **kw):
        col = original_get_db(*a, **kw)
        original_col_get = col.get

        def _tracked_get(*args, **kwargs):
            called["get"] = True
            return original_col_get(*args, **kwargs)

        col.get = _tracked_get
        return col

    monkeypatch.setattr(rag, "get_db", _track_get)

    rag._maybe_backfill_created_ts()

    # El marker debe haber evitado el scan
    assert called["get"] is False
    # Y el in-memory flag también se setea para esta run
    assert rag._CREATED_TS_BACKFILL_DONE is True


def test_backfill_marks_on_empty_vault():
    """Empty vault (col.count() == 0) → marker no se setea porque el
    early-return de `col.count() == 0` sale antes de llegar al mark.
    Esto es intencional: sin chunks no hay nada que backfillear, pero
    tampoco sabemos si el vault quedará permanentemente vacío."""
    rag._maybe_backfill_created_ts()
    # count=0 vault: no marker set (behaviour-documented test)
    assert rag._created_ts_backfill_persisted() is False


def test_backfill_marks_when_all_chunks_have_created_ts(monkeypatch):
    """Si todos los chunks ya tienen created_ts (caso común post-reset index),
    el backfill termina sin trabajo pero SIGUE marcando el SQL — así un restart
    futuro ya no re-escanea los 3600 chunks.

    Mockeamos `get_db` entero para aislarnos del schema de sqlite-vec; la
    lógica del marker no depende del backend real (cubierto por
    test_mark_and_check_round_trip + test_backfill_skips_when_marker_persisted
    en el integration path)."""
    mark_calls = []
    # Capture the real `_mark_created_ts_backfill_persisted` without recursing.
    real_mark = rag._mark_created_ts_backfill_persisted

    def _tracked_mark():
        mark_calls.append(True)
        real_mark()

    monkeypatch.setattr(rag, "_mark_created_ts_backfill_persisted", _tracked_mark)

    # Mock get_db → fake collection with all chunks already backfilled.
    class _FakeCol:
        _db = None  # not used in this path

        def count(self):
            return 3

        def get(self, **_kw):
            return {
                "ids": ["c1", "c2", "c3"],
                "metadatas": [
                    {"file": "a.md", "created_ts": 1000.0},
                    {"file": "b.md", "created_ts": 2000.0},
                    {"file": "c.md", "created_ts": 3000.0},
                ],
            }

    monkeypatch.setattr(rag, "get_db", lambda *a, **kw: _FakeCol())

    rag._maybe_backfill_created_ts()

    # Mark should fire even though there's nothing to backfill.
    assert mark_calls == [True]


def test_persisted_check_swallows_exceptions(monkeypatch):
    """Si la query SQL explota (DB lockeada, schema corrupto), `_created_ts_backfill_persisted`
    retorna False sin raisear — el caller corre el backfill una vez como fallback."""
    class _BrokenCol:
        class _db:
            @staticmethod
            def execute(*args, **kwargs):
                raise RuntimeError("DB locked")

    monkeypatch.setattr(rag, "get_db", lambda *a, **kw: _BrokenCol())
    # No excepción bubbles up
    assert rag._created_ts_backfill_persisted() is False


def test_mark_swallows_exceptions(monkeypatch):
    """Misma defensiva para el writer — si el write falla, silent_log pero no raisea."""
    class _BrokenCol:
        class _db:
            @staticmethod
            def execute(*args, **kwargs):
                raise RuntimeError("DB locked during mark")

            @staticmethod
            def commit():
                pass

    monkeypatch.setattr(rag, "get_db", lambda *a, **kw: _BrokenCol())
    # No excepción bubbles up
    rag._mark_created_ts_backfill_persisted()


def test_marker_name_is_a_sentinel_not_a_real_table():
    """Defensa: el marker NO debe colisionar con el nombre de ninguna tabla
    SQL real. El prefijo `_` indica "privado / sentinel"."""
    assert rag._CREATED_TS_BACKFILL_MARKER.startswith("_")
    # Y el valor concreto no debe aparecer en el DDL de ninguna tabla
    ddl_names = [name for name, _ in rag._TELEMETRY_DDL]
    assert rag._CREATED_TS_BACKFILL_MARKER not in ddl_names
