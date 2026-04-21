"""Regression tests for `_sql_write_with_retry` + the SQL writers that
adopted it after the 2026-04-21 pass:

- `log_query_event`, `log_behavior_event`, `log_impressions`,
  `record_brief_written`, `_log_contradictions`, `_brief_state_record`,
  `_surface_log_run`, `_filing_log`, `_proactive_log`,
  `_ambient_state_record`, `_ambient_log_event`, `_log_tune_event`,
  `_log_archive_event`.

Pre-fix every one of these dropped events silently on
`sqlite3.OperationalError("database is locked")`. The
`sql_state_errors.jsonl` file showed dozens of `*_sql_write_failed`
entries per day — each = one lost telemetry/behavior/feedback event.

We don't exhaustively retest every writer (each is ~4 lines around
`_ragvec_state_conn`); we cover the helper directly + a representative
hot-path writer (`log_query_event`).
"""
from __future__ import annotations

import sqlite3

import pytest

import rag


# ── _sql_write_with_retry ────────────────────────────────────────────────────


def test_sql_write_with_retry_succeeds_on_first(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _x: None)
    logged: list[tuple] = []
    monkeypatch.setattr(rag, "_log_sql_state_error",
                        lambda tag, **kw: logged.append((tag, kw)))
    calls = []
    rag._sql_write_with_retry(lambda: calls.append(1), "tag")
    assert len(calls) == 1
    assert logged == []


def test_sql_write_with_retry_retries_on_locked(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _x: None)
    logged: list[tuple] = []
    monkeypatch.setattr(rag, "_log_sql_state_error",
                        lambda tag, **kw: logged.append((tag, kw)))
    n = {"calls": 0}

    def flaky():
        n["calls"] += 1
        if n["calls"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return None

    rag._sql_write_with_retry(flaky, "tag")
    assert n["calls"] == 3
    assert logged == []  # third attempt succeeded


def test_sql_write_with_retry_logs_after_max_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _x: None)
    logged: list[tuple] = []
    monkeypatch.setattr(rag, "_log_sql_state_error",
                        lambda tag, **kw: logged.append((tag, kw)))

    def always_locked():
        raise sqlite3.OperationalError("database is locked")

    rag._sql_write_with_retry(always_locked, "stuck_tag")
    assert len(logged) == 1
    assert logged[0][0] == "stuck_tag"


def test_sql_write_with_retry_no_retry_on_non_lock(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _x: None)
    logged: list[tuple] = []
    monkeypatch.setattr(rag, "_log_sql_state_error",
                        lambda tag, **kw: logged.append((tag, kw)))
    n = {"calls": 0}

    def schema_err():
        n["calls"] += 1
        raise sqlite3.OperationalError("no such table: rag_missing")

    rag._sql_write_with_retry(schema_err, "schema_tag")
    # Only ONE attempt — non-lock errors short-circuit.
    assert n["calls"] == 1
    assert len(logged) == 1
    assert "no such table" in logged[0][1]["err"]


def test_sql_write_with_retry_swallows_generic_exception(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _x: None)
    logged: list[tuple] = []
    monkeypatch.setattr(rag, "_log_sql_state_error",
                        lambda tag, **kw: logged.append((tag, kw)))

    def boom():
        raise RuntimeError("unexpected")

    rag._sql_write_with_retry(boom, "boom_tag")
    assert logged == [("boom_tag", {"err": "RuntimeError('unexpected')"})]


# ── log_query_event integration ──────────────────────────────────────────────


def test_log_query_event_drops_silently_on_persistent_lock(tmp_path, monkeypatch):
    """Hot path: query event writer must NEVER raise back to the caller
    even under sustained lock contention."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr("time.sleep", lambda _x: None)
    monkeypatch.setattr(rag, "_sql_append_event",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            sqlite3.OperationalError("database is locked")))
    # Should return None (not raise) after exhausting retries.
    result = rag.log_query_event({"cmd": "test_q", "q": "hola"})
    assert result is None


def test_log_query_event_lands_in_sql_when_writer_is_healthy(tmp_path, monkeypatch):
    """End-to-end: a working writer persists the event to rag_queries."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    rag.log_query_event({"cmd": "test_ok", "q": "ping"})
    conn = sqlite3.connect(str(tmp_path / "ragvec.db"))
    try:
        rows = conn.execute(
            "SELECT cmd, q FROM rag_queries WHERE cmd = ?",
            ("test_ok",),
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("test_ok", "ping")]


# ── _fetch_weather_openmeteo: round(None) regression ─────────────────────────


def test_weather_openmeteo_handles_null_temperature(monkeypatch):
    """Open-Meteo occasionally returns `null` for `temperature_2m` during
    sensor outages. Pre-fix this raised `TypeError: type NoneType doesn't
    define __round__`, killing morning brief generation."""
    payloads = iter([
        # Geocoding: 1 hit.
        {"results": [{"latitude": -31.6, "longitude": -60.7, "name": "Santa Fe"}]},
        # Weather: null temperature + null daily series.
        {
            "current": {"temperature_2m": None, "weather_code": None},
            "daily": {
                "time": ["2026-04-21", "2026-04-22"],
                "temperature_2m_max": [None, 20.4],
                "temperature_2m_min": [None, 12.1],
                "precipitation_probability_max": [None, 30],
                "weather_code": [None, 3],
            },
        },
    ])

    class _FakeResp:
        def __init__(self, payload):
            import json as _json
            self._body = _json.dumps(payload).encode()

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    def fake_urlopen(url, timeout=None):
        return _FakeResp(next(payloads))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = rag._fetch_weather_openmeteo("Santa+Fe,Argentina")
    assert out is not None
    assert out["current"]["temp_C"] == "0"  # null → 0
    assert len(out["days"]) == 2
    # Day 0 is fully null → empty strings, but still rendered.
    assert out["days"][0]["minC"] == ""
    assert out["days"][0]["maxC"] == ""
    assert out["days"][0]["chanceofrain"] == 0
    # Day 1 is healthy.
    assert out["days"][1]["minC"] == "12"
    assert out["days"][1]["maxC"] == "20"
    assert out["days"][1]["chanceofrain"] == 30


# ── ingest_calendar.py: atomic OAuth credential write ────────────────────────


def test_ingest_calendar_atomic_creds_write(tmp_path, monkeypatch):
    """Verify the import shape — we don't run the real google client here,
    just check that `_get_calendar_service` references `tmp.replace(...)`
    instead of a bare `write_text`. Catching this regression early is
    cheaper than running an OAuth roundtrip in CI.
    """
    src = (
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "scripts" / "ingest_calendar.py"
    ).read_text(encoding="utf-8")
    # Pre-fix: a bare `creds_path.write_text(json.dumps(stored)`. Post-fix:
    # `tmp = creds_path.with_suffix(".json.tmp")` + `tmp.replace(creds_path)`.
    assert "tmp = creds_path.with_suffix(\".json.tmp\")" in src
    assert "tmp.replace(creds_path)" in src
    assert "creds_path.write_text(json.dumps(stored)" not in src
