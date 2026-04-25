"""Tests del budget de retry en `_persist_with_sqlite_retry` del web/server.

Post 2026-04-23: bumped attempts 3→8 + backoff 0.1-0.35→0.15-0.6s, +
reintentamos "disk I/O error" (antes caía al primer intento). Cambio
derivó del audit de sql_state_errors.jsonl que mostró 258
`queries_sql_write_failed` + 46 `memory_sql_write_failed` acumulados en
2 semanas por el budget demasiado corto.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

import rag
from web import server as server_mod


def test_persist_with_sqlite_retry_default_attempts_is_8():
    """Default bumped 3→8 para alinear con rag.py y cubrir WAL contention
    con 3+ writers concurrentes (queries + memory + cpu samplers)."""
    calls = {"n": 0}

    def writer():
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with patch("time.sleep"):
        server_mod._persist_with_sqlite_retry(writer, "test_tag")

    assert calls["n"] == 8, (
        f"expected 8 retry attempts (post 2026-04-23), got {calls['n']}"
    )


def test_persist_with_sqlite_retry_explicit_attempts_override():
    """Callers en el hot path pueden pasar `attempts=3` para preservar
    el comportamiento tight pre-bump (budget <1s)."""
    calls = {"n": 0}

    def writer():
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with patch("time.sleep"):
        server_mod._persist_with_sqlite_retry(writer, "test_tag", attempts=3)

    assert calls["n"] == 3


def test_persist_with_sqlite_retry_succeeds_early_stops_retrying():
    """Éxito en el intento N: no hay más retries aunque N < attempts."""
    calls = {"n": 0}

    def writer():
        calls["n"] += 1
        if calls["n"] < 4:
            raise sqlite3.OperationalError("database is locked")
        # success on attempt 4

    with patch("time.sleep"):
        server_mod._persist_with_sqlite_retry(writer, "test_tag")

    assert calls["n"] == 4, (
        f"should stop at attempt 4 (first success), got {calls['n']}"
    )


def test_persist_with_sqlite_retry_disk_io_is_transient():
    """Post 2026-04-23: "disk I/O error" también se reintenta (audit del
    JSONL mostró 92 ocurrencias — typically fsync contention transient)."""
    calls = {"n": 0}

    def writer():
        calls["n"] += 1
        raise sqlite3.OperationalError("disk I/O error")

    with patch("time.sleep"):
        server_mod._persist_with_sqlite_retry(writer, "test_tag")

    assert calls["n"] == 8, (
        f"disk I/O error must be treated as transient, got {calls['n']} "
        "attempts (expected full retry)"
    )


@pytest.mark.parametrize("msg", [
    "no such table: rag_behavior",
    "UNIQUE constraint failed: rag_queries.id",
    "database disk image is malformed",
])
def test_persist_with_sqlite_retry_non_transient_fail_fast(msg):
    """Schema drift / corruption / constraint — NO reintentar."""
    calls = {"n": 0}

    def writer():
        calls["n"] += 1
        raise sqlite3.OperationalError(msg)

    with patch("time.sleep"):
        server_mod._persist_with_sqlite_retry(writer, "test_tag")

    assert calls["n"] == 1, (
        f"{msg!r} is non-transient, expected 1 attempt got {calls['n']}"
    )


def test_persist_with_sqlite_retry_backoff_bounds():
    """Los sleeps entre retries caen en el rango [0.15, 0.6]s."""
    sleeps: list[float] = []

    def writer():
        raise sqlite3.OperationalError("database is locked")

    def fake_sleep(duration):
        sleeps.append(duration)

    with patch("time.sleep", fake_sleep):
        server_mod._persist_with_sqlite_retry(writer, "test_tag")

    # 7 sleeps entre 8 attempts
    assert len(sleeps) == 7, f"expected 7 sleeps, got {len(sleeps)}"
    for s in sleeps:
        assert 0.15 <= s <= 0.6, f"sleep {s}s fuera del rango [0.15, 0.6]"
