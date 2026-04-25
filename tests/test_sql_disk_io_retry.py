"""Tests: `_sql_write_with_retry` / `_sql_read_with_retry` reintentran
`disk I/O error` (no solo `database is locked`).

Auditoría 2026-04-22: `sql_state_errors.jsonl` tiene 56 entradas con
`OperationalError('disk I/O error')` — 47 son de hoy, en clusters de
múltiples errores en el mismo segundo (duplicados a 11:17:09, 13:48:03,
13:58:16, 14:01:31, 14:48:53 — sugiere fsync contention concurrente).

Diagnóstico:
  - DB integrity OK (`PRAGMA integrity_check` = ok)
  - Disk tiene 195 GB libres, no es disk full
  - `~/.local/share/obsidian-rag/` está en el filesystem local, no en
    iCloud Drive
  - El patrón de clusters sugiere: fsync falla transient cuando varios
    writers están flusheando al mismo tiempo (indexer + web daemon +
    ingesters + watch)

El retry wrapper ya tenía el contrato correcto para `database is locked`
(reintenta 5 veces con backoff jittered). Pero `disk I/O error` caía al
primer intento porque la condición era `"locked" not in str(exc).lower()
→ log_and_return`. En la mayoría de los casos fsync failures son transient
— al segundo intento, cuando el OS terminó lo que estaba haciendo, el
write pasa.

Fix: expandir la lista de errores reintentables a `("locked", "disk i/o
error")`. Conservative: errores tipo `schema drift`, `disk full`,
`no such table`, `unique constraint` siguen cayendo al primer intento
(esos NO se arreglan con retry).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


# ── write retry reintenta disk I/O error ─────────────────────────────────────


def test_sql_write_retry_handles_disk_io_error():
    """El write retry debe reintentar `disk I/O error` como hace con
    `database is locked`."""
    calls = {"n": 0}

    def writer():
        calls["n"] += 1
        raise sqlite3.OperationalError("disk I/O error")

    with patch("time.sleep"):
        rag._sql_write_with_retry(writer, "test_tag")

    # Post 2026-04-23 el default es 8 attempts (bumped de 5).
    assert calls["n"] == 8, (
        f"disk I/O error es transient (fsync contention) — debe reintentar "
        f"8 veces antes de darse por vencido. Got {calls['n']} calls."
    )


def test_sql_write_retry_disk_io_recovers_on_attempt_3():
    """Si el fsync transient se resuelve al tercer intento, el write
    completa sin loggear error."""
    calls = {"n": 0}
    logged: list[str] = []

    def writer():
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("disk I/O error")
        # success on 3rd

    def fake_log(tag, **kw):
        logged.append(tag)

    with patch("time.sleep"), patch.object(rag, "_log_sql_state_error", fake_log):
        rag._sql_write_with_retry(writer, "test_tag")

    assert calls["n"] == 3
    assert logged == [], (
        "retry exitoso no debe loggear error (diferencia vs falla final)"
    )


def test_sql_write_retry_preserves_non_transient_fail_fast():
    """Regression guard: errores NO-transient (schema drift, UNIQUE
    violation, no such table) deben seguir cayendo al primer intento
    — retryear sería gastar latencia sin beneficio."""
    for non_transient in [
        "no such table: foo",
        "UNIQUE constraint failed",
        "database disk image is malformed",  # corruption — not fixable via retry
    ]:
        calls = {"n": 0}

        def writer():
            calls["n"] += 1
            raise sqlite3.OperationalError(non_transient)

        with patch("time.sleep"):
            rag._sql_write_with_retry(writer, "test_tag")

        assert calls["n"] == 1, (
            f"{non_transient!r} no es transient; retry lo haría peor. "
            f"Got {calls['n']} calls."
        )


# ── read retry reintenta disk I/O error ──────────────────────────────────────


def test_sql_read_retry_handles_disk_io_error():
    calls = {"n": 0}

    def reader():
        calls["n"] += 1
        raise sqlite3.OperationalError("disk I/O error")

    with patch("time.sleep"):
        out = rag._sql_read_with_retry(reader, "test_tag", default="fallback")

    assert calls["n"] == 5
    assert out == "fallback"


def test_sql_read_retry_disk_io_recovers():
    calls = {"n": 0}

    def reader():
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("disk I/O error")
        return "recovered"

    with patch("time.sleep"):
        out = rag._sql_read_with_retry(reader, "test_tag", default=None)

    assert out == "recovered"
    assert calls["n"] == 3


def test_sql_read_retry_non_transient_fail_fast():
    """Simetría con write retry."""
    for non_transient in [
        "no such table: foo",
        "database disk image is malformed",
    ]:
        calls = {"n": 0}

        def reader():
            calls["n"] += 1
            raise sqlite3.OperationalError(non_transient)

        with patch("time.sleep"):
            rag._sql_read_with_retry(reader, "test_tag", default=None)

        assert calls["n"] == 1, f"{non_transient!r} should fail fast"


# ── Invariante: locked sigue reintentando (regression guard) ────────────────


def test_sql_write_retry_still_handles_database_locked():
    """El fix nuevo no debe romper el comportamiento original de
    reintentar `database is locked`. Post 2026-04-23: 8 attempts."""
    calls = {"n": 0}

    def writer():
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with patch("time.sleep"):
        rag._sql_write_with_retry(writer, "test_tag")

    assert calls["n"] == 8


def test_sql_read_retry_still_handles_database_locked():
    calls = {"n": 0}

    def reader():
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with patch("time.sleep"):
        rag._sql_read_with_retry(reader, "test_tag", default=None)

    assert calls["n"] == 5
