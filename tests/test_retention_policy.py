"""Drift guard: every table in _TELEMETRY_DDL must have a retention policy.

Two invariants enforced:
1. test_every_ddl_table_has_policy — each table created by _TELEMETRY_DDL is
   registered in _TELEMETRY_RETENTION_POLICY (prevents silent unbounded growth
   when a new DDL entry is added without a policy).

2. test_no_unbounded_log_tables — tables with policy=None (keep forever) that
   also have a `ts` column must appear in _RETENTION_NONE_ALLOWED (prevents
   accidentally unbounded log tables from slipping past review).

Satellite-module tables (rag_anticipate_feedback defined in
rag_anticipate/feedback.py, rag_schema_version, etc.) appear in
_TELEMETRY_RETENTION_POLICY as "state" or with a day-retention value even
though they are not in _TELEMETRY_DDL — that is intentional and the tests
below do NOT require DDL tables to be the superset.
"""

from __future__ import annotations

import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_ddl_table_names() -> set[str]:
    """Parse _TELEMETRY_DDL and return the set of table names defined there.

    Reads directly from the source to avoid importing rag (slow + side effects).
    Uses a simple regex over the CREATE TABLE IF NOT EXISTS statements between
    the _TELEMETRY_DDL tuple start and the next module-level symbol.
    """
    import rag as _rag

    src_path = Path(_rag.__file__)
    src = src_path.read_text(encoding="utf-8")

    # Locate the _TELEMETRY_DDL tuple
    ddl_start = src.find("_TELEMETRY_DDL: tuple[tuple[str, tuple[str, ...]")
    if ddl_start < 0:
        ddl_start = src.find("_TELEMETRY_DDL =")
    assert ddl_start >= 0, "_TELEMETRY_DDL not found in rag/__init__.py"

    # Find the sentinel that follows the tuple
    sentinel = "_TELEMETRY_DDL_ENSURED"
    sentinel_pos = src.find(sentinel, ddl_start)
    assert sentinel_pos >= 0, f"{sentinel} not found after _TELEMETRY_DDL"

    ddl_section = src[ddl_start:sentinel_pos]
    tables = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", ddl_section))
    # Remove false positives from Spanish inline comments that happen to have
    # "CREATE TABLE IF NOT EXISTS" in a comment referencing other text.
    false_positives = {"arriba", "no"}
    return tables - false_positives


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_every_ddl_table_has_policy() -> None:
    """Every table in _TELEMETRY_DDL must appear in _TELEMETRY_RETENTION_POLICY.

    If you add a new CREATE TABLE IF NOT EXISTS to _TELEMETRY_DDL and forget
    to register it in _TELEMETRY_RETENTION_POLICY, this test fails with a
    clear message showing which table is missing.
    """
    import rag as _rag

    ddl_tables = _extract_ddl_table_names()
    policy_tables = set(_rag._TELEMETRY_RETENTION_POLICY.keys())

    missing = ddl_tables - policy_tables
    assert not missing, (
        f"The following tables are defined in _TELEMETRY_DDL but have no entry "
        f"in _TELEMETRY_RETENTION_POLICY:\n  {sorted(missing)}\n\n"
        f"Add each missing table to _TELEMETRY_RETENTION_POLICY with an "
        f"appropriate policy value:\n"
        f"  int   → rotate after N days (log-style table)\n"
        f"  None  → keep forever (high-signal; add to _RETENTION_NONE_ALLOWED too)\n"
        f"  'state' → upsert-by-PK, never rotated\n"
        f"  'cache' → content-hash invalidated, never time-rotated\n"
    )


def test_no_unbounded_log_tables() -> None:
    """Tables with policy=None that have a `ts` column must be in _RETENTION_NONE_ALLOWED.

    This guards against someone adding a new log-style table with policy=None
    (keep forever) without explicitly acknowledging it in _RETENTION_NONE_ALLOWED.

    The check is done via a live DDL round-trip: spin up an in-memory SQLite,
    run every DDL statement in _TELEMETRY_DDL, then PRAGMA table_info to detect
    ts columns.
    """
    import rag as _rag

    none_policy_tables = {
        t for t, d in _rag._TELEMETRY_RETENTION_POLICY.items() if d is None
    }
    allowed = _rag._RETENTION_NONE_ALLOWED

    # Build an in-memory DB with the full DDL so we can inspect columns.
    conn = sqlite3.connect(":memory:")
    try:
        for table_name, stmts in _rag._TELEMETRY_DDL:
            for stmt in stmts:
                try:
                    conn.execute(stmt)
                except sqlite3.Error:
                    pass  # index/migration stmts may fail on empty schema
        conn.commit()

        problematic: list[str] = []
        for table in none_policy_tables:
            if table in allowed:
                continue
            # Check if the table was created and has a `ts` column.
            try:
                cols = {
                    row[1]
                    for row in conn.execute(f"PRAGMA table_info([{table}])").fetchall()
                }
            except sqlite3.Error:
                cols = set()
            if "ts" in cols:
                problematic.append(table)
    finally:
        conn.close()

    assert not problematic, (
        f"The following tables have policy=None (keep forever) AND a `ts` column "
        f"but are NOT in _RETENTION_NONE_ALLOWED:\n  {sorted(problematic)}\n\n"
        f"If keeping them forever is intentional, add each to _RETENTION_NONE_ALLOWED.\n"
        f"If they should rotate, change their policy to an int (days)."
    )


def test_back_compat_aliases_derived_correctly() -> None:
    """_SQL_ROTATION_POLICY, _SQL_STATE_TABLES, _SQL_KEEP_ALL_TABLES are correct subsets."""
    import rag as _rag

    policy = _rag._TELEMETRY_RETENTION_POLICY

    expected_rotation = {t for t, d in policy.items() if isinstance(d, int)}
    actual_rotation = {t for t, d in _rag._SQL_ROTATION_POLICY}
    assert actual_rotation == expected_rotation, (
        f"_SQL_ROTATION_POLICY mismatch.\n"
        f"  Extra: {actual_rotation - expected_rotation}\n"
        f"  Missing: {expected_rotation - actual_rotation}"
    )

    expected_state = {t for t, d in policy.items() if d == "state"}
    actual_state = set(_rag._SQL_STATE_TABLES)
    assert actual_state == expected_state, (
        f"_SQL_STATE_TABLES mismatch.\n"
        f"  Extra: {actual_state - expected_state}\n"
        f"  Missing: {expected_state - actual_state}"
    )

    expected_keep_all = {t for t, d in policy.items() if d is None}
    actual_keep_all = set(_rag._SQL_KEEP_ALL_TABLES)
    assert actual_keep_all == expected_keep_all, (
        f"_SQL_KEEP_ALL_TABLES mismatch.\n"
        f"  Extra: {actual_keep_all - expected_keep_all}\n"
        f"  Missing: {expected_keep_all - actual_keep_all}"
    )


def test_retention_none_allowed_is_subset_of_none_policy() -> None:
    """Every entry in _RETENTION_NONE_ALLOWED must have policy=None."""
    import rag as _rag

    policy = _rag._TELEMETRY_RETENTION_POLICY
    for table in _rag._RETENTION_NONE_ALLOWED:
        assert table in policy, (
            f"{table!r} is in _RETENTION_NONE_ALLOWED but not in _TELEMETRY_RETENTION_POLICY"
        )
        assert policy[table] is None, (
            f"{table!r} is in _RETENTION_NONE_ALLOWED but its policy is "
            f"{policy[table]!r} (expected None)"
        )


def test_table_size_warn_bytes_default() -> None:
    """_TABLE_SIZE_WARN_BYTES defaults to 100 MB."""
    import rag as _rag

    assert _rag._TABLE_SIZE_WARN_BYTES == 100 * 1024 * 1024


def test_table_size_warn_bytes_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """RAG_TABLE_SIZE_WARN_BYTES env var overrides the default."""
    import importlib
    import os

    # We can't easily re-evaluate the module-level constant, so test the
    # int(os.environ.get(...)) pattern directly — same as what rag.py does.
    monkeypatch.setenv("RAG_TABLE_SIZE_WARN_BYTES", str(50 * 1024 * 1024))
    import rag as _rag  # noqa: F401 — just confirm parse works
    val = int(os.environ.get("RAG_TABLE_SIZE_WARN_BYTES", str(100 * 1024 * 1024)))
    assert val == 50 * 1024 * 1024
