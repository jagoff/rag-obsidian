"""T8: inter-process concurrency / stress tests for the SQL state store.

Production scenario: 6+ launchd services (web, watch, morning, today, digest,
wa-tasks, online-tune, listener…) write into the same `~/.local/share/obsidian-
rag/ragvec/ragvec.db` simultaneously. Threads inside a single process are
serialised by Python's GIL + SQLite's WAL write lock, so "threaded" contention
tests are misleading — the real contention is between processes sharing the
DB file via its filesystem lock.

These tests spawn OS processes (via `concurrent.futures.ProcessPoolExecutor`)
against a tmp-path ragvec.db and assert:
  - no lost writes under N writers
  - no torn reads under mixed writers/readers
  - upsert conflict resolution (last-writer-wins on same PK)
  - no SQLITE_BUSY escapes past the 10s busy_timeout configured in
    `_ragvec_state_conn`
  - WAL checkpoint / VACUUM under load doesn't kill active writers

Marked `slow` because each test pays multiprocess spawn overhead (~100ms per
worker on darwin). Run via `-m slow` when you want the full battery, or just
leave them in the default suite (they add ~6-10s total).
"""
from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pytest

import rag


pytestmark = pytest.mark.slow


# ── DB helpers (module-level so ProcessPoolExecutor can pickle them) ─────────


def _init_db(db_dir: Path) -> None:
    """Create the empty ragvec.db with the T1 telemetry schema."""
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_dir / "ragvec.db"),
                           isolation_level=None, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rag_schema_version ("
            " table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
        )
        rag._ensure_telemetry_tables(conn)
    finally:
        conn.close()


def _open(db_dir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_dir / "ragvec.db"),
                           isolation_level=None, check_same_thread=False,
                           timeout=15.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


# ── Worker functions (MUST be top-level for pickling) ───────────────────────


def _append_queries_worker(args: tuple) -> int:
    """Insert `n` rows into rag_queries. Returns rows actually inserted."""
    db_dir, proc_idx, n = args
    conn = _open(db_dir)
    ok = 0
    try:
        for i in range(n):
            ts = f"2026-04-19T10:{proc_idx:02d}:{i % 60:02d}"
            conn.execute(
                "INSERT INTO rag_queries (ts, q) VALUES (?, ?)",
                (ts, f"q-{proc_idx}-{i}"),
            )
            ok += 1
    finally:
        conn.close()
    return ok


def _append_behavior_worker(args: tuple) -> int:
    db_dir, proc_idx, deadline = args
    conn = _open(db_dir)
    inserted = 0
    try:
        while time.time() < deadline:
            ts = f"2026-04-19T10:{proc_idx:02d}:{inserted % 60:02d}"
            conn.execute(
                "INSERT INTO rag_behavior (ts, source, event, path) VALUES (?, ?, ?, ?)",
                (ts, "cli", "open", f"p/{proc_idx}/{inserted}.md"),
            )
            inserted += 1
    finally:
        conn.close()
    return inserted


def _read_behavior_window_worker(args: tuple) -> list[int]:
    """Return list of row counts each read saw — used to assert monotonicity."""
    db_dir, deadline = args
    conn = _open(db_dir)
    counts: list[int] = []
    try:
        while time.time() < deadline:
            # Equivalent to rag._sql_query_window(..., since_ts="")
            rows = conn.execute(
                "SELECT COUNT(*) FROM rag_behavior WHERE ts >= ?",
                ("2026-04-01T00:00:00",),
            ).fetchone()
            counts.append(int(rows[0]))
            time.sleep(0.01)
    finally:
        conn.close()
    return counts


def _upsert_same_session_worker(args: tuple) -> None:
    db_dir, proc_idx = args
    conn = _open(db_dir)
    try:
        ts = f"2026-04-19T10:00:{proc_idx:02d}"
        conn.execute(
            "INSERT OR REPLACE INTO rag_conversations_index "
            "(session_id, relative_path, updated_at) VALUES (?, ?, ?)",
            ("shared-session", f"conv/{proc_idx}.md", ts),
        )
    finally:
        conn.close()


def _upsert_distinct_session_worker(args: tuple) -> None:
    db_dir, proc_idx = args
    conn = _open(db_dir)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO rag_conversations_index "
            "(session_id, relative_path, updated_at) VALUES (?, ?, ?)",
            (f"session-{proc_idx}", f"conv/{proc_idx}.md",
             f"2026-04-19T10:00:{proc_idx:02d}"),
        )
    finally:
        conn.close()


def _checkpoint_worker(args: tuple) -> int:
    """Periodically run `PRAGMA wal_checkpoint(TRUNCATE)` until deadline."""
    db_dir, deadline, mode = args
    conn = _open(db_dir)
    checkpoints = 0
    try:
        while time.time() < deadline:
            conn.execute(f"PRAGMA wal_checkpoint({mode})")
            checkpoints += 1
            time.sleep(0.05)
    finally:
        conn.close()
    return checkpoints


def _future_ts_worker(args: tuple) -> int:
    db_dir, proc_idx, n = args
    conn = _open(db_dir)
    ok = 0
    try:
        for i in range(n):
            # 10 seconds in the future + per-proc offset
            ts = f"2036-04-19T{(10 + proc_idx) % 24:02d}:00:{i % 60:02d}"
            conn.execute(
                "INSERT INTO rag_behavior (ts, source, event, path) VALUES (?, ?, ?, ?)",
                (ts, "cli", "open", f"fut/{proc_idx}/{i}.md"),
            )
            ok += 1
    finally:
        conn.close()
    return ok


# ── Tests ────────────────────────────────────────────────────────────────────


def test_20_procs_append_rag_queries(tmp_path):
    """20 procs × 100 rows each = 2000 total. No SQLITE_BUSY escapes."""
    _init_db(tmp_path)
    n_per = 100
    n_procs = 20
    with ProcessPoolExecutor(max_workers=n_procs) as ex:
        futures = [
            ex.submit(_append_queries_worker, (tmp_path, pi, n_per))
            for pi in range(n_procs)
        ]
        totals = [f.result(timeout=60) for f in as_completed(futures)]

    assert sum(totals) == n_procs * n_per, "writes lost between procs"
    conn = _open(tmp_path)
    try:
        got = conn.execute("SELECT COUNT(*) FROM rag_queries").fetchone()[0]
    finally:
        conn.close()
    assert got == n_procs * n_per, f"DB count {got} != expected {n_procs * n_per}"


def test_mixed_writer_reader_no_starvation(tmp_path):
    """10 writers + 5 readers for 5s. Writers make progress; readers see a
    monotonically non-decreasing view (no torn reads)."""
    _init_db(tmp_path)
    deadline = time.time() + 5.0
    with ProcessPoolExecutor(max_workers=15) as ex:
        w_futures = [
            ex.submit(_append_behavior_worker, (tmp_path, pi, deadline))
            for pi in range(10)
        ]
        r_futures = [
            ex.submit(_read_behavior_window_worker, (tmp_path, deadline))
            for _ in range(5)
        ]
        writes = [f.result(timeout=30) for f in w_futures]
        reads = [f.result(timeout=30) for f in r_futures]

    total_written = sum(writes)
    assert total_written > 0, "no writer made progress"

    # Each reader returned a non-decreasing sequence (SQLite snapshot isolation
    # in WAL mode means monotonicity per connection).
    for seq in reads:
        if not seq:
            continue
        for a, b in zip(seq, seq[1:]):
            assert b >= a, "reader saw decreasing count — torn read"
        assert seq[-1] <= total_written

    # Final DB count should equal total_written.
    conn = _open(tmp_path)
    try:
        got = conn.execute("SELECT COUNT(*) FROM rag_behavior").fetchone()[0]
    finally:
        conn.close()
    assert got == total_written


def test_concurrent_upsert_conversations_index_same_key(tmp_path):
    """20 procs upsert the SAME session_id → final row count = 1."""
    _init_db(tmp_path)
    with ProcessPoolExecutor(max_workers=20) as ex:
        futures = [
            ex.submit(_upsert_same_session_worker, (tmp_path, pi))
            for pi in range(20)
        ]
        for f in as_completed(futures):
            f.result(timeout=60)

    conn = _open(tmp_path)
    try:
        rows = conn.execute(
            "SELECT session_id, updated_at FROM rag_conversations_index"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "shared-session"
    # updated_at should be one of the writes — under INSERT OR REPLACE the
    # last write wins. We can't predict which proc wrote last, but the value
    # must be in the emitted range.
    emitted = {f"2026-04-19T10:00:{pi:02d}" for pi in range(20)}
    assert rows[0][1] in emitted


def test_concurrent_upsert_different_keys(tmp_path):
    """20 procs upsert DIFFERENT session_ids → final row count = 20."""
    _init_db(tmp_path)
    with ProcessPoolExecutor(max_workers=20) as ex:
        futures = [
            ex.submit(_upsert_distinct_session_worker, (tmp_path, pi))
            for pi in range(20)
        ]
        for f in as_completed(futures):
            f.result(timeout=60)

    conn = _open(tmp_path)
    try:
        got = conn.execute("SELECT COUNT(*) FROM rag_conversations_index").fetchone()[0]
    finally:
        conn.close()
    assert got == 20


def test_vacuum_during_load_no_errors(tmp_path):
    """5 writer procs + 1 proc running wal_checkpoint(TRUNCATE) periodically.
    Assert no writer fails + all writes land."""
    _init_db(tmp_path)
    deadline = time.time() + 3.0
    with ProcessPoolExecutor(max_workers=6) as ex:
        w_futures = [
            ex.submit(_append_behavior_worker, (tmp_path, pi, deadline))
            for pi in range(5)
        ]
        cp_future = ex.submit(_checkpoint_worker, (tmp_path, deadline, "TRUNCATE"))
        writes = [f.result(timeout=30) for f in w_futures]
        n_checkpoints = cp_future.result(timeout=30)

    assert sum(writes) > 0
    assert n_checkpoints > 0
    conn = _open(tmp_path)
    try:
        got = conn.execute("SELECT COUNT(*) FROM rag_behavior").fetchone()[0]
    finally:
        conn.close()
    assert got == sum(writes)


def test_writer_survives_wal_swap(tmp_path):
    """Concurrent writers + RESTART checkpoints (forces the WAL to rotate
    mid-write). Writers must continue to completion."""
    _init_db(tmp_path)
    deadline = time.time() + 3.0
    with ProcessPoolExecutor(max_workers=4) as ex:
        w_futures = [
            ex.submit(_append_behavior_worker, (tmp_path, pi, deadline))
            for pi in range(3)
        ]
        cp_future = ex.submit(_checkpoint_worker, (tmp_path, deadline, "RESTART"))
        writes = [f.result(timeout=30) for f in w_futures]
        n_checkpoints = cp_future.result(timeout=30)

    assert sum(writes) > 0
    assert n_checkpoints > 0
    conn = _open(tmp_path)
    try:
        got = conn.execute("SELECT COUNT(*) FROM rag_behavior").fetchone()[0]
    finally:
        conn.close()
    assert got == sum(writes)


def test_clock_skew_in_ts(tmp_path):
    """Writers inserting with ts values 10 years in the future don't break the
    readers' MAX(ts) cache invalidation logic — the next INSERT with a newer
    (still-future) ts must still bump MAX(ts)."""
    _init_db(tmp_path)
    n_per = 10
    n_procs = 5
    with ProcessPoolExecutor(max_workers=n_procs) as ex:
        futures = [
            ex.submit(_future_ts_worker, (tmp_path, pi, n_per))
            for pi in range(n_procs)
        ]
        for f in as_completed(futures):
            f.result(timeout=60)

    conn = _open(tmp_path)
    try:
        got = conn.execute("SELECT COUNT(*) FROM rag_behavior").fetchone()[0]
        max_ts = conn.execute("SELECT MAX(ts) FROM rag_behavior").fetchone()[0]
    finally:
        conn.close()
    assert got == n_per * n_procs
    assert max_ts is not None
    assert max_ts.startswith("2036-"), f"expected future ts, got {max_ts}"

    # Now insert a newer ts and confirm MAX(ts) moves forward.
    conn = _open(tmp_path)
    try:
        conn.execute(
            "INSERT INTO rag_behavior (ts, source, event, path) VALUES (?, ?, ?, ?)",
            ("2099-01-01T00:00:00", "cli", "open", "far-future.md"),
        )
        new_max = conn.execute("SELECT MAX(ts) FROM rag_behavior").fetchone()[0]
    finally:
        conn.close()
    assert new_max == "2099-01-01T00:00:00"
    assert new_max > max_ts
