#!/usr/bin/env python3
"""One-shot migration: split ragvec.db into ragvec.db (vec/meta) + telemetry.db (rag_*).

PROBLEM
-------
ragvec.db (206 MB) shares a single WAL between two radically different workloads:
  - Bulk-write indexer: meta_* + vec_* tables, holds long write transactions
  - Hot-path telemetry: rag_* tables, written on every query/chat/CLI call

Contention produces "database is locked" errors on the telemetry path.

SOLUTION
--------
Move all rag_* + system_memory_metrics tables to a separate telemetry.db in the
same directory. Each file gets its own WAL; the indexer's long transactions no
longer block telemetry writes.

PRECONDITIONS
-------------
1. Stop all obsidian-rag launchd services before running:
       launchctl unload ~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist
   Or pass --force to skip this check.
2. ragvec.db must exist and be > 0 bytes.
3. telemetry.db must NOT exist (or pass --force to backup + overwrite).

NOTE: ragvec.db must be in WAL journal mode (it always is in production).
      If it isn't, the script aborts — something is wrong with the setup.

ROLLBACK (30-day window)
------------------------
This script does NOT drop or truncate anything in ragvec.db. The original
rag_* tables remain there as a fallback. To roll back: delete telemetry.db
and revert any rag.py connection-routing change. Tables in ragvec.db are
untouched.

EXIT CODES
----------
0  success or --dry-run OK
1  preflight failure (services running / ragvec.db missing / telemetry.db exists)
2  row-count mismatch during copy (aborted + rolled back)
3  other error (journal mode wrong, disk full, DB corrupt, etc.)
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
import rag  # noqa: E402

# DB_PATH is the *directory* ~/.local/share/obsidian-rag/ragvec
DEFAULT_SOURCE_DIR = rag.DB_PATH.parent  # ~/.local/share/obsidian-rag

# Source of truth for all telemetry tables.
# system_memory_metrics IS included here (rag.py:3763) — the spec note
# suggesting it might not be was written before verification.
_TELEMETRY_TABLES: list[str] = [t[0] for t in rag._TELEMETRY_DDL]

# Ingester state tables that MUST stay in ragvec.db.
#
# scripts/ingest_{whatsapp,gmail,calendar,reminders}.py open their own
# sqlite3.connect(rag.DB_PATH / "ragvec.db") directly — they do NOT go
# through rag._ragvec_state_conn(). Moving these tables to telemetry.db
# while the ingester code still points at ragvec.db would:
#   - silently re-create empty tables in ragvec.db on the next ingest run
#     (CREATE TABLE IF NOT EXISTS), losing the incremental cursor
#   - force a full re-scan: 12990 WA messages + a 2-year Google Calendar
#     bootstrap triggered by a missing sync_token
#
# Their write frequency is ~1 row per 5 min — negligible WAL contention.
# The actual contention source is vec_* + meta_* bulk writes during indexing,
# which also live in ragvec.db (same side, no cross-DB conflict with telemetry).
_INGESTER_STATE_STAY_IN_RAGVEC: frozenset[str] = frozenset({
    "rag_whatsapp_state",
    "rag_gmail_state",
    "rag_calendar_state",
    "rag_reminders_state",
})

# rag_schema_version exists in both DBs with incompatible key spaces:
#   ragvec.db    → collection versions + backfill sentinels (SqliteVecClient)
#   telemetry.db → migration sentinels (_split_v1, etc.)
# Decision: do NOT copy it. _ensure_telemetry_tables() creates it empty in
# telemetry.db; only _split_v1 is added at the end of this script. This avoids
# merging two incompatible sentinel namespaces.
_SKIP_COPY: frozenset[str] = frozenset({"rag_schema_version"}) | _INGESTER_STATE_STAY_IN_RAGVEC


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class TableResult:
    name: str
    src_rows: int = 0
    dst_rows: int = 0
    status: str = "pending"   # ok | skip-missing | skip-excluded | mismatch | error | dry-run
    note: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_running_services() -> list[str]:
    """Return list of matching PIDs. Empty = no services running."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "com.fer.obsidian-rag"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    if out.returncode != 0:
        return []
    return [p for p in out.stdout.split() if p.strip()]


def _table_exists_in(conn: sqlite3.Connection, table: str, schema: str = "main") -> bool:
    master = f"{schema}.sqlite_master"
    row = conn.execute(
        f"SELECT 1 FROM {master} WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _copy_table(
    conn: sqlite3.Connection, table: str, verbose: bool
) -> TableResult:
    """Copy one table from main → tdb in its own BEGIN IMMEDIATE transaction.

    conn must already have tdb ATTACHed.

    Column mapping strategy (handles schema drift between source and dest):
      - For each column in tdb:
          * present in source   → SELECT that column from source
          * has a DDL default   → use that literal as the SELECT expression
          * NOT NULL, no default → type-based sentinel ('' / 0 / 0.0)
          * nullable, no default → NULL
    In production source and dest have identical schemas (both created by
    _TELEMETRY_DDL), so every column maps 1:1 and no sentinels are needed.

    Uses INSERT OR IGNORE to survive pre-existing rows (e.g. version rows
    inserted by _ensure_telemetry_tables into rag_schema_version).
    Rolls back + returns status='mismatch' if src_rows != dst_rows after insert.
    """
    r = TableResult(name=table)

    if not _table_exists_in(conn, table, "main"):
        r.status = "skip-missing"
        r.note = "not in source"
        return r

    r.src_rows = conn.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]

    src_col_names = {
        row[1]
        for row in conn.execute(f"PRAGMA main.table_info({table})").fetchall()
    }
    # tdb table_info rows: (cid, name, type, notnull, dflt_value, pk)
    tdb_col_info = conn.execute(f"PRAGMA tdb.table_info({table})").fetchall()

    insert_cols: list[str] = []
    select_exprs: list[str] = []
    for _, col_name, col_type, notnull, dflt_value, _ in tdb_col_info:
        insert_cols.append(col_name)
        if col_name in src_col_names:
            select_exprs.append(col_name)
        elif dflt_value is not None:
            select_exprs.append(str(dflt_value))
        elif notnull:
            base = col_type.upper().split("(")[0].strip()
            if base in ("INTEGER", "INT", "BIGINT", "SMALLINT"):
                select_exprs.append("0")
            elif base in ("REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL"):
                select_exprs.append("0.0")
            else:
                select_exprs.append("''")
        else:
            select_exprs.append("NULL")

    if not insert_cols:
        r.status = "error"
        r.note = "destination table has no columns"
        return r

    col_list = ", ".join(insert_cols)
    sel_list = ", ".join(select_exprs)
    try:
        conn.execute("BEGIN IMMEDIATE")
        sql = (
            f"INSERT OR IGNORE INTO tdb.{table} ({col_list})"
            f" SELECT {sel_list} FROM main.{table}"
        )
        if verbose:
            print(f"    SQL: {sql}")
        conn.execute(sql)
        r.dst_rows = conn.execute(f"SELECT COUNT(*) FROM tdb.{table}").fetchone()[0]
        if r.src_rows != r.dst_rows:
            conn.execute("ROLLBACK")
            r.status = "mismatch"
            r.note = f"src={r.src_rows} dst={r.dst_rows}"
            return r
        conn.execute("COMMIT")
        r.status = "ok"
    except sqlite3.Error as exc:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        r.status = "error"
        r.note = str(exc)

    return r


def _print_summary(results: list[TableResult]) -> None:
    if not results:
        return
    w = max(len(r.name) for r in results)
    header = f"{'Table':<{w}}  {'Src rows':>10}  {'Dst rows':>10}  {'Status':<14}  Note"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<{w}}  {r.src_rows:>10}  {r.dst_rows:>10}"
            f"  {r.status:<14}  {r.note}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Split ragvec.db: move rag_* telemetry tables to telemetry.db",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, metavar="PATH",
        help="Directory containing ragvec/ subdir (default: %(default)s)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Don't modify anything — print the plan + source row counts",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Override preflight checks (pgrep) and telemetry.db existence",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print every SQL statement executed",
    )
    args = ap.parse_args(argv)

    source_dir: Path = args.source_dir
    db_dir       = source_dir / "ragvec"
    ragvec_db    = db_dir / "ragvec.db"
    telemetry_db = db_dir / "telemetry.db"

    # ── Preflight ─────────────────────────────────────────────────────────────

    pids = _check_running_services()
    if pids and not args.force:
        print(
            "ERROR: obsidian-rag services are running. Stop them first:\n"
            "  launchctl unload ~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist\n"
            f"\nFound PIDs: {', '.join(pids)}\n"
            "\nPass --force to skip this check.",
            file=sys.stderr,
        )
        return 1

    if not ragvec_db.exists() or ragvec_db.stat().st_size == 0:
        print(f"ERROR: {ragvec_db} does not exist or is empty.", file=sys.stderr)
        return 1

    if telemetry_db.exists() and not args.dry_run:
        if not args.force:
            print(
                f"ERROR: {telemetry_db} already exists.\n"
                "Pass --force to back it up and overwrite.",
                file=sys.stderr,
            )
            return 1
        bak = telemetry_db.with_suffix(f".db.bak.{int(time.time())}")
        print(f"[force] Backing up existing telemetry.db → {bak.name}")
        os.replace(telemetry_db, bak)

    # ── Open ragvec.db (autocommit for explicit transaction control) ───────────

    try:
        conn_main = sqlite3.connect(
            str(ragvec_db), timeout=30.0, isolation_level=None
        )
    except sqlite3.Error as exc:
        print(f"ERROR: cannot open {ragvec_db}: {exc}", file=sys.stderr)
        return 3

    conn_main.execute("PRAGMA busy_timeout=60000")

    jmode = conn_main.execute("PRAGMA journal_mode").fetchone()[0]
    if jmode != "wal":
        print(
            f"ERROR: {ragvec_db} is in journal_mode={jmode!r}, expected 'wal'.\n"
            "The production DB is always WAL — something is wrong. Aborting.",
            file=sys.stderr,
        )
        conn_main.close()
        return 3

    if args.verbose:
        print(f"journal_mode={jmode!r}  ✓")

    # ── WAL checkpoint TRUNCATE ────────────────────────────────────────────────

    if not args.dry_run:
        ckpt = conn_main.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        busy, log_pages, ckpt_pages = ckpt
        print(
            f"WAL checkpoint: busy={busy}  log_pages={log_pages}"
            f"  ckpt_pages={ckpt_pages}"
        )
        if busy != 0:
            print(
                f"  WARNING: {busy} writer(s) were busy during checkpoint — "
                "pgrep passed so we continue."
            )

    # ── Build tables_to_copy list ──────────────────────────────────────────────

    # _TELEMETRY_TABLES already includes system_memory_metrics.
    # _SKIP_COPY excludes rag_schema_version + the 4 ingester state tables
    # that must stay in ragvec.db (ingest_*.py connect directly to ragvec.db).
    tables_to_copy: list[str] = [
        t for t in _TELEMETRY_TABLES if t not in _SKIP_COPY
    ]

    # ── Dry-run: print plan + row counts, exit ────────────────────────────────

    if args.dry_run:
        print("DRY-RUN")
        print(f"  source : {ragvec_db}  ({ragvec_db.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  dest   : {telemetry_db}  (will be created)")
        print(f"  tables : {len(tables_to_copy)} to migrate,"
              f" {len(_INGESTER_STATE_STAY_IN_RAGVEC)} stay in ragvec.db\n")
        results: list[TableResult] = []
        for t in tables_to_copy:
            if not _table_exists_in(conn_main, t):
                results.append(TableResult(name=t, status="skip-missing", note="not in source"))
            else:
                n = conn_main.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                results.append(TableResult(name=t, src_rows=n, status="dry-run"))
        for t in sorted(_INGESTER_STATE_STAY_IN_RAGVEC):
            results.append(TableResult(name=t, status="skip-stays-ragvec",
                                       note="ingest_*.py writes direct to ragvec.db"))
        _print_summary(results)
        conn_main.close()
        return 0

    # ── Create telemetry.db + schema ───────────────────────────────────────────

    print(f"\nCreating {telemetry_db} ...")

    try:
        conn_tdb = sqlite3.connect(str(telemetry_db), timeout=30.0)
    except sqlite3.Error as exc:
        print(f"ERROR: cannot create {telemetry_db}: {exc}", file=sys.stderr)
        conn_main.close()
        return 3

    conn_tdb.execute("PRAGMA busy_timeout=60000")
    conn_tdb.execute("PRAGMA journal_mode=WAL")
    conn_tdb.execute("PRAGMA synchronous=NORMAL")

    # rag_schema_version must exist before _ensure_telemetry_tables() runs
    # (it uses INSERT OR IGNORE into it to register table version rows).
    conn_tdb.execute(
        "CREATE TABLE IF NOT EXISTS rag_schema_version"
        " (table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
    )
    conn_tdb.commit()

    rag._ensure_telemetry_tables(conn_tdb)

    # Verify sentinel table was created
    if not conn_tdb.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rag_schema_version'"
    ).fetchone():
        print(
            "ERROR: rag_schema_version not found in telemetry.db after "
            "_ensure_telemetry_tables — aborting.",
            file=sys.stderr,
        )
        conn_tdb.close()
        conn_main.close()
        return 3

    conn_tdb.commit()
    conn_tdb.close()

    # ── ATTACH telemetry.db + copy tables ─────────────────────────────────────

    attach_sql = f"ATTACH DATABASE '{telemetry_db}' AS tdb"
    if args.verbose:
        print(f"  SQL: {attach_sql}")
    conn_main.execute(attach_sql)

    results = []
    had_error = False

    for table in tables_to_copy:
        if args.verbose:
            print(f"  Copying {table} ...")

        r = _copy_table(conn_main, table, args.verbose)
        results.append(r)

        if r.status == "ok":
            print(f"  ✓ {table:<40}  {r.src_rows:>8} rows")
        elif r.status == "skip-missing":
            print(f"  - {table:<40}  skip ({r.note})")
        elif r.status == "mismatch":
            print(
                f"\nERROR: row-count mismatch for '{table}': "
                f"src={r.src_rows} dst={r.dst_rows} — aborted + rolled back.",
                file=sys.stderr,
            )
            had_error = True
            break
        else:
            print(f"  ✗ {table:<40}  error: {r.note}", file=sys.stderr)
            had_error = True
            break

    if had_error:
        conn_main.execute("DETACH DATABASE tdb")
        conn_main.close()
        return 2

    # Add the 4 ingester state tables to the summary as explicit skips so the
    # output makes clear they were intentionally left in ragvec.db.
    for t in sorted(_INGESTER_STATE_STAY_IN_RAGVEC):
        results.append(TableResult(
            name=t, status="skip-stays-ragvec",
            note="ingest_*.py writes direct to ragvec.db",
        ))
        print(f"  ~ {t:<40}  stays in ragvec.db")

    # ── Write _split_v1 sentinel ───────────────────────────────────────────────

    conn_main.execute("BEGIN")
    conn_main.execute(
        "INSERT OR REPLACE INTO tdb.rag_schema_version"
        " (table_name, version) VALUES ('_split_v1', 1)"
    )
    conn_main.execute("COMMIT")
    if args.verbose:
        print("  Wrote sentinel _split_v1 to tdb.rag_schema_version")

    # ── Detach + close ─────────────────────────────────────────────────────────

    conn_main.execute("DETACH DATABASE tdb")
    conn_main.close()

    # ── Summary ────────────────────────────────────────────────────────────────

    print()
    _print_summary(results)
    print()

    ragvec_mb = ragvec_db.stat().st_size / 1024 / 1024
    tdb_mb    = telemetry_db.stat().st_size / 1024 / 1024
    print(f"ragvec.db   : {ragvec_db}  ({ragvec_mb:.1f} MB)")
    print(f"telemetry.db: {telemetry_db}  ({tdb_mb:.1f} MB)")
    print()

    # Verify sentinel
    conn_v = sqlite3.connect(str(telemetry_db))
    row = conn_v.execute(
        "SELECT table_name, version FROM rag_schema_version WHERE table_name='_split_v1'"
    ).fetchone()
    conn_v.close()
    if row:
        print(f"Sentinel OK: {row}")
    else:
        print("WARNING: sentinel _split_v1 not found in telemetry.db!", file=sys.stderr)

    copied = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skip-missing")
    print(f"\nDone: {copied} tables copied, {skipped} skipped (not in source).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
