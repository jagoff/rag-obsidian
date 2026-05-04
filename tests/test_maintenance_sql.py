"""T7: SQL-aware maintenance rotation + `.bak.<ts>` cleanup + rollback.

Covers:
  - `_sql_rotate_log_tables` — per-table retention, conditional VACUUM, flag gate.
  - `_cleanup_bak_files` — purge `.bak.<unix_ts>` files older than 30 days.
  - `_rollback_state_migration` — restore baks + DROP rag_* tables.
  - `run_maintenance` integration (post-T10 SQL-only, RAG_STATE_SQL removed).

Never touches the live DB — every fixture points DB_PATH at tmp_path and
redirects Path.home() via monkeypatch so the state_dir lookups land in a
throwaway directory.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pytest

import rag


# ── Fixtures ────────────────────────────────────────────────────────────────

def _open_db(path: Path) -> sqlite3.Connection:
    """Open + apply the T1 DDL so the test can populate rag_* tables."""
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_schema_version ("
        " table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
    )
    rag._ensure_telemetry_tables(conn)
    return conn


@pytest.fixture
def sql_env(tmp_path, monkeypatch):
    """Flag ON + DB_PATH + fake HOME so state_dir lands in tmp_path."""
    home = tmp_path / "home"
    state_dir = home / ".local/share/obsidian-rag"
    state_dir.mkdir(parents=True, exist_ok=True)
    db_dir = state_dir / "ragvec"
    db_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(rag, "DB_PATH", db_dir)
    monkeypatch.setenv("HOME", str(home))
    # Path.home() reads $HOME on POSIX so the env override is enough, but
    # some helpers cache Path.home() at import time. `_cleanup_bak_files`
    # computes state_dir at call time, so no extra patching required.
    # Initialise telemetry.db so _sql_rotate_log_tables / _rollback_state_migration
    # find the tables via _ragvec_state_conn().
    conn = _open_db(db_dir / rag._TELEMETRY_DB_FILENAME)
    conn.close()
    yield {"home": home, "state_dir": state_dir, "db_dir": db_dir,
            "db_path": db_dir / rag._TELEMETRY_DB_FILENAME}


def _seed_rows(conn: sqlite3.Connection, table: str, tss: list[str],
                extra_cols: dict | None = None) -> None:
    """Insert len(tss) rows into `table` with the given ts values.

    For tables with additional NOT NULL columns (queries.q, behavior.source +
    event, feedback.rating), caller supplies `extra_cols` with defaults.
    """
    extra_cols = extra_cols or {}
    cols = ["ts"] + list(extra_cols.keys())
    placeholders = ",".join("?" for _ in cols)
    sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    for ts in tss:
        vals = [ts] + [extra_cols[k] for k in extra_cols]
        conn.execute(sql, vals)
    conn.commit()


def _iso_days_ago(days: int, now: float | None = None) -> str:
    base = now if now is not None else time.time()
    return datetime.fromtimestamp(base - days * 86400).isoformat(timespec="seconds")


# ── Rotation ────────────────────────────────────────────────────────────────

def test_rotation_deletes_old_rows(sql_env):
    """rag_queries: seed rows at day 0, 30, 60, 120 → retention 90d deletes
    only the 120-day row. Keeps the semantics unambiguous around the cutoff."""
    conn = sqlite3.connect(str(sql_env["db_path"]))
    try:
        now = time.time()
        # Rows at day 0 (today), 30, 60, 120 ago.
        tss = [_iso_days_ago(d, now) for d in (0, 30, 60, 120)]
        _seed_rows(conn, "rag_queries", tss, extra_cols={"q": "seed"})
        assert conn.execute("SELECT COUNT(*) FROM rag_queries").fetchone()[0] == 4
    finally:
        conn.close()

    out = rag._sql_rotate_log_tables(dry_run=False, now_ts=now)
    # Only the 120-day row is strictly older than the 90-day cutoff.
    assert out["rows_deleted"]["rag_queries"] == 1

    conn = sqlite3.connect(str(sql_env["db_path"]))
    try:
        assert conn.execute("SELECT COUNT(*) FROM rag_queries").fetchone()[0] == 3
        # Newest row still present.
        assert conn.execute(
            "SELECT COUNT(*) FROM rag_queries WHERE ts >= ?", (_iso_days_ago(1, now),)
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_rotation_different_retentions_per_table(sql_env):
    """Assert each policy bucket deletes the expected count and keep-all
    tables are untouched."""
    conn = sqlite3.connect(str(sql_env["db_path"]))
    try:
        now = time.time()
        # Seed queries (90d retention) with 100d + 50d rows → 1 deletes.
        _seed_rows(conn, "rag_queries",
                    [_iso_days_ago(100, now), _iso_days_ago(50, now)],
                    {"q": "s"})
        # Seed memory (30d retention) with 40d + 10d rows → 1 deletes.
        _seed_rows(conn, "rag_memory_metrics",
                    [_iso_days_ago(40, now), _iso_days_ago(10, now)])
        # Seed feedback (KEEP ALL) with an ancient 500d row — must survive.
        _seed_rows(conn, "rag_feedback",
                    [_iso_days_ago(500, now)], {"rating": 1})
        # Seed brief_written (60d retention) with 70d + 40d rows → 1 deletes.
        _seed_rows(conn, "rag_brief_written",
                    [_iso_days_ago(70, now), _iso_days_ago(40, now)],
                    {"brief_type": "morning"})
    finally:
        conn.close()

    out = rag._sql_rotate_log_tables(dry_run=False, now_ts=now)
    assert out["rows_deleted"]["rag_queries"] == 1
    assert out["rows_deleted"]["rag_memory_metrics"] == 1
    assert out["rows_deleted"]["rag_brief_written"] == 1
    # feedback is not in the rotation policy at all
    assert "rag_feedback" not in out["rows_deleted"]

    conn = sqlite3.connect(str(sql_env["db_path"]))
    try:
        assert conn.execute("SELECT COUNT(*) FROM rag_feedback").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM rag_tune").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM rag_contradictions").fetchone()[0] == 0
        # Survivors for rotated tables: only the within-retention row.
        assert conn.execute("SELECT COUNT(*) FROM rag_queries").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM rag_memory_metrics").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM rag_brief_written").fetchone()[0] == 1
    finally:
        conn.close()




class _DummyCol:
    """Minimal stand-in for a SqliteVecCollection used by legacy maintenance
    steps (context-cache pruning, url pruning). All reads return empty."""
    def get(self, include=None):
        return {"ids": [], "metadatas": []}

    def count(self) -> int:
        return 0

    def delete(self, ids=None) -> None:
        return


def test_rotation_vacuum_gated_by_delta(sql_env, monkeypatch):
    """Small page_count * page_size delta → no VACUUM. Large delta → VACUUM runs."""
    # Small-delta run: patch the threshold so even a tiny DB triggers VACUUM
    # conditionally. First, verify the UNTOUCHED path with a high threshold.
    monkeypatch.setattr(rag, "_VACUUM_DELTA_BYTES", 10**12)  # 1 TB
    out = rag._sql_rotate_log_tables(dry_run=False)
    assert out["vacuum_ran"] is False

    # Large-delta path: threshold set to 0 forces VACUUM to run on any DB.
    monkeypatch.setattr(rag, "_VACUUM_DELTA_BYTES", 0)
    out = rag._sql_rotate_log_tables(dry_run=False)
    assert out["vacuum_ran"] is True
    assert out["vacuum_before_bytes"] is not None
    assert out["vacuum_after_bytes"] is not None

    # Sentinel persisted — running again should see delta ≈ 0 and SKIP.
    monkeypatch.setattr(rag, "_VACUUM_DELTA_BYTES", 1)
    out2 = rag._sql_rotate_log_tables(dry_run=False)
    assert out2["vacuum_ran"] is False


def test_rotation_whatsapp_scheduled_uses_created_at(sql_env):
    """Regresión: rag_whatsapp_scheduled tiene `created_at`, NO `ts`.
    El loop no debe abortar mid-run ni omitir las tablas siguientes.
    """
    import sqlite3 as _sqlite
    db_path = sql_env["db_path"]
    conn = _sqlite.connect(str(db_path))
    now = time.time()
    try:
        # Crear la tabla con el DDL real (created_at, sin columna ts).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rag_whatsapp_scheduled ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  created_at TEXT NOT NULL,"
            "  scheduled_for_utc TEXT NOT NULL DEFAULT '',"
            "  jid TEXT NOT NULL DEFAULT '',"
            "  message_text TEXT NOT NULL DEFAULT '',"
            "  status TEXT NOT NULL DEFAULT 'pending'"
            ")"
        )
        # Seed 2 rows: una antigua (40d, debe borrarse con retention=30) y una reciente.
        old_ts = _iso_days_ago(40, now)
        new_ts = _iso_days_ago(5, now)
        conn.execute(
            "INSERT INTO rag_whatsapp_scheduled (created_at, scheduled_for_utc, jid, message_text)"
            " VALUES (?, '', '', '')", (old_ts,)
        )
        conn.execute(
            "INSERT INTO rag_whatsapp_scheduled (created_at, scheduled_for_utc, jid, message_text)"
            " VALUES (?, '', '', '')", (new_ts,)
        )
        conn.commit()
    finally:
        conn.close()

    # No debe lanzar excepción y debe reportar 1 row deleted (no 0 ni abortar).
    out = rag._sql_rotate_log_tables(dry_run=False, now_ts=now)
    assert "rag_whatsapp_scheduled" in out["rows_deleted"], (
        "La tabla rag_whatsapp_scheduled no apareció en rows_deleted — "
        "posiblemente el loop abortó antes de llegar a ella."
    )
    assert out["rows_deleted"]["rag_whatsapp_scheduled"] == 1, (
        "Esperaba 1 row deletada (la antigua de 40d con retention=30d) "
        f"pero got {out['rows_deleted']['rag_whatsapp_scheduled']}"
    )
    # Verificar que el row reciente sobrevivió.
    conn = _sqlite.connect(str(db_path))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM rag_whatsapp_scheduled"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 1, f"Esperaba 1 row sobreviviente, got {count}"


def test_rotation_no_abort_on_custom_time_col(sql_env):
    """Con una tabla custom (time_col != ts), las tablas SIGUIENTES también rotan.
    Verifica que el loop no se interrumpa: rag_synthetic_queries (que SÍ usa ts)
    debe rotarse aunque rag_whatsapp_scheduled esté presente antes de ella.
    """
    import sqlite3 as _sqlite
    db_path = sql_env["db_path"]
    conn = _sqlite.connect(str(db_path))
    now = time.time()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rag_whatsapp_scheduled ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  created_at TEXT NOT NULL,"
            "  scheduled_for_utc TEXT NOT NULL DEFAULT '',"
            "  jid TEXT NOT NULL DEFAULT '',"
            "  message_text TEXT NOT NULL DEFAULT '',"
            "  status TEXT NOT NULL DEFAULT 'pending'"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rag_synthetic_queries ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  ts TEXT NOT NULL,"
            "  query TEXT NOT NULL DEFAULT '',"
            "  source_path TEXT NOT NULL DEFAULT ''"
            ")"
        )
        # Rows antiguos en ambas tablas.
        old_ts = _iso_days_ago(100, now)
        conn.execute(
            "INSERT INTO rag_whatsapp_scheduled (created_at, scheduled_for_utc, jid, message_text)"
            " VALUES (?, '', '', '')", (old_ts,)
        )
        conn.execute(
            "INSERT INTO rag_synthetic_queries (ts, query) VALUES (?, 'q')", (old_ts,)
        )
        conn.commit()
    finally:
        conn.close()

    out = rag._sql_rotate_log_tables(dry_run=False, now_ts=now)
    # Ambas tablas deben aparecer y tener 1 delete cada una.
    assert out["rows_deleted"].get("rag_whatsapp_scheduled") == 1
    assert out["rows_deleted"].get("rag_synthetic_queries") == 1


# ── .bak cleanup ────────────────────────────────────────────────────────────

def test_bak_cleanup_purges_old(sql_env):
    """40-day-old .bak.<ts> files → deleted. 20-day-old → kept."""
    state_dir = sql_env["state_dir"]
    now = time.time()
    old_ts = int(now - 40 * 86400)
    young_ts = int(now - 20 * 86400)

    old_f = state_dir / f"queries.jsonl.bak.{old_ts}"
    young_f = state_dir / f"behavior.jsonl.bak.{young_ts}"
    old_f.write_text("old\n", encoding="utf-8")
    young_f.write_text("young\n", encoding="utf-8")

    out = rag._cleanup_bak_files(dry_run=False, max_age_days=30, now_ts=now)
    assert out["deleted"] == 1
    assert not old_f.exists()
    assert young_f.exists()


def test_bak_cleanup_ignores_unexpected_extensions(sql_env):
    """Files not matching *.bak.<digits> must not be touched."""
    state_dir = sql_env["state_dir"]
    now = time.time()
    # Legacy `.bak` (no ts) — should NOT match.
    (state_dir / "foo.bak").write_text("bak", encoding="utf-8")
    # `.bak.notanumber` — should NOT match + counted as skipped.
    (state_dir / "foo.bak.notanumber").write_text("nope", encoding="utf-8")
    # `.baktemp` — regex guard.
    (state_dir / "foo.baktemp").write_text("no", encoding="utf-8")
    # Actual old .bak.<ts>.
    old_ts = int(now - 60 * 86400)
    legit = state_dir / f"queries.jsonl.bak.{old_ts}"
    legit.write_text("legit", encoding="utf-8")

    out = rag._cleanup_bak_files(dry_run=False, max_age_days=30, now_ts=now)
    # Only the legit one deleted.
    assert out["deleted"] == 1
    # notanumber counted as skipped.
    assert out["skipped_non_bak"] >= 1
    # Unrelated files survive.
    assert (state_dir / "foo.bak").exists()
    assert (state_dir / "foo.bak.notanumber").exists()
    assert (state_dir / "foo.baktemp").exists()
    assert not legit.exists()


# ── Rollback ────────────────────────────────────────────────────────────────

def _write_bak(state_dir: Path, name: str, ts: int, content: str = "x\n") -> Path:
    p = state_dir / f"{name}.bak.{ts}"
    p.write_text(content, encoding="utf-8")
    return p


def test_rollback_state_migration_restores_baks(sql_env, monkeypatch):
    """Set up a DB with rag_queries rows + a recent .bak.<ts> → rollback
    renames the bak back + drops the tables."""
    monkeypatch.setattr(rag, "_pgrep_obsidian_rag", lambda: [])  # no services
    state_dir = sql_env["state_dir"]
    now = time.time()
    recent_ts = int(now - 5 * 86400)

    bak = _write_bak(state_dir, "queries.jsonl", recent_ts, "original\n")

    # Seed a table row so we can assert the DROP actually ran.
    conn = sqlite3.connect(str(sql_env["db_path"]))
    try:
        _seed_rows(conn, "rag_queries", [_iso_days_ago(1, now)], {"q": "pre"})
    finally:
        conn.close()

    result = rag._rollback_state_migration(force=False, now_ts=now)
    assert result["ok"] is True
    assert result["files_restored"] == 1
    assert "rag_queries" in result["tables_dropped"]

    # Bak renamed back to original path.
    assert not bak.exists()
    assert (state_dir / "queries.jsonl").read_text(encoding="utf-8") == "original\n"

    # Tables gone.
    conn = sqlite3.connect(str(sql_env["db_path"]))
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "rag_queries" not in names
    finally:
        conn.close()


def test_rollback_refuses_when_services_running(sql_env, monkeypatch):
    monkeypatch.setattr(rag, "_pgrep_obsidian_rag", lambda: ["12345"])
    state_dir = sql_env["state_dir"]
    now = time.time()
    _write_bak(state_dir, "queries.jsonl", int(now - 86400))

    result = rag._rollback_state_migration(force=False, now_ts=now)
    assert result["ok"] is False
    assert result["refused"] is not None
    assert "12345" in result["refused"]


def test_rollback_force_proceeds(sql_env, monkeypatch):
    """pids matched but --force → rollback still runs."""
    monkeypatch.setattr(rag, "_pgrep_obsidian_rag", lambda: ["99999"])
    state_dir = sql_env["state_dir"]
    now = time.time()
    bak = _write_bak(state_dir, "queries.jsonl", int(now - 86400), "restored\n")

    result = rag._rollback_state_migration(force=True, now_ts=now)
    assert result["ok"] is True
    assert result["files_restored"] == 1
    assert not bak.exists()
    assert (state_dir / "queries.jsonl").read_text(encoding="utf-8") == "restored\n"


def test_rollback_refuses_when_no_baks(sql_env, monkeypatch):
    """No .bak.<ts> within 30d → refuse without --force."""
    monkeypatch.setattr(rag, "_pgrep_obsidian_rag", lambda: [])
    # state_dir is empty by default; no baks exist.
    result = rag._rollback_state_migration(force=False)
    assert result["ok"] is False
    assert result["refused"] is not None
    assert "bak" in result["refused"].lower()


# ── Maintenance summary integration ──────────────────────────────────────────

def test_maintenance_summary_includes_sql_section(sql_env, monkeypatch):
    """run_maintenance produces an `sql_rotation` key with `rows_deleted`
    per table (post-T10 SQL-only path, unconditional)."""
    # Seed a queries row + a memory row, both old.
    conn = sqlite3.connect(str(sql_env["db_path"]))
    now = time.time()
    try:
        _seed_rows(conn, "rag_queries", [_iso_days_ago(100, now)], {"q": "stale"})
        _seed_rows(conn, "rag_cpu_metrics", [_iso_days_ago(45, now)])
    finally:
        conn.close()

    # Stub the expensive / external calls.
    monkeypatch.setattr(rag, "auto_index_vault", lambda *a, **k: {
        "kind": "no_changes", "indexed": 0, "removed": 0, "scanned": 0, "took_ms": 0})
    monkeypatch.setattr(rag, "cleanup_sessions", lambda: 0)
    monkeypatch.setattr(rag, "_find_orphan_collections", lambda: [])
    monkeypatch.setattr(rag, "_prune_orphan_segment_dirs", lambda dry_run=False: {
        "count": 0, "bytes_freed": 0, "paths": []})
    monkeypatch.setattr(rag, "_vec_wal_checkpoint", lambda dry_run=False: {
        "ok": True, "before_bytes": 0, "after_bytes": 0})
    monkeypatch.setattr(rag, "_rebuild_feedback_golden_from_sql_feedback",
                         lambda conn: {"positives": [], "negatives": []})
    monkeypatch.setattr(rag, "find_dead_notes", lambda *a, **k: [])
    monkeypatch.setattr(rag, "get_db", lambda: _DummyCol())
    monkeypatch.setattr(rag, "get_urls_db", lambda: _DummyCol())
    monkeypatch.setattr(rag, "_prune_ignored_notes", lambda v: 0)
    monkeypatch.setattr(rag, "_prune_auto_index_state", lambda: 0)
    monkeypatch.setattr(rag, "_prune_filing_batches", lambda: 0)
    monkeypatch.setattr(rag, "_cleanup_tmp_files", lambda: 0)
    monkeypatch.setattr(rag, "_prune_url_orphans", lambda v: 0)
    monkeypatch.setattr(rag, "_prune_feedback_orphans", lambda v: 0)
    monkeypatch.setattr(rag, "_check_ollama_health", lambda: {"stub": "ok"})

    results = rag.run_maintenance(dry_run=False, skip_reindex=True, skip_logs=False)
    assert "sql_rotation" in results
    assert "rows_deleted" in results["sql_rotation"]
    assert results["sql_rotation"]["rows_deleted"]["rag_queries"] == 1
    assert results["sql_rotation"]["rows_deleted"]["rag_cpu_metrics"] == 1
    # bak cleanup ran (empty state_dir → 0 purged, but key present).
    assert "bak_cleanup" in results
    assert results["bak_cleanup"]["deleted"] == 0
