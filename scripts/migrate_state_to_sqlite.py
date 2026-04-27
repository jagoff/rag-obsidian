"""One-shot JSONL→SQLite migration for the rag_* telemetry tables.

Reads every legacy JSONL/JSON state file under ~/.local/share/obsidian-rag/
and imports it into the SQL tables created by T1 (_ensure_telemetry_tables).
On successful per-source commit, renames the source to .bak.<unix_ts> so a
re-run is naturally idempotent.

Concurrency: live services append to these files. Run with launchd services
stopped. Script refuses if `pgrep -f com.fer.obsidian-rag` returns a pid
unless `--force` is passed.

Per-source flow:
  open → BEGIN IMMEDIATE → stream-parse → map → append/upsert → COMMIT
  on success: os.replace(source, f"{source}.bak.{int(time.time())}")
  on failure: ROLLBACK, leave source untouched, continue to next source.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

# Import from the parent rag.py (sibling of scripts/)
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
import rag  # noqa: E402

DEFAULT_SOURCE_DIR = Path.home() / ".local/share/obsidian-rag"


# ── Source registry ──────────────────────────────────────────────────────────
# Order matters for one dependency: rag_feedback_golden reads MAX(ts) from
# rag_feedback at build time, so feedback must migrate before feedback_golden.
#
# `kind` values:
#   "log"   → append-only; row dict → _sql_append_event.
#   "state" → upsert on pk_cols.
#   "state_latest_jsonl" → JSONL stream with multiple records per pk, last
#                           wins (currently ambient_state).
#   "conversations_index" → flat JSON object {session_id: relative_path}.
#   "feedback_golden"    → JSON {positives, negatives}; expand to (path, rating).

@dataclass
class Source:
    name: str            # short name used in --only and summary table
    filename: str        # basename under source_dir
    table: str           # destination SQL table
    kind: str            # log | state_latest_jsonl | conversations_index | feedback_golden
    pk_cols: tuple[str, ...] = ()
    scalars: tuple[str, ...] = ()           # plain scalar columns copied 1:1
    jsons: tuple[tuple[str, str], ...] = () # (src_key, dst_col) pairs; dst_col ends in _json
    bool_ints: tuple[str, ...] = ()         # scalar keys to coerce bool → int
    required: tuple[str, ...] = ()          # keys that get a default value


def _generic_map(rec: dict, src: Source) -> dict:
    """Shared mapper for all log/state_latest_jsonl sources. Driven by Source fields."""
    out: dict = {k: rec[k] for k in src.scalars if k in rec}
    for bk in src.bool_ints:
        if bk in out and isinstance(out[bk], bool):
            out[bk] = int(out[bk])
    for src_key, dst_col in src.jsons:
        if src_key in rec:
            out[dst_col] = rec[src_key]
    known = set(src.scalars) | {k for k, _ in src.jsons}
    extra = {k: v for k, v in rec.items() if k not in known}
    if extra and "extra_json" in _telemetry_cols(src.table):
        out["extra_json"] = extra
    elif extra and "payload_json" in _telemetry_cols(src.table) and "payload_json" not in out:
        out["payload_json"] = extra
    # ts default
    if "ts" in src.scalars and "ts" not in out:
        out["ts"] = rec.get("ts") or _now_iso()
    for req in src.required:
        out.setdefault(req, "unknown" if req != "rating" else 0)
    return out


_TELEMETRY_COLS_CACHE: dict[str, set[str]] = {}


def _telemetry_cols(table: str) -> set[str]:
    """Return the set of column names for `table` by parsing T1's DDL once.
    Cheap + avoids opening the DB just to inspect columns."""
    if table in _TELEMETRY_COLS_CACHE:
        return _TELEMETRY_COLS_CACHE[table]
    cols: set[str] = set()
    for tname, stmts in rag._TELEMETRY_DDL:
        if tname != table:
            continue
        head = stmts[0]
        # Naively extract column names between first '(' and matching ')'.
        inner = head[head.index("(") + 1: head.rindex(")")]
        for piece in inner.split(","):
            tok = piece.strip().split()
            if tok and tok[0].isidentifier():
                cols.add(tok[0])
        break
    _TELEMETRY_COLS_CACHE[table] = cols
    return cols


def _map_eval(rec: dict, src: Source) -> dict:
    """eval.jsonl: {ts, singles:{hit5,mrr,n}, chains:{...}} → flattened row."""
    out: dict = {"ts": rec.get("ts") or _now_iso()}
    sg = rec.get("singles") or {}
    ch = rec.get("chains") or {}
    for src_key, dst in (("hit5", "singles_hit5"), ("mrr", "singles_mrr"), ("n", "singles_n")):
        if src_key in sg:
            out[dst] = sg[src_key]
    for src_key, dst in (
        ("hit5", "chains_hit5"), ("mrr", "chains_mrr"),
        ("chain_success", "chains_chain_success"),
        ("turns", "chains_turns"), ("chains", "chains_n"),
    ):
        if src_key in ch:
            out[dst] = ch[src_key]
    extra = {k: v for k, v in rec.items() if k not in {"ts", "singles", "chains"}}
    if extra:
        out["extra_json"] = extra
    return out


def _map_brief_state(rec: dict, src: Source) -> dict:
    """brief_state.jsonl legacy: {ts, key=brief_path\\x00cited_path}. Split to populate path/brief_type."""
    key = rec.get("key") or ""
    ts = rec.get("ts") or _now_iso()
    brief_path, cited_path = (key.split("\x00", 1) + [""])[:2] if "\x00" in key else (key, "")
    bt = "today" if brief_path.endswith("-evening.md") else (
        "morning" if "/04-Archive/99-obsidian-system/99-AI/reviews/" in brief_path or brief_path.startswith("04-Archive/99-obsidian-system/99-AI/reviews/") else "unknown"
    )
    return {"pair_key": key, "brief_type": bt, "kind": "cited",
            "path": cited_path, "first_ts": ts, "last_ts": ts}


SOURCES: tuple[Source, ...] = (
    Source("queries", "queries.jsonl", "rag_queries", "log",
           scalars=("ts", "cmd", "q", "session", "mode", "top_score", "t_retrieve",
                    "t_gen", "answer_len", "citation_repaired", "critique_fired",
                    "critique_changed"),
           jsons=(("variants", "variants_json"), ("paths", "paths_json"),
                  ("scores", "scores_json"), ("filters", "filters_json"),
                  ("bad_citations", "bad_citations_json"))),
    Source("behavior", "behavior.jsonl", "rag_behavior", "log",
           scalars=("ts", "source", "event", "path", "query", "rank", "dwell_s"),
           required=("source", "event")),
    Source("feedback", "feedback.jsonl", "rag_feedback", "log",
           scalars=("ts", "turn_id", "rating", "q", "scope"),
           jsons=(("paths", "paths_json"),), required=("rating",)),
    Source("tune", "tune.jsonl", "rag_tune", "log",
           scalars=("ts", "cmd", "samples", "seed", "n_cases", "delta",
                    "eval_hit5_singles", "eval_hit5_chains", "rolled_back"),
           jsons=(("baseline", "baseline_json"), ("best", "best_json"))),
    Source("contradictions", "contradictions.jsonl", "rag_contradictions", "log",
           scalars=("ts", "subject_path", "helper_raw", "skipped"),
           jsons=(("contradicts", "contradicts_json"),),
           required=("subject_path",)),
    Source("ambient", "ambient.jsonl", "rag_ambient", "log",
           scalars=("ts", "cmd", "path", "hash")),
    Source("brief_written", "brief_written.jsonl", "rag_brief_written", "log",
           scalars=("ts", "brief_type", "brief_path"),
           jsons=(("paths_cited", "paths_cited_json"),
                  ("citations_by_section", "citations_by_section_json")),
           required=("brief_type",)),
    Source("wa_tasks", "wa_tasks.jsonl", "rag_wa_tasks", "log",
           scalars=("ts", "since", "chats", "items", "path")),
    Source("archive", "archive.jsonl", "rag_archive_log", "log",
           scalars=("ts", "cmd", "min_age_days", "query_window_days", "folder",
                    "dry_run", "force", "gate", "n_candidates", "n_plan",
                    "n_applied", "n_skipped", "gated", "batch_path"),
           bool_ints=("dry_run", "force", "gated")),
    Source("filing", "filing.jsonl", "rag_filing_log", "log",
           scalars=("ts", "cmd", "path", "note", "folder", "confidence",
                    "upward_title", "upward_kind"),
           jsons=(("neighbors", "neighbors_json"),)),
    Source("eval", "eval.jsonl", "rag_eval_runs", "log"),   # custom mapper
    Source("surface", "surface.jsonl", "rag_surface_log", "log",
           scalars=("ts", "cmd", "n_pairs", "sim_threshold", "min_hops",
                    "top", "skip_young_days", "llm", "duration_ms"),
           bool_ints=("llm",)),
    Source("proactive", "proactive.jsonl", "rag_proactive_log", "log",
           scalars=("ts", "kind", "sent", "reason"),
           bool_ints=("sent",)),
    Source("cpu", "rag_cpu.jsonl", "rag_cpu_metrics", "log",
           scalars=("ts", "total_pct", "ncores", "interval_s"),
           jsons=(("by_category", "by_category_json"), ("top", "top_json"))),
    Source("memory", "rag_memory.jsonl", "rag_memory_metrics", "log",
           scalars=("ts", "total_mb"),
           jsons=(("by_category", "by_category_json"), ("top", "top_json"),
                  ("vm", "vm_json"))),
    # T1 DDL created this WITHOUT the rag_ prefix — keep the live name.
    Source("system_memory", "system_memory.jsonl", "system_memory_metrics", "log",
           scalars=("ts", "total_mb"),
           jsons=(("by_category", "by_category_json"), ("top", "top_json"),
                  ("vm", "vm_json"))),
    Source("ambient_state", "ambient_state.jsonl", "rag_ambient_state",
           "state_latest_jsonl", pk_cols=("path",),
           scalars=("path", "hash", "analyzed_at")),
    Source("brief_state", "brief_state.jsonl", "rag_brief_state",
           "state_latest_jsonl", pk_cols=("pair_key",)),  # custom mapper
    Source("conversations_index", "conversations_index.json",
           "rag_conversations_index", "conversations_index",
           pk_cols=("session_id",)),
    Source("feedback_golden", "feedback_golden.json", "rag_feedback_golden",
           "feedback_golden", pk_cols=("path", "rating")),
)


# Custom mapper registry (fallbacks to _generic_map).
_CUSTOM_MAPPERS = {
    "eval": _map_eval,
    "brief_state": _map_brief_state,
}


def _map(rec: dict, src: Source) -> dict:
    fn = _CUSTOM_MAPPERS.get(src.name)
    return fn(rec, src) if fn else _generic_map(rec, src)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """ISO-8601 second-precision to match the rest of the codebase."""
    import datetime as _dt
    return _dt.datetime.now().isoformat(timespec="seconds")


def _connect_db(db_path: Path):
    """Open ragvec.db the same way SqliteVecClient does and apply T1 DDL.

    The migration script targets the *live* DB file. We don't go through
    SqliteVecClient because its init triggers sqlite-vec extension load +
    collection creation which is irrelevant here (and noisy on a test DB).
    """
    import sqlite3 as _sqlite3
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_schema_version ("
        " table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
    )
    rag._ensure_telemetry_tables(conn)
    return conn


def _iter_jsonl(path: Path):
    """Yield parsed JSON records. Skip unparseable lines silently (matching
    the live code's best-effort tolerance; these files are append-only logs
    and can have a torn trailing write)."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _check_running_services() -> list[str]:
    """Return list of matching pids. Empty = nothing running."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "com.fer.obsidian-rag"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.split() if line.strip()]


# ── Per-source migration ─────────────────────────────────────────────────────

@dataclass
class SourceResult:
    name: str
    filename: str
    rows_read: int = 0
    rows_inserted: int = 0
    rows_skipped_dupe: int = 0
    status: str = "pending"  # migrated | skipped-missing | failed | dry-run
    error: str | None = None
    renamed_to: str | None = None


def _sql_insert_nocommit(conn, table: str, row: dict) -> None:
    """Inline INSERT that does NOT commit (unlike rag._sql_append_event which
    calls conn.commit()). Needed so the per-source transaction wraps all rows."""
    serialised = rag._sql_serialise_row(row)
    cols = list(serialised.keys())
    placeholders = ",".join("?" for _ in cols)
    col_sql = ",".join(cols)
    sql = f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})"
    conn.execute(sql, [serialised[c] for c in cols])


def _sql_upsert_nocommit(conn, table: str, row: dict) -> None:
    """Inline INSERT OR REPLACE with no commit. See _sql_insert_nocommit."""
    serialised = rag._sql_serialise_row(row)
    cols = list(serialised.keys())
    placeholders = ",".join("?" for _ in cols)
    col_sql = ",".join(cols)
    sql = f"INSERT OR REPLACE INTO {table} ({col_sql}) VALUES ({placeholders})"
    conn.execute(sql, [serialised[c] for c in cols])


def _migrate_log(conn, src: Source, path: Path, dry_run: bool) -> SourceResult:
    import sqlite3 as _sqlite3
    r = SourceResult(name=src.name, filename=src.filename)
    if dry_run:
        for _ in _iter_jsonl(path):
            r.rows_read += 1
        r.status = "dry-run"
        return r
    conn.execute("BEGIN IMMEDIATE")
    try:
        for rec in _iter_jsonl(path):
            r.rows_read += 1
            row = _map(rec, src)
            try:
                _sql_insert_nocommit(conn, src.table, row)
                r.rows_inserted += 1
            except _sqlite3.IntegrityError:
                r.rows_skipped_dupe += 1
        conn.execute("COMMIT")
        r.status = "migrated"
    except Exception as exc:
        try:
            conn.execute("ROLLBACK")
        except _sqlite3.OperationalError:
            pass
        r.status = "failed"
        r.error = f"{type(exc).__name__}: {exc}"
        raise
    return r


def _migrate_state_latest_jsonl(conn, src: Source, path: Path, dry_run: bool) -> SourceResult:
    """JSONL file where logically only the latest record per PK is authoritative.
    Fold in memory (path is small, <100KB typical), then upsert in one tx."""
    r = SourceResult(name=src.name, filename=src.filename)
    latest: dict = {}
    for rec in _iter_jsonl(path):
        r.rows_read += 1
        row = _map(rec, src)
        if not src.pk_cols:
            continue
        key = tuple(row.get(c) for c in src.pk_cols)
        latest[key] = row
    if dry_run:
        r.status = "dry-run"
        return r
    import sqlite3 as _sqlite3
    conn.execute("BEGIN IMMEDIATE")
    try:
        for row in latest.values():
            _sql_upsert_nocommit(conn, src.table, row)
            r.rows_inserted += 1
        conn.execute("COMMIT")
        r.status = "migrated"
    except Exception as exc:
        try:
            conn.execute("ROLLBACK")
        except _sqlite3.OperationalError:
            pass
        r.status = "failed"
        r.error = f"{type(exc).__name__}: {exc}"
        raise
    return r


def _migrate_conversations_index(conn, src: Source, path: Path, dry_run: bool) -> SourceResult:
    r = SourceResult(name=src.name, filename=src.filename)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        r.status = "failed"
        r.error = f"expected dict, got {type(data).__name__}"
        return r
    r.rows_read = len(data)
    if dry_run:
        r.status = "dry-run"
        return r
    now_iso = _now_iso()
    import sqlite3 as _sqlite3
    conn.execute("BEGIN IMMEDIATE")
    try:
        for session_id, rel_path in data.items():
            row = {
                "session_id": session_id,
                "relative_path": rel_path,
                "updated_at": now_iso,
            }
            _sql_upsert_nocommit(conn, src.table, row)
            r.rows_inserted += 1
        conn.execute("COMMIT")
        r.status = "migrated"
    except Exception as exc:
        try:
            conn.execute("ROLLBACK")
        except _sqlite3.OperationalError:
            pass
        r.status = "failed"
        r.error = f"{type(exc).__name__}: {exc}"
        raise
    return r


def _migrate_feedback_golden(conn, src: Source, path: Path, dry_run: bool) -> SourceResult:
    """feedback_golden.json → rag_feedback_golden + rag_feedback_golden_meta.

    JSON shape: {"positives": [{q, emb, paths}], "negatives": [...]}.
    Fan out: one (path, rating) row per path in each entry, using that
    entry's query embedding. Last-seen-wins on duplicate (path, rating) via
    INSERT OR REPLACE.

    `source_ts` = max(ts) of rag_feedback at build time (read AFTER feedback
    table is populated — the runner calls this source after 'feedback').
    Also writes `last_built_source_ts` to rag_feedback_golden_meta.
    """
    r = SourceResult(name=src.name, filename=src.filename)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    positives = data.get("positives") or []
    negatives = data.get("negatives") or []
    # Count logical rows = sum of path instances (dedup applied below is
    # silent on Insert-OR-Replace so we just count).
    r.rows_read = sum(len(e.get("paths", [])) for e in positives) \
        + sum(len(e.get("paths", [])) for e in negatives)
    if dry_run:
        r.status = "dry-run"
        return r
    source_ts = rag._sql_max_ts(conn, "rag_feedback") or _now_iso()
    built_at = _now_iso()
    import sqlite3 as _sqlite3
    conn.execute("BEGIN IMMEDIATE")
    try:
        seen: set[tuple[str, int]] = set()
        for rating, entries in ((1, positives), (-1, negatives)):
            for entry in entries:
                emb = entry.get("emb") or []
                paths = [p for p in (entry.get("paths") or []) if p]
                if not emb or not paths:
                    continue
                dim = len(emb)
                blob = struct.pack("<" + "f" * dim, *emb)
                for p in paths:
                    key = (p, rating)
                    # Count each unique pk once.
                    if key not in seen:
                        seen.add(key)
                    row = {
                        "path": p,
                        "rating": rating,
                        "embedding": blob,
                        "dim": dim,
                        "built_at": built_at,
                        "source_ts": source_ts,
                    }
                    _sql_upsert_nocommit(conn, src.table, row)
                    r.rows_inserted += 1
        # Meta kv
        _sql_upsert_nocommit(
            conn, "rag_feedback_golden_meta",
            {"k": "last_built_source_ts", "v": source_ts},
        )
        conn.execute("COMMIT")
        r.status = "migrated"
    except Exception as exc:
        try:
            conn.execute("ROLLBACK")
        except _sqlite3.OperationalError:
            pass
        r.status = "failed"
        r.error = f"{type(exc).__name__}: {exc}"
        raise
    return r


_MIGRATORS = {
    "log": _migrate_log,
    "state_latest_jsonl": _migrate_state_latest_jsonl,
    "conversations_index": _migrate_conversations_index,
    "feedback_golden": _migrate_feedback_golden,
}


# ── Reverse direction ────────────────────────────────────────────────────────

def _reverse_source(conn, src: Source, out_dir: Path) -> SourceResult:
    """Dump SQL table rows back to JSONL/JSON at the canonical filename.

    Used as a 30-day escape hatch. Does NOT drop tables. Overwrites target.
    """
    import sqlite3 as _sqlite3
    r = SourceResult(name=src.name, filename=src.filename)
    out_path = out_dir / src.filename
    out_dir.mkdir(parents=True, exist_ok=True)
    if src.kind == "conversations_index":
        rows = conn.execute(
            f"SELECT session_id, relative_path FROM {src.table} ORDER BY session_id"
        ).fetchall()
        obj = {sid: rp for sid, rp in rows}
        out_path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        r.rows_read = r.rows_inserted = len(obj)
        r.status = "reversed"
        return r
    if src.kind == "feedback_golden":
        rows = conn.execute(
            f"SELECT path, rating, embedding, dim FROM {src.table}"
        ).fetchall()
        # Reassemble one entry per (rating, emb). We can't recover the exact
        # query text (it's not in the table) — leave `q` blank; this is the
        # documented limitation of the reverse dump.
        out_obj = {"positives": [], "negatives": []}
        by_rating: dict[int, list] = {1: [], -1: []}
        for p, rating, blob, dim in rows:
            emb = list(struct.unpack("<" + "f" * dim, blob))
            by_rating.setdefault(rating, []).append({"q": "", "emb": emb, "paths": [p]})
        out_obj["positives"] = by_rating.get(1, [])
        out_obj["negatives"] = by_rating.get(-1, [])
        out_path.write_text(json.dumps(out_obj, ensure_ascii=False), encoding="utf-8")
        r.rows_read = r.rows_inserted = len(rows)
        r.status = "reversed"
        return r

    # JSONL output for log + state_latest_jsonl sources
    col_names = [c[1] for c in conn.execute(f"PRAGMA table_info({src.table})").fetchall()]
    prev_row_factory = conn.row_factory
    conn.row_factory = _sqlite3.Row
    try:
        if src.kind == "log":
            rows = list(conn.execute(f"SELECT * FROM {src.table} ORDER BY ts, id"))
        else:
            rows = list(conn.execute(f"SELECT * FROM {src.table}"))
    finally:
        conn.row_factory = prev_row_factory
    lines: list[str] = []
    for row in rows:
        rec = {c: row[c] for c in col_names if c not in ("id",)}
        # Inflate *_json columns
        flat: dict = {}
        for k, v in rec.items():
            if v is None:
                continue
            if k.endswith("_json") and isinstance(v, str):
                try:
                    inflated = json.loads(v)
                except json.JSONDecodeError:
                    inflated = v
                flat[k[:-5]] = inflated  # strip trailing _json
            else:
                flat[k] = v
        lines.append(json.dumps(flat, ensure_ascii=False, sort_keys=True))
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    r.rows_read = r.rows_inserted = len(rows)
    r.status = "reversed"
    return r


# ── Round-trip check ─────────────────────────────────────────────────────────

def _canonical_line(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    return json.dumps(rec, ensure_ascii=False, sort_keys=True)


def _round_trip_check(conn, results: list[SourceResult], source_dir: Path) -> list[str]:
    """Export each migrated table back to .roundtrip.jsonl, diff vs the .bak
    file we made during migration. Returns a list of human-readable failures.
    Empty list = all clean.
    """
    import tempfile
    failures: list[str] = []
    out_dir = Path(tempfile.mkdtemp(prefix="rag-migrate-rt-"))
    by_name = {s.name: s for s in SOURCES}
    for res in results:
        # A fresh migration surfaces .bak immediately (status='migrated').
        # A re-run finds the source already consumed (status='skipped-missing')
        # but a .bak should still be on disk from the earlier run — check
        # against that so drift detection works after the fact.
        if res.status not in ("migrated", "skipped-missing"):
            continue
        src = by_name[res.name]
        # Find the most recent .bak for this source.
        baks = sorted(source_dir.glob(f"{src.filename}.bak.*"), reverse=True)
        if not baks:
            # No backup found → nothing to compare against. For a fresh
            # migrated source this is a bug; for skipped-missing it just
            # means no prior run happened — silently skip.
            if res.status == "migrated":
                failures.append(f"{src.name}: no .bak.<ts> found for round-trip")
            continue
        bak = baks[0]
        rev = _reverse_source(conn, src, out_dir)
        rev_path = out_dir / src.filename
        # Semantics:
        # - log / state_latest_jsonl: canonicalise each line, compare as multiset
        # - conversations_index: compare dict equality
        # - feedback_golden: compare sets of (path, rating, dim) — the q field
        #   is not recoverable from SQL, so we relax that dimension.
        if src.kind == "log":
            a = sorted(filter(None, (_canonical_line(l) for l in bak.read_text(encoding="utf-8").splitlines())))
            b = sorted(filter(None, (_canonical_line(l) for l in rev_path.read_text(encoding="utf-8").splitlines())))
            if a != b:
                failures.append(f"{src.name}: round-trip diff (orig={len(a)} vs export={len(b)})")
        elif src.kind == "state_latest_jsonl":
            # Fold original by PK (latest wins), compare to exported dump.
            latest_orig: dict = {}
            for line in bak.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mapped = _map(rec, src)
                pk = tuple(mapped.get(c) for c in src.pk_cols)
                latest_orig[pk] = mapped
            # Exported records
            exp: dict = {}
            for line in rev_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pk = tuple(rec.get(c) for c in src.pk_cols)
                exp[pk] = rec
            if set(latest_orig.keys()) != set(exp.keys()):
                failures.append(f"{src.name}: PK mismatch orig={len(latest_orig)} exp={len(exp)}")
        elif src.kind == "conversations_index":
            orig = json.loads(bak.read_text(encoding="utf-8"))
            exp = json.loads(rev_path.read_text(encoding="utf-8"))
            if set(orig.keys()) != set(exp.keys()):
                failures.append(f"{src.name}: session id set drift")
            else:
                for sid in orig:
                    if orig[sid] != exp[sid]:
                        failures.append(f"{src.name}: path changed for {sid}")
        elif src.kind == "feedback_golden":
            orig = json.loads(bak.read_text(encoding="utf-8"))
            exp = json.loads(rev_path.read_text(encoding="utf-8"))
            orig_keys = {(p, 1) for e in orig.get("positives", []) for p in e.get("paths", [])}
            orig_keys |= {(p, -1) for e in orig.get("negatives", []) for p in e.get("paths", [])}
            exp_keys = {(e["paths"][0], 1) for e in exp.get("positives", [])}
            exp_keys |= {(e["paths"][0], -1) for e in exp.get("negatives", [])}
            if orig_keys != exp_keys:
                failures.append(f"{src.name}: (path,rating) set drift")
    return failures


# ── CLI ──────────────────────────────────────────────────────────────────────

def _print_summary_table(results: list[SourceResult]) -> None:
    cols = ("source", "rows_read", "rows_inserted", "rows_skipped_dupe", "status")
    print(f"{cols[0]:<32} {cols[1]:>10} {cols[2]:>14} {cols[3]:>18} {cols[4]:<12}")
    print("-" * 92)
    for r in results:
        print(f"{r.filename:<32} {r.rows_read:>10} {r.rows_inserted:>14} {r.rows_skipped_dupe:>18} {r.status:<12}")
        if r.error:
            print(f"    error: {r.error}")


def _print_summary_json(results: list[SourceResult]) -> None:
    out = [
        {
            "source": r.filename,
            "name": r.name,
            "rows_read": r.rows_read,
            "rows_inserted": r.rows_inserted,
            "rows_skipped_dupe": r.rows_skipped_dupe,
            "status": r.status,
            "error": r.error,
            "renamed_to": r.renamed_to,
        }
        for r in results
    ]
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _audit_trail(conn, results: list[SourceResult]) -> None:
    """Append a rag_tune row (cmd='migrate') summarising the run."""
    summary_payload = [
        {
            "name": r.name, "filename": r.filename,
            "rows_read": r.rows_read, "rows_inserted": r.rows_inserted,
            "rows_skipped_dupe": r.rows_skipped_dupe, "status": r.status,
        }
        for r in results
    ]
    try:
        rag._sql_append_event(conn, "rag_tune", {
            "ts": _now_iso(),
            "cmd": "migrate",
            "extra_json": {"migration": summary_payload},
        })
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--round-trip-check", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated source names (e.g. queries,behavior)")
    ap.add_argument("--force", action="store_true",
                    help="Proceed even if launchd rag services appear to be running")
    args = ap.parse_args(argv)

    source_dir: Path = args.source_dir
    db_path = source_dir / "ragvec" / "ragvec.db"

    selected: list[Source] = list(SOURCES)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = wanted - {s.name for s in SOURCES}
        if unknown:
            print(f"ERROR: unknown --only names: {sorted(unknown)}", file=sys.stderr)
            return 2
        selected = [s for s in SOURCES if s.name in wanted]

    # Preflight: warn/refuse if services are running.
    if not args.dry_run and not args.summary:
        pids = _check_running_services()
        if pids and not args.force:
            print(
                "WARNING: Stop launchd services before running this migration — "
                "see the preflight checklist.",
                file=sys.stderr,
            )
            print(f"Refusing to run: pgrep matched pids {pids}. Pass --force to override.",
                  file=sys.stderr)
            return 3

    try:
        conn = _connect_db(db_path)
    except Exception as exc:
        print(f"ERROR: cannot open {db_path}: {exc}", file=sys.stderr)
        return 1

    try:
        if args.reverse:
            results: list[SourceResult] = []
            for src in selected:
                try:
                    r = _reverse_source(conn, src, source_dir)
                except Exception as exc:
                    r = SourceResult(name=src.name, filename=src.filename,
                                     status="failed", error=f"{type(exc).__name__}: {exc}")
                results.append(r)
            if args.summary:
                _print_summary_json(results)
            else:
                _print_summary_table(results)
            return 0

        results = _run_migration(conn, selected, source_dir, dry_run=args.dry_run or args.summary)

        if args.summary:
            _print_summary_json(results)
        else:
            _print_summary_table(results)

        # Audit trail only on a non-dry run.
        if not (args.dry_run or args.summary):
            _audit_trail(conn, results)

        if args.round_trip_check and not (args.dry_run or args.summary):
            failures = _round_trip_check(conn, results, source_dir)
            if failures:
                print("\nROUND-TRIP FAILURES:", file=sys.stderr)
                for f in failures:
                    print(f"  - {f}", file=sys.stderr)
                return 4
            print("\nRound-trip OK.")

        # Exit 1 if any source failed.
        if any(r.status == "failed" for r in results):
            return 1
        return 0
    finally:
        conn.close()


def _run_migration(conn, selected: list[Source], source_dir: Path, dry_run: bool) -> list[SourceResult]:
    """Iterate sources in order, migrate each in its own transaction, rename
    source to .bak.<ts> on successful commit. Failed source stays untouched."""
    results: list[SourceResult] = []
    for src in selected:
        path = source_dir / src.filename
        if not path.is_file():
            r = SourceResult(name=src.name, filename=src.filename,
                             status="skipped-missing")
            results.append(r)
            continue
        migrator = _MIGRATORS[src.kind]
        try:
            r = migrator(conn, src, path, dry_run)
        except Exception as exc:
            print(f"ERROR migrating {src.filename}: {exc}", file=sys.stderr)
            traceback.print_exc()
            r = SourceResult(name=src.name, filename=src.filename,
                             status="failed", error=f"{type(exc).__name__}: {exc}")
            results.append(r)
            continue
        # Rename on success (never on dry-run / skipped / failed)
        if r.status == "migrated" and not dry_run:
            bak = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
            os.replace(path, bak)
            r.renamed_to = str(bak)
        results.append(r)
    return results


if __name__ == "__main__":
    sys.exit(main())
