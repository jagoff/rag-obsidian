from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# Telemetry DB lives next to ragvec.db (post-2026-04-21 split). We resolve
# the path dynamically via rag.DB_PATH + rag._TELEMETRY_DB_FILENAME so tests
# that monkeypatch DB_PATH (e.g. tmp_path fixtures) keep working without
# having to also patch this module. Lazy import avoids pulling rag.py at
# module load time — only when the writer first needs to open a conn.
def _resolve_telemetry_db_path() -> Path:
    import rag
    return rag.DB_PATH / rag._TELEMETRY_DB_FILENAME

_FRONTMATTER_KEYS = ("session_id", "created", "updated", "turns", "confidence_avg", "sources", "tags")
_TAGS = ("conversation", "rag-chat")


@dataclass(frozen=True)
class TurnData:
    question: str
    answer: str
    sources: list[dict]
    confidence: float
    timestamp: datetime


def slugify(text: str, *, max_len: int = 50) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "conversation"


def _iso_z(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_seconds_now() -> str:
    # Matches T1's convention for `updated_at` on rag_* tables.
    return datetime.now().isoformat(timespec="seconds")


def _open_sql_conn() -> sqlite3.Connection:
    # Autocommit mode + WAL + busy_timeout match SqliteVecClient's settings so
    # the writer coexists with the long-lived vec client without lock
    # contention. check_same_thread=False because the caller is a daemon
    # thread spawned per /api/chat turn.
    db_path = _resolve_telemetry_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path), isolation_level=None, check_same_thread=False, timeout=30.0
    )
    # busy_timeout FIRST — every subsequent PRAGMA then honours it instead
    # of returning SQLITE_BUSY immediately. Critical under multi-process
    # stampede where 20 workers all spawn conns within ~10ms of each other.
    conn.execute("PRAGMA busy_timeout=30000")
    # journal_mode=WAL briefly takes an exclusive lock to flip the header,
    # so skip the write once WAL is already active (idempotent). Saves
    # ~N² contention when multiple fresh conns all race to SET WAL.
    # Retry the PRAGMA explicitly on a lock conflict — sqlite's C layer can
    # return SQLITE_BUSY (OperationalError) for PRAGMA statements even with
    # busy_timeout set, specifically when multiple fresh processes race to
    # flip an unset journal_mode (verified flaky under mp.Pool(8) hammering).
    # Each iteration attempts afresh after a short jittered sleep.
    for attempt in range(10):
        try:
            cur = conn.execute("PRAGMA journal_mode")
            row = cur.fetchone()
            current_mode = (row[0] if row else "").lower()
            if current_mode == "wal":
                break
            conn.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == 9:
                raise
            import time as _t
            import random as _r
            _t.sleep(0.05 + _r.random() * 0.15)
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_conversations_table(conn: sqlite3.Connection) -> None:
    # Cheap CREATE IF NOT EXISTS guard in case the writer races ahead of
    # SqliteVecClient init on a fresh install. Schema mirrors T1's DDL.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_conversations_index ("
        " session_id TEXT PRIMARY KEY,"
        " relative_path TEXT NOT NULL,"
        " updated_at TEXT NOT NULL"
        ")"
    )


def persist_conversation_index_entry(session_id: str, relative_path: str) -> None:
    """Upsert (session_id → relative_path) into rag_conversations_index.

    SQL-only since T10. Safe to call concurrently — BEGIN IMMEDIATE +
    per-process busy_timeout=30s serialise writes cleanly; no JSON-on-disk
    lock needed. Under heavy multi-process stampede (verified flaky on
    mp.Pool(8) in tests), a stray `sqlite3.OperationalError("database is
    locked")` can still escape busy_timeout on the `CREATE TABLE IF NOT
    EXISTS` + `BEGIN IMMEDIATE` pair — we retry up to 10 times with
    jittered backoff before propagating. Each retry opens a fresh
    connection so a half-locked txn state can't survive.
    """
    import random as _r
    import time as _t
    last_exc: Exception | None = None
    for attempt in range(10):
        conn = _open_sql_conn()
        try:
            _ensure_conversations_table(conn)
            conn.execute("BEGIN IMMEDIATE")
            # _sql_upsert is the T1 primitive. Lazy-import to avoid hauling
            # rag.py into every web module at import time.
            import rag
            rag._sql_upsert(
                conn,
                "rag_conversations_index",
                {
                    "session_id": session_id,
                    "relative_path": relative_path,
                    "updated_at": _iso_seconds_now(),
                },
                pk_cols=("session_id",),
            )
            return
        except sqlite3.OperationalError as exc:
            last_exc = exc
            if "locked" not in str(exc).lower():
                raise
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
        finally:
            conn.close()
        _t.sleep(0.05 + _r.random() * 0.15)
    if last_exc is not None:
        raise last_exc


def get_conversation_path(session_id: str) -> str | None:
    """Return the vault-relative path for `session_id`, or None.

    SQL-only since T10.
    """
    conn = _open_sql_conn()
    try:
        _ensure_conversations_table(conn)
        row = conn.execute(
            "SELECT relative_path FROM rag_conversations_index WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row is not None else None


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        raise ValueError("missing frontmatter opening")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("missing frontmatter closing")
    block = text[4:end]
    body = text[end + 5 :]
    meta: dict = {}
    current_list: list | None = None
    current_key: str | None = None
    for line in block.split("\n"):
        if not line:
            continue
        if line.startswith("  - "):
            if current_list is None or current_key is None:
                raise ValueError(f"list item without key: {line!r}")
            current_list.append(line[4:].strip())
            continue
        if ": " in line or line.endswith(":"):
            key, _, rest = line.partition(":")
            key = key.strip()
            value = rest.strip()
            if not value:
                current_list = []
                current_key = key
                meta[key] = current_list
            elif value == "[]":
                # `_render_frontmatter` writes empty lists as inline `[]` literal
                # text. Without this branch the parser stored the string `"[]"`,
                # failing `isinstance(existing_sources, list)` on the next
                # turn with `ValueError("sources must be a list")`. Reported
                # via `conversation_turn_pending.jsonl` on zero-source turns.
                current_list = None
                current_key = None
                meta[key] = []
            else:
                current_list = None
                current_key = None
                meta[key] = value
            continue
        raise ValueError(f"unparseable frontmatter line: {line!r}")
    return meta, body


def _render_frontmatter(meta: dict) -> str:
    lines = ["---"]
    for key in _FRONTMATTER_KEYS:
        value = meta[key]
        if isinstance(value, list):
            if not value:
                lines.append(f"{key}: []")
            else:
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _render_turn_block(turn_n: int, turn: TurnData) -> str:
    hhmm = turn.timestamp.strftime("%H:%M")
    wikilinks = []
    seen: set[str] = set()
    for src in turn.sources:
        f = src.get("file", "")
        if not f or f in seen:
            continue
        seen.add(f)
        wikilinks.append(f"[[{f[:-3] if f.endswith('.md') else f}]]")
    sources_line = "**Sources**: " + (" · ".join(wikilinks) if wikilinks else "—")
    return (
        f"## Turn {turn_n} — {hhmm}\n\n"
        f"> {turn.question}\n\n"
        f"{turn.answer}\n\n"
        f"{sources_line}\n"
    )


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Write + fsync + rename: on a crash between write and rename, the tmp
    # is orphaned (GC-able) but the target is either the old file or the
    # new one — never half-written. Without fsync, the rename can land
    # before the data pages, leaving an empty file at `path` after crash.
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _union_sources(existing: list[str], turn_sources: list[dict]) -> list[str]:
    merged = set(existing)
    for src in turn_sources:
        f = src.get("file", "")
        if f:
            merged.add(f)
    return sorted(merged)


def _write_turn_body(
    vault_root: Path, folder: Path, session_id: str, turn: TurnData
) -> Path:
    rel = get_conversation_path(session_id)
    target = vault_root / rel if rel else None

    if target and target.exists():
        existing_text = target.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(existing_text)
        try:
            old_turns = int(meta["turns"])
            old_avg = float(meta["confidence_avg"])
            created = meta["created"]
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"malformed frontmatter in {target}: {exc}")
        # Defense-in-depth: historical notes persisted pre-sanitize may carry
        # `confidence_avg: -inf` / `nan`. Without this clamp, subsequent turns
        # propagate the invalid value forever (`-inf * k + x == -inf`).
        import math
        if math.isnan(old_avg) or math.isinf(old_avg):
            old_avg = 0.0
        new_turns = old_turns + 1
        new_avg = (old_avg * old_turns + turn.confidence) / new_turns
        existing_sources = meta.get("sources", [])
        if not isinstance(existing_sources, list):
            raise ValueError("sources must be a list")
        sources_union = _union_sources(existing_sources, turn.sources)
        new_meta = {
            "session_id": session_id,
            "created": created,
            "updated": _iso_z(turn.timestamp),
            "turns": new_turns,
            "confidence_avg": f"{new_avg:.3f}",
            "sources": sources_union,
            "tags": list(_TAGS),
        }
        block = _render_turn_block(new_turns, turn)
        body_trimmed = body.rstrip() + "\n\n"
        new_text = _render_frontmatter(new_meta) + "\n" + body_trimmed + block
        # SQL-first / disk-second: commit the index entry before touching
        # the .md file. If disk write fails (e.g. full vault), the index
        # still points to an existing (stale by one turn) note — not to a
        # ghost. The inverse order would create the .md and then fail the
        # index upsert, fragmenting the session on the next turn.
        persist_conversation_index_entry(session_id, str(target.relative_to(vault_root)))
        _atomic_write(target, new_text)
        return target

    slug = slugify(turn.question)
    filename = f"{turn.timestamp.strftime('%Y-%m-%d-%H%M')}-{slug}.md"
    target = folder / filename
    created_iso = _iso_z(turn.timestamp)
    sources_union = _union_sources([], turn.sources)
    new_meta = {
        "session_id": session_id,
        "created": created_iso,
        "updated": created_iso,
        "turns": 1,
        "confidence_avg": f"{turn.confidence:.3f}",
        "sources": sources_union,
        "tags": list(_TAGS),
    }
    block = _render_turn_block(1, turn)
    new_text = _render_frontmatter(new_meta) + "\n" + block
    # SQL-first: if the upsert fails the .md never gets written, so there's
    # no orphaned note on disk to mask the error. The caller (web/server.py)
    # will see the exception and log it as conversation_turn_error.
    persist_conversation_index_entry(session_id, str(target.relative_to(vault_root)))
    _atomic_write(target, new_text)
    return target


def write_turn(
    vault_root: Path,
    session_id: str,
    turn: TurnData,
    *,
    subfolder: str = "04-Archive/99-obsidian-system/99-Claude/conversations",
) -> Path:
    """Append one turn to the session's .md + upsert the SQL index.

    SQL-only since T10. Concurrency is bounded by one /api/chat per session
    at a time + SQLite's busy_timeout in the index upsert; the whole-body
    fcntl lock from the pre-T10 JSON path is gone.
    """
    folder = vault_root / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    return _write_turn_body(vault_root, folder, session_id, turn)
