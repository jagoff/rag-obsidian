from __future__ import annotations

import fcntl
import json
import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_INDEX_PATH: Path = Path.home() / ".local/share/obsidian-rag" / "conversations_index.json"

# ragvec.db lives next to sqlite-vec data. Kept in sync with rag.DB_PATH — if
# the env var OBSIDIAN_RAG_VAULT changes the vault at runtime the SQL writer
# still targets the same physical file used by rag.py + SqliteVecClient, since
# both resolve via Path.home() identically in this codebase.
_DB_PATH: Path = Path.home() / ".local/share/obsidian-rag" / "ragvec" / "ragvec.db"

_FRONTMATTER_KEYS = ("session_id", "created", "updated", "turns", "confidence_avg", "sources", "tags")
_TAGS = ("conversation", "rag-chat")

# 7-day cutover: when set, the index lookup + persistence go through SQL.
# JSON fallback on read still runs so sessions that started pre-cutover keep
# finding their on-disk .md. T10 will strip the JSON path entirely.
_SQL_FLAG_ENV = "RAG_STATE_SQL"


def _sql_enabled() -> bool:
    # Read env each call — tests flip the flag between runs, and this is
    # cheap (os.environ is a dict lookup).
    return os.environ.get(_SQL_FLAG_ENV, "").strip() == "1"


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


def _read_index_json() -> dict[str, str]:
    try:
        raw = _INDEX_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_index_json(mapping: dict[str, str]) -> None:
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _INDEX_PATH.with_suffix(_INDEX_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, _INDEX_PATH)


def _open_sql_conn() -> sqlite3.Connection:
    # Autocommit mode + WAL + busy_timeout match SqliteVecClient's settings so
    # the writer coexists with the long-lived vec client without lock
    # contention. check_same_thread=False because the caller is a daemon
    # thread spawned per /api/chat turn.
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(_DB_PATH), isolation_level=None, check_same_thread=False, timeout=30.0
    )
    # busy_timeout FIRST — every subsequent PRAGMA then honours it instead
    # of returning SQLITE_BUSY immediately. Critical under multi-process
    # stampede where 20 workers all spawn conns within ~10ms of each other.
    conn.execute("PRAGMA busy_timeout=30000")
    # journal_mode=WAL briefly takes an exclusive lock to flip the header,
    # so skip the write once WAL is already active (idempotent). Saves
    # ~N² contention when multiple fresh conns all race to SET WAL.
    cur = conn.execute("PRAGMA journal_mode")
    row = cur.fetchone()
    current_mode = (row[0] if row else "").lower()
    if current_mode != "wal":
        conn.execute("PRAGMA journal_mode=WAL")
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


def _persist_json_unlocked(session_id: str, relative_path: str) -> None:
    mapping = _read_index_json()
    mapping[session_id] = relative_path
    _write_index_json(mapping)


def _persist_index_from_within_write_turn(session_id: str, relative_path: str) -> None:
    # write_turn's JSON branch already holds the fcntl lock, so the JSON
    # persist must skip re-locking (flock on a fresh fd would block).
    # SQL mode has no outer lock — go through the public path to run the
    # BEGIN IMMEDIATE upsert.
    if _sql_enabled():
        persist_conversation_index_entry(session_id, relative_path)
    else:
        _persist_json_unlocked(session_id, relative_path)


def persist_conversation_index_entry(session_id: str, relative_path: str) -> None:
    """Upsert (session_id → relative_path) into the active index.

    Flag ON → SQL UPSERT inside BEGIN IMMEDIATE against rag_conversations_index.
    Flag OFF → atomic JSON replace under fcntl.flock (legacy path).

    Safe to call standalone (public API). When invoked from inside write_turn
    in JSON mode, the outer fcntl lock already guards the map and this function
    bypasses re-locking via the `_sql_enabled()` branch vs. legacy branch.
    """
    if _sql_enabled():
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
        finally:
            conn.close()
        return

    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _INDEX_PATH.with_suffix(_INDEX_PATH.suffix + ".lock")
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        try:
            _persist_json_unlocked(session_id, relative_path)
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)


def get_conversation_path(session_id: str) -> str | None:
    """Return the vault-relative path for `session_id`, or None.

    Flag ON → SELECT from rag_conversations_index; fall through to JSON if
    row missing (7-day cutover safety — pre-cutover sessions still resolve).
    Flag OFF → JSON lookup only.
    """
    if _sql_enabled():
        conn = _open_sql_conn()
        try:
            _ensure_conversations_table(conn)
            row = conn.execute(
                "SELECT relative_path FROM rag_conversations_index WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is not None:
            return row[0]
        # Cutover fallback: JSON may still hold sessions written pre-flag.
        return _read_index_json().get(session_id)

    return _read_index_json().get(session_id)


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
        _persist_index_from_within_write_turn(session_id, str(target.relative_to(vault_root)))
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
    _persist_index_from_within_write_turn(session_id, str(target.relative_to(vault_root)))
    _atomic_write(target, new_text)
    return target


def write_turn(
    vault_root: Path,
    session_id: str,
    turn: TurnData,
    *,
    subfolder: str = "00-Inbox/conversations",
) -> Path:
    folder = vault_root / subfolder
    folder.mkdir(parents=True, exist_ok=True)

    if _sql_enabled():
        # SQL path: row-level lock inside persist_conversation_index_entry
        # serialises index upserts. Same-session concurrent .md writes are
        # not a production concern (one /api/chat request per session at a
        # time) — removing the whole-body fcntl lock is the point of T5.
        return _write_turn_body(vault_root, folder, session_id, turn)

    # Legacy JSON path: whole-body fcntl lock serialises the .md
    # read-modify-write + index write. Unchanged until T10.
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _INDEX_PATH.with_suffix(_INDEX_PATH.suffix + ".lock")
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        try:
            return _write_turn_body(vault_root, folder, session_id, turn)
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
