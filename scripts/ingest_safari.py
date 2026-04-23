"""Safari ingester — Phase 2 of the cross-source corpus.

Reads `~/Library/Safari/History.db` (SQLite) for browsing history and
`~/Library/Safari/Bookmarks.plist` (binary plist) for bookmarks +
Reading List. Emits one chunk per URL (deduplicated across visits) and
one chunk per bookmark/reading-list entry.

Complements the inline Chrome ingester (`rag.py:_sync_chrome_history`)
which writes .md to the vault. This ingester uses the source-prefixed
chunk architecture (same pattern as whatsapp/calls/contacts) — smaller,
deduplicated, no vault pollution.

Ingest strategy:
  - History: SQLite JOIN `history_items ← history_visits` via `history_item`.
    Per URL: latest visit title, visit_count aggregate, first/last visit ts,
    filter `load_successful=1` + `redirect_source IS NULL` (drop redirect
    duplicates). Retention 180d on last-visit timestamp.
  - Bookmarks: recursive walk over `Children` tree, collecting
    `WebBookmarkTypeLeaf` entries with their folder path. Reading List
    is a named sub-tree ("com.apple.ReadingList") that gets tagged
    separately via `safari_type=reading_list`.
  - Chunk per item (<800 chars, atomic). Retention None for bookmarks
    (they're curated, don't age the same way).
  - Source weight: 0.80 (similar to calls — signal-rich but not as
    high-trust as calendar/contacts).
  - Recency halflife: 90d for history, None for bookmarks.

Incremental sync:
  - `rag_safari_history_state(history_item_id, content_hash, last_seen_ts,
    updated_at)`. Content hash over (title, url, visit_count, last_visit).
  - `rag_safari_bookmark_state(bookmark_uuid, content_hash, last_seen_ts,
    updated_at)`. Content hash over (title, url, folder_path).
  - Stale deletion per-table.

Invoked via `rag index --source safari [--reset] [--since ISO]
[--dry-run] [--skip-bookmarks] [--max-urls N]`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


# ── Config ─────────────────────────────────────────────────────────────────

DEFAULT_HISTORY_DB = Path.home() / "Library/Safari/History.db"
DEFAULT_BOOKMARKS_PLIST = Path.home() / "Library/Safari/Bookmarks.plist"
CHUNK_MAX_CHARS = 800
DOC_ID_PREFIX = "safari"
DEFAULT_HISTORY_RETENTION_DAYS = 180
DEFAULT_MAX_URLS = 5000  # cap per-run to avoid multi-minute indexing

# col.add batch size. 1000+ items in one transaction vs the web server's
# sporadic writes was measured on the dev host to reliably hit
# `database is locked` on the bookmarks pass (1073 rows typical). At 200
# each transaction holds the writer lock for < ~1s, leaving slots for
# the server to slot its own writes in. Rationale mirrors the Chrome
# sync (rag.py:_index_chrome_bookmarks batch_size=256 default).
_ADD_BATCH_SIZE = 50

COCOA_EPOCH_OFFSET = 978307200.0

# Reading List lives as a well-known subtree in Bookmarks.plist.
READING_LIST_TITLE = "com.apple.ReadingList"


# ── Data types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HistoryEntry:
    history_item_id: int
    url: str
    domain: str                  # host only
    title: str                   # latest non-empty visit title
    visit_count: int
    first_visit_ts: float        # Unix epoch
    last_visit_ts: float


@dataclass(frozen=True)
class Bookmark:
    uuid: str
    url: str
    title: str
    folder_path: str             # "/".join(parent folder titles)
    is_reading_list: bool = False


# ── Timestamp helpers ──────────────────────────────────────────────────────

def _cocoa_to_unix(raw: float | int | None) -> float:
    if raw is None:
        return 0.0
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if val <= 0:
        return 0.0
    return val + COCOA_EPOCH_OFFSET


def _domain(url: str) -> str:
    """Extract the host from a URL. Safe on malformed inputs — returns
    '' rather than raising (callers show the raw URL as fallback)."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    return (parsed.hostname or "").lower()


# ── History reader ─────────────────────────────────────────────────────────

# Aggregated per-URL view: most-recent non-empty title, total visits,
# first/last visit time. Drops redirects (we want landing pages, not
# intermediate 301s). `load_successful=1` cuts failed requests.
_HISTORY_SQL = """
SELECT
  hi.id,
  hi.url,
  COALESCE(hi.domain_expansion, ''),
  COALESCE(hi.visit_count, 0),
  MIN(hv.visit_time),
  MAX(hv.visit_time),
  -- Title: prefer the most-recent non-empty one.
  (
    SELECT title FROM history_visits
    WHERE history_item = hi.id
      AND title IS NOT NULL AND title != ''
      AND load_successful = 1
      AND redirect_source IS NULL
    ORDER BY visit_time DESC LIMIT 1
  )
FROM history_items hi
JOIN history_visits hv ON hv.history_item = hi.id
WHERE hv.load_successful = 1
  AND hv.redirect_source IS NULL
  AND hv.visit_time >= ?
GROUP BY hi.id
ORDER BY MAX(hv.visit_time) DESC
LIMIT ?
"""


def read_history(
    db_path: Path = DEFAULT_HISTORY_DB,
    *,
    since_unix_ts: float = 0.0,
    max_urls: int = DEFAULT_MAX_URLS,
) -> list[HistoryEntry]:
    """Load Safari history aggregated by URL. Filters:
      - `load_successful = 1` (drop 404s, net failures)
      - `redirect_source IS NULL` (drop 301/302 intermediate hops)
      - `visit_time >= since_unix_ts` in Cocoa epoch, converted inline
      - `max_urls` cap (most-recent-first)
    Returns [] if the DB is missing (host without Safari usage).
    """
    if not db_path.exists():
        return []
    cocoa_cutoff = max(0.0, since_unix_ts - COCOA_EPOCH_OFFSET)
    # Read-only + immutable — Safari may hold write locks.
    uri = f"file:{db_path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    try:
        out: list[HistoryEntry] = []
        for row in conn.execute(_HISTORY_SQL, (cocoa_cutoff, max_urls)):
            hid, url, domain_expansion, vcount, first_ts, last_ts, title = row
            if not url:
                continue
            out.append(HistoryEntry(
                history_item_id=int(hid),
                url=str(url),
                domain=_domain(url) or str(domain_expansion or ""),
                title=(title or "").strip(),
                visit_count=int(vcount or 0),
                first_visit_ts=_cocoa_to_unix(first_ts),
                last_visit_ts=_cocoa_to_unix(last_ts),
            ))
        return out
    finally:
        conn.close()


# ── Bookmarks reader ───────────────────────────────────────────────────────

def _walk_bookmarks(
    node: dict,
    folder_stack: list[str],
    out: list[Bookmark],
    *,
    under_reading_list: bool = False,
) -> None:
    """Recursively walk Safari's nested Bookmarks plist tree.
    `folder_stack` carries parent folder titles for display_path.
    `under_reading_list` flips True once we cross the RL subtree.
    """
    if not isinstance(node, dict):
        return
    btype = node.get("WebBookmarkType", "")
    title = (node.get("Title") or "").strip()
    uuid = node.get("WebBookmarkUUID") or ""

    if btype == "WebBookmarkTypeList":
        # Folder. Check if we're entering the RL subtree.
        next_stack = folder_stack
        rl_flag = under_reading_list
        if title == READING_LIST_TITLE:
            rl_flag = True
            next_stack = ["Reading List"]
        elif title:
            # Skip default/internal folders that don't add info.
            if title not in ("BookmarksBar", "BookmarksMenu"):
                next_stack = folder_stack + [title]
        for child in (node.get("Children") or []):
            _walk_bookmarks(child, next_stack, out,
                            under_reading_list=rl_flag)
    elif btype == "WebBookmarkTypeLeaf":
        url = (node.get("URLString") or "").strip()
        if not url:
            return
        # Title is inside URIDictionary for leaves.
        uri_dict = node.get("URIDictionary") or {}
        leaf_title = (uri_dict.get("title") or node.get("Title") or "").strip()
        # Reading List entries carry a ReadingListNonSync with preview text —
        # pull it if present for a richer chunk body.
        rl_preview = ""
        rl_meta = node.get("ReadingList") or {}
        if isinstance(rl_meta, dict):
            rl_preview = (rl_meta.get("PreviewText") or "").strip()
        extra = f" — {rl_preview}" if under_reading_list and rl_preview else ""
        out.append(Bookmark(
            uuid=str(uuid) if uuid else f"leaf:{hash(url):x}",
            url=url,
            title=(leaf_title + extra)[:400],  # soft cap
            folder_path="/".join(folder_stack) if folder_stack else "",
            is_reading_list=bool(under_reading_list),
        ))
    # WebBookmarkTypeProxy = History / RL placeholder link. Skip.


def read_bookmarks(plist_path: Path = DEFAULT_BOOKMARKS_PLIST) -> list[Bookmark]:
    if not plist_path.exists():
        return []
    try:
        with open(plist_path, "rb") as f:
            data = plistlib.load(f)
    except (plistlib.InvalidFileException, OSError):
        return []
    out: list[Bookmark] = []
    _walk_bookmarks(data, [], out)
    return out


# ── State ──────────────────────────────────────────────────────────────────

_HISTORY_STATE_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_safari_history_state ("
    " history_item_id INTEGER PRIMARY KEY,"
    " content_hash TEXT NOT NULL,"
    " last_seen_ts TEXT NOT NULL,"
    " updated_at TEXT NOT NULL"
    ")"
)

_BOOKMARK_STATE_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_safari_bookmark_state ("
    " bookmark_uuid TEXT PRIMARY KEY,"
    " content_hash TEXT NOT NULL,"
    " last_seen_ts TEXT NOT NULL,"
    " updated_at TEXT NOT NULL"
    ")"
)


def _ensure_state_tables(conn: sqlite3.Connection) -> None:
    conn.execute(_HISTORY_STATE_DDL)
    conn.execute(_BOOKMARK_STATE_DDL)


def _load_hashes(conn: sqlite3.Connection, table: str, key_col: str) -> dict[str, str]:
    cur = conn.execute(f"SELECT {key_col}, content_hash FROM {table}")
    return {str(row[0]): row[1] for row in cur.fetchall()}


def _upsert_hash(
    conn: sqlite3.Connection, table: str, key_col: str,
    key: str | int, h: str, now_iso: str,
) -> None:
    conn.execute(
        f"INSERT OR REPLACE INTO {table} "
        f"({key_col}, content_hash, last_seen_ts, updated_at) "
        f"VALUES (?, ?, ?, ?)",
        (key, h, now_iso, now_iso),
    )


def _delete_hash(conn: sqlite3.Connection, table: str, key_col: str, key: str | int) -> None:
    conn.execute(f"DELETE FROM {table} WHERE {key_col} = ?", (key,))


def _reset_state(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM rag_safari_history_state")
    conn.execute("DELETE FROM rag_safari_bookmark_state")


def _history_hash(h: HistoryEntry) -> str:
    payload = "\n".join([
        str(h.history_item_id), h.url, h.title,
        str(h.visit_count), f"{h.last_visit_ts:.0f}",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _bookmark_hash(b: Bookmark) -> str:
    payload = "\n".join([
        b.uuid, b.url, b.title, b.folder_path,
        f"{int(b.is_reading_list)}",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── Chunker ────────────────────────────────────────────────────────────────

def _format_history_body(h: HistoryEntry) -> str:
    headline = h.title or h.url
    parts: list[str] = [f"Safari: {headline}"]
    if h.title and h.title != h.url:
        parts.append(f"URL: {h.url}")
    if h.domain:
        parts.append(f"Dominio: {h.domain}")
    if h.last_visit_ts:
        dt = datetime.fromtimestamp(h.last_visit_ts, tz=timezone.utc).astimezone()
        parts.append(f"Última visita: {dt.strftime('%Y-%m-%d %H:%M')}")
    parts.append(f"Visitas: {h.visit_count}")
    body = "\n".join(parts)
    if len(body) > CHUNK_MAX_CHARS:
        body = body[:CHUNK_MAX_CHARS].rstrip()
    return body


def _format_bookmark_body(b: Bookmark) -> str:
    marker = "Reading List" if b.is_reading_list else "Bookmark"
    parts: list[str] = [f"Safari {marker}: {b.title or b.url}"]
    if b.title and b.title != b.url:
        parts.append(f"URL: {b.url}")
    if b.folder_path:
        parts.append(f"Carpeta: {b.folder_path}")
    dom = _domain(b.url)
    if dom:
        parts.append(f"Dominio: {dom}")
    body = "\n".join(parts)
    if len(body) > CHUNK_MAX_CHARS:
        body = body[:CHUNK_MAX_CHARS].rstrip()
    return body


def _history_embed_prefix(h: HistoryEntry, body: str) -> str:
    return (
        f"[source=safari | kind=history | domain={h.domain} | "
        f"visits={h.visit_count}] {body}"
    )


def _bookmark_embed_prefix(b: Bookmark, body: str) -> str:
    kind = "reading_list" if b.is_reading_list else "bookmark"
    return (
        f"[source=safari | kind={kind} | "
        f"folder={b.folder_path or '(root)'}] {body}"
    )


def _history_doc_id(h: HistoryEntry) -> str:
    return f"{DOC_ID_PREFIX}://history/{h.history_item_id}::0"


def _history_file_key(h: HistoryEntry) -> str:
    return f"{DOC_ID_PREFIX}://history/{h.history_item_id}"


def _bookmark_doc_id(b: Bookmark) -> str:
    kind = "rl" if b.is_reading_list else "bm"
    return f"{DOC_ID_PREFIX}://{kind}/{b.uuid}::0"


def _bookmark_file_key(b: Bookmark) -> str:
    kind = "rl" if b.is_reading_list else "bm"
    return f"{DOC_ID_PREFIX}://{kind}/{b.uuid}"


# ── Writer ─────────────────────────────────────────────────────────────────

def _add_batched(col, ids, embeddings, bodies, metas, source: str) -> None:
    """Split a potentially large col.add() into `_ADD_BATCH_SIZE` chunks
    with an exponential-backoff retry loop on `database is locked`.

    Each batch's col.add does 3 sqlite-vec writes per chunk (meta UPSERT
    + vec0 DELETE + vec0 INSERT). On the dev host with `rag serve`
    actively writing behavior events, the 1073-bookmark Safari pass
    reliably hit `SQLITE_BUSY` even at batch=200. Default SQLite
    `busy_timeout` on sqlite-vec is 0 (no wait), so any concurrent
    writer causes an immediate failure.

    Retry: up to 8 attempts per batch, sleep = 0.25 × 2^attempt seconds
    (0.25, 0.5, 1, 2, 4, 8, 16, 32s → ~64s cumulative max). If the last
    attempt still fails, re-raise — something beyond contention is wrong.

    Between batches sqlite-vec's writer lock releases briefly, letting
    the concurrent web server slot its own writes in.
    """
    import time as _time
    for i in range(0, len(ids), _ADD_BATCH_SIZE):
        j = i + _ADD_BATCH_SIZE
        last_exc: Exception | None = None
        for attempt in range(8):
            try:
                col.add(
                    ids=ids[i:j], embeddings=embeddings[i:j],
                    documents=bodies[i:j], metadatas=metas[i:j],
                )
                last_exc = None
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                last_exc = exc
                _time.sleep(0.25 * (2 ** attempt))
        if last_exc is not None:
            raise last_exc
        rag._extract_and_index_entities_for_chunks(
            bodies[i:j], ids[i:j], metas[i:j], source,
        )


def upsert_history(col, entries: list[HistoryEntry]) -> int:
    if not entries:
        return 0
    bodies = [_format_history_body(h) for h in entries]
    embed_texts = [_history_embed_prefix(h, b) for h, b in zip(entries, bodies)]
    embeddings = rag.embed(embed_texts)
    for h in entries:
        key = _history_file_key(h)
        try:
            existing = col.get(where={"file": key}, include=[])
            if existing.get("ids"):
                col.delete(ids=existing["ids"])
        except Exception:
            pass
    ids = [_history_doc_id(h) for h in entries]
    metas: list[dict] = []
    for h, body in zip(entries, bodies):
        metas.append({
            "file": _history_file_key(h),
            "note": f"Safari: {(h.title or h.url)[:60]}",
            "folder": "",
            "tags": "",
            "hash": _history_hash(h),
            "outlinks": "",
            "source": "safari",
            "kind": "history",
            "created_ts": h.last_visit_ts,
            "url": h.url,
            "domain": h.domain,
            "title": h.title,
            "visit_count": h.visit_count,
            "first_visit_ts": h.first_visit_ts,
            "last_visit_ts": h.last_visit_ts,
            "history_item_id": h.history_item_id,
            "parent": body,
        })
    _add_batched(col, ids, embeddings, bodies, metas, "safari")
    return len(entries)


def upsert_bookmarks(col, marks: list[Bookmark]) -> int:
    if not marks:
        return 0
    bodies = [_format_bookmark_body(b) for b in marks]
    embed_texts = [_bookmark_embed_prefix(b, body)
                    for b, body in zip(marks, bodies)]
    embeddings = rag.embed(embed_texts)
    for b in marks:
        key = _bookmark_file_key(b)
        try:
            existing = col.get(where={"file": key}, include=[])
            if existing.get("ids"):
                col.delete(ids=existing["ids"])
        except Exception:
            pass
    ids = [_bookmark_doc_id(b) for b in marks]
    metas: list[dict] = []
    for b, body in zip(marks, bodies):
        metas.append({
            "file": _bookmark_file_key(b),
            "note": f"Safari {'RL' if b.is_reading_list else 'BM'}: {(b.title or b.url)[:60]}",
            "folder": "",
            "tags": "",
            "hash": _bookmark_hash(b),
            "outlinks": "",
            "source": "safari",
            "kind": "reading_list" if b.is_reading_list else "bookmark",
            "created_ts": 0.0,  # bookmarks don't carry a creation ts in the plist
            "url": b.url,
            "domain": _domain(b.url),
            "title": b.title,
            "folder_path": b.folder_path,
            "is_reading_list": int(b.is_reading_list),
            "bookmark_uuid": b.uuid,
            "parent": body,
        })
    _add_batched(col, ids, embeddings, bodies, metas, "safari")
    return len(marks)


def delete_history(col, hist_item_ids: list[int]) -> int:
    if not hist_item_ids:
        return 0
    n = 0
    for hid in hist_item_ids:
        key = f"{DOC_ID_PREFIX}://history/{hid}"
        try:
            got = col.get(where={"file": key}, include=[])
            if got.get("ids"):
                col.delete(ids=got["ids"])
                n += 1
        except Exception:
            continue
    return n


def delete_bookmarks(col, keys: list[str]) -> int:
    """`keys` come from state DB — they're the full `file` key
    (`safari://bm/<uuid>` or `safari://rl/<uuid>`), not just the uuid."""
    if not keys:
        return 0
    n = 0
    for key in keys:
        try:
            got = col.get(where={"file": key}, include=[])
            if got.get("ids"):
                col.delete(ids=got["ids"])
                n += 1
        except Exception:
            continue
    return n


# ── Orchestration ──────────────────────────────────────────────────────────

def run(
    *,
    reset: bool = False,
    dry_run: bool = False,
    since_iso: str | None = None,
    retention_days: int | None = None,
    max_urls: int = DEFAULT_MAX_URLS,
    skip_bookmarks: bool = False,
    history_db: Path | None = None,
    bookmarks_plist: Path | None = None,
    vault_col=None,
    history_fetch_fn: Callable[[], list[HistoryEntry]] | None = None,
    bookmarks_fetch_fn: Callable[[], list[Bookmark]] | None = None,
    now: datetime | None = None,
) -> dict:
    t0 = time.perf_counter()
    summary: dict = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "reset": bool(reset),
        "dry_run": bool(dry_run),
        "history_fetched": 0,
        "history_indexed": 0,
        "history_unchanged": 0,
        "history_deleted": 0,
        "bookmarks_fetched": 0,
        "bookmarks_indexed": 0,
        "bookmarks_unchanged": 0,
        "bookmarks_deleted": 0,
        "reading_list_fetched": 0,
        "duration_s": 0.0,
    }

    retention_days = (
        DEFAULT_HISTORY_RETENTION_DAYS if retention_days is None else retention_days
    )
    current_now = (now or datetime.now()).timestamp()
    floor_ts = 0.0
    if retention_days and retention_days > 0:
        floor_ts = current_now - (retention_days * 86400)
    if since_iso:
        try:
            since_dt = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
            floor_ts = max(floor_ts, since_dt.timestamp())
        except ValueError:
            summary["error"] = f"--since inválido: {since_iso!r}"
            summary["duration_s"] = round(time.perf_counter() - t0, 2)
            return summary

    hdb = Path(history_db) if history_db is not None else DEFAULT_HISTORY_DB
    bpl = Path(bookmarks_plist) if bookmarks_plist is not None else DEFAULT_BOOKMARKS_PLIST

    h_fetch = history_fetch_fn if history_fetch_fn is not None else (
        lambda: read_history(hdb, since_unix_ts=floor_ts, max_urls=max_urls)
    )
    b_fetch = bookmarks_fetch_fn if bookmarks_fetch_fn is not None else (
        lambda: read_bookmarks(bpl)
    )

    history = h_fetch() or []
    summary["history_fetched"] = len(history)
    bookmarks = [] if skip_bookmarks else (b_fetch() or [])
    summary["bookmarks_fetched"] = sum(1 for b in bookmarks if not b.is_reading_list)
    summary["reading_list_fetched"] = sum(1 for b in bookmarks if b.is_reading_list)

    if not history and not bookmarks and not hdb.exists() and not bpl.exists():
        summary["error"] = "Safari DB / Bookmarks no encontrados"
        summary["duration_s"] = round(time.perf_counter() - t0, 2)
        return summary

    col = vault_col if vault_col is not None else rag.get_db()
    state_conn = sqlite3.connect(str(rag.DB_PATH / "ragvec.db"))
    _ensure_state_tables(state_conn)
    if reset:
        _reset_state(state_conn)
    state_conn.commit()

    # History diff.
    prior_hist_hashes = _load_hashes(
        state_conn, "rag_safari_history_state", "history_item_id"
    )
    live_hist_keys: set[str] = set()
    hist_to_write: list[HistoryEntry] = []
    for h in history:
        k = str(h.history_item_id)
        live_hist_keys.add(k)
        nh = _history_hash(h)
        if prior_hist_hashes.get(k) != nh:
            hist_to_write.append(h)
        else:
            summary["history_unchanged"] += 1
    stale_hist_ids = [int(k) for k in prior_hist_hashes if k not in live_hist_keys]

    # Bookmarks diff.
    prior_bm_hashes = _load_hashes(
        state_conn, "rag_safari_bookmark_state", "bookmark_uuid"
    )
    live_bm_keys: set[str] = set()
    bm_to_write: list[Bookmark] = []
    for b in bookmarks:
        live_bm_keys.add(b.uuid)
        nh = _bookmark_hash(b)
        if prior_bm_hashes.get(b.uuid) != nh:
            bm_to_write.append(b)
        else:
            summary["bookmarks_unchanged"] += 1
    # Stale uuids: we don't know if it was bm or rl retroactively; reconstruct
    # the key by trying both. Easier: iterate prior hashes and try the two
    # file keys (safari://bm/<uuid> and safari://rl/<uuid>) — delete any match.
    stale_bm_uuids = [u for u in prior_bm_hashes if u not in live_bm_keys]
    stale_bm_keys: list[str] = []
    for u in stale_bm_uuids:
        stale_bm_keys.append(f"{DOC_ID_PREFIX}://bm/{u}")
        stale_bm_keys.append(f"{DOC_ID_PREFIX}://rl/{u}")

    now_iso = (now or datetime.now()).isoformat(timespec="seconds")

    if not dry_run:
        summary["history_indexed"] = upsert_history(col, hist_to_write)
        for h in hist_to_write:
            _upsert_hash(state_conn, "rag_safari_history_state",
                         "history_item_id", h.history_item_id,
                         _history_hash(h), now_iso)
        summary["history_deleted"] = delete_history(col, stale_hist_ids)
        for hid in stale_hist_ids:
            _delete_hash(state_conn, "rag_safari_history_state",
                         "history_item_id", hid)

        summary["bookmarks_indexed"] = upsert_bookmarks(col, bm_to_write)
        for b in bm_to_write:
            _upsert_hash(state_conn, "rag_safari_bookmark_state",
                         "bookmark_uuid", b.uuid,
                         _bookmark_hash(b), now_iso)
        summary["bookmarks_deleted"] = delete_bookmarks(col, stale_bm_keys)
        for u in stale_bm_uuids:
            _delete_hash(state_conn, "rag_safari_bookmark_state",
                         "bookmark_uuid", u)
        state_conn.commit()
    else:
        summary["history_indexed"] = len(hist_to_write)
        summary["history_deleted"] = len(stale_hist_ids)
        summary["bookmarks_indexed"] = len(bm_to_write)
        summary["bookmarks_deleted"] = len(stale_bm_keys) // 2

    state_conn.close()
    summary["duration_s"] = round(time.perf_counter() - t0, 2)
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--since", type=str, default=None,
                    help="ISO-8601 hard floor for history visits")
    ap.add_argument("--retention-days", type=int, default=None,
                    help=f"Soft floor for history in days (default: {DEFAULT_HISTORY_RETENTION_DAYS})")
    ap.add_argument("--max-urls", type=int, default=DEFAULT_MAX_URLS,
                    help="Cap on history URLs per-run (default: 5000)")
    ap.add_argument("--skip-bookmarks", action="store_true")
    ap.add_argument("--history-db", type=str, default=None)
    ap.add_argument("--bookmarks-plist", type=str, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    summary = run(
        reset=bool(args.reset),
        dry_run=bool(args.dry_run),
        since_iso=args.since,
        retention_days=args.retention_days,
        max_urls=args.max_urls,
        skip_bookmarks=bool(args.skip_bookmarks),
        history_db=Path(args.history_db) if args.history_db else None,
        bookmarks_plist=Path(args.bookmarks_plist) if args.bookmarks_plist else None,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if "error" in summary:
        print(f"[error] {summary['error']}", file=sys.stderr)
        sys.exit(1)
    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}History: {summary['history_fetched']} URLs "
        f"({summary['history_indexed']} indexed · {summary['history_unchanged']} unchanged · "
        f"{summary['history_deleted']} deleted) · "
        f"Bookmarks: {summary['bookmarks_fetched']} "
        f"({summary['bookmarks_indexed']} indexed) · "
        f"RL: {summary['reading_list_fetched']} · "
        f"{summary['duration_s']}s"
    )


if __name__ == "__main__":
    main()
