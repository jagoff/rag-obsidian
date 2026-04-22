"""Apple Reminders ingester — Phase 1.d of the cross-source corpus.

User decision §10.1 (2026-04-20): Reminders is local-only (EventKit via
AppleScript, same path as the morning brief's `_fetch_reminders_due`).
Ingesting the full catalogue (both pending and completed) gives
`retrieve()` access to the historical record of every loop the user has
tracked — useful for "qué había agendado para la semana pasada" or
"cuándo terminé X".

Ingest strategy:
  - Reader: AppleScript against Reminders.app (same trust boundary /
    Automation permission as the morning brief). Pulls id, list,
    completed flag, due/completion/creation/modification dates, name,
    and body (notes) in a pipe-separated format.
  - Chunk per reminder (<800 chars, atomic). Same pattern as Calendar
    events — no splitting, no parent window.
  - Retention: None (§10.2). Reminders are small in volume and the
    user prunes them inside Reminders.app directly; the ingester
    mirrors the live state (delete-from-index when the id disappears).
  - Source weight: 0.90 (§10.3). Recency halflife 90d (§10.2 + §1288).
  - Scope: every list the user has, no opt-out (§10.5 "indexá todo").
    The `cross-source.yaml` escape hatch is documented but not wired
    yet — consistent with the Gmail/WA ingesters.

Incremental sync:
  - Per-reminder `content_hash` in `rag_reminders_state(id, hash,
    last_seen_ts, updated_at)`. On each run: fetch all reminders,
    compute hash of (name, body, due, completed, list). Upsert rows
    whose hash changed or are new; delete rows whose id is no longer
    present in the live catalogue.
  - No cursor / modification-date polling — Reminders.app's
    `modification date` is unreliable via AppleScript (changes on
    mere reads on some macOS versions). Content-hash diffing is
    simpler and bulletproof.

Invoked via `rag index --source reminders [--reset] [--dry-run]
[--include-completed/--only-pending]`. `--reset` wipes cursors and
forces a full re-scan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


# ── Config ─────────────────────────────────────────────────────────────────

CHUNK_MAX_CHARS = 800
DOC_ID_PREFIX = "reminders"

# AppleScript record separator — ASCII Unit Separator (0x1F). Avoids
# collisions with content; reminders almost never include it.
_AS_FS = "\u241F"  # visible form in logs: "␟" (symbol for US), emitted
                    # as the literal 0x241F codepoint from AppleScript.
# We actually emit the raw Unit Separator (chr(31)) in AppleScript because
# `character id 31` is well-supported there, and match against chr(31) in
# Python. The 0x241F variable is documentation-only.
_FIELD_SEP = chr(31)


# ── AppleScript reader ─────────────────────────────────────────────────────

# Pulls every reminder (pending + completed) with a stable field layout.
# Body is stripped of field-sep + newlines to keep parsing unambiguous.
# Dates emitted via `as string` — locale-formatted, parsed by
# `_parse_applescript_date()` downstream.
#
# Layout per line (chr(31)-separated, terminated by linefeed):
#   id ␟ list ␟ completed ␟ due ␟ completion ␟ created ␟ modified ␟ priority ␟ flagged ␟ name ␟ body
_REMINDERS_ALL_SCRIPT = '''
set _fs to character id 31
set _out to ""
tell application "Reminders"
  repeat with _list in lists
    try
      set _list_name to name of _list
      repeat with _r in (reminders of _list)
        try
          set _rid to id of _r
          set _name to name of _r
          set _completed to completed of _r
          set _due to ""
          try
            if (due date of _r) is not missing value then
              set _due to (due date of _r) as string
            end if
          end try
          set _comp to ""
          try
            if (completion date of _r) is not missing value then
              set _comp to (completion date of _r) as string
            end if
          end try
          set _created to ""
          try
            if (creation date of _r) is not missing value then
              set _created to (creation date of _r) as string
            end if
          end try
          set _modified to ""
          try
            if (modification date of _r) is not missing value then
              set _modified to (modification date of _r) as string
            end if
          end try
          set _prio to 0
          try
            set _prio to priority of _r
          end try
          set _flag to false
          try
            set _flag to flagged of _r
          end try
          set _body to ""
          try
            set _body to (body of _r) as string
          end try
          if (count of _body) > 1800 then
            set _body to text 1 thru 1800 of _body
          end if
          set _prev_tids to AppleScript's text item delimiters
          set AppleScript's text item delimiters to {return, linefeed, character id 9, _fs}
          set _bparts to text items of _body
          set AppleScript's text item delimiters to " "
          set _body to _bparts as text
          set AppleScript's text item delimiters to _prev_tids
          set _out to _out & _rid & _fs & _list_name & _fs & (_completed as string) & _fs & _due & _fs & _comp & _fs & _created & _fs & _modified & _fs & (_prio as string) & _fs & (_flag as string) & _fs & _name & _fs & _body & linefeed
        end try
      end repeat
    end try
  end repeat
end tell
return _out
'''


# ── Data types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Reminder:
    id: str
    list_name: str
    name: str
    body: str
    completed: bool
    flagged: bool
    priority: int
    due_ts: float           # 0.0 if no due date
    completion_ts: float    # 0.0 if not completed
    created_ts: float       # 0.0 if AppleScript didn't expose it
    modified_ts: float

    @property
    def anchor_ts(self) -> float:
        """Best available timestamp for the chunk's `created_ts` metadata
        (used by recency decay). Preference order:
          1. creation date (the "when was this loop born")
          2. due date (semantic anchor for timed reminders)
          3. modification date (fallback)
          4. completion date (for old completed items)
        """
        for v in (self.created_ts, self.due_ts, self.modified_ts, self.completion_ts):
            if v:
                return v
        return 0.0


# ── Parsing ────────────────────────────────────────────────────────────────

def _as_bool(s: str) -> bool:
    return s.strip().lower() in ("true", "yes", "1")


def _as_dt(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    dt = rag._parse_applescript_date(s)
    if dt is None:
        return 0.0
    try:
        return dt.timestamp()
    except (ValueError, OSError):
        return 0.0


def _parse_reminder_line(line: str) -> Reminder | None:
    """Parse one chr(31)-separated line emitted by `_REMINDERS_ALL_SCRIPT`.
    Returns None for malformed lines (too few fields, missing id/name)."""
    if not line:
        return None
    parts = line.split(_FIELD_SEP)
    if len(parts) < 11:
        return None
    (rid, list_name, completed, due, comp, created, modified,
     priority, flagged, name, body) = parts[:11]
    rid = rid.strip()
    name = name.strip()
    if not rid or not name:
        return None
    try:
        prio = int(priority.strip() or "0")
    except ValueError:
        prio = 0
    return Reminder(
        id=rid,
        list_name=list_name.strip(),
        name=name,
        body=body.strip(),
        completed=_as_bool(completed),
        flagged=_as_bool(flagged),
        priority=prio,
        due_ts=_as_dt(due),
        completion_ts=_as_dt(comp),
        created_ts=_as_dt(created),
        modified_ts=_as_dt(modified),
    )


def _default_fetch(timeout: float = 180.0) -> list[Reminder]:
    """Default fetcher — real AppleScript. Silently returns [] when Apple
    integrations are disabled or the script fails.

    Timeout default 180s: el AppleScript itera todas las reminders de todas
    las listas (36 en el host de test = 98s medidos el 2026-04-21). Setting
    muy bajo causaba un `0 fetched` silencioso en hosts con muchas listas.
    Subsecuente runs incrementales son rapidísimos (solo nuevas/cambiadas)
    pero el primer full-scan puede tardar minutos.
    """
    if not rag._apple_enabled():
        return []
    out = rag._osascript(_REMINDERS_ALL_SCRIPT, timeout=timeout)
    if not out:
        return []
    items: list[Reminder] = []
    for line in out.splitlines():
        r = _parse_reminder_line(line)
        if r is not None:
            items.append(r)
    return items


# ── State ──────────────────────────────────────────────────────────────────

_STATE_TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_reminders_state ("
    " reminder_id TEXT PRIMARY KEY,"
    " content_hash TEXT NOT NULL,"
    " last_seen_ts TEXT NOT NULL,"
    " updated_at TEXT NOT NULL"
    ")"
)


def _ensure_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(_STATE_TABLE_DDL)


def _load_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    cur = conn.execute("SELECT reminder_id, content_hash FROM rag_reminders_state")
    return {row[0]: row[1] for row in cur.fetchall()}


def _upsert_hash(
    conn: sqlite3.Connection, rid: str, h: str, now_iso: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO rag_reminders_state "
        "(reminder_id, content_hash, last_seen_ts, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (rid, h, now_iso, now_iso),
    )


def _delete_hash(conn: sqlite3.Connection, rid: str) -> None:
    conn.execute("DELETE FROM rag_reminders_state WHERE reminder_id = ?", (rid,))


def _reset_state(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM rag_reminders_state")


def _content_hash(r: Reminder) -> str:
    """Stable hash over the fields that would change `display_text` or
    metadata the scorer cares about. Excludes `modified_ts` on purpose —
    Reminders.app sometimes bumps modification date on idle reads."""
    payload = "\n".join([
        r.name, r.body, r.list_name,
        f"{r.completed}", f"{r.flagged}", f"{r.priority}",
        f"{r.due_ts:.0f}", f"{r.completion_ts:.0f}", f"{r.created_ts:.0f}",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── Chunker ────────────────────────────────────────────────────────────────

def _format_reminder_body(r: Reminder) -> str:
    """Human-readable transcript used as display_text."""
    status = "✓ completada" if r.completed else "⧗ pendiente"
    parts = [f"Tarea: {r.name}", f"Estado: {status}"]
    if r.list_name:
        parts.append(f"Lista: {r.list_name}")
    if r.due_ts:
        due_dt = datetime.fromtimestamp(r.due_ts)
        parts.append(f"Vence: {due_dt.strftime('%Y-%m-%d %H:%M')}")
    if r.completed and r.completion_ts:
        comp_dt = datetime.fromtimestamp(r.completion_ts)
        parts.append(f"Cerrada: {comp_dt.strftime('%Y-%m-%d %H:%M')}")
    if r.created_ts:
        cr_dt = datetime.fromtimestamp(r.created_ts)
        parts.append(f"Creada: {cr_dt.strftime('%Y-%m-%d')}")
    if r.priority:
        parts.append(f"Prioridad: {r.priority}")
    if r.flagged:
        parts.append("⚑ Destacada")
    if r.body:
        parts.append("---")
        parts.append(r.body)
    body = "\n".join(parts)
    if len(body) > CHUNK_MAX_CHARS:
        body = body[:CHUNK_MAX_CHARS].rstrip()
    return body


def _embed_prefix(r: Reminder, body: str) -> str:
    when = ""
    if r.due_ts:
        when = datetime.fromtimestamp(r.due_ts).strftime("%Y-%m-%d")
    status = "done" if r.completed else "open"
    return (
        f"[source=reminders | list={r.list_name} | status={status} | "
        f"title={r.name}{' | ' + when if when else ''}] {body}"
    )


def _reminder_doc_id(r: Reminder) -> str:
    return f"{DOC_ID_PREFIX}://{r.id}::0"


def _reminder_file_key(r: Reminder) -> str:
    return f"{DOC_ID_PREFIX}://{r.id}"


# ── Writer ─────────────────────────────────────────────────────────────────

def upsert_reminders(col, reminders: list[Reminder]) -> int:
    if not reminders:
        return 0
    bodies = [_format_reminder_body(r) for r in reminders]
    embed_texts = [_embed_prefix(r, b) for r, b in zip(reminders, bodies)]
    embeddings = rag.embed(embed_texts)

    for r in reminders:
        key = _reminder_file_key(r)
        try:
            existing = col.get(where={"file": key}, include=[])
            if existing.get("ids"):
                col.delete(ids=existing["ids"])
        except Exception:
            pass

    ids = [_reminder_doc_id(r) for r in reminders]
    metas: list[dict] = []
    for r, body in zip(reminders, bodies):
        metas.append({
            "file": _reminder_file_key(r),
            "note": f"Rem: {r.name}",
            "folder": "",
            "tags": "",
            "hash": _content_hash(r),
            "outlinks": "",
            "source": "reminders",
            "created_ts": r.anchor_ts,
            "reminder_id": r.id,
            "list_name": r.list_name,
            "title": r.name,
            "completed": int(r.completed),
            "flagged": int(r.flagged),
            "priority": r.priority,
            "due_ts": r.due_ts,
            "completion_ts": r.completion_ts,
            "creation_ts": r.created_ts,
            "modification_ts": r.modified_ts,
            "parent": body,
        })
    col.add(ids=ids, embeddings=embeddings, documents=bodies, metadatas=metas)
    # Entity extraction — people / projects mentioned in reminder bodies.
    # Gated by `_entity_extraction_enabled()` + silent-fail if gliner absent.
    rag._extract_and_index_entities_for_chunks(bodies, ids, metas, "reminders")
    return len(reminders)


def delete_reminders(col, reminder_ids: list[str]) -> int:
    if not reminder_ids:
        return 0
    n = 0
    for rid in reminder_ids:
        key = f"{DOC_ID_PREFIX}://{rid}"
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
    include_completed: bool = True,
    vault_col=None,
    fetch_fn: Callable[[], list[Reminder]] | None = None,
    now: datetime | None = None,
) -> dict:
    """Ingest Reminders.app reminders. Returns a summary dict.

    Auth:
      - Default: `_default_fetch()` runs the AppleScript.
      - Tests inject `fetch_fn=mock` returning a fixed `[Reminder]`.

    Scope:
      - `include_completed=True` (default): index both pending and done.
      - `include_completed=False`: skip completed (roughly matches what
        `rag agenda` shows today).
    """
    t0 = time.perf_counter()
    summary: dict = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "reset": bool(reset),
        "dry_run": bool(dry_run),
        "include_completed": bool(include_completed),
        "reminders_fetched": 0,
        "reminders_indexed": 0,
        "reminders_unchanged": 0,
        "reminders_deleted": 0,
        "duration_s": 0.0,
    }

    fetch = fetch_fn if fetch_fn is not None else _default_fetch
    reminders = fetch() or []
    summary["reminders_fetched"] = len(reminders)
    if not include_completed:
        reminders = [r for r in reminders if not r.completed]

    col = vault_col if vault_col is not None else rag.get_db()

    # rag.DB_PATH already points at `.../ragvec/` — do NOT append an extra
    # `ragvec/` segment (that produced `.../ragvec/ragvec/ragvec.db` and an
    # "unable to open database file" on a clean host).
    state_conn = sqlite3.connect(str(rag.DB_PATH / "ragvec.db"))
    _ensure_state_table(state_conn)
    if reset:
        _reset_state(state_conn)
    state_conn.commit()

    prior_hashes = _load_hashes(state_conn)
    live_ids = {r.id for r in reminders}

    # Split into upsert (new or changed) vs unchanged.
    to_write: list[Reminder] = []
    for r in reminders:
        h = _content_hash(r)
        if prior_hashes.get(r.id) != h:
            to_write.append(r)
        else:
            summary["reminders_unchanged"] += 1

    # Stale ids = in prior state but not in live fetch.
    stale_ids = [rid for rid in prior_hashes if rid not in live_ids]

    now_iso = (now or datetime.now()).isoformat(timespec="seconds")

    if not dry_run:
        summary["reminders_indexed"] = upsert_reminders(col, to_write)
        for r in to_write:
            _upsert_hash(state_conn, r.id, _content_hash(r), now_iso)
        summary["reminders_deleted"] = delete_reminders(col, stale_ids)
        for rid in stale_ids:
            _delete_hash(state_conn, rid)
        state_conn.commit()
    else:
        summary["reminders_indexed"] = len(to_write)
        summary["reminders_deleted"] = len(stale_ids)

    state_conn.close()
    summary["duration_s"] = round(time.perf_counter() - t0, 2)
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-pending", action="store_true",
                     help="Skip completed reminders (default: index both)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    summary = run(
        reset=bool(args.reset),
        dry_run=bool(args.dry_run),
        include_completed=not args.only_pending,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}{summary['reminders_fetched']} reminders · "
        f"{summary['reminders_indexed']} indexados · "
        f"{summary['reminders_unchanged']} sin cambios · "
        f"{summary['reminders_deleted']} borrados · "
        f"{summary['duration_s']}s"
    )


if __name__ == "__main__":
    main()
