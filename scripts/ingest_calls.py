"""Call History ingester — Phase 1.f of the cross-source corpus.

Reads `~/Library/Application Support/CallHistoryDB/CallHistory.storedata`
(Core Data SQLite) and indexes every call — phone, FaceTime audio/video,
incoming / outgoing / missed — as an atomic chunk with `source=calls`.

Numbers are resolved to human names via
`scripts.ingest_contacts.resolve_phone()`, so missed-call queries like
"¿me llamó Juli?" return `Llamada perdida de Juli — 2026-04-22 14:30`
instead of `Llamada perdida de +5491112345678`. If the number is
unknown (not in Contacts) the raw number + the value of `ZNAME` (if
Apple cached a name) are used as fallbacks.

Ingest strategy:
  - Reader: direct SQLite against CallHistory.storedata (read-only).
    Zero pyobjc / AppleScript.
  - Chunk per call (<800 chars, atomic). One row = one chunk.
  - Retention: 180 days (same bucket as WhatsApp/messages — the
    conversational-volume tier).
  - Source weight: 0.80 (between gmail 0.85 and whatsapp 0.75 — log
    entries carry less semantic content than either).
  - Recency halflife: 30 days (who-called-yesterday is critical, who-
    called-six-months-ago is trivia).

Incremental sync:
  - State: `rag_calls_state(call_uid, content_hash, last_seen_ts,
    updated_at)`. Each call is keyed by `ZUNIQUE_ID` (stable GUID
    Apple assigns at call time).
  - On each run: fetch calls within retention window, compute hash
    of (date, duration, direction, answered, address, service).
    Upsert changed/new, delete rows that dropped out of retention.
  - Calls are effectively immutable after they're logged (Apple
    doesn't retroactively mutate ZCALLRECORD rows), so `hash`
    mostly acts as a paranoid delete-and-replace guard.

Enrichment:
  - `resolve_phone(ZADDRESS)` → Contact → `display_name`.
  - Unknown numbers show `ZNAME` if present (Apple sometimes caches
    the name it resolved at call time), else the raw address.

Direction + state matrix (combining ZORIGINATED + ZANSWERED):
  - outgoing answered    → "saliente · atendida"
  - outgoing unanswered  → "saliente · sin respuesta"
  - incoming answered    → "entrante · atendida"
  - incoming unanswered  → "perdida"  ⚠️ the interesting one

Invoked via `rag index --source calls [--reset] [--dry-run]
[--since ISO]`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402
from scripts import ingest_contacts as _contacts  # noqa: E402


# ── Config ─────────────────────────────────────────────────────────────────

DEFAULT_CALLDB = (
    Path.home() / "Library/Application Support/CallHistoryDB/CallHistory.storedata"
)
CHUNK_MAX_CHARS = 800
DOC_ID_PREFIX = "calls"
DEFAULT_RETENTION_DAYS = 180

COCOA_EPOCH_OFFSET = 978307200.0  # 2001-01-01 UTC → Unix

# Service provider → human label.
_SERVICE_LABELS = {
    "com.apple.Telephony": "Teléfono",
    "com.apple.Telephony.GCS": "Teléfono (WiFi/GSM)",
    "com.apple.FaceTime": "FaceTime",
    "com.apple.FaceTime-Audio": "FaceTime Audio",
    "com.apple.FaceTime-Video": "FaceTime Video",
}


# ── Data types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Call:
    uid: str                # ZUNIQUE_ID — stable across phone restarts
    date_ts: float          # Unix epoch (converted from Cocoa ZDATE)
    duration_s: float       # 0.0 for calls that didn't connect
    address: str            # ZADDRESS — raw phone number or email (FaceTime)
    cached_name: str        # ZNAME — Apple's contacts lookup at call time
    originated: bool        # True = outgoing, False = incoming
    answered: bool          # True = call completed, False = not picked up
    service: str            # ZSERVICE_PROVIDER raw string
    call_type: int          # ZCALLTYPE int (1=phone, 8=FaceTime, etc.)
    handle_type: int        # ZHANDLE_TYPE (0=phone, 1=email)
    location: str           # ZLOCATION — sometimes "City, Country"
    read: bool              # ZREAD — user has acknowledged missed call

    @property
    def direction(self) -> str:
        return "saliente" if self.originated else "entrante"

    @property
    def state(self) -> str:
        if self.originated:
            return "atendida" if self.answered else "sin respuesta"
        return "atendida" if self.answered else "perdida"

    @property
    def is_missed(self) -> bool:
        return not self.originated and not self.answered

    @property
    def service_label(self) -> str:
        return _SERVICE_LABELS.get(self.service, self.service or "Llamada")


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


def _format_duration(seconds: float) -> str:
    """0s / 42s / 3m 42s / 1h 2m. Compact and human."""
    if seconds <= 0:
        return "0s"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


# ── Reader ─────────────────────────────────────────────────────────────────

_CALLS_SQL = (
    "SELECT ZUNIQUE_ID, ZDATE, COALESCE(ZDURATION, 0), COALESCE(ZADDRESS, ''), "
    "COALESCE(ZNAME, ''), COALESCE(ZORIGINATED, 0), COALESCE(ZANSWERED, 0), "
    "COALESCE(ZSERVICE_PROVIDER, ''), COALESCE(ZCALLTYPE, 0), "
    "COALESCE(ZHANDLE_TYPE, 0), COALESCE(ZLOCATION, ''), COALESCE(ZREAD, 0) "
    "FROM ZCALLRECORD WHERE ZDATE IS NOT NULL AND ZDATE >= ? "
    "ORDER BY ZDATE ASC"
)


def read_calls(
    db_path: Path = DEFAULT_CALLDB,
    *,
    since_unix_ts: float = 0.0,
) -> list[Call]:
    """Load calls from the Core Data SQLite. Filters by `since_unix_ts`
    (Unix epoch seconds); pass 0.0 to fetch everything currently stored.
    macOS/iOS rolls the table — don't expect history older than a few
    months. Returns [] if the DB file doesn't exist (host without call
    forwarding from an iPhone)."""
    if not db_path.exists():
        return []
    # Caller passes a Unix epoch; SQL uses Cocoa — convert the cutoff
    # rather than every row.
    cocoa_cutoff = max(0.0, since_unix_ts - COCOA_EPOCH_OFFSET)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out: list[Call] = []
        for row in conn.execute(_CALLS_SQL, (cocoa_cutoff,)):
            (uid, zdate, dur, addr, name, orig, ans, svc,
             ctype, htype, loc, read) = row
            if not uid:
                continue
            out.append(Call(
                uid=str(uid),
                date_ts=_cocoa_to_unix(zdate),
                duration_s=float(dur or 0),
                address=str(addr).strip(),
                cached_name=str(name).strip(),
                originated=bool(orig),
                answered=bool(ans),
                service=str(svc).strip(),
                call_type=int(ctype or 0),
                handle_type=int(htype or 0),
                location=str(loc).strip(),
                read=bool(read),
            ))
        return out
    finally:
        conn.close()


# ── State ──────────────────────────────────────────────────────────────────

_STATE_TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_calls_state ("
    " call_uid TEXT PRIMARY KEY,"
    " content_hash TEXT NOT NULL,"
    " last_seen_ts TEXT NOT NULL,"
    " updated_at TEXT NOT NULL"
    ")"
)


def _ensure_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(_STATE_TABLE_DDL)


def _load_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    cur = conn.execute("SELECT call_uid, content_hash FROM rag_calls_state")
    return {row[0]: row[1] for row in cur.fetchall()}


def _upsert_hash(conn: sqlite3.Connection, uid: str, h: str, now_iso: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO rag_calls_state "
        "(call_uid, content_hash, last_seen_ts, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (uid, h, now_iso, now_iso),
    )


def _delete_hash(conn: sqlite3.Connection, uid: str) -> None:
    conn.execute("DELETE FROM rag_calls_state WHERE call_uid = ?", (uid,))


def _reset_state(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM rag_calls_state")


def _content_hash(c: Call) -> str:
    payload = "\n".join([
        c.uid,
        f"{c.date_ts:.0f}",
        f"{c.duration_s:.2f}",
        c.address,
        c.cached_name,
        f"{c.originated}",
        f"{c.answered}",
        c.service,
        f"{c.call_type}",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── Enrichment ─────────────────────────────────────────────────────────────

def _resolve_contact_name(c: Call) -> tuple[str, str | None]:
    """Return `(display_name, contact_uid)` for a call. Uses
    `ingest_contacts.resolve_phone` first; falls back to cached ZNAME,
    then to the raw address. `contact_uid` is None when we couldn't
    resolve."""
    if c.address:
        match = _contacts.resolve_phone(c.address)
        if match is not None:
            return match.display_name, match.uid
    if c.cached_name:
        return c.cached_name, None
    return c.address or "(desconocido)", None


# ── Chunker ────────────────────────────────────────────────────────────────

def _format_call_body(c: Call) -> tuple[str, str | None]:
    """Return `(body, contact_uid)` — body is the human display_text,
    contact_uid is stored in metadata for lookups."""
    name, uid = _resolve_contact_name(c)
    # Direction-flavoured headline so "llamada perdida" matches BM25 too.
    if c.is_missed:
        headline = f"Llamada perdida de {name}"
    elif c.originated and c.answered:
        headline = f"Llamada saliente a {name}"
    elif c.originated and not c.answered:
        headline = f"Llamada saliente a {name} (sin respuesta)"
    elif c.answered:
        headline = f"Llamada entrante de {name}"
    else:  # unreachable; kept for type completeness
        headline = f"Llamada con {name}"
    dt = datetime.fromtimestamp(c.date_ts, tz=timezone.utc).astimezone()
    when = dt.strftime("%Y-%m-%d %H:%M")
    parts: list[str] = [f"{headline} — {c.service_label} — {when}"]
    if c.address:
        parts.append(f"Número: {c.address}")
    parts.append(f"Duración: {_format_duration(c.duration_s)}")
    parts.append(f"Dirección: {c.direction} · {c.state}")
    if c.location:
        parts.append(f"Ubicación: {c.location}")
    if not c.read and c.is_missed:
        parts.append("No leída")
    body = "\n".join(parts)
    if len(body) > CHUNK_MAX_CHARS:
        body = body[:CHUNK_MAX_CHARS].rstrip()
    return body, uid


def _embed_prefix(c: Call, body: str, contact_name: str) -> str:
    when = datetime.fromtimestamp(c.date_ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d")
    return (
        f"[source=calls | direction={c.direction} | state={c.state} | "
        f"who={contact_name} | date={when}] {body}"
    )


def _call_doc_id(c: Call) -> str:
    return f"{DOC_ID_PREFIX}://{c.uid}::0"


def _call_file_key(c: Call) -> str:
    return f"{DOC_ID_PREFIX}://{c.uid}"


# ── Writer ─────────────────────────────────────────────────────────────────

def upsert_calls(col, calls: list[Call]) -> int:
    if not calls:
        return 0
    rendered: list[tuple[str, str | None, str]] = []  # (body, uid, name)
    for c in calls:
        body, uid = _format_call_body(c)
        # Extract the resolved name from body headline (already computed
        # inside _format_call_body), but we also need it for the embed
        # prefix — cheaper to re-call resolve than parse headline.
        name, _ = _resolve_contact_name(c)
        rendered.append((body, uid, name))

    bodies = [r[0] for r in rendered]
    embed_texts = [_embed_prefix(c, b, n) for c, (b, _, n) in zip(calls, rendered)]
    embeddings = rag.embed(embed_texts)

    for c in calls:
        key = _call_file_key(c)
        try:
            existing = col.get(where={"file": key}, include=[])
            if existing.get("ids"):
                col.delete(ids=existing["ids"])
        except Exception:
            pass

    ids = [_call_doc_id(c) for c in calls]
    metas: list[dict] = []
    for c, (body, contact_uid, name) in zip(calls, rendered):
        metas.append({
            "file": _call_file_key(c),
            "note": f"Call: {name} ({c.direction} · {c.state})",
            "folder": "",
            "tags": "",
            "hash": _content_hash(c),
            "outlinks": "",
            "source": "calls",
            "created_ts": c.date_ts,
            "call_uid": c.uid,
            "contact_uid": contact_uid or "",
            "contact_name": name,
            "address": c.address,
            "direction": c.direction,
            "state": c.state,
            "missed": int(c.is_missed),
            "answered": int(c.answered),
            "originated": int(c.originated),
            "duration_s": c.duration_s,
            "service": c.service,
            "service_label": c.service_label,
            "call_type": c.call_type,
            "handle_type": c.handle_type,
            "location": c.location,
            "title": f"Llamada {c.state} — {name}",
            "parent": body,
        })
    col.add(ids=ids, embeddings=embeddings, documents=bodies, metadatas=metas)
    rag._extract_and_index_entities_for_chunks(bodies, ids, metas, "calls")
    return len(calls)


def delete_calls(col, uids: list[str]) -> int:
    if not uids:
        return 0
    n = 0
    for uid in uids:
        key = f"{DOC_ID_PREFIX}://{uid}"
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
    db_path: Path | None = None,
    vault_col=None,
    fetch_fn: Callable[[], list[Call]] | None = None,
    now: datetime | None = None,
) -> dict:
    """Ingest calls from CallHistory.storedata. Returns summary dict.

    - `since_iso`: hard floor (ISO-8601). Combined with `retention_days`
      — whichever is more recent wins.
    - `retention_days`: soft floor (default 180). Set to 0 / None to
      disable. Calls older than `now - retention` are neither fetched
      nor indexed, and rows in state DB past retention are deleted.
    """
    t0 = time.perf_counter()
    summary: dict = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "reset": bool(reset),
        "dry_run": bool(dry_run),
        "calls_fetched": 0,
        "calls_indexed": 0,
        "calls_unchanged": 0,
        "calls_deleted": 0,
        "missed_calls": 0,
        "duration_s": 0.0,
    }

    retention_days = (
        DEFAULT_RETENTION_DAYS if retention_days is None else retention_days
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

    db = Path(db_path) if db_path is not None else DEFAULT_CALLDB
    fetch = fetch_fn if fetch_fn is not None else (lambda: read_calls(db, since_unix_ts=floor_ts))
    calls = fetch() or []
    summary["calls_fetched"] = len(calls)
    summary["missed_calls"] = sum(1 for c in calls if c.is_missed)

    if not calls and not db.exists():
        summary["error"] = f"CallHistory DB no encontrada en {db}"
        summary["duration_s"] = round(time.perf_counter() - t0, 2)
        return summary

    col = vault_col if vault_col is not None else rag.get_db()
    state_conn = sqlite3.connect(str(rag.DB_PATH / "ragvec.db"))
    _ensure_state_table(state_conn)
    if reset:
        _reset_state(state_conn)
    state_conn.commit()

    prior_hashes = _load_hashes(state_conn)
    live_uids = {c.uid for c in calls}

    to_write: list[Call] = []
    for c in calls:
        h = _content_hash(c)
        if prior_hashes.get(c.uid) != h:
            to_write.append(c)
        else:
            summary["calls_unchanged"] += 1

    # Stale = in state DB but absent from live fetch. Happens either
    # because the call rolled past retention OR Apple pruned it.
    stale_uids = [uid for uid in prior_hashes if uid not in live_uids]
    now_iso = (now or datetime.now()).isoformat(timespec="seconds")

    if not dry_run:
        summary["calls_indexed"] = upsert_calls(col, to_write)
        for c in to_write:
            _upsert_hash(state_conn, c.uid, _content_hash(c), now_iso)
        summary["calls_deleted"] = delete_calls(col, stale_uids)
        for uid in stale_uids:
            _delete_hash(state_conn, uid)
        state_conn.commit()
    else:
        summary["calls_indexed"] = len(to_write)
        summary["calls_deleted"] = len(stale_uids)

    state_conn.close()
    summary["duration_s"] = round(time.perf_counter() - t0, 2)
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reset", action="store_true",
                    help="Wipe state DB + re-index every call within retention")
    ap.add_argument("--dry-run", action="store_true",
                    help="Count fetched/changed but skip writes")
    ap.add_argument("--since", type=str, default=None,
                    help="ISO-8601 hard floor (overrides retention if more recent)")
    ap.add_argument("--retention-days", type=int, default=None,
                    help=f"Soft floor in days (default: {DEFAULT_RETENTION_DAYS}). 0 to disable.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--db", type=str, default=None,
                    help="Override CallHistory.storedata path")
    args = ap.parse_args()

    summary = run(
        reset=bool(args.reset),
        dry_run=bool(args.dry_run),
        since_iso=args.since,
        retention_days=args.retention_days,
        db_path=Path(args.db) if args.db else None,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if "error" in summary:
        print(f"[error] {summary['error']}", file=sys.stderr)
        sys.exit(1)
    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}{summary['calls_fetched']} llamadas ("
        f"{summary['missed_calls']} perdidas) · "
        f"{summary['calls_indexed']} indexadas · "
        f"{summary['calls_unchanged']} sin cambios · "
        f"{summary['calls_deleted']} borradas · "
        f"{summary['duration_s']}s"
    )


if __name__ == "__main__":
    main()
