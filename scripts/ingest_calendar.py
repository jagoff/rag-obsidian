"""Google Calendar ingester — Phase 1.b of the cross-source corpus.

User decision §10.6 (2026-04-20): OAuth Google Calendar via the Claude
harness MCP pattern (same cred shape as the Gmail + Google Drive
ingesters already in rag.py). This is cloud-path — explicitly documented
as an exception to the "fully local" invariant in CLAUDE.md.

Setup (one-time, user):
  1. mkdir -p ~/.calendar-mcp
  2. Copy OAuth client JSON to ~/.calendar-mcp/gcp-oauth.keys.json
     (same format as ~/.gmail-mcp/gcp-oauth.keys.json)
  3. Run the OAuth flow once to produce ~/.calendar-mcp/credentials.json
     (or symlink from another MCP's tokens.json if same client_id).
  4. `rag index --source calendar --reset` to bootstrap.

Ingest strategy:
  - Reader: `events.list()` per configured calendar with
    singleEvents=True, timeMin=last 2y, orderBy=startTime. Paginates
    until pageToken is None or a hard cap of 5000 events.
  - Chunk per event (<800 chars). No splits — Calendar events are
    short + atomic. `parent` = event body (title + attendees +
    location + description) itself since there's no surrounding
    context that makes sense (unlike WA's ±10 messages).
  - Recurrence: singleEvents=True expands RRULE into individual
    instances within the window. Keeps indexing uniform; no special
    handling of master vs exception needed.
  - Retention: None (§10.2) — events don't age, all kept.
  - Source weight: 0.95 (§10.3).
  - No recency decay (§10.3 — events anchored in time, decay would
    de-rank legitimate future plans).

Incremental sync:
  - Cursor per calendar_id in rag_calendar_state(cal_id, sync_token,
    last_updated, updated_at). Uses Google Calendar's sync_token protocol
    (incremental since last fetch; server returns only changed events).
  - If sync_token is missing or expired (410 Gone), fall back to a full
    window scan and bootstrap a new token.
  - Deleted events: server sets status="cancelled"; we remove those
    rows from the index.

Invoked via `rag index --source calendar [--reset] [--calendar-id ID]
[--dry-run]`. `--reset` wipes cursors for all calendars and forces a
full re-scan.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


# ── Config ─────────────────────────────────────────────────────────────────

CALENDAR_CREDS_DIR = Path.home() / ".calendar-mcp"
CALENDAR_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
]

# Window for initial (pre-sync-token) bootstrap. Events with start times
# within [now - INITIAL_WINDOW_PAST_DAYS, now + INITIAL_WINDOW_FUTURE_DAYS]
# get indexed on first run; after that the sync_token picks up deltas.
INITIAL_WINDOW_PAST_DAYS = 365 * 2   # 2 years of history
INITIAL_WINDOW_FUTURE_DAYS = 180     # 6 months of upcoming

# Hard cap on events per calendar per run — defense against runaway lists.
MAX_EVENTS_PER_CAL = 5000

# Embed prefix cap (same philosophy as vault).
CHUNK_MAX_CHARS = 800

DOC_ID_PREFIX = "calendar"

HARDCODED_EXCLUDE_CAL_IDS = frozenset({
    # System-generated calendars that rarely carry useful signal.
    "addressbook#contacts@group.v.calendar.google.com",
    "en.usa#holiday@group.v.calendar.google.com",
})


# ── Data types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CalEvent:
    id: str
    calendar_id: str
    calendar_name: str
    title: str
    description: str
    location: str
    start_ts: float               # epoch; 0.0 for all-day w/ unparseable date
    end_ts: float
    attendees: list[str]          # email addresses
    is_all_day: bool
    status: str                   # "confirmed" | "tentative" | "cancelled"
    html_link: str


# ── State ──────────────────────────────────────────────────────────────────

_STATE_TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_calendar_state ("
    " calendar_id TEXT PRIMARY KEY,"
    " sync_token TEXT,"
    " last_updated TEXT,"
    " updated_at TEXT NOT NULL"
    ")"
)


def _ensure_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(_STATE_TABLE_DDL)


def _load_sync_token(conn: sqlite3.Connection, cal_id: str) -> str | None:
    row = conn.execute(
        "SELECT sync_token FROM rag_calendar_state WHERE calendar_id = ?",
        (cal_id,),
    ).fetchone()
    if not row:
        return None
    return row[0] if row[0] else None


def _save_sync_token(
    conn: sqlite3.Connection, cal_id: str, token: str | None, last_updated: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO rag_calendar_state "
        "(calendar_id, sync_token, last_updated, updated_at) VALUES (?, ?, ?, ?)",
        (
            cal_id, token, last_updated,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )


def _reset_cursors(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM rag_calendar_state")


# ── Auth ───────────────────────────────────────────────────────────────────

def _calendar_service():
    """Return authed Google Calendar API client, or None. Same cred-refresh
    pattern as `_gmail_service()` / `_drive_service()`. Reads OAuth keys
    from `~/.calendar-mcp/` (parallel to the Gmail MCP).

    Separate creds dir lets the user configure the Calendar OAuth flow
    without touching Gmail's. If the user prefers a single auth for both,
    symlinking tokens.json achieves that externally.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        return None
    creds_path = CALENDAR_CREDS_DIR / "credentials.json"
    oauth_path = CALENDAR_CREDS_DIR / "gcp-oauth.keys.json"
    if not creds_path.is_file() or not oauth_path.is_file():
        return None
    try:
        stored = json.loads(creds_path.read_text(encoding="utf-8"))
        oauth = json.loads(oauth_path.read_text(encoding="utf-8"))
        installed = oauth.get("installed") or oauth.get("web") or {}
        creds = Credentials(
            token=stored.get("access_token") or stored.get("token"),
            refresh_token=stored.get("refresh_token"),
            token_uri=installed.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=installed.get("client_id"),
            client_secret=installed.get("client_secret"),
            scopes=CALENDAR_SCOPES,
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            stored["access_token"] = creds.token
            stored["token"] = creds.token
            # Atomic tmp + replace. Sin esto, un kill -9 (o reboot) durante
            # `write_text` deja `credentials.json` half-written y la próxima
            # invocación cae en el outer `except: return None` → el ingester
            # silenciosamente deja de sincronizar Calendar y el usuario tiene
            # que re-OAuthear manualmente. Mismo patrón que `_save_vaults_config`.
            tmp = creds_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(stored), encoding="utf-8")
            tmp.replace(creds_path)
        return build("calendar", "v3", credentials=creds, cache_discovery=False)
    except Exception:
        return None


# ── Reader ─────────────────────────────────────────────────────────────────

def _parse_cal_dt(raw: dict) -> tuple[float, bool]:
    """Parse a Google Calendar `start`/`end` dict. Returns (epoch, is_all_day).

    Shape:
      - Timed event: {"dateTime": "...T...Z"|"...+hh:mm", "timeZone": "..."}
      - All-day:     {"date": "YYYY-MM-DD"}
    """
    if not isinstance(raw, dict):
        return 0.0, False
    if "date" in raw:
        try:
            dt = datetime.strptime(raw["date"], "%Y-%m-%d")
            return dt.timestamp(), True
        except (ValueError, TypeError):
            return 0.0, True
    if "dateTime" in raw:
        s = str(raw["dateTime"])
        try:
            dt = datetime.fromisoformat(
                s.replace("Z", "+00:00") if s.endswith("Z") else s
            )
            if dt.tzinfo is not None:
                dt = dt.astimezone().replace(tzinfo=None)
            return dt.timestamp(), False
        except (ValueError, TypeError):
            return 0.0, False
    return 0.0, False


def _parse_event(raw: dict, calendar_id: str, calendar_name: str) -> CalEvent | None:
    """Map a Google Calendar events.list() item → CalEvent. Returns None on
    anything that's not indexable (missing id, deleted-without-id, etc.)."""
    eid = raw.get("id")
    if not eid:
        return None
    status = str(raw.get("status") or "confirmed")
    start_ts, is_all_day = _parse_cal_dt(raw.get("start") or {})
    end_ts, _ = _parse_cal_dt(raw.get("end") or {})
    attendees_raw = raw.get("attendees") or []
    attendees: list[str] = []
    for a in attendees_raw:
        email = (a or {}).get("email")
        if email:
            attendees.append(str(email))
    return CalEvent(
        id=str(eid),
        calendar_id=calendar_id,
        calendar_name=calendar_name,
        title=str(raw.get("summary") or "(sin título)"),
        description=str(raw.get("description") or "").strip(),
        location=str(raw.get("location") or "").strip(),
        start_ts=start_ts,
        end_ts=end_ts,
        attendees=attendees,
        is_all_day=is_all_day,
        status=status,
        html_link=str(raw.get("htmlLink") or ""),
    )


def list_calendars(svc) -> list[tuple[str, str]]:
    """Return [(calendar_id, display_name), ...] for all calendars the
    user can access, excluding the hardcoded excludes list."""
    out: list[tuple[str, str]] = []
    try:
        r = svc.calendarList().list().execute()
        for item in r.get("items", []) or []:
            cid = item.get("id") or ""
            if cid in HARDCODED_EXCLUDE_CAL_IDS:
                continue
            name = item.get("summaryOverride") or item.get("summary") or cid
            out.append((cid, str(name)))
    except Exception as e:
        # Log OAuth/auth errors explicitly to help debug token issues
        import sys
        print(f"[error] list_calendars failed: {e}", file=sys.stderr)
    return out


def read_events(
    svc, calendar_id: str, calendar_name: str,
    *,
    sync_token: str | None = None,
    time_min_iso: str | None = None,
    time_max_iso: str | None = None,
    max_events: int = MAX_EVENTS_PER_CAL,
) -> tuple[list[CalEvent], list[str], str | None]:
    """Return (active_events, cancelled_ids, new_sync_token).

    Two modes:
      - `sync_token` is `__bootstrap_done__` (sentinel) → treat as bootstrap
        (Google Calendar API doesn't support true syncToken in non-incremental calls)
      - `sync_token` None or sentinel → bootstrap (full scan over [time_min, time_max])

    On 410 Gone (sync_token expired / too old), returns (None, None, None) to
    signal a bootstrap is needed.

    QUIRK: Google Calendar API does NOT return nextSyncToken in bootstrap mode
    (when using timeMin/timeMax). Our workaround:
      1. First run: bootstrap with 2y past + 6m future, return sentinel
      2. Caller saves sentinel + timestamp
      3. Next run: load sentinel, recognize it, but use smaller window (1d past + 6m future)
      4. This avoids re-fetching millions of old events while catching new/changed ones
    """
    active: list[CalEvent] = []
    cancelled: list[str] = []
    next_page: str | None = None
    next_sync: str | None = None

    # Sentinel just means "we've done at least one bootstrap before"
    # We treat it the same as None (bootstrap mode) but it tells the caller
    # to use a smaller time window for efficiency.
    is_sentinel = sync_token == "__bootstrap_done__"
    sync_token = None  # Always treat as bootstrap (Google doesn't return syncToken)

    while True:
        kwargs: dict = {
            "calendarId": calendar_id,
            "singleEvents": True,
            "maxResults": 250,
        }
        if sync_token and not is_sentinel:
            # Real incremental: use syncToken
            kwargs["syncToken"] = sync_token
            kwargs["orderBy"] = "startTime"
        else:
            # Bootstrap or post-bootstrap: use time range
            # (Google Calendar API doesn't support true incremental without syncToken)
            if time_min_iso:
                kwargs["timeMin"] = time_min_iso
            if time_max_iso:
                kwargs["timeMax"] = time_max_iso
            kwargs["orderBy"] = "startTime"
        if next_page:
            kwargs["pageToken"] = next_page

        try:
            r = svc.events().list(**kwargs).execute()
        except Exception as exc:
            msg = str(exc)
            if "410" in msg or "gone" in msg.lower() or "syncToken" in msg:
                return [], [], None   # caller should fall back to bootstrap
            raise

        for item in r.get("items", []) or []:
            status = str(item.get("status") or "confirmed")
            if status == "cancelled":
                if item.get("id"):
                    cancelled.append(str(item["id"]))
                continue
            ev = _parse_event(item, calendar_id, calendar_name)
            if ev is not None:
                active.append(ev)
                if len(active) >= max_events:
                    break
        if len(active) >= max_events:
            break
        next_page = r.get("nextPageToken")
        next_sync = r.get("nextSyncToken") or next_sync
        if not next_page:
            break

    # Sentinel return: Google Calendar doesn't give us a syncToken in bootstrap mode.
    # We return our sentinel so the caller knows to save state.
    if active or cancelled:
        if next_sync is None:
            next_sync = "__bootstrap_done__"

    return active, cancelled, next_sync


# ── Chunker ────────────────────────────────────────────────────────────────

def _format_event_body(ev: CalEvent) -> str:
    """Human-readable transcript used as display_text (what reranker + LLM see).

    Shape:
        Título: X
        Cuándo: DD/MM/YYYY HH:MM (o "todo el día")
        Dónde: location (if any)
        Con: email1, email2 (if any)
        ---
        {description}
    """
    when = ""
    if ev.start_ts:
        dt = datetime.fromtimestamp(ev.start_ts)
        if ev.is_all_day:
            when = dt.strftime("%Y-%m-%d (todo el día)")
        else:
            end_part = ""
            if ev.end_ts:
                end_dt = datetime.fromtimestamp(ev.end_ts)
                end_part = f" → {end_dt.strftime('%H:%M')}"
            when = f"{dt.strftime('%Y-%m-%d %H:%M')}{end_part}"
    parts = [f"Título: {ev.title}"]
    if when:
        parts.append(f"Cuándo: {when}")
    if ev.location:
        parts.append(f"Dónde: {ev.location}")
    if ev.attendees:
        shown = ", ".join(ev.attendees[:5])
        if len(ev.attendees) > 5:
            shown += f" (+{len(ev.attendees) - 5} más)"
        parts.append(f"Con: {shown}")
    if ev.description:
        parts.append("---")
        parts.append(ev.description)
    body = "\n".join(parts)
    if len(body) > CHUNK_MAX_CHARS:
        body = body[:CHUNK_MAX_CHARS].rstrip()
    return body


def _embed_prefix(ev: CalEvent, body: str) -> str:
    """Discriminator prefix used at embed time (§2.5). Keeps display body raw."""
    when = ""
    if ev.start_ts:
        when = datetime.fromtimestamp(ev.start_ts).strftime("%Y-%m-%d")
    return f"[source=calendar | title={ev.title} | {when}] {body}"


def _event_doc_id(ev: CalEvent) -> str:
    return f"{DOC_ID_PREFIX}://{ev.calendar_id}/{ev.id}::0"


def _event_file_key(ev: CalEvent) -> str:
    return f"{DOC_ID_PREFIX}://{ev.calendar_id}/{ev.id}"


# ── Writer ─────────────────────────────────────────────────────────────────

def upsert_events(col, events: list[CalEvent]) -> int:
    """Write/refresh events into the sqlite-vec collection. Returns count written."""
    if not events:
        return 0
    bodies = [_format_event_body(e) for e in events]
    embed_texts = [_embed_prefix(e, b) for e, b in zip(events, bodies)]
    embeddings = rag.embed(embed_texts)

    keys = [_event_file_key(ev) for ev in events]
    try:
        existing = col.get(where={"file": {"$in": keys}}, include=[])
        if existing.get("ids"):
            col.delete(ids=existing["ids"])
    except Exception:
        pass

    ids = [_event_doc_id(e) for e in events]
    metas: list[dict] = []
    for e, body in zip(events, bodies):
        metas.append({
            "file": _event_file_key(e),
            "note": f"Cal: {e.title}",
            "folder": "",
            "tags": "",
            "hash": "",
            "outlinks": "",
            "source": "calendar",
            "created_ts": e.start_ts,   # calendar's natural anchor is start
            "calendar_id": e.calendar_id,
            "calendar_name": e.calendar_name,
            "event_id": e.id,
            "title": e.title,
            "location": e.location,
            "attendees": ",".join(e.attendees),
            "is_all_day": int(e.is_all_day),
            "start_ts": e.start_ts,
            "end_ts": e.end_ts,
            "status": e.status,
            "html_link": e.html_link,
            "parent": body,
        })
    col.add(ids=ids, embeddings=embeddings, documents=bodies, metadatas=metas)
    # Entity extraction — attendees / organizers / locations in event bodies.
    # Gated by `_entity_extraction_enabled()` + silent-fail if gliner absent.
    rag._extract_and_index_entities_for_chunks(bodies, ids, metas, "calendar")
    return len(events)


def delete_cancelled(col, calendar_id: str, event_ids: list[str]) -> int:
    """Remove rows for cancelled events. Returns count deleted."""
    if not event_ids:
        return 0
    keys = [f"{DOC_ID_PREFIX}://{calendar_id}/{eid}" for eid in event_ids]
    try:
        got = col.get(where={"file": {"$in": keys}}, include=[])
        if got.get("ids"):
            col.delete(ids=got["ids"])
            return len(got["ids"])
    except Exception:
        pass
    return 0


# ── Orchestration ──────────────────────────────────────────────────────────

def run(
    *,
    reset: bool = False,
    calendar_id: str | None = None,
    dry_run: bool = False,
    vault_col=None,
    svc=None,                    # injectable for tests
    now: datetime | None = None,
) -> dict:
    """Ingest calendar events. Returns a summary dict.

    Auth:
      - Default: call `_calendar_service()` to build a real API client.
      - Tests inject `svc=mock` to skip auth.

    Scope:
      - `calendar_id=None` → scan every non-excluded calendar from calendarList.
      - `calendar_id="ID"` → scan only that calendar.
    """
    t0 = time.perf_counter()
    summary: dict = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "reset": bool(reset),
        "dry_run": bool(dry_run),
        "calendars_scanned": 0,
        "events_indexed": 0,
        "events_cancelled": 0,
        "bootstrapped": 0,
        "incremental": 0,
        "duration_s": 0.0,
    }
    service = svc if svc is not None else _calendar_service()
    if service is None:
        summary["error"] = (
            "calendar service unavailable — configure ~/.calendar-mcp/ "
            "{gcp-oauth.keys.json, credentials.json}"
        )
        summary["duration_s"] = round(time.perf_counter() - t0, 2)
        return summary

    col = vault_col if vault_col is not None else rag.get_db()

    state_conn = sqlite3.connect(str(rag.DB_PATH / "ragvec.db"))
    _ensure_state_table(state_conn)
    if reset:
        _reset_cursors(state_conn)
    state_conn.commit()

    # Resolve calendar set.
    if calendar_id:
        calendars = [(calendar_id, calendar_id)]
    else:
        calendars = list_calendars(service)

    now_dt = now or datetime.now()
    time_min_iso = (now_dt - timedelta(days=INITIAL_WINDOW_PAST_DAYS)).isoformat() + "Z"
    time_max_iso = (now_dt + timedelta(days=INITIAL_WINDOW_FUTURE_DAYS)).isoformat() + "Z"

    # Smaller window for post-bootstrap runs (1 day past to catch changes)
    time_min_incremental = (now_dt - timedelta(days=1)).isoformat() + "Z"
    time_max_incremental = (now_dt + timedelta(days=INITIAL_WINDOW_FUTURE_DAYS)).isoformat() + "Z"

    for cal_id, cal_name in calendars:
        summary["calendars_scanned"] += 1
        sync_token = _load_sync_token(state_conn, cal_id)
        mode_bootstrap = sync_token is None

        # Use smaller window if we have the sentinel (post-bootstrap)
        if sync_token == "__bootstrap_done__":
            effective_time_min = time_min_incremental
            effective_time_max = time_max_incremental
            mode_bootstrap = False  # Treat sentinel as incremental for reporting
        else:
            effective_time_min = time_min_iso
            effective_time_max = time_max_iso
            # mode_bootstrap already set above: True iff sync_token is None

        active, cancelled, new_sync = read_events(
            service, cal_id, cal_name,
            sync_token=sync_token,
            time_min_iso=effective_time_min,
            time_max_iso=effective_time_max,
        )

        if active == [] and cancelled == [] and new_sync is None and not mode_bootstrap:
            # Server said 410 Gone — fall back to bootstrap now.
            active, cancelled, new_sync = read_events(
                service, cal_id, cal_name,
                sync_token=None,
                time_min_iso=time_min_iso,
                time_max_iso=time_max_iso,
            )
            mode_bootstrap = True

        if mode_bootstrap:
            summary["bootstrapped"] += 1
        else:
            summary["incremental"] += 1

        if not dry_run:
            n = upsert_events(col, active)
            summary["events_indexed"] += n
            # Count all cancellations observed from the API (regardless of
            # whether the row existed in our index — avoids reporting 0 on
            # the first bootstrap where cancellations came in alongside the
            # initial sync).
            summary["events_cancelled"] += len(cancelled)
            delete_cancelled(col, cal_id, cancelled)
            # Always save state if we got a non-None new_sync (includes sentinel)
            if new_sync is not None:
                _save_sync_token(
                    state_conn, cal_id, new_sync,
                    datetime.now().isoformat(timespec="seconds"),
                )
                state_conn.commit()
        else:
            summary["events_indexed"] += len(active)
            summary["events_cancelled"] += len(cancelled)

    state_conn.close()
    summary["duration_s"] = round(time.perf_counter() - t0, 2)
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--calendar-id", default=None,
                     help="Restrict to one calendar (default: all non-excluded)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    summary = run(
        reset=bool(args.reset),
        calendar_id=args.calendar_id,
        dry_run=bool(args.dry_run),
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    prefix = "[dry-run] " if args.dry_run else ""
    if "error" in summary:
        print(f"[error] {summary['error']}")
        return
    print(
        f"{prefix}{summary['calendars_scanned']} calendarios · "
        f"{summary['events_indexed']} eventos · "
        f"{summary['events_cancelled']} cancelados · "
        f"{summary['bootstrapped']} bootstrap / {summary['incremental']} incremental · "
        f"{summary['duration_s']}s"
    )


if __name__ == "__main__":
    main()
