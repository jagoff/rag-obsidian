"""Gmail ingester — Phase 1.c of the cross-source corpus.

User decision §10.6 (2026-04-20): OAuth Google Gmail via the existing
`_gmail_service()` in rag.py (scope `gmail.modify`, so scope upgrade
not needed here — readonly would be enough, but we reuse the existing
service + creds to avoid dual-auth friction).

Ingest strategy:
  - Reader: `users().messages().list()` with `q="newer_than:365d"`
    scoped to INBOX + Sent. Paginates; hard cap 5000 messages per run.
  - Per-thread view: we group messages by `threadId` and emit one chunk
    per THREAD (not per individual message). Threads are the useful
    unit — a single email + reply + reply is one conversation, not
    three separate context blocks. Makes the chunker much simpler than
    WhatsApp's speaker-run merging.
  - Chunk body = subject + "De:" last sender + "Fecha:" + last 1-3
    message bodies, quoted-reply stripped. Cap 800 chars (same as vault).
  - `parent` = subject + first 1200 chars of the full thread (all
    messages concatenated). Feeds reranker more context than the chunk
    alone.
  - Recency decay: halflife 180d per §10.3 — gmail recency matters but
    less than WhatsApp.
  - Retention: 365d per §10.2 — old emails dropped at ingest time.
  - Source weight: 0.85.

Incremental sync:
  - Bootstrap: fetch historyId on first run, full scan by q="newer_than:365d".
  - Incremental: `users().history().list(startHistoryId=...)` returns
    added/modified messages since then. If the stored historyId is too
    old Gmail returns 404; fall back to a full re-scan.
  - Cursor table `rag_gmail_state(account_id, history_id, last_msg_id,
    updated_at)`. Single-account for now (user's primary).

Opt-out: sec 10.5 — "index everything". No label filters. Hardcoded skip
of the `CHAT` label (Google Hangouts chat logs, rarely useful and noisy).

Invoked via `rag index --source gmail [--reset] [--dry-run] [--max-messages N]`.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


# ── Config ─────────────────────────────────────────────────────────────────

INITIAL_WINDOW_DAYS = 365           # 1 year bootstrap (matches retention)
MAX_MESSAGES_PER_RUN = 5000         # hard cap per ingest run

# Labels we always skip. CHAT is Hangouts log noise; SPAM/TRASH would
# re-surface junk into the corpus.
HARDCODED_EXCLUDE_LABELS = frozenset({"CHAT", "SPAM", "TRASH"})

# Bot email guard (2026-04-30): skip emails from/to this address if the
# body contains [RAG_GENERATED] marker — prevents feedback loops. Env var
# OBSIDIAN_RAG_BOT_EMAIL defaults to empty (feature inactive if not set).
# Reason: when the system sends an email to a contact and that contact
# replies, the reply lands in Gmail, gets indexed, and the LLM might cite
# its own previous response back to the user (false self-referentiality).
BOT_EMAIL = os.environ.get("OBSIDIAN_RAG_BOT_EMAIL", "").strip()
BOT_EMAIL_MARKER = "[RAG_GENERATED]"

CHUNK_MAX_CHARS = 800
PARENT_MAX_CHARS = 1200
BODY_PREFIX_MAX = 600               # last-message body cap (for chunk body)

DOC_ID_PREFIX = "gmail"


# ── Data types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GmailMessage:
    id: str
    thread_id: str
    label_ids: list[str]
    subject: str
    sender: str
    to: str
    cc: str
    date_ts: float
    body: str                       # plain text, quoted-replies stripped


@dataclass(frozen=True)
class GmailThread:
    thread_id: str
    subject: str                    # from the latest message
    last_sender: str                # from the latest message
    last_ts: float
    messages: list[GmailMessage]    # oldest → newest
    folder: str                     # "INBOX" | "Sent" | mixed: "INBOX,Sent"


# ── State ──────────────────────────────────────────────────────────────────

_STATE_TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_gmail_state ("
    " account_id TEXT PRIMARY KEY,"
    " history_id TEXT,"
    " last_msg_id TEXT,"
    " updated_at TEXT NOT NULL"
    ")"
)


def _ensure_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(_STATE_TABLE_DDL)


def _load_history_id(conn: sqlite3.Connection, account_id: str) -> str | None:
    row = conn.execute(
        "SELECT history_id FROM rag_gmail_state WHERE account_id = ?",
        (account_id,),
    ).fetchone()
    return row[0] if row and row[0] else None


def _save_history_id(
    conn: sqlite3.Connection, account_id: str, history_id: str,
    last_msg_id: str | None = None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO rag_gmail_state (account_id, history_id, last_msg_id, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (
            account_id, history_id, last_msg_id,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )


def _save_history_id_cas(
    conn: sqlite3.Connection,
    account_id: str,
    new_history_id: str,
    expected_history_id: str | None,
    last_msg_id: str | None = None,
) -> bool:
    """Compare-and-swap del cursor de history.

    Audit 2026-04-25: el ``BEGIN IMMEDIATE`` solo cubría la escritura,
    pero el ``_load_history_id`` inicial NO estaba en la transacción.
    Si 2 ingesters concurrentes leían el mismo cursor, ambos procesaban
    el mismo rango de history y el segundo escribía sobre el primero
    (idempotente para el cursor pero el corpus tenía duplicados).

    Solución: en vez de una transacción larga (que tendría locked al
    state.db por todo el ingest, ~minutos), usamos optimistic
    concurrency: la escritura solo gana si el cursor no se movió
    desde que lo leímos al inicio.

    Args:
      account_id: identifica al usuario (multi-cuenta hipotético).
      new_history_id: el cursor que queremos escribir.
      expected_history_id: lo que leímos al inicio del run; None si
        era bootstrap (no había row).
      last_msg_id: pasado al INSERT OR UPDATE (informativo).

    Returns:
      True si la escritura tuvo efecto (no había conflicto).
      False si otro worker ya avanzó el cursor — el caller debe
      loggear y NO retry (el otro worker ya hizo el trabajo).
    """
    now_iso = datetime.now().isoformat(timespec="seconds")
    if expected_history_id is None:
        # Caso bootstrap: no había row. INSERT OR IGNORE de modo que
        # si otro worker bootsrapeó simultáneamente, su INSERT gana
        # y el nuestro es no-op (rowcount=0).
        cur = conn.execute(
            "INSERT OR IGNORE INTO rag_gmail_state "
            "(account_id, history_id, last_msg_id, updated_at) VALUES (?, ?, ?, ?)",
            (account_id, new_history_id, last_msg_id, now_iso),
        )
        return (cur.rowcount or 0) > 0
    # Caso incremental: había una row con `expected_history_id`. Solo
    # escribimos si nadie más la cambió mientras procesábamos. El WHERE
    # con history_id=expected es atomic en SQLite (un UPDATE es una
    # sola operación, sin race entre WHERE y SET).
    cur = conn.execute(
        "UPDATE rag_gmail_state SET history_id = ?, last_msg_id = ?, updated_at = ? "
        "WHERE account_id = ? AND history_id = ?",
        (new_history_id, last_msg_id, now_iso, account_id, expected_history_id),
    )
    return (cur.rowcount or 0) > 0


def _reset_cursor(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM rag_gmail_state")


# ── Body + quoted-reply stripping ──────────────────────────────────────────

_QUOTED_LINE_RE = re.compile(r"^\s*>.*$", re.MULTILINE)

# Common quoted-reply headers we strip from `body` to keep only the
# new content of each message. These lines plus everything after them
# get dropped.
_REPLY_HEADER_PATTERNS = [
    re.compile(r"^\s*On\s+\w+,?\s+.+\s+at\s+.+wrote:\s*$", re.MULTILINE),
    re.compile(r"^\s*El\s+\w+,?\s+.+escribió:\s*$", re.MULTILINE),
    re.compile(r"^\s*-+\s*Original Message\s*-+\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*-+\s*Mensaje original\s*-+\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*From:\s.+\s*$", re.MULTILINE),  # forwards
    re.compile(r"^\s*De:\s.+\s*$", re.MULTILINE),
]


def strip_quoted(text: str) -> str:
    """Remove quoted-reply sections from email body plain text.

    Heuristics (cheap, no external deps):
      - Truncate at the first match of a reply-header pattern.
      - Drop any remaining lines starting with `>`.
      - Collapse runs of 3+ blank lines → 1.
    """
    if not text:
        return ""
    # Truncate at first header match.
    cut_at = len(text)
    for pat in _REPLY_HEADER_PATTERNS:
        m = pat.search(text)
        if m and m.start() < cut_at:
            cut_at = m.start()
    text = text[:cut_at]
    # Drop quoted lines.
    text = _QUOTED_LINE_RE.sub("", text)
    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode_part_body(part: dict) -> str:
    """Decode a single MIME part's body to text. Prefers text/plain; falls
    back to text/html → strip tags."""
    mime = (part.get("mimeType") or "").lower()
    body = part.get("body") or {}
    data = body.get("data")
    if not data:
        # Multi-part: walk children.
        for sub in part.get("parts", []) or []:
            decoded = _decode_part_body(sub)
            if decoded:
                return decoded
        return ""
    try:
        raw = base64.urlsafe_b64decode(data + "===").decode(
            "utf-8", errors="replace"
        )
    except Exception:
        return ""
    if mime.startswith("text/html"):
        # Minimal tag strip — we don't need fidelity, just searchable text.
        raw = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
        raw = re.sub(r"<style[^>]*>.*?</style>",   " ", raw, flags=re.IGNORECASE | re.DOTALL)
        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = html.unescape(raw)
        raw = re.sub(r"\s+", " ", raw)
    return raw


def _extract_header(headers: list[dict], name: str) -> str:
    name_lc = name.lower()
    for h in headers or []:
        if (h.get("name") or "").lower() == name_lc:
            return str(h.get("value") or "")
    return ""


def _parse_rfc2822_date(raw: str) -> float:
    if not raw:
        return 0.0
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)
        return dt.timestamp()
    except Exception:
        return 0.0


def _parse_message(raw: dict) -> GmailMessage | None:
    """Map a Gmail users.messages.get() response → GmailMessage.
    Returns None on anything that's not indexable (missing id, no body)."""
    mid = raw.get("id")
    tid = raw.get("threadId")
    if not mid or not tid:
        return None
    payload = raw.get("payload") or {}
    headers = payload.get("headers") or []
    subject = _extract_header(headers, "Subject") or "(sin asunto)"
    sender = _extract_header(headers, "From")
    to = _extract_header(headers, "To")
    cc = _extract_header(headers, "Cc")
    date_raw = _extract_header(headers, "Date")
    # internalDate is ms-epoch string; prefer it over parsed Date header.
    try:
        internal_ms = int(raw.get("internalDate") or 0)
    except (TypeError, ValueError):
        internal_ms = 0
    ts = float(internal_ms / 1000.0) if internal_ms else _parse_rfc2822_date(date_raw)
    body = _decode_part_body(payload)
    body = strip_quoted(body)
    return GmailMessage(
        id=str(mid), thread_id=str(tid),
        label_ids=list(raw.get("labelIds") or []),
        subject=subject, sender=sender, to=to, cc=cc,
        date_ts=ts, body=body,
    )


def _messages_to_threads(msgs: list[GmailMessage]) -> list[GmailThread]:
    """Group messages by thread_id, sort each thread oldest→newest, derive
    the thread-level subject/last-sender/folder from the latest message."""
    by_tid: dict[str, list[GmailMessage]] = {}
    for m in msgs:
        by_tid.setdefault(m.thread_id, []).append(m)
    out: list[GmailThread] = []
    for tid, msgs_list in by_tid.items():
        msgs_list.sort(key=lambda m: m.date_ts)
        latest = msgs_list[-1]
        # Folder = union of labels seen across the thread.
        folders: list[str] = []
        for m in msgs_list:
            if "INBOX" in m.label_ids and "INBOX" not in folders:
                folders.append("INBOX")
            if "SENT" in m.label_ids and "Sent" not in folders:
                folders.append("Sent")
        out.append(GmailThread(
            thread_id=tid,
            subject=latest.subject,
            last_sender=latest.sender,
            last_ts=latest.date_ts,
            messages=msgs_list,
            folder=",".join(folders) or "INBOX",
        ))
    out.sort(key=lambda t: t.last_ts, reverse=True)
    return out


# ── Reader ─────────────────────────────────────────────────────────────────

def list_recent_messages(svc, *, days: int = INITIAL_WINDOW_DAYS,
                          max_results: int = MAX_MESSAGES_PER_RUN) -> list[str]:
    """Paginate users.messages.list() with q="newer_than:Nd in:anywhere
    -in:chats -in:spam -in:trash". Returns message IDs."""
    q = (
        f"newer_than:{days}d (in:inbox OR in:sent) "
        "-in:chats -in:spam -in:trash"
    )
    out: list[str] = []
    page_token: str | None = None
    while True:
        kwargs: dict = {"userId": "me", "q": q, "maxResults": 500}
        if page_token:
            kwargs["pageToken"] = page_token
        try:
            r = svc.users().messages().list(**kwargs).execute()
        except Exception:
            break
        for item in r.get("messages", []) or []:
            mid = item.get("id")
            if mid:
                out.append(mid)
        if len(out) >= max_results:
            out = out[:max_results]
            break
        page_token = r.get("nextPageToken")
        if not page_token:
            break
    return out


def fetch_message(svc, message_id: str) -> GmailMessage | None:
    """Full .get() for one message. Returns None on any error."""
    try:
        raw = svc.users().messages().get(
            userId="me", id=message_id, format="full",
        ).execute()
    except Exception:
        return None
    if not raw:
        return None
    labels = set(raw.get("labelIds") or [])
    if labels & HARDCODED_EXCLUDE_LABELS:
        return None
    return _parse_message(raw)


def fetch_messages_bulk(svc, ids: list[str]) -> list[GmailMessage]:
    """Sequential .get() per ID. Gmail API doesn't have a real batch path
    in googleapiclient's default discovery; at 500-1000 msgs this is
    still well under the 250 queries/user/second quota.

    Bot-email guard (2026-04-30): skip messages from/to BOT_EMAIL (if
    configured) when the body contains BOT_EMAIL_MARKER — prevents
    feedback loops where the bot's own generated responses get indexed
    and re-cited back to the user."""
    out: list[GmailMessage] = []
    for mid in ids:
        m = fetch_message(svc, mid)
        if m is not None:
            # Skip bot-generated messages if guard is active.
            if BOT_EMAIL and (m.sender == BOT_EMAIL or BOT_EMAIL in m.to):
                if BOT_EMAIL_MARKER in m.body:
                    continue  # Skip this message; it's bot-generated output.
            out.append(m)
    return out


def get_profile(svc) -> tuple[str, str | None]:
    """Return (email_address, history_id). Both empty strings on error."""
    try:
        p = svc.users().getProfile(userId="me").execute()
    except Exception:
        return "", None
    return str(p.get("emailAddress") or ""), p.get("historyId")


# ── Chunker ────────────────────────────────────────────────────────────────

def _format_thread_body(thread: GmailThread) -> str:
    """Chunk body — focuses on the thread latest + headers. Keeps it short
    so search surfaces the most recent / relevant turn."""
    latest = thread.messages[-1]
    when = ""
    if latest.date_ts:
        when = datetime.fromtimestamp(latest.date_ts).strftime("%Y-%m-%d %H:%M")
    parts = [f"Asunto: {thread.subject}"]
    if thread.last_sender:
        parts.append(f"De: {thread.last_sender}")
    if when:
        parts.append(f"Fecha: {when}")
    if latest.to:
        parts.append(f"Para: {latest.to[:200]}")
    parts.append(f"Folder: {thread.folder}")
    body_snippet = latest.body[:BODY_PREFIX_MAX] if latest.body else ""
    if body_snippet:
        parts.append("---")
        parts.append(body_snippet)
    body = "\n".join(parts)
    if len(body) > CHUNK_MAX_CHARS:
        body = body[:CHUNK_MAX_CHARS].rstrip()
    return body


def _format_thread_parent(thread: GmailThread) -> str:
    """Parent context — full thread concatenated, capped at PARENT_MAX_CHARS.
    Feeds reranker with the context of earlier turns."""
    segs: list[str] = [f"Asunto: {thread.subject}"]
    for m in thread.messages:
        when = datetime.fromtimestamp(m.date_ts).strftime("%Y-%m-%d %H:%M") if m.date_ts else ""
        segs.append(f"\n--- {m.sender} · {when} ---\n{m.body[:400]}")
    body = "\n".join(segs)
    if len(body) > PARENT_MAX_CHARS:
        body = body[:PARENT_MAX_CHARS].rstrip()
    return body


def _embed_prefix(thread: GmailThread, body: str) -> str:
    """Discriminator prefix (§2.5)."""
    return (
        f"[source=gmail | from={thread.last_sender} | subject={thread.subject}] "
        f"{body}"
    )


def _thread_doc_id(thread: GmailThread) -> str:
    return f"{DOC_ID_PREFIX}://thread/{thread.thread_id}::0"


def _thread_file_key(thread: GmailThread) -> str:
    return f"{DOC_ID_PREFIX}://thread/{thread.thread_id}"


# ── Writer ─────────────────────────────────────────────────────────────────

def upsert_threads(col, threads: list[GmailThread]) -> int:
    """Embed + upsert one chunk per thread. Idempotent — existing rows
    under the thread's file key are deleted before add."""
    if not threads:
        return 0
    bodies = [_format_thread_body(t) for t in threads]
    embed_texts = [_embed_prefix(t, b) for t, b in zip(threads, bodies)]

    # Retry embedding con backoff para Ollama transient errors (2026-04-30)
    embeddings = None
    for attempt in range(3):
        try:
            embeddings = rag.embed(embed_texts)
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)  # 2s, 4s
                continue
            raise

    for t in threads:
        try:
            existing = col.get(where={"file": _thread_file_key(t)}, include=[])
            if existing.get("ids"):
                col.delete(ids=existing["ids"])
        except Exception:
            pass

    ids = [_thread_doc_id(t) for t in threads]
    metas: list[dict] = []
    for t, body in zip(threads, bodies):
        metas.append({
            "file": _thread_file_key(t),
            "note": f"Mail: {t.subject[:60]}",
            "folder": t.folder,
            "tags": "",
            "hash": "",
            "outlinks": "",
            "source": "gmail",
            "created_ts": t.last_ts,
            "thread_id": t.thread_id,
            "subject": t.subject,
            "sender": t.last_sender,
            "last_ts": t.last_ts,
            "n_messages": len(t.messages),
            "parent": _format_thread_parent(t),
        })
    col.add(ids=ids, embeddings=embeddings, documents=bodies, metadatas=metas)
    # Entity extraction — senders / organizations mentioned in email threads.
    # Gated by `_entity_extraction_enabled()` + silent-fail if gliner absent.
    rag._extract_and_index_entities_for_chunks(bodies, ids, metas, "gmail")
    return len(threads)


def delete_threads(col, thread_ids: list[str]) -> int:
    n = 0
    for tid in thread_ids:
        key = f"{DOC_ID_PREFIX}://thread/{tid}"
        try:
            got = col.get(where={"file": key}, include=[])
            if got.get("ids"):
                col.delete(ids=got["ids"])
                n += 1
        except Exception:
            continue
    return n


# ── History-based incremental sync ────────────────────────────────────────

def apply_history(svc, start_history_id: str) -> tuple[list[str], list[str], str | None]:
    """Return (message_ids_to_fetch, thread_ids_to_delete, latest_history_id).

    Uses users.history.list() to walk deltas. On 404 (history too old),
    returns ([], [], None) so caller can bootstrap from scratch.
    """
    added_msgs: set[str] = set()
    removed_threads: set[str] = set()
    latest_hid: str | None = start_history_id
    page_token: str | None = None
    while True:
        kwargs: dict = {
            "userId": "me",
            "startHistoryId": start_history_id,
            "maxResults": 500,
        }
        if page_token:
            kwargs["pageToken"] = page_token
        try:
            r = svc.users().history().list(**kwargs).execute()
        except Exception as exc:
            msg = str(exc)
            if "404" in msg or "Not Found" in msg or "historyId" in msg.lower():
                return [], [], None
            return list(added_msgs), list(removed_threads), latest_hid
        for h in r.get("history", []) or []:
            latest_hid = h.get("id") or latest_hid
            for m in h.get("messagesAdded", []) or []:
                mid = (m.get("message") or {}).get("id")
                if mid:
                    added_msgs.add(mid)
            # We treat labelsRemoved to TRASH / SPAM as thread removal.
            for m in h.get("messagesDeleted", []) or []:
                tid = (m.get("message") or {}).get("threadId")
                if tid:
                    removed_threads.add(tid)
        latest_hid = r.get("historyId") or latest_hid
        page_token = r.get("nextPageToken")
        if not page_token:
            break
    return list(added_msgs), list(removed_threads), latest_hid


# ── Retention ─────────────────────────────────────────────────────────────

def _retention_cutoff(now: float | None = None) -> float:
    days = rag.SOURCE_RETENTION_DAYS.get("gmail")
    if days is None:
        return 0.0
    now_epoch = now if now is not None else time.time()
    return now_epoch - (days * 86400.0)


# ── Orchestration ──────────────────────────────────────────────────────────

def run(
    *,
    reset: bool = False,
    max_messages: int | None = None,
    dry_run: bool = False,
    vault_col=None,
    svc=None,
) -> dict:
    """Ingest gmail threads. See module docstring for the full contract."""
    t0 = time.perf_counter()
    summary: dict = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "reset": bool(reset),
        "dry_run": bool(dry_run),
        "messages_seen": 0,
        "threads_built": 0,
        "threads_indexed": 0,
        "threads_deleted": 0,
        "bootstrapped": False,
        "incremental": False,
        "duration_s": 0.0,
    }
    service = svc if svc is not None else rag._gmail_service()
    if service is None:
        summary["error"] = (
            "gmail service unavailable — configure ~/.gmail-mcp/ "
            "{gcp-oauth.keys.json, credentials.json}"
        )
        summary["duration_s"] = round(time.perf_counter() - t0, 2)
        return summary

    col = vault_col if vault_col is not None else rag.get_db()
    email, profile_hid = get_profile(service)
    account_id = email or "unknown@gmail"

    state_conn = sqlite3.connect(str(rag.DB_PATH / "ragvec.db"))
    _ensure_state_table(state_conn)
    if reset:
        _reset_cursor(state_conn)
    state_conn.commit()

    stored_hid = _load_history_id(state_conn, account_id)
    mode_bootstrap = stored_hid is None

    # Message IDs to fetch full payloads for.
    fetch_ids: list[str] = []
    delete_tids: list[str] = []

    # Fallback-bootstrap flag: True si entramos al bootstrap desde el
    # path incremental (history expired, 410 Gone). Se propaga al
    # cursor advancement abajo para que NO se confíe en `profile_hid`
    # (que fue capturado ANTES de la re-bootstrap y por ende puede
    # ser stale respecto del conjunto de mensajes que acabamos de
    # procesar).
    latest_hid: str | None = None
    fallback_bootstrap = False
    if mode_bootstrap:
        fetch_ids = list_recent_messages(
            service,
            days=INITIAL_WINDOW_DAYS,
            max_results=max_messages or MAX_MESSAGES_PER_RUN,
        )
        summary["bootstrapped"] = True
    else:
        add_ids, del_tids, latest_hid = apply_history(service, stored_hid)
        if latest_hid is None:
            # History expired → fallback bootstrap.
            fetch_ids = list_recent_messages(
                service, days=INITIAL_WINDOW_DAYS,
                max_results=max_messages or MAX_MESSAGES_PER_RUN,
            )
            summary["bootstrapped"] = True
            fallback_bootstrap = True
        else:
            fetch_ids = add_ids[: max_messages or MAX_MESSAGES_PER_RUN]
            delete_tids = del_tids
            summary["incremental"] = True

    summary["messages_seen"] = len(fetch_ids)

    msgs = fetch_messages_bulk(service, fetch_ids)
    # Retention: drop messages older than cutoff (defense in depth —
    # Gmail's `newer_than:Nd` should already filter, but on incremental
    # path old edits can slip through).
    retention_floor = _retention_cutoff()
    if retention_floor:
        msgs = [m for m in msgs if m.date_ts >= retention_floor]

    threads = _messages_to_threads(msgs)
    summary["threads_built"] = len(threads)

    if not dry_run:
        n = upsert_threads(col, threads)
        summary["threads_indexed"] = n
        d = delete_threads(col, delete_tids)
        summary["threads_deleted"] = d

        # Advance cursor. 2026-04-24 audit fix: SIEMPRE re-fetch el
        # historyId post-ingest via `get_profile(service)` en vez de
        # confiar en `profile_hid` (capturado al inicio del run, stale).
        # Relevante especialmente para:
        #
        # 1. `mode_bootstrap=True` (stored_hid era None): pre-fix usábamos
        #    el `profile_hid` pre-bootstrap. Si Gmail recibe mensajes
        #    DURANTE el fetch de 365 días, esos mensajes caen en un gap —
        #    el próximo `apply_history(profile_hid)` devuelve un rango
        #    que ya parcialmente procesamos, pero cualquier mensaje
        #    recibido entre `get_profile()` inicial y el upsert_threads
        #    queda sin cursor que lo capture.
        #
        # 2. `fallback_bootstrap=True` (incremental tiró 410 Gone):
        #    pre-fix el path incremental sí refetcheaba via
        #    `get_profile()`, pero el fallback bootstrap NO lo hacía.
        #    Usábamos `profile_hid` (stale) → próxima corrida mismo 410
        #    → loop infinito de bootstraps con mensajes perdidos en el
        #    gap entre cada refetch.
        #
        # Post-fix: `fresh_hid = get_profile(service)` siempre; caemos a
        # `latest_hid` (del apply_history) si disponible, y solo como
        # último recurso a `profile_hid` (si el get_profile post-ingest
        # falla por network). Así garantizamos que el cursor es lo más
        # adelantado posible sin perder mensajes.
        try:
            _, fresh_hid = get_profile(service)
        except Exception:
            fresh_hid = None
        new_hid = fresh_hid or latest_hid or profile_hid
        if new_hid:
            # 2026-04-25 audit: además del BEGIN IMMEDIATE en la
            # escritura, ahora usamos compare-and-swap contra el
            # `stored_hid` que leímos al inicio. Si 2 ingesters
            # corren en paralelo y ambos leen `stored_hid=X`, ambos
            # procesan el rango X→Y; el primero en commitear escribe
            # Y, el segundo intenta escribir Y también pero su CAS
            # falla (el cursor ya no está en X). Loggeamos el conflicto
            # y NO retry — el corpus quedará con duplicados de ese
            # run, pero el cursor es consistente y la próxima corrida
            # NO re-procesa el rango.
            try:
                state_conn.execute("BEGIN IMMEDIATE")
                won = _save_history_id_cas(
                    state_conn, account_id, str(new_hid),
                    expected_history_id=stored_hid,
                )
                state_conn.commit()
                if not won:
                    summary["cursor_cas_conflict"] = True
            except sqlite3.Error:
                try:
                    state_conn.rollback()
                except sqlite3.Error:
                    pass
                raise
    else:
        summary["threads_indexed"] = len(threads)
        summary["threads_deleted"] = len(delete_tids)

    state_conn.close()
    summary["duration_s"] = round(time.perf_counter() - t0, 2)
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--max-messages", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    summary = run(
        reset=bool(args.reset),
        max_messages=args.max_messages,
        dry_run=bool(args.dry_run),
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if "error" in summary:
        print(f"[error] {summary['error']}")
        return
    mode = "bootstrap" if summary["bootstrapped"] else "incremental"
    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}Gmail ({mode}): "
        f"{summary['messages_seen']} msgs seen · "
        f"{summary['threads_built']} threads · "
        f"{summary['threads_indexed']} indexados · "
        f"{summary['threads_deleted']} borrados · "
        f"{summary['duration_s']}s"
    )


if __name__ == "__main__":
    main()
