"""WhatsApp ingester — Phase 1.a of the cross-source corpus (§10.1 user
decision 2026-04-20: WhatsApp first).

Reads messages from the local `whatsapp-mcp` bridge SQLite store
(`~/repositories/whatsapp-mcp/whatsapp-bridge/store/messages.db`), groups
contiguous messages from the same sender into conversational chunks, and
upserts them into the main sqlite-vec collection with
`source="whatsapp"`. Each chunk's metadata carries `chat_jid`,
`chat_name`, `sender`, `is_from_me`, `created_ts`, plus a vault-style
`file` field (vault-relative pseudo-path `whatsapp://<jid>/<first_msg_id>`)
for downstream rendering.

Retention: SOURCE_RETENTION_DAYS["whatsapp"] = 180 days (design §10.2).
Messages older than that are filtered at read time and never indexed.
Re-runs are incremental via a cursor in `rag_whatsapp_state` (max
`timestamp` per chat_jid), so a full scan only happens on first run /
reset.

Conversational chunking (§2.6 option A, §3.3):
- Contiguous messages from the same sender within 5 minutes → merged.
- Change of speaker OR gap ≥5min → split.
- Isolated messages <30 chars merged with the temporally-closest neighbor
  in the same chat (within 5min window).
- Chunk size cap: 800 chars (same as vault). If a merged window exceeds
  that, split at the closest message boundary.

`parent` field = ±10 message window around the chunk's first message in
the same chat, capped at 1200 chars — the design-doc recommendation for
rerank context.

Invoked via `rag index --source whatsapp [--reset] [--since YYYY-MM-DD]`.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


# ── Config ─────────────────────────────────────────────────────────────────

DEFAULT_BRIDGE_DB = Path.home() / "repositories" / "whatsapp-mcp" / "whatsapp-bridge" / "store" / "messages.db"

# Conversational chunk boundaries.
CHUNK_SAME_SPEAKER_WINDOW_S = 300     # 5 min — messages closer than this merge
CHUNK_MIN_CHARS = 30                  # solo messages under this try to merge up/down
CHUNK_MAX_CHARS = 800                 # hard cap matches vault
CHUNK_MIN_MERGE_CHARS = 150           # target lower bound (vault chunk rule)

PARENT_WINDOW_MESSAGES = 10           # ±N messages around chunk for `parent`
PARENT_MAX_CHARS = 1200               # cap (vault convention)

# Doc-id prefix. `whatsapp://` + chat_jid + first-msg-id gives stable IDs
# even if the bridge DB compacts.
DOC_ID_PREFIX = "whatsapp"

# Chat JIDs we never want to index regardless of user opt-out decision.
#
# - `status@broadcast` — WhatsApp's internal story/status feed. Not
#   conversational, not something the user "said", just ambient noise.
# - `rag.WHATSAPP_BOT_JID` (RagNet group `120363426178035051@g.us`) —
#   the bot's own UI surface. Receives morning briefs, archive
#   notifications, reminder pushes, anticipatory agent prompts, draft
#   cards, AND the user's slash commands (`/help`, `/note`, `/cap`,
#   etc.) and the bot's responses to them. None of this is
#   conversational corpus content — indexing it creates a feedback
#   loop: bot pushes a brief → indexer chunks the brief → next
#   retrieve surfaces yesterday's brief as "context" → next brief
#   includes references to its own past output. The fetchers used by
#   the brief itself (`_fetch_whatsapp_unread`, `_fetch_whatsapp_today`,
#   `_fetch_whatsapp_window`) already exclude this JID at the SQL
#   level; the indexer was the last open path. Closed 2026-04-28
#   alongside the `RAG_DRAFT_VIA_RAGNET` redirect flag (which would
#   amplify the leak — testing 1-2 days of redirected ambient sends
#   could add hundreds of bot-output chunks to the corpus).
#
# Also content-level: any message whose first char is U+200B (zero-
# width space) is bot output (the listener anti-loop marker). Defense
# in depth for any future bot-to-non-RagNet-chat path; today the JID
# guard alone catches all production cases.
HARDCODED_EXCLUDE_JIDS = frozenset({
    "status@broadcast",
    rag.WHATSAPP_BOT_JID,
    # 2026-04-30: chat "Notes" (5493425153999-1539438783@g.us) — el user
    # lo usa como inbox de capture (notes-to-self); cada msg ahí es
    # materializado como `.md` en `00-Inbox/` por `whatsapp-listener` y
    # ya entra al RAG via la indexación normal del vault. Si lo
    # indexáramos también acá, tendríamos cada nota duplicada como
    # chunk WhatsApp + chunk vault — ruido en search results y disk.
    # Override este JID via env `WA_LISTENER_NOTES_CHAT_JID`; tener que
    # mantener ambos en sync (acá + listener) es aceptable porque cambia
    # ~nunca y el JID del grupo es estable.
    "5493425153999-1539438783@g.us",
})

# U+200B (zero-width space) anti-loop marker. The listener bot prefixes
# every outbound message with this char; any message in the bridge DB
# starting with it is bot output, not user content, and must not enter
# the corpus.
_ANTILOOP_MARKER = "\u200B"


# ── Data types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WAMessage:
    id: str
    chat_jid: str
    chat_name: str
    sender: str
    content: str
    timestamp: float              # epoch seconds
    is_from_me: bool
    media_type: str | None


@dataclass(frozen=True)
class WAChunk:
    chat_jid: str
    chat_name: str
    sender: str                   # dominant sender for the chunk
    first_msg_id: str
    last_msg_id: str
    first_ts: float
    last_ts: float
    body: str                     # canonical display text (Speaker: line format)
    parent: str                   # ±N-message window for rerank context


# ── State: incremental cursor per chat ─────────────────────────────────────

_STATE_TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_whatsapp_state ("
    " chat_jid TEXT PRIMARY KEY,"
    " last_ts REAL NOT NULL,"
    " last_msg_id TEXT,"
    " updated_at TEXT NOT NULL"
    ")"
)

# Cursor para el scan de imágenes (cita detector). Separado del cursor de
# mensajes porque: (a) el user puede hacer `--reset` para re-ingestar chats
# sin querer re-OCR todas las imágenes (caro: ~1-2s por imagen entre OCR +
# helper call), (b) queremos un reset independiente si en algún momento
# agregamos lógica nueva al detector y hay que re-procesar. Key fija
# `global` porque no nos interesa cursor per-chat — el sidecar
# `rag_cita_detections` ya deduplica a nivel hash de OCR.
_MEDIA_STATE_TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_wa_media_state ("
    " scope TEXT PRIMARY KEY,"
    " last_ts REAL NOT NULL,"
    " updated_at TEXT NOT NULL"
    ")"
)


def _ensure_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(_STATE_TABLE_DDL)
    conn.execute(_MEDIA_STATE_TABLE_DDL)


def _load_media_cursor(conn: sqlite3.Connection, scope: str = "global") -> float:
    row = conn.execute(
        "SELECT last_ts FROM rag_wa_media_state WHERE scope = ?", (scope,),
    ).fetchone()
    return float(row[0]) if row else 0.0


def _save_media_cursor(
    conn: sqlite3.Connection, last_ts: float, scope: str = "global",
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO rag_wa_media_state (scope, last_ts, updated_at) "
        "VALUES (?, ?, ?)",
        (scope, float(last_ts), datetime.now().isoformat(timespec="seconds")),
    )


def _load_cursor(conn: sqlite3.Connection, chat_jid: str) -> float:
    """Retorna el cursor (last_ts) para un chat_jid, o 0.0 si no existe.

    2026-04-24 audit: el row puede existir pero con `last_ts = NULL`
    (corrupción manual, migración parcial). Sin el guard `row[0] is not
    None`, `float(None)` lanza TypeError y el caller no distingue entre
    "no cursor" vs "cursor corrupto" — ambos terminan re-procesando
    TODOS los mensajes desde epoch. Con el guard, NULL se trata igual
    que missing (re-process desde 0), pero si el row futuro se
    arregla, seguimos leyendo correcto.
    """
    try:
        row = conn.execute(
            "SELECT last_ts FROM rag_whatsapp_state WHERE chat_jid = ?",
            (chat_jid,),
        ).fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except (sqlite3.Error, TypeError, ValueError):
        # DB corrupta / tipo inesperado → fallback seguro a 0.0.
        # El caller hará re-process desde epoch, worst-case lento pero
        # nunca crash.
        pass
    return 0.0


def _save_cursor(
    conn: sqlite3.Connection, chat_jid: str, last_ts: float, last_msg_id: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO rag_whatsapp_state (chat_jid, last_ts, last_msg_id, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (chat_jid, last_ts, last_msg_id, datetime.now().isoformat(timespec="seconds")),
    )


def _reset_cursors(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM rag_whatsapp_state")


# ── Reader ─────────────────────────────────────────────────────────────────

def _parse_bridge_ts(raw: object) -> float | None:
    """WhatsApp bridge writes `timestamp` as Go RFC3339 — SQLite returns it
    as a string. Parse into epoch float. Accept numeric input too (some
    older rows)."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    # Trim fractional seconds > microseconds (Go writes nanoseconds).
    # Example: '2026-04-20T23:11:05.123456789-03:00' → keep to microseconds.
    if "." in s:
        head, sep, rest = s.partition(".")
        frac = rest
        tz = ""
        for i, ch in enumerate(rest):
            if ch in "+-Z":
                frac = rest[:i]
                tz = rest[i:]
                break
        frac = frac[:6]  # microseconds
        s = f"{head}{sep}{frac}{tz}"
    s = s.replace("Z", "+00:00") if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt.timestamp()


def read_messages(
    bridge_db: Path,
    *,
    since_ts: float = 0.0,
    chat_jid: str | None = None,
    exclude_jids: frozenset[str] = HARDCODED_EXCLUDE_JIDS,
) -> list[WAMessage]:
    """Load messages from the bridge DB. `since_ts` filters by epoch (exclusive,
    `> since_ts`) to cleanly implement incremental reads from the cursor.
    `chat_jid`, when given, restricts to one chat (mainly for tests).
    Skips empty-content rows and excluded JIDs. Sorted (chat_jid, timestamp)."""
    if not bridge_db.is_file():
        return []
    conn = sqlite3.connect(f"file:{bridge_db}?mode=ro&immutable=1", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        q = (
            "SELECT m.id, m.chat_jid, m.sender, m.content, m.timestamp, "
            " m.is_from_me, m.media_type, COALESCE(c.name, '') AS chat_name "
            "FROM messages m LEFT JOIN chats c ON c.jid = m.chat_jid "
            "WHERE m.content IS NOT NULL AND m.content != '' "
        )
        params: list = []
        if chat_jid:
            q += " AND m.chat_jid = ? "
            params.append(chat_jid)
        q += " ORDER BY m.chat_jid, m.timestamp"
        rows = conn.execute(q, params).fetchall()
    finally:
        conn.close()

    out: list[WAMessage] = []
    for r in rows:
        jid = r["chat_jid"]
        if jid in exclude_jids:
            continue
        content = str(r["content"] or "")
        # Skip the bot's own anti-loop output. The listener prefixes every
        # outbound bot message with U+200B so it can ignore its own echoes;
        # the same marker tells us "this row was authored by the bot, not
        # a human". Drop unconditionally — these rows have no corpus value.
        if content.startswith(_ANTILOOP_MARKER):
            continue
        ts = _parse_bridge_ts(r["timestamp"])
        if ts is None or ts <= since_ts:
            continue
        out.append(WAMessage(
            id=str(r["id"]),
            chat_jid=jid,
            chat_name=str(r["chat_name"] or jid),
            sender=str(r["sender"] or ""),
            content=content,
            timestamp=ts,
            is_from_me=bool(r["is_from_me"]),
            media_type=r["media_type"],
        ))
    return out


# ── Chunker ────────────────────────────────────────────────────────────────

def _speaker_label(msg: WAMessage) -> str:
    if msg.is_from_me:
        return "yo"
    # Resolve JID → nombre legible via phone index de dossiers (99 Mentions).
    # Fallback cascade dentro de `rag._resolve_sender_to_name`:
    #   1. dossier match por phone digits  → "Maria" / "Grecia"
    #   2. last-4 masked                   → "…3891"  (evita filtrar PII)
    #   3. local-part del JID sin digits   → name literal
    #   4. empty sender                    → `fallback` (chat_name)
    #   5. all empty                       → "?"
    # Pasamos `chat_name` como fallback para 1-on-1 chats donde el sender
    # puede venir vacío y `chats.name` ya tiene el nombre de la persona.
    return rag._resolve_sender_to_name(msg.sender, fallback=msg.chat_name or "")


def _render_window(messages: list[WAMessage]) -> str:
    """Format messages as a `Speaker: content` transcript. Newlines between
    turns. Truncated by caller via MAX_CHARS constraints."""
    lines: list[str] = []
    for m in messages:
        lines.append(f"{_speaker_label(m)}: {m.content.strip()}")
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    # Prefer breaking at the last newline before the cap.
    cut = text.rfind("\n", 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    return text[:cut].rstrip()


def chunk_conversation(
    messages: Iterable[WAMessage],
    *,
    same_speaker_window_s: float = CHUNK_SAME_SPEAKER_WINDOW_S,
    min_merge_chars: int = CHUNK_MIN_MERGE_CHARS,
    max_chars: int = CHUNK_MAX_CHARS,
) -> list[list[WAMessage]]:
    """Group messages in the same chat into chunks. Returns a list of
    message-groups. Input must be pre-sorted by (chat_jid, timestamp)
    — grouping happens within each chat_jid independently.

    Rules (applied per chat_jid):
      - Start a new group when speaker changes OR gap ≥ same_speaker_window_s
        OR adding the next message would exceed max_chars.
      - After initial pass, merge any group whose body < min_merge_chars
        into its temporally-nearest neighbor within same_speaker_window_s.
    """
    # Group by chat first — chunker is strictly intra-chat.
    by_chat: dict[str, list[WAMessage]] = {}
    for m in messages:
        by_chat.setdefault(m.chat_jid, []).append(m)

    groups: list[list[WAMessage]] = []
    for chat_jid, msgs in by_chat.items():
        msgs.sort(key=lambda m: m.timestamp)
        local: list[list[WAMessage]] = []
        current: list[WAMessage] = []
        current_len = 0
        for m in msgs:
            start_new = False
            if not current:
                start_new = False  # open group
            else:
                prev = current[-1]
                gap = m.timestamp - prev.timestamp
                if gap >= same_speaker_window_s:
                    start_new = True
                elif m.sender != current[-1].sender or m.is_from_me != current[-1].is_from_me:
                    start_new = True
                else:
                    # would overflow?
                    projected = current_len + len(m.content) + 10
                    if projected > max_chars:
                        start_new = True
            if start_new:
                local.append(current)
                current = [m]
                current_len = len(m.content) + 10
            else:
                current.append(m)
                current_len += len(m.content) + 10
        if current:
            local.append(current)

        # Merge undersized groups with nearest temporal neighbor.
        merged = _merge_tiny_groups(local, min_merge_chars, same_speaker_window_s, max_chars)
        groups.extend(merged)
    return groups


def _merge_tiny_groups(
    groups: list[list[WAMessage]],
    min_chars: int,
    window_s: float,
    max_chars: int,
) -> list[list[WAMessage]]:
    """Second pass: groups whose rendered body is <min_chars get merged into
    the closest neighbor within `window_s` if the combined body stays
    under `max_chars`. Prefers the previous group (time-causal) for ties."""
    if not groups:
        return []
    out = list(groups)
    i = 0
    while i < len(out):
        body_len = sum(len(m.content) + 10 for m in out[i])
        if body_len >= min_chars or len(out[i]) == 0:
            i += 1
            continue
        # Can we merge with previous?
        prev = out[i - 1] if i > 0 else None
        nxt = out[i + 1] if i + 1 < len(out) else None
        merged = False
        if prev is not None:
            gap = out[i][0].timestamp - prev[-1].timestamp
            combined = sum(len(m.content) + 10 for m in prev) + body_len
            if gap <= window_s and combined <= max_chars:
                prev.extend(out[i])
                out.pop(i)
                merged = True
                continue
        if not merged and nxt is not None:
            gap = nxt[0].timestamp - out[i][-1].timestamp
            combined = body_len + sum(len(m.content) + 10 for m in nxt)
            if gap <= window_s and combined <= max_chars:
                out[i].extend(nxt)
                out.pop(i + 1)
                i += 1
                continue
        i += 1
    return out


def build_chunks(
    messages: list[WAMessage],
    *,
    same_speaker_window_s: float = CHUNK_SAME_SPEAKER_WINDOW_S,
    min_merge_chars: int = CHUNK_MIN_MERGE_CHARS,
    max_chars: int = CHUNK_MAX_CHARS,
    parent_window: int = PARENT_WINDOW_MESSAGES,
    parent_max_chars: int = PARENT_MAX_CHARS,
) -> list[WAChunk]:
    """Pipeline: group → render body + parent window → WAChunk dataclasses.

    `parent` uses the surrounding ±N messages in the same chat (not just the
    chunk's own messages) to give the reranker more semantic context for
    short messages like "dale" / "ok mañana".
    """
    if not messages:
        return []
    # Build per-chat index so we can look up ±N neighbors for the parent window.
    by_chat_sorted: dict[str, list[WAMessage]] = {}
    for m in messages:
        by_chat_sorted.setdefault(m.chat_jid, []).append(m)
    for lst in by_chat_sorted.values():
        lst.sort(key=lambda x: x.timestamp)
    # Index lookup: (chat_jid, msg_id) → position in sorted list.
    id_to_pos: dict[tuple[str, str], int] = {}
    for jid, lst in by_chat_sorted.items():
        for i, m in enumerate(lst):
            id_to_pos[(jid, m.id)] = i

    groups = chunk_conversation(
        messages, same_speaker_window_s=same_speaker_window_s,
        min_merge_chars=min_merge_chars, max_chars=max_chars,
    )

    out: list[WAChunk] = []
    for grp in groups:
        if not grp:
            continue
        first, last = grp[0], grp[-1]
        body = _render_window(grp)
        body = _truncate(body, max_chars)

        # Parent: ±parent_window msgs around the first message of the chunk.
        lst = by_chat_sorted[first.chat_jid]
        pos = id_to_pos[(first.chat_jid, first.id)]
        lo = max(0, pos - parent_window)
        hi = min(len(lst), pos + parent_window + 1)
        parent = _render_window(lst[lo:hi])
        parent = _truncate(parent, parent_max_chars)

        out.append(WAChunk(
            chat_jid=first.chat_jid,
            chat_name=first.chat_name,
            sender=_speaker_label(first),
            first_msg_id=first.id,
            last_msg_id=last.id,
            first_ts=first.timestamp,
            last_ts=last.timestamp,
            body=body,
            parent=parent,
        ))
    return out


# ── Index writer ───────────────────────────────────────────────────────────

def _embed_prefix(chunk: WAChunk) -> str:
    """Prefix used at embed time — same discriminator pattern proposed in §2.5.
    Does NOT affect display_text (the raw body) which is what the reranker
    + LLM see."""
    return (
        f"[source=whatsapp | chat={chunk.chat_name} | from={chunk.sender}] "
        f"{chunk.body}"
    )


def _chunk_doc_id(chunk: WAChunk, idx: int) -> str:
    """Stable ID: `whatsapp://<chat_jid>/<first_msg_id>::<idx>`.
    `idx` supports multi-chunk splits for single large groups (not yet used
    — kept for future-proofing)."""
    return f"{DOC_ID_PREFIX}://{chunk.chat_jid}/{chunk.first_msg_id}::{idx}"


def _chunk_file_key(chunk: WAChunk) -> str:
    """Value stored in meta["file"] so existing sqlite-vec queries that use
    `where={"file": ...}` keep working. Matches the doc-id prefix sans the
    `::idx` suffix."""
    return f"{DOC_ID_PREFIX}://{chunk.chat_jid}/{chunk.first_msg_id}"


def upsert_chunks(
    col,
    chunks: list[WAChunk],
) -> int:
    """Write chunks into the main sqlite-vec collection.

    For each chunk: delete any prior rows whose `file` matches (idempotent
    reruns), then add the new embeddings + metadata. Returns count of
    chunks written.
    """
    if not chunks:
        return 0
    # Generate all embeddings in one batch to amortize ollama overhead.
    embed_texts = [_embed_prefix(c) for c in chunks]
    embeddings = rag.embed(embed_texts)

    # Delete existing rows for these chunks (idempotent).
    for c in chunks:
        key = _chunk_file_key(c)
        try:
            existing = col.get(where={"file": key}, include=[])
            if existing.get("ids"):
                col.delete(ids=existing["ids"])
        except Exception:
            # Defensive: collection I/O issues are best-effort here — if
            # delete fails, the add below may produce duplicates which the
            # caller can catch via periodic maintenance. Better than
            # dropping the whole batch.
            pass

    ids = [_chunk_doc_id(c, i) for i, c in enumerate(chunks)]
    docs = [c.body for c in chunks]
    metas = []
    for c in chunks:
        metas.append({
            "file": _chunk_file_key(c),
            "note": f"WA: {c.chat_name}",
            "folder": "",                 # WA doesn't live in vault folders
            "tags": "",
            "hash": "",                   # no hash — messages are immutable per id
            "outlinks": "",
            "source": "whatsapp",
            "created_ts": c.first_ts,
            "chat_jid": c.chat_jid,
            "chat_name": c.chat_name,
            "sender": c.sender,
            "first_msg_id": c.first_msg_id,
            "last_msg_id": c.last_msg_id,
            "first_ts": c.first_ts,
            "last_ts": c.last_ts,
            "parent": c.parent,
        })

    col.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    # Entity extraction — keep `rag_entities` fresh as WA history grows.
    # Gated by `_entity_extraction_enabled()` + silent-fail if gliner absent.
    rag._extract_and_index_entities_for_chunks(docs, ids, metas, "whatsapp")
    return len(chunks)


# ── Image → cita detector ──────────────────────────────────────────────────
#
# El bridge de WhatsApp-MCP guarda los attachments en
# `<store>/<chat_jid>/<filename>`. La tabla `messages` tiene `media_type`
# (ej. 'image') + `filename`. Para cada imagen nueva desde el último
# cursor la OCR-eamos con Apple Vision y la pasamos al detector de citas
# — si devuelve `is_cita=True` con confidence ≥ umbral, se agenda evento
# directo vía `rag._maybe_create_cita_from_ocr` (mismo pipeline que el
# indexer del vault y `rag capture --image`).
#
# Dedup: el sidecar `rag_cita_detections` impide double-create aunque un
# forward de imagen la haga llegar por dos chats distintos.
#
# Silent-fail total: cualquier excepción en OCR, detector o propose
# calendar event NUNCA propaga — el ETL de WA sigue funcionando aunque
# el subsistema de citas esté caído.


def _read_recent_image_messages(
    bridge_db: Path,
    since_ts: float,
    *,
    max_images: int | None = None,
    exclude_jids: frozenset[str] = HARDCODED_EXCLUDE_JIDS,
) -> list[tuple[str, str, float]]:
    """Return [(chat_jid, filename, ts), ...] for every image message newer
    than `since_ts`. Sorted ascending by timestamp so a mid-run crash
    resumes cleanly with the max ts of the processed batch.
    """
    if not bridge_db.is_file():
        return []
    conn = sqlite3.connect(f"file:{bridge_db}?mode=ro&immutable=1", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT chat_jid, filename, timestamp FROM messages "
            "WHERE media_type = 'image' AND filename IS NOT NULL "
            "AND filename != '' "
            "ORDER BY timestamp ASC"
        ).fetchall()
    finally:
        conn.close()
    out: list[tuple[str, str, float]] = []
    for r in rows:
        jid = str(r["chat_jid"])
        if jid in exclude_jids:
            continue
        ts = _parse_bridge_ts(r["timestamp"])
        if ts is None or ts <= since_ts:
            continue
        out.append((jid, str(r["filename"]), ts))
        if max_images is not None and len(out) >= max_images:
            break
    return out


def scan_wa_images_for_citas(
    bridge_db: Path,
    state_conn: sqlite3.Connection,
    *,
    max_images: int | None = None,
    reset_cursor: bool = False,
) -> dict:
    """Escaneá imágenes nuevas del bridge WA, OCR + cita detector.

    Returns: dict con contadores {images_seen, ocr_ok, cita_created,
    duplicate, no_cita, errors}.

    Silent-fail: cada imagen está en su propio try/except. Una imagen
    corrupta o un helper caído NO bloquea el resto del batch ni el
    ingester de mensajes.

    Requiere que `rag._cita_detect_enabled()` sea True. Si `RAG_CITA_DETECT=0`
    devuelve un summary vacío sin leer el bridge.
    """
    summary = {
        "images_seen": 0, "ocr_ok": 0, "vlm_captioned": 0,
        "cita_created": 0, "reminder_created": 0, "note_classified": 0,
        "duplicate": 0, "ambiguous": 0, "no_cita": 0, "low_confidence": 0,
        "errors": 0,
    }
    if not rag._cita_detect_enabled():
        summary["skipped"] = "RAG_CITA_DETECT=0"
        return summary

    store_root = bridge_db.parent
    # Reset del budget VLM al arrancar el scan. Cada ingest pass tiene su
    # propio cap — evita que un WA ingest anterior en el mismo proceso
    # (por ej. daemon de cron que procesa varias veces por hora) se coma
    # el budget de este run.
    try:
        rag._vlm_caption_budget_reset()
    except Exception:
        pass
    try:
        last_ts = 0.0 if reset_cursor else _load_media_cursor(state_conn)
    except Exception:
        last_ts = 0.0

    try:
        records = _read_recent_image_messages(
            bridge_db, since_ts=last_ts, max_images=max_images,
        )
    except Exception as exc:
        # Defensive: bridge DB locked / corrupt → skip silently.
        try:
            rag._silent_log("wa_scan_images_read", exc)
        except Exception:
            pass
        summary["errors"] += 1
        return summary

    summary["images_seen"] = len(records)
    if not records:
        return summary

    max_ts_processed = last_ts
    for chat_jid, filename, ts in records:
        # Build absolute path. Bridge layout: <store>/<chat_jid>/<filename>.
        img_path = store_root / chat_jid / filename
        if not img_path.is_file():
            # File in DB but missing on disk — user may have pruned the
            # store, or the bridge is still downloading. Advance cursor
            # so we don't re-try forever.
            max_ts_processed = max(max_ts_processed, ts)
            continue

        # Wrapper unificado: OCR + fallback a VLM caption si el OCR no
        # fue suficiente. Muchas imágenes de WhatsApp son fotos puras
        # (selfies, paisajes, memes) donde OCR devuelve "" — el VLM
        # captionea y el detector cita puede decidir sobre eso.
        try:
            ocr_text, img_source = rag._image_text_or_caption(img_path)
        except Exception as exc:
            try:
                rag._silent_log(f"wa_scan_read:{img_path}", exc)
            except Exception:
                pass
            summary["errors"] += 1
            max_ts_processed = max(max_ts_processed, ts)
            continue

        if not ocr_text or len(ocr_text.strip()) < 20:
            max_ts_processed = max(max_ts_processed, ts)
            continue

        if img_source == "vlm":
            summary["vlm_captioned"] += 1
        else:
            summary["ocr_ok"] += 1

        try:
            result = rag._maybe_create_cita_from_ocr(
                ocr_text, img_path, source="whatsapp",
            )
        except Exception as exc:
            try:
                rag._silent_log(f"wa_scan_detect:{img_path}", exc)
            except Exception:
                pass
            summary["errors"] += 1
            max_ts_processed = max(max_ts_processed, ts)
            continue

        if result is not None:
            dec = result.get("decision") or ""
            if dec == "cita":
                summary["cita_created"] += 1
            elif dec == "reminder":
                summary["reminder_created"] += 1
            elif dec == "note":
                summary["note_classified"] += 1
            elif dec == "duplicate":
                summary["duplicate"] += 1
            elif dec == "ambiguous":
                summary["ambiguous"] += 1
            elif dec == "low_confidence":
                summary["low_confidence"] += 1
            elif dec == "no":
                summary["no_cita"] += 1
            else:
                summary["errors"] += 1
        max_ts_processed = max(max_ts_processed, ts)

    # Save cursor only if we actually advanced.
    if max_ts_processed > last_ts:
        try:
            _save_media_cursor(state_conn, max_ts_processed)
            state_conn.commit()
        except Exception as exc:
            try:
                rag._silent_log("wa_scan_cursor_save", exc)
            except Exception:
                pass
    return summary


# ── Orchestration ──────────────────────────────────────────────────────────

def _retention_cutoff(now: float | None = None) -> float:
    days = rag.SOURCE_RETENTION_DAYS.get("whatsapp")
    if days is None:
        return 0.0
    now_epoch = now if now is not None else time.time()
    return now_epoch - (days * 86400.0)


def run(
    *,
    bridge_db: Path | None = None,
    since_iso: str | None = None,
    reset: bool = False,
    max_chats: int | None = None,
    max_messages: int | None = None,
    dry_run: bool = False,
    vault_col=None,
    scan_images: bool = True,
    max_images: int | None = None,
) -> dict:
    """Incremental ingest. Returns a summary dict with counts + timing.

    Cursor logic:
      - reset=True → wipe rag_whatsapp_state (full re-scan)
      - since_iso  → override cursor uniformly across all chats
      - default    → use each chat's stored last_ts
    `max_chats` / `max_messages` bound the scope for dry-runs or staged rollouts.
    """
    t0 = time.perf_counter()
    db = bridge_db or DEFAULT_BRIDGE_DB
    summary: dict = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "bridge_db": str(db),
        "reset": bool(reset),
        "dry_run": bool(dry_run),
        "messages_read": 0,
        "messages_after_retention": 0,
        "chunks_built": 0,
        "chunks_written": 0,
        "chats_touched": 0,
        "duration_s": 0.0,
    }

    if not db.is_file():
        summary["error"] = f"bridge DB not found: {db}"
        summary["duration_s"] = round(time.perf_counter() - t0, 2)
        return summary

    col = vault_col if vault_col is not None else rag.get_db()

    # Global cursor baseline. Per-chat cursors layer on top via rag_whatsapp_state.
    since_global = 0.0
    if since_iso:
        try:
            since_global = datetime.fromisoformat(since_iso).timestamp()
        except ValueError:
            summary["error"] = f"invalid --since: {since_iso}"
            summary["duration_s"] = round(time.perf_counter() - t0, 2)
            return summary

    retention_floor = _retention_cutoff()

    # Cursor state is kept in the same sqlite-vec DB next to the vec tables.
    # This is the state DB, not the bridge — never confuse them.
    state_conn = sqlite3.connect(str(rag.DB_PATH / "ragvec.db"))
    state_conn.row_factory = sqlite3.Row
    _ensure_state_table(state_conn)
    if reset:
        _reset_cursors(state_conn)
    state_conn.commit()

    # First read: all messages above the retention floor + global since.
    raw_messages = read_messages(
        db,
        since_ts=max(since_global, retention_floor),
    )
    summary["messages_read"] = len(raw_messages)

    # Apply per-chat cursor (last_ts stored after previous run).
    filtered: list[WAMessage] = []
    chat_cursors: dict[str, float] = {}
    for m in raw_messages:
        cur = chat_cursors.get(m.chat_jid)
        if cur is None:
            cur = _load_cursor(state_conn, m.chat_jid) if not reset else 0.0
            chat_cursors[m.chat_jid] = cur
        if m.timestamp <= cur:
            continue
        filtered.append(m)
    summary["messages_after_retention"] = len(filtered)

    # max_chats / max_messages caps (dry runs, staged rollout).
    if max_chats is not None:
        seen_chats: list[str] = []
        capped: list[WAMessage] = []
        for m in filtered:
            if m.chat_jid not in seen_chats:
                if len(seen_chats) >= max_chats:
                    continue
                seen_chats.append(m.chat_jid)
            capped.append(m)
        filtered = capped
    if max_messages is not None:
        filtered = filtered[:max_messages]

    chunks = build_chunks(filtered)
    summary["chunks_built"] = len(chunks)
    summary["chats_touched"] = len({c.chat_jid for c in chunks})

    if dry_run:
        state_conn.close()
        summary["duration_s"] = round(time.perf_counter() - t0, 2)
        return summary

    written = upsert_chunks(col, chunks)
    summary["chunks_written"] = written

    # Sanity check post-upsert: si el batch afirma haber escrito N chunks
    # pero la DB no refleja al menos 1 de ellos, no avanzamos el cursor —
    # la próxima corrida va a re-intentar. Previene el desync silencioso
    # reportado 2026-04-24: el cursor indicaba "14394 procesados" pero la
    # collection tenía 0 chunks con source="whatsapp". Chequeamos una
    # muestra (el primer chunk_id) en lugar del conteo total — el get por
    # id es O(1) y suficiente para detectar la falla macro.
    if written > 0 and chunks:
        sample_id = _chunk_doc_id(chunks[0], 0)
        try:
            probe = col.get(ids=[sample_id], include=[])
            if not probe.get("ids"):
                summary["error"] = (
                    f"desync: upsert reportó {written} chunks pero el id "
                    f"{sample_id!r} no está en la collection — cursor NOT "
                    "advanced, próxima corrida reintenta"
                )
                summary["duration_s"] = round(time.perf_counter() - t0, 2)
                state_conn.close()
                return summary
        except Exception as exc:
            # Si la probe misma falla, tratamos como desync conservador
            # (igual de peligroso que un 0-rows response).
            summary["error"] = (
                f"desync probe failed: {exc!r} — cursor NOT advanced"
            )
            summary["duration_s"] = round(time.perf_counter() - t0, 2)
            state_conn.close()
            return summary

    # Advance per-chat cursors to the latest timestamp seen this run.
    latest: dict[str, tuple[float, str]] = {}
    for m in filtered:
        ts_id = latest.get(m.chat_jid)
        if ts_id is None or m.timestamp > ts_id[0]:
            latest[m.chat_jid] = (m.timestamp, m.id)
    # 2026-04-24 audit: wrap cursor writes en BEGIN IMMEDIATE para
    # serializar escrituras concurrentes (cron + manual `rag index
    # --source whatsapp` corriendo en paralelo). BEGIN IMMEDIATE
    # adquiere el write-lock de sqlite antes de cualquier escritura,
    # así que 2 ingesters no pueden escribir cursors al mismo tiempo.
    # Sin esto, ambos podían leer el cursor viejo, procesar los
    # mismos mensajes, y escribir el cursor final con los timestamps
    # propios — con las filas de upsert_chunks potencialmente
    # duplicadas. Ahora el 2do writer espera al 1ro.
    #
    # Rollback si la contención es excesiva (poco probable, son
    # writes de 1-10 cursors per run): cambiar a `BEGIN DEFERRED` o
    # eliminar el `execute("BEGIN IMMEDIATE")` → vuelve al commit
    # implícito del driver.
    try:
        state_conn.execute("BEGIN IMMEDIATE")
        for jid, (ts, mid) in latest.items():
            _save_cursor(state_conn, jid, ts, mid)
        state_conn.commit()
    except sqlite3.Error:
        # Rollback en error — la próxima corrida re-intenta con
        # cursor viejo (peor caso: re-procesa algunos mensajes,
        # idempotente via `col.delete()` en upsert_chunks).
        try:
            state_conn.rollback()
        except sqlite3.Error:
            pass
        raise

    # Image → cita detector. Corre DESPUÉS del ingest de mensajes para que
    # un fallo en el detector no bloquee la indexación normal de chats.
    # `reset=True` también resetea el cursor de imágenes (consistencia: el
    # user que pide full re-scan de mensajes probablemente también quiere
    # re-evaluar imágenes). Silent-fail — summary.image_scan queda vacío
    # si algo explota.
    image_summary: dict = {}
    if scan_images and rag._cita_detect_enabled():
        try:
            image_summary = scan_wa_images_for_citas(
                db, state_conn, max_images=max_images, reset_cursor=reset,
            )
        except Exception as exc:
            try:
                rag._silent_log("wa_scan_images_outer", exc)
            except Exception:
                pass
            image_summary = {"error": str(exc)}
    summary["image_scan"] = image_summary

    state_conn.close()

    summary["duration_s"] = round(time.perf_counter() - t0, 2)
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bridge-db", default=str(DEFAULT_BRIDGE_DB),
                     help="Path to whatsapp-mcp messages.db")
    ap.add_argument("--since", default=None,
                     help="ISO timestamp — only index messages newer than this")
    ap.add_argument("--reset", action="store_true",
                     help="Reset per-chat cursor (full re-scan)")
    ap.add_argument("--max-chats", type=int, default=None,
                     help="Cap to first N chats (staged rollout / dry run)")
    ap.add_argument("--max-messages", type=int, default=None,
                     help="Cap to first N messages after filtering")
    ap.add_argument("--dry-run", action="store_true",
                     help="Compute chunks but don't write to the index")
    ap.add_argument("--json", action="store_true",
                     help="Emit summary as JSON")
    ap.add_argument("--skip-images", action="store_true",
                     help="No escanear imágenes nuevas por citas")
    ap.add_argument("--max-images", type=int, default=None,
                     help="Cap a N imágenes por run (sobre todo en el primer "
                          "ingest tras instalar el detector)")
    args = ap.parse_args()

    summary = run(
        bridge_db=Path(args.bridge_db),
        since_iso=args.since,
        reset=bool(args.reset),
        max_chats=args.max_chats,
        max_messages=args.max_messages,
        dry_run=bool(args.dry_run),
        scan_images=not bool(args.skip_images),
        max_images=args.max_images,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}{summary['messages_read']} msgs read · "
        f"{summary['messages_after_retention']} after cursor/retention · "
        f"{summary['chunks_built']} chunks · "
        f"{summary['chunks_written']} written · "
        f"{summary['chats_touched']} chats · "
        f"{summary['duration_s']}s"
    )
    img = summary.get("image_scan") or {}
    if img.get("images_seen"):
        print(
            f"images: {img.get('images_seen', 0)} seen · "
            f"{img.get('ocr_ok', 0)} ocr · "
            f"{img.get('vlm_captioned', 0)} vlm · "
            f"{img.get('cita_created', 0)} citas · "
            f"{img.get('reminder_created', 0)} recordatorios · "
            f"{img.get('note_classified', 0)} notes · "
            f"{img.get('duplicate', 0)} dup · "
            f"{img.get('ambiguous', 0)} ambig · "
            f"{img.get('errors', 0)} err"
        )
    if "error" in summary:
        print(f"[error] {summary['error']}")


if __name__ == "__main__":
    main()
