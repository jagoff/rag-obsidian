"""Apple Contacts ingester — Phase 1.e of the cross-source corpus.

Reads every `AddressBook-v22.abcddb` SQLite under
`~/Library/Application Support/AddressBook/Sources/*/` (one per
account — iCloud, Google, local) and indexes each contact as an atomic
chunk with `source=contacts`.

Two jobs in one:
  1. **Corpus**: contacts queryable via `rag query` ("el teléfono de
     Juli", "quiénes son los de Moka", "cumpleaños de Astor").
  2. **Lookup helper**: `resolve_phone(number) -> Contact | None`
     used by `scripts/ingest_calls.py` (and future ingest_messages /
     whatsapp enrichment) to resolve raw phone digits into human
     names. Without this, call logs show `+5491112345678` instead of
     `Juli Pérez`.

Ingest strategy:
  - Reader: direct SQLite against every `*/AddressBook-v22.abcddb`.
    Zero cloud calls, zero pyobjc — the .abcddb is a plain SQLite
    file, no Core Data bridge needed.
  - Chunk per contact (<800 chars, atomic). Merges first+last+org+
    phones+emails+note into a single human-readable body.
  - Retention: None (contacts don't age, user prunes in Contacts.app).
  - Source weight: 0.95 (editorial trust — user-curated).
  - Recency halflife: None (contacts don't decay).

Incremental sync:
  - Per-contact `content_hash` in `rag_contacts_state(uid, hash,
    last_seen_ts, updated_at)`. On each run: fetch all contacts,
    hash normalised fields; upsert if changed, delete stale ids.
  - Same pattern as `ingest_reminders.py` — `ZMODIFICATIONDATE` is
    unreliable (bumps on idle iCloud sync).

Phone normalisation (for `resolve_phone`):
  - All digits kept, `+` allowed at the start, everything else
    stripped. Argentina mobile `+54 9 11 …` and landline `+54 11 …`
    both collapse to the same last-8 digits, so the matcher tries
    progressively shorter suffixes (exact → last-10 → last-8 →
    last-7) until a unique match is found. Ambiguous? Returns None
    rather than guessing (caller falls back to the raw number).

Invoked via `rag index --source contacts [--reset] [--dry-run]`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


# ── Config ─────────────────────────────────────────────────────────────────

DEFAULT_ABOOK_ROOT = Path.home() / "Library/Application Support/AddressBook/Sources"
CHUNK_MAX_CHARS = 800
DOC_ID_PREFIX = "contacts"

# Apple Core Data stores timestamps as seconds since the Cocoa epoch
# (2001-01-01 UTC). Unix epoch is 1970. Offset is exact.
COCOA_EPOCH_OFFSET = 978307200.0


# ── Data types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Phone:
    raw: str                # exact string from ZFULLNUMBER (with formatting)
    label: str              # "mobile", "work", "home", "iPhone", ...
    digits: str             # normalised (leading `+` kept, everything else digits)


@dataclass(frozen=True)
class Email:
    address: str
    label: str              # "work", "home", "other"


@dataclass(frozen=True)
class Contact:
    uid: str                # ZUNIQUEID (globally stable across sync)
    source_id: str          # subdir UUID under AddressBook/Sources/ (disambiguator)
    first_name: str
    last_name: str
    middle_name: str
    nickname: str
    organization: str
    job_title: str
    department: str
    note: str
    phones: tuple[Phone, ...] = field(default_factory=tuple)
    emails: tuple[Email, ...] = field(default_factory=tuple)
    birthday_ts: float = 0.0
    created_ts: float = 0.0
    modified_ts: float = 0.0

    @property
    def display_name(self) -> str:
        """Human name: first + middle + last, falls back to nickname,
        then organization, then '(sin nombre)'. Stable ordering."""
        parts = [self.first_name, self.middle_name, self.last_name]
        name = " ".join(p for p in parts if p).strip()
        if name:
            return name
        if self.nickname:
            return self.nickname
        if self.organization:
            return self.organization
        return "(sin nombre)"

    @property
    def anchor_ts(self) -> float:
        """Timestamp used as `created_ts` for recency decay. Contacts
        don't decay (halflife=None) but we still set this honestly."""
        return self.created_ts or self.modified_ts or 0.0

    def is_empty(self) -> bool:
        """Garbage record with no useful content? Some .abcddb rows are
        metadata / groups / linked placeholders with every field null.
        Filter them out to avoid polluting the corpus."""
        has_name = any([self.first_name, self.last_name, self.nickname,
                        self.middle_name])
        has_contact = bool(self.phones) or bool(self.emails)
        has_org = bool(self.organization)
        return not (has_name or has_contact or has_org)


# ── Timestamp conversion ───────────────────────────────────────────────────

def _cocoa_to_unix(raw: float | None) -> float:
    """Convert a Core Data `TIMESTAMP` column (Cocoa epoch, seconds since
    2001-01-01 UTC) to Unix epoch. Returns 0.0 for None / 0 / negative."""
    if raw is None:
        return 0.0
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if val <= 0:
        return 0.0
    return val + COCOA_EPOCH_OFFSET


# ── Phone normalisation ────────────────────────────────────────────────────

def normalise_phone(raw: str) -> str:
    """Normalise a phone string to digits (+ optional leading `+`).
    Everything else (spaces, parens, dashes, dots, letters) is stripped.
    Returns '' on empty / None / no-digits input.
    Examples:
      '+54 9 11 1234-5678' → '+5491112345678'
      '(415) 555-1234'     → '4155551234'
      'ext. 101'           → '101'
    """
    if not raw:
        return ""
    s = str(raw).strip()
    out_chars: list[str] = []
    for i, ch in enumerate(s):
        if ch == "+" and i == 0:
            out_chars.append("+")
        elif ch.isdigit():
            out_chars.append(ch)
    return "".join(out_chars)


def _phone_match_keys(digits: str) -> list[str]:
    """Return progressively-shorter suffix keys for phone lookup.
    Strategy: exact → last-10 → last-8 → last-7. Shorter than 7 = too
    many collisions to trust. The `+` prefix is dropped for suffix keys.
    Caller iterates and stops at the first unique match.
    """
    if not digits:
        return []
    raw = digits.lstrip("+")
    keys: list[str] = [digits]
    for n in (10, 8, 7):
        if len(raw) >= n:
            tail = raw[-n:]
            # Avoid dupes when full number IS already n digits.
            if tail and tail not in keys:
                keys.append(tail)
    return keys


# ── Reader ─────────────────────────────────────────────────────────────────

def _find_abcddb_files(root: Path = DEFAULT_ABOOK_ROOT) -> list[tuple[str, Path]]:
    """Locate every `AddressBook-v22.abcddb` SQLite under the AddressBook
    Sources dir. Returns list of `(source_id, absolute_path)` tuples.
    `source_id` is the parent-dir name (UUID-ish), used as a namespace
    disambiguator in the doc_id.
    """
    if not root.exists():
        return []
    out: list[tuple[str, Path]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        db = sub / "AddressBook-v22.abcddb"
        if db.exists():
            out.append((sub.name, db))
    return out


_PHONES_SQL = (
    "SELECT ZOWNER, ZFULLNUMBER, COALESCE(ZLABEL, '') FROM ZABCDPHONENUMBER "
    "WHERE ZFULLNUMBER IS NOT NULL AND ZFULLNUMBER != '' "
    "ORDER BY ZOWNER, ZORDERINGINDEX"
)

_EMAILS_SQL = (
    "SELECT ZOWNER, ZADDRESS, COALESCE(ZLABEL, '') FROM ZABCDEMAILADDRESS "
    "WHERE ZADDRESS IS NOT NULL AND ZADDRESS != '' "
    "ORDER BY ZOWNER, ZORDERINGINDEX"
)

_NOTES_SQL = "SELECT ZCONTACT, ZTEXT FROM ZABCDNOTE WHERE ZTEXT IS NOT NULL AND ZTEXT != ''"

_RECORDS_SQL = (
    "SELECT Z_PK, COALESCE(ZUNIQUEID,''), COALESCE(ZFIRSTNAME,''), "
    "COALESCE(ZLASTNAME,''), COALESCE(ZMIDDLENAME,''), "
    "COALESCE(ZNICKNAME,''), COALESCE(ZORGANIZATION,''), "
    "COALESCE(ZJOBTITLE,''), COALESCE(ZDEPARTMENT,''), "
    "ZBIRTHDAY, ZCREATIONDATE, ZMODIFICATIONDATE "
    "FROM ZABCDRECORD"
)


def _clean_label(raw: str) -> str:
    """Strip the CoreData label wrapper. Stored labels look like
    `_$!<Mobile>!$_` or `_$!<Work>!$_`. Return the inner word or the
    original if it's already plain ('iPhone', 'móvil')."""
    if not raw:
        return ""
    s = raw.strip()
    # `_$!<X>!$_` → `X`
    if s.startswith("_$!<") and s.endswith(">!$_"):
        return s[4:-4].strip().lower()
    return s.strip().lower()


def read_contacts_from_db(db_path: Path, source_id: str) -> list[Contact]:
    """Read every contact from a single `AddressBook-v22.abcddb` file."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # Build side-tables keyed by ZOWNER = ZABCDRECORD.Z_PK.
        phones_by_owner: dict[int, list[Phone]] = {}
        for owner, full, label in conn.execute(_PHONES_SQL):
            if owner is None:
                continue
            phones_by_owner.setdefault(int(owner), []).append(Phone(
                raw=str(full).strip(),
                label=_clean_label(label),
                digits=normalise_phone(full),
            ))

        emails_by_owner: dict[int, list[Email]] = {}
        for owner, addr, label in conn.execute(_EMAILS_SQL):
            if owner is None:
                continue
            emails_by_owner.setdefault(int(owner), []).append(Email(
                address=str(addr).strip(),
                label=_clean_label(label),
            ))

        notes_by_owner: dict[int, str] = {}
        try:
            for owner, text in conn.execute(_NOTES_SQL):
                if owner is None or not text:
                    continue
                notes_by_owner[int(owner)] = str(text).strip()
        except sqlite3.OperationalError:
            # Older schemas may not have ZABCDNOTE with ZTEXT column.
            # Silent-skip — notes aren't required, they're bonus signal.
            pass

        out: list[Contact] = []
        for row in conn.execute(_RECORDS_SQL):
            (pk, uid, first, last, middle, nick, org, job, dept,
             bday_raw, created_raw, modified_raw) = row
            if not uid:
                # Records without ZUNIQUEID are Core Data internals
                # (groups, link-placeholders). Skip — we can't stable-id them.
                continue
            c = Contact(
                uid=str(uid),
                source_id=source_id,
                first_name=str(first).strip(),
                last_name=str(last).strip(),
                middle_name=str(middle).strip(),
                nickname=str(nick).strip(),
                organization=str(org).strip(),
                job_title=str(job).strip(),
                department=str(dept).strip(),
                note=notes_by_owner.get(int(pk), ""),
                phones=tuple(phones_by_owner.get(int(pk), [])),
                emails=tuple(emails_by_owner.get(int(pk), [])),
                birthday_ts=_cocoa_to_unix(bday_raw),
                created_ts=_cocoa_to_unix(created_raw),
                modified_ts=_cocoa_to_unix(modified_raw),
            )
            if c.is_empty():
                continue
            out.append(c)
        return out
    finally:
        conn.close()


def _default_fetch(root: Path = DEFAULT_ABOOK_ROOT) -> list[Contact]:
    """Union all contacts from every AddressBook source. On UID collision
    across sources (rare — linked-card behaviour) keeps the first one
    seen (iteration order is sorted by source_id, so deterministic)."""
    seen_uids: set[str] = set()
    out: list[Contact] = []
    for source_id, db_path in _find_abcddb_files(root):
        try:
            for c in read_contacts_from_db(db_path, source_id):
                if c.uid in seen_uids:
                    continue
                seen_uids.add(c.uid)
                out.append(c)
        except sqlite3.DatabaseError as exc:
            # Defensive — don't abort the whole ingest if one DB is
            # corrupt / locked / upgraded-to-newer-schema.
            print(f"[contacts] skip {db_path}: {exc}", file=sys.stderr)
            continue
    return out


# ── State ──────────────────────────────────────────────────────────────────

_STATE_TABLE_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_contacts_state ("
    " contact_uid TEXT PRIMARY KEY,"
    " content_hash TEXT NOT NULL,"
    " last_seen_ts TEXT NOT NULL,"
    " updated_at TEXT NOT NULL"
    ")"
)


def _ensure_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(_STATE_TABLE_DDL)


def _load_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    cur = conn.execute("SELECT contact_uid, content_hash FROM rag_contacts_state")
    return {row[0]: row[1] for row in cur.fetchall()}


def _upsert_hash(conn: sqlite3.Connection, uid: str, h: str, now_iso: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO rag_contacts_state "
        "(contact_uid, content_hash, last_seen_ts, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (uid, h, now_iso, now_iso),
    )


def _delete_hash(conn: sqlite3.Connection, uid: str) -> None:
    conn.execute("DELETE FROM rag_contacts_state WHERE contact_uid = ?", (uid,))


def _reset_state(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM rag_contacts_state")


def _content_hash(c: Contact) -> str:
    """Stable hash over fields that change the chunk body or the
    phone-lookup index. Excludes `modified_ts` (noisy) and `source_id`
    (same contact across sources shouldn't force reindex)."""
    phones_part = "|".join(f"{p.raw}:{p.label}" for p in c.phones)
    emails_part = "|".join(f"{e.address}:{e.label}" for e in c.emails)
    payload = "\n".join([
        c.uid, c.first_name, c.last_name, c.middle_name, c.nickname,
        c.organization, c.job_title, c.department, c.note,
        phones_part, emails_part,
        f"{c.birthday_ts:.0f}",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── Chunker ────────────────────────────────────────────────────────────────

def _format_birthday(bday_ts: float) -> str:
    """Human birthday string. Apple stores YEARLESS birthdays as year 0001
    — emit only `MM-DD` in that case so the display doesn't say '0001'."""
    if not bday_ts:
        return ""
    dt = datetime.fromtimestamp(bday_ts, tz=timezone.utc)
    if dt.year <= 1:
        return dt.strftime("%m-%d")
    return dt.strftime("%Y-%m-%d")


def _format_contact_body(c: Contact) -> str:
    """Human-readable transcript used as display_text for retrieval."""
    parts: list[str] = [f"Contacto: {c.display_name}"]
    if c.organization:
        org_line = c.organization
        if c.job_title:
            org_line = f"{c.job_title} — {org_line}"
        if c.department:
            org_line += f" / {c.department}"
        parts.append(f"Organización: {org_line}")
    elif c.job_title:
        parts.append(f"Puesto: {c.job_title}")
    if c.nickname and c.nickname != c.first_name:
        parts.append(f"Apodo: {c.nickname}")
    if c.phones:
        for p in c.phones:
            label = f" ({p.label})" if p.label else ""
            parts.append(f"Teléfono{label}: {p.raw}")
    if c.emails:
        for e in c.emails:
            label = f" ({e.label})" if e.label else ""
            parts.append(f"Email{label}: {e.address}")
    bday = _format_birthday(c.birthday_ts)
    if bday:
        parts.append(f"Cumpleaños: {bday}")
    if c.note:
        parts.append("---")
        parts.append(c.note)
    body = "\n".join(parts)
    if len(body) > CHUNK_MAX_CHARS:
        body = body[:CHUNK_MAX_CHARS].rstrip()
    return body


def _embed_prefix(c: Contact, body: str) -> str:
    return (
        f"[source=contacts | name={c.display_name}"
        f"{' | org=' + c.organization if c.organization else ''}] {body}"
    )


def _contact_doc_id(c: Contact) -> str:
    return f"{DOC_ID_PREFIX}://{c.uid}::0"


def _contact_file_key(c: Contact) -> str:
    return f"{DOC_ID_PREFIX}://{c.uid}"


# ── Writer ─────────────────────────────────────────────────────────────────

def upsert_contacts(col, contacts: list[Contact]) -> int:
    if not contacts:
        return 0
    bodies = [_format_contact_body(c) for c in contacts]
    embed_texts = [_embed_prefix(c, b) for c, b in zip(contacts, bodies)]
    embeddings = rag.embed(embed_texts)

    for c in contacts:
        key = _contact_file_key(c)
        try:
            existing = col.get(where={"file": key}, include=[])
            if existing.get("ids"):
                col.delete(ids=existing["ids"])
        except Exception:
            pass

    ids = [_contact_doc_id(c) for c in contacts]
    metas: list[dict] = []
    for c, body in zip(contacts, bodies):
        metas.append({
            "file": _contact_file_key(c),
            "note": f"Contacto: {c.display_name}",
            "folder": "",
            "tags": "",
            "hash": _content_hash(c),
            "outlinks": "",
            "source": "contacts",
            "created_ts": c.anchor_ts,
            "contact_uid": c.uid,
            "source_id": c.source_id,
            "title": c.display_name,
            "organization": c.organization,
            "primary_phone": c.phones[0].raw if c.phones else "",
            "primary_email": c.emails[0].address if c.emails else "",
            "birthday_ts": c.birthday_ts,
            "creation_ts": c.created_ts,
            "modification_ts": c.modified_ts,
            "parent": body,
        })
    col.add(ids=ids, embeddings=embeddings, documents=bodies, metadatas=metas)
    rag._extract_and_index_entities_for_chunks(bodies, ids, metas, "contacts")
    return len(contacts)


def delete_contacts(col, uids: list[str]) -> int:
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


# ── Phone lookup (exported for other ingesters) ────────────────────────────

# Module-level cache. Rebuilt on first call; invalidated when
# `invalidate_phone_index()` is called by `run()` after a successful
# ingest. Safe to use across threads — the dict is read-mostly after
# build and Python atomic dict.get is fine.
_phone_index: dict[str, str] | None = None  # digit-key → contact UID
_contact_by_uid: dict[str, Contact] | None = None


def invalidate_phone_index() -> None:
    global _phone_index, _contact_by_uid
    _phone_index = None
    _contact_by_uid = None


def _build_phone_index(contacts: Iterable[Contact]) -> tuple[dict[str, str], dict[str, Contact]]:
    """Build the digit-key → UID map. Two-pass:

    Pass 1 — **phone-level dedup**. iCloud linked cards produce the
    same phone on multiple `ZUNIQUEID`s (one per AddressBook source).
    Group by exact `phone.digits` and pick a canonical UID per number.
    "Canonical" = contact with the longest `display_name`, tiebreak
    lexicographic on UID (stable, deterministic across runs).

    Pass 2 — **suffix fan-out**. For each `(digits → canonical_uid)`,
    emit suffix keys. Drop a key ONLY when it collides across
    genuinely different underlying numbers mapped to different
    canonical UIDs (e.g. an AR mobile and a US landline sharing the
    last-7 digits — that's a real ambiguity, better return None than
    mis-attribute).

    Before this two-pass setup the resolver fell back to "key appears
    on different UIDs → drop" which exploded on contacts with dual-
    source iCloud (measured on the host: 90/580 keys dropped, dropping
    real-world resolve rate to 3%)."""
    by_uid: dict[str, Contact] = {c.uid: c for c in contacts}

    # Pass 1: phone number → canonical UID.
    phone_to_uid: dict[str, str] = {}
    for c in contacts:
        for p in c.phones:
            if not p.digits:
                continue
            existing = phone_to_uid.get(p.digits)
            if existing is None:
                phone_to_uid[p.digits] = c.uid
                continue
            if existing == c.uid:
                continue
            existing_c = by_uid[existing]
            # Prefer the contact with the longer display name; tiebreak
            # by lexicographically smaller UID for determinism.
            if (len(c.display_name) > len(existing_c.display_name) or
                (len(c.display_name) == len(existing_c.display_name)
                 and c.uid < existing)):
                phone_to_uid[p.digits] = c.uid

    # Pass 2: suffix fan-out with per-key conflict tracking.
    index: dict[str, str] = {}
    conflict_keys: set[str] = set()
    for digits, uid in phone_to_uid.items():
        for key in _phone_match_keys(digits):
            if key in conflict_keys:
                continue
            prev = index.get(key)
            if prev is None:
                index[key] = uid
            elif prev != uid:
                # Genuine ambiguity — two different canonical UIDs
                # share this suffix. Drop so lookups return None.
                conflict_keys.add(key)
                index.pop(key, None)
    return index, by_uid


def resolve_phone(number: str, *, root: Path = DEFAULT_ABOOK_ROOT) -> Contact | None:
    """Resolve a raw phone number to a Contact, or None if unknown /
    ambiguous. Builds the lookup index lazily on first call; cached
    at module level until `invalidate_phone_index()` is called (ingest
    does this after every successful run).

    Matches progressively — exact digits first, then last-10 / last-8 /
    last-7. Ambiguous suffixes (same last-N digits across multiple
    contacts) are dropped at index-build time, so a match here is
    trustworthy.
    """
    global _phone_index, _contact_by_uid
    if _phone_index is None or _contact_by_uid is None:
        contacts = _default_fetch(root)
        _phone_index, _contact_by_uid = _build_phone_index(contacts)
    digits = normalise_phone(number)
    if not digits:
        return None
    for key in _phone_match_keys(digits):
        uid = _phone_index.get(key)
        if uid:
            return _contact_by_uid.get(uid)
    return None


# ── Orchestration ──────────────────────────────────────────────────────────

def run(
    *,
    reset: bool = False,
    dry_run: bool = False,
    root: Path | None = None,
    vault_col=None,
    fetch_fn: Callable[[], list[Contact]] | None = None,
    now: datetime | None = None,
) -> dict:
    """Ingest contacts from all AddressBook sources. Returns summary dict."""
    t0 = time.perf_counter()
    summary: dict = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "reset": bool(reset),
        "dry_run": bool(dry_run),
        "contacts_fetched": 0,
        "contacts_indexed": 0,
        "contacts_unchanged": 0,
        "contacts_deleted": 0,
        "sources_scanned": 0,
        "duration_s": 0.0,
    }

    ab_root = Path(root) if root is not None else DEFAULT_ABOOK_ROOT
    fetch = fetch_fn if fetch_fn is not None else (lambda: _default_fetch(ab_root))
    contacts = fetch() or []
    summary["contacts_fetched"] = len(contacts)
    summary["sources_scanned"] = len({c.source_id for c in contacts})

    if not contacts and ab_root == DEFAULT_ABOOK_ROOT and not ab_root.exists():
        summary["error"] = f"AddressBook no encontrado en {ab_root}"
        summary["duration_s"] = round(time.perf_counter() - t0, 2)
        return summary

    col = vault_col if vault_col is not None else rag.get_db()

    state_conn = sqlite3.connect(str(rag.DB_PATH / "ragvec.db"))
    _ensure_state_table(state_conn)
    if reset:
        _reset_state(state_conn)
    state_conn.commit()

    prior_hashes = _load_hashes(state_conn)
    live_uids = {c.uid for c in contacts}

    to_write: list[Contact] = []
    for c in contacts:
        h = _content_hash(c)
        if prior_hashes.get(c.uid) != h:
            to_write.append(c)
        else:
            summary["contacts_unchanged"] += 1

    stale_uids = [uid for uid in prior_hashes if uid not in live_uids]
    now_iso = (now or datetime.now()).isoformat(timespec="seconds")

    if not dry_run:
        summary["contacts_indexed"] = upsert_contacts(col, to_write)
        for c in to_write:
            _upsert_hash(state_conn, c.uid, _content_hash(c), now_iso)
        summary["contacts_deleted"] = delete_contacts(col, stale_uids)
        for uid in stale_uids:
            _delete_hash(state_conn, uid)
        state_conn.commit()
        # Rebuild phone-lookup cache so downstream ingesters in the
        # same process see the fresh data.
        invalidate_phone_index()
    else:
        summary["contacts_indexed"] = len(to_write)
        summary["contacts_deleted"] = len(stale_uids)

    state_conn.close()
    summary["duration_s"] = round(time.perf_counter() - t0, 2)
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reset", action="store_true",
                    help="Wipe state DB + re-embed every contact")
    ap.add_argument("--dry-run", action="store_true",
                    help="Count fetched/changed but skip writes")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--root", type=str, default=None,
                    help=f"Override AddressBook root (default: {DEFAULT_ABOOK_ROOT})")
    args = ap.parse_args()

    summary = run(
        reset=bool(args.reset),
        dry_run=bool(args.dry_run),
        root=Path(args.root) if args.root else None,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if "error" in summary:
        print(f"[error] {summary['error']}", file=sys.stderr)
        sys.exit(1)
    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}{summary['contacts_fetched']} contactos · "
        f"{summary['contacts_indexed']} indexados · "
        f"{summary['contacts_unchanged']} sin cambios · "
        f"{summary['contacts_deleted']} borrados · "
        f"{summary['sources_scanned']} fuentes · "
        f"{summary['duration_s']}s"
    )


if __name__ == "__main__":
    main()
