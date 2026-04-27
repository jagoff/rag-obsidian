"""Google Drive cross-source ingester — Phase 1.g (2026-04-24).

Indexa Google Docs / Sheets / Slides del user directo en la colección
vectorial, con ``source="drive"``. NO escribe archivos al vault — el
workflow del usuario (Obsidian + iCloud) movía periódicamente
``03-Resources/GoogleDrive/`` a ``.trash/`` (5 copias observadas del
19-24 de abril 2026), haciendo el approach file-based insostenible.

Design:
  - Reader: ``drive.files().list()`` con
    ``q="modifiedTime > '<iso>' and (mime = doc|sheet|slides) and trashed=false"``.
    Pagina. Cap de 500 archivos por run.
  - Chunking por-doc: ~800 chars con overlap 80, split por párrafos cuando
    se puede, hard-split si un párrafo único excede el target. Doc largos
    (hasta 128_000 chars del body-cap) generan múltiples chunks
    independientes — el retrieval va a pescar el chunk relevante en vez
    de intentar matchear un bloque de 50k.
  - doc_id: ``gdrive://file/<file_id>#chunk=<NNNN>`` (4-digit padded
    idx). Idempotent: la re-ingest del mismo file borra sus chunks
    previos antes de re-upsertar.
  - Embed prefix: ``[source=drive | kind=<doc|sheet|slides> | title=<name> |
    owner=<name>] <body>`` (mismo shape que gmail / whatsapp).
  - Recency: ``SOURCE_RECENCY_HALFLIFE_DAYS["drive"] = 90`` (entre email
    180d y chat 30d).
  - Retention: ``SOURCE_RETENTION_DAYS["drive"] = 365`` (un año, como
    email). El ingester borra chunks con ``created_ts`` más viejos que
    el cutoff cada run.
  - Source weight: ``SOURCE_WEIGHTS["drive"] = 0.85`` (confianza editorial
    alta, idem email).

Incremental sync:
  - Cursor en ``rag_gdrive_state(account_id, last_modified_seen,
    updated_at)`` — single-row.
  - Bootstrap (cursor vacío): ventana de N días (default 30) en
    ``modifiedTime``.
  - Incremental: listar ``modifiedTime > last_modified_seen``, orden
    asc para que el cursor avance monotonic.

Opt-out: ninguno por ahora (mismo criterio de "index everything" que
gmail + whatsapp). Los PDFs / xlsx / docx subidos no Google-native NO
se tocan — requieren deps extra (pdftotext, openpyxl, python-docx).

Invocación::
    rag index --source drive [--reset] [--dry-run] [--days 30]
    .venv/bin/python scripts/ingest_gdrive.py --days 90 --json
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


# ── Config ──────────────────────────────────────────────────────────────────

DOC_ID_PREFIX = "gdrive"
DEFAULT_DAYS = 30
DEFAULT_MAX_FILES = 500
DEFAULT_BODY_CAP = 128_000
CHUNK_TARGET = 800
CHUNK_OVERLAP = 80
PAGE_SIZE = 100
STATE_DB_FILE = "ragvec.db"

EXPORT_MIME = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

MIME_LABEL = {
    "application/vnd.google-apps.document": "doc",
    "application/vnd.google-apps.spreadsheet": "sheet",
    "application/vnd.google-apps.presentation": "slides",
}


# ── State table (cursor) ────────────────────────────────────────────────────

_STATE_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_gdrive_state ("
    " account_id TEXT PRIMARY KEY,"
    " last_modified_seen TEXT,"
    " updated_at TEXT NOT NULL"
    ")"
)


def _ensure_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(_STATE_DDL)


def _load_cursor(conn: sqlite3.Connection, account_id: str) -> str | None:
    row = conn.execute(
        "SELECT last_modified_seen FROM rag_gdrive_state WHERE account_id = ?",
        (account_id,),
    ).fetchone()
    return row[0] if row and row[0] else None


def _save_cursor(conn: sqlite3.Connection, account_id: str, last_iso: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO rag_gdrive_state "
        "(account_id, last_modified_seen, updated_at) VALUES (?, ?, ?)",
        (account_id, last_iso, datetime.now().isoformat(timespec="seconds")),
    )


def _reset_cursor(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM rag_gdrive_state")


# ── Chunking ────────────────────────────────────────────────────────────────

_PARA_RE = re.compile(r"\n\s*\n+")


def _chunk_body(text: str, target: int = CHUNK_TARGET, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split ``text`` into chunks ~target chars long at paragraph boundaries.

    - Short bodies (<= target) → single chunk.
    - Paragraph-level split via blank-line regex; concatenates short paragraphs
      until the target is reached, then emits.
    - Overlap: last ``overlap`` chars of the previous chunk seed the next one
      (context continuity across splits).
    - Hard-split fallback for single paragraphs bigger than target.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= target:
        return [text]

    paragraphs = _PARA_RE.split(text)
    chunks: list[str] = []
    buf = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # If this paragraph alone exceeds target, flush buf then hard-split.
        if len(p) > target:
            if buf:
                chunks.append(buf)
                buf = ""
            start = 0
            step = max(1, target - overlap)
            while start < len(p):
                chunks.append(p[start:start + target])
                start += step
            continue
        # Normal case: append to buf if it still fits.
        tentative = (buf + "\n\n" + p) if buf else p
        if len(tentative) <= target:
            buf = tentative
        else:
            if buf:
                chunks.append(buf)
            # Seed next chunk with overlap tail for continuity.
            seed = buf[-overlap:] if (buf and overlap > 0) else ""
            buf = (seed + "\n\n" + p) if seed else p

    if buf:
        chunks.append(buf)
    return chunks


# ── Drive API helpers ───────────────────────────────────────────────────────


def _build_service():
    """Returns (service, reason). service is None when creds/deps missing.

    To handle timeouts on .export() calls, callers should wrap them in
    try/except and catch socket.timeout or httplib2.RedirectMissingLocation.
    The timeout is enforced at the HTTP library level when the request hangs.
    """
    creds = rag._load_google_credentials()
    if creds is None:
        return None, "no_google_credentials"
    try:
        from googleapiclient.discovery import build
    except ImportError:
        return None, "google_api_missing"
    try:
        svc = build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as exc:
        return None, f"build_failed: {str(exc)[:120]}"
    return svc, ""


def _list_since(svc, *, since_iso: str, mime_types: list[str], max_files: int) -> list[dict]:
    """Paginated files.list starting from ``since_iso`` exclusive, asc order
    so the cursor advances monotonically."""
    mime_filter = " or ".join(f"mimeType = '{m}'" for m in mime_types)
    q = f"(modifiedTime > '{since_iso}') and ({mime_filter}) and trashed = false"
    fields = (
        "nextPageToken, "
        "files(id, name, mimeType, modifiedTime, createdTime, "
        "owners(displayName, emailAddress), webViewLink, size)"
    )
    out: list[dict] = []
    page_token: str | None = None
    while True:
        if len(out) >= max_files:
            return out[:max_files]
        resp = svc.files().list(
            q=q,
            orderBy="modifiedTime asc",
            pageSize=min(PAGE_SIZE, max_files - len(out)),
            fields=fields,
            pageToken=page_token,
        ).execute()
        out.extend(resp.get("files") or [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            return out[:max_files]


class _ExportTimeout(Exception):
    """Raised when an export operation times out."""
    pass


def _execute_with_timeout(request, timeout_secs: int = 30) -> object:
    """Execute a googleapiclient request with a timeout.

    Wraps the .execute() call with a signal-based timeout (Unix-only).
    Falls back to direct .execute() on Windows or if signal handling fails.
    """
    if sys.platform == "win32":
        # Signal alarms don't work on Windows; fall back to unprotected execute.
        return request.execute()

    def timeout_handler(signum, frame):
        raise _ExportTimeout()

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_secs)
    try:
        return request.execute()
    except _ExportTimeout:
        raise TimeoutError(f"export timed out after {timeout_secs}s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _export_body(svc, file_id: str, mime: str, body_cap: int, file_meta: dict | None = None) -> tuple[str, str]:
    """Export the document body as text (docs/slides) or CSV (sheets).
    Returns (body, err); err is empty string on success.

    Skips files larger than 50MB to avoid lengthy timeouts on exports.
    """
    export_mime = EXPORT_MIME.get(mime)
    if not export_mime:
        return "", f"unsupported_mime:{mime}"

    # Skip very large files (applies to binary exports, not Google Docs).
    if file_meta and mime not in EXPORT_MIME:  # Non-Google-native format
        file_size = int((file_meta.get("size") or "0") or "0")
        if file_size > 50_000_000:  # 50 MB
            return "", "file_too_large:>50MB"

    try:
        req = svc.files().export(fileId=file_id, mimeType=export_mime)
        raw = _execute_with_timeout(req, timeout_secs=30)
    except Exception as exc:
        exc_str = str(exc)
        # Timeout errors show up as socket errors, TimeoutError, or "timed out" in message.
        if isinstance(exc, TimeoutError) or \
           "timeout" in exc_str.lower() or "timed out" in exc_str.lower():
            return "", "export_timeout:>30s"
        return "", f"export_failed:{exc_str[:100]}"
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    raw = raw.strip()
    if len(raw) > body_cap:
        raw = raw[:body_cap] + f"\n\n<!-- truncated at {body_cap} chars -->"
    return raw, ""


# ── Identifiers ─────────────────────────────────────────────────────────────


def _file_key(file_id: str) -> str:
    """Logical identity of the Drive file — used as ``file`` meta so we can
    find all chunks of a single doc for delete+re-add on update."""
    return f"{DOC_ID_PREFIX}://file/{file_id}"


def _chunk_doc_id(file_id: str, idx: int) -> str:
    return f"{DOC_ID_PREFIX}://file/{file_id}#chunk={idx:04d}"


# ── Embed + upsert ──────────────────────────────────────────────────────────


def _embed_prefix(meta: dict, chunk_body: str) -> str:
    mime_label = MIME_LABEL.get(meta["mimeType"], "file")
    owner = (meta.get("owners") or [{}])[0].get("displayName", "?")
    return f"[source=drive | kind={mime_label} | title={meta['name']} | owner={owner}] {chunk_body}"


def _parse_mod_ts(mod_iso: str) -> float:
    """Parse RFC3339 modifiedTime → epoch seconds. Returns 0.0 on parse
    failure (retention cleanup treats 0 as 'very old', but we never set 0
    for real chunks because Drive always returns modifiedTime)."""
    if not mod_iso:
        return 0.0
    try:
        return datetime.fromisoformat(mod_iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def upsert_drive_file(col, meta: dict, body: str) -> int:
    """Chunk + embed + upsert one Drive file. Idempotent: pre-deletes any
    chunks under the same file_key before re-adding. Returns chunks written
    (0 if body is empty AND no prior chunks existed)."""
    file_id = meta["id"]
    file_key = _file_key(file_id)
    chunks = _chunk_body(body)

    # Delete existing chunks for this file (re-ingest = delete + re-add).
    try:
        existing = col.get(where={"file": file_key}, include=[])
        if existing.get("ids"):
            col.delete(ids=existing["ids"])
    except Exception:
        pass

    if not chunks:
        # Empty doc: nothing to upsert. We already deleted any previous chunks,
        # so a doc that was previously indexed and is now empty ends up cleanly
        # removed from the corpus.
        return 0

    mime_label = MIME_LABEL.get(meta["mimeType"], "file")
    owners = meta.get("owners") or []
    owner_name = (owners[0].get("displayName") if owners else None) or "?"
    owner_email = (owners[0].get("emailAddress") if owners else None) or ""
    mod_iso = meta.get("modifiedTime", "") or ""
    mod_ts = _parse_mod_ts(mod_iso)
    link = meta.get("webViewLink") or ""

    parent_text = (
        f"{meta.get('name', '?')} — {mime_label} por {owner_name}\n"
        f"Drive link: {link}\n\n{body[:1200]}"
    )

    embed_texts = [_embed_prefix(meta, c) for c in chunks]
    embeddings = rag.embed(embed_texts)
    ids = [_chunk_doc_id(file_id, i) for i in range(len(chunks))]
    metas: list[dict] = []
    for i, chunk in enumerate(chunks):
        metas.append({
            "file": file_key,
            "note": (meta.get("name") or "(sin nombre)")[:80],
            "folder": f"drive/{mime_label}",
            "tags": "google-drive,gdrive-archive",
            "hash": "",
            "outlinks": "",
            "source": "drive",
            "created_ts": mod_ts,
            "drive_id": file_id,
            "mime": mime_label,
            "modified_iso": mod_iso,
            "owner": owner_name,
            "owner_email": owner_email,
            "link": link,
            "chunk_idx": i,
            "total_chunks": len(chunks),
            "parent": parent_text,
        })

    col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metas)
    # Entity extraction — same hook as gmail ingester (silent-fails if gliner absent).
    try:
        rag._extract_and_index_entities_for_chunks(chunks, ids, metas, "drive")
    except Exception:
        pass
    return len(chunks)


def _retention_prune(col, *, now_ts: float | None = None) -> int:
    """Drop chunks with source='drive' whose created_ts is older than the
    retention cutoff. Returns chunks deleted."""
    retention_days = rag.SOURCE_RETENTION_DAYS.get("drive")
    if retention_days is None:
        return 0
    cutoff = (now_ts or time.time()) - (retention_days * 86400.0)
    try:
        got = col.get(where={"source": "drive"}, include=["metadatas"])
    except Exception:
        return 0
    stale_ids: list[str] = []
    for mid, meta in zip(got.get("ids", []) or [], got.get("metadatas", []) or []):
        ts = float((meta or {}).get("created_ts") or 0.0)
        if ts and ts < cutoff:
            stale_ids.append(mid)
    if not stale_ids:
        return 0
    try:
        col.delete(ids=stale_ids)
    except Exception:
        return 0
    return len(stale_ids)


# ── Orchestration ───────────────────────────────────────────────────────────


def run(
    *,
    reset: bool = False,
    days: int = DEFAULT_DAYS,
    max_files: int = DEFAULT_MAX_FILES,
    body_cap: int = DEFAULT_BODY_CAP,
    dry_run: bool = False,
    svc=None,
    col=None,
    state_conn=None,
) -> dict:
    """Main entry. Returns a summary dict consumed by `rag index --source drive`
    and `python scripts/ingest_gdrive.py`."""
    t0 = time.perf_counter()
    summary: dict = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "reset": bool(reset),
        "dry_run": bool(dry_run),
        "files_seen": 0,
        "files_indexed": 0,
        "chunks_written": 0,
        "files_failed": 0,
        "retention_deleted": 0,
        "errors": [],
        "bootstrapped": False,
        "days": days,
        "duration_s": 0.0,
    }

    if svc is None:
        built_svc, reason = _build_service()
        if built_svc is None:
            summary["error"] = reason
            summary["duration_s"] = round(time.perf_counter() - t0, 2)
            return summary
        svc = built_svc

    account_id = "me"  # single-user for now; multi-account is a Phase 2 concern
    if col is None:
        col = rag.get_db()

    state_conn_local = state_conn
    opened_state = False
    if state_conn_local is None:
        state_conn_local = sqlite3.connect(str(rag.DB_PATH / STATE_DB_FILE))
        opened_state = True
    try:
        _ensure_state_table(state_conn_local)
        if reset:
            _reset_cursor(state_conn_local)
            state_conn_local.commit()

        stored = _load_cursor(state_conn_local, account_id)
        mode_bootstrap = stored is None
        summary["bootstrapped"] = mode_bootstrap
        if mode_bootstrap:
            since_iso = (
                datetime.now(timezone.utc) - timedelta(days=days)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            since_iso = stored

        try:
            files = _list_since(
                svc, since_iso=since_iso,
                mime_types=list(EXPORT_MIME.keys()),
                max_files=max_files,
            )
        except Exception as exc:
            summary["error"] = f"list_failed: {str(exc)[:120]}"
            summary["duration_s"] = round(time.perf_counter() - t0, 2)
            return summary

        summary["files_seen"] = len(files)
        latest_mod = since_iso

        for f in files:
            if dry_run:
                summary["files_indexed"] += 1
                if (f.get("modifiedTime") or "") > latest_mod:
                    latest_mod = f["modifiedTime"]
                continue
            body, err = _export_body(svc, f["id"], f["mimeType"], body_cap, file_meta=f)
            if err:
                summary["files_failed"] += 1
                summary["errors"].append({
                    "id": f.get("id", "?"),
                    "name": f.get("name", "?"),
                    "err": err,
                })
                continue
            try:
                written = upsert_drive_file(col, f, body)
                summary["chunks_written"] += written
                summary["files_indexed"] += 1
            except Exception as exc:
                summary["files_failed"] += 1
                summary["errors"].append({
                    "id": f.get("id", "?"),
                    "name": f.get("name", "?"),
                    "err": f"upsert_failed:{str(exc)[:80]}",
                })
                continue
            if (f.get("modifiedTime") or "") > latest_mod:
                latest_mod = f["modifiedTime"]

        # Retention: drop old chunks. Skip on dry-run.
        if not dry_run:
            summary["retention_deleted"] = _retention_prune(col)

            # Save cursor only on real runs and only if we processed at least one file.
            # Empty runs keep the previous cursor so next run retries the same window.
            if summary["files_seen"] > 0:
                _save_cursor(state_conn_local, account_id, latest_mod)
                state_conn_local.commit()

        summary["duration_s"] = round(time.perf_counter() - t0, 2)
        return summary
    finally:
        if opened_state:
            state_conn_local.close()


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reset", action="store_true",
                    help="limpiar el cursor y re-indexar desde 0 (ventana --days)")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS,
                    help=f"ventana de bootstrap en días (default {DEFAULT_DAYS})")
    ap.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES,
                    help=f"tope de archivos por run (default {DEFAULT_MAX_FILES})")
    ap.add_argument("--body-cap", type=int, default=DEFAULT_BODY_CAP,
                    help=f"chars máx por doc antes de truncar (default {DEFAULT_BODY_CAP})")
    ap.add_argument("--dry-run", action="store_true",
                    help="listar sin exportar ni upsertar (no avanza cursor)")
    ap.add_argument("--json", action="store_true",
                    help="imprimir summary como JSON")
    args = ap.parse_args()

    summary = run(
        reset=bool(args.reset),
        days=args.days,
        max_files=args.max_files,
        body_cap=args.body_cap,
        dry_run=bool(args.dry_run),
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return

    if "error" in summary:
        print(f"[error] {summary['error']}")
        sys.exit(1)

    mode = "bootstrap" if summary["bootstrapped"] else "incremental"
    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}Drive ({mode}): "
        f"{summary['files_seen']} archivos · "
        f"+{summary['files_indexed']} indexados · "
        f"{summary['chunks_written']} chunks · "
        f"-{summary['retention_deleted']} retención · "
        f"{summary['files_failed']} fallaron · "
        f"{summary['duration_s']}s"
    )
    if summary["errors"]:
        print(f"  {len(summary['errors'])} errores (primeros 5):")
        for e in summary["errors"][:5]:
            print(f"    - {e['name']} ({e['id'][:8]}): {e['err']}")


if __name__ == "__main__":
    main()
