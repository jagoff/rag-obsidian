"""Google Drive monthly backfill ingester.

Pulls Google Docs / Sheets / Slides modified in the last N days (default
30) from the user's Drive and writes ONE markdown per doc under::

    <VAULT>/03-Resources/GoogleDrive/archive/YYYY-MM-DD__<slug>__<id8>.md

Each file carries YAML frontmatter (source, drive_id, mime, modified,
owner, link) + the exported body. Existing files are skipped when the
content is unchanged (`rag._atomic_write_if_changed`), so re-runs are
cheap.

This complements the daily ``_sync_gdrive_notes`` (which writes a
single summary with ~4 docs modified in the last 48h). The backfill
gives ``rag_query`` / the reranker a real view of the user's full
recent Drive activity instead of relying on the live
``drive_search`` tool (which hits the Drive API on every query,
~1-3s of latency).

After writing the markdown files, we run ``rag index`` so the new
files enter the semantic corpus immediately.

Known caveat (Obsidian cleanup): the user's workflow occasionally moves
``03-Resources/GoogleDrive/`` to ``.trash/`` (5 copies observed between
Apr 19-24, 2026). If that happens the archive folder vanishes and the
chunks become orphans on the next index. Re-run this script to rebuild.

Invoked as::

    .venv/bin/python scripts/ingest_gdrive.py                # 30d, reindex
    .venv/bin/python scripts/ingest_gdrive.py --days 90      # 90d window
    .venv/bin/python scripts/ingest_gdrive.py --dry-run      # listar sin bajar
    .venv/bin/python scripts/ingest_gdrive.py --no-reindex   # solo escribir
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


# ── Config ──────────────────────────────────────────────────────────────────

DEFAULT_DAYS = 30
DEFAULT_MAX_FILES = 500          # hard safety cap across the run
DEFAULT_BODY_CAP = 128_000       # chars per doc (avoids pathological 10MB sheets)
PAGE_SIZE = 100                  # Drive API max per page for our field set
ARCHIVE_SUBPATH = "03-Resources/GoogleDrive/archive"

# mimeType → exportable text mime. Three Google-native types only; uploaded
# PDFs / xlsx / docx need separate tooling (pdftotext, openpyxl, python-docx)
# that we don't depend on yet — kept out of scope for this backfill.
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


# ── Helpers ─────────────────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_WS_RE = re.compile(r"[\s_-]+")


def _slugify(name: str, max_len: int = 60) -> str:
    """Filename-safe slug. Lowercase, dash-separated, capped in length.
    Keeps unicode letters/digits so Spanish-named docs survive legibly."""
    name = name.strip().lower()
    name = _SLUG_RE.sub("", name)
    name = _WS_RE.sub("-", name).strip("-")
    if not name:
        name = "untitled"
    return name[:max_len].rstrip("-") or "untitled"


def _build_drive_service():
    """Returns (service, reason). service is None when creds / deps missing."""
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


def _list_recent(svc, *, days: int, mime_types: list[str], max_files: int) -> list[dict]:
    """Paginated files.list over modifiedTime window. Returns up to
    max_files raw file metadata dicts."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    mime_filter = " or ".join(f"mimeType = '{m}'" for m in mime_types)
    q = f"(modifiedTime > '{cutoff}') and ({mime_filter}) and trashed = false"
    fields = (
        "nextPageToken, "
        "files(id, name, mimeType, modifiedTime, createdTime, "
        "owners(displayName, emailAddress), webViewLink, size, parents)"
    )
    out: list[dict] = []
    page_token: str | None = None
    while True:
        resp = svc.files().list(
            q=q,
            orderBy="modifiedTime desc",
            pageSize=min(PAGE_SIZE, max_files - len(out)),
            fields=fields,
            pageToken=page_token,
        ).execute()
        batch = resp.get("files") or []
        out.extend(batch)
        if len(out) >= max_files:
            return out[:max_files]
        page_token = resp.get("nextPageToken")
        if not page_token:
            return out


def _export_body(svc, file_id: str, mime: str, body_cap: int) -> tuple[str, str]:
    """Returns (body, err). body is the decoded, capped export text."""
    export_mime = EXPORT_MIME.get(mime)
    if not export_mime:
        return "", f"unsupported_mime:{mime}"
    try:
        raw = svc.files().export(fileId=file_id, mimeType=export_mime).execute()
    except Exception as exc:
        return "", f"export_failed:{str(exc)[:100]}"
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    raw = raw.strip()
    if len(raw) > body_cap:
        raw = raw[:body_cap] + f"\n\n<!-- truncated at {body_cap} chars -->"
    return raw, ""


def _render_note(meta: dict, body: str, vault_relative_target: str) -> str:
    """Markdown with YAML frontmatter. Matches the rest of the vault's
    `source: google-drive` notes so the indexer treats it as drive-origin."""
    mime_label = MIME_LABEL.get(meta["mimeType"], "file")
    owner_list = meta.get("owners") or []
    owner_name = (owner_list[0].get("displayName") if owner_list else None) or "?"
    owner_email = (owner_list[0].get("emailAddress") if owner_list else None) or ""

    fm: list[str] = [
        "---",
        "source: google-drive",
        f"drive_id: {meta['id']}",
        f"mime: {mime_label}",
        f"modified: {meta.get('modifiedTime', '')}",
        f"created: {meta.get('createdTime', '')}",
        f"owner: {owner_name}",
    ]
    if owner_email:
        fm.append(f"owner_email: {owner_email}")
    if meta.get("webViewLink"):
        fm.append(f"link: {meta['webViewLink']}")
    fm.extend([
        "tags:",
        "- google-drive",
        "- gdrive-archive",
        f"- gdrive/{mime_label}",
        "---",
        "",
        f"# {meta['name']}",
        "",
        f"**Tipo:** {mime_label} · **Modificado:** {meta.get('modifiedTime', '')} · **Owner:** {owner_name}",
    ])
    if meta.get("webViewLink"):
        fm.append(f"**Link:** {meta['webViewLink']}")
    fm.extend(["", body if body else "_(sin contenido exportado)_", ""])
    return "\n".join(fm) + "\n"


def _target_path(vault_root: Path, meta: dict) -> Path:
    mod = meta.get("modifiedTime", "") or datetime.now(timezone.utc).isoformat()
    # modifiedTime is RFC3339 e.g. 2026-04-24T13:22:10.123Z — we just need YYYY-MM-DD
    date_prefix = mod[:10] if len(mod) >= 10 else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = _slugify(meta.get("name", "untitled"))
    id8 = meta["id"][:8]
    filename = f"{date_prefix}__{slug}__{id8}.md"
    return vault_root / ARCHIVE_SUBPATH / filename


# ── Runner ──────────────────────────────────────────────────────────────────

def run(
    *,
    days: int = DEFAULT_DAYS,
    max_files: int = DEFAULT_MAX_FILES,
    body_cap: int = DEFAULT_BODY_CAP,
    mime_types: list[str] | None = None,
    vault_root: Path | None = None,
    dry_run: bool = False,
    svc=None,
) -> dict:
    """Main entry. Returns a summary dict."""
    t0 = time.monotonic()
    mime_types = mime_types or list(EXPORT_MIME.keys())
    vault_root = vault_root or rag.VAULT_PATH

    if svc is None:
        svc, reason = _build_drive_service()
        if svc is None:
            return {"ok": False, "error": reason, "duration_s": 0.0}

    try:
        files = _list_recent(svc, days=days, mime_types=mime_types, max_files=max_files)
    except Exception as exc:
        return {"ok": False, "error": f"list_failed: {str(exc)[:120]}", "duration_s": round(time.monotonic() - t0, 2)}

    summary = {
        "ok": True,
        "files_seen": len(files),
        "files_written": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "bytes_written": 0,
        "errors": [],
        "written_paths": [],
        "days": days,
        "max_files": max_files,
        "dry_run": dry_run,
    }

    for f in files:
        target = _target_path(vault_root, f)
        if dry_run:
            summary["written_paths"].append(str(target))
            summary["files_skipped"] += 1
            continue

        body, err = _export_body(svc, f["id"], f["mimeType"], body_cap)
        if err:
            summary["files_failed"] += 1
            summary["errors"].append({"id": f["id"], "name": f.get("name"), "err": err})
            # still try to write a stub so reruns can pick up body later
            body = ""

        note = _render_note(f, body, str(target.relative_to(vault_root)))
        try:
            changed = rag._atomic_write_if_changed(target, note)
        except Exception as exc:
            summary["files_failed"] += 1
            summary["errors"].append({"id": f["id"], "name": f.get("name"), "err": f"write_failed:{str(exc)[:80]}"})
            continue

        if changed:
            summary["files_written"] += 1
            summary["bytes_written"] += len(note.encode("utf-8"))
            summary["written_paths"].append(str(target.relative_to(vault_root)))
        else:
            summary["files_skipped"] += 1

    summary["duration_s"] = round(time.monotonic() - t0, 2)
    return summary


def _run_reindex() -> tuple[bool, str]:
    """Invoke `rag index` in-process. Returns (ok, output)."""
    try:
        # Prefer the installed entrypoint so we use whatever venv the user
        # actually runs `rag` from. Fall back to `python -m rag` equivalent
        # if the installed binary isn't on PATH.
        proc = subprocess.run(
            ["rag", "index"],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        return proc.returncode == 0, (proc.stdout + proc.stderr).strip()[-4000:]
    except FileNotFoundError:
        # Fallback: call the click group via the module
        proc = subprocess.run(
            [sys.executable, "-c", "import rag; rag.cli(['index'])"],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        return proc.returncode == 0, (proc.stdout + proc.stderr).strip()[-4000:]
    except subprocess.TimeoutExpired:
        return False, "rag index timed out (30min)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS,
                    help=f"ventana de modificación en días (default {DEFAULT_DAYS})")
    ap.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES,
                    help=f"tope de archivos a bajar (default {DEFAULT_MAX_FILES})")
    ap.add_argument("--body-cap", type=int, default=DEFAULT_BODY_CAP,
                    help=f"chars máx por doc (default {DEFAULT_BODY_CAP})")
    ap.add_argument("--dry-run", action="store_true",
                    help="lista archivos sin bajar ni escribir")
    ap.add_argument("--no-reindex", action="store_true",
                    help="no correr `rag index` al final")
    ap.add_argument("--json", action="store_true",
                    help="imprimir summary como JSON")
    args = ap.parse_args()

    summary = run(
        days=args.days,
        max_files=args.max_files,
        body_cap=args.body_cap,
        dry_run=bool(args.dry_run),
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return

    if not summary.get("ok"):
        print(f"[error] {summary.get('error', 'unknown')}")
        sys.exit(1)

    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}Google Drive backfill ({args.days}d): "
        f"{summary['files_seen']} archivos · "
        f"{summary['files_written']} escritos · "
        f"{summary['files_skipped']} skipped · "
        f"{summary['files_failed']} fallaron · "
        f"{summary['bytes_written']/1024:.1f} KB · "
        f"{summary['duration_s']}s"
    )
    if summary["errors"]:
        print(f"  {len(summary['errors'])} errores (primeros 5):")
        for e in summary["errors"][:5]:
            print(f"    - {e['name']} ({e['id'][:8]}): {e['err']}")

    if args.dry_run or args.no_reindex:
        if not args.dry_run and summary["files_written"]:
            print("  (skipped reindex — correr `rag index` manualmente)")
        return

    if summary["files_written"] == 0:
        print("  (nada nuevo escrito, skip reindex)")
        return

    print("Reindexando corpus…")
    ok, out = _run_reindex()
    if ok:
        print("  reindex OK")
    else:
        print(f"  [warn] reindex falló:\n{out}")
        sys.exit(2)


if __name__ == "__main__":
    main()
