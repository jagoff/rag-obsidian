"""Gmail + Google Drive ETLs — extracted from rag/cross_source_etls.py 2026-05-09.

Snapshots recent Gmail messages and recently-modified Google Docs/Sheets/
Slides via the Google API client + OAuth (readonly scopes). Outputs land in
``99-obsidian/99-AI/external-ingest/{Gmail,GoogleDrive}/<YYYY-MM-DD>.md`` so
the regular ``_run_index`` rglob absorbs them.

Both ETLs share the OAuth handshake (``_load_google_credentials``) → splitting
them into the same module keeps that helper colocated with its consumers.

Silent-fail contract: helpers return ``None`` /
``{ok: False, reason: "..."}`` instead of raising. ``_atomic_write_if_changed``
and ``_etl_log_swallow`` are lazy-imported from ``rag.cross_source_etls`` to
avoid circular import. ``google.*`` and ``googleapiclient`` are lazy-imported
inside the functions so the module loads without them (the ETL is a no-op
when the deps are missing — same contract as ``spotipy`` / ``openpyxl``).

OAuth keys: re-uses the gmail-mcp client config so the user doesn't manage
two Google Cloud OAuth apps. Token is stored in our own config dir
(``_GOOGLE_TOKEN_PATH`` from ``rag._constants``) so the scopes
(``gmail.readonly`` + ``drive.readonly``) are independent of gmail-mcp's
own token.

Tests (``tests/test_integration_gmail.py``, ``tests/test_file_permissions.py``)
monkeypatch ``rag._load_google_credentials`` and ``rag._GOOGLE_TOKEN_PATH``
on the top-level ``rag`` module — ``_sync_gmail_notes`` /
``_sync_gdrive_notes`` re-resolve ``_load_google_credentials`` via
``sys.modules.get("rag")`` so the patch propagates regardless of where the
function lives.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rag._constants import _EXTERNAL_INGEST_BASE, _GOOGLE_TOKEN_PATH

__all__ = [
    "_GMAIL_VAULT_SUBPATH",
    "_GDRIVE_VAULT_SUBPATH",
    "_GOOGLE_KEYS_CANDIDATES",
    "_GOOGLE_SCOPES",
    "_google_keys_path",
    "_load_google_credentials",
    "_decode_gmail_body",
    "_sync_gmail_notes",
    "_sync_gdrive_notes",
]

_GMAIL_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/Gmail"
_GDRIVE_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/GoogleDrive"

# OAuth keys: reuse the gmail-mcp client config so the user doesn't manage two
# Google Cloud OAuth apps. Token is stored in our own config dir so the
# scopes (gmail + drive readonly) are independent of gmail-mcp's own token.
_GOOGLE_KEYS_CANDIDATES = (
    Path.home() / ".config/obsidian-rag/google_credentials.json",
    Path.home() / ".gmail-mcp/gcp-oauth.keys.json",
)
_GOOGLE_SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
)


def _google_keys_path() -> Path | None:
    for p in _GOOGLE_KEYS_CANDIDATES:
        if p.is_file():
            return p
    return None


def _load_google_credentials(allow_interactive: bool = True) -> "google.oauth2.credentials.Credentials | None":
    """Return Google OAuth `Credentials` for Gmail + Drive (readonly), or None.

    Lookup order: cached token → refresh if expired → first-time interactive
    browser flow (only when `allow_interactive` and stdin is a TTY). Token is
    persisted to `_GOOGLE_TOKEN_PATH` so subsequent runs are silent.
    """
    from rag import _silent_log, _write_secret_file  # lazy
    from rag.cross_source_etls import _etl_log_swallow

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        return None

    creds = None
    if _GOOGLE_TOKEN_PATH.is_file():
        try:
            creds = Credentials.from_authorized_user_file(
                str(_GOOGLE_TOKEN_PATH), list(_GOOGLE_SCOPES)
            )
        except Exception:
            creds = None
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _write_secret_file(_GOOGLE_TOKEN_PATH, creds.to_json())
            return creds
        except Exception as exc:
            _silent_log('google_token_refresh', exc)
    if not allow_interactive or not sys.stdin.isatty():
        return None
    keys = _google_keys_path()
    if not keys:
        return None
    try:
        flow = InstalledAppFlow.from_client_secrets_file(str(keys), list(_GOOGLE_SCOPES))
        creds = flow.run_local_server(port=0, open_browser=True)
    except Exception as exc:
        _etl_log_swallow("google_oauth_flow_failed", exc)
        return None
    _write_secret_file(_GOOGLE_TOKEN_PATH, creds.to_json())
    return creds


def _decode_gmail_body(payload: dict) -> str:
    """Walk a Gmail API `payload` tree, prefer text/plain, fall back to HTML
    stripped of tags. Returns empty string when the message has no body parts.
    """
    import base64
    def _decode(data: str) -> str:
        try:
            return base64.urlsafe_b64decode(data.encode("ascii")).decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _walk(node: dict, want_mime: str) -> str:
        if node.get("mimeType") == want_mime and (node.get("body") or {}).get("data"):
            return _decode(node["body"]["data"])
        for child in node.get("parts") or []:
            found = _walk(child, want_mime)
            if found:
                return found
        return ""

    plain = _walk(payload, "text/plain")
    if plain:
        return plain
    html = _walk(payload, "text/html")
    if not html:
        return ""
    # Drop <style> + <script> block contents before stripping tags.
    html = re.sub(
        r"<(style|script)\b[^>]*>.*?</\1\s*>", " ", html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return re.sub(r"<[^>]+>", " ", html)


def _sync_gmail_notes(vault_root: Path, hours: int = 48, max_messages: int = 30, body_cap: int = 5000) -> dict:
    """Snapshot recent Gmail to a daily note. Subject + headers + body (capped)
    per message. Hash-skipped when content unchanged.
    """
    from rag.cross_source_etls import _atomic_write_if_changed, _etl_log_swallow

    _cred_fn = getattr(sys.modules.get("rag"), "_load_google_credentials", _load_google_credentials)
    creds = _cred_fn()
    if creds is None:
        return {"ok": False, "reason": "no_google_credentials"}
    try:
        from googleapiclient.discovery import build
    except ImportError:
        return {"ok": False, "reason": "google_api_missing"}
    try:
        gm = build("gmail", "v1", credentials=creds, cache_discovery=False)
        days = max(1, int((hours + 23) // 24))
        resp = gm.users().messages().list(
            userId="me", q=f"newer_than:{days}d", maxResults=max_messages,
        ).execute()
        ids = [m["id"] for m in (resp.get("messages") or [])]
    except Exception as exc:
        return {"ok": False, "reason": f"list_failed: {str(exc)[:120]}"}
    if not ids:
        return {"ok": True, "files_written": 0, "reason": "no_messages"}

    messages: list[dict] = []
    for mid in ids:
        try:
            msg = gm.users().messages().get(
                userId="me", id=mid, format="full",
            ).execute()
        except Exception as exc:
            _etl_log_swallow("gmail_message_fetch", exc)
            continue
        headers = {h["name"].lower(): h["value"] for h in (msg.get("payload", {}).get("headers") or [])}
        body = _decode_gmail_body(msg.get("payload") or {})
        body = re.sub(r"\s+", " ", body).strip()[:body_cap]
        messages.append({
            "id": mid,
            "subject": headers.get("subject", "(sin subject)"),
            "from": headers.get("from", "?"),
            "date": headers.get("date", ""),
            "snippet": (msg.get("snippet") or "").strip(),
            "body": body,
        })

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    fm = [
        "---",
        "source: gmail",
        f"snapshot_at: {now.isoformat(timespec='seconds')}",
        f"window_hours: {hours}",
        f"message_count: {len(messages)}",
        "tags:",
        "- gmail",
        "- system-snapshot",
        "---",
        "",
        f"# Gmail — {today} (últimas {hours}h)",
        "",
    ]
    for m in messages:
        fm.append(f"## {m['subject']}")
        fm.append("")
        fm.append(f"**From:** {m['from']}  ")
        fm.append(f"**Date:** {m['date']}  ")
        if m["snippet"]:
            fm.append(f"**Snippet:** {m['snippet']}")
        fm.append("")
        if m["body"]:
            fm.append(m["body"])
            fm.append("")
    body_text = "\n".join(fm) + "\n"
    target = vault_root / _GMAIL_VAULT_SUBPATH / f"{today}.md"
    written = _atomic_write_if_changed(target, body_text)
    return {
        "ok": True,
        "files_written": 1 if written else 0,
        "messages": len(messages),
        "target": _GMAIL_VAULT_SUBPATH,
    }


def _sync_gdrive_notes(vault_root: Path, hours: int = 48, max_docs: int = 4, body_cap: int = 8000) -> dict:
    """Snapshot the last `max_docs` Google Docs/Sheets/Slides modified in the
    window. Title + exported text body per doc. Hash-skipped.
    """
    from rag.cross_source_etls import _atomic_write_if_changed

    _cred_fn = getattr(sys.modules.get("rag"), "_load_google_credentials", _load_google_credentials)
    creds = _cred_fn()
    if creds is None:
        return {"ok": False, "reason": "no_google_credentials"}
    try:
        from googleapiclient.discovery import build
    except ImportError:
        return {"ok": False, "reason": "google_api_missing"}
    try:
        dv = build("drive", "v3", credentials=creds, cache_discovery=False)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        mime_filter = " or ".join(
            f"mimeType = '{m}'" for m in (
                "application/vnd.google-apps.document",
                "application/vnd.google-apps.spreadsheet",
                "application/vnd.google-apps.presentation",
            )
        )
        q = f"(modifiedTime > '{cutoff}') and ({mime_filter}) and trashed = false"
        resp = dv.files().list(
            q=q, orderBy="modifiedTime desc", pageSize=max_docs,
            fields="files(id, name, mimeType, modifiedTime, owners(displayName), webViewLink)",
        ).execute()
        files = resp.get("files") or []
    except Exception as exc:
        return {"ok": False, "reason": f"list_failed: {str(exc)[:120]}"}
    if not files:
        return {"ok": True, "files_written": 0, "reason": "no_docs"}

    EXPORT_MIME = {
        "application/vnd.google-apps.document": "text/plain",
        "application/vnd.google-apps.spreadsheet": "text/csv",
        "application/vnd.google-apps.presentation": "text/plain",
    }
    docs: list[dict] = []
    for f in files:
        export_mime = EXPORT_MIME.get(f["mimeType"], "text/plain")
        try:
            body = dv.files().export(fileId=f["id"], mimeType=export_mime).execute()
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="ignore")
            body = body.strip()[:body_cap]
        except Exception:
            body = ""
        docs.append({
            "id": f["id"],
            "name": f.get("name", "(sin nombre)"),
            "mime": f["mimeType"].split(".")[-1],
            "modified": f.get("modifiedTime", ""),
            "owner": (f.get("owners") or [{}])[0].get("displayName", "?"),
            "link": f.get("webViewLink", ""),
            "body": body,
        })

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    fm = [
        "---",
        "source: google-drive",
        f"snapshot_at: {now.isoformat(timespec='seconds')}",
        f"window_hours: {hours}",
        f"doc_count: {len(docs)}",
        "tags:",
        "- google-drive",
        "- system-snapshot",
        "---",
        "",
        f"# Google Drive — {today} (últimos {len(docs)} docs últimas {hours}h)",
        "",
    ]
    for d in docs:
        fm.append(f"## {d['name']}")
        fm.append("")
        fm.append(f"**Tipo:** {d['mime']} · **Modificado:** {d['modified']} · **Owner:** {d['owner']}")
        if d["link"]:
            fm.append(f"**Link:** {d['link']}")
        fm.append("")
        if d["body"]:
            fm.append(d["body"])
            fm.append("")
    body_text = "\n".join(fm) + "\n"
    target = vault_root / _GDRIVE_VAULT_SUBPATH / f"{today}.md"
    written = _atomic_write_if_changed(target, body_text)
    return {
        "ok": True,
        "files_written": 1 if written else 0,
        "docs": len(docs),
        "target": _GDRIVE_VAULT_SUBPATH,
    }
