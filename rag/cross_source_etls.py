"""Cross-source ETLs — extracted from rag/__init__.py 2026-05-04.

Each function writes `.md` notes to the vault so the regular `_run_index`
rglob absorbs them. All helpers follow the same contract:
- Silent-fail: return ``{ok: False, reason: "..."}`` instead of raising.
- Hash-skip: only write if file content changed (``_atomic_write_if_changed``).
- Stats dict return for logging.

Imports from ``rag/__init__.py`` are lazy (inside each function body) to
avoid circular-import issues.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rag._constants import _GOOGLE_TOKEN_PATH

# Constantes de Spotify viven en `rag/integrations/spotify_etl.py` (split
# 2026-05-09). Import temprano necesario porque `_harden_oauth_cache_perms()`
# (line ~1093) las consume en module-load time, ANTES del re-export al final.
from rag.integrations.spotify_etl import (  # noqa: E402, F401
    _SPOTIFY_CREDS_PATH,
    _SPOTIFY_SCOPES,
    _SPOTIFY_TOKEN_PATH,
    _SPOTIFY_TOP_TTL_DAYS,
    _SPOTIFY_VAULT_SUBPATH,
)


def _etl_log_swallow(scope: str, exc: BaseException) -> None:
    """Wrapper local sobre `rag._silent_log` con lazy-import para evitar
    circular import. Cualquier fallo del logger se traga.

    Audit 2026-05-04: pre-fix había 35/59 except silent en este archivo
    con SOLO 1 logueado. Cuando un ETL fallaba (Gmail OAuth caducó, Drive
    API rate-limit, MOZE CSV mal formado, Spotify token expired), no
    quedaba traza en silent_errors.jsonl → debugging a ciegas.
    """
    try:
        from rag import _silent_log  # noqa: PLC0415 — lazy
        _silent_log(scope, exc)
    except Exception:  # pragma: no cover — never re-raise
        pass

__all__ = [
    # MOZE helpers
    "MOZE_BACKUP_DIR",
    "TARJETAS_BACKUP_DIR",
    "MOZE_VAULT_SUBPATH",
    "MOZE_MONTH_ES",
    "_moze_pnum",
    "_moze_fmt_ars",
    "_moze_parse_latest",
    "_moze_render_month",
    "_sync_moze_notes",
    # Credit card helpers
    "TARJETAS_VAULT_SUBPATH",
    "_CARD_BRAND_RE",
    "_CARD_LAST4_RE",
    "_parse_ars_or_usd",
    "_parse_card_date",
    "_parse_credit_card_xlsx",
    "_card_note_filename",
    "_card_render_note",
    "_sync_credit_cards_notes",
    # WhatsApp ETL
    "_WHATSAPP_ETL_SCRIPT",
    "_WHATSAPP_ETL_RE",
    "_sync_whatsapp_notes",
    # External-source ETL constants
    "_EXTERNAL_INGEST_BASE",
    "_REMINDERS_VAULT_SUBPATH",
    "_CALENDAR_VAULT_SUBPATH",
    "_CHROME_VAULT_SUBPATH",
    "_YOUTUBE_VAULT_SUBPATH",
    "_GMAIL_VAULT_SUBPATH",
    "_GDRIVE_VAULT_SUBPATH",
    "_GITHUB_VAULT_SUBPATH",
    "_CLAUDE_VAULT_SUBPATH",
    "_YOUTUBE_TRANSCRIPTS_SUBPATH",
    "_SPOTIFY_VAULT_SUBPATH",
    "_SPOTIFY_CREDS_PATH",
    "_SPOTIFY_TOKEN_PATH",
    "_SPOTIFY_SCOPES",
    "_SPOTIFY_TOP_TTL_DAYS",
    "_GOOGLE_KEYS_CANDIDATES",
    "_GOOGLE_TOKEN_PATH",
    "_GOOGLE_SCOPES",
    # OAuth / Chrome helpers
    "_harden_oauth_cache_perms",
    "_CHROME_HISTORY_PATH",
    "_CHROME_EPOCH_OFFSET_S",
    "_CHROME_SKIP_PREFIXES",
    "_CHROME_SKIP_PATTERNS",
    "_YOUTUBE_WATCH_RE",
    "_atomic_write_if_changed",
    # Reminders + Calendar ETLs
    "_sync_reminders_notes",
    "_sync_apple_calendar_notes",
    # Chrome helpers + ETL
    "_unix_to_chrome_ts",
    "_read_chrome_visits",
    "_sync_chrome_history",
    # Google helpers + ETLs
    "_google_keys_path",
    "_load_google_credentials",
    "_decode_gmail_body",
    "_sync_gmail_notes",
    "_sync_gdrive_notes",
    # GitHub
    "_GH_EVENT_LABELS",
    "_gh_run",
    "_sync_github_activity",
    # Claude Code transcripts
    "_CLAUDE_PROJECTS_ROOT",
    "_CLAUDE_INDEX_WINDOW_DAYS",
    "_CLAUDE_TURN_BODY_CAP",
    "_SECRET_PATTERNS",
    "_redact_secrets",
    "_claude_extract_turn",
    "_sync_claude_code_transcripts",
    # YouTube transcripts
    "_YT_TRANSCRIPT_LANG_PRIORITY",
    "_YT_TRANSCRIPT_BATCH",
    "_YT_VIDEO_ID_RE",
    "_YT_IP_BLOCKED_COOLDOWNS_SECONDS",
    "_check_yt_ip_cooldown",
    "_set_yt_ip_cooldown",
    "_collect_youtube_video_ids",
    "_fetch_yt_transcript_for_index",
    "_sync_youtube_transcripts",
    # Spotify
    "_spotify_client",
    "_sync_spotify_notes",
    # Screen Time
    "SCREENTIME_VAULT_SUBPATH",
    "_SCREENTIME_BACKFILL_DAYS",
    "_SCREENTIME_DAILY_RE",
    "_SCREENTIME_MONTHLY_RE",
    "_sync_screentime_notes",
    "_render_screentime_daily_md",
    "_render_screentime_monthly_md",
    "_render_screentime_index_md",
]

# ── MOZE finanzas ─────────────────────────────────────────────────────────────
# Movido a `rag/integrations/moze.py` (2026-05-09). Re-export al final del
# módulo preserva back-compat con `from rag.cross_source_etls import *` y
# `from rag.cross_source_etls import _sync_moze_notes`.

# `TARJETAS_BACKUP_DIR` movido a `rag/integrations/credit_cards.py`
# (2026-05-09) — re-exportado al final del módulo. La constante
# `_EXTERNAL_INGEST_BASE` queda accesible localmente vía import from
# `rag/_constants.py` (alias preservado para sub-módulos legacy).
from rag._constants import _EXTERNAL_INGEST_BASE  # noqa: E402, F401

# ── Resúmenes de tarjeta de crédito → notas mensuales ────────────────────────
# Movido a `rag/integrations/credit_cards.py` (2026-05-09). Re-export al final
# del módulo preserva back-compat con `from rag.cross_source_etls import *` y
# `from rag.cross_source_etls import _sync_credit_cards_notes`.


# ── WhatsApp ETL ──────────────────────────────────────────────────────────────

_WHATSAPP_ETL_SCRIPT = Path.home() / ".local/bin/whatsapp-to-vault"
_WHATSAPP_ETL_RE = re.compile(
    r"wrote\s+(\d+)\s+files,\s+(\d+)\s+unchanged,\s+(\d+)\s+\(chat, month\)\s+buckets,\s+(\d+)\s+chats"
)


def _sync_whatsapp_notes(vault_root: Path) -> dict:
    """Trigger the WhatsApp → vault ETL script and parse its summary line.

    Mirrors the MOZE pre-index pattern: produces `.md` files in
    `<vault>/99-obsidian/99-AI/external-ingest/WhatsApp/<chat>/YYYY-MM.md` so the regular rglob
    picks them up. Subprocess to keep it as a single source of truth — the
    same script that the `com.fer.whatsapp-vault-sync` launchd plist runs
    every 15 min. Silent-fail when the script is missing (other machines).
    """
    if not _WHATSAPP_ETL_SCRIPT.is_file():
        return {"ok": False, "reason": "script_missing"}
    try:
        proc = subprocess.run(
            [str(_WHATSAPP_ETL_SCRIPT)],
            capture_output=True, timeout=60, text=True,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return {"ok": False, "reason": str(exc)[:120]}
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return {"ok": False, "reason": err[:160] or f"rc={proc.returncode}"}
    m = _WHATSAPP_ETL_RE.search(out)
    if not m:
        return {"ok": True, "raw": out[:160]}
    return {
        "ok": True,
        "files_written": int(m.group(1)),
        "files_unchanged": int(m.group(2)),
        "buckets": int(m.group(3)),
        "chats": int(m.group(4)),
        "target": f"{_EXTERNAL_INGEST_BASE}/WhatsApp",
    }


# ── External-source ETLs ──────────────────────────────────────────────────────
# Same pattern as MOZE / WhatsApp: produce `.md` files inside the vault so the
# regular `_run_index` rglob absorbs them. Each helper is silent-fail and
# returns a stats dict for logging. Triggered from `_run_index` after the
# WhatsApp sync, before the vault scan.

_REMINDERS_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/Reminders"
_CALENDAR_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/Calendar"
_CHROME_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/Chrome"
# `_YOUTUBE_VAULT_SUBPATH` definido en `rag/integrations/youtube.py` —
# re-exportado al final del módulo para preservar back-compat con
# `from rag.cross_source_etls import _YOUTUBE_VAULT_SUBPATH`.
_GMAIL_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/Gmail"
_GDRIVE_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/GoogleDrive"
# `_GITHUB_VAULT_SUBPATH` definido en `rag/integrations/github.py` —
# re-exportado al final del módulo para preservar back-compat con
# `from rag.cross_source_etls import _GITHUB_VAULT_SUBPATH`.
# `_CLAUDE_VAULT_SUBPATH` definido en `rag/integrations/claude_code.py` —
# re-exportado al final del módulo para preservar back-compat con
# `from rag.cross_source_etls import _CLAUDE_VAULT_SUBPATH`.
# `_YOUTUBE_TRANSCRIPTS_SUBPATH` definido en `rag/integrations/youtube.py` —
# re-exportado al final del módulo para preservar back-compat con
# `from rag.cross_source_etls import _YOUTUBE_TRANSCRIPTS_SUBPATH`.
# Constantes `_SPOTIFY_*` definidas en `rag/integrations/spotify_etl.py` (split
# 2026-05-09) e importadas al tope del módulo arriba (necesario antes de que
# `_harden_oauth_cache_perms()` las consuma en module-load time).

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


def _harden_oauth_cache_perms() -> None:
    """One-shot chmod of OAuth cache files + their containing dir."""
    cfg_dir = Path.home() / ".config/obsidian-rag"
    if cfg_dir.is_dir():
        try:
            cfg_dir.chmod(0o700)
        except OSError as exc:
            _etl_log_swallow("oauth_cache_chmod_dir", exc)
    for tok in (_GOOGLE_TOKEN_PATH, _SPOTIFY_TOKEN_PATH):
        if tok.is_file():
            try:
                os.chmod(tok, 0o600)
            except OSError as exc:
                _etl_log_swallow("oauth_cache_chmod_token", exc)


_harden_oauth_cache_perms()

_CHROME_HISTORY_PATH = Path.home() / "Library/Application Support/Google/Chrome/Default/History"
# Chrome epoch is 1601-01-01 UTC microseconds (Windows FILETIME).
_CHROME_EPOCH_OFFSET_S = 11644473600
# URL prefixes / patterns we never want indexed — they're navigation noise.
_CHROME_SKIP_PREFIXES = (
    "chrome://", "chrome-extension://", "about:", "edge://", "view-source:",
    "data:", "javascript:", "file:///",
)
_CHROME_SKIP_PATTERNS = (
    re.compile(r"^https?://(www\.)?google\.[^/]+/search\?"),
    re.compile(r"^https?://(www\.)?google\.[^/]+/url\?"),
    re.compile(r"^https?://(www\.)?bing\.com/search\?"),
    re.compile(r"^https?://(duckduckgo\.com|search\.brave\.com)/\?"),
)
# `_YOUTUBE_WATCH_RE` definido en `rag/integrations/youtube.py` —
# re-exportado al final del módulo para preservar back-compat con
# `from rag.cross_source_etls import _YOUTUBE_WATCH_RE`.


def _atomic_write_if_changed(target: Path, body: str) -> bool:
    """Write `body` to `target` only if its contents changed. Returns True on
    write, False on skip. Indexing relies on hash-skip — rewriting bytes that
    haven't changed forces re-embed for nothing.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            if target.read_text(encoding="utf-8") == body:
                return False
        except OSError:
            pass
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, target)
    return True


def _sync_reminders_notes(vault_root: Path) -> dict:
    """Snapshot Apple Reminders to a daily note. Pending only, horizon 180 days
    + undated. Completed-reminders fetch is intentionally NOT included.
    """
    from rag import _apple_enabled, _fetch_reminders_due  # lazy
    if not _apple_enabled():
        return {"ok": False, "reason": "apple_disabled"}
    now = datetime.now()
    pending = _fetch_reminders_due(now, horizon_days=180, max_items=500)
    if not pending:
        return {"ok": True, "files_written": 0, "reason": "no_data"}

    by_bucket: dict[str, list[dict]] = {}
    for item in pending:
        by_bucket.setdefault(item["bucket"], []).append(item)

    today = now.strftime("%Y-%m-%d")
    fm_lines = [
        "---",
        "source: apple-reminders",
        f"snapshot_at: {now.isoformat(timespec='seconds')}",
        f"pending_count: {len(pending)}",
        "tags:",
        "- apple-reminders",
        "- system-snapshot",
        "---",
        "",
        f"# Apple Reminders — {today}",
        "",
    ]
    body_lines: list[str] = list(fm_lines)
    for bucket_key, label in (
        ("overdue", "Overdue"),
        ("today", "Hoy"),
        ("upcoming", "Próximos"),
        ("undated", "Sin fecha"),
    ):
        items = by_bucket.get(bucket_key) or []
        if not items:
            continue
        body_lines.append(f"## {label} ({len(items)})")
        body_lines.append("")
        for it in items:
            due = it["due"] or "—"
            list_tag = f" `[{it['list']}]`" if it.get("list") else ""
            body_lines.append(f"- **{it['name']}** · {due}{list_tag}")
        body_lines.append("")
    body = "\n".join(body_lines)

    target = vault_root / _REMINDERS_VAULT_SUBPATH / f"{today}.md"
    written = _atomic_write_if_changed(target, body)
    return {
        "ok": True,
        "files_written": 1 if written else 0,
        "pending": len(pending),
        "completed": 0,
        "target": _REMINDERS_VAULT_SUBPATH,
    }


def _sync_apple_calendar_notes(vault_root: Path, days_ahead: int = 90) -> dict:
    """Snapshot upcoming Apple Calendar events to per-week notes. Requires
    icalBuddy (`brew install ical-buddy`); returns silently when missing.
    """
    from rag import _apple_enabled, _icalbuddy_path, _fetch_calendar_ahead  # lazy
    if not _apple_enabled():
        return {"ok": False, "reason": "apple_disabled"}
    if not _icalbuddy_path():
        return {"ok": False, "reason": "icalbuddy_missing"}
    events = _fetch_calendar_ahead(days_ahead=days_ahead, max_events=200)
    if not events:
        return {"ok": True, "files_written": 0, "reason": "no_events"}
    now = datetime.now()
    iso_year, iso_week, _ = now.isocalendar()
    week_label = f"{iso_year}-W{iso_week:02d}"

    fm_lines = [
        "---",
        "source: apple-calendar",
        f"snapshot_at: {now.isoformat(timespec='seconds')}",
        f"window_days: {days_ahead}",
        f"event_count: {len(events)}",
        "tags:",
        "- apple-calendar",
        "- system-snapshot",
        "---",
        "",
        f"# Calendar — semana {week_label} (próximos {days_ahead}d)",
        "",
    ]
    body_lines: list[str] = list(fm_lines)
    current_label = None
    for ev in events:
        label = ev.get("date_label") or "(sin fecha)"
        if label != current_label:
            body_lines.append(f"## {label}")
            body_lines.append("")
            current_label = label
        time_range = ev.get("time_range") or ""
        time_part = f"`{time_range}` · " if time_range else ""
        body_lines.append(f"- {time_part}{ev.get('title', '(sin título)')}")
    body_lines.append("")
    body = "\n".join(body_lines)

    target = vault_root / _CALENDAR_VAULT_SUBPATH / f"{week_label}.md"
    written = _atomic_write_if_changed(target, body)
    return {
        "ok": True,
        "files_written": 1 if written else 0,
        "events": len(events),
        "target": _CALENDAR_VAULT_SUBPATH,
    }


def _unix_to_chrome_ts(unix_s: float) -> int:
    return int((unix_s + _CHROME_EPOCH_OFFSET_S) * 1_000_000)


def _read_chrome_visits(history_db: Path, hours: int = 48) -> list[dict]:
    """Read distinct URLs visited in the last `hours` from Chrome History.
    Chrome locks the SQLite while the browser runs — we copy to /tmp and read
    the snapshot. Empty list on any error.
    """
    from rag import _chrome_to_unix_ts  # lazy — defined in integrations.chrome_bookmarks
    if not history_db.is_file():
        return []
    import shutil
    import sqlite3 as _sqlite3
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "obsidian-rag-chrome-history.db"
    try:
        shutil.copy2(history_db, tmp)
    except OSError:
        return []
    try:
        con = _sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        con.row_factory = _sqlite3.Row
        cutoff = _unix_to_chrome_ts(time.time() - hours * 3600)
        rows = con.execute(
            "SELECT url, title, visit_count, last_visit_time "
            "FROM urls WHERE last_visit_time > ? "
            "ORDER BY last_visit_time DESC",
            (cutoff,),
        ).fetchall()
        con.close()
    except _sqlite3.Error:
        return []
    finally:
        try:
            tmp.unlink()
        except OSError as exc:
            _etl_log_swallow("chrome_history_tmp_unlink", exc)

    out: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        url = (r["url"] or "").strip()
        if not url or url in seen:
            continue
        if any(url.startswith(p) for p in _CHROME_SKIP_PREFIXES):
            continue
        if any(p.match(url) for p in _CHROME_SKIP_PATTERNS):
            continue
        seen.add(url)
        out.append({
            "url": url,
            "title": (r["title"] or "").strip() or url,
            "visit_count": int(r["visit_count"] or 0),
            "ts": _chrome_to_unix_ts(int(r["last_visit_time"] or 0)),
        })
    return out


def _sync_chrome_history(vault_root: Path, hours: int = 48) -> dict:
    """Daily snapshot of Chrome history (last `hours`, dedup by exact URL).
    Also derives a YouTube-only note from URLs matching watch?v=… so YouTube
    activity surfaces independently in retrieval. Hash-skipped when content
    matches the existing day file.
    """
    import sys as _sys
    _chrome_hist_path = getattr(_sys.modules.get("rag"), "_CHROME_HISTORY_PATH", _CHROME_HISTORY_PATH)
    visits = _read_chrome_visits(_chrome_hist_path, hours=hours)
    if not visits:
        return {"ok": False, "reason": "no_visits_or_chrome_locked"}
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    chrome_fm = [
        "---",
        "source: chrome-history",
        f"snapshot_at: {now.isoformat(timespec='seconds')}",
        f"window_hours: {hours}",
        f"url_count: {len(visits)}",
        "tags:",
        "- chrome-history",
        "- system-snapshot",
        "---",
        "",
        f"# Chrome history — {today} (últimas {hours}h)",
        "",
    ]
    chrome_lines: list[str] = list(chrome_fm)
    for v in visits:
        ts = datetime.fromtimestamp(v["ts"]).strftime("%H:%M")
        title = v["title"].replace("|", "·")
        chrome_lines.append(f"- `{ts}` [{title}]({v['url']})")
    chrome_body = "\n".join(chrome_lines) + "\n"

    chrome_target = vault_root / _CHROME_VAULT_SUBPATH / f"{today}.md"
    chrome_written = _atomic_write_if_changed(chrome_target, chrome_body)

    yt_videos: list[dict] = []
    seen_vid: set[str] = set()
    for v in visits:
        m = _YOUTUBE_WATCH_RE.match(v["url"])
        if not m:
            continue
        vid = m.group(2)
        if vid in seen_vid:
            continue
        seen_vid.add(vid)
        yt_videos.append({
            "video_id": vid,
            "title": v["title"],
            "url": f"https://www.youtube.com/watch?v={vid}",
            "ts": v["ts"],
        })

    yt_written = 0
    if yt_videos:
        yt_fm = [
            "---",
            "source: youtube-via-chrome",
            f"snapshot_at: {now.isoformat(timespec='seconds')}",
            f"window_hours: {hours}",
            f"video_count: {len(yt_videos)}",
            "tags:",
            "- youtube",
            "- system-snapshot",
            "---",
            "",
            f"# YouTube watched — {today} (últimas {hours}h, vía Chrome)",
            "",
        ]
        yt_lines: list[str] = list(yt_fm)
        for v in yt_videos:
            ts = datetime.fromtimestamp(v["ts"]).strftime("%H:%M")
            title = v["title"].replace("|", "·")
            yt_lines.append(f"- `{ts}` [{title}]({v['url']})")
        yt_body = "\n".join(yt_lines) + "\n"
        yt_target = vault_root / _YOUTUBE_VAULT_SUBPATH / f"{today}.md"
        yt_written = 1 if _atomic_write_if_changed(yt_target, yt_body) else 0

    return {
        "ok": True,
        "files_written": (1 if chrome_written else 0) + yt_written,
        "urls": len(visits),
        "youtube_videos": len(yt_videos),
        "target": _CHROME_VAULT_SUBPATH,
    }


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
    import sys as _sys
    _cred_fn = getattr(_sys.modules.get("rag"), "_load_google_credentials", _load_google_credentials)
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
    import sys as _sys
    _cred_fn = getattr(_sys.modules.get("rag"), "_load_google_credentials", _load_google_credentials)
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


# ── GitHub activity ───────────────────────────────────────────────────────────
# Movido a `rag/integrations/github.py` (2026-05-09). Re-export al final del
# módulo preserva back-compat con `from rag.cross_source_etls import *` y
# `from rag.cross_source_etls import _sync_github_activity`.


# ── Claude Code transcripts ───────────────────────────────────────────────────
# Movido a `rag/integrations/claude_code.py` (2026-05-09). Re-export al final
# del módulo preserva back-compat con `from rag.cross_source_etls import *`
# y `from rag.cross_source_etls import _sync_claude_code_transcripts`.


# ── YouTube transcripts ───────────────────────────────────────────────────────
# Movido a `rag/integrations/youtube.py` (2026-05-09). Re-export al final del
# módulo preserva back-compat con `from rag.cross_source_etls import *` y
# `from rag.cross_source_etls import _sync_youtube_transcripts`.



# ── Spotify ───────────────────────────────────────────────────────────────────
# Movido a `rag/integrations/spotify_etl.py` (2026-05-09). Re-export al final
# del módulo preserva back-compat con `from rag.cross_source_etls import *` y
# `from rag.cross_source_etls import _sync_spotify_notes`.


# ── Screen Time persistence (daily + monthly) ─────────────────────────────────
# Movido a `rag/integrations/screentime.py` (2026-05-09) para coalescer con
# `_collect_screentime` (mismo source DB, misma categorización). Re-exportado
# acá vía `__all__` en `rag/__init__.py` (que ya hace `from rag.cross_source_etls
# import *`) para preservar back-compat con call sites + monkeypatches viejos.
from rag.integrations.screentime import (  # noqa: F401, E402
    SCREENTIME_VAULT_SUBPATH,
    _SCREENTIME_BACKFILL_DAYS,
    _SCREENTIME_DAILY_RE,
    _SCREENTIME_MONTHLY_RE,
    _render_screentime_daily_md,
    _render_screentime_index_md,
    _render_screentime_monthly_md,
    _sync_screentime_notes,
)

# ── Claude Code transcripts (re-export) ──────────────────────────────────────
# Movido a `rag/integrations/claude_code.py` (2026-05-09). Re-exportado para
# preservar `from rag.cross_source_etls import _sync_claude_code_transcripts`
# y los monkeypatches en tests (`tests/test_external_etls.py`).
from rag.integrations.claude_code import (  # noqa: F401, E402
    _CLAUDE_INDEX_WINDOW_DAYS,
    _CLAUDE_PROJECTS_ROOT,
    _CLAUDE_TURN_BODY_CAP,
    _CLAUDE_VAULT_SUBPATH,
    _SECRET_PATTERNS,
    _claude_extract_turn,
    _redact_secrets,
    _sync_claude_code_transcripts,
)

# ── GitHub activity (re-export) ───────────────────────────────────────────────
# Movido a `rag/integrations/github.py` (2026-05-09). Re-exportado para
# preservar `from rag.cross_source_etls import _sync_github_activity` y los
# monkeypatches en tests (`tests/test_external_etls.py`).
from rag.integrations.github import (  # noqa: F401, E402
    _GH_EVENT_LABELS,
    _GITHUB_VAULT_SUBPATH,
    _gh_run,
    _sync_github_activity,
)

# ── YouTube transcripts (re-export) ──────────────────────────────────────────
# Movido a `rag/integrations/youtube.py` (2026-05-09). Re-exportado para
# preservar `from rag.cross_source_etls import _sync_youtube_transcripts`,
# acceso a constantes desde `_sync_chrome_history` (que usa `_YOUTUBE_VAULT_SUBPATH`
# + `_YOUTUBE_WATCH_RE`), y los monkeypatches en tests
# (`tests/test_external_etls.py`).
from rag.integrations.youtube import (  # noqa: F401, E402
    _YT_IP_BLOCKED_COOLDOWNS_SECONDS,
    _YT_TRANSCRIPT_BATCH,
    _YT_TRANSCRIPT_LANG_PRIORITY,
    _YT_VIDEO_ID_RE,
    _YOUTUBE_TRANSCRIPTS_SUBPATH,
    _YOUTUBE_VAULT_SUBPATH,
    _YOUTUBE_WATCH_RE,
    _check_yt_ip_cooldown,
    _collect_youtube_video_ids,
    _fetch_yt_transcript_for_index,
    _set_yt_ip_cooldown,
    _sync_youtube_transcripts,
)

# ── Spotify (re-export) ──────────────────────────────────────────────────────
# Movido a `rag/integrations/spotify_etl.py` (2026-05-09). Las constantes ya
# se importan arriba (necesario para `_harden_oauth_cache_perms`); este
# re-export trae las funciones para preservar
# `from rag.cross_source_etls import _sync_spotify_notes` + monkeypatches
# en `tests/test_external_etls.py`.
from rag.integrations.spotify_etl import (  # noqa: F401, E402
    _spotify_client,
    _sync_spotify_notes,
)

# ── MOZE finanzas (re-export) ────────────────────────────────────────────────
# Movido a `rag/integrations/moze.py` (2026-05-09). Re-exportado para
# preservar `from rag.cross_source_etls import _sync_moze_notes` + las
# constantes (`MOZE_BACKUP_DIR`, `MOZE_VAULT_SUBPATH`, etc.) usadas por
# call sites externos + tests (`tests/test_external_etls.py`,
# `tests/test_cross_source_*.py`).
from rag.integrations.moze import (  # noqa: F401, E402
    MOZE_BACKUP_DIR,
    MOZE_MONTH_ES,
    MOZE_VAULT_SUBPATH,
    _moze_cache_dir,
    _moze_fmt_ars,
    _moze_parse_latest,
    _moze_pnum,
    _moze_render_month,
    _sync_moze_notes,
)

# ── Credit cards (re-export) ─────────────────────────────────────────────────
# Movido a `rag/integrations/credit_cards.py` (2026-05-09). Re-exportado para
# preservar `from rag.cross_source_etls import _sync_credit_cards_notes` +
# las constantes (`TARJETAS_BACKUP_DIR`, `TARJETAS_VAULT_SUBPATH`, regex) y
# los call sites externos + tests.
from rag.integrations.credit_cards import (  # noqa: F401, E402
    TARJETAS_BACKUP_DIR,
    TARJETAS_VAULT_SUBPATH,
    _CARD_BRAND_RE,
    _CARD_LAST4_RE,
    _card_note_filename,
    _card_render_note,
    _parse_ars_or_usd,
    _parse_card_date,
    _parse_credit_card_xlsx,
    _sync_credit_cards_notes,
)
