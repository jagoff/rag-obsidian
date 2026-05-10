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
# `_CHROME_VAULT_SUBPATH` definido en `rag/integrations/chrome_history.py` —
# re-exportado al final del módulo.
# `_YOUTUBE_VAULT_SUBPATH` definido en `rag/integrations/youtube.py` —
# re-exportado al final del módulo para preservar back-compat con
# `from rag.cross_source_etls import _YOUTUBE_VAULT_SUBPATH`.
# `_GMAIL_VAULT_SUBPATH` y `_GDRIVE_VAULT_SUBPATH` definidos en
# `rag/integrations/google_apis.py` — re-exportados al final del módulo.
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

# `_GOOGLE_KEYS_CANDIDATES` y `_GOOGLE_SCOPES` definidos en
# `rag/integrations/google_apis.py` — re-exportados al final del módulo.


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

# Constantes `_CHROME_HISTORY_PATH`, `_CHROME_EPOCH_OFFSET_S`,
# `_CHROME_SKIP_PREFIXES`, `_CHROME_SKIP_PATTERNS` definidas en
# `rag/integrations/chrome_history.py` — re-exportadas al final del módulo.
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



# ── Chrome history ───────────────────────────────────────────────────────────
# Movido a `rag/integrations/chrome_history.py` (2026-05-09). Re-export al final
# del módulo preserva back-compat con `from rag.cross_source_etls import *` y
# `from rag.cross_source_etls import _sync_chrome_history`.


# ── Gmail + Google Drive ─────────────────────────────────────────────────────
# Movido a `rag/integrations/google_apis.py` (2026-05-09). Re-export al final
# del módulo preserva back-compat con `from rag.cross_source_etls import *` y
# `from rag.cross_source_etls import _sync_gmail_notes` /
# `_sync_gdrive_notes`.

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

# ── Gmail + Google Drive (re-export) ─────────────────────────────────────────
# Movido a `rag/integrations/google_apis.py` (2026-05-09). Re-exportado para
# preservar `from rag.cross_source_etls import _sync_gmail_notes` /
# `_sync_gdrive_notes` + las constantes (`_GMAIL_VAULT_SUBPATH`,
# `_GOOGLE_KEYS_CANDIDATES`, etc.) y los monkeypatches de los tests
# (`tests/test_integration_gmail.py`, `tests/test_file_permissions.py`).
from rag.integrations.google_apis import (  # noqa: F401, E402
    _GDRIVE_VAULT_SUBPATH,
    _GMAIL_VAULT_SUBPATH,
    _GOOGLE_KEYS_CANDIDATES,
    _GOOGLE_SCOPES,
    _decode_gmail_body,
    _google_keys_path,
    _load_google_credentials,
    _sync_gdrive_notes,
    _sync_gmail_notes,
)

# ── Chrome history (re-export) ───────────────────────────────────────────────
# Movido a `rag/integrations/chrome_history.py` (2026-05-09). Re-exportado para
# preservar `from rag.cross_source_etls import _sync_chrome_history` + las
# constantes (`_CHROME_HISTORY_PATH`, `_CHROME_VAULT_SUBPATH`, etc.) y los
# monkeypatches en `tests/test_external_etls.py`.
from rag.integrations.chrome_history import (  # noqa: F401, E402
    _CHROME_EPOCH_OFFSET_S,
    _CHROME_HISTORY_PATH,
    _CHROME_SKIP_PATTERNS,
    _CHROME_SKIP_PREFIXES,
    _CHROME_VAULT_SUBPATH,
    _read_chrome_visits,
    _sync_chrome_history,
    _unix_to_chrome_ts,
)
