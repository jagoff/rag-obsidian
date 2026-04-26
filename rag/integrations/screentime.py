"""Screen Time integration — leaf ETL extracted from `rag/__init__.py` (Phase 1b).

Source: macOS Screen Time logs at `~/Library/Application Support/Knowledge/
knowledgeC.db` (CoreDuet's [knowledge store](https://gist.github.com/mac4n6/9d44e3001b1d59d3eb1f49f5e54f4ada)).
Read-only access works without Full Disk Access as long as the file is
readable. Each row in `ZOBJECT` with `ZSTREAMNAME = '/app/usage'` represents
one foreground session — summing `ZENDDATE - ZSTARTDATE` per bundle gives
active-use seconds (NOT wall time — backgrounded apps don't count).

## Surfaces

- `_collect_screentime(start, end, db_path=None)` — per-app + per-category
  foreground usage in `[start, end)`. Returns `{available, total_secs,
  top_apps, categories}`. Sessions <5s ignored (filters spurious re-focuses).
  Unknown bundle IDs surface via their stem so brand-new apps don't hide in
  "otros".
- `_render_screentime_section(st)` — deterministic markdown for the morning
  brief. Empty if `st.available=False` or `total < 5min` (likely a sleeping
  Mac or fresh setup).
- `_screentime_app_label(bundle)` — bundle ID → human label. Falls back to
  the last dotted segment when unknown.
- `_screentime_category(bundle)` — bundle ID → coarse category (`code`,
  `notas`, `comms`, `browser`, `media`, `otros`).

## Invariants
- Silent-fail: missing DB, locked DB, sqlite error → `{available: False, ...}`.
  Never raise.
- The `immutable=1` URI flag lets us read even if macOS holds a write lock
  (read-only mode + WAL ignore — the snapshot may be slightly stale but
  morning briefs don't need second-level accuracy).
- Cocoa epoch offset: 978,307,200 seconds between 1970-01-01 (Unix) and
  2001-01-01 (Cocoa). DON'T forget to subtract it from `datetime.timestamp()`
  before passing to the SQL — the column is in Cocoa-seconds.

## Why deferred imports
`_fmt_hm` lives in `rag.__init__` (also used by `web/server.py` directly).
Module-level `from rag import _fmt_hm` here would deadlock the package load.
Function-body imports run after `rag.__init__` finishes loading.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


# ── Screen Time (knowledgeC.db) ─────────────────────────────────────────
# macOS logs foreground app usage at `/app/usage` in CoreDuet's knowledge
# store. Read-only access works without Full Disk Access as long as the
# file is readable. Values are per foreground session; summing gives
# active-use seconds (not wall time). Categories are heuristic — bundle
# ID prefix match. Unknown apps render as bundle ID stem so new apps
# surface instead of hiding in "otros".

SCREENTIME_DB = Path.home() / "Library/Application Support/Knowledge/knowledgeC.db"
# 978307200 = seconds between 1970-01-01 and 2001-01-01 (Cocoa epoch).
_SCREENTIME_COCOA_OFFSET = 978307200

_SCREENTIME_APP_LABELS = {
    "com.exafunction.windsurf": "Windsurf",
    "com.googlecode.iterm2": "iTerm",
    "com.apple.Terminal": "Terminal",
    "com.mitchellh.ghostty": "Ghostty",
    "com.microsoft.VSCode": "VS Code",
    "com.sublimetext.4": "Sublime",
    "com.jetbrains.pycharm": "PyCharm",
    "md.obsidian": "Obsidian",
    "com.google.Chrome": "Chrome",
    "com.apple.Safari": "Safari",
    "company.thebrowser.Browser": "Arc",
    "com.brave.Browser": "Brave",
    "net.whatsapp.WhatsApp": "WhatsApp",
    "com.apple.MobileSMS": "Messages",
    "com.tinyspeck.slackmacgap": "Slack",
    "com.hnc.Discord": "Discord",
    "ru.keepcoder.Telegram": "Telegram",
    "com.apple.mail": "Mail",
    "com.apple.iCal": "Calendar",
    "com.flexibits.fantastical2.mac": "Fantastical",
    "com.apple.reminders": "Reminders",
    "com.apple.Notes": "Notes",
    "com.apple.finder": "Finder",
    "com.apple.Photos": "Photos",
    "com.apple.Music": "Music",
    "com.spotify.client": "Spotify",
    "com.apple.QuickTimePlayerX": "QuickTime",
    "com.apple.systempreferences": "System Settings",
    "com.apple.ActivityMonitor": "Activity Monitor",
    "com.figma.Desktop": "Figma",
    "com.linear": "Linear",
    "notion.id": "Notion",
    "com.apple.podcasts": "Podcasts",
}

_SCREENTIME_CATEGORIES = {
    "code": {
        "com.exafunction.windsurf", "com.googlecode.iterm2", "com.apple.Terminal",
        "com.mitchellh.ghostty",
        "com.microsoft.VSCode", "com.sublimetext.4", "com.jetbrains.pycharm",
        "com.apple.dt.Xcode", "com.todesktop.230313mzl4w4u92",  # Cursor
    },
    "notas": {"md.obsidian", "com.apple.Notes", "notion.id"},
    "comms": {
        "net.whatsapp.WhatsApp", "com.apple.MobileSMS", "com.tinyspeck.slackmacgap",
        "com.hnc.Discord", "ru.keepcoder.Telegram", "com.apple.mail", "com.apple.FaceTime",
    },
    "browser": {
        "com.google.Chrome", "com.apple.Safari", "company.thebrowser.Browser",
        "com.brave.Browser", "org.mozilla.firefox",
    },
    "media": {
        "com.apple.Music", "com.spotify.client", "com.apple.QuickTimePlayerX",
        "com.apple.podcasts", "com.apple.TV",
    },
}


def _screentime_app_label(bundle: str) -> str:
    if bundle in _SCREENTIME_APP_LABELS:
        return _SCREENTIME_APP_LABELS[bundle]
    # Fallback: last dotted segment, title-cased ("com.foo.BarApp" → "BarApp")
    return bundle.rsplit(".", 1)[-1] if "." in bundle else bundle


def _screentime_category(bundle: str) -> str:
    for cat, bundles in _SCREENTIME_CATEGORIES.items():
        if bundle in bundles:
            return cat
    return "otros"


def _collect_screentime(
    start: datetime, end: datetime,
    db_path: Path | None = None,
) -> dict:
    """Per-app foreground usage for [start, end). Returns:

    ```
    {
        "available": bool,
        "total_secs": int,
        "top_apps": [{"bundle": str, "label": str, "secs": int}],
        "categories": {"code": int, "comms": int, ...},
    }
    ```

    Silent-degrades to `available=False` if the db is missing or locked.
    Only sessions >= 5s counted (filters spurious re-focuses). Unknown
    bundles surface via their stem so new apps aren't swept into "otros".
    """
    import sqlite3

    path = db_path or SCREENTIME_DB
    empty = {"available": False, "total_secs": 0, "top_apps": [], "categories": {}}
    if not path.is_file():
        return empty

    start_ts = start.timestamp() - _SCREENTIME_COCOA_OFFSET
    end_ts = end.timestamp() - _SCREENTIME_COCOA_OFFSET
    try:
        # immutable=1 lets us read even if macOS holds a write lock.
        uri = f"file:{path}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            rows = conn.execute(
                """
                SELECT ZVALUESTRING, SUM(ZENDDATE - ZSTARTDATE) AS secs
                FROM ZOBJECT
                WHERE ZSTREAMNAME = '/app/usage'
                  AND ZSTARTDATE >= ?
                  AND ZSTARTDATE < ?
                  AND (ZENDDATE - ZSTARTDATE) >= 5
                GROUP BY ZVALUESTRING
                ORDER BY secs DESC
                """,
                (start_ts, end_ts),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return empty

    top: list[dict] = []
    cats: dict[str, int] = {}
    total = 0
    for bundle, secs in rows:
        if not bundle or secs is None:
            continue
        s = int(round(float(secs)))
        if s <= 0:
            continue
        total += s
        top.append({
            "bundle": bundle,
            "label": _screentime_app_label(bundle),
            "secs": s,
        })
        cat = _screentime_category(bundle)
        cats[cat] = cats.get(cat, 0) + s

    return {
        "available": True,
        "total_secs": total,
        "top_apps": top[:10],
        "categories": cats,
    }


def _render_screentime_section(st: dict) -> str:
    """Deterministic "where time went" section. Empty if db unavailable
    or total < 5 min (likely sleeping Mac or brand-new setup).
    """
    from rag import _fmt_hm
    if not st or not st.get("available"):
        return ""
    total = int(st.get("total_secs") or 0)
    if total < 300:
        return ""

    lines = [f"## 🖥 Pantalla · {_fmt_hm(total)} activo"]
    top = (st.get("top_apps") or [])[:5]
    if top:
        parts = [f"{a['label']} {_fmt_hm(a['secs'])}" for a in top]
        lines.append("- " + " · ".join(parts))
    cats = st.get("categories") or {}
    if cats:
        order = ["code", "notas", "comms", "browser", "media", "otros"]
        pieces = []
        for k in order:
            v = cats.get(k, 0)
            if v >= 60:
                pieces.append(f"{k} {_fmt_hm(v)}")
        if pieces:
            lines.append("- " + " · ".join(pieces))
    return "\n".join(lines)
