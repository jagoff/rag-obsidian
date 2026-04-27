"""Chrome bookmarks + history integration — leaf ETL extracted from `rag/__init__.py`.

Source: local Chrome profile dirs at `~/Library/Application Support/Google/Chrome/`.
Each profile has a `Bookmarks` JSON tree and a `History` SQLite DB. We surface
two distinct signals:

- `_fetch_chrome_bookmarks_used`: top-n *bookmarked* URLs visited in the last N
  hours — high-intent reads vs ambient browsing. Joins the JSON tree with the
  History DB on URL.
- `_chrome_to_unix_ts`: helper to convert Chrome's microseconds-since-1601
  (Windows FILETIME / Webkit epoch) into a regular Unix timestamp. Lives here
  because every consumer of Chrome data hits it.
- `_chrome_bookmarks_root`: returns the Chrome profile root dir; tests
  monkey-patch this to point at a tmp dir with fake profiles.

## Invariants
- Silent-fail: missing file, locked SQLite (Chrome running), JSON decode error,
  permission denied → return `[]`. Never raise.
- We copy the History DB to a tmp file before reading because Chrome holds an
  exclusive write lock; reading the live file races and corrupts results.
- The Chrome epoch offset (11,644,473,600 s = 369 years × 365.2425 days) is
  re-defined locally inside `_fetch_chrome_bookmarks_used` to avoid import-time
  coupling to `rag.__init__`. `_chrome_to_unix_ts` reuses the constant from
  `rag.__init__` via a deferred import to preserve a single source of truth
  there (`_unix_to_chrome_ts` and others stay in core).

## Why deferred imports
`rag.__init__` re-exports these symbols at the bottom; module-level
`from rag import X` here would deadlock the package import. Inside function
bodies the import is fine — by the time the function runs, `rag.__init__` is
fully loaded.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path


def _chrome_to_unix_ts(chrome_us: int) -> float:
    from rag import _CHROME_EPOCH_OFFSET_S
    return (chrome_us / 1_000_000.0) - _CHROME_EPOCH_OFFSET_S


def _chrome_bookmarks_root() -> Path:
    return Path.home() / "Library" / "Application Support" / "Google" / "Chrome"


# ── Chrome bookmarks used (History join Bookmarks) ───────────────────────────
# Bookmarks live in a JSON tree, visits live in SQLite; join by URL to surface
# which *saved* pages the user reached for recently. Distinct signal from raw
# top-visited (ambient browsing) because bookmarks encode intent.
def _fetch_chrome_bookmarks_used(hours: int = 48, n: int = 5) -> list[dict]:
    """Top-n bookmarks whose URL was visited in the last `hours`.

    Pipeline:
    1. Flatten `Bookmarks` JSON (recursive across `roots.*`) into a URL→meta map.
    2. Copy `History` SQLite (WAL-safe) and query visits within the window.
    3. Inner-join by URL, sort by `last_visit` desc, truncate to n.

    Chrome's `visit_time` is microseconds since 1601-01-01 UTC — same epoch as
    `Bookmarks.date_added`, which is why the conversion constant is shared
    with `_fetch_chrome_top_week`. Silent-fail if either file is missing.
    """
    bm_path = Path.home() / "Library/Application Support/Google/Chrome/Default/Bookmarks"
    hist_path = Path.home() / "Library/Application Support/Google/Chrome/Default/History"
    if not bm_path.is_file() or not hist_path.is_file():
        return []

    try:
        tree = json.loads(bm_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    bookmarks: dict[str, dict] = {}

    def _walk(node: dict, folder: str) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "url":
            url = node.get("url") or ""
            if url and url not in bookmarks:
                bookmarks[url] = {
                    "name": node.get("name") or "",
                    "folder": folder.strip("/"),
                }
            return
        name = node.get("name") or ""
        sub_folder = f"{folder}/{name}" if name else folder
        for child in node.get("children") or []:
            _walk(child, sub_folder)

    for root in (tree.get("roots") or {}).values():
        _walk(root, "")

    if not bookmarks:
        return []

    CHROME_EPOCH_OFFSET = 11_644_473_600
    now_ts = time.time()
    window_chrome = int((now_ts - hours * 3600 + CHROME_EPOCH_OFFSET) * 1_000_000)

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=True) as tmp:
        shutil.copyfile(hist_path, tmp.name)
        conn = sqlite3.connect(f"file:{tmp.name}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT u.url AS url,
                       COUNT(v.id) AS visit_count,
                       MAX(v.visit_time) AS last_visit
                FROM urls u
                JOIN visits v ON v.url = u.id
                WHERE v.visit_time >= ?
                GROUP BY u.url
                ORDER BY last_visit DESC
                """,
                (window_chrome,),
            ).fetchall()
        finally:
            conn.close()

    out: list[dict] = []
    for r in rows:
        url = r["url"]
        meta = bookmarks.get(url)
        if not meta:
            continue
        last_unix = (r["last_visit"] / 1_000_000) - CHROME_EPOCH_OFFSET
        out.append({
            "name": meta["name"],
            "url": url,
            "folder": meta["folder"],
            "visit_count": int(r["visit_count"]),
            "last_visit_iso": datetime.fromtimestamp(last_unix).isoformat(timespec="seconds"),
        })
        if len(out) >= n:
            break
    return out


# ── YouTube watched today (Chrome history) ─────────────────────────────────
# Sibling of `_fetch_chrome_bookmarks_used` — same Chrome history DB, same
# tmp-copy pattern, but filters to YouTube watch URLs and a TODAY window
# (today 00:00 local → now). Used by both web (`_home_compute`) and CLI
# (`cmd_today`).


def _fetch_youtube_today(now: datetime, n: int = 5) -> list[dict]:
    """YouTube videos abiertos en Chrome HOY (today 00:00 local → now).

    Same shape as `_fetch_youtube_watched` (see `web/server.py`):
    list of {title, url, video_id, visit_count, last_visit_iso}, dedup
    por video_id, sorted by last_visit DESC.

    Reuses the Chrome history pattern (tmp copy + read-only SQLite + epoch
    conversion). Differs from the 7-day watched fetcher only in the lower
    bound of the visit_time window: hard cut at today_start instead of
    rolling N hours.
    """
    from urllib.parse import parse_qs, urlparse

    src = Path.home() / "Library/Application Support/Google/Chrome/Default/History"
    if not src.is_file():
        return []

    CHROME_EPOCH_OFFSET = 11_644_473_600
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    window_start_chrome = int(
        (today_start.timestamp() + CHROME_EPOCH_OFFSET) * 1_000_000
    )

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=True) as tmp:
        try:
            shutil.copyfile(src, tmp.name)
        except OSError:
            return []
        try:
            conn = sqlite3.connect(f"file:{tmp.name}?mode=ro", uri=True)
        except sqlite3.Error:
            return []
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT u.url AS url, u.title AS title,
                       COUNT(v.id) AS visit_count,
                       MAX(v.visit_time) AS last_visit
                FROM urls u
                JOIN visits v ON v.url = u.id
                WHERE v.visit_time >= ?
                  AND (
                       u.url LIKE '%://www.youtube.com/watch%'
                    OR u.url LIKE '%://youtube.com/watch%'
                    OR u.url LIKE '%://m.youtube.com/watch%'
                    OR u.url LIKE '%://youtu.be/%'
                  )
                GROUP BY u.url
                ORDER BY last_visit DESC
                LIMIT 50
                """,
                (window_start_chrome,),
            ).fetchall()
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    seen_ids: set[str] = set()
    out: list[dict] = []
    for r in rows:
        url = r["url"]
        try:
            parsed = urlparse(url)
            if parsed.netloc.endswith("youtu.be"):
                vid = parsed.path.strip("/").split("/")[0] or url
            else:
                vid = (parse_qs(parsed.query).get("v") or [url])[0]
        except Exception:
            vid = url
        if vid in seen_ids:
            continue
        seen_ids.add(vid)
        raw_title = (r["title"] or "").strip()
        title = (
            raw_title[:-len(" - YouTube")].rstrip()
            if raw_title.endswith(" - YouTube") else raw_title
        )
        last_unix = (r["last_visit"] / 1_000_000) - CHROME_EPOCH_OFFSET
        out.append({
            "title": title or url,
            "url": url,
            "video_id": vid,
            "visit_count": int(r["visit_count"]),
            "last_visit_iso": datetime.fromtimestamp(last_unix).isoformat(timespec="seconds"),
        })
        if len(out) >= n:
            break
    return out
