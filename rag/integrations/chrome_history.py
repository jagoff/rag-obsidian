"""Chrome history ETL — extracted from rag/cross_source_etls.py 2026-05-09.

Reads ``~/Library/Application Support/Google/Chrome/Default/History`` (SQLite,
copied to /tmp because Chrome locks the file while running) and writes a
daily markdown snapshot to
``99-obsidian/99-AI/external-ingest/Chrome/<YYYY-MM-DD>.md`` with the URLs
visited in the last ``hours`` window (default 48h).

Also derives a separate YouTube-watched note (``Chrome/YouTube/<YYYY-MM-DD>.md``)
from URLs matching ``_YOUTUBE_WATCH_RE`` so ``_sync_youtube_transcripts``
(``rag/integrations/youtube.py``) can fetch transcripts without re-parsing
the Chrome history.

Filters:
  - Skip prefixes: ``chrome://``, ``chrome-extension://``, ``about:``,
    ``data:``, etc. (navigation noise that's not retrieval-worthy).
  - Skip search-engine URLs (``google.com/search``, ``bing``, DDG, Brave).
    Search results pages are noise — the user lands on the result, not on
    the SERP.

Silent-fail contract: helpers return ``[]`` /
``{ok: False, reason: "..."}`` instead of raising. ``_atomic_write_if_changed``
and ``_etl_log_swallow`` are lazy-imported from ``rag.cross_source_etls`` to
avoid circular import. ``_YOUTUBE_VAULT_SUBPATH`` + ``_YOUTUBE_WATCH_RE``
are imported lazy from ``rag.integrations.youtube`` for the same reason.

Tests (``tests/test_external_etls.py``) monkeypatch
``rag._CHROME_HISTORY_PATH`` on the top-level ``rag`` module —
``_sync_chrome_history`` re-resolves it via ``sys.modules.get("rag")`` so
the patch propagates regardless of where the function lives.
"""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path

from rag._constants import _EXTERNAL_INGEST_BASE

__all__ = [
    "_CHROME_VAULT_SUBPATH",
    "_CHROME_HISTORY_PATH",
    "_CHROME_FLAVORS",
    "_CHROME_EPOCH_OFFSET_S",
    "_CHROME_SKIP_PREFIXES",
    "_CHROME_SKIP_PATTERNS",
    "_unix_to_chrome_ts",
    "_chrome_history_paths",
    "_read_chrome_visits",
    "_sync_chrome_history",
]

_CHROME_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/Chrome"

_CHROME_HISTORY_PATH = Path.home() / "Library/Application Support/Google/Chrome/Default/History"
# Chrome-family flavors instalados típicamente bajo `~/Library/Application Support/`.
# Cada uno puede tener varios profiles ("Default", "Profile 1", "Profile 2", ...).
# El label vacío significa Chrome stable (no se prefija el label en la attribution);
# los demás se prefijan ("Canary/Default", "Beta/Profile 1") para no confundir.
_CHROME_FLAVORS: tuple[tuple[str, str], ...] = (
    ("Google/Chrome", ""),
    ("Google/Chrome Canary", "Canary"),
    ("Google/Chrome Beta", "Beta"),
    ("Google/Chrome Dev", "Dev"),
    ("Chromium", "Chromium"),
    ("BraveSoftware/Brave-Browser", "Brave"),
)
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


def _unix_to_chrome_ts(unix_s: float) -> int:
    return int((unix_s + _CHROME_EPOCH_OFFSET_S) * 1_000_000)


def _chrome_history_paths() -> list[tuple[str, Path]]:
    """Return ``[(label, History_path), ...]`` para CADA profile de CADA
    Chrome-family browser instalado bajo ``~/Library/Application Support/``.

    Label format:
      - flavor stable + Default → ``"Default"``
      - flavor stable + Profile 1 → ``"Profile 1"``
      - flavor non-stable + Default → ``"Canary/Default"`` (etc.)

    Solo entries cuyo archivo ``History`` existe — Chrome lo crea cuando
    el browser arranca por primera vez.

    Back-compat con tests (``patch.object(rag, "_CHROME_HISTORY_PATH", ...)``):
    si el ``_CHROME_HISTORY_PATH`` resuelto al runtime es DISTINTO al
    default real (i.e. monkeypatched a un tmp_path), retornamos solo ese
    path como single-entry (test isolation tight). En real usage el
    constant es el default → walk all flavors.
    """
    import sys
    rag_mod = sys.modules.get("rag")
    legacy = getattr(rag_mod, "_CHROME_HISTORY_PATH", _CHROME_HISTORY_PATH)
    try:
        is_default = Path(legacy).resolve(strict=False) == _CHROME_HISTORY_PATH.resolve(strict=False)
    except (TypeError, ValueError):
        is_default = False
    if not is_default:
        # Monkeypatched or user-overridden — honor it as the single source.
        if Path(legacy).is_file():
            return [("Default", Path(legacy))]
        return []

    base = Path.home() / "Library" / "Application Support"
    found: list[tuple[str, Path]] = []
    for flavor_dir, flavor_label in _CHROME_FLAVORS:
        root = base / flavor_dir
        if not root.is_dir():
            continue
        try:
            children = sorted(root.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            # Chrome profile dirs son SOLO "Default" o "Profile N" (N ≥ 1).
            # Otros (System Profile, GrShaderCache, etc.) NO son profiles.
            if child.name != "Default" and not child.name.startswith("Profile "):
                continue
            hist = child / "History"
            if not hist.is_file():
                continue
            label = f"{flavor_label}/{child.name}" if flavor_label else child.name
            found.append((label, hist))
    return found


def _read_chrome_visits(history_db: Path, hours: int = 48) -> list[dict]:
    """Read distinct URLs visited in the last `hours` from Chrome History.
    Chrome locks the SQLite while the browser runs — we copy to /tmp and read
    the snapshot. Empty list on any error.
    """
    from rag import _chrome_to_unix_ts  # lazy — defined in integrations.chrome_bookmarks
    from rag.cross_source_etls import _etl_log_swallow

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
    from rag.cross_source_etls import _atomic_write_if_changed
    from rag.integrations.youtube import _YOUTUBE_VAULT_SUBPATH, _YOUTUBE_WATCH_RE

    # Iteramos sobre todos los profiles de TODOS los Chrome-family browsers
    # (Default + Profile N de Chrome stable + Canary + Beta + Dev + Chromium
    # + Brave). El user puede tener sesiones simultáneas en varios y queremos
    # capturarlas TODAS para que el brief no tenga blind spots. Dedupe por
    # URL exacta — si dos profiles abrieron la misma URL, gana la más
    # reciente (mayor `ts`).
    #
    # Re-resolve via `sys.modules["rag"]` para que tests que monkeypatchan
    # `rag._chrome_history_paths` propaguen al call site interno (mismo
    # shim pattern que `_CHROME_HISTORY_PATH`).
    _paths_fn = getattr(sys.modules.get("rag"), "_chrome_history_paths", _chrome_history_paths)
    paths = _paths_fn()
    if not paths:
        return {"ok": False, "reason": "no_visits_or_chrome_locked"}
    by_url: dict[str, dict] = {}
    for _label, hist_path in paths:
        for v in _read_chrome_visits(hist_path, hours=hours):
            url = v.get("url")
            if not url:
                continue
            prev = by_url.get(url)
            if prev is None or (v.get("ts") or 0) > (prev.get("ts") or 0):
                by_url[url] = v
    if not by_url:
        return {"ok": False, "reason": "no_visits_or_chrome_locked"}
    visits = sorted(by_url.values(), key=lambda v: v.get("ts") or 0, reverse=True)
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
