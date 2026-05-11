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

    Multi-profile (2026-05-11): iteramos cada `(profile, Bookmarks, History)`
    pair de cada Chrome-family browser. Bookmarks de TODOS los profiles se
    funden; visits del MISMO profile que el bookmark dominan (no asumimos
    que Canary y stable comparten visits). Dedupe final por URL.
    """
    # Pair Bookmarks + History por profile. Solo profiles donde AMBOS
    # archivos existen entran al join.
    from rag.integrations.chrome_history import _CHROME_FLAVORS
    profile_pairs: list[tuple[Path, Path]] = []
    base = Path.home() / "Library" / "Application Support"
    for flavor_dir, _label in _CHROME_FLAVORS:
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
            if child.name != "Default" and not child.name.startswith("Profile "):
                continue
            bm = child / "Bookmarks"
            hist = child / "History"
            if bm.is_file() and hist.is_file():
                profile_pairs.append((bm, hist))
    if not profile_pairs:
        return []

    # Cargamos la primera pair "primaria" (Default de Chrome stable, casi
    # siempre primero por orden de _CHROME_FLAVORS + iterdir). Mantenemos
    # el shape del JSON tree de la primera para back-compat de tests
    # antiguos que asumían un solo árbol. Las demás pairs se procesan
    # adelante en el merge.
    bm_path, hist_path = profile_pairs[0]
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

    # Merge bookmarks from other profiles too — el user puede tener una
    # cuenta personal en Default y otra de laburo en Profile 1.
    for extra_bm, _extra_hist in profile_pairs[1:]:
        try:
            extra_tree = json.loads(extra_bm.read_text(encoding="utf-8"))
        except Exception:
            continue
        for root in (extra_tree.get("roots") or {}).values():
            _walk(root, "")

    if not bookmarks:
        return []

    CHROME_EPOCH_OFFSET = 11_644_473_600
    now_ts = time.time()
    window_chrome = int((now_ts - hours * 3600 + CHROME_EPOCH_OFFSET) * 1_000_000)

    # Por cada History DB del profile_pairs, mergeamos rows. El visit_count
    # se suma (mismo URL visto en 2 profiles = más intent total); el
    # last_visit gana el máximo.
    by_url: dict[str, dict] = {}
    for _bm_path, profile_hist in profile_pairs:
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=True) as tmp:
            try:
                shutil.copyfile(profile_hist, tmp.name)
            except OSError:
                continue
            try:
                conn = sqlite3.connect(f"file:{tmp.name}?mode=ro", uri=True)
            except sqlite3.Error:
                continue
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
            except sqlite3.Error:
                rows = []
            finally:
                conn.close()
        for r in rows:
            url = r["url"]
            if url not in bookmarks:
                continue
            prev = by_url.get(url)
            vc = int(r["visit_count"] or 0)
            lv = int(r["last_visit"] or 0)
            if prev is None:
                by_url[url] = {"visit_count": vc, "last_visit": lv}
            else:
                prev["visit_count"] += vc
                if lv > prev["last_visit"]:
                    prev["last_visit"] = lv

    ranked = sorted(
        by_url.items(),
        key=lambda kv: kv[1]["last_visit"],
        reverse=True,
    )

    out: list[dict] = []
    for url, agg in ranked:
        meta = bookmarks[url]
        last_unix = (agg["last_visit"] / 1_000_000) - CHROME_EPOCH_OFFSET
        out.append({
            "name": meta["name"],
            "url": url,
            "folder": meta["folder"],
            "visit_count": agg["visit_count"],
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

    from rag.integrations.chrome_history import _chrome_history_paths

    paths = _chrome_history_paths()
    if not paths:
        return []

    CHROME_EPOCH_OFFSET = 11_644_473_600
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    window_start_chrome = int(
        (today_start.timestamp() + CHROME_EPOCH_OFFSET) * 1_000_000
    )

    # Mergeamos rows de todos los profiles. Cada profile → snapshot a tmp
    # + query. Falla silenciosa por profile (Chrome stable abierto pero
    # Canary disponible, etc.) — el peor caso es []. Cada try/except
    # loguea via _silent_log para que el operador pueda diagnosticar.
    rows: list = []
    for _label, src in paths:
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=True) as tmp:
            try:
                shutil.copyfile(src, tmp.name)
            except OSError as exc:
                # Bug Hunt 2026-05-08 (M Int chrome_bookmarks): pre-fix los
                # 3 except branches swallowing total sin traza. Cuando un
                # profile de Chrome tenía la History DB locked (browser
                # corriendo + WAL mode), el panel YouTube/bookmarks devolvía
                # empty sin razón visible → operador no podía diferenciar
                # "user no usó YT hoy" de "Chrome bloqueando la copy".
                # Compat lazy import. Continuamos al siguiente profile —
                # multi-profile (2026-05-11) no aborta si un profile rompe.
                try:
                    from rag import _silent_log
                    _silent_log("chrome_youtube_copy_failed", exc)
                except Exception:
                    pass
                continue
            try:
                conn = sqlite3.connect(f"file:{tmp.name}?mode=ro", uri=True)
            except sqlite3.Error as exc:
                try:
                    from rag import _silent_log
                    _silent_log("chrome_youtube_connect_failed", exc)
                except Exception:
                    pass
                continue
            try:
                conn.row_factory = sqlite3.Row
                rows.extend(conn.execute(
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
                ).fetchall())
            except sqlite3.Error as exc:
                try:
                    from rag import _silent_log
                    _silent_log("chrome_youtube_query_failed", exc)
                except Exception:
                    pass
            finally:
                conn.close()

    # Sort rows across profiles by last_visit desc before dedup so the
    # video_id ganador es el más recientemente visto en cualquier profile.
    rows.sort(key=lambda r: r["last_visit"] or 0, reverse=True)

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


def _fetch_chrome_today_domains(now: datetime, top_n: int = 8) -> list[dict]:
    """URLs visitados HOY (00:00 → now) agrupados por dominio. Una sola
    fila por dominio con su top title como sample, para el evening brief.

    Reusa la lectura snapshotted-to-/tmp de Chrome History — Chrome
    bloquea el DB mientras está abierto. Filtra los mismos prefixes /
    SERPs que `_read_chrome_visits` (chrome://, google.com/search, etc.)
    via re-uso de `_CHROME_SKIP_PREFIXES` + `_CHROME_SKIP_PATTERNS`.

    Returns list of `{domain, visits, sample_title, sample_url}` ordered
    por visits DESC. Empty list cuando no hay browsing hoy o Chrome
    locked / no instalado.
    """
    from urllib.parse import urlparse

    from rag.integrations.chrome_history import (
        _CHROME_SKIP_PREFIXES,
        _CHROME_SKIP_PATTERNS,
        _chrome_history_paths,
    )

    paths = _chrome_history_paths()
    if not paths:
        return []

    # Local epoch offset (igual que `_fetch_youtube_today` arriba — evita
    # tocar el import de `rag` durante boot del package).
    CHROME_EPOCH_OFFSET = 11_644_473_600
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    window_start_chrome = int(
        (today_start.timestamp() + CHROME_EPOCH_OFFSET) * 1_000_000
    )

    # Mergeamos rows de cada profile (multi-profile 2026-05-11). Cada
    # profile → snapshot + query — fail silencioso por profile.
    rows: list = []
    for _label, src in paths:
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=True) as tmp:
            try:
                shutil.copyfile(src, tmp.name)
            except OSError as exc:
                try:
                    from rag import _silent_log
                    _silent_log("chrome_today_copy_failed", exc)
                except Exception:
                    pass
                continue
            try:
                conn = sqlite3.connect(f"file:{tmp.name}?mode=ro", uri=True)
            except sqlite3.Error as exc:
                try:
                    from rag import _silent_log
                    _silent_log("chrome_today_connect_failed", exc)
                except Exception:
                    pass
                continue
            try:
                conn.row_factory = sqlite3.Row
                rows.extend(conn.execute(
                    """
                    SELECT u.url AS url, u.title AS title,
                           COUNT(v.id) AS visit_count
                    FROM urls u
                    JOIN visits v ON v.url = u.id
                    WHERE v.visit_time >= ?
                    GROUP BY u.url
                    ORDER BY visit_count DESC
                    LIMIT 200
                    """,
                    (window_start_chrome,),
                ).fetchall())
            except sqlite3.Error as exc:
                try:
                    from rag import _silent_log
                    _silent_log("chrome_today_query_failed", exc)
                except Exception:
                    pass
            finally:
                conn.close()

    by_domain: dict[str, dict] = {}
    for r in rows:
        url = (r["url"] or "").strip()
        if not url:
            continue
        if any(url.startswith(p) for p in _CHROME_SKIP_PREFIXES):
            continue
        if any(p.match(url) for p in _CHROME_SKIP_PATTERNS):
            continue
        try:
            host = urlparse(url).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
        except Exception:
            continue
        if not host:
            continue
        bucket = by_domain.get(host)
        title = (r["title"] or "").strip() or url
        visits = int(r["visit_count"] or 0)
        if bucket is None:
            by_domain[host] = {
                "domain": host,
                "visits": visits,
                "sample_title": title,
                "sample_url": url,
            }
        else:
            bucket["visits"] += visits
            # Keep the first (most-visited) title as sample.
    ranked = sorted(by_domain.values(), key=lambda d: d["visits"], reverse=True)
    return ranked[:top_n]
