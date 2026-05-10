"""Chrome bookmarks indexing — extracted from `rag/__init__.py` (Wave 7 split, 2026-05-10).

Chrome guarda bookmarks como un JSON tree por profile en
`~/Library/Application Support/Google/Chrome/<Profile>/Bookmarks`. Walkeamos
el tree, flatten a `{url, title, folder_breadcrumb, date_added}`, y escribimos
al URL collection (`source="bookmark"`) — así una query semántica surfacea
notas Y bookmarks en un ranking unificado.

## Public API

- `chrome_bookmark_files(root=None)` → list[(profile_name, bookmarks_path)]
- `parse_chrome_bookmarks(path)` → list[dict]
- `_webkit_ts_to_iso(ts)` → str (ISO-8601 from Chrome's microseconds-since-1601)
- `_bookmark_embed_text(title, folder_breadcrumb, url)` → str (input para embed)
- `_index_chrome_bookmarks(col_urls, profile, bookmarks, batch_size=256)` → int
- `sync_chrome_bookmarks(profile=None)` → dict (resumen por profile)

## Diferencia con `rag/integrations/chrome_bookmarks.py`

`rag/integrations/chrome_bookmarks.py` tiene el **cross-source ETL path**:
`_fetch_chrome_bookmarks_used` (top-N bookmarked URLs visited en últimas N
horas) + `_chrome_to_unix_ts` + `_chrome_bookmarks_root`. Es el path que
alimenta señales de "anticipatory agent" / brief.

Este módulo (`rag/chrome_bookmarks_index.py`) es el **indexing path**:
parsea el tree completo y mete cada bookmark al URL collection para que
`rag links` / retrieval cross-source los surface. Distinto consumer.

## Re-export

`rag/__init__.py` re-exporta vía `from rag.chrome_bookmarks_index import (...)`
con nombres explícitos. Tests (`tests/test_bookmarks.py`) hacen
`monkeypatch.setattr(rag, "_chrome_bookmarks_root", ...)` y
`monkeypatch.setattr(rag, "get_urls_db", ...)` — siguen funcionando porque
las funciones acá leen ambos via `rag.X` deferred (re-resolve por turno).

## Dependencias deferred

`_chrome_bookmarks_root`, `embed`, `get_urls_db`, `SqliteVecCollection` viven
en `rag.__init__` y se importan deferred dentro de cada función para evitar
circular import.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

__all__ = [
    "chrome_bookmark_files",
    "parse_chrome_bookmarks",
    "_webkit_ts_to_iso",
    "_bookmark_embed_text",
    "_index_chrome_bookmarks",
    "sync_chrome_bookmarks",
]


def chrome_bookmark_files(root: Path | None = None) -> list[tuple[str, Path]]:
    """Return (profile_name, bookmarks_path) tuples for every Chrome profile
    that has a Bookmarks file. Empty list if Chrome is not installed.
    """
    import rag  # noqa: PLC0415

    base = root or rag._chrome_bookmarks_root()
    if not base.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        bm = child / "Bookmarks"
        if bm.is_file():
            found.append((child.name, bm))
    return found


def parse_chrome_bookmarks(path: Path) -> list[dict]:
    """Flatten Chrome's Bookmarks JSON tree into [{url, title, folder, date_added}].

    Chrome stores three top-level roots: bookmark_bar, other, synced. Folders
    nest arbitrarily. Each leaf is `{type: url, url, name, date_added}` — date
    is Webkit epoch (microseconds since 1601-01-01). Folders are `{type: folder,
    name, children}`. Invalid JSON or missing structure yields [].
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    roots = (data or {}).get("roots") or {}
    out: list[dict] = []

    def _walk(node: dict, breadcrumb: list[str]) -> None:
        if not isinstance(node, dict):
            return
        ntype = node.get("type")
        if ntype == "url":
            url = (node.get("url") or "").strip()
            if not url or not url.startswith(("http://", "https://")):
                return
            title = (node.get("name") or "").strip()
            out.append({
                "url": url,
                "title": title,
                "folder": " > ".join(breadcrumb),
                "date_added": _webkit_ts_to_iso(node.get("date_added")),
            })
            return
        if ntype == "folder":
            label = (node.get("name") or "").strip()
            children = node.get("children") or []
            sub = breadcrumb + ([label] if label else [])
            for c in children:
                _walk(c, sub)

    for root_key, root_node in roots.items():
        if not isinstance(root_node, dict):
            continue
        label = root_node.get("name") or root_key
        for child in root_node.get("children") or []:
            _walk(child, [str(label)])
    return out


def _webkit_ts_to_iso(ts: object) -> str:
    """Chrome's date_added is microseconds since 1601-01-01 UTC. Convert to
    ISO-8601; return '' on bad input.
    """
    if not ts:
        return ""
    try:
        micros = int(ts)
    except (TypeError, ValueError):
        return ""
    # Seconds between 1601-01-01 and 1970-01-01
    epoch_delta = 11644473600
    secs = micros / 1_000_000 - epoch_delta
    try:
        return datetime.fromtimestamp(secs).isoformat(timespec="seconds")
    except (OSError, OverflowError, ValueError):
        return ""


def _bookmark_embed_text(title: str, folder_breadcrumb: str, url: str) -> str:
    """Render the text that bge-m3 sees. Folder breadcrumb gives topical hints
    the title alone often lacks ('Rust async' + 'Programming > Languages').
    """
    parts = []
    if folder_breadcrumb:
        parts.append(folder_breadcrumb)
    if title:
        parts.append(title)
    # Include bare URL as a weak signal — domain words sometimes carry meaning.
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    if host:
        parts.append(host)
    return " | ".join(parts) if parts else url


def _index_chrome_bookmarks(
    col_urls,
    profile: str,
    bookmarks: list[dict],
    batch_size: int = 256,
) -> int:
    """Replace all `source=bookmark` rows for this Chrome profile with the
    current set. Idempotent — re-running the sync after adding bookmarks
    surfaces only diffs.
    """
    import rag  # noqa: PLC0415

    file_id = f"chrome-bookmark::{profile}"
    existing = col_urls.get(where={"file": file_id}, include=[])
    if existing.get("ids"):
        col_urls.delete(ids=existing["ids"])
    if not bookmarks:
        return 0
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    seen_urls: set[str] = set()
    for bm in bookmarks:
        url = bm["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        ids.append(f"{file_id}::{h}")
        docs.append(_bookmark_embed_text(bm["title"], bm["folder"], url))
        metas.append({
            "file": file_id,
            "note": bm["title"] or url,
            "folder": f"chrome/{profile}",
            "tags": "bookmark",
            "url": url,
            "anchor": bm["title"],
            "line": 0,
            "source": "bookmark",
            "profile": profile,
            "bookmark_folder": bm["folder"],
            "date_added": bm["date_added"],
        })
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_docs = docs[i:i + batch_size]
        batch_metas = metas[i:i + batch_size]
        embeddings = rag.embed(batch_docs)
        col_urls.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_docs,
            metadatas=batch_metas,
        )
    return len(ids)


def sync_chrome_bookmarks(profile: str | None = None) -> dict:
    """Parse Chrome's Bookmarks files for every profile (or only `profile`) and
    replace the corresponding rows in the URL collection. Returns summary dict.
    """
    import rag  # noqa: PLC0415

    col_urls = rag.get_urls_db()
    pairs = chrome_bookmark_files()
    if not pairs:
        return {"profiles": 0, "total": 0, "per_profile": {}}
    if profile is not None:
        pairs = [p for p in pairs if p[0] == profile]
    per_profile: dict[str, int] = {}
    total = 0
    for prof, path in pairs:
        bookmarks = parse_chrome_bookmarks(path)
        n = _index_chrome_bookmarks(col_urls, prof, bookmarks)
        per_profile[prof] = n
        total += n
    return {"profiles": len(pairs), "total": total, "per_profile": per_profile}
