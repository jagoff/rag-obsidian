"""Route registration for the memo dashboard endpoints."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from fastapi import Depends, Request
from fastapi.responses import FileResponse, RedirectResponse

_memo_cleanup_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _open_memvec_rw() -> sqlite3.Connection:
    from web.memo_dashboard import _HAVE_SQLITE_VEC, _memvec_db

    db = _memvec_db()
    conn = sqlite3.connect(str(db), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    if _HAVE_SQLITE_VEC:
        try:
            import sqlite_vec  # type: ignore

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except Exception:
            pass
    return conn


def _row_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def _history_event(op: str, row: dict, *, delta: dict | None = None) -> str | None:
    from web.memo_dashboard import _history_db

    try:
        db = _history_db()
        db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db), timeout=10.0) as conn:
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, "
                "op TEXT NOT NULL, record_id TEXT NOT NULL, title TEXT, "
                "type TEXT, delta_json TEXT)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_record ON events(record_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_op ON events(op)")
            conn.execute(
                "INSERT INTO events(ts, op, record_id, title, type, delta_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    _now_iso(),
                    op,
                    row.get("id", ""),
                    row.get("title", ""),
                    row.get("type", ""),
                    json.dumps(delta, ensure_ascii=False) if delta else None,
                ),
            )
    except Exception as exc:
        return f"history log failed: {type(exc).__name__}: {exc}"
    return None


def _graph_drop_for_memoria(memo_id: str) -> str | None:
    from web.memo_dashboard import _memo_dir

    db = _memo_dir() / "graph.db"
    if not db.exists():
        return None
    try:
        with sqlite3.connect(str(db), timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("BEGIN IMMEDIATE")
            old = [
                r["entity_id"]
                for r in conn.execute(
                    "SELECT entity_id FROM entity_memoria WHERE memoria_id = ?",
                    (memo_id,),
                ).fetchall()
            ]
            conn.execute("DELETE FROM entity_memoria WHERE memoria_id = ?", (memo_id,))
            for eid in old:
                conn.execute(
                    "UPDATE entities SET mention_count = MAX(0, mention_count - 1) "
                    "WHERE id = ?",
                    (eid,),
                )
            conn.commit()
    except Exception as exc:
        return f"graph cleanup failed: {type(exc).__name__}: {exc}"
    return None


def _delete_memo_direct(memo_id: str) -> tuple[str | None, list[str]]:
    from web.memo_dashboard import _invalidate_caches, _resolve_meta_row, _safe_resolve

    warnings: list[str] = []
    conn = _open_memvec_rw()
    try:
        row = _resolve_meta_row(conn, memo_id)
        if row is None:
            raise ValueError(f"not found or ambiguous: {memo_id}")
        payload = _row_dict(row)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM meta WHERE id = ?", (payload["id"],))
            conn.execute("DELETE FROM vec WHERE id = ?", (payload["id"],))
            conn.execute("DELETE FROM fts WHERE id = ?", (payload["id"],))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    full = _safe_resolve(payload.get("path", ""))
    if full and full.exists():
        try:
            full.unlink()
        except OSError as exc:
            warnings.append(f"file delete failed: {type(exc).__name__}: {exc}")

    for maybe_warning in (
        _history_event("delete", payload),
        _graph_drop_for_memoria(payload["id"]),
    ):
        if maybe_warning:
            warnings.append(maybe_warning)
    _invalidate_caches()
    try:
        from web.memo_v06 import cache_invalidate

        cache_invalidate()
    except ImportError:
        pass
    return payload["id"], warnings


def _update_memo_tags_direct(memo_id: str, tags: list[str]) -> tuple[str, list[str]]:
    from web.memo_dashboard import (
        _body_for_path,
        _invalidate_caches,
        _parse_tags,
        _resolve_meta_row,
        _rewrite_frontmatter,
    )

    warnings: list[str] = []
    tags = sorted({str(t).strip() for t in tags if str(t).strip()})
    conn = _open_memvec_rw()
    try:
        row = _resolve_meta_row(conn, memo_id)
        if row is None:
            raise ValueError(f"not found or ambiguous: {memo_id}")
        payload = _row_dict(row)
        old_tags = _parse_tags(payload.get("tags"))
        updated = _now_iso()
        body = _body_for_path(payload.get("path", ""))
        tags_json = json.dumps(tags, ensure_ascii=False)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "UPDATE meta SET tags = ?, updated = ? WHERE id = ?",
                (tags_json, updated, payload["id"]),
            )
            conn.execute("DELETE FROM fts WHERE id = ?", (payload["id"],))
            conn.execute(
                "INSERT INTO fts(id, title, tags, body) VALUES (?, ?, ?, ?)",
                (payload["id"], payload.get("title", ""), " ".join(tags), body),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    payload["updated"] = updated
    try:
        _rewrite_frontmatter(payload, tags, updated)
    except OSError as exc:
        warnings.append(f"frontmatter update failed: {type(exc).__name__}: {exc}")
    maybe_warning = _history_event("update", payload, delta={"tags": [old_tags, tags]})
    if maybe_warning:
        warnings.append(maybe_warning)
    _invalidate_caches()
    return payload["id"], warnings


def _memo_score_for_merge(row: dict) -> tuple[int, str]:
    from web.memo_dashboard import _parse_iso, _parse_tags, _read_body_meta, _score_memo

    tags = _parse_tags(row.get("tags"))
    size, _ = _read_body_meta(row.get("path", ""))
    scored = _score_memo(
        type_=(row.get("type") or "").lower(),
        tags=tags,
        body_size=size,
        updated_dt=_parse_iso(row.get("updated")),
        in_dupe_cluster=True,
    )
    return int(scored["score"]), str(row.get("updated") or "")


def memo_api(limit: int = 30, type: str | None = None) -> dict:
    """Snapshot de las memorias del MCP `memo`."""
    from web.memo_dashboard import snapshot

    return snapshot(limit=limit, type_filter=type)


def memo_note_api(id: str | None = None, path: str | None = None) -> dict:
    """Detalle de una memoria."""
    from web.memo_dashboard import note_detail

    return note_detail(memo_id=id, path=path)


def memo_search_api(q: str = "", limit: int = 20) -> dict:
    """FTS5 search sobre memo."""
    from web.memo_dashboard import search

    return search(query=q, limit=limit)


def memo_timemachine_snapshot_api(
    date: str | None = None,
    type: str | None = None,
    limit: int = 200,
) -> dict:
    """Time-machine: corpus reconstruido al `date`."""
    from web.memo_v06 import snapshot

    return snapshot(date=date, type_=type, limit=limit)


def memo_timemachine_diff_api(from_date: str, to_date: str | None = None) -> dict:
    """Time-machine diff entre dos fechas."""
    from web.memo_v06 import timemachine_diff

    return timemachine_diff(from_date=from_date, to_date=to_date)


def memo_graph_api(
    limit_nodes: int = 80,
    min_count: int = 2,
    type: str | None = None,
) -> dict:
    """Knowledge graph de memo."""
    from web.memo_v06 import graph

    return graph(limit_nodes=limit_nodes, min_count=min_count, type_filter=type)


def memo_temporal_timeline_api(days: int = 30) -> dict:
    """Timeline saves/updates/deletes por día."""
    from web.memo_v06 import temporal_timeline

    return temporal_timeline(days=days)


def memo_temporal_stale_api(days: int = 90, limit: int = 30) -> dict:
    """Memorias sin update >N días."""
    from web.memo_v06 import temporal_stale

    return temporal_stale(days_threshold=days, limit=limit)


def memo_delete_api(ids: list[str]) -> dict:
    """Borra una o más memorias por id."""
    deleted: list[str] = []
    errors: list[dict] = []
    warnings: list[dict] = []
    for mid in ids:
        if not mid or len(mid) < 4:
            errors.append({"id": mid, "error": "id muy corto"})
            continue
        try:
            deleted_id, item_warnings = _delete_memo_direct(mid)
            if deleted_id:
                deleted.append(deleted_id)
            for warning in item_warnings:
                warnings.append({"id": deleted_id or mid, "warning": warning})
        except Exception as exc:
            errors.append({"id": mid, "error": f"{type(exc).__name__}: {exc}"})
    return {"ok": True, "deleted": deleted, "errors": errors, "warnings": warnings}


def memo_cleanup_api(apply_dead: bool = True, apply_dupes: bool = False) -> dict:
    """Cleanup periódico de memo."""
    if not _memo_cleanup_lock.acquire(blocking=False):
        return {"ok": False, "error": "cleanup ya está corriendo"}
    try:
        repo = Path(__file__).resolve().parent.parent
        scripts_dir = repo / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from memo_cleanup import run_cleanup  # type: ignore

        results = run_cleanup(
            dry_run=False,
            verbose=False,
            apply_dead=apply_dead,
            apply_dupes=apply_dupes,
        )
        try:
            from web.memo_dashboard import _snapshot_cache

            if hasattr(_snapshot_cache, "clear"):
                _snapshot_cache.clear()
        except ImportError:
            pass
        try:
            from web.memo_v06 import cache_invalidate

            cache_invalidate()
        except ImportError:
            pass
        return {"ok": True, "results": results}
    except ImportError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        _memo_cleanup_lock.release()


def memo_merge_api(pairs: list[dict[str, str]]) -> dict:
    """Fusiona pares de near-dupes de memo."""
    merged: list[dict] = []
    errors: list[dict] = []
    warnings: list[dict] = []
    skipped: list[dict] = []
    deleted_in_batch: set[str] = set()

    for pair in pairs:
        a_id = pair.get("a")
        b_id = pair.get("b")
        if not a_id or not b_id:
            errors.append({"pair": pair, "error": "missing id"})
            continue

        if a_id in deleted_in_batch or b_id in deleted_in_batch:
            skipped.append({
                "pair": pair,
                "reason": "una memoria del par ya fue borrada en otro par overlapping",
            })
            continue

        try:
            from web.memo_dashboard import _parse_tags, _resolve_meta_row

            conn = _open_memvec_rw()
            try:
                a_row = _resolve_meta_row(conn, a_id)
                b_row = _resolve_meta_row(conn, b_id)
            finally:
                conn.close()

            if a_row is None or b_row is None:
                skipped.append({
                    "pair": pair,
                    "reason": "memoria no encontrada (probablemente borrada por otro proceso)",
                })
                continue

            a_memo = _row_dict(a_row)
            b_memo = _row_dict(b_row)
            a_score, a_updated = _memo_score_for_merge(a_memo)
            b_score, b_updated = _memo_score_for_merge(b_memo)

            keep_id = a_id
            delete_id = b_id
            if b_score > a_score:
                keep_id, delete_id = b_id, a_id
            elif b_score == a_score and b_updated > a_updated:
                keep_id, delete_id = b_id, a_id

            keep_raw_tags = a_memo.get("tags") if keep_id == a_id else b_memo.get("tags")
            delete_raw_tags = b_memo.get("tags") if keep_id == a_id else a_memo.get("tags")
            keep_tags = set(_parse_tags(keep_raw_tags))
            delete_tags = set(_parse_tags(delete_raw_tags))
            merged_tags = sorted(list(keep_tags | delete_tags))

            if merged_tags:
                _, item_warnings = _update_memo_tags_direct(keep_id, merged_tags)
                for warning in item_warnings:
                    warnings.append({"pair": pair, "warning": warning})

            deleted_id, item_warnings = _delete_memo_direct(delete_id)
            if deleted_id:
                merged.append({"kept": keep_id, "deleted": deleted_id})
                deleted_in_batch.add(deleted_id)
                for warning in item_warnings:
                    warnings.append({"pair": pair, "warning": warning})
            else:
                errors.append({"pair": pair, "error": f"delete failed: {delete_id}"})
        except Exception as exc:
            errors.append({"pair": pair, "error": f"{type(exc).__name__}: {exc}"})

    return {
        "ok": True,
        "merged": merged,
        "errors": errors,
        "warnings": warnings,
        "skipped": skipped,
    }


async def _mem_vault_legacy_redirect(request: Request, path: str = ""):
    """Legacy: /mem-vault/... -> /memo."""
    qs = request.url.query
    target = "/memo"
    if qs:
        target = f"{target}?{qs}"
    return RedirectResponse(url=target, status_code=308)


async def _memory_legacy_redirect(request: Request, path: str = ""):
    """Legacy: /memory/... -> /memo."""
    qs = request.url.query
    target = "/memo"
    if qs:
        target = f"{target}?{qs}"
    return RedirectResponse(url=target, status_code=308)


def register_memo_routes(
    app,
    static_dir: Path,
    require_admin_token: Callable,
) -> dict[str, object]:
    """Register memo routes and return handlers for web.server compatibility."""
    static_dir = Path(static_dir)

    def memo_page() -> FileResponse:
        """HTML del dashboard /memo. Hidrata via /api/memo."""
        return FileResponse(static_dir / "memo.html")

    app.get("/memo")(memo_page)
    app.get("/api/memo")(memo_api)
    app.get("/api/memo/note")(memo_note_api)
    app.get("/api/memo/search")(memo_search_api)
    app.get("/api/memo/timemachine/snapshot")(memo_timemachine_snapshot_api)
    app.get("/api/memo/timemachine/diff")(memo_timemachine_diff_api)
    app.get("/api/memo/graph")(memo_graph_api)
    app.get("/api/memo/temporal/timeline")(memo_temporal_timeline_api)
    app.get("/api/memo/temporal/stale")(memo_temporal_stale_api)
    app.post("/api/memo/delete", dependencies=[Depends(require_admin_token)])(memo_delete_api)
    app.post("/api/memo/cleanup", dependencies=[Depends(require_admin_token)])(memo_cleanup_api)
    app.post("/api/memo/merge", dependencies=[Depends(require_admin_token)])(memo_merge_api)
    app.get("/mem-vault")(_mem_vault_legacy_redirect)
    app.get("/mem-vault/{path:path}")(_mem_vault_legacy_redirect)
    app.get("/memory")(_memory_legacy_redirect)
    app.get("/memory/{path:path}")(_memory_legacy_redirect)

    return {
        "memo_page": memo_page,
        "memo_api": memo_api,
        "memo_note_api": memo_note_api,
        "memo_search_api": memo_search_api,
        "memo_timemachine_snapshot_api": memo_timemachine_snapshot_api,
        "memo_timemachine_diff_api": memo_timemachine_diff_api,
        "memo_graph_api": memo_graph_api,
        "memo_temporal_timeline_api": memo_temporal_timeline_api,
        "memo_temporal_stale_api": memo_temporal_stale_api,
        "memo_delete_api": memo_delete_api,
        "memo_cleanup_api": memo_cleanup_api,
        "memo_merge_api": memo_merge_api,
        "_mem_vault_legacy_redirect": _mem_vault_legacy_redirect,
        "_memory_legacy_redirect": _memory_legacy_redirect,
    }
