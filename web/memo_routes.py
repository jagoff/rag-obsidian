"""Route registration for the memo dashboard endpoints."""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from fastapi import Depends, Request
from fastapi.responses import FileResponse, RedirectResponse

_memo_cleanup_lock = threading.Lock()


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
    """Borra una o más memorias por id. Llama `memo delete --yes` per id."""
    deleted: list[str] = []
    errors: list[dict] = []
    for mid in ids:
        if not mid or len(mid) < 4:
            errors.append({"id": mid, "error": "id muy corto"})
            continue
        try:
            r = subprocess.run(
                ["memo", "delete", "--yes", mid],
                capture_output=True,
                timeout=30,
            )
            if r.returncode == 0:
                deleted.append(mid)
            else:
                errors.append({"id": mid, "error": r.stderr.decode()[:300]})
        except subprocess.SubprocessError as exc:
            errors.append({"id": mid, "error": f"{type(exc).__name__}: {exc}"})
    try:
        from web.memo_v06 import cache_invalidate

        cache_invalidate()
    except ImportError:
        pass
    return {"ok": True, "deleted": deleted, "errors": errors}


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
            from web.memo_dashboard import note_detail

            a_detail = note_detail(memo_id=a_id)
            b_detail = note_detail(memo_id=b_id)

            if not a_detail.get("ok") or not b_detail.get("ok"):
                skipped.append({
                    "pair": pair,
                    "reason": "memoria no encontrada (probablemente borrada por otro proceso)",
                })
                continue

            a_memo = a_detail.get("memo", {})
            b_memo = b_detail.get("memo", {})
            a_score = a_memo.get("score", 0)
            b_score = b_memo.get("score", 0)
            a_updated = a_memo.get("updated", "")
            b_updated = b_memo.get("updated", "")

            keep_id = a_id
            delete_id = b_id
            if b_score > a_score:
                keep_id, delete_id = b_id, a_id
            elif b_score == a_score and b_updated > a_updated:
                keep_id, delete_id = b_id, a_id

            keep_tags = set(a_memo.get("tags", []) if keep_id == a_id else b_memo.get("tags", []))
            delete_tags = set(
                b_memo.get("tags", []) if keep_id == a_id else a_memo.get("tags", [])
            )
            merged_tags = sorted(list(keep_tags | delete_tags))

            if merged_tags:
                tag_args: list[str] = []
                for tag in merged_tags:
                    tag_args.extend(["-t", tag])
                upd = subprocess.run(
                    ["memo", "update", keep_id, *tag_args],
                    capture_output=True,
                    timeout=30,
                )
                if upd.returncode != 0:
                    warnings.append({
                        "pair": pair,
                        "warning": f"tag merge failed: {upd.stderr.decode()[:1500]}",
                    })

            result = subprocess.run(
                ["memo", "delete", "--yes", delete_id],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                merged.append({"kept": keep_id, "deleted": delete_id})
                deleted_in_batch.add(delete_id)
            else:
                errors.append({"pair": pair, "error": result.stderr.decode()})
        except subprocess.SubprocessError as exc:
            errors.append({"pair": pair, "error": str(exc)})

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
