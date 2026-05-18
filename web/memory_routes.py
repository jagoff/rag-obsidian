"""Read-only routes for unified memory summaries."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field


_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100


class DistillConversationsRequest(BaseModel):
    apply: bool = False
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    limit: int = Field(10, ge=1, le=100)
    require_missing_source: bool = True


def _utc_iso_from_timestamp(ts: float) -> str:
    return (
        datetime.fromtimestamp(ts, timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00",
        "Z",
    )


def _coerce_limit(limit: Any) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = _DEFAULT_LIMIT
    return max(1, min(value, _MAX_LIMIT))


def _error_section(exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "count": 0,
        "latest": [],
        "error": f"{type(exc).__name__}: {exc}",
    }


def _count_from_snapshot(snapshot: dict[str, Any]) -> int:
    candidates = [
        snapshot.get("count"),
        (snapshot.get("totals") or {}).get("all")
        if isinstance(snapshot.get("totals"), dict)
        else None,
    ]
    for candidate in candidates:
        try:
            return max(0, int(candidate))
        except (TypeError, ValueError):
            continue
    recent = snapshot.get("recent")
    if isinstance(recent, list):
        return len(recent)
    return 0


def _memo_section(limit: int) -> dict[str, Any]:
    try:
        from web import memo_dashboard

        snapshot = memo_dashboard.snapshot(limit=limit)
        if not isinstance(snapshot, dict):
            return {
                "ok": False,
                "count": 0,
                "latest": [],
                "error": f"memo snapshot returned {type(snapshot).__name__}",
            }
        ok = bool(snapshot.get("ok", True))
        recent = snapshot.get("recent")
        latest = recent[:limit] if isinstance(recent, list) else []
        section = dict(snapshot)
        section["ok"] = ok
        section["count"] = _count_from_snapshot(snapshot) if ok else 0
        section["latest"] = latest
        section["snapshot"] = snapshot
        if not ok:
            section.setdefault("error", "memo snapshot unavailable")
        return section
    except Exception as exc:
        return _error_section(exc)


def _conversations_section(limit: int) -> dict[str, Any]:
    try:
        import rag

        with rag._ragvec_state_conn() as conn:
            count_row = conn.execute(
                "SELECT COUNT(*) FROM rag_conversations_index",
            ).fetchone()
            rows = conn.execute(
                """
                SELECT session_id, relative_path, updated_at
                FROM rag_conversations_index
                ORDER BY updated_at DESC, session_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return {
            "ok": True,
            "count": int(count_row[0] if count_row else 0),
            "latest": [
                {
                    "session_id": row[0],
                    "relative_path": row[1],
                    "updated_at": row[2],
                }
                for row in rows
            ],
        }
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return {"ok": True, "count": 0, "latest": []}
        return _error_section(exc)
    except Exception as exc:
        return _error_section(exc)


def _runbooks_section(limit: int) -> dict[str, Any]:
    try:
        import rag
        from rag.conversation_distiller import RUNBOOKS_DIR

        vault_path = Path(rag.VAULT_PATH)
        runbooks_root = vault_path / RUNBOOKS_DIR
        if not vault_path.is_dir() or not runbooks_root.is_dir():
            return {
                "ok": True,
                "count": 0,
                "latest": [],
                "root": str(runbooks_root),
            }

        rows: list[dict[str, Any]] = []
        for path in runbooks_root.rglob("*.md"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
                relative_path = str(path.relative_to(vault_path))
            except OSError:
                continue
            rows.append(
                {
                    "relative_path": relative_path,
                    "updated_at": _utc_iso_from_timestamp(stat.st_mtime),
                    "mtime": stat.st_mtime,
                    "bytes": stat.st_size,
                },
            )
        rows.sort(key=lambda row: (row["mtime"], row["relative_path"]), reverse=True)
        return {
            "ok": True,
            "count": len(rows),
            "latest": rows[:limit],
            "root": str(runbooks_root),
        }
    except Exception as exc:
        return _error_section(exc)


def _summary_for_sections(sections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "memo": int(sections["memo"].get("count") or 0),
        "conversations": int(sections["conversations"].get("count") or 0),
        "runbooks": int(sections["runbooks"].get("count") or 0),
    }
    errors = [name for name, section in sections.items() if not section.get("ok")]
    if errors:
        status = "partial"
    elif any(counts.values()):
        status = "ok"
    else:
        status = "empty"
    return {
        "status": status,
        "counts": counts,
        "memo_count": counts["memo"],
        "conversations_count": counts["conversations"],
        "runbooks_count": counts["runbooks"],
        "errors": errors,
    }


def api_memory_unified(limit: int = _DEFAULT_LIMIT) -> dict[str, Any]:
    limit = _coerce_limit(limit)
    sections = {
        "memo": _memo_section(limit),
        "conversations": _conversations_section(limit),
        "runbooks": _runbooks_section(limit),
    }
    return {
        "ok": True,
        "generated_at": _utc_now_iso(),
        "summary": _summary_for_sections(sections),
        "sections": sections,
    }


def api_memory_distill_conversations(req: DistillConversationsRequest) -> dict[str, Any]:
    try:
        import rag
        from rag.conversation_distiller import run_distillation

        result = run_distillation(
            Path(rag.VAULT_PATH),
            apply=bool(req.apply),
            min_confidence=float(req.min_confidence),
            limit=int(req.limit),
            require_missing_source=bool(req.require_missing_source),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"conversation distillation failed: {exc}") from exc
    return {
        "ok": True,
        "generated_at": _utc_now_iso(),
        "result": result,
    }


def register_memory_routes(app, require_admin_token=None) -> dict[str, object]:
    admin_dep = [Depends(require_admin_token)] if require_admin_token is not None else []
    app.get("/api/memory/unified")(api_memory_unified)
    app.post(
        "/api/memory/distill-conversations",
        dependencies=admin_dep,
    )(api_memory_distill_conversations)
    return {
        "api_memory_unified": api_memory_unified,
        "api_memory_distill_conversations": api_memory_distill_conversations,
    }
