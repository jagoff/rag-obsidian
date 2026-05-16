"""Side-effect action routes for reminders, calendar, model unload, and VLM."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime as _dt
from pathlib import Path as _Path
from typing import Callable

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel

__all__ = [
    "ollama_unload",
    "ollama_restart_compat",
    "ReminderCreateRequest",
    "create_reminder",
    "CalendarCreateRequest",
    "create_calendar_event",
    "VlmDescribeRequest",
    "vlm_describe",
    "ReminderCompleteRequest",
    "complete_reminder",
    "ReminderDeleteRequest",
    "delete_reminder",
    "CalendarDeleteRequest",
    "delete_calendar_event",
    "ActionRouteDeps",
    "register_action_routes",
]


@dataclass(frozen=True)
class ActionRouteDeps:
    """Callbacks supplied by the live server module at registration time."""

    rate_limit_behavior: Callable[[Request], None]
    bust_home_cache: Callable[[], None]


_DEPS: ActionRouteDeps | None = None


def _deps() -> ActionRouteDeps:
    if _DEPS is None:
        raise RuntimeError("action routes not registered")
    return _DEPS


def _rate_limit_behavior(request: Request) -> None:
    _deps().rate_limit_behavior(request)


def _bust_home_cache() -> None:
    try:
        _deps().bust_home_cache()
    except Exception as exc:
        try:
            import rag as _rag_mod  # noqa: PLC0415

            _rag_mod._silent_log("home_cache_bust", exc)
        except ImportError:
            pass


def ollama_unload() -> dict:
    """Evict in-process chat models and drop the reranker if possible."""
    freed = []
    try:
        from rag.llm_backend import get_backend

        mlx_backend = get_backend()
        mlx_loaded = list(getattr(mlx_backend, "_loaded", {}).keys())
        if mlx_backend.unload(None):
            freed.extend(mlx_loaded or ["mlx"])
    except (ImportError, RuntimeError) as exc:
        freed.append(f"mlx_clear (fail: {exc})")
    try:
        import rag as _rag
        from rag import maybe_unload_reranker

        _rag._reranker_last_use = 0.0
        reranker_dropped = maybe_unload_reranker()
    except ImportError:
        reranker_dropped = False
    return {"ok": True, "freed_models": freed, "reranker_dropped": reranker_dropped}


def ollama_restart_compat() -> dict:
    """Compat endpoint for old panic-button clients."""
    raise HTTPException(
        status_code=410,
        detail="Ollama restart no existe post-MLX; usa /api/ollama/unload.",
    )


class ReminderCreateRequest(BaseModel):
    text: str
    due: str | None = None
    due_iso: str | None = None
    list: str | None = None
    priority: int | None = None
    notes: str | None = None
    recurrence: dict | None = None


def create_reminder(req: ReminderCreateRequest, request: Request) -> dict:
    _rate_limit_behavior(request)
    from rag import _create_reminder

    due_dt: _dt | None = None
    if req.due_iso:
        try:
            due_dt = _dt.fromisoformat(req.due_iso)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"due_iso invalido: {exc}") from exc

    ok, res = _create_reminder(
        req.text,
        due_token=req.due if not due_dt else None,
        list_name=req.list,
        due_dt=due_dt,
        priority=req.priority,
        notes=req.notes,
        recurrence=req.recurrence,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=res)
    _bust_home_cache()
    return {"ok": True, "id": res}


class CalendarCreateRequest(BaseModel):
    title: str
    start_iso: str
    end_iso: str | None = None
    calendar: str | None = None
    location: str | None = None
    notes: str | None = None
    all_day: bool = False
    recurrence: dict | None = None


def create_calendar_event(req: CalendarCreateRequest, request: Request) -> dict:
    _rate_limit_behavior(request)
    from rag import _create_calendar_event

    try:
        start_dt = _dt.fromisoformat(req.start_iso)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"start_iso invalido: {exc}") from exc
    end_dt: _dt | None = None
    if req.end_iso:
        try:
            end_dt = _dt.fromisoformat(req.end_iso)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"end_iso invalido: {exc}") from exc

    ok, res = _create_calendar_event(
        req.title,
        start_dt,
        end_dt,
        calendar=req.calendar,
        location=req.location,
        notes=req.notes,
        all_day=req.all_day,
        recurrence=req.recurrence,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=res)
    return {"ok": True, "uid": res}


class VlmDescribeRequest(BaseModel):
    path: str


def vlm_describe(req: VlmDescribeRequest, request: Request) -> dict:
    _rate_limit_behavior(request)
    try:
        raw = _Path(req.path).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=f"path invalido: {exc}") from exc
    if not raw.is_file():
        raise HTTPException(status_code=404, detail="path no es archivo")

    import os as _os  # noqa: PLC0415
    _vault_root = _Path(
        _os.environ.get(
            "OBSIDIAN_RAG_VAULT",
            _Path.home()
            / "Library"
            / "Mobile Documents"
            / "iCloud~md~obsidian"
            / "Documents"
            / "Notes",
        )
    ).resolve()
    allowed_roots = [
        _Path.home() / "repos" / "whatsapp-mcp" / "whatsapp-bridge" / "store",
        _vault_root / "00-Inbox" / "attachments",
    ]
    inside = False
    for root in allowed_roots:
        try:
            raw.relative_to(root.resolve())
            inside = True
            break
        except ValueError:
            continue
    if not inside:
        raise HTTPException(
            status_code=403,
            detail="path fuera de allowlist (bridge media + vault attachments only)",
        )

    if raw.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".gif", ".bmp"}:
        raise HTTPException(status_code=400, detail=f"extension no soportada: {raw.suffix}")

    from rag.ocr import _image_text_or_caption

    try:
        text, source = _image_text_or_caption(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"_image_text_or_caption fallo: {exc}",
        ) from exc
    return {"text": text, "source": source, "path": str(raw)}


class ReminderCompleteRequest(BaseModel):
    reminder_id: str


def complete_reminder(req: ReminderCompleteRequest) -> dict:
    from rag import _complete_reminder

    ok, msg = _complete_reminder(req.reminder_id)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    _bust_home_cache()
    return {"ok": True, "message": msg}


class ReminderDeleteRequest(BaseModel):
    reminder_id: str


def delete_reminder(req: ReminderDeleteRequest) -> dict:
    from rag import _delete_reminder

    ok, msg = _delete_reminder(req.reminder_id)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    _bust_home_cache()
    return {"ok": True, "message": msg}


class CalendarDeleteRequest(BaseModel):
    event_uid: str


def delete_calendar_event(req: CalendarDeleteRequest) -> dict:
    from rag import _delete_calendar_event

    ok, msg = _delete_calendar_event(req.event_uid)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"ok": True, "message": msg}


def register_action_routes(
    app,
    require_admin_token,
    deps: ActionRouteDeps,
) -> dict[str, object]:
    global _DEPS
    _DEPS = deps
    admin_dep = [Depends(require_admin_token)]
    app.post("/api/ollama/unload", dependencies=admin_dep)(ollama_unload)
    app.post("/api/ollama/restart", dependencies=admin_dep)(ollama_restart_compat)
    app.post("/api/reminders/create", dependencies=admin_dep)(create_reminder)
    app.post("/api/calendar/create", dependencies=admin_dep)(create_calendar_event)
    app.post("/api/vlm/describe", dependencies=admin_dep)(vlm_describe)
    app.post("/api/reminders/complete", dependencies=admin_dep)(complete_reminder)
    app.post("/api/reminders/delete", dependencies=admin_dep)(delete_reminder)
    app.post("/api/calendar/delete", dependencies=admin_dep)(delete_calendar_event)
    return {name: globals()[name] for name in __all__ if name != "register_action_routes"}
