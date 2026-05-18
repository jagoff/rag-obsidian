"""Routes for server-backed browser layout persistence."""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel


class LayoutItem(BaseModel):
    key: str
    value: Any = None


class LayoutSnapshot(BaseModel):
    state: dict[str, Any]


def api_ui_layout_get(page: str) -> dict[str, Any]:
    try:
        from rag.ui_layout_state import load_layout_with_updated_at, ui_layout_db_path

        state, updated_at = load_layout_with_updated_at(page)
        return {
            "ok": True,
            "page": page,
            "path": str(ui_layout_db_path()),
            "state": state,
            "updated_at": updated_at,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def api_ui_layout_set(page: str, item: LayoutItem) -> dict[str, Any]:
    try:
        from rag.ui_layout_state import (
            load_layout_with_updated_at,
            save_layout_item,
            ui_layout_db_path,
        )

        changed = save_layout_item(page, item.key, item.value)
        state, updated_at = load_layout_with_updated_at(page)
        return {
            "ok": True,
            "changed": changed,
            "page": page,
            "path": str(ui_layout_db_path()),
            "state": state,
            "updated_at": updated_at,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def api_ui_layout_snapshot(page: str, snapshot: LayoutSnapshot) -> dict[str, Any]:
    try:
        from rag.ui_layout_state import (
            load_layout_with_updated_at,
            replace_layout,
            ui_layout_db_path,
        )

        replace_layout(page, snapshot.state)
        state, updated_at = load_layout_with_updated_at(page)
        return {
            "ok": True,
            "page": page,
            "path": str(ui_layout_db_path()),
            "state": state,
            "updated_at": updated_at,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def api_ui_layout_clear(page: str) -> dict[str, Any]:
    try:
        from rag.ui_layout_state import clear_layout, ui_layout_db_path

        changed = clear_layout(page)
        return {
            "ok": True,
            "changed": changed,
            "page": page,
            "path": str(ui_layout_db_path()),
            "state": {},
            "updated_at": {},
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def register_ui_layout_routes(app) -> dict[str, object]:
    app.get("/api/ui-layout/{page}")(api_ui_layout_get)
    app.post("/api/ui-layout/{page}")(api_ui_layout_set)
    app.post("/api/ui-layout/{page}/snapshot")(api_ui_layout_snapshot)
    app.delete("/api/ui-layout/{page}")(api_ui_layout_clear)
    return {
        "api_ui_layout_get": api_ui_layout_get,
        "api_ui_layout_set": api_ui_layout_set,
        "api_ui_layout_snapshot": api_ui_layout_snapshot,
        "api_ui_layout_clear": api_ui_layout_clear,
    }
