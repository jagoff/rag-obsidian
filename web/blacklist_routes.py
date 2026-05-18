"""Routes for the system-wide blacklist admin UI."""
from __future__ import annotations

from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


class BlacklistItem(BaseModel):
    kind: str
    value: str


BLACKLIST_KINDS = [
    {"key": "chats", "label": "Grupos / chats"},
    {"key": "people", "label": "Personas"},
    {"key": "topics", "label": "Temas"},
    {"key": "words", "label": "Palabras exactas"},
    {"key": "fuzzy_words", "label": "Palabras parecidas"},
    {"key": "paths", "label": "Paths exactos"},
    {"key": "path_prefixes", "label": "Prefijos de path"},
    {"key": "path_globs", "label": "Globs de path"},
]


def blacklist_page(static_dir: Path) -> FileResponse:
    return FileResponse(static_dir / "blacklist.html")


def api_blacklist() -> dict:
    from rag.exclusions import blacklist_path, load_blacklist

    return {
        "path": str(blacklist_path()),
        "kinds": BLACKLIST_KINDS,
        "config": load_blacklist(),
    }


def api_blacklist_add(item: BlacklistItem) -> dict:
    value = item.value.strip()
    if not value:
        raise HTTPException(status_code=400, detail="value vacío")
    try:
        from rag.exclusions import add_blacklist_item, blacklist_path, load_blacklist

        changed = add_blacklist_item(item.kind, value)
        return {
            "ok": True,
            "changed": changed,
            "path": str(blacklist_path()),
            "config": load_blacklist(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def api_blacklist_delete(item: BlacklistItem) -> dict:
    value = item.value.strip()
    if not value:
        raise HTTPException(status_code=400, detail="value vacío")
    try:
        from rag.exclusions import blacklist_path, load_blacklist, remove_blacklist_item

        changed = remove_blacklist_item(item.kind, value)
        return {
            "ok": True,
            "changed": changed,
            "path": str(blacklist_path()),
            "config": load_blacklist(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def register_blacklist_routes(app, static_dir: Path, require_admin_token) -> dict[str, object]:
    static_dir = Path(static_dir)

    def page() -> FileResponse:
        return blacklist_page(static_dir)

    app.get("/blacklist")(page)
    app.get("/api/blacklist", dependencies=[Depends(require_admin_token)])(api_blacklist)
    app.post("/api/blacklist", dependencies=[Depends(require_admin_token)])(api_blacklist_add)
    app.post(
        "/api/blacklist/delete",
        dependencies=[Depends(require_admin_token)],
    )(api_blacklist_delete)

    return {
        "blacklist_page": page,
        "api_blacklist": api_blacklist,
        "api_blacklist_add": api_blacklist_add,
        "api_blacklist_delete": api_blacklist_delete,
    }
