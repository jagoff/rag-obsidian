"""Basic static, health, mirror, and PWA route registration."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, RedirectResponse


def health_check() -> dict:
    """Liveness probe used by bootstrap and external monitors."""
    return {"status": "ok", "service": "obsidian-rag-web"}


def api_mirror(date: str | None = None, refresh: int = 0) -> dict:
    from rag.mirror import assemble_mirror

    return assemble_mirror(date=date, use_cache=(refresh != 1))


def api_screen_capture(obs_id: int):
    """Serve a rag_screen_observations PNG by row id."""
    import sqlite3 as _sqlite3

    try:
        import rag as _rag

        db_path = _rag.DB_PATH / "telemetry.db"
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"rag import failed: {exc}")

    try:
        con = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        try:
            row = con.execute(
                "SELECT image_path FROM rag_screen_observations WHERE id = ?",
                (int(obs_id),),
            ).fetchone()
        finally:
            con.close()
    except _sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=f"db error: {exc}")

    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="image not found")

    img_path = Path(row[0])
    canonical_dir = Path.home() / ".local/share/obsidian-rag/screen_captures"
    try:
        img_path.resolve().relative_to(canonical_dir.resolve())
    except (ValueError, OSError):
        raise HTTPException(status_code=403, detail="image outside allowed dir")

    if not img_path.exists():
        raise HTTPException(status_code=404, detail="image file missing")

    return FileResponse(
        path=str(img_path),
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600, immutable"},
    )


def api_mirror_insights(date: str | None = None) -> dict:
    from rag.mirror import assemble_mirror, generate_insights

    mirror = assemble_mirror(date=date, use_cache=True)
    insights = generate_insights(mirror)
    return {"date": mirror.get("date"), **insights}


def api_mirror_whatsapp(date: str | None = None) -> dict:
    from rag.mirror import _source_whatsapp

    if date is None:
        try:
            from rag import mood as _mood

            date = _mood._today_local()
        except Exception:
            date = datetime.now().date().isoformat()
    return {
        "date": date,
        "computed_at": time.time(),
        "whatsapp": _source_whatsapp(date),
    }


def register_basic_routes(app, static_dir: Path) -> dict[str, object]:
    """Register basic routes and return handlers for web.server compatibility."""
    static_dir = Path(static_dir)

    def home_page() -> FileResponse:
        return FileResponse(static_dir / "home.v2.html")

    def home_v1_page() -> RedirectResponse:
        # home.html (pre-ESM) fue reemplazado completamente por home.v2.html + módulos ESM.
        # /v1 redirige a / para no romper bookmarks existentes.
        return RedirectResponse("/", status_code=301)

    def home_v2_page() -> RedirectResponse:
        # /v2 es alias permanente de /
        return RedirectResponse("/", status_code=301)

    def chat_page() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    def mirror_page() -> FileResponse:
        return FileResponse(static_dir / "mirror.html")

    def wa_page() -> FileResponse:
        return FileResponse(static_dir / "wa.html")

    def scheduled_page() -> RedirectResponse:
        return RedirectResponse("/dashboard#sec-wa-scheduled-card", status_code=307)

    def manifest() -> FileResponse:
        return FileResponse(
            static_dir / "manifest.webmanifest",
            media_type="application/manifest+json",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    def service_worker() -> FileResponse:
        return FileResponse(
            static_dir / "sw.js",
            media_type="application/javascript",
            headers={
                "Cache-Control": "no-cache",
                "Service-Worker-Allowed": "/",
            },
        )

    app.get("/health")(health_check)
    app.get("/api/health")(health_check)
    app.get("/")(home_page)
    app.get("/v1")(home_v1_page)
    app.get("/v2")(home_v2_page)
    app.get("/chat")(chat_page)
    app.get("/mirror")(mirror_page)
    app.get("/wa")(wa_page)
    app.get("/wzp")(wa_page)
    app.get("/scheduled")(scheduled_page)
    app.get("/api/mirror")(api_mirror)
    app.get("/api/screen-capture/{obs_id}")(api_screen_capture)
    app.get("/api/mirror/insights")(api_mirror_insights)
    app.get("/api/mirror/whatsapp")(api_mirror_whatsapp)
    app.get("/manifest.webmanifest")(manifest)
    app.get("/sw.js")(service_worker)

    return {
        "health_check": health_check,
        "home_page": home_page,
        "home_v1_page": home_v1_page,
        "home_v2_page": home_v2_page,
        "chat_page": chat_page,
        "mirror_page": mirror_page,
        "wa_page": wa_page,
        "scheduled_page": scheduled_page,
        "api_mirror": api_mirror,
        "api_screen_capture": api_screen_capture,
        "api_mirror_insights": api_mirror_insights,
        "manifest": manifest,
        "service_worker": service_worker,
    }
