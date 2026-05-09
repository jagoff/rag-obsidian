"""Admin auth — protege endpoints destructivos del web server.

Extracto de [`web/server.py`](web/server.py) (Phase W3b, refactor modular
2026-05-09). Vive acá:

- `_ADMIN_TOKEN_PATH` — ruta del archivo 0o600 con el token persistido.
- `_load_or_create_admin_token()` — read-or-generate con permisos 0o600.
- `_ADMIN_TOKEN` — cargado al import.
- `_require_admin_token(request)` — dependency FastAPI (Bearer token).
- `_is_localhost_request(request)` — clasifica request por loopback.
- `admin_token(request)` — handler GET `/api/admin/token` (sin decorador;
  el `@app.get(...)` se aplica en `web/server.py` para evitar circular
  import).

Re-exportado desde [`web/server.py`](web/server.py) con
`from web._admin import (...)` para preservar back-compat con tests
(`web.server._require_admin_token`, `web.server._ADMIN_TOKEN`, etc.).

Cross-module monkeypatch (gotcha): los tests hacen
`patch("web.server._is_localhost_request", return_value=True)`. Para que
el patch se propague al endpoint `admin_token` (que llama
`_is_localhost_request` internamente), el handler hace lazy lookup
`from web import server as _ws; _ws._is_localhost_request(request)`.
Mismo trato para `_ADMIN_TOKEN_PATH` (test
`test_admin_token_file_created_on_boot` monkeypatcha la ruta sobre
`web.server`).
"""
from __future__ import annotations

import secrets
import sys
from pathlib import Path

from fastapi import HTTPException, Request


_ADMIN_TOKEN_PATH = Path.home() / ".config" / "obsidian-rag" / "admin_token.txt"


def _resolve_token_path() -> Path:
    """Resolve la ruta del token vía lookup dinámico al módulo `web.server`.

    Soporta el patrón de monkeypatch que usan los tests:
    `monkeypatch.setattr(web.server, "_ADMIN_TOKEN_PATH", token_path)`.
    Si `web.server` todavía no está importado (orden de import al
    bootstrap), cae al `_ADMIN_TOKEN_PATH` local.
    """
    try:
        from web import server as _ws  # noqa: PLC0415
        return getattr(_ws, "_ADMIN_TOKEN_PATH", _ADMIN_TOKEN_PATH)
    except Exception:
        return _ADMIN_TOKEN_PATH


def _load_or_create_admin_token() -> str:
    """Lee el token de _ADMIN_TOKEN_PATH o lo genera (write-once, chmod 600).

    Se llama una vez al import — el resultado queda en _ADMIN_TOKEN.
    """
    token_path = _resolve_token_path()
    try:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                return token
        # Generar token nuevo
        token = secrets.token_urlsafe(32)
        # Escribir con tmp+replace para atomicidad
        tmp = token_path.with_suffix(".tmp")
        tmp.write_text(token + "\n", encoding="utf-8")
        tmp.chmod(0o600)
        tmp.replace(token_path)
        # Print al stderr UNA sola vez para que el user lo vea en el log
        sys.stderr.write(
            f"\n[obsidian-rag] Admin token generado: {token}\n"
            f"  Guardado en: {token_path}\n"
            f"  Usar en: Authorization: Bearer {token}\n\n"
        )
        return token
    except Exception as exc:
        # Fallback: token en memoria (no persistido). Suficiente para la sesión.
        sys.stderr.write(
            f"[obsidian-rag] WARNING: no pude escribir admin_token ({exc}); "
            "usando token efímero en memoria.\n"
        )
        return secrets.token_urlsafe(32)


_ADMIN_TOKEN: str = _load_or_create_admin_token()


def _require_admin_token(request: Request) -> None:
    """FastAPI dependency — 401 si el request no trae el Bearer token correcto.

    Usar como: `@app.post("/api/...", dependencies=[Depends(_require_admin_token)])`.
    Compara con `secrets.compare_digest` para evitar timing attacks.
    """
    auth = request.headers.get("Authorization", "")
    scheme, _, provided = auth.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        provided.encode(), _ADMIN_TOKEN.encode()
    ):
        raise HTTPException(
            status_code=401,
            detail=(
                "Se requiere Authorization: Bearer <admin_token>. "
                "El token está en ~/.config/obsidian-rag/admin_token.txt"
            ),
        )


def _is_localhost_request(request: Request) -> bool:
    """True si el request viene de loopback (127.0.0.1, ::1, localhost).

    Endpoints que sirven el admin_token al frontend confían en esto: el browser
    del user accede vía localhost o ra.ai (mapeado a 127.0.0.1 en /etc/hosts).
    Cualquier otro origin (LAN expose, tunnel) NO recibe el token.
    """
    if not request.client:
        return False
    host = (request.client.host or "").strip().lower()
    return host in {"127.0.0.1", "::1", "localhost"}


def admin_token(request: Request):
    """Devuelve el admin_token al frontend SI el request es de loopback.

    El frontend (`/static/*.js`) llama este endpoint una vez al boot,
    cachea el token en memoria y lo manda como `Authorization: Bearer X`
    en los 8 endpoints admin (auto-fix-devin, reindex, ollama/*, etc.).

    Restringido a localhost — un device remoto en LAN/tunnel NO puede leer
    el token aunque acceda al frontend (el browser del LAN-user pegaría 403).

    Rate limit (Bug Hunt 2026-05-08 H-1): aún siendo loopback-only, evita
    polling agresivo desde una extensión maliciosa o bug del frontend que
    podría exfiltrar el token via timing/fingerprint si el host está
    comprometido. Reusa `_BEHAVIOR_BUCKETS` (120 req/60s).

    NOTA refactor (2026-05-09): hace lazy lookup `web.server._is_localhost_request`
    + `_check_rate_limit` + buckets para preservar monkeypatch desde los tests
    (`patch("web.server._is_localhost_request", ...)`). El endpoint queda
    decorado `@app.get(...)` desde `web/server.py` para evitar circular
    import al import-time del módulo.
    """
    # Lazy import para preservar el patrón del monkeypatch en tests.
    from web import server as _ws  # noqa: PLC0415
    if not _ws._is_localhost_request(request):
        raise HTTPException(status_code=403, detail="solo accesible desde localhost")
    client_ip = (request.client.host if request.client else "unknown")
    _ws._check_rate_limit(
        _ws._BEHAVIOR_BUCKETS, client_ip,
        _ws._BEHAVIOR_RATE_LIMIT, _ws._BEHAVIOR_RATE_WINDOW,
    )
    return {"token": _ws._ADMIN_TOKEN}


__all__ = [
    "_ADMIN_TOKEN",
    "_ADMIN_TOKEN_PATH",
    "_load_or_create_admin_token",
    "_require_admin_token",
    "_is_localhost_request",
    "admin_token",
]
