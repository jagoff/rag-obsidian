"""Wrapper httpx para los endpoints del bridge Go (whatsapp-mcp).

Centraliza timeouts + serialización + manejo de errores estructurado.
Los call sites previos (`send.py`, `scheduled.py`) hablan directo via
`urllib`/`requests` por razones históricas — este módulo es el camino
nuevo para `/wa` y endpoints derivados; los viejos pueden migrarse de a
poco sin urgencia.

Soporta tanto el modo sync (handlers FastAPI sync) como async (SSE
poller). Cada función devuelve un dict ya parseado o levanta
`BridgeError` con un mensaje legible.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx


def _bridge_base() -> str:
    """Devuelve `http://localhost:<port>` del bridge actual.

    Lee `OBSIDIAN_RAG_WA_BRIDGE_PORT` (default 8088). Sin cachear porque
    los tests monkeypatchean el env var.
    """
    port = int(os.environ.get("OBSIDIAN_RAG_WA_BRIDGE_PORT", "8088"))
    return f"http://localhost:{port}"


class BridgeError(RuntimeError):
    """Levantada cuando el bridge devuelve 4xx/5xx o el request explota."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


_DEFAULT_TIMEOUT = httpx.Timeout(5.0, connect=2.0, read=5.0, write=5.0, pool=5.0)
_LONG_TIMEOUT = httpx.Timeout(30.0, connect=2.0, read=30.0, write=30.0, pool=5.0)


def _check_ok(r: httpx.Response) -> dict[str, Any]:
    if r.status_code >= 400:
        body_excerpt = (r.text or "")[:200]
        raise BridgeError(
            f"bridge {r.status_code} on {r.url.path}: {body_excerpt}", status=r.status_code
        )
    if r.headers.get("Content-Type", "").startswith("application/json"):
        try:
            return r.json()
        except Exception as e:
            raise BridgeError(f"bridge returned malformed JSON on {r.url.path}: {e}") from e
    return {}


def _raise_request_error(path: str, exc: httpx.RequestError) -> None:
    if isinstance(exc, httpx.TimeoutException):
        raise BridgeError(f"bridge timeout on {path}: {exc}") from exc
    raise BridgeError(f"bridge unreachable on {path}: {exc}") from exc


def _request_json(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: httpx.Timeout = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    try:
        with httpx.Client(base_url=_bridge_base(), timeout=timeout) as c:
            return _check_ok(c.request(method, path, json=json_body, params=params))
    except httpx.RequestError as exc:
        _raise_request_error(path, exc)


# --- Sync API (para handlers FastAPI sync) ---


def health() -> dict[str, Any]:
    return _request_json("GET", "/api/health")


def send_text(jid: str, text: str, reply_to: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"recipient": jid, "message": text}
    if reply_to:
        body["reply_to"] = reply_to
    return _request_json("POST", "/api/send", json_body=body, timeout=_LONG_TIMEOUT)


def send_media(
    jid: str,
    media_path: str,
    caption: str = "",
    reply_to: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Manda un archivo. El bridge espera path absoluto en el server local."""
    body: dict[str, Any] = {
        "recipient": jid,
        "media_path": str(Path(media_path).resolve()),
    }
    if caption:
        body["message"] = caption
    if reply_to:
        body["reply_to"] = reply_to
    return _request_json("POST", "/api/send", json_body=body, timeout=_LONG_TIMEOUT)


def send_ptt(jid: str, opus_path: str, reply_to: dict[str, Any] | None = None) -> dict[str, Any]:
    """Manda un voice note. El bridge ya hardcodea `PTT=true` cuando el
    `media_type` es audio; basta con pasar el `.ogg/opus`.
    """
    return send_media(jid, opus_path, caption="", reply_to=reply_to)


def react(
    jid: str, message_id: str, sender_jid: str, from_me: bool, emoji: str
) -> dict[str, Any]:
    body = {
        "recipient": jid,
        "message_id": message_id,
        "sender_jid": sender_jid,
        "from_me": from_me,
        "emoji": emoji,
    }
    return _request_json("POST", "/api/react", json_body=body)


def revoke(jid: str, message_id: str) -> dict[str, Any]:
    body = {"recipient": jid, "message_id": message_id}
    return _request_json("POST", "/api/revoke", json_body=body)


def typing(jid: str, state: str) -> dict[str, Any]:
    """Send presence update. State: 'composing', 'recording', 'paused'."""
    body = {"recipient": jid, "state": state}
    return _request_json("POST", "/api/typing", json_body=body)


def download_media(message_id: str, chat_jid: str) -> dict[str, Any]:
    """Pide al bridge que descargue el media de un mensaje. Devuelve
    dict con `filename` y `path` (path absoluto en el server local)
    para que el caller los sirva como FileResponse.
    """
    body = {"message_id": message_id, "chat_jid": chat_jid}
    return _request_json("POST", "/api/download", json_body=body, timeout=_LONG_TIMEOUT)


def get_avatar_bytes(jid: str, full: bool = False, refresh: bool = False) -> bytes | None:
    """Devuelve los bytes del JPEG del avatar, o None si el bridge dice
    404 (el contacto no tiene foto pública).
    """
    params = {"jid": jid, "size": "full" if full else "preview"}
    if refresh:
        params["refresh"] = "1"
    try:
        with httpx.Client(base_url=_bridge_base(), timeout=_LONG_TIMEOUT) as c:
            r = c.get("/api/avatar", params=params)
    except httpx.RequestError as exc:
        _raise_request_error("/api/avatar", exc)
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        raise BridgeError(f"avatar {r.status_code} for {jid}", status=r.status_code)
    return r.content
