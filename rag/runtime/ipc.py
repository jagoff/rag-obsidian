"""Unix domain socket server para el supervisor.

Permite al CLI (``rag supervisor trigger <job>``) y a la web UI
(``/api/supervisor/run/<job>``) disparar jobs adentro del supervisor sin
pagar el cold-start de un proceso nuevo.

Protocolo:

- Socket path: ``~/.local/share/obsidian-rag/supervisor.sock``
- Permisos: ``0o600`` (user-only).
- Mensajes: JSON line-delimited (``\n`` separator). Cada request es UNA
  línea JSON; cada response es UNA línea JSON.
- Schema request:  ``{"action": "run", "job": "<label>"}`` o
                  ``{"action": "status"}`` o ``{"action": "ping"}``.
- Schema response: ``{"ok": true|false, "result": <any>, "error": str|null}``.

Handlers se registran via ``@ipc.handler("<action>")`` o adentro del
supervisor importan ``register_handler`` directo.

Diseño:

- ``IPCServer.serve()`` es bloqueante (asyncio loop) — el supervisor lo
  corre en un thread daemon.
- Cada conexión es one-shot: leer 1 línea, responder, cerrar. Si querés
  multiple requests, abrí múltiple sockets. Mantiene el server simple +
  evita slow-loris.
- Timeout server-side: 5s para parsear request, 30s total per conexión.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SOCKET_PATH",
    "IPCServer",
    "client_call",
    "handler",
    "register_handler",
]

DEFAULT_SOCKET_PATH = Path.home() / ".local/share/obsidian-rag/supervisor.sock"

_HANDLERS: dict[str, Callable[[dict[str, Any]], Any]] = {}


def register_handler(action: str, fn: Callable[[dict[str, Any]], Any]) -> None:
    """Registra ``fn`` como handler para ``action``.

    El handler recibe el dict del request (sin la key ``action``) y
    retorna lo que va al campo ``result`` de la response. Si raisea,
    el server responde ``{"ok": false, "error": str(exc)}``.
    """
    if action in _HANDLERS:
        logger.info("ipc: re-registering handler for %s", action)
    _HANDLERS[action] = fn


def handler(action: str) -> Callable[[Callable[[dict[str, Any]], Any]],
                                     Callable[[dict[str, Any]], Any]]:
    """Decorator equivalente a ``register_handler``."""
    def _wrap(fn: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
        register_handler(action, fn)
        return fn
    return _wrap


def _registered_handlers() -> dict[str, Callable[[dict[str, Any]], Any]]:
    """Snapshot — solo para tests."""
    return dict(_HANDLERS)


def _reset_handlers() -> None:
    """Solo para tests."""
    _HANDLERS.clear()


class IPCServer:
    """Async Unix socket server.

    Uso:

    ```python
    server = IPCServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # ... supervisor corre ...
    server.shutdown()
    ```
    """

    def __init__(self, socket_path: Path | str = DEFAULT_SOCKET_PATH):
        self.socket_path = Path(socket_path)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.AbstractServer | None = None
        self._stop = threading.Event()

    def serve_forever(self) -> None:
        """Entrypoint thread-friendly. Bloquea hasta ``shutdown()``."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve_async())
        finally:
            self._loop.close()
            self._loop = None

    async def _serve_async(self) -> None:
        # Cleanup stale socket file (previous crash leaves it).
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError as exc:
                logger.warning("ipc: failed to unlink stale socket: %s", exc)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=str(self.socket_path),
        )
        # Permisos restrictivos — solo el user dueño puede conectar.
        try:
            os.chmod(self.socket_path, 0o600)
        except OSError as exc:
            logger.warning("ipc: failed to chmod socket: %s", exc)
        logger.info("ipc: listening on %s", self.socket_path)
        try:
            async with self._server:
                await self._server.serve_forever()
        except asyncio.CancelledError:
            pass

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not line:
                return
            response = await self._dispatch(line.decode("utf-8"))
            writer.write((json.dumps(response) + "\n").encode("utf-8"))
            await writer.drain()
        except asyncio.TimeoutError:
            err = json.dumps({"ok": False, "error": "request timeout"})
            writer.write((err + "\n").encode("utf-8"))
        except Exception as exc:  # noqa: BLE001 — never propagate to server
            logger.exception("ipc: connection handler crashed")
            err = json.dumps({"ok": False, "error": f"server: {exc}"})
            try:
                writer.write((err + "\n").encode("utf-8"))
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _dispatch(self, raw: str) -> dict[str, Any]:
        try:
            req = json.loads(raw.strip() or "{}")
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"invalid json: {exc}"}
        if not isinstance(req, dict):
            return {"ok": False, "error": "request must be a JSON object"}
        action = req.get("action")
        if not action:
            return {"ok": False, "error": "missing 'action' field"}
        fn = _HANDLERS.get(action)
        if fn is None:
            return {"ok": False, "error": f"unknown action: {action}"}
        try:
            payload = {k: v for k, v in req.items() if k != "action"}
            # Run handler en thread pool — los handlers son sync (calls a
            # rag.* potencialmente bloqueantes) y no queremos bloquear el
            # event loop.
            result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: fn(payload),
            )
            return {"ok": True, "result": result, "error": None}
        except Exception as exc:  # noqa: BLE001 — surface to client
            logger.exception("ipc: handler %s raised", action)
            return {"ok": False, "error": str(exc), "result": None}

    def shutdown(self) -> None:
        if self._loop is None or self._server is None:
            return
        loop = self._loop

        def _stop() -> None:
            if self._server is not None:
                self._server.close()

        loop.call_soon_threadsafe(_stop)


# ── Cliente ────────────────────────────────────────────────────────────────


def client_call(
    action: str,
    *,
    socket_path: Path | str = DEFAULT_SOCKET_PATH,
    timeout: float = 30.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Cliente sincrónico para invocar el supervisor desde CLI / web.

    Retorna el dict de response. Si el socket no existe (supervisor down),
    levanta ``FileNotFoundError`` — el caller decide si fallback.
    """
    sp = Path(socket_path)
    if not sp.exists():
        raise FileNotFoundError(f"supervisor socket not found: {sp}")

    request = {"action": action, **kwargs}
    payload = (json.dumps(request) + "\n").encode("utf-8")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(sp))
        sock.sendall(payload)
        # Read until newline.
        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        raw = b"".join(chunks).decode("utf-8").strip()
    finally:
        sock.close()

    if not raw:
        return {"ok": False, "error": "empty response from server"}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"invalid json from server: {exc}"}
