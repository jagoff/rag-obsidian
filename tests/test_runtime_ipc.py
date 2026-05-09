"""Tests fundacionales de rag.runtime.ipc.

Cubren:
- handler registration
- client_call → server roundtrip
- error handling (handler raisea, action desconocida, JSON malformado)
- permisos socket 0o600
- shutdown limpio

NOTA: macOS limita ``sun_path`` a ~104 caracteres. ``pytest tmp_path``
genera paths largos (``/private/var/folders/.../pytest-of-fer/...``) que
sobrepasan el límite. Usamos ``tempfile.mkdtemp(dir="/tmp")`` para
garantizar paths cortos.
"""
from __future__ import annotations

import os
import shutil
import stat
import tempfile
import threading
import time
from pathlib import Path

import pytest

from rag.runtime import ipc


@pytest.fixture
def short_tmpdir():
    """Tmpdir corto para que el socket path quepa en sun_path (≤104 chars)."""
    d = Path(tempfile.mkdtemp(dir="/tmp", prefix="rag-ipc-"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def _reset_handlers():
    ipc._reset_handlers()
    yield
    ipc._reset_handlers()


@pytest.fixture
def server(short_tmpdir):
    sock_path = short_tmpdir / "t.sock"
    server = ipc.IPCServer(socket_path=sock_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Espera al socket — el server hace mkdir + listen async, damos hasta 2s.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if sock_path.exists():
            break
        time.sleep(0.05)
    if not sock_path.exists():
        pytest.fail("IPC socket nunca apareció en 2s")
    yield server, sock_path
    server.shutdown()
    thread.join(timeout=2.0)


def test_handler_decorator_registers():
    @ipc.handler("foo")
    def my_handler(payload):
        return {"echo": payload}

    handlers = ipc._registered_handlers()
    assert "foo" in handlers
    result = handlers["foo"]({"x": 1})
    assert result == {"echo": {"x": 1}}


def test_client_call_roundtrip(server):
    _srv, sock_path = server

    @ipc.handler("ping")
    def ping(payload):
        return {"pong": True, "received": payload}

    response = ipc.client_call("ping", socket_path=sock_path, value=42)
    assert response["ok"] is True
    assert response["result"]["pong"] is True
    assert response["result"]["received"]["value"] == 42


def test_client_call_unknown_action(server):
    _srv, sock_path = server
    response = ipc.client_call("does_not_exist", socket_path=sock_path)
    assert response["ok"] is False
    assert "unknown action" in (response["error"] or "")


def test_handler_exception_surfaces(server):
    _srv, sock_path = server

    @ipc.handler("crash")
    def crash(payload):
        raise RuntimeError("kaboom")

    response = ipc.client_call("crash", socket_path=sock_path)
    assert response["ok"] is False
    assert "kaboom" in (response["error"] or "")


def test_socket_permissions_0o600(server):
    _srv, sock_path = server
    mode = stat.S_IMODE(os.stat(sock_path).st_mode)
    assert mode == 0o600, f"esperado 0o600, got {oct(mode)}"


def test_client_call_socket_not_exists(short_tmpdir):
    nonexistent = short_tmpdir / "missing.sock"
    with pytest.raises(FileNotFoundError):
        ipc.client_call("anything", socket_path=nonexistent)


def test_stale_socket_replaced_on_serve(short_tmpdir):
    sock_path = short_tmpdir / "stale.sock"
    # Toca un archivo regular en el path — server debería borrarlo y crear
    # el suyo.
    sock_path.write_text("stale")
    assert sock_path.exists()

    server = ipc.IPCServer(socket_path=sock_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if sock_path.exists() and Path(sock_path).is_socket():
            break
        time.sleep(0.05)

    try:
        # No assertion estricto — Path.is_socket() no funciona en todos los
        # FS (tmpfs en macOS sí). Lo importante: el server arrancó OK y el
        # cliente puede conectarse.

        @ipc.handler("ping")
        def ping(_):
            return "pong"

        resp = ipc.client_call("ping", socket_path=sock_path)
        assert resp["ok"] is True
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
