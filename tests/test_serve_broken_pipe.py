"""Regression test for the `rag serve` broken-pipe handling (2026-04-26).

Context: the WhatsApp listener posts to `rag serve /query` (port 7832).
When the phone network flaps, or when whisper transcription pushes the
listener past its HTTP client timeout, the listener closes the TCP socket
before the server has finished assembling the response. Stdlib's
`http.server.BaseHTTPRequestHandler.wfile.write()` then raises
`BrokenPipeError`, and stdlib's `socketserver.BaseServer.handle_error()`
prints a full traceback to stderr → which launchd captures into
`serve.error.log` as an "error".

Before the fix we saw **two** tracebacks per flapped request: one for the
failed 200 write, a second from the except-clause that tried to send a
500 on the same dead socket. This test guards against regressions by
checking three invariants in source:

1. A `_QuietHTTPServer` subclass exists and overrides `handle_error`.
2. The `_CLIENT_GONE` tuple explicitly lists the three disconnect types
   we swallow (BrokenPipeError, ConnectionResetError, ConnectionAbortedError).
3. The `do_POST` body short-circuits on `_CLIENT_GONE` BEFORE the generic
   `Exception` handler — otherwise the 500 fallback still runs and writes
   on the dead socket.

Plus one behavioural test: spin up a real TCP server using the same
handle_error logic and assert that a `BrokenPipeError` raised inside a
handler is silently swallowed (no stderr output), while a different
exception still bubbles through to the stdlib traceback path.
"""
from __future__ import annotations

import io
import socket
import socketserver
import sys
import threading
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "rag" / "__init__.py").read_text(encoding="utf-8")


# ── Source-level invariants ────────────────────────────────────────────────


def _serve_source_slice() -> str:
    """Return the source of the `serve()` function only."""
    idx = SOURCE.find("def serve(host: str, port: int):")
    assert idx >= 0, "`def serve(...)` not found in rag/__init__.py"
    # serve() ends when the next top-level `def ` appears.
    end = SOURCE.find("\ndef ", idx + 1)
    return SOURCE[idx:end if end > 0 else len(SOURCE)]


def test_quiethttpserver_subclass_exists():
    body = _serve_source_slice()
    assert "class _QuietHTTPServer(HTTPServer):" in body, (
        "Expected `_QuietHTTPServer(HTTPServer)` subclass inside serve() "
        "to override handle_error and silence BrokenPipe tracebacks."
    )
    assert "def handle_error(self, request, client_address):" in body, (
        "_QuietHTTPServer must override handle_error()."
    )
    assert "server = _QuietHTTPServer((host, port), _Handler)" in body, (
        "serve() must instantiate _QuietHTTPServer, not the plain HTTPServer."
    )


def test_client_gone_tuple_covers_three_disconnect_types():
    body = _serve_source_slice()
    assert "_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)" in body, (
        "_CLIENT_GONE tuple must list all three stdlib disconnect exceptions. "
        "If any one is missing the log will still show a traceback for that variant."
    )


def test_do_post_catches_client_gone_before_generic_exception():
    """The except order matters: if `except Exception` comes first, the
    BrokenPipeError gets caught there, we try to write a 500, and THAT
    raises another BrokenPipe → double traceback. The `except _CLIENT_GONE`
    clause must be listed FIRST so it wins."""
    body = _serve_source_slice()
    # Locate the do_POST method.
    start = body.find("def do_POST(self):")
    assert start >= 0, "do_POST missing from serve()"
    # The method ends at the next `def ` inside the handler class.
    end = body.find("\n        def ", start + 1)
    post = body[start:end if end > 0 else start + 2000]

    client_gone_idx = post.find("except _CLIENT_GONE:")
    generic_idx = post.find("except Exception as exc:")
    assert client_gone_idx >= 0, (
        "do_POST must have `except _CLIENT_GONE:` to silently ignore "
        "dropped connections."
    )
    assert generic_idx >= 0, "do_POST must still handle generic exceptions."
    assert client_gone_idx < generic_idx, (
        "Exception handler order is wrong: `except _CLIENT_GONE` must "
        "come BEFORE `except Exception` (a BrokenPipeError IS an Exception "
        "subclass, so generic-first swallows it and then crashes on the "
        "500 fallback write). Got "
        f"_CLIENT_GONE@{client_gone_idx}, Exception@{generic_idx}."
    )


def test_do_post_500_fallback_wrapped_in_client_gone_guard():
    """Even the 500 fallback write can hit a dead socket. Must be wrapped."""
    body = _serve_source_slice()
    # Heuristic: find `self._json(500,` and verify it's followed by a matching
    # `except _CLIENT_GONE:` inside a try block.
    idx = body.find("self._json(500, {\"error\": str(exc)})")
    assert idx >= 0, "do_POST must still attempt to return a 500 with the error."
    window = body[idx:idx + 400]
    assert "except _CLIENT_GONE:" in window, (
        "The `self._json(500, ...)` call must be inside a "
        "`try: ... except _CLIENT_GONE: pass` block — otherwise a client "
        "that disconnects between the handler failure and the 500 write "
        "still produces a traceback."
    )


# ── Behavioural test: real server swallows BrokenPipe silently ─────────────


class _BrokenPipeHandler(BaseHTTPRequestHandler):
    """Handler that always raises BrokenPipeError — simulates the
    `self.wfile.write()` failure against a dead client socket."""

    def do_GET(self):  # noqa: N802 (stdlib naming)
        raise BrokenPipeError(32, "Broken pipe")

    def log_message(self, format, *args):  # noqa: A002
        pass


class _OtherErrorHandler(BaseHTTPRequestHandler):
    """Handler that raises a non-disconnect exception — must still be
    forwarded to stdlib's handle_error for visibility."""

    def do_GET(self):  # noqa: N802
        raise RuntimeError("oops, something else broke")

    def log_message(self, format, *args):  # noqa: A002
        pass


class _QuietHTTPServer(HTTPServer):
    """Mirror of the in-serve() subclass (duplicated here on purpose so
    the test doesn't need to reach into a closure-local class)."""

    _CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

    def handle_error(self, request, client_address):
        exc_type = sys.exc_info()[0]
        if exc_type is not None and issubclass(exc_type, self._CLIENT_GONE):
            return
        super().handle_error(request, client_address)


def _run_one_request(handler_cls) -> str:
    """Spin up _QuietHTTPServer on a random port, send one GET, capture
    everything stdlib's handle_error writes to stderr, and return it."""
    # `socketserver` writes the traceback directly to sys.stderr, so a
    # plain `redirect_stderr` + `io.StringIO` catches it.
    server = _QuietHTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.handle_request, daemon=True)
    captured = io.StringIO()
    with redirect_stderr(captured):
        thread.start()
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
                s.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
                try:
                    s.recv(1024)  # likely empty / reset
                except OSError:
                    pass
        finally:
            thread.join(timeout=2.0)
            server.server_close()
    return captured.getvalue()


def test_brokenpipe_error_produces_no_stderr_traceback():
    """The whole point of the fix: no noise on disconnect."""
    output = _run_one_request(_BrokenPipeHandler)
    assert "Traceback" not in output, (
        "A BrokenPipeError inside a handler leaked a traceback to stderr. "
        f"Output was:\n{output}"
    )
    assert "BrokenPipeError" not in output, (
        f"Expected complete silence on BrokenPipeError, got:\n{output}"
    )


def test_other_exceptions_still_produce_stderr_traceback():
    """Regression guard: we must NOT have accidentally suppressed ALL errors.
    Unrelated exceptions (RuntimeError, ValueError, ...) must still reach
    stdlib's handle_error and print a traceback — otherwise real bugs go
    invisible."""
    output = _run_one_request(_OtherErrorHandler)
    assert "Traceback" in output, (
        "RuntimeError was silenced — real exceptions must still be logged. "
        f"Output was:\n{output!r}"
    )
    assert "RuntimeError" in output, (
        f"Expected RuntimeError in stderr output, got:\n{output!r}"
    )
