"""Tests for the `_silent_log` helper introduced as a replacement for
`except Exception: pass` at sites where silent failure could mask a real
bug (contradict JSON parse, reranker unload, feedback golden rebuild).

Contract:
- Writes one JSON line to SILENT_ERRORS_LOG_PATH via the existing
  _LOG_QUEUE (same non-blocking path as queries.jsonl / behavior.jsonl).
- Must never raise — if JSON serialisation or queue put throws, we drop
  the record. This is the same "best-effort observability" contract that
  lets callers keep using it at P0 sites (a silent log of a silent log
  would be absurd).
- Truncates long exception messages to 500 chars so a 1 MB exception
  string from a broken subsystem can't balloon the jsonl file.
"""
from __future__ import annotations

import json
import threading

import pytest

import rag


@pytest.fixture
def silent_log_path(tmp_path, monkeypatch):
    path = tmp_path / "silent_errors.jsonl"
    monkeypatch.setattr(rag, "SILENT_ERRORS_LOG_PATH", path)
    return path


@pytest.fixture
def flush_log():
    """Drain the background log queue after each test so writes are visible."""
    yield
    rag._LOG_QUEUE.join()


def test_silent_log_writes_record(silent_log_path, flush_log):
    """Happy path: helper writes one JSON line with where + exc_type + exc."""
    try:
        raise ValueError("boom")
    except ValueError as exc:
        rag._silent_log("test_where", exc)

    rag._LOG_QUEUE.join()
    assert silent_log_path.is_file()
    lines = silent_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    # Default sin with_traceback no agrega `traceback` al record
    assert "traceback" not in rec


def test_silent_log_with_traceback_includes_stack_frames(silent_log_path, flush_log):
    """`with_traceback=True` agrega el campo `traceback` con frames del stack.

    Útil para sitios como `graph_expand.outer` donde `exc_type` + `exc` no
    alcanzan para diagnosticar (TypeError genérico, sin contexto del
    callsite). El default sigue siendo False para no inflar los logs en
    hot paths.
    """
    def inner():
        raise TypeError("'NoneType' object is not subscriptable")

    try:
        inner()
    except TypeError as exc:
        rag._silent_log("graph_expand.outer", exc, with_traceback=True)

    rag._LOG_QUEUE.join()
    lines = silent_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["where"] == "graph_expand.outer"
    assert rec["exc_type"] == "TypeError"
    # Traceback debería mencionar la función inner() del stack
    assert "traceback" in rec
    assert "inner" in rec["traceback"]
    # Cap a 2KB
    assert len(rec["traceback"]) <= 2000


def test_silent_log_record_has_default_fields(silent_log_path, flush_log):
    """Backward compat: el shape default sigue siendo where + exc_type + exc + ts."""
    try:
        raise ValueError("boom")
    except ValueError as exc:
        rag._silent_log("test_where", exc)

    rag._LOG_QUEUE.join()
    lines = silent_log_path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    assert rec["where"] == "test_where"
    assert rec["exc_type"] == "ValueError"
    assert rec["exc"] == "boom"
    assert rec["ts"]  # ISO timestamp present


def test_silent_log_truncates_long_messages(silent_log_path, flush_log):
    """Bounded record size — prevents a subsystem emitting 1 MB stacktraces
    in exc.args[0] from bloating silent_errors.jsonl."""
    big = "x" * 5000
    try:
        raise RuntimeError(big)
    except RuntimeError as exc:
        rag._silent_log("truncation_test", exc)

    rag._LOG_QUEUE.join()
    rec = json.loads(silent_log_path.read_text(encoding="utf-8").splitlines()[0])
    assert len(rec["exc"]) == 500


def test_silent_log_never_raises_on_unserialisable_where(silent_log_path):
    """If json.dumps blows up on the record (e.g. a caller passes a bytes
    `where`), the helper must swallow it — not re-raise into a silent-fail
    code path that would now crash."""
    class Weird:
        def __repr__(self):
            raise RuntimeError("cannot repr")

    # `where` is always a str in our call sites, but if someone pipes in
    # something weird we still must not raise.
    try:
        raise ValueError("x")
    except ValueError as exc:
        # Force a serialisation failure by making exc.__str__ explode.
        bad_exc = ValueError()
        bad_exc.args = (Weird(),)
        # Must not raise.
        rag._silent_log("serialisation_crash", bad_exc)


def test_silent_log_never_raises_on_queue_full(monkeypatch, silent_log_path):
    """If _LOG_QUEUE is full / put_nowait throws, helper drops silently."""
    class FullQueue:
        def put_nowait(self, item):
            raise RuntimeError("queue full")

    monkeypatch.setattr(rag, "_LOG_QUEUE", FullQueue())

    try:
        raise ValueError("x")
    except ValueError as exc:
        rag._silent_log("queue_full", exc)  # must not raise


def test_silent_log_concurrent_writes(silent_log_path, flush_log):
    """10 threads each logging 10 times → 100 lines, all parseable. Verifies
    we reuse the same single-writer queue pattern as log_query_event."""
    def worker(i: int) -> None:
        for j in range(10):
            try:
                raise RuntimeError(f"t{i}-{j}")
            except RuntimeError as exc:
                rag._silent_log(f"thread_{i}", exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rag._LOG_QUEUE.join()

    lines = silent_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    # Every line is valid JSON with the expected keys.
    for line in lines:
        rec = json.loads(line)
        assert rec["exc_type"] == "RuntimeError"
        assert rec["where"].startswith("thread_")
