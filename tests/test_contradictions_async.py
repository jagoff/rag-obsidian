"""Tests for the async contradiction-check daemon in rag.py.

Production behaviour we guard (mirror of test_web_conv_shutdown.py):

  * `_spawn_contradiction_worker` registers every worker thread in
    `_CONTRA_WRITERS` and removes it on completion (success or failure),
    so the tracker can't leak.
  * `_drain_contradiction_workers` (the atexit hook) joins in-flight
    workers up to ~10s before letting the process exit. A wedged worker
    never blocks shutdown; it's logged and skipped.
  * `_dispatch_contradiction_check` honours the `_CONTRADICTION_ASYNC`
    env toggle — sync when off (tests), daemon when on (production).
  * `_retry_pending_contradictions` drains the JSONL retry file and
    re-enters the daemon queue on startup.
  * `_update_contradicts_frontmatter`'s idempotence guard closes the
    watch→daemon→watch loop when the contradict set hasn't changed.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import rag


# ── Tracker / drain primitives ─────────────────────────────────────────────

def _reset_tracker() -> None:
    with rag._CONTRA_WRITERS_LOCK:
        rag._CONTRA_WRITERS.clear()
    # Clear the shutdown event in case a prior test (or a real atexit
    # invocation in the suite) tripped it. Without this, every spawned
    # worker would short-circuit to the "deferred-shutdown" branch.
    rag._CONTRA_SHUTDOWN_EVENT.clear()


def _register_fake_worker(work_time: float) -> threading.Thread:
    """Spawn a daemon thread that sleeps `work_time`s, mirroring the
    bookkeeping in `_spawn_contradiction_worker` without touching the
    chat model / file write path."""
    def _body() -> None:
        try:
            time.sleep(work_time)
        finally:
            with rag._CONTRA_WRITERS_LOCK:
                rag._CONTRA_WRITERS.discard(threading.current_thread())

    t = threading.Thread(target=_body, daemon=True, name=f"fake-contradict-{work_time}")
    with rag._CONTRA_WRITERS_LOCK:
        rag._CONTRA_WRITERS.add(t)
    t.start()
    return t


def test_tracker_self_removes_on_completion():
    """A worker that finishes on its own must not linger in the tracker."""
    _reset_tracker()
    t = _register_fake_worker(0.01)
    t.join(timeout=1.0)
    assert not t.is_alive()
    with rag._CONTRA_WRITERS_LOCK:
        assert t not in rag._CONTRA_WRITERS


def test_drain_with_no_workers_is_noop():
    """Drain on an empty tracker must return immediately."""
    _reset_tracker()
    t0 = time.monotonic()
    rag._drain_contradiction_workers()
    assert time.monotonic() - t0 < 0.5


def test_drain_waits_for_quick_workers():
    """A handful of short workers must all finish inside the 10s budget.
    Drain must return once they're all done, not after the full cap."""
    _reset_tracker()
    workers = [_register_fake_worker(0.05) for _ in range(3)]
    t0 = time.monotonic()
    rag._drain_contradiction_workers()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"drain took {elapsed:.2f}s, expected <2s"
    for t in workers:
        assert not t.is_alive()
    with rag._CONTRA_WRITERS_LOCK:
        assert not rag._CONTRA_WRITERS


def test_drain_does_not_exceed_cap_with_wedged_worker():
    """If a worker is stuck (ollama hung, file moved), drain must return
    inside ~10s, NOT hang the process. daemon=True lets the OS reap on exit."""
    _reset_tracker()
    # Shorten the cap so the test doesn't actually take 10s.
    wedged = _register_fake_worker(30.0)
    t0 = time.monotonic()
    rag._drain_contradiction_workers(timeout=1.0)
    elapsed = time.monotonic() - t0
    assert 0.9 < elapsed < 2.0, f"drain took {elapsed:.2f}s, expected ~1s"
    assert wedged.is_alive(), "wedged worker should still be running"
    # Cleanup — the daemon thread dies on process exit anyway.
    with rag._CONTRA_WRITERS_LOCK:
        rag._CONTRA_WRITERS.discard(wedged)


# ── Spawn / register / self-release ───────────────────────────────────────

def test_spawn_contradiction_worker_registers_and_releases(monkeypatch):
    """`_spawn_contradiction_worker` must track the thread during execution
    and release it when `_check_and_flag_contradictions` returns,
    regardless of success/failure."""
    _reset_tracker()
    barrier = threading.Event()

    def fake_check(col, path, text, doc_id):
        barrier.wait(timeout=2.0)
        return None

    monkeypatch.setattr(rag, "_check_and_flag_contradictions", fake_check)

    t = rag._spawn_contradiction_worker(None, Path("/tmp/x.md"), "body", "x.md")
    with rag._CONTRA_WRITERS_LOCK:
        assert t in rag._CONTRA_WRITERS
    barrier.set()
    t.join(timeout=2.0)
    assert not t.is_alive()
    with rag._CONTRA_WRITERS_LOCK:
        assert t not in rag._CONTRA_WRITERS


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_spawn_releases_on_check_exception(monkeypatch, tmp_path):
    """An exception inside `_check_and_flag_contradictions` must still
    discharge the tracker AND spill the payload to the pending JSONL —
    otherwise a recurrent failure would leak one entry per save."""
    _reset_tracker()

    def fake_check(col, path, text, doc_id):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(rag, "_check_and_flag_contradictions", fake_check)
    # Redirect the pending file so we don't clobber the user's real one.
    pending = tmp_path / "contradiction_pending.jsonl"
    monkeypatch.setattr(rag, "_CONTRA_PENDING_PATH", pending)

    t = rag._spawn_contradiction_worker(None, Path("/tmp/y.md"), "body", "y.md")
    t.join(timeout=2.0)
    assert not t.is_alive()
    with rag._CONTRA_WRITERS_LOCK:
        assert t not in rag._CONTRA_WRITERS
    # Pending file must have one entry recording the failure.
    assert pending.is_file()
    lines = [l for l in pending.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["doc_id_prefix"] == "y.md"
    assert "simulated failure" in rec["error"]


def test_worker_defers_when_shutdown_event_set(monkeypatch, tmp_path):
    """If `_CONTRA_SHUTDOWN_EVENT` was already set when the worker starts,
    it must skip the chat call and write a `deferred-shutdown` entry to
    the pending JSONL. Pre-fix, mid-flight 30-120s LLM calls during atexit
    would produce `contradiction_shutdown_timeout` errors in telemetry.
    """
    _reset_tracker()

    invocations: list[tuple] = []

    def fake_check(col, path, text, doc_id):
        invocations.append((col, path, text, doc_id))

    monkeypatch.setattr(rag, "_check_and_flag_contradictions", fake_check)
    pending = tmp_path / "contradiction_pending.jsonl"
    monkeypatch.setattr(rag, "_CONTRA_PENDING_PATH", pending)

    # Simulate the drain having fired before the worker started.
    rag._CONTRA_SHUTDOWN_EVENT.set()
    try:
        t = rag._spawn_contradiction_worker(None, Path("/tmp/late.md"), "body", "late.md")
        t.join(timeout=2.0)
    finally:
        rag._CONTRA_SHUTDOWN_EVENT.clear()

    assert not t.is_alive()
    # Chat call was skipped entirely.
    assert invocations == []
    # Pending file has a single deferred entry.
    assert pending.is_file()
    lines = [l for l in pending.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["error"] == "deferred-shutdown"
    assert rec["doc_id_prefix"] == "late.md"


def test_drain_sets_shutdown_event(monkeypatch):
    """`_drain_contradiction_workers()` must flip the shutdown event so
    in-flight workers can short-circuit. Verifies the event is set even
    when there are no pending workers (the early-return branch)."""
    _reset_tracker()
    rag._CONTRA_SHUTDOWN_EVENT.clear()
    assert not rag._CONTRA_SHUTDOWN_EVENT.is_set()

    rag._drain_contradiction_workers(timeout=0.1)

    assert rag._CONTRA_SHUTDOWN_EVENT.is_set()
    # Reset for downstream tests.
    rag._CONTRA_SHUTDOWN_EVENT.clear()


# ── Dispatcher (sync vs async) ────────────────────────────────────────────

def test_dispatch_sync_when_env_disabled(monkeypatch):
    """`_CONTRADICTION_ASYNC=0` → runs synchronously, returns the tuple."""
    _reset_tracker()
    monkeypatch.setenv("_CONTRADICTION_ASYNC", "0")
    calls: list[tuple] = []

    def fake_check(col, path, text, doc_id):
        calls.append((col, path, text, doc_id))
        return ("new-raw", "new-hash")

    monkeypatch.setattr(rag, "_check_and_flag_contradictions", fake_check)
    out = rag._dispatch_contradiction_check(None, Path("/tmp/a.md"), "body", "a.md")
    assert out == ("new-raw", "new-hash")
    assert len(calls) == 1
    # No daemon thread registered in the sync path.
    with rag._CONTRA_WRITERS_LOCK:
        assert not rag._CONTRA_WRITERS


def test_dispatch_async_when_env_enabled(monkeypatch):
    """Default env → spawns daemon, returns None immediately."""
    _reset_tracker()
    monkeypatch.setenv("_CONTRADICTION_ASYNC", "1")
    barrier = threading.Event()
    invoked = threading.Event()

    def fake_check(col, path, text, doc_id):
        invoked.set()
        barrier.wait(timeout=2.0)
        return None

    monkeypatch.setattr(rag, "_check_and_flag_contradictions", fake_check)
    out = rag._dispatch_contradiction_check(None, Path("/tmp/b.md"), "body", "b.md")
    assert out is None
    # Let the worker finish before the test exits.
    invoked.wait(timeout=2.0)
    barrier.set()
    with rag._CONTRA_WRITERS_LOCK:
        pending = list(rag._CONTRA_WRITERS)
    for t in pending:
        t.join(timeout=2.0)


# ── Retry of pending payloads ─────────────────────────────────────────────

def test_retry_pending_contradictions_drains_file(monkeypatch, tmp_path):
    """On startup, the JSONL retry file must be consumed line-by-line,
    each valid entry re-dispatched as a daemon worker, and the file
    deleted (or rewritten with only the invalid lines)."""
    _reset_tracker()
    pending = tmp_path / "contradiction_pending.jsonl"
    f = tmp_path / "note.md"
    f.write_text("body\n", encoding="utf-8")
    pending.write_text(
        json.dumps({
            "ts": "2026-04-21T00:00:00",
            "path": str(f),
            "text": "body",
            "doc_id_prefix": "note.md",
            "error": "prior failure",
        }) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag, "_CONTRA_PENDING_PATH", pending)

    invoked = threading.Event()

    def fake_check(col, path, text, doc_id):
        invoked.set()

    monkeypatch.setattr(rag, "_check_and_flag_contradictions", fake_check)
    n = rag._retry_pending_contradictions(None)
    assert n == 1
    # The file is gone (all entries consumed).
    assert not pending.is_file()
    # The daemon actually ran.
    assert invoked.wait(timeout=2.0)
    # Tracker is clean.
    with rag._CONTRA_WRITERS_LOCK:
        pending_threads = list(rag._CONTRA_WRITERS)
    for t in pending_threads:
        t.join(timeout=2.0)


def test_retry_skips_vanished_files(monkeypatch, tmp_path):
    """If the path in the pending entry no longer exists on disk, drop
    the entry silently — don't spawn a worker that will fail immediately."""
    _reset_tracker()
    pending = tmp_path / "contradiction_pending.jsonl"
    pending.write_text(
        json.dumps({
            "ts": "2026-04-21T00:00:00",
            "path": str(tmp_path / "gone.md"),
            "text": "body",
            "doc_id_prefix": "gone.md",
        }) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag, "_CONTRA_PENDING_PATH", pending)
    n = rag._retry_pending_contradictions(None)
    assert n == 0
    with rag._CONTRA_WRITERS_LOCK:
        assert not rag._CONTRA_WRITERS


# ── Idempotence guard on frontmatter write ────────────────────────────────

def test_frontmatter_contradicts_set_empty(tmp_path):
    """No frontmatter or no contradicts key → empty set."""
    f = tmp_path / "a.md"
    f.write_text("# title\nbody\n", encoding="utf-8")
    assert rag._frontmatter_contradicts_set(f) == set()


def test_frontmatter_contradicts_set_populated(tmp_path):
    """Existing contradicts: list parses into a set of strings."""
    f = tmp_path / "b.md"
    f.write_text(
        "---\ncontradicts:\n- path/one.md\n- path/two.md\n---\n\nbody\n",
        encoding="utf-8",
    )
    assert rag._frontmatter_contradicts_set(f) == {"path/one.md", "path/two.md"}


def test_check_and_flag_skips_when_set_unchanged(monkeypatch, tmp_path):
    """The idempotence short-circuit: if on-disk contradicts already
    match the newly-detected set, no write, no log, no console alert.
    Prevents the async re-index loop."""
    # Aislar DB_PATH para que cualquier _log_contradictions accidental
    # caiga en tmp_path en lugar de la DB real del usuario. La idempotency
    # guard hace que este test no llegue a loguear, pero defensa en
    # profundidad — un día alguien edita `_check_and_flag_contradictions`
    # y rompe la guard, y queremos que el daño quede contenido al tmp.
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    f = tmp_path / "c.md"
    f.write_text(
        "---\ncontradicts:\n- path/one.md\n---\n\n" + "body " * 100,
        encoding="utf-8",
    )
    # Mock the detector to return a single contradiction with the SAME path.
    monkeypatch.setattr(
        rag, "find_contradictions_for_note",
        lambda *a, **kw: [{"path": "path/one.md", "note": "one", "snippet": "…", "why": "…"}],
    )
    # If the write were attempted we'd see mtime bump; capture.
    mtime_before = f.stat().st_mtime
    rag._check_and_flag_contradictions(None, f, "body " * 100, "c.md")
    mtime_after = f.stat().st_mtime
    assert mtime_after == mtime_before, "file was rewritten despite unchanged set"


def test_check_and_flag_writes_when_set_differs(monkeypatch, tmp_path):
    """If detected set differs, frontmatter IS rewritten."""
    # Aislar DB_PATH: este test SÍ pasa por `_log_contradictions` porque el
    # set detectado difiere del frontmatter. Pre-fix (2026-04-29) el test
    # escribía 261 rows fake (`subject_path="d.md"`, `contradicts_json` con
    # `new/two.md`) a la DB real `~/.local/share/obsidian-rag/ragvec/telemetry.db`
    # cada `pytest` — envenenando `_load_contradiction_priors()` y el tune
    # del ranker (esos paths no existen en el vault, son fixtures del test).
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    f = tmp_path / "d.md"
    f.write_text(
        "---\ncontradicts:\n- old/one.md\n---\n\n" + "body " * 100,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        rag, "find_contradictions_for_note",
        lambda *a, **kw: [{"path": "new/two.md", "note": "two", "snippet": "…", "why": "…"}],
    )
    rag._check_and_flag_contradictions(None, f, "body " * 100, "d.md")
    assert rag._frontmatter_contradicts_set(f) == {"new/two.md"}
