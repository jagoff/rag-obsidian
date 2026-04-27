"""Concurrency tests for the three module-level caches that gained locks:
`_context_cache`, `_synthetic_q_cache`, `_mentions_cache`.

The production scenario driving these: `web/server.py` calls into `rag` from
multiple request-handler threads (FastAPI + uvicorn's threadpool), and
`rag index` / `rag watch` touch the same caches from background threads.
Without locks a `json.dumps(_cache)` running on one thread and a
`_cache[k] = v` running on another raises `RuntimeError: dictionary changed
size during iteration`; a lazy-init can double-parse the JSON file and
discard one thread's result.

These tests use `threading` (not multiprocessing) because the locks are
in-process. GIL or no GIL, the bugs are reproducible under thread pressure.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import rag


# ── _context_cache ───────────────────────────────────────────────────────────


def _reset_context_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force `_load_context_cache` to re-read from the on-disk file we control."""
    monkeypatch.setattr(rag, "CONTEXT_CACHE_PATH", tmp_path / "ctx.json")
    monkeypatch.setattr(rag, "_context_cache", None)
    monkeypatch.setattr(rag, "_context_cache_dirty", False)


def test_context_cache_has_lock():
    """Regression guard: if someone removes the lock, this fails fast."""
    assert isinstance(rag._context_cache_lock, type(threading.Lock()))


def test_context_cache_lazy_init_runs_once_under_contention(tmp_path, monkeypatch):
    """20 threads calling `_load_context_cache` on a cold module must end up
    with the SAME dict object (one load, shared reference)."""
    _reset_context_cache(tmp_path, monkeypatch)
    (tmp_path / "ctx.json").write_text(json.dumps({"seed": "summary"}))

    results: list[dict] = []
    ready = threading.Event()
    errors: list[Exception] = []

    def worker():
        ready.wait()
        try:
            results.append(rag._load_context_cache())
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    ready.set()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors
    assert len(results) == 20
    first = results[0]
    assert all(r is first for r in results), "lock failed: parallel re-loads"
    assert first == {"seed": "summary"}


def test_context_cache_save_during_mutation_is_safe(tmp_path, monkeypatch):
    """A thread calling `_save_context_cache` must not see `RuntimeError:
    dictionary changed size during iteration` while other threads mutate."""
    _reset_context_cache(tmp_path, monkeypatch)
    # Pre-populate via the lazy loader so the subsequent mutation path doesn't
    # race the initial lazy load itself (that's the other test).
    rag._load_context_cache()

    errors: list[Exception] = []
    stop = threading.Event()

    def mutator(start: int):
        i = start
        while not stop.is_set():
            # Matches the production write path in `get_context_summary`.
            with rag._context_cache_lock:
                rag._context_cache[f"h{i}"] = "x" * 40
                rag._context_cache_dirty = True
            i += 1

    def saver():
        try:
            while not stop.is_set():
                rag._save_context_cache()
        except Exception as e:
            errors.append(e)

    writers = [threading.Thread(target=mutator, args=(s * 10_000,)) for s in range(4)]
    savers = [threading.Thread(target=saver) for _ in range(2)]
    for t in writers + savers:
        t.start()
    # Short window is plenty: without the lock this regresses within ~100ms
    # on a warm process.
    threading.Event().wait(0.3)
    stop.set()
    for t in writers + savers:
        t.join(timeout=5.0)

    assert not errors, f"save crashed mid-mutation: {errors[:3]}"


# ── _synthetic_q_cache ───────────────────────────────────────────────────────


def _reset_sq_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag, "SYNTHETIC_Q_CACHE_PATH", tmp_path / "sq.json")
    monkeypatch.setattr(rag, "_synthetic_q_cache", None)
    monkeypatch.setattr(rag, "_synthetic_q_cache_dirty", False)


def test_synthetic_q_cache_has_lock():
    assert isinstance(rag._synthetic_q_cache_lock, type(threading.Lock()))


def test_synthetic_q_cache_lazy_init_runs_once_under_contention(tmp_path, monkeypatch):
    _reset_sq_cache(tmp_path, monkeypatch)
    (tmp_path / "sq.json").write_text(json.dumps({"seed": ["¿qué?"]}))

    results: list[dict] = []
    ready = threading.Event()
    errors: list[Exception] = []

    def worker():
        ready.wait()
        try:
            results.append(rag._load_synthetic_q_cache())
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    ready.set()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors
    assert len(results) == 20
    first = results[0]
    assert all(r is first for r in results)
    assert first == {"seed": ["¿qué?"]}


def test_synthetic_q_cache_save_during_mutation_is_safe(tmp_path, monkeypatch):
    _reset_sq_cache(tmp_path, monkeypatch)
    rag._load_synthetic_q_cache()

    errors: list[Exception] = []
    stop = threading.Event()

    def mutator(start: int):
        i = start
        while not stop.is_set():
            with rag._synthetic_q_cache_lock:
                rag._synthetic_q_cache[f"h{i}"] = [f"q-{i}"]
                rag._synthetic_q_cache_dirty = True
            i += 1

    def saver():
        try:
            while not stop.is_set():
                rag._save_synthetic_q_cache()
        except Exception as e:
            errors.append(e)

    writers = [threading.Thread(target=mutator, args=(s * 10_000,)) for s in range(4)]
    savers = [threading.Thread(target=saver) for _ in range(2)]
    for t in writers + savers:
        t.start()
    threading.Event().wait(0.3)
    stop.set()
    for t in writers + savers:
        t.join(timeout=5.0)

    assert not errors, f"save crashed mid-mutation: {errors[:3]}"


# ── _mentions_cache ──────────────────────────────────────────────────────────


def test_mentions_cache_has_lock():
    assert isinstance(rag._mentions_cache_lock, type(threading.Lock()))


def test_mentions_cache_concurrent_load_is_consistent(tmp_path, monkeypatch):
    """N threads calling `_load_mentions_index` simultaneously must all get
    the same {token: path} mapping, and nothing must crash."""
    vault = tmp_path / "vault"
    (vault / rag._MENTIONS_FOLDER).mkdir(parents=True)
    (vault / rag._MENTIONS_FOLDER / "Alice.md").write_text("- apellido: Smith\n")
    (vault / rag._MENTIONS_FOLDER / "Bob.md").write_text("- apellido: Jones\n")
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "_mentions_cache", None)

    results: list[dict] = []
    ready = threading.Event()
    errors: list[Exception] = []

    def worker():
        ready.wait()
        try:
            results.append(rag._load_mentions_index(vault))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    ready.set()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors
    assert len(results) == 16
    assert all(r == results[0] for r in results)
    assert rag._fold("alice") in results[0]
    assert rag._fold("bob") in results[0]


# ── _corpus_cache single-flight ───────────────────────────────────────────────


class _FakeCol:
    """Minimal SqliteVecCollection stub for corpus-cache tests.

    `col.get()` is the expensive call (~50-200ms in production). We make it
    trackable via a counter so the test can assert single-flight semantics.
    """

    def __init__(self, docs: list[str], metas: list[dict]):
        self._docs = docs
        self._metas = metas
        self.get_call_count = 0
        self._lock = threading.Lock()
        self.id = "fake-col-id"

    def count(self) -> int:
        return len(self._docs)

    def get(self, include=None):
        with self._lock:
            self.get_call_count += 1
        return {
            "documents": list(self._docs),
            "ids": [f"id{i}" for i in range(len(self._docs))],
            "metadatas": list(self._metas),
        }


def test_corpus_cache_has_rlock():
    """Guard: _corpus_cache_lock must be an RLock (allows recursive acquisition
    from within the same thread, needed for the single-flight pattern)."""
    import _thread

    assert isinstance(rag._corpus_cache_lock, type(threading.RLock()))


def test_corpus_cache_single_flight_on_concurrent_miss(monkeypatch):
    """Two threads hitting a cold cache must trigger col.get() exactly once.

    Pre-fix: both threads passed the 'cached is None' check, released the
    lock, then raced through BM25Okapi rebuild in parallel — col.get() was
    called twice and the second overwrite discarded the first result.
    Post-fix: the lock is held for the entire rebuild, so the second thread
    re-checks after the first finishes and returns the cached dict.
    """
    col = _FakeCol(
        docs=["nota sobre correr", "nota sobre meditar"],
        metas=[
            {"note": "Correr", "file": "01-Projects/correr.md", "tags": "salud", "folder": "01-Projects"},
            {"note": "Meditar", "file": "02-Areas/meditar.md", "tags": "bienestar", "folder": "02-Areas"},
        ],
    )

    # Force cold cache so both threads see a miss.
    monkeypatch.setattr(rag, "_corpus_cache", None)

    # Gate: both threads start col.get() only after both have started.
    results: list[dict] = []
    ready = threading.Event()
    errors: list[Exception] = []

    def worker():
        ready.wait()
        try:
            results.append(rag._load_corpus(col))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    ready.set()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"workers raised: {errors}"
    assert len(results) == 8

    # All threads must return the same dict object (shared reference from cache).
    first = results[0]
    assert all(r is first for r in results), (
        "single-flight broken: threads returned different dict objects"
    )

    # col.get() must have been called exactly once regardless of concurrency.
    assert col.get_call_count == 1, (
        f"col.get() called {col.get_call_count} times — parallel rebuilds occurred"
    )


def test_corpus_cache_invalidate_wins_over_concurrent_rebuild(monkeypatch):
    """An invalidation that arrives while the rebuild lock is held must NOT
    be overwritten by the stale rebuild result.

    Pre-fix: invalidate() cleared the cache under its own lock acquisition,
    but the rebuilding thread then did a second `with _corpus_cache_lock: _corpus_cache = new_cache`
    — restoring the cache that invalidate had just cleared.
    Post-fix: a single lock covers the entire rebuild, so the sequence
    cannot interleave.

    We simulate this by: (1) doing a full rebuild to warm the cache,
    (2) invalidating, (3) asserting the cache is None (not restored).
    """
    col = _FakeCol(
        docs=["una nota"],
        metas=[{"note": "Una", "file": "01-Projects/una.md", "tags": "", "folder": "01-Projects"}],
    )

    monkeypatch.setattr(rag, "_corpus_cache", None)
    monkeypatch.setattr(rag, "_pagerank_cache", None)
    monkeypatch.setattr(rag, "_pagerank_cache_cid", None)

    # Warm the cache.
    rag._load_corpus(col)
    assert rag._corpus_cache is not None

    # Invalidate — must zero out the cache.
    rag._invalidate_corpus_cache()
    assert rag._corpus_cache is None, (
        "invalidate() did not clear the cache — rebuild may have overwritten it"
    )
