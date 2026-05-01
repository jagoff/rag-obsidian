"""Protege la invariante documentada en CLAUDE.md:

    > per variant: sqlite-vec sem + BM25 (accent-normalised, GIL-serialised
    > — do NOT parallelise)

y el comentario inline en `retrieve()` (rag.py, sobre el loop
`for v, q_embed in zip(variants, variant_embeds)`):

    > Sequential — sqlite-vec + BM25Okapi both hold a GIL-bound mutex, so
    > ThreadPoolExecutor over these serialises anyway AND adds per-task
    > overhead. Measured: parallel 3× slower than sequential on M3 Max.

Historia: se midió 3× slower en paralelo sobre M3 Max — introducir
`ThreadPoolExecutor` sobre `bm25_search` es una "optimización" que regresa
performance silenciosamente (no rompe tests, solo empeora P50/P95). Estos
tests bloquean esa regresión:

1. `test_retrieve_per_variant_loop_is_sequential_source_inspection`
   — inspecciona el source de `rag.retrieve()` y bloquea la aparición de
   `ThreadPoolExecutor` / `concurrent.futures` / `threading.Thread(` /
   `asyncio.gather` dentro del bloque per-variant. Protección directa contra
   "alguien optimiza con un pool".

2. `test_bm25_search_concurrent_consistency`
   — 10 threads llamando `bm25_search` concurrentemente con un corpus
   monkeypatcheado deben devolver resultados idénticos. Protege correctness
   bajo concurrencia (web server llama retrieve desde múltiples threads).

3. `test_bm25_search_serial_when_guarded` (instrumentation-based)
   — si alguna vez rag.py agrega un lock/guard explícito (`_bm25_lock` o
   similar) en `bm25_search`, este test verifica que el lock efectivamente
   serializa. Skip por default (no hay guard hoy); se auto-activa cuando
   alguno de los nombres conocidos aparezca en el módulo.
"""
from __future__ import annotations

import inspect
import os
import threading
import time

import pytest

# Memory-pressure watchdog no se dispara en unit tests, pero varios paths de
# rag.py lo consultan al import; dejar el flag explícito evita sorpresas si
# conftest cambia.
os.environ.setdefault("RAG_MEMORY_PRESSURE_DISABLE", "1")

import rag


# ── Test 1: source inspection — the direct "optimization" guard ──────────────


def test_retrieve_per_variant_loop_is_sequential_source_inspection():
    """The per-variant BM25 dispatch inside retrieve() MUST remain a plain
    `for` loop. Wrapping in a thread pool measured 3× slower on M3 Max
    (CLAUDE.md). This test blocks the regression at review time.
    """
    src = inspect.getsource(rag.retrieve)
    # Presence check first — if retrieve() is refactored, this assertion fires
    # BEFORE the banned-pattern scan so the test maintainer updates the anchor.
    assert "for v, q_embed in zip(variants, variant_embeds)" in src, (
        "The per-variant BM25/sem loop in rag.retrieve() has been refactored; "
        "update this invariant test to track the new shape."
    )
    banned = (
        "ThreadPoolExecutor",
        "concurrent.futures",
        "asyncio.gather",
        # threading.Thread(  — narrow match: allow `threading.Lock` / `RLock`
        # (legitimate guards), block explicit Thread spawns over the loop.
        "threading.Thread(",
        "multiprocessing.Pool",
    )
    # Scope the check to the per-variant block only. A `ThreadPoolExecutor`
    # elsewhere in retrieve() (unrelated fan-out) would not violate the
    # BM25/sem invariant — don't false-positive on it.
    loop_start = src.index("for v, q_embed in zip(variants, variant_embeds)")
    tail_markers = ('_timing["sem_ms"]', "_timing['sem_ms']")
    loop_end = -1
    for marker in tail_markers:
        idx = src.find(marker, loop_start)
        if idx != -1:
            loop_end = idx
            break
    assert loop_end > loop_start, (
        "Could not locate end of per-variant loop; expected a "
        "`_timing['sem_ms']` write after the loop body."
    )
    loop_src = src[loop_start:loop_end]
    for pattern in banned:
        assert pattern not in loop_src, (
            f"Banned parallelism primitive {pattern!r} found inside the "
            f"per-variant BM25/sem dispatch of rag.retrieve(). BM25 is "
            f"GIL-serialised per CLAUDE.md line 126 (`do NOT parallelise` — "
            f"measured 3× slower on M3 Max). Keep the plain `for v, q_embed` "
            f"loop."
        )


# ── Test 2: concurrent correctness ───────────────────────────────────────────


class _DeterministicBM25:
    """Minimal stub compatible with BM25Okapi's surface used by
    `bm25_search`. `get_scores` returns a fresh list per call so concurrent
    callers cannot mutate shared state — mimics rank_bm25's numpy array
    allocation pattern."""

    def __init__(self, scores):
        self._scores = scores

    def get_scores(self, _tokens):
        return list(self._scores)


def _fake_corpus(n: int = 20) -> dict:
    ids = [f"id_{i}" for i in range(n)]
    # Monotone-decreasing scores → deterministic top-k ordering.
    scores = [float(n - i) for i in range(n)]
    metas = [
        {"file": f"folder/note_{i}.md", "tags": "", "created_ts": 1000.0 + i}
        for i in range(n)
    ]
    return {
        "bm25": _DeterministicBM25(scores),
        "ids": ids,
        "metas": metas,
        "tags": set(),
        "folders": set(),
        "vocab": set(),
    }


def test_bm25_search_concurrent_consistency(monkeypatch):
    """10 threads calling `bm25_search` concurrently must all get the same
    top-k. A torn-read race on the corpus cache (watchdog invalidates while
    a query thread is mid-get_scores) would produce truncated/shuffled ID
    lists. Passes today thanks to `_corpus_cache_lock` + rank_bm25's
    internal thread-safety; blocks regressions that remove either.
    """
    corpus = _fake_corpus(n=20)
    monkeypatch.setattr(rag, "_load_corpus", lambda _col: corpus)

    NUM_THREADS = 10
    results: list = [None] * NUM_THREADS
    errors: list = []
    start_barrier = threading.Barrier(NUM_THREADS)

    def worker(i: int) -> None:
        try:
            start_barrier.wait(timeout=5.0)
            results[i] = rag.bm25_search(
                col=None, query="lenguaje seguro y rapido", k=5,
                folder=None, tag=None, date_range=None,
            )
        except BaseException as exc:  # noqa: BLE001 — re-raised via post-join check
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"bm25_search raised under concurrency: {errors!r}"
    assert all(r is not None for r in results), "one or more threads never wrote a result"

    reference = results[0]
    for i, r in enumerate(results[1:], start=1):
        assert r == reference, (
            f"Thread {i} returned different results than thread 0 — "
            f"bm25_search is not concurrency-safe. "
            f"Expected {reference!r}, got {r!r}."
        )
    # Sanity: top-1 is the highest-score id (n=20, scores monotone decreasing
    # with index → id_0 wins).
    assert reference[0] == "id_0"


# ── Test 3: guard instrumentation (conditional) ──────────────────────────────


_KNOWN_GUARD_NAMES = (
    "_bm25_lock",
    "_bm25_search_lock",
    "_bm25_guard",
    "_bm25_semaphore",
    "_bm25_inflight_lock",
)


def _find_bm25_guard():
    """Return (name, obj) if rag.py exposes a runtime guard for bm25_search."""
    for name in _KNOWN_GUARD_NAMES:
        obj = getattr(rag, name, None)
        if obj is not None:
            return name, obj
    return None


def test_bm25_search_serial_when_guarded(monkeypatch):
    """When a future rag-retrieval commit adds a module-level lock/guard on
    `bm25_search` (e.g. `_bm25_lock = threading.Lock()` wrapped around the
    body), this test verifies the guard *actually* serialises callers.
    Skip cleanly when no guard exists — Test 1 + Test 2 cover the invariant
    meanwhile.
    """
    guard_info = _find_bm25_guard()
    if guard_info is None:
        pytest.skip(
            "No bm25_search guard present in rag.py today. Protection lives "
            "at the retrieve()-caller level (see "
            "test_retrieve_per_variant_loop_is_sequential_source_inspection). "
            "This test auto-activates when rag.py exposes one of: "
            + ", ".join(_KNOWN_GUARD_NAMES)
        )

    active = 0
    max_active = 0
    counter_lock = threading.Lock()

    class _TracingBM25(_DeterministicBM25):
        def get_scores(self, tokens):
            nonlocal active, max_active
            with counter_lock:
                active += 1
                if active > max_active:
                    max_active = active
            try:
                # Sleep releases the GIL — without a guard, multiple threads
                # would land in this block simultaneously.
                time.sleep(0.02)
                return super().get_scores(tokens)
            finally:
                with counter_lock:
                    active -= 1

    corpus = _fake_corpus(n=10)
    corpus["bm25"] = _TracingBM25([float(10 - i) for i in range(10)])
    monkeypatch.setattr(rag, "_load_corpus", lambda _col: corpus)

    NUM_THREADS = 8
    errors: list = []
    successes: list = [None] * NUM_THREADS
    start_barrier = threading.Barrier(NUM_THREADS)

    def worker(i: int) -> None:
        try:
            start_barrier.wait(timeout=5.0)
            successes[i] = rag.bm25_search(
                col=None, query="x", k=3,
                folder=None, tag=None, date_range=None,
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    # 2026-05-01: el lock cambió de `blocking=False + raise` a
    # `acquire(timeout=30.0)`. El nuevo comportamiento serializa los
    # callers en lugar de fallar — TODOS los threads deben completar OK.
    # max_active sigue siendo ≤1 porque el lock garantiza exclusión mutua.
    guard_name = guard_info[0]
    assert not errors, (
        f"bm25_search debería esperar (acquire timeout=30s) en lugar de "
        f"fallar bajo contención, pero {len(errors)} threads tiraron "
        f"excepción: {errors!r}"
    )
    assert all(r is not None for r in successes), (
        f"Algún thread no completó: successes={successes!r}"
    )
    assert max_active <= 1, (
        f"rag.{guard_name} is present but did NOT serialise callers: "
        f"observed max_active={max_active} concurrent get_scores calls "
        f"under {NUM_THREADS} threads. The guard should wrap the whole "
        f"bm25_search body (not just the cache read)."
    )
    # Sanity: el lock corrió cada thread serialmente (max_active==1) y
    # todos los threads vieron al mismo top-k (consistency bajo serialización).
    reference = successes[0]
    for i, r in enumerate(successes[1:], start=1):
        assert r == reference, (
            f"Thread {i} returned different results than thread 0 — "
            f"serialised but not consistent. Expected {reference!r}, got {r!r}."
        )
