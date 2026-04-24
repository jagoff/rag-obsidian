"""Concurrency tests for `_AGENT_PENDING_WRITES` (the `rag do` agent-tool
pending-writes buffer).

Hoy el único caller productivo es `do()` en un solo thread — pero las
`_agent_tool_*` son importadas por `web/tools.py` y los read-tools ya
corren en un thread pool. Si alguna vez `propose_write` / `append_to_note`
se exponen al web (o si los tools de `rag do` pasan a paralelo), una
race en `.append()` contra un snapshot `list(...)` puede perder entries
o crashear con `RuntimeError: list changed size during iteration`.

Pattern copiado de `tests/test_cache_concurrency.py` — threads reales
(no multiprocessing) porque el lock es in-process.
"""
from __future__ import annotations

import threading

import rag


def test_agent_pending_writes_has_lock():
    """Regression guard: si alguien saca el lock, este falla rápido."""
    assert isinstance(
        rag._AGENT_PENDING_WRITES_lock, type(threading.Lock())
    )


def test_concurrent_appends_preserve_all_entries(monkeypatch):
    """N threads haciendo append concurrente → buffer tiene exactamente N entries."""
    monkeypatch.setattr(rag, "_AGENT_PENDING_WRITES", [])

    n_threads = 50
    ready = threading.Event()
    errors: list[Exception] = []

    def worker(idx: int) -> None:
        ready.wait()
        try:
            with rag._AGENT_PENDING_WRITES_lock:
                rag._AGENT_PENDING_WRITES.append({"idx": idx, "kind": "create"})
        except Exception as e:  # pragma: no cover — defensive
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    ready.set()
    for t in threads:
        t.join()

    assert not errors
    assert len(rag._AGENT_PENDING_WRITES) == n_threads
    seen_idx = {w["idx"] for w in rag._AGENT_PENDING_WRITES}
    assert seen_idx == set(range(n_threads))


def test_snapshot_iteration_safe_during_concurrent_appends(monkeypatch):
    """Snapshot bajo lock + iteración afuera — el snapshot nunca cambia de size
    mientras se itera, aunque otros threads sigan apendeando."""
    monkeypatch.setattr(rag, "_AGENT_PENDING_WRITES", [])

    stop = threading.Event()
    errors: list[Exception] = []

    def appender() -> None:
        try:
            while not stop.is_set():
                with rag._AGENT_PENDING_WRITES_lock:
                    rag._AGENT_PENDING_WRITES.append({"kind": "create", "path": "x.md"})
        except Exception as e:  # pragma: no cover — defensive
            errors.append(e)

    def reader() -> None:
        try:
            for _ in range(50):
                with rag._AGENT_PENDING_WRITES_lock:
                    snapshot = list(rag._AGENT_PENDING_WRITES)
                # Iterar fuera del lock — si snapshot no fuera un shallow copy,
                # explotaría con "list changed size during iteration".
                for _item in snapshot:
                    pass
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=appender) for _ in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()

    # Dejar correr un poco + parar.
    import time
    time.sleep(0.1)
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert not errors, f"race condition detectada: {errors}"


def test_clear_and_append_interleaving_is_atomic(monkeypatch):
    """Thread A limpia, threads B-N apendean. Post-join, buffer tiene SOLO
    entries apendeadas POST-clear (no deberíamos ver residuos pre-clear)."""
    monkeypatch.setattr(rag, "_AGENT_PENDING_WRITES", [])

    # Seed pre-clear con entries "viejos"
    for i in range(10):
        with rag._AGENT_PENDING_WRITES_lock:
            rag._AGENT_PENDING_WRITES.append({"phase": "pre", "i": i})

    clear_done = threading.Event()
    errors: list[Exception] = []

    def clearer() -> None:
        try:
            with rag._AGENT_PENDING_WRITES_lock:
                rag._AGENT_PENDING_WRITES.clear()
            clear_done.set()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def appender(idx: int) -> None:
        # Esperar al clear antes de apendear — si no, dependemos del scheduler.
        clear_done.wait()
        try:
            with rag._AGENT_PENDING_WRITES_lock:
                rag._AGENT_PENDING_WRITES.append({"phase": "post", "i": idx})
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=clearer)]
    threads += [threading.Thread(target=appender, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Todas las entries finales deben ser post-clear.
    phases = {w["phase"] for w in rag._AGENT_PENDING_WRITES}
    assert phases == {"post"}
    assert len(rag._AGENT_PENDING_WRITES) == 20
