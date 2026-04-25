"""Tests for the cooperative lockfile guard used by the Anticipatory Agent.

Cubre:
1. anticipate_lock acquired cuando lockfile no existe
2. anticipate_lock writes pid + ts to file
3. Lock released on context exit (lockfile sigue existiendo pero unlocked)
4. Second concurrent acquire returns acquired=False (test con thread)
5. Lock released after exception en context body
6. timeout_seconds=0 fails inmediatamente si held
7. timeout_seconds=0.5 espera y eventualmente acquire
8. lock_status sin lockfile → held=False, pid=None
9. lock_status mientras held → held=True, pid correcto
10. lock_status post-release → held=False
11. Concurrent threads: solo uno acquire a la vez

NOTA sobre threads + flock: en Linux/macOS `fcntl.flock` es per-process
(advisory). Dos threads en el MISMO process se ven como un sólo holder y
pueden re-acquirir. Por eso los tests de "concurrencia real" se hacen con
subprocesses (vía multiprocessing), no con threads. Los tests con threads
sólo verifican que el contextmanager no se rompa cuando se llama desde
varios threads.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from rag_anticipate import lockfile


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_lock(tmp_path, monkeypatch):
    """Punto de aislamiento del lockfile real del user.

    Apunta `LOCK_PATH` a `tmp_path/anticipate.lock` — cada test arranca con
    un lockfile inexistente y nunca toca `~/.local/share/obsidian-rag/`.
    """
    p = tmp_path / "anticipate.lock"
    monkeypatch.setattr(lockfile, "LOCK_PATH", p)
    return p


# ── Helpers (subprocess workers) ─────────────────────────────────────────────
#
# Los workers están a top-level del módulo para que `multiprocessing` (con
# spawn/fork) los pueda picklear correctamente. NO usar closures.


def _hold_then_release(lock_path: str, hold_seconds: float, ready_path: str) -> None:
    """Worker: acquire, escribir un fichero 'ready' para que el padre sepa
    que el hijo agarró el lock, mantener el lock `hold_seconds`, y liberar."""
    from rag_anticipate import lockfile as _lf

    _lf.LOCK_PATH = Path(lock_path)
    with _lf.anticipate_lock() as ok:
        Path(ready_path).write_text("1" if ok else "0", encoding="utf-8")
        if ok:
            time.sleep(hold_seconds)


def _try_acquire_blocking(lock_path: str, timeout_seconds: float, result_path: str) -> None:
    """Worker: intenta acquire con `timeout_seconds` y escribe 'true'/'false'."""
    from rag_anticipate import lockfile as _lf

    _lf.LOCK_PATH = Path(lock_path)
    with _lf.anticipate_lock(timeout_seconds=timeout_seconds) as ok:
        Path(result_path).write_text("true" if ok else "false", encoding="utf-8")


def _try_acquire_nonblocking(lock_path: str, result_path: str) -> None:
    """Worker: intenta acquire non-blocking y escribe 'true'/'false'."""
    from rag_anticipate import lockfile as _lf

    _lf.LOCK_PATH = Path(lock_path)
    with _lf.anticipate_lock() as ok:
        Path(result_path).write_text("true" if ok else "false", encoding="utf-8")


def _spawn_ctx() -> multiprocessing.context.BaseContext:
    """Use 'spawn' to avoid inheriting fds from the parent (más reproducible
    en macOS donde 'fork' es deprecated)."""
    return multiprocessing.get_context("spawn")


# ── Tests ────────────────────────────────────────────────────────────────────


def test_acquire_when_no_lockfile(tmp_lock):
    """1. anticipate_lock acquired cuando lockfile no existe."""
    assert not tmp_lock.exists()
    with lockfile.anticipate_lock() as acquired:
        assert acquired is True
    # Después del context el lockfile sigue existiendo (sólo se libera el lock).
    assert tmp_lock.exists()


def test_writes_pid_and_ts(tmp_lock):
    """2. anticipate_lock writes pid + ts to file."""
    before = int(time.time())
    with lockfile.anticipate_lock() as acquired:
        assert acquired is True
        content = tmp_lock.read_text(encoding="utf-8").strip()
    after = int(time.time())

    assert "pid=" in content
    assert "ts=" in content
    parts = dict(p.split("=", 1) for p in content.split() if "=" in p)
    assert int(parts["pid"]) == os.getpid()
    ts = int(parts["ts"])
    assert before <= ts <= after


def test_released_on_context_exit(tmp_lock):
    """3. Lock released on context exit — segundo acquire del MISMO process
    debería volver a triunfar (y un subprocess también)."""
    with lockfile.anticipate_lock() as acquired:
        assert acquired is True

    # Same process re-acquire debería funcionar.
    with lockfile.anticipate_lock() as acquired_again:
        assert acquired_again is True

    # Y un subprocess debería poder acquirir limpio (lock liberado de verdad).
    ctx = _spawn_ctx()
    result_path = tmp_lock.parent / "result.txt"
    p = ctx.Process(
        target=_try_acquire_nonblocking,
        args=(str(tmp_lock), str(result_path)),
    )
    p.start()
    p.join(timeout=5)
    assert p.exitcode == 0
    assert result_path.read_text(encoding="utf-8") == "true"


def test_concurrent_acquire_fails(tmp_lock):
    """4. Second concurrent acquire returns acquired=False.

    Usa subprocess (no threads) porque flock es per-process. El padre arranca
    al hijo, espera a que el hijo escriba el 'ready' file (= ya tiene el
    lock), e intenta acquire — debe fallar.
    """
    ctx = _spawn_ctx()
    ready_path = tmp_lock.parent / "ready.txt"

    holder = ctx.Process(
        target=_hold_then_release,
        args=(str(tmp_lock), 1.5, str(ready_path)),
    )
    holder.start()
    try:
        # Esperar a que el hijo escriba 'ready' = ya tiene el lock.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if ready_path.exists() and ready_path.read_text(encoding="utf-8") == "1":
                break
            time.sleep(0.02)
        else:
            holder.terminate()
            pytest.fail("holder subprocess never signalled ready")

        # Ahora el padre intenta acquire non-blocking → debe fallar.
        with lockfile.anticipate_lock() as acquired:
            assert acquired is False
    finally:
        holder.join(timeout=5)


def test_released_after_exception(tmp_lock):
    """5. Lock released after exception en context body."""

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        with lockfile.anticipate_lock() as acquired:
            assert acquired is True
            raise _Boom("kaboom")

    # Subprocess debería poder acquirir limpio.
    ctx = _spawn_ctx()
    result_path = tmp_lock.parent / "result.txt"
    p = ctx.Process(
        target=_try_acquire_nonblocking,
        args=(str(tmp_lock), str(result_path)),
    )
    p.start()
    p.join(timeout=5)
    assert p.exitcode == 0
    assert result_path.read_text(encoding="utf-8") == "true"


def test_timeout_zero_fails_immediately(tmp_lock):
    """6. timeout_seconds=0 fails inmediatamente si held (no espera)."""
    ctx = _spawn_ctx()
    ready_path = tmp_lock.parent / "ready.txt"

    holder = ctx.Process(
        target=_hold_then_release,
        args=(str(tmp_lock), 1.5, str(ready_path)),
    )
    holder.start()
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if ready_path.exists() and ready_path.read_text(encoding="utf-8") == "1":
                break
            time.sleep(0.02)
        else:
            holder.terminate()
            pytest.fail("holder subprocess never signalled ready")

        t0 = time.time()
        with lockfile.anticipate_lock(timeout_seconds=0.0) as acquired:
            assert acquired is False
        elapsed = time.time() - t0
        # Non-blocking debería retornar bien rápido (<200ms con holgura).
        assert elapsed < 0.2, f"expected fast fail, took {elapsed:.3f}s"
    finally:
        holder.join(timeout=5)


def test_timeout_waits_and_eventually_acquires(tmp_lock):
    """7. timeout_seconds>0 espera hasta que el holder libere y acquire."""
    ctx = _spawn_ctx()
    ready_path = tmp_lock.parent / "ready.txt"

    # Holder mantiene el lock 0.4s.
    holder = ctx.Process(
        target=_hold_then_release,
        args=(str(tmp_lock), 0.4, str(ready_path)),
    )
    holder.start()
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if ready_path.exists() and ready_path.read_text(encoding="utf-8") == "1":
                break
            time.sleep(0.02)
        else:
            holder.terminate()
            pytest.fail("holder subprocess never signalled ready")

        t0 = time.time()
        with lockfile.anticipate_lock(timeout_seconds=2.0) as acquired:
            elapsed = time.time() - t0
            assert acquired is True
            # Debió esperar > 0 (porque al inicio estaba held), pero < 2s
            # (porque el holder libera a ~0.4s).
            assert elapsed < 2.0
    finally:
        holder.join(timeout=5)


def test_lock_status_no_lockfile(tmp_lock):
    """8. lock_status sin lockfile → held=False, pid=None, ts=None."""
    assert not tmp_lock.exists()
    status = lockfile.lock_status()
    assert status == {"held": False, "pid": None, "ts": None}


def test_lock_status_while_held(tmp_lock):
    """9. lock_status mientras held → held=True, pid correcto.

    Necesita un proceso externo (flock per-process: si lo testeamos en el
    mismo process, lock_status() reabre el fd y agarra el lock libremente).
    """
    ctx = _spawn_ctx()
    ready_path = tmp_lock.parent / "ready.txt"

    holder = ctx.Process(
        target=_hold_then_release,
        args=(str(tmp_lock), 1.5, str(ready_path)),
    )
    holder.start()
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if ready_path.exists() and ready_path.read_text(encoding="utf-8") == "1":
                break
            time.sleep(0.02)
        else:
            holder.terminate()
            pytest.fail("holder subprocess never signalled ready")

        status = lockfile.lock_status()
        assert status["held"] is True
        assert status["pid"] == holder.pid
        assert isinstance(status["ts"], int)
        assert status["ts"] > 0
    finally:
        holder.join(timeout=5)


def test_lock_status_post_release(tmp_lock):
    """10. lock_status post-release → held=False (file existe pero unlocked)."""
    with lockfile.anticipate_lock() as acquired:
        assert acquired is True

    assert tmp_lock.exists()
    status = lockfile.lock_status()
    assert status["held"] is False
    # PID + ts del último holder siguen siendo legibles del file.
    assert status["pid"] == os.getpid()
    assert isinstance(status["ts"], int)


def test_concurrent_processes_only_one_acquires(tmp_lock):
    """11. Concurrent processes: solo uno acquire a la vez.

    Lanza N procesos simultáneamente; cada uno intenta non-blocking acquire
    y reporta true/false. Exactamente UNO debe haber acquired.
    """
    ctx = _spawn_ctx()
    n = 4
    procs = []
    result_paths = []

    # Para forzar competencia: un primer holder mantiene el lock, y dentro
    # de su ventana lanzamos N "tryers" non-blocking. Todos deben fallar.
    ready_path = tmp_lock.parent / "ready.txt"
    holder = ctx.Process(
        target=_hold_then_release,
        args=(str(tmp_lock), 1.0, str(ready_path)),
    )
    holder.start()
    try:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if ready_path.exists() and ready_path.read_text(encoding="utf-8") == "1":
                break
            time.sleep(0.02)
        else:
            holder.terminate()
            pytest.fail("holder subprocess never signalled ready")

        for i in range(n):
            rp = tmp_lock.parent / f"result_{i}.txt"
            result_paths.append(rp)
            p = ctx.Process(
                target=_try_acquire_nonblocking,
                args=(str(tmp_lock), str(rp)),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join(timeout=5)
            assert p.exitcode == 0

        # Mientras el holder tiene el lock, NINGÚN tryer debió haber acquired.
        results = [rp.read_text(encoding="utf-8") for rp in result_paths]
        assert all(r == "false" for r in results), f"unexpected: {results}"
    finally:
        holder.join(timeout=5)


def test_acquire_after_holder_releases(tmp_lock):
    """12 (bonus). Después de que el holder libera, un nuevo acquire triunfa
    inmediatamente — verifica que el path "release del subprocess termina
    el lock" funciona.
    """
    ctx = _spawn_ctx()
    ready_path = tmp_lock.parent / "ready.txt"

    holder = ctx.Process(
        target=_hold_then_release,
        args=(str(tmp_lock), 0.3, str(ready_path)),
    )
    holder.start()
    holder.join(timeout=5)
    assert holder.exitcode == 0

    # Holder ya terminó → el lock está libre.
    with lockfile.anticipate_lock() as acquired:
        assert acquired is True
