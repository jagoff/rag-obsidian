"""Cooperative lockfile guard for anticipate runs.

Pattern: file lock con `fcntl.flock(LOCK_EX | LOCK_NB)`. Si otro
proceso ya tiene el lock, el segundo aborta inmediatamente (no espera).
Lockfile contiene PID + timestamp del owner — útil para diagnostics.

Uso:
    from rag_anticipate.lockfile import anticipate_lock
    with anticipate_lock() as acquired:
        if not acquired:
            return  # otro process tiene el lock
        # ... do work ...
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path

LOCK_PATH = Path.home() / ".local/share/obsidian-rag/anticipate.lock"


def resolve_lock_path() -> Path:
    """Return the active anticipate lock path.

    `LOCK_PATH` stays as the production default and as the legacy monkeypatch
    point for tests. `RAG_ANTICIPATE_LOCK_PATH` lets test runs and sandboxes
    isolate themselves from a real launchd daemon on the same machine.
    """
    override = os.environ.get("RAG_ANTICIPATE_LOCK_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return LOCK_PATH


@contextmanager
def anticipate_lock(*, timeout_seconds: float = 0.0):
    """Try to acquire the anticipate lock.

    timeout_seconds=0 → non-blocking (default).
    timeout_seconds>0 → waiting up to N seconds.

    yields True si acquired, False si no se pudo.
    Always releases on exit (even on exception)."""
    import fcntl
    import time

    lock_path = resolve_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    acquired = False
    try:
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.time() + timeout_seconds
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                # Write PID + ts para diagnostics
                os.lseek(fd, 0, 0)
                os.ftruncate(fd, 0)
                info = f"pid={os.getpid()} ts={int(time.time())}\n"
                os.write(fd, info.encode("utf-8"))
                break
            except (BlockingIOError, OSError):
                if time.time() >= deadline:
                    break
                time.sleep(0.05)
        yield acquired
    finally:
        if fd is not None:
            try:
                if acquired:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            except Exception:
                pass


def lock_status() -> dict:
    """Diagnostic — read lockfile contents sin acquirir."""
    lock_path = resolve_lock_path()
    if not lock_path.is_file():
        return {"held": False, "pid": None, "ts": None}
    try:
        content = lock_path.read_text(encoding="utf-8").strip()
        # Parse "pid=N ts=T"
        parts = dict(p.split("=", 1) for p in content.split() if "=" in p)
        # Try acquiring non-blocking to see if held
        import fcntl

        fd = os.open(str(lock_path), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = False
            fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            held = True
        finally:
            os.close(fd)
        return {
            "held": held,
            "pid": int(parts.get("pid", "0")) if parts.get("pid", "").isdigit() else None,
            "ts": int(parts.get("ts", "0")) if parts.get("ts", "").isdigit() else None,
        }
    except Exception:
        return {"held": False, "pid": None, "ts": None}
