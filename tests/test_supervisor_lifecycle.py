"""Tests del lifecycle del supervisor + IPC handlers built-in.

Cubren:
- ``Supervisor.start()`` registra jobs + arranca IPC server + scheduler.
- IPC handler ``ping`` responde rápido.
- IPC handler ``status`` devuelve los jobs registrados.
- IPC handler ``run <job>`` dispara sincrónico via scheduler.run_now.
- Shutdown graceful — IPC server thread termina.

NOTA: usa ``short_tmpdir`` igual que ``test_runtime_ipc`` para evitar
``AF_UNIX path too long`` en pytest.
"""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

from rag.runtime import ipc
from rag.runtime.scheduler import Scheduler, cron, interval


@pytest.fixture
def short_tmpdir():
    d = Path(tempfile.mkdtemp(dir="/tmp", prefix="rag-sup-"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolate_telemetry_db(tmp_path, monkeypatch):
    """Aisla `rag_supervisor_jobs` a un tmp_path per-test.

    Pre-fix (audit 2026-05-11): el handler IPC ``run`` invoca
    ``scheduler.run_now()`` → ``_persist_run()`` →
    ``insert_supervisor_job_run()``, que resolvía ``OBSIDIAN_RAG_DB_PATH``
    en runtime y escribía a la telemetry.db de PROD. Resultado: 16+ rows
    con ``job_label='trigger_test'`` apareciendo en prod cada vez que la
    suite corría. Idéntico patrón al fix de ``test_runtime_scheduler.py``.
    """
    monkeypatch.setenv("OBSIDIAN_RAG_DB_PATH", str(tmp_path))


@pytest.fixture(autouse=True)
def _reset_state():
    import sys as _sys
    Scheduler.reset_global()
    ipc._reset_handlers()
    # Remover jobs modules de sys.modules para que los decorators
    # vuelvan a registrarse contra el nuevo Scheduler.global_instance().
    for mod in list(_sys.modules):
        if mod.startswith("rag.runtime.jobs"):
            del _sys.modules[mod]
    yield
    Scheduler.reset_global()
    ipc._reset_handlers()
    for mod in list(_sys.modules):
        if mod.startswith("rag.runtime.jobs"):
            del _sys.modules[mod]


def _start_supervisor_with_socket(sock_path: Path):
    """Arranca un supervisor con socket override."""
    from rag.runtime import supervisor

    # Patch DEFAULT_SOCKET_PATH para que IPCServer use sock_path.
    sup = supervisor.Supervisor()
    # Reemplazamos manualmente el server para usar el sock corto del test.
    server = ipc.IPCServer(socket_path=sock_path)
    sup._ipc_server = server
    sup._ipc_thread = threading.Thread(
        target=server.serve_forever, name="rag-ipc-test", daemon=True,
    )
    sup._ipc_thread.start()

    # Importar jobs + register IPC handlers built-in usando la API
    # del supervisor.
    supervisor._import_jobs()
    supervisor._register_builtin_ipc()

    Scheduler.global_instance().start()
    return sup


def _wait_for_socket(sock_path: Path, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if sock_path.exists():
            return
        time.sleep(0.05)
    pytest.fail(f"socket {sock_path} no apareció en {timeout}s")


def test_supervisor_ping(short_tmpdir):
    sock = short_tmpdir / "s.sock"
    sup = _start_supervisor_with_socket(sock)
    try:
        _wait_for_socket(sock)
        resp = ipc.client_call("ping", socket_path=sock)
        assert resp["ok"] is True
        assert resp["result"]["pong"] is True
        assert resp["result"]["ts"] > 0
    finally:
        sup.shutdown(timeout=2.0)


def test_supervisor_status_lists_registered_jobs(short_tmpdir):
    sock = short_tmpdir / "s.sock"

    @interval(seconds=900, label="my_test_job")
    def my_job():
        return None

    sup = _start_supervisor_with_socket(sock)
    try:
        _wait_for_socket(sock)
        resp = ipc.client_call("status", socket_path=sock)
        assert resp["ok"] is True
        labels = {j["label"] for j in resp["result"]["jobs"]}
        assert "my_test_job" in labels
        assert resp["result"]["uptime_s"] >= 0
    finally:
        sup.shutdown(timeout=2.0)


def test_supervisor_jobs_list(short_tmpdir):
    sock = short_tmpdir / "s.sock"

    @interval(seconds=60, label="alpha_job")
    def a():
        pass

    @cron(hour=3, label="beta_job")
    def b():
        pass

    sup = _start_supervisor_with_socket(sock)
    try:
        _wait_for_socket(sock)
        resp = ipc.client_call("jobs", socket_path=sock)
        assert resp["ok"] is True
        labels = resp["result"]["labels"]
        assert "alpha_job" in labels
        assert "beta_job" in labels
    finally:
        sup.shutdown(timeout=2.0)


def test_supervisor_run_triggers_job(short_tmpdir):
    sock = short_tmpdir / "s.sock"
    counter = {"n": 0}

    @interval(seconds=900, label="trigger_test")
    def my_job():
        counter["n"] += 1
        return {"counter": counter["n"]}

    sup = _start_supervisor_with_socket(sock)
    try:
        _wait_for_socket(sock)
        resp = ipc.client_call("run", socket_path=sock, job="trigger_test")
        assert resp["ok"] is True
        assert resp["result"]["ok"] is True
        assert resp["result"]["result"] == {"counter": 1}
        assert counter["n"] == 1
    finally:
        sup.shutdown(timeout=2.0)


def test_supervisor_run_unknown_job(short_tmpdir):
    sock = short_tmpdir / "s.sock"
    sup = _start_supervisor_with_socket(sock)
    try:
        _wait_for_socket(sock)
        resp = ipc.client_call("run", socket_path=sock, job="ghost_job")
        # IPC dice ok=true (handler corrió OK), pero el resultado del
        # scheduler dice ok=false (job no existe).
        assert resp["ok"] is True
        assert resp["result"]["ok"] is False
        assert "unknown job" in resp["result"]["error"]
    finally:
        sup.shutdown(timeout=2.0)


def test_supervisor_shutdown_clean(short_tmpdir):
    sock = short_tmpdir / "s.sock"
    sup = _start_supervisor_with_socket(sock)
    _wait_for_socket(sock)
    sup.shutdown(timeout=2.0)
    # Después del shutdown, el socket no debería responder. Damos un
    # poquito de tiempo para que el server thread termine.
    time.sleep(0.3)
    # Cliente puede o bien fallar con ConnectionRefused (socket file
    # quedó pero el server cerró) o FileNotFoundError (server unlinkó).
    with pytest.raises((FileNotFoundError, ConnectionRefusedError, OSError)):
        ipc.client_call("ping", socket_path=sock, timeout=1.0)


def test_import_jobs_loads_drift_watcher():
    """Smoke: ``_import_jobs()`` registra el drift_watcher job."""
    from rag.runtime import supervisor
    Scheduler.reset_global()
    n = supervisor._import_jobs()
    assert n >= 1
    sched = Scheduler.global_instance()
    assert "drift_watcher" in sched.jobs(), (
        f"expected drift_watcher in jobs, got: {list(sched.jobs())}"
    )
