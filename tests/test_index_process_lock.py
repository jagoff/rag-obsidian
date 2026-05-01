"""Tests del mutex `INDEX_PROCESS_LOCK` que protege `rag index`.

El lock previene que dos invocaciones de `rag index` corran simultáneamente
contra el mismo dataset, doblando la carga sobre Ollama embeddings y la
contención sobre el sqlite-vec write lock. Bug histórico (2026-05-01):
una sesión devin paralela mantuvo un retry-loop bash de `rag index
--no-contradict` 3h mientras el plist scheduled `ingest-safari` disparaba
otro `rag index --source safari` cada 6h — Ollama saturado, chat web pegado.

Ver también el docstring de `INDEX_PROCESS_LOCK` y `_index_process_lock()`
en `rag/__init__.py`.
"""
from __future__ import annotations

import os
import threading

import pytest

from rag import (
    INDEX_PROCESS_LOCK,
    _index_process_lock,
    _peek_index_lock_holder,
)


@pytest.fixture
def isolated_lock_path(tmp_path, monkeypatch):
    """Reemplaza `INDEX_PROCESS_LOCK` por un tmp path para no chocar
    con cualquier `rag index` real corriendo en la máquina de tests."""
    lock_path = tmp_path / "index.lock"
    # The constant is bound at import time; rebind in the module's namespace.
    import rag
    monkeypatch.setattr(rag, "INDEX_PROCESS_LOCK", lock_path)
    yield lock_path
    # tmp_path is auto-cleaned by pytest


def test_acquires_when_free(isolated_lock_path):
    """Lock libre → adquirible sin problemas."""
    with _index_process_lock() as fh:
        assert fh is not None
        assert isolated_lock_path.exists()


def test_writes_holder_metadata(isolated_lock_path):
    """El lock file contiene PID + ISO timestamp del holder."""
    with _index_process_lock():
        holder = _peek_index_lock_holder()
        assert str(os.getpid()) in holder
        # ISO 8601 UTC timestamp shape: 2026-...T...+00:00
        assert "T" in holder
        assert "+00:00" in holder or "Z" in holder


def test_blocks_second_acquisition(isolated_lock_path):
    """Mientras el lock está tomado, una segunda adquisición debe levantar
    `BlockingIOError`. Eso es lo que la CLI captura para imprimir el mensaje
    `Otro rag index activo` y abortar con exit 1."""
    with _index_process_lock():
        with pytest.raises(BlockingIOError):
            with _index_process_lock():
                pass  # no debería llegar acá


def test_releases_on_normal_exit(isolated_lock_path):
    """Tras salir del `with`, el lock se libera y un `rag index` posterior
    puede adquirirlo. Sin esto, el lock quedaría tomado para siempre y el
    plist scheduled rebotaría todas las invocaciones siguientes."""
    with _index_process_lock():
        pass
    # Re-acquirable
    with _index_process_lock():
        pass


def test_releases_on_exception_inside_with(isolated_lock_path):
    """Si el body del `with` levanta una excepción, el lock igual se libera
    (es lo que garantiza el `__exit__` del context manager). Verificamos
    que la próxima adquisición no rebota."""
    with pytest.raises(RuntimeError):
        with _index_process_lock():
            raise RuntimeError("simulated crash mid-index")
    # El lock se debe haber liberado a pesar de la excepción
    with _index_process_lock():
        pass


def test_blocks_across_threads(isolated_lock_path):
    """`fcntl.flock` es per-process en algunos sistemas y per-FD en otros.
    En macOS/Linux, dos threads del MISMO proceso abriendo el MISMO archivo
    en file descriptors distintos sí compiten. Verificamos que el comportamiento
    funciona como mutex inter-thread (que es lo que querríamos in real-world
    si dos threads dentro del mismo Python intentan correr `rag index`)."""
    holding = threading.Event()
    blocked = threading.Event()
    second_failed = threading.Event()

    def hold_lock():
        with _index_process_lock():
            holding.set()
            blocked.wait(timeout=5)

    def try_second():
        holding.wait(timeout=5)
        try:
            with _index_process_lock():
                # Debería NO llegar acá
                pass
        except BlockingIOError:
            second_failed.set()

    t1 = threading.Thread(target=hold_lock)
    t2 = threading.Thread(target=try_second)
    t1.start()
    t2.start()
    # Esperar a que el segundo thread intente y rebote
    second_failed.wait(timeout=5)
    blocked.set()  # liberar el primer thread
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert second_failed.is_set()


def test_peek_holder_returns_empty_when_no_file(tmp_path, monkeypatch):
    """`_peek_index_lock_holder` debe retornar string vacío si el archivo
    no existe — el caller usa eso para mostrar 'desconocido' en el error."""
    import rag
    monkeypatch.setattr(rag, "INDEX_PROCESS_LOCK", tmp_path / "nonexistent.lock")
    assert _peek_index_lock_holder() == ""


def test_peek_holder_returns_empty_when_empty_file(tmp_path, monkeypatch):
    """Archivo vacío (caso edge: holder murió antes de escribir) → empty string."""
    import rag
    lock_path = tmp_path / "empty.lock"
    lock_path.touch()
    monkeypatch.setattr(rag, "INDEX_PROCESS_LOCK", lock_path)
    assert _peek_index_lock_holder() == ""


class TestIndexEmbedClient:
    """`_index_embed_client()` cachea un `ollama.Client(timeout=120)` para
    los embeddings del index pipeline. El timeout existe para que un Ollama
    saturado no cuelgue el `rag index` en `recvfrom` indefinidamente —
    ver el docstring de la función para el incidente del 2026-05-01.
    """

    def test_returns_client_with_120s_timeout(self):
        """El client tiene timeout=120s consistente con `_index_chat_client`."""
        from rag import _index_embed_client
        client = _index_embed_client()
        assert client is not None
        # ollama.Client expone _client (httpx) con timeout
        timeout = client._client.timeout
        # httpx.Timeout puede ser un objeto compuesto. El read timeout es lo
        # que importa para `recvfrom` infinito.
        assert timeout.read == 120.0 or float(timeout.read) == 120.0

    def test_caches_singleton(self):
        """Llamadas sucesivas devuelven el mismo Client (no re-crea)."""
        from rag import _index_embed_client
        c1 = _index_embed_client()
        c2 = _index_embed_client()
        assert c1 is c2
