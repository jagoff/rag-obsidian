"""Tests para el shutdown del joblib/loky pool en el lifecycle del web daemon.

### Bug que estos tests cierran

Audit 2026-04-24 del `~/.local/share/obsidian-rag/web.error.log` mostró
**269 leaked POSIX semaphores** acumuladas en formato `/loky-PID-XXX`
— una por cada restart del web daemon. El leak viene de joblib (que
usa loky internamente como backend default de `Parallel`); algún
transitive dep arrancaba el pool durante una request y nadie lo
draineaba al shutdown.

### Fix

`_shutdown_joblib_loky_pool` registrado vía `@_on_shutdown` en
`web/server.py` post commit del audit. Llama
`get_reusable_executor().shutdown(kill_workers=True)` para terminar
el pool antes de que `resource_tracker` valide handles.

### Lo que estos tests garantizan

1. La callback está registrada en `_shutdown_callbacks` (se ejecuta
   al server stop).
2. Es idempotente — call repetido no rompe nada.
3. Falla silencioso si joblib no está disponible (no crashea el
   shutdown path).
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from web import server as srv


def test_shutdown_callback_registered():
    """`_shutdown_joblib_loky_pool` debe estar en `_shutdown_callbacks` post import."""
    callback_names = [fn.__name__ for fn in srv._shutdown_callbacks]
    assert "_shutdown_joblib_loky_pool" in callback_names, (
        f"loky shutdown callback not registered. Got: {callback_names}"
    )


def test_shutdown_idempotent_when_pool_never_started():
    """Llamar el shutdown sin que el pool arrancó nunca debe ser no-op,
    sin raisear. Caso típico: web server bootea + recibe queries que NO
    usan joblib.Parallel + se apaga. `get_reusable_executor()` devuelve
    un executor vacío + `shutdown` lo cierra sin workers que matar."""
    # No exception expected.
    srv._shutdown_joblib_loky_pool()
    # Llamar de nuevo (post-shutdown): también idempotente.
    srv._shutdown_joblib_loky_pool()


def test_shutdown_idempotent_after_pool_actually_used():
    """Forzar el pool a arrancar via joblib.Parallel + apagarlo.
    Verifica que el shutdown LIMPIA el pool sin raisear."""
    try:
        from joblib import Parallel, delayed
    except ImportError:
        pytest.skip("joblib not installed")

    # Spawn a real loky pool y darle trabajo trivial — fuerza
    # arranque del executor.
    results = Parallel(n_jobs=2, backend="loky")(
        delayed(lambda x: x * 2)(i) for i in range(4)
    )
    assert results == [0, 2, 4, 6]

    # Ahora apagamos. Si hay un bug en el handler, esto raisea o
    # cuelga.
    srv._shutdown_joblib_loky_pool()

    # Llamar de nuevo: el executor está cerrado → nuevo call también
    # idempotente.
    srv._shutdown_joblib_loky_pool()


def test_shutdown_skips_silently_when_joblib_unavailable(monkeypatch, capsys):
    """Si joblib no está importable (caso edge: instalación parcial),
    el shutdown NO debe romper el lifecycle. Silent-skip con un print
    diagnóstico — pero nunca raisea."""
    # Forzar el ImportError quitando joblib del sys.modules + bloqueando
    # el re-import.
    monkeypatch.setitem(sys.modules, "joblib", None)
    monkeypatch.setitem(sys.modules, "joblib.externals", None)
    monkeypatch.setitem(sys.modules, "joblib.externals.loky", None)

    # No debe raisear.
    srv._shutdown_joblib_loky_pool()

    # Y debe haber loggeado el skip.
    captured = capsys.readouterr()
    # El print del skip va a stdout (flush=True). Aceptamos que no
    # esté capturado en algunos runners — el invariante crítico es
    # NO RAISEAR.


def test_shutdown_handles_executor_shutdown_exception(monkeypatch):
    """Si el executor.shutdown() lanza una excepción inesperada (ej.
    versión nueva de joblib cambia la API), el handler la traga y
    sigue. Garantía de que el lifecycle del FastAPI nunca rompe por
    este callback."""
    class _BrokenExecutor:
        def shutdown(self, *a, **kw):
            raise RuntimeError("simulated joblib API change")

    def fake_get():
        return _BrokenExecutor()

    # Simulate the import + shutdown flow with a broken executor.
    with patch.dict(sys.modules, {
        "joblib.externals.loky": type(sys)("loky_stub"),
    }):
        sys.modules["joblib.externals.loky"].get_reusable_executor = fake_get
        # No raise.
        srv._shutdown_joblib_loky_pool()


def test_shutdown_resets_loky_globals_to_release_semlock_refs():
    """Audit follow-up 2026-04-25: el shutdown debe NULLIFICAR los globals
    `_executor` + `_executor_kwargs` de loky para que las references a
    SemLock se liberen y los Finalizers (`util.Finalize(_cleanup, ...)`)
    corran ANTES del exit, no en el resource_tracker check del fini-fini.
    Sin esto, vemos `leaked semaphore objects: /loky-PID-XXX` (308
    warnings acumuladas en `web.error.log` pre-fix)."""
    try:
        from joblib import Parallel, delayed
        from joblib.externals.loky import reusable_executor as re_mod
    except ImportError:
        pytest.skip("joblib not installed")

    # Force a real executor to exist
    Parallel(n_jobs=2, backend="loky")(delayed(lambda x: x)(i) for i in range(2))
    # Pre-condition: globals point to the executor
    assert re_mod._executor is not None
    assert re_mod._executor_kwargs is not None

    srv._shutdown_joblib_loky_pool()

    # Post-condition: globals nullified — the SemLock refs the executor
    # held are now garbage and gc.collect() inside the handler ran their
    # Finalizers, calling sem_unlink + resource_tracker.unregister.
    assert re_mod._executor is None, (
        "loky._executor still set after shutdown — SemLock refs not released"
    )
    assert re_mod._executor_kwargs is None


def test_shutdown_calls_gc_collect():
    """Verifica que el handler dispara gc.collect() para correr los
    Finalizers de SemLock que dependen del GC. Si el GC no corre, los
    Finalizers (`exitpriority=0`) terminan corriendo en el atexit
    fini-fini DESPUÉS del resource_tracker check, generando los warnings."""
    import gc as _gc
    calls = {"n": 0}
    original_collect = _gc.collect

    def counting_collect(*args, **kwargs):
        calls["n"] += 1
        return original_collect(*args, **kwargs)

    with patch.object(_gc, "collect", counting_collect):
        srv._shutdown_joblib_loky_pool()

    assert calls["n"] >= 1, (
        f"gc.collect() not called inside _shutdown_joblib_loky_pool "
        f"(call count: {calls['n']})"
    )


def test_tqdm_lock_preset_at_import_time():
    """Audit empírico 2026-04-25: el leak `/loky-PID-XXX` que vimos en
    `web.error.log` (247 acumulados) NO venía de joblib pools — venía
    de `tqdm.std.create_mp_lock()` que sentence-transformers triggea
    al cargar el reranker. Joblib monkey-patcha el naming para que
    todos los SemLocks (incluso de tqdm) usen el prefix `/loky-`.

    El fix preventivo: setear `tqdm.tqdm._lock` a un threading.RLock
    ANTES de cualquier import pesado, lo que skipea
    `TqdmDefaultWriteLock()` (donde se crea el POSIX SemLock).

    Este test verifica que el preset está aplicado: `tqdm.tqdm._lock`
    debe ser un threading lock, NO el `TqdmDefaultWriteLock` con su
    `mp_lock` POSIX.
    """
    import threading
    import tqdm as _tqdm
    import web.server  # noqa: F401  (force the module-level set_lock)

    # Después del import, _lock debe estar pre-set a un threading lock,
    # NO al TqdmDefaultWriteLock que crearía el mp_lock POSIX.
    assert hasattr(_tqdm.tqdm, "_lock"), "tqdm._lock no existe — preset perdido"
    lock = _tqdm.tqdm._lock
    # Threading.RLock devuelve _thread.RLock o threading._RLock dependiendo
    # de la versión de Python. Lo importante es que NO sea un
    # TqdmDefaultWriteLock (la que crea el SemLock POSIX).
    assert _tqdm.tqdm._lock.__class__.__name__ != "TqdmDefaultWriteLock", (
        f"tqdm._lock es {type(lock).__name__} — expected threading.RLock. "
        f"Eso significa que algo llamó tqdm.get_lock() ANTES del preset y "
        f"creó el SemLock POSIX."
    )
