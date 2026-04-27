"""Helpers compartidos de shutdown del proceso.

Centraliza la limpieza del leak de POSIX semaphores `/loky-PID-XXX` que
`resource_tracker` reporta en stderr al exit de los daemons de larga
duración (web server + watch + serve). Importado por:

- `rag/__init__.py` — registrado via `atexit.register()` para cualquier
  invocación del CLI (watch, serve, query, chat, etc.).
- `web/server.py` — registrado via `@_on_shutdown` dentro del lifespan
  de FastAPI.

### Origen del bug (resumen)

`tqdm` (transitively pulled por sentence-transformers / transformers /
sklearn) crea un `multiprocessing.RLock()` lazy en su primer
`tqdm.get_lock()`. Ese RLock es un POSIX named semaphore.

Adicionalmente, `joblib.externals.loky.backend.__init__` monkey-patchea
`multiprocessing.synchronize.SemLock._make_name` para que TODOS los
SemLocks creados después usen el prefix `/loky-PID-XXX`. Resultado: la
warning `leaked semaphore objects: {/loky-PID-XXX}` aparece en cada
clean shutdown (aunque tqdm no sea de joblib y viceversa — el monkey-
patch contamina el namespace global de SemLocks).

### Preset complementario

Este módulo solo cubre el *teardown*. El *preset* (pre-setear
`tqdm.tqdm._lock` a un `threading.RLock()` para evitar la creación del
SemLock en primer lugar) vive inline en el bootstrap de cada entry
point porque tiene que ejecutarse ANTES de que cualquier dep pesado
toque tqdm — no se puede factorizar sin romper ese orden de carga.

### Factorización 2026-04-26

Extracted de dos bloques duplicados (`web/server.py:7530-7599` y
`rag/__init__.py:96-136`) post commits `22a3b0b` (web) + `4d66199`
(rag). Mantener las dos copias diverge inevitablemente — ej. si mañana
joblib cambia la API del `get_reusable_executor`, hay que patchear
dos lados. Un solo file garantiza paridad.
"""
from __future__ import annotations


def shutdown_joblib_loky_pool() -> None:
    """Drain el joblib/loky reusable executor al exit del proceso.

    Origen: sentence-transformers / transformers / sklearn usan joblib
    internamente; algunos paths arrancan un `loky` pool (multiprocessing
    workers) durante el re-indexing. Si nadie drena ese pool al exit,
    `resource_tracker` ve el SemLock del pool huérfano y warnea en
    stderr (visible en `watch.error.log`, `web.error.log`, etc.).

    Fix en tres pasos:

    1. ``executor.shutdown(wait=True, kill_workers=True)`` — termina los
       workers + marca el executor como shut-down.
    2. Nullificar los globals ``_executor`` + ``_executor_kwargs`` del
       módulo ``reusable_executor`` para liberar las refs a SemLock.
       Sin esto, los `util.Finalize(SemLock._cleanup, ...)` con
       `exitpriority=0` no corren antes del resource_tracker check del
       exit fini-fini.
    3. ``gc.collect()`` — fuerza los Finalizers de SemLock a correr
       AHORA, no en el atexit del intérprete.

    Idempotente: si el pool nunca arrancó (caso típico — la mayoría de
    procesos no usan `joblib.Parallel`), `get_reusable_executor()`
    devuelve un executor vacío y `shutdown` es no-op.

    Silent-fail: si joblib no está instalado o cambia su API, el handler
    traga la excepción y sigue. El leak vuelve pero no rompemos el
    shutdown path.

    Trade-off: si una request EN VUELO está usando `joblib.Parallel`,
    ``kill_workers=True`` la corta abruptamente. Es lo correcto durante
    SIGTERM — los user-facing handlers ya tienen sus propios timeouts
    y graceful aborts.
    """
    try:
        from joblib.externals.loky import get_reusable_executor
        from joblib.externals.loky import reusable_executor as _re_mod
        import gc as _gc

        executor = get_reusable_executor()
        executor.shutdown(wait=True, kill_workers=True)

        try:
            _re_mod._executor = None
            _re_mod._executor_kwargs = None
        except Exception:
            pass
        try:
            _gc.collect()
        except Exception:
            pass
    except Exception:
        # Silent-skip: el leak no es bloqueante, no queremos romper el
        # shutdown si joblib se desinstala o cambia su API.
        pass
