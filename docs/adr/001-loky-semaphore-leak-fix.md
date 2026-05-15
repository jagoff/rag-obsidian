# ADR-001: Loky Semaphore Leak Fix

## Contexto

Los daemons de larga duración (`rag watch`, `rag serve`) acumulaban warnings de `resource_tracker` en cada shutdown limpio:

```
leaked semaphore objects: {/loky-PID-XXX}
```

**Root cause**: `tqdm` (transitivamente importado por `sentence-transformers` / `transformers`) crea un `multiprocessing.RLock()` lazy en su primer `tqdm.get_lock()`. Ese RLock es un **POSIX named semaphore** que Python NO libera automáticamente al `exit()` del proceso.

Adicionalmente, `joblib.externals.loky.backend.__init__` monkey-patchea `multiprocessing.synchronize.SemLock._make_name` para que **todos** los SemLocks creados después usen el prefix `/loky-PID-XXX`.

## Evidencia

- Pre-fix: 9 warnings acumuladas en `watch.error.log` (CLI daemon).
- Pre-fix: 247 leaks acumulados en `web.error.log` (web daemon).
- Trace empírico (2026-04-25) con monkey-patch de `SemLock.__init__` confirma que el SemLock se crea en `tqdm/std.py:121 create_mp_lock` durante el primer `cls.get_lock()` que dispara `sentence-transformers` al cargar el reranker.

## Fix

Pre-setear el lock interno de `tqdm` a un `threading.RLock` (in-process, no semaphore POSIX) **antes** de que cualquier dependencia pesada toque `tqdm`:

```python
try:
    import tqdm
    tqdm.tqdm.set_lock(threading.RLock())
except Exception:
    pass
```

`TQDM_DISABLE=1` (ya seteado en top-level) desactiva las progress bars, pero **no** previene la creación del SemLock vía `tqdm.get_lock()` que `sentence-transformers` llama durante la carga del reranker/embedder.

### Side effect

Las progress bars de `tqdm` en este proceso ya no son inter-process safe. El web daemon es single-process (`uvicorn` con `workers=1`), así que no hay pérdida real. Si alguna vez se introduce multi-process workers, este lock necesita re-evaluación.

## Aplicación

- `rag/__init__.py` (espejo para todas las invocaciones CLI, 2026-04-26).
- `web/server.py` (web daemon, 2026-04-25).
- El teardown (drain del pool `joblib/loky`) vive en `rag/_shutdown.py`.

## Estado

**Accepted** — aplicado en producción desde 2026-04-26.
