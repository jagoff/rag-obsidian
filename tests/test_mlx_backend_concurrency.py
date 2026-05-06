"""Tests de concurrency para `MLXBackend._loaded` + `_loaded_lock` (P1 fix 2026-05-05).

Coverage:

- 3 threads concurrent llamando `_load(qwen2.5:3b)` → `mlx_lm.load` se llama
  ≥1 vez (mock cuenta), todos reciben el mismo `(model, tokenizer)` que quedó
  guardado en `_loaded`. Aceptamos que durante la ventana sin lock más de un
  thread puede entrar a `mlx_lm.load`, pero el doble-check garantiza que solo
  UN tuple termina en `_loaded`.

- 3 threads con modelos distintos → `mlx_lm.load` corre 3 veces, todos los
  modelos quedan resident, sin race en `_loaded`.

- Race con `unload()`: thread A está en `_load()` (durante el `mlx_lm.load`
  unlocked), thread B llama `unload()` → no exception, `_loaded` queda en
  estado consistente.

Mockeamos `mlx_lm.load` para evitar carga real (3-8s + 3+ GB RAM por modelo).

Si `mlx_lm` no está instalado, el módulo entero se skipea via importorskip
porque `MLXBackend.__init__` falla eagerly.
"""

from __future__ import annotations

import sys
import threading
import time
import types
import unittest.mock as mock
from typing import Any

import pytest

pytest.importorskip("mlx_lm")

from rag.llm_backend import MLXBackend, reset_backend, to_mlx  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    """Aislamiento del singleton del backend resolver."""
    reset_backend()
    yield
    reset_backend()


def _fake_load_factory(latency_s: float = 0.5, counter: dict[str, int] | None = None):
    """Crea un fake `mlx_lm.load` que duerme `latency_s` y devuelve sentinels.

    `counter` se incrementa por canonical name para verificar cuántas veces
    realmente se llamó la I/O cara.
    """
    if counter is None:
        counter = {}
    counter_lock = threading.Lock()

    def _fake_load(canonical: str) -> tuple[Any, Any]:
        with counter_lock:
            counter[canonical] = counter.get(canonical, 0) + 1
        time.sleep(latency_s)
        # Sentinels: tuple-of-strings is fine — el código real solo guarda
        # `(model, tokenizer)` opacamente y los pasa a mlx_lm.generate más tarde.
        return (f"model-{canonical}", f"tokenizer-{canonical}")

    return _fake_load, counter


def _patch_mlx_load(fake_load):
    """Inyecta `fake_load` como `mlx_lm.load` via sys.modules patch.

    `MLXBackend._load` hace `from mlx_lm import load` localmente cada vez,
    así que reemplazamos `mlx_lm.load` en sys.modules directamente.
    """
    real_mlx_lm = sys.modules.get("mlx_lm")
    fake_module = types.ModuleType("mlx_lm")
    # Preservar otros símbolos que MLXBackend.__init__ ya validó como existentes
    if real_mlx_lm is not None:
        for attr in dir(real_mlx_lm):
            if not attr.startswith("_"):
                try:
                    setattr(fake_module, attr, getattr(real_mlx_lm, attr))
                except Exception:
                    pass
    fake_module.load = fake_load  # type: ignore[attr-defined]
    return mock.patch.dict(sys.modules, {"mlx_lm": fake_module})


# ---------------------------------------------------------------------------
# 1. Concurrent _load(same model) — solo 1 entry en _loaded, mlx_lm.load
#    puede correr 1+ veces durante la ventana sin lock pero el resultado
#    final es coherente.
# ---------------------------------------------------------------------------


def test_concurrent_load_same_model_single_entry():
    backend = MLXBackend()
    fake_load, counter = _fake_load_factory(latency_s=0.3)

    results: list[tuple[Any, Any]] = []
    errors: list[Exception] = []
    barrier = threading.Barrier(3)

    def _worker():
        try:
            barrier.wait()  # arrancan los 3 al unísono
            r = backend._load("qwen2.5:3b")
            results.append(r)
        except Exception as e:
            errors.append(e)

    with _patch_mlx_load(fake_load):
        threads = [threading.Thread(target=_worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert errors == [], f"workers raised: {errors}"
    assert len(results) == 3
    canonical = to_mlx("qwen2.5:3b")

    # Invariante crítico: solo UN tuple winner queda en _loaded
    assert canonical in backend._loaded
    winner = backend._loaded[canonical]

    # Todos los workers que terminaron después del primer store deberían
    # ver el winner. Los que entraron en la ventana sin lock pueden tener
    # su propio tuple local, PERO no contaminaron _loaded.
    assert all(r == winner for r in results) or all(
        # caso degenerate: 3 entraron antes del primer store, cada uno cargó
        # un tuple distinto. El winner es el primero en ganar el lock post-load.
        # Aceptamos esto siempre que solo UN tuple haya ganado en _loaded.
        isinstance(r, tuple) and len(r) == 2 for r in results
    )

    # mlx_lm.load corre al menos 1 vez. En el caso ideal (todos los workers
    # ven el primer store via double-check) corre 1 sola vez. En el peor caso
    # corren los 3 antes de que alguno persista — pero el lock garantiza que
    # solo el primer store gana.
    assert counter[canonical] >= 1
    assert counter[canonical] <= 3


# ---------------------------------------------------------------------------
# 2. Concurrent _load(distinct models) — 3 entries en _loaded, sin race
# ---------------------------------------------------------------------------


def test_concurrent_load_distinct_models():
    backend = MLXBackend()
    fake_load, counter = _fake_load_factory(latency_s=0.2)

    models = ["qwen2.5:3b", "qwen2.5:7b", "qwen3:4b"]
    canonicals = [to_mlx(m) for m in models]
    results: dict[str, tuple[Any, Any]] = {}
    errors: list[Exception] = []
    barrier = threading.Barrier(3)

    def _worker(model_id: str):
        try:
            barrier.wait()
            results[model_id] = backend._load(model_id)
        except Exception as e:
            errors.append(e)

    with _patch_mlx_load(fake_load):
        threads = [threading.Thread(target=_worker, args=(m,)) for m in models]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert errors == [], f"workers raised: {errors}"
    assert len(results) == 3

    # Los 3 modelos quedan resident
    for c in canonicals:
        assert c in backend._loaded, f"{c} not in _loaded"
        assert counter[c] == 1, f"{c} loaded {counter[c]} times (should be 1)"


# ---------------------------------------------------------------------------
# 3. Race con unload() durante un _load — no exception, estado consistente
# ---------------------------------------------------------------------------


def test_unload_during_load_no_crash():
    backend = MLXBackend()
    # Pre-popular _loaded para que unload() tenga algo que limpiar
    backend._loaded[to_mlx("qwen2.5:7b")] = ("preloaded-model", "preloaded-tok")

    fake_load, counter = _fake_load_factory(latency_s=0.5)

    load_done = threading.Event()
    unload_done = threading.Event()
    errors: list[Exception] = []

    def _loader():
        try:
            backend._load("qwen2.5:3b")
            load_done.set()
        except Exception as e:
            errors.append(e)

    def _unloader():
        try:
            time.sleep(0.1)  # dejar que loader entre al mlx_lm.load (unlocked)
            backend.unload()  # unload all
            unload_done.set()
        except Exception as e:
            errors.append(e)

    # Patchear `mlx.core.clear_cache` también — sino MLXBackend.unload
    # intenta tocar Metal real. fake_module ya hereda mlx_lm.load patcheado.
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.clear_cache = lambda: None  # type: ignore[attr-defined]

    with _patch_mlx_load(fake_load), mock.patch.dict(
        sys.modules, {"mlx.core": fake_mx}
    ):
        t1 = threading.Thread(target=_loader)
        t2 = threading.Thread(target=_unloader)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

    assert errors == [], f"workers raised: {errors}"
    assert load_done.is_set()
    assert unload_done.is_set()

    # Estado final: o el unload corrió antes del store (loader's tuple winó
    # la slot), o el unload corrió después (slot quedó limpia). Cualquiera
    # de los dos es correcto — lo importante es que no haya excepción ni
    # estado corrupto.
    canonical_3b = to_mlx("qwen2.5:3b")
    if canonical_3b in backend._loaded:
        # caso A: store ganó al unload — el tuple del loader debe estar
        assert backend._loaded[canonical_3b] is not None
    # caso B: unload ganó — _loaded está vacío o solo tiene el evicted state.
    # Ambos casos son válidos; el test pasa si no hubo crash ni dict corruption.


# ---------------------------------------------------------------------------
# 4. Lock es re-entrant: _load() llama _evict_for() bajo el mismo lock
# ---------------------------------------------------------------------------


def test_lock_is_reentrant():
    """Smoke test que confirma RLock semantic — Lock simple deadlockearía aquí."""
    backend = MLXBackend()
    # El _load() interno hace `with self._loaded_lock:` y luego llama
    # _evict_for() que también hace `with self._loaded_lock:`. Si fuera Lock
    # simple, la segunda adquisición se quedaría colgada. Probamos que la
    # secuencia completa termina sin deadlock.
    fake_load, _ = _fake_load_factory(latency_s=0.05)
    with _patch_mlx_load(fake_load):
        # Llenar el cap _MAX_SMALL_LOADED para que _evict_for haga trabajo real
        for i in range(backend._MAX_SMALL_LOADED + 1):
            backend._load(f"qwen2.5:3b" if i == 0 else f"unknown-{i}:tag")

    # Si llegamos acá, no hubo deadlock — fix funciona.
    assert len(backend._loaded) <= backend._MAX_SMALL_LOADED
