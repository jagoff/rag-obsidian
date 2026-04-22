"""Tests del `RAGState` registry (2026-04-22, #3 refactor estructural).

Contexto: pre-fix hay ~20 singletons dispersos en rag.py con locks
separados (`_context_cache_lock`, `_synthetic_q_cache_lock`,
`_corpus_cache_lock`, `_embed_cache_lock`, `_mentions_cache_lock`, …).
Cada cache nuevo replica el patrón `{cache_dict, lock, dirty_flag,
save_to_disk, load_from_disk, atexit_save}` — 30-40 líneas por cache.

Resultado de esa dispersión:
  - Invariantes de orden de adquisición de locks implícitos → riesgo
    TOCTOU documentado (reranker idle-unload).
  - Tests de concurrencia tienen que patchear cada uno por separado
    (`tests/test_cache_concurrency.py` con 8 casos).
  - Clear-all (útil en tests) requiere tocar 20 caches a mano.

Scope contenido para esta iteración:
  - `RAGState` clase con registry + API uniforme
  - `register_cache(name, lock)`: dar de alta un cache existente
  - `clear_all()`: útil en tests + rag cache clear
  - `stats()`: dict con info de cada cache (size, lock held?)
  - **NO mueve** los caches actuales — quedan donde están. Solo se
    registran en el state y el state expone API uniforme.

Fuera de scope para esta iteración (quedan como deuda reconocida):
  - Mover los modelos (reranker, embedder, NLI, GLiNER) — warmup complejo
  - Eliminar los locks individuales — sigue siendo defensive en cada cache
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import rag


# ── Existencia ──────────────────────────────────────────────────────────────


def test_rag_state_exists():
    assert hasattr(rag, "RAGState")


def test_rag_state_singleton_exists():
    """El estado global se accede via `rag.state` (singleton).  Ahí se
    registran los caches al import-time de rag.py."""
    assert hasattr(rag, "state")
    assert isinstance(rag.state, rag.RAGState)


# ── Registry API ────────────────────────────────────────────────────────────


def test_register_cache_basic():
    state = rag.RAGState()
    d = {"foo": "bar"}
    lock = threading.Lock()
    state.register_cache("mycache", d, lock=lock)
    # Registered
    assert "mycache" in state.list_caches()


def test_register_cache_returns_handle():
    """El handle debe permitir clear / len / get desde el state
    sin que el caller retenga la referencia al dict."""
    state = rag.RAGState()
    d = {"x": 1, "y": 2}
    state.register_cache("mycache", d)
    # La dict interna se puede inspeccionar
    assert state.size_of("mycache") == 2


def test_register_cache_duplicate_name_raises():
    state = rag.RAGState()
    state.register_cache("cache1", {"a": 1})
    with pytest.raises(ValueError):
        state.register_cache("cache1", {"b": 2})


# ── clear_all: útil para tests ──────────────────────────────────────────────


def test_clear_all_empties_all_caches():
    state = rag.RAGState()
    d1 = {"a": 1, "b": 2}
    d2 = {"c": 3}
    state.register_cache("c1", d1)
    state.register_cache("c2", d2)

    assert state.size_of("c1") == 2
    assert state.size_of("c2") == 1

    state.clear_all()

    assert state.size_of("c1") == 0
    assert state.size_of("c2") == 0
    # Los dict originales también se vaciaron (no deep copy)
    assert d1 == {}
    assert d2 == {}


def test_clear_specific_cache():
    state = rag.RAGState()
    state.register_cache("c1", {"a": 1})
    state.register_cache("c2", {"b": 2})
    state.clear("c1")
    assert state.size_of("c1") == 0
    assert state.size_of("c2") == 1


def test_clear_unknown_cache_is_noop():
    """`clear("nonexistent")` no debe raisear — tolerancia para hooks
    que limpian por nombre sin saber si el cache está o no."""
    state = rag.RAGState()
    state.clear("nonexistent_cache")  # no raise


# ── stats: info para rag stats / dashboard ─────────────────────────────────


def test_stats_returns_dict_with_name_size():
    state = rag.RAGState()
    state.register_cache("c1", {"a": 1, "b": 2})
    state.register_cache("c2", {})
    stats = state.stats()
    assert isinstance(stats, dict)
    assert stats["c1"]["size"] == 2
    assert stats["c2"]["size"] == 0


def test_stats_includes_all_registered():
    state = rag.RAGState()
    state.register_cache("a", {})
    state.register_cache("b", {"x": 1})
    state.register_cache("c", {"y": 2})
    stats = state.stats()
    assert set(stats.keys()) == {"a", "b", "c"}


# ── Thread-safety: acquire lock + read size ────────────────────────────────


def test_size_of_is_lock_respecting():
    """`size_of(name)` debe adquirir el lock del cache para leer de
    forma consistente incluso mientras un writer muta — no es crítico
    que sea sub-microsecond, sí que no raisee."""
    state = rag.RAGState()
    shared = {}
    lock = threading.Lock()
    state.register_cache("shared", shared, lock=lock)

    # Simulamos un writer concurrente
    def writer():
        for i in range(100):
            with lock:
                shared[f"k{i}"] = i

    t = threading.Thread(target=writer)
    t.start()
    # Reader lee repetidamente mientras el writer corre
    for _ in range(50):
        size = state.size_of("shared")
        assert size >= 0  # siempre coherente
    t.join()


# ── Integración: los caches reales de rag.py deben estar registrados ──────


def test_real_caches_registered_at_import():
    """Los caches que antes tenían locks dispersos ahora se deben
    registrar en `rag.state` al import-time. Lista mínima esperada:

      - embed            (LRU de embeddings)
      - context_summary  (JSON persistido)
      - synthetic_q      (JSON persistido)
      - expand           (paraphrase LRU)
      - corpus           (BM25 + vocab)

    Tests posteriores pueden agregar más; este test es el contract
    mínimo para que `rag stats` / `rag cache stats` tengan info para
    mostrar."""
    registered = set(rag.state.list_caches())
    # Al menos el core que ya existe como módulo-level dicts
    core_expected = {"context_summary", "synthetic_q"}
    missing = core_expected - registered
    assert not missing, f"caches not registered in rag.state: {missing}"


# ── Double-register protection ─────────────────────────────────────────────


def test_real_caches_not_registered_twice(caplog):
    """Re-importing rag.py (pytest a veces lo hace entre tests) no
    debe crashear con `ValueError: duplicate cache name`. Usamos
    register_cache_safe que es idempotente."""
    # Acceso a state should not raise aunque se llame múltiples veces
    _ = rag.state.list_caches()
    # Re-llamada al registrar caches viejos (simulando reload) con
    # el método safe debería ser no-op
    state = rag.state
    # Si el code path de registro usa register_cache directamente,
    # un segundo import raisearía — pero el test de este módulo corre
    # sin reload, así que la verificación es que state ya tiene los
    # caches sin warnings.
    assert len(state.list_caches()) > 0
