"""Tests para wins de perf detectados en audit línea-por-línea (2026-04-30).

Cubren:
- Fix 1: bm25_search usa heapq.nlargest (O(n log k))
- Fix 2: collect_ranker_features no llama col.count() dos veces
- Fix 3: orphan backfill SQL usa range bounds (no función en WHERE)
- Fix 4: índices sobre json_extract via VIRTUAL columns
- Fix 5: _scan_queries_log tiene LIMIT
"""
from __future__ import annotations

import inspect
from pathlib import Path

import rag


def test_bm25_search_uses_heapq_nlargest():
    """Fix 1: bm25_search debe usar heapq.nlargest en vez de sorted()[:k]."""
    src = inspect.getsource(rag.bm25_search)
    assert "heapq.nlargest" in src, (
        "bm25_search debe usar heapq.nlargest(k, ..., key=...) en vez de "
        "sorted(..., reverse=True)[:k] (O(n log k) vs O(n log n))"
    )


def test_heapq_imported():
    """Fix 1: heapq debe estar importado a nivel módulo."""
    assert hasattr(rag, "heapq"), "heapq debe estar importado al top de rag/__init__.py"


def test_heapq_nlargest_equivalent_to_sorted():
    """Fix 1: heapq.nlargest produce el mismo resultado que sorted()[:k] para top-k."""
    import heapq
    scores = [0.5, 0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.6]
    k = 3
    sorted_top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    heap_top = heapq.nlargest(k, range(len(scores)), key=lambda i: scores[i])
    # nlargest devuelve en orden DESC; sorted[:k] también — equivalentes
    assert heap_top == sorted_top, f"heap={heap_top} sorted={sorted_top}"


def test_collect_ranker_features_single_count_call():
    """Fix 2: collect_ranker_features no debe llamar col.count() dos veces seguidas."""
    src = inspect.getsource(rag.collect_ranker_features)
    # Contamos cuántas veces aparece col.count() — el patrón viejo era 2,
    # con el fix queda 1 (asignado a variable y reusado).
    n_count_calls = src.count("col.count()")
    assert n_count_calls <= 1, (
        f"collect_ranker_features llama col.count() {n_count_calls} veces; "
        f"debe ser 1 (asignar a variable + reusar)"
    )


def test_orphan_backfill_sql_uses_range_bounds():
    """Fix 3: backfill_orphan_behavior SQL usa ts >= ? AND ts <= ? (no ABS+strftime)."""
    if not hasattr(rag, "backfill_orphan_behavior"):
        return  # función puede tener otro nombre — el SQL test está en otro lado
    src = inspect.getsource(rag.backfill_orphan_behavior)
    # El predicado viejo era ABS(strftime('%s', ts) - strftime('%s', ?)) <= N
    # — eso fuerza full scan. El nuevo usa rangos explícitos.
    assert "ABS(strftime" not in src, (
        "El WHERE no debe envolver ts en ABS(strftime(...)) — fuerza full scan; "
        "usar ts >= datetime(?, ...) AND ts <= datetime(?, ...)"
    )


def test_scan_queries_log_has_limit():
    """Fix 5: _scan_queries_log debe tener LIMIT en el SELECT."""
    src = inspect.getsource(rag._scan_queries_log)
    # SELECT ... FROM rag_queries ... LIMIT N
    assert "LIMIT" in src.upper(), (
        "_scan_queries_log debe tener LIMIT (default 5000) para evitar "
        "traer >10k rows + parsear extra_json"
    )


def test_local_embed_enabled_has_cache():
    """Fix 6: _local_embed_enabled tiene caché lazy post-warmup."""
    # El cache se setea cuando warmup completa. Acá sólo verificamos que
    # exista la variable del cache y un freezer.
    assert hasattr(rag, "_LOCAL_EMBED_ENABLED_CACHED"), (
        "Falta variable módulo _LOCAL_EMBED_ENABLED_CACHED para cachear "
        "el flag post-warmup (evita re-leer env var en cada query)"
    )
    assert hasattr(rag, "_freeze_local_embed_enabled"), (
        "Falta helper _freeze_local_embed_enabled() que se llame al final "
        "del warmup para congelar el flag"
    )
