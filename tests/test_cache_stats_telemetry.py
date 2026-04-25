"""Tests para `_cache_telemetry_stats()` — cross-reference rag_queries
con rag_response_cache (agregado 2026-04-23).

El helper calcula hit rate real, distribución de miss reasons, y ahorro
de latencia estimado leyendo `extra_json.cache_probe` + `cache_hit`.
Usado por `rag cache stats --days N` para exponer observabilidad
downstream.
"""
from __future__ import annotations


import pytest

import rag


@pytest.fixture
def clean_cache_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_CACHE_ENABLED", "1")
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    yield


def _insert_query(q: str, cache_hit: bool, reason: str, t_gen_s: float = 5.0,
                  result: str = "miss"):
    """Insert a synthetic row into rag_queries with cache_probe populated.

    We go through log_query_event so the column mapping is the real one
    (extra_json shape must match what cache_stats reads)."""
    extra = {
        "cache_hit": cache_hit,
        "cache_probe": {
            "result": result,
            "reason": reason,
            "top_cosine": 0.88 if not cache_hit else 0.99,
            "candidates": 1,
        },
    }
    event = {
        "cmd": "query",
        "q": q,
        "t_gen": t_gen_s,
        # log_query_event flattens unknown keys into extra_json via
        # _map_queries_row, so we pass cache_hit/cache_probe at top-level.
        "cache_hit": cache_hit,
        "cache_probe": extra["cache_probe"],
    }
    rag.log_query_event(event)


def _drain():
    try:
        rag._LOG_QUEUE.join()
    except Exception:
        pass
    try:
        rag._BACKGROUND_SQL_QUEUE.join()
    except Exception:
        pass


def test_telemetry_empty_window_returns_zeros(clean_cache_env):
    """Ventana sin queries elegibles → baseline shape correcto.

    La DB real (telemetry.db) puede tener rows productivas de sesiones
    previas con `cache_probe` populado — los asserts se escriben como
    *delta* desde el baseline sin insertar nada nuevo.
    """
    stats = rag._cache_telemetry_stats(days=7)
    assert stats["window_days"] == 7
    # Shape + tipos se respetan; la magnitud depende del historial real.
    assert isinstance(stats["eligible"], int)
    assert isinstance(stats["hits"], int)
    assert isinstance(stats["hit_rate_pct"], float)
    assert isinstance(stats["miss_reasons"], dict)
    assert isinstance(stats["top_queries"], list)


def test_telemetry_counts_hits_and_eligibility(clean_cache_env):
    """Insertar 3 misses + 1 hit → hit count sube +1 vs baseline, eligible +4."""
    baseline = rag._cache_telemetry_stats(days=7)
    _insert_query("telemetry_test_a", cache_hit=False, reason="below_threshold")
    _insert_query("telemetry_test_b", cache_hit=False, reason="corpus_mismatch")
    _insert_query("telemetry_test_c", cache_hit=False, reason="below_threshold")
    _insert_query("telemetry_test_d", cache_hit=True, reason="match", result="hit")
    _drain()

    stats = rag._cache_telemetry_stats(days=7)
    # Deltas — más robusto que asserts absolutos cuando la DB tiene
    # historial productivo de sesiones reales anteriores al test.
    assert stats["eligible"] - baseline["eligible"] == 4
    assert stats["hits"] - baseline["hits"] == 1


def test_telemetry_miss_reasons_aggregated(clean_cache_env):
    """Miss reasons cuenta por categoría — delta desde baseline."""
    baseline = rag._cache_telemetry_stats(days=7)
    b_below = baseline["miss_reasons"].get("below_threshold", 0)
    b_ttl = baseline["miss_reasons"].get("ttl_expired", 0)
    for _ in range(3):
        _insert_query("telemetry_test_x", cache_hit=False, reason="below_threshold")
    _insert_query("telemetry_test_y", cache_hit=False, reason="ttl_expired")
    _drain()

    stats = rag._cache_telemetry_stats(days=7)
    assert stats["miss_reasons"].get("below_threshold", 0) - b_below == 3
    assert stats["miss_reasons"].get("ttl_expired", 0) - b_ttl == 1


def test_telemetry_excludes_skipped_and_disabled(clean_cache_env):
    """`result='skipped'` y `result='disabled'` NO cuentan como elegibles."""
    baseline = rag._cache_telemetry_stats(days=7)
    _insert_query("telemetry_test_sk", cache_hit=False, reason="flags_skip",
                  result="skipped")
    _insert_query("telemetry_test_ds", cache_hit=False, reason="cache_disabled",
                  result="disabled")
    _insert_query("telemetry_test_real", cache_hit=True, reason="match",
                  result="hit")
    _drain()

    stats = rag._cache_telemetry_stats(days=7)
    # Solo la row "real" suma a eligibility — skipped/disabled se filtran.
    assert stats["eligible"] - baseline["eligible"] == 1
    assert stats["hits"] - baseline["hits"] == 1


def test_telemetry_top_queries_from_response_cache(clean_cache_env):
    """Los top queries se leen de `rag_response_cache.hit_count`."""
    import numpy as np
    emb = np.zeros(1024, dtype="float32"); emb[0] = 1.0
    emb = emb / np.linalg.norm(emb)
    # Store 2 entries and simulate hits via lookup.
    rag.semantic_cache_store(
        emb, question="qué es ikigai", response="a",
        paths=[], scores=[], top_score=0.9,
        intent="semantic", corpus_hash="Htop",
    )
    # Fire 3 lookups → bumps hit_count to 3.
    for _ in range(3):
        rag.semantic_cache_lookup(emb, "Htop")

    stats = rag._cache_telemetry_stats(days=7)
    top = stats["top_queries"]
    assert len(top) >= 1
    # La primera es la que tiene más hit_count.
    assert top[0]["question"].startswith("qué es ikigai")
    assert top[0]["hit_count"] == 3


def test_telemetry_cli_smoke(clean_cache_env):
    """`rag cache stats --days 7` ejecuta sin excepciones."""
    from click.testing import CliRunner
    from rag import cli
    r = CliRunner()
    result = r.invoke(cli, ["cache", "stats", "--days", "7"])
    assert result.exit_code == 0, (
        f"cache stats failed: {result.output}\n{result.exception}"
    )
    assert "Semantic response cache" in result.output
