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
    # Contamos sólo apariciones en líneas de código (no en comentarios).
    n_count_calls = sum(
        1
        for line in src.splitlines()
        if "col.count()" in line and not line.lstrip().startswith("#")
    )
    assert n_count_calls <= 1, (
        f"collect_ranker_features llama col.count() {n_count_calls} veces "
        f"en código (no comments); debe ser 1 (asignar a variable + reusar)"
    )


def test_behavior_backfill_match_uses_range_bounds():
    """Fix 3: _behavior_backfill_find_match SQL usa ts >= datetime(?, ...) range bounds.

    Antes el WHERE envolvía ts en ABS(strftime(...)) — fuerza full scan
    de ix_rag_queries_ts (SCAN COVERING INDEX). Con range bounds explícitos
    el plan pasa a SEARCH USING COVERING INDEX (ts>? AND ts<?).
    """
    src = inspect.getsource(rag._behavior_backfill_find_match)
    # El WHERE no debe usar ABS(strftime(...)) — eso fuerza full scan.
    # Permitimos ABS(strftime(...)) en el ORDER BY (afecta sort de resultset
    # chico, no el scan). Para distinguir: contamos solo apariciones en
    # líneas que tengan WHERE / AND.
    bad_where_lines = [
        line for line in src.splitlines()
        if ("WHERE" in line or " AND " in line)
        and "ABS(strftime" in line
        and "ORDER BY" not in line
    ]
    assert not bad_where_lines, (
        f"El WHERE/AND no debe envolver ts en ABS(strftime(...)): "
        f"{bad_where_lines}"
    )
    # Y debe haber range bounds explícitos (datetime con +N/-N seconds).
    assert "datetime(?" in src or "datetime(?, '" in src, (
        "Falta range bound tipo datetime(?, '-N seconds') / datetime(?, '+N seconds')"
    )


def test_behavior_backfill_query_plan_uses_index():
    """Fix 3: EXPLAIN QUERY PLAN del SQL nuevo confirma SEARCH (range bounds)."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE rag_queries (id INTEGER PRIMARY KEY, ts TEXT, "
        "q TEXT, session TEXT, paths_json TEXT)"
    )
    conn.execute("CREATE INDEX ix_rag_queries_ts ON rag_queries(ts)")
    conn.execute(
        "CREATE INDEX ix_rag_queries_session_ts "
        "ON rag_queries(session, ts)"
    )
    plan = list(conn.execute(
        "EXPLAIN QUERY PLAN SELECT id FROM rag_queries "
        "WHERE ts >= datetime(?, '-600 seconds') "
        "AND ts <= datetime(?, '+600 seconds')",
        ("2026-04-30T10:00", "2026-04-30T10:00"),
    ))
    plan_str = " ".join(str(r) for r in plan)
    assert "SEARCH" in plan_str and "ts>" in plan_str.lower() or "ts >" in plan_str, (
        f"Plan esperado SEARCH con range bounds, fue: {plan_str}"
    )
    # Same-session debe usar ix_rag_queries_session_ts.
    plan_session = list(conn.execute(
        "EXPLAIN QUERY PLAN SELECT id FROM rag_queries "
        "WHERE session = ? "
        "AND ts >= datetime(?, '-600 seconds') "
        "AND ts <= datetime(?, '+600 seconds')",
        ("s1", "2026-04-30T10:00", "2026-04-30T10:00"),
    ))
    plan_session_str = " ".join(str(r) for r in plan_session)
    assert "ix_rag_queries_session_ts" in plan_session_str, (
        f"Same-session debería usar ix_rag_queries_session_ts: {plan_session_str}"
    )


def test_scan_queries_log_has_limit():
    """Fix 5: _scan_queries_log debe tener LIMIT en el SELECT."""
    src = inspect.getsource(rag._scan_queries_log)
    # SELECT ... FROM rag_queries ... LIMIT N
    assert "LIMIT" in src.upper(), (
        "_scan_queries_log debe tener LIMIT (default 5000) para evitar "
        "traer >10k rows + parsear extra_json"
    )


def test_turn_id_expression_index_in_ddl():
    """Fix 4: DDL de rag_queries incluye índice sobre json_extract(extra_json, '$.turn_id')."""
    ddl = "\n".join(
        s for table_name, stmts in rag._TELEMETRY_DDL
        for s in stmts
    )
    assert "ix_rag_queries_turn_id" in ddl, (
        "Falta índice ix_rag_queries_turn_id sobre json_extract(extra_json, '$.turn_id')"
    )
    assert "json_extract(extra_json, '$.turn_id')" in ddl, (
        "El índice debe ser exactamente sobre la expresión que la query usa"
    )


def test_corrective_path_expression_index_in_ddl():
    """Fix 4: DDL de rag_feedback incluye índice sobre json_extract corrective_path."""
    ddl = "\n".join(
        s for table_name, stmts in rag._TELEMETRY_DDL
        for s in stmts
    )
    assert "ix_rag_feedback_corrective" in ddl, (
        "Falta índice ix_rag_feedback_corrective sobre json_extract(extra_json, '$.corrective_path')"
    )


def test_expression_index_actually_used_by_planner():
    """Fix 4: SQLite usa el expression index en EXPLAIN QUERY PLAN."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE rag_queries (id INTEGER PRIMARY KEY, ts TEXT, "
        "q TEXT, session TEXT, extra_json TEXT)"
    )
    conn.execute(
        "CREATE INDEX ix_rag_queries_turn_id "
        "ON rag_queries(json_extract(extra_json, '$.turn_id'))"
    )
    plan = list(conn.execute(
        "EXPLAIN QUERY PLAN SELECT q FROM rag_queries "
        "WHERE json_extract(extra_json, '$.turn_id') = ?",
        ("abc",),
    ))
    plan_str = " ".join(str(r) for r in plan)
    assert "ix_rag_queries_turn_id" in plan_str, (
        f"Plan debería usar ix_rag_queries_turn_id, fue: {plan_str}"
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


def test_freeze_local_embed_enabled_caches_value(monkeypatch):
    """Fix 6: _freeze_local_embed_enabled persiste (env_raw, decision)."""
    # Reset
    rag._LOCAL_EMBED_ENABLED_CACHED = None
    monkeypatch.setenv("RAG_LOCAL_EMBED", "1")
    rag._freeze_local_embed_enabled()
    cached = rag._LOCAL_EMBED_ENABLED_CACHED
    assert isinstance(cached, tuple) and len(cached) == 2, (
        f"Cache debe ser tupla (env_raw, bool), fue {cached!r}"
    )
    assert cached == ("1", True), f"Cache esperado ('1', True), fue {cached!r}"
    # Mientras la env var sea la misma, el cache hit.
    assert rag._local_embed_enabled() is True
    # Si la env var cambia, el cache se invalida automáticamente.
    monkeypatch.setenv("RAG_LOCAL_EMBED", "0")
    assert rag._local_embed_enabled() is False, (
        "Cache invalidation: cambiar la env var debería re-evaluar"
    )
    # Reset para no contaminar otros tests
    rag._LOCAL_EMBED_ENABLED_CACHED = None


def test_local_embed_enabled_falls_back_to_env_when_uncached(monkeypatch):
    """Fix 6: sin cache, _local_embed_enabled lee env var (back-compat)."""
    rag._LOCAL_EMBED_ENABLED_CACHED = None
    monkeypatch.setenv("RAG_LOCAL_EMBED", "1")
    assert rag._local_embed_enabled() is True
    monkeypatch.setenv("RAG_LOCAL_EMBED", "0")
    assert rag._local_embed_enabled() is False
    monkeypatch.delenv("RAG_LOCAL_EMBED", raising=False)
    assert rag._local_embed_enabled() is False
    rag._LOCAL_EMBED_ENABLED_CACHED = None
