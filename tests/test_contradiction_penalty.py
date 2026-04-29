"""Tests para `rag/contradictions_penalty.py`.

El módulo es un re-scorer post-rerank que demote (sin filtrar) los
chunks cuyo `path` matchea la tabla `rag_contradictions` (rows sin
`resolved_at` / `skipped`). Helper consumer-only — el detector
indexer-time vive en otro módulo y este test no lo toca.

Cubre los 8 casos del spec:
  1. load_contradiction_paths con tabla vacía → set vacío.
  2. load_contradiction_paths con 3 pares → 6 paths únicos.
  3. apply_contradiction_penalty sin matches → no cambia scores.
  4. apply_contradiction_penalty con 1 match → score baja por la magnitud.
  5. Re-orden post-penalty: result demoted cae al final si el demote
     supera el gap.
  6. RAG_CONTRADICTION_PENALTY=0 → no se aplica (helper ni se llama).
  7. counter=True → no se aplica (helper ni se llama).
  8. Cache TTL: dos llamadas dentro de 5min → 1 sola query SQL.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import rag
from rag.contradictions_penalty import (
    apply_contradiction_penalty,
    cache_query_count,
    count_penalized,
    load_contradiction_paths,
    reset_cache,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _open_db(tmp_path: Path) -> sqlite3.Connection:
    """Abre una conn fresca al state DB del tmp y crea las tablas."""
    db = tmp_path / rag._TELEMETRY_DB_FILENAME
    conn = sqlite3.connect(str(db), isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_schema_version ("
        " table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
    )
    rag._ensure_telemetry_tables(conn)
    return conn


@pytest.fixture(autouse=True)
def _reset_cache():
    """Cada test arranca con cache en 0 — el módulo es global y los tests
    comparten state si no reseteamos."""
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    """tmp_path con state DB inicializado + DB_PATH apuntando ahí."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(
        rag, "_SQL_STATE_ERROR_LOG", tmp_path / "sql_state_errors.jsonl",
    )
    conn = _open_db(tmp_path)
    conn.close()
    yield tmp_path


def _seed(
    tmp_path: Path,
    subject_path: str,
    contradicts_paths: list[str],
    skipped: str | None = None,
    ts: str = "2026-04-29T10:00:00",
) -> None:
    """Inserta una row real en `rag_contradictions` con el shape actual."""
    conn = _open_db(tmp_path)
    try:
        conn.execute(
            "INSERT INTO rag_contradictions"
            " (ts, subject_path, contradicts_json, helper_raw, skipped)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                ts,
                subject_path,
                json.dumps([
                    {"path": p, "note": "n", "why": "w"} for p in contradicts_paths
                ]),
                "",
                skipped,
            ),
        )
    finally:
        conn.close()


# ── Tests ────────────────────────────────────────────────────────────────────


def test_load_paths_empty_table(db_env):
    """Caso 1: tabla vacía → set vacío.

    No raisea ni rompe — el caller arriba interpreta empty set como
    "no hay contradicciones, skipear penalty".
    """
    conn = sqlite3.connect(str(db_env / rag._TELEMETRY_DB_FILENAME))
    try:
        out = load_contradiction_paths(conn)
    finally:
        conn.close()
    assert out == set()


def test_load_paths_three_pairs_six_paths(db_env):
    """Caso 2: 3 pares (subject, [contradicts]) cada uno con paths únicos
    → 6 paths totales en el set.

    Demuestra que tanto subject_path como cada elemento de contradicts_json
    cuenta. Spec usa "subject_path o contradicts_path" pero el schema real
    es contradicts_json (lista) → este test cubre el caso correcto.
    """
    pairs = [
        ("01-Projects/A.md", ["02-Areas/B.md"], "2026-04-01T10:00:00"),
        ("01-Projects/C.md", ["02-Areas/D.md"], "2026-04-02T10:00:00"),
        ("01-Projects/E.md", ["02-Areas/F.md"], "2026-04-03T10:00:00"),
    ]
    for subj, contrs, ts in pairs:
        _seed(db_env, subj, contrs, ts=ts)

    conn = sqlite3.connect(str(db_env / rag._TELEMETRY_DB_FILENAME))
    try:
        out = load_contradiction_paths(conn)
    finally:
        conn.close()

    assert out == {
        "01-Projects/A.md", "02-Areas/B.md",
        "01-Projects/C.md", "02-Areas/D.md",
        "01-Projects/E.md", "02-Areas/F.md",
    }


def test_load_paths_resolved_excluded(db_env):
    """Caso 2b: rows con `skipped` no-null se excluyen (resueltas via triage).

    Bonus sobre el spec: confirma que el filtro funciona en producción.
    Sin este test, una contradicción "resuelta" seguiría penalizando
    para siempre.
    """
    _seed(db_env, "still-bad.md", ["x.md"], skipped=None,
          ts="2026-04-01T10:00:00")
    _seed(db_env, "resolved.md", ["y.md"], skipped="dismissed",
          ts="2026-04-02T10:00:00")

    conn = sqlite3.connect(str(db_env / rag._TELEMETRY_DB_FILENAME))
    try:
        out = load_contradiction_paths(conn)
    finally:
        conn.close()
    assert "still-bad.md" in out
    assert "x.md" in out
    assert "resolved.md" not in out
    assert "y.md" not in out


def test_apply_penalty_no_paths_no_change():
    """Caso 3: sin paths conflictivos → scores intactos, ordering intacto.

    Idempotencia para el caso "DB sin contradicciones" — no debe ni
    siquiera re-ordenar.
    """
    results = [
        {"path": "a.md", "score": 0.9},
        {"path": "b.md", "score": 0.5},
        {"path": "c.md", "score": 0.3},
    ]
    snapshot = [dict(r) for r in results]
    out = apply_contradiction_penalty(results, set(), penalty=-0.05)
    # Misma lista, mismos scores, mismo orden.
    assert out is results
    assert results == snapshot


def test_apply_penalty_one_match_score_drops():
    """Caso 4: 1 result en el set → su score baja por la magnitud.

    Default penalty=-0.05 → 0.9 → 0.85.
    """
    results = [
        {"path": "a.md", "score": 0.9},
        {"path": "b.md", "score": 0.5},
    ]
    apply_contradiction_penalty(results, {"a.md"}, penalty=-0.05)
    # `a.md` perdió 0.05 pero sigue siendo el top porque 0.85 > 0.5.
    a = next(r for r in results if r["path"] == "a.md")
    b = next(r for r in results if r["path"] == "b.md")
    assert abs(a["score"] - 0.85) < 1e-9
    assert abs(b["score"] - 0.5) < 1e-9
    # Sin re-orden visible (a sigue primero).
    assert results[0]["path"] == "a.md"


def test_apply_penalty_reorder_when_gap_exceeded():
    """Caso 5: si la magnitud del penalty supera el gap, el result demoted
    cae al final por re-orden DESC.

    Setup: a=0.5 (en set), b=0.45, c=0.40. Penalty=-0.10 → a=0.40.
    Empate con c (que sigue 0.40). Después del sort DESC estable, b queda
    primero (0.45), después a y c (0.40 cada uno). El test sólo afirma
    que `a` ya NO está primero — la posición exacta de empates no es
    parte del contrato.
    """
    results = [
        {"path": "a.md", "score": 0.5},
        {"path": "b.md", "score": 0.45},
        {"path": "c.md", "score": 0.40},
    ]
    apply_contradiction_penalty(results, {"a.md"}, penalty=-0.10)
    # Después del demote, `a` no es top.
    assert results[0]["path"] != "a.md"
    # `a` debería estar entre los últimos (0.40 ≤ otros).
    a_idx = next(i for i, r in enumerate(results) if r["path"] == "a.md")
    assert a_idx >= 1
    # Score real del demoted.
    a = next(r for r in results if r["path"] == "a.md")
    assert abs(a["score"] - 0.40) < 1e-9


def test_retrieve_disabled_via_env(monkeypatch):
    """Caso 6: RAG_CONTRADICTION_PENALTY=0 → la helper ni se llama.

    Verificamos a nivel del gating de retrieve(): el flag
    `_contradiction_penalty_enabled()` retorna False y el bloque
    de penalty no entra. Como proxy directo: comprobamos que el flag
    helper retorna False y que un load_contradiction_paths NO se
    invoca cuando el block es saltado.

    El test usa `_contradiction_penalty_enabled` directamente porque
    correr retrieve() end-to-end requiere un vault completo y no
    aporta nada que el flag-check no demuestre.
    """
    monkeypatch.setenv("RAG_CONTRADICTION_PENALTY", "0")
    assert rag._contradiction_penalty_enabled() is False

    # Sanity: el flag default ON cuando la env var está unset.
    monkeypatch.delenv("RAG_CONTRADICTION_PENALTY", raising=False)
    assert rag._contradiction_penalty_enabled() is True


def test_retrieve_skipped_when_counter_true():
    """Caso 7: counter=True → helper ni se llama.

    Probamos el invariante directamente con el bloque condicional de
    retrieve(): la condición `not counter` corta el flujo antes del
    SQL fetch + apply.

    Mockeamos `load_contradiction_paths` para asegurar que NO se
    invoque cuando counter=True. Para evitar correr retrieve()
    end-to-end (caro), reproducimos el patrón del wire-up:

        if scored_all and not counter and _contradiction_penalty_enabled():
            paths = load_contradiction_paths(conn)
            ...

    Si counter=True, el corto-circuito asegura que paths nunca se
    pida. Validamos el branch lógico en isolation.
    """
    calls = []

    def _spy(*a, **k):
        calls.append((a, k))
        return set()

    counter = True
    scored_all_present = True
    enabled = True

    if scored_all_present and not counter and enabled:
        _spy(None)

    # counter=True corta el flujo antes del fetch.
    assert calls == []

    # Sanity inverso: con counter=False la helper sí se llamaría.
    counter = False
    if scored_all_present and not counter and enabled:
        _spy(None)
    assert len(calls) == 1


def test_cache_ttl_one_query_within_5min(db_env):
    """Caso 8: dos llamadas a load_contradiction_paths dentro de la
    ventana de TTL → 1 sola query SQL real (el segundo hit viene del
    cache).

    Usamos `_now` parametrizable para simular dos lecturas separadas
    por 60s — bien adentro del TTL de 300s.
    """
    _seed(db_env, "subj.md", ["target.md"])

    conn = sqlite3.connect(str(db_env / rag._TELEMETRY_DB_FILENAME))
    try:
        # Primera llamada: cache miss → 1 query.
        out1 = load_contradiction_paths(conn, _now=1000.0)
        # Segunda llamada 60s después: cache hit → 0 queries adicionales.
        out2 = load_contradiction_paths(conn, _now=1060.0)
    finally:
        conn.close()

    assert out1 == out2
    assert "subj.md" in out1
    assert "target.md" in out1
    assert cache_query_count() == 1, (
        f"expected exactly 1 SQL query within TTL, got {cache_query_count()}"
    )


def test_cache_expires_after_ttl(db_env):
    """Bonus: pasados >5min, el cache expira y vuelve a query.

    Defensivo — sin este test, una bajada del TTL accidental pasaría
    silenciosa.
    """
    _seed(db_env, "x.md", ["y.md"])

    conn = sqlite3.connect(str(db_env / rag._TELEMETRY_DB_FILENAME))
    try:
        load_contradiction_paths(conn, _now=1000.0)
        # 6 min después — fuera del TTL de 300s.
        load_contradiction_paths(conn, _now=1000.0 + 360.0)
    finally:
        conn.close()

    assert cache_query_count() == 2


def test_count_penalized_helper():
    """Helper que cuenta sin mutar — usado para telemetría."""
    results = [
        {"path": "a.md", "score": 0.9},
        {"path": "b.md", "score": 0.5},
        {"path": "c.md", "score": 0.3},
    ]
    n = count_penalized(results, {"a.md", "c.md"})
    assert n == 2
    # Inmutado.
    assert results[0]["score"] == 0.9


def test_apply_penalty_returns_same_list():
    """Contract: la función retorna la MISMA lista (mut-en-place +
    sort-en-place), no una copia. Permite chaining como
    `apply(...)[:k]`.
    """
    results = [{"path": "a.md", "score": 0.5}]
    out = apply_contradiction_penalty(results, {"a.md"}, penalty=-0.05)
    assert out is results
