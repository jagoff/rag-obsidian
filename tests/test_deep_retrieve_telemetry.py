"""Regression test del audit 2026-04-25 — deep_retrieve telemetry persistence.

`deep_retrieve()` etiqueta el dict resultado con `deep_retrieve_iterations`
+ `deep_retrieve_exit_reason` (commit `e81251f`), pero los callers (CLI
`query`/`chat`/`serve` + web `cmd=web`) no estaban propagando esos campos
al payload de `log_query_event`. Resultado: imposible medir empíricamente
el hit rate del high-confidence bypass o cuántas queries gastan más de 1
iteración del sufficiency loop.

Este test asegura que:

1. `log_query_event` con esos dos campos en el dict los persiste a
   `rag_queries.extra_json` (no se rompen por keys nuevas; el mapper
   `_to_rag_queries_row` los rutea automáticamente al blob JSON).
2. Cuando no se proveen (caller no corrió deep), los campos quedan
   `null` en JSON o ausentes, sin bloquear el insert.
3. Es queryable via `json_extract` para analytics tipo "qué fracción
   de queries de los últimos 7d corrió >1 iteración?".
"""
from __future__ import annotations

import json
import sqlite3

import rag


def _connect(tmp_path):
    return sqlite3.connect(str(tmp_path / "telemetry.db"))


def test_deep_retrieve_iterations_persisted_in_extra_json(tmp_path, monkeypatch):
    """Cuando deep_retrieve corrió y exitó por high_confidence_bypass,
    ambos campos quedan recuperables desde extra_json."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "0")

    rag.log_query_event({
        "cmd": "web",
        "q": "test query bypass",
        "session": "sess-bypass",
        "deep_retrieve_iterations": 1,
        "deep_retrieve_exit_reason": "high_confidence_bypass",
    })

    conn = _connect(tmp_path)
    try:
        row = conn.execute(
            "SELECT extra_json FROM rag_queries WHERE q = ?",
            ("test query bypass",),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None, "esperaba 1 row insertada"
    extra = json.loads(row[0])
    assert extra.get("deep_retrieve_iterations") == 1
    assert extra.get("deep_retrieve_exit_reason") == "high_confidence_bypass"


def test_deep_retrieve_exit_reason_supports_all_branches(tmp_path, monkeypatch):
    """Las 6 razones de salida que `deep_retrieve()` puede emitir deben
    persistir como string literal sin truncarse ni mutar."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "0")

    # Las 6 razones que el código puede setear (rag/__init__.py:19410+)
    reasons = [
        ("no_docs", 1),
        ("high_confidence_bypass", 1),
        ("low_confidence_bypass", 1),
        ("sufficient", 2),
        ("no_sub_query", 2),
        ("no_new_chunks", 2),
        ("timeout", 2),
        ("max_iters", 3),
    ]
    for reason, iters in reasons:
        rag.log_query_event({
            "cmd": "query",
            "q": f"test {reason}",
            "deep_retrieve_iterations": iters,
            "deep_retrieve_exit_reason": reason,
        })

    conn = _connect(tmp_path)
    try:
        rows = conn.execute(
            "SELECT q, extra_json FROM rag_queries WHERE q LIKE 'test %'"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == len(reasons)
    by_q = {q: json.loads(blob) for q, blob in rows}
    for reason, iters in reasons:
        extra = by_q[f"test {reason}"]
        assert extra["deep_retrieve_iterations"] == iters, reason
        assert extra["deep_retrieve_exit_reason"] == reason, reason


def test_log_query_event_without_deep_fields_does_not_break(tmp_path, monkeypatch):
    """Cuando deep_retrieve no corrió (caller no llamó la función o ya
    tenía un fast_path), el payload no tiene esos dos campos. El insert
    debe seguir funcionando — son opcionales."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "0")

    rag.log_query_event({
        "cmd": "web",
        "q": "fast path no deep",
        "session": "sess-fast",
        "fast_path": True,
        # NO se setean deep_retrieve_iterations/exit_reason
    })

    conn = _connect(tmp_path)
    try:
        row = conn.execute(
            "SELECT extra_json FROM rag_queries WHERE q = ?",
            ("fast path no deep",),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    extra = json.loads(row[0])
    # Aceptamos cualquiera de: ausencia, o None explícito.
    assert extra.get("deep_retrieve_iterations") in (None, ""), extra
    assert extra.get("deep_retrieve_exit_reason") in (None, ""), extra
    # El otro campo (fast_path) sí está, demostrando que el insert
    # no se descartó.
    assert extra.get("fast_path") is True


def test_deep_retrieve_telemetry_queryable_via_json_extract(tmp_path, monkeypatch):
    """SQL aggregate típico que vamos a correr en el dashboard:
    "% queries que corrieron >1 iteración del sufficiency loop".
    Tiene que funcionar con la columna extra_json + json_extract.
    """
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setenv("RAG_LOG_QUERY_ASYNC", "0")

    # 5 queries: 3 con bypass (iters=1), 2 con loop (iters=2,3)
    payloads = [
        ("q1", 1, "high_confidence_bypass"),
        ("q2", 1, "high_confidence_bypass"),
        ("q3", 1, "no_docs"),
        ("q4", 2, "sufficient"),
        ("q5", 3, "max_iters"),
    ]
    for q, iters, reason in payloads:
        rag.log_query_event({
            "cmd": "web", "q": q,
            "deep_retrieve_iterations": iters,
            "deep_retrieve_exit_reason": reason,
        })

    conn = _connect(tmp_path)
    try:
        # Misma query SQL que iría en el dashboard.
        n_multi = conn.execute(
            "SELECT COUNT(*) FROM rag_queries "
            "WHERE CAST(json_extract(extra_json, '$.deep_retrieve_iterations') AS INTEGER) > 1"
        ).fetchone()[0]
        n_total = conn.execute(
            "SELECT COUNT(*) FROM rag_queries "
            "WHERE json_extract(extra_json, '$.deep_retrieve_iterations') IS NOT NULL"
        ).fetchone()[0]
        n_bypass = conn.execute(
            "SELECT COUNT(*) FROM rag_queries "
            "WHERE json_extract(extra_json, '$.deep_retrieve_exit_reason') = 'high_confidence_bypass'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert n_total == 5, "5 queries con telemetría deep"
    assert n_multi == 2, "q4 + q5 corrieron >1 iter"
    assert n_bypass == 2, "q1 + q2 hit el bypass"
