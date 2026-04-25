"""Tests for SQL transient-lock retry behaviour on telemetry writes.

Auditoría 2026-04-22 encontró en ``sql_state_errors.jsonl``:

  - 314x ``semantic_cache_store_failed`` con ``database is locked``
  - 115x ``queries_sql_write_failed`` (ya tenía retry de 3 intentos, pero
    el jittered backoff ``0.1-0.35s`` × 3 = ~0.75s máximo no alcanzaba)
  - 52x ``feedback_golden_sql_read_failed`` (read-only, no tenía retry)
  - 46x ``memory_sql_write_failed`` (ya usa retry)
  - 34x ``behavior_sql_write_failed`` (ya usa retry)

Este módulo cubre las tres brechas:

  1. ``semantic_cache_store`` envuelve su INSERT en
     ``_sql_write_with_retry`` — antes silent-failaba al primer
     ``database is locked``.
  2. ``_sql_write_with_retry`` usa 5 intentos (vs 3) y backoff más
     largo (base 0.15s, random 0.35s) — contention window medida en
     producción supera los 0.75s del esquema viejo.
  3. ``get_feedback_golden_snapshot`` (read) usa un helper nuevo
     ``_sql_read_with_retry`` que reintenta ``database is locked``
     con el mismo patrón.

No tocamos ``_sql_write_with_retry`` aguas arriba para evitar trastocar
14 writers en producción: subimos el upper-bound de attempts vía kwarg
default sólo — el llamador sigue llamando ``_sql_write_with_retry(_do,
"tag")`` y hereda la mejora automáticamente.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch


import rag


# ── _sql_write_with_retry: attempts + backoff invariants ──────────────────────


def test_sql_write_with_retry_default_attempts_is_8():
    """Evolución del default:

    - Pre-2026-04-22: 3 intentos (budget ~0.75s). Demasiado corto bajo
      contention sostenida.
    - 2026-04-22: bumped a 5 (budget ~1.3s). Todavía quedaba corto bajo
      3+ writers concurrentes (queries + memory + cpu samplers alineados).
    - 2026-04-23: bumped a 8 (budget ~4s). Match del `_persist_with_sqlite_retry`
      de web/server.py. Hot-path callers que no toleran 4s pasan
      `attempts=3` explícito.
    """
    calls = {"n": 0}

    def writer():
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with patch("time.sleep"):  # compress the backoff
        rag._sql_write_with_retry(writer, "test_tag")

    assert calls["n"] == 8, \
        f"expected 8 retry attempts (post 2026-04-23), got {calls['n']}"


def test_sql_write_with_retry_succeeds_on_attempt_3(tmp_path, monkeypatch):
    """Retry terminates early once the write succeeds."""
    calls = {"n": 0}

    def writer():
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        # success on 3rd

    with patch("time.sleep"):
        rag._sql_write_with_retry(writer, "test_tag")

    assert calls["n"] == 3


def test_sql_write_with_retry_backoff_upper_bound_increased():
    """The sleep between retries must at least be able to exceed 0.15s
    (previously floor was 0.1s). Spot-check by capturing the sleep arg."""
    sleeps: list[float] = []

    def writer():
        raise sqlite3.OperationalError("database is locked")

    def fake_sleep(duration):
        sleeps.append(duration)

    with patch("time.sleep", fake_sleep):
        rag._sql_write_with_retry(writer, "test_tag")

    # 7 sleeps between 8 attempts (post 2026-04-23 bump). Each must be
    # >=0.15 (floor) and <=0.6 (ceiling = 0.15 + 0.45). Tests las
    # constantes sin hard-codearlas al call site.
    assert len(sleeps) == 7, f"expected 7 backoff sleeps, got {len(sleeps)}"
    for s in sleeps:
        assert 0.15 <= s <= 0.6, f"sleep {s}s outside [0.15, 0.6]"


def test_sql_write_with_retry_non_transient_errors_fail_fast():
    """Schema drift, constraint violations, corruption etc. must NOT retry —
    they don't resolve by waiting, and retrying just wastes latency.

    Post 2026-04-22 `disk I/O error` se considera transient y se reintenta
    (ver test_sql_disk_io_retry.py para esa expansión + justificación
    empírica). Este test usa errores que SÍ son permanentes."""
    for non_transient in (
        "no such table: rag_queries",  # schema drift
        "UNIQUE constraint failed",     # dup row
        "database disk image is malformed",  # corruption
    ):
        calls = {"n": 0}

        def writer():
            calls["n"] += 1
            raise sqlite3.OperationalError(non_transient)

        with patch("time.sleep"):
            rag._sql_write_with_retry(writer, "test_tag")

        assert calls["n"] == 1, \
            f"{non_transient!r} is not transient — should fail fast"


# ── semantic_cache_store: now uses retry wrapper ──────────────────────────────


def test_semantic_cache_store_uses_retry_on_lock(monkeypatch):
    """Pre-2026-04-22 semantic_cache_store silently failed on the first
    ``database is locked`` — which accounted for 314 of the errors in
    sql_state_errors.jsonl. After the fix, the INSERT must run inside
    ``_sql_write_with_retry``."""
    import numpy as np

    monkeypatch.setenv("RAG_CACHE_ENABLED", "1")
    q_emb = np.zeros(1024, dtype="float32")
    q_emb[0] = 1.0
    q_emb = q_emb / np.linalg.norm(q_emb)

    retry_calls = {"n": 0}
    orig = rag._sql_write_with_retry

    def tracking_wrapper(fn, tag, **kw):
        retry_calls["n"] += 1
        retry_calls["tag"] = tag
        return orig(fn, tag, **kw)

    monkeypatch.setattr(rag, "_sql_write_with_retry", tracking_wrapper)

    # Provide valid args so the function reaches the SQL path.
    rag.semantic_cache_store(
        q_embedding=q_emb.tolist(),
        question="test",
        response="resp",
        paths=["a.md"],
        scores=[0.9],
        top_score=0.9,
        intent="semantic",
        corpus_hash="dead" * 16,
    )

    assert retry_calls["n"] == 1, \
        f"semantic_cache_store must route its INSERT through " \
        f"_sql_write_with_retry (got {retry_calls['n']} calls)"
    assert retry_calls["tag"] == "semantic_cache_store_failed"


# ── _sql_read_with_retry: new helper for READs on a contended WAL ─────────────


def test_sql_read_with_retry_exists():
    assert hasattr(rag, "_sql_read_with_retry"), \
        "post 2026-04-22 rag.py must expose _sql_read_with_retry for " \
        "read-path lock contention (feedback_golden, corpus_hash, etc.)"


def test_sql_read_with_retry_returns_value_on_success():
    def reader():
        return {"ok": True}

    assert rag._sql_read_with_retry(reader, "x", default=None) == {"ok": True}


def test_sql_read_with_retry_returns_default_on_persistent_lock():
    def reader():
        raise sqlite3.OperationalError("database is locked")

    with patch("time.sleep"):
        out = rag._sql_read_with_retry(reader, "test_tag", default="fallback")
    assert out == "fallback"


def test_sql_read_with_retry_returns_on_transient_lock():
    calls = {"n": 0}

    def reader():
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return "recovered"

    with patch("time.sleep"):
        out = rag._sql_read_with_retry(reader, "test_tag", default=None)
    assert out == "recovered"
    assert calls["n"] == 3


def test_sql_read_with_retry_non_transient_propagates_as_default():
    """Non-transient errors degrade to default — same contract as
    _sql_write_with_retry, so reads never raise into caller land.

    Post 2026-04-22 `disk I/O error` es transient y se reintenta;
    usamos un error realmente no-transient aqui."""
    def reader():
        raise sqlite3.OperationalError("no such table: rag_queries")

    with patch("time.sleep"):
        out = rag._sql_read_with_retry(reader, "test_tag", default=[])
    assert out == []


def test_sql_read_with_retry_unexpected_exception_returns_default():
    def reader():
        raise ValueError("broken row")

    with patch("time.sleep"):
        out = rag._sql_read_with_retry(reader, "test_tag", default={"k": "v"})
    assert out == {"k": "v"}


def test_sql_read_with_retry_default_attempts_is_5():
    calls = {"n": 0}

    def reader():
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with patch("time.sleep"):
        rag._sql_read_with_retry(reader, "test_tag", default=None)

    assert calls["n"] == 5
