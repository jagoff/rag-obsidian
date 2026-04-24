"""Tests para el wiring async de behavior/impressions/metrics writers.

Audit 2026-04-24: `sql_state_errors.jsonl` tenía 1756 entries en 6 días
con la distribución:

    402 semantic_cache_store_failed    (ya async desde 2026-04-22)
    316 queries_sql_write_failed       (async desde 2026-04-22)
    156 impression_sql_write_failed    ← ahora async
     66 memory_sql_write_failed        ← ahora async
     34 cpu_sql_write_failed           ← ahora async
     34 behavior_sql_write_failed      ← ahora async

Este archivo verifica:
1. Los 4 writers nuevos respetan el env var override (`RAG_LOG_BEHAVIOR_ASYNC`,
   `RAG_METRICS_ASYNC`).
2. El path async enqueueá en `_BACKGROUND_SQL_QUEUE`; el path sync NO.
3. Los defaults del helper son permisivos (val not in {"0","false","no"}).
"""
from __future__ import annotations

import rag


# ── _log_behavior_event_background_default ───────────────────────────────────


def test_behavior_async_default_on_when_unset(monkeypatch):
    monkeypatch.delenv("RAG_LOG_BEHAVIOR_ASYNC", raising=False)
    assert rag._log_behavior_event_background_default() is True


def test_behavior_async_respects_zero(monkeypatch):
    monkeypatch.setenv("RAG_LOG_BEHAVIOR_ASYNC", "0")
    assert rag._log_behavior_event_background_default() is False


def test_behavior_async_respects_false(monkeypatch):
    monkeypatch.setenv("RAG_LOG_BEHAVIOR_ASYNC", "false")
    assert rag._log_behavior_event_background_default() is False


def test_behavior_async_respects_no(monkeypatch):
    monkeypatch.setenv("RAG_LOG_BEHAVIOR_ASYNC", "no")
    assert rag._log_behavior_event_background_default() is False


def test_behavior_async_is_on_for_arbitrary_truthy(monkeypatch):
    """Mismo pattern que RAG_LOG_QUERY_ASYNC: cualquier string fuera de
    {0,false,no} es ON. `1`, `true`, `yes`, o strings vacíos → async."""
    for val in ("1", "true", "yes", "anything", ""):
        monkeypatch.setenv("RAG_LOG_BEHAVIOR_ASYNC", val)
        assert rag._log_behavior_event_background_default() is True, (
            f"value {val!r} should be truthy"
        )


# ── _metrics_background_default (web/server.py) ──────────────────────────────


def test_metrics_async_default_on_when_unset(monkeypatch):
    from web import server as srv
    monkeypatch.delenv("RAG_METRICS_ASYNC", raising=False)
    assert srv._metrics_background_default() is True


def test_metrics_async_respects_zero(monkeypatch):
    from web import server as srv
    monkeypatch.setenv("RAG_METRICS_ASYNC", "0")
    assert srv._metrics_background_default() is False


# ── log_behavior_event enqueues on async path ────────────────────────────────


def test_log_behavior_event_enqueues_when_async(monkeypatch):
    """En modo async, `log_behavior_event` debe llamar
    `_enqueue_background_sql` en vez de `_sql_write_with_retry`."""
    monkeypatch.setenv("RAG_LOG_BEHAVIOR_ASYNC", "1")
    calls = {"enqueue": 0, "sync": 0}

    def fake_enqueue(fn, tag):
        calls["enqueue"] += 1
        assert tag == "behavior_sql_write_failed"

    def fake_sync(fn, tag):
        calls["sync"] += 1

    monkeypatch.setattr(rag, "_enqueue_background_sql", fake_enqueue)
    monkeypatch.setattr(rag, "_sql_write_with_retry", fake_sync)

    rag.log_behavior_event({"source": "cli", "event": "open", "path": "x.md"})

    assert calls["enqueue"] == 1, f"expected 1 enqueue, got {calls}"
    assert calls["sync"] == 0


def test_log_behavior_event_uses_sync_when_override(monkeypatch):
    monkeypatch.setenv("RAG_LOG_BEHAVIOR_ASYNC", "0")
    calls = {"enqueue": 0, "sync": 0}

    def fake_enqueue(fn, tag):
        calls["enqueue"] += 1

    def fake_sync(fn, tag):
        calls["sync"] += 1
        assert tag == "behavior_sql_write_failed"

    monkeypatch.setattr(rag, "_enqueue_background_sql", fake_enqueue)
    monkeypatch.setattr(rag, "_sql_write_with_retry", fake_sync)

    rag.log_behavior_event({"source": "cli", "event": "open", "path": "x.md"})

    assert calls["sync"] == 1, f"expected 1 sync write, got {calls}"
    assert calls["enqueue"] == 0


# ── log_impressions enqueues on async path ───────────────────────────────────


def test_log_impressions_enqueues_when_async(monkeypatch):
    monkeypatch.setenv("RAG_LOG_BEHAVIOR_ASYNC", "1")
    # Bypass throttle — nueva query+top1 cada call.
    monkeypatch.setattr(rag, "_impression_last_seen", {})
    calls = {"enqueue": 0, "sync": 0}

    def fake_enqueue(fn, tag):
        calls["enqueue"] += 1
        assert tag == "impression_sql_write_failed"

    def fake_sync(fn, tag):
        calls["sync"] += 1

    monkeypatch.setattr(rag, "_enqueue_background_sql", fake_enqueue)
    monkeypatch.setattr(rag, "_sql_write_with_retry", fake_sync)

    rag.log_impressions("query_x", ["a.md", "b.md"])

    assert calls["enqueue"] == 1
    assert calls["sync"] == 0


# ── Web samplers ─────────────────────────────────────────────────────────────


def test_memory_persist_enqueues_when_async(monkeypatch):
    from web import server as srv
    monkeypatch.setenv("RAG_METRICS_ASYNC", "1")
    calls = {"enqueue": 0, "sync": 0}

    def fake_enqueue(fn, tag):
        calls["enqueue"] += 1
        assert tag == "memory_sql_write_failed"

    def fake_sync(fn, tag):
        calls["sync"] += 1

    monkeypatch.setattr(srv, "_enqueue_background_sql", fake_enqueue)
    monkeypatch.setattr(srv, "_persist_with_sqlite_retry", fake_sync)

    srv._memory_persist({"ts": "2026-04-24T14:00:00", "wired_pct": 0.5})

    assert calls["enqueue"] == 1, f"expected 1 enqueue, got {calls}"
    assert calls["sync"] == 0


def test_cpu_persist_enqueues_when_async(monkeypatch):
    from web import server as srv
    monkeypatch.setenv("RAG_METRICS_ASYNC", "1")
    calls = {"enqueue": 0, "sync": 0}

    def fake_enqueue(fn, tag):
        calls["enqueue"] += 1
        assert tag == "cpu_sql_write_failed"

    def fake_sync(fn, tag):
        calls["sync"] += 1

    monkeypatch.setattr(srv, "_enqueue_background_sql", fake_enqueue)
    monkeypatch.setattr(srv, "_persist_with_sqlite_retry", fake_sync)

    srv._cpu_persist({"ts": "2026-04-24T14:00:00", "user_pct": 0.3})

    assert calls["enqueue"] == 1, f"expected 1 enqueue, got {calls}"
    assert calls["sync"] == 0
