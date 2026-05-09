"""Tests F4.1+F4.2+F4.3 — SQL triggers + bridge hook integrados.

Verifica:
- ``start_all_watchers()`` arranca 3 threads.
- IPC handler ``status_sql_watchers`` registrado.
- Threshold mínimo en routing-rules (5 rows) y wa-tasks (3 mensajes).
- Subscribers async no bloquean.
- Opt-out via env var.
"""
from __future__ import annotations

import sys

import pytest

from rag.runtime import ipc


@pytest.fixture(autouse=True)
def _reset():
    ipc._reset_handlers()
    if "rag.runtime.jobs._sql_triggers" in sys.modules:
        sl = sys.modules["rag.runtime.jobs._sql_triggers"]
        ipc.register_handler("status_sql_watchers", sl.status_sql_watchers_handler)
    yield
    ipc._reset_handlers()


def test_module_imports_and_registers_handlers():
    import rag.runtime.jobs._sql_triggers  # noqa: F401
    assert "status_sql_watchers" in ipc._registered_handlers()


def test_status_handler_returns_three_watchers():
    import rag.runtime.jobs._sql_triggers as mod
    result = mod.status_sql_watchers_handler({})
    assert "routing" in result
    assert "drift" in result
    assert "wa" in result
    for k in ("routing", "drift", "wa"):
        assert "thread_alive" in result[k]
        assert "polls_count" in result[k]


def test_routing_handler_skips_below_threshold(monkeypatch):
    """5 rows mínimos para gatear extract-rules."""
    import rag.runtime.jobs._sql_triggers as mod
    n_calls = {"i": 0}

    def fake_routing():
        n_calls["i"] += 1
        return {"exit_code": 0}

    monkeypatch.setattr(
        "rag.runtime.jobs.frequent.routing_rules_job",
        fake_routing,
    )

    # Threshold no alcanzado — sub no debería disparar.
    mod._on_feedback_inserted({"new_rows": 3})
    assert n_calls["i"] == 0


def test_routing_handler_fires_above_threshold(monkeypatch):
    import rag.runtime.jobs._sql_triggers as mod
    n_calls = {"i": 0}

    def fake_routing():
        n_calls["i"] += 1
        return {"exit_code": 0}

    monkeypatch.setattr(
        "rag.runtime.jobs.frequent.routing_rules_job",
        fake_routing,
    )

    mod._on_feedback_inserted({"new_rows": 7})
    assert n_calls["i"] == 1


def test_drift_handler_fires_on_any_eval_run(monkeypatch):
    """Cada eval run dispara — sin threshold, runs son raros."""
    import rag.runtime.jobs._sql_triggers as mod
    n_calls = {"i": 0}

    def fake_drift():
        n_calls["i"] += 1
        return {"alerts": 0, "kinds": []}

    monkeypatch.setattr(
        "rag.runtime.jobs.drift_watcher.drift_watcher_job",
        fake_drift,
    )

    mod._on_eval_run_completed({"new_rows": 1})
    assert n_calls["i"] == 1


def test_wa_handler_skips_below_threshold(monkeypatch):
    import rag.runtime.jobs._sql_triggers as mod
    n_calls = {"i": 0}

    def fake_wa():
        n_calls["i"] += 1
        return {"exit_code": 0}

    monkeypatch.setattr(
        "rag.runtime.jobs.frequent.wa_tasks_job",
        fake_wa,
    )

    mod._on_wa_message_inbound({"new_rows": 1})
    assert n_calls["i"] == 0
    mod._on_wa_message_inbound({"new_rows": 5})
    assert n_calls["i"] == 1


def test_handler_exception_does_not_propagate(monkeypatch):
    """Si el job raisea, handler captura — no rompe el watcher thread."""
    import rag.runtime.jobs._sql_triggers as mod

    def crashy_routing():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "rag.runtime.jobs.frequent.routing_rules_job",
        crashy_routing,
    )

    # NO debería raisar.
    mod._on_feedback_inserted({"new_rows": 10})


def test_disabled_via_env_var(monkeypatch):
    """RAG_SQL_WATCHERS_DISABLED=1 → start_all_watchers retorna False
    para los 3."""
    import rag.runtime.jobs._sql_triggers as mod

    monkeypatch.setenv("RAG_SQL_WATCHERS_DISABLED", "1")
    result = mod.start_all_watchers()
    assert result == {"routing": False, "drift": False, "wa": False}
