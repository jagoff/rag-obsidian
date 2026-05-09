"""Tests de paridad — drift_watcher in-supervisor vs script standalone.

Garantizan que el wrapper del job en ``rag/runtime/jobs/drift_watcher.py``
delega a ``scripts/drift_watcher.py:evaluate()`` correctamente:

- Si ``evaluate()`` retorna ``[]`` → handler retorna ``alerts: 0``.
- Si retorna lista de alerts → handler retorna count + kinds.
- Si raisea → handler captura, retorna error en signals (no escala).

Tests no requieren un supervisor corriendo — se invoca el handler
directo.
"""
from __future__ import annotations

import sys

import pytest

from rag.runtime.scheduler import Scheduler


@pytest.fixture(autouse=True)
def _reset():
    Scheduler.reset_global()
    # Quitar drift_watcher de sys.modules para forzar re-import limpio.
    sys.modules.pop("drift_watcher", None)
    sys.modules.pop("rag.runtime.jobs.drift_watcher", None)
    yield
    Scheduler.reset_global()


def _import_handler():
    import rag.runtime.jobs.drift_watcher as mod
    return mod.drift_watcher_job


def test_handler_registered_with_6h_interval():
    _import_handler()
    job = Scheduler.global_instance().get_job("drift_watcher")
    assert job is not None
    assert job.trigger_kind == "interval"
    # APScheduler IntervalTrigger acepta hours=6 → equivalente a 21600s.
    assert job.trigger_args == {"hours": 6}


def test_handler_returns_zero_alerts_on_empty_evaluate(monkeypatch):
    handler = _import_handler()
    import rag.runtime.jobs.drift_watcher as mod

    fake_dw = type("M", (), {"evaluate": staticmethod(lambda: [])})
    monkeypatch.setattr(mod, "_import_evaluate", lambda: fake_dw)

    result = handler()
    assert result == {"alerts": 0, "kinds": []}


def test_handler_aggregates_kinds_from_alerts(monkeypatch):
    handler = _import_handler()
    import rag.runtime.jobs.drift_watcher as mod

    fake_alerts = [
        {"kind": "singles", "delta": -0.06},
        {"kind": "chains", "delta": -0.08},
        {"kind": "singles", "delta": -0.07},
    ]
    fake_dw = type("M", (), {"evaluate": staticmethod(lambda: fake_alerts)})
    monkeypatch.setattr(mod, "_import_evaluate", lambda: fake_dw)

    result = handler()
    assert result["alerts"] == 3
    assert sorted(result["kinds"]) == ["chains", "singles"]


def test_handler_captures_evaluate_exception(monkeypatch):
    handler = _import_handler()
    import rag.runtime.jobs.drift_watcher as mod

    def _raise():
        raise RuntimeError("DB locked")

    fake_dw = type("M", (), {"evaluate": staticmethod(_raise)})
    monkeypatch.setattr(mod, "_import_evaluate", lambda: fake_dw)

    result = handler()
    assert result["alerts"] == 0
    assert result["kinds"] == []
    assert "DB locked" in (result.get("error") or "")


def test_handler_handles_import_failure(monkeypatch):
    handler = _import_handler()
    import rag.runtime.jobs.drift_watcher as mod

    def _fail_import():
        raise ImportError("scripts/ no en sys.path")

    monkeypatch.setattr(mod, "_import_evaluate", _fail_import)

    result = handler()
    assert result["alerts"] == 0
    assert "scripts/" in (result.get("error") or "")


def test_handler_returns_dict_for_non_list_evaluate(monkeypatch):
    handler = _import_handler()
    import rag.runtime.jobs.drift_watcher as mod

    # Si evaluate() devuelve algo raro (string, dict, None), no crashear.
    fake_dw = type("M", (), {"evaluate": staticmethod(lambda: None)})
    monkeypatch.setattr(mod, "_import_evaluate", lambda: fake_dw)

    result = handler()
    assert result["alerts"] == 0
    assert result["kinds"] == []
