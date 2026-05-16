"""Tests fundacionales de rag.runtime.scheduler.

Cubren:
- decorators registran al singleton
- ``run_now`` dispatch sync con telemetría
- exit_code distinto cuando handler raisea
- jobs registrados se listan correctamente
- shutdown limpio
"""
from __future__ import annotations

import pytest

from rag.runtime.scheduler import Job, Scheduler, cron, interval


@pytest.fixture(autouse=True)
def _reset_scheduler():
    """Reset singleton entre tests para evitar state leak."""
    Scheduler.reset_global()
    yield
    Scheduler.reset_global()


@pytest.fixture(autouse=True)
def _isolate_telemetry_db(tmp_path, monkeypatch):
    """Aisla `rag_supervisor_jobs` a un tmp_path per-test.

    Pre-fix (audit 2026-05-11): `run_now()` → `_persist_run()` →
    `insert_supervisor_job_run()` escribía a la telemetry.db de PROD
    porque ningún fixture seteaba `OBSIDIAN_RAG_DB_PATH`. Resultado:
    15+ rows con `job_label IN ('failing_job', 'bad_job', 'counter_job',
    'stat_job')` apareciendo en prod cada vez que la suite corría.
    `stale_etl_signal` no las flaggeaba (no son labels conocidos), pero
    ensuciaban queries de analytics + auditoría.
    """
    monkeypatch.setenv("OBSIDIAN_RAG_DB_PATH", str(tmp_path))


def test_cron_decorator_registers_job():
    @cron(hour=3, minute=0, label="test_cron_job")
    def handler():
        return "ok"

    sched = Scheduler.global_instance()
    assert "test_cron_job" in sched.jobs()
    job = sched.get_job("test_cron_job")
    assert job is not None
    assert job.trigger_kind == "cron"
    assert job.trigger_args == {"hour": 3, "minute": 0}


def test_interval_decorator_registers_job():
    @interval(seconds=900, label="test_interval_job")
    def handler():
        return "ok"

    job = Scheduler.global_instance().get_job("test_interval_job")
    assert job is not None
    assert job.trigger_kind == "interval"
    assert job.trigger_args == {"seconds": 900}


def test_interval_requires_unit():
    with pytest.raises(ValueError, match="seconds/minutes/hours"):
        @interval(label="bad")
        def handler():
            pass


def test_run_now_returns_ok_for_clean_handler():
    counter = {"n": 0}

    @cron(hour=3, label="counter_job")
    def handler():
        counter["n"] += 1
        return {"value": counter["n"]}

    result = Scheduler.global_instance().run_now("counter_job")
    assert result["ok"] is True
    assert result["error"] is None
    assert result["result"] == {"value": 1}
    assert isinstance(result["duration_s"], float)
    assert result["duration_s"] > 0
    assert counter["n"] == 1


def test_run_now_captures_exception():
    @cron(hour=3, label="bad_job")
    def handler():
        raise RuntimeError("boom")

    result = Scheduler.global_instance().run_now("bad_job")
    assert result["ok"] is False
    assert "boom" in (result["error"] or "")


def test_run_now_honors_handler_exit_code_payload():
    sched = Scheduler(headless=True)
    sched.register_job(Job(
        label="subprocess_job",
        handler=lambda: {"exit_code": 2, "last_stderr": "boom"},
        trigger_kind="cron",
        trigger_args={"hour": 3},
    ))

    result = sched.run_now("subprocess_job")
    job = sched.get_job("subprocess_job")

    assert result["ok"] is False
    assert result["error"] == "boom"
    assert job is not None
    assert job.last_exit_code == 2
    assert job.fails_count == 1


def test_run_now_unknown_job():
    result = Scheduler.global_instance().run_now("does_not_exist")
    assert result["ok"] is False
    assert "unknown job" in (result["error"] or "")


def test_job_stats_updated_after_run():
    @interval(seconds=60, label="stat_job")
    def handler():
        return None

    sched = Scheduler.global_instance()
    sched.run_now("stat_job")
    sched.run_now("stat_job")

    job = sched.get_job("stat_job")
    assert job is not None
    assert job.runs_count == 2
    assert job.fails_count == 0


def test_job_fails_count_increments_on_error():
    @interval(seconds=60, label="failing_job")
    def handler():
        raise ValueError("nope")

    sched = Scheduler.global_instance()
    sched.run_now("failing_job")
    sched.run_now("failing_job")

    job = sched.get_job("failing_job")
    assert job is not None
    assert job.runs_count == 2
    assert job.fails_count == 2


def test_headless_mode_no_apscheduler_dependency():
    sched = Scheduler(headless=True)
    sched.register_job(Job(
        label="manual",
        handler=lambda: 42,
        trigger_kind="cron",
        trigger_args={"hour": 3},
    ))
    sched.start()  # No-op en headless
    result = sched.run_now("manual")
    assert result["ok"] is True
    assert result["result"] == 42
    sched.shutdown()  # Idempotente


def test_re_register_overwrites():
    @cron(hour=3, label="dup")
    def handler_v1():
        return "v1"

    @cron(hour=4, label="dup")
    def handler_v2():
        return "v2"

    job = Scheduler.global_instance().get_job("dup")
    assert job is not None
    assert job.trigger_args == {"hour": 4, "minute": 0}
    result = Scheduler.global_instance().run_now("dup")
    assert result["result"] == "v2"
