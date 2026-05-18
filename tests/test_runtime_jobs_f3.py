"""Tests F3 — migración de 18 jobs adicionales al supervisor.

Cubren:
- ``frequent.py`` — 11 hot-path frecuentes registrados con schedules
  matcheando los plists viejos.
- ``proactive.py`` — 7 weekly/monthly registrados con day_of_week +
  hour + minute correctos.
- ``briefs.py`` — 3 briefs Mon-Fri / Sun.
- ``housekeeping.py`` — 3 housekeeping daily/weekly.
- Total post-F3: 34 jobs (incluyendo drift_watcher F1 + 9 nightly F2/Gx).

NO testeamos invocación real — eso es shadow A/B en producción.
"""
from __future__ import annotations

import logging
import sys
import threading

import pytest

from rag.runtime.scheduler import Scheduler


@pytest.fixture(autouse=True)
def _reset():
    Scheduler.reset_global()
    for mod in list(sys.modules):
        if mod.startswith("rag.runtime.jobs"):
            del sys.modules[mod]
    yield
    Scheduler.reset_global()
    for mod in list(sys.modules):
        if mod.startswith("rag.runtime.jobs"):
            del sys.modules[mod]


def _import_all_jobs():
    import rag.runtime.jobs.drift_watcher  # noqa: F401
    import rag.runtime.jobs.nightly  # noqa: F401
    import rag.runtime.jobs.frequent  # noqa: F401
    import rag.runtime.jobs.proactive  # noqa: F401
    import rag.runtime.jobs.briefs  # noqa: F401
    import rag.runtime.jobs.housekeeping  # noqa: F401


def test_total_jobs_post_f3():
    _import_all_jobs()
    sched = Scheduler.global_instance()
    expected = {
        # F1
        "drift_watcher",
        # F2 nightly
        "auto_harvest", "whisper_vocab", "implicit_feedback",
        "online_tune", "maintenance", "calibrate",
        "identity_fingerprint_refresh", "reranker_finetune", "drafts_finetune",
        # F3.1 frequent
        "anticipate", "routing_rules", "wa_fast", "ingest_whatsapp",
        "ingest_cross_source", "mood_poll", "spotify_poll", "wa_tasks",
        "vault_image_captioner", "wa_voice_backfill", "screen_observer",
        # F3.2 proactive
        "emergent", "patterns", "archive", "distill",
        "active_learning_nudge", "active_learning_suggest_goldens",
        "brief_auto_tune",
        # F3.3 briefs + housekeeping
        "morning", "today", "digest",
        "vault_cleanup", "wake_up", "consolidate",
    }
    actual = set(sched.jobs())
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"jobs faltantes: {missing}"
    assert not extra, f"jobs inesperados: {extra}"


def test_frequent_intervals():
    _import_all_jobs()
    sched = Scheduler.global_instance()
    # Cada job interval con su cadencia esperada.
    cases = {
        "anticipate": {"minutes": 15},
        "routing_rules": {"minutes": 5},
        "wa_fast": {"minutes": 5},
        "ingest_whatsapp": {"minutes": 15},
        "ingest_cross_source": {"hours": 1},
        "mood_poll": {"minutes": 30},
        "spotify_poll": {"minutes": 5},
        "wa_tasks": {"minutes": 30},
    }
    for label, expected_args in cases.items():
        job = sched.get_job(label)
        assert job is not None, f"{label} no registrado"
        assert job.trigger_kind == "interval"
        for k, v in expected_args.items():
            assert job.trigger_args.get(k) == v, (
                f"{label}: {k}={job.trigger_args.get(k)} esperado {v}"
            )


def test_ingest_whatsapp_treats_index_lock_busy_as_skip(monkeypatch):
    import rag.runtime.jobs.frequent as frequent

    seen: dict[str, object] = {}

    def fake_run(args, *, timeout, extra_env=None, benign_failure_markers=(),
                 benign_skip_reason=""):
        seen["args"] = args
        seen["timeout"] = timeout
        seen["extra_env"] = extra_env
        seen["benign_failure_markers"] = benign_failure_markers
        seen["benign_skip_reason"] = benign_skip_reason
        return {"exit_code": 0, "skipped": True, "skip_reason": benign_skip_reason}

    monkeypatch.setattr(frequent, "_run_subprocess", fake_run)

    result = frequent.ingest_whatsapp_job()

    assert result["exit_code"] == 0
    assert result["skip_reason"] == "index_lock_busy"
    assert seen["args"][-2:] == ["--source", "whatsapp"]
    assert seen["extra_env"]["RAG_INDEX_LOCK_WAIT_SECONDS"] == "0"
    assert frequent._INDEX_LOCK_BUSY_MARKER in seen["benign_failure_markers"]
    assert seen["benign_skip_reason"] == "index_lock_busy"


def test_ingest_cross_source_does_not_wait_on_index_lock(monkeypatch):
    import rag.runtime.jobs.frequent as frequent

    seen: dict[str, object] = {}

    def fake_run(args, *, timeout, extra_env=None, **kwargs):
        seen["args"] = args
        seen["timeout"] = timeout
        seen["extra_env"] = extra_env
        return {"exit_code": 0}

    monkeypatch.setattr(frequent, "_run_subprocess", fake_run)

    result = frequent.ingest_cross_source_job()

    assert result["exit_code"] == 0
    assert seen["args"][-1] == "ingest-cross-source"
    assert seen["timeout"] == 1800
    assert seen["extra_env"]["RAG_INDEX_LOCK_WAIT_SECONDS"] == "0"
    assert seen["extra_env"]["RAG_SAFARI_BOOKMARK_LOCK_BUDGET_S"] == "60"
    assert seen["extra_env"]["RAG_SAFARI_BOOKMARK_MAX_WRITE"] == "250"


def test_wa_tasks_job_is_single_flight(monkeypatch):
    import rag.runtime.jobs.frequent as frequent

    started = threading.Event()
    release = threading.Event()
    n_calls = {"i": 0}

    def slow_run(args, *, timeout, extra_env=None, **kwargs):
        n_calls["i"] += 1
        started.set()
        assert release.wait(timeout=2)
        return {"exit_code": 0}

    monkeypatch.setattr(frequent, "_run_subprocess", slow_run)

    first = threading.Thread(target=frequent.wa_tasks_job)
    first.start()
    assert started.wait(timeout=2)

    second = frequent.wa_tasks_job()
    release.set()
    first.join(timeout=2)

    assert not first.is_alive()
    assert n_calls["i"] == 1
    assert second["skipped"] is True
    assert second["skip_reason"] == "already_running"


def test_anticipate_skips_when_swap_pressure(monkeypatch):
    import rag.runtime.jobs.frequent as frequent

    monkeypatch.delenv("RAG_ANTICIPATE_DISABLED", raising=False)
    monkeypatch.setenv("RAG_ANTICIPATE_PRESSURE_GUARD", "1")
    monkeypatch.setenv("RAG_ANTICIPATE_MAX_MEMORY_PCT", "90")
    monkeypatch.setenv("RAG_ANTICIPATE_MAX_SWAP_GB", "1.5")
    monkeypatch.setattr(frequent, "_running_process_count", lambda needle: 0)
    monkeypatch.setattr(frequent, "_runtime_pressure_snapshot", lambda: (75.0, 2.7))
    monkeypatch.setattr(
        frequent,
        "_run_guarded_subprocess",
        lambda *a, **kw: pytest.fail("anticipate should not start under swap"),
    )

    result = frequent.anticipate_job()

    assert result["skipped"] is True
    assert result["skip_reason"] == "swap_pressure"
    assert result["swap_gb"] == 2.7


def test_anticipate_allows_stale_swap_when_memory_clear(monkeypatch):
    import rag.runtime.jobs.frequent as frequent

    monkeypatch.delenv("RAG_ANTICIPATE_DISABLED", raising=False)
    monkeypatch.setenv("RAG_ANTICIPATE_PRESSURE_GUARD", "1")
    monkeypatch.setenv("RAG_ANTICIPATE_MAX_SWAP_GB", "1.5")
    monkeypatch.setattr(frequent, "_running_process_count", lambda needle: 0)
    monkeypatch.setattr(frequent, "_runtime_pressure_snapshot", lambda: (12.0, 5.1))
    monkeypatch.setattr(
        frequent,
        "_run_guarded_subprocess",
        lambda *a, **kw: {"exit_code": 0, "guarded": True},
    )

    result = frequent.anticipate_job()

    assert result == {"exit_code": 0, "guarded": True}


def test_anticipate_skips_when_existing_run(monkeypatch):
    import rag.runtime.jobs.frequent as frequent

    monkeypatch.delenv("RAG_ANTICIPATE_DISABLED", raising=False)
    monkeypatch.setattr(frequent, "_running_process_count", lambda needle: 1)
    monkeypatch.setattr(
        frequent,
        "_run_guarded_subprocess",
        lambda *a, **kw: pytest.fail("anticipate should not start twice"),
    )

    result = frequent.anticipate_job()

    assert result["skipped"] is True
    assert result["skip_reason"] == "already_running"
    assert result["running_instances"] == 1


def test_anticipate_process_match_ignores_prompt_text():
    import rag.runtime.jobs.frequent as frequent

    cmd = (
        "/opt/homebrew/bin/codex exec --color never "
        "error='/Users/fer/repos/rag/.venv/bin/rag anticipate run: swap high'"
    )

    assert frequent._command_matches_needle(
        cmd,
        frequent._ANTICIPATE_PROCESS_NEEDLE,
    ) is False


def test_anticipate_process_match_accepts_real_invocations():
    import rag.runtime.jobs.frequent as frequent

    assert frequent._command_matches_needle(
        "/Users/fer/repos/rag/.venv/bin/rag anticipate run",
        frequent._ANTICIPATE_PROCESS_NEEDLE,
    ) is True
    assert frequent._command_matches_needle(
        "/Users/fer/repos/rag/.venv/bin/python "
        "/Users/fer/repos/rag/.venv/bin/rag anticipate run",
        frequent._ANTICIPATE_PROCESS_NEEDLE,
    ) is True
    assert frequent._command_matches_needle(
        "/Users/fer/repos/rag/.venv/bin/python -m rag anticipate run",
        frequent._ANTICIPATE_PROCESS_NEEDLE,
    ) is True


def test_anticipate_uses_guarded_runner_when_pressure_clear(monkeypatch):
    import rag.runtime.jobs.frequent as frequent

    seen: dict[str, object] = {}

    def fake_guarded(args, *, timeout, extra_env=None, poll_interval=5.0):
        seen["args"] = args
        seen["timeout"] = timeout
        seen["extra_env"] = extra_env
        return {"exit_code": 0, "guarded": True}

    monkeypatch.delenv("RAG_ANTICIPATE_DISABLED", raising=False)
    monkeypatch.setenv("RAG_ANTICIPATE_TIMEOUT_S", "123")
    monkeypatch.setattr(frequent, "_running_process_count", lambda needle: 0)
    monkeypatch.setattr(frequent, "_runtime_pressure_snapshot", lambda: (10.0, 0.0))
    monkeypatch.setattr(frequent, "_run_guarded_subprocess", fake_guarded)

    result = frequent.anticipate_job()

    assert result == {"exit_code": 0, "guarded": True}
    assert seen["args"][-2:] == ["anticipate", "run"]
    assert seen["timeout"] == 123
    assert seen["extra_env"]["RAG_ANTICIPATE_PRESSURE_GUARD"] == "1"


def test_guarded_anticipate_swap_abort_is_reported_as_skip(monkeypatch, caplog):
    import rag.runtime.jobs.frequent as frequent

    class FakeProc:
        pid = 12345
        returncode = None

        def __init__(self) -> None:
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise frequent.subprocess.TimeoutExpired(["rag"], timeout)
            self.returncode = -15
            return "", ""

    monkeypatch.setenv("RAG_ANTICIPATE_ABORT_SWAP_GB", "1.5")
    monkeypatch.setattr(frequent.subprocess, "Popen", lambda *a, **kw: FakeProc())
    monkeypatch.setattr(frequent, "_process_tree_rss_gb", lambda pid: None)
    monkeypatch.setattr(frequent, "_runtime_pressure_snapshot", lambda: (75.0, 5.1))
    monkeypatch.setattr(frequent, "_terminate_process_group", lambda proc: None)

    with caplog.at_level(logging.WARNING):
        result = frequent._run_guarded_subprocess(
            ["rag", "anticipate", "run"],
            timeout=30,
            poll_interval=0.01,
        )

    assert result["exit_code"] == 0
    assert result["skipped"] is True
    assert result["skip_reason"] == "swap_pressure"
    assert result["raw_exit_code"] == -9
    assert result["pressure_reason"] == "swap 5.10GB >= 1.50GB"
    assert "guarded job exit" not in caplog.text


def test_proactive_calendar_schedules():
    _import_all_jobs()
    sched = Scheduler.global_instance()
    cases = {
        # day_of_week APScheduler: Mon=0..Sun=6.
        "emergent": {"day_of_week": 4, "hour": 10, "minute": 0},  # Vie
        "patterns": {"day_of_week": 6, "hour": 20, "minute": 0},  # Dom
        "archive": {"day": 1, "hour": 23, "minute": 0},
        "distill": {"day_of_week": 6, "hour": 22, "minute": 30},  # Dom
        "active_learning_nudge": {"day_of_week": 0, "hour": 10, "minute": 0},  # Lun
        "active_learning_suggest_goldens": {"day_of_week": 0, "hour": 11, "minute": 0},
        "brief_auto_tune": {"day_of_week": 6, "hour": 3, "minute": 0},  # Dom
    }
    for label, expected_args in cases.items():
        job = sched.get_job(label)
        assert job is not None, f"{label} no registrado"
        for k, v in expected_args.items():
            assert job.trigger_args.get(k) == v, (
                f"{label}: {k}={job.trigger_args.get(k)} esperado {v}"
            )


def test_briefs_use_mon_fri_for_weekday_briefs():
    _import_all_jobs()
    sched = Scheduler.global_instance()
    morning = sched.get_job("morning")
    today = sched.get_job("today")
    digest = sched.get_job("digest")
    assert morning is not None
    assert today is not None
    assert digest is not None
    assert morning.trigger_args.get("day_of_week") == "mon-fri"
    assert today.trigger_args.get("day_of_week") == "mon-fri"
    assert digest.trigger_args.get("day_of_week") == "sun"
    assert morning.trigger_args.get("hour") == 7
    assert today.trigger_args.get("hour") == 22
    assert digest.trigger_args.get("hour") == 22


def test_housekeeping_schedules():
    _import_all_jobs()
    sched = Scheduler.global_instance()
    cases = {
        "vault_cleanup": {"hour": 2, "minute": 0},
        "wake_up": {"hour": 4, "minute": 0},
        "consolidate": {"day_of_week": "mon", "hour": 6, "minute": 0},
    }
    for label, expected_args in cases.items():
        job = sched.get_job(label)
        assert job is not None, f"{label} no registrado"
        for k, v in expected_args.items():
            assert job.trigger_args.get(k) == v, (
                f"{label}: {k}={job.trigger_args.get(k)} esperado {v}"
            )


def test_no_duplicate_labels():
    """No debería haber dos jobs con el mismo label."""
    _import_all_jobs()
    sched = Scheduler.global_instance()
    labels = list(sched.jobs())
    assert len(labels) == len(set(labels)), "labels duplicados encontrados"
