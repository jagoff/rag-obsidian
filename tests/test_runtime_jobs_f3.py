"""Tests F3 — migración de 18 jobs adicionales al supervisor.

Cubren:
- ``frequent.py`` — 8 hot-path frecuentes registrados con schedules
  matcheando los plists viejos.
- ``proactive.py`` — 7 weekly/monthly registrados con day_of_week +
  hour + minute correctos.
- ``briefs.py`` — 3 briefs Mon-Fri / Sun.
- ``housekeeping.py`` — 3 housekeeping daily/weekly.
- Total post-F3: 28 jobs (incluyendo drift_watcher F1 + 6 nightly F2).

NO testeamos invocación real — eso es shadow A/B en producción.
"""
from __future__ import annotations

import sys

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
        # F3.1 frequent
        "anticipate", "routing_rules", "wa_fast", "ingest_whatsapp",
        "ingest_cross_source", "mood_poll", "spotify_poll", "wa_tasks",
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
