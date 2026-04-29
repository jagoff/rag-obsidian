"""Tests for rag/scheduler.py — unified in-process scheduler.

Tests use mocks so no APScheduler daemon or real subprocess is started.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_yaml(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.dump(content, fh)


# ── yaml parsing ─────────────────────────────────────────────────────────────

class TestLoadSchedulerConfig:
    def test_returns_default_when_file_missing(self, tmp_path):
        from rag.scheduler import load_scheduler_config, _DEFAULT_JOBS
        missing = tmp_path / "nope.yaml"
        result = load_scheduler_config(missing)
        assert "morning" in result
        assert len(result) == len(_DEFAULT_JOBS)

    def test_parses_cron_job(self, tmp_path):
        cfg = tmp_path / "sched.yaml"
        _write_yaml(cfg, {"jobs": {
            "morning": {"cron": "0 7 * * 1-5", "command": "rag morning", "enabled": True}
        }})
        from rag.scheduler import load_scheduler_config
        jobs = load_scheduler_config(cfg)
        assert "morning" in jobs
        assert jobs["morning"]["cron"] == "0 7 * * 1-5"
        assert jobs["morning"]["command"] == "rag morning"

    def test_parses_interval_job(self, tmp_path):
        cfg = tmp_path / "sched.yaml"
        _write_yaml(cfg, {"jobs": {
            "anticipate": {
                "interval_minutes": 10,
                "command": "rag anticipate run",
                "enabled": True,
            }
        }})
        from rag.scheduler import load_scheduler_config
        jobs = load_scheduler_config(cfg)
        assert jobs["anticipate"]["interval_minutes"] == 10

    def test_skips_disabled_jobs(self, tmp_path):
        cfg = tmp_path / "sched.yaml"
        _write_yaml(cfg, {"jobs": {
            "morning": {"cron": "0 7 * * 1-5", "command": "rag morning", "enabled": False},
            "active": {"cron": "0 8 * * *", "command": "rag other", "enabled": True},
        }})
        from rag.scheduler import load_scheduler_config
        jobs = load_scheduler_config(cfg)
        assert "morning" not in jobs
        assert "active" in jobs

    def test_skips_job_without_command(self, tmp_path):
        cfg = tmp_path / "sched.yaml"
        _write_yaml(cfg, {"jobs": {
            "broken": {"cron": "0 7 * * *", "enabled": True},
        }})
        from rag.scheduler import load_scheduler_config
        jobs = load_scheduler_config(cfg)
        assert "broken" not in jobs

    def test_skips_job_without_schedule(self, tmp_path):
        cfg = tmp_path / "sched.yaml"
        _write_yaml(cfg, {"jobs": {
            "broken": {"command": "rag foo", "enabled": True},
        }})
        from rag.scheduler import load_scheduler_config
        jobs = load_scheduler_config(cfg)
        assert "broken" not in jobs

    def test_skips_invalid_cfg_type(self, tmp_path):
        cfg = tmp_path / "sched.yaml"
        _write_yaml(cfg, {"jobs": {"bad": "not a dict"}})
        from rag.scheduler import load_scheduler_config
        jobs = load_scheduler_config(cfg)
        assert "bad" not in jobs

    def test_default_jobs_all_have_cron_or_interval(self):
        from rag.scheduler import _DEFAULT_JOBS
        for name, cfg in _DEFAULT_JOBS.items():
            assert "cron" in cfg or "interval_minutes" in cfg, (
                f"job {name!r} has neither cron nor interval_minutes"
            )

    def test_default_jobs_all_have_command(self):
        from rag.scheduler import _DEFAULT_JOBS
        for name, cfg in _DEFAULT_JOBS.items():
            assert "command" in cfg, f"job {name!r} missing command"


# ── job registration ──────────────────────────────────────────────────────────

class TestBuildScheduler:
    def test_registers_cron_job(self, tmp_path):
        from rag.scheduler import build_scheduler
        jobs = {
            "morning": {
                "cron": "0 7 * * 1-5",
                "command": "rag morning",
            }
        }
        sched = build_scheduler(jobs, tmp_path)
        sched.start()
        try:
            job_ids = [j.id for j in sched.get_jobs()]
            assert "morning" in job_ids
        finally:
            sched.shutdown(wait=False)

    def test_registers_interval_job(self, tmp_path):
        from rag.scheduler import build_scheduler
        jobs = {
            "anticipate": {
                "interval_minutes": 10,
                "command": "rag anticipate run",
            }
        }
        sched = build_scheduler(jobs, tmp_path)
        sched.start()
        try:
            job_ids = [j.id for j in sched.get_jobs()]
            assert "anticipate" in job_ids
        finally:
            sched.shutdown(wait=False)

    def test_registers_all_default_jobs(self, tmp_path):
        from rag.scheduler import build_scheduler, _DEFAULT_JOBS
        sched = build_scheduler(_DEFAULT_JOBS, tmp_path)
        sched.start()
        try:
            registered = {j.id for j in sched.get_jobs()}
            for name in _DEFAULT_JOBS:
                assert name in registered, f"job {name!r} not registered"
        finally:
            sched.shutdown(wait=False)


# ── telemetry logging ─────────────────────────────────────────────────────────

class TestLogRun:
    def test_exit_code_zero_logged(self, tmp_path):
        from rag.scheduler import _ensure_scheduler_table, _log_run, read_last_runs
        _ensure_scheduler_table(tmp_path)
        _log_run(tmp_path, "morning", "2026-04-29T07:00:00+00:00",
                 "2026-04-29T07:01:00+00:00", 0, 60.0, None)
        rows = read_last_runs(tmp_path, n=10)
        assert len(rows) == 1
        assert rows[0]["job_name"] == "morning"
        assert rows[0]["exit_code"] == 0
        assert rows[0]["error_msg"] is None

    def test_nonzero_exit_code_logged(self, tmp_path):
        from rag.scheduler import _ensure_scheduler_table, _log_run, read_last_runs
        _ensure_scheduler_table(tmp_path)
        _log_run(tmp_path, "ingest-gmail", "2026-04-29T08:00:00+00:00",
                 "2026-04-29T08:00:30+00:00", 1, 30.0, "exit_code=1")
        rows = read_last_runs(tmp_path, n=10)
        assert rows[0]["exit_code"] == 1
        assert rows[0]["error_msg"] == "exit_code=1"

    def test_multiple_jobs_logged(self, tmp_path):
        from rag.scheduler import _ensure_scheduler_table, _log_run, read_last_runs
        _ensure_scheduler_table(tmp_path)
        _log_run(tmp_path, "morning", "2026-04-29T07:00:00+00:00",
                 "2026-04-29T07:01:00+00:00", 0, 60.0, None)
        _log_run(tmp_path, "anticipate", "2026-04-29T07:10:00+00:00",
                 "2026-04-29T07:10:05+00:00", 0, 5.0, None)
        rows = read_last_runs(tmp_path, n=10)
        assert len(rows) == 2
        names = {r["job_name"] for r in rows}
        assert names == {"morning", "anticipate"}


# ── _run_job ─────────────────────────────────────────────────────────────────

class TestRunJob:
    def test_successful_job_logs_exit_zero(self, tmp_path):
        from rag.scheduler import _ensure_scheduler_table, _run_job, read_last_runs
        _ensure_scheduler_table(tmp_path)
        job_cfg = {"command": "true", "timeout_minutes": 1}
        _run_job("test-ok", job_cfg, tmp_path)
        rows = read_last_runs(tmp_path, n=10)
        assert len(rows) == 1
        assert rows[0]["exit_code"] == 0

    def test_failing_job_logs_nonzero_exit(self, tmp_path):
        from rag.scheduler import _ensure_scheduler_table, _run_job, read_last_runs
        _ensure_scheduler_table(tmp_path)
        job_cfg = {"command": "false", "timeout_minutes": 1}
        _run_job("test-fail", job_cfg, tmp_path)
        rows = read_last_runs(tmp_path, n=10)
        assert rows[0]["exit_code"] != 0

    def test_timeout_logs_minus_two(self, tmp_path):
        from rag.scheduler import _ensure_scheduler_table, _run_job, read_last_runs
        _ensure_scheduler_table(tmp_path)
        # sleep 10 will timeout with a 0.01min budget
        job_cfg = {"command": "sleep 10", "timeout_minutes": 0.001}
        _run_job("test-timeout", job_cfg, tmp_path)
        rows = read_last_runs(tmp_path, n=10)
        assert rows[0]["exit_code"] == -2
        assert rows[0]["error_msg"] is not None

    def test_shell_command_works(self, tmp_path):
        from rag.scheduler import _ensure_scheduler_table, _run_job, read_last_runs
        _ensure_scheduler_table(tmp_path)
        job_cfg = {
            "command": "echo hi && true",
            "timeout_minutes": 1,
            "shell": True,
        }
        _run_job("test-shell", job_cfg, tmp_path)
        rows = read_last_runs(tmp_path, n=10)
        assert rows[0]["exit_code"] == 0


# ── _ensure_scheduler_table ───────────────────────────────────────────────────

class TestEnsureSchedulerTable:
    def test_creates_table_and_indexes(self, tmp_path):
        from rag.scheduler import _ensure_scheduler_table
        _ensure_scheduler_table(tmp_path)
        conn = sqlite3.connect(str(tmp_path / "telemetry.db"))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "rag_scheduler_runs" in tables
        indexes = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()}
        assert "ix_rag_scheduler_runs_ts" in indexes
        assert "ix_rag_scheduler_runs_job" in indexes
        conn.close()

    def test_idempotent(self, tmp_path):
        from rag.scheduler import _ensure_scheduler_table
        _ensure_scheduler_table(tmp_path)
        _ensure_scheduler_table(tmp_path)  # second call must not raise


# ── plist factory ─────────────────────────────────────────────────────────────

class TestSchedulerPlist:
    def test_plist_contains_rag_bin(self):
        from rag.scheduler import scheduler_plist
        xml = scheduler_plist("/usr/local/bin/rag")
        assert "/usr/local/bin/rag" in xml

    def test_plist_has_keepalive_true(self):
        from rag.scheduler import scheduler_plist
        xml = scheduler_plist("/usr/local/bin/rag")
        assert "<true/>" in xml
        assert "KeepAlive" in xml

    def test_plist_has_scheduler_command(self):
        from rag.scheduler import scheduler_plist
        xml = scheduler_plist("/usr/local/bin/rag")
        assert "<string>scheduler</string>" in xml

    def test_plist_has_rag_use_scheduler_env(self):
        from rag.scheduler import scheduler_plist
        xml = scheduler_plist("/usr/local/bin/rag")
        assert "RAG_USE_SCHEDULER" in xml

    def test_plist_registered_in_services_spec_when_flag_set(self):
        import os
        import rag as rag_module
        env_bak = os.environ.get("RAG_USE_SCHEDULER")
        os.environ["RAG_USE_SCHEDULER"] = "1"
        try:
            specs = rag_module._services_spec("/usr/local/bin/rag")
            labels = {s[0] for s in specs}
            assert "com.fer.obsidian-rag-scheduler" in labels
        finally:
            if env_bak is None:
                os.environ.pop("RAG_USE_SCHEDULER", None)
            else:
                os.environ["RAG_USE_SCHEDULER"] = env_bak


# ── rag_scheduler_runs in _TELEMETRY_DDL ─────────────────────────────────────

class TestSchedulerDDL:
    def test_table_in_telemetry_ddl(self):
        import rag
        table_names = {name for name, _ in rag._TELEMETRY_DDL}
        assert "rag_scheduler_runs" in table_names

    def test_table_created_by_ensure_telemetry_tables(self, tmp_path):
        import rag
        import sqlite3
        db_file = tmp_path / "telemetry.db"
        conn = sqlite3.connect(str(db_file))
        conn.execute("PRAGMA journal_mode=WAL")
        rag._ensure_telemetry_tables(conn)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "rag_scheduler_runs" in tables
        conn.close()


# ── write_default_scheduler_yaml ─────────────────────────────────────────────

class TestWriteDefaultSchedulerYaml:
    def test_creates_file(self, tmp_path):
        from rag.scheduler import _DEFAULT_JOBS
        import rag.scheduler as sched_mod
        target = tmp_path / "sched.yaml"
        orig = sched_mod._SCHEDULER_YAML
        sched_mod._SCHEDULER_YAML = target
        sched_mod._CONFIG_DIR = tmp_path
        try:
            sched_mod._write_default_scheduler_yaml()
            assert target.exists()
            with open(target) as fh:
                parsed = yaml.safe_load(fh)
            assert "jobs" in parsed
            assert "morning" in parsed["jobs"]
        finally:
            sched_mod._SCHEDULER_YAML = orig

    def test_does_not_overwrite_existing(self, tmp_path):
        import rag.scheduler as sched_mod
        target = tmp_path / "sched.yaml"
        target.write_text("existing: content\n")
        orig_yaml = sched_mod._SCHEDULER_YAML
        orig_dir = sched_mod._CONFIG_DIR
        sched_mod._SCHEDULER_YAML = target
        sched_mod._CONFIG_DIR = tmp_path
        try:
            sched_mod._write_default_scheduler_yaml()
            # Should not overwrite
            assert target.read_text() == "existing: content\n"
        finally:
            sched_mod._SCHEDULER_YAML = orig_yaml
            sched_mod._CONFIG_DIR = orig_dir
