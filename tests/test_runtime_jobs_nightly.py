"""Tests F2 — nightly batch jobs migrados al supervisor.

Cubren:
- Los 6 jobs se registran al import.
- Cada job tiene el cron schedule esperado (matchea el plist viejo).
- ``_run_subprocess`` captura stdout/stderr en formato esperable por
  ``rag_supervisor_jobs.signals``.
- ``implicit_feedback`` corre los 3 sub-jobs en serie con worst exit_code.
- Timeout config (online_tune más generoso que default).

NO testeamos la invocación real del binario ``rag`` — eso es shadow A/B
en producción. Acá solo validamos el wiring + el wrapper subprocess.
"""
from __future__ import annotations

import sys

import pytest

from rag.runtime.scheduler import Scheduler


@pytest.fixture(autouse=True)
def _reset():
    Scheduler.reset_global()
    sys.modules.pop("rag.runtime.jobs.nightly", None)
    yield
    Scheduler.reset_global()
    sys.modules.pop("rag.runtime.jobs.nightly", None)


def _import_nightly():
    import rag.runtime.jobs.nightly as mod
    return mod


def test_six_jobs_registered():
    _import_nightly()
    sched = Scheduler.global_instance()
    expected = {
        "auto_harvest",
        "whisper_vocab",
        "implicit_feedback",
        "online_tune",
        "maintenance",
        "calibrate",
        # 2026-05-10: weekly reranker LoRA fine-tune (Sun 04:30).
        "reranker_finetune",
        # 2026-05-11: weekly drafts DPO+LoRA fine-tune (Sun 04:00).
        "drafts_finetune",
    }
    actual = set(sched.jobs())
    assert expected.issubset(actual), (
        f"missing jobs: {expected - actual}"
    )


def test_schedules_match_plists():
    """Cada job tiene cron schedule equivalente al plist viejo."""
    _import_nightly()
    sched = Scheduler.global_instance()
    expected_schedules = {
        "auto_harvest": {"hour": 3, "minute": 0},
        "whisper_vocab": {"hour": 3, "minute": 15},
        "implicit_feedback": {"hour": 3, "minute": 25},
        "online_tune": {"hour": 3, "minute": 30},
        "maintenance": {"hour": 4, "minute": 0},
        "drafts_finetune": {"hour": 4, "minute": 0, "day_of_week": 6},
        "reranker_finetune": {"hour": 4, "minute": 30, "day_of_week": 6},
        "calibrate": {"hour": 5, "minute": 0},
    }
    for label, expected_args in expected_schedules.items():
        job = sched.get_job(label)
        assert job is not None, f"job {label} no registrado"
        assert job.trigger_kind == "cron"
        for k, v in expected_args.items():
            assert job.trigger_args.get(k) == v, (
                f"{label}: trigger_args[{k}]={job.trigger_args.get(k)} "
                f"esperado {v}"
            )


def test_run_subprocess_captures_signals(monkeypatch):
    """``_run_subprocess`` retorna dict con stdout_lines/stderr_lines."""
    mod = _import_nightly()
    import subprocess as sp

    class _Result:
        def __init__(self):
            self.returncode = 0
            self.stdout = "line1\nline2\nline3\n"
            self.stderr = ""

    monkeypatch.setattr(sp, "run", lambda *a, **kw: _Result())
    out = mod._run_subprocess(["echo", "test"])
    assert out["exit_code"] == 0
    assert out["stdout_lines"] == 3
    assert out["stderr_lines"] == 0
    assert out["last_stderr"] is None


def test_run_subprocess_captures_failure(monkeypatch):
    mod = _import_nightly()
    import subprocess as sp

    class _Result:
        def __init__(self):
            self.returncode = 2
            self.stdout = ""
            self.stderr = "Error: db locked\nstack...\nfinal: aborted\n"

    monkeypatch.setattr(sp, "run", lambda *a, **kw: _Result())
    out = mod._run_subprocess(["false"])
    assert out["exit_code"] == 2
    assert out["last_stderr"] is not None
    assert "aborted" in out["last_stderr"]


def test_run_subprocess_keeps_stdout_tail_on_failure(monkeypatch):
    mod = _import_nightly()
    import subprocess as sp

    class _Result:
        def __init__(self):
            self.returncode = 1
            self.stdout = "first line\nOtro `rag index` ya está activo\n"
            self.stderr = ""

    monkeypatch.setattr(sp, "run", lambda *a, **kw: _Result())
    out = mod._run_subprocess(["rag", "index"])
    assert out["exit_code"] == 1
    assert out["last_stdout"] is not None
    assert "rag index" in out["last_stdout"]


def test_run_subprocess_can_treat_known_failure_as_skip(monkeypatch):
    mod = _import_nightly()
    import subprocess as sp

    class _Result:
        def __init__(self):
            self.returncode = 1
            self.stdout = "Otro `rag index` ya está activo\n"
            self.stderr = ""

    monkeypatch.setattr(sp, "run", lambda *a, **kw: _Result())
    out = mod._run_subprocess(
        ["rag", "index"],
        benign_failure_markers=("Otro `rag index` ya está activo",),
        benign_skip_reason="index_lock_busy",
    )
    assert out["exit_code"] == 0
    assert out["raw_exit_code"] == 1
    assert out["skipped"] is True
    assert out["skip_reason"] == "index_lock_busy"


def test_run_subprocess_handles_timeout(monkeypatch):
    mod = _import_nightly()
    import subprocess as sp

    def _raise_timeout(*args, **kwargs):
        raise sp.TimeoutExpired(cmd=args[0], timeout=10)

    monkeypatch.setattr(sp, "run", _raise_timeout)
    out = mod._run_subprocess(["sleep", "999"], timeout=10)
    assert out["exit_code"] == -1
    assert "timeout" in (out["last_stderr"] or "")


def test_implicit_feedback_runs_three_subs_in_series(monkeypatch):
    mod = _import_nightly()
    calls = []

    def _fake_run(args, *, timeout=None, extra_env=None):
        calls.append(list(args))
        return {
            "exit_code": 0,
            "stdout_lines": 1,
            "stderr_lines": 0,
            "last_stderr": None,
        }

    monkeypatch.setattr(mod, "_run_subprocess", _fake_run)
    result = mod.implicit_feedback_job()
    # detect-requery debe correr antes de infer-implicit porque escribe
    # follow_up_query, consumido por corrective_paths.
    assert len(calls) == 3
    subs = [c[2] for c in calls]  # rag_bin, "feedback", <sub>
    assert subs == ["detect-requery", "infer-implicit", "classify-sessions"]
    assert result["exit_code"] == 0
    assert result["n_subs_ok"] == 3


def test_implicit_feedback_worst_exit_propagates(monkeypatch):
    mod = _import_nightly()
    n = {"i": 0}

    def _fake_run(args, *, timeout=None, extra_env=None):
        n["i"] += 1
        return {
            "exit_code": 0 if n["i"] != 2 else 5,
            "stdout_lines": 0,
            "stderr_lines": 0,
            "last_stderr": None,
        }

    monkeypatch.setattr(mod, "_run_subprocess", _fake_run)
    result = mod.implicit_feedback_job()
    # 3 subs corrieron igual aunque el 2do falló — los 3 son idempotentes.
    assert n["i"] == 3
    assert result["exit_code"] == 5
    assert result["n_subs_ok"] == 2


def test_online_tune_uses_extended_timeout(monkeypatch):
    mod = _import_nightly()
    captured = {}

    def _fake_run(args, *, timeout=None, extra_env=None):
        captured["timeout"] = timeout
        captured["args"] = args
        return {"exit_code": 0, "stdout_lines": 0, "stderr_lines": 0,
                "last_stderr": None}

    monkeypatch.setattr(mod, "_run_subprocess", _fake_run)
    mod.online_tune_job()
    # online-tune tarda 24min warm en M-chip, le dimos 45min de timeout.
    assert captured["timeout"] == 2700
    assert "tune" in captured["args"]
    assert "--online" in captured["args"]


def test_drafts_finetune_gated_off(monkeypatch):
    """RAG_AUTO_FINETUNE_DRAFTS=0 → exit_code 0 sin tocar nada."""
    mod = _import_nightly()
    monkeypatch.setenv("RAG_AUTO_FINETUNE_DRAFTS", "0")
    # Si llamara subprocess o SQL, este fake explotaría.
    monkeypatch.setattr(
        mod, "_run_subprocess",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no debería ejecutarse")),
    )
    result = mod.drafts_finetune_job()
    assert result["exit_code"] == 0
    assert result["phase"] == "gated_off"


def test_drafts_finetune_skip_insufficient_signal(monkeypatch):
    """N pares < threshold → skip silent."""
    mod = _import_nightly()
    monkeypatch.setenv("RAG_AUTO_FINETUNE_DRAFTS", "1")
    monkeypatch.setenv("RAG_DRAFTS_FINETUNE_MIN_PAIRS", "100")

    # Fake augment OK + count = 3.
    monkeypatch.setattr(
        mod, "_run_subprocess",
        lambda *a, **kw: {"exit_code": 0, "stdout_lines": 0, "stderr_lines": 0,
                          "last_stderr": None},
    )
    # Fake SQL count → 3 rows.
    import contextlib

    class _FakeConn:
        def execute(self, *a, **kw):
            class _Cursor:
                def fetchone(self_inner):
                    return (3,)
            return _Cursor()

    @contextlib.contextmanager
    def _fake_state_conn():
        yield _FakeConn()

    import rag
    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_state_conn)
    result = mod.drafts_finetune_job()
    assert result["exit_code"] == 0
    assert result["phase"] == "skip_insufficient_signal"
    assert result["n_pairs"] == 3
    assert result["min_pairs"] == 100


def test_drafts_finetune_triggers_when_enough(monkeypatch):
    """N pares ≥ threshold → augment + finetune subprocess corren."""
    mod = _import_nightly()
    monkeypatch.setenv("RAG_AUTO_FINETUNE_DRAFTS", "1")
    monkeypatch.setenv("RAG_DRAFTS_FINETUNE_MIN_PAIRS", "10")

    calls: list[list[str]] = []

    def _fake_run(args, *, timeout=None, extra_env=None):
        calls.append(args)
        return {"exit_code": 0, "stdout_lines": 5, "stderr_lines": 0,
                "last_stderr": None}

    monkeypatch.setattr(mod, "_run_subprocess", _fake_run)

    import contextlib

    class _FakeConn:
        def execute(self, *a, **kw):
            class _Cursor:
                def fetchone(self_inner):
                    return (42,)
            return _Cursor()

    @contextlib.contextmanager
    def _fake_state_conn():
        yield _FakeConn()

    import rag
    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_state_conn)
    result = mod.drafts_finetune_job()
    assert result["exit_code"] == 0
    assert result["phase"] == "trained"
    assert result["n_pairs"] == 42
    # augment + finetune corren (en ese orden).
    assert len(calls) == 2
    assert any("augment_drafts_dataset.py" in arg for arg in calls[0])
    assert any("finetune_drafts.py" in arg for arg in calls[1])
