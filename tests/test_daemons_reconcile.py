"""Tests para rag daemons reconcile / doctor / retry / kickstart-overdue.

Cubre los write paths del control plane: _compute_reconcile_actions,
_execute_reconcile_action, _doctor_diagnose y los 4 subcomandos CLI.

Isolación de DB_PATH: snap+restore manual (NO monkeypatch.setattr) para
evitar el warning falso _stabilize_rag_state. Mismo patrón que
tests/test_rag_log_sql_read.py::sql_env.
"""
from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import rag


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _open_db(tmp_path: Path) -> sqlite3.Connection:
    db = tmp_path / rag._TELEMETRY_DB_FILENAME
    conn = sqlite3.connect(str(db), isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_schema_version ("
        " table_name TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0)"
    )
    rag._ensure_telemetry_tables(conn)
    return conn


@pytest.fixture
def sql_env(tmp_path):
    """Redirect DB_PATH a tmp db. Snap+restore manual para evitar warning."""
    snap_db = rag.DB_PATH
    rag.DB_PATH = tmp_path
    conn = _open_db(tmp_path)
    try:
        yield tmp_path, conn
    finally:
        conn.close()
        rag.DB_PATH = snap_db


def _fake_status(
    label: str,
    state: str = "running",
    runs: int = 1,
    last_exit: int | None = 0,
    overdue: bool = False,
    category: str = "managed",
    last_tick_iso: str | None = "2026-01-01T00:00:00",
    expected_cadence_s: int | None = 300,
) -> dict:
    return {
        "label": label,
        "category": category,
        "state": state,
        "runs": runs,
        "last_exit": last_exit,
        "last_tick_iso": last_tick_iso,
        "overdue": overdue,
        "expected_cadence_s": expected_cadence_s,
    }


# ── _compute_reconcile_actions ────────────────────────────────────────────────


def test_compute_reconcile_actions_bootstrap_missing_with_plist():
    """state=missing + plist en disco → acción bootstrap."""
    label = "com.fer.obsidian-rag-test-bootstrap"

    with (
        __import__("unittest.mock", fromlist=["patch"]).patch.object(
            rag, "_all_daemon_labels", return_value=[(label, "managed")]
        ),
        __import__("unittest.mock", fromlist=["patch"]).patch.object(
            rag, "_gather_daemon_status",
            return_value=_fake_status(label, state="missing", last_exit=None),
        ),
        __import__("unittest.mock", fromlist=["patch"]).patch.object(
            rag, "_plist_on_disk", return_value=True
        ),
    ):
        actions = rag._compute_reconcile_actions(gentle=False)

    assert len(actions) == 1
    assert actions[0]["kind"] == "bootstrap"
    assert actions[0]["label"] == label
    assert "plist on disk" in actions[0]["reason"]


def test_compute_reconcile_actions_bootout_loaded_without_plist():
    """state=running + plist NO en disco → acción bootout."""
    from unittest.mock import patch as _patch
    label = "com.fer.obsidian-rag-test-bootout"

    with (
        _patch.object(rag, "_all_daemon_labels", return_value=[(label, "managed")]),
        _patch.object(rag, "_gather_daemon_status", return_value=_fake_status(label, state="running")),
        _patch.object(rag, "_plist_on_disk", return_value=False),
    ):
        actions = rag._compute_reconcile_actions(gentle=False)

    assert len(actions) == 1
    assert actions[0]["kind"] == "bootout"
    assert "no plist on disk" in actions[0]["reason"]


def test_compute_reconcile_actions_gentle_skips_bootout():
    """gentle=True: mismo input que bootout → NO genera bootout."""
    from unittest.mock import patch as _patch
    label = "com.fer.obsidian-rag-test-gentle"

    with (
        _patch.object(rag, "_all_daemon_labels", return_value=[(label, "managed")]),
        _patch.object(rag, "_gather_daemon_status", return_value=_fake_status(label, state="running")),
        _patch.object(rag, "_plist_on_disk", return_value=False),
    ):
        actions = rag._compute_reconcile_actions(gentle=True)

    assert actions == []


def test_compute_reconcile_actions_kickstart_on_failed_exit():
    """last_exit=1 AND runs=2 → acción kickstart."""
    from unittest.mock import patch as _patch
    label = "com.fer.obsidian-rag-test-kick"

    with (
        _patch.object(rag, "_all_daemon_labels", return_value=[(label, "managed")]),
        _patch.object(
            rag, "_gather_daemon_status",
            return_value=_fake_status(label, state="running", runs=2, last_exit=1),
        ),
        _patch.object(rag, "_plist_on_disk", return_value=True),
    ):
        actions = rag._compute_reconcile_actions(gentle=False)

    assert len(actions) == 1
    assert actions[0]["kind"] == "kickstart"
    assert "last_exit=1" in actions[0]["reason"]
    assert "runs=2" in actions[0]["reason"]


def test_compute_reconcile_actions_kickstart_on_overdue():
    """overdue=True → acción kickstart."""
    from unittest.mock import patch as _patch
    label = "com.fer.obsidian-rag-test-overdue"

    with (
        _patch.object(rag, "_all_daemon_labels", return_value=[(label, "managed")]),
        _patch.object(
            rag, "_gather_daemon_status",
            return_value=_fake_status(label, state="running", overdue=True),
        ),
        _patch.object(rag, "_plist_on_disk", return_value=True),
    ):
        actions = rag._compute_reconcile_actions(gentle=False)

    assert len(actions) == 1
    assert actions[0]["kind"] == "kickstart"
    assert "overdue" in actions[0]["reason"]


# ── _execute_reconcile_action ─────────────────────────────────────────────────


def test_execute_reconcile_action_bootstrap_ok(monkeypatch):
    """kind=bootstrap exit=0 → ok=True."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)

    result = rag._execute_reconcile_action({
        "label": "com.fer.obsidian-rag-test",
        "kind": "bootstrap",
        "plist_path": Path("/tmp/fake.plist"),
    })

    assert result["ok"] is True
    assert result["exit_code"] == 0


def test_execute_reconcile_action_bootstrap_ealready(monkeypatch):
    """kind=bootstrap exit=37 (EALREADY) → ok=True."""
    mock_proc = MagicMock()
    mock_proc.returncode = 37
    mock_proc.stderr = "service already loaded"
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)

    result = rag._execute_reconcile_action({
        "label": "com.fer.obsidian-rag-test",
        "kind": "bootstrap",
        "plist_path": Path("/tmp/fake.plist"),
    })

    assert result["ok"] is True
    assert result["exit_code"] == 37


def test_execute_reconcile_action_bootout_no_exists(monkeypatch):
    """kind=bootout exit=3 (no existe = ya bootouted) → ok=True."""
    mock_proc = MagicMock()
    mock_proc.returncode = 3
    mock_proc.stderr = "Could not find service"
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)

    result = rag._execute_reconcile_action({
        "label": "com.fer.obsidian-rag-test",
        "kind": "bootout",
        "plist_path": None,
    })

    assert result["ok"] is True
    assert result["exit_code"] == 3


# ── daemons_reconcile CLI ─────────────────────────────────────────────────────


def test_daemons_reconcile_defaults_to_dry_run(sql_env):
    """Sin --apply ni --dry-run → dry-run implícito, sin subprocess, sin log."""
    from unittest.mock import patch as _patch
    runner = CliRunner()
    action = {
        "label": "com.fer.obsidian-rag-test",
        "kind": "bootstrap",
        "reason": "plist on disk, not loaded",
        "current_state": {"state": "missing", "category": "managed"},
        "plist_path": Path("/tmp/fake.plist"),
    }

    with (
        _patch.object(rag, "_compute_reconcile_actions", return_value=[action]),
        _patch.object(subprocess, "run") as mock_run,
    ):
        result = runner.invoke(rag.daemons_reconcile, [])

    assert result.exit_code == 0
    assert "dry-run" in result.output.lower()
    mock_run.assert_not_called()


def test_daemons_reconcile_apply_calls_subprocess_and_logs(sql_env):
    """--apply con 1 action → subprocess llamado + _log_daemon_run_event exactamente 1 vez."""
    from unittest.mock import patch as _patch
    runner = CliRunner()
    label = "com.fer.obsidian-rag-test-apply"
    action = {
        "label": label,
        "kind": "bootstrap",
        "reason": "plist on disk, not loaded",
        "current_state": {"state": "missing", "category": "managed"},
        "plist_path": Path("/tmp/fake.plist"),
    }

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stderr = ""

    post_status = _fake_status(label, state="running")

    log_calls: list[dict] = []

    def _capturing_log(label, action, **kwargs):
        log_calls.append({"label": label, "action": action, **kwargs})

    with (
        _patch.object(rag, "_compute_reconcile_actions", return_value=[action]),
        _patch.object(subprocess, "run", return_value=mock_proc),
        _patch.object(rag, "_gather_daemon_status", return_value=post_status),
        _patch.object(rag, "_log_daemon_run_event", side_effect=_capturing_log),
    ):
        result = runner.invoke(rag.daemons_reconcile, ["--apply"])

    assert result.exit_code == 0
    assert len(log_calls) == 1
    assert log_calls[0]["action"] == "reconcile_bootstrap"
    assert log_calls[0]["label"] == label


# ── daemons_doctor CLI ────────────────────────────────────────────────────────


def test_daemons_doctor_all_healthy():
    """Todos sanos → mensaje de OK sin diagnósticos."""
    from unittest.mock import patch as _patch
    runner = CliRunner()
    healthy = _fake_status("com.fer.obsidian-rag-web", state="running", last_exit=0, overdue=False)

    with (
        _patch.object(rag, "_all_daemon_labels", return_value=[("com.fer.obsidian-rag-web", "managed")]),
        _patch.object(rag, "_gather_daemon_status", return_value=healthy),
    ):
        result = runner.invoke(rag.daemons_doctor, [])

    assert result.exit_code == 0
    assert "✓" in result.output
    assert "sanos" in result.output


def test_daemons_doctor_safari_lock():
    """ingest-safari con last_exit=1 → diagnóstico menciona 'database lock'."""
    from unittest.mock import patch as _patch
    runner = CliRunner()
    label = "com.fer.obsidian-rag-ingest-safari"
    row = _fake_status(label, state="running", last_exit=1, overdue=False)

    with (
        _patch.object(rag, "_all_daemon_labels", return_value=[(label, "managed")]),
        _patch.object(rag, "_gather_daemon_status", return_value=row),
    ):
        result = runner.invoke(rag.daemons_doctor, [])

    assert result.exit_code == 0
    assert "database lock" in result.output
    assert "rag daemons retry" in result.output


# ── daemons_retry CLI ─────────────────────────────────────────────────────────


def test_daemons_retry_calls_kickstart_and_logs(sql_env):
    """'ingest-safari' → llama launchctl con full label y loggea action=retry."""
    from unittest.mock import patch as _patch
    runner = CliRunner()
    full_label = "com.fer.obsidian-rag-ingest-safari"

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stderr = ""

    log_calls: list[dict] = []

    def _capturing_log(label, action, **kwargs):
        log_calls.append({"label": label, "action": action, **kwargs})

    captured_cmds: list[list] = []

    def _mock_run(cmd, **kw):
        captured_cmds.append(cmd)
        return mock_proc

    with (
        _patch.object(rag, "_all_daemon_labels", return_value=[(full_label, "managed")]),
        _patch.object(rag, "_gather_daemon_status", return_value=_fake_status(full_label)),
        _patch.object(subprocess, "run", side_effect=_mock_run),
        _patch.object(rag, "_log_daemon_run_event", side_effect=_capturing_log),
    ):
        result = runner.invoke(rag.daemons_retry, ["ingest-safari"])

    assert result.exit_code == 0
    # Verificar que se llamó kickstart con el full label
    assert any(full_label in " ".join(cmd) and "kickstart" in " ".join(cmd)
               for cmd in captured_cmds)
    # Log
    assert len(log_calls) == 1
    assert log_calls[0]["action"] == "retry"
    assert log_calls[0]["label"] == full_label


def test_daemons_retry_unknown_label_raises():
    """Label que no existe en spec → click.BadParameter."""
    from unittest.mock import patch as _patch
    runner = CliRunner()

    with _patch.object(rag, "_all_daemon_labels", return_value=[("com.fer.obsidian-rag-web", "managed")]):
        result = runner.invoke(rag.daemons_retry, ["label-no-existe"])

    assert result.exit_code != 0


# ── daemons_kickstart_overdue CLI ─────────────────────────────────────────────


def test_daemons_kickstart_overdue_none():
    """0 overdue → mensaje sin acción."""
    from unittest.mock import patch as _patch
    runner = CliRunner()
    row = _fake_status("com.fer.obsidian-rag-web", overdue=False)

    with (
        _patch.object(rag, "_all_daemon_labels", return_value=[("com.fer.obsidian-rag-web", "managed")]),
        _patch.object(rag, "_gather_daemon_status", return_value=row),
    ):
        result = runner.invoke(rag.daemons_kickstart_overdue, [])

    assert result.exit_code == 0
    assert "ningún" in result.output


def test_daemons_kickstart_overdue_two(sql_env):
    """2 overdue → 2 kickstart calls + 2 log events."""
    from unittest.mock import patch as _patch
    runner = CliRunner()

    labels = [
        "com.fer.obsidian-rag-morning",
        "com.fer.obsidian-rag-today",
    ]
    row_map = {
        labels[0]: _fake_status(labels[0], overdue=True),
        labels[1]: _fake_status(labels[1], overdue=True),
    }

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stderr = ""

    log_calls: list[dict] = []

    def _capturing_log(label, action, **kwargs):
        log_calls.append({"label": label, "action": action, **kwargs})

    def _fake_gather(label, category):
        return row_map[label]

    with (
        _patch.object(rag, "_all_daemon_labels", return_value=[(l, "managed") for l in labels]),
        _patch.object(rag, "_gather_daemon_status", side_effect=_fake_gather),
        _patch.object(subprocess, "run", return_value=mock_proc),
        _patch.object(rag, "_log_daemon_run_event", side_effect=_capturing_log),
    ):
        result = runner.invoke(rag.daemons_kickstart_overdue, [])

    assert result.exit_code == 0
    assert len(log_calls) == 2
    for lc in log_calls:
        assert lc["action"] == "kickstart"
