"""Tests for `rag daemons status` — T2 del control plane de daemons launchd.

Cubre:
  1. _services_spec_manual() shape y count.
  2. No overlap entre _services_spec_manual() y _services_spec().
  3. Parser de salida de `launchctl print` — estado running + runs + exit code.
  4. Parser con stderr de "Could not find service" → state="missing".
  5. Last tick desde mtime de logfile reciente → overdue=False.
  6. Last tick de logfile hace 2 días con cadence 3600s → overdue=True.
  7. --json produce JSON parseable con keys correctas.
  8. --unhealthy-only filtra correctamente.
  9. _log_daemon_run_event se llama una vez por run.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import rag


# ── DB_PATH isolation (patrón obligatorio, ver CLAUDE.md) ─────────────────────

@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path):
    snap = rag.DB_PATH
    rag.DB_PATH = tmp_path
    try:
        yield tmp_path
    finally:
        rag.DB_PATH = snap


# ── 1. _services_spec_manual() shape y count ──────────────────────────────────

def test_services_spec_manual_returns_7_entries():
    # 2026-05-04 consolidation: 7 → 3 manual entries.
    specs = rag._services_spec_manual()
    assert len(specs) == 3


def test_services_spec_manual_shape():
    specs = rag._services_spec_manual()
    for s in specs:
        assert "label" in s, f"missing 'label' key: {s}"
        assert "category" in s, f"missing 'category' key: {s}"
        assert s["category"] == "manual_keep", f"unexpected category: {s}"
        assert s["label"].startswith("com.fer.obsidian-rag-"), f"bad label prefix: {s}"


def test_services_spec_manual_known_labels():
    # 2026-05-04 consolidation: cloudflare-tunnel*, lgbm-train, paraphrases-train
    # removidos del manual spec. Quedan 3.
    specs = rag._services_spec_manual()
    labels = {s["label"] for s in specs}
    expected = {
        "com.fer.obsidian-rag-synth-refresh",
        "com.fer.obsidian-rag-spotify-poll",
        "com.fer.obsidian-rag-log-rotate",
    }
    assert labels == expected


# ── 2. No overlap con _services_spec() ─────────────────────────────────────────

def test_no_overlap_with_services_spec():
    rag_bin = rag._rag_binary()
    managed_labels = {label for (label, _fname, _xml) in rag._services_spec(rag_bin)}
    manual_labels = {s["label"] for s in rag._services_spec_manual()}
    overlap = managed_labels & manual_labels
    assert not overlap, f"labels aparecen en ambas specs: {overlap}"


# ── 3. Parser de launchctl print — running + runs + exit ──────────────────────

def test_parse_launchctl_print_running():
    stdout = (
        "{\n"
        "    path = /Library/LaunchAgents/com.fer.obsidian-rag-web.plist\n"
        "    state = running\n"
        "    program = /Users/fer/.local/bin/rag\n"
        "    runs = 22\n"
        "    last exit code = 0\n"
        "}\n"
    )
    result = rag._parse_launchctl_print(stdout)
    assert result["state"] == "running"
    assert result["runs"] == 22
    assert result["last_exit"] == 0


def test_parse_launchctl_print_not_running():
    stdout = (
        "{\n"
        "    state = not running\n"
        "    runs = 5\n"
        "    last exit code = 1\n"
        "}\n"
    )
    result = rag._parse_launchctl_print(stdout)
    assert result["state"] == "not running"
    assert result["runs"] == 5
    assert result["last_exit"] == 1


def test_parse_launchctl_print_never_exited():
    stdout = (
        "    state = not running\n"
        "    runs = 0\n"
        "    last exit code = (never exited)\n"
    )
    result = rag._parse_launchctl_print(stdout)
    assert result["state"] == "not running"
    assert result["runs"] == 0
    assert result["last_exit"] == "(never exited)"


# ── 4. Mock subprocess con "Could not find service" → state="missing" ─────────

def test_gather_daemon_status_missing(tmp_path):
    """Daemon no bootstrappeado: subprocess retorna exit≠0 con stderr del tipo
    'Could not find service ...' → state='missing', runs y last_exit None."""
    mock_proc = MagicMock()
    mock_proc.returncode = 113
    mock_proc.stdout = ""
    mock_proc.stderr = "Could not find service com.fer.obsidian-rag-fake in domain for system"

    with patch("subprocess.run", return_value=mock_proc):
        row = rag._gather_daemon_status("com.fer.obsidian-rag-fake", "managed")

    assert row["state"] == "missing"
    assert row["runs"] is None
    assert row["last_exit"] is None
    assert row["label"] == "com.fer.obsidian-rag-fake"
    assert row["category"] == "managed"


# ── 5. Last tick reciente → overdue=False ──────────────────────────────────────

def test_overdue_false_when_recent_log(tmp_path):
    """Logfile con mtime ahora, cadencia 3600s → overdue=False."""
    log_file = tmp_path / "fake-daemon.log"
    log_file.write_text("ok\n")
    # mtime es ~ahora por defecto

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "state = running\nruns = 10\nlast exit code = 0\n"
    mock_proc.stderr = ""

    plist_content = (
        "<?xml version='1.0'?>\n"
        "<plist><dict>\n"
        "<key>Label</key><string>com.fer.obsidian-rag-fake-daemon</string>\n"
        "<key>StartInterval</key><integer>3600</integer>\n"
        "</dict></plist>\n"
    )
    plist_path = tmp_path / "com.fer.obsidian-rag-fake-daemon.plist"
    plist_path.write_text(plist_content)

    with (
        patch("subprocess.run", return_value=mock_proc),
        patch.object(rag, "_daemon_log_path", return_value=log_file),
        patch.object(rag, "_plist_cadence_seconds", return_value=3600),
    ):
        row = rag._gather_daemon_status("com.fer.obsidian-rag-fake-daemon", "managed")

    assert row["overdue"] is False
    assert row["last_tick_iso"] is not None


# ── 6. Last tick hace 2 días con cadence 3600s → overdue=True ─────────────────

def test_overdue_true_when_stale_log(tmp_path):
    """Logfile con mtime hace 2 días, cadencia 3600s → overdue=True."""
    log_file = tmp_path / "fake-daemon.log"
    log_file.write_text("old\n")
    # Forzar mtime a hace 2 días
    stale_ts = (datetime.now() - timedelta(days=2)).timestamp()
    os.utime(str(log_file), (stale_ts, stale_ts))

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "state = not running\nruns = 1\nlast exit code = 0\n"
    mock_proc.stderr = ""

    with (
        patch("subprocess.run", return_value=mock_proc),
        patch.object(rag, "_daemon_log_path", return_value=log_file),
        patch.object(rag, "_plist_cadence_seconds", return_value=3600),
    ):
        row = rag._gather_daemon_status("com.fer.obsidian-rag-fake-daemon", "managed")

    assert row["overdue"] is True


# ── 7. --json produce JSON parseable con keys correctas ───────────────────────

def test_daemons_status_json_output():
    """Invocación con --json debe emitir JSON válido con las keys esperadas."""
    from click.testing import CliRunner

    # Stub mínimo para no invocar launchctl real
    fake_row = {
        "label": "com.fer.obsidian-rag-web",
        "category": "managed",
        "state": "running",
        "runs": 5,
        "last_exit": 0,
        "last_tick_iso": "2026-05-01T08:00:00",
        "overdue": False,
        "expected_cadence_s": None,
    }

    with (
        patch.object(rag, "_gather_daemon_status", return_value=fake_row),
        patch.object(rag, "_services_spec", return_value=[
            ("com.fer.obsidian-rag-web", "web.plist", "<plist/>"),
        ]),
        patch.object(rag, "_services_spec_manual", return_value=[]),
        patch.object(rag, "_log_daemon_run_event"),
    ):
        runner = CliRunner()
        result = runner.invoke(rag.daemons_status, ["--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 1
    keys = set(data[0].keys())
    expected_keys = {
        "label", "category", "state", "runs",
        "last_exit", "last_tick_iso", "overdue", "expected_cadence_s",
    }
    assert expected_keys <= keys


# ── 8. --unhealthy-only filtra correctamente ──────────────────────────────────

def test_unhealthy_only_filters():
    """Un daemon healthy + uno overdue → solo el overdue en output."""
    from click.testing import CliRunner

    healthy = {
        "label": "com.fer.obsidian-rag-watch",
        "category": "managed",
        "state": "running",
        "runs": 100,
        "last_exit": 0,
        "last_tick_iso": "2026-05-01T10:00:00",
        "overdue": False,
        "expected_cadence_s": 30,
    }
    overdue = {
        "label": "com.fer.obsidian-rag-calibrate",
        "category": "managed",
        "state": "not running",
        "runs": 0,
        "last_exit": "(never exited)",
        "last_tick_iso": None,
        "overdue": True,
        "expected_cadence_s": 3600,
    }

    call_count = [0]
    labels_called = []

    def fake_gather(label, category):
        call_count[0] += 1
        labels_called.append(label)
        if "watch" in label:
            return healthy
        return overdue

    with (
        patch.object(rag, "_gather_daemon_status", side_effect=fake_gather),
        patch.object(rag, "_services_spec", return_value=[
            ("com.fer.obsidian-rag-watch", "watch.plist", "<plist/>"),
            ("com.fer.obsidian-rag-calibrate", "calibrate.plist", "<plist/>"),
        ]),
        patch.object(rag, "_services_spec_manual", return_value=[]),
        patch.object(rag, "_log_daemon_run_event"),
    ):
        runner = CliRunner()
        result = runner.invoke(rag.daemons_status, ["--json", "--unhealthy-only"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["label"] == "com.fer.obsidian-rag-calibrate"


# ── 9. _log_daemon_run_event se llama una vez por run ─────────────────────────

def test_log_daemon_run_event_called_once():
    """Independientemente del número de daemons, _log_daemon_run_event
    debe llamarse exactamente UNA vez por invocación del comando status."""
    from click.testing import CliRunner

    fake_row = {
        "label": "com.fer.obsidian-rag-web",
        "category": "managed",
        "state": "running",
        "runs": 3,
        "last_exit": 0,
        "last_tick_iso": None,
        "overdue": False,
        "expected_cadence_s": None,
    }

    log_calls = []

    def fake_log(**kwargs):
        log_calls.append(kwargs)

    with (
        patch.object(rag, "_gather_daemon_status", return_value=fake_row),
        patch.object(rag, "_services_spec", return_value=[
            ("com.fer.obsidian-rag-web", "web.plist", "<plist/>"),
            ("com.fer.obsidian-rag-morning", "morning.plist", "<plist/>"),
        ]),
        patch.object(rag, "_services_spec_manual", return_value=[]),
        patch.object(rag, "_log_daemon_run_event", side_effect=fake_log),
    ):
        runner = CliRunner()
        result = runner.invoke(rag.daemons_status, ["--json"])

    assert result.exit_code == 0, result.output
    assert len(log_calls) == 1, f"se esperaba 1 call, got {len(log_calls)}: {log_calls}"
    assert log_calls[0]["action"] == "status_check"
    assert log_calls[0]["label"] == "<status_run>"
