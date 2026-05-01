"""Tests para `scripts/wake_hook.py` — el sidecar daemon que detecta
wakes user-visible y dispara `rag daemons kickstart-overdue`.

No corremos pmset real ni invocamos `rag daemons` real — todo mockeado.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Importar el módulo con un path sys.path hack (script files sin __init__.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from scripts import wake_hook  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Redirige STATE_PATH a tmp_path para no chocar con el real."""
    monkeypatch.setattr(wake_hook, "STATE_PATH", tmp_path / "wake-state.json")
    yield


def _mock_pmset_output(lines: list[str]) -> MagicMock:
    """Build a fake subprocess.run result."""
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = "\n".join(lines)
    proc.stderr = ""
    return proc


# ── _last_display_on ──────────────────────────────────────────────────────

def test_last_display_on_extracts_last_match():
    """Extrae la fecha de la ÚLTIMA "Display is turned on" del log."""
    out = _mock_pmset_output([
        "2026-04-30 18:00:00 -0300 Notification        \tDisplay is turned on",
        "2026-05-01 09:53:25 -0300 Notification        \tDisplay is turned on",
        "2026-05-01 14:00:00 -0300 Other line that doesnt match",
    ])
    with patch.object(wake_hook.subprocess, "run", return_value=out):
        result = wake_hook._last_display_on()
    assert result == "2026-05-01 09:53:25 -0300"


def test_last_display_on_returns_none_when_no_match():
    """Si no hay "Display is turned on" → None."""
    out = _mock_pmset_output(["random line", "another random"])
    with patch.object(wake_hook.subprocess, "run", return_value=out):
        result = wake_hook._last_display_on()
    assert result is None


def test_last_display_on_silent_fail_on_pmset_error():
    """pmset returncode≠0 → None, no raisea."""
    proc = MagicMock(returncode=1, stdout="", stderr="permission denied")
    with patch.object(wake_hook.subprocess, "run", return_value=proc):
        result = wake_hook._last_display_on()
    assert result is None


def test_last_display_on_silent_fail_on_timeout():
    """subprocess.TimeoutExpired → None, no raisea."""
    import subprocess
    with patch.object(
        wake_hook.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(cmd="pmset", timeout=15),
    ):
        result = wake_hook._last_display_on()
    assert result is None


# ── state file ─────────────────────────────────────────────────────────────

def test_read_state_returns_empty_dict_when_missing():
    assert wake_hook._read_state() == {}


def test_read_state_handles_corrupt_json():
    """JSON corrupto → dict vacío."""
    wake_hook.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    wake_hook.STATE_PATH.write_text("not json {")
    assert wake_hook._read_state() == {}


def test_write_then_read_state_roundtrip():
    wake_hook._write_state({"last_wake": "2026-05-01 09:53:25 -0300"})
    assert wake_hook._read_state() == {"last_wake": "2026-05-01 09:53:25 -0300"}


# ── check_once ─────────────────────────────────────────────────────────────

def test_check_once_no_fire_when_state_matches():
    """Si el last_wake del state == el actual → no dispara kickstart."""
    wake_hook._write_state({"last_wake": "2026-05-01 09:53:25 -0300"})
    out = _mock_pmset_output([
        "2026-05-01 09:53:25 -0300 Notification        \tDisplay is turned on",
    ])
    kick_mock = MagicMock(return_value=True)
    with patch.object(wake_hook.subprocess, "run", return_value=out), \
         patch.object(wake_hook, "_kickstart_overdue", kick_mock):
        fired = wake_hook.check_once()
    assert fired is False
    kick_mock.assert_not_called()


def test_check_once_fires_on_new_wake():
    """Wake nuevo (>=last_wake del state) → dispara kickstart + actualiza state."""
    wake_hook._write_state({"last_wake": "2026-04-30 18:00:00 -0300"})
    out = _mock_pmset_output([
        "2026-05-01 09:53:25 -0300 Notification        \tDisplay is turned on",
    ])
    kick_mock = MagicMock(return_value=True)
    with patch.object(wake_hook.subprocess, "run", return_value=out), \
         patch.object(wake_hook, "_kickstart_overdue", kick_mock):
        fired = wake_hook.check_once()
    assert fired is True
    kick_mock.assert_called_once()
    state = wake_hook._read_state()
    assert state["last_wake"] == "2026-05-01 09:53:25 -0300"
    assert state["last_kickstart_ok"] is True


def test_check_once_no_fire_when_no_pmset_data():
    """pmset no devuelve "Display is turned on" → skip silencioso."""
    out = _mock_pmset_output(["unrelated line"])
    kick_mock = MagicMock()
    with patch.object(wake_hook.subprocess, "run", return_value=out), \
         patch.object(wake_hook, "_kickstart_overdue", kick_mock):
        fired = wake_hook.check_once()
    assert fired is False
    kick_mock.assert_not_called()


def test_check_once_persists_state_even_if_kickstart_fails():
    """Si kickstart falla, state.last_wake igual se actualiza para no
    re-disparar el mismo wake en el próximo tick (idempotency over success)."""
    wake_hook._write_state({"last_wake": "2026-04-30 18:00:00 -0300"})
    out = _mock_pmset_output([
        "2026-05-01 09:53:25 -0300 Notification        \tDisplay is turned on",
    ])
    kick_mock = MagicMock(return_value=False)  # simulate failure
    with patch.object(wake_hook.subprocess, "run", return_value=out), \
         patch.object(wake_hook, "_kickstart_overdue", kick_mock):
        fired = wake_hook.check_once()
    assert fired is False  # devuelve False porque kickstart falló
    state = wake_hook._read_state()
    assert state["last_wake"] == "2026-05-01 09:53:25 -0300"  # pero state actualizado
    assert state["last_kickstart_ok"] is False
