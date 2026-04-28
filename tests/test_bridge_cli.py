"""Tests for `rag bridge` CLI group (Bug 5 — auto-recovery + status).

Covers:
- `rag bridge --help` doesn't crash + lists subcommands.
- `_wa_bridge_http_probe()` interprets `/api/health` JSON responses
  correctly: connected, not_paired, socket-dropped, 404 (old binary),
  unreachable.
- `rag bridge status` returns exit code 1 when probe says disconnected.
- `rag bridge reauth` requires a TTY (refuses to run when stdin/stdout
  aren't TTYs).

Mocks:
- `urllib.request.urlopen` to fake the bridge HTTP response.
- `subprocess.run` for `launchctl print` to fake "loaded/not loaded".
- `sys.stdin.isatty()` for the TTY check.

We do NOT touch `~/.local/bin/whatsapp-bridge`, the real bridge daemon,
or the actual `store/whatsapp.db`. All filesystem ops in `bridge_reauth`
are guarded behind the TTY check, which we exercise via the no-TTY
exit-2 path.
"""
from __future__ import annotations

import io
import json
from unittest import mock

import pytest
from click.testing import CliRunner

import rag


def test_bridge_group_help_lists_subcommands():
    """`rag bridge --help` should show both `status` and `reauth`."""
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["bridge", "--help"])
    assert result.exit_code == 0, result.output
    assert "status" in result.output
    assert "reauth" in result.output


@pytest.mark.parametrize("body,expected_ok,detail_substr", [
    # All-green: connected + logged in.
    ('{"connected": true, "logged_in": true, "jid": "5491234567890.0:0@s.whatsapp.net"}',
     True, "connected"),
    # Logged-in but socket dropped — bridge can recover without QR.
    ('{"connected": false, "logged_in": true, "jid": "5491234567890.0:0@s.whatsapp.net"}',
     False, "auto-reconnect pending"),
    # No paired device — needs `rag bridge reauth`.
    ('{"connected": false, "logged_in": false, "jid": ""}',
     False, "not_paired"),
])
def test_http_probe_parses_health_json(body, expected_ok, detail_substr):
    """The probe should read `/api/health` JSON and classify correctly."""
    fake_resp = mock.MagicMock()
    fake_resp.read.return_value = body.encode("utf-8")
    fake_resp.status = 200
    fake_resp.__enter__ = lambda self: self
    fake_resp.__exit__ = lambda *args: None

    with mock.patch("urllib.request.urlopen", return_value=fake_resp):
        ok, detail = rag._wa_bridge_http_probe()

    assert ok is expected_ok
    assert detail_substr in detail


def test_http_probe_handles_old_bridge_binary_404():
    """If the bridge predates `/api/health`, probe returns clear hint."""
    import urllib.error
    err = urllib.error.HTTPError(
        url="http://localhost:8080/api/health",
        code=404,
        msg="Not Found",
        hdrs=None,
        fp=io.BytesIO(b"404 page not found\n"),
    )
    with mock.patch("urllib.request.urlopen", side_effect=err):
        ok, detail = rag._wa_bridge_http_probe()
    assert ok is False
    assert "no_health_endpoint" in detail
    assert "go build" in detail  # actionable hint


def test_http_probe_handles_unreachable_bridge():
    """Daemon offline → ConnectionRefused → probe returns False + hint."""
    import urllib.error
    err = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    with mock.patch("urllib.request.urlopen", side_effect=err):
        ok, detail = rag._wa_bridge_http_probe()
    assert ok is False
    assert "unreachable" in detail


def test_bridge_status_exits_1_when_disconnected(tmp_path):
    """`rag bridge status` should exit non-zero when bridge is down so cron
    jobs / monitoring scripts can detect it.
    """
    # Point sentinel + log at non-existent paths so we skip those branches.
    nonexistent_sentinel = tmp_path / "no-sentinel"
    nonexistent_log = tmp_path / "no-log"
    with (
        mock.patch.object(rag, "_wa_bridge_is_loaded", return_value=False),
        mock.patch.object(rag, "_wa_bridge_http_probe",
                          return_value=(False, "unreachable: ...")),
        mock.patch.object(rag, "_WA_BRIDGE_SENTINEL", nonexistent_sentinel),
        mock.patch.object(rag, "_WA_BRIDGE_LOG", nonexistent_log),
    ):
        runner = CliRunner()
        result = runner.invoke(rag.cli, ["bridge", "status"])

    assert result.exit_code == 1, result.output
    assert "disconnected" in result.output
    assert "rag bridge reauth" in result.output


def test_bridge_status_ok_when_connected(tmp_path):
    """When the probe says connected, exit code is 0 (no nag)."""
    nonexistent_sentinel = tmp_path / "no-sentinel"
    nonexistent_log = tmp_path / "no-log"
    with (
        mock.patch.object(rag, "_wa_bridge_is_loaded", return_value=True),
        mock.patch.object(rag, "_wa_bridge_http_probe",
                          return_value=(True, "connected (jid=...)")),
        mock.patch.object(rag, "_WA_BRIDGE_SENTINEL", nonexistent_sentinel),
        mock.patch.object(rag, "_WA_BRIDGE_LOG", nonexistent_log),
    ):
        runner = CliRunner()
        result = runner.invoke(rag.cli, ["bridge", "status"])

    assert result.exit_code == 0, result.output
    assert "connected" in result.output
    assert "✓ loaded" in result.output


def test_bridge_reauth_refuses_without_tty():
    """Reauth requires interactive TTY (QR uses Unicode half-blocks).

    `CliRunner.invoke` doesn't attach a real TTY, so this path always
    triggers the early-exit check.
    """
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["bridge", "reauth", "--yes"])
    assert result.exit_code == 2, result.output
    assert "TTY" in result.output


# NOTE: there's no `test_bridge_reauth_aborts_on_missing_binary` because
# Click's CliRunner unconditionally replaces sys.stdin/stdout with mocks
# that return `isatty() == False` — so any reauth path past the TTY
# guard can't be exercised in unit tests without significant refactoring
# (extract `_is_interactive()` helper, etc.). The binary-existence
# check is trivial enough that we cover it via code review instead.
