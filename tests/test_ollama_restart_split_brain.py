"""Tests para `_ollama_restart_if_stuck` — split-brain guard (commit 2026-05-01).

Background: el auto-recovery de obsidian-rag respawnea ollama via `brew
services restart` cuando detecta wedge. Pre-fix, no chequeaba si Ollama.app
también estaba corriendo, dejando dos daemons binding `:11434` con
SO_REUSEPORT → kernel load-balancea connections random → modelos cargados
distintos en cada daemon → cold-load + 90s timeouts.

El fix mata Ollama.app explícitamente antes de restartear homebrew, y
chequea post-restart que solo haya 1 PID.
"""
from __future__ import annotations

from unittest import mock

import pytest


def _make_run_response(returncode: int = 0, stdout: str = "", stderr: str = ""):
    """Build a CompletedProcess-shaped object for subprocess.run mocks."""
    m = mock.MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


class TestSplitBrainGuard:
    """Verifica que cuando homebrew daemon está loaded, _ollama_restart_if_stuck
    mate Ollama.app antes de hacer `brew services restart`.
    """

    def test_kills_ollama_app_before_brew_restart(self, monkeypatch):
        """Caso: launchctl list reporta homebrew.mxcl.ollama loaded.
        El restart debe llamar pkill -f /Applications/Ollama.app ANTES de
        `brew services restart ollama`.
        """
        from web import server

        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(list(cmd))
            # launchctl list → homebrew daemon visible
            if cmd[:2] == ["launchctl", "list"]:
                return _make_run_response(returncode=0, stdout="123\t0\thomebrew.mxcl.ollama\n")
            # brew services restart → ok
            if cmd[:3] == ["/opt/homebrew/bin/brew", "services", "restart"]:
                return _make_run_response(returncode=0)
            # pgrep ollama serve → 1 daemon (post-restart sanity check)
            if cmd[:2] == ["pgrep", "-f"] and "ollama serve" in cmd[2]:
                return _make_run_response(returncode=0, stdout="999\n")
            # pkill, sleeps, etc → ok
            return _make_run_response(returncode=0)

        monkeypatch.setattr(server.subprocess, "run", fake_run)
        monkeypatch.setattr(server, "_ollama_alive", lambda timeout=2.0: True)
        monkeypatch.setattr(server.time, "sleep", lambda *_: None)

        result = server._ollama_restart_if_stuck()
        assert result is True

        cmds_run = [tuple(c[:3]) if len(c) >= 3 else tuple(c) for c in calls]
        # Buscamos los índices en orden.
        kill_app_idx = next(
            (i for i, c in enumerate(calls)
             if c[:3] == ["pkill", "-9", "-f"]
             and len(c) >= 4 and "/Applications/Ollama.app" in c[3]),
            -1,
        )
        brew_idx = next(
            (i for i, c in enumerate(calls)
             if c[:3] == ["/opt/homebrew/bin/brew", "services", "restart"]),
            -1,
        )
        assert kill_app_idx >= 0, f"no encontré pkill /Applications/Ollama.app — calls: {cmds_run}"
        assert brew_idx >= 0, f"no encontré brew services restart — calls: {cmds_run}"
        assert kill_app_idx < brew_idx, (
            f"pkill Ollama.app debería ir ANTES de brew restart, pero indices: "
            f"kill={kill_app_idx} brew={brew_idx}"
        )

    def test_falls_back_to_app_when_homebrew_not_loaded(self, monkeypatch):
        """Caso: launchctl list NO reporta homebrew.mxcl.ollama. La función
        debe matar todos los procesos ollama y reabrir Ollama.app.
        """
        from web import server

        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            calls.append(list(cmd))
            # launchctl list → SIN homebrew daemon
            if cmd[:2] == ["launchctl", "list"]:
                return _make_run_response(returncode=0, stdout="(no ollama here)\n")
            if cmd[:2] == ["pgrep", "-f"] and "ollama serve" in cmd[2]:
                return _make_run_response(returncode=0, stdout="888\n")
            return _make_run_response(returncode=0)

        monkeypatch.setattr(server.subprocess, "run", fake_run)
        monkeypatch.setattr(server, "_ollama_alive", lambda timeout=2.0: True)
        monkeypatch.setattr(server.time, "sleep", lambda *_: None)

        result = server._ollama_restart_if_stuck()
        assert result is True

        # Tiene que haber un `pkill -9 -f ollama` (sin path específico) +
        # `open -a Ollama` después.
        pkill_idx = next(
            (i for i, c in enumerate(calls)
             if c[:3] == ["pkill", "-9", "-f"] and (len(c) > 3 and c[3] == "ollama")),
            -1,
        )
        open_idx = next(
            (i for i, c in enumerate(calls)
             if c[:3] == ["open", "-a", "Ollama"]),
            -1,
        )
        assert pkill_idx >= 0, f"no encontré pkill -9 -f ollama — calls: {calls}"
        assert open_idx >= 0, f"no encontré open -a Ollama — calls: {calls}"
        assert pkill_idx < open_idx, "pkill debería ir antes que open -a"

    def test_post_restart_warns_on_double_daemon(self, monkeypatch, capsys):
        """Si post-restart pgrep reporta 2+ PIDs de ollama serve, loggear
        warning greppable. La función igual retorna True (mejor degraded
        mode que sin chat).
        """
        from web import server

        def fake_run(cmd, *args, **kwargs):
            if cmd[:2] == ["launchctl", "list"]:
                return _make_run_response(returncode=0, stdout="999\t0\thomebrew.mxcl.ollama\n")
            if cmd[:3] == ["/opt/homebrew/bin/brew", "services", "restart"]:
                return _make_run_response(returncode=0)
            # pgrep returns TWO PIDs → split-brain post-restart
            if cmd[:2] == ["pgrep", "-f"] and "ollama serve" in cmd[2]:
                return _make_run_response(returncode=0, stdout="111\n222\n")
            return _make_run_response(returncode=0)

        monkeypatch.setattr(server.subprocess, "run", fake_run)
        monkeypatch.setattr(server, "_ollama_alive", lambda timeout=2.0: True)
        monkeypatch.setattr(server.time, "sleep", lambda *_: None)

        result = server._ollama_restart_if_stuck()
        assert result is True  # degraded mode, no aborta

        captured = capsys.readouterr()
        assert "ollama-restart-warn" in captured.out
        assert "split-brain" in captured.out
        assert "111" in captured.out and "222" in captured.out

    def test_returns_false_if_alive_check_never_passes(self, monkeypatch):
        """Si después del restart el daemon nunca contesta, retorna False."""
        from web import server

        def fake_run(cmd, *args, **kwargs):
            if cmd[:2] == ["launchctl", "list"]:
                return _make_run_response(returncode=0, stdout="999\t0\thomebrew.mxcl.ollama\n")
            return _make_run_response(returncode=0)

        monkeypatch.setattr(server.subprocess, "run", fake_run)
        monkeypatch.setattr(server, "_ollama_alive", lambda timeout=2.0: False)
        monkeypatch.setattr(server.time, "sleep", lambda *_: None)

        result = server._ollama_restart_if_stuck()
        assert result is False
