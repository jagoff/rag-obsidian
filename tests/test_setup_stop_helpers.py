"""Tests para funciones helpers de `rag stop`."""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from rag.cli.setup import _find_cloudflare_tunnel_labels, _cleanup_staled_locks


class TestFindCloudflaredTunnelLabels:
    """Tests para _find_cloudflare_tunnel_labels()."""

    def test_returns_empty_tuple_when_no_plists(self, tmp_path):
        """Retorna () cuando no hay plists de cloudflare."""
        with patch("rag._LAUNCH_AGENTS_DIR", tmp_path):
            result = _find_cloudflare_tunnel_labels()
            assert result == ()

    def test_finds_cloudflare_plists(self, tmp_path):
        """Encuentra plists com.fer.obsidian-rag-cloudflare-*.plist."""
        # Crear algunos plists de prueba
        (tmp_path / "com.fer.obsidian-rag-cloudflare-tunnel.plist").touch()
        (tmp_path / "com.fer.obsidian-rag-cloudflare-watcher.plist").touch()
        (tmp_path / "com.fer.obsidian-rag-web.plist").touch()  # No debe incluirse

        with patch("rag._LAUNCH_AGENTS_DIR", tmp_path):
            result = _find_cloudflare_tunnel_labels()

        assert len(result) == 2
        assert "com.fer.obsidian-rag-cloudflare-tunnel" in result
        assert "com.fer.obsidian-rag-cloudflare-watcher" in result
        assert "com.fer.obsidian-rag-web" not in result

    def test_returns_sorted_labels(self, tmp_path):
        """Retorna labels ordenados alfabéticamente."""
        (tmp_path / "com.fer.obsidian-rag-cloudflare-zulu.plist").touch()
        (tmp_path / "com.fer.obsidian-rag-cloudflare-alpha.plist").touch()
        (tmp_path / "com.fer.obsidian-rag-cloudflare-bravo.plist").touch()

        with patch("rag._LAUNCH_AGENTS_DIR", tmp_path):
            result = _find_cloudflare_tunnel_labels()

        assert result == (
            "com.fer.obsidian-rag-cloudflare-alpha",
            "com.fer.obsidian-rag-cloudflare-bravo",
            "com.fer.obsidian-rag-cloudflare-zulu",
        )


class TestCleanupStaledLocks:
    """Tests para _cleanup_staled_locks()."""

    def test_skips_nonexistent_files(self, tmp_path):
        """No falla si los lock files no existen."""
        lock_path = tmp_path / "nonexistent.lock"
        assert not lock_path.exists()

        with patch.object(Path, "home", return_value=tmp_path.parent):
            # Solo debe fallar si intenta hacer algo con el path no existente,
            # pero la func chequea exists() primero
            _cleanup_staled_locks()

    def test_removes_lock_for_dead_pid(self, tmp_path):
        """Borra .pid lock si el PID no existe."""
        lock_dir = tmp_path / ".local" / "share" / "obsidian-rag"
        lock_dir.mkdir(parents=True, exist_ok=True)
        pid_file = lock_dir / "supervisor.pid"

        # Escribir un PID que seguramente no existe (ej. 999999)
        pid_file.write_text("999999")

        with patch.object(Path, "home", return_value=tmp_path):
            _cleanup_staled_locks()

        # El lock debe haber sido borrado porque el PID no existe
        assert not pid_file.exists()

    def test_keeps_lock_for_live_pid(self, tmp_path):
        """No borra .pid lock si el proceso sigue vivo."""
        lock_dir = tmp_path / ".local" / "share" / "obsidian-rag"
        lock_dir.mkdir(parents=True, exist_ok=True)
        pid_file = lock_dir / "supervisor.pid"

        # Escribir el PID del proceso actual (que está vivo)
        pid_file.write_text(str(os.getpid()))

        with patch.object(Path, "home", return_value=tmp_path):
            _cleanup_staled_locks()

        # El lock debe seguir existiendo porque el PID está vivo
        assert pid_file.exists()
