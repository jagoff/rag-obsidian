"""Integration tests para `_print_access_urls()` en `rag start`."""

import pytest
from unittest.mock import patch, MagicMock
from rag.cli.setup import _print_access_urls


class TestPrintAccessUrls:
    """Tests para el output de URLs en `rag start`."""

    def test_print_access_urls_basic_output(self, capsys):
        """Verifica que _print_access_urls imprime secciones esperadas."""
        with patch("rag.cli._start_helpers.health_probe_web", return_value=(False, 0)):
            with patch("rag.cli._start_helpers.read_plist_env_var", return_value=None):
                with patch("rag.cli._start_helpers.get_cloudflared_url", return_value=None):
                    with patch("rag.cli._start_helpers.get_lan_ip", return_value=None):
                        _print_access_urls()

        captured = capsys.readouterr()
        output = captured.out

        # Verificar que se imprimen las secciones esperadas
        assert "▸ acceso" in output
        assert "▸ herramientas" in output
        assert "Verificar:" in output
        assert "chat" in output
        assert "dashboard" in output
        assert "atlas" in output
        assert "mirror" in output
        assert "memory" in output
        assert "admin token" in output
        assert "MCP server" in output
        assert "obsidian-rag-mcp" in output

    def test_print_access_urls_with_health_ok(self, capsys):
        """Si health_probe_web devuelve ok=True, muestra latencia."""
        with patch("rag.cli._start_helpers.health_probe_web", return_value=(True, 45)):
            with patch("rag.cli._start_helpers.read_plist_env_var", return_value=None):
                with patch("rag.cli._start_helpers.get_cloudflared_url", return_value=None):
                    with patch("rag.cli._start_helpers.get_lan_ip", return_value=None):
                        _print_access_urls()

        captured = capsys.readouterr()
        assert "45 ms" in captured.out or "✓" in captured.out

    def test_print_access_urls_with_lan_ip(self, capsys):
        """Si BIND_HOST=0.0.0.0 y hay LAN IP, muestra URLs con LAN IP."""
        with patch("rag.cli._start_helpers.health_probe_web", return_value=(False, 0)):
            with patch("rag.cli._start_helpers.read_plist_env_var", return_value="0.0.0.0"):
                with patch("rag.cli._start_helpers.get_lan_ip", return_value="192.168.0.5"):
                    with patch("rag.cli._start_helpers.get_cloudflared_url", return_value=None):
                        _print_access_urls()

        captured = capsys.readouterr()
        output = captured.out

        # Debe mostrar LAN IP
        assert "192.168.0.5" in output
        assert "LAN" in output

    def test_print_access_urls_with_tunnel(self, capsys):
        """Si cloudflared URL existe, la muestra."""
        tunnel_url = "https://random-words-1234.trycloudflare.com"
        with patch("rag.cli._start_helpers.health_probe_web", return_value=(False, 0)):
            with patch("rag.cli._start_helpers.read_plist_env_var", return_value=None):
                with patch("rag.cli._start_helpers.get_cloudflared_url", return_value=tunnel_url):
                    with patch("rag.cli._start_helpers.get_lan_ip", return_value=None):
                        _print_access_urls()

        captured = capsys.readouterr()
        assert tunnel_url in captured.out
        assert "tunnel" in captured.out
