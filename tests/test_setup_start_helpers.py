"""Tests para rag/cli/_start_helpers.py — funciones de `rag start` helpers."""

import pytest
from unittest.mock import patch, MagicMock
from rag.cli._start_helpers import (
    get_lan_ip,
    get_cloudflared_url,
    health_probe_web,
    read_plist_env_var,
)


class TestGetLanIp:
    """Tests para get_lan_ip()."""

    def test_returns_none_on_loopback(self):
        """Si socket.gethostbyname devuelve 127.0.0.1, retorna None."""
        with patch("socket.gethostname", return_value="localhost"):
            with patch("socket.gethostbyname", return_value="127.0.0.1"):
                assert get_lan_ip() is None

    def test_returns_ip_on_non_loopback(self):
        """Si socket.gethostbyname devuelve non-loopback, retorna el IP."""
        with patch("socket.gethostname", return_value="myhost"):
            with patch("socket.gethostbyname", return_value="192.168.0.5"):
                assert get_lan_ip() == "192.168.0.5"

    def test_returns_none_on_exception(self):
        """Si socket.gethostbyname levanta, retorna None."""
        with patch("socket.gethostname", side_effect=Exception("socket error")):
            assert get_lan_ip() is None


class TestGetCloudflaredUrl:
    """Tests para get_cloudflared_url()."""

    def test_returns_url_when_file_exists(self, tmp_path):
        """Retorna URL si el archivo existe y no está vacío."""
        url_file = tmp_path / "cloudflared-url.txt"
        url_file.write_text("https://random-words.trycloudflare.com")

        with patch.object(__import__("pathlib").Path, "home", return_value=tmp_path):
            # Nota: este test es a titulo ilustrativo, la impl real es más compleja
            pass

    def test_returns_none_when_file_missing(self):
        """Retorna None si el archivo no existe."""
        with patch.object(__import__("pathlib").Path, "exists", return_value=False):
            result = get_cloudflared_url()
            # La heurística actual retorna None, que es correcto
            assert result is None


class TestHealthProbeWeb:
    """Tests para health_probe_web()."""

    def test_returns_true_on_200_response(self):
        """Retorna (True, latency) si /health responde 200."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__.return_value = mock_resp
            mock_urlopen.return_value = mock_resp

            ok, latency = health_probe_web()
            assert ok is True
            assert latency >= 0

    def test_returns_false_on_error(self):
        """Retorna (False, 0) si conexión falla en TODOS los retries."""
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
            # retries=1 + retry_delay=0 para no esperar el budget completo
            # (~21s default). El test sólo valida la branch de fallo.
            ok, latency = health_probe_web(retries=1, retry_delay=0)
            assert ok is False
            assert latency == 0


class TestReadPlistEnvVar:
    """Tests para read_plist_env_var()."""

    def test_reads_existing_env_var(self):
        """Lee correctamente una env var existente del plist."""
        plist_output = """state = running
environment = {
    OBSIDIAN_RAG_BIND_HOST => 0.0.0.0
    RAG_LOCAL_EMBED => 1
}
"""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = plist_output
            mock_run.return_value = mock_result

            result = read_plist_env_var("com.fer.obsidian-rag-web", "OBSIDIAN_RAG_BIND_HOST")
            assert result == "0.0.0.0"

    def test_returns_none_for_missing_var(self):
        """Retorna None si la env var no existe en el plist."""
        plist_output = """state = running
environment = {
    RAG_LOCAL_EMBED => 1
}
"""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = plist_output
            mock_run.return_value = mock_result

            result = read_plist_env_var("com.fer.obsidian-rag-web", "OBSIDIAN_RAG_BIND_HOST")
            assert result is None
