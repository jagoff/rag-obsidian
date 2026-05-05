"""Tests para YouTube transcript circuit-breaker.

Cuando YouTube bloquea la IP (IpBlocked), el sistema debe:
1. Detectar el bloqueo en el primer video que lo cause
2. Registrar un cooldown exponencial (4h → 8h → 16h → 24h)
3. En próximos runs dentro del cooldown, abort early sin tocar la red
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from rag.cross_source_etls import (
    _check_yt_ip_cooldown,
    _set_yt_ip_cooldown,
    _fetch_yt_transcript_for_index,
    _sync_youtube_transcripts,
    _YT_IP_BLOCKED_COOLDOWNS_SECONDS,
)
import rag


@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path):
    """Aísla DB_PATH para evitar contaminar telemetry.db real."""
    snap = rag.DB_PATH
    rag.DB_PATH = tmp_path / "ragvec"
    try:
        yield
    finally:
        rag.DB_PATH = snap


class TestYtCircuitBreakerCooldown:
    """Tests para las funciones de cooldown."""

    def test_cooldown_check_returns_inactive_when_no_row(self):
        """Sin registro en tabla → cooldown inactivo."""
        result = _check_yt_ip_cooldown()
        assert isinstance(result, dict)
        assert result["active"] is False

    def test_cooldown_set_and_check_returns_active(self):
        """Después de setear cooldown, check retorna activo."""
        _set_yt_ip_cooldown()
        result = _check_yt_ip_cooldown()
        assert isinstance(result, dict)
        assert result["active"] is True
        assert "blocked_until_ts" in result
        assert result["retry_count"] == 0

    def test_cooldown_expiry_detection(self):
        """Cuando cooldown expira, check retorna inactivo."""
        # Set cooldown muy corto (0.1 segundos)
        import rag as _rag_module
        try:
            conn = _rag_module._ragvec_state_conn()
            now = time.time()
            conn.execute(
                """
                INSERT OR REPLACE INTO rag_yt_transcript_cooldown (id, blocked_until_ts, retry_count, last_blocked_at)
                VALUES (0, ?, 0, ?)
                """,
                (now + 0.1, now),
            )
            conn.commit()
            conn.close()
        except Exception:
            pytest.skip("No DB available")

        # Esperar a que expire
        time.sleep(0.2)

        result = _check_yt_ip_cooldown()
        assert isinstance(result, dict)
        assert result["active"] is False

    def test_cooldown_exponential_backoff(self):
        """Múltiples bloqueos después de cooldown expirado aumentan exponencialmente.

        Nota: llamadas a _set_yt_ip_cooldown() dentro del MISMO cooldown window
        son idempotentes — no bumpean retry_count. Necesitamos esperar a que
        expire entre cada llamada para que el exponential backoff funcione.
        """
        durations = []

        for i in range(4):
            _set_yt_ip_cooldown()
            result = _check_yt_ip_cooldown()
            assert result["active"] is True
            assert result["retry_count"] == i

            # Extract duration
            blocked_until = result["blocked_until_ts"]
            now = time.time()
            duration = blocked_until - now
            durations.append(duration)

            # Esperar a que expire el cooldown antes de la próxima llamada
            # (así la siguiente _set_yt_ip_cooldown() bumpea el counter)
            time.sleep(max(duration + 0.1, 0.2))

        # Verificar que cada cooldown es aproximadamente el esperado
        # (allow ~1s de tolerancia por timing)
        assert durations[0] < _YT_IP_BLOCKED_COOLDOWNS_SECONDS[0] + 1
        assert durations[1] < _YT_IP_BLOCKED_COOLDOWNS_SECONDS[1] + 1
        assert durations[2] < _YT_IP_BLOCKED_COOLDOWNS_SECONDS[2] + 1
        # El 4to+ se clampea a 24h
        assert durations[3] < _YT_IP_BLOCKED_COOLDOWNS_SECONDS[3] + 1


class TestYtTranscriptFetchDetectsIpBlocked:
    """Tests para IpBlocked detection en _fetch_yt_transcript_for_index."""

    def test_ip_blocked_exception_triggers_cooldown(self):
        """Si api.list() levanta IpBlocked, registra cooldown."""
        from youtube_transcript_api._errors import IpBlocked

        with patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api.list.side_effect = IpBlocked("YouTube is blocking your IP")
            mock_api_class.return_value = mock_api

            # Antes de fetch, cooldown está inactivo
            result_before = _check_yt_ip_cooldown()
            assert result_before["active"] is False

            # Fetch detecta IpBlocked
            result = _fetch_yt_transcript_for_index("dummyvideo123")
            assert result is None

            # Después de fetch, cooldown está activo
            result_after = _check_yt_ip_cooldown()
            assert result_after["active"] is True

    def test_other_exceptions_do_not_trigger_cooldown(self):
        """Excepciones que NO sean IpBlocked no activan cooldown."""
        with patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_api_class:
            mock_api = MagicMock()
            mock_api.list.side_effect = RuntimeError("Some other error")
            mock_api_class.return_value = mock_api

            result_before = _check_yt_ip_cooldown()
            assert result_before["active"] is False

            # Fetch encuentra RuntimeError (no IpBlocked)
            result = _fetch_yt_transcript_for_index("dummyvideo123")
            assert result is None

            # Cooldown sigue inactivo
            result_after = _check_yt_ip_cooldown()
            assert result_after["active"] is False


class TestYtCircuitBreakerIdempotence:
    """Tests para idempotencia de _set_yt_ip_cooldown dentro del mismo window."""

    def test_set_cooldown_multiple_calls_within_window_idempotent(self):
        """Múltiples llamadas a _set_yt_ip_cooldown dentro del MISMO cooldown window
        son idempotentes — retry_count no bumpea.
        """
        # Primera llamada — establece retry_count=0, cooldown 4h
        _set_yt_ip_cooldown()
        result1 = _check_yt_ip_cooldown()
        assert result1["active"] is True
        assert result1["retry_count"] == 0
        blocked_until_1 = result1["blocked_until_ts"]

        # Segunda llamada INMEDIATA (dentro del mismo window)
        # No debe cambiar retry_count ni blocked_until_ts
        _set_yt_ip_cooldown()
        result2 = _check_yt_ip_cooldown()
        assert result2["active"] is True
        assert result2["retry_count"] == 0  # No bumpeó
        assert result2["blocked_until_ts"] == blocked_until_1  # Sin cambios

        # Tercera y cuarta llamadas — mismo resultado
        _set_yt_ip_cooldown()
        _set_yt_ip_cooldown()
        result3 = _check_yt_ip_cooldown()
        assert result3["retry_count"] == 0
        assert result3["blocked_until_ts"] == blocked_until_1


class TestSyncYoutubeTranscriptsCircuitBreakerAbortEarly:
    """Tests para _sync_youtube_transcripts abort early cuando cooldown activo."""

    def test_sync_returns_early_when_cooldown_active(self, tmp_path):
        """Cuando cooldown activo, sync retorna early sin network calls."""
        vault_root = tmp_path / "vault"
        vault_root.mkdir()

        # Activar cooldown
        _set_yt_ip_cooldown()
        result_before = _check_yt_ip_cooldown()
        assert result_before["active"] is True

        # Sync debe abortar sin tocar videos
        result = _sync_youtube_transcripts(vault_root)
        assert result["ok"] is True
        assert result["reason"] == "yt_ip_cooldown_active"
        assert result["files_written"] == 0
        assert result["fetched_this_run"] == 0

    def test_sync_proceeds_when_cooldown_expired(self, tmp_path):
        """Cuando cooldown expiró, sync procede normal (sin net call en test)."""
        vault_root = tmp_path / "vault"
        vault_root.mkdir()

        # Crear un cooldown expirado
        import rag as _rag_module
        try:
            conn = _rag_module._ragvec_state_conn()
            now = time.time()
            # Set expired timestamp (pasado)
            conn.execute(
                """
                INSERT OR REPLACE INTO rag_yt_transcript_cooldown (id, blocked_until_ts, retry_count, last_blocked_at)
                VALUES (0, ?, 0, ?)
                """,
                (now - 1000, now - 1000),
            )
            conn.commit()
            conn.close()
        except Exception:
            pytest.skip("No DB available")

        # Sync debe proceder (aunque sin videos para fetch)
        result = _sync_youtube_transcripts(vault_root)
        assert result["ok"] is True
        # Sin videos, debería retornar "no_videos"
        assert result.get("reason") == "no_videos"

    def test_sync_loop_breaks_on_first_ip_blocked(self, tmp_path):
        """Cuando se detecta IpBlocked en el primer video del batch,
        el loop debe romper y NO procesar el resto de los videos.

        Esto se verifica mockeando YouTubeTranscriptApi para que lance IpBlocked.
        El loop debe:
        1. Intentar el primer video
        2. _fetch_yt_transcript_for_index atrapa IpBlocked, llama _set_yt_ip_cooldown()
        3. El loop chequea cooldown y rompe
        4. Videos posteriores nunca se procesan
        """
        vault_root = tmp_path / "vault"
        vault_root.mkdir()

        # Limpiar cooldown previo de otros tests
        import rag as _rag_module
        try:
            conn = _rag_module._ragvec_state_conn()
            conn.execute("DELETE FROM rag_yt_transcript_cooldown WHERE id = 0")
            conn.commit()
            conn.close()
        except Exception as e:
            pass  # Silent fail, table may not exist in fresh DB

        # Crear 3 videos en el vault (en la ruta que _collect_youtube_video_ids busca)
        yt_dir = vault_root / "04-Archive/99-obsidian-system/99-AI/external-ingest/YouTube"
        yt_dir.mkdir(parents=True)
        (yt_dir / "2026-05-04.md").write_text(
            "# YouTube Daily\n"
            "[Video 1](https://www.youtube.com/watch?v=abc123defgh)\n"
            "[Video 2](https://www.youtube.com/watch?v=xyz456ijklm)\n"
            "[Video 3](https://www.youtube.com/watch?v=uvw789nopqr)\n"
        )

        from youtube_transcript_api._errors import IpBlocked

        # Mockeamos el YouTubeTranscriptApi para que lance IpBlocked en el primer call a list()
        with patch("youtube_transcript_api.YouTubeTranscriptApi") as mock_api_class:
            mock_api = MagicMock()
            # Primera llamada a api.list() lanza IpBlocked directamente
            # El resto nunca ocurrirá porque el loop debe romper
            mock_api.list.side_effect = [
                IpBlocked("YouTube is blocking your IP"),  # Primera llamada lanza excepción
                MagicMock(),  # Nunca debería llega acá
                MagicMock(),  # Nunca debería llegar acá
            ]
            mock_api_class.return_value = mock_api

            result = _sync_youtube_transcripts(vault_root, batch=10)

        # El loop debe haber intentado el primer video y luego roto
        # fetched_this_run cuenta cada intento (incluso los que fallan)
        assert result["fetched_this_run"] == 1, f"Expected 1 fetch, got {result['fetched_this_run']}"
        assert result["failed_this_run"] == 1
        # El cooldown debe estar activo
        cooldown = _check_yt_ip_cooldown()
        assert cooldown["active"] is True, f"Expected cooldown active, got {cooldown}"
        # Verificar que solo se hizo 1 intento de fetch (el loop no continuó)
        assert mock_api.list.call_count == 1, f"Expected 1 call to list(), got {mock_api.list.call_count}"
