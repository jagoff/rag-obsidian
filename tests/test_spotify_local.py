"""Spotify local poller — AppleScript parser + SQLite writer/reader.

Mockea `subprocess.run` para devolver fake AppleScript output sin tocar
el desktop app. La SQLite real (telemetry.db) se reusa porque el writer
usa `_ragvec_state_conn()` (configurable solo via DB_PATH a nivel
proceso). Cleanup: cada test borra rows que insertó al final.
"""
from __future__ import annotations

import subprocess
import time
from unittest.mock import patch

import pytest

from rag.integrations import spotify_local as sl


def _fake_run(stdout: str):
    """Build a `CompletedProcess`-like fake for `subprocess.run`."""
    cp = subprocess.CompletedProcess(args=[], returncode=0)
    cp.stdout = stdout
    cp.stderr = ""
    return cp


class TestNowPlaying:
    def test_not_running(self):
        with patch("subprocess.run", return_value=_fake_run("NOT_RUNNING")):
            assert sl.now_playing() is None

    def test_no_track(self):
        with patch("subprocess.run", return_value=_fake_run("NO_TRACK")):
            assert sl.now_playing() is None

    def test_empty_output(self):
        with patch("subprocess.run", return_value=_fake_run("")):
            assert sl.now_playing() is None

    def test_timeout(self):
        def _raise(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="osascript", timeout=2.0)
        with patch("subprocess.run", side_effect=_raise):
            assert sl.now_playing() is None

    def test_osascript_missing(self):
        def _raise(*a, **kw):
            raise FileNotFoundError("osascript")
        with patch("subprocess.run", side_effect=_raise):
            assert sl.now_playing() is None

    def test_full_metadata_playing(self):
        out = (
            "Pensando en Ti|Canserbero|Vida|playing|"
            "spotify:track:2auNjgM4xIOANxtZT4Xe9z|244000|45.2|"
            "https://i.scdn.co/image/ab67616d0000b2730f38ea3b0b59d4b59b4da2e6"
        )
        with patch("subprocess.run", return_value=_fake_run(out)):
            np = sl.now_playing()
        assert np is not None
        assert np["name"] == "Pensando en Ti"
        assert np["artist"] == "Canserbero"
        assert np["album"] == "Vida"
        assert np["state"] == "playing"
        assert np["track_id"] == "spotify:track:2auNjgM4xIOANxtZT4Xe9z"
        assert np["duration_ms"] == 244000
        assert np["position_s"] == 45.2
        assert np["art_url"].startswith("https://i.scdn.co/")

    def test_paused_track(self):
        out = "Algo|Artista|Album|paused|spotify:track:abc|180000|90.0|"
        with patch("subprocess.run", return_value=_fake_run(out)):
            np = sl.now_playing()
        assert np is not None
        assert np["state"] == "paused"

    def test_truncated_output_returns_none(self):
        # AppleScript transient — only 3 fields. Bail.
        with patch("subprocess.run", return_value=_fake_run("X|Y|Z")):
            assert sl.now_playing() is None

    def test_empty_artist_skipped(self):
        # Spotify a veces emite "" en transitions; el parser lo descarta.
        out = "Track||Album|playing|spotify:track:abc|180000|0|"
        with patch("subprocess.run", return_value=_fake_run(out)):
            assert sl.now_playing() is None

    def test_malformed_duration_swallowed(self):
        # Duration no parseable → None pero el resto se conserva.
        out = "X|Y|Z|playing|spotify:track:abc|notanumber|45|"
        with patch("subprocess.run", return_value=_fake_run(out)):
            np = sl.now_playing()
        assert np is not None
        assert np["duration_ms"] is None
        assert np["position_s"] == 45.0


@pytest.fixture
def cleanup_spotify_log():
    """Borra el row test después del test para no contaminar telemetry.db."""
    from rag import _ragvec_state_conn
    test_track_ids: list[str] = []
    yield test_track_ids
    if not test_track_ids:
        return
    try:
        with _ragvec_state_conn() as conn:
            placeholders = ",".join("?" * len(test_track_ids))
            conn.execute(
                f"DELETE FROM rag_spotify_log WHERE track_id IN ({placeholders})",
                test_track_ids,
            )
    except Exception:
        pass


class TestRecordNowPlaying:
    def test_skips_when_not_running(self):
        with patch.object(sl, "now_playing", return_value=None):
            res = sl.record_now_playing()
        assert res["recorded"] is False
        assert "spotify_not_running" in res["reason"]

    def test_skips_paused(self):
        np = {
            "name": "T", "artist": "A", "album": "Al",
            "state": "paused", "track_id": "spotify:track:test_paused",
            "duration_ms": 180000, "position_s": 0, "art_url": None,
        }
        with patch.object(sl, "now_playing", return_value=np):
            res = sl.record_now_playing()
        assert res["recorded"] is False
        assert res["reason"] == "state_paused"

    def test_inserts_new_row_on_first_play(self, cleanup_spotify_log):
        track_id = f"spotify:track:test_insert_{int(time.time())}"
        cleanup_spotify_log.append(track_id)
        np = {
            "name": "TestTrack", "artist": "TestArtist", "album": "TestAlbum",
            "state": "playing", "track_id": track_id,
            "duration_ms": 200000, "position_s": 5.0, "art_url": "http://x/a.jpg",
        }
        with patch.object(sl, "now_playing", return_value=np):
            res = sl.record_now_playing()
        assert res["recorded"] is True
        assert res["updated"] is False
        assert res["track"] == "TestTrack"
        recent = sl.recent_tracks_today(20)
        match = [t for t in recent if t["track_id"] == track_id]
        assert len(match) == 1
        assert match[0]["name"] == "TestTrack"
        assert match[0]["artist"] == "TestArtist"

    def test_updates_last_seen_on_repeat_within_window(self, cleanup_spotify_log):
        # Mismo track_id en 2 polls consecutivos < 5min → 1 row con
        # last_seen actualizado, NO una row nueva.
        track_id = f"spotify:track:test_repeat_{int(time.time())}"
        cleanup_spotify_log.append(track_id)
        np = {
            "name": "Repeat", "artist": "R", "album": "Al",
            "state": "playing", "track_id": track_id,
            "duration_ms": 200000, "position_s": 0, "art_url": None,
        }
        with patch.object(sl, "now_playing", return_value=np):
            r1 = sl.record_now_playing()
            time.sleep(0.05)
            r2 = sl.record_now_playing()
        assert r1["updated"] is False
        assert r2["updated"] is True
        recent = sl.recent_tracks_today(20)
        match = [t for t in recent if t["track_id"] == track_id]
        assert len(match) == 1, "Solo debe haber 1 row para el mismo track_id"

    def test_inserts_again_when_track_changes(self, cleanup_spotify_log):
        track1 = f"spotify:track:test_chg1_{int(time.time())}"
        track2 = f"spotify:track:test_chg2_{int(time.time())}"
        cleanup_spotify_log.extend([track1, track2])
        np1 = {
            "name": "T1", "artist": "A", "album": "Al", "state": "playing",
            "track_id": track1, "duration_ms": 100000, "position_s": 0,
            "art_url": None,
        }
        np2 = {**np1, "track_id": track2, "name": "T2"}
        with patch.object(sl, "now_playing", return_value=np1):
            sl.record_now_playing()
        with patch.object(sl, "now_playing", return_value=np2):
            sl.record_now_playing()
        recent = sl.recent_tracks_today(20)
        m1 = [t for t in recent if t["track_id"] == track1]
        m2 = [t for t in recent if t["track_id"] == track2]
        assert len(m1) == 1
        assert len(m2) == 1


class TestRecentTracksToday:
    def test_empty_when_no_data(self, cleanup_spotify_log):
        # No track_ids para limpiar — el test asume que la tabla puede
        # tener datos legítimos del día. Solo verifica que el orden y
        # límite funcionan, no que esté vacía.
        result = sl.recent_tracks_today(0)
        assert result == []

    def test_returns_today_only(self, cleanup_spotify_log):
        # Insert 1 row HOY + 1 row con date manual de ayer.
        # Verificamos que solo el de hoy aparece.
        from rag import _ragvec_state_conn
        track_today = f"spotify:track:test_today_{int(time.time())}"
        track_yesterday = f"spotify:track:test_yest_{int(time.time())}"
        cleanup_spotify_log.extend([track_today, track_yesterday])
        np = {
            "name": "Today", "artist": "A", "album": "Al", "state": "playing",
            "track_id": track_today, "duration_ms": 100000, "position_s": 0,
            "art_url": None,
        }
        with patch.object(sl, "now_playing", return_value=np):
            sl.record_now_playing()
        # Inject yesterday row directly
        with _ragvec_state_conn() as conn:
            now = time.time() - 86400
            conn.execute(
                "INSERT INTO rag_spotify_log "
                "(track_id, name, artist, album, state, duration_ms,"
                " first_seen, last_seen, date) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (track_yesterday, "Yest", "B", "Al", "playing", 100000,
                 now, now, time.strftime("%Y-%m-%d", time.localtime(now))),
            )
        recent = sl.recent_tracks_today(50)
        ids = {t["track_id"] for t in recent}
        assert track_today in ids
        assert track_yesterday not in ids
