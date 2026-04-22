"""Tests for STT MVP — `rag transcribe` + transcribe_audio() (2026-04-22).

Mocks faster-whisper so the tests don't need the real ~480MB model or
ffmpeg on PATH. Covers:

  1. `transcribe_audio()` — happy path (calls whisper, caches result)
  2. Cache hit on unchanged (path, mtime) — skips whisper entirely
  3. Cache bypass with use_cache=False — forces re-transcription
  4. FileNotFoundError on bogus path
  5. RuntimeError with install hint when faster-whisper not importable
  6. `rag transcribe` CLI command — JSON output, text output, error exits
  7. Cache invalidation on mtime change
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def audio_file(tmp_path):
    """A dummy audio file — faster-whisper is mocked so content doesn't
    matter, but the file needs to exist + have a real mtime."""
    p = tmp_path / "voice-memo.m4a"
    p.write_bytes(b"fake audio content - mock whisper won't care")
    return p


@pytest.fixture
def telemetry_db(tmp_path, monkeypatch):
    """Isolated DB so the rag_audio_transcripts table writes don't
    pollute the real telemetry.db."""
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)
    # Reset the module-level whisper cache so each test loads a fresh mock.
    rag._whisper_model_cache.clear()
    yield tmp_path
    rag._whisper_model_cache.clear()


@pytest.fixture
def mock_whisper():
    """Replaces faster_whisper.WhisperModel with a mock that returns a
    predictable transcript. Segments are an iterator of objects with
    `.text` attributes, mirroring the real API."""
    class _Segment:
        def __init__(self, text):
            self.text = text

    class _Info:
        def __init__(self, language="es", duration=12.4):
            self.language = language
            self.duration = duration

    fake_model = MagicMock()
    fake_model.transcribe.return_value = (
        iter([_Segment(" hola "), _Segment("cómo estás "), _Segment(" che ")]),
        _Info(language="es", duration=12.4),
    )

    with patch.object(rag, "_load_whisper_model", return_value=fake_model):
        yield fake_model


# ── transcribe_audio() ──────────────────────────────────────────────────────


def test_transcribe_audio_happy_path(audio_file, telemetry_db, mock_whisper):
    result = rag.transcribe_audio(audio_file)
    assert result["text"] == "hola cómo estás che"
    assert result["language"] == "es"
    assert result["duration_s"] == 12.4
    assert result["model"] == "small"
    assert result["cached"] is False
    # Mock was called exactly once.
    mock_whisper.transcribe.assert_called_once()


def test_transcribe_audio_caches_result(audio_file, telemetry_db, mock_whisper):
    """First call transcribes; second call hits cache (no mock invocation)."""
    first = rag.transcribe_audio(audio_file)
    assert first["cached"] is False
    assert mock_whisper.transcribe.call_count == 1

    # Second call on the same unchanged file → cache hit.
    second = rag.transcribe_audio(audio_file)
    assert second["cached"] is True
    assert second["text"] == first["text"]
    assert second["language"] == first["language"]
    # Mock was NOT re-invoked.
    assert mock_whisper.transcribe.call_count == 1


def test_transcribe_audio_bypass_cache(audio_file, telemetry_db, mock_whisper):
    """use_cache=False forces re-transcription even with a fresh cache row."""
    rag.transcribe_audio(audio_file)
    assert mock_whisper.transcribe.call_count == 1

    rag.transcribe_audio(audio_file, use_cache=False)
    assert mock_whisper.transcribe.call_count == 2


def test_transcribe_audio_cache_invalidates_on_mtime_change(
    audio_file, telemetry_db, mock_whisper,
):
    """If the file is modified (mtime bumps), cache miss → re-transcribe."""
    rag.transcribe_audio(audio_file)
    assert mock_whisper.transcribe.call_count == 1

    # Simulate file modification by bumping mtime.
    new_mtime = time.time() + 100
    import os
    os.utime(audio_file, (new_mtime, new_mtime))

    rag.transcribe_audio(audio_file)
    assert mock_whisper.transcribe.call_count == 2


def test_transcribe_audio_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        rag.transcribe_audio(tmp_path / "nope.m4a")
    assert "not found" in str(exc_info.value).lower()


def test_transcribe_audio_missing_dep_raises_runtime_error(
    audio_file, telemetry_db, monkeypatch,
):
    """When faster-whisper isn't installed, _load_whisper_model raises
    RuntimeError with a useful install hint. transcribe_audio propagates."""
    # Simulate: _load_whisper_model itself calls faster_whisper import,
    # so we patch the import to fail.
    def _raise_import_error(_name):
        raise RuntimeError(
            "faster-whisper no está instalado. "
            "`uv tool install --reinstall --editable '.[stt]'`"
        )

    monkeypatch.setattr(rag, "_load_whisper_model", _raise_import_error)
    with pytest.raises(RuntimeError) as exc_info:
        rag.transcribe_audio(audio_file)
    assert "faster-whisper" in str(exc_info.value)
    assert "[stt]" in str(exc_info.value)


def test_transcribe_audio_empty_text_not_cached(
    audio_file, telemetry_db,
):
    """Audio that produces empty text (silence, garbage) should NOT write
    a cache row. Otherwise a future fix to the transcription wouldn't
    kick in."""
    class _Info:
        language = "es"
        duration = 3.0

    empty_model = MagicMock()
    empty_model.transcribe.return_value = (iter([]), _Info())

    with patch.object(rag, "_load_whisper_model", return_value=empty_model):
        result = rag.transcribe_audio(audio_file)

    assert result["text"] == ""
    # No row written.
    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM rag_audio_transcripts WHERE audio_path = ?",
            (str(audio_file.resolve()),),
        ).fetchone()
    assert row[0] == 0


# ── `rag transcribe` CLI ────────────────────────────────────────────────────


def test_cli_transcribe_prints_text(audio_file, telemetry_db, mock_whisper):
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["transcribe", str(audio_file)])
    assert result.exit_code == 0
    assert "hola cómo estás che" in result.output


def test_cli_transcribe_json_output(audio_file, telemetry_db, mock_whisper):
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["transcribe", str(audio_file), "--json"])
    assert result.exit_code == 0
    # JSON is one line in output.
    parsed = json.loads(result.output.strip().split("\n")[0])
    assert parsed["text"] == "hola cómo estás che"
    assert parsed["language"] == "es"
    assert parsed["duration_s"] == 12.4
    assert parsed["cached"] is False


def test_cli_transcribe_missing_file_exits_2(telemetry_db):
    """Click's `exists=True` type kicks in → Click's own exit 2 before
    our code runs."""
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["transcribe", "/nope/not-here.m4a"])
    assert result.exit_code == 2
    # Click's error message format.
    assert "does not exist" in result.output.lower() or "no such" in result.output.lower()


def test_cli_transcribe_missing_dep_exits_6(audio_file, telemetry_db, monkeypatch):
    """The custom SystemExit(6) signals "stt extras not installed" so
    scripts can distinguish from a generic error."""
    def _raise(*a, **kw):
        raise RuntimeError("faster-whisper no está instalado. ...")
    monkeypatch.setattr(rag, "transcribe_audio", _raise)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["transcribe", str(audio_file)])
    assert result.exit_code == 6
    # Error message reached the user via console.print().
    assert "faster-whisper" in (result.output or "")


def test_cli_transcribe_forwards_model_and_lang(
    audio_file, telemetry_db, mock_whisper,
):
    """--model and --lang flags must propagate to transcribe_audio."""
    called = {}

    def _capture(path, *, model, language, use_cache):
        called["model"] = model
        called["language"] = language
        called["use_cache"] = use_cache
        return {
            "text": "captured", "language": language, "duration_s": 1.0,
            "model": model, "cached": False, "transcribed_at": time.time(),
        }

    with patch.object(rag, "transcribe_audio", _capture):
        runner = CliRunner()
        result = runner.invoke(rag.cli, [
            "transcribe", str(audio_file),
            "--model", "base", "--lang", "en",
        ])
    assert result.exit_code == 0
    assert called["model"] == "base"
    assert called["language"] == "en"
    assert called["use_cache"] is True


def test_cli_transcribe_no_cache_flag(audio_file, telemetry_db, mock_whisper):
    """--no-cache must propagate as use_cache=False."""
    captured = {}

    def _capture(path, *, model, language, use_cache):
        captured["use_cache"] = use_cache
        return {
            "text": "x", "language": "es", "duration_s": 1.0,
            "model": model, "cached": False, "transcribed_at": time.time(),
        }

    with patch.object(rag, "transcribe_audio", _capture):
        runner = CliRunner()
        result = runner.invoke(rag.cli, [
            "transcribe", str(audio_file), "--no-cache",
        ])
    assert result.exit_code == 0
    assert captured["use_cache"] is False


# ── Schema: the table is in the DDL set ─────────────────────────────────────


def test_rag_audio_transcripts_table_exists(telemetry_db):
    """Belt + suspenders that the new DDL entry registered the table."""
    with rag._ragvec_state_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rag_audio_transcripts'"
        ).fetchone()
    assert row is not None


def test_rag_audio_transcripts_columns(telemetry_db):
    """Column shape contract — if someone reorders PRAGMA-level, this
    test surfaces it before a production write with a different schema
    silently corrupts data."""
    with rag._ragvec_state_conn() as conn:
        cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(rag_audio_transcripts)"
        ).fetchall()]
    assert cols == [
        "audio_path", "mtime", "text", "language",
        "duration_s", "model", "transcribed_at",
    ]
