"""Tests for the `rag log --silent-errors` CLI — exposes the _silent_log
helper's output as a human-readable table / summary.

See rag.py:_render_silent_errors_log + the `log` command.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import rag


@pytest.fixture
def seeded_silent_errors(tmp_path, monkeypatch):
    """Point SILENT_ERRORS_LOG_PATH at a tmp file populated with three
    entries spanning two `where` buckets and two exception types."""
    p = tmp_path / "silent_errors.jsonl"
    lines = [
        {"ts": "2026-04-20T10:00:00", "where": "feedback_golden_embed",
         "exc_type": "ConnectionError", "exc": "cannot reach ollama"},
        {"ts": "2026-04-20T10:01:00", "where": "feedback_golden_embed",
         "exc_type": "ConnectionError", "exc": "cannot reach ollama"},
        {"ts": "2026-04-20T10:02:00", "where": "context_cache_load",
         "exc_type": "JSONDecodeError", "exc": "Expecting value"},
    ]
    p.write_text("\n".join(json.dumps(e) for e in lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(rag, "SILENT_ERRORS_LOG_PATH", p)
    return p


def test_render_silent_errors_log_list_mode(seeded_silent_errors, capsys):
    """Default (non-summary) mode prints one row per entry, tail N."""
    rag._render_silent_errors_log(n=10, summary=False)
    out = capsys.readouterr().out
    # All three entries should show up.
    assert "feedback_golden_embed" in out
    assert "context_cache_load" in out
    assert "ConnectionError" in out
    assert "JSONDecodeError" in out
    assert "cannot reach ollama" in out


def test_render_silent_errors_log_summary_mode(seeded_silent_errors, capsys):
    """With summary=True, groups by (where, exc_type) with counts."""
    rag._render_silent_errors_log(n=10, summary=True)
    out = capsys.readouterr().out
    # feedback_golden_embed × ConnectionError appears twice → count=2
    assert "feedback_golden_embed" in out
    assert "context_cache_load" in out
    # Count column is rendered to the left of the where
    # (allow for ANSI/whitespace variations from Rich).
    assert "2" in out  # 2 connection errors
    assert "1" in out  # 1 json decode error


def test_render_silent_errors_log_respects_n(seeded_silent_errors, capsys):
    """n caps the number of rows in list mode (keeping the newest)."""
    rag._render_silent_errors_log(n=1, summary=False)
    out = capsys.readouterr().out
    # Only the newest (context_cache_load / JSONDecodeError) should render.
    assert "JSONDecodeError" in out
    # The two older feedback_golden_embed entries must be trimmed.
    # Rich tables can wrap, so we check the cell content explicitly.
    assert "ConnectionError" not in out


def test_render_silent_errors_log_missing_file(tmp_path, monkeypatch, capsys):
    """If silent_errors.jsonl doesn't exist, print a friendly hint
    instead of crashing — this is the common case on a clean install."""
    missing = tmp_path / "never_created.jsonl"
    monkeypatch.setattr(rag, "SILENT_ERRORS_LOG_PATH", missing)
    rag._render_silent_errors_log(n=5, summary=False)
    out = capsys.readouterr().out
    assert "No hay silent_errors.jsonl" in out


def test_render_silent_errors_log_empty_file(tmp_path, monkeypatch, capsys):
    """Empty (zero-byte) file: also prints a friendly hint, no crash."""
    p = tmp_path / "silent_errors.jsonl"
    p.write_text("", encoding="utf-8")
    monkeypatch.setattr(rag, "SILENT_ERRORS_LOG_PATH", p)
    rag._render_silent_errors_log(n=5, summary=False)
    out = capsys.readouterr().out
    assert "vacío" in out or "empty" in out.lower()


def test_render_silent_errors_log_malformed_lines_are_dropped(
    tmp_path, monkeypatch, capsys
):
    """A single malformed JSON line in the middle must not abort the
    render — the log is append-only and partial writes can happen on
    disk-full or process-kill mid-fsync."""
    p = tmp_path / "silent_errors.jsonl"
    good = json.dumps({
        "ts": "2026-04-20T10:00:00", "where": "ok_row",
        "exc_type": "RuntimeError", "exc": "boom",
    })
    p.write_text(
        good + "\n"
        + "not json at all\n"                      # malformed middle
        + "{invalid but json-ish\n"                # malformed middle
        + good + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag, "SILENT_ERRORS_LOG_PATH", p)
    rag._render_silent_errors_log(n=10, summary=False)
    out = capsys.readouterr().out
    # Good rows survived.
    assert "ok_row" in out
    assert "RuntimeError" in out
