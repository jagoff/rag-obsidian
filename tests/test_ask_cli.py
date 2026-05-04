"""Feature #19 del 2026-04-23 — `rag ask` minimalist alias tests.

Validates:
- `rag ask "..."` delegates to query() with default args.
- --quick maps to no_deep=True (multi stays off by default).
- --source passes through.
- --session / --continue passed through.
- --plain disables banner.
- Banner rendered (non-plain mode) with truncated preview.
"""
from __future__ import annotations


from click.testing import CliRunner

import rag


# ── CLI delegation ───────────────────────────────────────────────────────


def test_ask_delegates_to_query(monkeypatch):
    """rag ask "..." → calls query() with the question."""
    captured: dict = {}

    def fake_query(**kwargs):
        captured.update(kwargs)

    # Replace the query callback (Click command func) entirely.
    monkeypatch.setattr(rag.query, "callback", fake_query)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ask", "cuál es mi ikigai?"])
    assert result.exit_code == 0, result.output
    assert captured.get("question") == "cuál es mi ikigai?"


def test_ask_quick_sets_no_deep(monkeypatch):
    captured: dict = {}

    def fake_query(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(rag.query, "callback", fake_query)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ask", "--quick", "test"])
    assert result.exit_code == 0, result.output
    assert captured.get("no_deep") is True
    assert "no_multi" not in captured


def test_ask_default_no_quick_flags_off(monkeypatch):
    captured: dict = {}

    def fake_query(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(rag.query, "callback", fake_query)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ask", "test"])
    assert result.exit_code == 0, result.output
    assert captured.get("no_deep") is False
    assert "no_multi" not in captured


def test_ask_source_passes_through(monkeypatch):
    captured: dict = {}

    def fake_query(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(rag.query, "callback", fake_query)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ask", "--source", "whatsapp", "x"])
    assert result.exit_code == 0
    assert captured.get("source_opt") == "whatsapp"


def test_ask_session_and_continue_pass_through(monkeypatch):
    captured: dict = {}

    def fake_query(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(rag.query, "callback", fake_query)

    runner = CliRunner()
    # --session
    runner.invoke(rag.cli, ["ask", "--session", "abc123", "x"])
    assert captured["session_id"] == "abc123"
    # --continue
    captured.clear()
    runner.invoke(rag.cli, ["ask", "--continue", "x"])
    assert captured["continue_"] is True


def test_ask_plain_mode_skips_banner(monkeypatch):
    captured: dict = {}

    def fake_query(**kwargs):
        captured.update(kwargs)
        click.echo("query-output-here")

    import click
    monkeypatch.setattr(rag.query, "callback", fake_query)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ask", "--plain", "test question"])
    assert result.exit_code == 0
    # Banner character ❯ should NOT appear in plain mode.
    assert "❯" not in result.output
    assert captured.get("plain") is True


def test_ask_non_plain_shows_banner(monkeypatch):
    captured: dict = {}

    def fake_query(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(rag.query, "callback", fake_query)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ask", "test question"])
    assert result.exit_code == 0
    # Banner present.
    assert "❯" in result.output
    assert "test question" in result.output


def test_ask_banner_truncates_long_question(monkeypatch):
    captured: dict = {}

    def fake_query(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(rag.query, "callback", fake_query)

    long_q = "x" * 200
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ask", long_q])
    assert result.exit_code == 0
    # The banner shouldn't contain 200 x's — should be truncated with "...".
    assert "..." in result.output


def test_ask_defaults_match_query_defaults(monkeypatch):
    """Ensure the delegate passes safe defaults for all `query` kwargs."""
    captured: dict = {}

    def fake_query(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(rag.query, "callback", fake_query)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["ask", "q"])
    assert result.exit_code == 0, result.output
    # Values matching query's default options.
    assert captured.get("hyde") is False
    assert captured.get("multi") is False
    assert captured.get("no_auto_filter") is False
    assert captured.get("raw") is False
    assert captured.get("loose") is False
    assert captured.get("force") is False
    assert captured.get("counter") is False
    assert captured.get("critique") is False
    assert captured.get("no_cache") is False
    assert captured.get("vault_scope") is None
    assert captured.get("k") == 5
