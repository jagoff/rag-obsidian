"""Feature #4 del 2026-04-23 — agent loop upgrade: planning system prompt +
unproductive-streak detection + inline "thinking" display.

Validates:
- System prompt includes the new planning/reflection rules
- Unproductive streak counter flips after 3 errors and nudges the LLM
- Non-error tool output resets the streak
- The LLM's inline content (plan / reflection) renders in stdout when
  accompanied by tool_calls
"""
from __future__ import annotations

from unittest.mock import MagicMock

from click.testing import CliRunner

import rag


def _fake_resp(tool_calls: list, content: str = "") -> MagicMock:
    msg = MagicMock()
    msg.tool_calls = tool_calls or []
    msg.content = content
    r = MagicMock()
    r.message = msg
    return r


def _fake_tc(name: str, args: dict) -> MagicMock:
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = args
    return tc


def _install_base(monkeypatch):
    monkeypatch.setattr(rag, "warmup_async", lambda: None)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "test-model")


def _mocked_tools(monkeypatch, *, search_result: str = "(ok) hit"):
    captured = {"search_calls": 0}

    def fake_search(query, k=5):
        captured["search_calls"] += 1
        return search_result
    fake_search.__name__ = "_agent_tool_search"
    monkeypatch.setattr(rag, "_agent_tool_search", fake_search)

    def fake_read_note(path):
        return f"(mock) {path}"
    fake_read_note.__name__ = "_agent_tool_read_note"
    monkeypatch.setattr(rag, "_agent_tool_read_note", fake_read_note)

    def fake_list_notes(folder=None, tag=None, limit=30):
        return "(mock) listado"
    fake_list_notes.__name__ = "_agent_tool_list_notes"
    monkeypatch.setattr(rag, "_agent_tool_list_notes", fake_list_notes)

    def fake_weather(location=None):
        return "(mock) clima"
    fake_weather.__name__ = "_agent_tool_weather"
    monkeypatch.setattr(rag, "_agent_tool_weather", fake_weather)

    def fake_propose_write(path, content, rationale=""):
        return f"(mock) propuesta {path}"
    fake_propose_write.__name__ = "_agent_tool_propose_write"
    monkeypatch.setattr(rag, "_agent_tool_propose_write", fake_propose_write)

    return captured


# ── system prompt ────────────────────────────────────────────────────────

def test_agent_system_prompt_has_planning_rule():
    """New prompt asks for a plan in the first message."""
    assert "PLAN" in rag._AGENT_SYSTEM
    assert "self-reflection" in rag._AGENT_SYSTEM.lower() or (
        "no encontré" in rag._AGENT_SYSTEM
    )


def test_agent_system_prompt_has_stop_rule():
    """New prompt tells the LLM to stop after 3 unproductive calls."""
    assert "tres tool calls" in rag._AGENT_SYSTEM.lower() or (
        "tres" in rag._AGENT_SYSTEM.lower()
        and "no aportan" in rag._AGENT_SYSTEM.lower()
    )


# ── unproductive streak ─────────────────────────────────────────────────

def test_unproductive_streak_triggers_nudge(monkeypatch):
    """3 consecutive error tool results → LLM gets a user-role nudge
    telling it to stop and answer honestly."""
    _install_base(monkeypatch)

    # search always returns an error string (unproductive)
    def failing_search(query, k=5):
        return "Error: algo explotó"
    failing_search.__name__ = "_agent_tool_search"
    monkeypatch.setattr(rag, "_agent_tool_search", failing_search)

    # other tools defaults (unused here but required)
    _mocked_tools(monkeypatch)
    monkeypatch.setattr(rag, "_agent_tool_search", failing_search)

    # LLM calls search 3 times in a row, THEN gives a final answer after nudge
    responses = [
        _fake_resp([_fake_tc("_agent_tool_search", {"query": "a"})]),
        _fake_resp([_fake_tc("_agent_tool_search", {"query": "b"})]),
        _fake_resp([_fake_tc("_agent_tool_search", {"query": "c"})]),
        # After the nudge, LLM responds without tool_calls
        _fake_resp([], content="No pude encontrarlo."),
    ]
    fake_client = MagicMock()
    fake_client.chat.side_effect = responses
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["do", "--yes", "--max-iterations", "10", "test streak"],
    )
    assert result.exit_code == 0, result.output
    # Evidence that the nudge fired.
    assert "sin progreso" in result.output
    # Final answer made it through.
    assert "no pude encontrarlo" in result.output.lower()


def test_successful_tool_resets_streak(monkeypatch):
    """A successful tool call (non-error) resets the streak counter
    so subsequent errors don't accumulate across productive calls."""
    _install_base(monkeypatch)

    # Alternating: error, success, error. After 3 iter shouldn't trigger.
    search_calls = [0]
    def alternating_search(query, k=5):
        search_calls[0] += 1
        if search_calls[0] % 2 == 1:
            return "Error: nope"
        return "(ok) found it"
    alternating_search.__name__ = "_agent_tool_search"
    monkeypatch.setattr(rag, "_agent_tool_search", alternating_search)
    _mocked_tools(monkeypatch)
    monkeypatch.setattr(rag, "_agent_tool_search", alternating_search)

    responses = [
        _fake_resp([_fake_tc("_agent_tool_search", {"query": "1"})]),  # Error
        _fake_resp([_fake_tc("_agent_tool_search", {"query": "2"})]),  # OK (resets)
        _fake_resp([_fake_tc("_agent_tool_search", {"query": "3"})]),  # Error
        _fake_resp([_fake_tc("_agent_tool_search", {"query": "4"})]),  # OK (resets)
        _fake_resp([], content="Listo."),
    ]
    fake_client = MagicMock()
    fake_client.chat.side_effect = responses
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["do", "--yes", "--max-iterations", "10", "alternating test"],
    )
    assert result.exit_code == 0, result.output
    # Nudge did NOT fire — success reset the streak.
    assert "sin progreso" not in result.output


def test_sin_resultados_also_counts_as_unproductive(monkeypatch):
    """Not just 'Error:' — semantic search empties also count as
    unproductive for streak tracking."""
    _install_base(monkeypatch)

    def empty_search(query, k=5):
        return "Sin resultados."
    empty_search.__name__ = "_agent_tool_search"
    monkeypatch.setattr(rag, "_agent_tool_search", empty_search)
    _mocked_tools(monkeypatch)
    monkeypatch.setattr(rag, "_agent_tool_search", empty_search)

    responses = [
        _fake_resp([_fake_tc("_agent_tool_search", {"query": "a"})]),
        _fake_resp([_fake_tc("_agent_tool_search", {"query": "b"})]),
        _fake_resp([_fake_tc("_agent_tool_search", {"query": "c"})]),
        _fake_resp([], content="Nada."),
    ]
    fake_client = MagicMock()
    fake_client.chat.side_effect = responses
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["do", "--yes", "--max-iterations", "10", "empty search test"],
    )
    assert result.exit_code == 0, result.output
    assert "sin progreso" in result.output


# ── inline text display ─────────────────────────────────────────────────

def test_inline_plan_text_rendered_when_accompanied_by_tool_calls(monkeypatch):
    """When the LLM writes content + tool_calls together (plan phase),
    the inline text should be printed so the user sees the reasoning."""
    _install_base(monkeypatch)
    _mocked_tools(monkeypatch)

    PLAN = "Mi plan: primero busco, después leo el archivo, después resumo."
    responses = [
        _fake_resp(
            [_fake_tc("_agent_tool_search", {"query": "foo"})],
            content=PLAN,
        ),
        _fake_resp([], content="Listo."),
    ]
    fake_client = MagicMock()
    fake_client.chat.side_effect = responses
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["do", "--yes", "--max-iterations", "4", "plan test"],
    )
    assert result.exit_code == 0, result.output
    # Plan text appears in stdout.
    assert "mi plan" in result.output.lower()


# ── env override ────────────────────────────────────────────────────────

def test_unproductive_cap_env_override(monkeypatch):
    """RAG_AGENT_UNPRODUCTIVE_CAP tunes how many errors in a row trigger
    the nudge."""
    _install_base(monkeypatch)
    monkeypatch.setenv("RAG_AGENT_UNPRODUCTIVE_CAP", "2")  # stricter: 2 is enough

    def failing_search(query, k=5):
        return "Error: nope"
    failing_search.__name__ = "_agent_tool_search"
    monkeypatch.setattr(rag, "_agent_tool_search", failing_search)
    _mocked_tools(monkeypatch)
    monkeypatch.setattr(rag, "_agent_tool_search", failing_search)

    responses = [
        _fake_resp([_fake_tc("_agent_tool_search", {"query": "a"})]),
        _fake_resp([_fake_tc("_agent_tool_search", {"query": "b"})]),
        _fake_resp([], content="Ok paro."),
    ]
    fake_client = MagicMock()
    fake_client.chat.side_effect = responses
    monkeypatch.setattr(rag, "_chat_capped_client", lambda: fake_client)

    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["do", "--yes", "--max-iterations", "10", "cap 2"],
    )
    assert result.exit_code == 0, result.output
    # Nudge fired after 2 errors, not 3.
    assert "sin progreso" in result.output
