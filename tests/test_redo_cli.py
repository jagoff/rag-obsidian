"""Tests for the `/redo [hint]` CLI slash command (2026-04-22).

The handler lives inline in `chat()` around rag.py:~19606 (same pattern
as `/inbox`, `/open`, `/copy` — direct if-elif dispatch). Because the
full loop is complex to exercise E2E (readline + LLM + streaming), we
test the *parsing + base-question augmentation logic* here the same
way test_corrective_path_chat.py does for the feedback flow.

Parallel to the web /redo implemented in commit 904bc8b — same shape:
  - /redo on its own → re-ask the last question verbatim
  - /redo <hint>    → append "— enfocá en: <hint>" to the base question

Edge cases:
  - /redo when last_question is empty → error message, no action
  - /redo with trailing whitespace → treated as /redo with no hint
  - consecutive redos with hints → accumulate (documented tradeoff)

The handler and its help entry must stay in sync: the test in
test_chat_slash_commands.py::test_chat_help_covers_all_slash_commands
catches the help-drift half; these tests catch the parsing half.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


def _simulate_redo_handler(question: str, last_question: str) -> tuple[str, str | None, bool]:
    """Replicate the parsing logic from the /redo handler at rag.py:~19606.

    Returns (effective_question, hint, handled). `handled=False` means
    the handler emitted the "no previous answer" error path and the
    caller should continue to the next iteration without invoking
    retrieve/LLM. When handled=True, `effective_question` is what the
    retrieve pipeline sees and `hint` is what we'd log to telemetry.

    Kept in sync with rag.py: if the format string ("— enfocá en: ")
    changes there, update the assertion in these tests too.
    """
    if not (question == "/redo" or question.startswith("/redo ")):
        raise ValueError("not a /redo command")
    if not last_question:
        return ("", None, False)
    hint = question[len("/redo"):].strip()
    if hint:
        return (f"{last_question} — enfocá en: {hint}", hint, True)
    return (last_question, None, True)


# ── /redo basic parsing ──────────────────────────────────────────────────────


def test_redo_without_hint_reuses_last_question():
    """Bare `/redo` re-asks the original question verbatim."""
    out, hint, handled = _simulate_redo_handler("/redo",
                                                "¿cuándo es el cumple de Astor?")
    assert handled is True
    assert out == "¿cuándo es el cumple de Astor?"
    assert hint is None


def test_redo_with_trailing_space_treated_as_bare():
    """`/redo ` (just a trailing space) should behave like `/redo`."""
    out, hint, handled = _simulate_redo_handler("/redo ",
                                                "¿cuándo es el cumple de Astor?")
    assert handled is True
    assert out == "¿cuándo es el cumple de Astor?"
    assert hint is None


def test_redo_with_hint_appends_soft_steer():
    """`/redo <hint>` concatenates the hint with the canonical separator.
    This MUST match the web /redo format (web/server.py:~3899) so the
    behaviour is identical across surfaces."""
    out, hint, handled = _simulate_redo_handler(
        "/redo enfocate en el aniversario", "¿qué onda Astor?",
    )
    assert handled is True
    assert hint == "enfocate en el aniversario"
    # Separator format is the contract — web and CLI share it.
    assert " — enfocá en: " in out
    assert out == "¿qué onda Astor? — enfocá en: enfocate en el aniversario"


def test_redo_multiword_hint():
    """Multi-word hints (spaces inside the argument) are preserved."""
    out, hint, handled = _simulate_redo_handler(
        "/redo si hay mensaje y cuál es el tono",
        "¿hablé con María ayer?",
    )
    assert handled is True
    assert hint == "si hay mensaje y cuál es el tono"
    assert "si hay mensaje y cuál es el tono" in out


# ── /redo guard: no previous question ────────────────────────────────────────


def test_redo_without_last_question_returns_not_handled():
    """First-turn `/redo` should surface the "no hay respuesta previa"
    path — handled=False means the caller should `continue` without
    invoking retrieve/LLM. This prevents the placeholder `last_question=""`
    from being sent to the retriever (would parse as empty query)."""
    out, hint, handled = _simulate_redo_handler("/redo", "")
    assert handled is False
    assert out == ""
    assert hint is None


def test_redo_with_hint_but_no_last_question_also_blocked():
    """Same guard applies even if the user typed a hint — without a
    base question to regenerate, the hint is meaningless."""
    out, hint, handled = _simulate_redo_handler("/redo enfocate en X", "")
    assert handled is False


# ── consecutive redos: the documented accumulation tradeoff ──────────────────


def test_consecutive_redos_accumulate_hints_documented():
    """Redo N acts on `last_question`, which includes any prior hint.

    This is a DOCUMENTED tradeoff (see the comment block at rag.py:~19587).
    Two redos with hints chain as:
      turn 1: "q"  →  last_q = "q"
      /redo H1   →  last_q = "q — enfocá en: H1"
      /redo H2   →  last_q = "q — enfocá en: H1 — enfocá en: H2"

    The justification is simplicity — the alternative requires extra
    mutable state (`_redo_base_q` that survives the turn), and in
    practice consecutive hint-redos are rare. The user can `/cls` to
    reset. This test pins that behaviour so nobody "fixes" it without
    realising it was deliberate.
    """
    # First redo with hint.
    out1, _, _ = _simulate_redo_handler(
        "/redo enfocá en el aniversario", "¿qué onda Astor?",
    )
    assert out1 == "¿qué onda Astor? — enfocá en: enfocá en el aniversario"

    # Second redo uses `out1` as the new last_question — hint accumulates.
    out2, _, _ = _simulate_redo_handler("/redo más técnico", out1)
    assert "— enfocá en: enfocá en el aniversario" in out2
    assert "— enfocá en: más técnico" in out2
    # The chain is visible. If the user is surprised they can `/cls`.


# ── /redo appears in help (regression guard) ─────────────────────────────────


def test_redo_documented_in_help():
    """The slash command must be listed in `/help` so users discover it.
    Duplicates the coverage loop in test_chat_slash_commands.py but
    pinned here explicitly so a /redo-focused test run catches the drift."""
    out = rag._chat_help_text()
    assert "/redo" in out, "/redo must appear in `/help` — keep docs in sync"
    assert "pista" in out.lower() or "hint" in out.lower(), (
        "help should describe the optional pista/hint argument"
    )
