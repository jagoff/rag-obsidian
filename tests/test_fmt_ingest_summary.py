"""Tests for the minimal `_fmt_ingest_summary` helper.

Pure function, no IO — we exercise the grammar directly and confirm
the user-facing guarantees:
  - zero deltas are suppressed (no `· +0 · -0` noise)
  - `total` is optional (calendar/gmail don't expose one)
  - `dry_run=True` prefixes `dry · ` exactly once
  - `extra` slot renders between deltas and duration
  - duration rounds to 2 decimal places
"""
from __future__ import annotations

import rag


def test_minimal_no_deltas_no_total():
    # When everything is zero the only remaining tokens are the name
    # and the duration — the one-liner should collapse cleanly.
    assert rag._fmt_ingest_summary("calendar", duration_s=0.0) == "calendar · 0.0s"


def test_minimal_no_deltas_with_total():
    assert rag._fmt_ingest_summary(
        "contacts", total=588, duration_s=0.01,
    ) == "contacts: 588 · 0.01s"


def test_indexed_only_suppresses_deleted_zero():
    # +N without a -0 companion.
    assert rag._fmt_ingest_summary(
        "reminders", total=45, indexed=3, duration_s=0.8,
    ) == "reminders: 45 · +3 · 0.8s"


def test_deleted_only_suppresses_indexed_zero():
    assert rag._fmt_ingest_summary(
        "reminders", total=45, deleted=2, duration_s=0.8,
    ) == "reminders: 45 · -2 · 0.8s"


def test_both_deltas():
    assert rag._fmt_ingest_summary(
        "calls", total=36, indexed=3, deleted=1, duration_s=10.8,
    ) == "calls: 36 · +3 · -1 · 10.8s"


def test_extra_renders_between_deltas_and_duration():
    assert rag._fmt_ingest_summary(
        "calls", total=36, indexed=3, extra="1 missed", duration_s=10.8,
    ) == "calls: 36 · +3 · 1 missed · 10.8s"


def test_extra_without_deltas():
    # Example: calendar in bootstrap mode, no events changed.
    assert rag._fmt_ingest_summary(
        "calendar", extra="bootstrap", duration_s=2.3,
    ) == "calendar · bootstrap · 2.3s"


def test_dry_run_prefix():
    assert rag._fmt_ingest_summary(
        "contacts", total=588, indexed=588, duration_s=0.01, dry_run=True,
    ) == "dry · contacts: 588 · +588 · 0.01s"


def test_dry_run_with_empty_extra_does_not_add_blank_segment():
    # Falsy extra must not produce `· ·` or a stray trailing separator.
    out = rag._fmt_ingest_summary(
        "calls", total=0, indexed=0, extra="", duration_s=0.0,
    )
    assert "· ·" not in out
    assert out == "calls: 0 · 0.0s"


def test_duration_rounds_to_two_decimals():
    assert rag._fmt_ingest_summary(
        "whatsapp", total=1, duration_s=1.23456,
    ) == "whatsapp: 1 · 1.23s"


def test_no_color_markup_in_output():
    # The whole point of the minimal rewrite: no rich markup leaks
    # into the string — readers without Rich see the same thing.
    out = rag._fmt_ingest_summary(
        "calls", total=36, indexed=3, extra="5 missed",
        deleted=1, duration_s=10.8, dry_run=True,
    )
    for banned in ("[bold]", "[/bold]", "[green]", "[yellow]",
                    "[dim]", "[red]", "[/]"):
        assert banned not in out
