"""Tests for per-source confidence threshold scaffolding (Phase 1.f).

All values currently equal the global baseline (0.015) — this is a pure
scaffolding change. The tests lock in the helper contract so future
tuning passes don't accidentally break the gate.
"""
from __future__ import annotations

import pytest

import rag


def test_threshold_helper_unknown_source_returns_global():
    assert rag.confidence_threshold_for_source(None) == rag.CONFIDENCE_RERANK_MIN
    assert rag.confidence_threshold_for_source("") == rag.CONFIDENCE_RERANK_MIN
    assert rag.confidence_threshold_for_source("not-a-real-source") == rag.CONFIDENCE_RERANK_MIN


@pytest.mark.parametrize("src", [
    "vault", "calendar", "gmail", "reminders", "whatsapp", "messages",
])
def test_threshold_helper_known_sources_defined(src):
    # Every VALID_SOURCE must have an explicit entry — defensive: missing
    # entries would silently fall through to the baseline, hiding bugs.
    t = rag.confidence_threshold_for_source(src)
    assert isinstance(t, float)
    assert 0 < t < 1


def test_threshold_helper_all_sources_covered():
    # The per-source dict should have an entry for every VALID_SOURCE.
    assert set(rag.CONFIDENCE_RERANK_MIN_PER_SOURCE) == rag.VALID_SOURCES


def test_threshold_scaffolding_currently_equals_baseline():
    """Lock in the 'no-op scaffolding' invariant. Remove this test when
    per-source values diverge from the baseline (they should diverge
    after Phase 1.f calibration)."""
    for src, val in rag.CONFIDENCE_RERANK_MIN_PER_SOURCE.items():
        assert val == rag.CONFIDENCE_RERANK_MIN, (
            f"{src} threshold drifted to {val} — update this test "
            "to reflect calibrated value + document in CLAUDE.md"
        )
