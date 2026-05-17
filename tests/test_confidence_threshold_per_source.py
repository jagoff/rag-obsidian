"""Tests for per-source confidence threshold.

Vault stays at the global MLX reranker baseline (0.35). Short-body sources
(whatsapp / calendar / reminders / messages / calls) use a lower gate because
their relevant matches score below prose chunks. Gmail / Drive / finances are
intermediate; contacts / safari sit slightly higher.

The tests lock in the helper contract + the calibrated values so
future tuning passes go through the explicit data-driven channel
(`rag eval` + behavior priors) rather than silent edits.
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


def test_threshold_calibrated_values_w3_9():
    """Lock in the current MLX-scale calibrated values. Drift here is
    intentional only when accompanied by an `rag eval` run + CLAUDE.md
    doc update — fail loud otherwise."""
    expected = {
        "vault":     0.35,
        "obsidian":  0.35,
        "memory":    0.35,
        "whatsapp":  0.20,
        "calendar":  0.20,
        "reminders": 0.20,
        "messages":  0.20,
        "calls":     0.20,
        "gmail":     0.25,
        "drive":     0.25,
        "finances":  0.25,
        "contacts":  0.28,
        "safari":    0.28,
        "pillow":    0.35,
        "health":    0.35,
    }
    assert rag.CONFIDENCE_RERANK_MIN_PER_SOURCE == expected, (
        "Per-source thresholds drifted from W3.9 calibration. "
        "Run `rag eval` to validate before updating expected dict."
    )
    # Vault must NOT regress below the baseline — vault is the dominant
    # source and a refused vault hit is a false negative we can't tolerate.
    assert rag.CONFIDENCE_RERANK_MIN_PER_SOURCE["vault"] == rag.CONFIDENCE_RERANK_MIN
    # Short-body sources must be < baseline (rationale of the calibration).
    for src in ("whatsapp", "calendar", "reminders", "messages", "calls"):
        assert rag.CONFIDENCE_RERANK_MIN_PER_SOURCE[src] < rag.CONFIDENCE_RERANK_MIN, (
            f"{src} should be lower than baseline (short-body)"
        )
