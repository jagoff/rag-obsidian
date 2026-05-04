"""Tests for per-source confidence threshold (Phase 1.f / W3.9).

Vault stays at the global baseline (0.015). Short-body sources
(whatsapp / calendar / reminders / messages / calls) calibrated to
0.008 because bge-reranker-v2-m3 systematically scores their bodies
in the 0.02-0.10 band (vs vault's 0.10+). gmail / drive intermediate
at 0.010, contacts / safari at 0.012.

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
    """Lock in the W3.9 (2026-04-29) calibrated values. Drift here is
    intentional only when accompanied by an `rag eval` run + CLAUDE.md
    doc update — fail loud otherwise."""
    expected = {
        "vault":     0.015,   # baseline (vault-prose calibrated)
        "whatsapp":  0.008,   # bodies cortos ~143 chars
        "calendar":  0.008,   # eventos cortos
        "reminders": 0.008,   # ítems cortos
        "messages":  0.008,   # iMessage/SMS, mismo patrón que WA
        "calls":     0.008,   # entries muy cortas
        "gmail":     0.010,   # threads más largos, scores intermedios
        "drive":     0.010,   # docs cortos en la fase actual del ingester
        "contacts":  0.012,   # bodies medianos, signal alto
        "safari":    0.012,   # title + body de bookmark/history
        "pillow":    0.015,   # sleep tracker, baseline (no eval data yet)
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
