"""Feature #14 del 2026-04-23 — Adaptive k tests.

Validates:
- _compute_adaptive_k: truncates on significant drop, respects min_k,
  never exceeds k_default, handles edge cases (empty, short, negative).
"""
from __future__ import annotations

import pytest

import rag


# ── _compute_adaptive_k ──────────────────────────────────────────────────


def test_empty_returns_default():
    assert rag._compute_adaptive_k([], k_default=5) == 5


def test_short_returns_len():
    assert rag._compute_adaptive_k([1.0], k_default=5, min_k=2) == 1
    assert rag._compute_adaptive_k([1.0, 0.9], k_default=5, min_k=2) == 2


def test_obvious_drop_between_1_and_2_clamps_to_min_k():
    # Score 1.2 then big drop to 0.1: (1.2-0.1)/1.2 = 0.917 >> 0.35.
    # k=1 but min_k=2 clamps.
    scores = [1.2, 0.1, 0.05, 0.02]
    assert rag._compute_adaptive_k(
        scores, k_default=5, min_k=2, gap_ratio=0.35
    ) == 2


def test_drop_between_2_and_3_returns_2():
    # 1.2 → 1.1 (small), 1.1 → 0.2 (big), 0.2 → 0.1 (small).
    # Drop at index 2: (1.1-0.2)/1.1 = 0.818 > 0.35 → k=2.
    scores = [1.2, 1.1, 0.2, 0.1]
    assert rag._compute_adaptive_k(
        scores, k_default=5, min_k=2, gap_ratio=0.35
    ) == 2


def test_no_significant_drop_returns_full_n():
    # Gradual decay — no drop > 0.35.
    scores = [1.0, 0.9, 0.81, 0.73]
    assert rag._compute_adaptive_k(
        scores, k_default=4, min_k=2, gap_ratio=0.35
    ) == 4


def test_n_clamped_to_k_default():
    # If len > k_default, only scans the first k_default.
    scores = [1.0, 0.99, 0.98, 0.97, 0.96, 0.1, 0.1]
    # k_default=4, no drop in first 4 → k=4.
    assert rag._compute_adaptive_k(
        scores, k_default=4, min_k=2, gap_ratio=0.35
    ) == 4


def test_min_k_respected_when_k_default_smaller():
    # If k_default=3 but scores suggest k=1, we clamp to min_k=2.
    scores = [1.0, 0.05, 0.04, 0.03]
    assert rag._compute_adaptive_k(
        scores, k_default=3, min_k=2, gap_ratio=0.35
    ) == 2


def test_negative_prev_skipped():
    """All-negative scores → no clear signal, fallback to k_default."""
    scores = [-0.1, -0.3, -0.5]
    # prev=-0.1 is <=0 → continue. prev=-0.3 also <=0 → continue.
    # Loop ends without finding drop. Returns n.
    assert rag._compute_adaptive_k(
        scores, k_default=3, min_k=2, gap_ratio=0.35
    ) == 3


def test_mixed_sign_works_on_positive_prev():
    """First positive prev, drop on next → truncates."""
    scores = [0.8, -0.1, -0.5]
    # i=1: prev=0.8, cur=-0.1, drop=(0.8-(-0.1))/0.8 = 1.125 >> 0.35 → k=1.
    # min_k=2 clamps → k=2.
    assert rag._compute_adaptive_k(
        scores, k_default=3, min_k=2, gap_ratio=0.35
    ) == 2


def test_gap_ratio_tunable():
    # Aggressive gap_ratio=0.15 flags smaller drops.
    scores = [1.0, 0.85, 0.5, 0.4]
    # i=1: drop = (1.0-0.85)/1.0 = 0.15 → equal to threshold → truncate.
    k = rag._compute_adaptive_k(scores, k_default=4, min_k=1, gap_ratio=0.15)
    assert k == 1


def test_min_k_never_exceeded():
    """min_k=5 but only 3 scores → return 3 (n), not min_k."""
    scores = [1.0, 0.9, 0.8]
    # n = min(len(scores), k_default) = min(3, 5) = 3
    # n <= min_k (3 <= 5) → return n = 3
    k = rag._compute_adaptive_k(scores, k_default=5, min_k=5, gap_ratio=0.35)
    assert k == 3


def test_uniform_scores_no_truncation():
    """Equal scores → no drop → k=len."""
    scores = [0.5, 0.5, 0.5, 0.5]
    assert rag._compute_adaptive_k(
        scores, k_default=4, min_k=2, gap_ratio=0.35
    ) == 4
