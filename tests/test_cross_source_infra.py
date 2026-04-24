"""Tests for Phase 1.0 cross-source shared infrastructure.

Covers the pure helpers (`normalize_source`, `source_weight`,
`source_recency_multiplier`) + the `apply_weighted_scores` integration
(source multiplier applied composably on the final score).

End-to-end retrieve() + --source filter tests live in
`tests/test_retrieve_source_filter.py` (next commit).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

import rag


# ── Constants ───────────────────────────────────────────────────────────────

def test_valid_sources_contains_expected():
    # Phase 1 registered 6 sources + Phase 1e/f added contacts + calls
    # + Phase 2 added safari + drive (Google Docs/Sheets/Slides corpus,
    # commit de03db1 2026-04-23). Anchor test so future additions don't
    # silently grow the surface.
    assert rag.VALID_SOURCES == frozenset(
        {"vault", "calendar", "gmail", "whatsapp", "reminders", "messages",
         "contacts", "calls", "safari", "drive"}
    )


def test_source_weights_dict_covers_every_valid_source():
    # Hierarchy: vault > contacts ≈ calendar > reminders > gmail ≈ drive
    #          > safari ≈ calls > whatsapp = messages.
    assert set(rag.SOURCE_WEIGHTS) == rag.VALID_SOURCES
    assert rag.SOURCE_WEIGHTS["vault"] == 1.00
    assert rag.SOURCE_WEIGHTS["contacts"] == 0.95
    assert rag.SOURCE_WEIGHTS["calendar"] == 0.95
    assert rag.SOURCE_WEIGHTS["reminders"] == 0.90
    assert rag.SOURCE_WEIGHTS["gmail"] == 0.85
    # drive: Docs/Sheets/Slides son user-authored + high trust, mismo band que email.
    assert rag.SOURCE_WEIGHTS["drive"] == 0.85
    assert rag.SOURCE_WEIGHTS["safari"] == 0.80
    assert rag.SOURCE_WEIGHTS["calls"] == 0.80
    assert rag.SOURCE_WEIGHTS["whatsapp"] == 0.75
    assert rag.SOURCE_WEIGHTS["messages"] == 0.75


def test_recency_halflife_and_retention_keyed_on_every_source():
    assert set(rag.SOURCE_RECENCY_HALFLIFE_DAYS) == rag.VALID_SOURCES
    assert set(rag.SOURCE_RETENTION_DAYS) == rag.VALID_SOURCES
    # Vault/Calendar opt out of decay; WhatsApp has a short halflife.
    assert rag.SOURCE_RECENCY_HALFLIFE_DAYS["vault"] is None
    assert rag.SOURCE_RECENCY_HALFLIFE_DAYS["calendar"] is None
    assert rag.SOURCE_RECENCY_HALFLIFE_DAYS["whatsapp"] == 30.0


# ── normalize_source ─────────────────────────────────────────────────────────

def test_normalize_source_returns_vault_for_legacy_missing_field():
    assert rag.normalize_source(None) == "vault"
    assert rag.normalize_source("") == "vault"
    assert rag.normalize_source(123) == "vault"  # non-string fallback


def test_normalize_source_accepts_valid():
    assert rag.normalize_source("whatsapp") == "whatsapp"
    assert rag.normalize_source("gmail") == "gmail"


def test_normalize_source_rejects_unknown():
    # Defensive: unknown string → vault (not crash).
    assert rag.normalize_source("facebook") == "vault"
    assert rag.normalize_source("Gmail") == "vault"  # case-sensitive


def test_normalize_source_custom_default():
    assert rag.normalize_source(None, default="whatsapp") == "whatsapp"


# ── source_weight ────────────────────────────────────────────────────────────

def test_source_weight_returns_expected():
    assert rag.source_weight("vault") == 1.00
    assert rag.source_weight("whatsapp") == 0.75


def test_source_weight_unknown_returns_defensive_half():
    # Should not crash — 0.50 picked because it's below every real source.
    assert rag.source_weight("facebook") == 0.50
    assert rag.source_weight("") == 0.50


# ── source_recency_multiplier ───────────────────────────────────────────────

def test_recency_multiplier_returns_one_for_halflife_none():
    # Vault opts out of decay — always 1.0 regardless of age.
    old_ts = time.time() - 1000 * 86400  # 1000 days ago
    assert rag.source_recency_multiplier("vault", old_ts) == 1.0
    assert rag.source_recency_multiplier("calendar", old_ts) == 1.0


def test_recency_multiplier_respects_halflife():
    # WhatsApp halflife is 30d → age=30 should be 0.5, age=60 → 0.25.
    now = time.time()
    age_30d = now - 30 * 86400
    age_60d = now - 60 * 86400
    assert rag.source_recency_multiplier("whatsapp", age_30d, now=now) == pytest.approx(0.5, abs=1e-3)
    assert rag.source_recency_multiplier("whatsapp", age_60d, now=now) == pytest.approx(0.25, abs=1e-3)


def test_recency_multiplier_age_zero_returns_one():
    now = time.time()
    assert rag.source_recency_multiplier("whatsapp", now, now=now) == 1.0


def test_recency_multiplier_accepts_iso_string():
    now = datetime.now()
    created = (now - timedelta(days=30)).isoformat(timespec="seconds")
    mult = rag.source_recency_multiplier("whatsapp", created, now=now.timestamp())
    assert mult == pytest.approx(0.5, abs=1e-2)


def test_recency_multiplier_accepts_iso_z_timezone():
    # Zulu-style Z suffix (used by web/conversation_writer frontmatter).
    now = datetime.now()
    iso_z = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    mult = rag.source_recency_multiplier("whatsapp", iso_z, now=now.timestamp())
    # ISO Z is UTC; test clock is local — the offset may be a few hours but
    # still within the same ~30d bucket, so multiplier ∈ [0.4, 0.6]
    assert 0.4 <= mult <= 0.6


def test_recency_multiplier_missing_ts_is_safe_one():
    # No created_ts → no decay (don't amplify missing-data into bias).
    assert rag.source_recency_multiplier("whatsapp", None) == 1.0
    assert rag.source_recency_multiplier("whatsapp", "") == 1.0
    assert rag.source_recency_multiplier("whatsapp", "not-a-date") == 1.0


def test_recency_multiplier_future_created_ts_clamped_to_one():
    # Slightly wrong clocks / cross-host writes → age is negative; we
    # clamp at age=0 (multiplier=1) rather than producing >1 boosts.
    now = time.time()
    future = now + 86400  # 1 day in the future
    assert rag.source_recency_multiplier("whatsapp", future, now=now) == 1.0


# ── apply_weighted_scores integration ───────────────────────────────────────

def _make_feat(path: str, rerank: float, source: str, created_ts: float | str | None = None) -> dict:
    """Build a minimal feat dict matching compute_feats() output."""
    return {
        "path": path, "note": path.split("/")[-1].replace(".md", ""),
        "rerank": rerank, "recency_raw": 0.0, "tag_hits": 0,
        "fb_pos_cos": 0.0, "fb_neg_cos": 0.0, "ignored": False,
        "has_recency_cue": False, "graph_pagerank": 0.0,
        "click_prior": 0.0, "click_prior_folder": 0.0,
        "click_prior_hour": 0.0, "dwell_score": 0.0,
        "meta": {"file": path, "source": source, "created_ts": created_ts},
    }


def test_apply_weighted_scores_source_multiplier_downweights_nonvault():
    """Two candidates with identical rerank; vault should rank above WA."""
    w = rag.RankerWeights.defaults()
    feats = [
        _make_feat("whatsapp-chat.md", 0.60, "whatsapp"),
        _make_feat("02-Areas/Note.md", 0.60, "vault"),
    ]
    top = rag.apply_weighted_scores(feats, w, k=2)
    # Vault should win: 0.60 * 1.0 = 0.60 vs 0.60 * 0.75 = 0.45 (+ recency, negligible with no created_ts)
    assert top[0]["meta"]["source"] == "vault"
    assert top[1]["meta"]["source"] == "whatsapp"
    # Vault score unchanged, WA score ~0.75 of rerank
    assert top[0]["score"] == pytest.approx(0.60, abs=1e-3)
    assert top[1]["score"] == pytest.approx(0.45, abs=1e-3)


def test_apply_weighted_scores_recency_decays_whatsapp():
    """WA chunk 30 days old should halve (0.75 * 0.5 = 0.375× baseline)."""
    w = rag.RankerWeights.defaults()
    old_ts = time.time() - 30 * 86400
    fresh_ts = time.time()
    feats = [
        _make_feat("wa-fresh.md", 0.80, "whatsapp", fresh_ts),
        _make_feat("wa-old.md",   0.80, "whatsapp", old_ts),
    ]
    top = rag.apply_weighted_scores(feats, w, k=2)
    assert top[0]["path"] == "wa-fresh.md"
    # Fresh: 0.80 * 0.75 * 1.0 = 0.60
    assert top[0]["score"] == pytest.approx(0.60, abs=5e-3)
    # Old: 0.80 * 0.75 * 0.5 = 0.30
    assert top[1]["score"] == pytest.approx(0.30, abs=5e-3)


def test_apply_weighted_scores_vault_default_no_op_for_legacy_meta():
    """Legacy rows without `source` field get treated as vault → no decay,
    no weight. Ensures the v11 collection with no source field doesn't
    regress."""
    w = rag.RankerWeights.defaults()
    # meta has no "source" key at all — simulates pre-Phase-1 rows.
    legacy_feat = {
        "path": "02-Areas/Legacy.md", "note": "Legacy", "rerank": 0.50,
        "recency_raw": 0.0, "tag_hits": 0,
        "fb_pos_cos": 0.0, "fb_neg_cos": 0.0, "ignored": False,
        "has_recency_cue": False, "graph_pagerank": 0.0,
        "click_prior": 0.0, "click_prior_folder": 0.0,
        "click_prior_hour": 0.0, "dwell_score": 0.0,
        "meta": {"file": "02-Areas/Legacy.md"},  # no "source" field
    }
    top = rag.apply_weighted_scores([legacy_feat], w, k=1)
    # Vault baseline: 0.50 * 1.0 * 1.0 = 0.50 (unchanged from pre-Phase-1).
    assert top[0]["score"] == pytest.approx(0.50, abs=1e-3)
