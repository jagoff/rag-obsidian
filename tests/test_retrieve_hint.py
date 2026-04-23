"""Tests for `_build_retrieve_hint` + its wiring into the SSE `status`
event (2026-04-22).

Measured real latency (rag_queries, 30d): web retrieve p90 = 25s. During
that wait, the user saw a generic "buscando…" ticker with no context
about what was being searched. The hint maps the classified intent to
a short action phrase so long waits feel intentional rather than opaque.

Tested surfaces:
  1. `_build_retrieve_hint(intent)` — pure function, deterministic
  2. Hint length ≤48 chars (fits mobile thinking-line in 1 row)
  3. `semantic` + unknown intents → None (no incremental info vs legacy)
  4. SSE contract: when hint is present, `status {stage:"retrieving"}`
     carries `hint` + `intent` extra fields
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_web_server = pytest.importorskip("web.server")


# ── Pure mapping function ───────────────────────────────────────────────────


def test_hint_maps_known_intents():
    """Each of the known intent labels maps to a non-empty hint string."""
    mapping = {
        "count": "Contando",
        "list": "Listando",
        "recent": "recientes",
        "agenda": "agenda",
        "entity_lookup": "persona",
        "comparison": "Comparando",
        "synthesis": "Sintetizando",
        "create": "creación",
    }
    for intent, expected_substr in mapping.items():
        hint = _web_server._build_retrieve_hint(intent)
        assert hint is not None, f"{intent!r} must produce a hint"
        assert expected_substr.lower() in hint.lower(), (
            f"{intent!r} hint {hint!r} should contain {expected_substr!r}"
        )


def test_hint_none_for_semantic():
    """`semantic` is the default retrieve path — the legacy dynamic
    ticker copy is better than a flat generic hint for it."""
    assert _web_server._build_retrieve_hint("semantic") is None


def test_hint_none_for_unknown_intent():
    """Unknown intent strings don't crash — return None so the client
    falls back to legacy ticker copy."""
    assert _web_server._build_retrieve_hint("some_future_intent") is None
    assert _web_server._build_retrieve_hint(None) is None
    assert _web_server._build_retrieve_hint("") is None


def test_hint_length_cap_for_mobile():
    """Each hint must stay ≤ 48 chars so it fits a single line on mobile.
    Longer labels would wrap and fight the seconds-counter for space."""
    for intent in ("count", "list", "recent", "agenda",
                   "entity_lookup", "comparison", "synthesis", "create"):
        hint = _web_server._build_retrieve_hint(intent)
        assert hint is not None
        assert len(hint) <= 48, (
            f"{intent!r} hint {hint!r} is {len(hint)} chars; "
            "mobile thinking-line fits ≤48"
        )


def test_hint_has_no_trailing_period():
    """Ticker's seconds suffix would read awkwardly after a period. The
    hints end with ellipsis instead."""
    for intent in ("count", "list", "recent", "agenda",
                   "entity_lookup", "comparison", "synthesis", "create"):
        hint = _web_server._build_retrieve_hint(intent)
        assert hint is not None
        assert not hint.endswith("."), (
            f"{intent!r} hint ends with period: {hint!r}"
        )


# ── Integration hooks: the /api/chat retrieving emitter ─────────────────────


def test_status_retrieving_carries_hint_and_intent():
    """The endpoint's retrieving-status emitter builds a dict with
    hint + intent when the classifier returns a non-semantic intent.
    This is a source-level check — the full SSE stream is too heavy
    to exercise E2E without a real model."""
    src = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    assert '"stage": "retrieving"' in src
    # The enrichment block (comments dropped) — match key shapes.
    assert "_retrieving_status[\"hint\"]" in src
    assert "_retrieving_status[\"intent\"]" in src
    assert "_build_retrieve_hint(" in src


def test_status_retrieving_omits_hint_when_none():
    """When `_build_retrieve_hint` returns None (semantic / unknown
    intent), the status dict must NOT carry `hint` — clients treat
    missing hint as "use legacy ticker copy". The conditional shape
    is enforced by the `if _early_hint` guards in the emitter."""
    src = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
    # Find the block that builds the status dict
    idx = src.find('_retrieving_status = {"stage": "retrieving"}')
    assert idx >= 0, "retrieving status init not found"
    nearby = src[idx : idx + 800]
    # Both enrichment assignments must be guarded by truthiness
    # checks — an unguarded assignment would always ship the keys
    # (None or not) and break the client's fallback logic.
    assert "if _early_hint:" in nearby, (
        "hint assignment must be guarded by `if _early_hint:`"
    )
    assert "if _early_intent:" in nearby, (
        "intent assignment must be guarded by `if _early_intent:`"
    )
