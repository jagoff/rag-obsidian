"""Tests for copy-as-implicit-positive (2026-04-22).

When the user copies text from a RAG response, we emit a behavior event
with event='copy'. It's folded into the ranker-vivo priors as a positive
(clicks++, impressions++) — the same arithmetic as an `open`, because
copying content is at least as strong a signal as clicking to read it.

Two sources emit the event:
  - web: document-level Cmd+C listener in web/static/app.js (gates on
    selection length ≥20 chars and being inside a `.turn[data-turn-id]`)
  - CLI: the /copy slash handler in rag.py:~19431 logs behavior after
    pbcopy succeeds, inferring the top vault-relative source as path

Invariants pinned here:
  1. 'copy' is in _BEHAVIOR_POSITIVE → CTR numerator
  2. Endpoint /api/behavior accepts event='copy'
  3. copy events affect the priors snapshot the same way opens do
  4. A copy without a path is a no-op (nothing to attribute)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag


@pytest.fixture
def behavior_env(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)
    rag._behavior_priors_cache = None
    rag._behavior_priors_cache_key = None
    rag._behavior_priors_cache_key_sql = None
    yield tmp_path
    rag._behavior_priors_cache = None
    rag._behavior_priors_cache_key_sql = None


# ── Event-type membership ────────────────────────────────────────────────────


def test_copy_is_a_positive_event():
    """'copy' must live in _BEHAVIOR_POSITIVE so the CTR accumulator
    counts it as a click + impression, not just a denominator bump."""
    assert "copy" in rag._BEHAVIOR_POSITIVE


def test_copy_not_in_negative_or_impression_only():
    """Belt + suspenders: copy is NOT negative, NOT impression-only."""
    assert "copy" not in rag._BEHAVIOR_NEGATIVE
    assert "copy" not in rag._BEHAVIOR_IMPRESSION_ONLY


# ── Aggregator: copy folds into CTR like open ─────────────────────────────────


def test_copy_event_increases_click_prior(behavior_env):
    """A single copy event on a path should push its CTR above the Laplace
    baseline of 1/11 ≈ 0.091 (new path, 1 click, 1 impression = +1 click
    from the copy-counts-as-click rule)."""
    rag.log_behavior_event({
        "source": "cli",
        "event": "copy",
        "query": "qué onda con X",
        "path": "02-Areas/X.md",
        "rank": 1,
    })
    priors = rag._load_behavior_priors()
    ctr = priors.get("click_prior", {}).get("02-Areas/X.md")
    assert ctr is not None
    # (1+1)/(1+10) = 2/11 ≈ 0.181 — higher than the empty baseline 0.091
    assert ctr > 0.15, f"copy event didn't register as click: ctr={ctr}"


def test_copy_and_open_produce_same_ctr(behavior_env):
    """Semantically equivalent: one copy on path A and one open on path B
    should yield the same CTR (both are positive single-clicks)."""
    rag.log_behavior_event({
        "source": "web", "event": "copy",
        "query": "q1", "path": "A.md", "rank": 1,
    })
    rag.log_behavior_event({
        "source": "web", "event": "open",
        "query": "q2", "path": "B.md", "rank": 1,
    })
    priors = rag._load_behavior_priors()
    ctr_a = priors.get("click_prior", {}).get("A.md")
    ctr_b = priors.get("click_prior", {}).get("B.md")
    assert ctr_a == ctr_b, (
        f"copy and open should contribute equal signal: A={ctr_a} B={ctr_b}"
    )


def test_copy_without_path_is_skipped_by_aggregator(behavior_env):
    """Aggregator requires a non-empty path; a copy with empty path
    (shouldn't happen in practice but guard against it) must not raise
    or inflate any bucket."""
    rag.log_behavior_event({
        "source": "web", "event": "copy",
        "query": "q", "path": "", "rank": 1,
    })
    priors = rag._load_behavior_priors()
    # No path → no entry in click_prior.
    assert priors.get("click_prior", {}) == {}


# ── Endpoint accepts event='copy' ────────────────────────────────────────────


_fastapi = pytest.importorskip("fastapi.testclient")
_web_server = pytest.importorskip("web.server")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client():
    return TestClient(_web_server.app)


def test_behavior_endpoint_accepts_copy_event(client, behavior_env):
    """The client-side copy listener posts {event:'copy', ...} — the
    endpoint must accept it (not 400 on unknown-event)."""
    resp = client.post("/api/behavior", json={
        "source": "web",
        "event": "copy",
        "query": "qué onda con Astor",
        "path": "02-Areas/Astor.md",
        "rank": 1,
        "session": "web:abc123",
    })
    assert resp.status_code == 200, resp.text


def test_behavior_endpoint_rejects_unknown_event(client, behavior_env):
    """Sanity check — a bogus event name still gets rejected (catch
    any regression that opens the whitelist too wide)."""
    resp = client.post("/api/behavior", json={
        "source": "web",
        "event": "bogus_event_xyz",
        "path": "x.md",
    })
    assert resp.status_code == 400


def test_behavior_endpoint_rejects_copy_with_cross_source_path(client, behavior_env):
    """Cross-source ids (calendar://, whatsapp://) aren't vault-relative
    paths and the existing traversal guard must reject them."""
    resp = client.post("/api/behavior", json={
        "source": "web",
        "event": "copy",
        "path": "calendar://event-abc",
    })
    # The server checks for "/" at start or ".." — "://" contains "/".
    # calendar://x → split("/") gives ["calendar:", "", "x"] — first
    # non-empty element is "calendar:", not "..", so it falls through
    # to the VAULT_PATH.resolve() relative check which fails.
    assert resp.status_code == 400, (
        f"cross-source path should be rejected, got {resp.status_code}: {resp.text}"
    )
