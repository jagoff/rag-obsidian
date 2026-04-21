"""Tests for `log_impressions()` + impression events folded into behavior priors.

Before this change, CTR was (interactions_as_clicks+1)/(interactions+10), which
inflated already-active paths since non-interactions didn't count as
impressions. `retrieve()` now emits one `impression` event per surfaced path,
and `_compute_behavior_priors_from_rows` treats them as denominator-only.
"""
from __future__ import annotations

import time

import pytest

import rag


@pytest.fixture
def behavior_env(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    # Ensure tables exist for this isolated DB.
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)
    rag._impression_last_seen.clear()
    yield tmp_path
    rag._impression_last_seen.clear()


def _rows(tmp_path) -> list[dict]:
    with rag._ragvec_state_conn() as conn:
        cur = conn.execute(
            "SELECT ts, source, event, query, path, rank FROM rag_behavior "
            "ORDER BY id ASC"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def test_log_impressions_writes_one_row_per_path(behavior_env):
    rag.log_impressions("tell me about alpha", ["a.md", "b.md", "c.md"], source="cli")
    rows = _rows(behavior_env)
    events = [r for r in rows if r["event"] == "impression"]
    assert len(events) == 3
    # Ranks start at 1.
    assert [r["rank"] for r in events] == [1, 2, 3]
    assert {r["path"] for r in events} == {"a.md", "b.md", "c.md"}


def test_log_impressions_empty_is_noop(behavior_env):
    rag.log_impressions("", ["a.md"])
    rag.log_impressions("q", [])
    assert _rows(behavior_env) == []


def test_log_impressions_caps_batch_at_cap(behavior_env):
    paths = [f"n{i}.md" for i in range(20)]
    rag.log_impressions("many", paths, cap=5)
    rows = _rows(behavior_env)
    assert len(rows) == 5
    assert [r["path"] for r in rows] == paths[:5]


def test_log_impressions_throttles_repeat_query(behavior_env):
    """Same (query, top1_path) within 30s must NOT double-log."""
    rag.log_impressions("repeat", ["a.md", "b.md"])
    rag.log_impressions("repeat", ["a.md", "c.md"])  # same top1 → throttled
    rows = _rows(behavior_env)
    assert len(rows) == 2, f"expected 2 rows, got {len(rows)}: {rows}"


def test_log_impressions_not_throttled_across_queries(behavior_env):
    rag.log_impressions("query-A", ["a.md"])
    rag.log_impressions("query-B", ["a.md"])  # different query → not throttled
    rows = _rows(behavior_env)
    assert len(rows) == 2


def test_impression_events_count_as_denominator_only(behavior_env):
    """Path with 10 impressions + 0 clicks must have low CTR; path with 5/5 must be higher."""
    # p1: 5 impressions, 0 clicks → low CTR
    for i in range(5):
        rag.log_impressions(f"q{i}", ["p1.md"])  # distinct queries → not throttled
    # p2: 5 impressions + 5 opens → high CTR
    for i in range(5):
        rag.log_impressions(f"q{i}", ["p2.md"])
        rag.log_behavior_event({
            "source": "cli", "event": "open", "query": f"q{i}", "path": "p2.md",
        })
    # Recompute priors from the fresh rows.
    rag._behavior_priors_cache = None
    rag._behavior_priors_cache_key = None
    rag._behavior_priors_cache_key_sql = None
    priors = rag._load_behavior_priors()
    ctr_p1 = priors.get("click_prior", {}).get("p1.md")
    ctr_p2 = priors.get("click_prior", {}).get("p2.md")
    assert ctr_p1 is not None
    assert ctr_p2 is not None
    # p1: (0+1)/(5+10) = 1/15 ≈ 0.067; p2: (5+1)/(10+10) = 6/20 = 0.30
    assert ctr_p2 > ctr_p1, f"click prior not strictly higher for clicked path: p1={ctr_p1} p2={ctr_p2}"
    assert ctr_p1 < 0.10
    assert ctr_p2 > 0.20
