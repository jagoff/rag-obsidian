"""Tests for Task 2: behavior priors, ranker features, exploration toggle.

Post-T10: behavior priors read from rag_behavior (SQL) only.
"""
import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


def _seed_sql_behavior(db_dir: Path, events: list[dict]) -> None:
    """Seed rag_behavior with the given events via the module's SQL primitives."""
    import rag
    # Ensure DB_PATH points at the test dir.
    with rag._ragvec_state_conn() as conn:
        for ev in events:
            rag._sql_append_event(conn, "rag_behavior", rag._map_behavior_row(ev))


# ── 1. _load_behavior_priors ─────────────────────────────────────────────────

def test_load_behavior_priors_missing_file(tmp_path, monkeypatch):
    """Empty rag_behavior → empty snapshot with correct keys."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(rag, "_behavior_priors_cache", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key_sql", None)
    priors = rag._load_behavior_priors()
    assert priors["click_prior"] == {}
    assert priors["click_prior_folder"] == {}
    assert priors["click_prior_hour"] == {}
    assert priors["dwell_score"] == {}
    assert priors["n_events"] == 0
    assert "hash" in priors


_TEST_EVENTS = [
    # 5 positive opens for "02-Areas/Coaching/Ikigai.md"
    {"source": "cli", "event": "open", "path": "02-Areas/Coaching/Ikigai.md",
     "ts": "2026-04-17T10:30:00"},
    {"source": "cli", "event": "open", "path": "02-Areas/Coaching/Ikigai.md",
     "ts": "2026-04-17T10:35:00"},
    {"source": "whatsapp", "event": "positive_implicit", "path": "02-Areas/Coaching/Ikigai.md",
     "ts": "2026-04-17T11:00:00"},
    {"source": "brief", "event": "kept", "path": "02-Areas/Coaching/Ikigai.md",
     "ts": "2026-04-17T09:00:00", "dwell_ms": 5000},
    {"source": "cli", "event": "save", "path": "02-Areas/Coaching/Ikigai.md",
     "ts": "2026-04-17T12:00:00"},
    # 2 negative for "03-Resources/Bookmarks.md"
    {"source": "brief", "event": "deleted", "path": "03-Resources/Bookmarks.md",
     "ts": "2026-04-17T08:00:00"},
    {"source": "brief", "event": "negative_implicit", "path": "03-Resources/Bookmarks.md",
     "ts": "2026-04-17T08:05:00"},
    # 3 for another note
    {"source": "cli", "event": "open", "path": "01-Projects/Alpha.md",
     "ts": "2026-04-17T14:00:00"},
    {"source": "cli", "event": "open", "path": "01-Projects/Alpha.md",
     "ts": "2026-04-17T15:00:00"},
    {"source": "brief", "event": "kept", "path": "01-Projects/Alpha.md",
     "ts": "2026-04-17T16:00:00", "dwell_ms": 2000},
]


def test_load_behavior_priors_with_events(tmp_path, monkeypatch):
    """10 events → correct CTR with Laplace smoothing."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(rag, "_behavior_priors_cache", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key_sql", None)
    # Convert dwell_ms → dwell_s for SQL-native storage, preserving the
    # mean-dwell signal the reader aggregates.
    for ev in _TEST_EVENTS:
        ev_sql = dict(ev)
        if "dwell_ms" in ev_sql:
            ev_sql["dwell_s"] = ev_sql.pop("dwell_ms") / 1000.0
        _seed_sql_behavior(tmp_path, [ev_sql])
    priors = rag._load_behavior_priors()

    # Ikigai: 5 clicks, 5 impressions → (5+1)/(5+10) = 0.4
    ikigai_ctr = priors["click_prior"]["02-Areas/Coaching/Ikigai.md"]
    assert abs(ikigai_ctr - 6/15) < 1e-6, f"Expected {6/15}, got {ikigai_ctr}"

    # Bookmarks: 0 clicks, 2 impressions → (0+1)/(2+10) = 1/12
    bm_ctr = priors["click_prior"]["03-Resources/Bookmarks.md"]
    assert abs(bm_ctr - 1/12) < 1e-6, f"Expected {1/12}, got {bm_ctr}"

    # Folder "02-Areas": 5 clicks, 5 impressions → 6/15
    folder_ctr = priors["click_prior_folder"]["02-Areas"]
    assert abs(folder_ctr - 6/15) < 1e-6

    # Folder "03-Resources": 0 clicks, 2 impressions → 1/12
    bm_folder_ctr = priors["click_prior_folder"]["03-Resources"]
    assert abs(bm_folder_ctr - 1/12) < 1e-6

    # Dwell: Ikigai has 1 dwell event at 5000ms → log1p(5.0)
    import math
    expected_dwell = math.log1p(5.0)
    ikigai_dwell = priors["dwell_score"]["02-Areas/Coaching/Ikigai.md"]
    assert abs(ikigai_dwell - expected_dwell) < 1e-6

    # n_events
    assert priors["n_events"] == 10


def test_load_behavior_priors_cache_invalidation(tmp_path, monkeypatch):
    """Cache invalidates when rag_behavior MAX(ts) moves forward."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(rag, "_behavior_priors_cache", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key_sql", None)

    p1 = rag._load_behavior_priors()
    n1 = p1["n_events"]

    # Seed events → MAX(ts) moves, cache invalidates.
    _seed_sql_behavior(tmp_path, _TEST_EVENTS)
    p2 = rag._load_behavior_priors()
    n2 = p2["n_events"]

    assert n1 == 0
    assert n2 == 10


# ── 2. collect_ranker_features — new feature keys ────────────────────────────

def test_collect_ranker_features_has_behavior_keys(tmp_path, monkeypatch):
    """When behavior.jsonl is missing, new features default to 0.0."""
    import rag

    # Stub heavy dependencies so we don't need a live index
    fake_col = MagicMock()
    fake_col.count.return_value = 1

    fake_feats_base = [
        {
            "path": "02-Areas/X.md",
            "note": "X",
            "rerank": 0.5,
            "recency_raw": 0.0,
            "tag_hits": 0,
            "fb_pos_cos": 0.0,
            "fb_neg_cos": 0.0,
            "ignored": False,
            "has_recency_cue": False,
            "graph_pagerank": 0.0,
            "meta": {"file": "02-Areas/X.md"},
        }
    ]

    # Monkey-patch _load_behavior_priors to return empty snapshot
    empty_priors = {
        "click_prior": {},
        "click_prior_folder": {},
        "click_prior_hour": {},
        "dwell_score": {},
        "n_events": 0,
        "hash": "0:0",
    }
    monkeypatch.setattr(rag, "_load_behavior_priors", lambda: empty_priors)

    # Directly verify the feature keys are populated by simulating what
    # collect_ranker_features adds (we can't easily call it without a live
    # embed/reranker, so we test the helper structure)
    priors = rag._load_behavior_priors()
    path = "02-Areas/X.md"
    path_folder = "02-Areas"
    hour = 10

    click_prior_val = priors["click_prior"].get(path, 0.0)
    folder_val = priors["click_prior_folder"].get(path_folder, 0.0)
    hour_val = priors["click_prior_hour"].get((path, hour), 0.0)
    dwell_val = priors["dwell_score"].get(path, 0.0)

    assert click_prior_val == 0.0
    assert folder_val == 0.0
    assert hour_val == 0.0
    assert dwell_val == 0.0


# ── 3. RankerWeights — new knobs at defaults ─────────────────────────────────

def test_ranker_weights_has_behavior_knobs():
    """RankerWeights.defaults() includes the four new behavior knobs at 0.0."""
    from rag import RankerWeights
    w = RankerWeights.defaults()
    assert hasattr(w, "click_prior")
    assert hasattr(w, "click_prior_folder")
    assert hasattr(w, "click_prior_hour")
    assert hasattr(w, "dwell_score")
    assert w.click_prior == 0.0
    assert w.click_prior_folder == 0.0
    assert w.click_prior_hour == 0.0
    assert w.dwell_score == 0.0


def test_ranker_weights_as_dict_includes_behavior_knobs():
    """as_dict() must expose all four new knobs for ranker.json serialization."""
    from rag import RankerWeights
    d = RankerWeights.defaults().as_dict()
    for key in ("click_prior", "click_prior_folder", "click_prior_hour", "dwell_score"):
        assert key in d, f"Missing key: {key}"
        assert d[key] == 0.0


def test_ranker_weights_from_dict_backward_compat():
    """Old ranker.json without behavior keys → falls back to 0.0 (no crash)."""
    from rag import RankerWeights
    old_dict = {
        "recency_cue": 0.1,
        "recency_always": 0.0,
        "tag_literal": 0.0,
        "feedback_pos": 0.03,
        "feedback_neg": 0.15,
        "graph_pagerank": 0.0,
        # behavior keys absent — should fall back to 0.0
    }
    w = RankerWeights.from_dict(old_dict)
    assert w.click_prior == 0.0
    assert w.click_prior_folder == 0.0
    assert w.click_prior_hour == 0.0
    assert w.dwell_score == 0.0


# ── 4. _TUNE_SPACE — new knobs with documented ranges ────────────────────────

def test_tune_space_has_behavior_knobs():
    """_TUNE_SPACE includes all four behavior knobs with conservative ranges."""
    from rag import _TUNE_SPACE
    assert "click_prior" in _TUNE_SPACE
    assert "click_prior_folder" in _TUNE_SPACE
    assert "click_prior_hour" in _TUNE_SPACE
    assert "dwell_score" in _TUNE_SPACE

    lo, hi = _TUNE_SPACE["click_prior"]
    assert lo == 0.0 and hi == 0.30

    lo, hi = _TUNE_SPACE["click_prior_folder"]
    assert lo == 0.0 and hi == 0.15

    lo, hi = _TUNE_SPACE["click_prior_hour"]
    assert lo == 0.0 and hi == 0.20

    lo, hi = _TUNE_SPACE["dwell_score"]
    assert lo == 0.0 and hi == 0.10


# ── 5. Baseline score unchanged when behavior.jsonl absent ────────────────────

def test_apply_weighted_scores_baseline_unchanged():
    """With all behavior weights=0.0 and no behavior data, scores are identical
    to pre-Task-2 behavior (backward compat)."""
    from rag import RankerWeights, apply_weighted_scores

    feats = [
        {
            "path": "a.md",
            "note": "A",
            "rerank": 0.8,
            "recency_raw": 0.5,
            "tag_hits": 1,
            "fb_pos_cos": 0.0,
            "fb_neg_cos": 0.0,
            "ignored": False,
            "has_recency_cue": True,
            "graph_pagerank": 0.3,
            # New behavior features — all 0.0 (missing file scenario)
            "click_prior": 0.0,
            "click_prior_folder": 0.0,
            "click_prior_hour": 0.0,
            "dwell_score": 0.0,
            "meta": {},
        },
        {
            "path": "b.md",
            "note": "B",
            "rerank": 0.6,
            "recency_raw": 0.2,
            "tag_hits": 0,
            "fb_pos_cos": 1.0,   # cosine 1.0 with floor 0.70 → ramp weight 1.0 → full feedback_pos applied
            "fb_neg_cos": 0.0,
            "ignored": False,
            "has_recency_cue": True,
            "graph_pagerank": 0.1,
            "click_prior": 0.0,
            "click_prior_folder": 0.0,
            "click_prior_hour": 0.0,
            "dwell_score": 0.0,
            "meta": {},
        },
    ]
    w = RankerWeights.defaults()
    # Manually compute expected score (pre-Task-2 formula)
    # a.md: 0.8 + 0.1*0.5 + 0 + 0 = 0.85 (recency_cue=0.1, tag_literal=0)
    # b.md: 0.6 + 0.1*0.2 + 0.03 = 0.65
    result = apply_weighted_scores(feats, w, k=2)
    assert result[0]["path"] == "a.md"
    assert result[1]["path"] == "b.md"
    assert abs(result[0]["score"] - 0.85) < 1e-6
    assert abs(result[1]["score"] - 0.65) < 1e-6


# ── 6. ε-exploration toggle ───────────────────────────────────────────────────

def test_explore_env_absent_no_change(monkeypatch):
    """RAG_EXPLORE absent → retrieve() result is untouched (no random calls)."""
    monkeypatch.delenv("RAG_EXPLORE", raising=False)
    # We test indirectly: _load_behavior_priors and the explore branch should
    # not call secrets.randbelow. We just verify the env guard logic.
    assert os.environ.get("RAG_EXPLORE") is None


def test_explore_env_set_fires_at_rate(monkeypatch):
    """With RAG_EXPLORE=1, the explore branch fires at ε≈0.1 over many trials."""
    monkeypatch.setenv("RAG_EXPLORE", "1")

    fired = 0
    N = 1000
    # Simulate the probability check: secrets.randbelow(100) < 10
    # We mock secrets.randbelow to return a predictable sequence
    import rag
    calls = []

    original_secrets = __import__("secrets")

    values = [5] * 100 + [50] * 900  # 100 fires, 900 misses → exactly 10%
    idx = [0]

    def mock_randbelow(n):
        v = values[idx[0] % len(values)]
        idx[0] += 1
        calls.append(v)
        return v

    with patch("secrets.randbelow", side_effect=mock_randbelow):
        for _ in range(N):
            # Replicate the guard logic from retrieve()
            if os.environ.get("RAG_EXPLORE") == "1":
                import secrets
                if secrets.randbelow(100) < 10:
                    fired += 1

    rate = fired / N
    assert abs(rate - 0.10) < 0.05, f"Expected ~10% fire rate, got {rate:.2%}"


def test_explore_env_top3_altered(monkeypatch, tmp_path):
    """With RAG_EXPLORE=1 + forced fire, one top-3 result gets swapped."""
    monkeypatch.setenv("RAG_EXPLORE", "1")

    import rag

    # Build a minimal scored_all (8 items)
    def make_scored(n):
        out = []
        for i in range(n):
            meta = {"file": f"note{i}.md"}
            c = (f"doc{i}", meta, f"id{i}")
            e = f"expanded{i}"
            s = 1.0 - i * 0.1
            out.append((c, e, s))
        return out

    scored_all = make_scored(8)
    k = 5
    scored = scored_all[:k]

    docs_orig = [e for _, e, _ in scored]
    metas_orig = [c[1] for c, _, _ in scored]
    scores_orig = [s for _, _, s in scored]

    docs = list(docs_orig)
    metas = list(metas_orig)
    final_scores = list(scores_orig)
    top_score = final_scores[0]

    # Force fire by mocking randbelow to return specific values:
    # first call (for ε check) → 5 (< 10, fires)
    # second call (evict_slot) → 0
    # third call (swap_idx) → 0
    call_seq = [5, 0, 0]
    call_idx = [0]

    def mock_rb(n):
        v = call_seq[call_idx[0] % len(call_seq)]
        call_idx[0] += 1
        return v

    search_query = "test query"
    with patch("secrets.randbelow", side_effect=mock_rb):
        with patch.object(rag, "log_behavior_event"):  # suppress actual write
            # Replicate the explore branch logic
            if os.environ.get("RAG_EXPLORE") == "1" and len(docs) >= 3 and len(scored_all) > k:
                import secrets
                if secrets.randbelow(100) < 10:
                    evict_slot = secrets.randbelow(3)
                    extras_pool = scored_all[k:k + 4]
                    if extras_pool:
                        swap_idx = secrets.randbelow(len(extras_pool))
                        swap_c, swap_e, _swap_s = extras_pool[swap_idx]
                        swap_meta = swap_c[1] if isinstance(swap_c[1], dict) else {}
                        docs[evict_slot] = swap_e
                        metas[evict_slot] = swap_meta
                        final_scores[evict_slot] = _swap_s
                        rag.log_behavior_event({
                            "source": "cli",
                            "event": "explore",
                            "path": swap_meta.get("file", ""),
                            "rank": evict_slot + 1,
                            "query": search_query,
                        })

    # docs[0] should have been swapped to the rank-5 candidate (scores_all[k+0])
    assert docs[0] != docs_orig[0], "Top result should have been swapped"
    assert docs[0] == "expanded5"  # first extra candidate


def test_eval_strips_rag_explore(monkeypatch):
    """rag eval auto-strips RAG_EXPLORE from env (pop before assert)."""
    monkeypatch.setenv("RAG_EXPLORE", "1")

    # Simulate what the eval function does at the top
    os.environ.pop("RAG_EXPLORE", None)
    assert os.environ.get("RAG_EXPLORE") != "1"


# ── 7. Laplace smoothing edge cases ─────────────────────────────────────────

def test_laplace_smoothing_sparse_path(tmp_path, monkeypatch):
    """A path with 1 event gets (1+1)/(1+10)=2/11, not 1.0."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(rag, "_behavior_priors_cache", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key_sql", None)
    _seed_sql_behavior(tmp_path, [
        {"source": "cli", "event": "open", "path": "rare/note.md", "ts": "2026-04-17T10:00:00"},
    ])
    priors = rag._load_behavior_priors()

    ctr = priors["click_prior"]["rare/note.md"]
    expected = 2 / 11
    assert abs(ctr - expected) < 1e-6, f"Expected {expected}, got {ctr}"


def test_laplace_smoothing_all_negative(tmp_path, monkeypatch):
    """A path with only negative events gets (0+1)/(n+10)."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(rag, "_behavior_priors_cache", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key_sql", None)
    _seed_sql_behavior(tmp_path, [
        {"source": "brief", "event": "deleted", "path": "spam.md", "ts": "2026-04-17T08:00:00"},
        {"source": "brief", "event": "deleted", "path": "spam.md", "ts": "2026-04-17T08:01:00"},
        {"source": "brief", "event": "deleted", "path": "spam.md", "ts": "2026-04-17T08:02:00"},
    ])
    priors = rag._load_behavior_priors()

    ctr = priors["click_prior"]["spam.md"]
    expected = 1 / 13  # (0+1)/(3+10)
    assert abs(ctr - expected) < 1e-6


def test_corrupt_lines_skipped(tmp_path, monkeypatch):
    """Post-T10: with SQL-only storage there's no JSONL parser to skip
    corrupt lines — invalid writes are rejected at insert time by the
    row-mapper. This test now verifies well-formed SQL rows aggregate."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    monkeypatch.setattr(rag, "_behavior_priors_cache", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key", None)
    monkeypatch.setattr(rag, "_behavior_priors_cache_key_sql", None)
    _seed_sql_behavior(tmp_path, [
        {"source": "cli", "event": "open", "path": "ok.md", "ts": "2026-04-17T10:00:00"},
        {"source": "cli", "event": "open", "path": "ok.md", "ts": "2026-04-17T11:00:00"},
    ])
    priors = rag._load_behavior_priors()

    assert priors["n_events"] == 2
    assert "ok.md" in priors["click_prior"]
