"""Tests para el ranker auto-tuning — RankerWeights, feature extraction,
apply_weighted_scores, feedback augmentation, y el sweep del `rag tune`.

Lo que NO cubrimos acá: `collect_ranker_features` end-to-end (requiere ChromaDB
poblada + reranker real; test de integración aparte). Cubrimos la función pura
de scoring (`apply_weighted_scores`), la mecánica de sampling/refinement,
y el parsing de feedback.jsonl.
"""
import json
from pathlib import Path

import pytest

import rag


# ── RankerWeights ────────────────────────────────────────────────────────────


def test_defaults_preserve_legacy_behaviour():
    w = rag.RankerWeights.defaults()
    assert w.recency_cue == 0.1
    assert w.recency_always == 0.0
    assert w.tag_literal == 0.0
    assert w.feedback_pos == rag.FEEDBACK_POSITIVE_BOOST
    assert w.feedback_neg == rag.FEEDBACK_NEGATIVE_PENALTY


def test_roundtrip_save_load(tmp_path, monkeypatch):
    cfg = tmp_path / "ranker.json"
    monkeypatch.setattr(rag, "RANKER_CONFIG_PATH", cfg)
    rag._invalidate_ranker_weights()
    w = rag.RankerWeights(
        recency_cue=0.2, recency_always=0.05, tag_literal=0.08,
        feedback_pos=0.04, feedback_neg=0.20,
    )
    w.save(metadata={"source": "test"})
    assert cfg.is_file()
    data = json.loads(cfg.read_text())
    assert data["weights"]["tag_literal"] == 0.08
    assert data["metadata"]["source"] == "test"
    loaded = rag.RankerWeights.load()
    assert loaded.tag_literal == 0.08
    assert loaded.recency_always == 0.05


def test_load_missing_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RANKER_CONFIG_PATH", tmp_path / "nope.json")
    rag._invalidate_ranker_weights()
    w = rag.RankerWeights.load()
    assert w.as_dict() == rag.RankerWeights.defaults().as_dict()


def test_load_malformed_returns_defaults(tmp_path, monkeypatch):
    cfg = tmp_path / "ranker.json"
    cfg.write_text("{this is not valid json")
    monkeypatch.setattr(rag, "RANKER_CONFIG_PATH", cfg)
    rag._invalidate_ranker_weights()
    w = rag.RankerWeights.load()
    assert w.as_dict() == rag.RankerWeights.defaults().as_dict()


def test_cache_invalidation_on_mtime(tmp_path, monkeypatch):
    import time
    cfg = tmp_path / "ranker.json"
    monkeypatch.setattr(rag, "RANKER_CONFIG_PATH", cfg)
    rag._invalidate_ranker_weights()
    rag.RankerWeights(feedback_pos=0.10).save()
    first = rag.get_ranker_weights()
    assert first.feedback_pos == 0.10
    time.sleep(0.02)
    rag.RankerWeights(feedback_pos=0.20).save()
    second = rag.get_ranker_weights()
    assert second.feedback_pos == 0.20


# ── match_literal_tags ──────────────────────────────────────────────────────


def test_explicit_hashtag_matches():
    # #tag markers accept ≥2 chars (explicit intent); bare tokens require ≥4
    # so "aws" alone is skipped but "#aws" hits.
    vocab = {"finops", "aws", "coaching"}
    assert rag.match_literal_tags("#finops y #aws", vocab) == {"finops", "aws"}
    # Bare "aws" (3 chars) dropped to avoid noise on generic short words.
    assert rag.match_literal_tags("notas de aws", vocab) == set()


def test_bare_token_matches_when_long_enough():
    vocab = {"finops", "ik"}
    assert rag.match_literal_tags("notas sobre finops", vocab) == {"finops"}
    # "ik" tag is 2-char — skipped by the bare-token path (≥4 chars gate).
    assert rag.match_literal_tags("algo de ik", vocab) == set()


def test_accent_insensitive():
    vocab = {"canción"}
    assert rag.match_literal_tags("letra de cancion", vocab) == {"canción"}


def test_empty_vocab_returns_empty():
    assert rag.match_literal_tags("finops", set()) == set()


def test_noise_tokens_ignored():
    vocab = {"finops"}
    # "que" (3 chars) shouldn't accidentally match even if in vocab
    assert rag.match_literal_tags("que pasa hoy", vocab) == set()


# ── apply_weighted_scores ──────────────────────────────────────────────────


def _feat(path, rerank, **kw):
    # fb_pos_cos/fb_neg_cos: cosine similarity to matching golden entry (0.0 = no match).
    # Cosine 1.0 with default floor 0.70 gives ramp weight 1.0, preserving old bool=True math.
    base = {
        "path": path, "note": path.rsplit("/", 1)[-1],
        "rerank": rerank, "recency_raw": 0.0,
        "tag_hits": 0, "fb_pos_cos": 0.0, "fb_neg_cos": 0.0,
        "ignored": False, "has_recency_cue": False,
        "meta": {"file": path, "tags": ""},
    }
    base.update(kw)
    return base


def test_default_weights_sort_by_rerank_only():
    feats = [_feat("a.md", 0.3), _feat("b.md", 0.5), _feat("c.md", 0.1)]
    top = rag.apply_weighted_scores(feats, rag.RankerWeights.defaults(), k=3)
    assert [t["path"] for t in top] == ["b.md", "a.md", "c.md"]


def test_recency_always_lifts_fresh_note():
    # Base ranking: a > b. Give b a big recency_raw; with recency_always=0 it
    # stays second; with recency_always >> the rerank gap, b jumps to top.
    feats = [
        _feat("a.md", 0.5, recency_raw=0.1),
        _feat("b.md", 0.4, recency_raw=1.0),
    ]
    w_off = rag.RankerWeights(recency_always=0.0)
    w_on = rag.RankerWeights(recency_always=0.5)
    assert rag.apply_weighted_scores(feats, w_off, k=1)[0]["path"] == "a.md"
    assert rag.apply_weighted_scores(feats, w_on, k=1)[0]["path"] == "b.md"


def test_tag_boost_changes_order():
    feats = [
        _feat("a.md", 0.5, tag_hits=0),
        _feat("b.md", 0.4, tag_hits=2),
    ]
    w_off = rag.RankerWeights(tag_literal=0.0)
    w_on = rag.RankerWeights(tag_literal=0.3)   # 0.3 * 2 = 0.6 → b wins
    assert rag.apply_weighted_scores(feats, w_off, k=1)[0]["path"] == "a.md"
    assert rag.apply_weighted_scores(feats, w_on, k=1)[0]["path"] == "b.md"


def test_feedback_signals_applied_symmetrically():
    # fb_pos_cos=1.0 with default floor=0.70 → ramp weight = (1.0-0.70)/(1.0-0.70) = 1.0
    # so full feedback_pos/neg is applied, preserving the old bool=True semantics.
    feats = [
        _feat("a.md", 0.5, fb_neg_cos=1.0),    # penalty
        _feat("b.md", 0.4, fb_pos_cos=1.0),    # boost
    ]
    w = rag.RankerWeights(feedback_pos=0.2, feedback_neg=0.3)
    top = rag.apply_weighted_scores(feats, w, k=2)
    # a: 0.5 - 0.3 = 0.2; b: 0.4 + 0.2 = 0.6 → b wins
    assert [t["path"] for t in top] == ["b.md", "a.md"]


def test_ignored_paths_dropped():
    feats = [
        _feat("a.md", 0.9, ignored=True),
        _feat("b.md", 0.1),
    ]
    top = rag.apply_weighted_scores(feats, rag.RankerWeights.defaults(), k=5)
    assert [t["path"] for t in top] == ["b.md"]


def test_recency_cue_only_applies_when_flag_true():
    feats = [_feat("a.md", 0.3, recency_raw=1.0, has_recency_cue=True),
             _feat("b.md", 0.5, recency_raw=0.0, has_recency_cue=False)]
    # cue_weight 0.3, recency_raw 1.0 → a boosted by 0.3 → 0.6 vs b=0.5
    w = rag.RankerWeights(recency_cue=0.3, recency_always=0.0)
    top = rag.apply_weighted_scores(feats, w, k=1)
    assert top[0]["path"] == "a.md"


# ── feedback.jsonl augmentation ─────────────────────────────────────────────


def test_augmentation_extracts_corrective_paths(tmp_path, monkeypatch):
    fb = tmp_path / "feedback.jsonl"
    fb.write_text(
        json.dumps({
            "ts": "2026-04-15", "q": "como instalo claude peers",
            "rating": -1, "paths": ["wrong.md"],
            "corrective_path": "01-Projects/mcp/peers.md",
        }) + "\n"
        + json.dumps({
            "ts": "2026-04-15", "q": "qué es ikigai",
            "rating": 1, "paths": ["02-Areas/ikigai.md"],
            # sin corrective_path → se ignora
        }) + "\n"
        + json.dumps({
            "ts": "2026-04-15", "q": "sesión",
            "scope": "session",
            "corrective_path": "x.md",
        }) + "\n"
        + json.dumps({
            "ts": "2026-04-15", "q": "como instalo claude peers",
            "corrective_path": "02-Areas/mcp-peers.md",  # duplicate query
        }) + "\n"
    )
    monkeypatch.setattr(rag, "FEEDBACK_PATH", fb)
    cases = rag._feedback_augmented_cases()
    assert len(cases) == 1
    assert cases[0]["question"] == "como instalo claude peers"
    assert cases[0]["expected"] == ["01-Projects/mcp/peers.md"]


def test_augmentation_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "FEEDBACK_PATH", tmp_path / "nope.jsonl")
    assert rag._feedback_augmented_cases() == []


def test_augmentation_skips_short_queries(tmp_path, monkeypatch):
    fb = tmp_path / "feedback.jsonl"
    fb.write_text(
        json.dumps({"q": "hi", "corrective_path": "x.md"}) + "\n"
    )
    monkeypatch.setattr(rag, "FEEDBACK_PATH", fb)
    assert rag._feedback_augmented_cases(min_len=4) == []


# ── _score_case + _aggregate ────────────────────────────────────────────────


def test_score_case_hit_and_rr():
    feats = [_feat("right.md", 0.9), _feat("wrong.md", 0.5)]
    hit, rr, rec = rag._score_case(
        feats, {"right.md"}, rag.RankerWeights.defaults(), k=5,
    )
    assert hit is True
    assert rr == 1.0   # rank 1
    assert rec == 1.0


def test_score_case_miss():
    feats = [_feat("a.md", 0.9), _feat("b.md", 0.5)]
    hit, rr, rec = rag._score_case(
        feats, {"c.md"}, rag.RankerWeights.defaults(), k=5,
    )
    assert hit is False
    assert rr == 0.0
    assert rec == 0.0


def test_score_case_partial_recall():
    feats = [
        _feat("a.md", 0.9), _feat("b.md", 0.7),
        _feat("c.md", 0.5), _feat("d.md", 0.3),
    ]
    expected = {"a.md", "c.md"}
    hit, rr, rec = rag._score_case(
        feats, expected, rag.RankerWeights.defaults(), k=4,
    )
    assert hit is True
    assert rr == 1.0    # a.md at rank 1
    assert rec == 1.0   # both found


def test_aggregate_averages():
    cm = [(True, 1.0, 1.0), (True, 0.5, 0.5), (False, 0.0, 0.0)]
    agg = rag._aggregate(cm)
    assert agg["hit"] == pytest.approx(2 / 3)
    assert agg["mrr"] == pytest.approx(0.5)
    assert agg["recall"] == pytest.approx(0.5)


def test_objective_formula_matches_spec():
    # Spec: hit + 0.5 * MRR. Both are in [0, 1]; objective ranges [0, 1.5].
    assert rag._objective({"hit": 1.0, "mrr": 0.0}) == 1.0
    assert rag._objective({"hit": 0.0, "mrr": 1.0}) == 0.5
    assert rag._objective({"hit": 0.8, "mrr": 0.4}) == pytest.approx(1.0)


# ── sweep mechanics ─────────────────────────────────────────────────────────


def test_sample_weights_deterministic_with_seed():
    import random
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    w1 = rag._sample_weights(rng1)
    w2 = rag._sample_weights(rng2)
    assert w1.as_dict() == w2.as_dict()


def test_sample_weights_respects_search_space():
    import random
    rng = random.Random(0)
    for _ in range(50):
        w = rag._sample_weights(rng)
        for dim, (lo, hi) in rag._TUNE_SPACE.items():
            val = getattr(w, dim)
            assert lo <= val <= hi


def test_coordinate_refine_improves_or_stays():
    # Cases where tag_literal > 0 strictly improves hit
    feats_good = [_feat("right.md", 0.4, tag_hits=3),
                  _feat("wrong.md", 0.5, tag_hits=0)]
    feats_bad = [_feat("c.md", 0.9, tag_hits=0)]
    cases = [
        {"feats": feats_good, "expected": {"right.md"}},
        {"feats": feats_bad, "expected": {"c.md"}},
    ]
    for c in cases:
        feats = c["feats"]; expected = c["expected"]
        c["metrics"] = lambda w, f=feats, e=expected: rag._score_case(f, e, w, 5)

    start = rag.RankerWeights.defaults()  # tag_literal=0 → misses right.md
    start_score = rag._objective(rag._aggregate([c["metrics"](start) for c in cases]))
    refined = rag._coordinate_refine(start, cases, k=5, steps=10)
    refined_score = rag._objective(rag._aggregate([c["metrics"](refined) for c in cases]))
    assert refined_score >= start_score
    # With enough steps, should find tag_literal > 0 since that wins the "good" case.
    assert refined.tag_literal > 0.0


def test_log_tune_event_append(tmp_path, monkeypatch):
    """Post-T10: tune log writes to rag_tune (SQL)."""
    import sqlite3
    log = tmp_path / "tune.jsonl"
    monkeypatch.setattr(rag, "TUNE_LOG_PATH", log)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    rag._log_tune_event({"foo": "bar"})
    rag._log_tune_event({"foo": "baz"})
    conn = sqlite3.connect(str(tmp_path / "ragvec.db"))
    conn.row_factory = sqlite3.Row
    try:
        rows = list(conn.execute(
            "SELECT ts, extra_json FROM rag_tune ORDER BY id"
        ).fetchall())
    finally:
        conn.close()
    assert len(rows) == 2
    extra = json.loads(rows[0]["extra_json"])
    assert extra["foo"] == "bar"
    assert rows[0]["ts"]
