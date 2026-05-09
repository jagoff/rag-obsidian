"""Tests para rag/llm_judge.py — LLM-as-judge condicional post-rerank."""

import pytest

from rag.llm_judge import (
    _alpha,
    _enabled,
    _parse_judge_response,
    _trigger_threshold,
    blend_scores,
    judge_and_blend,
    should_fire_judge,
)


# ── _enabled ──────────────────────────────────────────────────────────────────


def test_enabled_off_by_default(monkeypatch):
    monkeypatch.delenv("RAG_LLM_JUDGE", raising=False)
    assert _enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "TRUE", "Yes"])
def test_enabled_truthy_values(monkeypatch, val):
    monkeypatch.setenv("RAG_LLM_JUDGE", val)
    assert _enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "", "2"])
def test_enabled_falsy_values(monkeypatch, val):
    monkeypatch.setenv("RAG_LLM_JUDGE", val)
    assert _enabled() is False


# ── _trigger_threshold / _alpha ───────────────────────────────────────────────


def test_trigger_threshold_default(monkeypatch):
    monkeypatch.delenv("RAG_LLM_JUDGE_THRESHOLD", raising=False)
    assert _trigger_threshold() == pytest.approx(0.5)


def test_trigger_threshold_custom(monkeypatch):
    monkeypatch.setenv("RAG_LLM_JUDGE_THRESHOLD", "0.3")
    assert _trigger_threshold() == pytest.approx(0.3)


def test_trigger_threshold_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("RAG_LLM_JUDGE_THRESHOLD", "not-a-float")
    assert _trigger_threshold() == pytest.approx(0.5)


def test_alpha_default(monkeypatch):
    monkeypatch.delenv("RAG_LLM_JUDGE_ALPHA", raising=False)
    assert _alpha() == pytest.approx(0.5)


def test_alpha_clamped_below_zero(monkeypatch):
    monkeypatch.setenv("RAG_LLM_JUDGE_ALPHA", "-1.0")
    assert _alpha() == pytest.approx(0.0)


def test_alpha_clamped_above_one(monkeypatch):
    monkeypatch.setenv("RAG_LLM_JUDGE_ALPHA", "2.5")
    assert _alpha() == pytest.approx(1.0)


# ── should_fire_judge ─────────────────────────────────────────────────────────


def test_should_fire_judge_off_when_disabled(monkeypatch):
    monkeypatch.delenv("RAG_LLM_JUDGE", raising=False)
    assert should_fire_judge(top_score=0.1, n_candidates=10) is False


def test_should_fire_judge_off_when_score_too_high(monkeypatch):
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    monkeypatch.setenv("RAG_LLM_JUDGE_THRESHOLD", "0.5")
    assert should_fire_judge(top_score=0.6, n_candidates=10) is False


def test_should_fire_judge_off_when_too_few_candidates(monkeypatch):
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    monkeypatch.setenv("RAG_LLM_JUDGE_MIN_CANDIDATES", "5")
    assert should_fire_judge(top_score=0.2, n_candidates=3) is False


def test_should_fire_judge_on_when_conditions_met(monkeypatch):
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    monkeypatch.setenv("RAG_LLM_JUDGE_THRESHOLD", "0.5")
    monkeypatch.setenv("RAG_LLM_JUDGE_MIN_CANDIDATES", "5")
    assert should_fire_judge(top_score=0.3, n_candidates=8) is True


# ── _parse_judge_response ─────────────────────────────────────────────────────


def test_parse_valid_json():
    raw = '{"scores": [8, 3, 7, 5, 9]}'
    result = _parse_judge_response(raw, expected_n=5)
    assert result == [8.0, 3.0, 7.0, 5.0, 9.0]


def test_parse_json_with_markdown_fence():
    raw = '```json\n{"scores": [10, 4]}\n```'
    result = _parse_judge_response(raw, expected_n=2)
    assert result == [10.0, 4.0]


def test_parse_empty_string_returns_none():
    assert _parse_judge_response("", expected_n=3) is None
    assert _parse_judge_response("   ", expected_n=3) is None


def test_parse_wrong_length_returns_none():
    raw = '{"scores": [8, 3]}'
    assert _parse_judge_response(raw, expected_n=5) is None


def test_parse_invalid_json_returns_none():
    assert _parse_judge_response("not json at all", expected_n=3) is None


def test_parse_scores_clamped_to_range():
    raw = '{"scores": [-5, 15, 5]}'
    result = _parse_judge_response(raw, expected_n=3)
    assert result == [0.0, 10.0, 5.0]


def test_parse_non_numeric_score_defaults_to_five():
    raw = '{"scores": ["high", 7, 3]}'
    result = _parse_judge_response(raw, expected_n=3)
    assert result is not None
    assert result[0] == pytest.approx(5.0)


def test_parse_missing_scores_key_returns_none():
    raw = '{"rankings": [8, 3]}'
    assert _parse_judge_response(raw, expected_n=2) is None


# ── blend_scores ──────────────────────────────────────────────────────────────


def test_blend_scores_alpha_half():
    ce = [0.8, 0.6, 0.4]
    llm = [10.0, 5.0, 0.0]
    result = blend_scores(ce, llm, alpha=0.5)
    # 0.5 * 0.8 + 0.5 * (10/10) = 0.4 + 0.5 = 0.9
    assert result[0] == pytest.approx(0.9)
    # 0.5 * 0.6 + 0.5 * (5/10) = 0.3 + 0.25 = 0.55
    assert result[1] == pytest.approx(0.55)
    # 0.5 * 0.4 + 0.5 * 0.0 = 0.2
    assert result[2] == pytest.approx(0.2)


def test_blend_scores_mismatched_lengths_returns_ce():
    ce = [0.8, 0.6]
    llm = [10.0, 5.0, 3.0]
    result = blend_scores(ce, llm, alpha=0.5)
    assert result == ce


def test_blend_scores_alpha_one_returns_ce_only():
    ce = [0.7, 0.3]
    llm = [0.0, 10.0]
    result = blend_scores(ce, llm, alpha=1.0)
    assert result == pytest.approx(ce)


def test_blend_scores_alpha_zero_returns_llm_only():
    ce = [0.0, 0.0]
    llm = [8.0, 2.0]
    result = blend_scores(ce, llm, alpha=0.0)
    assert result == pytest.approx([0.8, 0.2])


# ── judge_and_blend ───────────────────────────────────────────────────────────


def _make_fake_helper_response(content: str):
    class _Msg:
        def __init__(self, c):
            self.content = c

    class _Resp:
        def __init__(self, c):
            self.message = _Msg(c)

    return _Resp(content)


def test_judge_and_blend_returns_ce_on_llm_failure(monkeypatch):
    import rag

    def _fail(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(rag, "_helper_client", lambda: type("C", (), {"chat": staticmethod(_fail)})())

    candidates = [("doc text", {"path": "a.md"})] * 6
    ce_scores = [0.8, 0.6, 0.5, 0.4, 0.3, 0.2]

    final_scores, telemetry = judge_and_blend("test query", candidates, ce_scores)
    assert final_scores == ce_scores
    assert telemetry["llm_judge_fired"] is False
    assert telemetry["llm_judge_parse_failed"] is True


def test_judge_and_blend_blends_on_success(monkeypatch):
    import rag

    scores_json = '{"scores": [9, 7, 5, 3, 1, 0]}'

    class _FakeClient:
        def chat(self, **kwargs):
            return _make_fake_helper_response(scores_json)

    monkeypatch.setattr(rag, "_helper_client", lambda: _FakeClient())
    monkeypatch.setattr(rag, "HELPER_MODEL", "test-model")
    monkeypatch.setattr(rag, "HELPER_OPTIONS", {})
    monkeypatch.setattr(rag, "LLM_KEEP_ALIVE", -1)

    candidates = [("doc text", {"path": "a.md"})] * 6
    ce_scores = [0.8, 0.6, 0.5, 0.4, 0.3, 0.2]

    final_scores, telemetry = judge_and_blend("test query", candidates, ce_scores, alpha=0.5)
    assert telemetry["llm_judge_fired"] is True
    assert len(final_scores) == len(ce_scores)
    # Top score should be blended value of first candidate
    assert final_scores[0] == pytest.approx(0.5 * 0.8 + 0.5 * (9.0 / 10.0))
