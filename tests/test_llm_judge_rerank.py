"""Tests para el modulo `rag.llm_judge` + integracion en `retrieve()`.

Casos cubiertos:
  (a) `should_fire_judge()` dispara cuando top_score bajo + flag ON.
  (b) `should_fire_judge()` NO dispara cuando top_score alto, flag OFF,
      o pool chica.
  (c) Blending formula correcta (alpha 0.5 ↔ 50/50, alpha 1 ↔ pure CE).
  (d) JSON parse failure cae graceful (return None).
  (e) `judge_and_blend()` extiende correctamente cuando pool > judge cap.
  (f) Integration: el dataclass RetrieveResult expone los campos.

NO se hacen llamadas reales a ollama — todo monkey-patched.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

import rag
from rag import llm_judge


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def reset_env(monkeypatch):
    """Limpia env vars del LLM judge antes de cada test."""
    for var in (
        "RAG_LLM_JUDGE",
        "RAG_LLM_JUDGE_THRESHOLD",
        "RAG_LLM_JUDGE_ALPHA",
        "RAG_LLM_JUDGE_POOL",
        "RAG_LLM_JUDGE_MIN_CANDIDATES",
        "RAG_LLM_JUDGE_DEBUG",
    ):
        monkeypatch.delenv(var, raising=False)


def _mock_helper_client_returning(json_text: str):
    """Devuelve un cliente mock cuya `chat()` retorna `json_text` en
    el `.message.content`."""
    client = MagicMock()
    resp = MagicMock()
    resp.message.content = json_text
    client.chat.return_value = resp
    return client


# ─────────────────────────────────────────────────────────────────────────
# (a) should_fire_judge: dispara cuando top_score bajo + flag ON
# ─────────────────────────────────────────────────────────────────────────


def test_should_fire_judge_disparates_when_top_score_low_and_flag_on(
    reset_env, monkeypatch,
):
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    # top_score 0.3 < default threshold 0.5, n=10 > min 5 → fire
    assert llm_judge.should_fire_judge(top_score=0.3, n_candidates=10) is True


def test_should_fire_judge_respects_custom_threshold(reset_env, monkeypatch):
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    monkeypatch.setenv("RAG_LLM_JUDGE_THRESHOLD", "0.2")
    # top_score 0.3 >= custom 0.2 → no fire
    assert llm_judge.should_fire_judge(top_score=0.3, n_candidates=10) is False
    # top_score 0.1 < 0.2 → fire
    assert llm_judge.should_fire_judge(top_score=0.1, n_candidates=10) is True


# ─────────────────────────────────────────────────────────────────────────
# (b) should_fire_judge: NO dispara en condiciones de skip
# ─────────────────────────────────────────────────────────────────────────


def test_should_fire_judge_no_fire_when_flag_off(reset_env):
    # Sin flag, no dispara aunque score bajo + pool grande
    assert llm_judge.should_fire_judge(top_score=0.1, n_candidates=20) is False


def test_should_fire_judge_no_fire_when_top_score_high(reset_env, monkeypatch):
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    # top_score 0.7 >= default threshold 0.5 → cross-encoder se decidio
    assert llm_judge.should_fire_judge(top_score=0.7, n_candidates=10) is False


def test_should_fire_judge_no_fire_when_pool_small(reset_env, monkeypatch):
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    # n=3 < default min 5 → no fire
    assert llm_judge.should_fire_judge(top_score=0.1, n_candidates=3) is False


def test_should_fire_judge_respects_custom_min_candidates(reset_env, monkeypatch):
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    monkeypatch.setenv("RAG_LLM_JUDGE_MIN_CANDIDATES", "10")
    # n=8 < custom 10 → no fire
    assert llm_judge.should_fire_judge(top_score=0.1, n_candidates=8) is False
    # n=12 >= 10 → fire
    assert llm_judge.should_fire_judge(top_score=0.1, n_candidates=12) is True


def test_should_fire_judge_flag_truthy_variants(reset_env, monkeypatch):
    """RAG_LLM_JUDGE acepta 1/true/yes (case-insensitive)."""
    for val in ("1", "true", "TRUE", "yes", "YES"):
        monkeypatch.setenv("RAG_LLM_JUDGE", val)
        assert llm_judge.should_fire_judge(0.1, 10) is True, val
    for val in ("0", "false", "no", "", "off"):
        monkeypatch.setenv("RAG_LLM_JUDGE", val)
        assert llm_judge.should_fire_judge(0.1, 10) is False, val


# ─────────────────────────────────────────────────────────────────────────
# (c) Blending formula
# ─────────────────────────────────────────────────────────────────────────


def test_blend_scores_alpha_half(reset_env):
    """alpha=0.5 ↔ promedio simple ce + (llm/10)."""
    ce = [0.4, 0.3, 0.5]
    llm_scores = [10.0, 5.0, 0.0]  # normalizado: 1.0, 0.5, 0.0
    result = llm_judge.blend_scores(ce, llm_scores, alpha=0.5)
    assert result == pytest.approx([0.5 * 0.4 + 0.5 * 1.0,
                                    0.5 * 0.3 + 0.5 * 0.5,
                                    0.5 * 0.5 + 0.5 * 0.0])


def test_blend_scores_alpha_one_is_pure_ce(reset_env):
    """alpha=1.0 → ignora el LLM (pure cross-encoder)."""
    ce = [0.4, 0.3, 0.5]
    llm_scores = [10.0, 0.0, 5.0]  # totalmente diferente
    result = llm_judge.blend_scores(ce, llm_scores, alpha=1.0)
    assert result == pytest.approx(ce)


def test_blend_scores_alpha_zero_is_pure_llm(reset_env):
    """alpha=0 → ignora el cross-encoder (pure LLM normalizado)."""
    ce = [0.4, 0.3, 0.5]
    llm_scores = [10.0, 0.0, 5.0]
    result = llm_judge.blend_scores(ce, llm_scores, alpha=0.0)
    assert result == pytest.approx([1.0, 0.0, 0.5])


def test_blend_scores_clamps_alpha_out_of_range(reset_env):
    """alpha fuera de [0,1] queda capado defensivamente."""
    ce = [0.4]
    llm = [10.0]
    # alpha 1.5 → clamp a 1.0 → result = 1.0 * 0.4 + 0.0 * 1.0 = 0.4
    assert llm_judge.blend_scores(ce, llm, alpha=1.5) == pytest.approx([0.4])
    # alpha -0.5 → clamp a 0.0 → result = 0.0 * 0.4 + 1.0 * 1.0 = 1.0
    assert llm_judge.blend_scores(ce, llm, alpha=-0.5) == pytest.approx([1.0])


def test_blend_scores_mismatched_lengths_returns_ce_intact(reset_env):
    """Defensa contra caller con bug: largos distintos → CE intacto."""
    ce = [0.4, 0.3]
    llm = [10.0]
    result = llm_judge.blend_scores(ce, llm)
    assert result == ce


def test_blend_scores_alpha_from_env(reset_env, monkeypatch):
    """Sin alpha explicito, lee RAG_LLM_JUDGE_ALPHA."""
    monkeypatch.setenv("RAG_LLM_JUDGE_ALPHA", "0.3")
    ce = [1.0]
    llm = [10.0]  # normalizado 1.0
    # 0.3 * 1.0 + 0.7 * 1.0 = 1.0
    assert llm_judge.blend_scores(ce, llm) == pytest.approx([1.0])
    # Con scores distintos
    ce2 = [0.5]
    llm2 = [0.0]  # normalizado 0.0
    # 0.3 * 0.5 + 0.7 * 0.0 = 0.15
    assert llm_judge.blend_scores(ce2, llm2) == pytest.approx([0.15])


# ─────────────────────────────────────────────────────────────────────────
# (d) JSON parse failure → graceful return None
# ─────────────────────────────────────────────────────────────────────────


def test_parse_judge_response_valid_json(reset_env):
    raw = '{"scores": [9, 7, 3]}'
    out = llm_judge._parse_judge_response(raw, expected_n=3)
    assert out == [9.0, 7.0, 3.0]


def test_parse_judge_response_clamps_out_of_range(reset_env):
    raw = '{"scores": [15, -3, 5]}'
    out = llm_judge._parse_judge_response(raw, expected_n=3)
    assert out == [10.0, 0.0, 5.0]


def test_parse_judge_response_size_mismatch_returns_none(reset_env):
    """Si el modelo devuelve cantidad distinta de scores → None."""
    raw = '{"scores": [9, 7]}'
    assert llm_judge._parse_judge_response(raw, expected_n=5) is None


def test_parse_judge_response_invalid_json_returns_none(reset_env):
    """Texto raro sin JSON → None."""
    assert llm_judge._parse_judge_response("not json at all", 3) is None
    assert llm_judge._parse_judge_response("", 3) is None
    assert llm_judge._parse_judge_response("   ", 3) is None


def test_parse_judge_response_repair_fenced_json(reset_env):
    """Modelo a veces wrapea en ```json ... ``` aunque le pidamos puro."""
    raw = '```json\n{"scores": [5, 5, 5]}\n```'
    out = llm_judge._parse_judge_response(raw, expected_n=3)
    assert out == [5.0, 5.0, 5.0]


def test_parse_judge_response_extra_keys_ok(reset_env):
    """Si el modelo agrega keys extras pero `scores` es valido, OK."""
    raw = '{"reasoning": "...", "scores": [9, 8, 7], "version": 1}'
    out = llm_judge._parse_judge_response(raw, expected_n=3)
    assert out == [9.0, 8.0, 7.0]


def test_parse_judge_response_string_score_falls_back_to_neutral(reset_env):
    """Si un score viene como string raro, default a 5 (neutral) en
    vez de tirar todo el batch."""
    raw = '{"scores": [9, "high", 3]}'
    out = llm_judge._parse_judge_response(raw, expected_n=3)
    assert out == [9.0, 5.0, 3.0]


# ─────────────────────────────────────────────────────────────────────────
# (e) llm_judge_candidates → graceful on failures
# ─────────────────────────────────────────────────────────────────────────


def test_llm_judge_candidates_silent_on_helper_exception(reset_env, monkeypatch):
    """Si _helper_client().chat() raisea, devolvemos None."""
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    client = MagicMock()
    client.chat.side_effect = RuntimeError("ollama down")
    monkeypatch.setattr(rag, "_helper_client", lambda: client)

    result = llm_judge.llm_judge_candidates(
        "test query",
        [("doc1 body", {"file": "a.md", "note": "A"})],
    )
    assert result is None


def test_llm_judge_candidates_silent_on_parse_fail(reset_env, monkeypatch):
    """Si el modelo devuelve JSON invalido, devolvemos None."""
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    client = _mock_helper_client_returning("not even close to json")
    monkeypatch.setattr(rag, "_helper_client", lambda: client)

    result = llm_judge.llm_judge_candidates(
        "test query",
        [("doc1 body", {"file": "a.md", "note": "A"})],
    )
    assert result is None


def test_llm_judge_candidates_returns_scores_on_valid_response(
    reset_env, monkeypatch,
):
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    client = _mock_helper_client_returning('{"scores": [9, 5, 3]}')
    monkeypatch.setattr(rag, "_helper_client", lambda: client)

    result = llm_judge.llm_judge_candidates(
        "test query",
        [
            ("doc1", {"file": "a.md", "note": "A"}),
            ("doc2", {"file": "b.md", "note": "B"}),
            ("doc3", {"file": "c.md", "note": "C"}),
        ],
    )
    assert result == [9.0, 5.0, 3.0]


def test_llm_judge_candidates_empty_input(reset_env):
    """Empty candidates → None."""
    assert llm_judge.llm_judge_candidates("query", []) is None


# ─────────────────────────────────────────────────────────────────────────
# (e cont'd) judge_and_blend full flow
# ─────────────────────────────────────────────────────────────────────────


def test_judge_and_blend_happy_path(reset_env, monkeypatch):
    """Judge corre OK, scores son blended."""
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    client = _mock_helper_client_returning('{"scores": [10, 0, 5]}')
    monkeypatch.setattr(rag, "_helper_client", lambda: client)

    pairs = [
        ("doc a", {"file": "a.md", "note": "A"}),
        ("doc b", {"file": "b.md", "note": "B"}),
        ("doc c", {"file": "c.md", "note": "C"}),
    ]
    ce = [0.4, 0.5, 0.3]
    blended, telemetry = llm_judge.judge_and_blend(
        "query", pairs, ce, alpha=0.5,
    )
    # 0.5 * 0.4 + 0.5 * 1.0 = 0.7
    # 0.5 * 0.5 + 0.5 * 0.0 = 0.25
    # 0.5 * 0.3 + 0.5 * 0.5 = 0.4
    assert blended == pytest.approx([0.7, 0.25, 0.4])
    assert telemetry["llm_judge_fired"] is True
    assert telemetry["llm_judge_parse_failed"] is False
    assert telemetry["llm_judge_top_score_before"] == pytest.approx(0.5)
    # max blended = 0.7
    assert telemetry["llm_judge_top_score_after"] == pytest.approx(0.7)


def test_judge_and_blend_parse_fail_returns_ce_intact(reset_env, monkeypatch):
    """Parse fail → blended == ce, telemetry.parse_failed=True."""
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    client = _mock_helper_client_returning("garbage")
    monkeypatch.setattr(rag, "_helper_client", lambda: client)

    pairs = [("doc", {"file": "a.md", "note": "A"})]
    ce = [0.4]
    blended, telemetry = llm_judge.judge_and_blend("q", pairs, ce)
    assert blended == ce
    assert telemetry["llm_judge_fired"] is False
    assert telemetry["llm_judge_parse_failed"] is True


def test_judge_and_blend_extends_when_pool_capped(reset_env, monkeypatch):
    """Cuando candidates > pool_cap, blendeamos el head + dejamos tail
    intacto con CE-only scores."""
    monkeypatch.setenv("RAG_LLM_JUDGE", "1")
    monkeypatch.setenv("RAG_LLM_JUDGE_POOL", "2")
    # Mock devuelve solo 2 scores (matching capped pool)
    client = _mock_helper_client_returning('{"scores": [10, 0]}')
    monkeypatch.setattr(rag, "_helper_client", lambda: client)

    pairs = [
        ("a", {"file": "a.md"}),
        ("b", {"file": "b.md"}),
        ("c", {"file": "c.md"}),
        ("d", {"file": "d.md"}),
    ]
    ce = [0.4, 0.5, 0.3, 0.2]
    blended, telemetry = llm_judge.judge_and_blend(
        "q", pairs, ce, alpha=0.5,
    )
    # First 2 blended: [0.5*0.4 + 0.5*1.0, 0.5*0.5 + 0.5*0.0] = [0.7, 0.25]
    # Last 2 untouched: [0.3, 0.2]
    assert blended == pytest.approx([0.7, 0.25, 0.3, 0.2])
    assert telemetry["llm_judge_fired"] is True


# ─────────────────────────────────────────────────────────────────────────
# (f) RetrieveResult dataclass exposes the fields
# ─────────────────────────────────────────────────────────────────────────


def test_retrieve_result_has_llm_judge_fields():
    """Dataclass fields para LLM judge existen + defaults correctos."""
    import dataclasses

    fields_by_name = {f.name: f for f in dataclasses.fields(rag.RetrieveResult)}
    assert "llm_judge_fired" in fields_by_name
    assert "llm_judge_ms" in fields_by_name
    assert "llm_judge_top_score_before" in fields_by_name
    assert "llm_judge_top_score_after" in fields_by_name
    assert "llm_judge_parse_failed" in fields_by_name
    assert "llm_judge_n_candidates" in fields_by_name

    # Defaults: judge no disparo
    rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=0.0,
    )
    assert rr.llm_judge_fired is False
    assert rr.llm_judge_ms == 0
    assert rr.llm_judge_top_score_before == 0.0
    assert rr.llm_judge_top_score_after == 0.0
    assert rr.llm_judge_parse_failed is False
    assert rr.llm_judge_n_candidates == 0


def test_retrieve_result_get_legacy_compat():
    """Los call sites legacy con `result.get('llm_judge_fired')` siguen
    funcionando vía el __getitem__ wrapper."""
    rr = rag.RetrieveResult(
        docs=[], metas=[], scores=[], confidence=0.0,
        llm_judge_fired=True, llm_judge_ms=234,
    )
    assert rr.get("llm_judge_fired") is True
    assert rr.get("llm_judge_ms") == 234
    # Default for missing
    assert rr.get("nonexistent_field", "default") == "default"


# ─────────────────────────────────────────────────────────────────────────
# Master gate OFF → judge no llamado
# ─────────────────────────────────────────────────────────────────────────


def test_master_gate_off_skips_helper_call(reset_env, monkeypatch):
    """Sin RAG_LLM_JUDGE=1, should_fire_judge() es False y el caller
    nunca llama al helper. Verificamos via spy."""
    # Default OFF (env limpiado por fixture)
    spy = MagicMock()
    monkeypatch.setattr(rag, "_helper_client", spy)
    # No llamamos llm_judge_candidates directo — verificamos should_fire
    assert llm_judge.should_fire_judge(0.1, 10) is False
    spy.assert_not_called()
