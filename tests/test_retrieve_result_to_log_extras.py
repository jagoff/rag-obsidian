"""Tests para retrieve_result_to_log_extras (2026-05-04).

Verifica que la función extrae correctamente del RetrieveResult los campos
que el gen() del web necesita para poblar rag_queries.extra_json — los 6
features que antes quedaban fuera del log web por el dict manual.

Coverage de los contratos:
  - defaults seguros con un RR vacío
  - anaphora_resolved/original/rewritten desde rr.extras
  - contradiction_penalty_applied / mmr_applied desde campos del dataclass
  - temporal_intent desde el dataclass
  - decomposed / n_sub_queries / decompose_ms desde filters_applied + timing
  - llm_judge_* desde los campos del dataclass
  - silent-fail ante excepciones (nunca lanza, devuelve {})
"""
import pytest

from rag import RetrieveResult, retrieve_result_to_log_extras


# ── helpers ──────────────────────────────────────────────────────────────────

def _empty_rr(**kwargs) -> RetrieveResult:
    """RR mínimo válido con overrides opcionales."""
    defaults = dict(docs=[], metas=[], scores=[], confidence=0.5)
    defaults.update(kwargs)
    return RetrieveResult(**defaults)


# ── defaults seguros ─────────────────────────────────────────────────────────

def test_empty_rr_returns_all_keys():
    """Con un RR vacío, todas las claves esperadas deben estar presentes."""
    extras = retrieve_result_to_log_extras(_empty_rr())
    expected_keys = {
        "anaphora_resolved", "anaphora_original", "anaphora_rewritten",
        "contradiction_penalty_applied", "mmr_applied", "temporal_intent",
        "decomposed", "n_sub_queries", "decompose_ms",
        "llm_judge_fired", "llm_judge_ms",
        "llm_judge_top_score_before", "llm_judge_top_score_after",
        "llm_judge_parse_failed", "llm_judge_n_candidates",
        # Quick Win #4 — typo correction telemetry surfaced via el bridge
        # (refactor 2026-05-04 evening). Reemplaza las globals legacy
        # `_expand_last_llm_typo_*[0]` que tenían data race en el web.
        "llm_typo_corrected", "llm_typo_original", "llm_typo_corrected_text",
        # filters_applied surfacing (2026-05-04, agregado para que el web
        # log_query_event reciba `filters_json` con los filtros inferidos).
        "filters",
    }
    assert expected_keys == set(extras.keys())


def test_empty_rr_defaults_are_safe():
    """Con RR vacío, los valores deben ser los ceros / False / 'neutral' seguros."""
    extras = retrieve_result_to_log_extras(_empty_rr())
    assert extras["anaphora_resolved"] is False
    assert extras["anaphora_original"] == ""
    assert extras["anaphora_rewritten"] == ""
    assert extras["contradiction_penalty_applied"] == 0
    assert extras["mmr_applied"] == 0
    assert extras["temporal_intent"] == "neutral"
    assert extras["decomposed"] is False
    assert extras["n_sub_queries"] == 0
    assert extras["decompose_ms"] == 0
    assert extras["llm_judge_fired"] is False
    assert extras["llm_judge_ms"] == 0
    assert extras["llm_judge_top_score_before"] == 0.0
    assert extras["llm_judge_top_score_after"] == 0.0
    assert extras["llm_judge_parse_failed"] is False
    assert extras["llm_judge_n_candidates"] == 0


# ── anaphora desde extras ─────────────────────────────────────────────────────

def test_anaphora_resolved_propagates():
    rr = _empty_rr(extras={
        "anaphora_resolved": True,
        "anaphora_original": "y mañana?",
        "anaphora_rewritten": "qué eventos tengo mañana?",
    })
    extras = retrieve_result_to_log_extras(rr)
    assert extras["anaphora_resolved"] is True
    assert extras["anaphora_original"] == "y mañana?"
    assert extras["anaphora_rewritten"] == "qué eventos tengo mañana?"


def test_anaphora_not_resolved_is_false():
    rr = _empty_rr(extras={
        "anaphora_resolved": False,
        "anaphora_original": "test",
        "anaphora_rewritten": "test",
    })
    extras = retrieve_result_to_log_extras(rr)
    assert extras["anaphora_resolved"] is False


# ── contradiction_penalty_applied ────────────────────────────────────────────

def test_contradiction_penalty_propagates():
    rr = _empty_rr(contradiction_penalty_applied=3)
    assert retrieve_result_to_log_extras(rr)["contradiction_penalty_applied"] == 3


def test_contradiction_penalty_zero_by_default():
    assert retrieve_result_to_log_extras(_empty_rr())["contradiction_penalty_applied"] == 0


# ── mmr_applied ──────────────────────────────────────────────────────────────

def test_mmr_applied_propagates():
    rr = _empty_rr(mmr_applied=5)
    assert retrieve_result_to_log_extras(rr)["mmr_applied"] == 5


# ── temporal_intent ──────────────────────────────────────────────────────────

def test_temporal_intent_recent():
    rr = _empty_rr(temporal_intent="recent")
    assert retrieve_result_to_log_extras(rr)["temporal_intent"] == "recent"


def test_temporal_intent_historical():
    rr = _empty_rr(temporal_intent="historical")
    assert retrieve_result_to_log_extras(rr)["temporal_intent"] == "historical"


def test_temporal_intent_neutral_default():
    rr = _empty_rr(temporal_intent="neutral")
    assert retrieve_result_to_log_extras(rr)["temporal_intent"] == "neutral"


# ── decomposed / query decomposition ─────────────────────────────────────────

def test_decomposed_propagates():
    rr = _empty_rr(
        filters_applied={"decomposed": True, "n_sub_queries": 3},
        timing={"decompose_ms": 120},
    )
    extras = retrieve_result_to_log_extras(rr)
    assert extras["decomposed"] is True
    assert extras["n_sub_queries"] == 3
    assert extras["decompose_ms"] == 120


def test_decomposed_false_when_absent():
    rr = _empty_rr(filters_applied={}, timing={})
    extras = retrieve_result_to_log_extras(rr)
    assert extras["decomposed"] is False
    assert extras["n_sub_queries"] == 0
    assert extras["decompose_ms"] == 0


# ── llm_judge_* ──────────────────────────────────────────────────────────────

def test_llm_judge_fields_propagate():
    rr = _empty_rr(
        llm_judge_fired=True,
        llm_judge_ms=450,
        llm_judge_top_score_before=0.42,
        llm_judge_top_score_after=0.71,
        llm_judge_parse_failed=False,
        llm_judge_n_candidates=20,
    )
    extras = retrieve_result_to_log_extras(rr)
    assert extras["llm_judge_fired"] is True
    assert extras["llm_judge_ms"] == 450
    assert abs(extras["llm_judge_top_score_before"] - 0.42) < 1e-6
    assert abs(extras["llm_judge_top_score_after"] - 0.71) < 1e-6
    assert extras["llm_judge_parse_failed"] is False
    assert extras["llm_judge_n_candidates"] == 20


def test_llm_judge_parse_failed_propagates():
    rr = _empty_rr(llm_judge_fired=True, llm_judge_parse_failed=True)
    extras = retrieve_result_to_log_extras(rr)
    assert extras["llm_judge_parse_failed"] is True


# ── tipo de valores de retorno ────────────────────────────────────────────────

def test_return_types_are_python_primitives():
    """Las claves del dict deben ser tipos Python básicos, no numpy/torch."""
    rr = _empty_rr(
        contradiction_penalty_applied=2,
        mmr_applied=1,
        temporal_intent="recent",
        llm_judge_fired=True,
        llm_judge_ms=300,
    )
    extras = retrieve_result_to_log_extras(rr)
    assert isinstance(extras["anaphora_resolved"], bool)
    assert isinstance(extras["contradiction_penalty_applied"], int)
    assert isinstance(extras["mmr_applied"], int)
    assert isinstance(extras["temporal_intent"], str)
    assert isinstance(extras["decomposed"], bool)
    assert isinstance(extras["n_sub_queries"], int)
    assert isinstance(extras["decompose_ms"], int)
    assert isinstance(extras["llm_judge_fired"], bool)
    assert isinstance(extras["llm_judge_ms"], int)
    assert isinstance(extras["llm_judge_top_score_before"], float)
    assert isinstance(extras["llm_judge_top_score_after"], float)
    assert isinstance(extras["llm_judge_parse_failed"], bool)
    assert isinstance(extras["llm_judge_n_candidates"], int)


# ── silent-fail contract ──────────────────────────────────────────────────────

def test_silent_fail_on_broken_object():
    """Si el objeto no es un RetrieveResult válido, debe devolver {} sin lanzar."""
    result = retrieve_result_to_log_extras(None)  # type: ignore[arg-type]
    assert isinstance(result, dict)
    # puede ser vacío o con defaults — lo importante es no lanzar


def test_dict_with_unknown_values_degrades_gracefully():
    """Un dict plano (mock de test legacy) no debe crashear."""
    fake = {"docs": [], "metas": [], "scores": [], "confidence": 0.5}

    class FakeRR:
        extras = {}
        filters_applied = {}
        timing = {}

        def get(self, key, default=None):
            return default

    result = retrieve_result_to_log_extras(FakeRR())  # type: ignore[arg-type]
    assert isinstance(result, dict)
