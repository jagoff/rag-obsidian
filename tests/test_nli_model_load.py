"""Tests for NLI model loader + unloader (Improvement #1, Fase B.1).

Cubre: get_nli_model(), maybe_unload_nli_model(), idle TTL, NEVER_UNLOAD env,
ground_claims_nli() con inference real (mockeada) vs fallback.

Los tests NO requieren sentence_transformers instalado — mockean el model
a través del setter directo `rag._nli_model = mock` (más simple que patch
del import dinámico, que queda encerrado dentro de get_nli_model).
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

import rag


class _MockCrossEncoder:
    """Stand-in para CrossEncoder — expone .predict(pairs, convert_to_numpy=True)."""

    def __init__(self, scores=None):
        self.predict_calls = []
        self._scores = scores  # np.ndarray shape (N, 3) o None → default entail=1.0

    def predict(self, pairs, convert_to_numpy=False):
        self.predict_calls.append(list(pairs))
        import numpy as np
        if self._scores is not None:
            return self._scores
        return np.array([[0.9, 0.05, 0.05]] * len(pairs))


@pytest.fixture
def reset_nli_state():
    """Reset NLI global state before + after each test."""
    prev_model = rag._nli_model
    prev_last_use = rag._nli_last_use
    rag._nli_model = None
    rag._nli_last_use = 0.0
    yield
    rag._nli_model = prev_model
    rag._nli_last_use = prev_last_use


def test_get_nli_model_returns_cached_after_first_call(reset_nli_state):
    """Segunda call NO re-inicia — devuelve el mismo objeto."""
    rag._nli_model = _MockCrossEncoder()
    m1 = rag.get_nli_model()
    m2 = rag.get_nli_model()
    assert m1 is m2


def test_get_nli_model_refreshes_last_use(reset_nli_state):
    """Cada call refresca _nli_last_use."""
    rag._nli_model = _MockCrossEncoder()
    rag._nli_last_use = 0.0
    rag.get_nli_model()
    assert rag._nli_last_use > 0.0

    snapshot = rag._nli_last_use
    time.sleep(0.005)
    rag.get_nli_model()
    assert rag._nli_last_use >= snapshot


def test_maybe_unload_nli_model_noop_when_none(reset_nli_state):
    """Si _nli_model is None → False sin side effects."""
    rag._nli_model = None
    assert rag.maybe_unload_nli_model() is False


def test_maybe_unload_nli_model_noop_recent(reset_nli_state, monkeypatch):
    """Si idle < TTL → False, modelo intacto."""
    monkeypatch.setattr(rag, "_NLI_IDLE_TTL", 60.0)
    monkeypatch.setattr(rag, "_NLI_NEVER_UNLOAD", False)
    rag._nli_model = _MockCrossEncoder()
    rag._nli_last_use = time.time()
    assert rag.maybe_unload_nli_model() is False
    assert rag._nli_model is not None


def test_maybe_unload_nli_model_evicts_when_idle(reset_nli_state, monkeypatch):
    """Idle > TTL → True, modelo eliminado."""
    monkeypatch.setattr(rag, "_NLI_IDLE_TTL", 0.01)
    monkeypatch.setattr(rag, "_NLI_NEVER_UNLOAD", False)
    rag._nli_model = _MockCrossEncoder()
    rag._nli_last_use = time.time() - 10.0
    assert rag.maybe_unload_nli_model() is True
    assert rag._nli_model is None


def test_maybe_unload_nli_model_force_bypasses_ttl(reset_nli_state, monkeypatch):
    """force=True ignora TTL y NEVER_UNLOAD."""
    monkeypatch.setattr(rag, "_NLI_IDLE_TTL", 900.0)
    monkeypatch.setattr(rag, "_NLI_NEVER_UNLOAD", True)
    rag._nli_model = _MockCrossEncoder()
    rag._nli_last_use = time.time()
    assert rag.maybe_unload_nli_model(force=True) is True
    assert rag._nli_model is None


def test_maybe_unload_respects_never_unload(reset_nli_state, monkeypatch):
    """NEVER_UNLOAD=True + no force → no unload aunque idle > TTL."""
    monkeypatch.setattr(rag, "_NLI_IDLE_TTL", 0.01)
    monkeypatch.setattr(rag, "_NLI_NEVER_UNLOAD", True)
    rag._nli_model = _MockCrossEncoder()
    rag._nli_last_use = time.time() - 10.0
    assert rag.maybe_unload_nli_model(force=False) is False
    assert rag._nli_model is not None


def test_ground_claims_nli_model_none_falls_back_neutral(reset_nli_state, monkeypatch):
    """Si get_nli_model devuelve None → todos neutral."""
    monkeypatch.setattr(rag, "get_nli_model", lambda: None)
    claims = [rag.Claim(text="This is a claim that is long enough")]
    docs = ["Some evidence text here"]
    metas = [{"file": "test.md"}]
    result = rag.ground_claims_nli(claims, docs, metas)
    assert result is not None
    assert result.claims_total == 1
    assert result.claims[0].verdict == "neutral"
    assert result.claims_supported == 0
    assert result.claims_contradicted == 0


def test_ground_claims_nli_classifies_entails(reset_nli_state, monkeypatch):
    """Scores [0.9, 0.05, 0.05] con threshold 0.7 → entails."""
    import numpy as np
    mock = _MockCrossEncoder(scores=np.array([[0.9, 0.05, 0.05]]))
    monkeypatch.setattr(rag, "get_nli_model", lambda: mock)
    claims = [rag.Claim(text="The project ends in June")]
    docs = ["The project ends on June 30, 2024"]
    metas = [{"file": "note.md", "chunk_id": "c1"}]
    result = rag.ground_claims_nli(claims, docs, metas, threshold_contradicts=0.7)
    assert result.claims[0].verdict == "entails"
    assert result.claims_supported == 1


def test_ground_claims_nli_classifies_contradicts(reset_nli_state, monkeypatch):
    """Scores [0.05, 0.05, 0.9] con threshold 0.7 → contradicts."""
    import numpy as np
    mock = _MockCrossEncoder(scores=np.array([[0.05, 0.05, 0.9]]))
    monkeypatch.setattr(rag, "get_nli_model", lambda: mock)
    claims = [rag.Claim(text="The project has 3 phases")]
    docs = ["The project has 2 phases"]
    metas = [{"file": "note.md"}]
    result = rag.ground_claims_nli(claims, docs, metas, threshold_contradicts=0.7)
    assert result.claims[0].verdict == "contradicts"
    assert result.claims_contradicted == 1


def test_ground_claims_nli_neutral_below_threshold(reset_nli_state, monkeypatch):
    """Scores por debajo de threshold → neutral."""
    import numpy as np
    mock = _MockCrossEncoder(scores=np.array([[0.4, 0.5, 0.1]]))
    monkeypatch.setattr(rag, "get_nli_model", lambda: mock)
    claims = [rag.Claim(text="Some unrelated claim here")]
    docs = ["Completely different topic doc"]
    metas = [{"file": "note.md"}]
    result = rag.ground_claims_nli(claims, docs, metas, threshold_contradicts=0.7)
    assert result.claims[0].verdict == "neutral"


def test_ground_claims_nli_refusal_skipped(reset_nli_state, monkeypatch):
    """Refusal claims → neutral SIN llamar model.predict."""
    mock = _MockCrossEncoder()
    monkeypatch.setattr(rag, "get_nli_model", lambda: mock)
    claims = [rag.Claim(text="No encontré esto en el vault.", is_refusal=True)]
    docs = ["Some doc"]
    metas = [{"file": "note.md"}]
    result = rag.ground_claims_nli(claims, docs, metas)
    assert result.claims[0].verdict == "neutral"
    assert len(mock.predict_calls) == 0, "model.predict should NOT be called for refusals"


def test_ground_claims_nli_records_nli_ms(reset_nli_state, monkeypatch):
    """nli_ms field ≥ 0 tras inference."""
    import numpy as np
    mock = _MockCrossEncoder(scores=np.array([[0.9, 0.05, 0.05]]))
    monkeypatch.setattr(rag, "get_nli_model", lambda: mock)
    claims = [rag.Claim(text="A claim that is long enough")]
    docs = ["Evidence text here"]
    metas = [{"file": "note.md"}]
    result = rag.ground_claims_nli(claims, docs, metas)
    assert result.nli_ms >= 0


def test_ground_claims_nli_evidence_span_truncated(reset_nli_state, monkeypatch):
    """evidence_span truncado a ≤200 chars."""
    import numpy as np
    long_doc = "x" * 500
    mock = _MockCrossEncoder(scores=np.array([[0.9, 0.05, 0.05]]))
    monkeypatch.setattr(rag, "get_nli_model", lambda: mock)
    claims = [rag.Claim(text="A claim that is long enough")]
    docs = [long_doc]
    metas = [{"file": "note.md"}]
    result = rag.ground_claims_nli(claims, docs, metas)
    assert len(result.claims[0].evidence_span) <= 200


def test_ground_claims_nli_multi_doc_picks_best_evidence(reset_nli_state, monkeypatch):
    """Con múltiples docs, elige el de mayor entail score como evidencia."""
    import numpy as np
    mock = _MockCrossEncoder(scores=np.array([
        [0.9, 0.05, 0.05],  # doc 0: alto entail
        [0.1, 0.8, 0.1],    # doc 1: neutral
        [0.2, 0.3, 0.5],    # doc 2: parcial contradict
    ]))
    monkeypatch.setattr(rag, "get_nli_model", lambda: mock)
    claims = [rag.Claim(text="A claim that is long enough")]
    docs = ["Doc 1", "Doc 2", "Doc 3"]
    metas = [
        {"file": "f1.md", "chunk_id": "c1"},
        {"file": "f2.md", "chunk_id": "c2"},
        {"file": "f3.md", "chunk_id": "c3"},
    ]
    result = rag.ground_claims_nli(claims, docs, metas)
    assert result.claims[0].verdict == "entails"
    assert result.claims[0].evidence_chunk_id == "c1"


def test_ground_claims_nli_empty_inputs():
    """Empty inputs → None."""
    assert rag.ground_claims_nli([], [], []) is None
    assert rag.ground_claims_nli([rag.Claim(text="x")], [], []) is None
    assert rag.ground_claims_nli([], ["doc"], [{"file": "f.md"}]) is None


def test_ground_claims_nli_max_claims_gate():
    """> max_claims → None."""
    claims = [rag.Claim(text=f"Claim {i} long enough") for i in range(25)]
    result = rag.ground_claims_nli(claims, ["doc"], [{"file": "f.md"}], max_claims=20)
    assert result is None


def test_torch_mps_empty_cache_no_error():
    """_torch_mps_empty_cache() swallows exceptions."""
    rag._torch_mps_empty_cache()  # no raise


def test_ground_claims_nli_inference_error_falls_back(reset_nli_state, monkeypatch):
    """Si model.predict() raises → claim marcado neutral, no crash."""
    mock = MagicMock()
    mock.predict.side_effect = RuntimeError("predict failed")
    monkeypatch.setattr(rag, "get_nli_model", lambda: mock)
    claims = [rag.Claim(text="A claim that is long enough")]
    docs = ["Evidence"]
    metas = [{"file": "note.md"}]
    result = rag.ground_claims_nli(claims, docs, metas)
    assert result is not None
    assert result.claims[0].verdict == "neutral"
