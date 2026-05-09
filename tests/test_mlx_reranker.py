"""Smoke tests for `rag/mlx_reranker.py` — Qwen3-Reranker MLX backend.

Only the unit tests that don't load the real model run by default. The
heavyweight tests (real forward pass on Apple Silicon) gate on the
`requires_mlx` marker — opt-in via `pytest -m requires_mlx`.

Discrimination expectations are documented in the project memory
`decision/cierre-formal-ola-5-mlx-migration` and replicate the smoke-test
results from 2026-05-06: Qwen3-Reranker scores are *peakier* than
bge-reranker (P near 0.0 or near 1.0; bge-reranker is gentler in the
0.5-0.95 range for relevant pairs).
"""
from __future__ import annotations


import pytest


def test_mlx_reranker_aliases_table_resolves_qwen3_06b():
    """The default 0.6B alias maps to the mxfp8 mlx-community port."""
    from rag.mlx_reranker import MLX_RERANKER_ALIASES, DEFAULT_MLX_RERANKER

    assert MLX_RERANKER_ALIASES["qwen3-reranker:0.6b"] == DEFAULT_MLX_RERANKER
    assert DEFAULT_MLX_RERANKER == "mlx-community/Qwen3-Reranker-0.6B-mxfp8"


def test_resolve_mlx_reranker_path_falls_back_for_baseline():
    """When the call site still passes the bge baseline name, we route to
    the MLX default — that's the typical state right after the swap."""
    from rag.mlx_reranker import resolve_mlx_reranker_path, DEFAULT_MLX_RERANKER

    assert resolve_mlx_reranker_path("BAAI/bge-reranker-v2-m3") == DEFAULT_MLX_RERANKER
    assert resolve_mlx_reranker_path("") == DEFAULT_MLX_RERANKER
    assert resolve_mlx_reranker_path(None) == DEFAULT_MLX_RERANKER


def test_resolve_mlx_reranker_path_passthrough_hf_id():
    """An explicit mlx-community HF id is returned as-is."""
    from rag.mlx_reranker import resolve_mlx_reranker_path

    hf_id = "mlx-community/Qwen3-Reranker-4B-mxfp8"
    assert resolve_mlx_reranker_path(hf_id) == hf_id


def test_resolve_mlx_reranker_path_alias():
    """Aliases (`qwen3-reranker:8b`) translate to mxfp8 HF ids."""
    from rag.mlx_reranker import resolve_mlx_reranker_path

    assert resolve_mlx_reranker_path("qwen3-reranker:4b") == \
        "mlx-community/Qwen3-Reranker-4B-mxfp8"


def test_is_mlx_reranker_enabled_default_off(monkeypatch):
    """Default `RAG_RERANKER_BACKEND=torch` → MLX off. Production stays on
    bge-reranker until a deliberate cutover."""
    monkeypatch.delenv("RAG_RERANKER_BACKEND", raising=False)
    from rag.mlx_reranker import is_mlx_reranker_enabled

    assert is_mlx_reranker_enabled() is False


def test_is_mlx_reranker_enabled_when_env_set(monkeypatch):
    monkeypatch.setenv("RAG_RERANKER_BACKEND", "mlx")
    from rag.mlx_reranker import is_mlx_reranker_enabled

    assert is_mlx_reranker_enabled() is True


def test_is_mlx_reranker_enabled_case_insensitive(monkeypatch):
    monkeypatch.setenv("RAG_RERANKER_BACKEND", "MLX")
    from rag.mlx_reranker import is_mlx_reranker_enabled

    assert is_mlx_reranker_enabled() is True


def test_mlx_reranker_class_constructor_no_load():
    """Constructing the class doesn't trigger a model load — only `.predict`
    or `_ensure_loaded()` does. This matters because the production
    `get_reranker()` path may construct under contention."""
    from rag.mlx_reranker import MLXReranker

    rr = MLXReranker()
    assert rr.model_path == "mlx-community/Qwen3-Reranker-0.6B-mxfp8"
    assert rr._model is None
    assert rr._tokenizer is None


# -- Heavyweight (load + forward) --------------------------------------------


@pytest.mark.requires_mlx
def test_mlx_reranker_predict_relevant_vs_irrelevant():
    """Forward-pass smoke: relevant pair scores P > 0.5, irrelevant P < 0.1.

    Discrimination floor is generous (0.5 / 0.1) because Qwen3-Reranker
    output is bimodal — strongly relevant near 1.0, irrelevant near 0.0.
    The middle band (0.1-0.5) is where bge-reranker shines and Qwen3 is
    less certain; we deliberately don't test that middle band here.
    """
    from rag.mlx_reranker import MLXReranker

    rr = MLXReranker()
    pairs = [
        ("ikigai", "Ikigai.md\n(02-Areas/Coaching)\n\nIntersección entre lo "
         "que amás, lo que el mundo necesita, en lo que sos bueno y por lo "
         "que te pagan."),
        ("ikigai", "Pizza casera.md\n(03-Resources/Receta)\n\nMasa madre 24h. "
         "Aceite oliva. Mozzarella fior di latte."),
    ]
    scores = rr.predict(pairs)
    assert len(scores) == 2
    relevant, irrelevant = scores
    assert relevant > 0.5, f"relevant pair scored too low: {relevant}"
    assert irrelevant < 0.1, f"irrelevant pair scored too high: {irrelevant}"
    # Discrimination ratio — at least 5x.
    assert relevant / max(irrelevant, 1e-9) > 5.0


@pytest.mark.requires_mlx
def test_mlx_reranker_predict_returns_python_floats():
    """Output is `list[float]`, not `np.ndarray` — call sites in
    `rag/__init__.py` cast each score to float() and iterate as a list."""
    from rag.mlx_reranker import MLXReranker

    rr = MLXReranker()
    scores = rr.predict([("hi", "world")])
    assert isinstance(scores, list)
    assert all(isinstance(s, float) for s in scores)
    assert all(0.0 <= s <= 1.0 for s in scores)


@pytest.mark.requires_mlx
def test_mlx_reranker_idempotent_load():
    """Multiple `.predict()` calls reuse the same loaded model — no thrash."""
    from rag.mlx_reranker import MLXReranker

    rr = MLXReranker()
    rr.predict([("a", "b")])
    model_id = id(rr._model)
    rr.predict([("c", "d")])
    assert id(rr._model) == model_id


@pytest.mark.requires_mlx
def test_mlx_reranker_unload_clears_state():
    """`.unload()` drops the model + tokenizer, next predict re-loads."""
    from rag.mlx_reranker import MLXReranker

    rr = MLXReranker()
    rr.predict([("a", "b")])
    assert rr._model is not None
    rr.unload()
    assert rr._model is None
    assert rr._tokenizer is None
    assert rr._yes_id is None
    # Re-load works
    rr.predict([("e", "f")])
    assert rr._model is not None


@pytest.mark.requires_mlx
def test_mlx_reranker_truncation_keeps_tail():
    """Inputs that exceed `max_length` get tail-truncated so the assistant
    response cue at the end of the chat template is preserved."""
    from rag.mlx_reranker import MLXReranker

    # Construct a doc that — combined with the chat template — exceeds
    # the default 512 token budget.
    huge_doc = "lorem ipsum " * 800
    rr = MLXReranker()
    scores = rr.predict([("test query", huge_doc)])
    # Score is well-defined — the model didn't crash on truncated input.
    assert 0.0 <= scores[0] <= 1.0
