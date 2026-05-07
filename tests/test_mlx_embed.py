"""Tests for MLXBackend.embed() — Phase 1 Scope B.

All tests mock `mlx_lm.load` and the model internals so they never
load a real HuggingFace model. The embedding logic (last-real-token
pooling, L2-normalisation, shape validation) is exercised with a
FakeEmbedModel that returns deterministic numpy arrays.

Mark: @pytest.mark.requires_mlx — skipped when mlx_lm not installed.
"""

from __future__ import annotations

import math
import unittest.mock as mock

import pytest

mlx_lm = pytest.importorskip("mlx_lm")  # noqa: F841

import mlx.core as mx  # type: ignore[import-not-found]
import numpy as np

from rag.llm_backend import MLXBackend, reset_backend

pytestmark = pytest.mark.requires_mlx

# ---------------------------------------------------------------------------
# Shared fake infrastructure
# ---------------------------------------------------------------------------

_HIDDEN_DIM = 1024
_VOCAB_SIZE = 32000


def _make_fake_hidden(batch_size: int, seq_len: int, dim: int = _HIDDEN_DIM) -> mx.array:
    """Deterministic fake hidden-state (batch, seq, dim)."""
    np.random.seed(42)
    arr = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
    return mx.array(arr)


class _FakeEmbedModel:
    """Minimal fake that replaces the mlx-lm Qwen3-Embedding model.

    The `.model(input_ids)` call (body forward, bypassing lm_head) is what
    MLXBackend.embed() invokes. We return a deterministic tensor with the
    right shape. The actual content is a fixed random matrix — enough to
    verify pooling + normalisation logic without real weights.
    """

    class _Body:
        def __call__(self, input_ids: mx.array) -> mx.array:
            batch, seq = input_ids.shape
            return _make_fake_hidden(batch, seq)

    def __init__(self) -> None:
        self.model = self._Body()


class _FakeTokenizer:
    """Fake HuggingFace tokenizer for embed tests.

    Returns numpy arrays (same format as the real tokenizer with
    `return_tensors="np"`).
    """

    pad_token_id = 0

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "np",
    ) -> dict[str, np.ndarray]:
        batch = len(texts)
        # Simulate variable-length tokenisation: each token = 1 token per char,
        # max 20 tokens. Shorter sequences are right-padded with 0.
        lengths = [min(len(t), 20) for t in texts]
        max_len = max(lengths)
        input_ids = np.zeros((batch, max_len), dtype=np.int32)
        attention_mask = np.zeros((batch, max_len), dtype=np.int32)
        for i, length in enumerate(lengths):
            input_ids[i, :length] = np.arange(1, length + 1, dtype=np.int32)
            attention_mask[i, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture(autouse=True)
def _reset_backend():
    reset_backend()
    yield
    reset_backend()


@pytest.fixture
def backend_with_fake_embed(monkeypatch):
    """MLXBackend with mlx_lm.load patched to return FakeEmbedModel + FakeTokenizer."""
    fake_model = _FakeEmbedModel()
    fake_tokenizer = _FakeTokenizer()

    with mock.patch("mlx_lm.load", return_value=(fake_model, fake_tokenizer)):
        backend = MLXBackend()
        yield backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mlx_backend_embed_returns_1024_dim(backend_with_fake_embed):
    """Smoke: single input returns a 1024-dim vector."""
    resp = backend_with_fake_embed.embed(
        model="qwen3-embedding:0.6b",
        inputs=["hola mundo"],
    )
    assert "embeddings" in resp
    assert len(resp["embeddings"]) == 1
    assert len(resp["embeddings"][0]) == _HIDDEN_DIM


def test_mlx_backend_embed_l2_normalized(backend_with_fake_embed):
    """Each output vector must have L2-norm ≈ 1.0."""
    resp = backend_with_fake_embed.embed(
        model="qwen3-embedding:0.6b",
        inputs=["hola mundo", "que tal"],
    )
    for vec in resp["embeddings"]:
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-4, f"L2-norm not 1.0: got {norm}"


def test_mlx_backend_embed_batch(backend_with_fake_embed):
    """3-input batch returns 3 vectors."""
    inputs = ["primer texto", "segundo texto", "tercer texto con más palabras"]
    resp = backend_with_fake_embed.embed(
        model="qwen3-embedding:0.6b",
        inputs=inputs,
    )
    assert len(resp["embeddings"]) == 3
    for vec in resp["embeddings"]:
        assert len(vec) == _HIDDEN_DIM


def test_mlx_backend_embed_cosine_similarity_same_text(backend_with_fake_embed):
    """Same text embedded in two separate calls → cosine similarity == 1.0.

    The fake model returns a deterministic tensor seeded at 42, producing the
    same hidden state for batch_size=1 each call. Both embeddings are normalised
    to unit vectors; their dot product (= cosine similarity) must be exactly 1.0
    (within floating-point precision).

    This validates that the pooling + L2-normalisation pipeline is deterministic:
    the same input always produces the same output vector.
    """
    resp_a = backend_with_fake_embed.embed(
        model="qwen3-embedding:0.6b",
        inputs=["hola mundo"],
    )
    resp_b = backend_with_fake_embed.embed(
        model="qwen3-embedding:0.6b",
        inputs=["hola mundo"],
    )
    a = resp_a["embeddings"][0]
    b = resp_b["embeddings"][0]
    dot = sum(x * y for x, y in zip(a, b))
    # Same input → same unit vector → dot product == 1.0
    assert dot > 0.9999, f"Expected cosine-sim ≈ 1.0 for identical inputs, got {dot}"


def test_mlx_backend_embed_model_alias_resolved(monkeypatch):
    """MLX_MODEL_ALIAS maps qwen3-embedding:0.6b to the HF id."""
    from rag.llm_backend import MLX_MODEL_ALIAS

    assert "qwen3-embedding:0.6b" in MLX_MODEL_ALIAS
    # Migración 2026-05-06: 4bit-DWQ (cos ~0.97 vs PyTorch fp16, requiere
    # reindex) → 8bit (cos ≥0.9977, bit-equivalente funcional). Validación
    # detallada en docstring de rag/mlx_embed.py.
    assert "Qwen3-Embedding-0.6B-8bit" in MLX_MODEL_ALIAS["qwen3-embedding:0.6b"]


def test_mlx_backend_embed_caches_model(backend_with_fake_embed):
    """Second call with the same model does NOT reload (cache hit)."""
    with mock.patch("mlx_lm.load", return_value=(_FakeEmbedModel(), _FakeTokenizer())) as load_mock:
        backend = backend_with_fake_embed
        # Prime the cache
        backend.embed(model="qwen3-embedding:0.6b", inputs=["primer call"])
        call_count_after_first = load_mock.call_count

        # Second call — should hit _loaded_embed, no mlx_lm.load
        backend.embed(model="qwen3-embedding:0.6b", inputs=["segundo call"])
        # mlx_lm.load must NOT have been called again for the second embed
        assert load_mock.call_count == call_count_after_first


def test_mlx_backend_embed_unload_clears_embed_cache(backend_with_fake_embed):
    """unload(None) clears both chat _loaded and _loaded_embed."""
    backend = backend_with_fake_embed
    backend.embed(model="qwen3-embedding:0.6b", inputs=["un texto"])
    assert len(backend._loaded_embed) == 1

    backend.unload(None)
    assert len(backend._loaded_embed) == 0
    assert len(backend._loaded) == 0


# Ola 7+ (2026-05-06): `test_ollama_backend_embed_passthrough` borrado.
# Ejercía `OllamaBackend.embed()` que ya no existe — la clase se retiró
# en Ola 7 y los tipos `Message`/`ChatResponse` ahora son pydantic
# locales (no `from ollama._types import ...`).
