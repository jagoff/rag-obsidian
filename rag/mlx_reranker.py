"""MLX cross-encoder reranker — Qwen3-Reranker-0.6B-mxfp8.

Drop-in replacement for `sentence_transformers.CrossEncoder.predict(pairs)`.
Output shape: numpy.ndarray of float scores (probability 0-1), one per pair.

## Why a separate module

`rag/llm_backend.py` is chat/generate/embed scoped — it doesn't deal with
ranking. The reranker has its own load/cache/predict lifecycle (cross-
encoder, not a generative LLM in the chat sense), and shares the
title-prefix invariant from `project_rerank_title_prefix` memory.

## How the score is computed

Qwen3-Reranker is a fine-tuned generative LM trained to emit `yes`/`no`
in response to a prompt of the form:

    <Instruct>: Given a web search query, retrieve relevant passages...
    <Query>: {query}
    <Document>: {document}

Wrapped in the Qwen3 chat template with `enable_thinking=False`. After a
forward pass we read the logits at the last position and compute:

    P(relevant) = sigmoid(logit_yes - logit_no)

Result is a probability in [0, 1]. **NOT** a logit like bge-reranker-v2-m3
returns — downstream consumers (ranker.json weights, CONFIDENCE_RERANK_MIN
threshold) must be re-calibrated. See CLAUDE.md §Reranker invariants and
the project memory `decision/cierre-formal-ola-5-mlx-migration` for the
risk discussion.

## Discrimination smoke (2026-05-06)

Verified on `mlx-community/Qwen3-Reranker-0.6B-mxfp8`:
- Strongly relevant pair (e.g. "ikigai" → Ikigai.md note): P=0.9876
- Irrelevant pair (e.g. "info finops" → Pizza casera.md): P≈0.0000
- Discrimination 4-5 orders of magnitude — much peakier than bge-reranker.

## Threading

`MLXReranker` is thread-safe for concurrent `.predict()` calls — the
underlying `mlx_lm` model is reentrant for forward passes (no shared
mutable state in the model.eval() path). The lazy `load()` is guarded
by `_load_lock` so only one thread pays the cold-load cost.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Iterable

# Default model — exported so `_resolve_reranker_model_path` can pick it
# up when `RAG_RERANKER_BACKEND=mlx` (mxfp8 quant ~600 MB on disk).
DEFAULT_MLX_RERANKER = "mlx-community/Qwen3-Reranker-0.6B-mxfp8"

# Larger tier candidates for A/B if quality regresses on the 0.6B.
# Not used by default. Naming preserved from huggingface mlx-community.
MLX_RERANKER_ALIASES: dict[str, str] = {
    "qwen3-reranker:0.6b": "mlx-community/Qwen3-Reranker-0.6B-mxfp8",
    "qwen3-reranker:4b":   "mlx-community/Qwen3-Reranker-4B-mxfp8",
    "qwen3-reranker:8b":   "mlx-community/Qwen3-Reranker-8B-mxfp8",
    "jina-reranker:v3":    "mlx-community/jina-reranker-v3-4bit-mxfp4",
}

_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)


class MLXReranker:
    """Drop-in replacement for `sentence_transformers.CrossEncoder` using MLX.

    API surface kept minimal — only `.predict(pairs, show_progress_bar=...)`
    matches the production call sites in `rag/__init__.py`.

    Thread-safe lazy load + reentrant forward.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MLX_RERANKER,
        max_length: int = 512,
        instruction: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.max_length = max_length
        self.instruction = instruction or _INSTRUCTION
        self._model = None
        self._tokenizer = None
        self._yes_id: int | None = None
        self._no_id: int | None = None
        self._load_lock = threading.Lock()
        self._last_use: float = 0.0

    # -- internal -----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from mlx_lm import load as _mlx_load
            self._model, self._tokenizer = _mlx_load(self.model_path)
            # The model emits literal "yes" / "no" tokens — capture their IDs
            # once so each predict() call doesn't re-tokenize the string.
            # `add_special_tokens=False` matters: with default the encode adds
            # BOS that shifts the ID we want.
            self._yes_id = self._tokenizer.encode("yes", add_special_tokens=False)[-1]
            self._no_id = self._tokenizer.encode("no", add_special_tokens=False)[-1]

    def _build_prompt(self, query: str, doc: str) -> str:
        user_content = (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ]
        # `enable_thinking=False` is mandatory: with thinking on, the first
        # generation token after the assistant header is `<think>`, not the
        # yes/no judgement we score. Smoke tests on 2026-05-06 showed P
        # collapsing to ~0.001 across the board with thinking enabled.
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _score_pair(self, query: str, doc: str) -> float:
        import mlx.core as mx
        prompt = self._build_prompt(query, doc)
        ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        # Truncate from the right end of the document if the prompt exceeds
        # max_length. Right-truncation preserves the trailing assistant
        # prompt suffix that the model uses to know where to predict.
        if len(ids) > self.max_length:
            # Keep the last `max_length` tokens — preserves Instruct/Query
            # framing because they are short relative to the body and live
            # at the *front*. With Qwen3-Reranker we accept losing the head
            # of the doc body before losing the response cue at the tail.
            # An alternative would be to truncate the doc string before
            # template formatting, but that requires re-tokenising the
            # assistant suffix and is harder to keep correct.
            ids = ids[-self.max_length :]
        arr = mx.array([ids])
        logits = self._model(arr)[0, -1, :]
        yes = float(logits[self._yes_id])
        no = float(logits[self._no_id])
        return 1.0 / (1.0 + math.exp(-(yes - no)))

    # -- public -------------------------------------------------------------

    def predict(
        self,
        pairs: Iterable[tuple[str, str]],
        show_progress_bar: bool = False,  # noqa: ARG002 — API parity with CrossEncoder
        batch_size: int | None = None,    # noqa: ARG002
        **kwargs: Any,                     # noqa: ARG002
    ) -> list[float]:
        """Score each (query, document) pair. Returns a list of probabilities.

        Returned type is `list[float]` rather than `np.ndarray` to avoid
        forcing a numpy dependency on call sites that consume the result —
        the existing call sites in `rag/__init__.py` iterate the result and
        cast each score to `float()` already.
        """
        self._ensure_loaded()
        self._last_use = time.time()
        return [self._score_pair(q, d) for q, d in pairs]

    def unload(self) -> None:
        """Drop the model from memory + clear MLX cache. Idempotent."""
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            self._yes_id = None
            self._no_id = None
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass

    @property
    def last_use(self) -> float:
        """Wall-time epoch of last `.predict()` call. Used by idle-unload."""
        return self._last_use


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def is_mlx_reranker_enabled() -> bool:
    """True when `RAG_RERANKER_BACKEND=mlx` (env-controlled, default off
    until eval gate validates).

    The default is `torch` (sentence-transformers + bge-reranker-v2-m3 on
    MPS) so production stays on the calibrated baseline until a deliberate
    cutover.
    """
    return os.environ.get("RAG_RERANKER_BACKEND", "torch").lower() == "mlx"


def resolve_mlx_reranker_path(model: str | None = None) -> str:
    """Resolve a reranker name (Ollama-style, alias, or HF id) to the MLX HF id.

    Falls back to `DEFAULT_MLX_RERANKER` when the input is the bge-reranker
    name or empty — that's the typical state right after the swap when call
    sites still pass `RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"`.
    """
    if not model or model.startswith("BAAI/"):
        return DEFAULT_MLX_RERANKER
    if model.startswith("mlx-community/"):
        return model
    return MLX_RERANKER_ALIASES.get(model, model)


__all__ = [
    "DEFAULT_MLX_RERANKER",
    "MLX_RERANKER_ALIASES",
    "MLXReranker",
    "is_mlx_reranker_enabled",
    "resolve_mlx_reranker_path",
]
