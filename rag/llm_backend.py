"""LLM backend abstraction — Ollama → MLX migration.

Created 2026-05-05 as part of `99-AI/system/mlx-migration/dispatch.md`.

## Status: Ola 2 (en curso) — MLXBackend functional

This module defines a thin interface (`LLMBackend`) over the LLM call
contract used across `rag/__init__.py`. Two concrete backends:

- `OllamaBackend` — wraps the existing `ollama` Python client. **Default
  during the migration window** so master stays green even if MLX models
  haven't finished downloading.
- `MLXBackend` — uses `mlx-lm` (Apple MLX). Functional as of Ola 2:
  `chat()`, `chat_stream()`, `generate()` all implemented + smoke-tested
  OK for all 4 models (2026-05-05). **Default once Ola 4 eval gate passes**.

Switch via env var `RAG_LLM_BACKEND={ollama,mlx}` (default `ollama`
during the migration window, `mlx` post-cutover).

## Scope (NOT embeddings)

This backend covers `chat()` + `generate()` only. Embeddings (`bge-m3`
today) are out of scope — see `99-AI/system/embedding-swap-qwen3-8b/`
for the parallel embedding migration.

## Invariants preserved

- HELPER_OPTIONS (temperature=0, seed=42) — eval reproducibility floor.
- CHAT_OPTIONS (num_ctx=4096, num_predict=768) — VRAM-budgeted.
- `keep_alive=-1` semantics → emulated via resident-process + LRU on
  the MLX side (MLX has no native keep_alive). Eviction policy:
  Qwen3-30B (~17 GB) is single-tenant (_BIG_MODELS), evicts everything
  else on load. Small models LRU-capped at _MAX_SMALL_LOADED=3.

## Model name aliasing

The backend accepts both Ollama-style names (`qwen2.5:3b`) and MLX HF
IDs (`mlx-community/Qwen2.5-3B-Instruct-4bit`). `to_mlx()` / `to_ollama()`
resolve between them via `MLX_MODEL_ALIAS` table.

## What MLX does NOT support (vs Ollama)

- `tools=[...]` (tool-calling): kwarg is dropped by `_mlx_chat_via_backend`.
  Call sites that need tool-calling JSON schema still go via Ollama or need
  a custom Qwen3 parser (Ola 2 work).
- `keep_alive=0` (model unload trick): ignored silently — no-op.
- Grammar-constrained JSON decode: `format='json'` uses system-prompt nudge
  + `_extract_json()` post-gen instead.

## TODO (Ola 2 remaining + Ola 3+)

- [ ] Migrate 14 raw `ollama.chat(stream=True, ...)` call sites to `get_backend().chat_stream(...)`
- [ ] Tool-calling format adapter (Qwen3 `<tool_call>` JSON schema)
- [ ] Idle-unload watchdog thread (RAG_MLX_IDLE_TTL enforcement)
- [ ] Bench harness wiring (benchmarks/bench_mlx_vs_ollama.py)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Model aliases: Ollama name ↔ MLX HuggingFace ID
# ---------------------------------------------------------------------------

MLX_MODEL_ALIAS: dict[str, str] = {
    # Helper tier (deterministic, temp=0, seed=42)
    "qwen2.5:3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    # Chat default
    "qwen2.5:7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    # High-quality tier (contradictions, brief JSON, `rag do`, HyDE re-test)
    "command-r:latest": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    "command-r": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    "qwen2.5:14b": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    # Experimental (A/B vs the 3B helper, NOT default until eval CIs OK)
    "qwen3:4b": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
}

OLLAMA_MODEL_ALIAS: dict[str, str] = {v: k for k, v in MLX_MODEL_ALIAS.items()}


def to_mlx(model: str) -> str:
    """Resolve any model name to its MLX HuggingFace ID."""
    if model.startswith("mlx-community/"):
        return model
    return MLX_MODEL_ALIAS.get(model, model)


def to_ollama(model: str) -> str:
    """Resolve any model name to its Ollama short name."""
    return OLLAMA_MODEL_ALIAS.get(model, model)


# ---------------------------------------------------------------------------
# Backend contract
# ---------------------------------------------------------------------------


@dataclass
class ChatOptions:
    """Sampling + context options. Mirrors Ollama's `options` dict.

    Defaults match `HELPER_OPTIONS` (temperature=0, seed=42) — call sites
    that need chat-tier sampling pass `CHAT_OPTIONS` overrides.
    """

    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 4096
    num_predict: int = 768
    top_p: float = 1.0
    stop: tuple[str, ...] = ()


class LLMBackend(ABC):
    """Common interface for Ollama / MLX backends."""

    name: str = "abstract"

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: ChatOptions | None = None,
        keep_alive: str | int = -1,
        format: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Chat completion. Returns Ollama-shape dict for compat."""

    @abstractmethod
    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: ChatOptions | None = None,
        keep_alive: str | int = -1,
        format: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Streaming chat. Yields ChatResponse objects.

        Each intermediate yield has `done=False` and `.message.content`
        containing the incremental token piece. The terminal yield has
        `done=True, done_reason='stop'` and empty `.message.content`.

        Call sites iterate with:
            for chunk in backend.chat_stream(...):
                piece = chunk.message.content
        """

    @abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        options: ChatOptions | None = None,
        keep_alive: str | int = -1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Raw generate (no chat template)."""

    @abstractmethod
    def list_available(self) -> list[str]:
        """Return list of locally-available model names (canonical)."""


# ---------------------------------------------------------------------------
# OllamaBackend (legacy — default during migration window)
# ---------------------------------------------------------------------------


class OllamaBackend(LLMBackend):
    """Wraps `ollama` Python client. Identity passthrough."""

    name = "ollama"

    def __init__(self) -> None:
        import ollama  # type: ignore[import-not-found]

        self._ollama = ollama

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: ChatOptions | None = None,
        keep_alive: str | int = -1,
        format: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        opts = self._opts_to_dict(options)
        call_kwargs = dict(
            model=to_ollama(model),
            messages=messages,
            options=opts,
            keep_alive=keep_alive,
            **kwargs,
        )
        if format is not None:
            call_kwargs["format"] = format
        return self._ollama.chat(**call_kwargs)

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: ChatOptions | None = None,
        keep_alive: str | int = -1,
        format: str | None = None,
        **kwargs: Any,
    ) -> Any:
        opts = self._opts_to_dict(options)
        call_kwargs = dict(
            model=to_ollama(model),
            messages=messages,
            options=opts,
            keep_alive=keep_alive,
            stream=True,
            **kwargs,
        )
        if format is not None:
            call_kwargs["format"] = format
        yield from self._ollama.chat(**call_kwargs)

    def generate(
        self,
        model: str,
        prompt: str,
        options: ChatOptions | None = None,
        keep_alive: str | int = -1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        opts = self._opts_to_dict(options)
        return self._ollama.generate(
            model=to_ollama(model),
            prompt=prompt,
            options=opts,
            keep_alive=keep_alive,
            **kwargs,
        )

    def list_available(self) -> list[str]:
        return [m.model for m in self._ollama.list().models]

    @staticmethod
    def _opts_to_dict(options: ChatOptions | None) -> dict[str, Any]:
        if options is None:
            options = ChatOptions()
        d: dict[str, Any] = {
            "temperature": options.temperature,
            "seed": options.seed,
            "num_ctx": options.num_ctx,
            "num_predict": options.num_predict,
            "top_p": options.top_p,
        }
        if options.stop:
            d["stop"] = list(options.stop)
        return d


# ---------------------------------------------------------------------------
# MLXBackend — Ola 2 implementation pending
# ---------------------------------------------------------------------------


class MLXBackend(LLMBackend):
    """Apple MLX backend via `mlx-lm`. Resident-models + LRU eviction.

    Returns ollama-shape responses (`ChatResponse` / `GenerateResponse`)
    so call sites that read `r["message"]["content"]` or `r.message.content`
    keep working without changes.

    ## Determinism

    `temperature=0` → greedy decoding (argmax) → bit-exact reproducible
    on the same hardware. `seed` is honored via `mx.random.seed()` when
    `temperature > 0` (sampling path).

    ## VRAM management

    `_loaded` holds resident `(model, tokenizer)` tuples keyed by HF repo
    id. Eviction policy:

    - When loading a `_BIG_MODELS` member (Qwen3-30B-A3B ~17 GB):
      evict ALL other models first (pure single-tenant on big-tier).
    - Otherwise: LRU cap at `_MAX_LOADED` (default 3 small models OR
      1 big + 0 small).

    `_idle_ttl` (env `RAG_MLX_IDLE_TTL`, default 1800s) is a hint for
    a future watchdog thread; not yet enforced. Eviction today is
    purely capacity-driven.

    ## Streaming semantics (`chat_stream`)

    `chat_stream(...)` yields `ChatResponse` objects via
    `mlx_lm.stream_generate`. Each intermediate yield has `done=False`
    and `.message.content` containing the incremental token piece
    (may be a sub-word fragment, a word, or multiple tokens depending
    on the MLX flush granularity). The terminal yield has
    `done=True, done_reason='stop'` and empty `.message.content`.

    Chat template formatting and JSON-mode nudge work identically to
    `chat()` — the same `_apply_chat_template` helper is called.
    """

    name = "mlx"

    # Models that NEVER coexist with anything else (pure single-tenant)
    _BIG_MODELS = frozenset(
        {
            "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
        }
    )

    _MAX_SMALL_LOADED = 3

    def __init__(self) -> None:
        # Lazy import to avoid hard dep when backend is `ollama`
        try:
            import mlx_lm  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "mlx-lm not installed. Run `uv pip install '.[mlx]'` or "
                "set RAG_LLM_BACKEND=ollama."
            ) from e

        # OrderedDict to preserve LRU order: oldest first, newest last
        from collections import OrderedDict

        self._loaded: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._idle_ttl = int(os.environ.get("RAG_MLX_IDLE_TTL", "1800"))

    def _load(self, model_id: str) -> tuple[Any, Any]:
        """Get or load (model, tokenizer) for `model_id`. LRU bump on hit."""
        canonical = to_mlx(model_id)
        if canonical in self._loaded:
            self._loaded.move_to_end(canonical)
            return self._loaded[canonical]

        # Eviction BEFORE load (free RAM first)
        self._evict_for(canonical)

        from mlx_lm import load  # type: ignore[import-not-found]

        model, tokenizer = load(canonical)
        self._loaded[canonical] = (model, tokenizer)
        return (model, tokenizer)

    def _evict_for(self, incoming: str) -> None:
        """Evict resident models to make room for `incoming`."""
        is_big = incoming in self._BIG_MODELS

        if is_big:
            # Big models are single-tenant: evict everything else.
            for k in list(self._loaded.keys()):
                if k != incoming:
                    self._loaded.pop(k, None)
            return

        # Incoming is small: evict any big, then LRU-trim small ones.
        for k in list(self._loaded.keys()):
            if k in self._BIG_MODELS:
                self._loaded.pop(k, None)

        while len(self._loaded) >= self._MAX_SMALL_LOADED:
            self._loaded.popitem(last=False)  # evict LRU (oldest)

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: ChatOptions | None = None,
        keep_alive: str | int = -1,
        format: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from ollama._types import ChatResponse, Message  # type: ignore[import-not-found]

        opts = options or ChatOptions()
        mlx_model, tokenizer = self._load(model)
        prompt = self._apply_chat_template(tokenizer, messages, format=format)
        text = self._mlx_generate(mlx_model, tokenizer, prompt, opts)
        if format == "json":
            text = self._extract_json(text)
        return ChatResponse(
            model=to_ollama(model),
            message=Message(role="assistant", content=text),
            done=True,
            done_reason="stop",
        )

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: ChatOptions | None = None,
        keep_alive: str | int = -1,
        format: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from mlx_lm import stream_generate  # type: ignore[import-not-found]
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]
        from ollama._types import ChatResponse, Message  # type: ignore[import-not-found]

        opts = options or ChatOptions()
        mlx_model, tokenizer = self._load(model)
        prompt = self._apply_chat_template(tokenizer, messages, format=format)

        sampler = make_sampler(temp=opts.temperature, top_p=opts.top_p)
        if opts.temperature > 0:
            import mlx.core as mx  # type: ignore[import-not-found]

            mx.random.seed(opts.seed)

        ollama_model = to_ollama(model)
        for response in stream_generate(
            mlx_model,
            tokenizer,
            prompt,
            max_tokens=opts.num_predict,
            sampler=sampler,
        ):
            yield ChatResponse(
                model=ollama_model,
                message=Message(role="assistant", content=response.text),
                done=False,
            )

        yield ChatResponse(
            model=ollama_model,
            message=Message(role="assistant", content=""),
            done=True,
            done_reason="stop",
        )

    def generate(
        self,
        model: str,
        prompt: str,
        options: ChatOptions | None = None,
        keep_alive: str | int = -1,
        **kwargs: Any,
    ) -> Any:
        from ollama._types import GenerateResponse  # type: ignore[import-not-found]

        opts = options or ChatOptions()
        mlx_model, tokenizer = self._load(model)
        text = self._mlx_generate(mlx_model, tokenizer, prompt, opts)
        return GenerateResponse(
            model=to_ollama(model),
            response=text,
            done=True,
            done_reason="stop",
        )

    @staticmethod
    def _apply_chat_template(
        tokenizer: Any,
        messages: list[dict[str, str]],
        format: str | None = None,
    ) -> str:
        """Apply tokenizer's chat template; nudge JSON mode in system msg."""
        msgs = list(messages)
        if format == "json":
            # Best-effort JSON mode for models without native grammar.
            # Prepend a strong instruction; parser does repair downstream.
            extra = (
                "Respond with ONLY a single valid JSON object. "
                "No prose, no markdown fences, no commentary."
            )
            if msgs and msgs[0].get("role") == "system":
                msgs[0] = {
                    **msgs[0],
                    "content": msgs[0]["content"] + "\n\n" + extra,
                }
            else:
                msgs = [{"role": "system", "content": extra}, *msgs]

        return tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )

    @staticmethod
    def _mlx_generate(model: Any, tokenizer: Any, prompt: str, opts: ChatOptions) -> str:
        from mlx_lm import generate  # type: ignore[import-not-found]
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]

        sampler = make_sampler(temp=opts.temperature, top_p=opts.top_p)
        # Seed for reproducibility when sampling (temp>0). Greedy (temp=0)
        # is bit-exact deterministic on same hardware regardless of seed.
        if opts.temperature > 0:
            import mlx.core as mx  # type: ignore[import-not-found]

            mx.random.seed(opts.seed)

        return generate(
            model,
            tokenizer,
            prompt,
            max_tokens=opts.num_predict,
            sampler=sampler,
            verbose=False,
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Best-effort: strip markdown fences + isolate first {...} block.

        Models without native grammar mode often wrap JSON in ```json ... ```
        or prepend prose. The downstream parser in `rag/__init__.py` already
        handles malformed JSON via repair; this just gives it a cleaner
        starting point.
        """
        s = text.strip()
        # Strip ```json ... ``` fences
        if s.startswith("```"):
            s = s.split("\n", 1)[-1] if "\n" in s else s
            if s.endswith("```"):
                s = s[:-3]
            s = s.strip()
        # Isolate first balanced {...} (naive, parser does the heavy lifting)
        first = s.find("{")
        last = s.rfind("}")
        if first >= 0 and last > first:
            return s[first : last + 1]
        return s

    def list_available(self) -> list[str]:
        # Scan ~/.cache/huggingface/hub/ for mlx-community models
        from pathlib import Path

        hub = Path.home() / ".cache" / "huggingface" / "hub"
        if not hub.exists():
            return []
        out: list[str] = []
        for d in hub.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            if "mlx-community" not in name:
                continue
            # `models--mlx-community--Qwen2.5-3B-Instruct-4bit` →
            # `mlx-community/Qwen2.5-3B-Instruct-4bit`
            stripped = name.removeprefix("models--")
            parts = stripped.split("--")
            if len(parts) >= 2:
                out.append(f"{parts[0]}/{'-'.join(parts[1:])}")
        return out


# ---------------------------------------------------------------------------
# Backend resolver
# ---------------------------------------------------------------------------


_BACKEND_SINGLETON: LLMBackend | None = None


def get_backend() -> LLMBackend:
    """Return the active LLM backend (singleton).

    Selection: env var `RAG_LLM_BACKEND` ∈ {ollama, mlx}. Default
    `ollama` during the migration window; flips to `mlx` post-cutover
    (Ola 5).
    """
    global _BACKEND_SINGLETON
    if _BACKEND_SINGLETON is not None:
        return _BACKEND_SINGLETON

    choice = os.environ.get("RAG_LLM_BACKEND", "ollama").lower()
    if choice == "mlx":
        _BACKEND_SINGLETON = MLXBackend()
    elif choice == "ollama":
        _BACKEND_SINGLETON = OllamaBackend()
    else:
        raise ValueError(
            f"RAG_LLM_BACKEND must be 'ollama' or 'mlx' (got {choice!r})"
        )
    return _BACKEND_SINGLETON


def reset_backend() -> None:
    """Force re-resolution on next get_backend() call. Tests only."""
    global _BACKEND_SINGLETON
    _BACKEND_SINGLETON = None


__all__ = [
    "ChatOptions",
    "LLMBackend",
    "MLXBackend",
    "MLX_MODEL_ALIAS",
    "OLLAMA_MODEL_ALIAS",
    "OllamaBackend",
    "get_backend",
    "reset_backend",
    "to_mlx",
    "to_ollama",
]
