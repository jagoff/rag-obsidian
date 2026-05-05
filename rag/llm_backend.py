"""LLM backend abstraction — Ollama → MLX migration.

Created 2026-05-05 as part of `99-AI/system/mlx-migration/dispatch.md`.

## Status: Ola 1 scaffold (compatibility-mode default Ollama)

This module defines a thin interface (`LLMBackend`) over the LLM call
contract used across `rag/__init__.py`. Two concrete backends:

- `OllamaBackend` — wraps the existing `ollama` Python client. **Default
  during the migration window** so master stays green even if MLX models
  haven't finished downloading.
- `MLXBackend` — uses `mlx-lm` (Apple MLX). **Default once the four MLX
  models are local + Ola 4 eval gate passes**.

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
  Qwen3-30B (~17 GB) NEVER coexists with qwen2.5:7b (~4.3 GB) when
  unified RAM <32 GB.

## Model name aliasing

To avoid touching 28 call sites in Ola 1, the backend accepts both
Ollama-style names (`qwen2.5:3b`) and MLX-style names
(`mlx-community/Qwen2.5-3B-Instruct-4bit`). `_resolve_alias()` maps
between them based on `MLX_MODEL_ALIAS` table. Call sites get migrated
to canonical MLX names in Ola 2.

## TODO (Ola 2+)

- [ ] Implement `MLXBackend.chat()` with mlx_lm.generate + chat template
- [ ] Implement `MLXBackend.generate()` (raw, no chat template)
- [ ] LRU eviction for resident MLX processes
- [ ] JSON-mode robustness (parse + repair) parity with Ollama
- [ ] Tool-calling format adapter (command-r → Qwen3 schema)
- [ ] Bench harness wiring (benchmarks/mlx_vs_ollama.py)
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
    """Apple MLX backend via `mlx-lm`. Resident-process + LRU eviction.

    Ola 2 work — scaffold only. Raises NotImplementedError until the
    Ola 2 agents wire `mlx_lm.generate` + chat template + JSON mode.
    """

    name = "mlx"

    # Models that NEVER coexist in unified RAM <32 GB (LRU eviction key)
    _BIG_MODELS = frozenset(
        {
            "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
        }
    )

    def __init__(self) -> None:
        # Lazy import to avoid hard dep when backend is `ollama`
        try:
            import mlx_lm  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "mlx-lm not installed. Run `uv add mlx-lm` or set "
                "RAG_LLM_BACKEND=ollama."
            ) from e

        self._loaded: dict[str, Any] = {}  # model_id → (model, tokenizer)
        self._idle_ttl = int(os.environ.get("RAG_MLX_IDLE_TTL", "1800"))

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: ChatOptions | None = None,
        keep_alive: str | int = -1,
        format: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError("Ola 2 work — see dispatch.md")

    def generate(
        self,
        model: str,
        prompt: str,
        options: ChatOptions | None = None,
        keep_alive: str | int = -1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError("Ola 2 work — see dispatch.md")

    def list_available(self) -> list[str]:
        # Scan ~/.cache/huggingface/hub/ for mlx-community models
        from pathlib import Path

        hub = Path.home() / ".cache" / "huggingface" / "hub"
        if not hub.exists():
            return []
        return [
            d.name.replace("models--", "").replace("--", "/")
            for d in hub.iterdir()
            if d.is_dir() and "mlx-community" in d.name
        ]


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
