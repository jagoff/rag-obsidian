"""LLM backend abstraction — Apple MLX in-process.

Single concrete backend: `MLXBackend` (mlx-lm). Cubre chat / generate /
embed / VLM (granite via mlx-vlm). Modelos: Qwen2.5-3B / 7B,
Qwen3-30B-A3B, Qwen3-Embedding-0.6B.

## Invariants

- HELPER_OPTIONS (temperature=0, seed=42) — eval reproducibility floor.
- CHAT_OPTIONS (num_ctx=4096, num_predict=384) — VRAM-budgeted.
- `keep_alive` kwarg: no-op (MLX in-process). Eviction via LRU + idle
  TTL (`RAG_MLX_IDLE_TTL`, default 1800s). Qwen3-30B (~17 GB) es
  single-tenant (`_BIG_MODELS`); small models LRU-capped a
  `_MAX_SMALL_LOADED=3`.

## Model name aliasing

Acepta tanto el alias corto (`qwen2.5:3b`) como el MLX HF id
(`mlx-community/Qwen2.5-3B-Instruct-4bit`). `to_mlx()` / `to_short_name()`
resuelven entre los dos vía `MLX_MODEL_ALIAS`.

## Tool-calling

`tools=[...]` se propaga a `tokenizer.apply_chat_template(tools=...)` —
Qwen2.5/3-Instruct templates lo soportan nativo. El modelo emite
`<tool_call>{...}</tool_call>` y `parse_tool_calls()` los extrae a
`Message.ToolCall`. Si el template no acepta `tools=`, el caller detecta
la ausencia de `tool_calls` en la respuesta y degrada gracefully.

## Format JSON

`format='json'` usa system-prompt nudge + `_extract_json()` post-gen
(MLX no tiene grammar-constrained decode nativo).
"""

from __future__ import annotations

import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Qwen3 <think> block stripping
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """Remove Qwen3-style <think>...</think> reasoning blocks before JSON parse.

    Qwen3-30B (HQ tier MLX) can emit thinking blocks before the actual
    response when greedy decoding falls into a thinking-mode pattern.
    These blocks contain prose AND brace characters that confuse JSON
    extractors and structured-output parsers downstream.

    No-op si no hay `<think>` tag (passthrough). NO strippea tags sin
    cerrar — output incompleto queda as-is.
    """
    if not text or "<think>" not in text.lower():
        return text
    return _THINK_BLOCK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Model aliases: short alias ↔ MLX HuggingFace ID
# ---------------------------------------------------------------------------

MLX_MODEL_ALIAS: dict[str, str] = {
    # Helper tier (deterministic, temp=0, seed=42)
    "qwen2.5:3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    # Chat default
    "qwen2.5:7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    # High-quality tier (contradictions, brief JSON, `rag do`, HyDE re-test).
    # DWQ (Dynamic Weight Quantization) recupera ~1-2pp de calidad vs 4bit
    # estándar al mismo costo de memoria/latencia. Rollback: cambiar suffix
    # `-DWQ` por `` (vacío) para volver al 4bit estándar.
    "command-r:latest": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ",
    "command-r": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ",
    "qwen2.5:14b": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ",
    # Listener WA usa `qwen2.5:14b-instruct` (DRAFT_HELPER_MODEL en plist) —
    # ruteamos al mismo MLX HQ tier que `qwen2.5:14b`. Necesario para que el
    # endpoint Ollama-compat (`/ollama/api/chat`) resuelva sin 404.
    "qwen2.5:14b-instruct": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ",
    "qwen3:30b-a3b": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ",
    "qwen3:30b": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ",
    # Experimental (A/B vs the 3B helper, NOT default until eval CIs OK).
    # DWQ-2510 = revision Oct 2025 con mejor recall en tasks JSON-structured.
    "qwen3:4b": "mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510",
    # Embedder — active path is MLX in-process via MLXEmbedder
    # (`rag.mlx_embed`). 8-bit quant elegida porque cosine ≥0.9977 vs
    # PyTorch fp16 reference (~bit-equivalente, NO requiere reindex);
    # variantes 4bit-DWQ (~0.97) y mxfp8 (~0.98) sí lo requerirían.
    # Validación en commit que migró el embedder a MLX (2026-05-06).
    "qwen3-embedding:0.6b": "mlx-community/Qwen3-Embedding-0.6B-8bit",
}

SHORT_NAME_ALIAS: dict[str, str] = {v: k for k, v in MLX_MODEL_ALIAS.items()}


def list_cached_mlx_models() -> list[str]:
    """Return cached MLX HuggingFace model IDs without importing MLX runtime."""
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


def to_mlx(model: str) -> str:
    """Resolve any model name to its MLX HuggingFace ID."""
    if model.startswith("mlx-community/"):
        return model
    return MLX_MODEL_ALIAS.get(model, model)


def to_short_name(model: str) -> str:
    """Resolve any MLX HuggingFace ID back to its short alias name.

    Inverse de `to_mlx()`. Útil para preservar el nombre corto en
    `ChatResponse.model` (más legible en logs/feedback que el HF id full).
    """
    return SHORT_NAME_ALIAS.get(model, model)


# ---------------------------------------------------------------------------
# Backend contract
# ---------------------------------------------------------------------------


@dataclass
class ChatOptions:
    """Sampling + context options. Mirrors the backend's `options` dict.

    Defaults match `HELPER_OPTIONS` (temperature=0, seed=42) — call sites
    that need chat-tier sampling pass `CHAT_OPTIONS` overrides.
    """

    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 4096
    # 384 matches CHAT_OPTIONS["num_predict"] in rag/__init__.py (median
    # answer_len=38 chars, p90 approx 200 tokens).
    num_predict: int = 384
    top_p: float = 1.0
    stop: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Response types (MLX response types)
# ---------------------------------------------------------------------------
# These mirror the shape that MLXBackend.chat() / generate() / chat_stream()
# return. Pydantic BaseModel preserva la API que los call sites ya esperan
# de los tipos MLX-shape:
#   - attribute access (resp.message.content, resp.done, ...)
#   - `tc.model_dump()` para serializar tool_calls a JSON
#   - nested ToolCall + Function dentro de Message para que
#     `parse_tool_calls()` pueda construir `Message.ToolCall(function=
#     Message.ToolCall.Function(name=..., arguments=...))`.


class _ToolCallFunction(BaseModel):
    """Function payload dentro de un tool call (name + arguments)."""

    name: str
    arguments: dict[str, Any] = {}


class _ToolCall(BaseModel):
    """Tool call emitido por el modelo (function name + arguments dict)."""

    function: _ToolCallFunction


class Message(BaseModel):
    """Single chat message (assistant turn). Mirrors Message shape."""

    role: str
    content: str | None = None
    tool_calls: list[_ToolCall] | None = None


# Attribute aliases post-class-body (pydantic v2 trata las asignaciones in-body
# como fields). Esto preserva la sintaxis MLX-shape original que los call
# sites ya esperan: `Message.ToolCall(function=Message.ToolCall.Function(...))`.
_ToolCall.Function = _ToolCallFunction  # type: ignore[attr-defined]
Message.ToolCall = _ToolCall  # type: ignore[attr-defined]


class ChatResponse(BaseModel):
    """Non-streaming chat completion response. MLX-shape response."""

    model: str
    message: Message
    done: bool
    done_reason: str | None = None


class GenerateResponse(BaseModel):
    """Raw generate (no chat template) response. MLX-shape response."""

    model: str
    response: str
    done: bool
    done_reason: str | None = None


class LLMBackend(ABC):
    """Common interface for LLM backends."""

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
        """Chat completion. Returns `ChatResponse` (pydantic BaseModel)."""

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
    def embed(
        self,
        model: str,
        inputs: list[str],
        keep_alive: str | int = -1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Batch embedding. Returns MLX-shape: {"embeddings": [[float, ...], ...]}.

        Each inner list is L2-normalised 1024-dim (qwen3-embedding compatible).
        """

    @abstractmethod
    def list_available(self) -> list[str]:
        """Return list of locally-available model names (canonical)."""

    @abstractmethod
    def unload(self, model: str | None = None) -> bool:
        """Best-effort: free RAM/VRAM held by `model` (or all if None).

        Returns True if anything was unloaded. Errors swallowed — never raise.
        """


# ---------------------------------------------------------------------------
# Forward lock global — serializa cualquier inference Metal MLX in-process
# (chat / chat_stream / generate / embed). Compartido por MLXBackend y
# por MLXEmbedder (rag/mlx_embed.py) para que un forward de embed y uno de
# chat NO colisionen en el mismo device. Sin este lock, el ThreadPool de
# `_home_compute` (web/server.py) lanza ~14 fetchers concurrentes — varios
# invocan MLX → command buffers Metal colisionan → kIOGPUCommandBufferCallback
# ErrorHang + InnocentVictim. El cold-load (`_load`) queda fuera del lock
# para que cargas paralelas de modelos distintos no se serialicen.
# ---------------------------------------------------------------------------
_MLX_FORWARD_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# MLXBackend
# ---------------------------------------------------------------------------


class MLXBackend(LLMBackend):
    """Apple MLX backend via `mlx-lm`. Resident-models + LRU eviction.

    Returns MLX-shape responses (`ChatResponse` / `GenerateResponse`)
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
            "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ",
        }
    )

    _MAX_SMALL_LOADED = 3

    def __init__(self) -> None:
        # Lazy import — mlx-lm es opcional en pyproject (extra `mlx`).
        try:
            import mlx_lm  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "mlx-lm not installed. Run `uv tool install --reinstall "
                "--editable '.[mlx]'`."
            ) from e
        except RuntimeError as e:
            raise RuntimeError(
                "mlx-lm is installed but MLX could not initialize a Metal device. "
                "Run this on an Apple Silicon/macOS session with Metal available."
            ) from e

        # OrderedDict to preserve LRU order: oldest first, newest last
        from collections import OrderedDict

        self._loaded: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._idle_ttl = int(os.environ.get("RAG_MLX_IDLE_TTL", "1800"))

        # Race-condition guard for `_loaded` mutation (P1 fix 2026-05-05).
        #
        # Under the multi-thread web server, `_load()`/`_evict_for()`/`unload()`
        # can interleave: thread A loading qwen2.5:7b while thread B loads
        # qwen2.5:3b while the memory watchdog calls `unload()`. The sequence
        # `check → evict → store` is NOT atomic without a lock — two threads
        # could each `mlx_lm.load()` the same model duplicating VRAM, or
        # `unload()` could pop a tuple while another thread is reading it.
        #
        # RLock (not Lock) because `_load()` calls `_evict_for()` while holding
        # the lock — re-entrancy avoids deadlock. Note: the heavy `mlx_lm.load()`
        # call itself is intentionally OUT of the lock (see `_load()` docstring)
        # so concurrent `chat()` calls on already-resident models don't block
        # behind a 3-8s cold load running in another thread.
        self._loaded_lock = threading.RLock()
        # Forward lock compartido a nivel módulo (`_MLX_FORWARD_LOCK`) — el
        # mismo lock vive en `rag.mlx_embed.MLXEmbedder` para que un embed y
        # un chat NO colisionen en Metal. Memo `obsidian_rag_web_service_gpu_hang_loop`
        # (2026-05-06): residual del web crash loop por concurrencia MLX en
        # `_home_compute`. El HTTP shim resuelve qwen2.5:7b puntual (proc aparte);
        # este lock cubre lo que queda in-process. NO toca `_load()` — solo el
        # forward — para que cold-load en otro thread no serialice contra forwards.
        self._forward_lock = _MLX_FORWARD_LOCK
        self._last_used: dict[str, float] = {}
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self._watchdog_lock = threading.Lock()
        # Fix #4: sampler cache — same (temp, top_p) reuses the same sampler object
        self._sampler_cache: dict[tuple[float, float], Any] = {}
        # Embedder model cache — separate from chat `_loaded` to avoid eviction
        # interference (embedder is small ~400MB; chat models LRU at _MAX_SMALL_LOADED).
        # `_last_used` tracks idle time for the same watchdog as chat models.
        self._loaded_embed: dict[str, tuple[Any, Any]] = {}
        self._start_watchdog()

    def _start_watchdog(self) -> None:
        """Daemon thread: evict idle MLX models cada _idle_ttl/4 segundos.

        Idempotente: chequea `_watchdog_thread` para no spawnear múltiples
        threads. Daemon=True → muere al exit del proceso. Errores en el
        loop se swallowean — el watchdog NUNCA debe matar el proceso si
        la eviction falla.

        Disable: `RAG_MLX_IDLE_TTL=0` o `RAG_MLX_IDLE_DISABLE=1`.
        """
        if os.environ.get("RAG_MLX_IDLE_DISABLE", "").strip() in ("1", "true", "yes"):
            return
        if self._idle_ttl <= 0:
            return
        with self._watchdog_lock:
            if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
                return
            # Check 4× per TTL window. Min 60s para no consumir CPU.
            interval = max(60, self._idle_ttl // 4)

            def _loop() -> None:
                while not self._watchdog_stop.wait(interval):
                    try:
                        self._evict_idle()
                    except Exception:
                        # Watchdog NUNCA raises. Silent-fail mantiene proceso vivo.
                        pass

            self._watchdog_thread = threading.Thread(
                target=_loop, name="mlx-idle-watchdog", daemon=True,
            )
            self._watchdog_thread.start()

    def _evict_idle(self) -> None:
        """Evict resident models cuyo last_used > _idle_ttl segundos ago.

        Llamado por el watchdog daemon. Mantiene `_loaded`, `_loaded_embed`
        y `_last_used` en sync. No-op si TTL <= 0.

        Bug Hunt 2026-05-08 H1: el path previo sólo hacía `_loaded.pop()`
        sin `mx.clear_cache()`. En Apple Silicon, Metal mantiene las
        wired pages aunque el OrderedDict drop-eé las refs Python — el
        VRAM nunca se libera bajo idle, sólo bajo memory pressure
        watchdog. Fix: replicar el pattern de `unload()` (clear cache +
        gc) fuera del lock cuando se evictó algo. También cleanup de
        `_last_used` orfanas (Bug Hunt M2).
        """
        if self._idle_ttl <= 0:
            return
        now = time.monotonic()
        with self._loaded_lock:
            stale = [
                mid for mid, ts in self._last_used.items()
                if (now - ts) > self._idle_ttl
                and (mid in self._loaded or mid in self._loaded_embed)
            ]
            for mid in stale:
                self._loaded.pop(mid, None)
                self._loaded_embed.pop(mid, None)
                self._last_used.pop(mid, None)
            # Cleanup keys orfanas en _last_used (Bug Hunt M2): si un
            # `unload(model)` previo pop-eó de los dicts pero falló al
            # limpiar `_last_used`, se queda atorada.
            orphan = [
                mid for mid in list(self._last_used)
                if mid not in self._loaded and mid not in self._loaded_embed
            ]
            for mid in orphan:
                self._last_used.pop(mid, None)
        if stale:
            # Liberar VRAM Metal — fuera del lock para no bloquear
            # otros load/chat. mx.clear_cache + gc.collect es safe
            # cuando no hay refs vivas a buffers MLX (las refs se
            # dropearon arriba al pop).
            try:
                import gc as _gc
                import mlx.core as _mx  # type: ignore[import-not-found]
                _mx.clear_cache()
                _gc.collect()
            except Exception:
                pass

    def _load(self, model_id: str) -> tuple[Any, Any]:
        """Get or load (model, tokenizer) for `model_id`. LRU bump on hit.

        Concurrency contract (P1 fix 2026-05-05):

        1. The fast-path lookup + LRU bump runs under `_loaded_lock` — atomic.
        2. Eviction (`_evict_for`) runs under the same lock (re-entrant via RLock).
        3. The heavy `mlx_lm.load(canonical)` call (3-8s cold load) is OUT of
           the lock so concurrent `chat()` calls on already-resident models
           do NOT block behind it.
        4. After loading, a double-check inside the lock ensures that if a
           second thread loaded the same model during the unlocked window,
           we discard the duplicate and return the version stored first.
           This caps duplicated `mlx_lm.load()` calls to at most N for N
           racing threads on the same fresh model — but only ONE result
           wins the `_loaded` slot, no leak.
        """
        canonical = to_mlx(model_id)

        # Fast path: already resident → LRU bump + return under lock.
        with self._loaded_lock:
            if canonical in self._loaded:
                self._loaded.move_to_end(canonical)
                self._last_used[canonical] = time.monotonic()
                return self._loaded[canonical]
            # Eviction BEFORE load (free RAM first); reentrant via RLock.
            self._evict_for(canonical)

        # Heavy I/O OUTSIDE the lock so concurrent chats on resident models
        # don't block behind a 3-8s cold load.
        from mlx_lm import load  # type: ignore[import-not-found]

        model, tokenizer = load(canonical)

        # Double-check under lock: another thread may have loaded the same
        # model while we were waiting on `mlx_lm.load`. If so, return the
        # winner and let `(model, tokenizer)` we just loaded get GC'd.
        with self._loaded_lock:
            if canonical in self._loaded:
                self._loaded.move_to_end(canonical)
                self._last_used[canonical] = time.monotonic()
                return self._loaded[canonical]
            self._loaded[canonical] = (model, tokenizer)
            self._last_used[canonical] = time.monotonic()
            return (model, tokenizer)

    def _evict_for(self, incoming: str) -> None:
        """Evict resident models to make room for `incoming`.

        Holds `_loaded_lock` (RLock) — safe to call while already holding it
        from `_load()`. Direct external callers are also protected.
        """
        with self._loaded_lock:
            is_big = incoming in self._BIG_MODELS

            if is_big:
                # Big models are single-tenant: evict everything else,
                # including the embedder — it stays in VRAM otherwise and
                # inflates usage when Qwen3-30B (~17 GB) loads.
                for k in list(self._loaded.keys()):
                    if k != incoming:
                        self._loaded.pop(k, None)
                        self._last_used.pop(k, None)
                for k in list(self._loaded_embed.keys()):
                    self._loaded_embed.pop(k, None)
                    self._last_used.pop(k, None)
                return

            # Incoming is small: evict any big, then LRU-trim small ones.
            for k in list(self._loaded.keys()):
                if k in self._BIG_MODELS:
                    self._loaded.pop(k, None)
                    self._last_used.pop(k, None)

            while len(self._loaded) >= self._MAX_SMALL_LOADED:
                k, _ = self._loaded.popitem(last=False)  # evict LRU (oldest)
                self._last_used.pop(k, None)

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: ChatOptions | None = None,
        keep_alive: str | int = -1,
        format: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        opts = options or ChatOptions()
        mlx_model, tokenizer = self._load(model)
        prompt = self._apply_chat_template(
            tokenizer, messages, format=format, tools=tools,
        )
        with self._forward_lock:
            text = self._mlx_generate(mlx_model, tokenizer, prompt, opts)
        self._bump_last_used(model)

        tool_calls = None
        content = text
        if tools:
            from rag.mlx_tool_calls import parse_tool_calls, strip_tool_call_blocks

            tool_calls = parse_tool_calls(text)
            if tool_calls:
                content = strip_tool_call_blocks(text)

        if format == "json" and not tool_calls:
            content = self._extract_json(content if content is not None else text)

        msg_kwargs: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg_kwargs["tool_calls"] = tool_calls
        return ChatResponse(
            model=to_short_name(model),
            message=Message(**msg_kwargs),
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
        # HTTP shim path (2026-05-08): cuando RAG_MLX_HTTP_SHIM=1 y el
        # modelo está mapeado a un puerto en RAG_MLX_HTTP_SHIM_PORTS,
        # routeamos via OpenAI API local. Razón: in-process MLX bajo
        # carga sostenida (chat /api/chat con embed + rerank + generate
        # en serie) trippea el watchdog de Metal (kIOGPUCommandBufferCallback
        # ErrorHang) y crashea el rag-web. Cada `mlx_lm.server` proceso
        # tiene su propio Metal context, sin contention con el embedder
        # in-process del web.
        _shim_yield = self._maybe_chat_stream_http_shim(model, messages, options, format)
        if _shim_yield is not None:
            yield from _shim_yield
            return

        from mlx_lm import stream_generate  # type: ignore[import-not-found]
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]

        opts = options or ChatOptions()
        mlx_model, tokenizer = self._load(model)
        prompt = self._apply_chat_template(tokenizer, messages, format=format)

        sampler = make_sampler(temp=opts.temperature, top_p=opts.top_p)
        if opts.temperature > 0:
            import mlx.core as mx  # type: ignore[import-not-found]

            mx.random.seed(opts.seed)

        short_name = to_short_name(model)
        # acquire/release explícito (no `with`) para garantizar que el lock se
        # libera incluso si el caller corta el iterador antes del done=True
        # (GeneratorExit) — el `finally` corre en ambos paths.
        self._forward_lock.acquire()
        try:
            for response in stream_generate(
                mlx_model,
                tokenizer,
                prompt,
                max_tokens=opts.num_predict,
                sampler=sampler,
                prefill_step_size=128,
            ):
                yield ChatResponse(
                    model=short_name,
                    message=Message(role="assistant", content=response.text),
                    done=False,
                )

            self._bump_last_used(model)
            yield ChatResponse(
                model=short_name,
                message=Message(role="assistant", content=""),
                done=True,
                done_reason="stop",
            )
        finally:
            self._forward_lock.release()

    def generate(
        self,
        model: str,
        prompt: str,
        options: ChatOptions | None = None,
        keep_alive: str | int = -1,
        **kwargs: Any,
    ) -> Any:
        opts = options or ChatOptions()
        mlx_model, tokenizer = self._load(model)
        with self._forward_lock:
            text = self._mlx_generate(mlx_model, tokenizer, prompt, opts)
        return GenerateResponse(
            model=to_short_name(model),
            response=text,
            done=True,
            done_reason="stop",
        )

    @staticmethod
    def _shim_port_for(model: str) -> int | None:
        """Return mlx_lm.server port for a model name, or None if not mapped.

        Mapping read from `RAG_MLX_HTTP_SHIM_PORTS` env var (default:
        `qwen2.5:7b=8082,qwen2.5:3b=8083`). Match accepts both short name
        and HF canonical (mlx-community/...). Override per deploy.
        """
        if os.environ.get("RAG_MLX_HTTP_SHIM", "").strip() not in ("1", "true", "yes"):
            return None
        raw = os.environ.get(
            "RAG_MLX_HTTP_SHIM_PORTS",
            "qwen2.5:7b=8082,qwen2.5:3b=8083",
        )
        mapping: dict[str, int] = {}
        for entry in raw.split(","):
            entry = entry.strip()
            if "=" not in entry:
                continue
            name, port = entry.split("=", 1)
            try:
                mapping[name.strip().lower()] = int(port.strip())
            except ValueError:
                continue
        short = to_short_name(model).lower()
        canonical = to_mlx(model).lower()
        return mapping.get(short) or mapping.get(canonical)

    def _maybe_chat_stream_http_shim(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: ChatOptions | None,
        format: str | None,
    ):
        """If model has a shim port, yield streamed ChatResponse via HTTP.

        Returns None when shim disabled or model not mapped — caller falls
        through to in-process MLX path.
        """
        port = self._shim_port_for(model)
        if port is None:
            return None
        try:
            import urllib.request
            import json as _json
        except ImportError:
            return None

        opts = options or ChatOptions()
        # Inject JSON-mode nudge into system message (mirror in-process behavior).
        msgs = list(messages)
        if format == "json":
            extra = (
                "Respond with ONLY a single valid JSON object. "
                "No prose, no markdown fences, no commentary."
            )
            if msgs and msgs[0].get("role") == "system":
                msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\n" + extra}
            else:
                msgs = [{"role": "system", "content": extra}, *msgs]

        canonical = to_mlx(model)
        body = {
            "model": canonical,
            "messages": msgs,
            "stream": True,
            "max_tokens": opts.num_predict,
            "temperature": opts.temperature,
            "top_p": opts.top_p,
        }
        if format == "json":
            body["response_format"] = {"type": "json_object"}
        if opts.temperature > 0:
            body["seed"] = opts.seed

        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        short_name = to_short_name(model)

        def _gen():
            req = urllib.request.Request(
                url,
                data=_json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = _json.loads(payload)
                    except Exception:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    if not delta:
                        continue
                    yield ChatResponse(
                        model=short_name,
                        message=Message(role="assistant", content=delta),
                        done=False,
                    )
            self._bump_last_used(model)
            yield ChatResponse(
                model=short_name,
                message=Message(role="assistant", content=""),
                done=True,
                done_reason="stop",
            )

        return _gen()

    @staticmethod
    def _apply_chat_template(
        tokenizer: Any,
        messages: list[dict[str, str]],
        format: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Apply tokenizer's chat template; nudge JSON mode in system msg.

        When `tools` is non-empty, propagated to `apply_chat_template(tools=...)`
        — Qwen2.5/3 chat templates render the schema and instruct the model to
        emit `<tool_call>{...}</tool_call>` blocks. If the tokenizer's template
        rejects the kwarg (older HF tokenizer or stripped chat_template), fall
        back to the no-tools render — caller will see no `tool_calls` in the
        response and can route accordingly.
        """
        msgs = list(messages)
        if format == "json":
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

        if tools:
            try:
                return tokenizer.apply_chat_template(
                    msgs,
                    add_generation_prompt=True,
                    tokenize=False,
                    tools=tools,
                )
            except (TypeError, ValueError):
                # Tokenizer template doesn't accept `tools=` — render without
                # the schema. The caller's `parse_tool_calls()` will return
                # None and the response degrades gracefully to plain text.
                pass

        return tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )

    def _mlx_generate(self, model: Any, tokenizer: Any, prompt: str, opts: ChatOptions) -> str:
        """Run MLX generate with a cached sampler (Fix #4)."""
        from mlx_lm import generate  # type: ignore[import-not-found]

        sampler = self._get_sampler(opts.temperature, opts.top_p)
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
            prefill_step_size=128,
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Best-effort: strip think blocks + fences + isolate first {...} block.

        Qwen3-30B (HQ tier) can emit <think>...</think> blocks before the JSON
        -- strip those first since they may contain brace characters that confuse
        the naive {{ ... }} slicer. Then strips markdown fences.
        The downstream parser in `rag/__init__.py` handles malformed JSON via
        repair; this just gives it a cleaner starting point.
        """
        s = strip_think_blocks(text.strip())
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
        return list_cached_mlx_models()

    def unload(self, model: str | None = None) -> bool:
        """Pop `model` (or all) from `_loaded` + clear MLX Metal cache.

        Apple Silicon Metal holds wired pages for resident MLX models. The
        memory watchdog calls this under pressure to release VRAM. After
        eviction, `mx.clear_cache()` releases the Metal allocator's
        reserved pool — without this, popping the OrderedDict only drops
        the Python references; Metal keeps the pages wired until
        gc.collect() runs which is non-deterministic.

        Concurrency: the `_loaded` mutation runs under `_loaded_lock`
        (P1 fix 2026-05-05). `mx.clear_cache()` and `gc.collect()` run
        OUTSIDE the lock — they can be slow and don't touch the dict.

        Returns True iff anything was unloaded.
        """
        try:
            with self._loaded_lock:
                if not self._loaded and not self._loaded_embed:
                    return False
                if model is None:
                    self._loaded.clear()
                    self._loaded_embed.clear()
                    self._last_used.clear()
                    unloaded_any = True
                else:
                    canonical = to_mlx(model)
                    unloaded_any = False
                    if canonical in self._loaded:
                        self._loaded.pop(canonical, None)
                        self._last_used.pop(canonical, None)
                        unloaded_any = True
                    if canonical in self._loaded_embed:
                        self._loaded_embed.pop(canonical, None)
                        self._last_used.pop(canonical, None)
                        unloaded_any = True
            # Cache-clear + GC outside the lock: slow ops, no `_loaded` access.
            try:
                import mlx.core as mx  # type: ignore[import-not-found]

                mx.clear_cache()
            except Exception:
                pass
            try:
                import gc

                gc.collect()
            except Exception:
                pass
            return unloaded_any
        except Exception:
            return False

    def embed(
        self,
        model: str,
        inputs: list[str],
        keep_alive: str | int = -1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Batch embedding via MLX Qwen3-Embedding model.

        Returns MLX-shape {"embeddings": [[float, ...], ...]} where each
        inner list is a L2-normalised 1024-dim vector.

        Model is cached in `_loaded_embed` (separate from chat `_loaded` dict)
        and tracked via `_last_used` for watchdog eviction. Uses last-real-token
        pooling with attention-mask-derived lengths to handle padded batches
        correctly — naive last-token pooling on a padded sequence would read the
        pad token embedding, not the real final token.
        """
        import mlx.core as mx  # type: ignore[import-not-found]

        canonical = to_mlx(model)

        # Load (or retrieve cached) embedder model + tokenizer.
        # Double-checked under lock: concurrent requests can both pass the
        # `not in` check and trigger two mlx_lm.load() calls without the
        # lock, wasting ~5-8s and potentially leaving a dangling reference.
        with self._loaded_lock:
            if canonical not in self._loaded_embed:
                from mlx_lm import load  # type: ignore[import-not-found]

                embed_model, embed_tokenizer = load(canonical)
                self._loaded_embed[canonical] = (embed_model, embed_tokenizer)
            embed_model, embed_tokenizer = self._loaded_embed[canonical]
            self._last_used[canonical] = time.monotonic()

        # mlx-lm wraps HF tokenizers in `TokenizerWrapper` (not directly
        # callable). Reach for the inner `_tokenizer` (a PreTrainedTokenizer)
        # which supports the standard batch __call__ interface with padding /
        # truncation / return_tensors.
        inner_tokenizer = getattr(embed_tokenizer, "_tokenizer", embed_tokenizer)

        # Tokenize the batch. Use padding so all sequences share a length;
        # `return_tensors="np"` gives numpy arrays that mlx.array accepts.
        # `padding=True` pads shorter sequences to the longest in the batch;
        # `truncation=True` caps at model max_length (32768 for qwen3-emb).
        encoded = inner_tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        input_ids = mx.array(encoded["input_ids"])           # (batch, seq)
        attention_mask = mx.array(encoded["attention_mask"])  # (batch, seq)

        batch_size = input_ids.shape[0]

        # Forward pass through the embedding body (bypass lm_head). Bajo el
        # lock global para no colisionar con un chat MLX concurrente.
        with self._forward_lock:
            hidden = embed_model.model(input_ids)  # (batch, seq, hidden_dim)

        # Last-real-token pooling. For each row find the index of the last
        # non-pad token. `attention_mask` is 1 for real tokens, 0 for pads.
        # sum(axis=1) gives the count of real tokens; subtract 1 for 0-based index.
        seq_lengths = mx.sum(attention_mask, axis=1) - 1  # shape (batch,)

        # mx.eval needed so seq_lengths is concrete before the index gather
        mx.eval(seq_lengths)
        seq_lengths_np = seq_lengths.tolist()

        # Gather pooled[b] = hidden[b, seq_lengths[b], :]
        pooled_rows: list[Any] = []
        for b in range(batch_size):
            pooled_rows.append(hidden[b, int(seq_lengths_np[b]), :])
        # Stack into (batch, hidden_dim)
        pooled = mx.stack(pooled_rows, axis=0)

        # L2-normalise each row
        norms = mx.sqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True))
        normalised = pooled / norms

        # Validate output dimension (1024 for qwen3-embedding:0.6b)
        hidden_dim = normalised.shape[-1]
        if hidden_dim != 1024:
            raise RuntimeError(
                f"MLXBackend.embed: expected 1024-dim output, got {hidden_dim}. "
                f"Model: {canonical}"
            )

        mx.eval(normalised)
        result = normalised.tolist()
        # Liberar tensores MLX (hidden B×T×D, pooled B×D, normalised B×D)
        # antes de retornar. Sin clear_cache los buffers Metal permanecen
        # hasta presión externa → fragmentation acumulada en embed calls
        # frecuentes (queries, tune, etc.).
        mx.clear_cache()
        return {"embeddings": result}

    def _get_sampler(self, temperature: float, top_p: float) -> Any:
        """Return a cached make_sampler() result for (temperature, top_p).

        The HELPER path always calls with temp=0, top_p=1.0 — the same
        sampler object is reused across requests (Fix #4). No explicit lock:
        CPython dict ops are GIL-atomic; a first-call race creates at most
        one extra identical sampler object (benign).
        """
        key = (float(temperature), float(top_p))
        try:
            # Fast path: already cached
            cached = self._sampler_cache.get(key)
            if cached is not None:
                return cached
        except AttributeError:
            # First call before _sampler_cache is set (shouldn't happen,
            # but defensively fall through to create)
            pass
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]

        sampler = make_sampler(temp=temperature, top_p=top_p)
        try:
            self._sampler_cache[key] = sampler
        except AttributeError:
            pass
        return sampler

    def _bump_last_used(self, model: str) -> None:
        """Update `_last_used` timestamp for `model` post-generation.

        Called at the end of `chat()` and `chat_stream()` so the idle-unload
        TTL restarts from the *end* of inference, not the start.
        """
        canonical = to_mlx(model)
        with self._loaded_lock:
            if canonical in self._loaded:
                self._last_used[canonical] = time.monotonic()

    def shutdown_watchdog(self) -> None:
        """Signal the watchdog thread to stop. Call from reset_backend() in tests."""
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=2)
            self._watchdog_thread = None


# ---------------------------------------------------------------------------
# Backend resolver
# ---------------------------------------------------------------------------


_BACKEND_SINGLETON: LLMBackend | None = None
_BACKEND_SINGLETON_LOCK = threading.Lock()


def get_backend() -> LLMBackend:
    """Return the active LLM backend (singleton). Único disponible: MLX."""
    global _BACKEND_SINGLETON
    if _BACKEND_SINGLETON is not None:
        return _BACKEND_SINGLETON
    with _BACKEND_SINGLETON_LOCK:
        if _BACKEND_SINGLETON is None:
            _BACKEND_SINGLETON = MLXBackend()
    return _BACKEND_SINGLETON


def reset_backend() -> None:
    """Force re-resolution on next get_backend() call. Tests only."""
    global _BACKEND_SINGLETON
    if isinstance(_BACKEND_SINGLETON, MLXBackend):
        _BACKEND_SINGLETON.shutdown_watchdog()
    _BACKEND_SINGLETON = None


__all__ = [
    "ChatOptions",
    "ChatResponse",
    "GenerateResponse",
    "LLMBackend",
    "Message",
    "MLXBackend",
    "MLX_MODEL_ALIAS",
    "get_backend",
    "list_cached_mlx_models",
    "reset_backend",
    "strip_think_blocks",
    "to_mlx",
    "to_short_name",
]
