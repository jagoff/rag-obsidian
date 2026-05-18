"""LoRA fine-tune del modelo de drafts WhatsApp — extracted from `rag/__init__.py` (Wave 9 split, 2026-05-10).

Mismo patrón que el reranker LoRA: adapter PEFT vive bajo XDG data
(no cache porque la signal es no-regenerable), loader runtime hace
silent-fail si peft no está / adapter no existe / RAG_DRAFTS_FT OFF.

## Importante

El listener TS sigue usando `qwen2.5:14b` para generar drafts en
producción. El adapter fine-tuned NO sustituye eso. El único call-site
del adapter es `/api/draft/preview` (web/server.py), que el user llama
manualmente para A/B "qué hubiera salido del baseline" vs "qué hubiera
salido del fine-tuned". NO está en el hot path.

## Backends

- **`mlx`** (default cuando existe `drafts_ft_mlx/`) — `mlx_lm.load` con
  `adapter_path=DRAFTS_FT_ADAPTER_DIR_MLX`, mucho más simple que PEFT.
- **`peft`** (rollback) — transformers + peft path histórico.

Override explícito via `RAG_DRAFTS_FT_BACKEND={mlx,peft}`.

## Public API

- `generate_draft_preview(*, original_conversation, bot_draft_baseline,
  max_new_tokens=200)` → str — endpoint principal
- `_drafts_ft_enabled()` → bool — flag operacional `RAG_DRAFTS_FT`
- `_drafts_ft_adapter_available()` → bool — cheap check existencia
- `_drafts_ft_backend()` → str ("mlx" | "peft") — backend resolver
- `_load_drafts_ft_model()` → (model, tokenizer) | (None, None) — PEFT lazy load
- `_generate_draft_preview_mlx(...)` → str — MLX path

## Estado en `rag.__init__` (state-at-owner pattern)

`DRAFTS_FT_ADAPTER_DIR`, `DRAFTS_FT_ADAPTER_DIR_MLX`,
`DRAFTS_FT_BASE_MODEL`, `DRAFTS_FT_BASE_MODEL_MLX`, `_drafts_ft_model`,
`_drafts_ft_tokenizer`, `_drafts_ft_lock`, `_drafts_ft_load_failed`
viven en `rag/__init__.py` para preservar compat con tests
(`tests/test_finetune_drafts.py`) que hacen
`monkeypatch.setattr(rag, "DRAFTS_FT_ADAPTER_DIR", ...)`,
`monkeypatch.setattr(rag, "_drafts_ft_model", None)`, etc.

Las funciones acá leen/escriben via `import rag; rag.X` deferred.
"""

from __future__ import annotations

import os

__all__ = [
    "_drafts_ft_enabled",
    "_drafts_ft_adapter_available",
    "_drafts_ft_backend",
    "_load_drafts_ft_model",
    "_generate_draft_preview_mlx",
    "generate_draft_preview",
]


def _drafts_ft_enabled() -> bool:
    """True iff `RAG_DRAFTS_FT=1` (truthy). Default OFF.

    Es el flag operacional que el user setea manualmente cuando quiere
    A/B vía `/api/draft/preview`. Sin él, el endpoint hace echo del
    bot_draft_baseline sin tocar el modelo (zero-cost passthrough).
    """
    return os.environ.get("RAG_DRAFTS_FT", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _drafts_ft_adapter_available() -> bool:
    """Cheap check: existe el dir + adapter_config.json. NO valida que
    el adapter sea cargeable (eso lo hace _load_drafts_ft_model en el
    primer use). Útil para `rag drafts stats` que solo quiere saber si
    hay metadata para mostrar.
    """
    import rag  # noqa: PLC0415

    try:
        return (
            rag.DRAFTS_FT_ADAPTER_DIR.is_dir()
            and (rag.DRAFTS_FT_ADAPTER_DIR / "adapter_config.json").is_file()
        )
    except Exception:
        return False


def _load_drafts_ft_model():
    """Lazy-load del modelo fine-tuned + tokenizer. Thread-safe.

    Returns:
        (model, tokenizer) tuple, or (None, None) si:
          - peft / transformers no están instalados
          - el adapter dir no existe / falta adapter_config.json
          - la carga del PEFT model raiseó (corrupt safetensors,
            base model dist mismatch, etc.)

    Conservative: NUNCA raisea. Loguea a `silent_errors.jsonl` y
    devuelve (None, None) — el caller (preview endpoint) interpreta
    eso como "fallback a echo del baseline".

    Cuando el load falla por RuntimeError (state_dict mismatch entre
    el adapter LoRA y el base model), setea `_drafts_ft_load_failed`
    a True para evitar reintentos per-request con el mismo error.
    Reiniciar el proceso resetea el sentinel.
    """
    import rag  # noqa: PLC0415

    with rag._drafts_ft_lock:
        if rag._drafts_ft_model is not None:
            return rag._drafts_ft_model, rag._drafts_ft_tokenizer
        if rag._drafts_ft_load_failed:
            return None, None
        if not _drafts_ft_adapter_available():
            rag._silent_log(
                "drafts_ft_adapter_missing",
                FileNotFoundError(str(rag.DRAFTS_FT_ADAPTER_DIR)),
            )
            return None, None
        try:
            from peft import PeftModel  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:
            rag._silent_log("drafts_ft_peft_missing", exc)
            return None, None
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(rag.DRAFTS_FT_ADAPTER_DIR))
        except Exception:
            # Si el adapter dir no tiene tokenizer save (training viejo),
            # fall back al base model tokenizer.
            try:
                from transformers import AutoTokenizer as _AT
                tokenizer = _AT.from_pretrained(rag.DRAFTS_FT_BASE_MODEL)
            except Exception as exc:
                rag._silent_log("drafts_ft_tokenizer_load_failed", exc)
                return None, None
        try:
            base = AutoModelForCausalLM.from_pretrained(
                rag.DRAFTS_FT_BASE_MODEL, torch_dtype="auto",
            )
            model = PeftModel.from_pretrained(base, str(rag.DRAFTS_FT_ADAPTER_DIR))
            model.eval()
            rag._drafts_ft_model = model
            rag._drafts_ft_tokenizer = tokenizer
            return rag._drafts_ft_model, rag._drafts_ft_tokenizer
        except Exception as exc:
            import traceback as _tb
            detail = _tb.format_exc()
            rag._silent_log(
                "drafts_ft_load_failed",
                f"{exc!r}\n\nTraceback completo:\n{detail}\n"
                f"Adapter: {rag.DRAFTS_FT_ADAPTER_DIR}\n"
                f"Base model: {rag.DRAFTS_FT_BASE_MODEL}\n"
                "RAG_DRAFTS_FT desactivado en runtime — reiniciar proceso para reintentar.",
            )
            rag._drafts_ft_load_failed = True
            return None, None


def _drafts_ft_backend() -> str:
    """Resuelve el backend del drafts adapter.

    Default: `mlx` cuando exista el adapter MLX (Ola 10 path nativo).
    Fallback: `peft` cuando solo exista el PEFT histórico. Override
    explícito via `RAG_DRAFTS_FT_BACKEND={mlx,peft}`.
    """
    import rag  # noqa: PLC0415

    explicit = os.environ.get("RAG_DRAFTS_FT_BACKEND", "").strip().lower()
    if explicit in ("mlx", "peft"):
        return explicit
    # Auto-detect: MLX adapter dir tiene prioridad si existe
    try:
        if (rag.DRAFTS_FT_ADAPTER_DIR_MLX / "adapters.safetensors").is_file() or \
           (rag.DRAFTS_FT_ADAPTER_DIR_MLX / "adapter_config.json").is_file():
            return "mlx"
    except Exception:
        pass
    return "peft"


def _generate_draft_preview_mlx(
    *, original_conversation: str, bot_draft_baseline: str,
    max_new_tokens: int = 200,
) -> str:
    """Path MLX-native del preview generator (Fase 1.2 Ola 10).

    Usa `mlx_lm.load` con `adapter_path=DRAFTS_FT_ADAPTER_DIR_MLX` +
    `mlx_lm.generate`. Mucho más simple que el PEFT path porque
    mlx_lm carga adapter LoRA + base model en un solo call.
    """
    import rag  # noqa: PLC0415

    try:
        from mlx_lm import generate as _mlx_generate, load as _mlx_load  # type: ignore[import-not-found]
    except ImportError as exc:
        rag._silent_log("drafts_ft_mlx_import_missing", exc)
        return bot_draft_baseline

    try:
        from rag.llm_backend import (  # noqa: PLC0415
            _MLX_FORWARD_LOCK,
            clear_mlx_cache_safely,
        )
    except Exception as exc:
        rag._silent_log("drafts_ft_mlx_runtime_guard_missing", exc)
        return bot_draft_baseline

    model = None
    tokenizer = None
    try:
        clear_mlx_cache_safely(collect=True)
        # `mlx_lm.load()` also allocates large Metal buffers. Keep the one-off
        # adapter load serialized with chat/embed/rerank/VLM work, then clear
        # after the preview so repeated A/B calls do not accumulate allocator
        # pressure.
        with _MLX_FORWARD_LOCK:
            model, tokenizer = _mlx_load(
                rag.DRAFTS_FT_BASE_MODEL_MLX,
                adapter_path=str(rag.DRAFTS_FT_ADAPTER_DIR_MLX),
            )
    except Exception as exc:
        rag._silent_log("drafts_ft_mlx_load_failed", exc)
        clear_mlx_cache_safely(collect=True)
        return bot_draft_baseline
    prompt = (
        f"{original_conversation}\n\n"
        f"## Borrador del bot:\n"
        f"{bot_draft_baseline}\n\n"
        f"## Mensaje final que mandó Fer:\n"
    )
    try:
        with _MLX_FORWARD_LOCK:
            out = _mlx_generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                verbose=False,
            )
        text = str(out or "").strip()
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text
    except Exception as exc:
        rag._silent_log("drafts_ft_mlx_generate_failed", exc)
        return bot_draft_baseline
    finally:
        model = None
        tokenizer = None
        clear_mlx_cache_safely(collect=True)


def generate_draft_preview(
    *, original_conversation: str, bot_draft_baseline: str,
    max_new_tokens: int = 200,
) -> str:
    """Genera el output del modelo fine-tuned para A/B manual del user.

    Contract:
      - Si `RAG_DRAFTS_FT` está OFF → echo del baseline.
      - Backend `mlx` (default cuando existe `drafts_ft_mlx/`) usa
        `mlx_lm.load` + `mlx_lm.generate`. Backend `peft` (fallback)
        usa el path histórico transformers + peft.
      - Si el adapter no existe / dependencias faltan → echo del baseline +
        log a silent_errors.jsonl.
      - Si el modelo carga pero la generación crashea → echo del
        baseline + log.
      - Si todo OK → devuelve el output del modelo (string post-strip
        del prompt, máx max_new_tokens).

    Note: NUNCA raisea. El endpoint preview siempre devuelve algo
    útil al caller (mínimo el baseline echo).
    """
    import rag  # noqa: PLC0415

    if not _drafts_ft_enabled():
        return bot_draft_baseline

    if _drafts_ft_backend() == "mlx":
        return _generate_draft_preview_mlx(
            original_conversation=original_conversation,
            bot_draft_baseline=bot_draft_baseline,
            max_new_tokens=max_new_tokens,
        )

    # Path histórico PEFT (rollback)
    model, tokenizer = _load_drafts_ft_model()
    if model is None or tokenizer is None:
        return bot_draft_baseline
    try:
        import torch
        prompt = (
            f"{original_conversation}\n\n"
            f"## Borrador del bot:\n"
            f"{bot_draft_baseline}\n\n"
            f"## Mensaje final que mandó Fer:\n"
        )
        inp = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=1024)
        # Move inputs al device del modelo. Sin esto, en MPS / CUDA, el
        # forward pass crashea con `RuntimeError: Placeholder storage has
        # not been allocated on MPS device` porque el tokenizer retorna
        # tensores en CPU y `generate` no auto-convierte (transformers
        # 5.1.0). Bug observado el 2026-05-01 con LoRA adapter en M3 Max.
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        inp = {k: v.to(model_device) for k, v in inp.items()}
        out = model.generate(
            **inp, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        full = tokenizer.decode(out[0], skip_special_tokens=True)
        if full.startswith(prompt):
            return full[len(prompt):].strip()
        return full.strip()
    except Exception as exc:
        rag._silent_log("drafts_ft_generate_failed", exc)
        return bot_draft_baseline
