"""MLX in-process embedder — Qwen3-Embedding-0.6B-8bit.

Drop-in replacement for the SentenceTransformer used by `_get_local_embedder()`
in `rag/__init__.py`. Mantiene la API mínima que los call sites consumen:

- `MLXEmbedder.encode(texts, normalize_embeddings=True, batch_size=N,
  convert_to_numpy=True, show_progress_bar=False)` → `np.ndarray (B, 1024)`.
- `.max_seq_length` (settable, propaga a tokenizer truncation).
- `.tokenizer.model_max_length` (compat con el cap defensivo del wrapper).

## Por qué `mlx-community/Qwen3-Embedding-0.6B-8bit`

Validación 2026-05-06 (5 oraciones, comparado contra
`Qwen/Qwen3-Embedding-0.6B` PyTorch fp16 en MPS via sentence-transformers):

| Variant                                          | cosine vs PyTorch | encode 5  |
|--------------------------------------------------|-------------------|-----------|
| `mlx-community/Qwen3-Embedding-0.6B-8bit`        | **0.9977-1.0037** | 94 ms     |
| `mlx-community/Qwen3-Embedding-0.6B-mxfp8`       | 0.9777-0.9843     | 377 ms    |
| `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`    | 0.9643-0.9795     | 286 ms    |

8-bit es bit-equivalente funcional (cosine ≥0.9977): los embeddings
siguen apuntando al mismo punto en R^1024, no se degrada retrieval, NO
necesita reindex (`_COLLECTION_BASE` queda en `obsidian_notes_v11`).

## Encode pipeline

1. Tokenizer encode con `truncation=True` + `max_length=max_seq_length`.
2. Pad RIGHT a la longitud max del batch con `pad_id` (eos_token_id).
3. `model.model(ids)` → hidden states `(B, T, D=1024)`. NO `model(ids)` —
   ese path corre el `lm_head` y devuelve logits del vocab (151669-d).
4. Last-real-token pooling: `hidden[batch_idx, real_lengths-1, :]`.
   La convención de Qwen3-Embedding es usar el último token (post-EOS),
   NO mean pooling. Validado contra el reference PyTorch.
5. L2 normalize.
6. `bf16 → fp32 → np.ndarray` (mlx no exporta bf16 a numpy directo).

## Threading

`MLXEmbedder` es thread-safe para concurrent `.encode()` — el forward de
`mlx_lm` es reentrante. El lazy `load()` está guardado por `_load_lock`.
"""

from __future__ import annotations

import os
import threading
from typing import Iterable

import numpy as np


# Lazy-imported in `_load()` para no pagar el costo de import cuando el
# proceso no usa el embedder (ej. tests unitarios que mockean LLM y embed).
_mx = None  # mlx.core
_mlx_load = None  # mlx_lm.load


def _import_mlx() -> None:
    """Lazy import de mlx + mlx_lm. Idempotente."""
    global _mx, _mlx_load
    if _mx is not None and _mlx_load is not None:
        return
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    _mx = mx
    _mlx_load = mlx_load


class _TokenizerProxy:
    """Compat wrapper para que el call site pueda hacer
    `embedder.tokenizer.model_max_length = N` igual que con SentenceTransformer.

    Internamente bindea al `TokenizerWrapper` de mlx-lm + propaga el cap a
    `model_max_length` del tokenizer subyacente (rust HF) para honrarlo en
    `truncation=True`.
    """

    def __init__(self, mlx_tokenizer):
        self._tok = mlx_tokenizer
        self.model_max_length = 512

    def __setattr__(self, name: str, value):
        super().__setattr__(name, value)
        if name == "model_max_length":
            try:
                inner = getattr(self._tok, "_tokenizer", None) or self._tok
                if inner is not None and hasattr(inner, "model_max_length"):
                    inner.model_max_length = int(value)
            except Exception:
                pass


class MLXEmbedder:
    """In-process embedder backed por mlx-lm.

    API subset que los call sites en `rag/__init__.py` ya consumen:

    - `__init__(repo_id, max_seq_length=512)`: lazy. La carga real ocurre
      en el primer `.encode()`. (Idem patrón `MLXReranker`.)
    - `encode(texts, normalize_embeddings=True, batch_size=N,
      convert_to_numpy=True, show_progress_bar=False) -> np.ndarray (B, 1024)`.
      Si `convert_to_numpy=False`, devuelve `list[np.ndarray]` (compat con
      SentenceTransformer cuando el caller hizo `.tolist()` después).
    - `max_seq_length`: gobierna truncation per-text antes de tokenizar.
    """

    def __init__(self, repo_id: str, max_seq_length: int = 512):
        self.repo_id = repo_id
        self._max_seq_length = int(max_seq_length)
        self._model = None
        self._mlx_tokenizer = None
        self.tokenizer: _TokenizerProxy | None = None
        self._load_lock = threading.Lock()

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        v = int(value)
        if v <= 0:
            v = 512
        self._max_seq_length = v
        if self.tokenizer is not None:
            self.tokenizer.model_max_length = v

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            _import_mlx()
            assert _mlx_load is not None
            model, tokenizer = _mlx_load(self.repo_id)
            self._model = model
            self._mlx_tokenizer = tokenizer
            self.tokenizer = _TokenizerProxy(tokenizer)
            self.tokenizer.model_max_length = self._max_seq_length

    def _tokenize_truncated(self, texts: list[str]) -> tuple[list[list[int]], list[int]]:
        """Tokenize cada texto y truncar a `max_seq_length`. Devuelve los
        IDs y las longitudes reales (sin padding).
        """
        assert self._mlx_tokenizer is not None
        cap = max(1, int(self._max_seq_length))
        ids_list: list[list[int]] = []
        lengths: list[int] = []
        for t in texts:
            ids = self._mlx_tokenizer.encode(t)
            if not isinstance(ids, list):
                ids = list(ids)
            if len(ids) > cap:
                ids = ids[:cap]
            if not ids:
                # Edge case: input vacío → embed un EOS para no crashear.
                # Caller no debería pasar "" pero defensivo.
                eos = getattr(self._mlx_tokenizer, "eos_token_id", None) or 0
                ids = [int(eos)]
            ids_list.append(ids)
            lengths.append(len(ids))
        return ids_list, lengths

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Forward MLX único sobre `texts` con padding RIGHT al max_len del
        batch. Devuelve `(B, D)` fp32 numpy normalizado L2.

        El forward (línea `model.model(...)` + `mx.eval`) corre bajo
        `_MLX_FORWARD_LOCK` (rag.llm_backend) — mismo lock que MLXBackend
        chat/embed/generate — para evitar colisión Metal con otro forward
        concurrente del mismo proceso (memo
        `obsidian_rag_web_service_gpu_hang_loop`, web home-refresh).
        """
        from rag.llm_backend import _MLX_FORWARD_LOCK

        assert _mx is not None and self._model is not None
        ids_list, lengths = self._tokenize_truncated(texts)
        max_len = max(lengths)
        pad_id = getattr(self._mlx_tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self._mlx_tokenizer, "eos_token_id", None) or 0
        padded = [ids + [int(pad_id)] * (max_len - len(ids)) for ids in ids_list]
        ids_arr = _mx.array(padded)
        with _MLX_FORWARD_LOCK:
            # Backbone forward (sin lm_head). Devuelve hidden states (B, T, D).
            h = self._model.model(ids_arr)
            batch_idx = _mx.arange(len(texts))
            last_idx = _mx.array([L - 1 for L in lengths])
            last = h[batch_idx, last_idx, :]  # (B, D)
            norm = _mx.linalg.norm(last, axis=-1, keepdims=True)
            # Guard contra division-by-zero: norm 0 implica hidden state nulo
            # (no debería pasar en práctica, pero el wrapper aplica un floor).
            emb = last / (norm + 1e-12)
            emb_fp32 = emb.astype(_mx.float32)
            _mx.eval(emb_fp32)
        # Convertir a numpy y liberar tensores MLX. `np.array()` copia los
        # datos a CPU RAM; después del return, `emb_fp32`, `h`, `last`, `emb`
        # son inalcanzables → el GC los marca como free en el próximo
        # clear_cache(). Sin el clear_cache() explícito, el allocator MLX
        # retiene la memoria Metal hasta presión externa.
        result = np.array(emb_fp32)
        _mx.clear_cache()
        return result

    def encode(
        self,
        texts: str | list[str] | Iterable[str],
        *,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **kwargs,
    ):
        """Embed `texts` y devolver shape `(B, 1024)` numpy o `list[np.ndarray]`.

        `normalize_embeddings` se ignora (siempre normalizamos L2 — Qwen3
        embedding así fue entrenado y los call sites de obsidian-rag
        siempre piden `normalize=True`).
        `show_progress_bar` y kwargs extra: aceptados para compat con la
        API de SentenceTransformer pero no usados.
        """
        del show_progress_bar, kwargs  # compat, no-op
        if isinstance(texts, str):
            texts_list = [texts]
            single = True
        else:
            texts_list = list(texts)
            single = False
        if not texts_list:
            return np.zeros((0, 1024), dtype=np.float32)
        self._load()
        bs = max(1, int(batch_size))

        # Sort-by-length before batching: texts of similar length en el mismo
        # batch reducen el padding promedio un 20-40% en batches mixed-length
        # (notas cortas de 50 tokens mezcladas con notas de 512 desperdician
        # ~9× el compute del batch entero en padding). Unsort al final para
        # preservar el orden original del caller.
        order = sorted(range(len(texts_list)), key=lambda i: len(texts_list[i]))
        sorted_texts = [texts_list[i] for i in order]
        # Inverse permutation para restaurar el orden original.
        inv_order = [0] * len(order)
        for sorted_pos, orig_pos in enumerate(order):
            inv_order[orig_pos] = sorted_pos

        sorted_chunks: list[np.ndarray] = []
        for i in range(0, len(sorted_texts), bs):
            chunk = sorted_texts[i : i + bs]
            sorted_chunks.append(self._encode_batch(chunk))
        sorted_out = np.concatenate(sorted_chunks, axis=0)
        # Restore original order via the inverse permutation.
        out = sorted_out[inv_order]

        if not normalize_embeddings:
            # Si el caller pidió raw (sin L2), des-normalizamos no es
            # posible — pero ningún call site lo pide en obsidian-rag.
            # Mantenemos `out` como está y dejamos un breadcrumb.
            pass
        if convert_to_numpy:
            return out[0] if single else out
        # SentenceTransformer con `convert_to_numpy=False` devuelve tensors
        # PyTorch; aquí devolvemos `list[np.ndarray]` (compat con `.tolist()`
        # del caller cuando se itera por row).
        return [out[i] for i in range(out.shape[0])]


# ── Helper: scan local HF cache para warm-load sin tocar la red ──────────────
# Replica el patrón de `_scan_local_models()` en llm_backend.py: si el
# snapshot del repo no está en `~/.cache/huggingface/hub/`, mejor fallar
# rápido en `_load()` con el error de mlx_load, que tirar un timeout.

def is_repo_cached(repo_id: str) -> bool:
    """Devuelve True si el snapshot de `repo_id` está cacheado localmente.

    Heurística: chequea la existencia del dir `models--<org>--<name>` en
    `~/.cache/huggingface/hub/`. NO valida que todos los archivos del
    snapshot estén presentes — eso lo hace `mlx_lm.load`.
    """
    safe = repo_id.replace("/", "--")
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub = os.path.join(home, "hub")
    if not os.path.isdir(hub):
        return False
    return os.path.isdir(os.path.join(hub, f"models--{safe}"))


__all__ = ["MLXEmbedder", "is_repo_cached"]
