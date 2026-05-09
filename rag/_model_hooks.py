"""Reload hooks por tier — registrados al import.

Cada hook corre cuando `rag.models.swap(tier, ...)` o `set_env(tier, ...)`
mutan el modelo activo. Responsabilidad: invalidar caches in-process +
unloadear modelos viejos del backend MLX (libera VRAM/RAM).

Se importa al final de `rag/__init__.py` cuando las constantes ya están
definidas (`HELPER_MODEL`, `EMBED_MODEL`, etc.).
"""

from __future__ import annotations

import os

from rag import models as _models


def _on_chat_swap(old: str, new: str) -> None:
    """Reset cache de `resolve_chat_model()` + unload MLX del modelo viejo."""
    import rag
    rag._CHAT_MODEL_RESOLVED = None
    try:
        from rag.llm_backend import MLXBackend, to_mlx
        if old:
            MLXBackend().unload(to_mlx(old))
    except Exception:
        pass


def _on_helper_swap(old: str, new: str) -> None:
    """Mutar `HELPER_MODEL` + `_LOOKUP_MODEL` (si no hay override env) + unload."""
    import rag
    rag.HELPER_MODEL = new
    if not os.environ.get("RAG_LOOKUP_MODEL", "").strip():
        rag._LOOKUP_MODEL = new
    try:
        from rag.llm_backend import MLXBackend, to_mlx
        if old:
            MLXBackend().unload(to_mlx(old))
    except Exception:
        pass


def _on_embed_swap(old: str, new: str) -> None:
    """Mutar `EMBED_MODEL` + clear embed cache + unload embedder MLX.

    Validación previa de `models.swap()` ya garantiza que `new` está en la
    `_EMBED_WHITELIST` (dim 1024 lockstep con `_COLLECTION_BASE`). Si pasaste
    `unsafe=True`, el corpus va a fallar al próximo retrieve por dim mismatch
    — corré `rag index --full` después.
    """
    import rag
    rag.EMBED_MODEL = new
    # Clear el embed cache (vault embeddings). Algunos paths usan diccionarios
    # módulo-globales — limpieza defensiva.
    for attr in ("_embed_cache", "_embed_cache_model"):
        if hasattr(rag, attr):
            obj = getattr(rag, attr)
            if isinstance(obj, dict):
                obj.clear()
            elif obj is not None:
                setattr(rag, attr, None)
    # Unload del embedder MLX viejo si está cargado
    try:
        from rag.llm_backend import MLXBackend, to_mlx
        if old:
            MLXBackend().unload(to_mlx(old))
    except Exception:
        pass


def _on_rerank_swap(old: str, new: str) -> None:
    """Mutar `RERANKER_MODEL` + reset reranker singleton."""
    import rag
    rag.RERANKER_MODEL = new
    # Reset PT reranker singleton (si existe)
    for attr in ("_reranker_singleton", "_reranker", "_RERANKER_OBJ"):
        if hasattr(rag, attr) and getattr(rag, attr) is not None:
            setattr(rag, attr, None)
    # Reset MLX reranker singleton
    try:
        import rag.mlx_reranker as _mr
        for attr in ("_RERANKER_SINGLETON", "_reranker_singleton"):
            if hasattr(_mr, attr) and getattr(_mr, attr) is not None:
                setattr(_mr, attr, None)
    except Exception:
        pass


def _on_stt_swap(old: str, new: str) -> None:
    """Clear cache de wrappers whisper."""
    try:
        import rag.whisper as _w
        _w._whisper_model_cache.clear()
    except Exception:
        pass


def _on_vlm_swap(old: str, new: str) -> None:
    """Re-set `RAG_VLM_MODEL` + reset VLM singleton de ocr.py."""
    os.environ["RAG_VLM_MODEL"] = new
    try:
        import rag.ocr as _ocr
        # Reset singletons del VLM cargado (granite via mlx-vlm)
        for attr in ("_VLM_MODEL_OBJ", "_VLM_PROCESSOR"):
            if hasattr(_ocr, attr):
                setattr(_ocr, attr, None)
        # Refresh la constante module-level que apunta al HF id
        if hasattr(_ocr, "VLM_MODEL"):
            _ocr.VLM_MODEL = new
    except Exception:
        pass


def install_hooks() -> None:
    """Registrar todos los reload hooks. Llamado al final de `rag/__init__.py`."""
    _models.register_reload_hook("chat", _on_chat_swap)
    _models.register_reload_hook("helper", _on_helper_swap)
    _models.register_reload_hook("embed", _on_embed_swap)
    _models.register_reload_hook("rerank", _on_rerank_swap)
    _models.register_reload_hook("stt", _on_stt_swap)
    _models.register_reload_hook("vlm", _on_vlm_swap)
