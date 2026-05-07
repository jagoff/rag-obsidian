"""Tests del helper `chat_keep_alive()` post-Ola 8 (cero-Ollama).

Histórico:
- Ola 0-7: `chat_keep_alive()` clampaba LLM_KEEP_ALIVE=-1 a "20m" cuando
  el modelo resuelto estaba en `_LARGE_CHAT_MODELS` (command-r, qwen3:30b-a3b).
  Guard contra el 2026-04-17 Mac-freeze regression con Ollama wired memory.
- Ola 8 (2026-05-06): los modelos grandes Ollama fueron purgados del disco;
  MLX backend in-process maneja eviction propio (idle-unload + LRU). Quitamos
  el clamp + el frozenset `_LARGE_CHAT_MODELS`. `chat_keep_alive()` queda
  como passthrough trivial — devuelve `LLM_KEEP_ALIVE` raw.

Estos tests validan ese contrato simple post-Ola 8.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag  # noqa: E402


def test_chat_keep_alive_returns_llm_keep_alive_default():
    """Sin model arg: devuelve `LLM_KEEP_ALIVE` configurado por env."""
    assert rag.chat_keep_alive() == rag.LLM_KEEP_ALIVE


def test_chat_keep_alive_with_model_arg_returns_llm_keep_alive():
    """Con model arg explícito: ignora el arg y devuelve `LLM_KEEP_ALIVE`.
    El parámetro queda solo por compat de firma con call sites históricos."""
    assert rag.chat_keep_alive("qwen2.5:7b") == rag.LLM_KEEP_ALIVE
    assert rag.chat_keep_alive("qwen2.5:3b") == rag.LLM_KEEP_ALIVE
    assert rag.chat_keep_alive("any-model") == rag.LLM_KEEP_ALIVE


def test_llm_keep_alive_default_is_minus_one():
    """Default actualizado a `-1` (forever) post-Ola 8 — MLX in-process,
    sin riesgo de wired-memory por daemon Ollama."""
    # Solo si el env var no está set por el shell del runner. Si está, lo
    # respetamos — esto es un default check.
    import os
    if not os.environ.get("RAG_LLM_KEEP_ALIVE") and not os.environ.get("OLLAMA_KEEP_ALIVE"):
        assert rag.LLM_KEEP_ALIVE == -1
