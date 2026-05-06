"""Tests for rag/llm_backend.py — LLMBackend abstraction.

Post-Ola 7 (2026-05-06): `OllamaBackend` retirado. Solo MLX vivo.
`test_get_backend_mlx_when_env_set` uses pytest.importorskip because
it exercises MLXBackend.__init__ which does `import mlx_lm` eagerly.
"""

from __future__ import annotations

import pytest

from rag.llm_backend import (
    MLX_MODEL_ALIAS,
    OLLAMA_MODEL_ALIAS,
    ChatOptions,
    MLXBackend,
    get_backend,
    reset_backend,
    to_mlx,
    to_ollama,
)

# ---------------------------------------------------------------------------
# Autouse: reset singleton before every test so tests don't pollute each other
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_backend()
    yield
    reset_backend()


# ---------------------------------------------------------------------------
# 1. to_mlx alias resolution
# ---------------------------------------------------------------------------


def test_to_mlx_resolves_aliases():
    assert to_mlx("qwen2.5:3b") == "mlx-community/Qwen2.5-3B-Instruct-4bit"
    assert to_mlx("command-r") == "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
    # Already canonical — identity
    assert to_mlx("mlx-community/Foo") == "mlx-community/Foo"
    # Unknown Ollama name — identity passthrough
    assert to_mlx("unknown:7b") == "unknown:7b"


# ---------------------------------------------------------------------------
# 2. to_ollama inverse mapping
# ---------------------------------------------------------------------------


def test_to_ollama_inverse():
    # MLX canonical → Ollama short name
    assert to_ollama("mlx-community/Qwen2.5-3B-Instruct-4bit") == "qwen2.5:3b"
    assert to_ollama("mlx-community/Qwen2.5-7B-Instruct-4bit") == "qwen2.5:7b"
    # Unknown MLX id — identity passthrough
    assert to_ollama("mlx-community/SomethingElse") == "mlx-community/SomethingElse"
    # Round-trip: to_mlx → to_ollama should give back the original for known names
    original = "qwen2.5:3b"
    assert to_ollama(to_mlx(original)) == original


# ---------------------------------------------------------------------------
# 3. get_backend default → MLXBackend (post-cutover 2026-05-05)
# ---------------------------------------------------------------------------


def test_get_backend_default_mlx(monkeypatch):
    """Post-cutover 2026-05-05: el default de get_backend es MLX. La conftest
    autouse `_force_ollama_backend_for_tests` setea ollama por test, así que
    aquí explícitamente desetea para verificar el verdadero default."""
    pytest.importorskip("mlx_lm")
    monkeypatch.delenv("RAG_LLM_BACKEND", raising=False)

    from rag.llm_backend import MLXBackend

    backend = get_backend()
    assert isinstance(backend, MLXBackend)
    assert backend.name == "mlx"


# ---------------------------------------------------------------------------
# 4. get_backend mlx env → MLXBackend (skipped when mlx_lm missing)
# ---------------------------------------------------------------------------


def test_get_backend_mlx_when_env_set(monkeypatch):
    pytest.importorskip("mlx_lm")

    monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")

    backend = get_backend()
    assert isinstance(backend, MLXBackend)
    assert backend.name == "mlx"


# ---------------------------------------------------------------------------
# 5. invalid backend env → fallback a MLX con warning (post-Ola 7)
# ---------------------------------------------------------------------------


def test_get_backend_invalid_falls_back_to_mlx(monkeypatch, caplog):
    """Post-Ola 7: cualquier valor distinto de 'mlx' (incluyendo 'ollama'
    y 'invalid') loguea warning y vuelve a MLX. Antes esto raisearía
    ValueError; ahora MLX es el único backend vivo."""
    pytest.importorskip("mlx_lm")
    monkeypatch.setenv("RAG_LLM_BACKEND", "invalid")

    backend = get_backend()
    assert isinstance(backend, MLXBackend)


# ---------------------------------------------------------------------------
# 6. ChatOptions defaults
# ---------------------------------------------------------------------------


def test_chat_options_defaults():
    opts = ChatOptions()
    assert opts.temperature == 0.0
    assert opts.seed == 42
    assert opts.num_ctx == 4096
    # 384 matches CHAT_OPTIONS["num_predict"] in rag/__init__.py — was 768 (mismatch bug)
    assert opts.num_predict == 384


# ---------------------------------------------------------------------------
# 7. MLXBackend._extract_json strips markdown fences
# ---------------------------------------------------------------------------


def test_mlx_extract_json_strips_fences():
    raw = "```json\n{\"a\":1}\n```"
    result = MLXBackend._extract_json(raw)
    assert result == '{"a":1}'


# ---------------------------------------------------------------------------
# 8. MLXBackend._extract_json isolates first {...} block
# ---------------------------------------------------------------------------


def test_mlx_extract_json_isolates_block():
    raw = 'prose {"a":1} more'
    result = MLXBackend._extract_json(raw)
    assert result == '{"a":1}'


# ---------------------------------------------------------------------------
# 9. MLX_MODEL_ALIAS completeness
# ---------------------------------------------------------------------------


def test_mlx_alias_table_complete():
    required_keys = {
        "qwen2.5:3b",
        "qwen2.5:7b",
        "command-r",
        "command-r:latest",
        "qwen2.5:14b",
        "qwen3:4b",
    }
    assert required_keys <= set(MLX_MODEL_ALIAS.keys())


# ---------------------------------------------------------------------------
# 10. reset_backend clears singleton
# ---------------------------------------------------------------------------


def test_reset_backend_clears_singleton(monkeypatch):
    """Post-Ola 7: solo MLX vivo. El test verifica que reset_backend()
    fuerza re-resolver el singleton (instancias distintas pre/post reset)."""
    pytest.importorskip("mlx_lm")
    monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")

    b1 = get_backend()
    assert isinstance(b1, MLXBackend)

    reset_backend()

    b2 = get_backend()
    assert isinstance(b2, MLXBackend)
    # They're different objects because reset_backend cleared the singleton
    assert b1 is not b2
