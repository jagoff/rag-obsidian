"""Tests for rag/llm_backend.py — LLMBackend abstraction.

Post-Ola 7 (2026-05-06): `OllamaBackend` retirado. Solo MLX vivo.
`test_get_backend_mlx_when_env_set` uses pytest.importorskip because
it exercises MLXBackend.__init__ which does `import mlx_lm` eagerly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from rag.llm_backend import (
    MLX_MODEL_ALIAS,
    ChatOptions,
    MLXBackend,
    get_backend,
    list_cached_mlx_models,
    reset_backend,
    to_mlx,
    to_short_name,
)


def _requires_mlx_lm() -> None:
    try:
        pytest.importorskip("mlx_lm")
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip("requires MLX Metal device")
        raise

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
    assert to_mlx("command-r") == "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ"
    # Already canonical — identity
    assert to_mlx("mlx-community/Foo") == "mlx-community/Foo"
    # Unknown short alias — identity passthrough
    assert to_mlx("unknown:7b") == "unknown:7b"


# ---------------------------------------------------------------------------
# 2. to_short_name inverse mapping
# ---------------------------------------------------------------------------


def test_to_short_name_inverse():
    # MLX canonical → short alias name
    assert to_short_name("mlx-community/Qwen2.5-3B-Instruct-4bit") == "qwen2.5:3b"
    assert to_short_name("mlx-community/Qwen2.5-7B-Instruct-4bit") == "qwen2.5:7b"
    # Unknown MLX id — identity passthrough
    assert to_short_name("mlx-community/SomethingElse") == "mlx-community/SomethingElse"
    # Round-trip: to_mlx → to_short_name should give back the original for known names
    original = "qwen2.5:3b"
    assert to_short_name(to_mlx(original)) == original


# ---------------------------------------------------------------------------
# 3. get_backend default → MLXBackend (post-cutover 2026-05-05)
# ---------------------------------------------------------------------------


def test_get_backend_default_mlx(monkeypatch):
    """Post-cutover 2026-05-05: el default de get_backend es MLX. La conftest
    autouse `_force_ollama_backend_for_tests` setea ollama por test, así que
    aquí explícitamente desetea para verificar el verdadero default."""
    _requires_mlx_lm()
    monkeypatch.delenv("RAG_LLM_BACKEND", raising=False)

    from rag.llm_backend import MLXBackend

    backend = get_backend()
    assert isinstance(backend, MLXBackend)
    assert backend.name == "mlx"


# ---------------------------------------------------------------------------
# 4. get_backend mlx env → MLXBackend (skipped when mlx_lm missing)
# ---------------------------------------------------------------------------


def test_get_backend_mlx_when_env_set(monkeypatch):
    _requires_mlx_lm()

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
    _requires_mlx_lm()
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
# 10. cache scan does not require MLX runtime
# ---------------------------------------------------------------------------


def test_list_cached_mlx_models_scans_hf_cache_without_mlx(monkeypatch, tmp_path):
    hub = tmp_path / ".cache" / "huggingface" / "hub"
    (hub / "models--mlx-community--Qwen2.5-7B-Instruct-4bit").mkdir(parents=True)
    (hub / "models--BAAI--bge-reranker-v2-m3").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert list_cached_mlx_models() == [
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
    ]


# ---------------------------------------------------------------------------
# 11. reset_backend clears singleton
# ---------------------------------------------------------------------------


def test_reset_backend_clears_singleton(monkeypatch):
    """Post-Ola 7: solo MLX vivo. El test verifica que reset_backend()
    fuerza re-resolver el singleton (instancias distintas pre/post reset)."""
    _requires_mlx_lm()
    monkeypatch.setenv("RAG_LLM_BACKEND", "mlx")

    b1 = get_backend()
    assert isinstance(b1, MLXBackend)

    reset_backend()

    b2 = get_backend()
    assert isinstance(b2, MLXBackend)
    # They're different objects because reset_backend cleared the singleton
    assert b1 is not b2


def test_mlx_forward_lock_survives_module_reload():
    """Reloads must not split the process-wide Metal critical section."""
    code = (
        "import importlib; "
        "import rag.llm_backend as lb; "
        "lock = lb._MLX_FORWARD_LOCK; "
        "reloaded = importlib.reload(lb); "
        "raise SystemExit(0 if reloaded._MLX_FORWARD_LOCK is lock else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
