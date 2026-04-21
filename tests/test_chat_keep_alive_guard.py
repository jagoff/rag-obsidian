"""Tests for the `chat_keep_alive()` guard against the 2026-04-17 Mac-freeze
regression.

The regression: `OLLAMA_KEEP_ALIVE=-1` pins the chat model as wired memory
in macOS unified-memory systems; for models ≥~10 GB (command-r, qwen3:30b-a3b)
this blocks the kernel from swapping, triggering beachballs.

Post-2026-04-21: code-level default of OLLAMA_KEEP_ALIVE is "-1" (since the
default chat model is qwen2.5:7b ≈4.7 GB, safe). The guard auto-clamps to
"20m" if resolve_chat_model() falls back to a large model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

import rag  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_chat_cache():
    """Ensure every test starts from a clean resolve_chat_model() cache so
    monkeypatched preferences take effect without cross-contamination."""
    rag._CHAT_MODEL_RESOLVED = None
    yield
    rag._CHAT_MODEL_RESOLVED = None


def test_large_chat_models_frozenset_is_not_empty():
    """Sanity — the allowlist isn't accidentally empty after refactor."""
    assert len(rag._LARGE_CHAT_MODELS) >= 2
    # Both families the 2026-04-17 incident touched must be listed.
    assert any("command-r" in m for m in rag._LARGE_CHAT_MODELS)
    assert any("qwen3" in m for m in rag._LARGE_CHAT_MODELS)


def test_small_model_passthrough(monkeypatch):
    """Non-large models return OLLAMA_KEEP_ALIVE raw (no clamp)."""
    monkeypatch.setattr(rag, "OLLAMA_KEEP_ALIVE", -1)
    assert rag.chat_keep_alive("qwen2.5:7b") == -1
    assert rag.chat_keep_alive("qwen2.5:14b") == -1
    assert rag.chat_keep_alive("phi4:latest") == -1
    # string keep_alive also passes through
    monkeypatch.setattr(rag, "OLLAMA_KEEP_ALIVE", "24h")
    assert rag.chat_keep_alive("qwen2.5:7b") == "24h"


def test_large_model_clamps_to_20m(monkeypatch):
    """Large models force-clamp regardless of OLLAMA_KEEP_ALIVE value.
    Covers every known entry in _LARGE_CHAT_MODELS."""
    monkeypatch.setattr(rag, "OLLAMA_KEEP_ALIVE", -1)
    for large_model in rag._LARGE_CHAT_MODELS:
        assert rag.chat_keep_alive(large_model) == rag._LARGE_KEEP_ALIVE, \
            f"Failed to clamp {large_model}"


def test_large_model_clamps_even_if_user_set_24h(monkeypatch):
    """Regression — the user's explicit `OLLAMA_KEEP_ALIVE=24h` must not
    override the clamp. That's the whole point: protect the Mac from
    wired-memory pinning even when the user opted-in globally."""
    monkeypatch.setattr(rag, "OLLAMA_KEEP_ALIVE", "24h")
    assert rag.chat_keep_alive("command-r:latest") == "20m"
    assert rag.chat_keep_alive("qwen3:30b-a3b") == "20m"


def test_large_model_respects_explicit_override_env(monkeypatch):
    """RAG_KEEP_ALIVE_LARGE_MODEL is the explicit opt-out for users with
    >64 GB who know what they're doing."""
    monkeypatch.setattr(rag, "OLLAMA_KEEP_ALIVE", -1)
    monkeypatch.setenv("RAG_KEEP_ALIVE_LARGE_MODEL", "4h")
    assert rag.chat_keep_alive("command-r:latest") == "4h"
    # Integer also accepted (seconds)
    monkeypatch.setenv("RAG_KEEP_ALIVE_LARGE_MODEL", "3600")
    assert rag.chat_keep_alive("command-r:latest") == 3600


def test_large_model_ignores_empty_override(monkeypatch):
    """Empty-string override must not poison the default — falls back to
    the baked-in _LARGE_KEEP_ALIVE."""
    monkeypatch.setattr(rag, "OLLAMA_KEEP_ALIVE", -1)
    monkeypatch.setenv("RAG_KEEP_ALIVE_LARGE_MODEL", "")
    assert rag.chat_keep_alive("command-r:latest") == rag._LARGE_KEEP_ALIVE


def test_no_model_installed_degrades_gracefully(monkeypatch):
    """resolve_chat_model() raises RuntimeError when no chat model exists.
    chat_keep_alive() must not crash — it returns OLLAMA_KEEP_ALIVE raw
    because we can't classify an unknown model. The caller will fail on
    the ollama.chat() anyway, so this is not papering over a real bug."""
    monkeypatch.setattr(rag, "OLLAMA_KEEP_ALIVE", -1)

    def _raise():
        raise RuntimeError("no models installed")

    monkeypatch.setattr(rag, "resolve_chat_model", _raise)
    # No model param → triggers the except path
    assert rag.chat_keep_alive() == -1


def test_default_resolve_when_model_not_passed(monkeypatch):
    """When model=None, the helper resolves via resolve_chat_model().
    Preference order should return qwen2.5:7b (small) in the test env,
    so default should pass through OLLAMA_KEEP_ALIVE unchanged."""
    monkeypatch.setattr(rag, "OLLAMA_KEEP_ALIVE", -1)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "qwen2.5:7b")
    assert rag.chat_keep_alive() == -1


def test_default_resolve_large_model_clamps(monkeypatch):
    """When model=None and resolve_chat_model() returns a large model
    (the fallback scenario that motivated this guard), clamp fires."""
    monkeypatch.setattr(rag, "OLLAMA_KEEP_ALIVE", -1)
    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "command-r:latest")
    assert rag.chat_keep_alive() == "20m"


def test_large_keep_alive_parses_to_same_value_as_env():
    """Defensive: the baked-in _LARGE_KEEP_ALIVE should parse cleanly via
    _parse_keep_alive (i.e. ollama accepts it). Catches a typo refactor that
    would silently cause ollama to fall back to its internal 5m default."""
    # _parse_keep_alive returns int for numeric strings, else the raw string.
    parsed = rag._parse_keep_alive(rag._LARGE_KEEP_ALIVE)
    assert isinstance(parsed, (int, str))
    # Our cap uses a duration string, so it should stay a string.
    assert parsed == rag._LARGE_KEEP_ALIVE
