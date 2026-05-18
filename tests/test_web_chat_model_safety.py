"""Regression tests for web chat model safety guardrails."""
from __future__ import annotations

from web import server as server_mod


def test_resolve_web_chat_model_guards_30b_default(monkeypatch):
    monkeypatch.delenv("RAG_WEB_ALLOW_HEAVY_CHAT_MODEL", raising=False)
    monkeypatch.setattr(server_mod, "_read_chat_model_override", lambda: None)
    monkeypatch.setattr(server_mod, "WEB_CHAT_MODEL", "qwen3:30b-a3b")
    monkeypatch.setattr(server_mod, "_WEB_SAFE_CHAT_MODEL", "qwen2.5:7b")
    monkeypatch.setattr(
        server_mod,
        "_web_mlx_model_available",
        lambda model: model == "qwen2.5:7b",
    )
    server_mod._WEB_HEAVY_MODEL_WARNED.clear()

    assert server_mod._resolve_web_chat_model() == "qwen2.5:7b"


def test_resolve_web_chat_model_allows_30b_with_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("RAG_WEB_ALLOW_HEAVY_CHAT_MODEL", "1")
    monkeypatch.setattr(server_mod, "_read_chat_model_override", lambda: None)
    monkeypatch.setattr(server_mod, "WEB_CHAT_MODEL", "qwen3:30b-a3b")
    monkeypatch.setattr(server_mod, "_WEB_SAFE_CHAT_MODEL", "qwen2.5:7b")
    monkeypatch.setattr(server_mod, "_web_mlx_model_available", lambda model: True)
    server_mod._WEB_HEAVY_MODEL_WARNED.clear()

    assert server_mod._resolve_web_chat_model() == "qwen3:30b-a3b"

