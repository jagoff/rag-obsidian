"""Tests para (C) — index parcial de conversations bot-only, gateado por
``RAG_INDEX_CONVERSATIONS_BOT_ONLY=1``.
"""
from __future__ import annotations

import os

import rag


CONV_PATH = "04-Archive/99-obsidian-system/99-AI/conversations/2026-01-01-foo.md"


def test_excluded_by_default(monkeypatch):
    monkeypatch.delenv("RAG_INDEX_CONVERSATIONS_BOT_ONLY", raising=False)
    assert rag.is_excluded(CONV_PATH) is True
    assert rag._conversations_indexable(CONV_PATH) is False


def test_indexable_when_env_set(monkeypatch):
    monkeypatch.setenv("RAG_INDEX_CONVERSATIONS_BOT_ONLY", "1")
    assert rag._conversations_indexable(CONV_PATH) is True
    assert rag.is_excluded(CONV_PATH) is False


def test_other_system_paths_still_excluded_with_env(monkeypatch):
    monkeypatch.setenv("RAG_INDEX_CONVERSATIONS_BOT_ONLY", "1")
    # Otras paths de 99-obsidian-system siguen excluidas.
    assert rag.is_excluded("04-Archive/99-obsidian-system/99-Templates/foo.md") is True
    # Memory exception sigue intacta.
    assert rag.is_excluded("04-Archive/99-obsidian-system/99-AI/memory/x.md") is False


def test_strip_user_queries():
    raw = (
        "---\nsession_id: abc\ncreated: 2026-01-01\n---\n\n"
        "## Turn 1\n\n"
        "> esta es la pregunta del user\n\n"
        "Esta es la respuesta del bot.\n"
        "Tiene varias líneas.\n\n"
        "**Sources**: [[foo]] · [[bar]]\n"
    )
    out = rag._strip_conversation_user_queries(raw)
    assert "## Turn 1" in out
    assert "respuesta del bot" in out
    assert "varias líneas" in out
    # Removed:
    assert "esta es la pregunta del user" not in out
    assert "**Sources**" not in out


def test_strip_preserves_frontmatter_and_headers():
    raw = "---\ntitle: x\n---\n\n## Turn 1\n\n> q\n\ncontent\n"
    out = rag._strip_conversation_user_queries(raw)
    assert out.startswith("---\n")
    assert "## Turn 1" in out
    assert "content" in out
    assert "> q" not in out
