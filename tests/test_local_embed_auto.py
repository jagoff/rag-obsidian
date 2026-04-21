"""Tests for auto-enabling RAG_LOCAL_EMBED on query-like CLI subcommands.

Bug motivation: the in-process SentenceTransformer embed path (`query_embed_local`)
is ~10-30ms vs ~140ms via ollama HTTP. It's set in the web + serve launchd plists
but NOT in CLI query/chat paths — meaning every `rag query` and `rag chat` call
pays the 140ms tax unnecessarily. This module tests the auto-enable heuristic.

Invariants tested:

1. `_local_embed_enabled()` reads the env var freshly on each call (not cached
   at import). This is required so the CLI group can set the env var before
   retrieve() reads it.
2. The `cli()` group auto-enables the env var for a known allow-list of
   query-like subcommands (query, chat, do, pendientes, prep, links).
3. Bulk/index subcommands (index, watch) do NOT auto-enable — their path must
   stay on ollama to avoid VRAM pressure from embedding thousands of chunks.
4. An explicit RAG_LOCAL_EMBED=0 (or "false"/"no") disables auto-enable even
   for query-like commands — user override wins.
5. An explicit RAG_LOCAL_EMBED=1 preserved for all commands (symmetric).
"""
import os

import pytest

import rag

# Note: conftest.py::_snapshot_rag_local_embed_env autouse-snapshots + restores
# the env var after every test, covering the side-effect that
# `_maybe_auto_enable_local_embed` has on `os.environ`.


# ── _local_embed_enabled() helper contract ────────────────────────────────────


def test_local_embed_enabled_reads_env_freshly(monkeypatch):
    """The function must re-read os.environ each call — otherwise the CLI
    group can't mutate the flag between import and subcommand dispatch."""
    monkeypatch.delenv("RAG_LOCAL_EMBED", raising=False)
    assert rag._local_embed_enabled() is False

    monkeypatch.setenv("RAG_LOCAL_EMBED", "1")
    assert rag._local_embed_enabled() is True

    monkeypatch.setenv("RAG_LOCAL_EMBED", "0")
    assert rag._local_embed_enabled() is False

    monkeypatch.setenv("RAG_LOCAL_EMBED", "")
    assert rag._local_embed_enabled() is False


@pytest.mark.parametrize("falsy", ["", "0", "false", "no"])
def test_local_embed_disabled_for_falsy_values(monkeypatch, falsy):
    monkeypatch.setenv("RAG_LOCAL_EMBED", falsy)
    assert rag._local_embed_enabled() is False


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "TRUE"])
def test_local_embed_enabled_for_truthy_values(monkeypatch, truthy):
    monkeypatch.setenv("RAG_LOCAL_EMBED", truthy)
    assert rag._local_embed_enabled() is True


# ── Auto-enable allow-list ────────────────────────────────────────────────────


def test_query_like_cmds_is_a_set_of_strings():
    assert isinstance(rag._LOCAL_EMBED_AUTO_CMDS, (set, frozenset))
    assert all(isinstance(c, str) for c in rag._LOCAL_EMBED_AUTO_CMDS)
    # Must contain the core interactive query paths
    assert "query" in rag._LOCAL_EMBED_AUTO_CMDS
    assert "chat" in rag._LOCAL_EMBED_AUTO_CMDS


def test_query_like_cmds_excludes_bulk_paths():
    """Indexing + watch must NOT auto-enable (bulk chunk embedding stays on
    ollama per CLAUDE.md — avoids memory pressure from 10k+ embeds)."""
    assert "index" not in rag._LOCAL_EMBED_AUTO_CMDS
    assert "watch" not in rag._LOCAL_EMBED_AUTO_CMDS


# ── maybe_auto_enable_local_embed() behaviour ─────────────────────────────────


def test_auto_enable_sets_env_for_query_subcmd(monkeypatch):
    """Unset env + query-like subcommand → env becomes '1'."""
    monkeypatch.delenv("RAG_LOCAL_EMBED", raising=False)
    rag._maybe_auto_enable_local_embed("query")
    assert os.environ.get("RAG_LOCAL_EMBED") == "1"


def test_auto_enable_does_nothing_for_index(monkeypatch):
    """index/watch/ingest subcommands must NOT toggle the flag."""
    monkeypatch.delenv("RAG_LOCAL_EMBED", raising=False)
    rag._maybe_auto_enable_local_embed("index")
    assert os.environ.get("RAG_LOCAL_EMBED") is None
    rag._maybe_auto_enable_local_embed("watch")
    assert os.environ.get("RAG_LOCAL_EMBED") is None


def test_auto_enable_respects_explicit_disable(monkeypatch):
    """User explicitly set RAG_LOCAL_EMBED=0 → do NOT flip it back on."""
    monkeypatch.setenv("RAG_LOCAL_EMBED", "0")
    rag._maybe_auto_enable_local_embed("query")
    assert os.environ.get("RAG_LOCAL_EMBED") == "0"


def test_auto_enable_respects_explicit_enable(monkeypatch):
    """User already set RAG_LOCAL_EMBED=1 → stays '1' (no-op, not reset)."""
    monkeypatch.setenv("RAG_LOCAL_EMBED", "1")
    rag._maybe_auto_enable_local_embed("query")
    assert os.environ.get("RAG_LOCAL_EMBED") == "1"


def test_auto_enable_handles_none_subcmd(monkeypatch):
    """ctx.invoked_subcommand is None when user runs bare `rag` (no subcmd)."""
    monkeypatch.delenv("RAG_LOCAL_EMBED", raising=False)
    rag._maybe_auto_enable_local_embed(None)
    assert os.environ.get("RAG_LOCAL_EMBED") is None


def test_auto_enable_for_chat_do_pendientes(monkeypatch):
    """Other core interactive paths are in the allow-list too."""
    for cmd in ("chat", "do", "pendientes"):
        monkeypatch.delenv("RAG_LOCAL_EMBED", raising=False)
        rag._maybe_auto_enable_local_embed(cmd)
        assert os.environ.get("RAG_LOCAL_EMBED") == "1", f"cmd={cmd}"
