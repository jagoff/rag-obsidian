"""Tests for GC#2.B (2026-04-22) — `_resolve_reranker_model_path()` routing.

Invariants:
1. No env, no symlink → baseline BAAI/bge-reranker-v2-m3 wins.
2. `RAG_RERANKER_FT_PATH` env override wins when the directory exists.
3. `RAG_RERANKER_FT_PATH` pointing at non-existent path → fall back.
4. `~/.cache/obsidian-rag/reranker-ft-current` symlink wins when env unset.
5. Broken symlink (target deleted) → fall back to baseline.
6. Env wins over symlink (explicit > implicit).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import rag


@pytest.fixture
def ft_home(tmp_path, monkeypatch):
    """Redirect ~/.cache/obsidian-rag to a tmp dir for each test."""
    cache = tmp_path / ".cache" / "obsidian-rag"
    cache.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("RAG_RERANKER_FT_PATH", raising=False)
    yield cache


def test_baseline_when_no_env_no_symlink(ft_home):
    assert rag._resolve_reranker_model_path() == rag.RERANKER_MODEL


def test_env_override_wins_when_dir_exists(ft_home, tmp_path, monkeypatch):
    ft_dir = tmp_path / "my-ft"
    ft_dir.mkdir()
    monkeypatch.setenv("RAG_RERANKER_FT_PATH", str(ft_dir))
    assert rag._resolve_reranker_model_path() == str(ft_dir)


def test_env_override_nonexistent_falls_back(ft_home, monkeypatch):
    monkeypatch.setenv("RAG_RERANKER_FT_PATH", "/nonexistent/path")
    assert rag._resolve_reranker_model_path() == rag.RERANKER_MODEL


def test_symlink_used_when_env_unset(ft_home, tmp_path):
    ft_dir = tmp_path / "my-ft"
    ft_dir.mkdir()
    current = ft_home / "reranker-ft-current"
    current.symlink_to(ft_dir)
    resolved = rag._resolve_reranker_model_path()
    # Symlink resolves to the real dir (.resolve() canonicalises).
    assert Path(resolved).resolve() == ft_dir.resolve()


def test_broken_symlink_falls_back(ft_home, tmp_path):
    ft_dir = tmp_path / "deleted"
    ft_dir.mkdir()
    current = ft_home / "reranker-ft-current"
    current.symlink_to(ft_dir)
    ft_dir.rmdir()  # break the link
    assert rag._resolve_reranker_model_path() == rag.RERANKER_MODEL


def test_env_wins_over_symlink(ft_home, tmp_path, monkeypatch):
    """Explicit env path overrides the implicit symlink."""
    env_dir = tmp_path / "via-env"
    env_dir.mkdir()
    symlink_dir = tmp_path / "via-symlink"
    symlink_dir.mkdir()
    (ft_home / "reranker-ft-current").symlink_to(symlink_dir)
    monkeypatch.setenv("RAG_RERANKER_FT_PATH", str(env_dir))
    assert rag._resolve_reranker_model_path() == str(env_dir)
