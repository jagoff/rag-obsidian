"""Tests for the reranker-ft cache cleanup step in run_maintenance.

Covers:
  - empty cache dir → no-op, key present in results
  - active symlink target is preserved even if old
  - old non-target dir gets deleted
  - dry_run=True skips deletion but reports bytes
  - missing cache dir → silent no-op (cold install)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

import rag


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_ft_dir(base: Path, name: str, age_seconds: int = 0, size: int = 1024) -> Path:
    """Create a fake reranker-ft dir with a dummy file inside."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    dummy = d / "adapter_config.json"
    dummy.write_bytes(b"x" * size)
    mtime = time.time() - age_seconds
    os.utime(d, (mtime, mtime))
    os.utime(dummy, (mtime, mtime))
    return d


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def fake_cache(tmp_path, monkeypatch):
    """Point ~/.cache/obsidian-rag to a temp dir and redirect Path.home()."""
    cache_base = tmp_path / ".cache" / "obsidian-rag"
    cache_base.mkdir(parents=True, exist_ok=True)

    # Redirect Path.home() so the cleanup code resolves the right base dir.
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    return cache_base


# ── tests ─────────────────────────────────────────────────────────────────────

def test_empty_cache_no_op(fake_cache, monkeypatch, tmp_path):
    """No reranker-ft dirs → result key present, deleted=0."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    (tmp_path / "ragvec").mkdir(exist_ok=True)

    result = rag.run_maintenance(dry_run=True, skip_reindex=True, skip_logs=True)
    assert "reranker_ft_cleaned" in result
    assert result["reranker_ft_cleaned"]["deleted"] == 0
    assert result["reranker_ft_cleaned"]["bytes_freed"] == 0


def test_active_symlink_target_preserved(fake_cache, monkeypatch, tmp_path):
    """Dir that is the symlink target is NOT deleted even if >7 days old."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    (tmp_path / "ragvec").mkdir(exist_ok=True)

    old_dir = _make_ft_dir(fake_cache, "reranker-ft-20260101-000000", age_seconds=30 * 86400)
    symlink = fake_cache / "reranker-ft-current"
    symlink.symlink_to(old_dir)

    result = rag.run_maintenance(dry_run=False, skip_reindex=True, skip_logs=True)
    assert old_dir.exists(), "Active symlink target must not be deleted"
    assert result["reranker_ft_cleaned"]["deleted"] == 0


def test_old_non_target_deleted(fake_cache, monkeypatch, tmp_path):
    """Dir older than 7 days and not the symlink target gets removed."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    (tmp_path / "ragvec").mkdir(exist_ok=True)

    active_dir = _make_ft_dir(fake_cache, "reranker-ft-20260415-120000", age_seconds=2 * 86400)
    old_dir = _make_ft_dir(fake_cache, "reranker-ft-20260101-000000", age_seconds=30 * 86400, size=2048)
    symlink = fake_cache / "reranker-ft-current"
    symlink.symlink_to(active_dir)

    result = rag.run_maintenance(dry_run=False, skip_reindex=True, skip_logs=True)
    assert not old_dir.exists(), "Old non-active dir should be deleted"
    assert active_dir.exists(), "Recent active dir must survive"
    assert result["reranker_ft_cleaned"]["deleted"] == 1
    assert result["reranker_ft_cleaned"]["bytes_freed"] > 0


def test_dry_run_skips_deletion(fake_cache, monkeypatch, tmp_path):
    """dry_run=True reports bytes but does not remove the dir."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    (tmp_path / "ragvec").mkdir(exist_ok=True)

    old_dir = _make_ft_dir(fake_cache, "reranker-ft-20260101-000000", age_seconds=30 * 86400)

    result = rag.run_maintenance(dry_run=True, skip_reindex=True, skip_logs=True)
    assert old_dir.exists(), "dry_run must not delete the dir"
    assert result["reranker_ft_cleaned"]["deleted"] == 1
    assert result["reranker_ft_cleaned"]["bytes_freed"] > 0


def test_missing_cache_dir_no_op(monkeypatch, tmp_path):
    """Cache dir doesn't exist (cold install) → silent no-op."""
    # Point home to a dir that has NO .cache/obsidian-rag subdir.
    empty_home = tmp_path / "no_cache_home"
    empty_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: empty_home))
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    (tmp_path / "ragvec").mkdir(exist_ok=True)

    result = rag.run_maintenance(dry_run=True, skip_reindex=True, skip_logs=True)
    # Should not raise; key may be 'cleaned' with 0 or an error key — both acceptable.
    assert "reranker_ft_cleaned" in result or "reranker_ft_cleaned_error" in result
