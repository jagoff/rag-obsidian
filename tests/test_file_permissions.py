"""Tests for state-dir + OAuth token permission hardening.

See rag.py:428-471 (_ensure_state_dir_secure + _write_secret_file) and
4868-4890 (_harden_oauth_cache_perms). queries.jsonl + behavior.jsonl
contain literal user prompts + open/save events (PII); OAuth refresh
tokens grant indefinite Gmail/Drive/Spotify access. Neither should be
world- or group-readable.
"""
from __future__ import annotations

import os
from pathlib import Path


import rag


def _mode(p: Path) -> str:
    return oct(p.stat().st_mode & 0o777)


def test_state_dir_is_0o700_after_import():
    """Module import must leave ~/.local/share/obsidian-rag at 0o700.

    This is the directory holding queries.jsonl + behavior.jsonl
    (PII) and tokens. On shared hosts it must not be world-readable.
    """
    assert rag._STATE_DIR.is_dir()
    assert _mode(rag._STATE_DIR) == "0o700", (
        f"state dir {rag._STATE_DIR} must be 0o700, got {_mode(rag._STATE_DIR)}"
    )


def test_ensure_state_dir_secure_tightens_existing_loose_perms(tmp_path, monkeypatch):
    """If the state dir pre-exists with umask-022 perms (0o755),
    a fresh import (via re-running the helper) must chmod to 0o700."""
    fake_home = tmp_path
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    # Simulate a user whose state dir was created before hardening.
    loose = fake_home / ".local/share/obsidian-rag"
    loose.mkdir(parents=True)
    os.chmod(loose, 0o755)
    assert _mode(loose) == "0o755"

    # Run helper — should chmod to 0o700.
    returned = rag._ensure_state_dir_secure()
    assert returned == loose
    assert _mode(loose) == "0o700"


def test_write_secret_file_creates_0o600_atomically(tmp_path):
    """OAuth token writes must land at 0o600 and be visible atomically.

    Verifies:
      - final file at path is 0o600
      - final file has expected content
      - no stale .tmp.* siblings are left behind
      - rewriting overwrites cleanly
    """
    target = tmp_path / "nested/dir/google_token.json"
    rag._write_secret_file(target, '{"refresh_token": "first"}')
    assert target.is_file()
    assert _mode(target) == "0o600"
    assert target.read_text() == '{"refresh_token": "first"}'
    # Parent dir also tightened.
    assert _mode(target.parent) == "0o700"
    # No leftover tmp files.
    leftovers = [p for p in target.parent.iterdir() if p.name != target.name]
    assert leftovers == []

    # Rewrite — still 0o600, content updated.
    rag._write_secret_file(target, '{"refresh_token": "second"}')
    assert target.read_text() == '{"refresh_token": "second"}'
    assert _mode(target) == "0o600"


def test_write_secret_file_survives_chmod_failure(tmp_path, monkeypatch):
    """On FS that reject chmod (NFS, SMB, FUSE) we swallow the error
    rather than crash — the write still lands, perms just stay at
    whatever umask gave us. Same policy as _ensure_state_dir_secure."""
    target = tmp_path / "token.json"

    def fake_chmod(*_a, **_kw):
        raise OSError("filesystem doesn't support chmod")

    monkeypatch.setattr(os, "chmod", fake_chmod)
    # Must not raise — just write.
    rag._write_secret_file(target, "{}")
    assert target.is_file()
    assert target.read_text() == "{}"


def test_harden_oauth_cache_perms_is_idempotent(tmp_path, monkeypatch):
    """_harden_oauth_cache_perms runs on every import. Must be idempotent
    + never blow up if tokens are missing (brand-new install)."""
    fake_home = tmp_path
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    cfg = fake_home / ".config/obsidian-rag"
    cfg.mkdir(parents=True)
    goog = cfg / "google_token.json"
    spot = cfg / "spotify_token.json"
    goog.write_text("{}")
    spot.write_text("{}")
    # Pre-hardening state: world-readable.
    os.chmod(cfg, 0o755)
    os.chmod(goog, 0o644)
    os.chmod(spot, 0o644)

    # We have to re-point the module globals at the fake paths for this run.
    monkeypatch.setattr(rag, "_GOOGLE_TOKEN_PATH", goog)
    monkeypatch.setattr(rag, "_SPOTIFY_TOKEN_PATH", spot)

    rag._harden_oauth_cache_perms()
    assert _mode(cfg) == "0o700"
    assert _mode(goog) == "0o600"
    assert _mode(spot) == "0o600"

    # Second call — no change, no exception.
    rag._harden_oauth_cache_perms()
    assert _mode(cfg) == "0o700"
    assert _mode(goog) == "0o600"


def test_harden_oauth_cache_perms_noop_when_tokens_missing(tmp_path, monkeypatch):
    """Fresh install: no token files yet. Must not crash."""
    fake_home = tmp_path
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(rag, "_GOOGLE_TOKEN_PATH", fake_home / "nope_google.json")
    monkeypatch.setattr(rag, "_SPOTIFY_TOKEN_PATH", fake_home / "nope_spotify.json")
    rag._harden_oauth_cache_perms()  # must not raise
