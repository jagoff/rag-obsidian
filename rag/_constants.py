"""Shared constants used by multiple sub-modules.

Lives separate from rag/__init__.py to avoid circular imports between
sub-modules (plists, cross_source_etls, archive, postprocess, etc.).
"""
from __future__ import annotations

from pathlib import Path

# Google OAuth token path (Calendar + Gmail + Drive integrations).
# Single source of truth — plists.py and cross_source_etls.py import from here.
_GOOGLE_TOKEN_PATH = Path.home() / ".config/obsidian-rag" / "google_token.json"
