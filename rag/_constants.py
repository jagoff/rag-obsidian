"""Shared constants used by multiple sub-modules.

Lives separate from rag/__init__.py to avoid circular imports between
sub-modules (plists, cross_source_etls, archive, postprocess, etc.).
"""
from __future__ import annotations

from pathlib import Path

# Google OAuth token path (Calendar + Gmail + Drive integrations).
# Single source of truth — plists.py and cross_source_etls.py import from here.
_GOOGLE_TOKEN_PATH = Path.home() / ".config/obsidian-rag" / "google_token.json"

# External-source ETL output base under the vault. All cross-source ETL
# writers (gmail, calendar, drive, screentime, github, claude_code,
# youtube, spotify, reminders, chrome) put notes under
# `<vault>/99-obsidian/99-AI/external-ingest/<source>/...` so the regular
# `_run_index` rglob absorbs them. Lives en `_constants` para que tanto
# `cross_source_etls` (writer principal) como sub-módulos en
# `rag/integrations/*` que ahora alojan algunos ETLs específicos
# (e.g. screentime) compartan la misma raíz sin import circular.
_EXTERNAL_INGEST_BASE = "99-obsidian/99-AI/external-ingest"
