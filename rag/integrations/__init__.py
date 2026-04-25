"""obsidian-rag integrations — leaf modules extracted from rag.py (Phase 1 of monolith split, 2026-04-25).

Each module here owns the side-effect-only ETL for one external source:
whatsapp / gmail / drive / calendar / reminders / apple_mail / weather /
screentime / chrome_bookmarks. They are imported lazily by the orchestrators
in `rag.__init__` (the rag.py monolith renamed) via a re-export block at the
bottom of __init__.py — every `rag.<symbol>` reference still resolves so
existing tests (which monkey-patch attributes directly on `rag`) keep working.
"""

