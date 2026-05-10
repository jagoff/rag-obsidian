"""WhatsApp → vault ETL — extracted from rag/cross_source_etls.py 2026-05-09.

Thin wrapper around the external ``~/.local/bin/whatsapp-to-vault`` script
(written in TypeScript, lives in `~/whatsapp-listener`), which reads the
WhatsApp bridge SQLite (`~/repositories/whatsapp-mcp/whatsapp-bridge/store/messages.db`)
and emits per-chat-per-month markdown buckets at
``99-obsidian/99-AI/external-ingest/WhatsApp/<chat>/<YYYY-MM>.md``.

Why subprocess vs reimplementing the parser in Python: the same script also
runs from the ``com.fer.whatsapp-vault-sync`` launchd plist every 15 min.
Single source of truth — if we duplicated the logic in Python the plist
output and the index-time output could drift.

Silent-fail contract: returns ``{ok: False, reason: "..."}`` instead of
raising. Other machines (no listener installed) hit ``script_missing`` and
the rest of ``_run_index`` continues normally.

Note: this is the index-time wrapper (called from the regular RAG indexer).
The actual listener daemon, scheduled-send worker, and tasks-extraction loop
live under ``rag/integrations/whatsapp/`` (12 sub-modules).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from rag._constants import _EXTERNAL_INGEST_BASE

__all__ = [
    "_WHATSAPP_ETL_SCRIPT",
    "_WHATSAPP_ETL_RE",
    "_sync_whatsapp_notes",
]

_WHATSAPP_ETL_SCRIPT = Path.home() / ".local/bin/whatsapp-to-vault"
_WHATSAPP_ETL_RE = re.compile(
    r"wrote\s+(\d+)\s+files,\s+(\d+)\s+unchanged,\s+(\d+)\s+\(chat, month\)\s+buckets,\s+(\d+)\s+chats"
)


def _sync_whatsapp_notes(vault_root: Path) -> dict:
    """Trigger the WhatsApp → vault ETL script and parse its summary line.

    Mirrors the MOZE pre-index pattern: produces `.md` files in
    `<vault>/99-obsidian/99-AI/external-ingest/WhatsApp/<chat>/YYYY-MM.md` so the regular rglob
    picks them up. Subprocess to keep it as a single source of truth — the
    same script that the `com.fer.whatsapp-vault-sync` launchd plist runs
    every 15 min. Silent-fail when the script is missing (other machines).
    """
    if not _WHATSAPP_ETL_SCRIPT.is_file():
        return {"ok": False, "reason": "script_missing"}
    try:
        proc = subprocess.run(
            [str(_WHATSAPP_ETL_SCRIPT)],
            capture_output=True, timeout=60, text=True,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return {"ok": False, "reason": str(exc)[:120]}
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        return {"ok": False, "reason": err[:160] or f"rc={proc.returncode}"}
    m = _WHATSAPP_ETL_RE.search(out)
    if not m:
        return {"ok": True, "raw": out[:160]}
    return {
        "ok": True,
        "files_written": int(m.group(1)),
        "files_unchanged": int(m.group(2)),
        "buckets": int(m.group(3)),
        "chats": int(m.group(4)),
        "target": f"{_EXTERNAL_INGEST_BASE}/WhatsApp",
    }
