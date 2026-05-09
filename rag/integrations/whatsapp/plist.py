"""WhatsApp launchd plist generators.

Surface:

- ``_wa_tasks_plist(rag_bin)`` — string del plist para
  ``com.fer.obsidian-rag-wa-tasks``. Cron 30min, lee delta del bridge SQLite,
  escribe `00-Inbox/WA-YYYY-MM-DD.md` con tasks/questions/commitments.

Generators consumidos por `rag/plists/_spec.py` (``_services_spec()`` lo
re-exporta vía lazy import) y por el setup loop (`rag setup` que escribe
los plists a `~/Library/LaunchAgents/`).
"""

from __future__ import annotations

from pathlib import Path


def _wa_tasks_plist(rag_bin: str) -> str:
    """WhatsApp action-item extractor — every 30min.

    Reads delta from the bridge SQLite since last run and distills tasks/
    questions/commitments to `00-Inbox/WA-YYYY-MM-DD.md`. Cheap: one
    qwen2.5:3b call per chat with new inbound messages (capped at 12
    chats). `ambient: skip` in the output frontmatter prevents the
    WhatsApp push loop.
    """
    from rag import _RAG_LOG_DIR
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-wa-tasks</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>wa-tasks</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
    <key>RAG_LLM_BACKEND</key><string>mlx</string>
    <key>HF_HUB_OFFLINE</key><string>1</string>
    <key>TRANSFORMERS_OFFLINE</key><string>1</string>
  </dict>
  <key>StartInterval</key><integer>1800</integer>
  <key>RunAtLoad</key><false/>
  <key>ThrottleInterval</key><integer>60</integer>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/wa-tasks.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/wa-tasks.error.log</string>
</dict>
</plist>
"""


__all__ = [
    "_wa_tasks_plist",
]
