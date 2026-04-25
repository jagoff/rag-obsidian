"""Apple Mail integration — leaf ETL extracted from `rag/__init__.py` (Phase 1b).

Source: macOS Mail.app via `osascript`. Queries the unified `inbox` (one
single AS query across all accounts — ~0.5s) for unread messages received
in the last 36h, and tags any sender matching `MAIL_VIP_CONFIG_PATH` so
the brief renderer floats them to the top.

## Invariants
- Silent-fail: Apple env disabled (`OBSIDIAN_RAG_NO_APPLE=1`), osascript
  timeout, no Mail.app installed, permission denial → return `[]`.
  Never raise.
- Body capped to 600 chars in AppleScript, stripped of `|`/newline/tab so
  the pipe-separated parse stays unambiguous; Python normalises HTML +
  whitespace and caps to 200 chars for the preview.
- VIPs are sorted ABOVE non-VIPs before the `max_items` truncation — so
  important senders survive even when the inbox is noisy.

## Why deferred imports
`_apple_enabled`, `_osascript`, `_load_mail_vips`, `_strip_html_to_preview`,
and `_is_vip_sender` all live in `rag.__init__`. Module-level imports here
would deadlock the package load. Function-body imports run after
`rag.__init__` finishes loading, so they always succeed.
"""

from __future__ import annotations


# Use Mail's unified `inbox` alias — single query across all accounts, ~0.5s.
# Previous per-account iteration was 10-20× slower and still missed Gmail
# (no dedicated INBOX mailbox — Gmail uses labels).
# Body is truncated to 600 chars in AS and stripped of `|`/newlines/tabs so the
# pipe-separated parse stays unambiguous; Python normalises HTML+whitespace and
# caps to 200 chars for the preview.
_MAIL_SCRIPT = '''
set _cutoff to (current date) - (36 * hours)
set _out to ""
tell application "Mail"
  try
    repeat with _msg in (messages of inbox whose read status is false and date received > _cutoff)
      try
        set _subject to subject of _msg
        set _sender to sender of _msg
        set _received to (date received of _msg) as string
        set _body to ""
        try
          set _body to (content of _msg) as string
        end try
        if (count of _body) > 600 then
          set _body to text 1 thru 600 of _body
        end if
        set _prev_tids to AppleScript's text item delimiters
        set AppleScript's text item delimiters to {return, linefeed, character id 9, "|"}
        set _bparts to text items of _body
        set AppleScript's text item delimiters to " "
        set _body to _bparts as text
        set AppleScript's text item delimiters to _prev_tids
        set _out to _out & _subject & "|" & _sender & "|" & _received & "|" & _body & linefeed
      end try
    end repeat
  end try
end tell
return _out
'''


def _fetch_mail_unread(max_items: int = 10) -> list[dict]:
    """Unread messages received in the last 36h from Apple Mail INBOX
    across all accounts. Each item carries ``subject``, ``sender``,
    ``received``, ``body_preview`` (≤200 chars, HTML stripped) and
    ``is_vip`` (sender matches an entry in ``MAIL_VIP_CONFIG_PATH``).
    VIPs are sorted to the top before the ``max_items`` cap so they
    survive truncation.
    """
    from rag import (
        _apple_enabled,
        _is_vip_sender,
        _load_mail_vips,
        _osascript,
        _strip_html_to_preview,
    )
    if not _apple_enabled():
        return []
    out = _osascript(_MAIL_SCRIPT, timeout=20.0)
    if not out:
        return []
    vips = _load_mail_vips()
    items: list[dict] = []
    for line in out.splitlines():
        parts = line.split("|", 3)
        if len(parts) < 2:
            continue
        subject = parts[0].strip()
        sender = parts[1].strip()
        received = parts[2].strip() if len(parts) > 2 else ""
        body_raw = parts[3] if len(parts) > 3 else ""
        if not subject:
            continue
        items.append({
            "subject": subject,
            "sender": sender,
            "received": received,
            "body_preview": _strip_html_to_preview(body_raw, cap=200),
            "is_vip": _is_vip_sender(sender, vips),
        })
    items.sort(key=lambda m: 0 if m.get("is_vip") else 1)
    return items[:max_items]
