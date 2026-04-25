"""Apple Calendar integration — leaf ETL extracted from `rag/__init__.py` (Phase 1b).

Source: macOS Calendar.app via [icalBuddy](https://hasseg.org/icalBuddy/),
a CLI that walks the EventKit DB and emits text-formatted events. Required
binary path is resolved by `_icalbuddy_path()` (which lives in `rag.__init__`
because it's also used by `_sync_apple_calendar_notes`); silent-fail when
icalBuddy is not installed. The user can `brew install ical-buddy` to enable.

Two surfaces:

- `_fetch_calendar_today(max_events)` — events for today only, sorted by
  start time, capped at N. Used by morning briefs.
- `_fetch_calendar_ahead(days_ahead, max_events)` — events for today through
  today+N days, with relative-date labels (e.g. "tomorrow", "Sat Apr 19").
  Used by week / next-week views.

## Invariants
- Silent-fail: `OBSIDIAN_RAG_NO_APPLE=1`, missing icalBuddy, subprocess timeout
  or non-zero exit, malformed output → return `[]`. Never raise.
- icalBuddy timeout is 10s per call; the EventKit DB usually answers in <500ms
  so 10s is a defensive ceiling for cold-launches when Calendar.app hasn't
  cached its calendars yet.

## Why deferred imports
`_apple_enabled` and `_icalbuddy_path` live in `rag.__init__`. We import them
inside the function bodies via `from rag import …` so:

1. Module-level imports here would trigger a circular import during the
   `rag.__init__` package load.
2. Tests `monkeypatch.setattr(rag, "_icalbuddy_path", ...)` need the lookup to
   happen at call time on the `rag` module's namespace — `from rag import X`
   inside a function body does exactly that (re-resolves the attribute on each
   call), so the patch is honored.
"""

from __future__ import annotations

import re


def _fetch_calendar_today(max_events: int = 15) -> list[dict]:
    """Events scheduled for today via icalBuddy. Returns [] if icalBuddy is
    not installed — the user can `brew install ical-buddy` to enable.

    Output parsing handles the default icalBuddy format:
        Event title
            list: CalendarName
            date: 14/04/2026 at 09:30 - 10:00
    """
    from rag import _apple_enabled, _icalbuddy_path
    if not _apple_enabled():
        return []
    icb = _icalbuddy_path()
    if not icb:
        return []
    import subprocess
    try:
        res = subprocess.run(
            [
                icb,
                "-npn",                        # no property names
                "-nc",                          # no calendar names inline
                "-nrd",                         # no relative dates
                "-ea",                          # exclude all-day events? no, include.
                "-iep", "title,datetime",       # include only: title + datetime
                "-b", "",                       # no bullet prefix
                "eventsToday",
            ],
            capture_output=True, text=True, timeout=10.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if res.returncode != 0:
        return []
    out = (res.stdout or "").strip()
    if not out:
        return []
    events: list[dict] = []
    current: dict | None = None
    for raw in out.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        # title lines start at col 0; property lines are indented
        if not line.startswith(" ") and not line.startswith("\t"):
            if current and current.get("title"):
                events.append(current)
            current = {"title": line.strip(), "start": "", "end": ""}
            continue
        # property line — look for date/time range
        stripped = line.strip()
        if current is None:
            continue
        # Formats seen: "today at 09:30 - 10:00", "14/04/2026 at 09:30 - 10:00",
        # or bare "09:30 - 10:00"
        m = re.search(r"(\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?)\s*-\s*(\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?)", stripped)
        if m:
            current["start"] = m.group(1)
            current["end"] = m.group(2)
    if current and current.get("title"):
        events.append(current)
    events.sort(key=lambda e: e["start"] or "99:99")
    return events[:max_events]


def _fetch_calendar_ahead(days_ahead: int, max_events: int = 40) -> list[dict]:
    """icalBuddy `eventsToday+N` with relative-date labels. Returns
    [{title, date_label, time_range}]. Silent-fail per contract."""
    from rag import _apple_enabled, _icalbuddy_path
    if not _apple_enabled() or days_ahead < 0:
        return []
    icb = _icalbuddy_path()
    if not icb:
        return []
    import subprocess as _sp
    query = "eventsToday" if days_ahead == 0 else f"eventsToday+{days_ahead}"
    try:
        res = _sp.run(
            [icb, "-nc", "-iep", "title,datetime", "-b", "", query],
            capture_output=True, text=True, timeout=10.0,
        )
    except (FileNotFoundError, _sp.TimeoutExpired, OSError):
        return []
    if res.returncode != 0 or not (res.stdout or "").strip():
        return []
    events: list[dict] = []
    current: dict | None = None
    for raw in (res.stdout or "").splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if not line.startswith(" ") and not line.startswith("\t"):
            if current and current.get("title"):
                events.append(current)
            current = {"title": line.strip(), "date_label": "", "time_range": ""}
            continue
        stripped = line.strip()
        if current is None:
            continue
        m = re.search(
            r"(\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?)\s*-\s*(\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?)",
            stripped,
        )
        if m:
            current["time_range"] = f"{m.group(1)}–{m.group(2)}"
            label = stripped.split(" at ", 1)[0].strip()
            if label and label != stripped:
                current["date_label"] = label
        else:
            current["date_label"] = stripped
    if current and current.get("title"):
        events.append(current)
    return events[:max_events]
