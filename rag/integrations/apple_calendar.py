"""Apple Calendar ETL — extracted from rag/cross_source_etls.py 2026-05-09.

Snapshots upcoming events from Apple Calendar (default 90 days ahead, max
200 events) to a per-week markdown note under
``99-obsidian/99-AI/external-ingest/Calendar/<YYYY-Www>.md`` so the regular
``_run_index`` rglob absorbs them.

Requires ``icalBuddy`` (``brew install ical-buddy``); silent-fails when the
binary is missing.

Silent-fail contract: returns ``{ok: False, reason: "..."}`` instead of
raising. ``_atomic_write_if_changed`` is lazy-imported from
``rag.cross_source_etls``. ``_apple_enabled``, ``_icalbuddy_path``, and
``_fetch_calendar_ahead`` are lazy-imported from ``rag`` top-level (defined
in ``rag/integrations/calendar.py`` — distinct from this ETL wrapper, which
just wires the fetcher into the per-week vault note).

Naming: ``apple_calendar.py`` (no ``calendar.py``) to avoid shadowing the
existing fetcher integration ``rag/integrations/calendar.py`` that this
wrapper consumes (same naming pattern as
``apple_reminders.py`` / ``spotify_etl.py``).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rag._constants import _EXTERNAL_INGEST_BASE

__all__ = [
    "_CALENDAR_VAULT_SUBPATH",
    "_sync_apple_calendar_notes",
]

_CALENDAR_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/Calendar"


def _sync_apple_calendar_notes(vault_root: Path, days_ahead: int = 90) -> dict:
    """Snapshot upcoming Apple Calendar events to per-week notes. Requires
    icalBuddy (`brew install ical-buddy`); returns silently when missing.
    """
    from rag import _apple_enabled, _icalbuddy_path, _fetch_calendar_ahead  # lazy
    from rag.cross_source_etls import _atomic_write_if_changed

    if not _apple_enabled():
        return {"ok": False, "reason": "apple_disabled"}
    if not _icalbuddy_path():
        return {"ok": False, "reason": "icalbuddy_missing"}
    events = _fetch_calendar_ahead(days_ahead=days_ahead, max_events=200)
    if not events:
        return {"ok": True, "files_written": 0, "reason": "no_events"}
    now = datetime.now()
    iso_year, iso_week, _ = now.isocalendar()
    week_label = f"{iso_year}-W{iso_week:02d}"

    fm_lines = [
        "---",
        "source: apple-calendar",
        f"snapshot_at: {now.isoformat(timespec='seconds')}",
        f"window_days: {days_ahead}",
        f"event_count: {len(events)}",
        "tags:",
        "- apple-calendar",
        "- system-snapshot",
        "---",
        "",
        f"# Calendar — semana {week_label} (próximos {days_ahead}d)",
        "",
    ]
    body_lines: list[str] = list(fm_lines)
    current_label = None
    for ev in events:
        label = ev.get("date_label") or "(sin fecha)"
        if label != current_label:
            body_lines.append(f"## {label}")
            body_lines.append("")
            current_label = label
        time_range = ev.get("time_range") or ""
        time_part = f"`{time_range}` · " if time_range else ""
        body_lines.append(f"- {time_part}{ev.get('title', '(sin título)')}")
    body_lines.append("")
    body = "\n".join(body_lines)

    target = vault_root / _CALENDAR_VAULT_SUBPATH / f"{week_label}.md"
    written = _atomic_write_if_changed(target, body)
    return {
        "ok": True,
        "files_written": 1 if written else 0,
        "events": len(events),
        "target": _CALENDAR_VAULT_SUBPATH,
    }
