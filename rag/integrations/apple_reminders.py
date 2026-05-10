"""Apple Reminders ETL — extracted from rag/cross_source_etls.py 2026-05-09.

Snapshots pending Apple Reminders (horizon 180 days + undated, max 500 items)
to a daily markdown note under
``99-obsidian/99-AI/external-ingest/Reminders/<YYYY-MM-DD>.md`` so the
regular ``_run_index`` rglob absorbs them. Items are bucketed by due-date
proximity (overdue / today / upcoming / undated).

Completed reminders are intentionally NOT fetched — the index would bloat
with stale entries that the user can't act on.

Silent-fail contract: returns ``{ok: False, reason: "..."}`` instead of
raising. ``_atomic_write_if_changed`` is lazy-imported from
``rag.cross_source_etls``. ``_apple_enabled`` and ``_fetch_reminders_due``
are lazy-imported from ``rag`` top-level (defined in
``rag/integrations/reminders.py`` — distinct from this ETL wrapper, which
just wires the fetcher into the per-day vault note).

Naming: ``apple_reminders.py`` (no ``reminders.py``) to avoid shadowing
the existing fetcher integration ``rag/integrations/reminders.py`` that
this wrapper consumes.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rag._constants import _EXTERNAL_INGEST_BASE

__all__ = [
    "_REMINDERS_VAULT_SUBPATH",
    "_sync_reminders_notes",
]

_REMINDERS_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/Reminders"


def _sync_reminders_notes(vault_root: Path) -> dict:
    """Snapshot Apple Reminders to a daily note. Pending only, horizon 180 days
    + undated. Completed-reminders fetch is intentionally NOT included.
    """
    from rag import _apple_enabled, _fetch_reminders_due  # lazy
    from rag.cross_source_etls import _atomic_write_if_changed

    if not _apple_enabled():
        return {"ok": False, "reason": "apple_disabled"}
    now = datetime.now()
    pending = _fetch_reminders_due(now, horizon_days=180, max_items=500)
    if not pending:
        return {"ok": True, "files_written": 0, "reason": "no_data"}

    by_bucket: dict[str, list[dict]] = {}
    for item in pending:
        by_bucket.setdefault(item["bucket"], []).append(item)

    today = now.strftime("%Y-%m-%d")
    fm_lines = [
        "---",
        "source: apple-reminders",
        f"snapshot_at: {now.isoformat(timespec='seconds')}",
        f"pending_count: {len(pending)}",
        "tags:",
        "- apple-reminders",
        "- system-snapshot",
        "---",
        "",
        f"# Apple Reminders — {today}",
        "",
    ]
    body_lines: list[str] = list(fm_lines)
    for bucket_key, label in (
        ("overdue", "Overdue"),
        ("today", "Hoy"),
        ("upcoming", "Próximos"),
        ("undated", "Sin fecha"),
    ):
        items = by_bucket.get(bucket_key) or []
        if not items:
            continue
        body_lines.append(f"## {label} ({len(items)})")
        body_lines.append("")
        for it in items:
            due = it["due"] or "—"
            list_tag = f" `[{it['list']}]`" if it.get("list") else ""
            body_lines.append(f"- **{it['name']}** · {due}{list_tag}")
        body_lines.append("")
    body = "\n".join(body_lines)

    target = vault_root / _REMINDERS_VAULT_SUBPATH / f"{today}.md"
    written = _atomic_write_if_changed(target, body)
    return {
        "ok": True,
        "files_written": 1 if written else 0,
        "pending": len(pending),
        "completed": 0,
        "target": _REMINDERS_VAULT_SUBPATH,
    }
