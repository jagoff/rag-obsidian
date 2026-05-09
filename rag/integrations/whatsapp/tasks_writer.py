"""WhatsApp tasks writer — append a section to `00-Inbox/WA-YYYY-MM-DD.md`.

Surface:

- ``_wa_chat_month_link(jid, label, ts_iso)`` — wikilink al chat-month note
  sincronizado por `whatsapp-to-vault` (vive en
  `99-obsidian/99-AI/external-ingest/WhatsApp/<slug>/YYYY-MM.md`).
- ``_wa_tasks_write_note(vault, run_ts, by_chat, extractions)`` — append
  timestamped section a `00-Inbox/WA-YYYY-MM-DD.md`. Crea el archivo con
  frontmatter ``ambient: skip`` (anti-loop del WA push) en el primer write
  del día. Runs posteriores agregan bajo nuevo heading ``## HH:MM``.

Devuelve ``(path, created, new_items)``. Si todas las extractions vinieron
vacías, no escribe nada y retorna ``(path, False, 0)``.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def _wa_chat_month_link(jid: str, label: str, ts_iso: str) -> str:
    """Wikilink to the vault-sync'd chat note for the message's month.

    Falls back to just the label if the month can't be parsed. The link
    target mirrors `whatsapp-to-vault`'s layout:
    `99-obsidian/99-AI/external-ingest/WhatsApp/<slug>/YYYY-MM.md`.
    """
    slug_src = label if any(ch.isalpha() for ch in label) else (jid.split("@")[0] or "sin-nombre")
    # Same slug rule as vault-sync: strip non-word/dash/dot/space.
    slug = re.sub(r"[^\w\-\. ]+", "", slug_src).strip()
    slug = re.sub(r"\s+", " ", slug)[:80] or "sin-nombre"
    try:
        dt = datetime.fromisoformat(ts_iso[:19].replace(" ", "T"))
        ym = dt.strftime("%Y-%m")
    except Exception:
        return f"[[{label}]]"
    return f"[[99-obsidian/99-AI/external-ingest/WhatsApp/{slug}/{ym}|{label}]]"


def _wa_tasks_write_note(
    vault: Path,
    run_ts: datetime,
    by_chat: list[dict],
    extractions: list[dict],
) -> tuple[Path, bool, int]:
    """Append a timestamped section to `00-Inbox/WA-YYYY-MM-DD.md`.

    Creates the file with frontmatter on first write of the day. Later
    runs append under a new `## HH:MM` heading so the same-day history is
    preserved. Returns ``(path, created, new_items)``. If every extraction
    came back empty, writes nothing and returns `(path, False, 0)`.
    """
    from rag import INBOX_FOLDER
    # Re-resolve via package namespace so monkeypatches en
    # `_wa_chat_month_link` propagan al call site.
    from rag.integrations.whatsapp import _wa_chat_month_link
    total_items = sum(
        len(e["tasks"]) + len(e["questions"]) + len(e["commitments"])
        for e in extractions
    )
    date_str = run_ts.strftime("%Y-%m-%d")
    note_path = vault / INBOX_FOLDER / f"WA-{date_str}.md"
    if total_items == 0:
        return note_path, False, 0

    lines: list[str] = []
    section = f"## {run_ts.strftime('%H:%M')} — {sum(1 for e in extractions if any(e[k] for k in ('tasks','questions','commitments')))} chats\n"
    lines.append(section)
    for chat, ext in zip(by_chat, extractions):
        if not any(ext[k] for k in ("tasks", "questions", "commitments")):
            continue
        first_new_ts = next(
            (m["ts"] for m in chat["messages"] if m["new"] and not m["is_from_me"]),
            chat["messages"][-1]["ts"] if chat["messages"] else "",
        )
        link = _wa_chat_month_link(chat["jid"], chat["label"], first_new_ts)
        lines.append(f"### {link}\n")
        for t in ext["tasks"]:
            lines.append(f"- [ ] {t}")
        for q in ext["questions"]:
            lines.append(f"- ❓ {q}")
        for c in ext["commitments"]:
            lines.append(f"- 📌 {c}")
        lines.append("")

    note_path.parent.mkdir(parents=True, exist_ok=True)
    created = not note_path.exists()
    if created:
        header = [
            "---",
            "source: whatsapp",
            "type: wa-tasks",
            f"date: {date_str}",
            "ambient: skip",
            "tags:",
            "- whatsapp",
            "- tasks/wa",
            "---",
            "",
            f"# WhatsApp — tareas {date_str}",
            "",
        ]
        body = "\n".join(header + lines) + "\n"
        note_path.write_text(body, encoding="utf-8")
    else:
        existing = note_path.read_text(encoding="utf-8")
        if not existing.endswith("\n"):
            existing += "\n"
        note_path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")
    return note_path, created, total_items


__all__ = [
    "_wa_chat_month_link",
    "_wa_tasks_write_note",
]
