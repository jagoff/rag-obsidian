"""WhatsApp tasks writer — append a section to `00-Inbox/WA-YYYY-MM-DD.md`.

Surface:

- ``_wa_chat_month_link(jid, label, ts_iso)`` — wikilink al chat-month note
  sincronizado por el ETL nativo de RAG (vive en
  `99-obsidian/99-AI/external-ingest/WhatsApp/<slug>/YYYY-MM.md`).
- ``_wa_tasks_write_note(vault, run_ts, by_chat, extractions)`` — append
  timestamped section a `00-Inbox/WA-YYYY-MM-DD.md`. Crea el archivo con
  frontmatter ``ambient: skip`` (anti-loop del WA push) en el primer write
  del día. Runs posteriores agregan bajo nuevo heading ``## HH:MM``.
- ``_wa_promises_persist(by_chat, extractions, run_ts)`` — INSERT a
  ``rag_promises`` (status='pending') las promesas extraídas en la misma
  pasada. Idempotente por (source_msg_id, promise_text). Silent-fail. La
  signal ``commitment_deadline`` (en ``rag_anticipate/signals/``) lee de
  esta tabla y emite push WA cuando una promesa está overdue / vence pronto.

Devuelve ``(path, created, new_items)``. Si todas las extractions vinieron
vacías, no escribe nada y retorna ``(path, False, 0)``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


def _wa_chat_month_link(jid: str, label: str, ts_iso: str) -> str:
    """Wikilink to the native WhatsApp chat note for the message's month.

    Falls back to just the label if the month can't be parsed. The link
    target mirrors `rag.integrations.whatsapp_etl`'s layout:
    `99-obsidian/99-AI/external-ingest/WhatsApp/<slug>/YYYY-MM.md`.
    """
    slug_src = label if any(ch.isalpha() for ch in label) else (jid.split("@")[0] or "sin-nombre")
    # Same slug rule as the native vault sync: strip non-word/dash/dot/space.
    slug = re.sub(r"[^\w\-\. ]+", "", slug_src).strip()
    slug = re.sub(r"\s+", " ", slug)[:80] or "sin-nombre"
    if slug in {".", ".."}:
        slug = "sin-nombre"
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


_DIRECTION_NORMALIZE = {"out": "outgoing", "in": "incoming"}


def _wa_promises_persist(
    by_chat: list[dict],
    extractions: list[dict],
    run_ts: datetime,
) -> int:
    """Persist `promises[]` from the combined extractor to `rag_promises`.

    Args:
        by_chat: Same ``by_chat`` que recibe ``_wa_tasks_write_note``. Cada
            entry tiene ``jid`` + ``label``. Pareado posicionalmente con
            ``extractions``.
        extractions: Output del extractor combined — cada entry trae
            ``promises: list[dict]`` con ``text``, ``when_text``, ``direction``
            ("out"/"in"), ``msg_id``, ``msg_ts``, ``speaker``.
        run_ts: Timestamp del run (se usa como ``ts`` y como anchor para
            ``_parse_promise_when`` cuando ``msg_ts`` no parsea).

    Returns:
        Cantidad de rows nuevas insertadas (0 si todas eran dupes o no había
        promesas).

    Idempotencia: dedup por ``(source_msg_id, promise_text)`` — si una
    promesa ya está pending para ese msg+text, skip (la corrida anterior
    la persistió). Permite que ``wa-tasks`` corra cada 30min sin duplicar.

    Silent-fail: cualquier error SQL → log + return 0. Nunca rompe el
    pipeline de wa-tasks (la idea es que el write a Inbox y el log a
    rag_wa_tasks sigan funcionando aunque el persist de promises falle).

    Why direction normalize: el extractor LLM emite ``"out"``/``"in"`` (el
    prompt dice "direction: out si yo prometo a otro"); la signal
    ``commitment_deadline_signal`` filtra por ``"outgoing"``/``"incoming"``
    (terminología más explícita, persiste en SQL). Mapeamos al persist
    para que el contrato del schema sea estable y la signal lea sin
    ambigüedad.
    """
    from rag import _ragvec_state_conn, _silent_log
    from rag.integrations.whatsapp import _parse_promise_when

    rows: list[tuple] = []
    for chat, ext in zip(by_chat, extractions):
        promises = (ext or {}).get("promises") or []
        if not promises:
            continue
        chat_jid = (chat.get("jid") or "").strip()
        chat_label = (chat.get("label") or "").strip()
        for p in promises:
            text = (p.get("text") or "").strip()
            if not text:
                continue
            direction = _DIRECTION_NORMALIZE.get(
                (p.get("direction") or "").strip().lower(),
                "",
            )
            if direction not in ("outgoing", "incoming"):
                continue
            when_text = (p.get("when_text") or "").strip()
            anchor_iso = (p.get("msg_ts") or "")[:19].replace(" ", "T")
            try:
                anchor_dt = datetime.fromisoformat(anchor_iso) if anchor_iso else run_ts
            except Exception:
                anchor_dt = run_ts
            try:
                due_dt, due_conf = _parse_promise_when(when_text, anchor=anchor_dt)
                due_iso = due_dt.isoformat(timespec="seconds")
            except Exception:
                due_iso = ""
                due_conf = 0.0
            extra = json.dumps(
                {
                    "when_text": when_text,
                    "speaker": (p.get("speaker") or "").strip(),
                    "chat_label": chat_label,
                },
                ensure_ascii=False,
            )
            rows.append((
                run_ts.isoformat(timespec="seconds"),
                chat_jid,
                chat_label,
                text,
                direction,
                due_iso or None,
                float(due_conf) if due_conf else None,
                str(p.get("msg_id") or ""),
                chat_jid,
                "pending",
                extra,
            ))
    if not rows:
        return 0

    inserted = 0
    try:
        with _ragvec_state_conn() as conn:
            for r in rows:
                source_msg_id = r[7]
                promise_text = r[3]
                cur = conn.execute(
                    "SELECT 1 FROM rag_promises "
                    "WHERE source_msg_id = ? AND promise_text = ? "
                    "  AND status = 'pending' "
                    "LIMIT 1",
                    (source_msg_id, promise_text),
                )
                if cur.fetchone():
                    continue
                conn.execute(
                    "INSERT INTO rag_promises "
                    "(ts, contact_jid, contact_name, promise_text, direction, "
                    " due_ts, due_confidence, source_msg_id, source_chat_jid, "
                    " status, extra_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    r,
                )
                inserted += 1
            conn.commit()
    except Exception as exc:
        try:
            _silent_log("wa_promises_persist_failed", exc)
        except Exception:
            pass
        return 0
    return inserted


__all__ = [
    "_wa_chat_month_link",
    "_wa_promises_persist",
    "_wa_tasks_write_note",
]
