"""WhatsApp -> vault ETL.

Reads the local WhatsApp bridge SQLite and emits deterministic per-chat,
per-month Markdown buckets at
``99-obsidian/99-AI/external-ingest/WhatsApp/<slug>/<YYYY-MM>.md``.

This used to shell out to ``~/.local/bin/whatsapp-to-vault``. The bridge
export now lives here so `rag index` and `rag start` no longer depend on the
separate TypeScript helper or the legacy ``com.fer.whatsapp-vault-sync``
launchd job.

Silent-fail contract: returns ``{ok: False, reason: "..."}`` instead of
raising. Other machines without a bridge DB simply skip this ETL.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rag._constants import _EXTERNAL_INGEST_BASE

if TYPE_CHECKING:  # pragma: no cover
    from scripts.ingest_whatsapp import WAMessage

__all__ = [
    "DEFAULT_BRIDGE_DB",
    "_WHATSAPP_ETL_RE",
    "_WHATSAPP_ETL_SCRIPT",
    "_sync_whatsapp_notes",
    "_wa_chat_slug",
]

# Deprecated compatibility exports for callers that imported these private
# names through `rag.cross_source_etls`. They are intentionally unused.
_WHATSAPP_ETL_SCRIPT = None
_WHATSAPP_ETL_RE = None

# Tests may monkeypatch this. In production, `None` means "ask the native
# WhatsApp ingester for its canonical bridge path" lazily, avoiding import
# cycles while `rag` is still initializing.
DEFAULT_BRIDGE_DB: Path | None = None

_WHATSAPP_TARGET_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/WhatsApp"


def _ingester():
    from scripts import ingest_whatsapp as _wa_ingest  # noqa: PLC0415

    return _wa_ingest


def _bridge_db_path() -> Path:
    if DEFAULT_BRIDGE_DB is not None:
        return Path(DEFAULT_BRIDGE_DB)
    return Path(_ingester().DEFAULT_BRIDGE_DB)


def _chat_label(msg: "WAMessage") -> str:
    return (msg.chat_name or msg.chat_jid.split("@", 1)[0] or "sin-nombre").strip()


def _wa_chat_slug(jid: str, label: str) -> str:
    """Slug rule shared with WhatsApp task links.

    Keep this in lockstep with `_wa_chat_month_link` so task notes resolve to
    the monthly buckets written here.
    """
    slug_src = label if any(ch.isalpha() for ch in label) else (jid.split("@", 1)[0] or "sin-nombre")
    slug = re.sub(r"[^\w\-\. ]+", "", slug_src).strip()
    slug = re.sub(r"\s+", " ", slug)[:80] or "sin-nombre"
    if slug in {".", ".."}:
        return "sin-nombre"
    return slug


def _msg_dt(msg: "WAMessage") -> datetime:
    return datetime.fromtimestamp(float(msg.timestamp))


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _yaml_scalar(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _format_message(msg: "WAMessage") -> list[str]:
    iw = _ingester()
    dt = _msg_dt(msg)
    speaker = iw._speaker_label(msg)
    content = (msg.content or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    if msg.media_type and not content.startswith("["):
        content = f"[{msg.media_type}] {content}".strip()
    parts = content.splitlines() or [""]
    lines = [f"- {dt.strftime('%H:%M')} - {speaker}: {parts[0]}"]
    lines.extend(f"  {part}" for part in parts[1:])
    return lines


def _render_bucket(slug: str, ym: str, messages: list["WAMessage"]) -> str:
    messages = sorted(messages, key=lambda m: (m.timestamp, m.id))
    labels = sorted({_chat_label(m) for m in messages if _chat_label(m)})
    jids = sorted({m.chat_jid for m in messages})
    chat_label = labels[0] if len(labels) == 1 else slug
    first_dt = _msg_dt(messages[0])
    last_dt = _msg_dt(messages[-1])

    out: list[str] = [
        "---",
        "source: whatsapp",
        "type: whatsapp-chat-month",
        f"chat: {_yaml_scalar(chat_label)}",
    ]
    if len(jids) == 1:
        out.append(f"chat_jid: {_yaml_scalar(jids[0])}")
    else:
        out.append("chat_jids:")
        out.extend(f"- {_yaml_scalar(jid)}" for jid in jids)
    out.extend([
        f"month: {_yaml_scalar(ym)}",
        f"message_count: {len(messages)}",
        f"first_ts: {_yaml_scalar(_iso(first_dt))}",
        f"last_ts: {_yaml_scalar(_iso(last_dt))}",
        "ambient: skip",
        "tags:",
        "- whatsapp",
        "---",
        "",
        f"# WhatsApp - {chat_label} - {ym}",
        "",
    ])

    last_day = ""
    for msg in messages:
        day = _msg_dt(msg).strftime("%Y-%m-%d")
        if day != last_day:
            if last_day:
                out.append("")
            out.extend([f"## {day}", ""])
            last_day = day
        out.extend(_format_message(msg))
    out.append("")
    return "\n".join(out)


def _write_if_changed(path: Path, body: str) -> bool:
    if path.is_file() and path.read_text(encoding="utf-8") == body:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return True


def _sync_whatsapp_notes(vault_root: Path) -> dict:
    """Sync WhatsApp monthly Markdown buckets into the vault."""
    bridge_db = _bridge_db_path()
    if not bridge_db.is_file():
        return {"ok": False, "reason": "bridge_db_missing"}

    try:
        messages = _ingester().read_messages(bridge_db, since_ts=0.0)
    except Exception as exc:  # pragma: no cover - defensive: corrupt DB/schema.
        return {"ok": False, "reason": str(exc)[:160]}

    target_root = Path(vault_root) / _WHATSAPP_TARGET_SUBPATH
    buckets: dict[tuple[str, str], list["WAMessage"]] = defaultdict(list)
    chats: set[str] = set()
    for msg in messages:
        label = _chat_label(msg)
        slug = _wa_chat_slug(msg.chat_jid, label)
        ym = _msg_dt(msg).strftime("%Y-%m")
        buckets[(slug, ym)].append(msg)
        chats.add(msg.chat_jid)

    files_written = 0
    files_unchanged = 0
    for (slug, ym), bucket_msgs in sorted(buckets.items()):
        body = _render_bucket(slug, ym, bucket_msgs)
        path = target_root / slug / f"{ym}.md"
        if _write_if_changed(path, body):
            files_written += 1
        else:
            files_unchanged += 1

    return {
        "ok": True,
        "files_written": files_written,
        "files_unchanged": files_unchanged,
        "buckets": len(buckets),
        "chats": len(chats),
        "target": _WHATSAPP_TARGET_SUBPATH,
    }
