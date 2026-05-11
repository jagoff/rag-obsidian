"""Claude.ai web conversations ETL.

Parses the ZIP that Claude.ai produces via Settings → Privacy → Export
account data (``conversations.json``) and writes per-conversation markdown
to ``99-obsidian/99-AI/external-ingest/Claude-Web/<title-slug>-<uuid8>.md``
so the regular ``_run_index`` rglob absorbs them.

User flow (manual, opt-in):
  1. Request export at https://claude.ai/settings/data-privacy-controls.
  2. Anthropic emails a ZIP within ~24h.
  3. User drops the ZIP into ``~/.claude-ai-export/`` (or sets
     ``RAG_CLAUDE_WEB_EXPORT_DIR`` to point elsewhere).
  4. Next ``rag index`` (or daemon tick) picks up the newest ZIP, parses
     conversations.json, redacts secrets, writes one .md per conversation.

Schema tolerance: supports both ``message["text"]`` (older export format)
and ``message["content"]`` block list (newer format mirroring the API).

Silent-fail contract: every helper returns ``{ok: False, reason: "..."}``
instead of raising. ``_atomic_write_if_changed`` (lazy-imported from
``rag.cross_source_etls``) handles hash-skip dedup. ``_redact_secrets``
re-used from ``rag.integrations.claude_code`` so secret patterns stay in
a single place.

Tests (``tests/test_external_etls.py``) monkeypatch constants on the
``rag`` module top-level — call sites use ``sys.modules.get("rag")`` to
re-resolve them at call time so the patch propagates here.
"""
from __future__ import annotations

import contextlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path

from rag._constants import _EXTERNAL_INGEST_BASE
from rag.integrations.claude_code import _redact_secrets

__all__ = [
    "_CLAUDE_WEB_VAULT_SUBPATH",
    "_CLAUDE_WEB_EXPORT_DIR",
    "_CLAUDE_WEB_INDEX_WINDOW_DAYS",
    "_CLAUDE_WEB_TURN_BODY_CAP",
    "_claude_web_extract_turn",
    "_claude_web_slug",
    "_claude_web_pick_export",
    "_claude_web_load_conversations",
    "_sync_claude_web_conversations",
]

_CLAUDE_WEB_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/Claude-Web"
_CLAUDE_WEB_EXPORT_DIR = Path.home() / ".claude-ai-export"
_CLAUDE_WEB_INDEX_WINDOW_DAYS = 365
_CLAUDE_WEB_TURN_BODY_CAP = 8000

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def _claude_web_slug(title: str, uuid: str) -> str:
    """Filesystem-safe slug ``<title>-<uuid8>``. Falls back to the uuid
    alone when the title is empty / unicode-only. uuid8 disambiguates
    conversations that share a title.
    """
    base = _SLUG_RE.sub("-", (title or "").strip()).strip("-").lower()
    if len(base) > 60:
        base = base[:60].rstrip("-")
    uuid_short = (uuid or "").replace("-", "")[:8] or "unk"
    return f"{base}-{uuid_short}" if base else f"untitled-{uuid_short}"


def _claude_web_extract_turn(msg: dict) -> tuple[str, str, str] | None:
    """Pull (role, ts, body) from one chat_messages entry. Returns None
    when the message has no renderable body. Tolerant to both legacy
    ``text`` field and newer ``content`` block list (text + tool_use).
    """
    sender = (msg.get("sender") or "").lower()
    if sender == "human":
        role = "user"
    elif sender == "assistant":
        role = "assistant"
    else:
        return None
    ts = (msg.get("created_at") or msg.get("updated_at") or "").replace("T", " ").split(".")[0]

    body_parts: list[str] = []
    text_field = msg.get("text")
    if isinstance(text_field, str) and text_field.strip():
        body_parts.append(text_field)
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and block.get("text"):
                if block["text"] not in body_parts:
                    body_parts.append(block["text"])
            elif btype == "tool_use":
                tool = block.get("name", "?")
                body_parts.append(f"[tool_use:{tool}]")
            elif btype == "tool_result":
                body_parts.append("[tool_result]")
            elif btype == "thinking" and block.get("thinking"):
                body_parts.append(f"[thinking]\n{block['thinking']}")
    # Attachments / files surface as references so the user knows context
    # was passed in (helpful for retrieval — e.g. "le pasé el PDF X").
    for ref in (msg.get("attachments") or []):
        name = ref.get("file_name") or ref.get("name") or ""
        if name:
            body_parts.append(f"[attachment: {name}]")
    for ref in (msg.get("files") or []):
        name = ref.get("file_name") or ref.get("name") or ""
        if name:
            body_parts.append(f"[file: {name}]")

    body = "\n\n".join(p for p in body_parts if p).strip()
    body = _redact_secrets(body)
    _body_cap = getattr(sys.modules.get("rag"), "_CLAUDE_WEB_TURN_BODY_CAP", _CLAUDE_WEB_TURN_BODY_CAP)
    if len(body) > _body_cap:
        body = body[:_body_cap] + "\n\n[…body truncado]"
    if not body:
        return None
    return role, ts, body


def _claude_web_pick_export(export_dir: Path) -> Path | None:
    """Return the most recent export source under ``export_dir``: either a
    ``.zip`` file or an already-extracted directory containing
    ``conversations.json``. Returns None if nothing usable found.

    Prefers .zip over extracted dir when both exist with the same mtime
    base — extracting the user's freshly dropped ZIP is always the
    intended action.
    """
    if not export_dir.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for entry in export_dir.iterdir():
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if entry.is_file() and entry.suffix.lower() == ".zip":
            candidates.append((mtime, entry))
        elif entry.is_dir() and (entry / "conversations.json").is_file():
            candidates.append((mtime, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _claude_web_load_conversations(source: Path) -> list[dict] | None:
    """Read ``conversations.json`` from a .zip or extracted directory.
    Returns None on any read failure (silent-fail).
    """
    try:
        if source.is_file() and source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as zf:
                target_name = None
                for name in zf.namelist():
                    if name.endswith("conversations.json"):
                        target_name = name
                        break
                if not target_name:
                    return None
                with zf.open(target_name) as fh:
                    data = json.load(fh)
        elif source.is_dir():
            with (source / "conversations.json").open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            return None
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    return data


def _sync_claude_web_conversations(vault_root: Path) -> dict:
    """Materialize claude.ai web conversations from the latest export ZIP
    (or extracted dir) under ``~/.claude-ai-export/`` as per-conversation
    markdown so ``_run_index`` absorbs them.

    Window: only conversations whose ``updated_at`` falls within
    ``_CLAUDE_WEB_INDEX_WINDOW_DAYS`` (default 365 — exports are manual
    and infrequent, so we keep a wide window).
    """
    from rag.cross_source_etls import _atomic_write_if_changed

    _rag = sys.modules.get("rag")
    _export_dir = getattr(_rag, "_CLAUDE_WEB_EXPORT_DIR", _CLAUDE_WEB_EXPORT_DIR)
    if not _export_dir.is_dir():
        return {"ok": False, "reason": "no_claude_web_export_dir"}

    source = _claude_web_pick_export(_export_dir)
    if source is None:
        return {"ok": False, "reason": "no_export_found"}

    conversations = _claude_web_load_conversations(source)
    if conversations is None:
        return {"ok": False, "reason": "export_unreadable"}

    cutoff_epoch = time.time() - (_CLAUDE_WEB_INDEX_WINDOW_DAYS * 86400)
    written = 0
    skipped = 0
    total = 0
    empty = 0

    for conv in conversations:
        if not isinstance(conv, dict):
            continue
        total += 1
        uuid = conv.get("uuid") or ""
        name = conv.get("name") or ""
        created_at = (conv.get("created_at") or "").split(".")[0].replace("T", " ")
        updated_at = (conv.get("updated_at") or "").split(".")[0].replace("T", " ")

        # Window filter — parse updated_at if present, else accept.
        if updated_at:
            try:
                # ISO 8601 with optional TZ — strip Z, treat as UTC.
                upd_iso = updated_at.replace("Z", "").strip()
                # Two formats observed: "YYYY-MM-DD HH:MM:SS" and "YYYY-MM-DDTHH:MM:SS"
                upd_iso = upd_iso.replace("T", " ")
                upd_epoch = time.mktime(time.strptime(upd_iso[:19], "%Y-%m-%d %H:%M:%S"))
            except (ValueError, OverflowError):
                upd_epoch = None
            if upd_epoch is not None and upd_epoch < cutoff_epoch:
                continue

        msgs = conv.get("chat_messages") or []
        if not isinstance(msgs, list) or not msgs:
            empty += 1
            continue

        turns: list[tuple[str, str, str]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            t = _claude_web_extract_turn(m)
            if t:
                turns.append(t)
        if not turns:
            empty += 1
            continue

        slug = _claude_web_slug(name, uuid)
        fm = [
            "---",
            "source: claude-web",
            f"conversation_uuid: {uuid}",
            f"title: {json.dumps(name, ensure_ascii=False)}",
            f"created_at: {created_at}",
            f"updated_at: {updated_at}",
            f"turn_count: {len(turns)}",
            "tags:",
            "- claude-web",
            "- system-snapshot",
            "---",
            "",
            f"# {name or '(sin título)'}",
            "",
        ]
        for role, ts, body in turns:
            fm.append(f"## {role} · {ts}")
            fm.append("")
            fm.append(body)
            fm.append("")
        body_text = "\n".join(fm) + "\n"
        target = vault_root / _CLAUDE_WEB_VAULT_SUBPATH / f"{slug}.md"
        if _atomic_write_if_changed(target, body_text):
            written += 1
        else:
            skipped += 1

    # Prune vault notes whose conversation UUID isn't in the current export
    # (covers manual deletions on claude.ai web). We only prune if the
    # export covered ≥10 conversations — otherwise a malformed/empty
    # export wouldn't wipe the vault.
    pruned = 0
    vault_dir = vault_root / _CLAUDE_WEB_VAULT_SUBPATH
    if vault_dir.is_dir() and total >= 10:
        current_uuids = {(c.get("uuid") or "").replace("-", "")[:8] for c in conversations if isinstance(c, dict)}
        for md_file in vault_dir.glob("*.md"):
            # slug pattern: ``<title>-<uuid8>.md`` — extract trailing 8-hex
            stem = md_file.stem
            tail = stem.rsplit("-", 1)[-1]
            if len(tail) == 8 and tail not in current_uuids:
                with contextlib.suppress(OSError):
                    md_file.unlink()
                    pruned += 1

    return {
        "ok": True,
        "files_written": written,
        "conversations_seen": total,
        "skipped": skipped,
        "empty": empty,
        "pruned": pruned,
        "source": str(source.name),
        "target": _CLAUDE_WEB_VAULT_SUBPATH,
    }
