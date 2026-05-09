"""Claude Code transcripts ETL — extracted from rag/cross_source_etls.py 2026-05-09.

Walks `~/.claude/projects/<slug>/*.jsonl` modified within the last 30 days,
redacts common secret shapes, and writes per-session markdown to the vault
under `99-obsidian/99-AI/external-ingest/Claude/<project>/<session_id>.md`
so the regular `_run_index` rglob absorbs them.

Silent-fail contract: every helper returns ``{ok: False, reason: "..."}``
instead of raising. ``_atomic_write_if_changed`` (lazy-imported from
``rag.cross_source_etls``) handles hash-skip dedup.

Tests (``tests/test_external_etls.py``) monkeypatch the constants on the
``rag`` module top-level — call sites use ``sys.modules.get("rag")`` to
re-resolve them at call time so the patch propagates here.
"""
from __future__ import annotations

import contextlib
import json
import re
import sys
import time
from pathlib import Path

from rag._constants import _EXTERNAL_INGEST_BASE

__all__ = [
    "_CLAUDE_VAULT_SUBPATH",
    "_CLAUDE_PROJECTS_ROOT",
    "_CLAUDE_INDEX_WINDOW_DAYS",
    "_CLAUDE_TURN_BODY_CAP",
    "_SECRET_PATTERNS",
    "_redact_secrets",
    "_claude_extract_turn",
    "_sync_claude_code_transcripts",
]

_CLAUDE_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/Claude"
_CLAUDE_PROJECTS_ROOT = Path.home() / ".claude/projects"
_CLAUDE_INDEX_WINDOW_DAYS = 30
_CLAUDE_TURN_BODY_CAP = 8000

_SECRET_PATTERNS = [
    (re.compile(r"sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}"), "[REDACTED-OPENAI/ANTHROPIC]"),
    (re.compile(r"ghp_[A-Za-z0-9]{30,}"),                  "[REDACTED-GH-PAT]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),          "[REDACTED-GH-PAT-NEW]"),
    (re.compile(r"AKIA[A-Z0-9]{16}"),                      "[REDACTED-AWS-KEY]"),
    (re.compile(r"AIza[A-Za-z0-9_\-]{35}"),                "[REDACTED-GOOGLE-API]"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),         "[REDACTED-SLACK]"),
    (re.compile(r"(?i)(?<![A-Za-z0-9_])(?:api[_-]?key|secret|password|token)\s*[:=]\s*[\"']?[A-Za-z0-9_\-./]{16,}"),
                                                            "[REDACTED-KV]"),
]


def _redact_secrets(text: str) -> str:
    for pat, rep in _SECRET_PATTERNS:
        text = pat.sub(rep, text)
    return text


def _claude_extract_turn(record: dict) -> tuple[str, str, str] | None:
    """Pull (role, ts, body) from one Claude Code transcript line. Returns
    None when the record is internal (tool result, summary, etc.) and
    shouldn't be rendered as a chat turn.
    """
    rec_type = record.get("type") or ""
    msg = record.get("message") or {}
    if rec_type not in ("user", "assistant"):
        return None
    role = msg.get("role") or rec_type
    ts = (record.get("timestamp") or "").replace("T", " ").split(".")[0]
    content = msg.get("content")
    if isinstance(content, str):
        body = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
            elif block.get("type") == "tool_use":
                tool = block.get("name", "?")
                parts.append(f"[tool_use:{tool}]")
            elif block.get("type") == "tool_result":
                parts.append("[tool_result]")
        body = "\n".join(parts)
    else:
        body = ""
    body = _redact_secrets(body.strip())
    _body_cap = getattr(sys.modules.get("rag"), "_CLAUDE_TURN_BODY_CAP", _CLAUDE_TURN_BODY_CAP)
    if len(body) > _body_cap:
        body = body[:_body_cap] + "\n\n[…body truncado]"
    if not body:
        return None
    return role, ts, body


def _sync_claude_code_transcripts(vault_root: Path) -> dict:
    """Convert Claude Code session JSONL → per-session markdown. Walks
    `~/.claude/projects/<slug>/*.jsonl` modified within the last 30 days,
    redacts common secret shapes, hash-skips via `_atomic_write_if_changed`.
    """
    from rag.cross_source_etls import _atomic_write_if_changed

    _rag = sys.modules.get("rag")
    _projects_root = getattr(_rag, "_CLAUDE_PROJECTS_ROOT", _CLAUDE_PROJECTS_ROOT)
    if not _projects_root.is_dir():
        return {"ok": False, "reason": "no_claude_projects_dir"}
    cutoff_mtime = time.time() - (_CLAUDE_INDEX_WINDOW_DAYS * 86400)
    written = 0
    total = 0
    skipped = 0
    for project_dir in sorted(_projects_root.iterdir()):
        if not project_dir.is_dir():
            continue
        for jsonl in sorted(project_dir.glob("*.jsonl")):
            try:
                stat = jsonl.stat()
            except OSError:
                continue
            if stat.st_mtime < cutoff_mtime:
                continue
            total += 1
            try:
                lines = jsonl.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            turns: list[tuple[str, str, str]] = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = _claude_extract_turn(rec)
                if t:
                    turns.append(t)
            if not turns:
                skipped += 1
                continue
            session_id = jsonl.stem
            started = turns[0][1] or "?"
            ended = turns[-1][1] or "?"
            fm = [
                "---",
                "source: claude-code",
                f"project: {project_dir.name}",
                f"session_id: {session_id}",
                f"started_at: {started}",
                f"ended_at: {ended}",
                f"turn_count: {len(turns)}",
                "tags:",
                "- claude-code",
                "- system-snapshot",
                "---",
                "",
                f"# Claude Code session — {project_dir.name} / {session_id}",
                "",
            ]
            for role, ts, body in turns:
                fm.append(f"## {role} · {ts}")
                fm.append("")
                fm.append(body)
                fm.append("")
            body_text = "\n".join(fm) + "\n"
            target = vault_root / _CLAUDE_VAULT_SUBPATH / project_dir.name / f"{session_id}.md"
            if _atomic_write_if_changed(target, body_text):
                written += 1
            else:
                skipped += 1

    # Prune vault transcripts whose source JSONL is gone or older than the window.
    pruned = 0
    vault_claude_dir = vault_root / _CLAUDE_VAULT_SUBPATH
    if vault_claude_dir.is_dir():
        for project_vault_dir in vault_claude_dir.iterdir():
            if not project_vault_dir.is_dir():
                continue
            source_project = _projects_root / project_vault_dir.name
            for md_file in project_vault_dir.glob("*.md"):
                session_id = md_file.stem
                source_jsonl = source_project / f"{session_id}.jsonl"
                stale = False
                if not source_jsonl.is_file():
                    stale = True
                else:
                    try:
                        if source_jsonl.stat().st_mtime < cutoff_mtime:
                            stale = True
                    except OSError:
                        stale = True
                if stale:
                    with contextlib.suppress(OSError):
                        md_file.unlink()
                        pruned += 1
            with contextlib.suppress(OSError):
                if not any(project_vault_dir.iterdir()):
                    project_vault_dir.rmdir()

    if not total and not pruned:
        return {"ok": True, "files_written": 0, "reason": "no_recent_sessions"}
    return {
        "ok": True,
        "files_written": written,
        "sessions_seen": total,
        "skipped": skipped,
        "pruned": pruned,
        "target": _CLAUDE_VAULT_SUBPATH,
    }
