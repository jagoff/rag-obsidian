"""GitHub activity ETL — extracted from rag/cross_source_etls.py 2026-05-09.

Snapshots recent GitHub activity (push/PR/issues/stars) plus open PRs via the
already-authenticated `gh` CLI and writes a daily markdown to the vault under
`99-obsidian/99-AI/external-ingest/GitHub/<YYYY-MM-DD>.md` so the regular
`_run_index` rglob absorbs it.

Silent-fail contract: every helper returns ``{ok: False, reason: "..."}``
instead of raising. ``_atomic_write_if_changed`` (lazy-imported from
``rag.cross_source_etls``) handles hash-skip dedup.

Tests (``tests/test_external_etls.py``) patch ``rag.subprocess.run`` —
module-singleton, so calls through ``subprocess.run`` here see the patch
regardless of which file the call lives in.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rag._constants import _EXTERNAL_INGEST_BASE

__all__ = [
    "_GITHUB_VAULT_SUBPATH",
    "_GH_EVENT_LABELS",
    "_gh_run",
    "_sync_github_activity",
]

_GITHUB_VAULT_SUBPATH = f"{_EXTERNAL_INGEST_BASE}/GitHub"

_GH_EVENT_LABELS = {
    "PushEvent": "push",
    "PullRequestEvent": "pull-request",
    "IssueCommentEvent": "issue-comment",
    "IssuesEvent": "issue",
    "PullRequestReviewEvent": "pr-review",
    "PullRequestReviewCommentEvent": "pr-review-comment",
    "CreateEvent": "create",
    "DeleteEvent": "delete",
    "ForkEvent": "fork",
    "WatchEvent": "star",
    "ReleaseEvent": "release",
}


def _gh_run(args: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    """Run a `gh` command. Returns (rc, stdout, stderr)."""
    try:
        proc = subprocess.run(
            ["gh", *args], capture_output=True, timeout=timeout, text=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return 127, "", str(exc)[:160]
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _sync_github_activity(vault_root: Path, hours: int = 48) -> dict:
    """Snapshot recent GitHub activity (push/PR/issues/stars) plus open PRs.
    Uses the already-authenticated `gh` CLI; silent-fail on missing/unauth.
    """
    from rag.cross_source_etls import _atomic_write_if_changed

    rc, login_out, _ = _gh_run(["api", "user", "--jq", ".login"])
    if rc != 0:
        return {"ok": False, "reason": "gh_unavailable_or_unauth"}
    user = login_out.strip()
    if not user:
        return {"ok": False, "reason": "gh_no_login"}

    rc, events_raw, err = _gh_run(["api", f"users/{user}/events?per_page=100"])
    if rc != 0:
        return {"ok": False, "reason": f"events_failed: {err[:120]}"}
    try:
        events = json.loads(events_raw)
    except json.JSONDecodeError:
        return {"ok": False, "reason": "events_parse_failed"}

    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    fresh: list[dict] = []
    for ev in events:
        ts_raw = ev.get("created_at", "")
        try:
            ts = datetime.strptime(ts_raw, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        if ts < cutoff:
            continue
        fresh.append({
            "ts": ts,
            "type": ev.get("type", "?"),
            "repo": (ev.get("repo") or {}).get("name", "?"),
            "payload": ev.get("payload") or {},
        })

    rc, pr_raw, _ = _gh_run([
        "api", "search/issues",
        "-X", "GET",
        "-f", f"q=is:pr is:open author:{user}",
    ])
    open_prs: list[dict] = []
    if rc == 0:
        try:
            for it in (json.loads(pr_raw).get("items") or [])[:20]:
                open_prs.append({
                    "title": it.get("title", "?"),
                    "url": it.get("html_url", ""),
                    "repo": (it.get("repository_url", "") or "").split("/repos/")[-1],
                    "number": it.get("number"),
                })
        except json.JSONDecodeError:
            pass

    if not fresh and not open_prs:
        return {"ok": True, "files_written": 0, "reason": "no_activity"}

    by_type: dict[str, list[dict]] = {}
    for ev in fresh:
        by_type.setdefault(ev["type"], []).append(ev)

    today = datetime.now().strftime("%Y-%m-%d")
    fm = [
        "---",
        "source: github",
        f"snapshot_date: {today}",
        f"window_hours: {hours}",
        f"event_count: {len(fresh)}",
        f"open_pr_count: {len(open_prs)}",
        "tags:",
        "- github",
        "- system-snapshot",
        "---",
        "",
        f"# GitHub activity — {today} (últimas {hours}h, usuario {user})",
        "",
    ]
    for ev_type, items in sorted(by_type.items()):
        label = _GH_EVENT_LABELS.get(ev_type, ev_type)
        fm.append(f"## {label} ({len(items)})")
        fm.append("")
        for ev in items:
            ts = ev["ts"].strftime("%Y-%m-%d %H:%M")
            repo = ev["repo"]
            p = ev["payload"]
            detail = ""
            if ev_type == "PushEvent":
                commits = p.get("commits") or []
                msgs = " · ".join((c.get("message", "").split("\n", 1)[0])[:80] for c in commits[:3])
                detail = f"{len(commits)} commit(s) {msgs}"
            elif ev_type in ("PullRequestEvent", "PullRequestReviewEvent", "PullRequestReviewCommentEvent"):
                pr = p.get("pull_request") or {}
                detail = f"{p.get('action','?')} #{pr.get('number','?')} {pr.get('title','')[:80]}"
            elif ev_type == "IssuesEvent":
                iss = p.get("issue") or {}
                detail = f"{p.get('action','?')} #{iss.get('number','?')} {iss.get('title','')[:80]}"
            elif ev_type == "IssueCommentEvent":
                iss = p.get("issue") or {}
                detail = f"comentó #{iss.get('number','?')} {iss.get('title','')[:80]}"
            elif ev_type == "WatchEvent":
                detail = "starred"
            elif ev_type == "CreateEvent":
                detail = f"creó {p.get('ref_type','?')} {p.get('ref','') or ''}"
            elif ev_type == "ReleaseEvent":
                rel = p.get("release") or {}
                detail = f"release {rel.get('tag_name','')}"
            else:
                detail = ""
            fm.append(f"- `{ts}` {repo} — {detail}")
        fm.append("")

    if open_prs:
        fm.append(f"## Open PRs ({len(open_prs)})")
        fm.append("")
        for pr in open_prs:
            fm.append(f"- {pr['repo']}#{pr['number']} [{pr['title']}]({pr['url']})")
        fm.append("")

    body = "\n".join(fm) + "\n"
    target = vault_root / _GITHUB_VAULT_SUBPATH / f"{today}.md"
    written = _atomic_write_if_changed(target, body)
    return {
        "ok": True,
        "files_written": 1 if written else 0,
        "events": len(fresh),
        "open_prs": len(open_prs),
        "target": _GITHUB_VAULT_SUBPATH,
    }
