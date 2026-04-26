#!/usr/bin/env python3
"""
Phase 1 of PARA vault audit: LLM batch classifier.

Reads ~/.local/share/obsidian-rag/para_audit/inventory.jsonl
Writes ~/.local/share/obsidian-rag/para_audit/proposals.jsonl

Only emits lines where proposed_bucket != current_bucket.
No moves are applied here — review first.
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"
OLLAMA_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 256}

PARA_BUCKETS = {"01-Projects", "02-Areas", "03-Resources", "04-Archive"}

SYSTEM_PROMPT = """You classify Obsidian notes into PARA buckets (Tiago Forte framework).

The four valid bucket values are exactly:
  01-Projects   — active, short-term effort with a goal and a finish line (e.g. "Launch v2", "Trip plan")
  02-Areas      — ongoing responsibility with no end date (e.g. "Health", "Finances", "Family")
  03-Resources  — reference material or topic of interest (e.g. "Rust", "Productivity techniques")
  04-Archive    — completed, abandoned, or no longer active item from any of the above

Decision rules:
  - If the note describes an active bounded effort with a goal → 01-Projects
  - If the note describes an ongoing role or standard to maintain → 02-Areas
  - If the note is reference/learning content with no active goal → 03-Resources
  - If the note is finished, abandoned, or outdated → 04-Archive
  - When evidence is weak, KEEP the current bucket and report confidence below 0.5

Output a single JSON object with three keys:
  proposed_bucket — one string, exactly one of the four bucket names above
  confidence      — number between 0 and 1
  reason          — short string, 80 chars max

Example output:
{"proposed_bucket": "03-Resources", "confidence": 0.82, "reason": "reference notes on Rust ownership, no active goal"}

Reply with the JSON object only. No prose before or after."""

USER_TEMPLATE = """Current bucket: {bucket}
Title: {title}
First 400 chars of body:
{body_excerpt}

Classify this note. Output JSON only."""

SKIP_REASONS = {
    "skip_zone": "is_in_skip_zone",
    "archive_optout": "has_archive_optout",
    "type_optout": "has_type_optout",
    "short_body": "body_chars < 100",
    "reviews": "bucket == 04-Archive/99-obsidian-system/99-Claude/reviews",
}


def extract_json(text: str) -> dict | None:
    """Extract first {...} block from text and parse as JSON."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first balanced {...} block
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def read_body_excerpt(path: str, max_chars: int = 400) -> str:
    """Open note, skip frontmatter (lines until second ---), return first max_chars of body."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        print(f"  [warn] cannot read {path}: {e}", file=sys.stderr)
        return ""

    body_lines = []
    in_frontmatter = False
    fm_closes = 0

    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if i == 0 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped == "---":
                fm_closes += 1
                if fm_closes >= 1:
                    in_frontmatter = False
            continue
        body_lines.append(line)

    body = "".join(body_lines)
    return body[:max_chars].strip()


def classify_note(item: dict, vault_root: Path) -> dict | None:
    """
    Call Ollama to classify a single note.
    Returns proposal dict or None on parse failure.
    """
    bucket = item["bucket"]
    title = item.get("title") or Path(item["path"]).stem
    abs_path = vault_root / item["path"]

    body_excerpt = read_body_excerpt(str(abs_path))
    user_msg = USER_TEMPLATE.format(
        bucket=bucket,
        title=title,
        body_excerpt=body_excerpt if body_excerpt else "(empty body)",
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "keep_alive": -1,
        "format": "json",
        "options": OLLAMA_OPTIONS,
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
    except Exception as e:
        print(f"  [error] HTTP call failed for {item['path']}: {e}", file=sys.stderr)
        return None

    try:
        resp_json = json.loads(raw)
        content = resp_json["message"]["content"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [error] bad Ollama response for {item['path']}: {e}", file=sys.stderr)
        return None

    parsed = extract_json(content)
    if parsed is None:
        print(
            f"  [error] JSON parse failed for {item['path']}. Response: {content[:200]!r}",
            file=sys.stderr,
        )
        return None

    proposed = parsed.get("proposed_bucket", "").strip()
    if proposed not in PARA_BUCKETS:
        print(
            f"  [error] unknown proposed_bucket {proposed!r} for {item['path']}",
            file=sys.stderr,
        )
        return None

    confidence = float(parsed.get("confidence", 0.0))
    reason = str(parsed.get("reason", ""))[:80]

    return {
        "path": item["path"],
        "current_bucket": bucket,
        "proposed_bucket": proposed,
        "confidence": round(confidence, 3),
        "reason": reason,
    }


def should_skip(item: dict) -> str | None:
    """Return skip reason string or None if note should be classified."""
    if item.get("is_in_skip_zone"):
        return "skip_zone"
    if item.get("has_archive_optout"):
        return "archive_optout"
    if item.get("has_type_optout"):
        return "type_optout"
    if (item.get("body_chars") or 0) < 100:
        return "short_body"
    if item.get("bucket") == "04-Archive/99-obsidian-system/99-Claude/reviews":
        return "reviews"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PARA audit Phase 1: classify vault notes via local LLM."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Process only the first N notes (0 = all, for testing).",
    )
    parser.add_argument(
        "--vault",
        type=str,
        default="",
        metavar="PATH",
        help="Override vault path (default: OBSIDIAN_RAG_VAULT or standard iCloud path).",
    )
    args = parser.parse_args()

    default_vault = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
    vault_root = Path(args.vault or os.environ.get("OBSIDIAN_RAG_VAULT") or default_vault).expanduser()
    if not vault_root.exists():
        print(f"[error] vault not found: {vault_root}", file=sys.stderr)
        sys.exit(1)

    state_dir = Path.home() / ".local/share/obsidian-rag/para_audit"
    inventory_path = state_dir / "inventory.jsonl"
    proposals_path = state_dir / "proposals.jsonl"

    if not inventory_path.exists():
        print(
            f"[error] inventory not found: {inventory_path}\n"
            "Run scripts/para_audit_inventory.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    state_dir.mkdir(parents=True, exist_ok=True)

    # Load inventory
    items: list[dict] = []
    with open(inventory_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] bad inventory line: {e}", file=sys.stderr)

    if args.limit > 0:
        items = items[: args.limit]

    total = len(items)
    print(f"Inventory loaded: {total} notes. Output → {proposals_path}")
    print(f"Model: {MODEL} | Ollama: {OLLAMA_URL}")
    print()

    # Track skip counts
    skip_counts: dict[str, int] = {k: 0 for k in SKIP_REASONS}
    proposals_written = 0

    # Transition tracking: "01-Projects → 04-Archive" → count
    transitions: dict[str, int] = {}
    confidences: list[float] = []
    high_conf_count = 0

    # Existing proposals to allow resuming (skip already-processed paths)
    seen_paths: set[str] = set()
    if proposals_path.exists():
        with open(proposals_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    seen_paths.add(row["path"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if seen_paths:
            print(f"Resuming: {len(seen_paths)} already-processed paths will be skipped.")

    processed = 0
    skipped_total = 0

    with open(proposals_path, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(items):
            # Skip already processed (resume support)
            if item.get("path") in seen_paths:
                processed += 1
                skipped_total += 1
                skip_counts.setdefault("already_done", 0)
                skip_counts["already_done"] = skip_counts.get("already_done", 0) + 1
                continue

            # Progress every 50 notes
            if idx > 0 and idx % 50 == 0:
                print(
                    f"[{idx}/{total}] processed={processed} proposed={proposals_written} "
                    f"skipped={skipped_total}"
                )

            skip_reason = should_skip(item)
            if skip_reason:
                skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1
                skipped_total += 1
                processed += 1
                continue

            proposal = classify_note(item, vault_root)
            processed += 1

            if proposal is None:
                # parse error — already logged to stderr
                skipped_total += 1
                skip_counts["parse_error"] = skip_counts.get("parse_error", 0) + 1
                continue

            # Only emit real proposals (bucket change)
            if proposal["proposed_bucket"] != proposal["current_bucket"]:
                out_f.write(json.dumps(proposal, ensure_ascii=False) + "\n")
                out_f.flush()
                proposals_written += 1

                key = f"{proposal['current_bucket']} → {proposal['proposed_bucket']}"
                transitions[key] = transitions.get(key, 0) + 1
                confidences.append(proposal["confidence"])
                if proposal["confidence"] >= 0.7:
                    high_conf_count += 1

    # Final progress line
    print(f"[{total}/{total}] processed={processed} proposed={proposals_written} skipped={skipped_total}")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total notes in inventory : {total}")
    print(f"Total processed          : {processed}")
    print(f"Total proposals emitted  : {proposals_written}")
    print(f"Total skipped            : {skipped_total}")
    print()
    print("Skip breakdown:")
    skip_labels = {
        "skip_zone": "  skip zone (Claude memory/system files)",
        "archive_optout": "  has archive:never",
        "type_optout": "  excluded type (moc/dashboard/etc.)",
        "short_body": "  body < 100 chars",
        "reviews": "  bucket 04-Archive/99-obsidian-system/99-Claude/reviews (auto-generated)",
        "parse_error": "  LLM parse error",
        "already_done": "  already processed (resume)",
    }
    for reason, count in sorted(skip_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            label = skip_labels.get(reason, f"  {reason}")
            print(f"{label}: {count}")
    print()

    if proposals_written > 0:
        print("Proposed transitions:")
        for key, count in sorted(transitions.items(), key=lambda x: -x[1]):
            print(f"  {key}: {count}")
        print()
        mean_conf = sum(confidences) / len(confidences)
        print(f"Mean confidence         : {mean_conf:.3f}")
        print(f"High confidence (≥0.7)  : {high_conf_count} / {proposals_written}")
    else:
        print("No bucket changes proposed.")

    print()
    print(f"Proposals written to: {proposals_path}")


if __name__ == "__main__":
    main()
