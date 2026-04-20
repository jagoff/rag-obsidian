#!/usr/bin/env python3
"""
Phase 0: PARA-method vault inventory scanner.

Walks 01-Projects, 02-Areas, 03-Resources, 04-Archive and emits one JSON
object per line to ~/.local/share/obsidian-rag/para_audit/inventory.jsonl.
Stdlib-only — no venv required.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

VAULT_DEFAULT = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
OUTPUT_DIR = Path.home() / ".local/share/obsidian-rag/para_audit"
OUTPUT_FILE = OUTPUT_DIR / "inventory.jsonl"

PARA_BUCKETS = ["01-Projects", "02-Areas", "03-Resources", "04-Archive"]

# Hard-skip patterns (vault-relative path segments)
CLAUDE_SKIP_PREFIX = "04-Archive/99-obsidian-system/99-Claude"

# Root-level 00-Inbox Claude memory filenames (but we skip 00-Inbox entirely anyway)
INBOX_SKIP_PATTERNS = re.compile(
    r"^(feedback_|reference_|project_|user_|MEMORY\.md)"
)

# Wikilink regex
WIKILINK_RE = re.compile(r"\[\[([^\[\]\|]+)(?:\|[^\[\]]+)?\]\]")

# Frontmatter boundary
FM_START = re.compile(r"^---[ \t]*\r?\n")
FM_END = re.compile(r"\n---[ \t]*\r?\n")

# Type opt-out values
TYPE_OPTOUTS = {"moc", "index", "permanent", "dashboard", "morning-brief",
                "weekly-digest", "prep", "conversation"}


def parse_frontmatter(text: str):
    """Return (keys_list, body_text). Naive key-only YAML parse."""
    if not FM_START.match(text):
        return [], text

    end_match = FM_END.search(text, 4)
    if not end_match:
        return [], text

    fm_block = text[4:end_match.start()]
    body = text[end_match.end():]

    keys = []
    fm_data = {}
    for line in fm_block.splitlines():
        m = re.match(r"^([A-Za-z_][\w\-]*)[ \t]*:[ \t]*(.*)", line)
        if m:
            key = m.group(1)
            val = m.group(2).strip()
            keys.append(key)
            fm_data[key] = val

    return keys, body, fm_data


def is_skip_zone(rel_path: str) -> bool:
    return rel_path.startswith(CLAUDE_SKIP_PREFIX)


def file_record(vault: Path, md_path: Path) -> dict:
    rel = md_path.relative_to(vault)
    rel_str = str(rel)

    parts = rel.parts
    bucket = parts[0]
    subfolder = str(Path(*parts[1:-1])) if len(parts) > 2 else ""

    text = md_path.read_text(encoding="utf-8", errors="replace")
    size_chars = len(text)

    result = parse_frontmatter(text)
    if len(result) == 3:
        fm_keys, body, fm_data = result
    else:
        fm_keys, body = result
        fm_data = {}

    body_chars = len(body)
    title = fm_data.get("title") or md_path.stem

    has_archive_optout = fm_data.get("archive", "").strip().lower() == "never"
    type_val = fm_data.get("type", "").strip().lower()
    has_type_optout = type_val in TYPE_OPTOUTS

    outlinks_count = len(WIKILINK_RE.findall(body))

    stat = md_path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    btime_raw = getattr(stat, "st_birthtime", stat.st_mtime)
    ctime = datetime.fromtimestamp(btime_raw, tz=timezone.utc).isoformat()

    return {
        "path": rel_str,
        "bucket": bucket,
        "subfolder": subfolder,
        "title": title,
        "size_chars": size_chars,
        "body_chars": body_chars,
        "frontmatter_keys": fm_keys,
        "has_archive_optout": has_archive_optout,
        "has_type_optout": has_type_optout,
        "is_in_skip_zone": is_skip_zone(rel_str),
        "outlinks_count": outlinks_count,
        "modified_ts": mtime,
        "created_ts": ctime,
    }


def main():
    parser = argparse.ArgumentParser(description="PARA vault inventory scanner")
    parser.add_argument("--vault", type=Path, default=None,
                        help="Override vault path (default: OBSIDIAN_RAG_VAULT or standard path)")
    args = parser.parse_args()

    vault_env = os.environ.get("OBSIDIAN_RAG_VAULT")
    if args.vault:
        vault = args.vault.expanduser().resolve()
    elif vault_env:
        vault = Path(vault_env).expanduser().resolve()
    else:
        vault = VAULT_DEFAULT.resolve()

    if not vault.is_dir():
        print(f"ERROR: vault not found at {vault}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0

    for bucket_name in PARA_BUCKETS:
        bucket_dir = vault / bucket_name
        if not bucket_dir.is_dir():
            continue
        for md_path in sorted(bucket_dir.rglob("*.md")):
            try:
                rec = file_record(vault, md_path)
            except Exception as exc:
                print(f"WARN: skipping {md_path}: {exc}", file=sys.stderr)
                skipped += 1
                continue

            if rec["is_in_skip_zone"]:
                skipped += 1
                continue

            records.append(rec)

    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    bucket_counts: Counter = Counter()
    archive_optout = 0
    type_optout = 0
    total_outlinks = 0

    for rec in records:
        bucket_counts[rec["bucket"]] += 1
        if rec["has_archive_optout"]:
            archive_optout += 1
        if rec["has_type_optout"]:
            type_optout += 1
        total_outlinks += rec["outlinks_count"]

    total = len(records)
    mean_links = total_outlinks / total if total else 0.0

    print(f"Vault: {vault}")
    print(f"Output: {OUTPUT_FILE}")
    print("")
    print(f"{'Bucket':<20} {'Notes':>6}")
    print(f"{'-'*28}")
    for bucket in PARA_BUCKETS:
        print(f"  {bucket:<18} {bucket_counts[bucket]:>6}")
    print(f"{'-'*28}")
    print(f"  {'TOTAL':<18} {total:>6}")
    print("")
    print(f"  Archive opt-out (archive: never): {archive_optout}")
    print(f"  Type opt-out (moc/index/etc):     {type_optout}")
    print(f"  Skipped (skip-zone + errors):     {skipped}")
    print(f"  Mean wikilinks per note:          {mean_links:.2f}")


if __name__ == "__main__":
    main()
