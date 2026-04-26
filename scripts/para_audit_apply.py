#!/usr/bin/env python3
"""
Phase 3b: PARA-method bulk mover.

Reads approved proposals from ~/.local/share/obsidian-rag/para_audit/proposals_approved.jsonl
and physically moves notes between PARA buckets.

Dry-run is the DEFAULT. Pass --apply to mutate.
"""

import argparse
import glob
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

VAULT_DEFAULT = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
AUDIT_DIR = Path.home() / ".local/share/obsidian-rag/para_audit"
PROPOSALS_FILE = AUDIT_DIR / "proposals_approved.jsonl"
MOVES_LOG = AUDIT_DIR / "moves_applied.jsonl"

PARA_BUCKETS = {"01-Projects", "02-Areas", "03-Resources", "04-Archive"}

HARD_SKIP_SEGMENTS = {"99-obsidian-system", "99-Claude"}
HARD_SKIP_PREFIXES = ("04-Archive/99-obsidian-system/99-Claude/reviews/",)
HARD_SKIP_EXTENSIONS = {".icloud"}

OPT_OUT_ARCHIVE_NEVER = re.compile(r"^archive:\s*never\s*$", re.MULTILINE)
OPT_OUT_TYPES = {"moc", "index", "permanent", "dashboard", "morning-brief",
                 "weekly-digest", "prep", "conversation"}
OPT_OUT_TYPE_RE = re.compile(r"^type:\s*(\S+)\s*$", re.MULTILINE)
OPT_OUT_PINNED_RE = re.compile(r"^pinned:\s*true\s*$", re.MULTILINE)

FRONTMATTER_FENCE_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_icloud_placeholders(vault: Path) -> list[str]:
    pattern = str(vault / "**" / ".*.icloud")
    return glob.glob(pattern, recursive=True)


def is_hard_skip(vault_rel: str) -> str | None:
    parts = Path(vault_rel).parts
    for seg in parts:
        if seg in HARD_SKIP_SEGMENTS:
            return f"hard-skip: path contains '{seg}'"
    for prefix in HARD_SKIP_PREFIXES:
        if vault_rel.startswith(prefix):
            return f"hard-skip: under {prefix.rstrip('/')}"
    if Path(vault_rel).suffix == ".icloud":
        return "hard-skip: .icloud extension"
    return None


def is_opt_out(content: str) -> str | None:
    fm_match = FRONTMATTER_FENCE_RE.match(content)
    if not fm_match:
        return None
    fm = fm_match.group(1)
    if OPT_OUT_ARCHIVE_NEVER.search(fm):
        return "opt-out: archive: never"
    if OPT_OUT_PINNED_RE.search(fm):
        return "opt-out: pinned: true"
    type_match = OPT_OUT_TYPE_RE.search(fm)
    if type_match:
        note_type = type_match.group(1).strip().strip('"\'')
        if note_type in OPT_OUT_TYPES:
            return f"opt-out: type: {note_type}"
    return None


def stamp_frontmatter(content: str, original_path: str, ts: str) -> str:
    extra = f"para_reclassified_at: {ts}\npara_reclassified_from: {original_path}"
    fm_match = FRONTMATTER_FENCE_RE.match(content)
    if fm_match:
        fm_body = fm_match.group(1)
        after_fm = content[fm_match.end():]
        new_fm = f"---\n{fm_body}\n{extra}\n---\n"
        return new_fm + after_fm
    else:
        return f"---\n{extra}\n---\n\n{content}"


def compute_destination(vault: Path, vault_rel: str, proposed_bucket: str) -> Path:
    parts = Path(vault_rel).parts
    # parts[0] is the current bucket; subpath is everything after
    subpath = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
    return vault / proposed_bucket / subpath


def load_proposals(limit: int | None) -> list[dict]:
    if not PROPOSALS_FILE.exists():
        print(f"ERROR: proposals file not found: {PROPOSALS_FILE}", file=sys.stderr)
        sys.exit(1)
    proposals = []
    with open(PROPOSALS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            proposals.append(json.loads(line))
    if limit is not None:
        proposals = proposals[:limit]
    return proposals


def append_move_log(entry: dict) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MOVES_LOG, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3b: Apply approved PARA-method note moves."
    )
    parser.add_argument("--apply", action="store_true",
                        help="Actually move files. Without this flag, dry-run only.")
    parser.add_argument("--vault", type=Path, default=None,
                        help="Override vault path (default: OBSIDIAN_RAG_VAULT or standard location).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N proposals (for testing).")
    args = parser.parse_args()

    vault = args.vault
    if vault is None:
        env_vault = __import__("os").environ.get("OBSIDIAN_RAG_VAULT")
        vault = Path(env_vault) if env_vault else VAULT_DEFAULT
    vault = vault.expanduser().resolve()

    if not vault.is_dir():
        print(f"ERROR: vault not found: {vault}", file=sys.stderr)
        sys.exit(1)

    dry_run = not args.apply
    mode_label = "DRY-RUN" if dry_run else "APPLY"
    print(f"Mode: {mode_label}  |  Vault: {vault}")
    if dry_run:
        print("Pass --apply to actually move files.\n")

    # iCloud placeholder check
    placeholders = check_icloud_placeholders(vault)
    if placeholders:
        msg = (
            f"iCloud sync in progress — pause iCloud Drive "
            f"(System Settings > Apple ID > iCloud Drive > pause sync) and retry.\n"
            f"Found {len(placeholders)} placeholder(s): {placeholders[:3]}{'...' if len(placeholders) > 3 else ''}"
        )
        if dry_run:
            print(f"WARNING: {msg}\n")
        else:
            print(f"ABORT: {msg}", file=sys.stderr)
            sys.exit(2)

    proposals = load_proposals(args.limit)
    print(f"Proposals loaded: {len(proposals)}\n")

    counts = {
        "moved": 0,
        "would_move": 0,
        "skip_hard": 0,
        "skip_opt_out": 0,
        "skip_collision": 0,
        "skip_missing_src": 0,
        "skip_same_bucket": 0,
    }

    for prop in proposals:
        vault_rel: str = prop["path"]
        current_bucket: str = prop.get("current_bucket", "")
        proposed_bucket: str = prop["proposed_bucket"]
        confidence: float = prop.get("confidence", 0.0)
        reason: str = prop.get("reason", "")

        # Same-bucket no-op
        if current_bucket == proposed_bucket:
            print(f"[SKIP] {vault_rel} → (same bucket: {proposed_bucket})")
            counts["skip_same_bucket"] += 1
            continue

        # Hard-skip check
        hard_skip_reason = is_hard_skip(vault_rel)
        if hard_skip_reason:
            print(f"[SKIP] {vault_rel} — {hard_skip_reason}")
            counts["skip_hard"] += 1
            continue

        src = vault / vault_rel
        if not src.exists():
            print(f"[SKIP] {vault_rel} — missing source file")
            counts["skip_missing_src"] += 1
            continue

        # Read content for opt-out check + frontmatter stamping
        content = src.read_text(encoding="utf-8", errors="replace")

        opt_out_reason = is_opt_out(content)
        if opt_out_reason:
            print(f"[SKIP] {vault_rel} — {opt_out_reason}")
            counts["skip_opt_out"] += 1
            continue

        dst = compute_destination(vault, vault_rel, proposed_bucket)

        # Collision check
        if dst.exists():
            print(f"[SKIP] {vault_rel} — collision: destination exists: {dst.relative_to(vault)}",
                  file=sys.stderr)
            counts["skip_collision"] += 1
            continue

        dst_rel = str(dst.relative_to(vault))
        tag = "DRY" if dry_run else "MOVED"

        if dry_run:
            print(f"[DRY]  {vault_rel} → {dst_rel}  (conf={confidence:.2f}, reason={reason})")
            counts["would_move"] += 1
        else:
            ts = now_iso()
            stamped = stamp_frontmatter(content, vault_rel, ts)
            # Create destination directory
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Write stamped content, then move (atomic per-note: write new, remove old)
            # Using shutil.move after writing stamped content:
            # Write to a temp location in same dir, then rename
            tmp = src.with_suffix(src.suffix + ".para_tmp")
            try:
                tmp.write_text(stamped, encoding="utf-8")
                shutil.move(str(tmp), str(dst))
                src.unlink(missing_ok=True)
            except Exception as exc:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                print(f"\nERROR moving {vault_rel}: {exc}", file=sys.stderr)
                print(
                    f"Stopping. Moved so far: {counts['moved']}. "
                    f"To reverse: read {MOVES_LOG} and run `mv to_path from_path` for each entry.",
                    file=sys.stderr,
                )
                sys.exit(3)

            log_entry = {
                "ts": ts,
                "from_path": vault_rel,
                "to_path": dst_rel,
                "current_bucket": current_bucket,
                "proposed_bucket": proposed_bucket,
                "confidence": confidence,
                "reason": reason,
            }
            append_move_log(log_entry)
            print(f"[MOVED] {vault_rel} → {dst_rel}  (conf={confidence:.2f}, reason={reason})")
            counts["moved"] += 1

    print()
    print("─" * 60)
    print(f"Total proposals:   {len(proposals)}")
    if dry_run:
        print(f"Would move:        {counts['would_move']}")
    else:
        print(f"Moved:             {counts['moved']}")
    skip_total = (counts["skip_hard"] + counts["skip_opt_out"] +
                  counts["skip_collision"] + counts["skip_missing_src"] +
                  counts["skip_same_bucket"])
    print(f"Skipped total:     {skip_total}")
    if counts["skip_hard"]:
        print(f"  hard-skip:       {counts['skip_hard']}")
    if counts["skip_opt_out"]:
        print(f"  opt-out:         {counts['skip_opt_out']}")
    if counts["skip_collision"]:
        print(f"  collision:       {counts['skip_collision']}")
    if counts["skip_missing_src"]:
        print(f"  missing source:  {counts['skip_missing_src']}")
    if counts["skip_same_bucket"]:
        print(f"  same bucket:     {counts['skip_same_bucket']}")

    if not dry_run and counts["moved"] > 0:
        print(f"\nAudit log:         {MOVES_LOG}")
    print()
    print(
        "To reverse: read moves_applied.jsonl and run `mv to_path from_path` for each entry. "
        "Or grep frontmatter `para_reclassified_from` to find them."
    )


if __name__ == "__main__":
    main()
