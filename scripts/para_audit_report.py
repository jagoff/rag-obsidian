#!/usr/bin/env python3
"""
Phase 5: PARA-method audit final report.

Aggregates the audit log files produced by Phases 0-4 and prints a human
report (or JSON via --json) summarizing what changed for the user.

No LLM calls. Stdlib only. Read-only — never mutates the vault.

Inputs (under ~/.local/share/obsidian-rag/):
  para_audit/inventory.jsonl           Phase 0 baseline snapshot
  para_audit/proposals.jsonl           Phase 1 LLM proposals
  para_audit/proposals_approved.jsonl  Phase 2 confidence-filtered proposals
  para_audit/moves_applied.jsonl       Phase 3b actually-moved notes
  filing_batches/archive-*.jsonl       Phase 3a archive wave audit
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

VAULT_DEFAULT = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
STATE_DIR = Path.home() / ".local/share/obsidian-rag"
AUDIT_DIR = STATE_DIR / "para_audit"
FILING_DIR = STATE_DIR / "filing_batches"

INVENTORY_PATH = AUDIT_DIR / "inventory.jsonl"
PROPOSALS_PATH = AUDIT_DIR / "proposals.jsonl"
APPROVED_PATH = AUDIT_DIR / "proposals_approved.jsonl"
MOVES_PATH = AUDIT_DIR / "moves_applied.jsonl"
ARCHIVE_GLOB = str(FILING_DIR / "archive-*.jsonl")

PARA_BUCKETS = ["01-Projects", "02-Areas", "03-Resources", "04-Archive"]
RECENT_WINDOW_DAYS = 7

WIKILINK_RE = re.compile(r"\[\[([^\[\]\|#]+)(?:#[^\[\]\|]*)?(?:\|[^\[\]]+)?\]\]")
FM_START = re.compile(r"^---[ \t]*\r?\n")
FM_END = re.compile(r"\n---[ \t]*\r?\n")
FM_RECLASSIFIED_KEY = "para_reclassified_at"


# --------------------------------------------------------------------------- #
# ANSI helpers
# --------------------------------------------------------------------------- #

class Style:
    """ANSI styles. `enabled` toggled by --no-color or non-TTY."""
    enabled: bool = True

    @classmethod
    def wrap(cls, code: str, text: str) -> str:
        if not cls.enabled:
            return text
        return f"\x1b[{code}m{text}\x1b[0m"

    @classmethod
    def bold(cls, t: str) -> str:    return cls.wrap("1", t)
    @classmethod
    def dim(cls, t: str) -> str:     return cls.wrap("2", t)
    @classmethod
    def red(cls, t: str) -> str:     return cls.wrap("31", t)
    @classmethod
    def green(cls, t: str) -> str:   return cls.wrap("32", t)
    @classmethod
    def yellow(cls, t: str) -> str:  return cls.wrap("33", t)
    @classmethod
    def blue(cls, t: str) -> str:    return cls.wrap("34", t)
    @classmethod
    def magenta(cls, t: str) -> str: return cls.wrap("35", t)
    @classmethod
    def cyan(cls, t: str) -> str:    return cls.wrap("36", t)


def osc8(target_path: Path, label: str) -> str:
    """OSC 8 hyperlink to a local file. Falls back to plain label if disabled."""
    if not Style.enabled:
        return label
    uri = "file://" + str(target_path)
    return f"\x1b]8;;{uri}\x1b\\{label}\x1b]8;;\x1b\\"


def section_header(title: str) -> str:
    return Style.bold(Style.cyan(f"━━ {title} "))


def signed(n: int) -> str:
    if n > 0:
        return Style.green(f"+{n}")
    if n < 0:
        return Style.red(str(n))
    return Style.dim("0")


# --------------------------------------------------------------------------- #
# JSONL loading
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path) -> list[dict] | None:
    """Return list of parsed JSON objects, or None if the file is missing."""
    if not path.exists():
        return None
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate garbage; just skip the row
                continue
    return rows


def load_archive_batches(window_days: int) -> tuple[list[dict] | None, list[Path]]:
    """Glob archive-*.jsonl, keep files mtime within window. Returns (rows_flat, files_used)."""
    matches = sorted(glob.glob(ARCHIVE_GLOB))
    if not matches:
        return (None, [])
    cutoff = datetime.now().timestamp() - window_days * 86400
    used: list[Path] = []
    rows: list[dict] = []
    for m in matches:
        p = Path(m)
        try:
            if p.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        loaded = load_jsonl(p)
        if not loaded:
            continue
        used.append(p)
        rows.extend(loaded)
    return (rows, used)


# --------------------------------------------------------------------------- #
# Vault walk (read-only)
# --------------------------------------------------------------------------- #

def walk_vault_now(vault: Path) -> tuple[Counter, dict[str, list[Path]]]:
    """Return (bucket counts, title -> [absolute paths])."""
    bucket_counts: Counter = Counter()
    title_index: dict[str, list[Path]] = defaultdict(list)

    for bucket in PARA_BUCKETS:
        root = vault / bucket
        if not root.exists():
            continue
        for path in root.rglob("*.md"):
            # skip dot-prefixed segments
            rel = path.relative_to(vault)
            if any(part.startswith(".") for part in rel.parts):
                continue
            bucket_counts[bucket] += 1
            title_index[path.stem.lower()].append(path)
    return bucket_counts, title_index


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Naive YAML key:value parse. Returns ({key: raw_value}, body_text)."""
    if not FM_START.match(text):
        return {}, text
    end_match = FM_END.search(text, 4)
    if not end_match:
        return {}, text
    fm_block = text[4:end_match.start()]
    body = text[end_match.end():]
    data: dict[str, str] = {}
    for line in fm_block.splitlines():
        m = re.match(r"^([A-Za-z_][\w\-]*)[ \t]*:[ \t]*(.*)", line)
        if m:
            data[m.group(1)] = m.group(2).strip()
    return data, body


def extract_wikilink_targets(body: str) -> set[str]:
    """Return lowercased link targets (no anchor, no alias)."""
    return {m.group(1).strip().lower() for m in WIKILINK_RE.finditer(body)}


# --------------------------------------------------------------------------- #
# Section computations
# --------------------------------------------------------------------------- #

def compute_bucket_deltas(inventory: list[dict] | None,
                          bucket_counts_now: Counter) -> dict:
    """Before/after per bucket from inventory + live walk."""
    if inventory is None:
        return {"skipped": True}
    before: Counter = Counter()
    for row in inventory:
        b = row.get("bucket")
        if b in PARA_BUCKETS:
            before[b] += 1
    deltas = {}
    for b in PARA_BUCKETS:
        a = bucket_counts_now.get(b, 0)
        bc = before.get(b, 0)
        deltas[b] = {"before": bc, "after": a, "delta": a - bc}
    return {"skipped": False, "buckets": deltas,
            "total_before": sum(before.values()),
            "total_after": sum(bucket_counts_now.values())}


def compute_top_transitions(moves: list[dict] | None, top_n: int = 10) -> dict:
    if moves is None:
        return {"skipped": True}
    transitions: Counter = Counter()
    for row in moves:
        src = row.get("current_bucket") or "?"
        dst = row.get("proposed_bucket") or "?"
        transitions[(src, dst)] += 1
    top = transitions.most_common(top_n)
    return {"skipped": False, "total_moves": sum(transitions.values()),
            "transitions": [{"from": s, "to": d, "count": c} for (s, d), c in top]}


def compute_archive_summary(rows: list[dict] | None,
                            files_used: list[Path]) -> dict:
    if rows is None:
        return {"skipped": True}
    return {"skipped": False,
            "total_archived": len(rows),
            "batch_files": [str(p) for p in files_used]}


def compute_skipped_proposals(proposals: list[dict] | None,
                              approved: list[dict] | None,
                              inventory: list[dict] | None) -> dict:
    """Count proposals that did NOT make it to approved + classify why."""
    if proposals is None:
        return {"skipped": True}

    approved_paths = set()
    if approved is not None:
        approved_paths = {row.get("path") for row in approved if row.get("path")}

    inv_by_path: dict[str, dict] = {}
    if inventory is not None:
        inv_by_path = {row.get("path"): row for row in inventory if row.get("path")}

    breakdown: Counter = Counter()
    total_skipped = 0
    for row in proposals:
        path = row.get("path")
        if not path:
            continue
        if path in approved_paths:
            continue
        total_skipped += 1
        reason = "low_confidence"
        inv = inv_by_path.get(path)
        if inv:
            if inv.get("is_in_skip_zone"):
                reason = "hard_skip_zone"
            elif inv.get("has_archive_optout"):
                reason = "frontmatter_archive_optout"
            elif inv.get("has_type_optout"):
                reason = "frontmatter_type_optout"
        # detect explicit reason on the proposal row itself
        if row.get("skip_reason"):
            reason = row.get("skip_reason")
        breakdown[reason] += 1
    return {"skipped": False,
            "total_skipped": total_skipped,
            "breakdown": dict(breakdown)}


def compute_wikilink_integrity(moves: list[dict] | None,
                               vault: Path,
                               title_index: dict[str, list[Path]]) -> dict:
    """For each moved note still on disk, scan its wikilinks and flag broken targets."""
    if moves is None:
        return {"skipped": True}

    broken: list[dict] = []
    notes_scanned = 0
    notes_missing = 0

    for row in moves:
        to_path_rel = row.get("to_path")
        if not to_path_rel:
            continue
        abs_path = (vault / to_path_rel).resolve() if not Path(to_path_rel).is_absolute() else Path(to_path_rel)
        if not abs_path.exists():
            notes_missing += 1
            continue
        try:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            notes_missing += 1
            continue
        notes_scanned += 1
        _, body = parse_frontmatter(text)
        targets = extract_wikilink_targets(body)
        for tgt in targets:
            if tgt not in title_index:
                broken.append({"from": str(abs_path.relative_to(vault)),
                               "missing_target": tgt})
    return {"skipped": False,
            "notes_scanned": notes_scanned,
            "notes_missing_on_disk": notes_missing,
            "broken_count": len(broken),
            "broken_links": broken[:50]}  # cap output


def compute_frontmatter_audit(moves: list[dict] | None,
                              vault: Path) -> dict:
    """Sanity check: count moved notes carrying the para_reclassified_at stamp."""
    if moves is None:
        return {"skipped": True}
    expected = 0
    stamped = 0
    missing_stamp_paths: list[str] = []
    for row in moves:
        to_path_rel = row.get("to_path")
        if not to_path_rel:
            continue
        expected += 1
        abs_path = vault / to_path_rel
        if not abs_path.exists():
            continue
        try:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fm, _ = parse_frontmatter(text)
        if FM_RECLASSIFIED_KEY in fm:
            stamped += 1
        else:
            missing_stamp_paths.append(to_path_rel)
    return {"skipped": False,
            "expected": expected,
            "stamped": stamped,
            "missing_stamp": expected - stamped,
            "missing_stamp_sample": missing_stamp_paths[:10]}


def compute_action_items(skipped: dict,
                         wikilink: dict,
                         frontmatter: dict,
                         transitions: dict) -> list[str]:
    items: list[str] = []
    if not wikilink.get("skipped") and wikilink.get("broken_count", 0) > 0:
        n = wikilink["broken_count"]
        items.append(
            f"{n} wikilink(s) point to titles no longer present in the vault — "
            f"review the broken-links table above and rename/delete as needed."
        )
    if not frontmatter.get("skipped") and frontmatter.get("missing_stamp", 0) > 0:
        n = frontmatter["missing_stamp"]
        items.append(
            f"{n} moved note(s) are missing the `{FM_RECLASSIFIED_KEY}` frontmatter "
            f"stamp — Phase 3b may have skipped them; spot-check the sample list."
        )
    if not skipped.get("skipped"):
        bd = skipped.get("breakdown", {})
        low = bd.get("low_confidence", 0)
        if low >= 20:
            items.append(
                f"{low} proposal(s) skipped for low confidence — re-run "
                f"`para_audit_classify.py` with a tweaked prompt or lower threshold "
                f"if you want to revisit them."
            )
    if not transitions.get("skipped") and transitions.get("total_moves", 0) == 0:
        items.append("No moves were applied — Phase 3b either ran in dry-run mode "
                     "or every proposal was rejected.")
    if not items:
        items.append("No follow-up needed. Vault is in a clean PARA shape.")
    return items


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_header(vault: Path, inventory: list[dict] | None) -> None:
    print(section_header("Header"))
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    total = len(inventory) if inventory is not None else 0
    print(f"  Generated: {Style.bold(now)}")
    print(f"  Vault:     {osc8(vault, str(vault))}")
    if inventory is None:
        print(f"  Inventory: {Style.yellow('[skipped — no inventory.jsonl]')}")
    else:
        print(f"  Notes scanned (Phase 0 baseline): {Style.bold(str(total))}")
    print()


def render_bucket_deltas(deltas: dict) -> None:
    print(section_header("Bucket counts (before -> after)"))
    if deltas.get("skipped"):
        print(f"  {Style.yellow('[skipped]')} inventory.jsonl missing — cannot compute deltas.")
        print()
        return
    bd = deltas["buckets"]
    print(f"  {'Bucket':<14} {'Before':>8} {'After':>8} {'Delta':>10}")
    print(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 10}")
    for bucket in PARA_BUCKETS:
        info = bd[bucket]
        delta_str = signed(info["delta"])
        # right-pad the colored delta string properly
        plain_delta = f"{info['delta']:+d}" if info["delta"] != 0 else "0"
        pad = max(0, 10 - len(plain_delta))
        print(f"  {bucket:<14} {info['before']:>8} {info['after']:>8} "
              f"{' ' * pad}{delta_str}")
    print(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 10}")
    diff = deltas["total_after"] - deltas["total_before"]
    plain_diff = f"{diff:+d}" if diff != 0 else "0"
    pad = max(0, 10 - len(plain_diff))
    print(f"  {'TOTAL':<14} {deltas['total_before']:>8} "
          f"{deltas['total_after']:>8} {' ' * pad}{signed(diff)}")
    print()


def render_top_transitions(top: dict) -> None:
    print(section_header("Top transitions by volume"))
    if top.get("skipped"):
        print(f"  {Style.yellow('[skipped]')} moves_applied.jsonl missing.")
        print()
        return
    if top["total_moves"] == 0:
        print(f"  {Style.dim('No moves recorded.')}")
        print()
        return
    print(f"  Total moves: {Style.bold(str(top['total_moves']))}")
    print()
    print(f"  {'#':>3}  {'Transition':<40} {'Count':>8}")
    print(f"  {'-' * 3}  {'-' * 40} {'-' * 8}")
    for i, t in enumerate(top["transitions"], 1):
        arrow = Style.dim("->")
        label = f"{t['from']} {arrow} {t['to']}"
        # account for invisible ANSI in arrow when padding
        pad = max(0, 40 - (len(t['from']) + 4 + len(t['to'])))
        print(f"  {i:>3}.  {label}{' ' * pad} {t['count']:>8}")
    print()


def render_archive_summary(arch: dict) -> None:
    print(section_header("Archive wave (last 7d)"))
    if arch.get("skipped"):
        print(f"  {Style.yellow('[skipped]')} no filing_batches/archive-*.jsonl found.")
        print()
        return
    print(f"  Notes archived: {Style.bold(str(arch['total_archived']))}")
    print(f"  Batch files used:")
    if not arch["batch_files"]:
        print(f"    {Style.dim('(none in window)')}")
    for f in arch["batch_files"]:
        p = Path(f)
        print(f"    - {osc8(p, p.name)}")
    print()


def render_skipped(skipped: dict) -> None:
    print(section_header("Skipped proposals"))
    if skipped.get("skipped"):
        print(f"  {Style.yellow('[skipped]')} proposals.jsonl missing.")
        print()
        return
    total = skipped["total_skipped"]
    print(f"  Total skipped: {Style.bold(str(total))}")
    if total == 0:
        print()
        return
    print()
    print(f"  {'Reason':<32} {'Count':>8}")
    print(f"  {'-' * 32} {'-' * 8}")
    for reason, count in sorted(skipped["breakdown"].items(),
                                key=lambda kv: -kv[1]):
        print(f"  {reason:<32} {count:>8}")
    print()


def render_wikilink(wikilink: dict, vault: Path) -> None:
    print(section_header("Wikilink integrity"))
    if wikilink.get("skipped"):
        print(f"  {Style.yellow('[skipped]')} no moves to scan.")
        print()
        return
    print(f"  Moved notes scanned:    {Style.bold(str(wikilink['notes_scanned']))}")
    if wikilink["notes_missing_on_disk"]:
        print(f"  Missing from disk:      {Style.red(str(wikilink['notes_missing_on_disk']))}")
    bc = wikilink["broken_count"]
    if bc == 0:
        print(f"  Broken wikilinks:       {Style.green('0')} — all targets resolve.")
        print()
        return
    print(f"  Broken wikilinks:       {Style.red(str(bc))}")
    print()
    print(f"  {'Source note':<60} {'Missing target':<30}")
    print(f"  {'-' * 60} {'-' * 30}")
    for entry in wikilink["broken_links"]:
        src = entry["from"]
        tgt = entry["missing_target"]
        src_abs = vault / src
        # truncate label visually but keep link to full path
        label = src if len(src) <= 60 else "..." + src[-57:]
        print(f"  {osc8(src_abs, f'{label:<60}')} "
              f"{Style.yellow(tgt[:30])}")
    if bc > len(wikilink["broken_links"]):
        extra = bc - len(wikilink["broken_links"])
        print(f"  {Style.dim(f'... ({extra} more not shown)')}")
    print()


def render_frontmatter(fm: dict) -> None:
    print(section_header("Frontmatter audit (para_reclassified_at)"))
    if fm.get("skipped"):
        print(f"  {Style.yellow('[skipped]')} no moves to scan.")
        print()
        return
    expected = fm["expected"]
    stamped = fm["stamped"]
    missing = fm["missing_stamp"]
    if expected == 0:
        print(f"  {Style.dim('No moved notes to audit.')}")
        print()
        return
    pct = (stamped / expected * 100.0) if expected else 0.0
    color = Style.green if pct >= 95 else (Style.yellow if pct >= 80 else Style.red)
    print(f"  Expected stamps: {expected}")
    print(f"  Found:           {color(str(stamped))} ({pct:.1f}%)")
    print(f"  Missing:         {missing}")
    if fm["missing_stamp_sample"]:
        print()
        print(f"  Sample of un-stamped notes:")
        for p in fm["missing_stamp_sample"]:
            print(f"    - {p}")
    print()


def render_action_items(items: list[str]) -> None:
    print(section_header("Action items"))
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")
    print()


# --------------------------------------------------------------------------- #
# JSON output
# --------------------------------------------------------------------------- #

def emit_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 5 PARA audit final report (read-only)."
    )
    parser.add_argument("--vault", type=Path,
                        default=Path(os.environ.get("OBSIDIAN_RAG_VAULT",
                                                    str(VAULT_DEFAULT))),
                        help="Path to vault (default: $OBSIDIAN_RAG_VAULT or iCloud).")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colors and OSC 8 hyperlinks.")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON instead of terminal output.")
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        Style.enabled = False
    if args.json:
        # JSON output should never carry ANSI even on a TTY
        Style.enabled = False

    vault: Path = args.vault.expanduser()
    if not vault.exists():
        print(f"Vault not found: {vault}", file=sys.stderr)
        return 2

    # Load all inputs (each may be None if the user skipped that phase)
    inventory = load_jsonl(INVENTORY_PATH)
    proposals = load_jsonl(PROPOSALS_PATH)
    approved = load_jsonl(APPROVED_PATH)
    moves = load_jsonl(MOVES_PATH)
    archive_rows, archive_files = load_archive_batches(RECENT_WINDOW_DAYS)

    # Walk the vault NOW (read-only)
    bucket_counts_now, title_index = walk_vault_now(vault)

    # Compute every section
    deltas = compute_bucket_deltas(inventory, bucket_counts_now)
    transitions = compute_top_transitions(moves)
    archive = compute_archive_summary(archive_rows, archive_files)
    skipped = compute_skipped_proposals(proposals, approved, inventory)
    wikilink = compute_wikilink_integrity(moves, vault, title_index)
    frontmatter = compute_frontmatter_audit(moves, vault)
    actions = compute_action_items(skipped, wikilink, frontmatter, transitions)

    if args.json:
        emit_json({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "vault": str(vault),
            "notes_scanned_baseline": len(inventory) if inventory else 0,
            "bucket_deltas": deltas,
            "top_transitions": transitions,
            "archive_wave": archive,
            "skipped_proposals": skipped,
            "wikilink_integrity": wikilink,
            "frontmatter_audit": frontmatter,
            "action_items": actions,
        })
        return 0

    render_header(vault, inventory)
    render_bucket_deltas(deltas)
    render_top_transitions(transitions)
    render_archive_summary(archive)
    render_skipped(skipped)
    render_wikilink(wikilink, vault)
    render_frontmatter(frontmatter)
    render_action_items(actions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
