#!/usr/bin/env python3
"""
PARA AUDIT — PHASE 3A: ARCHIVE WAVE RUNBOOK
============================================

Independent of the LLM-driven proposals, the archive wave reuses the existing
`rag dead` + `rag archive` tooling. It identifies notes by link-graph + retrieval
history (different signal than the LLM classification), so it's complementary
not redundant.

User commands (run in this order):

  1. rag dead --min-age-days 365
     # Read-only list of archive candidates. Review the output.

  2. rag archive --apply --gate 20
     # Moves dead notes → 04-Archive/, stamps frontmatter (archived_at,
     # archived_from, archived_reason). Gate >20 candidates → dry-run unless
     # --force is added. Audit log: ~/.local/share/obsidian-rag/filing_batches/
     # archive-*.jsonl

  3. (Optional, if you trust the candidates and exceeded the gate)
     rag archive --apply --force --gate 999

This wave runs BEFORE the bulk mover (Phase 3b) so the proposals.jsonl
universe shrinks (notes already archived can't be re-classified).

──────────────────────────────────────────────────────────────────────────────
PHASE 2 + 3a REVIEW SCRIPT — what this file does
──────────────────────────────────────────────────────────────────────────────

Reads ~/.local/share/obsidian-rag/para_audit/proposals.jsonl (produced by
Phase 1 LLM classification), filters by confidence, groups by transition type,
and writes the approved subset to proposals_approved.jsonl as the gate before
Phase 3b applies the bulk move.

Stdlib-only — no venv required.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

VAULT_DEFAULT = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
AUDIT_DIR = Path.home() / ".local/share/obsidian-rag/para_audit"
PROPOSALS_FILE = AUDIT_DIR / "proposals.jsonl"
APPROVED_FILE = AUDIT_DIR / "proposals_approved.jsonl"
DAILY_NOTE_REL = "00-Inbox/Daily note.md"

# ── Skip-zone: trust the classifier for most notes, but hard-exclude the same
#    path the inventory scanner already skipped. Notes in the skip zone should
#    never appear in proposals.jsonl because the Phase 1 classifier was given
#    the inventory output; but we double-check here as a belt-and-suspenders
#    guard so we never write a skip-zone path into proposals_approved.jsonl.
CLAUDE_SKIP_PREFIX = "04-Archive/99-obsidian-system/99-Claude"

PARA_BUCKETS = {"01-Projects", "02-Areas", "03-Resources", "04-Archive"}

# Wikilink regex (matches [[Title]] and [[Title|alias]] — captures bare title)
WIKILINK_RE = re.compile(r"\[\[([^\[\]\|]+)(?:\|[^\[\]]+)?\]\]")

# ── ANSI color helpers ─────────────────────────────────────────────────────────

_USE_COLOR = True  # toggled by --no-color


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def bold(t: str) -> str:    return _c("1", t)
def dim(t: str) -> str:     return _c("2", t)
def cyan(t: str) -> str:    return _c("36", t)
def yellow(t: str) -> str:  return _c("33", t)
def green(t: str) -> str:   return _c("32", t)
def red(t: str) -> str:     return _c("31", t)
def magenta(t: str) -> str: return _c("35", t)


def osc8(abs_path: str, label: str) -> str:
    """OSC 8 hyperlink so the user can click to open in their terminal."""
    if not _USE_COLOR:
        return label
    return f"\x1b]8;;file://{abs_path}\x1b\\{label}\x1b]8;;\x1b\\"


def rule(char: str = "─", width: int = 72) -> str:
    return dim(char * width)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_proposals(path: Path) -> list[dict]:
    """Read proposals.jsonl; skip malformed lines with a warning."""
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                records.append(rec)
            except json.JSONDecodeError as exc:
                print(f"WARN line {lineno}: {exc}", file=sys.stderr)
    return records


def read_note_body(vault: Path, rel_path: str, max_chars: int = 100) -> str:
    """Return first max_chars chars of note body (after frontmatter), or empty string."""
    abs_path = vault / rel_path
    if not abs_path.exists():
        return ""
    try:
        text = abs_path.read_text(encoding="utf-8", errors="replace")
        # Strip frontmatter if present
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                text = text[end + 4:].lstrip("\n")
        return text[:max_chars].replace("\n", " ").strip()
    except OSError:
        return ""


def load_daily_note_titles(vault: Path) -> set[str]:
    """Extract wikilink targets from 00-Inbox/Daily note.md as a set of lower-case titles."""
    daily = vault / DAILY_NOTE_REL
    if not daily.exists():
        return set()
    try:
        text = daily.read_text(encoding="utf-8", errors="replace")
        return {m.lower() for m in WIKILINK_RE.findall(text)}
    except OSError:
        return set()


def is_in_skip_zone(rel_path: str) -> bool:
    return rel_path.startswith(CLAUDE_SKIP_PREFIX)


# ── Display helpers ────────────────────────────────────────────────────────────

def transition_key(rec: dict) -> str:
    cur = rec.get("current_bucket", "?")
    prop = rec.get("proposed_bucket", "?")
    return f"{cur} → {prop}"


def print_transition_block(
    key: str,
    proposals: list[dict],
    vault: Path,
    daily_titles: set[str],
) -> None:
    # Sort by confidence descending; sample top 5
    by_conf = sorted(proposals, key=lambda r: r.get("confidence", 0.0), reverse=True)
    sample = by_conf[:5]

    print()
    print(rule())
    print(f"  {bold(cyan(key))}  {dim(f'({len(proposals)} proposals)')}")
    print(rule())

    for rec in sample:
        rel = rec.get("path", "")
        title = Path(rel).stem if rel else rec.get("title", "?")
        conf = rec.get("confidence", 0.0)
        reason = rec.get("reason", "")
        snippet = read_note_body(vault, rel, max_chars=100)
        abs_path = str((vault / rel).resolve()) if rel else ""

        conf_color = green if conf >= 0.85 else (yellow if conf >= 0.75 else dim)
        link = osc8(abs_path, title) if abs_path else title

        # Flag if note title appears in daily note wikilinks
        in_daily = title.lower() in daily_titles

        daily_flag = f"  {yellow('  ★ in Daily note')}" if in_daily else ""
        print(f"  {link}{daily_flag}")
        print(f"    {dim('conf:')} {conf_color(f'{conf:.2f}')}"
              f"  {dim('reason:')} {reason[:90]}")
        if snippet:
            print(f"    {dim('body:  ' + snippet[:97] + ('…' if len(snippet) >= 97 else ''))}")
        print()


def print_daily_note_section(flagged: list[dict], vault: Path) -> None:
    if not flagged:
        return
    print()
    print(rule("═"))
    print(f"  {bold(yellow('PROPOSALS REFERENCED IN DAILY NOTE — review carefully'))}")
    print(rule("═"))
    for rec in flagged:
        rel = rec.get("path", "")
        title = Path(rel).stem if rel else rec.get("title", "?")
        conf = rec.get("confidence", 0.0)
        trans = transition_key(rec)
        abs_path = str((vault / rel).resolve()) if rel else ""
        link = osc8(abs_path, title) if abs_path else title
        print(f"  {link}  {dim(trans)}  {dim(f'conf={conf:.2f}')}")
    print()


def print_summary_table(
    all_proposals: list[dict],
    approved: list[dict],
    by_transition: dict[str, list[dict]],
    threshold: float,
) -> None:
    total = len(all_proposals)
    n_approved = len(approved)
    n_filtered = total - n_approved

    print()
    print(rule())
    print(f"  {bold('SUMMARY')}")
    print(rule())
    print(f"  {'Transition':<40} {'Proposals':>9}  {'Approved':>8}")
    print(f"  {dim('-' * 60)}")

    for key in sorted(by_transition.keys()):
        group = by_transition[key]
        n_in = sum(1 for r in group if r.get("confidence", 0.0) >= threshold
                   and not is_in_skip_zone(r.get("path", "")))
        print(f"  {key:<40} {len(group):>9}  {n_in:>8}")

    print(f"  {dim('-' * 60)}")
    print(f"  {'TOTAL':<40} {total:>9}  {n_approved:>8}")
    print()
    print(f"  {dim('Filtered out (confidence < ' + str(threshold) + '):')}  {n_filtered}")
    print(f"  {dim('Skip-zone excluded:')}  "
          f"{sum(1 for r in all_proposals if is_in_skip_zone(r.get('path', '')))}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global _USE_COLOR

    parser = argparse.ArgumentParser(
        description="PARA audit Phase 2+3a: review LLM proposals + write approved gate file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.7, metavar="FLOAT",
        help="Minimum confidence to include in approved set (default: 0.70)",
    )
    parser.add_argument(
        "--vault", type=Path, default=None,
        help="Override vault path (default: OBSIDIAN_RAG_VAULT env or standard iCloud path)",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colors and OSC 8 hyperlinks",
    )
    args = parser.parse_args()

    if args.no_color:
        _USE_COLOR = False

    # Resolve vault
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

    # Load proposals
    if not PROPOSALS_FILE.exists():
        print(f"ERROR: proposals file not found: {PROPOSALS_FILE}", file=sys.stderr)
        print("Run Phase 1 (LLM classification) first.", file=sys.stderr)
        sys.exit(1)

    all_proposals = load_proposals(PROPOSALS_FILE)
    if not all_proposals:
        print("No proposals found in proposals.jsonl — nothing to review.")
        sys.exit(0)

    threshold = args.min_confidence

    # Load daily note wikilinks for cross-reference
    daily_titles = load_daily_note_titles(vault)
    daily_note_exists = (vault / DAILY_NOTE_REL).exists()

    # Group ALL proposals by transition (for summary counts)
    by_transition: dict[str, list[dict]] = defaultdict(list)
    for rec in all_proposals:
        by_transition[transition_key(rec)].append(rec)

    # Compute approved set: confidence >= threshold AND not in skip zone
    # We trust the Phase 1 classifier to have excluded skip-zone notes, but
    # double-check here as a hard guard (see comment near CLAUDE_SKIP_PREFIX).
    approved = [
        r for r in all_proposals
        if r.get("confidence", 0.0) >= threshold
        and not is_in_skip_zone(r.get("path", ""))
    ]

    # Collect flagged (in daily note) from approved set for the special section
    flagged_daily: list[dict] = []
    approved_titles = {Path(r.get("path", "")).stem.lower() for r in approved}
    if daily_titles:
        for rec in approved:
            title = Path(rec.get("path", "")).stem.lower()
            if title in daily_titles:
                flagged_daily.append(rec)

    # ── Print header ──────────────────────────────────────────────────────────
    print()
    print(bold(cyan("PARA AUDIT — PHASE 2 REVIEW")))
    print(dim(f"proposals.jsonl:  {PROPOSALS_FILE}"))
    print(dim(f"vault:            {vault}"))
    print(dim(f"min-confidence:   {threshold}"))
    print(dim(f"total proposals:  {len(all_proposals)}"))
    print(dim(f"daily note:       {'found' if daily_note_exists else 'not found'} ({DAILY_NOTE_REL})"))

    # ── Per-transition blocks ─────────────────────────────────────────────────
    # Only print blocks that have at least one approved proposal so noise
    # transitions that are all low-confidence don't clutter the output.
    approved_by_transition: dict[str, list[dict]] = defaultdict(list)
    for rec in approved:
        approved_by_transition[transition_key(rec)].append(rec)

    for key in sorted(approved_by_transition.keys()):
        group = approved_by_transition[key]
        print_transition_block(key, group, vault, daily_titles)

    # ── Daily-note cross-reference section ───────────────────────────────────
    print_daily_note_section(flagged_daily, vault)

    # ── Write approved gate file ──────────────────────────────────────────────
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with APPROVED_FILE.open("w", encoding="utf-8") as fh:
        for rec in approved:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    approved_link = osc8(str(APPROVED_FILE), str(APPROVED_FILE))
    print(f"  {bold(green('Approved gate file written:'))} {approved_link}")
    print(f"  {dim(str(len(approved)) + ' proposals written')}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print_summary_table(all_proposals, approved, by_transition, threshold)

    # ── Final pause line ──────────────────────────────────────────────────────
    print(bold(yellow(
        "PAUSE: review proposals_approved.jsonl before proceeding to apply."
    )))
    print()


if __name__ == "__main__":
    main()
