"""Mide cobertura de keywords de relación alrededor de wikilinks en el vault.

Decision gate para typed-edges feature: si <30% de edges tiene señal explícita
en ±80 chars, typed-edges no vale el esfuerzo.

Uso:
    .venv/bin/python scripts/measure_wikilink_context.py [--window 80] [--sample N]
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path


WIKILINK_RE = re.compile(r"\[\[([^\]|#^]+)(?:[#^][^\]|]*)?(?:\|[^\]]+)?\]\]")

FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
CODEBLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


KEYWORDS: dict[str, list[str]] = {
    "contradicts": [
        r"\bcontradic\w*", r"\brefuta\w*", r"\bdesment\w*", r"\ben contra de\b",
        r"\bcontradict\w*", r"\brefute\w*", r"\bopposes?\b", r"\bdisagrees?\b",
        r"\bpero\b.{0,20}", r"\bsin embargo\b",
    ],
    "supersedes": [
        r"\breemplaz\w*", r"\bsupera\w*", r"\bdeprecat\w*", r"\bobsolet\w*",
        r"\bsupersed\w*", r"\breplaces?\b", r"\boverrides?\b", r"\bretires?\b",
        r"\bactualiza\w*", r"\bupdates?\b",
    ],
    "extends": [
        r"\bextiend\w*", r"\bamplí\w*", r"\bexpand\w*", r"\bcontinu\w*",
        r"\bextends?\b", r"\bbuilds on\b", r"\bbased on\b", r"\bbasad\w* en\b",
        r"\bse apoya en\b", r"\bderiva\w*",
    ],
    "cites": [
        r"\bcita\w*", r"\bsegún\b", r"\bde acuerdo\b", r"\bconforme a\b",
        r"\bcites?\b", r"\bquotes?\b", r"\baccording to\b", r"\bper\b",
        r"\bvía\b", r"\bfuente\b",
    ],
    "responds_to": [
        r"\bresponde a\b", r"\bresponds? to\b", r"\breplies to\b",
        r"\bcontesta\w*", r"\baborda\w*", r"\baddresses\b",
        r"\ben respuesta a\b", r"\bin response to\b",
    ],
    "related": [
        r"\bver también\b", r"\bcf\.", r"\bsee also\b", r"\brelacionad\w*",
        r"\brelated\b", r"\bsimilar a\b", r"\btal como\b", r"\bcompare\b",
        r"\bcompar\w*", r"\bcomo en\b",
    ],
}

COMPILED: dict[str, list[re.Pattern]] = {
    label: [re.compile(p, re.IGNORECASE) for p in pats]
    for label, pats in KEYWORDS.items()
}


def strip_frontmatter(text: str) -> tuple[str, int]:
    m = FRONTMATTER_RE.match(text)
    if m:
        return text[m.end():], m.end()
    return text, 0


def classify_context(context: str) -> str | None:
    """Return first matching label or None. Priority order: strong→weak."""
    priority = ["contradicts", "supersedes", "extends", "responds_to", "cites", "related"]
    for label in priority:
        for pat in COMPILED[label]:
            if pat.search(context):
                return label
    return None


def scan_file(path: Path, window: int) -> list[dict]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    body, fm_end = strip_frontmatter(raw)
    body = CODEBLOCK_RE.sub(lambda m: " " * len(m.group(0)), body)

    records: list[dict] = []
    for m in WIKILINK_RE.finditer(body):
        start, end = m.span()
        lo = max(0, start - window)
        hi = min(len(body), end + window)
        ctx = body[lo:start] + body[end:hi]
        label = classify_context(ctx)
        records.append({
            "path": str(path),
            "link": m.group(1).strip(),
            "offset": fm_end + start,
            "label": label,
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default=None, help="Vault path (default from env or CLAUDE.md default)")
    ap.add_argument("--window", type=int, default=80)
    ap.add_argument("--sample", type=int, default=0, help="Sample N files (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=15, help="Show N example classified snippets")
    args = ap.parse_args()

    vault = args.vault or os.environ.get("OBSIDIAN_RAG_VAULT") or str(
        Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
    )
    vault_p = Path(vault)
    if not vault_p.exists():
        print(f"vault not found: {vault_p}", file=sys.stderr)
        return 1

    files = [
        p for p in vault_p.rglob("*.md")
        if not any(seg.startswith(".") for seg in p.relative_to(vault_p).parts)
    ]
    print(f"vault: {vault_p}")
    print(f"total md files: {len(files)}")
    if args.sample > 0 and args.sample < len(files):
        random.seed(args.seed)
        files = random.sample(files, args.sample)
        print(f"sampled: {len(files)}")

    total_links = 0
    label_counts: Counter[str] = Counter()
    examples: dict[str, list[dict]] = {k: [] for k in KEYWORDS}
    per_file_density: list[tuple[str, int, int]] = []  # (path, total, classified)

    for p in files:
        recs = scan_file(p, args.window)
        if not recs:
            continue
        classified = 0
        for r in recs:
            total_links += 1
            if r["label"]:
                label_counts[r["label"]] += 1
                classified += 1
                if len(examples[r["label"]]) < args.show:
                    # re-extract the snippet for display
                    raw = p.read_text(encoding="utf-8", errors="ignore")
                    body, _ = strip_frontmatter(raw)
                    body = CODEBLOCK_RE.sub(lambda m: " " * len(m.group(0)), body)
                    off = r["offset"] - (len(raw) - len(body))
                    lo = max(0, off - args.window)
                    hi = min(len(body), off + len(r["link"]) + 4 + args.window)
                    snippet = body[lo:hi].replace("\n", " ").strip()
                    examples[r["label"]].append({
                        "file": p.name,
                        "link": r["link"],
                        "ctx": snippet[:240],
                    })
            else:
                label_counts["_unclassified"] += 1
        per_file_density.append((str(p.relative_to(vault_p)), len(recs), classified))

    classified_total = total_links - label_counts["_unclassified"]
    coverage = (classified_total / total_links * 100) if total_links else 0.0

    print("\n" + "=" * 60)
    print(f"RESULTS (window=±{args.window} chars)")
    print("=" * 60)
    print(f"total wikilinks     : {total_links}")
    print(f"classified (any)    : {classified_total}  ({coverage:.1f}%)")
    print(f"unclassified        : {label_counts['_unclassified']}")
    print()
    print("by label:")
    for label in ["contradicts", "supersedes", "extends", "responds_to", "cites", "related"]:
        c = label_counts[label]
        pct = (c / total_links * 100) if total_links else 0.0
        print(f"  {label:14s}: {c:5d}  ({pct:.1f}%)")

    print("\nDECISION GATE:")
    if coverage < 30:
        verdict = "ABANDON — señal escasa (<30%). Typed-edges no rompe techo de chains."
    elif coverage < 40:
        verdict = "MARGINAL (30-40%). Probar spike 1-día con los tipos más densos antes de comprometer."
    else:
        verdict = "PROCEED — señal suficiente (≥40%). Avanzar a prototipo heurístico + intent-gated expansion."
    print(f"  {verdict}")

    print("\nexamples (top by label):")
    for label, items in examples.items():
        if not items:
            continue
        print(f"\n--- {label} ---")
        for it in items[:3]:
            print(f"  [{it['file']}] -> [[{it['link']}]]")
            print(f"    ctx: {it['ctx']}")

    # Density top-10
    if per_file_density:
        print("\nfiles with most classified edges:")
        ranked = sorted(per_file_density, key=lambda x: -x[2])[:10]
        for path, total, cls in ranked:
            print(f"  {cls:3d}/{total:3d}  {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
