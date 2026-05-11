#!/usr/bin/env python3
"""CLI standalone para `rag/vault_filer.py` — dry-run + apply gated.

Uso:
  scripts/vault_filer_cli.py plan [--limit N]      # default dry-run
  scripts/vault_filer_cli.py apply --yes [--limit N]

`apply` requiere `--yes` para confirmar — sin él aborta antes de mover.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _fmt_table(items: list[dict]) -> str:
    if not items:
        return "(empty inbox)"
    rows = []
    for r in items:
        path = Path(r.get("path", "")).name
        dest = r.get("destination", "?")
        conf = r.get("confidence", 0.0)
        reason = (r.get("reason", "") or "")[:60]
        rows.append(f"  {path[:40]:40s}  →  {dest[:40]:40s}  conf={conf:.2f}  {reason}")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan", help="dry-run, lista lo que movería")
    p_plan.add_argument("--limit", type=int, default=None)
    p_plan.add_argument("--json", action="store_true")
    p_apply = sub.add_parser("apply", help="ejecuta movimientos")
    p_apply.add_argument("--limit", type=int, default=None)
    p_apply.add_argument("--yes", action="store_true",
                         help="skip confirm interactivo")
    p_apply.add_argument("--threshold", type=float, default=0.65,
                         help="confidence mínimo para mover (default 0.65)")
    args = ap.parse_args()

    from rag import vault_filer  # noqa: PLC0415

    items = vault_filer.plan(limit=args.limit)
    if args.cmd == "plan":
        if args.json:
            print(json.dumps(items, ensure_ascii=False, indent=2))
        else:
            print(f"[filer] {len(items)} notes en 00-Inbox/")
            print(_fmt_table(items))
            high_conf = [r for r in items if r.get("confidence", 0) >= 0.65 and r.get("destination") != "stay"]
            print(f"\n[filer] {len(high_conf)} con confidence ≥ 0.65 → candidatas a apply")
        return 0

    # apply
    high_conf = [r for r in items if r.get("confidence", 0) >= args.threshold and r.get("destination") != "stay"]
    print(f"[filer] plan ready: {len(items)} scanned, {len(high_conf)} above threshold {args.threshold}")
    print(_fmt_table(high_conf))
    if not args.yes:
        print("\n[filer] aborting — pasá --yes para confirmar")
        return 1
    stats = vault_filer.apply_plan(items, threshold=args.threshold)
    print(f"\n[filer] moved={stats['moved']} skipped_low_conf={stats['skipped_low_conf']} "
          f"skipped_stay={stats['skipped_stay']} errors={stats['errors']}")
    return 0 if stats.get("errors", 0) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
