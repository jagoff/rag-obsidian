#!/usr/bin/env python3
"""Empirical threshold calibration for `CONFIDENCE_RERANK_MIN`,
`CONFIDENCE_DEEP_THRESHOLD`, `GRAPH_EXPANSION_GATE`, and related gates.

Reads `rag_queries` from the live state DB and reports the percentile
distribution of `top_score` for the `retrieve()`-producing commands
(query/chat/web), so you can re-calibrate the gates without re-running
a full eval.

Usage:
  .venv/bin/python scripts/calibrate_thresholds.py [--days 30]

Output (stdout):
  - Count + date range
  - Percentiles (p01, p05, p10, ..., p95, p99) of top_score
  - Current threshold values and where they fall in the percentile grid
  - Flagged queries (bad_citations, gated_low_confidence, critique_changed) rate

Does NOT change anything. Thresholds live in `rag.py` — tweak by hand
after looking at the numbers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = int(round(q * (len(sorted_values) - 1)))
    idx = max(0, min(len(sorted_values) - 1, idx))
    return sorted_values[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30,
                    help="window size in days (default 30)")
    args = ap.parse_args()

    cutoff = (
        rag.datetime.now() - rag.timedelta(days=args.days)  # type: ignore[attr-defined]
    ).isoformat(timespec="seconds")

    try:
        with rag._ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT top_score, critique_changed, bad_citations_json, "
                "extra_json, cmd, ts FROM rag_queries "
                "WHERE ts >= ? AND cmd IN ('query', 'chat', 'web') "
                "  AND top_score IS NOT NULL "
                "ORDER BY top_score ASC",
                (cutoff,),
            ).fetchall()
    except Exception as e:
        print(f"error reading rag_queries: {e}", file=sys.stderr)
        return 2

    scores = [float(r[0]) for r in rows if r[0] is not None]
    n = len(scores)
    print(f"# Threshold calibration ({args.days}d window)")
    print(f"# rows: {n}")
    if n == 0:
        print("No retrieval-bearing queries in window. Nothing to calibrate.")
        return 1

    print(f"# min: {scores[0]:.4f}   max: {scores[-1]:.4f}   n={n}")
    print()
    print("| Percentile | top_score |")
    print("|---|---|")
    for pct in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        v = _percentile(scores, pct / 100)
        print(f"| p{pct:02d} | {v:.4f} |")

    print()
    print("## Current thresholds")
    thresholds = {
        "CONFIDENCE_RERANK_MIN": rag.CONFIDENCE_RERANK_MIN,
        "CONFIDENCE_DEEP_THRESHOLD": rag.CONFIDENCE_DEEP_THRESHOLD,
        "GRAPH_EXPANSION_GATE": rag.GRAPH_EXPANSION_GATE,
    }
    for name, value in thresholds.items():
        # Percentile of `value` in the distribution.
        rank = sum(1 for s in scores if s < value)
        pct_rank = rank / n * 100
        print(f"- `{name}` = {value}  →  "
              f"p{pct_rank:.1f} of observed top_score ({rank}/{n} below)")

    # Query-quality signals.
    import json as _json
    gated = 0
    bad_cites = 0
    for r in rows:
        # critique_changed is its own column; bad_citations sits in the JSON.
        try:
            bc_json = r[2] or "[]"
            if _json.loads(bc_json):
                bad_cites += 1
        except Exception:
            pass
        try:
            extra = _json.loads(r[3] or "{}")
            if extra.get("gated_low_confidence"):
                gated += 1
        except Exception:
            pass
    crit_changed = sum(1 for r in rows if r[1])
    print()
    print("## Query-quality rates")
    print(f"- gated_low_confidence: {gated}/{n} = {gated/n*100:.1f}%")
    print(f"- bad_citations:        {bad_cites}/{n} = {bad_cites/n*100:.1f}%")
    print(f"- critique_changed:     {crit_changed}/{n} = {crit_changed/n*100:.1f}%")

    print()
    print("## Interpretation")
    below_gate = sum(1 for s in scores if s < rag.CONFIDENCE_RERANK_MIN)
    if rag.CONFIDENCE_RERANK_MIN < scores[0]:
        print("- CONFIDENCE_RERANK_MIN below observed min — never triggers, consider raising.")
    elif below_gate / n > 0.10:
        print(
            f"- CONFIDENCE_RERANK_MIN sits below {below_gate/n*100:.1f}% of scores; "
            "observed `gated_low_confidence` may under-report "
            "(paths: --force or chat bypass gate)."
        )
    else:
        print(f"- CONFIDENCE_RERANK_MIN in a comfortable band ({below_gate/n*100:.1f}% below).")
    deep_below = sum(1 for s in scores if s < rag.CONFIDENCE_DEEP_THRESHOLD)
    if deep_below / n > 0.40:
        print(
            f"- CONFIDENCE_DEEP_THRESHOLD triggers deep retrieval on {deep_below/n*100:.1f}% "
            "of queries — consider lowering if the extra LLM round-trips hurt p95."
        )
    else:
        print(f"- CONFIDENCE_DEEP_THRESHOLD triggers on {deep_below/n*100:.1f}% — OK.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
