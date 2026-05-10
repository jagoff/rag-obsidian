#!/usr/bin/env python3
"""Real-time progress monitor for `rag eval` background runs.

Usage:
    python scripts/eval_progress.py /tmp/eval-baseline-bge2.log

Reads the log file and prints a one-line status of how many singles +
chains have completed, with an ETA based on pace.

Counts query completions by counting the rich `✓`/`✗` markers that the
eval prints per-query. Chains are counted by `[chain ...]` markers.

Refreshes every 5 seconds. Exits when the eval finishes (i.e. when the
final `Singles:` + `Chains:` summary lines are present).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


# Defaults — golden set sizes (safe to hardcode; see queries.yaml).
EXPECTED_SINGLES = 53
EXPECTED_CHAINS = 8  # adjust if golden grows


def _count_progress(log_text: str) -> tuple[int, int, bool]:
    """Returns (singles_done, chains_done, finished)."""
    # Each single emits one ✓ or ✗ row to the table. Chains print a
    # row per turn — N turns × M chains in the table. We count the
    # higher-level "Evaluando chains…" progress bar lines.
    singles_done = log_text.count("✓") + log_text.count("✗")
    # Cap at EXPECTED_SINGLES — chain rows also use the same markers.
    singles_done = min(singles_done, EXPECTED_SINGLES)
    chains_done = log_text.count("Chain ")
    finished = "Singles: hit@" in log_text and (
        "Chains:" in log_text or EXPECTED_CHAINS == 0
    )
    return singles_done, chains_done, finished


def main(log_path_str: str) -> None:
    log_path = Path(log_path_str)
    if not log_path.exists():
        print(f"[err] log file not found: {log_path}", flush=True)
        sys.exit(1)
    t_start = time.time()
    while True:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"[err] cannot read log: {exc}", flush=True)
            time.sleep(5)
            continue
        s, c, done = _count_progress(text)
        elapsed = time.time() - t_start
        total_units = EXPECTED_SINGLES + EXPECTED_CHAINS
        units_done = s + c
        pct = 100.0 * units_done / total_units if total_units else 0.0
        eta = ""
        if units_done > 0 and not done:
            rate = units_done / elapsed
            remaining = total_units - units_done
            eta_s = remaining / rate if rate > 0 else 0
            eta = f"ETA ~{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
        bar_w = 30
        filled = int(bar_w * pct / 100.0)
        bar = "█" * filled + "░" * (bar_w - filled)
        print(
            f"\r[{bar}] {pct:5.1f}%  singles {s:3d}/{EXPECTED_SINGLES}  "
            f"chains {c:2d}/{EXPECTED_CHAINS}  "
            f"elapsed {int(elapsed)//60}m{int(elapsed)%60:02d}s  {eta}",
            end="",
            flush=True,
        )
        if done:
            print("\n[done] eval finished, see log for results", flush=True)
            return
        time.sleep(5)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log-path>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
