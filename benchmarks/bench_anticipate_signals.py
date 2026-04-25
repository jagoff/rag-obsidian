#!/usr/bin/env python3
"""Benchmark Anticipate signals — mide latency de cada signal individual.

Uso:
    python benchmarks/bench_anticipate_signals.py [--iterations 5] [--json]
    python benchmarks/bench_anticipate_signals.py --signal calendar --iterations 10

Output (default texto):
    Signal benchmarks (5 iterations each)
    ====================================
    calendar       cold=42ms warm_p50=12ms warm_p95=18ms emit=2 candidates
    echo           cold=312ms warm_p50=156ms warm_p95=210ms emit=0 candidates
    commitment     cold=1830ms warm_p50=1750ms warm_p95=1920ms emit=1 candidate
    ...
    Total agent run (all signals serial): cold=2280ms warm_p50=2040ms

Output (--json): dict con structure idéntica para parsing.

Iter strategy: Run 1 = cold (fresh process). Iter 2..N = warm (mismo process,
caches calientes). Reportar cold separado + p50/p95 sobre los warm.
"""

from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

# Permite `import rag` cuando se invoca el script directamente desde el repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def bench_signal(name: str, signal_fn, iterations: int = 5) -> dict:
    """Mide latency de una signal. Returns dict con cold + warm stats."""
    timings = []
    counts = []
    now = datetime.now()

    # Iter 1: cold
    t0 = time.perf_counter()
    try:
        result = signal_fn(now)
    except Exception as exc:
        return {"name": name, "error": repr(exc), "cold_ms": None, "warm_p50_ms": None}
    cold_ms = (time.perf_counter() - t0) * 1000
    counts.append(len(result))

    # Iter 2..N: warm
    for _ in range(iterations - 1):
        t0 = time.perf_counter()
        try:
            r = signal_fn(now)
        except Exception:
            r = []
        timings.append((time.perf_counter() - t0) * 1000)
        counts.append(len(r))

    return {
        "name": name,
        "cold_ms": round(cold_ms, 1),
        "warm_p50_ms": round(statistics.median(timings), 1) if timings else None,
        "warm_p95_ms": round(_percentile(timings, 95), 1) if timings else None,
        "warm_avg_ms": round(statistics.mean(timings), 1) if timings else None,
        "warm_min_ms": round(min(timings), 1) if timings else None,
        "warm_max_ms": round(max(timings), 1) if timings else None,
        "emit_counts": counts,
        "emit_total": sum(counts),
    }


def _percentile(values: list, pct: int) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_v) - 1)
    return sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f)


def bench_all(iterations: int = 5, signal_filter: str | None = None) -> dict:
    """Run benchmark sobre todas las signals registered."""
    import rag
    signals = rag._ANTICIPATE_SIGNALS
    if signal_filter:
        signals = tuple((n, fn) for (n, fn) in signals if n == signal_filter)

    out = {
        "iterations": iterations,
        "signals": {},
        "total_cold_ms": None,
        "total_warm_p50_ms": None,
    }

    # Per-signal
    for name, fn in signals:
        out["signals"][name] = bench_signal(name, fn, iterations=iterations)

    # Total agent run (all serial via anticipate_run_impl)
    if not signal_filter:
        try:
            t0 = time.perf_counter()
            rag.anticipate_run_impl(dry_run=True, force=True)
            cold_total = (time.perf_counter() - t0) * 1000
            warm_totals = []
            for _ in range(iterations - 1):
                t0 = time.perf_counter()
                rag.anticipate_run_impl(dry_run=True, force=True)
                warm_totals.append((time.perf_counter() - t0) * 1000)
            out["total_cold_ms"] = round(cold_total, 1)
            out["total_warm_p50_ms"] = round(statistics.median(warm_totals), 1) if warm_totals else None
        except Exception:
            pass

    return out


def render_text(bench: dict) -> str:
    lines = [f"Signal benchmarks ({bench['iterations']} iterations each)"]
    lines.append("=" * 60)
    for name, s in bench["signals"].items():
        if s.get("error"):
            lines.append(f"{name:15s} ERROR: {s['error']}")
            continue
        lines.append(
            f"{name:15s} cold={s['cold_ms']}ms warm_p50={s['warm_p50_ms']}ms "
            f"warm_p95={s['warm_p95_ms']}ms emit_total={s['emit_total']}"
        )
    if bench.get("total_cold_ms"):
        lines.append("")
        lines.append(f"Total agent run (all signals serial): cold={bench['total_cold_ms']}ms warm_p50={bench['total_warm_p50_ms']}ms")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--signal", type=str, default=None, help="Solo bench una signal")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    bench = bench_all(iterations=args.iterations, signal_filter=args.signal)

    if args.json:
        print(json.dumps(bench, indent=2))
    else:
        print(render_text(bench))


if __name__ == "__main__":
    main()
