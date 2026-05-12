#!/usr/bin/env python3
"""Benchmark de modelos MLX para chat.

Compara latencia de diferentes modelos MLX con las mismas queries.
"""
import argparse
import subprocess
import sys
import time

QUERIES = [
    "qué información tengo sobre guitarras",
    "qué dice mi nota sobre ikigai",
    "comandos CLI de claude code",
    "letra de muros fractales",
    "qué es ELEVA",
    "hola, cómo va",
    "gracias",
]


def run_query(model: str, query: str) -> dict:
    """Ejecuta una query con el modelo especificado y mide latencia."""
    cmd = ["rag", "chat", "--model", model, query]
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
        wall_ms = (time.monotonic() - t0) * 1000
        reply = result.stdout or result.stderr
        return {
            "wall_ms": wall_ms,
            "reply_preview": reply[:160].replace("\n", " ") if reply else "",
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "wall_ms": (time.monotonic() - t0) * 1000,
            "reply_preview": "TIMEOUT",
            "success": False,
        }
    except Exception as e:
        return {
            "wall_ms": (time.monotonic() - t0) * 1000,
            "reply_preview": str(e),
            "success": False,
        }


def benchmark_model(model: str) -> list[dict]:
    """Benchmarca un modelo con todas las queries."""
    print(f"\n=== Benchmark: {model} ===")
    results = []
    for q in QUERIES:
        print(f"  {q[:50]!r} ...", end=" ", flush=True)
        res = run_query(model, q)
        wall = res["wall_ms"]
        print(f"{wall:.0f}ms" if res["success"] else "FAIL")
        res["query"] = q
        results.append(res)
    return results


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def print_comparison(results_by_model: dict[str, list[dict]]) -> None:
    """Imprime tabla comparativa de latencia."""
    models = list(results_by_model.keys())
    if not models:
        print("No results to compare.")
        return

    print("\n=== Comparison ===\n")
    header = f"{'Query':<38} | " + " | ".join(f"{m:>9}" for m in models)
    print(header)
    print("-" * len(header))

    for i, q in enumerate(QUERIES):
        row = f"{q[:36]:<38} |"
        for model in models:
            res = results_by_model[model][i]
            ms = res["wall_ms"] if res["success"] else None
            row += f" {ms:>9.0f}" if ms else "     FAIL"
        print(row)

    print(f"\n=== Stats ===\n")

    for model in models:
        vals = [r["wall_ms"] for r in results_by_model[model] if r["success"]]
        if vals:
            v_sorted = sorted(vals)
            print(
                f"{model:<15} | min {v_sorted[0]/1000:.1f}s"
                f"  p50 {percentile(v_sorted, 50)/1000:.1f}s"
                f"  p95 {percentile(v_sorted, 95)/1000:.1f}s"
                f"  max {v_sorted[-1]/1000:.1f}s"
            )
        else:
            print(f"{model:<15} | N/A")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de modelos MLX para chat")
    parser.add_argument(
        "models",
        nargs="+",
        help="Modelos a comparar (ej: qwen2.5:7b qwen3:30b-a3b phi4:latest)",
    )
    args = parser.parse_args()

    results_by_model = {}
    for model in args.models:
        results_by_model[model] = benchmark_model(model)

    print_comparison(results_by_model)


if __name__ == "__main__":
    main()
