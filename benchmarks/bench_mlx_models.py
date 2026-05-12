#!/usr/bin/env python3
"""Benchmark de modelos MLX para chat usando el endpoint HTTP.

Compara latencia de diferentes modelos MLX con las mismas queries.
"""
import argparse
import os
import subprocess
import sys
import time
import uuid
import requests

BASE_URL = "http://localhost:8765"
QUERIES = [
    "qué información tengo sobre guitarras",
    "qué dice mi nota sobre ikigai",
    "comandos CLI de claude code",
    "letra de muros fractales",
    "qué es ELEVA",
    "hola, cómo va",
    "gracias",
]


def swap_model(model: str) -> None:
    """Cambia el modelo de chat usando rag models swap."""
    print(f"  Cambiando a {model}...")
    subprocess.run(["rag", "models", "swap", "chat", model], check=True)
    time.sleep(2)  # Esperar a que el cambio surta efecto


def chat_request(query: str) -> dict:
    """POST /api/chat y mide latencia."""
    session_id = f"bench-{uuid.uuid4().hex[:8]}"
    payload = {"question": query, "session_id": session_id}
    t0 = time.monotonic()
    parts: list[str] = []
    try:
        with requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=180, stream=True) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                try:
                    evt = __import__("json").loads(raw[6:])
                except Exception:
                    continue
                delta = evt.get("delta")
                if delta:
                    parts.append(delta)
        wall_ms = (time.monotonic() - t0) * 1000
        reply = "".join(parts)
        return {
            "wall_ms": wall_ms,
            "reply_preview": reply[:160].replace("\n", " ") if reply else "",
            "success": True,
        }
    except Exception as e:
        wall_ms = (time.monotonic() - t0) * 1000
        return {
            "wall_ms": wall_ms,
            "reply_preview": str(e),
            "success": False,
        }


def benchmark_model(model: str) -> list[dict]:
    """Benchmarca un modelo con todas las queries."""
    print(f"\n=== Benchmark: {model} ===")
    swap_model(model)
    results = []
    for q in QUERIES:
        print(f"  {q[:50]!r} ...", end=" ", flush=True)
        res = chat_request(q)
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

    # Guardar el modelo original para restaurarlo después
    original_model = "qwen2.5:7b"  # Default

    results_by_model = {}
    try:
        for model in args.models:
            results_by_model[model] = benchmark_model(model)
    finally:
        # Restaurar el modelo original
        print(f"\nRestaurando modelo original: {original_model}")
        swap_model(original_model)

    print_comparison(results_by_model)


if __name__ == "__main__":
    main()
