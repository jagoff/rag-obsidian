#!/usr/bin/env python3
"""Standalone benchmark: ollama vs MLX backend for /api/chat.

Usage:
    python benchmarks/bench_mlx_vs_ollama.py            # full bench (restarts service)
    python benchmarks/bench_mlx_vs_ollama.py --dry-run  # 2 queries, no restart

Requires: web service already installed as com.fer.obsidian-rag-web launchd.
"""

import argparse
import os
import plistlib
import re
import subprocess
import sys
import time
import uuid

import requests

PLIST = "/Users/fer/Library/LaunchAgents/com.fer.obsidian-rag-web.plist"
BASE_URL = "http://localhost:8765"
WARMUP_LABEL = "__warmup__"

QUERIES = [
    "qué información tengo sobre guitarras",   # historial drift
    "qué dice mi nota sobre ikigai",            # RAG simple
    "comandos CLI de claude code",              # tech reference
    "letra de muros fractales",                 # vault-specific
    "qué es ELEVA",                             # entity lookup
    "hola, cómo va",                            # chitchat
    "gracias",                                  # one-word
]

# --- drift detection ---

CJK_RANGE = re.compile(
    r"[\u4e00-\u9fff"          # CJK Unified
    r"\u3040-\u309f"           # Hiragana
    r"\u30a0-\u30ff"           # Katakana
    r"\uac00-\ud7af"           # Korean
    r"\u0400-\u04ff"           # Cyrillic
    r"\u0600-\u06ff"           # Arabic
    r"]"
)

PT_IT_HINTS = re.compile(
    r"\b(você|também|aqui está|eis aqui|però|anche|qualcosa|perché|"
    r"isso|então|como você|tudo bem|prego|grazie|buongiorno)\b",
    re.IGNORECASE,
)


def detect_drift(text: str) -> str | None:
    """Return drift label or None."""
    if CJK_RANGE.search(text):
        return "CJK/Cyrillic/Arabic"
    m = PT_IT_HINTS.search(text)
    if m:
        return f"PT/IT hint: {m.group()!r}"
    return None


# --- plist helpers ---

def _read_plist() -> tuple[bytes, dict]:
    with open(PLIST, "rb") as f:
        raw = f.read()
    return raw, plistlib.loads(raw)


def _write_plist(d: dict) -> None:
    with open(PLIST, "wb") as f:
        plistlib.dump(d, f)


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def restart_web_with_env(env_patch: dict[str, str | None]) -> None:
    """Apply env_patch to plist in-memory and restart the launchd service."""
    _, d = _read_plist()
    env = d.setdefault("EnvironmentVariables", {})
    for k, v in env_patch.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    _write_plist(d)

    uid = os.getuid()
    target = f"gui/{uid}"
    _launchctl("bootout", target, PLIST)
    time.sleep(3)
    _launchctl("bootstrap", target, PLIST)
    time.sleep(10)  # let server start + model load


def wait_for_server(timeout: int = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/api/vaults", timeout=3)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    print("ERROR: web service did not come up in time", file=sys.stderr)
    sys.exit(1)


# --- request helper ---

def chat_request(query: str, vault_scope: str = "work") -> dict:
    """POST /api/chat and parse SSE stream. Returns timing dict.

    Endpoint streams Server-Sent Events: each line `data: {...}\\n` is JSON
    with optional `delta` (token), `id`, `stage`, etc. The full reply is
    the concatenation of all `delta` fields. Wall is end-to-end (until stream
    closes).
    """
    import json as _json
    session_id = f"bench-{uuid.uuid4().hex[:8]}"
    payload = {"question": query, "session_id": session_id, "vault_scope": vault_scope}
    t0 = time.monotonic()
    parts: list[str] = []
    ttft_ms: float | None = None
    with requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=180, stream=True) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            try:
                evt = _json.loads(raw[6:])
            except Exception:
                continue
            delta = evt.get("delta")
            if delta:
                if ttft_ms is None:
                    ttft_ms = (time.monotonic() - t0) * 1000
                parts.append(delta)
    wall_ms = (time.monotonic() - t0) * 1000
    reply = "".join(parts)
    return {
        "wall_ms": wall_ms,
        "ttft_ms": ttft_ms,
        "timing_ms": None,
        "reply_preview": reply[:160].replace("\n", " "),
        "drift": detect_drift(reply),
    }


# --- benchmark runner ---

def run_round(label: str, queries: list[str], dry_run: bool) -> list[dict]:
    results = []
    warmup_done = False
    for q in queries:
        is_warmup = not warmup_done
        tag = WARMUP_LABEL if is_warmup else q
        print(f"  [{label}] {'(warmup) ' if is_warmup else ''}{q[:50]!r} ...", end=" ", flush=True)
        try:
            res = chat_request(q)
        except Exception as exc:
            print(f"ERROR: {exc}")
            res = {"wall_ms": None, "ttft_ms": None, "timing_ms": None, "reply_preview": "", "drift": None}
        res["query"] = q
        res["is_warmup"] = is_warmup
        wall = res["wall_ms"]
        print(f"{wall:.0f}ms" if wall is not None else "FAIL", "DRIFT!" if res["drift"] else "")
        results.append(res)
        warmup_done = True
    return results


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def print_table(ollama_res: list[dict], mlx_res: list[dict]) -> bool:
    """Print comparison table. Returns True if MLX wins (p50 faster + no drift)."""
    # Filter warmup
    ollama_real = [r for r in ollama_res if not r["is_warmup"]]
    mlx_real = [r for r in mlx_res if not r["is_warmup"]]

    print("\n=== Comparison (warmup excluded) ===\n")
    header = f"{'Query':<38} | {'ollama':>9} | {'MLX':>9} | {'speedup':>8} | drift"
    print(header)
    print("-" * len(header))

    speedups: list[float] = []
    drift_count = 0

    for o, m in zip(ollama_real, mlx_real):
        q = o["query"][:36]
        o_ms = o["wall_ms"]
        m_ms = m["wall_ms"]
        if o_ms and m_ms:
            sp = o_ms / m_ms
            speedups.append(sp)
            sp_str = f"{sp:.2f}x"
        else:
            sp_str = "N/A"
        drift_label = f"FAIL ({m['drift']})" if m["drift"] else "OK"
        if m["drift"]:
            drift_count += 1
        o_str = f"{o_ms:.0f} ms" if o_ms else "FAIL"
        m_str = f"{m_ms:.0f} ms" if m_ms else "FAIL"
        print(f"{q:<38} | {o_str:>9} | {m_str:>9} | {sp_str:>8} | {drift_label}")

    n = len(ollama_real)
    print(f"\n=== Stats (warmup excluded, N={n}) ===\n")

    def _stats_row(label: str, vals: list[float | None]) -> None:
        v = sorted(x for x in vals if x is not None)
        if not v:
            print(f"{label:<8} | N/A")
            return
        print(
            f"{label:<8} | min {v[0]/1000:.1f}s"
            f"  p50 {percentile(v, 50)/1000:.1f}s"
            f"  p95 {percentile(v, 95)/1000:.1f}s"
            f"  max {v[-1]/1000:.1f}s"
        )

    _stats_row("ollama", [r["wall_ms"] for r in ollama_real])
    _stats_row("MLX", [r["wall_ms"] for r in mlx_real])

    if speedups:
        sp_sorted = sorted(speedups)
        print(
            f"{'speedup':<8} | min {sp_sorted[0]:.2f}x"
            f"  p50 {percentile(sp_sorted, 50):.2f}x"
            f"  p95 {percentile(sp_sorted, 95):.2f}x"
            f"  max {sp_sorted[-1]:.2f}x"
        )

    print(f"\nDrift events: {drift_count}/{n} (MLX)")

    print("\n=== Decision ===")
    ollama_p50 = percentile(sorted(r["wall_ms"] for r in ollama_real if r["wall_ms"]), 50)
    mlx_p50 = percentile(sorted(r["wall_ms"] for r in mlx_real if r["wall_ms"]), 50)
    mlx_wins = mlx_p50 < ollama_p50 and drift_count == 0
    pct = abs(ollama_p50 - mlx_p50) / ollama_p50 * 100 if ollama_p50 else 0
    direction = "faster" if mlx_p50 < ollama_p50 else "slower"
    print(
        f"MLX p50 {direction} than ollama p50 by {pct:.0f}%"
        f" ({'no' if drift_count == 0 else drift_count} drift events)."
    )
    if mlx_wins:
        print("RECOMMENDATION: flip plist (Task 7).")
    else:
        print("RECOMMENDATION: keep ollama backend.")
    return mlx_wins


def main() -> None:
    parser = argparse.ArgumentParser(description="ollama vs MLX latency benchmark")
    parser.add_argument("--dry-run", action="store_true", help="2 queries, no service restart")
    args = parser.parse_args()

    queries = QUERIES[:2] if args.dry_run else QUERIES

    # Read original plist for guaranteed restore
    original_plist_bytes, _ = _read_plist()

    def restore_plist() -> None:
        with open(PLIST, "wb") as f:
            f.write(original_plist_bytes)
        if not args.dry_run:
            uid = os.getuid()
            _launchctl("bootout", f"gui/{uid}", PLIST)
            time.sleep(3)
            _launchctl("bootstrap", f"gui/{uid}", PLIST)

    try:
        # --- Round A: ollama ---
        print("\n=== Round A — ollama backend ===")
        if not args.dry_run:
            print("Restarting service (no RAG_WEB_USE_MLX)...")
            restart_web_with_env({"RAG_WEB_USE_MLX": None})
            wait_for_server()
        ollama_results = run_round("ollama", queries, args.dry_run)

        # --- Round B: MLX ---
        print("\n=== Round B — MLX backend ===")
        if not args.dry_run:
            print("Restarting service with RAG_WEB_USE_MLX=1 (cold load ~12s)...")
            restart_web_with_env({"RAG_WEB_USE_MLX": "1"})
            wait_for_server(timeout=90)
            # Extra wait for MLX model cold load
            print("Waiting for MLX model warm-up...")
            time.sleep(12)
        mlx_results = run_round("MLX", queries, args.dry_run)

    finally:
        print("\nRestoring original plist...")
        restore_plist()
        print("Plist restored.")

    print_table(ollama_results, mlx_results)


if __name__ == "__main__":
    main()
