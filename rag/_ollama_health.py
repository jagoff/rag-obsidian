"""Latency degradation watchdog para ollama daemon.

Bug detectado en eval 2026-04-28: el daemon ollama se atasca después
de 5-7 queries tool-heavy consecutivas (KV cache fragmentation o
internal wedge). El restart manual (`brew services restart ollama`)
recupera la performance. Este módulo automatiza la detección + restart.

API pública (importable desde rag.__init__ via lazy import):
- `start_latency_degradation_watchdog(...)` — arranca el thread daemon.
- `_latency_degradation_check()` — single-shot check (testeable).
- `_percentile(vals, p)` — utility.

ENV VARS para tunear:
- RAG_LATENCY_WATCHDOG_DISABLE=1 → no arrancar el watchdog
- RAG_LATENCY_WATCHDOG_INTERVAL=30 (segundos entre checks)
- RAG_LATENCY_WATCHDOG_WINDOW_MIN=10 (minutos de ventana reciente)
- RAG_LATENCY_WATCHDOG_THRESHOLD=1.8 (ratio recent/baseline para trigger)
- RAG_LATENCY_WATCHDOG_MIN_RECENT=5 (cantidad mínima de queries en ventana para evaluar — evita false positives con poca data)
- RAG_LATENCY_WATCHDOG_COOLDOWN=300 (segundos entre restarts permitidos — evita thrashing)
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional


_started = False
_start_lock = threading.Lock()
_last_restart_ts: float = 0.0
_last_restart_lock = threading.Lock()


def _percentile(vals: list[float], p: float) -> float | None:
    """Calcula percentile p (0-100) via nearest-rank en vals ordenados."""
    if not vals:
        return None
    sorted_vals = sorted(float(v) for v in vals if v is not None)
    if not sorted_vals:
        return None
    idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * p / 100.0) - 1))
    return float(sorted_vals[idx])


def _read_recent_query_latencies(window_minutes: int) -> tuple[list[float], list[float]]:
    """Read p95-relevant query latencies from `rag_queries` SQLite.

    Returns `(recent_vals, baseline_vals)` donde recent es la ventana
    de los últimos `window_minutes` y baseline es 7 días.

    Si la tabla no existe o está vacía, retorna `([], [])`.
    """
    try:
        # Lazy import — no querer ciclo con rag/__init__.py
        from rag import _ragvec_state_conn  # type: ignore
    except Exception:
        return [], []
    try:
        with _ragvec_state_conn() as conn:
            recent_rows = conn.execute(
                """
                SELECT CAST(json_extract(extra_json, '$.total_ms') AS INTEGER)
                FROM rag_queries
                WHERE cmd LIKE 'web%'
                  AND ts >= datetime('now', ?)
                  AND json_extract(extra_json, '$.total_ms') IS NOT NULL
                """,
                (f"-{int(window_minutes)} minutes",),
            ).fetchall()
            baseline_rows = conn.execute(
                """
                SELECT CAST(json_extract(extra_json, '$.total_ms') AS INTEGER)
                FROM rag_queries
                WHERE cmd LIKE 'web%'
                  AND ts >= datetime('now', '-7 days')
                  AND json_extract(extra_json, '$.total_ms') IS NOT NULL
                """
            ).fetchall()
    except Exception:
        return [], []
    recent = [r[0] for r in recent_rows if r[0] is not None]
    baseline = [r[0] for r in baseline_rows if r[0] is not None]
    return recent, baseline


def _restart_ollama_daemon() -> tuple[bool, str]:
    """Reinicia el daemon ollama via launchctl. Retorna (ok, detail).

    Usa `launchctl kickstart -k gui/<uid>/homebrew.mxcl.ollama` que es
    más liviano que `brew services restart` (no toca el plist) y respeta
    KeepAlive=true del plist.
    """
    try:
        uid = os.getuid()
    except Exception:
        return False, "no_uid"
    try:
        result = subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/homebrew.mxcl.ollama"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode == 0:
            return True, "kickstarted"
        return False, f"rc={result.returncode}: {result.stderr.strip()[:80]}"
    except Exception as exc:
        return False, f"exception: {exc!r}"


def _latency_degradation_check(
    *,
    window_minutes: int = 10,
    threshold: float = 1.8,
    min_recent: int = 5,
    cooldown_seconds: int = 300,
) -> dict:
    """Single-shot check + maybe restart. Retorna dict con resultado.

    Estructura del retorno:
      {
        "action": "skip" | "ok" | "restart_attempted" | "restart_skipped_cooldown",
        "reason": str,
        "p95_recent_ms": int|None,
        "p95_baseline_ms": int|None,
        "ratio": float|None,
        "n_recent": int,
        "n_baseline": int,
      }

    Esta es la función testeable. El loop la llama cada N segundos.
    """
    out: dict = {
        "action": "skip",
        "reason": "",
        "p95_recent_ms": None,
        "p95_baseline_ms": None,
        "ratio": None,
        "n_recent": 0,
        "n_baseline": 0,
    }
    recent, baseline = _read_recent_query_latencies(window_minutes)
    out["n_recent"] = len(recent)
    out["n_baseline"] = len(baseline)
    if len(recent) < min_recent:
        out["reason"] = f"insufficient_recent (n={len(recent)} < {min_recent})"
        return out
    p95_recent = _percentile(recent, 95)
    p95_baseline = _percentile(baseline, 95) if baseline else None
    out["p95_recent_ms"] = int(p95_recent) if p95_recent else None
    out["p95_baseline_ms"] = int(p95_baseline) if p95_baseline else None
    if not p95_baseline or p95_baseline <= 0:
        out["reason"] = "no_baseline"
        return out
    if not p95_recent:
        out["reason"] = "no_p95_recent"
        return out
    ratio = p95_recent / p95_baseline
    out["ratio"] = round(ratio, 2)
    if ratio < threshold:
        out["action"] = "ok"
        out["reason"] = f"ratio={ratio:.2f} < threshold={threshold}"
        return out
    # Need to restart — check cooldown.
    global _last_restart_ts
    with _last_restart_lock:
        now_t = time.time()
        if now_t - _last_restart_ts < cooldown_seconds:
            out["action"] = "restart_skipped_cooldown"
            out["reason"] = f"last_restart_was_{int(now_t - _last_restart_ts)}s_ago"
            return out
        _last_restart_ts = now_t
    ok, detail = _restart_ollama_daemon()
    out["action"] = "restart_attempted"
    out["reason"] = f"restart_ok={ok} detail={detail}"
    return out


def _watchdog_loop(
    interval: int,
    window_minutes: int,
    threshold: float,
    min_recent: int,
    cooldown_seconds: int,
) -> None:
    while True:
        try:
            time.sleep(interval)
            result = _latency_degradation_check(
                window_minutes=window_minutes,
                threshold=threshold,
                min_recent=min_recent,
                cooldown_seconds=cooldown_seconds,
            )
            if result["action"] in ("restart_attempted", "restart_skipped_cooldown"):
                print(
                    f"[ollama-health-watchdog] {result['action']}: "
                    f"p95_recent={result['p95_recent_ms']}ms "
                    f"p95_baseline={result['p95_baseline_ms']}ms "
                    f"ratio={result['ratio']} reason={result['reason']}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[ollama-health-watchdog] error: {exc!r}", flush=True)


def start_latency_degradation_watchdog() -> bool:
    """Idempotent. Returns True if started, False if skipped/disabled."""
    global _started
    with _start_lock:
        if _started:
            return True
        if os.environ.get("RAG_LATENCY_WATCHDOG_DISABLE") == "1":
            return False
        try:
            interval = int(os.environ.get("RAG_LATENCY_WATCHDOG_INTERVAL", "30"))
            window = int(os.environ.get("RAG_LATENCY_WATCHDOG_WINDOW_MIN", "10"))
            threshold = float(os.environ.get("RAG_LATENCY_WATCHDOG_THRESHOLD", "1.8"))
            min_recent = int(os.environ.get("RAG_LATENCY_WATCHDOG_MIN_RECENT", "5"))
            cooldown = int(os.environ.get("RAG_LATENCY_WATCHDOG_COOLDOWN", "300"))
        except ValueError:
            interval, window, threshold, min_recent, cooldown = 30, 10, 1.8, 5, 300
        t = threading.Thread(
            target=_watchdog_loop,
            args=(interval, window, threshold, min_recent, cooldown),
            name="ollama-health-watchdog",
            daemon=True,
        )
        t.start()
        _started = True
        print(
            f"[ollama-health-watchdog] started "
            f"interval={interval}s window={window}min threshold={threshold}x "
            f"min_recent={min_recent} cooldown={cooldown}s",
            flush=True,
        )
        return True
