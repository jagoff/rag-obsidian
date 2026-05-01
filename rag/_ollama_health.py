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
- RAG_LATENCY_WATCHDOG_THRESHOLD=3.0 (ratio recent/baseline para trigger;
  3.0 reemplazó 1.8 el 2026-05-01 — ver Trade-off abajo)
- RAG_LATENCY_WATCHDOG_MIN_RECENT=5 (cantidad mínima de queries en ventana para evaluar — evita false positives con poca data)
- RAG_LATENCY_WATCHDOG_COOLDOWN=1800 (segundos entre restarts permitidos — evita
  thrashing; 1800s=30min reemplazó 300s el 2026-05-01 — ver Trade-off abajo)

Trade-off threshold + cooldown (2026-05-01):
  El watchdog usa `pkill -9 -f ollama` que mata el daemon + sus runners,
  cerrando cualquier conexión TCP en vuelo. Si el web server está
  streamando una respuesta /api/chat al user en ese instante, el cliente
  ve "Server disconnected without sending a response" y la query muere.

  El loop autodestructivo observado 2026-05-01:
    1. Watchdog detecta p95_recent / p95_baseline >= 1.8x → restart.
    2. pkill mata ollama. Web server tiene 1+ requests en vuelo → mueren.
    3. Cold-load del modelo qwen2.5:7b post-restart toma 8-15s. La query
       que el user reintenta paga ese cold-load → total_ms ~100s.
    4. Esa query queda en `rag_queries.extra_json.total_ms`. Sube p95.
    5. 5 minutos después (cooldown=300s), watchdog corre check. p95 sigue
       arriba (queries lentas post-restart en window). Ratio >= 1.8 →
       restart OTRA VEZ.
    6. Goto 2.

  Confirmado en logs:
    [ollama-health-watchdog] restart_attempted: p95_recent=106503ms
      p95_baseline=52134ms ratio=2.04 reason=restart_ok=True
    [ollama-health-watchdog] restart_skipped_cooldown: ...
    [ollama-health-watchdog] restart_attempted: p95_recent=...

  Defaults nuevos (3.0x + 1800s):
    - 3.0x es threshold honesto para "ollama realmente roto". 1.8x es
      ruido de cold-load + cualquier query con tool-decide o reformulator
      que paga 30-60s extra.
    - 30 min de cooldown da tiempo suficiente para que las queries
      lentas post-restart salgan del window de 10 min y el p95 se
      estabilice antes del próximo check.

  Si el restart legítimamente NO funcionó (degradación real persistente
  como saturación de unified memory), el watchdog va a re-trigger 30 min
  después. Eso es correcto — un retry rápido (5 min) no agrega valor
  porque el cold-load apenas terminó y el sistema ya está warm de nuevo.
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

# In-flight chat counter — el web server llama begin_chat()/end_chat()
# alrededor de cada /api/chat handler. El watchdog NO restartea ollama
# mientras haya >=1 request activa, porque pkill -9 cierra la conexión
# TCP del runner y el cliente ve "Server disconnected without sending a
# response" — exactamente el bug que el watchdog dice arreglar.
# Observado 2026-05-01: query del user "podes recomendarme series"
# disparó retrieve OK con conf=0.385 sobre `Best Series all times.md`,
# pero el watchdog mató ollama mid-synthesis (`phase=synthesis
# exc=Server disconnected ttft_ms=8371`) → user vio "synthesis falló".
_in_flight_count = 0
_in_flight_lock = threading.Lock()


def begin_chat() -> None:
    """Marca que una request /api/chat empezó. Idempotent reentry-safe.

    Llamar al inicio del handler, en pareja con `end_chat()` en finally.
    El watchdog usa este counter para evitar restartar ollama mid-stream.
    """
    global _in_flight_count
    with _in_flight_lock:
        _in_flight_count += 1


def end_chat() -> None:
    """Marca que una request /api/chat terminó (éxito o error)."""
    global _in_flight_count
    with _in_flight_lock:
        _in_flight_count = max(0, _in_flight_count - 1)


def in_flight_chats() -> int:
    """Cantidad de chat handlers actualmente activos (testeable)."""
    with _in_flight_lock:
        return _in_flight_count


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
    """Reinicia el daemon ollama. Retorna (ok, detail).

    Detecta el deployment activo:
      1. homebrew.mxcl.ollama loaded → kickstart launchctl
      2. Ollama.app running (sin homebrew) → kill + open -a Ollama
      3. Neither → fallback: kill todos + open -a Ollama (best effort)

    Pre-2026-04-28 wave-4 hardcoded la ruta homebrew. Cuando el user
    deshabilitó el plist (porque el daemon duplicado causaba hangs), el
    watchdog quedó inútil. Auto-detección lo arregla.
    """
    try:
        uid = os.getuid()
    except Exception:
        return False, "no_uid"

    # 1) Detect homebrew deployment.
    try:
        check = subprocess.run(
            ["launchctl", "list"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        homebrew_loaded = (
            check.returncode == 0
            and "homebrew.mxcl.ollama" in check.stdout
        )
    except Exception:
        homebrew_loaded = False

    if homebrew_loaded:
        try:
            result = subprocess.run(
                ["launchctl", "kickstart", "-k",
                 f"gui/{uid}/homebrew.mxcl.ollama"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            if result.returncode == 0:
                return True, "kickstarted_homebrew"
        except Exception as exc:
            pass  # fallthrough to .app path

    # 2) Ollama.app path: kill all ollama procs then reopen.
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "ollama"],
            capture_output=True, timeout=5, check=False,
        )
        time.sleep(2)  # let processes exit cleanly
        subprocess.run(
            ["open", "-a", "Ollama"],
            capture_output=True, timeout=10, check=False,
        )
        return True, "restarted_app"
    except Exception as exc:
        return False, f"exception: {exc!r}"


def _latency_degradation_check(
    *,
    window_minutes: int = 10,
    threshold: float = 3.0,
    min_recent: int = 5,
    cooldown_seconds: int = 1800,
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
    # Need to restart — gate (a): NO matar ollama si hay /api/chat en
    # vuelo. pkill -9 cierra el socket del runner; si una respuesta SSE
    # está streaming a un cliente, el cliente ve "Server disconnected"
    # y la query muere. Skipear acá deja al request actual completar;
    # el próximo check (30s) re-evalúa con el counter en cero.
    n_in_flight = in_flight_chats()
    if n_in_flight > 0:
        out["action"] = "restart_skipped_in_flight"
        out["reason"] = f"in_flight={n_in_flight}"
        return out
    # Gate (b): cooldown.
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
            if result["action"] in (
                "restart_attempted",
                "restart_skipped_cooldown",
                "restart_skipped_in_flight",
            ):
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
            threshold = float(os.environ.get("RAG_LATENCY_WATCHDOG_THRESHOLD", "3.0"))
            min_recent = int(os.environ.get("RAG_LATENCY_WATCHDOG_MIN_RECENT", "5"))
            cooldown = int(os.environ.get("RAG_LATENCY_WATCHDOG_COOLDOWN", "1800"))
        except ValueError:
            interval, window, threshold, min_recent, cooldown = 30, 10, 3.0, 5, 1800
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
