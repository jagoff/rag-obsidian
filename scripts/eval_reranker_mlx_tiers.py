#!/usr/bin/env python3
"""Calibration sweep + auto-escalation: Qwen3-Reranker MLX tier finder.

Fase 2 del plan MLX-full-migration (Ola 10, 2026-05-07). Corre `rag eval`
con `RAG_RERANKER_BACKEND=mlx` para cada tier disponible (`qwen3-reranker:0.6b` →
`qwen3-reranker:4b` → `qwen3-reranker:8b`) y reporta cuál pasa el floor
(singles 56.60% / chains 72.00% lower-CI).

## Por qué auto-escalación

Qwen3-Reranker MLX devuelve P[0,1] post-sigmoid (vs logits unbounded de
bge). Sin sweep eval no sabemos si el 0.6B mantiene calidad — la
distribución bimodal puede dejar el threshold equivalente
(`CONFIDENCE_RERANK_MIN=0.015`) descalibrado. La user picked
escalar tier (no recalibrar threshold per-source). Este script
automatiza el sweep tier por tier hasta encontrar uno que apruebe.

## Cómo correr

```bash
.venv/bin/python scripts/eval_reranker_mlx_tiers.py --baseline-floor-singles 0.434 \\
    --baseline-floor-chains 0.560
```

(Los floors son lower-CI bounds. Default: 56.60% / 72.00% per `CLAUDE.md`
section "Eval baselines".)

## Output

- Tabla con resultado por tier (singles hit@5, chains hit@5, P95 latencia).
- Tier ganador (el más chico que pasa floor) → recomendación de cutover.
- Si ninguno pasa: reporta gap + sugiere recalibración manual o quedar en bge.

## NO hace cutover automático

Cambiar default `RAG_RERANKER_BACKEND` requiere edit de código (`rag/mlx_reranker.py`)
y bump de `_FILTER_VERSION`. Este script solo reporta; el operador hace el
cambio explícito tras revisar los números.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ── Constantes ─────────────────────────────────────────────────────────

# Default tiers a probar, en orden de costo creciente. Si el 0.6B pasa, ganamos
# el menor VRAM. El operador puede pasar `--tiers` para overridear.
DEFAULT_TIERS = ["qwen3-reranker:0.6b", "qwen3-reranker:4b", "qwen3-reranker:8b"]

# Floor por default desde CLAUDE.md (lower-CI bound, post-MLX 2026-05-05).
DEFAULT_FLOOR_SINGLES = 0.434
DEFAULT_FLOOR_CHAINS = 0.560
DEFAULT_MAX_P95_MS = 2500

EVAL_LOG_PATH = Path.home() / ".local/share/obsidian-rag/eval.jsonl"


# ── Eval invocation ────────────────────────────────────────────────────


def run_eval(*, tier: str, max_p95_ms: int) -> dict:
    """Ejecuta `rag eval --latency --max-p95-ms N` con `RAG_RERANKER_BACKEND=mlx`
    + tier seleccionado. Devuelve un dict con métricas parseadas del último
    record de `eval.jsonl`.
    """
    env = os.environ.copy()
    env["RAG_RERANKER_BACKEND"] = "mlx"
    env["RAG_MLX_RERANKER_MODEL"] = tier  # alias resuelto por `resolve_mlx_reranker_path`
    # Disable explore para mediciones determinísticas
    env.pop("RAG_EXPLORE", None)

    pre_lines = 0
    if EVAL_LOG_PATH.is_file():
        pre_lines = sum(1 for _ in EVAL_LOG_PATH.open())

    cmd = [
        sys.executable, "-m", "rag", "eval",
        "--latency",
        "--max-p95-ms", str(max_p95_ms),
    ]
    print(f"[eval] tier={tier} → {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, env=env)
    elapsed_min = (time.time() - t0) / 60
    print(f"[eval] tier={tier} exit_code={rc} elapsed={elapsed_min:.1f}min", flush=True)

    # Parse last entry de eval.jsonl que se haya agregado durante este run
    if not EVAL_LOG_PATH.is_file():
        return {"tier": tier, "exit_code": rc, "error": "no eval.jsonl"}
    with EVAL_LOG_PATH.open() as fh:
        all_lines = list(fh)
    new_lines = all_lines[pre_lines:]
    if not new_lines:
        return {"tier": tier, "exit_code": rc, "error": "no new eval.jsonl record"}
    try:
        record = json.loads(new_lines[-1])
    except Exception as exc:
        return {"tier": tier, "exit_code": rc, "error": f"parse fail: {exc}"}
    record["tier"] = tier
    record["exit_code"] = rc
    record["elapsed_min"] = round(elapsed_min, 2)
    return record


# ── Floor evaluation ───────────────────────────────────────────────────


def passes_floor(
    record: dict,
    *,
    floor_singles: float,
    floor_chains: float,
) -> tuple[bool, list[str]]:
    """Verifica que el record cumple los floors lower-CI.

    Returns (passes, reasons_failed). Si passes=True, reasons_failed=[].
    """
    reasons: list[str] = []
    if record.get("error"):
        return False, [f"error: {record['error']}"]
    if record.get("exit_code", 0) != 0:
        reasons.append(f"exit_code={record['exit_code']}")
    singles = record.get("singles") or {}
    chains = record.get("chains") or {}
    s_lo = (singles.get("hit5_ci") or [None, None])[0]
    c_lo = (chains.get("hit5_ci") or [None, None])[0]
    if s_lo is None or c_lo is None:
        reasons.append("hit5_ci missing")
        return False, reasons
    if s_lo < floor_singles:
        reasons.append(f"singles lower-CI {s_lo:.3f} < floor {floor_singles:.3f}")
    if c_lo < floor_chains:
        reasons.append(f"chains lower-CI {c_lo:.3f} < floor {floor_chains:.3f}")
    return len(reasons) == 0, reasons


# ── Reporting ──────────────────────────────────────────────────────────


def fmt_record(record: dict) -> str:
    """Una línea por record para la tabla final."""
    if record.get("error"):
        return f"  {record['tier']:24s}  ERROR: {record['error']}"
    s = record.get("singles") or {}
    c = record.get("chains") or {}
    p95 = record.get("p95_ms")
    p95_str = f"P95={p95:.0f}ms" if isinstance(p95, (int, float)) else "P95=?"
    s_h = (s.get("hit5") or 0.0) * 100
    s_lo = ((s.get("hit5_ci") or [0, 0])[0]) * 100
    c_h = (c.get("hit5") or 0.0) * 100
    c_lo = ((c.get("hit5_ci") or [0, 0])[0]) * 100
    return (
        f"  {record['tier']:24s}  "
        f"singles {s_h:.1f}% [lo={s_lo:.1f}%]  "
        f"chains {c_h:.1f}% [lo={c_lo:.1f}%]  "
        f"{p95_str}  ({record.get('elapsed_min', '?')}min)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep reranker MLX tiers + auto-escalation.",
    )
    parser.add_argument("--tiers", default=",".join(DEFAULT_TIERS),
                        help="Tiers a probar separados por coma (orden = ascendente cost)")
    parser.add_argument("--baseline-floor-singles", type=float,
                        default=DEFAULT_FLOOR_SINGLES,
                        help=f"Floor singles lower-CI (default: {DEFAULT_FLOOR_SINGLES})")
    parser.add_argument("--baseline-floor-chains", type=float,
                        default=DEFAULT_FLOOR_CHAINS,
                        help=f"Floor chains lower-CI (default: {DEFAULT_FLOOR_CHAINS})")
    parser.add_argument("--max-p95-ms", type=int, default=DEFAULT_MAX_P95_MS,
                        help=f"Latency gate (default: {DEFAULT_MAX_P95_MS}ms)")
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    if not tiers:
        print("ERROR: lista de tiers vacía", file=sys.stderr)
        raise SystemExit(2)

    print(f"[sweep] tiers: {tiers}")
    print(f"[sweep] floors: singles>={args.baseline_floor_singles:.3f} "
          f"chains>={args.baseline_floor_chains:.3f}")
    print(f"[sweep] max P95: {args.max_p95_ms}ms\n")

    records: list[dict] = []
    winner: str | None = None

    for tier in tiers:
        record = run_eval(tier=tier, max_p95_ms=args.max_p95_ms)
        records.append(record)
        ok, reasons = passes_floor(
            record,
            floor_singles=args.baseline_floor_singles,
            floor_chains=args.baseline_floor_chains,
        )
        if ok:
            print(f"\n[sweep] ✅ tier={tier} pasa floor — STOP escalación.\n")
            winner = tier
            break
        else:
            print(f"\n[sweep] ❌ tier={tier} NO pasa: {'; '.join(reasons)}\n")

    print("\n" + "=" * 80)
    print("RESUMEN SWEEP RERANKER MLX")
    print("=" * 80)
    for r in records:
        print(fmt_record(r))
    print()
    if winner:
        print(f"🏆 GANADOR: {winner}")
        print("\nPara hacer cutover manual, en `rag/mlx_reranker.py`:")
        print(f"  DEFAULT_MLX_RERANKER = MLX_RERANKER_ALIASES[\"{winner}\"]")
        print("\nY en `is_mlx_reranker_enabled()` cambiar default `torch` → `mlx`.")
        print("\nNo olvidar bumpear `_FILTER_VERSION` por cache invalidation.")
    else:
        print("⚠️  NINGÚN TIER PASA FLOOR.")
        print("\nOpciones:")
        print("  1. Quedar en bge-reranker-v2-m3 (status quo, ningún cambio).")
        print("  2. Recalibrar `CONFIDENCE_RERANK_MIN` per-source con sweep dedicado.")
        print("  3. Esperar mlx-community release de bge-reranker-v2-m3 MLX-port.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
