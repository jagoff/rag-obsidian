#!/usr/bin/env python3
"""Sugerir entries para queries.yaml desde feedback positivo en producción.

Bootstraps el golden set automáticamente: el feedback humano (rating=+1)
sobre query+top_path en `rag_feedback` es señal directa de que esa query
debería estar en el golden set apuntando a esa path.

Heurística:
  - Lee `rag_feedback` con `rating=+1` desde N días atrás (default 30).
  - Para cada row, extrae query + top path (`paths_json[0]`).
  - Filtra: path existe en vault + no es path "ruidoso" (Tokens.md,
    Links.md — aparecen como dummy en queries lookups).
  - Dedupea contra queries.yaml existente (por question lowercase).
  - Emite YAML snippet ready-to-paste a stdout, ordenado por más reciente.

NO escribe a queries.yaml directamente — el user revisa el snippet y
appendea a mano (o lo ignora). Append automático sería peligroso porque
una query ambigua + un único thumbs-up puede meter una entry mediocre
que infla la baseline sin agregar coverage real.

Uso típico:
  .venv/bin/python scripts/suggest_goldens.py --days 30

Para activar automáticamente como WhatsApp push semanal (ver ticket
"Active-learning bootstrap golden set"):
  .venv/bin/python scripts/suggest_goldens.py --days 7 --json | \\
    jq -r '.candidates[] | "  - question: \\"\\(.q)\\"\\n    expected:\\n      - \\(.expected)"'
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

DATA_DIR = Path.home() / ".local/share/obsidian-rag"
DB_PATH = DATA_DIR / "ragvec/telemetry.db"
GOLDEN_PATH = Path(__file__).resolve().parent.parent / "queries.yaml"
DEFAULT_VAULT = (
    Path.home()
    / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
)

# Paths que aparecen como ruido en lookups — top scores fallidos suelen
# meter Links.md / Tokens.md como filler. Skip para no contaminar el
# golden set.
NOISY_PATHS = frozenset({
    "03-Resources/Links.md",
    "03-Resources/Tokens.md",
})


def _existing_questions(yaml_path: Path) -> set[str]:
    """Set de preguntas (lowercase, stripped) ya en queries.yaml."""
    if not yaml_path.is_file():
        return set()
    try:
        data = yaml.safe_load(yaml_path.read_text())
    except Exception as exc:
        print(f"WARN: no pude parsear {yaml_path}: {exc}", file=sys.stderr)
        return set()
    qs = set()
    for entry in (data or {}).get("queries", []):
        q = (entry.get("question") or "").strip().lower()
        if q:
            qs.add(q)
    return qs


def _vault_path() -> Path:
    """Resolve vault root via env var u el default. No falla si no existe."""
    import os
    override = os.environ.get("OBSIDIAN_RAG_VAULT", "").strip()
    if override:
        return Path(override).expanduser()
    return DEFAULT_VAULT


def _harvest_candidates(days: int) -> list[dict]:
    """Lee rag_feedback, filtra positivos recientes, devuelve candidates."""
    if not DB_PATH.is_file():
        return []
    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat(
        timespec="seconds"
    )
    candidates: list[dict] = []
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
        try:
            rows = conn.execute(
                "SELECT ts, q, paths_json, extra_json "
                "FROM rag_feedback "
                "WHERE rating = 1 AND ts >= ? "
                "ORDER BY ts DESC",
                (cutoff_iso,),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"WARN: read rag_feedback falló: {exc}", file=sys.stderr)
        return []

    seen_pairs: set[tuple[str, str]] = set()
    for ts, q, paths_json, _extra_json in rows:
        if not q or not paths_json:
            continue
        try:
            paths = json.loads(paths_json)
        except (TypeError, ValueError):
            continue
        if not paths or not isinstance(paths, list):
            continue
        top_path = (paths[0] or "").strip() if isinstance(paths[0], str) else ""
        if not top_path or top_path in NOISY_PATHS:
            continue
        q_norm = q.strip().lower()
        pair = (q_norm, top_path)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        candidates.append({
            "ts": ts,
            "q": q.strip(),
            "expected": top_path,
            "paths_full": paths[:5],
        })
    return candidates


def _filter_against_vault_and_yaml(
    candidates: list[dict],
    vault: Path,
    existing: set[str],
) -> tuple[list[dict], dict]:
    """Filter candidates: path debe existir + question no debe duplicar."""
    keep: list[dict] = []
    drop_no_path = 0
    drop_dup_q = 0
    for c in candidates:
        q_norm = c["q"].strip().lower()
        if q_norm in existing:
            drop_dup_q += 1
            continue
        path = vault / c["expected"]
        if not path.is_file():
            drop_no_path += 1
            continue
        keep.append(c)
    summary = {
        "n_candidates_raw": len(candidates),
        "n_keep": len(keep),
        "n_drop_dup_q": drop_dup_q,
        "n_drop_path_missing": drop_no_path,
    }
    return keep, summary


def _emit_yaml(candidates: list[dict], limit: int) -> str:
    """Render a YAML snippet ready to paste into queries.yaml."""
    lines = []
    for c in candidates[:limit]:
        # Escape quotes pero mantener legible. queries.yaml usa double-quoted
        # strings consistentemente.
        q_esc = c["q"].replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  - question: "{q_esc}"')
        lines.append("    expected:")
        lines.append(f'      - {c["expected"]}')
        lines.append(f'    note: "auto-suggested {c["ts"][:10]} from rag_feedback rating=+1"')
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days", type=int, default=30,
        help="Ventana de días sobre rag_feedback (default 30).",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Máximo de candidates a emitir (default 10).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emitir summary + candidates en JSON (machine-readable).",
    )
    args = parser.parse_args()

    candidates = _harvest_candidates(args.days)
    vault = _vault_path()
    existing = _existing_questions(GOLDEN_PATH)
    keep, summary = _filter_against_vault_and_yaml(candidates, vault, existing)

    if args.json:
        out = {
            **summary,
            "vault": str(vault),
            "golden_path": str(GOLDEN_PATH),
            "candidates": keep[:args.limit],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    # Texto humano-friendly
    print(f"# active-learning suggest_goldens — last {args.days}d")
    print(f"# raw candidates: {summary['n_candidates_raw']} · "
          f"keep: {summary['n_keep']} · "
          f"drop_dup_q: {summary['n_drop_dup_q']} · "
          f"drop_path_missing: {summary['n_drop_path_missing']}")
    print(f"# vault: {vault}")
    print(f"# golden: {GOLDEN_PATH}")
    print()
    if not keep:
        print("# Sin candidates nuevos. (Ya están en queries.yaml o no hay "
              "feedback positivo reciente.)")
        return 0
    print("# Append a `queries:` en queries.yaml después de revisar:")
    print()
    print(_emit_yaml(keep, args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
