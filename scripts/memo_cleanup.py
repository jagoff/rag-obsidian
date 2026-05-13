#!/usr/bin/env python3
"""Cleanup periódico de memo — dead memorias + near-dupes.

Dead memorias: nunca usadas (recall_count = 0) + creadas hace > N días.
Near-dupes: pares con distancia < threshold (default 0.12).

Este script es read-safe: usa las funciones de memo_dashboard.py para
detectar, luego llama `memo delete` para borrar. Todas las acciones se
loguean para auditoría.

Usage:
    python scripts/memo_cleanup.py --dry-run
    python scripts/memo_cleanup.py --apply
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Agregar el repo al path para importar rag
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from web.memo_dashboard import (
    _ACTIVE_DAYS,
    _DEAD_MIN_AGE_DAYS,
    _DUPE_DIST_THRESHOLD,
    _build_recall_index,
    _compute_dupe_map,
    _history_db,
    _memvec_db,
    _memo_dir,
    _open_ro,
    _parse_iso,
    _vault_path,
)

# ── Configuración ────────────────────────────────────────────────────────────

# Dead memories: borrar si nunca usadas + creadas hace > X días
_DEAD_AGE_THRESHOLD_DAYS = 30  # más conservador que el _DEAD_MIN_AGE_DAYS de 2d

# Near-dupes: fusionar automáticamente si distancia < X (más estricto que el threshold de dashboard)
_AUTO_MERGE_DIST_THRESHOLD = 0.10  # muy conservador: solo casi-idénticas

# Safe delete: solo borrar dead memories si score < X (evitar borrar memorias de alta calidad)
_DEAD_MAX_SCORE = 40  # memorias con score >= 40 se preservan aunque estén dead

# ── Helpers ───────────────────────────────────────────────────────────────────


def _memo_delete(memo_id: str, dry_run: bool = False) -> bool:
    """Llama `memo delete` para borrar una memoria."""
    cmd = ["memo", "delete", "--yes", memo_id]
    if dry_run:
        print(f"  [DRY-RUN] Would delete: {memo_id}")
        return True
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True
        print(f"  [ERROR] memo delete {memo_id} failed: {result.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] memo delete {memo_id} timed out")
        return False
    except Exception as e:
        print(f"  [ERROR] memo delete {memo_id} exception: {e}")
        return False


def _detect_dead_memories(
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Detecta memorias que nunca fueron usadas + creadas hace > X días.

    Returns:
        {
            "candidates": list[dict],  # memorias candidatas a borrar
            "safe_to_delete": list[dict],  # memorias que pasan el filtro de score
            "preserved": list[dict],  # memorias dead pero con score alto (preservadas)
        }
    """
    from web.memo_dashboard import (
        _parse_tags,
        _read_body_meta,
        _score_memo,
        _TYPE_ACTIONABLE,
        _TYPE_INFORMATIONAL,
        _SIZE_MIN_OK,
        _SIZE_MAX_OK,
        _FRESH_DAYS,
        _STALE_DAYS,
        _W_ACTIONABLE,
        _W_TAGS,
        _W_SIZE,
        _W_FRESH,
        _W_UNIQUE,
    )

    mv = _open_ro(_memvec_db())
    if mv is None:
        return {"error": f"memvec.db not found at {_memvec_db()}"}

    try:
        all_rows = mv.execute(
            "SELECT id, title, type, tags, updated, created, path FROM meta"
        ).fetchall()

        # Build recall index
        recall_index = _build_recall_index([r["id"] for r in all_rows])

        # Detect dead memories
        now_utc = datetime.now(timezone.utc)
        cutoff_dead = now_utc - timedelta(days=_DEAD_AGE_THRESHOLD_DAYS)

        candidates = []
        safe_to_delete = []
        preserved = []

        for r in all_rows:
            rinfo = recall_index.get(r["id"], {})
            cnt = int(rinfo.get("count", 0))

            # Solo considerar memorias nunca usadas
            if cnt > 0:
                continue

            created_dt = _parse_iso(r["created"])
            if not created_dt:
                continue
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)

            # Solo considerar memorias creadas hace > X días
            if created_dt > cutoff_dead:
                continue

            # Calcular score para decidir si es seguro borrar
            tags = _parse_tags(r["tags"])
            type_ = (r["type"] or "").lower()
            updated_dt = _parse_iso(r["updated"])
            size, _ = _read_body_meta(r["path"])

            scored = _score_memo(
                type_=type_,
                tags=tags,
                body_size=size,
                updated_dt=updated_dt,
                in_dupe_cluster=False,  # no necesitamos esto para dead
            )

            candidate = {
                "id": r["id"],
                "title": r["title"] or "(sin título)",
                "type": type_,
                "created": r["created"],
                "age_days": (now_utc - created_dt).days,
                "score": scored["score"],
                "score_breakdown": scored["breakdown"],
                "path": r["path"],
            }

            candidates.append(candidate)

            if scored["score"] < _DEAD_MAX_SCORE:
                safe_to_delete.append(candidate)
            else:
                preserved.append(candidate)

    finally:
        mv.close()

    return {
        "candidates": candidates,
        "safe_to_delete": safe_to_delete,
        "preserved": preserved,
    }


def _detect_near_dupes(
    threshold: float = _AUTO_MERGE_DIST_THRESHOLD,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Detecta pares de near-dupes bajo un threshold.

    Returns:
        {
            "pairs": list[dict],  # pares de near-dupes
            "safe_to_merge": list[dict],  # pares que pasan el filtro de auto-merge
        }
    """
    dupe_map = _compute_dupe_map()
    pairs = dupe_map["pairs"]
    near_dupe_ids = dupe_map["near_dupe_ids"]

    # Filtrar pares por threshold más estricto para auto-merge
    safe_to_merge = []
    for a, b, dist in pairs:
        if dist < threshold:
            safe_to_merge.append({"a": a, "b": b, "distance": dist})

    return {
        "pairs": pairs,
        "safe_to_merge": safe_to_merge,
    }


def run_cleanup(
    dry_run: bool = False,
    verbose: bool = False,
    apply_dead: bool = True,
    apply_dupes: bool = False,  # off por default: merge es más destructivo
) -> dict[str, Any]:
    """Ejecuta cleanup de memo.

    Args:
        dry_run: Reportar sin modificar nada.
        verbose: Mostrar detalles.
        apply_dead: Aplicar cleanup de dead memories.
        apply_dupes: Aplicar cleanup de near-dupes (off por default).

    Returns:
        Dict con resultados del cleanup.
    """
    results = {
        "dry_run": dry_run,
        "dead": {"detected": 0, "deleted": 0, "preserved": 0, "errors": 0},
        "dupes": {"detected": 0, "merged": 0, "errors": 0},
    }

    # ── Dead memories ─────────────────────────────────────────────────────────
    if apply_dead:
        dead_report = _detect_dead_memories(dry_run=dry_run, verbose=verbose)
        if "error" in dead_report:
            results["dead"]["error"] = dead_report["error"]
        else:
            results["dead"]["detected"] = len(dead_report["candidates"])
            results["dead"]["preserved"] = len(dead_report["preserved"])
            safe_to_delete = dead_report["safe_to_delete"]

            print(f"\nDead memories: {len(safe_to_delete)} safe to delete, {len(dead_report['preserved'])} preserved (score >= {_DEAD_MAX_SCORE})")

            for candidate in safe_to_delete:
                if verbose:
                    print(f"  Deleting: {candidate['title']} (score={candidate['score']}, age={candidate['age_days']}d)")
                if _memo_delete(candidate["id"], dry_run=dry_run):
                    results["dead"]["deleted"] += 1
                else:
                    results["dead"]["errors"] += 1

    # ── Near-dupes ───────────────────────────────────────────────────────────
    if apply_dupes:
        dupe_report = _detect_near_dupes(
            threshold=_AUTO_MERGE_DIST_THRESHOLD,
            dry_run=dry_run,
            verbose=verbose,
        )
        results["dupes"]["detected"] = len(dupe_report["pairs"])
        safe_to_merge = dupe_report["safe_to_merge"]

        print(f"\nNear-dupes: {len(safe_to_merge)} safe to merge (dist < {_AUTO_MERGE_DIST_THRESHOLD})")

        # NOTA: Auto-merge de near-dupes requiere más lógica:
        # - Elegir cuál mantener (más reciente, más tags, mejor score)
        # - Fusionar tags del borrado al mantenido
        # - Borrar el duplicado
        # Por ahora solo reportamos; el usuario puede usar `memo consolidate` manualmente.
        print(f"  [INFO] Auto-merge not implemented yet. Use `memo consolidate` for manual review.")

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Cleanup periódico de memo — dead memorias + near-dupes")
    parser.add_argument("--dry-run", action="store_true", help="Reportar sin modificar nada")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mostrar detalles")
    parser.add_argument("--apply-dead", action="store_true", default=True, help="Aplicar cleanup de dead memories (default: True)")
    parser.add_argument("--apply-dupes", action="store_true", help="Aplicar cleanup de near-dupes (default: False)")
    args = parser.parse_args()

    print(f"Memo cleanup: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print(f"Memo dir: {_memo_dir()}")
    print(f"Vault path: {_vault_path()}")

    results = run_cleanup(
        dry_run=args.dry_run,
        verbose=args.verbose,
        apply_dead=args.apply_dead,
        apply_dupes=args.apply_dupes,
    )

    print(f"\nResults:")
    print(f"  Dead memories: {results['dead']['deleted']} deleted, {results['dead']['preserved']} preserved, {results['dead']['errors']} errors")
    print(f"  Near-dupes: {results['dupes']['detected']} detected, {results['dupes']['merged']} merged, {results['dupes']['errors']} errors")

    if results["dead"]["errors"] > 0 or results["dupes"]["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
