"""Housekeeping factories: maintenance / vault-cleanup / consolidate.

Cadena: vault-cleanup 02:00 → maintenance 04:00 (después del ciclo SQL
nightly) → consolidate Lunes 06:00. Limpieza estructural del vault y
del ragvec.db (WAL checkpoint, log-table rotation, conditional VACUUM).

Migrado de rag/plists/_legacy.py en Phase 3 commit 3 (2026-05-09).
"""
from __future__ import annotations

from rag.plists._render import _logs, _render_plist

__all__ = [
    "_consolidate_plist",
    "_maintenance_plist",
    "_vault_cleanup_plist",
]


def _maintenance_plist(rag_bin: str) -> str:
    """Daily housekeeping — every day at 04:00, after online-tune.

    Background (2026-04-21 hardening pass): with 15 services writing to
    ragvec.db concurrently (watch, serve, 4 ingesters, morning/today,
    etc.), the WAL grows unbounded between manual invocations. Observed
    in production: 126 MB WAL against a 206 MB main DB, none of the
    rotatable tables (rag_queries, rag_behavior, rag_contradictions)
    trimmed, auto_vacuum=0. Reads degrade as sqlite scans the WAL on
    each query; external backup tools that only copy `ragvec.db` miss
    126 MB of data.

    `rag maintenance` does: (1) WAL checkpoint(TRUNCATE) — compacts the
    -wal file back to KBs; (2) log-table rotation — deletes rows older
    than the configured TTL from the 6 rotatable telemetry tables;
    (3) conditional VACUUM — only if page_count*page_size exceeds
    last_vacuum_size by >500 MB (VACUUM takes an exclusive lock, so
    we gate it). See `_vec_wal_checkpoint` + `_rotate_telemetry_logs`
    in rag.py for the implementation.

    Scheduled at 04:00 specifically so online-tune (03:30) has fully
    released its SQL connections before VACUUM can acquire the
    exclusive lock. `RunAtLoad=false` — first run happens at the next
    04:00, so `rag setup` doesn't block on a potentially-long VACUUM.
    """
    out, err = _logs("maintenance")
    return _render_plist({
        "label": "com.fer.obsidian-rag-maintenance",
        "program_arguments": [rag_bin, "maintenance"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
        },
        "schedule": {
            "calendar": {"Hour": 4, "Minute": 0},
        },
        "run_at_load": False,
        "keep_alive": False,
        "process_type": "Background",
        "low_priority_io": True,
        "stdout_path": out,
        "stderr_path": err,
    })


def _vault_cleanup_plist(rag_bin: str) -> str:
    """Daily vault transient-folder cleanup — every day at 02:00.

    Mueve archivos viejos en `99-obsidian/99-AI/{{tmp,
    conversations, sessions, plans, system, reviews}}/` y wipe completo
    de `Wiki/` al `.trash/` del vault. `memory/` y `skills/` están
    explícitamente protegidos. Reversible: los archivos quedan en
    `<vault>/.trash/` hasta que el user vacíe la papelera de Obsidian.

    Schedule a las 02:00 — antes del ciclo de housekeeping del SQL
    (auto-harvest 03:00 → implicit 03:25 → online-tune 03:30 →
    maintenance 04:00 → calibrate 04:30) para no competir por I/O en
    iCloud durante esa ventana. Solo toca FS del vault, no abre
    ragvec.db, así que no hay race con el ciclo SQL.

    `RunAtLoad=false` para que `rag setup` no dispare un cleanup
    inmediato — la primera corrida es a la próxima 02:00 AM, dándole
    al user tiempo de auditar el plist + revertir si quiere.

    Lógica completa en `scripts/cleanup_vault_transient.py` — TTLs y
    políticas por carpeta documentados ahí. Para auditar qué se va a
    borrar sin tocar nada: `rag vault-cleanup --dry-run --json`.
    """
    out, err = _logs("vault-cleanup")
    return _render_plist({
        "label": "com.fer.obsidian-rag-vault-cleanup",
        "program_arguments": [rag_bin, "vault-cleanup"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
        },
        "schedule": {
            "calendar": {"Hour": 2, "Minute": 0},
        },
        "run_at_load": False,
        "keep_alive": False,
        "process_type": "Background",
        "low_priority_io": True,
        "stdout_path": out,
        "stderr_path": err,
    })


def _consolidate_plist(rag_bin: str) -> str:
    """Weekly episodic-memory consolidation — Mondays 06:00 local. Promotes
    recurring conversation clusters from
    99-obsidian/99-AI/conversations/ to PARA and
    archives the originals (see plans/episodic-memory.md Phase 2)."""
    out, err = _logs("consolidate")
    return _render_plist({
        "label": "com.fer.obsidian-rag-consolidate",
        "program_arguments": [rag_bin, "consolidate", "--apply"],
        "env": {
            "NO_COLOR": "1",
            "TERM": "dumb",
            "RAG_STATE_SQL": "1",
            "RAG_LLM_BACKEND": "mlx",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "schedule": {
            "calendar": {"Weekday": 1, "Hour": 6, "Minute": 0},
        },
        "run_at_load": False,
        "keep_alive": False,
        "process_type": "Background",
        "low_priority_io": True,
        "stdout_path": out,
        "stderr_path": err,
    })
