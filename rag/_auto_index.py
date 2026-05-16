"""Auto-index — detección + reindex incremental sin que el user corra `rag index`.

Phase 2c de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer el auto-index orchestrator + state persistence desde
`rag/__init__.py`.

## Estrategia

Dos casos:
  1. Vault vacío en scope → first-time full index (silent excepto progreso).
  2. Vault con contenido + archivos modificados desde último check →
     reindex incremental SOLO de esos archivos. mtime es el filtro
     barato; hash gate dentro de `_index_single_file` evita reembedding
     si nada cambió.

Persistimos `last_check_at` por vault para no leer todos los archivos
cada vez que arranca el chat.

## Lazy imports

Este módulo depende de muchos symbols de `rag/__init__.py`:
- `_with_vault` (context manager con mutación de globals — DEBE quedar
  en `__init__.py` porque usa `globals()` que apunta a su propio
  namespace).
- `_index_single_file`, `is_excluded`, `get_db_for`,
  `_collection_write_lock`, `LockHeldError`, `_log_collection_op`,
  `_silent_log`, `_wiki_ingest_indexed_files`.

Lazy imports adentro de `auto_index_vault` evitan circular import.

## Re-export

`rag/__init__.py` hace `from rag._auto_index import *  # noqa`.
Preserva 100% compat con call sites históricos
(`rag.auto_index_vault(path)`, `rag.AUTO_INDEX_STATE_PATH`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = [
    "AUTO_INDEX_STATE_PATH",
    "_auto_index_state_load",
    "_auto_index_state_save",
    "_vault_state_key",
    "auto_index_vault",
]


AUTO_INDEX_STATE_PATH = Path.home() / ".local/share/obsidian-rag/auto_index_state.json"


def _auto_index_state_load() -> dict:
    """Lee el state de auto-index. Estructura: {vault_hash: last_check_ts}.
    Tolerante a archivo faltante o corrupto — devuelve {} en cualquier error.
    """
    try:
        import rag as _rag  # noqa: PLC0415
        path = getattr(_rag, "AUTO_INDEX_STATE_PATH", AUTO_INDEX_STATE_PATH)
    except Exception:
        path = AUTO_INDEX_STATE_PATH
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _auto_index_state_save(state: dict) -> None:
    from rag import _silent_log  # noqa: PLC0415
    try:
        import rag as _rag  # noqa: PLC0415
        path = getattr(_rag, "AUTO_INDEX_STATE_PATH", AUTO_INDEX_STATE_PATH)
    except Exception:
        path = AUTO_INDEX_STATE_PATH

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(state, indent=2), encoding="utf-8",
        )
    except Exception as exc:
        _silent_log("auto_index_state_save", exc)


def _vault_state_key(vault_path: Path) -> str:
    """Clave estable para el state — sha256[:16] del path absoluto."""
    return hashlib.sha256(str(vault_path.resolve()).encode()).hexdigest()[:16]


def auto_index_vault(vault_path: Path) -> dict:
    """Detecta cambios en `vault_path` y reindexa lo necesario. Retorna
    {scanned, indexed, removed, kind, took_ms} donde kind ∈
    {"first_time", "incremental", "no_changes", "skipped_lock_held"}.

    Estrategia:
      - Vault vacío → first_time, escanea todo + indexa.
      - Vault con contenido → mtime-based incremental: solo lee archivos
        cuyo mtime > last_check_at. Para cada uno, _index_single_file
        ya hace su propio hash gate (skip si content no cambió de verdad).
      - Limpia orphans (archivos en el índice que ya no están en disco).
      - Actualiza last_check_at al final.

    If the collection write lock is held by another process (e.g. rag index
    --reset), skips this cycle immediately and returns kind="skipped_lock_held".
    Watch will trigger again on the next file change; chat has already started.
    """
    import time as _t  # noqa: PLC0415

    from rag import (  # noqa: PLC0415
        LockHeldError,
        _collection_write_lock,
        _index_single_file,
        _log_collection_op,
        _silent_log,
        _wiki_ingest_indexed_files,
        _with_vault,
        get_db_for,
        is_excluded,
    )

    t0 = _t.perf_counter()

    # Non-blocking lock probe: if a reset or another heavy writer holds the
    # lock, skip this entire auto-index cycle rather than racing or waiting.
    # Individual _index_single_file calls acquire the lock themselves (10s
    # timeout), so watch-triggered single-file writes still serialize correctly.
    try:
        with _collection_write_lock(blocking=False):
            pass  # just probe; real acquire happens inside _index_single_file
    except LockHeldError:
        _log_collection_op("auto_index_skipped", str(vault_path), {"reason": "write_lock_held"})
        return {
            "scanned": 0,
            "indexed": 0,
            "removed": 0,
            "kind": "skipped_lock_held",
            "took_ms": int((_t.perf_counter() - t0) * 1000),
        }
    state = _auto_index_state_load()
    key = _vault_state_key(vault_path)
    last_check = float(state.get(key, 0.0))

    with _with_vault(vault_path):
        # Usar get_db_for(vault_path) en vez de get_db() — el singleton de
        # get_db() cachea por DB_PATH e ignora COLLECTION_NAME del swap,
        # entonces auto_index de un vault secundario terminaba operando
        # sobre la colección del vault default. La orphan-cleanup veía
        # on_disk del vault X contra indexed_files del vault default y
        # borraba todo lo que no matcheara (bug: drenaba home al autoindexar
        # work).
        col = get_db_for(vault_path)
        first_time = col.count() == 0

        # Listar md files (rglob es rápido en APFS, ~50ms para 500 archivos).
        # Audit 2026-04-26: skip symlinks para evitar ciclos (a→b→a).
        # Python 3.13+ rglob soporta `recurse_symlinks=False`. En 3.13 o
        # menor caemos al manual `is_symlink()` check.
        md_files: list[Path] = []
        try:
            _md_iter = vault_path.rglob("*.md", recurse_symlinks=False)
        except TypeError:
            _md_iter = vault_path.rglob("*.md")
        for p in _md_iter:
            try:
                if p.is_symlink():
                    continue
                rel = p.relative_to(vault_path)
            except ValueError:
                continue
            except OSError:
                continue
            if is_excluded(str(rel)):
                continue
            md_files.append(p)

        # Bootstrap: si la colección ya tiene chunks pero last_check es 0
        # (estado no existía o fue borrado), asumimos que el index está al
        # día y solo persistimos el timestamp actual. Sin esto, la primera
        # corrida de auto_index escanea TODOS los archivos via
        # _index_single_file (col.get por archivo = ~25s en 500 notas) —
        # dispara un hang aparente al arrancar chat.
        if not first_time and last_check == 0.0:
            state[key] = _t.time()
            _auto_index_state_save(state)
            return {
                "scanned": len(md_files),
                "indexed": 0,
                "removed": 0,
                "kind": "no_changes",
                "took_ms": int((_t.perf_counter() - t0) * 1000),
            }

        # mtime-filter SOLO si no es first_time (ahí hay que indexar todo).
        if first_time:
            candidates = md_files
        else:
            candidates = [p for p in md_files if p.stat().st_mtime > last_check]

        indexed = 0
        indexed_paths: list[Path] = []
        for p in candidates:
            try:
                status = _index_single_file(col, p, skip_contradict=True)
            except Exception as _exc:
                from rag import _silent_log  # noqa: PLC0415
                _silent_log("auto_index_single_file_failed", _exc)
                continue
            if status == "indexed":
                indexed += 1
                indexed_paths.append(p)

        # Orphans: archivos en el índice que ya no están en disco. Solo
        # vale chequear cuando el vault ya estaba indexado (skip first_time).
        # SAFETY GATE (2026-04-18): iCloud sync puede hacer que rglob devuelva
        # 0 archivos temporariamente — sin este guard, TODOS los chunks se
        # marcan como orphans y se borra la colección entera. Bailout si
        # on_disk está vacío (no hay archivos visibles) o si on_disk cubre
        # <10% del set indexado (sync parcial). Logeamos y dejamos que la
        # próxima corrida limpie cuando sync haya terminado.
        removed = 0
        if not first_time:
            on_disk = {str(p.relative_to(vault_path)) for p in md_files}
            existing = col.get(include=["metadatas"])
            indexed_files = {m.get("file", "") for m in existing["metadatas"]}
            indexed_files.discard("")
            if not on_disk:
                print(
                    f"[auto_index] {vault_path.name}: rglob 0 files — "
                    f"skip orphan cleanup (iCloud sync in progress?)",
                    flush=True,
                )
                orphans = set()
            elif indexed_files and len(on_disk) < 0.1 * len(indexed_files):
                print(
                    f"[auto_index] {vault_path.name}: on_disk={len(on_disk)} "
                    f"vs indexed={len(indexed_files)} — too many orphans, "
                    f"skip cleanup (likely sync glitch)",
                    flush=True,
                )
                orphans = set()
            else:
                orphans = indexed_files - on_disk
            # Double-check per-orphan via is_file() inside the write lock.
            # rglob on iCloud can return a partial listing during sync (file
            # exists but momentarily invisible) — without this guard that
            # single missing path gets deleted from the index. is_file() on
            # the exact path hits macOS's fileProvider and reliably reflects
            # the real on-disk state.
            if orphans:
                with _collection_write_lock():
                    confirmed = [o for o in orphans if not (vault_path / o).is_file()]
                    if confirmed:
                        stale = col.get(where={"file": {"$in": confirmed}}, include=[])
                        if stale["ids"]:
                            col.delete(ids=stale["ids"])
                            removed += len(stale["ids"])

    state[key] = _t.time()
    _auto_index_state_save(state)

    # ── LLM Wiki layer hook ──────────────────────────────────────────────────
    # After the sqlite-vec indexing settles, compile a wiki page per newly
    # indexed source and refresh Wiki/index.md. Silent-fail end-to-end — the
    # indexing above already succeeded; if the helper model is wedged or
    # something else blows up, we log to silent_errors.jsonl and return the
    # usual stats untouched. Kill-switch: OBSIDIAN_RAG_WIKI_ENABLED=0.
    if indexed_paths:
        try:
            _wiki_ingest_indexed_files(vault_path, indexed_paths)
        except Exception as exc:
            _silent_log("auto_index_wiki_hook", exc)

    if first_time and indexed > 0:
        kind = "first_time"
    elif indexed == 0 and removed == 0:
        kind = "no_changes"
    else:
        kind = "incremental"

    return {
        "scanned": len(md_files),
        "indexed": indexed,
        "removed": removed,
        "kind": kind,
        "took_ms": int((_t.perf_counter() - t0) * 1000),
    }
