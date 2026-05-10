"""Maintenance operations — extracted from `rag/__init__.py` (Wave 8 split, 2026-05-10).

Housekeeping unificado: rotate logs JSONL, prune caches obsoletas, cleanup
tmp files, prune URL/feedback orphans, find/prune orphan collections,
run WAL checkpoints. Safe to run unattended (cron/launchd).

## Public API

Rotation:
- `_rotate_jsonl(path, max_bytes)` → str | None

Pruning (return count of items removed):
- `_prune_context_cache(col)`
- `_prune_synthetic_q_cache(col)`
- `_prune_ignored_notes(vault)`
- `_prune_auto_index_state()`
- `_prune_filing_batches(ttl_days)`
- `_cleanup_tmp_files()` (no-arg)
- `_cleanup_chat_uploads(*, ttl_days)` → dict
- `_check_embedder_health()` → dict
- `_prune_url_orphans(vault)`
- `_prune_feedback_orphans(vault)`

Orphan detection / cleanup:
- `_find_orphan_collections()` → list[str]
- `_find_orphan_segment_dirs()` → list[(Path, int)]
- `_prune_orphan_segment_dirs(dry_run)` → dict

WAL checkpoints (post-split 2026-04-21 — ragvec.db + telemetry.db
each have their own WAL):
- `_wal_checkpoint_for(sqlite_path, *, dry_run)` → dict
- `_vec_wal_checkpoint(dry_run)` → dict
- `_telemetry_wal_checkpoint(dry_run)` → dict

## Pattern

Las funciones acceden a colaboradores (paths, helpers, globals con
dirty flags) via `import rag` deferred. Tests
(`tests/test_maintenance_sql.py`, `tests/test_vec_cleanup.py`,
`tests/test_chat_uploads_cleanup.py`) hacen
`monkeypatch.setattr(rag, "DB_PATH", ...)` y todos los patches
funcionan porque `rag.DB_PATH` se re-resuelve en cada invocación.

## Constantes

- `_JSONL_ROTATE_BYTES` (10 MB) — threshold rotación logs
- `_FILING_BATCH_TTL_DAYS` (90) — TTL filing batches

## Dependencies en `rag.__init__` (deferred)

- Cache state: `_load_context_cache`, `_save_context_cache`,
  `_context_cache_lock`, `_context_cache_dirty`,
  `_load_synthetic_q_cache`, `_save_synthetic_q_cache`,
  `_synthetic_q_cache_lock`, `_synthetic_q_cache_dirty`
- Vault: `VAULT_PATH`, `_load_vaults_config`, `is_excluded`,
  `load_ignored_paths`, `save_ignored_paths`
- Auto-index: `_auto_index_state_load`, `_auto_index_state_save`,
  `_vault_state_key`
- Paths: `FILING_BATCHES_DIR`, `SESSIONS_DIR`, `FEEDBACK_PATH`,
  `DB_PATH`, `_TELEMETRY_DB_FILENAME`
- DB: `SqliteVecClient`, `get_urls_db`, `_COLLECTION_BASE`,
  `_URLS_COLLECTION_BASE`, `_collection_name_for_vault`,
  `_urls_collection_name_for_vault`, `COLLECTION_NAME`,
  `URLS_COLLECTION_NAME`, `_log_collection_op`
- Embed: `_get_local_embedder`, `EMBED_MODEL`
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

__all__ = [
    "_JSONL_ROTATE_BYTES",
    "_FILING_BATCH_TTL_DAYS",
    "_rotate_jsonl",
    "_prune_context_cache",
    "_prune_synthetic_q_cache",
    "_prune_ignored_notes",
    "_prune_auto_index_state",
    "_prune_filing_batches",
    "_cleanup_tmp_files",
    "_cleanup_chat_uploads",
    "_check_embedder_health",
    "_prune_url_orphans",
    "_prune_feedback_orphans",
    "_find_orphan_collections",
    "_find_orphan_segment_dirs",
    "_prune_orphan_segment_dirs",
    "_wal_checkpoint_for",
    "_vec_wal_checkpoint",
    "_telemetry_wal_checkpoint",
]


_JSONL_ROTATE_BYTES = 10 * 1024 * 1024  # 10 MB threshold for log rotation
_FILING_BATCH_TTL_DAYS = 90


def _rotate_jsonl(path: Path, max_bytes: int = _JSONL_ROTATE_BYTES) -> str | None:
    """Rotate a JSONL file if it exceeds max_bytes. Keeps the most recent half
    of the lines and moves the rest to path.1. Returns description or None."""
    if not path.is_file():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size < max_bytes:
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return None
    keep = len(lines) // 2
    old = path.with_suffix(path.suffix + ".1")
    old.write_text("\n".join(lines[:len(lines) - keep]) + "\n", encoding="utf-8")
    path.write_text("\n".join(lines[len(lines) - keep:]) + "\n", encoding="utf-8")
    return f"{len(lines) - keep} lines kept, {len(lines) - keep} archived → {old.name} ({size / 1024:.0f} KB → {path.stat().st_size / 1024:.0f} KB)"


def _prune_context_cache(col) -> int:
    """Remove context_summaries entries whose file hash is no longer in the index."""
    import rag  # noqa: PLC0415

    cache = rag._load_context_cache()
    if not cache:
        return 0
    existing = col.get(include=["metadatas"])
    live_hashes: set[str] = set()
    for m in existing["metadatas"]:
        h = m.get("hash", "")
        if h:
            live_hashes.add(h)
    with rag._context_cache_lock:
        stale = [k for k in cache if k not in live_hashes]
        for k in stale:
            del cache[k]
        if stale:
            rag._context_cache_dirty = True
    if stale:
        rag._save_context_cache()
    return len(stale)


def _prune_synthetic_q_cache(col) -> int:
    """Remove synthetic_questions entries whose file hash is no longer in the index."""
    import rag  # noqa: PLC0415

    cache = rag._load_synthetic_q_cache()
    if not cache:
        return 0
    existing = col.get(include=["metadatas"])
    live_hashes: set[str] = set()
    for m in existing["metadatas"]:
        h = m.get("hash", "")
        if h:
            live_hashes.add(h)
    with rag._synthetic_q_cache_lock:
        stale = [k for k in cache if k not in live_hashes]
        for k in stale:
            del cache[k]
        if stale:
            rag._synthetic_q_cache_dirty = True
    if stale:
        rag._save_synthetic_q_cache()
    return len(stale)


def _prune_ignored_notes(vault: Path) -> int:
    """Remove entries from ignored_notes.json whose files no longer exist."""
    import rag  # noqa: PLC0415

    paths = rag.load_ignored_paths()
    if not paths:
        return 0
    stale = {p for p in paths if not (vault / p).exists()}
    if stale:
        rag.save_ignored_paths(paths - stale)
    return len(stale)


def _prune_auto_index_state() -> int:
    """Remove auto_index_state entries for vaults not in the registry or default."""
    import rag  # noqa: PLC0415

    state = rag._auto_index_state_load()
    if not state:
        return 0
    valid_keys = {rag._vault_state_key(rag.VAULT_PATH)}
    try:
        vcfg = rag._load_vaults_config()
        for vpath in vcfg["vaults"].values():
            valid_keys.add(rag._vault_state_key(Path(vpath)))
    except Exception:
        pass
    stale = [k for k in state if k not in valid_keys]
    if stale:
        for k in stale:
            del state[k]
        rag._auto_index_state_save(state)
    return len(stale)


def _prune_filing_batches(ttl_days: int = _FILING_BATCH_TTL_DAYS) -> int:
    """Delete filing batch files older than ttl_days."""
    import rag  # noqa: PLC0415

    if not rag.FILING_BATCHES_DIR.is_dir():
        return 0
    cutoff = time.time() - ttl_days * 86400
    removed = 0
    for f in rag.FILING_BATCHES_DIR.glob("*.jsonl"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def _cleanup_tmp_files() -> int:
    """Remove orphan .tmp files from state directories (crashed writes)."""
    import rag  # noqa: PLC0415

    state_dir = Path.home() / ".local/share/obsidian-rag"
    removed = 0
    if not state_dir.is_dir():
        return 0
    for tmp in state_dir.glob("*.tmp"):
        try:
            # Only remove if older than 1 hour (avoid racing with live writes)
            if time.time() - tmp.stat().st_mtime > 3600:
                tmp.unlink()
                removed += 1
        except OSError:
            continue
    # Also check sessions dir
    if rag.SESSIONS_DIR.is_dir():
        for tmp in rag.SESSIONS_DIR.glob("*.tmp"):
            try:
                if time.time() - tmp.stat().st_mtime > 3600:
                    tmp.unlink()
                    removed += 1
            except OSError:
                continue
    return removed


def _cleanup_chat_uploads(*, ttl_days: int | None = None) -> dict:
    """Borra imágenes en chat-uploads/ con mtime > ttl_days.

    Las imágenes se guardan en ``~/.local/share/obsidian-rag/chat-uploads/``
    cuando el user las sube via ``/api/chat/upload-image``. Sin TTL, el dir
    crece sin bound (audit 2026-04-25 R2-Security #6). Idempotente — si
    una imagen ya fue copiada al vault con un naming cronológico
    (``00-Inbox/<timestamp>-<hash8>.ext``), borrar el cache no pierde data.

    Args:
      ttl_days: días de antigüedad mínima para borrar. Default lee la env
        var ``RAG_CHAT_UPLOADS_TTL_DAYS``, fallback 30.

    Returns:
      ``{"deleted": N, "bytes_freed": M, "errors": [...]}``. Nunca tira —
      si el dir no existe devuelve ceros; errores per-archivo se acumulan
      en ``errors`` para que el caller pueda loguearlos sin abortar la
      maintenance entera.
    """
    if ttl_days is None:
        ttl_days = int(os.environ.get("RAG_CHAT_UPLOADS_TTL_DAYS", "30"))
    upload_dir = Path.home() / ".local" / "share" / "obsidian-rag" / "chat-uploads"
    if not upload_dir.is_dir():
        return {"deleted": 0, "bytes_freed": 0, "errors": []}

    now = time.time()
    cutoff = now - (ttl_days * 86400)
    deleted = 0
    bytes_freed = 0
    errors: list[str] = []
    for fpath in upload_dir.iterdir():
        if not fpath.is_file():
            continue
        try:
            mtime = fpath.stat().st_mtime
            if mtime < cutoff:
                size = fpath.stat().st_size
                fpath.unlink()
                deleted += 1
                bytes_freed += size
        except Exception as e:  # noqa: BLE001 - best-effort, capturamos per-file
            errors.append(f"{fpath.name}: {e}")

    return {"deleted": deleted, "bytes_freed": bytes_freed, "errors": errors}


def _check_embedder_health() -> dict:
    """Verify the in-process embedder loads OK.

    Devuelve `{EMBED_MODEL: "ok"}` si carga, `{EMBED_MODEL: "missing"}` si
    `_get_local_embedder()` retorna None (HF cache no presente, MPS unavailable,
    etc), `{"error": ...}` si la carga levanta excepción.
    """
    import rag  # noqa: PLC0415

    try:
        model = rag._get_local_embedder()
    except Exception as e:
        return {"error": str(e)}
    return {rag.EMBED_MODEL: "ok" if model is not None else "missing"}


def _prune_url_orphans(vault: Path) -> int:
    """Remove URL collection entries for files no longer on disk."""
    import rag  # noqa: PLC0415

    try:
        col_urls = rag.get_urls_db()
        all_urls = col_urls.get(include=["metadatas"])
    except Exception:
        return 0
    if not all_urls["ids"]:
        return 0
    on_disk = {
        str(p.relative_to(vault))
        for p in vault.rglob("*.md")
        if not rag.is_excluded(str(p.relative_to(vault)))
    }
    orphan_ids = [
        id_ for id_, meta in zip(all_urls["ids"], all_urls["metadatas"])
        if meta.get("file", "") not in on_disk
    ]
    if orphan_ids:
        # Chunk deletes for predictable memory usage.
        for i in range(0, len(orphan_ids), 5000):
            col_urls.delete(ids=orphan_ids[i:i + 5000])
    return len(orphan_ids)


def _prune_feedback_orphans(vault: Path) -> int:
    """Remove feedback.jsonl entries whose paths all point to deleted notes.
    Rewrites the file in place, preserving entries that have at least one live path."""
    import rag  # noqa: PLC0415

    if not rag.FEEDBACK_PATH.is_file():
        return 0
    on_disk = {
        str(p.relative_to(vault))
        for p in vault.rglob("*.md")
        if not rag.is_excluded(str(p.relative_to(vault)))
    }
    lines = rag.FEEDBACK_PATH.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    removed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            kept.append(line)
            continue
        paths = ev.get("paths") or []
        # Keep if any path still exists, or if it's a session-scope rating (no paths)
        if not paths or any(p in on_disk for p in paths):
            kept.append(line)
        else:
            removed += 1
    if removed:
        rag.FEEDBACK_PATH.write_text("\n".join(kept) + "\n" if kept else "", encoding="utf-8")
    return removed


def _find_orphan_collections() -> list[str]:
    """Detect sqlite-vec collections from old schema versions or removed vaults.

    Safety invariants
    -----------------
    1. Base names are never orphaned: _COLLECTION_BASE ("obsidian_notes_v9")
       and _URLS_COLLECTION_BASE ("obsidian_urls_v1") are unconditionally added
       to `known` so that if this process was launched with OBSIDIAN_RAG_VAULT
       pointing at a non-default vault, the default vault's unsuffixed collection
       is still protected and not classified as an orphan.

    2. No recency guard is implemented — sqlite-vec has no creation timestamp in
       schema.

    3. Audit: every candidate orphan is logged via _log_collection_op before
       being returned to the caller (which may delete it).
    """
    import rag  # noqa: PLC0415

    try:
        client = rag.SqliteVecClient(path=str(rag.DB_PATH))
        all_cols = [c.name for c in client.list_collections()]
    except Exception:
        return []
    known = {rag.COLLECTION_NAME, rag.URLS_COLLECTION_NAME}
    # Unconditionally protect the base names for the default vault.
    # If this process was started with OBSIDIAN_RAG_VAULT set to a work vault,
    # COLLECTION_NAME / URLS_COLLECTION_NAME resolve to the work-vault suffixed
    # names.  The unsuffixed default-vault collections would then fall through
    # as unknown and get wrongly classified as orphans without this guard.
    known.add(rag._COLLECTION_BASE)
    known.add(rag._URLS_COLLECTION_BASE)
    # Also include collections for registered vaults
    try:
        vcfg = rag._load_vaults_config()
        for vpath in vcfg["vaults"].values():
            vp = Path(vpath)
            known.add(rag._collection_name_for_vault(vp))
            known.add(rag._urls_collection_name_for_vault(vp))
    except Exception:
        pass
    orphans = [c for c in all_cols if c not in known]
    # Audit every candidate before returning to caller — caller may delete them.
    for cname in orphans:
        rag._log_collection_op("orphan_candidate", cname)
    return orphans


def _find_orphan_segment_dirs() -> list[tuple[Path, int]]:
    """sqlite-vec has no HNSW segment dirs on disk; stub retained
    for call-site compatibility with _prune_orphan_segment_dirs."""
    return []


def _prune_orphan_segment_dirs(dry_run: bool = False) -> dict:
    """Delete orphan HNSW segment dirs. Returns {count, bytes_freed, paths}."""
    orphans = _find_orphan_segment_dirs()
    total = sum(sz for _, sz in orphans)
    out: dict = {
        "count": len(orphans),
        "bytes_freed": total,
        "paths": [p.name for p, _ in orphans],
    }
    if dry_run or not orphans:
        return out
    import shutil as _shutil
    removed = 0
    freed = 0
    for path, sz in orphans:
        try:
            _shutil.rmtree(path)
            removed += 1
            freed += sz
        except OSError:
            pass
    out["count"] = removed
    out["bytes_freed"] = freed
    return out


def _wal_checkpoint_for(sqlite_path: Path, *, dry_run: bool = False) -> dict:
    """Run `PRAGMA wal_checkpoint(TRUNCATE)` against an arbitrary sqlite file.

    Safer than VACUUM — doesn't require exclusive lock and truncates the
    write-ahead log after flushing. Returns {before_bytes, after_bytes, ok}.
    Helper for post-split maintenance: ragvec.db and telemetry.db each have
    their own WAL and need independent checkpoints.
    """
    wal_path = sqlite_path.with_suffix(sqlite_path.suffix + "-wal")
    if not sqlite_path.is_file():
        return {"ok": False, "reason": "no sqlite file"}
    before = sqlite_path.stat().st_size + (wal_path.stat().st_size if wal_path.is_file() else 0)
    if dry_run:
        return {"ok": True, "before_bytes": before, "after_bytes": before, "dry_run": True}
    import sqlite3 as _sqlite
    try:
        conn = _sqlite.connect(str(sqlite_path), timeout=5.0)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        return {"ok": False, "reason": str(e), "before_bytes": before}
    after = sqlite_path.stat().st_size + (wal_path.stat().st_size if wal_path.is_file() else 0)
    return {"ok": True, "before_bytes": before, "after_bytes": after}


def _vec_wal_checkpoint(dry_run: bool = False) -> dict:
    """Run `PRAGMA wal_checkpoint(TRUNCATE)` on ragvec.db (sqlite-vec workload)."""
    import rag  # noqa: PLC0415

    return _wal_checkpoint_for(rag.DB_PATH / "ragvec.db", dry_run=dry_run)


def _telemetry_wal_checkpoint(dry_run: bool = False) -> dict:
    """Run `PRAGMA wal_checkpoint(TRUNCATE)` on telemetry.db.

    Post 2026-04-21 split — telemetry.db is the hot-path writes target (queries,
    behavior, memory/cpu metrics). Separate WAL from ragvec.db, so an
    unconditional checkpoint is fired in run_maintenance to keep the WAL
    bounded even when --skip-logs bypasses _sql_rotate_log_tables.
    """
    import rag  # noqa: PLC0415

    return _wal_checkpoint_for(rag.DB_PATH / rag._TELEMETRY_DB_FILENAME, dry_run=dry_run)
