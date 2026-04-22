#!/usr/bin/env python3
"""One-shot GLiNER entity extraction backfill over the active corpus.

Reads every chunk from the active sqlite-vec collection, runs zero-shot NER
via GLiNER (urchade/gliner_multi-v2.1), and upserts results into the
``rag_entities`` + ``rag_entity_mentions`` tables inside
``~/.local/share/obsidian-rag/ragvec/ragvec.db``.

Usage::

    .venv/bin/python scripts/backfill_entities.py [--dry-run] [--limit N] [--vault NAME]

    # Smoke-test — no writes:
    .venv/bin/python scripts/backfill_entities.py --dry-run --limit 10

    # Full backfill on default vault (can take several minutes on large corpora):
    .venv/bin/python scripts/backfill_entities.py

    # Target a non-default vault registered in vaults.json:
    .venv/bin/python scripts/backfill_entities.py --vault work

Idempotence
-----------
``rag_entity_mentions`` uses ``INSERT OR IGNORE`` keyed on
``(entity_id, chunk_id)``, so re-running does **not** create duplicate mention
rows.  ``mention_count`` on ``rag_entities`` IS incremented on each pass even
for already-processed chunks, so the count will be inflated after multiple runs.
For a fully clean re-run, manually truncate both tables before invoking.

Side-effects
------------
* Writes only to ``rag_entities`` and ``rag_entity_mentions``.
* The sqlite-vec collection (chunk embeddings) is opened read-only.
* Does **not** set or consult ``RAG_EXTRACT_ENTITIES`` — that env-var gates the
  incremental hot-path inside ``rag index``; the backfill calls helpers directly.

When to run
-----------
Once, after bulk-indexing a corpus that pre-dates ``RAG_EXTRACT_ENTITIES=1``.
Subsequent ``rag index`` runs populate entity tables incrementally via
``_extract_and_index_entities_for_chunks``, so the backfill is not needed again
unless the tables are wiped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402

from rich.progress import (  # noqa: E402
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _resolve_collection(vault_name: str | None):
    """Return the sqlite-vec collection for *vault_name*, or the default."""
    if vault_name is None:
        return rag.get_db()
    cfg = rag._load_vaults_config()
    vaults = cfg.get("vaults", {})
    if vault_name not in vaults:
        known = ", ".join(vaults.keys()) if vaults else "(none registered)"
        print(
            f"[error] vault '{vault_name}' not found in registry. "
            f"Registered vaults: {known}",
            file=sys.stderr,
        )
        sys.exit(1)
    return rag.get_db_for(Path(vaults[vault_name]))


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="One-shot GLiNER entity backfill over the active corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --dry-run --limit 10   # smoke-test, no writes\n"
            "  %(prog)s                         # full backfill (may take minutes)\n"
            "  %(prog)s --vault work            # target a non-default vault\n"
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report chunk count without writing to the database.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N chunks (useful for testing).",
    )
    ap.add_argument(
        "--vault",
        default=None,
        metavar="NAME",
        help=(
            "Vault name to backfill (must be registered in vaults.json). "
            "Defaults to the currently active vault."
        ),
    )
    args = ap.parse_args()

    # ── GLiNER availability gate ──────────────────────────────────────────
    # _get_gliner_model() is sticky-fail: if the first load fails it returns
    # None forever for the lifetime of the process — no retry, no exception.
    model = rag._get_gliner_model()
    if model is None:
        print(
            "GLiNER no disponible; backfill abortado sin cambios.\n"
            "Instalá el paquete con: pip install gliner",
            file=sys.stderr,
        )
        sys.exit(0)

    # ── Load all chunks from the collection (read-only) ───────────────────
    col = _resolve_collection(args.vault)
    data = col.get(include=["documents", "metadatas"])
    ids: list[str] = data["ids"] or []
    docs: list[str | None] = data["documents"] or []
    metas: list[dict] = data["metadatas"] or []

    if args.limit is not None:
        ids = ids[: args.limit]
        docs = docs[: args.limit]
        metas = metas[: args.limit]

    total = len(ids)

    if args.dry_run:
        print(f"would process {total} chunks")
        return

    if total == 0:
        print("No chunks in collection — nothing to do.")
        return

    # ── Extraction + upsert loop ──────────────────────────────────────────
    # Open a single connection for the whole run (mirrors the batch pattern
    # used by _extract_and_index_entities_for_chunks).  GLiNER inference per
    # chunk keeps the connection alive for the duration — acceptable under WAL
    # mode which allows concurrent readers.
    t0 = time.perf_counter()
    chunks_ok = 0

    with rag._ragvec_state_conn() as conn:
        before_entities: int = conn.execute(
            "SELECT COUNT(*) FROM rag_entities"
        ).fetchone()[0]
        before_mentions: int = conn.execute(
            "SELECT COUNT(*) FROM rag_entity_mentions"
        ).fetchone()[0]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}[/cyan]/[cyan]{task.total}[/cyan]"),
            TimeElapsedColumn(),
            redirect_stderr=False,  # let per-chunk error prints reach stderr directly
        ) as progress:
            task = progress.add_task("Extrayendo entidades\u2026", total=total)

            for chunk_id, doc, meta in zip(ids, docs, metas):
                try:
                    # Use display_text (raw body) stored as `documents` in the
                    # collection — confirmed by _extract_and_index_entities_for_chunks
                    # which explicitly picks display_text (c[1]) over the embed_text
                    # prefix that is never persisted in the collection.
                    raw = rag._extract_entities_single(doc or "")
                    if raw:
                        entities = rag._cluster_entities(raw)
                        source = str(meta.get("source") or "vault")
                        ts = float(
                            meta.get("created_ts") or meta.get("ts") or time.time()
                        )
                        snippet = (doc or "")[:200]
                        rag._upsert_entities_for_chunk(
                            conn, entities, chunk_id, source, ts, snippet
                        )
                    chunks_ok += 1
                except Exception as exc:
                    print(f"[chunk error] {chunk_id}: {exc}", file=sys.stderr)
                finally:
                    progress.advance(task)

        after_entities: int = conn.execute(
            "SELECT COUNT(*) FROM rag_entities"
        ).fetchone()[0]
        after_mentions: int = conn.execute(
            "SELECT COUNT(*) FROM rag_entity_mentions"
        ).fetchone()[0]

    elapsed = time.perf_counter() - t0
    new_entities = after_entities - before_entities
    new_mentions = after_mentions - before_mentions

    print(f"\u2713 {new_entities} entidades \u00fanicas creadas")
    print(f"\u2713 {new_mentions} mentions escritas")
    print(f"\u2713 {chunks_ok} chunks procesados en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
