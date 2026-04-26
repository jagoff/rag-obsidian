"""Hard negative mining para training data del LightGBM lambdarank.

Para cada `(query, positive_path)` en `rag_synthetic_queries`, busca los
chunks que están **cerca en embedding pero NO son la nota source** —
esos son los negatives MÁS informativos para entrenar el ranker.

Diseño:
  1. Embed la query con bge-m3.
  2. Top-K nearest neighbors en la collection.
  3. Filtrar el positive + duplicates (cosine > 0.95 al positive).
  4. Tomar los top-N "hard negatives" (default 5 por query).
  5. Persistir a `rag_synthetic_negatives` con cosine score.

Costo: ~50-60ms por query → 5000 queries = ~5 min total.
Idempotency: re-runs sobre la misma synthetic_query_id no duplican.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_NEGATIVES_PER_QUERY = 5
DEFAULT_SEARCH_K = 20
DEFAULT_DUPLICATE_COSINE_THRESHOLD = 0.95


def _ensure_negatives_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_synthetic_negatives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            synthetic_query_id INTEGER NOT NULL,
            query TEXT NOT NULL,
            positive_path TEXT NOT NULL,
            neg_path TEXT NOT NULL,
            cosine_to_query REAL,
            cosine_to_positive REAL,
            UNIQUE(synthetic_query_id, neg_path)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_synth_neg_query "
        "ON rag_synthetic_negatives(synthetic_query_id)"
    )


def mine_hard_negatives_for_synthetic(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
    negatives_per_query: int = DEFAULT_NEGATIVES_PER_QUERY,
    search_k: int = DEFAULT_SEARCH_K,
    duplicate_cosine_threshold: float = DEFAULT_DUPLICATE_COSINE_THRESHOLD,
    embed_fn: Callable[[str], list[float]] | None = None,
    nearest_neighbors_fn: Callable[[list[float], int], list[dict[str, Any]]] | None = None,
    dry_run: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Mine hard negatives para los pairs en rag_synthetic_queries."""
    _ensure_negatives_table(conn)

    if embed_fn is None:
        embed_fn = _default_embed_fn
    if nearest_neighbors_fn is None:
        nearest_neighbors_fn = _default_nearest_neighbors_fn

    sql = """
        SELECT id, query, note_path
        FROM rag_synthetic_queries
        WHERE id NOT IN (
            SELECT DISTINCT synthetic_query_id FROM rag_synthetic_negatives
        )
        ORDER BY id ASC
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    queries = conn.execute(sql).fetchall()

    metrics: dict[str, int] = {
        "n_queries_examined": len(queries),
        "n_queries_no_neighbors": 0,
        "n_queries_with_negatives": 0,
        "n_negatives_inserted": 0,
        "n_filtered_self": 0,
        "n_filtered_duplicate": 0,
    }
    sample_pairs: list[dict[str, Any]] = []
    now_iso = datetime.now().isoformat(timespec="seconds")

    for i, (synth_id, query, positive_path) in enumerate(queries):
        if progress_callback is not None:
            progress_callback(i, len(queries), query)

        try:
            q_embed = embed_fn(query)
        except Exception as exc:
            logger.warning("embed failed for synth_id=%d: %s", synth_id, exc)
            continue

        try:
            neighbors = nearest_neighbors_fn(q_embed, search_k)
        except Exception as exc:
            logger.warning("NN search failed for synth_id=%d: %s", synth_id, exc)
            continue

        if not neighbors:
            metrics["n_queries_no_neighbors"] += 1
            continue

        filtered: list[dict[str, Any]] = []
        for n in neighbors:
            n_path = n.get("path")
            n_cosine = float(n.get("cosine", 0.0))
            if not n_path:
                continue
            if n_path == positive_path:
                metrics["n_filtered_self"] += 1
                continue
            if n_cosine > duplicate_cosine_threshold:
                metrics["n_filtered_duplicate"] += 1
                continue
            filtered.append(n)

        if not filtered:
            metrics["n_queries_no_neighbors"] += 1
            continue

        hard_negatives = filtered[:negatives_per_query]
        metrics["n_queries_with_negatives"] += 1

        for n in hard_negatives:
            sample_pairs.append({
                "synth_id": synth_id,
                "query": query[:80],
                "positive": positive_path,
                "negative": n["path"],
                "cosine_to_query": round(float(n.get("cosine", 0.0)), 3),
            })
            if dry_run:
                continue
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO rag_synthetic_negatives "
                    "(ts, synthetic_query_id, query, positive_path, neg_path, "
                    "cosine_to_query, cosine_to_positive) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        now_iso, synth_id, query, positive_path,
                        n["path"], float(n.get("cosine", 0.0)), None,
                    ),
                )
                if conn.total_changes > 0:
                    metrics["n_negatives_inserted"] += 1
            except sqlite3.IntegrityError:
                pass

    return {
        **metrics,
        "sample_pairs": sample_pairs[:20],
        "n_total_pairs": len(sample_pairs),
        "dry_run": dry_run,
    }


def _default_embed_fn(text: str) -> list[float]:
    import rag
    embeds = rag.embed([text])
    if not embeds:
        return []
    return list(embeds[0])


def _default_nearest_neighbors_fn(
    embedding: list[float], k: int
) -> list[dict[str, Any]]:
    import rag

    col = rag.get_db()
    if col.count() == 0:
        return []
    res = col.query(
        query_embeddings=[embedding],
        n_results=k * 3,
        include=["documents", "metadatas", "distances"],
    )
    if not res or not res.get("ids"):
        return []
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    seen_paths: set[str] = set()
    out: list[dict[str, Any]] = []
    for meta, dist in zip(metas, distances):
        path = (meta or {}).get("path")
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        cosine = 1.0 - float(dist) / 2.0
        out.append({"path": path, "cosine": cosine})
        if len(out) >= k:
            break
    return out


def get_negatives_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    _ensure_negatives_table(conn)
    n_total = conn.execute(
        "SELECT COUNT(*) FROM rag_synthetic_negatives"
    ).fetchone()[0]
    n_unique_queries = conn.execute(
        "SELECT COUNT(DISTINCT synthetic_query_id) FROM rag_synthetic_negatives"
    ).fetchone()[0]
    avg_cosine = conn.execute(
        "SELECT AVG(cosine_to_query) FROM rag_synthetic_negatives"
    ).fetchone()[0]
    return {
        "n_total": n_total,
        "n_unique_queries": n_unique_queries,
        "avg_cosine_to_query": round(float(avg_cosine), 3) if avg_cosine else 0.0,
    }
