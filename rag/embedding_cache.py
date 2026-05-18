"""Persistent exact-match embedding cache for index-time payloads.

The cache is intentionally conservative: a hit requires the same namespace,
model identity, text SHA-256 and text length. Callers pass the final text that
would be embedded, so optional chunk enrichments naturally invalidate via the
text hash. Stored vectors are float32 blobs, matching sqlite-vec storage.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

__all__ = [
    "EmbeddingCacheStats",
    "embed_texts_cached",
]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_embedding_cache (
    namespace TEXT NOT NULL,
    model_id TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    text_len INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    created_ts REAL NOT NULL,
    last_hit_ts REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(namespace, model_id, text_hash)
)
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS ix_rag_embedding_cache_last_hit "
    "ON rag_embedding_cache(last_hit_ts)",
)

_INIT_LOCKS: dict[str, threading.Lock] = {}
_INIT_LOCKS_GUARD = threading.Lock()


@dataclass(slots=True)
class EmbeddingCacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    errors: int = 0


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _INIT_LOCKS_GUARD:
        lock = _INIT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _INIT_LOCKS[key] = lock
        return lock


def _connect(cache_path: Path) -> sqlite3.Connection:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    with _lock_for(cache_path):
        conn.execute(_SCHEMA)
        for sql in _INDEXES:
            conn.execute(sql)
        conn.commit()
    return conn


def _cache_path(db_dir: Path) -> Path:
    return Path(db_dir) / "embedding_cache.db"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _embedding_to_blob(embedding: Iterable[float]) -> tuple[bytes, int]:
    import numpy as np

    arr = np.asarray(list(embedding), dtype="<f4")
    return arr.tobytes(), int(arr.shape[0])


def _blob_to_embedding(blob: bytes, dim: int) -> list[float]:
    import numpy as np

    return np.frombuffer(blob, dtype="<f4", count=int(dim)).astype("float32").tolist()


def _prune_if_needed(conn: sqlite3.Connection, max_rows: int) -> None:
    if max_rows <= 0:
        return
    row = conn.execute("SELECT COUNT(*) FROM rag_embedding_cache").fetchone()
    count = int(row[0] if row else 0)
    over = count - max_rows
    if over <= 0:
        return
    conn.execute(
        "DELETE FROM rag_embedding_cache WHERE rowid IN ("
        " SELECT rowid FROM rag_embedding_cache"
        " ORDER BY last_hit_ts ASC LIMIT ?"
        ")",
        (over,),
    )


def embed_texts_cached(
    texts: list[str],
    *,
    db_dir: Path,
    model_id: str,
    namespace: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    max_rows: int | None = None,
) -> tuple[list[list[float]], EmbeddingCacheStats]:
    """Embed texts using a persistent exact-match cache.

    Any cache failure degrades to ``embed_fn(texts)``. This function never
    returns partial results: output length always matches input length or the
    underlying embedder exception is propagated.
    """
    stats = EmbeddingCacheStats()
    if not texts:
        return [], stats
    if not model_id or os.environ.get("RAG_INDEX_EMBED_CACHE", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "",
    ):
        return embed_fn(texts), stats

    try:
        cache_path = _cache_path(db_dir)
        conn = _connect(cache_path)
    except Exception:
        stats.errors += 1
        return embed_fn(texts), stats

    now = time.time()
    hashes = [_text_hash(t) for t in texts]
    lengths = [len(t.encode("utf-8", errors="surrogatepass")) for t in texts]
    result: list[list[float] | None] = [None] * len(texts)
    missing_by_hash: dict[str, int] = {}

    try:
        unique_hashes = list(dict.fromkeys(hashes))
        if unique_hashes:
            placeholders = ",".join("?" * len(unique_hashes))
            rows = conn.execute(
                "SELECT text_hash, text_len, dim, embedding "
                "FROM rag_embedding_cache "
                "WHERE namespace = ? AND model_id = ? "
                f"AND text_hash IN ({placeholders})",
                [namespace, model_id, *unique_hashes],
            ).fetchall()
        else:
            rows = []
        cached: dict[str, tuple[int, int, bytes]] = {
            str(h): (int(text_len), int(dim), blob)
            for h, text_len, dim, blob in rows
        }
        hit_hashes: set[str] = set()
        for idx, h in enumerate(hashes):
            item = cached.get(h)
            if item is None or item[0] != lengths[idx]:
                missing_by_hash.setdefault(h, idx)
                continue
            result[idx] = _blob_to_embedding(item[2], item[1])
            hit_hashes.add(h)
        stats.hits = sum(1 for v in result if v is not None)
        stats.misses = len(texts) - stats.hits
        if hit_hashes:
            conn.executemany(
                "UPDATE rag_embedding_cache "
                "SET last_hit_ts = ?, hits = hits + 1 "
                "WHERE namespace = ? AND model_id = ? AND text_hash = ?",
                [(now, namespace, model_id, h) for h in hit_hashes],
            )
            conn.commit()
    except Exception:
        stats.errors += 1
        try:
            conn.close()
        except Exception:
            pass
        return embed_fn(texts), stats

    if missing_by_hash:
        missing_pairs = [
            (h, texts[idx], lengths[idx])
            for h, idx in missing_by_hash.items()
        ]
        fresh = embed_fn([text for _, text, _ in missing_pairs])
        if len(fresh) != len(missing_pairs):
            try:
                conn.close()
            except Exception:
                pass
            raise RuntimeError(
                "embedding cache embed_fn length mismatch: "
                f"{len(fresh)} embeddings for {len(missing_pairs)} texts"
            )
        fresh_by_hash = {h: emb for (h, _text, _length), emb in zip(missing_pairs, fresh)}
        for idx, h in enumerate(hashes):
            if result[idx] is None:
                result[idx] = fresh_by_hash[h]
        try:
            rows_to_store = []
            for h, _text, length in missing_pairs:
                blob, dim = _embedding_to_blob(fresh_by_hash[h])
                rows_to_store.append((
                    namespace,
                    model_id,
                    h,
                    length,
                    dim,
                    blob,
                    now,
                    now,
                    0,
                ))
            conn.executemany(
                "INSERT OR REPLACE INTO rag_embedding_cache "
                "(namespace, model_id, text_hash, text_len, dim, embedding, "
                " created_ts, last_hit_ts, hits) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows_to_store,
            )
            if max_rows is None:
                try:
                    max_rows = int(os.environ.get("RAG_INDEX_EMBED_CACHE_MAX_ROWS", "200000"))
                except (TypeError, ValueError):
                    max_rows = 200000
            _prune_if_needed(conn, int(max_rows))
            conn.commit()
            stats.stores = len(rows_to_store)
        except Exception:
            stats.errors += 1
    final = [v for v in result if v is not None]
    if len(final) != len(texts):
        try:
            conn.close()
        except Exception:
            pass
        raise RuntimeError(
            "embedding cache internal length mismatch: "
            f"{len(final)} for {len(texts)}"
        )
    try:
        conn.close()
    except Exception:
        pass
    return final, stats
