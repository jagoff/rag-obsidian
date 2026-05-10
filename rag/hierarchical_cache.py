"""Hierarchical multi-level cache for RAG retrieval results.

Game Changer #3 (2026-05-10): 3-tier cache to reduce latency + LLM calls.

Cache Levels
------------

L1 (query → top-k IDs):
  - Key: normalized query + filters (folder, tag, source, date_range)
  - Value: list of top-k result IDs (not full objects)
  - Size: LRU 256 entries
  - TTL: 1h (high churn, fast invalidation)
  - Hit: skip semantic search entirely, jump to rerank step
  - Purpose: cache the expensive semantic search (~50-100ms)

L2 (query + IDs → full result objects):
  - Key: L1 key + list of result IDs
  - Value: full result objects (with embeddings, metadata, etc.)
  - Size: LRU 512 entries
  - TTL: 24h (medium churn)
  - Hit: skip vector DB lookup, jump to rerank step
  - Purpose: cache the vector DB fetch (~20-50ms)

L3 (query + feedback → learned weights):
  - Key: normalized query + user feedback (thumbs up/down)
  - Value: adjusted re-ranking weights for similar queries
  - Size: LRU 128 entries
  - TTL: 7d (slow churn, learning over time)
  - Hit: use learned weights instead of default weights
  - Purpose: adaptive personalization based on user feedback

Cache Invalidation
-----------------

- Manual: `rag cache --clear [l1|l2|l3|all]`
- Automatic: TTL-based eviction
- Manual: `rag cache --stats` to inspect hit rates

Integration
-----------

Called from `retrieve()` in `rag/__init__.py`:
  1. Check L1 for top-k IDs
  2. If L1 miss, do semantic search
  3. Check L2 for full objects (if we have IDs from L1)
  4. If L2 miss, fetch from vector DB
  5. Check L3 for learned weights
  6. Apply weights in rerank step
  7. Store results in L1 + L2 on first retrieval
  8. Update L3 on user feedback
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

# L1: query → top-k IDs
L1_MAX_SIZE = 256
L1_TTL_SECONDS = 3600  # 1h

# L2: query + IDs → full result objects
L2_MAX_SIZE = 512
L2_TTL_SECONDS = 86400  # 24h

# L3: query + feedback → learned weights
L3_MAX_SIZE = 128
L3_TTL_SECONDS = 604800  # 7d

# ── LRU Cache Implementation ───────────────────────────────────────────────────


class _LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int, ttl_seconds: int):
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get value if key exists and not expired. Returns None on miss."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, timestamp = self._cache[key]
            if time.time() - timestamp > self._ttl_seconds:
                # Expired
                del self._cache[key]
                self._misses += 1
                return None

            # Hit - move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """Set value for key. Evicts oldest if at capacity."""
        with self._lock:
            # Remove if exists (will be re-added at end)
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            # Add at end (most recently used)
            self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, int | float]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


# ── Global Cache Instances ─────────────────────────────────────────────────────

_l1_cache = _LRUCache(L1_MAX_SIZE, L1_TTL_SECONDS)
_l2_cache = _LRUCache(L2_MAX_SIZE, L2_TTL_SECONDS)
_l3_cache = _LRUCache(L3_MAX_SIZE, L3_TTL_SECONDS)


# ── Key Generation ─────────────────────────────────────────────────────────────


def _make_query_key(
    query: str,
    folder: str | None = None,
    tag: str | None = None,
    source: str | None = None,
    date_range: tuple[float, float] | None = None,
) -> str:
    """Generate a normalized cache key from query + filters."""
    # Normalize query
    normalized = query.strip().lower()
    
    # Build filter dict
    filters = {
        "folder": folder,
        "tag": tag,
        "source": source,
        "date_range": date_range,
    }
    
    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}
    
    # Hash the combination
    key_data = f"{normalized}:{json.dumps(filters, sort_keys=True)}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def _make_l2_key(query_key: str, result_ids: list[str]) -> str:
    """Generate L2 cache key from L1 key + result IDs."""
    # Sort IDs for consistency
    sorted_ids = sorted(result_ids)
    key_data = f"{query_key}:{','.join(sorted_ids)}"
    return hashlib.sha256(key_data.encode()).hexdigest()


def _make_l3_key(query: str, feedback: str) -> str:
    """Generate L3 cache key from query + user feedback."""
    key_data = f"{query.strip().lower()}:{feedback}"
    return hashlib.sha256(key_data.encode()).hexdigest()


# ── L1: Query → Top-K IDs ─────────────────────────────────────────────────────


def l1_get(
    query: str,
    folder: str | None = None,
    tag: str | None = None,
    source: str | None = None,
    date_range: tuple[float, float] | None = None,
) -> list[str] | None:
    """Get top-k result IDs from L1 cache. Returns None on miss."""
    key = _make_query_key(query, folder, tag, source, date_range)
    result = _l1_cache.get(key)
    if result is None:
        return None
    # Validate it's a list of strings
    if not isinstance(result, list):
        return None
    return result


def l1_set(
    query: str,
    result_ids: list[str],
    folder: str | None = None,
    tag: str | None = None,
    source: str | None = None,
    date_range: tuple[float, float] | None = None,
) -> None:
    """Store top-k result IDs in L1 cache."""
    key = _make_query_key(query, folder, tag, source, date_range)
    _l1_cache.set(key, result_ids)


# ── L2: Query + IDs → Full Result Objects ───────────────────────────────────────


def l2_get(
    query: str,
    result_ids: list[str],
    folder: str | None = None,
    tag: str | None = None,
    source: str | None = None,
    date_range: tuple[float, float] | None = None,
) -> list[dict[str, Any]] | None:
    """Get full result objects from L2 cache. Returns None on miss."""
    query_key = _make_query_key(query, folder, tag, source, date_range)
    l2_key = _make_l2_key(query_key, result_ids)
    result = _l2_cache.get(l2_key)
    if result is None:
        return None
    # Validate it's a list of dicts
    if not isinstance(result, list):
        return None
    return result


def l2_set(
    query: str,
    result_ids: list[str],
    results: list[dict[str, Any]],
    folder: str | None = None,
    tag: str | None = None,
    source: str | None = None,
    date_range: tuple[float, float] | None = None,
) -> None:
    """Store full result objects in L2 cache."""
    query_key = _make_query_key(query, folder, tag, source, date_range)
    l2_key = _make_l2_key(query_key, result_ids)
    _l2_cache.set(l2_key, results)


# ── L3: Query + Feedback → Learned Weights ────────────────────────────────────


def l3_get(query: str, feedback: str) -> dict[str, float] | None:
    """Get learned weights from L3 cache. Returns None on miss."""
    key = _make_l3_key(query, feedback)
    result = _l3_cache.get(key)
    if result is None:
        return None
    # Validate it's a dict with float values
    if not isinstance(result, dict):
        return None
    return result


def l3_set(query: str, feedback: str, weights: dict[str, float]) -> None:
    """Store learned weights in L3 cache."""
    key = _make_l3_key(query, feedback)
    _l3_cache.set(key, weights)


# ── Cache Management ───────────────────────────────────────────────────────────


def clear_cache(level: str = "all") -> None:
    """Clear cache level(s). Valid levels: 'l1', 'l2', 'l3', 'all'."""
    level = level.lower()
    if level in ("l1", "all"):
        _l1_cache.clear()
    if level in ("l2", "all"):
        _l2_cache.clear()
    if level in ("l3", "all"):
        _l3_cache.clear()


def get_stats() -> dict[str, dict[str, int | float]]:
    """Return statistics for all cache levels."""
    return {
        "l1": _l1_cache.stats(),
        "l2": _l2_cache.stats(),
        "l3": _l3_cache.stats(),
    }


# ── CLI Integration ────────────────────────────────────────────────────────────


def _cli_clear(args: list[str]) -> str:
    """CLI handler for cache clear."""
    level = args[0] if args else "all"
    clear_cache(level)
    return f"Cleared cache level: {level}"


def _cli_stats(args: list[str]) -> str:
    """CLI handler for cache stats."""
    stats = get_stats()
    lines = ["Cache Statistics:"]
    for level, data in stats.items():
        lines.append(f"\n{level.upper()}:")
        lines.append(f"  Size: {data['size']}/{data['max_size']}")
        lines.append(f"  Hits: {data['hits']}")
        lines.append(f"  Misses: {data['misses']}")
        lines.append(f"  Hit rate: {data['hit_rate']:.2%}")
    return "\n".join(lines)


# ── Test Helpers ────────────────────────────────────────────────────────────────


def _reset_for_tests() -> None:
    """Clear all caches for tests."""
    clear_cache("all")
