"""Thread-safe cache helpers.

Extracted from web/server.py (2026-05-15) to allow reuse across the codebase.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class ThreadSafeCache:
    """Thread-safe cache with TTL and single-flight refresh."""

    def __init__(self, ttl: float = 60.0):
        self._lock = threading.RLock()
        self._cache: dict = {"ts": 0.0, "payload": None}
        self._ttl = ttl
        self._refreshing = False

    def get(self, _key: Any = None) -> tuple[float, dict] | None:
        """Get cached payload if fresh. Returns (ts, payload) or None."""
        with self._lock:
            entry = self._cache
            if entry["ts"] == 0.0:
                return None
            if time.time() - entry["ts"] > self._ttl:
                return None
            return (entry["ts"], entry["payload"])

    def put(self, *args: Any) -> None:
        """Update cache with new payload."""
        if len(args) == 1:
            payload = args[0]
        elif len(args) == 2:
            # Back-compat with pre-extraction call sites that passed a
            # sentinel key to the single-entry cache.
            payload = args[1]
        else:
            raise TypeError("put() expects payload or (key, payload)")
        with self._lock:
            self._cache = {"ts": time.time(), "payload": payload}
            self._refreshing = False

    def clear(self) -> None:
        """Reset the cache to an empty single-entry state."""
        with self._lock:
            self._cache = {"ts": 0.0, "payload": None}
            self._refreshing = False

    def __getitem__(self, key: str) -> Any:
        """Dict-style compatibility for older tests/call sites."""
        with self._lock:
            return self._cache[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-style compatibility for older tests/call sites."""
        if key not in ("ts", "payload"):
            raise KeyError(key)
        with self._lock:
            self._cache[key] = value

    def is_refreshing(self) -> bool:
        """Check if a refresh is in progress (single-flight guard)."""
        with self._lock:
            return self._refreshing

    def set_refreshing(self, value: bool) -> None:
        """Set refresh flag (single-flight guard)."""
        with self._lock:
            self._refreshing = value


class ThreadSafeCacheMultiKey:
    """Thread-safe cache with TTL and multi-key support.

    Used for caches that need to store multiple values keyed by a tuple
    (or any hashable). Automatically evicts stale entries on each ``put``,
    plus optional LRU-style eviction by age when ``max_size`` is set
    (absolute cap on the number of live entries).
    """

    def __init__(self, ttl: float = 60.0, max_size: int | None = None):
        self._lock = threading.RLock()
        self._cache: dict[Any, dict] = {}
        self._ttl = ttl
        self._max_size = max_size
        self._refreshing: dict[Any, bool] = {}

    def get(self, key: Any) -> tuple[float, Any] | None:
        """Get cached value for key if fresh, else None."""
        now = time.time()
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                if (now - entry["ts"]) < self._ttl:
                    return (entry["ts"], entry["payload"])
        return None

    def put(self, key: Any, payload: Any) -> None:
        """Update cached value for key and evict stale entries."""
        now = time.time()
        with self._lock:
            self._cache[key] = {"ts": now, "payload": payload}
            self._refreshing.pop(key, None)
            # Evict stale entries (TTL + small grace).
            cutoff = now - (self._ttl + 5.0)
            stale = [k for k, v in self._cache.items() if v["ts"] < cutoff]
            for k in stale:
                self._cache.pop(k, None)
                self._refreshing.pop(k, None)
            # Absolute cap: evict oldest entries by write time if over max_size.
            if self._max_size is not None and len(self._cache) > self._max_size:
                sorted_keys = sorted(
                    self._cache.items(), key=lambda kv: kv[1]["ts"],
                )
                excess = len(self._cache) - self._max_size
                for k, _ in sorted_keys[:excess]:
                    self._cache.pop(k, None)
                    self._refreshing.pop(k, None)

    def clear(self) -> None:
        """Reset the cache + refresh flags."""
        with self._lock:
            self._cache.clear()
            self._refreshing.clear()

    def delete(self, key: Any) -> None:
        """Remove key explicitly (idempotent)."""
        with self._lock:
            self._cache.pop(key, None)
            self._refreshing.pop(key, None)

    def __getitem__(self, key: str) -> Any:
        """Single-entry dict compatibility over the `default` key."""
        if key not in ("ts", "payload"):
            raise KeyError(key)
        with self._lock:
            entry = self._cache.get("default") or {"ts": 0.0, "payload": None}
            return entry[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Single-entry dict compatibility over the `default` key."""
        if key not in ("ts", "payload"):
            raise KeyError(key)
        with self._lock:
            entry = self._cache.setdefault("default", {"ts": 0.0, "payload": None})
            entry[key] = value

    def start_refresh(self, key: Any) -> bool:
        """Mark refresh as in progress for key. Returns True if we won the race."""
        with self._lock:
            if self._refreshing.get(key, False):
                return False
            self._refreshing[key] = True
            return True
