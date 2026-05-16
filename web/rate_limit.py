"""Shared sliding-window rate limiting primitives for web routes."""
from __future__ import annotations

import collections as _collections
import threading as _threading
import time

from fastapi import HTTPException

__all__ = [
    "_LRU_RATE_BUCKET_MAX",
    "_LRURateBucket",
    "_BEHAVIOR_BUCKETS",
    "_BEHAVIOR_RATE_LIMIT",
    "_BEHAVIOR_RATE_WINDOW",
    "_CHAT_BUCKETS",
    "_CHAT_RATE_LIMIT",
    "_CHAT_RATE_WINDOW",
    "_RATE_LIMIT_LOCK",
    "_check_rate_limit",
]

_LRU_RATE_BUCKET_MAX = 5000


class _LRURateBucket:
    """Bounded per-IP bucket compatible with defaultdict(deque) call sites."""

    def __init__(self, max_size: int = _LRU_RATE_BUCKET_MAX) -> None:
        self._data: _collections.OrderedDict[str, _collections.deque] = (
            _collections.OrderedDict()
        )
        self._max_size = max_size

    def __getitem__(self, key: str) -> _collections.deque:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        bucket: _collections.deque = _collections.deque()
        self._data[key] = bucket
        if len(self._data) > self._max_size:
            self._data.popitem(last=False)
        return bucket

    def __setitem__(self, key: str, value) -> None:
        if not isinstance(value, _collections.deque):
            value = _collections.deque(value)
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._max_size:
            self._data.popitem(last=False)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def clear(self) -> None:
        self._data.clear()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


_BEHAVIOR_BUCKETS: _LRURateBucket = _LRURateBucket()
_BEHAVIOR_RATE_LIMIT = 120
_BEHAVIOR_RATE_WINDOW = 60.0

_CHAT_BUCKETS: _LRURateBucket = _LRURateBucket()
_CHAT_RATE_LIMIT = 30
_CHAT_RATE_WINDOW = 60.0

_RATE_LIMIT_LOCK = _threading.Lock()


def _check_rate_limit(bucket, ip: str, limit: int, window: float) -> None:
    """Sliding-window rate limit per IP. Raises HTTPException 429 on breach."""
    now = time.time()
    cutoff = now - window
    with _RATE_LIMIT_LOCK:
        events = bucket[ip]
        while events and events[0] < cutoff:
            events.popleft()
        if len(events) >= limit:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        events.append(now)
