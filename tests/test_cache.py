from __future__ import annotations

import pytest

from rag.cache import ThreadSafeCache


def test_thread_safe_cache_accepts_legacy_single_key_get_put():
    cache = ThreadSafeCache(ttl=60.0)

    cache.put("default", {"ok": True})

    cached = cache.get("default")
    assert cached is not None
    _, payload = cached
    assert payload == {"ok": True}


def test_thread_safe_cache_rejects_invalid_put_arity():
    cache = ThreadSafeCache(ttl=60.0)

    with pytest.raises(TypeError):
        cache.put("a", "b", "c")
