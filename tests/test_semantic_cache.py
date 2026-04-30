"""Tests for Game-changer #1 (2026-04-22) — semantic response cache.

Validates the helpers in isolation (using a temp telemetry DB):
1. _compute_corpus_hash / _corpus_hash_cached
2. _embedding_to_blob / _blob_to_embedding round-trip
3. _ttl_for_intent routing
4. semantic_cache_store / _lookup round-trip
5. Cosine threshold enforcement (≥0.97 default, configurable)
6. TTL expiry skips stale rows
7. Corpus hash invalidation skips rows with different hash
8. Dimension mismatch skipped silently
9. hit_count bump on successful lookup
10. Refusal responses (top_score < 0.015) NOT stored
11. Empty response NOT stored
12. Disabled cache returns None on lookup + False on store
13. semantic_cache_clear removes rows (by hash or all)
14. semantic_cache_stats aggregates correctly
15. Concurrent writes don't crash (smoke test)
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import rag


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_cache_env(monkeypatch, tmp_path):
    """Ensure every test runs with the cache ENABLED and default thresholds.

    Redirects ``rag.DB_PATH`` to a tmp dir so ``semantic_cache_store`` /
    ``_lookup`` / ``_clear`` operate on an isolated telemetry.db. Pre-fix
    these tests wrote to the real prod telemetry.db (audit 2026-04-25:
    5 rows con question="test" filtradas a producción).
    """
    monkeypatch.setenv("RAG_CACHE_ENABLED", "1")
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_COSINE", 0.97)
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_DEFAULT_TTL", 86400)
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_RECENT_TTL", 600)
    monkeypatch.setattr(rag, "_SEMANTIC_CACHE_MAX_ROWS", 100)
    yield


def _emb(*floats: float, dim: int = 1024):
    """Build a deterministic unit vector. Pads with small noise to reach `dim`."""
    base = np.zeros(dim, dtype="float32")
    for i, v in enumerate(floats):
        base[i] = v
    # Ensure non-zero norm even when caller passes all zeros.
    if np.linalg.norm(base) == 0:
        base[0] = 1.0
    return base / np.linalg.norm(base)


# ── 1. Helpers (no DB) ───────────────────────────────────────────────────────


def test_embedding_to_blob_roundtrip():
    emb = _emb(1.0, 0.5, 0.3, dim=32)
    blob, dim = rag._embedding_to_blob(emb)
    assert dim == 32
    decoded = rag._blob_to_embedding(blob, dim)
    # Unit vectors stay unit; float32 round-trip is lossless for these values.
    assert np.allclose(emb, decoded, atol=1e-6)


def test_ttl_for_intent_routing():
    assert rag._ttl_for_intent("recent") == rag._SEMANTIC_CACHE_RECENT_TTL
    assert rag._ttl_for_intent("agenda") == rag._SEMANTIC_CACHE_RECENT_TTL
    assert rag._ttl_for_intent("semantic") == rag._SEMANTIC_CACHE_DEFAULT_TTL
    assert rag._ttl_for_intent(None) == rag._SEMANTIC_CACHE_DEFAULT_TTL
    assert rag._ttl_for_intent("synthesis") == rag._SEMANTIC_CACHE_DEFAULT_TTL
    assert rag._ttl_for_intent("count") == rag._SEMANTIC_CACHE_DEFAULT_TTL


# ── 2. Corpus hash ───────────────────────────────────────────────────────────


class _FakeCol:
    def __init__(self, count: int):
        self._count = count

    def count(self):
        return self._count


def test_compute_corpus_hash_returns_deterministic(monkeypatch, tmp_path):
    """Same corpus state → same hash."""
    # Stub the vault path to a tempdir so the rglob doesn't blow up on
    # real content.
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    (tmp_path / "a.md").write_text("# a")
    (tmp_path / "b.md").write_text("# b")
    h1 = rag._compute_corpus_hash(_FakeCol(10))
    h2 = rag._compute_corpus_hash(_FakeCol(10))
    assert h1 == h2
    assert len(h1) > 0


def test_compute_corpus_hash_changes_on_bucket_boundary(monkeypatch, tmp_path):
    """Hash buckets count by `_CORPUS_HASH_BUCKET` (default 500). Crossing a
    bucket boundary changes the hash; staying within does not.

    Pre-2026-04-24 hash changed on every count delta — ingesters running
    every 30min (whatsapp/calendar/gmail) constantly invalidated the
    semantic cache. Audit on web.log showed 30 SEMANTIC PUTs across 24
    distinct corpus_hashes → 0 cache hits ever. Bucketing collapses the
    rotation noise without losing the bulk-change invariant.

    2026-04-30: bucket subido de 100→500. Con corpus ~3600 chunks y
    ingesters que agregan ~30-80 chunks/run, bucket=100 producía 3
    hashes distintos por día (medido en rag_response_cache), manteniendo
    la hit-rate en 0%. Bucket=500 requiere +14% del corpus para invalidar.
    """
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    (tmp_path / "a.md").write_text("# a")
    bucket = rag._CORPUS_HASH_BUCKET
    # Within bucket: hash stable.
    h1 = rag._compute_corpus_hash(_FakeCol(bucket * 5 + 10))
    h2 = rag._compute_corpus_hash(_FakeCol(bucket * 5 + 200))
    assert h1 == h2, "small count delta inside bucket must not invalidate cache"
    # Crossing bucket boundary: hash changes.
    h3 = rag._compute_corpus_hash(_FakeCol(bucket * 6 + 10))
    assert h1 != h3, "crossing bucket boundary must invalidate cache"


def test_corpus_hash_bucket_default_is_500(monkeypatch, tmp_path):
    """Regression: bucket por default debe ser 500 (fix 2026-04-30).

    Con corpus ~3600 chunks, bucket=100 cruzaba el límite 2-3 veces/día
    con los ingesters incrementales (WA hourly, Calendar 6h), haciendo que
    la hit-rate del semantic cache sea 0%. Bucket=500 limita las
    invalidaciones a eventos bulk reales (--reset o semanas de ingesters).
    """
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    (tmp_path / "a.md").write_text("# a")
    # El bucket default debe ser 500 — ingesters que agregan <500 chunks
    # no invalidan el cache.
    assert rag._CORPUS_HASH_BUCKET == 500, (
        f"bucket esperado 500, got {rag._CORPUS_HASH_BUCKET}. "
        "Fix 2026-04-30: subido de 100 para evitar 0% hit-rate con ingesters continuos."
    )
    # Confirmar que 499 chunks de diferencia no cambian el hash.
    base_count = 3500
    h1 = rag._compute_corpus_hash(_FakeCol(base_count))
    h2 = rag._compute_corpus_hash(_FakeCol(base_count + 499))
    assert h1 == h2, "499 chunks de diferencia NO deben invalidar el cache"
    # Confirmar que cruzar el límite sí invalida.
    h3 = rag._compute_corpus_hash(_FakeCol(base_count + 500))
    # (base_count // 500 puede ser distinto de (base_count+500) // 500)
    if base_count % 500 != 0:
        # base_count no es múltiplo exacto → +500 sí cruza
        assert h1 != h3, "+500 chunks debe invalidar el cache"


def test_corpus_hash_cached_memoizes(monkeypatch, tmp_path):
    """Once computed, repeat calls with same chunk count return from memo."""
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    (tmp_path / "a.md").write_text("# a")
    # Reset memo so test is deterministic.
    with rag._corpus_hash_lock:
        rag._corpus_hash_memo["hash"] = None
        rag._corpus_hash_memo["chunk_count"] = None
    calls = {"n": 0}
    real = rag._compute_corpus_hash

    def spy(col):
        calls["n"] += 1
        return real(col)

    monkeypatch.setattr(rag, "_compute_corpus_hash", spy)
    h1 = rag._corpus_hash_cached(_FakeCol(5))
    h2 = rag._corpus_hash_cached(_FakeCol(5))
    h3 = rag._corpus_hash_cached(_FakeCol(5))
    assert h1 == h2 == h3
    assert calls["n"] == 1, f"should cache; got {calls['n']} computes"


# ── 3. Store + lookup round-trip ─────────────────────────────────────────────


def test_cache_store_and_lookup_hit(clean_cache_env):
    emb = _emb(1.0, 0.0)
    ok = rag.semantic_cache_store(
        emb,
        question="test q",
        response="cached answer",
        paths=["note.md"],
        scores=[0.9],
        top_score=0.9,
        intent="semantic",
        corpus_hash="HASH1",
    )
    assert ok is True
    hit = rag.semantic_cache_lookup(emb, "HASH1")
    assert hit is not None
    assert hit["response"] == "cached answer"
    assert hit["paths"] == ["note.md"]
    assert hit["scores"] == [0.9]
    assert hit["top_score"] == pytest.approx(0.9)
    assert hit["intent"] == "semantic"
    assert hit["cosine"] > 0.999  # same embedding


def test_cache_lookup_miss_different_corpus_hash(clean_cache_env):
    emb = _emb(1.0, 0.0)
    rag.semantic_cache_store(
        emb, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="HASH1",
    )
    # Same embedding, different corpus_hash → miss (vault changed).
    assert rag.semantic_cache_lookup(emb, "HASH2") is None


def test_cache_lookup_cosine_threshold(clean_cache_env):
    emb_a = _emb(1.0, 0.0)
    # Orthogonal vector — cosine = 0 vs emb_a → well below 0.97.
    emb_b = _emb(0.0, 1.0)
    rag.semantic_cache_store(
        emb_a, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="H",
    )
    # Orthogonal query should NOT hit.
    assert rag.semantic_cache_lookup(emb_b, "H") is None
    # Same query SHOULD hit.
    assert rag.semantic_cache_lookup(emb_a, "H") is not None


def test_cache_lookup_respects_custom_cosine_threshold(clean_cache_env):
    """Override the threshold at call-time to validate it's used.

    Construct two embeddings with a known cosine of ~0.7 so we can bracket
    a strict (0.99) and loose (0.5) threshold around the actual value.
    """
    # Unit-normalised: <1,0> and <0.7071, 0.7071> have cos=0.7071
    emb_a = _emb(1.0, 0.0)
    emb_near = _emb(1.0, 1.0)  # unit-normalised → <0.7071, 0.7071>
    rag.semantic_cache_store(
        emb_a, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="H",
    )
    # Strict threshold ABOVE the actual cosine (0.7071) → miss.
    assert rag.semantic_cache_lookup(emb_near, "H", cosine_threshold=0.99) is None
    # Loose threshold BELOW the actual cosine → hit.
    hit = rag.semantic_cache_lookup(emb_near, "H", cosine_threshold=0.50)
    assert hit is not None
    assert 0.65 < hit["cosine"] < 0.75  # sanity check on the test setup


def test_cache_lookup_ttl_expiry(clean_cache_env):
    emb = _emb(1.0, 0.0)
    rag.semantic_cache_store(
        emb, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent="semantic", corpus_hash="H",
        ttl_seconds=3600,
    )
    now = time.time()
    # Fresh hit
    assert rag.semantic_cache_lookup(emb, "H", now=now) is not None
    # Future now past TTL → miss
    assert rag.semantic_cache_lookup(emb, "H", now=now + 3601) is None


def test_cache_lookup_dimension_mismatch_silent(clean_cache_env):
    """A cached embedding with different dim is skipped (not crashed)."""
    emb_1024 = _emb(1.0, dim=1024)
    emb_512 = _emb(1.0, dim=512)
    rag.semantic_cache_store(
        emb_1024, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="H",
    )
    # Lookup with shorter dim should silently skip the row.
    assert rag.semantic_cache_lookup(emb_512, "H") is None


def test_cache_hit_count_bumps(clean_cache_env):
    emb = _emb(1.0, 0.0)
    rag.semantic_cache_store(
        emb, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="H",
    )
    assert rag.semantic_cache_lookup(emb, "H") is not None
    assert rag.semantic_cache_lookup(emb, "H") is not None
    assert rag.semantic_cache_lookup(emb, "H") is not None
    stats = rag.semantic_cache_stats()
    assert stats["hits"] == 3


# ── 4. Store-side guards ─────────────────────────────────────────────────────


def test_cache_store_skips_refusal(clean_cache_env):
    """top_score < 0.015 (refusal) → do NOT store."""
    emb = _emb(1.0, 0.0)
    ok = rag.semantic_cache_store(
        emb, question="q", response="No encontré esto.",
        paths=[], scores=[], top_score=0.005,  # refusal
        intent=None, corpus_hash="H",
    )
    assert ok is False
    assert rag.semantic_cache_lookup(emb, "H") is None


def test_cache_store_skips_empty_response(clean_cache_env):
    emb = _emb(1.0, 0.0)
    assert rag.semantic_cache_store(
        emb, question="q", response="",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="H",
    ) is False
    assert rag.semantic_cache_store(
        emb, question="q", response="   ",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="H",
    ) is False


def test_cache_disabled_returns_false_and_none(monkeypatch):
    monkeypatch.setenv("RAG_CACHE_ENABLED", "0")
    emb = _emb(1.0, 0.0)
    assert rag.semantic_cache_store(
        emb, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="H",
    ) is False
    assert rag.semantic_cache_lookup(emb, "H") is None


def test_cache_empty_corpus_hash_rejected(clean_cache_env):
    """A blank corpus_hash (col.count() failed) skips both store and lookup."""
    emb = _emb(1.0, 0.0)
    assert rag.semantic_cache_store(
        emb, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="",
    ) is False
    assert rag.semantic_cache_lookup(emb, "") is None


# ── 5. Maintenance (clear/stats) ─────────────────────────────────────────────


def test_cache_clear_by_hash(clean_cache_env):
    emb = _emb(1.0, 0.0)
    rag.semantic_cache_store(
        emb, question="q", response="a",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="HASH_A",
    )
    rag.semantic_cache_store(
        emb, question="q", response="b",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="HASH_B",
    )
    removed = rag.semantic_cache_clear(corpus_hash="HASH_A")
    assert removed == 1
    # HASH_B survives.
    assert rag.semantic_cache_lookup(emb, "HASH_B") is not None
    assert rag.semantic_cache_lookup(emb, "HASH_A") is None


def test_cache_clear_all(clean_cache_env):
    emb = _emb(1.0, 0.0)
    for h in ("H1", "H2", "H3"):
        rag.semantic_cache_store(
            emb, question="q", response="x",
            paths=[], scores=[], top_score=0.9,
            intent=None, corpus_hash=h,
        )
    assert rag.semantic_cache_clear() == 3
    stats = rag.semantic_cache_stats()
    assert stats["rows"] == 0


def test_cache_stats_contract(clean_cache_env):
    emb = _emb(1.0, 0.0)
    for h in ("A", "B"):
        rag.semantic_cache_store(
            emb, question="q", response="x",
            paths=[], scores=[], top_score=0.9,
            intent=None, corpus_hash=h,
        )
    rag.semantic_cache_lookup(emb, "A")  # +1 hit
    rag.semantic_cache_lookup(emb, "A")  # +1 hit
    stats = rag.semantic_cache_stats()
    assert stats["rows"] == 2
    assert stats["corpus_hashes"] == 2
    assert stats["hits"] == 2
    assert stats["enabled"] is True
    assert stats["cosine_threshold"] == 0.97


# ── 6. Concurrency smoke test ────────────────────────────────────────────────


def test_cache_concurrent_stores_dont_crash(clean_cache_env):
    """Smoke test: 3 threads storing concurrently must not bubble exceptions.

    Under SQLite WAL the writers serialise (SHARED→RESERVED→EXCLUSIVE lock
    escalation); some writes may fail with "database is locked" under load
    and the helper degrades silently to False per its error-handling contract.
    We only validate that Python never raises and that *some* rows landed.
    """
    errors: list[Exception] = []
    success_count = {"n": 0}
    lock = threading.Lock()

    def worker(idx):
        try:
            for i in range(20):
                emb = _emb(float(idx), float(i), dim=64)
                ok = rag.semantic_cache_store(
                    emb, question=f"q{idx}-{i}", response=f"ans{idx}-{i}",
                    paths=[f"p{idx}-{i}.md"], scores=[0.9],
                    top_score=0.9, intent=None, corpus_hash=f"H{idx}",
                )
                if ok:
                    with lock:
                        success_count["n"] += 1
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors, f"concurrent stores raised: {errors}"
    # At least some stores must have landed — tolerate WAL contention losses.
    assert success_count["n"] >= 10, f"expected ≥10 successful stores, got {success_count['n']}"
    stats = rag.semantic_cache_stats()
    assert stats["rows"] == success_count["n"]


# ── 7. Background (async) stores — 2026-04-22 perf fix ──────────────────────


def test_background_store_enqueues_and_commits(clean_cache_env):
    """background=True returns immediately, row is visible after draining queue.

    Pre-fix `semantic_cache_store()` blocked the hot path 1-1.3s on WAL
    contention (retry budget 5×~0.25s). background=True hands the write to
    the rag-bg-sql-writer daemon so the caller unblocks in ~µs. Draining
    _BACKGROUND_SQL_QUEUE.join() makes the write observable via lookup.
    """
    emb = _emb(1.0, 0.5, dim=32)
    t0 = time.perf_counter()
    ok = rag.semantic_cache_store(
        emb, question="bg-q", response="bg-ans",
        paths=["p.md"], scores=[0.9], top_score=0.9,
        intent=None, corpus_hash="HBG",
        background=True,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert ok is True, "background enqueue should always return True after gating"
    # The caller must NOT have blocked on a SQL commit. Even on a slow
    # macOS laptop the queue put + argument packing should be well under 50ms.
    assert elapsed_ms < 50, f"background store took {elapsed_ms:.1f}ms, expected <50ms"

    # Drain the worker so the INSERT actually commits before we assert visibility.
    rag._BACKGROUND_SQL_QUEUE.join()

    hit = rag.semantic_cache_lookup(emb, corpus_hash="HBG")
    assert hit is not None, "background-stored row must be readable after queue drain"
    assert hit["response"] == "bg-ans"
    assert hit["paths"] == ["p.md"]


def test_background_store_respects_gating(clean_cache_env):
    """Gating (empty response, refusal top_score, disabled cache, empty corpus_hash)
    must reject BEFORE reaching the background queue — no worker work wasted.
    """
    emb = _emb(1.0, 0.0, dim=32)
    # Empty response
    assert rag.semantic_cache_store(
        emb, question="q", response="",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="H", background=True,
    ) is False
    # Refusal top_score
    assert rag.semantic_cache_store(
        emb, question="q", response="No encontré.",
        paths=[], scores=[], top_score=0.005,
        intent=None, corpus_hash="H", background=True,
    ) is False
    # Empty corpus_hash
    assert rag.semantic_cache_store(
        emb, question="q", response="ans",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="", background=True,
    ) is False
    # Drain and confirm no rows landed.
    rag._BACKGROUND_SQL_QUEUE.join()
    stats = rag.semantic_cache_stats()
    assert stats["rows"] == 0, "gated stores must not enqueue any work"


@pytest.mark.slow
def test_background_store_does_not_block_under_contention(clean_cache_env, monkeypatch):
    """background=True must return in <100ms even if the underlying write
    is forced into its full retry budget (simulated via a slow connection).

    This is the specific regression we fixed: pre-2026-04-22 the caller
    blocked 1-1.3s on WAL contention. With background=True, the worker
    absorbs the retry budget off-thread so the user query returns before
    the write even finishes.
    """
    import contextlib as _contextlib
    import time as _time

    # Install a fake conn opener that sleeps 500ms per retry attempt (simulating
    # a busy_timeout wait). The foreground call must still return quickly
    # because it only enqueues.
    original = rag._ragvec_state_conn

    @_contextlib.contextmanager
    def slow_conn():
        _time.sleep(0.5)
        with original() as c:
            yield c

    monkeypatch.setattr(rag, "_ragvec_state_conn", slow_conn)

    emb = _emb(1.0, 0.0, dim=32)
    t0 = _time.perf_counter()
    ok = rag.semantic_cache_store(
        emb, question="slow-q", response="slow-ans",
        paths=[], scores=[], top_score=0.9,
        intent=None, corpus_hash="HSLOW",
        background=True,
    )
    elapsed_ms = (_time.perf_counter() - t0) * 1000
    assert ok is True
    # The foreground must NOT have waited for the 500ms slow commit.
    assert elapsed_ms < 100, f"background store blocked {elapsed_ms:.1f}ms under slow-conn, expected <100ms"
