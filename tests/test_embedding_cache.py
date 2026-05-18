from __future__ import annotations


def test_embed_texts_cached_stores_and_reuses_exact_matches(tmp_path, monkeypatch):
    from rag.embedding_cache import embed_texts_cached

    monkeypatch.setenv("RAG_INDEX_EMBED_CACHE", "1")
    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(len(t)), float(sum(t.encode("utf-8")))] for t in texts]

    texts = ["a", "bb", "a"]
    out1, stats1 = embed_texts_cached(
        texts,
        db_dir=tmp_path,
        model_id="model-a",
        namespace="index-payload-v1",
        embed_fn=fake_embed,
    )

    assert calls == [["a", "bb"]]
    assert stats1.hits == 0
    assert stats1.misses == 3
    assert stats1.stores == 2

    def fail_embed(texts: list[str]) -> list[list[float]]:
        raise AssertionError(f"unexpected embed call: {texts!r}")

    out2, stats2 = embed_texts_cached(
        texts,
        db_dir=tmp_path,
        model_id="model-a",
        namespace="index-payload-v1",
        embed_fn=fail_embed,
    )

    assert out2 == out1
    assert stats2.hits == 3
    assert stats2.misses == 0
    assert stats2.stores == 0


def test_index_payload_cache_is_explicitly_enabled_under_pytest(tmp_path, monkeypatch):
    import rag

    monkeypatch.setenv("RAG_INDEX_EMBED_CACHE", "1")
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "ragvec")
    monkeypatch.setattr(rag, "EMBED_MODEL", "test-model")
    monkeypatch.setattr(rag, "_resolve_embed_backend", lambda: "mlx")

    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(rag, "embed", fake_embed)
    texts = ["same", "other", "same"]

    out1 = rag._embed_index_payloads_with_persistent_cache(texts)
    assert calls == [["same", "other"]]
    calls.clear()
    out2 = rag._embed_index_payloads_with_persistent_cache(texts)

    assert out1 == out2
    assert calls == []


def test_index_cache_default_is_off_under_pytest(monkeypatch):
    import rag

    monkeypatch.delenv("RAG_INDEX_EMBED_CACHE", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")

    assert rag._index_embed_cache_enabled() is False

    monkeypatch.setenv("RAG_INDEX_EMBED_CACHE", "1")
    assert rag._index_embed_cache_enabled() is True


def test_index_auto_mlx_batch_tuning():
    import rag

    assert rag._index_auto_batch_target(backend="mlx", mem_gb=36) == 96
    assert rag._index_auto_embed_slice_size(96, backend="mlx", mem_gb=36) == 48
    assert rag._index_auto_local_embed_batch_size(backend="mlx", mem_gb=36) == 48
