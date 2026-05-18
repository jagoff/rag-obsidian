from __future__ import annotations


def test_run_index_full_safe_sets_resource_defaults_temporarily(monkeypatch):
    import rag

    monkeypatch.delenv("RAG_INDEX_LLM_ENRICHMENTS", raising=False)
    keys = tuple(rag._index_full_safe_defaults().keys())
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("RAG_INDEX_FULL_SAFE", raising=False)

    captured: dict[str, str | None] = {}

    def fake_run_index_with_env(reset: bool, no_contradict: bool) -> dict:
        captured.update({key: rag.os.environ.get(key) for key in keys})
        return {"reset": reset, "no_contradict": no_contradict}

    monkeypatch.setattr(rag, "_run_index_with_env", fake_run_index_with_env)

    result = rag._run_index(reset=True, no_contradict=False)

    assert result == {"reset": True, "no_contradict": False}
    assert captured["RAG_EXTRACT_ENTITIES"] == "0"
    assert captured["OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY"] == "1"
    assert captured["OBSIDIAN_RAG_SKIP_SYNTHETIC_Q"] == "1"
    assert captured["RAG_CONTEXTUAL_RETRIEVAL"] == "0"
    assert captured["RAG_INDEX_EMBED_SLICE_SIZE"] == "auto"
    assert captured["RAG_INDEX_FILE_CHUNK_SLICE_SIZE"] == "128"
    assert captured["RAG_INDEX_LOCAL_EMBED_BATCH"] == "auto"
    assert captured["RAG_INDEX_BATCH_EMBEDS"] == "1"
    assert captured["RAG_INDEX_BATCH_SIZE"] == "auto"
    assert captured["RAG_INDEX_BATCH_SLEEP_MS"] == "50"
    for key in keys:
        assert rag.os.environ.get(key) is None


def test_run_index_full_safe_respects_explicit_overrides(monkeypatch):
    import rag

    monkeypatch.delenv("RAG_INDEX_FULL_SAFE", raising=False)
    monkeypatch.setenv("RAG_EXTRACT_ENTITIES", "1")
    monkeypatch.setenv("RAG_INDEX_EMBED_SLICE_SIZE", "32")

    captured: dict[str, str | None] = {}

    def fake_run_index_with_env(reset: bool, no_contradict: bool) -> dict:
        del reset, no_contradict
        captured["entities"] = rag.os.environ.get("RAG_EXTRACT_ENTITIES")
        captured["slice"] = rag.os.environ.get("RAG_INDEX_EMBED_SLICE_SIZE")
        captured["sleep"] = rag.os.environ.get("RAG_INDEX_BATCH_SLEEP_MS")
        return {}

    monkeypatch.setattr(rag, "_run_index_with_env", fake_run_index_with_env)

    rag._run_index(reset=True, no_contradict=False)

    assert captured == {"entities": "1", "slice": "32", "sleep": "50"}
    assert rag.os.environ.get("RAG_EXTRACT_ENTITIES") == "1"
    assert rag.os.environ.get("RAG_INDEX_EMBED_SLICE_SIZE") == "32"
    assert rag.os.environ.get("RAG_INDEX_BATCH_SLEEP_MS") is None


def test_run_index_full_enrichment_safe_can_be_disabled(monkeypatch):
    import rag

    monkeypatch.delenv("RAG_INDEX_LLM_ENRICHMENTS", raising=False)
    for key in rag._index_full_safe_defaults():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAG_INDEX_FULL_SAFE", "0")

    captured: dict[str, str | None] = {}

    def fake_run_index_with_env(reset: bool, no_contradict: bool) -> dict:
        del reset, no_contradict
        captured["entities"] = rag.os.environ.get("RAG_EXTRACT_ENTITIES")
        captured["slice"] = rag.os.environ.get("RAG_INDEX_EMBED_SLICE_SIZE")
        return {}

    monkeypatch.setattr(rag, "_run_index_with_env", fake_run_index_with_env)

    rag._run_index(reset=True, no_contradict=False)

    assert captured == {"entities": "0", "slice": "auto"}


def test_run_index_safe_can_be_disabled_entirely(monkeypatch):
    import rag

    monkeypatch.delenv("RAG_INDEX_LLM_ENRICHMENTS", raising=False)
    for key in rag._index_full_safe_defaults():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAG_INDEX_FULL_SAFE", "0")
    monkeypatch.setenv("RAG_INDEX_SAFE", "0")

    captured: dict[str, str | None] = {}

    def fake_run_index_with_env(reset: bool, no_contradict: bool) -> dict:
        del reset, no_contradict
        captured["entities"] = rag.os.environ.get("RAG_EXTRACT_ENTITIES")
        captured["slice"] = rag.os.environ.get("RAG_INDEX_EMBED_SLICE_SIZE")
        return {}

    monkeypatch.setattr(rag, "_run_index_with_env", fake_run_index_with_env)

    rag._run_index(reset=True, no_contradict=False)

    assert captured == {"entities": None, "slice": None}


def test_run_index_incremental_applies_base_safe(monkeypatch):
    import rag

    monkeypatch.delenv("RAG_INDEX_LLM_ENRICHMENTS", raising=False)
    for key in rag._index_full_safe_defaults():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("RAG_INDEX_SAFE", raising=False)

    captured: dict[str, str | None] = {}

    def fake_run_index_with_env(reset: bool, no_contradict: bool) -> dict:
        del reset, no_contradict
        captured["entities"] = rag.os.environ.get("RAG_EXTRACT_ENTITIES")
        captured["slice"] = rag.os.environ.get("RAG_INDEX_EMBED_SLICE_SIZE")
        captured["file_slice"] = rag.os.environ.get("RAG_INDEX_FILE_CHUNK_SLICE_SIZE")
        captured["local_batch"] = rag.os.environ.get("RAG_INDEX_LOCAL_EMBED_BATCH")
        captured["batch"] = rag.os.environ.get("RAG_INDEX_BATCH_EMBEDS")
        captured["batch_size"] = rag.os.environ.get("RAG_INDEX_BATCH_SIZE")
        captured["abort"] = rag.os.environ.get("RAG_INDEX_ABORT_ON_MEMORY_PRESSURE")
        captured["used_abort"] = rag.os.environ.get("RAG_INDEX_ABORT_USED_PCT")
        captured["swap_guard"] = rag.os.environ.get("RAG_MEMORY_PRESSURE_SWAP_GB")
        captured["sleep"] = rag.os.environ.get("RAG_INDEX_MEMORY_PRESSURE_SLEEP_S")
        captured["swap_abort"] = rag.os.environ.get("RAG_INDEX_ABORT_SWAP_GB")
        captured["rss_abort"] = rag.os.environ.get("RAG_INDEX_ABORT_SELF_RSS_GB")
        captured["skip_ctx"] = rag.os.environ.get("OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY")
        captured["skip_synth"] = rag.os.environ.get("OBSIDIAN_RAG_SKIP_SYNTHETIC_Q")
        captured["contextual"] = rag.os.environ.get("RAG_CONTEXTUAL_RETRIEVAL")
        captured["skip_contra"] = rag.os.environ.get("RAG_INDEX_SKIP_CONTRADICTIONS")
        return {}

    monkeypatch.setattr(rag, "_run_index_with_env", fake_run_index_with_env)

    rag._run_index(reset=False, no_contradict=False)

    assert captured == {
        "entities": "0",
        "slice": "auto",
        "file_slice": "128",
        "local_batch": "auto",
        "batch": "1",
        "batch_size": "auto",
        "abort": "1",
        "used_abort": "0",
        "swap_guard": "0",
        "sleep": "1",
        "swap_abort": "0",
        "rss_abort": "18.0",
        "skip_ctx": "1",
        "skip_synth": "1",
        "contextual": "0",
        "skip_contra": "1",
    }


def test_run_index_safe_allows_llm_enrichment_opt_in(monkeypatch):
    import rag

    for key in rag._index_full_safe_defaults():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAG_INDEX_LLM_ENRICHMENTS", "1")

    captured: dict[str, str | None] = {}

    def fake_run_index_with_env(reset: bool, no_contradict: bool) -> dict:
        del reset, no_contradict
        captured["skip_ctx"] = rag.os.environ.get("OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY")
        captured["skip_synth"] = rag.os.environ.get("OBSIDIAN_RAG_SKIP_SYNTHETIC_Q")
        captured["contextual"] = rag.os.environ.get("RAG_CONTEXTUAL_RETRIEVAL")
        captured["skip_contra"] = rag.os.environ.get("RAG_INDEX_SKIP_CONTRADICTIONS")
        captured["slice"] = rag.os.environ.get("RAG_INDEX_EMBED_SLICE_SIZE")
        return {}

    monkeypatch.setattr(rag, "_run_index_with_env", fake_run_index_with_env)

    rag._run_index(reset=False, no_contradict=False)

    assert captured == {
        "skip_ctx": None,
        "skip_synth": None,
        "contextual": None,
        "skip_contra": None,
        "slice": "auto",
    }


def test_index_contradictions_disabled_by_safe_env(monkeypatch):
    import rag

    monkeypatch.setenv("RAG_INDEX_SKIP_CONTRADICTIONS", "1")

    assert rag._index_contradictions_enabled(reset=False, no_contradict=False) is False


def test_index_contradictions_can_be_enabled_explicitly(monkeypatch):
    import rag

    monkeypatch.setenv("RAG_INDEX_SKIP_CONTRADICTIONS", "0")

    assert rag._index_contradictions_enabled(reset=False, no_contradict=False) is True
    assert rag._index_contradictions_enabled(reset=True, no_contradict=False) is False
    assert rag._index_contradictions_enabled(reset=False, no_contradict=True) is False


def test_run_index_inner_slices_large_file_chunks(tmp_path, monkeypatch):
    import rag

    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "large.md"
    note.write_text("# Large\n\nbody " * 100, encoding="utf-8")

    class FakeCollection:
        def __init__(self):
            self.add_batch_sizes: list[int] = []

        def get(self, **kwargs):
            del kwargs
            return {"ids": [], "metadatas": []}

        def add(self, *, ids, embeddings, documents, metadatas):
            assert len(ids) == len(embeddings) == len(documents) == len(metadatas)
            self.add_batch_sizes.append(len(ids))

        def delete(self, **kwargs):
            del kwargs

    col = FakeCollection()
    url_col = FakeCollection()
    embed_call_sizes: list[int] = []

    def fake_embed(texts: list[str]):
        embed_call_sizes.append(len(texts))
        return [[0.0] for _ in texts]

    chunks = [(f"embed {i}", f"display {i}", f"parent {i}") for i in range(130)]

    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "get_urls_db", lambda: url_col)
    monkeypatch.setattr(rag, "embed", fake_embed)
    monkeypatch.setattr(rag, "semantic_chunks", lambda *a, **kw: chunks)
    monkeypatch.setattr(rag, "get_context_summary", lambda *a, **kw: "")
    monkeypatch.setattr(rag, "get_synthetic_questions", lambda *a, **kw: [])
    monkeypatch.setattr(
        rag._contextual_retrieval,
        "contextualize_chunks",
        lambda **kw: kw["embed_texts"],
    )
    monkeypatch.setattr(rag, "_save_context_cache", lambda: None)
    monkeypatch.setattr(rag, "_save_synthetic_q_cache", lambda: None)

    monkeypatch.setenv("RAG_SKIP_CROSS_SOURCE_ETLS", "1")
    monkeypatch.setenv("RAG_INDEX_BATCH_EMBEDS", "0")
    monkeypatch.setenv("RAG_INDEX_FILE_CHUNK_SLICE_SIZE", "64")
    monkeypatch.setenv("RAG_INDEX_EMBED_SLICE_SIZE", "8")
    monkeypatch.setenv("RAG_EXTRACT_ENTITIES", "0")
    monkeypatch.setenv("OBSIDIAN_RAG_SKIP_SYNTHETIC_Q", "1")
    monkeypatch.setenv("RAG_INDEX_MEMORY_GUARD_INTERVAL_S", "0")

    rag._run_index_inner(reset=False, no_contradict=True, col=col)

    assert col.add_batch_sizes == [64, 64, 2]
    assert max(embed_call_sizes) <= 8


def test_run_index_inner_removes_stale_chunks_when_file_becomes_empty(
    tmp_path,
    monkeypatch,
):
    import rag

    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "emptying.md"
    note.write_text("# Note\n\nbody " * 30, encoding="utf-8")

    client = rag.SqliteVecClient(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="bulk_empty_main", metadata={"hnsw:space": "cosine"}
    )
    url_col = client.get_or_create_collection(
        name="bulk_empty_urls", metadata={"hnsw:space": "cosine"}
    )

    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "get_urls_db", lambda: url_col)
    monkeypatch.setattr(rag, "embed", lambda texts: [[0.0] for _ in texts])
    monkeypatch.setattr(rag, "_save_context_cache", lambda: None)
    monkeypatch.setattr(rag, "_save_synthetic_q_cache", lambda: None)
    monkeypatch.setenv("RAG_SKIP_CROSS_SOURCE_ETLS", "1")
    monkeypatch.setenv("RAG_INDEX_NICE", "0")
    monkeypatch.setenv("RAG_INDEX_BATCH_EMBEDS", "0")
    monkeypatch.setenv("RAG_INDEX_MEMORY_GUARD_INTERVAL_S", "0")
    monkeypatch.setenv("RAG_INDEX_PREFLIGHT_MEMORY_GUARD", "0")
    monkeypatch.setenv("RAG_EXTRACT_ENTITIES", "0")
    monkeypatch.setenv("OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY", "1")
    monkeypatch.setenv("OBSIDIAN_RAG_SKIP_SYNTHETIC_Q", "1")

    rag._run_index_inner(reset=False, no_contradict=True, col=col)
    assert col.get(where={"file": "emptying.md"}, include=[])["ids"]

    note.write_text("---\ntag: empty\n---\n", encoding="utf-8")
    result = rag._run_index_inner(reset=False, no_contradict=True, col=col)

    assert result["empty_files"] == 1
    assert col.get(where={"file": "emptying.md"}, include=[])["ids"] == []


def test_run_index_inner_deletes_stale_urls_when_file_loses_links(
    tmp_path,
    monkeypatch,
):
    import rag

    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "links.md"
    note.write_text("# Links\n\n[old](https://example.com/old) body " * 20, encoding="utf-8")

    client = rag.SqliteVecClient(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="bulk_url_main", metadata={"hnsw:space": "cosine"}
    )
    url_col = client.get_or_create_collection(
        name="bulk_url_urls", metadata={"hnsw:space": "cosine"}
    )

    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "get_urls_db", lambda: url_col)
    monkeypatch.setattr(rag, "embed", lambda texts: [[0.0] for _ in texts])
    monkeypatch.setattr(rag, "_save_context_cache", lambda: None)
    monkeypatch.setattr(rag, "_save_synthetic_q_cache", lambda: None)
    monkeypatch.setenv("RAG_SKIP_CROSS_SOURCE_ETLS", "1")
    monkeypatch.setenv("RAG_INDEX_NICE", "0")
    monkeypatch.setenv("RAG_INDEX_BATCH_EMBEDS", "0")
    monkeypatch.setenv("RAG_INDEX_MEMORY_GUARD_INTERVAL_S", "0")
    monkeypatch.setenv("RAG_INDEX_PREFLIGHT_MEMORY_GUARD", "0")
    monkeypatch.setenv("RAG_EXTRACT_ENTITIES", "0")
    monkeypatch.setenv("OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY", "1")
    monkeypatch.setenv("OBSIDIAN_RAG_SKIP_SYNTHETIC_Q", "1")

    rag._run_index_inner(reset=False, no_contradict=True, col=col)
    assert url_col.get(where={"file": "links.md"}, include=[])["ids"]

    note.write_text("# Links\n\nsin links ahora " * 20, encoding="utf-8")
    rag._run_index_inner(reset=False, no_contradict=True, col=col)

    assert url_col.get(where={"file": "links.md"}, include=[])["ids"] == []


def test_run_index_inner_skips_unchanged_local_non_vault_source(
    tmp_path,
    monkeypatch,
):
    import rag

    vault = tmp_path / "vault"
    memory_dir = vault / "99-obsidian" / "99-AI" / "memory"
    memory_dir.mkdir(parents=True)
    note = memory_dir / "decision.md"
    raw = "# Decision\n\nKeep the cached hash stable.\n"
    note.write_text(raw, encoding="utf-8")
    rel = str(note.relative_to(vault))
    h = rag._file_hash_with_images(raw, note, vault)

    class FakeCollection:
        def __init__(self):
            self.get_calls: list[dict] = []
            self.deleted: list[dict] = []

        def get(self, **kwargs):
            self.get_calls.append(kwargs)
            return {
                "ids": ["memory::0", "gmail::0"],
                "metadatas": [
                    {"file": rel, "hash": h, "source": "memory"},
                    {
                        "file": "gmail://message/thread-1",
                        "hash": "uri-hash",
                        "source": "gmail",
                    },
                ],
            }

        def add(self, **kwargs):
            raise AssertionError(f"unchanged local note was reindexed: {kwargs}")

        def delete(self, **kwargs):
            self.deleted.append(kwargs)

    col = FakeCollection()
    url_col = FakeCollection()

    def unexpected(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unchanged local note should skip chunking/embedding")

    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "get_urls_db", lambda: url_col)
    monkeypatch.setattr(rag, "is_excluded", lambda rel_path: False)
    monkeypatch.setattr(rag, "_load_vaults_config", lambda: {})
    monkeypatch.setattr(rag, "semantic_chunks", unexpected)
    monkeypatch.setattr(rag, "embed", unexpected)
    monkeypatch.setattr(rag, "_save_context_cache", lambda: None)
    monkeypatch.setattr(rag, "_save_synthetic_q_cache", lambda: None)
    monkeypatch.setattr(rag, "_invalidate_corpus_cache", lambda: None)
    monkeypatch.setattr(rag, "_vlm_caption_budget_reset", lambda: None)

    monkeypatch.setenv("RAG_SKIP_CROSS_SOURCE_ETLS", "1")
    monkeypatch.setenv("RAG_INDEX_NICE", "0")
    monkeypatch.setenv("RAG_INDEX_BATCH_EMBEDS", "0")
    monkeypatch.setenv("RAG_INDEX_MEMORY_GUARD_INTERVAL_S", "0")
    monkeypatch.setenv("RAG_INDEX_PREFLIGHT_MEMORY_GUARD", "0")
    monkeypatch.setenv("RAG_EXTRACT_ENTITIES", "0")
    monkeypatch.setenv("OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY", "1")
    monkeypatch.setenv("OBSIDIAN_RAG_SKIP_SYNTHETIC_Q", "1")
    monkeypatch.setenv("RAG_CONTEXTUAL_RETRIEVAL", "0")

    result = rag._run_index_inner(reset=False, no_contradict=True, col=col)

    assert result["added_chunks"] == 0
    assert result["updated_files"] == 0
    assert result["orphans"] == 0
    assert result["total_files"] == 1
    assert col.deleted == []
    assert url_col.deleted == []
    assert col.get_calls == [{"include": ["metadatas"]}]


def test_get_db_cache_is_namespaced_by_collection(tmp_path, monkeypatch):
    import rag

    calls: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, path):
            self.path = path

        def get_or_create_collection(self, *, name, metadata):
            del metadata
            calls.append((self.path, name))
            return {"collection": name}

    db_path = tmp_path / "ragvec"
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    monkeypatch.setattr(rag, "COLLECTION_RESET_SENTINEL", tmp_path / "sentinel")
    monkeypatch.setattr(rag, "SqliteVecClient", FakeClient)
    monkeypatch.setattr(rag, "_db_singleton", None)
    monkeypatch.setattr(rag, "_db_singleton_created_at", 0.0)
    monkeypatch.setattr(rag, "COLLECTION_NAME", "obsidian_notes_v12")

    home = rag.get_db()
    home_again = rag.get_db()
    monkeypatch.setattr(rag, "COLLECTION_NAME", "obsidian_notes_v12_b02bbec8")
    finances = rag.get_db()

    assert home == {"collection": "obsidian_notes_v12"}
    assert home_again is home
    assert finances == {"collection": "obsidian_notes_v12_b02bbec8"}
    assert calls == [
        (str(db_path), "obsidian_notes_v12"),
        (str(db_path), "obsidian_notes_v12_b02bbec8"),
    ]


def test_run_index_inner_does_not_mass_delete_after_empty_scan(
    tmp_path,
    monkeypatch,
):
    import rag

    vault = tmp_path / "empty-vault"
    vault.mkdir()

    class FakeCollection:
        def __init__(self):
            self.deleted: list[dict] = []

        def get(self, **kwargs):
            assert kwargs == {"include": ["metadatas"]}
            return {
                "ids": ["old::0", "old::1"],
                "metadatas": [
                    {"file": "old-a.md", "hash": "a", "source": "vault"},
                    {"file": "old-b.md", "hash": "b", "source": "memory"},
                ],
            }

        def add(self, **kwargs):
            raise AssertionError(kwargs)

        def delete(self, **kwargs):
            self.deleted.append(kwargs)

    col = FakeCollection()
    url_col = FakeCollection()

    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "get_urls_db", lambda: url_col)
    monkeypatch.setattr(rag, "is_excluded", lambda rel_path: False)
    monkeypatch.setattr(rag, "_load_vaults_config", lambda: {})
    monkeypatch.setattr(rag, "_save_context_cache", lambda: None)
    monkeypatch.setattr(rag, "_save_synthetic_q_cache", lambda: None)
    monkeypatch.setattr(rag, "_invalidate_corpus_cache", lambda: None)
    monkeypatch.setattr(rag, "_vlm_caption_budget_reset", lambda: None)
    monkeypatch.setattr(rag, "_silent_log", lambda *args, **kwargs: None)

    monkeypatch.setenv("RAG_SKIP_CROSS_SOURCE_ETLS", "1")
    monkeypatch.setenv("RAG_INDEX_NICE", "0")
    monkeypatch.setenv("RAG_INDEX_MEMORY_GUARD_INTERVAL_S", "0")
    monkeypatch.setenv("RAG_INDEX_PREFLIGHT_MEMORY_GUARD", "0")
    monkeypatch.delenv("RAG_INDEX_ALLOW_EMPTY_ORPHAN_CLEANUP", raising=False)

    result = rag._run_index_inner(reset=False, no_contradict=True, col=col)

    assert result["added_chunks"] == 0
    assert result["orphans"] == 0
    assert result["total_files"] == 0
    assert col.deleted == []
    assert url_col.deleted == []
