"""Tests for collection safety invariants.

Covers:
1. _find_orphan_collections never orphans base collections.
2. _find_orphan_collections protects base collections even when COLLECTION_NAME is vault-suffixed.
3. _log_collection_op writes a valid JSON line to the ops log.
4. Sentinel invalidates _db_singleton so get_db() returns a fresh collection.
5. _collection_write_lock is mutually exclusive between threads.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import rag


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_fake_collection(name: str) -> MagicMock:
    m = MagicMock()
    m.name = name
    return m


# ── test 1: base collections are never orphaned ────────────────────────────────


def test_find_orphan_never_orphans_base_collections(tmp_path, monkeypatch):
    """obsidian_notes_v9 and obsidian_urls_v1 must never appear in orphan list."""
    orphan_name = "obsidian_notes_v7_abc12345"
    all_cols = [
        _make_fake_collection(rag._COLLECTION_BASE),
        _make_fake_collection(rag._URLS_COLLECTION_BASE),
        _make_fake_collection(orphan_name),
    ]

    fake_client = MagicMock()
    fake_client.list_collections.return_value = all_cols

    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "chroma")
    # Redirect ops log so we don't touch the real one.
    monkeypatch.setattr(rag, "COLLECTION_OPS_LOG", tmp_path / "ops.log")

    with patch("rag.SqliteVecClient", return_value=fake_client):
        result = rag._find_orphan_collections()

    assert rag._COLLECTION_BASE not in result
    assert rag._URLS_COLLECTION_BASE not in result
    assert orphan_name in result


# ── test 2: base collections protected even with vault-suffixed COLLECTION_NAME ─


def test_find_orphan_protects_base_when_collection_name_is_suffixed(tmp_path, monkeypatch):
    """Even when COLLECTION_NAME is obsidian_notes_v9_741d239c (work vault),
    the unsuffixed obsidian_notes_v9 must stay in `known` and not be orphaned."""
    suffixed_name = "obsidian_notes_v9_741d239c"
    suffixed_urls = "obsidian_urls_v1_741d239c"
    orphan_name = "obsidian_notes_v7_deadbeef"

    all_cols = [
        _make_fake_collection(rag._COLLECTION_BASE),    # default vault — must be protected
        _make_fake_collection(rag._URLS_COLLECTION_BASE),
        _make_fake_collection(suffixed_name),           # current vault — must be protected
        _make_fake_collection(suffixed_urls),
        _make_fake_collection(orphan_name),
    ]

    fake_client = MagicMock()
    fake_client.list_collections.return_value = all_cols

    monkeypatch.setattr(rag, "COLLECTION_NAME", suffixed_name)
    monkeypatch.setattr(rag, "URLS_COLLECTION_NAME", suffixed_urls)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "chroma")
    monkeypatch.setattr(rag, "COLLECTION_OPS_LOG", tmp_path / "ops.log")

    with patch("rag.SqliteVecClient", return_value=fake_client):
        result = rag._find_orphan_collections()

    assert rag._COLLECTION_BASE not in result, "base collection wrongly classified as orphan"
    assert rag._URLS_COLLECTION_BASE not in result, "base URL collection wrongly classified as orphan"
    assert suffixed_name not in result, "current-vault collection wrongly classified as orphan"
    assert orphan_name in result


# ── test 3: _log_collection_op writes valid JSON ───────────────────────────────


def test_log_collection_op_writes_valid_json_line(tmp_path, monkeypatch):
    """_log_collection_op must append one JSON line with expected fields."""
    ops_log = tmp_path / "ops.log"
    monkeypatch.setattr(rag, "COLLECTION_OPS_LOG", ops_log)

    rag._log_collection_op("test_op", "test_coll", {"note": "x"})

    lines = ops_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["op"] == "test_op"
    assert entry["collection"] == "test_coll"
    assert entry.get("extra") == {"note": "x"} or entry.get("note") == "x"
    assert "ts" in entry
    assert "pid" in entry
    assert isinstance(entry["stack"], list)
    assert len(entry["stack"]) <= 5


# ── test 4: sentinel invalidates _db_singleton ────────────────────────────────


def test_sentinel_invalidates_db_singleton(tmp_path, monkeypatch):
    """Writing a newer sentinel should force get_db() to create a fresh collection."""
    call_counter = {"n": 0}

    def _make_col():
        call_counter["n"] += 1
        col = MagicMock()
        col.id = f"col-id-{call_counter['n']}"
        return col

    fake_client = MagicMock()
    fake_client.get_or_create_collection.side_effect = lambda **_: _make_col()

    sentinel_path = tmp_path / "sentinel"
    monkeypatch.setattr(rag, "COLLECTION_RESET_SENTINEL", sentinel_path)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path / "chroma")
    (tmp_path / "chroma").mkdir(parents=True, exist_ok=True)

    # Reset module-level singleton so the test is isolated.
    monkeypatch.setattr(rag, "_db_singleton", None)
    monkeypatch.setattr(rag, "_db_singleton_created_at", 0.0)

    with patch("rag.SqliteVecClient", return_value=fake_client):
        col_a = rag.get_db()
        id_a = col_a.id

        # Write sentinel with a future mtime so it appears newer than the cached handle.
        future_ts = time.time() + 60
        sentinel_path.write_text("9999999999 new-uuid-abc")
        os.utime(sentinel_path, (future_ts, future_ts))

        col_b = rag.get_db()
        id_b = col_b.id

    assert id_a != id_b, "sentinel did not invalidate the cached collection"


# ── test 5: _collection_write_lock is mutually exclusive ──────────────────────


def test_collection_write_lock_is_exclusive(tmp_path, monkeypatch):
    """Two threads entering the lock must not overlap; total time >= hold_time * 2."""
    monkeypatch.setattr(rag, "COLLECTION_WRITE_LOCK", tmp_path / "lock")

    hold_secs = 0.5
    events: list[tuple[str, float]] = []

    def _worker():
        with rag._collection_write_lock(timeout=10):
            events.append(("enter", time.monotonic()))
            time.sleep(hold_secs)
            events.append(("exit", time.monotonic()))

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)

    start = time.monotonic()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    elapsed = time.monotonic() - start

    assert elapsed >= hold_secs * 2, (
        f"Expected total elapsed >= {hold_secs * 2:.2f}s but got {elapsed:.2f}s — "
        "lock may not be exclusive"
    )

    # Verify the two critical sections did not overlap.
    enter_times = [ts for name, ts in events if name == "enter"]
    exit_times = [ts for name, ts in events if name == "exit"]
    assert len(enter_times) == 2
    # The second enter must happen after the first exit.
    min_exit = min(exit_times)
    max_enter_second = max(enter_times)
    # One thread entered after the other exited.
    assert max_enter_second >= min_exit, (
        "Second thread entered lock before first thread released it"
    )


# ── test 6: corpus cache invalidates on collection UUID change ────────────────


def test_corpus_cache_invalidates_on_collection_id_change(monkeypatch):
    """Concurrent `index --reset` deletes + recreates the collection with a
    fresh UUID. If the post-reindex chunk count lands on the same number,
    the pre-reset BM25 cache is stale but `count` alone wouldn't detect it.
    The cache must compare against the current collection UUID too — otherwise
    a long-lived chat process keeps serving BM25 ids that have been wiped.
    """
    monkeypatch.setattr(rag, "_corpus_cache", None)

    def _fake_col(cid: str, n: int) -> MagicMock:
        col = MagicMock()
        col.id = cid
        col.count.return_value = n
        col.get.return_value = {
            "ids": [f"doc::{i}" for i in range(n)],
            "documents": [f"body {i}" for i in range(n)],
            "metadatas": [
                {"note": f"Note{i}", "file": f"f{i}.md", "folder": "X", "tags": "", "outlinks": ""}
                for i in range(n)
            ],
        }
        return col

    col_a = _fake_col("uuid-A", 5)
    corpus_a = rag._load_corpus(col_a)
    assert corpus_a["collection_id"] == "uuid-A"
    assert corpus_a["ids"] == [f"doc::{i}" for i in range(5)]

    # Same count but new UUID (simulates delete + recreate with identical row count).
    col_b = _fake_col("uuid-B", 5)
    col_b.get.return_value = {
        "ids": [f"fresh::{i}" for i in range(5)],
        "documents": [f"fresh body {i}" for i in range(5)],
        "metadatas": [
            {"note": f"Fresh{i}", "file": f"g{i}.md", "folder": "Y", "tags": "", "outlinks": ""}
            for i in range(5)
        ],
    }
    corpus_b = rag._load_corpus(col_b)
    assert corpus_b["collection_id"] == "uuid-B"
    assert corpus_b["ids"] == [f"fresh::{i}" for i in range(5)], (
        "BM25 cache did not rebuild after collection UUID change — stale ids would be served"
    )

    # Same UUID + same count → cache reused (no rebuild).
    col_b.get.reset_mock()
    corpus_b2 = rag._load_corpus(col_b)
    assert corpus_b2 is corpus_b
    col_b.get.assert_not_called()
