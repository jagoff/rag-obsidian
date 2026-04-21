"""Tests for orphan collection detection + WAL checkpoint (sqlite-vec backend)."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



# ---------------------------------------------------------------------------
# _find_orphan_segment_dirs — stub under sqlite-vec
# ---------------------------------------------------------------------------

def test_find_orphan_segment_dirs_returns_empty():
    """sqlite-vec stores everything in ragvec.db; no HNSW dirs exist."""
    import rag
    assert rag._find_orphan_segment_dirs() == []


# ---------------------------------------------------------------------------
# _find_orphan_collections
# ---------------------------------------------------------------------------

def test_find_orphan_collections_no_db(tmp_path, monkeypatch):
    """Missing ragvec.db → returns [] without raising."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    result = rag._find_orphan_collections()
    assert result == []


def test_find_orphan_collections_known_protected(tmp_path, monkeypatch):
    """Active collection names are never classified as orphans."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    client = rag.SqliteVecClient(path=str(tmp_path))
    # Create the active collections so list_collections returns them.
    client.get_or_create_collection(rag.COLLECTION_NAME)
    client.get_or_create_collection(rag.URLS_COLLECTION_NAME)
    result = rag._find_orphan_collections()
    assert result == []


def test_find_orphan_collections_detects_stale(tmp_path, monkeypatch):
    """Collections not in the known set are returned as orphans."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    client = rag.SqliteVecClient(path=str(tmp_path))
    client.get_or_create_collection("obsidian_notes_v7_deadbeef")  # stale schema
    result = rag._find_orphan_collections()
    assert "obsidian_notes_v7_deadbeef" in result


# ---------------------------------------------------------------------------
# _vec_wal_checkpoint
# ---------------------------------------------------------------------------

def test_wal_checkpoint_missing_db(tmp_path, monkeypatch):
    """No ragvec.db → returns ok=False."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    result = rag._vec_wal_checkpoint(dry_run=False)
    assert not result["ok"]


def test_wal_checkpoint_dry_run(tmp_path, monkeypatch):
    """dry_run=True on an existing ragvec.db returns ok=True, sizes equal."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    # SqliteVecClient creates ragvec.db inside tmp_path.
    client = rag.SqliteVecClient(path=str(tmp_path))
    client.get_or_create_collection("test_col")

    result = rag._vec_wal_checkpoint(dry_run=True)
    assert result["ok"]
    assert result.get("dry_run")
    assert result["before_bytes"] == result["after_bytes"]


def test_wal_checkpoint_live(tmp_path, monkeypatch):
    """Live checkpoint on a WAL-mode ragvec.db returns ok=True."""
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    db_path = tmp_path / "ragvec.db"

    # Write some data so the WAL has content.
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (id INTEGER, blob BLOB)")
    for i in range(100):
        conn.execute("INSERT INTO t VALUES (?, ?)", (i, b"z" * 1024))
    conn.commit()
    # Do NOT close yet so WAL has unflushed content.

    result = rag._vec_wal_checkpoint(dry_run=False)
    assert result["ok"], result
    conn.close()
