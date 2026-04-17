"""Tests for orphan HNSW segment dir detection + WAL checkpoint."""
import sqlite3
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


def _make_fake_chroma(root: Path, live_ids: list[str], orphan_ids: list[str]):
    """Create a fake chroma.sqlite3 with a `segments` table + UUID dirs on disk."""
    root.mkdir(parents=True, exist_ok=True)
    sqlite_path = root / "chroma.sqlite3"
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute("CREATE TABLE segments (id TEXT, collection TEXT, type TEXT)")
    for sid in live_ids:
        conn.execute("INSERT INTO segments VALUES (?, 'coll1', 'vector')", (sid,))
    conn.commit()
    conn.close()

    for sid in live_ids + orphan_ids:
        d = root / sid
        d.mkdir()
        (d / "data.bin").write_bytes(b"x" * 1024)


def test_find_orphan_segment_dirs(tmp_path, monkeypatch):
    import rag
    live = [str(uuid.uuid4()) for _ in range(2)]
    orphan = [str(uuid.uuid4()) for _ in range(3)]
    _make_fake_chroma(tmp_path, live, orphan)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    found = rag._find_orphan_segment_dirs()
    found_names = {p.name for p, _ in found}
    assert found_names == set(orphan)
    # size is nonzero per orphan
    for _, sz in found:
        assert sz >= 1024


def test_find_orphan_ignores_non_uuid_dirs(tmp_path, monkeypatch):
    import rag
    _make_fake_chroma(tmp_path, live_ids=[str(uuid.uuid4())], orphan_ids=[])
    # A non-UUID subdir shouldn't be flagged (e.g., some unrelated dir)
    (tmp_path / "random-subdir").mkdir()
    (tmp_path / "random-subdir" / "x.bin").write_bytes(b"y" * 512)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    found = rag._find_orphan_segment_dirs()
    assert found == []


def test_find_orphan_missing_sqlite(tmp_path, monkeypatch):
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    # no chroma.sqlite3 → returns empty, doesn't raise
    assert rag._find_orphan_segment_dirs() == []


def test_prune_orphan_segment_dirs_dry_run(tmp_path, monkeypatch):
    import rag
    live = [str(uuid.uuid4())]
    orphan = [str(uuid.uuid4()) for _ in range(2)]
    _make_fake_chroma(tmp_path, live, orphan)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    result = rag._prune_orphan_segment_dirs(dry_run=True)
    assert result["count"] == 2
    assert result["bytes_freed"] >= 2 * 1024
    # Dirs still exist
    for sid in orphan:
        assert (tmp_path / sid).exists()


def test_prune_orphan_segment_dirs_live(tmp_path, monkeypatch):
    import rag
    live = [str(uuid.uuid4())]
    orphan = [str(uuid.uuid4()) for _ in range(2)]
    _make_fake_chroma(tmp_path, live, orphan)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    result = rag._prune_orphan_segment_dirs(dry_run=False)
    assert result["count"] == 2
    for sid in orphan:
        assert not (tmp_path / sid).exists()
    # Live dir untouched
    for sid in live:
        assert (tmp_path / sid).exists()


def test_prune_orphan_no_orphans(tmp_path, monkeypatch):
    import rag
    live = [str(uuid.uuid4())]
    _make_fake_chroma(tmp_path, live, orphan_ids=[])
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    result = rag._prune_orphan_segment_dirs(dry_run=False)
    assert result["count"] == 0
    assert result["bytes_freed"] == 0


def test_wal_checkpoint_dry_run(tmp_path, monkeypatch):
    import rag
    sqlite_path = tmp_path / "chroma.sqlite3"
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute("CREATE TABLE x (id INTEGER)")
    conn.execute("INSERT INTO x VALUES (1)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    result = rag._chroma_wal_checkpoint(dry_run=True)
    assert result["ok"]
    assert result.get("dry_run")
    assert result["before_bytes"] == result["after_bytes"]


def test_wal_checkpoint_missing_sqlite(tmp_path, monkeypatch):
    import rag
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    result = rag._chroma_wal_checkpoint(dry_run=False)
    assert not result["ok"]


def test_wal_checkpoint_live(tmp_path, monkeypatch):
    """WAL checkpoint on a real sqlite in WAL mode compacts the -wal file."""
    import rag
    sqlite_path = tmp_path / "chroma.sqlite3"
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE x (id INTEGER, blob BLOB)")
    for i in range(100):
        conn.execute("INSERT INTO x VALUES (?, ?)", (i, b"z" * 1024))
    conn.commit()
    # do NOT close yet so WAL has content
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    result = rag._chroma_wal_checkpoint(dry_run=False)
    assert result["ok"], result
    conn.close()
