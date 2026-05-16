from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta


def _schema_graph(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            mention_count INTEGER NOT NULL DEFAULT 0,
            first_seen TEXT,
            last_seen TEXT,
            UNIQUE(name, type)
        );
        CREATE TABLE entity_memoria (
            entity_id INTEGER NOT NULL,
            memoria_id TEXT NOT NULL,
            occurrences INTEGER NOT NULL DEFAULT 1,
            extracted_at TEXT NOT NULL,
            UNIQUE(entity_id, memoria_id)
        );
        """
    )


def test_memo_graph_reads_sqlite_directly_without_memo_package(tmp_path, monkeypatch):
    import web.memo_v06 as memo_v06

    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path))
    memo_v06.cache_invalidate()
    db = tmp_path / "graph.db"
    with sqlite3.connect(db) as conn:
        _schema_graph(conn)
        conn.executemany(
            "INSERT INTO entities(id, name, type, mention_count) VALUES (?, ?, ?, ?)",
            [
                (1, "MLX", "technology", 3),
                (2, "memo", "project", 2),
                (3, "Qwen", "technology", 1),
            ],
        )
        conn.executemany(
            "INSERT INTO entity_memoria(entity_id, memoria_id, extracted_at) "
            "VALUES (?, ?, ?)",
            [
                (1, "m1", "2026-05-16T00:00:00Z"),
                (2, "m1", "2026-05-16T00:00:00Z"),
                (1, "m2", "2026-05-16T00:00:00Z"),
                (2, "m2", "2026-05-16T00:00:00Z"),
                (3, "m2", "2026-05-16T00:00:00Z"),
                (1, "m3", "2026-05-16T00:00:00Z"),
            ],
        )

    out = memo_v06.graph(limit_nodes=10, min_count=1)

    assert out["ok"] is True
    assert out["stats"] == {"entities": 3, "links": 6, "memorias": 3}
    assert [n["name"] for n in out["nodes"]] == ["MLX", "memo", "Qwen"]
    edge_by_pair = {
        tuple(sorted((out["nodes"][e["source"]]["name"], out["nodes"][e["target"]]["name"]))): e["weight"]
        for e in out["edges"]
    }
    assert edge_by_pair[("MLX", "memo")] == 2
    assert edge_by_pair[("MLX", "Qwen")] == 1
    assert edge_by_pair[("Qwen", "memo")] == 1


def test_memo_graph_missing_db_returns_empty_graph(tmp_path, monkeypatch):
    import web.memo_v06 as memo_v06

    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path))
    memo_v06.cache_invalidate()

    out = memo_v06.graph()

    assert out["ok"] is True
    assert out["nodes"] == []
    assert out["edges"] == []
    assert out["stats"]["entities"] == 0


def test_memo_temporal_timeline_reads_history_db_directly(tmp_path, monkeypatch):
    import web.memo_v06 as memo_v06

    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path))
    memo_v06.cache_invalidate()
    db = tmp_path / "history.db"
    now = datetime.now(UTC)
    yesterday = now - timedelta(days=1)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, op TEXT, "
            "record_id TEXT, title TEXT, type TEXT, delta_json TEXT)"
        )
        conn.executemany(
            "INSERT INTO events(ts, op, record_id) VALUES (?, ?, ?)",
            [
                (now.isoformat(), "save", "a"),
                (now.isoformat(), "update", "a"),
                (yesterday.isoformat(), "delete", "b"),
            ],
        )

    out = memo_v06.temporal_timeline(days=3)

    assert out["ok"] is True
    totals = out["total"]
    assert totals == {"saves": 1, "updates": 1, "deletes": 1}

