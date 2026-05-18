from __future__ import annotations

import json
import sqlite3
from datetime import datetime


def _init_memo_state(state_dir, data_dir) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(state_dir / "memvec.db") as conn:
        conn.executescript(
            """
            CREATE TABLE meta (
                id TEXT PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                tags TEXT NOT NULL,
                created TEXT NOT NULL,
                updated TEXT NOT NULL,
                body_hash TEXT NOT NULL,
                extra_json TEXT
            );
            CREATE TABLE vec(id TEXT PRIMARY KEY, embedding BLOB);
            CREATE VIRTUAL TABLE fts USING fts5(
                id UNINDEXED, title, tags, body,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
    with sqlite3.connect(state_dir / "graph.db") as conn:
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


def _insert_memo(state_dir, data_dir, *, id_, path, title, tags, updated=None) -> None:
    now = updated or datetime.now().astimezone().isoformat(timespec="milliseconds")
    body = f"Body for {title}\n"
    (data_dir / path).write_text(
        "---\n"
        f"id: {id_}\n"
        f"title: {title}\n"
        "tags:\n"
        + "".join(f"- {tag}\n" for tag in tags)
        + "type: note\n"
        f"created: {now}\n"
        f"updated: {now}\n"
        "---\n\n"
        + body,
        encoding="utf-8",
    )
    with sqlite3.connect(state_dir / "memvec.db") as conn:
        conn.execute(
            "INSERT INTO meta("
            "id, path, title, type, tags, created, updated, body_hash, extra_json"
            ") VALUES (?, ?, ?, 'note', ?, ?, ?, 'hash', NULL)",
            (id_, path, title, json.dumps(tags), now, now),
        )
        conn.execute("INSERT INTO vec(id, embedding) VALUES (?, X'00')", (id_,))
        conn.execute(
            "INSERT INTO fts(id, title, tags, body) VALUES (?, ?, ?, ?)",
            (id_, title, " ".join(tags), body),
        )


def test_memo_delete_api_deletes_db_file_history_and_graph(tmp_path, monkeypatch):
    import web.memo_routes as memo_routes

    state_dir = tmp_path / "state"
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MEMO_STATE_DIR", str(state_dir))
    monkeypatch.setenv("MEMO_DATA_DIR", str(data_dir))
    _init_memo_state(state_dir, data_dir)
    mid = "a" * 32
    _insert_memo(state_dir, data_dir, id_=mid, path="a.md", title="A", tags=["one"])
    with sqlite3.connect(state_dir / "graph.db") as conn:
        conn.execute(
            "INSERT INTO entities(id, name, type, mention_count) VALUES (1, 'memo', 'project', 1)"
        )
        conn.execute(
            "INSERT INTO entity_memoria(entity_id, memoria_id, extracted_at) VALUES (1, ?, 'now')",
            (mid,),
        )

    out = memo_routes.memo_delete_api([mid])

    assert out["errors"] == []
    assert out["deleted"] == [mid]
    assert not (data_dir / "a.md").exists()
    with sqlite3.connect(state_dir / "memvec.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM meta WHERE id = ?", (mid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM vec WHERE id = ?", (mid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM fts WHERE id = ?", (mid,)).fetchone()[0] == 0
    with sqlite3.connect(state_dir / "history.db") as conn:
        op = conn.execute("SELECT op FROM events WHERE record_id = ?", (mid,)).fetchone()[0]
        assert op == "delete"
    with sqlite3.connect(state_dir / "graph.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM entity_memoria WHERE memoria_id = ?", (mid,)
        ).fetchone()[0] == 0
        assert conn.execute("SELECT mention_count FROM entities WHERE id = 1").fetchone()[0] == 0


def test_memo_merge_api_unions_tags_and_deletes_loser(tmp_path, monkeypatch):
    import web.memo_routes as memo_routes

    state_dir = tmp_path / "state"
    data_dir = tmp_path / "data"
    monkeypatch.setenv("MEMO_STATE_DIR", str(state_dir))
    monkeypatch.setenv("MEMO_DATA_DIR", str(data_dir))
    _init_memo_state(state_dir, data_dir)
    keep = "a" * 32
    drop = "b" * 32
    _insert_memo(
        state_dir,
        data_dir,
        id_=keep,
        path="keep.md",
        title="Keep",
        tags=["one", "two"],
        updated="2026-05-17T12:00:00+00:00",
    )
    _insert_memo(
        state_dir,
        data_dir,
        id_=drop,
        path="drop.md",
        title="Drop",
        tags=["two", "three"],
        updated="2026-05-17T11:00:00+00:00",
    )

    out = memo_routes.memo_merge_api([{"a": keep, "b": drop}])

    assert out["errors"] == []
    assert out["merged"] == [{"kept": keep, "deleted": drop}]
    assert not (data_dir / "drop.md").exists()
    keep_text = (data_dir / "keep.md").read_text(encoding="utf-8")
    assert '"three"' in keep_text
    with sqlite3.connect(state_dir / "memvec.db") as conn:
        tags = json.loads(conn.execute("SELECT tags FROM meta WHERE id = ?", (keep,)).fetchone()[0])
        assert tags == ["one", "three", "two"]
        assert conn.execute("SELECT COUNT(*) FROM meta WHERE id = ?", (drop,)).fetchone()[0] == 0
    with sqlite3.connect(state_dir / "history.db") as conn:
        ops = [r[0] for r in conn.execute("SELECT op FROM events ORDER BY id")]
        assert ops == ["update", "delete"]
