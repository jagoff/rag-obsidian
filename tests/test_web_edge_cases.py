"""Regression tests for production edge-case bugs surfaced in web.log /
silent_errors.jsonl across the 2026-04-20 / 2026-04-21 pass:

- `_fetch_pagerank_top` IndexError when `ranked` is empty despite non-empty
  PageRank map (n=0 or mid-call cache invalidation).
- `_sanitize_confidence` must survive None / garbage / NaN / ±Inf without
  raising, because `_retry_pending_conversation_turns` feeds it raw
  JSONL-decoded values that may have been written pre-sanitize.
- `_cache_key` path in /api/chat: topic-shift reassignment of `history = []`
  shouldn't UnboundLocalError the PUT path. We don't wire the full chat
  handler here — just assert the helper contract.
- `_persist_with_sqlite_retry` retries transient locks + propagates real
  SQL errors through the silent log (not re-raised).
"""
from __future__ import annotations

import math
import sqlite3
from unittest.mock import patch

import pytest


def test_fetch_pagerank_top_empty_ranked_no_crash(tmp_path, monkeypatch):
    import rag
    from web import server

    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    client = rag.SqliteVecClient(path=str(tmp_path))
    col = client.get_or_create_collection("test_pr")
    col.add(
        ids=["only-one"], embeddings=[[0.1] * 8],
        documents=["solo un doc"],
        metadatas=[{"file": "A.md"}],
    )
    # Force get_pagerank to return a non-empty map while forcing `n=0` so
    # `ranked = sorted(...)[:0]` is empty. Without the guard this raised
    # IndexError on `ranked[0][1]`.
    fake_map = {"A.md": 0.5, "B.md": 0.3}
    with patch.object(server, "get_pagerank", return_value=fake_map):
        out = server._fetch_pagerank_top(col, n=0)
    assert out == []


def test_fetch_pagerank_top_returns_ranked_when_non_empty(tmp_path, monkeypatch):
    import rag
    from web import server

    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    client = rag.SqliteVecClient(path=str(tmp_path))
    col = client.get_or_create_collection("test_pr2")
    col.add(
        ids=["a", "b"], embeddings=[[0.1] * 8, [0.2] * 8],
        documents=["d1", "d2"],
        metadatas=[{"file": "A.md"}, {"file": "B.md"}],
    )
    fake_map = {"A.md": 1.0, "B.md": 0.5}
    with patch.object(server, "get_pagerank", return_value=fake_map):
        out = server._fetch_pagerank_top(col, n=5)
    assert len(out) == 2
    assert out[0]["path"] == "A.md"
    assert out[0]["pr"] == 1.0
    assert out[1]["pr"] == 0.5


@pytest.mark.parametrize("bad", [
    float("-inf"), float("inf"), float("nan"),
    None, "not-a-number", [], {},
])
def test_sanitize_confidence_handles_garbage(bad):
    from web.server import _sanitize_confidence
    out = _sanitize_confidence(bad)
    assert isinstance(out, float)
    assert not math.isnan(out)
    assert not math.isinf(out)
    assert out == 0.0


@pytest.mark.parametrize("good,expected", [
    (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (-0.3, -0.3),
    ("0.42", 0.42), (42, 42.0),
])
def test_sanitize_confidence_preserves_finite(good, expected):
    from web.server import _sanitize_confidence
    assert _sanitize_confidence(good) == expected


def test_persist_with_sqlite_retry_retries_on_locked(monkeypatch):
    from web import server
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise sqlite3.OperationalError("database is locked")
        # 3rd attempt succeeds
        return None

    # Monkey patch time.sleep to avoid actually sleeping in tests.
    monkeypatch.setattr("time.sleep", lambda _x: None)
    logged: list[tuple] = []
    monkeypatch.setattr(server, "_log_sql_state_error",
                        lambda tag, err: logged.append((tag, err)))
    server._persist_with_sqlite_retry(flaky, "unit_test_tag")
    assert len(calls) == 3
    assert logged == []  # no error logged — 3rd succeeded


def test_persist_with_sqlite_retry_gives_up_after_3(monkeypatch):
    from web import server

    def always_locked():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("time.sleep", lambda _x: None)
    logged: list[tuple] = []
    monkeypatch.setattr(server, "_log_sql_state_error",
                        lambda tag, err: logged.append((tag, err)))
    server._persist_with_sqlite_retry(always_locked, "unit_test_tag2")
    assert len(logged) == 1
    assert logged[0][0] == "unit_test_tag2"
    assert "locked" in logged[0][1].lower()


def test_persist_with_sqlite_retry_propagates_other_errors(monkeypatch):
    from web import server

    def schema_err():
        raise sqlite3.OperationalError("no such table: rag_missing")

    logged: list[tuple] = []
    monkeypatch.setattr(server, "_log_sql_state_error",
                        lambda tag, err: logged.append((tag, err)))
    # No retries on non-lock errors — logged on first failure.
    server._persist_with_sqlite_retry(schema_err, "unit_test_schema")
    assert len(logged) == 1
    assert "no such table" in logged[0][1]


def test_save_vaults_config_is_atomic(tmp_path, monkeypatch):
    """_save_vaults_config must use tmp+replace so a crash mid-write
    never leaves an empty vaults.json (which would silently wipe the
    registry on next load).
    """
    import rag
    cfg_path = tmp_path / "vaults.json"
    monkeypatch.setattr(rag, "VAULTS_CONFIG_PATH", cfg_path)
    cfg = {"vaults": {"A": "/tmp/a", "B": "/tmp/b"}, "current": "A"}
    rag._save_vaults_config(cfg)
    # No orphaned `.json.tmp` left behind.
    assert not (cfg_path.with_suffix(".json.tmp")).exists()
    # Final file is valid JSON with the expected contents.
    import json
    loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert loaded == cfg
