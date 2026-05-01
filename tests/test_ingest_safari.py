"""Tests for scripts/ingest_safari.py + `rag index --source safari`.

No live Safari DB reads — tests build a synthetic `History.db` SQLite
and a Bookmarks.plist file on disk. Covers:

  - URL domain extraction (happy, malformed, empty)
  - Cocoa epoch → Unix timestamp conversion
  - history SQL: aggregates per URL, filters load_successful=1 +
    redirect_source IS NULL + since floor, latest non-empty title wins
  - max_urls cap respected
  - Bookmarks plist walker: nested folders, Reading List detection,
    URIDictionary title extraction, PreviewText for RL entries
  - upsert_history + upsert_bookmarks write source=safari + kind meta
  - delete_* removes rows
  - run() orchestration: first pass, unchanged, changed, stale delete,
    separate bookmark vs reading_list tracking
  - integration: safari appears in rag.VALID_SOURCES + weights + retention
"""
from __future__ import annotations

import plistlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import rag
from scripts import ingest_safari as s


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_vault_col(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    (tmp_path / "ragvec").mkdir()
    conn = sqlite3.connect(str(tmp_path / "ragvec" / "ragvec.db"))
    conn.close()

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="saf_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)

    def _fake_embed(texts):
        return [[float(len(t) % 10) / 10] + [0.0] * 7 for t in texts]
    monkeypatch.setattr(rag, "embed", _fake_embed)
    return col


# ── URL parsing ─────────────────────────────────────────────────────────

def test_domain_happy_paths():
    assert s._domain("https://www.example.com/foo") == "www.example.com"
    assert s._domain("http://DOMAIN.IO/") == "domain.io"
    assert s._domain("https://sub.ex.ai:8080/path?q=1") == "sub.ex.ai"


def test_domain_edge_cases():
    # malformed / empty / non-URL strings shouldn't raise
    assert s._domain("") == ""
    assert s._domain(None) == ""  # type: ignore[arg-type]
    assert s._domain("not a url at all") == ""


# ── Cocoa timestamp ─────────────────────────────────────────────────────

def test_cocoa_to_unix_real_value():
    got = s._cocoa_to_unix(798581338.74)
    assert abs(got - 1776888538.74) < 0.001


def test_cocoa_to_unix_zero_and_negative():
    assert s._cocoa_to_unix(None) == 0.0
    assert s._cocoa_to_unix(0) == 0.0
    assert s._cocoa_to_unix(-500) == 0.0


# ── History reader ──────────────────────────────────────────────────────

def _make_fake_history_db(path: Path, rows: list[tuple]) -> None:
    """Write a synthetic History.db. `rows` is:
    [(hist_item_id, url, visit_count, visits=[(visit_time_cocoa, title,
    load_successful, redirect_source_or_None)])].
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript("""
            CREATE TABLE history_items (
                id INTEGER PRIMARY KEY,
                url TEXT NOT NULL,
                domain_expansion TEXT,
                visit_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE history_visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                history_item INTEGER NOT NULL,
                visit_time REAL NOT NULL,
                title TEXT,
                load_successful INTEGER NOT NULL DEFAULT 1,
                redirect_source INTEGER,
                redirect_destination INTEGER
            );
        """)
        for (hid, url, vcount, visits) in rows:
            conn.execute(
                "INSERT INTO history_items (id, url, visit_count) VALUES (?,?,?)",
                (hid, url, vcount),
            )
            for vt_cocoa, vtitle, vload, vredir in visits:
                conn.execute(
                    "INSERT INTO history_visits (history_item, visit_time, "
                    "title, load_successful, redirect_source) VALUES (?,?,?,?,?)",
                    (hid, vt_cocoa, vtitle, vload, vredir),
                )
        conn.commit()
    finally:
        conn.close()


def _cocoa(year, month, day, hour=12):
    return (datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp()
            - s.COCOA_EPOCH_OFFSET)


def test_read_history_aggregates_per_url(tmp_path):
    db = tmp_path / "History.db"
    _make_fake_history_db(db, [
        (1, "https://example.com/", 3, [
            (_cocoa(2026, 4, 1), "Old title",    1, None),
            (_cocoa(2026, 4, 20), "Latest title", 1, None),
            (_cocoa(2026, 4, 10), "Middle",       1, None),
        ]),
    ])
    h = s.read_history(db)
    assert len(h) == 1
    assert h[0].url == "https://example.com/"
    assert h[0].title == "Latest title"  # most-recent non-empty wins
    assert h[0].visit_count == 3
    assert h[0].first_visit_ts < h[0].last_visit_ts


def test_read_history_drops_failed_loads(tmp_path):
    db = tmp_path / "History.db"
    _make_fake_history_db(db, [
        (1, "https://broken.com/", 1, [
            (_cocoa(2026, 4, 20), "Network Error", 0, None),
        ]),
        (2, "https://ok.com/", 1, [
            (_cocoa(2026, 4, 20), "Real page", 1, None),
        ]),
    ])
    h = s.read_history(db)
    urls = {e.url for e in h}
    assert urls == {"https://ok.com/"}


def test_read_history_drops_redirects(tmp_path):
    db = tmp_path / "History.db"
    _make_fake_history_db(db, [
        (1, "https://t.co/xyz", 1, [
            (_cocoa(2026, 4, 20), "", 1, 5),   # has redirect_source → skip
        ]),
        (2, "https://landing.com/", 1, [
            (_cocoa(2026, 4, 20), "Landing", 1, None),
        ]),
    ])
    h = s.read_history(db)
    urls = {e.url for e in h}
    assert urls == {"https://landing.com/"}


def test_read_history_applies_since_floor(tmp_path):
    db = tmp_path / "History.db"
    _make_fake_history_db(db, [
        (1, "https://old.com/", 1, [(_cocoa(2025, 1, 1), "Old", 1, None)]),
        (2, "https://new.com/", 1, [(_cocoa(2026, 4, 1), "New", 1, None)]),
    ])
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    h = s.read_history(db, since_unix_ts=cutoff)
    assert len(h) == 1
    assert h[0].url == "https://new.com/"


def test_read_history_max_urls_cap(tmp_path):
    db = tmp_path / "History.db"
    rows = [
        (i, f"https://site{i}.com/", 1,
         [(_cocoa(2026, 4, i % 28 + 1), f"Site {i}", 1, None)])
        for i in range(1, 11)
    ]
    _make_fake_history_db(db, rows)
    h = s.read_history(db, max_urls=3)
    assert len(h) == 3


def test_read_history_missing_db_returns_empty(tmp_path):
    assert s.read_history(tmp_path / "nope.db") == []


# ── Bookmarks reader ────────────────────────────────────────────────────

def _make_fake_bookmarks_plist(path: Path, tree: dict) -> None:
    with open(path, "wb") as f:
        plistlib.dump(tree, f)


def _bm_leaf(url: str, title: str = "", uuid: str | None = None,
             reading_list_preview: str | None = None) -> dict:
    node: dict = {
        "WebBookmarkType": "WebBookmarkTypeLeaf",
        "WebBookmarkUUID": uuid or f"uuid-{abs(hash(url)) % 10**8:08d}",
        "URLString": url,
        "URIDictionary": {"title": title} if title else {},
    }
    if reading_list_preview is not None:
        node["ReadingList"] = {"PreviewText": reading_list_preview}
    return node


def _bm_folder(title: str, children: list, uuid: str | None = None) -> dict:
    return {
        "WebBookmarkType": "WebBookmarkTypeList",
        "WebBookmarkUUID": uuid or f"f-{abs(hash(title)) % 10**8:08d}",
        "Title": title,
        "Children": children,
    }


def test_read_bookmarks_flat_tree(tmp_path):
    plist = tmp_path / "Bookmarks.plist"
    # Real Safari Bookmarks.plist has the top-level root with Title="".
    _make_fake_bookmarks_plist(plist, _bm_folder("", [
        _bm_folder("BookmarksBar", [
            _bm_folder("00 Unsorted", [
                _bm_leaf("https://example.com/", "Example"),
                _bm_leaf("https://github.com/", "GitHub"),
            ]),
        ]),
    ]))
    bms = s.read_bookmarks(plist)
    assert len(bms) == 2
    # BookmarksBar is stripped from folder_path (internal folder), but
    # the subfolder "00 Unsorted" remains.
    titles = {b.title for b in bms}
    assert titles == {"Example", "GitHub"}
    folders = {b.folder_path for b in bms}
    assert folders == {"00 Unsorted"}
    assert all(not b.is_reading_list for b in bms)


def test_read_bookmarks_reading_list_subtree(tmp_path):
    plist = tmp_path / "Bookmarks.plist"
    _make_fake_bookmarks_plist(plist, _bm_folder("", [
        _bm_folder("BookmarksBar", [
            _bm_leaf("https://example.com/", "Normal BM"),
        ]),
        _bm_folder("com.apple.ReadingList", [
            _bm_leaf("https://long-article.com/",
                     "Long-form article",
                     reading_list_preview="First paragraph preview..."),
        ]),
    ]))
    bms = s.read_bookmarks(plist)
    rl = [b for b in bms if b.is_reading_list]
    bm = [b for b in bms if not b.is_reading_list]
    assert len(rl) == 1
    assert len(bm) == 1
    assert rl[0].folder_path == "Reading List"
    # Preview text is appended to title with " — ".
    assert "First paragraph preview" in rl[0].title


def test_read_bookmarks_missing_plist_returns_empty(tmp_path):
    assert s.read_bookmarks(tmp_path / "no.plist") == []


def test_read_bookmarks_skips_proxy_nodes(tmp_path):
    # Proxy nodes (History / RL shortcut) shouldn't emit a Bookmark.
    plist = tmp_path / "Bookmarks.plist"
    _make_fake_bookmarks_plist(plist, _bm_folder("", [
        {"WebBookmarkType": "WebBookmarkTypeProxy",
         "Title": "Historial"},
        _bm_folder("BookmarksBar", [
            _bm_leaf("https://real.com/", "Real"),
        ]),
    ]))
    bms = s.read_bookmarks(plist)
    assert len(bms) == 1
    assert bms[0].url == "https://real.com/"


# ── Content hash ────────────────────────────────────────────────────────

def test_history_hash_stable():
    h1 = s.HistoryEntry(1, "u", "d", "t", 3, 100.0, 200.0)
    h2 = s.HistoryEntry(1, "u", "d", "t", 3, 100.0, 200.0)
    assert s._history_hash(h1) == s._history_hash(h2)


def test_history_hash_changes_on_title_edit():
    h1 = s.HistoryEntry(1, "u", "d", "Old", 3, 100.0, 200.0)
    h2 = s.HistoryEntry(1, "u", "d", "New", 3, 100.0, 200.0)
    assert s._history_hash(h1) != s._history_hash(h2)


def test_bookmark_hash_includes_reading_list_flag():
    b1 = s.Bookmark("u1", "https://x.com/", "T", "F")
    b2 = s.Bookmark("u1", "https://x.com/", "T", "F", is_reading_list=True)
    assert s._bookmark_hash(b1) != s._bookmark_hash(b2)


# ── Body formatting ─────────────────────────────────────────────────────

def test_format_history_body_has_headline_and_url():
    h = s.HistoryEntry(1, "https://example.com/", "example.com",
                        "Example site", 5,
                        1700000000.0, 1800000000.0)
    body = s._format_history_body(h)
    assert "Safari: Example site" in body
    assert "URL: https://example.com/" in body
    assert "Dominio: example.com" in body
    assert "Visitas: 5" in body


def test_format_history_body_title_falls_back_to_url():
    h = s.HistoryEntry(1, "https://noname.com/", "noname.com", "", 1,
                        1700000000.0, 1800000000.0)
    body = s._format_history_body(h)
    assert body.startswith("Safari: https://noname.com/")


def test_format_bookmark_body_reading_list_marker():
    b = s.Bookmark("u1", "https://longread.com/", "Long-form",
                    "Reading List", is_reading_list=True)
    body = s._format_bookmark_body(b)
    assert body.startswith("Safari Reading List:")


def test_format_bookmark_body_bookmark_marker():
    b = s.Bookmark("u1", "https://x.com/", "X", "00 Unsorted")
    body = s._format_bookmark_body(b)
    assert body.startswith("Safari Bookmark:")


def test_format_bookmark_body_truncates():
    b = s.Bookmark("u1", "https://x.com/", "T", "y" * 2000)
    body = s._format_bookmark_body(b)
    assert len(body) <= s.CHUNK_MAX_CHARS


# ── Writer ──────────────────────────────────────────────────────────────

def test_upsert_history_writes_metadata(tmp_vault_col):
    col = tmp_vault_col
    h = s.HistoryEntry(42, "https://news.com/article",
                        "news.com", "Breaking news",
                        5, 1_700_000_000, 1_800_000_000)
    n = s.upsert_history(col, [h])
    assert n == 1
    got = col.get(where={"file": "safari://history/42"}, include=["metadatas"])
    assert got["ids"] == ["safari://history/42::0"]
    meta = got["metadatas"][0]
    assert meta["source"] == "safari"
    assert meta["kind"] == "history"
    assert meta["url"] == "https://news.com/article"
    assert meta["visit_count"] == 5
    assert meta["domain"] == "news.com"


def test_upsert_bookmarks_writes_reading_list_flag(tmp_vault_col):
    col = tmp_vault_col
    bm = s.Bookmark("uuid-bm", "https://bookmark.com/", "BM",
                     "00 Unsorted")
    rl = s.Bookmark("uuid-rl", "https://readlist.com/", "RL",
                     "Reading List", is_reading_list=True)
    s.upsert_bookmarks(col, [bm, rl])
    got_bm = col.get(where={"file": "safari://bm/uuid-bm"}, include=["metadatas"])
    got_rl = col.get(where={"file": "safari://rl/uuid-rl"}, include=["metadatas"])
    assert got_bm["metadatas"][0]["kind"] == "bookmark"
    assert got_bm["metadatas"][0]["is_reading_list"] == 0
    assert got_rl["metadatas"][0]["kind"] == "reading_list"
    assert got_rl["metadatas"][0]["is_reading_list"] == 1


def test_upsert_history_idempotent(tmp_vault_col):
    col = tmp_vault_col
    h = s.HistoryEntry(1, "https://x.com/", "x.com", "X", 1, 0, 0)
    s.upsert_history(col, [h])
    s.upsert_history(col, [h])
    got = col.get(where={"file": "safari://history/1"}, include=[])
    assert len(got["ids"]) == 1


def test_delete_history_removes_rows(tmp_vault_col):
    col = tmp_vault_col
    h = s.HistoryEntry(1, "https://x.com/", "x.com", "X", 1, 0, 0)
    s.upsert_history(col, [h])
    assert s.delete_history(col, [1]) == 1
    assert col.get(where={"file": "safari://history/1"}, include=[])["ids"] == []


# ── run() orchestration ────────────────────────────────────────────────

def _mk_hist(hid, url="https://x.com/", title="T", vcount=1):
    return s.HistoryEntry(hid, url, s._domain(url) or "x.com",
                           title, vcount,
                           1_700_000_000.0, 1_800_000_000.0)


def _mk_bm(uuid, url="https://b.com/", title="B",
             folder="Folder", is_rl=False):
    return s.Bookmark(uuid, url, title, folder, is_reading_list=is_rl)


def test_run_first_pass_indexes_both(tmp_vault_col):
    summary = s.run(
        history_fetch_fn=lambda: [_mk_hist(1), _mk_hist(2)],
        bookmarks_fetch_fn=lambda: [_mk_bm("u1"), _mk_bm("u2", is_rl=True)],
    )
    assert summary["history_fetched"] == 2
    assert summary["history_indexed"] == 2
    assert summary["bookmarks_fetched"] == 1
    assert summary["reading_list_fetched"] == 1
    assert summary["bookmarks_indexed"] == 2  # bm + rl both counted


def test_run_skip_bookmarks_flag(tmp_vault_col):
    summary = s.run(
        history_fetch_fn=lambda: [_mk_hist(1)],
        bookmarks_fetch_fn=lambda: [_mk_bm("u1")],
        skip_bookmarks=True,
    )
    assert summary["bookmarks_fetched"] == 0
    assert summary["bookmarks_indexed"] == 0
    assert summary["history_indexed"] == 1


def test_run_second_pass_unchanged_is_noop(tmp_vault_col):
    h = _mk_hist(1)
    s.run(history_fetch_fn=lambda: [h], bookmarks_fetch_fn=lambda: [])
    summary = s.run(history_fetch_fn=lambda: [h], bookmarks_fetch_fn=lambda: [])
    assert summary["history_indexed"] == 0
    assert summary["history_unchanged"] == 1


def test_run_detects_changed_history(tmp_vault_col):
    h1 = _mk_hist(1, title="Old title")
    h2 = _mk_hist(1, title="New title")
    s.run(history_fetch_fn=lambda: [h1], bookmarks_fetch_fn=lambda: [])
    summary = s.run(history_fetch_fn=lambda: [h2], bookmarks_fetch_fn=lambda: [])
    assert summary["history_indexed"] == 1
    assert summary["history_unchanged"] == 0


def test_run_deletes_stale_entries(tmp_vault_col):
    h1 = _mk_hist(1)
    h2 = _mk_hist(2)
    s.run(history_fetch_fn=lambda: [h1, h2], bookmarks_fetch_fn=lambda: [])
    summary = s.run(
        history_fetch_fn=lambda: [h1],  # h2 rolled off
        bookmarks_fetch_fn=lambda: [],
    )
    assert summary["history_deleted"] == 1


def test_run_reset_forces_reindex(tmp_vault_col):
    h = _mk_hist(1)
    s.run(history_fetch_fn=lambda: [h], bookmarks_fetch_fn=lambda: [])
    summary = s.run(
        history_fetch_fn=lambda: [h], bookmarks_fetch_fn=lambda: [],
        reset=True,
    )
    assert summary["history_indexed"] == 1
    assert summary["history_unchanged"] == 0


def test_run_dry_run_no_writes(tmp_vault_col):
    col = tmp_vault_col
    summary = s.run(
        history_fetch_fn=lambda: [_mk_hist(1)],
        bookmarks_fetch_fn=lambda: [_mk_bm("u1")],
        dry_run=True,
    )
    assert summary["history_indexed"] == 1
    assert summary["bookmarks_indexed"] == 1
    assert col.get(where={"file": "safari://history/1"}, include=[])["ids"] == []


def test_run_invalid_since_iso_errors(tmp_vault_col):
    summary = s.run(
        history_fetch_fn=lambda: [],
        bookmarks_fetch_fn=lambda: [],
        since_iso="bogus",
    )
    assert "error" in summary
    assert "--since" in summary["error"]


def test_run_missing_sources_error(tmp_vault_col):
    # Both paths explicitly point at non-existent files → error.
    missing = Path("/nonexistent/History.db")
    missing_plist = Path("/nonexistent/Bookmarks.plist")
    summary = s.run(
        history_db=missing, bookmarks_plist=missing_plist,
    )
    assert "error" in summary


# ── Entity extraction isolation (lock-fix regression tests) ─────────────

def test_add_batched_entity_extraction_called_once(tmp_vault_col, monkeypatch):
    """Entity extraction must run ONCE per _add_batched call, not once per
    batch. Pre-fix: it was called inside the batch loop → N calls per run."""
    col = tmp_vault_col
    call_log: list = []

    def _fake_extract(bodies, ids, metas, source):
        call_log.append({"ids": list(ids), "source": source})

    monkeypatch.setattr(rag, "_extract_and_index_entities_for_chunks", _fake_extract)

    # Build 3 batches worth of entries (batch size is 50; use 3 items so
    # there is a single batch in practice, but the key invariant is that
    # the mock is called exactly ONCE regardless of how many batches exist).
    n = s._ADD_BATCH_SIZE * 3 + 7  # force 4 batches
    ids = [f"safari://history/{i}::0" for i in range(n)]
    bodies = [f"body {i}" for i in range(n)]
    metas = [{"file": f"safari://history/{i}", "source": "safari"} for i in range(n)]
    dim = 8
    embeddings = [[0.1] * dim for _ in range(n)]

    s._add_batched(col, ids, embeddings, bodies, metas, "safari")

    assert len(call_log) == 1, (
        f"_extract_and_index_entities_for_chunks called {len(call_log)} times; expected 1"
    )
    # All IDs must be present in the single call.
    assert set(call_log[0]["ids"]) == set(ids)
    assert call_log[0]["source"] == "safari"


def test_add_batched_entity_extraction_failure_does_not_raise(tmp_vault_col, monkeypatch, tmp_path):
    """If entity extraction raises OperationalError (database is locked),
    _add_batched must NOT re-raise — the corpus chunks are already persisted.
    The error must be logged via rag._silent_log."""
    col = tmp_vault_col
    silent_calls: list = []

    def _exploding_extract(bodies, ids, metas, source):
        raise sqlite3.OperationalError("database is locked")

    def _fake_silent_log(tag, exc):
        silent_calls.append({"tag": tag, "exc": str(exc)})

    monkeypatch.setattr(rag, "_extract_and_index_entities_for_chunks", _exploding_extract)
    monkeypatch.setattr(rag, "_silent_log", _fake_silent_log)

    ids = ["safari://history/99::0"]
    bodies = ["body 99"]
    metas = [{"file": "safari://history/99", "source": "safari"}]
    embeddings = [[0.5] * 8]

    # Must not raise even though entity extraction explodes.
    s._add_batched(col, ids, embeddings, bodies, metas, "safari")

    # The chunk must be in the corpus (corpus write succeeded before the
    # entity extraction attempt).
    got = col.get(where={"file": "safari://history/99"}, include=[])
    assert got["ids"] == ["safari://history/99::0"], "Chunk missing from corpus post-failure"

    # Silent-fail must have been called with the safari-specific tag.
    assert len(silent_calls) == 1
    assert silent_calls[0]["tag"] == "safari_entity_extraction"
    assert "locked" in silent_calls[0]["exc"]


# ── Integration ─────────────────────────────────────────────────────────

def test_valid_sources_includes_safari():
    assert "safari" in rag.VALID_SOURCES


def test_source_weight_and_halflife_registered():
    assert rag.SOURCE_WEIGHTS["safari"] == 0.80
    assert rag.SOURCE_RECENCY_HALFLIFE_DAYS["safari"] == 90.0
    assert rag.SOURCE_RETENTION_DAYS["safari"] == 180
