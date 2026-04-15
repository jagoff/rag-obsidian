"""Tests for Chrome bookmarks integration into the URL sub-index."""
import json
from pathlib import Path

import chromadb
import pytest

import rag


@pytest.fixture
def fake_embed(monkeypatch):
    def _embed(texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setattr(rag, "embed", _embed)


@pytest.fixture
def tmp_urls_col(tmp_path, monkeypatch, fake_embed):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="urls_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_urls_db", lambda: col)
    return col


def _bookmarks_json(tree: dict) -> str:
    return json.dumps({"roots": tree})


def _write_chrome_tree(base: Path, profile: str, tree: dict) -> Path:
    pdir = base / profile
    pdir.mkdir(parents=True, exist_ok=True)
    f = pdir / "Bookmarks"
    f.write_text(_bookmarks_json(tree), encoding="utf-8")
    return f


# ── chrome_bookmark_files ────────────────────────────────────────────────────


def test_chrome_bookmark_files_empty_when_chrome_missing(tmp_path):
    assert rag.chrome_bookmark_files(tmp_path / "nope") == []


def test_chrome_bookmark_files_lists_profiles_with_bookmarks(tmp_path):
    _write_chrome_tree(tmp_path, "Default", {"bookmark_bar": {"children": []}})
    _write_chrome_tree(tmp_path, "Profile 1", {"bookmark_bar": {"children": []}})
    # A profile dir without a Bookmarks file is skipped.
    (tmp_path / "Guest Profile").mkdir()
    found = rag.chrome_bookmark_files(tmp_path)
    names = [name for name, _ in found]
    assert names == ["Default", "Profile 1"]


# ── parse_chrome_bookmarks ───────────────────────────────────────────────────


def test_parse_chrome_bookmarks_basic_tree(tmp_path):
    tree = {
        "bookmark_bar": {
            "name": "Bookmarks bar",
            "children": [
                {
                    "type": "url",
                    "name": "Claude docs",
                    "url": "https://docs.anthropic.com/",
                    "date_added": "13348080000000000",  # 2024-01-01-ish
                },
                {
                    "type": "folder",
                    "name": "Work",
                    "children": [
                        {
                            "type": "url",
                            "name": "Anthropic console",
                            "url": "https://console.anthropic.com/",
                        },
                    ],
                },
            ],
        },
        "other": {"name": "Other", "children": []},
    }
    f = _write_chrome_tree(tmp_path, "Default", tree)
    bms = rag.parse_chrome_bookmarks(f)
    assert len(bms) == 2
    titles = sorted(b["title"] for b in bms)
    assert titles == ["Anthropic console", "Claude docs"]
    nested = next(b for b in bms if b["title"] == "Anthropic console")
    assert nested["folder"].endswith("Work")


def test_parse_chrome_bookmarks_filters_non_http(tmp_path):
    tree = {
        "bookmark_bar": {
            "name": "Bar",
            "children": [
                {"type": "url", "name": "javascript", "url": "javascript:void(0)"},
                {"type": "url", "name": "ftp", "url": "ftp://example.com"},
                {"type": "url", "name": "ok", "url": "https://ok.example"},
            ],
        },
    }
    f = _write_chrome_tree(tmp_path, "Default", tree)
    bms = rag.parse_chrome_bookmarks(f)
    urls = [b["url"] for b in bms]
    assert urls == ["https://ok.example"]


def test_parse_chrome_bookmarks_handles_malformed_json(tmp_path):
    pdir = tmp_path / "Default"
    pdir.mkdir()
    (pdir / "Bookmarks").write_text("not json at all", encoding="utf-8")
    assert rag.parse_chrome_bookmarks(pdir / "Bookmarks") == []


def test_parse_chrome_bookmarks_missing_roots(tmp_path):
    pdir = tmp_path / "Default"
    pdir.mkdir()
    (pdir / "Bookmarks").write_text(json.dumps({"nope": {}}), encoding="utf-8")
    assert rag.parse_chrome_bookmarks(pdir / "Bookmarks") == []


def test_webkit_ts_to_iso_conversion():
    # 13348080000000000 microseconds ≈ 2024-01-01
    iso = rag._webkit_ts_to_iso("13348080000000000")
    assert iso.startswith("2024-01-01") or iso.startswith("2023-12-")
    assert rag._webkit_ts_to_iso(None) == ""
    assert rag._webkit_ts_to_iso("garbage") == ""


def test_bookmark_embed_text_combines_fields():
    txt = rag._bookmark_embed_text(
        title="Claude docs",
        folder_breadcrumb="Bookmarks bar > Work",
        url="https://docs.anthropic.com/claude/api",
    )
    assert "Claude docs" in txt
    assert "Work" in txt
    assert "docs.anthropic.com" in txt


# ── _index_chrome_bookmarks (idempotent) ─────────────────────────────────────


def _bms(*pairs):
    return [{"url": u, "title": t, "folder": "Bar", "date_added": ""} for u, t in pairs]


def test_index_chrome_bookmarks_writes_rows(tmp_urls_col):
    col = tmp_urls_col
    n = rag._index_chrome_bookmarks(col, "Default", _bms(
        ("https://a.example", "A"),
        ("https://b.example", "B"),
    ))
    assert n == 2
    got = col.get(where={"source": "bookmark"}, include=["metadatas"])
    urls = sorted((m or {}).get("url", "") for m in got["metadatas"])
    assert urls == ["https://a.example", "https://b.example"]


def test_index_chrome_bookmarks_dedups_duplicate_urls(tmp_urls_col):
    col = tmp_urls_col
    n = rag._index_chrome_bookmarks(col, "Default", _bms(
        ("https://a.example", "A"),
        ("https://a.example", "A duplicate"),
    ))
    assert n == 1


def test_index_chrome_bookmarks_is_idempotent(tmp_urls_col):
    col = tmp_urls_col
    rag._index_chrome_bookmarks(col, "Default", _bms(
        ("https://a.example", "A"), ("https://b.example", "B"),
    ))
    # Re-sync with different set — old rows must be replaced, not added.
    rag._index_chrome_bookmarks(col, "Default", _bms(
        ("https://c.example", "C"),
    ))
    got = col.get(where={"source": "bookmark"}, include=["metadatas"])
    urls = sorted((m or {}).get("url", "") for m in got["metadatas"])
    assert urls == ["https://c.example"]


def test_index_chrome_bookmarks_keeps_other_profiles(tmp_urls_col):
    col = tmp_urls_col
    rag._index_chrome_bookmarks(col, "Default", _bms(("https://a.example", "A")))
    rag._index_chrome_bookmarks(col, "Profile 1", _bms(("https://b.example", "B")))
    # Re-sync Default only — Profile 1 must stay intact.
    rag._index_chrome_bookmarks(col, "Default", _bms(("https://z.example", "Z")))
    got = col.get(where={"source": "bookmark"}, include=["metadatas"])
    per_profile: dict[str, list[str]] = {}
    for m in got["metadatas"]:
        per_profile.setdefault((m or {}).get("profile", ""), []).append(m["url"])
    assert sorted(per_profile["Default"]) == ["https://z.example"]
    assert sorted(per_profile["Profile 1"]) == ["https://b.example"]


def test_index_chrome_bookmarks_empty_clears_profile(tmp_urls_col):
    col = tmp_urls_col
    rag._index_chrome_bookmarks(col, "Default", _bms(("https://a.example", "A")))
    rag._index_chrome_bookmarks(col, "Default", [])
    got = col.get(where={"source": "bookmark"}, include=[])
    assert got["ids"] == []


# ── sync_chrome_bookmarks (end-to-end with fake Chrome dir) ──────────────────


def test_sync_chrome_bookmarks_all_profiles(tmp_urls_col, tmp_path, monkeypatch):
    # Point the resolver at our tmp_path and populate two profiles.
    monkeypatch.setattr(rag, "_chrome_bookmarks_root", lambda: tmp_path)
    _write_chrome_tree(tmp_path, "Default", {
        "bookmark_bar": {"name": "Bar", "children": [
            {"type": "url", "name": "A", "url": "https://a.example"},
        ]},
    })
    _write_chrome_tree(tmp_path, "Profile 1", {
        "bookmark_bar": {"name": "Bar", "children": [
            {"type": "url", "name": "B", "url": "https://b.example"},
            {"type": "url", "name": "C", "url": "https://c.example"},
        ]},
    })
    stats = rag.sync_chrome_bookmarks()
    assert stats["profiles"] == 2
    assert stats["total"] == 3
    assert stats["per_profile"] == {"Default": 1, "Profile 1": 2}


def test_sync_chrome_bookmarks_specific_profile(tmp_urls_col, tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "_chrome_bookmarks_root", lambda: tmp_path)
    _write_chrome_tree(tmp_path, "Default", {
        "bookmark_bar": {"name": "Bar", "children": [
            {"type": "url", "name": "A", "url": "https://a.example"},
        ]},
    })
    _write_chrome_tree(tmp_path, "Profile 1", {
        "bookmark_bar": {"name": "Bar", "children": [
            {"type": "url", "name": "B", "url": "https://b.example"},
        ]},
    })
    stats = rag.sync_chrome_bookmarks(profile="Profile 1")
    assert stats["profiles"] == 1
    assert stats["total"] == 1
    assert "Profile 1" in stats["per_profile"]


def test_sync_chrome_bookmarks_chrome_missing(tmp_urls_col, tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "_chrome_bookmarks_root", lambda: tmp_path / "nope")
    stats = rag.sync_chrome_bookmarks()
    assert stats == {"profiles": 0, "total": 0, "per_profile": {}}


# ── find_urls source filter ──────────────────────────────────────────────────


def test_row_matches_source_legacy_rows_are_notes():
    assert rag._row_matches_source({}, "note") is True
    assert rag._row_matches_source({"source": None}, "note") is True
    assert rag._row_matches_source({"source": "bookmark"}, "note") is False
    assert rag._row_matches_source({"source": "bookmark"}, "bookmark") is True


def test_find_urls_filters_by_source(tmp_urls_col, monkeypatch):
    col = tmp_urls_col
    # 2 note rows (one with explicit source, one legacy), 2 bookmark rows.
    col.add(
        ids=["note::1", "note::legacy", "bm::1", "bm::2"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]] * 4,
        documents=["note one", "note legacy", "bookmark one", "bookmark two"],
        metadatas=[
            {"file": "n1.md", "url": "https://n1.example",
             "anchor": "", "line": 1, "source": "note"},
            {"file": "n2.md", "url": "https://n2.example",
             "anchor": "", "line": 1},  # legacy — no source field
            {"file": "chrome-bookmark::Default", "url": "https://b1.example",
             "anchor": "B1", "line": 0, "source": "bookmark",
             "profile": "Default"},
            {"file": "chrome-bookmark::Default", "url": "https://b2.example",
             "anchor": "B2", "line": 0, "source": "bookmark",
             "profile": "Default"},
        ],
    )

    # Force reranker to a degenerate mock so scores come straight from distances.
    class _FakeReranker:
        def predict(self, pairs, show_progress_bar=False):
            return [1.0] * len(pairs)

    monkeypatch.setattr(rag, "get_reranker", lambda: _FakeReranker())
    monkeypatch.setattr(rag, "_maybe_backfill_urls", lambda: None)

    # No filter → all four surface (subject to per-file cap).
    items = rag.find_urls("anything", k=10)
    assert len(items) >= 3

    # source=note → legacy row (no field) + explicit note.
    notes_only = rag.find_urls("anything", k=10, source="note")
    urls = {it["url"] for it in notes_only}
    assert urls == {"https://n1.example", "https://n2.example"}

    # source=bookmark → only bookmark rows.
    bms_only = rag.find_urls("anything", k=10, source="bookmark")
    urls = {it["url"] for it in bms_only}
    assert urls == {"https://b1.example", "https://b2.example"}
    assert all(it["source"] == "bookmark" for it in bms_only)
    assert all(it["profile"] == "Default" for it in bms_only)
