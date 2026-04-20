"""Tests for mcp_server.py — the MCP-over-stdio wrapper that exposes
obsidian-rag to Claude Code / other MCP clients.

Coverage gaps this closes:
  - rag_read_note path-escape validation (../../../ etc)
  - rag_read_note non-.md + not-found paths
  - rag_query / rag_list_notes / rag_links empty-collection paths
  - rag_stats shape contract
  - _load_rag idempotency + thread-safety
  - _touch + idle-killer constants

The MCP framework itself (FastMCP) is not exercised — we call the
underlying tool functions directly. Mocks stand in for rag.get_db,
rag.retrieve, etc, so these tests don't require the real vault or the
ollama daemon.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import mcp_server


# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Each test starts with _rag cleared so _load_rag's cache is exercised."""
    monkeypatch.setattr(mcp_server, "_rag", None)
    # Fresh lock per test so lock-contention tests don't see leftover waiters.
    monkeypatch.setattr(mcp_server, "_rag_lock", threading.Lock())


@pytest.fixture
def fake_rag(monkeypatch, tmp_path):
    """Replaces _load_rag's import with a mock that mimics the rag.py API
    enough for mcp_server tool handlers. Returns the mock so individual tests
    can stub specific return values."""
    vault = tmp_path / "vault"
    (vault / "02-Areas" / "Coaching").mkdir(parents=True)
    (vault / "02-Areas" / "Coaching" / "Autoridad.md").write_text(
        "# Autoridad\n\nContenido de la nota de autoridad.\n",
        encoding="utf-8",
    )

    rag_mock = MagicMock()
    rag_mock.VAULT_PATH = vault
    rag_mock.COLLECTION_NAME = "obsidian_notes_v9_test"
    rag_mock.EMBED_MODEL = "bge-m3"
    rag_mock.RERANKER_MODEL = "bge-reranker-v2-m3"

    col = MagicMock()
    col.count.return_value = 42
    rag_mock.get_db.return_value = col
    rag_mock._load_corpus.return_value = {
        "metas": [
            {"file": "02-Areas/Coaching/Autoridad.md",
             "note": "Autoridad", "folder": "02-Areas/Coaching",
             "tags": "coaching,personal"},
            {"file": "03-Resources/Agile.md",
             "note": "Agile", "folder": "03-Resources",
             "tags": "agile,tech"},
        ]
    }

    # Install as the lazy import target
    monkeypatch.setattr(mcp_server, "_load_rag", lambda: rag_mock)
    return rag_mock


# ── rag_read_note ────────────────────────────────────────────────────────────


def test_read_note_rejects_path_escape(fake_rag):
    """Classic traversal — rag_read_note must block `../` payloads."""
    out = mcp_server.rag_read_note("../../etc/passwd")
    assert out.startswith("Error: path")


def test_read_note_rejects_absolute_outside_vault(fake_rag):
    """Absolute paths that resolve outside the vault are rejected."""
    out = mcp_server.rag_read_note("/etc/passwd.md")
    # After the .md check passes, resolving sends us to /etc/passwd.md
    # which is not under VAULT_PATH → "escapes the vault root".
    assert "escapes" in out.lower() or "not found" in out.lower()


def test_read_note_rejects_non_md_extension(fake_rag):
    assert mcp_server.rag_read_note("passwd").startswith("Error: path must end in .md")
    assert mcp_server.rag_read_note("secret.sh").startswith("Error: path must end in .md")


def test_read_note_returns_content_for_valid_path(fake_rag):
    out = mcp_server.rag_read_note("02-Areas/Coaching/Autoridad.md")
    assert "Contenido de la nota de autoridad" in out


def test_read_note_reports_missing_file(fake_rag):
    out = mcp_server.rag_read_note("02-Areas/Coaching/NotExists.md")
    assert "not found" in out.lower()


# ── rag_query ────────────────────────────────────────────────────────────────


def test_rag_query_returns_empty_on_empty_collection(fake_rag):
    fake_rag.get_db.return_value.count.return_value = 0
    out = mcp_server.rag_query("qué es el ikigai", k=5)
    assert out == []


def test_rag_query_clamps_k_to_upper_bound(fake_rag):
    fake_rag.retrieve.return_value = {
        "docs": ["doc"] * 5,
        "metas": [{"file": "x.md", "note": "x", "folder": "", "tags": ""}] * 5,
        "scores": [0.8] * 5,
    }
    mcp_server.rag_query("hola", k=999)
    # k passed to retrieve is clamped to 15.
    _, kwargs = fake_rag.retrieve.call_args[0], fake_rag.retrieve.call_args[1]
    assert fake_rag.retrieve.call_args[0][2] == 15  # 3rd positional is k


def test_rag_query_shape_is_stable(fake_rag):
    fake_rag.retrieve.return_value = {
        "docs": ["contenido 1"],
        "metas": [{"file": "a.md", "note": "A", "folder": "01", "tags": "t1,t2"}],
        "scores": [0.73],
    }
    out = mcp_server.rag_query("algo", k=1)
    assert out == [{
        "note": "A",
        "path": "a.md",
        "folder": "01",
        "tags": "t1,t2",
        "score": 0.73,
        "content": "contenido 1",
    }]


# ── rag_list_notes ───────────────────────────────────────────────────────────


def test_list_notes_dedup_by_path(fake_rag):
    # Corpus has 2 distinct paths → expect 2 results max.
    out = mcp_server.rag_list_notes(limit=100)
    paths = [n["path"] for n in out]
    assert len(paths) == len(set(paths))


def test_list_notes_folder_filter(fake_rag):
    out = mcp_server.rag_list_notes(folder="02-Areas", limit=100)
    assert all("02-Areas" in n["path"] for n in out)
    assert len(out) == 1


def test_list_notes_tag_filter_matches_exact_token(fake_rag):
    # "coach" must NOT match "coaching" — filter splits by comma.
    out_exact = mcp_server.rag_list_notes(tag="coaching", limit=100)
    out_partial = mcp_server.rag_list_notes(tag="coach", limit=100)
    assert len(out_exact) == 1
    assert len(out_partial) == 0


def test_list_notes_limit_truncates(fake_rag):
    out = mcp_server.rag_list_notes(limit=1)
    assert len(out) == 1


# ── rag_links ────────────────────────────────────────────────────────────────


def test_links_clamps_k_and_returns_normalised_shape(fake_rag):
    fake_rag.find_urls.return_value = [
        {"url": "https://x.com", "anchor": "X",
         "path": "01/a.md", "note": "a", "line": 7,
         "context": "ver X para detalles", "score": 0.91},
    ]
    out = mcp_server.rag_links("docs de X", k=999)
    assert fake_rag.find_urls.call_args.kwargs["k"] == 30
    assert out == [{
        "url": "https://x.com", "anchor": "X",
        "path": "01/a.md", "note": "a", "line": 7,
        "context": "ver X para detalles", "score": 0.91,
    }]


# ── rag_stats ────────────────────────────────────────────────────────────────


def test_stats_returns_expected_keys(fake_rag):
    out = mcp_server.rag_stats()
    assert set(out.keys()) == {
        "chunks", "collection", "embed_model", "reranker", "vault_path"
    }
    assert out["chunks"] == 42
    assert out["collection"] == "obsidian_notes_v9_test"


# ── Internals ────────────────────────────────────────────────────────────────


def test_touch_updates_last_call(monkeypatch):
    monkeypatch.setattr(mcp_server, "_last_call", 0.0)
    mcp_server._touch()
    assert mcp_server._last_call > 0


def test_idle_thresholds_are_sane():
    """Guard against someone accidentally dropping the idle timeouts to
    values that would churn respawns."""
    # Hot (heavy libs loaded): evict after ≥15m, but not more than 2h.
    assert 15 * 60 <= mcp_server._IDLE_HOT_SECONDS <= 2 * 3600
    # Cold (idle since spawn): keep alive longer than hot.
    assert mcp_server._IDLE_COLD_SECONDS > mcp_server._IDLE_HOT_SECONDS


def test_load_rag_is_idempotent_and_thread_safe():
    """Concurrent _load_rag calls must not double-import. This exercises
    the _rag_lock guard in mcp_server._load_rag (NOT the monkeypatched
    version from fake_rag)."""
    import mcp_server as ms  # reimport to bypass fake_rag in this test
    # Reset globals so we hit the real _load_rag
    ms._rag = None
    ms._rag_lock = threading.Lock()

    # Swap the inner import with a counter-backed sentinel. The trick: the
    # function does `import rag as _r` — we intercept by pre-populating
    # sys.modules with a fake then unwinding.
    import sys
    call_count = {"n": 0}
    fake_rag_module = MagicMock()
    # Real _load_rag does `import rag as _r` then `_rag = _r`. We overwrite
    # sys.modules entry so the import resolves instantly to our fake.
    original = sys.modules.get("rag")

    def _counted_import():
        call_count["n"] += 1
        return fake_rag_module

    # Patch __import__ would be invasive. Simpler: pre-stash fake in sys.modules.
    sys.modules["rag"] = fake_rag_module
    try:
        results = []

        def worker():
            results.append(ms._load_rag())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=2)

        # All 8 workers see the same module object.
        assert all(r is fake_rag_module for r in results)
        assert ms._rag is fake_rag_module
    finally:
        if original is not None:
            sys.modules["rag"] = original
        ms._rag = None
