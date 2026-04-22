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


# ── Write tools (2026-04-22) ─────────────────────────────────────────────────
# New tools that have filesystem / Apple-DB side effects. Validation hooks
# tested here with a fake rag module — full E2E (real capture_note writing
# to a real vault) is covered in test_capture_cmd.py / test_reminders_api.py.


def test_rag_capture_empty_text_returns_error(fake_rag):
    out = mcp_server.rag_capture("")
    assert out["created"] is False
    assert "empty" in out["error"].lower()


def test_rag_capture_whitespace_only_text_returns_error(fake_rag):
    out = mcp_server.rag_capture("   \n\t  ")
    assert out["created"] is False


def test_rag_capture_delegates_to_rag_capture_note(fake_rag, tmp_path):
    """Happy path: rag_capture calls rag.capture_note, auto-indexes, and
    returns the vault-relative path."""
    vault = fake_rag.VAULT_PATH
    written = vault / "00-Inbox" / "2026-04-22-1234-test.md"
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_text("dummy", encoding="utf-8")
    fake_rag.capture_note.return_value = written

    out = mcp_server.rag_capture("una idea nueva",
                                 tags=["mcp"], source="test")
    assert out["created"] is True
    assert out["path"] == "00-Inbox/2026-04-22-1234-test.md"
    # capture_note received the text + tags + source.
    fake_rag.capture_note.assert_called_once()
    call_kwargs = fake_rag.capture_note.call_args.kwargs
    assert call_kwargs["tags"] == ["mcp"]
    assert call_kwargs["source"] == "test"
    # Auto-index fires after write.
    fake_rag._index_single_file.assert_called_once()


def test_rag_capture_propagates_value_error(fake_rag):
    """ValueError from capture_note surfaces as {created:false, error:...}"""
    fake_rag.capture_note.side_effect = ValueError("disk full or whatever")
    out = mcp_server.rag_capture("text")
    assert out["created"] is False
    assert "disk full" in out["error"]


def test_rag_save_note_empty_text_or_title_returns_error(fake_rag):
    assert mcp_server.rag_save_note("", "title")["created"] is False
    assert mcp_server.rag_save_note("body", "")["created"] is False
    assert mcp_server.rag_save_note("body", "   ")["created"] is False


def test_rag_save_note_rejects_absolute_folder(fake_rag):
    out = mcp_server.rag_save_note(
        "body", "title", folder="/etc/passwd",
    )
    assert out["created"] is False
    assert "invalid folder" in out["error"] or "vault root" in out["error"]


def test_rag_save_note_rejects_traversal(fake_rag):
    out = mcp_server.rag_save_note(
        "body", "title", folder="../../escape",
    )
    assert out["created"] is False


def test_rag_save_note_writes_and_indexes(fake_rag, tmp_path, monkeypatch):
    """Happy path: writes a well-formed note + auto-indexes."""
    # Use a real _slug (the fake_rag fixture mocks everything, including _slug).
    import rag as real_rag
    fake_rag._slug = real_rag._slug

    out = mcp_server.rag_save_note(
        "body del contenido", "Mi Nota", folder="02-Areas/Test",
        tags=["experiment", "mcp"],
    )
    assert out["created"] is True
    assert out["path"].startswith("02-Areas/Test/")
    assert out["path"].endswith(".md")
    written = fake_rag.VAULT_PATH / out["path"]
    assert written.is_file()
    body = written.read_text(encoding="utf-8")
    assert "# Mi Nota" in body
    assert "body del contenido" in body
    assert "type: note" in body
    # Tags are in frontmatter.
    assert "  - experiment" in body
    assert "  - mcp" in body
    # Auto-index fires.
    fake_rag._index_single_file.assert_called_once()


def test_rag_save_note_collision_suffix(fake_rag, tmp_path):
    """Second save with same title gets -2 suffix."""
    import rag as real_rag
    fake_rag._slug = real_rag._slug

    out1 = mcp_server.rag_save_note("a", "Same Title", folder="00-Inbox")
    out2 = mcp_server.rag_save_note("b", "Same Title", folder="00-Inbox")
    assert out1["created"] is True
    assert out2["created"] is True
    assert out1["path"] != out2["path"]
    assert out2["path"].endswith("-2.md")


def test_rag_create_reminder_forwards_args_and_parses_json(fake_rag):
    """rag_create_reminder is a thin wrapper: it passes kwargs to
    propose_reminder and parses the JSON response to a dict."""
    fake_rag.propose_reminder.return_value = (
        '{"kind":"reminder","created":true,"reminder_id":"R-123",'
        '"fields":{"title":"comprar café","due_iso":"2026-04-23T10:00:00"}}'
    )
    out = mcp_server.rag_create_reminder(
        title="comprar café", when="mañana 10am",
        priority=5, notes="sin filtro",
    )
    assert out["created"] is True
    assert out["reminder_id"] == "R-123"
    assert out["kind"] == "reminder"

    fake_rag.propose_reminder.assert_called_once()
    call_kwargs = fake_rag.propose_reminder.call_args.kwargs
    assert call_kwargs["title"] == "comprar café"
    assert call_kwargs["when"] == "mañana 10am"
    # The MCP arg is `reminder_list` but propose_reminder takes `list`.
    assert "list" in call_kwargs
    assert call_kwargs["priority"] == 5
    assert call_kwargs["notes"] == "sin filtro"


def test_rag_create_reminder_handles_ambiguous_date(fake_rag):
    """needs_clarification passthrough."""
    fake_rag.propose_reminder.return_value = (
        '{"kind":"reminder","needs_clarification":true,'
        '"proposal_id":"prop-abc","fields":{"title":"X","due_iso":null}}'
    )
    out = mcp_server.rag_create_reminder(title="X", when="algún día")
    assert out.get("needs_clarification") is True
    assert out["proposal_id"] == "prop-abc"
    # created is NOT present in ambiguous case.
    assert "created" not in out or out["created"] is False


def test_rag_create_reminder_malformed_json_still_returns_dict(fake_rag):
    """If propose_reminder ever returns non-JSON, we don't crash."""
    fake_rag.propose_reminder.return_value = "not json at all {"
    out = mcp_server.rag_create_reminder(title="X")
    assert out["created"] is False
    assert "json" in out["error"].lower()
    assert out["raw"] == "not json at all {"


def test_rag_create_event_forwards_args(fake_rag):
    fake_rag.propose_calendar_event.return_value = (
        '{"kind":"event","created":true,"event_uid":"E-777",'
        '"fields":{"title":"reunión","start_iso":"2026-04-24T10:00:00"}}'
    )
    out = mcp_server.rag_create_event(
        title="reunión", start="jueves 10am",
        location="Oficina", all_day=False,
    )
    assert out["created"] is True
    assert out["event_uid"] == "E-777"
    fake_rag.propose_calendar_event.assert_called_once()
    call_kwargs = fake_rag.propose_calendar_event.call_args.kwargs
    assert call_kwargs["title"] == "reunión"
    assert call_kwargs["start"] == "jueves 10am"
    assert call_kwargs["location"] == "Oficina"
    assert call_kwargs["all_day"] is False


def test_rag_followup_delegates_and_filters(fake_rag):
    """rag_followup calls find_followup_loops + optionally filters by status
    + honors limit."""
    fake_rag.find_followup_loops.return_value = [
        {"status": "stale", "age_days": 40, "kind": "todo",
         "source_note": "A.md", "loop_text": "llamar a Juan"},
        {"status": "activo", "age_days": 10, "kind": "checkbox",
         "source_note": "B.md", "loop_text": "[ ] revisar contrato"},
        {"status": "resolved", "age_days": 5, "kind": "inline",
         "source_note": "C.md", "loop_text": "tenía que cerrar X"},
    ]
    # No filter, default limit → all 3.
    all_loops = mcp_server.rag_followup(days=30)
    assert len(all_loops) == 3

    stale_only = mcp_server.rag_followup(days=30, status="stale")
    assert len(stale_only) == 1
    assert stale_only[0]["status"] == "stale"

    # Limit respected.
    limited = mcp_server.rag_followup(days=30, limit=2)
    assert len(limited) == 2


def test_rag_followup_empty_list_is_ok(fake_rag):
    fake_rag.find_followup_loops.return_value = []
    assert mcp_server.rag_followup(days=30) == []
