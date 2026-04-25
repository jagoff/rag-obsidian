"""Feature #16 del 2026-04-23 — `rag auto-tag` tests.

Validates:
- _auto_tag_note constrains output to tag_vocab in strict mode
- _auto_tag_note allows new tags with allow_new=True
- _scan_untagged_notes returns only untagged + skips cross-source
- CLI --dry-run doesn't write, --apply with --yes writes
- Graceful handling of LLM failures / empty vocab
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import rag


class _FakeResp:
    def __init__(self, content: str):
        m = MagicMock()
        m.content = content
        self.message = m


class _FakeClient:
    def __init__(self):
        self._responses: list = []

    def set_next(self, content: str | Exception):
        self._responses.append(content)

    def chat(self, **kwargs):
        if not self._responses:
            return _FakeResp('{"tags": []}')
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return _FakeResp(nxt)


@pytest.fixture
def fake_helper(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(rag, "_helper_client", lambda: client)
    return client


# ── _auto_tag_note ───────────────────────────────────────────────────────


def test_auto_tag_empty_body_returns_empty(fake_helper, tmp_path):
    p = tmp_path / "note.md"
    p.write_text("")
    result = rag._auto_tag_note(p, "", {"proyecto", "urgente"})
    assert result == []


def test_auto_tag_empty_vocab_returns_empty(fake_helper, tmp_path):
    p = tmp_path / "note.md"
    p.write_text("some content")
    result = rag._auto_tag_note(p, "content", set())
    assert result == []


def test_auto_tag_returns_vocab_tags_only(fake_helper, tmp_path):
    """Strict mode: drops tags outside vocab."""
    fake_helper.set_next(json.dumps({
        "tags": ["proyecto", "INVENTADO", "urgente", "otro-no"]
    }))
    p = tmp_path / "note.md"
    p.write_text("content")
    result = rag._auto_tag_note(
        p, "body here", {"proyecto", "urgente"}, allow_new=False,
    )
    assert "proyecto" in result
    assert "urgente" in result
    assert "INVENTADO" not in [t.lower() for t in result]
    assert "otro-no" not in result


def test_auto_tag_allow_new_accepts_anything(fake_helper, tmp_path):
    fake_helper.set_next(json.dumps({
        "tags": ["proyecto", "tag-nuevo"]
    }))
    p = tmp_path / "note.md"
    p.write_text("content")
    result = rag._auto_tag_note(
        p, "body", {"proyecto"}, allow_new=True,
    )
    assert "proyecto" in result
    assert "tag-nuevo" in result


def test_auto_tag_ollama_exception_returns_empty(fake_helper, tmp_path):
    fake_helper.set_next(RuntimeError("ollama down"))
    p = tmp_path / "note.md"
    p.write_text("content")
    result = rag._auto_tag_note(p, "body", {"proyecto"})
    assert result == []


def test_auto_tag_invalid_json_returns_empty(fake_helper, tmp_path):
    fake_helper.set_next("not JSON {{{")
    p = tmp_path / "note.md"
    p.write_text("content")
    result = rag._auto_tag_note(p, "body", {"proyecto"})
    assert result == []


def test_auto_tag_deduplicates(fake_helper, tmp_path):
    fake_helper.set_next(json.dumps({
        "tags": ["proyecto", "proyecto", "PROYECTO"]
    }))
    p = tmp_path / "note.md"
    p.write_text("content")
    result = rag._auto_tag_note(p, "body", {"proyecto"})
    # Deduplicated after lowercasing.
    assert result == ["proyecto"]


def test_auto_tag_respects_max_tags(fake_helper, tmp_path):
    fake_helper.set_next(json.dumps({
        "tags": ["a", "b", "c", "d", "e", "f", "g"]
    }))
    p = tmp_path / "note.md"
    p.write_text("content")
    result = rag._auto_tag_note(
        p, "body", {"a", "b", "c", "d", "e", "f", "g"},
        max_tags=3,
    )
    assert len(result) == 3


def test_auto_tag_strips_hash_prefix(fake_helper, tmp_path):
    fake_helper.set_next(json.dumps({
        "tags": ["#proyecto", "#urgente"]
    }))
    p = tmp_path / "note.md"
    p.write_text("content")
    result = rag._auto_tag_note(
        p, "body", {"proyecto", "urgente"}, allow_new=False,
    )
    assert "proyecto" in result
    assert "urgente" in result


# ── _scan_untagged_notes ─────────────────────────────────────────────────


def _fake_corpus(metas: list[dict], docs: list[str]) -> dict:
    return {
        "docs": docs, "metas": metas,
        "count": len(docs), "collection_id": "t", "tags": set(),
    }


def test_scan_untagged_finds_untagged_only(monkeypatch):
    metas = [
        {"file": "a.md", "tags": ""},       # untagged
        {"file": "b.md", "tags": "proyecto"},  # tagged
    ]
    docs = ["a body", "b body"]
    monkeypatch.setattr(rag, "_load_corpus", lambda col: _fake_corpus(metas, docs))
    out = rag._scan_untagged_notes(None)
    paths = [p for p, _ in out]
    assert "a.md" in paths
    assert "b.md" not in paths


def test_scan_untagged_skips_cross_source(monkeypatch):
    metas = [
        {"file": "real-note.md", "tags": ""},
        {"file": "whatsapp://jid/msg1", "tags": ""},
        {"file": "gmail://msg2", "tags": ""},
    ]
    docs = ["x", "y", "z"]
    monkeypatch.setattr(rag, "_load_corpus", lambda col: _fake_corpus(metas, docs))
    out = rag._scan_untagged_notes(None)
    paths = [p for p, _ in out]
    assert paths == ["real-note.md"]


def test_scan_untagged_respects_limit(monkeypatch):
    metas = [{"file": f"n{i}.md", "tags": ""} for i in range(10)]
    docs = [f"body {i}" for i in range(10)]
    monkeypatch.setattr(rag, "_load_corpus", lambda col: _fake_corpus(metas, docs))
    out = rag._scan_untagged_notes(None, limit=3)
    assert len(out) == 3


def test_scan_untagged_deduplicates_chunks(monkeypatch):
    """A note with 3 chunks counts as 1."""
    metas = [{"file": "m.md", "tags": ""}] * 3
    docs = ["c1", "c2", "c3"]
    monkeypatch.setattr(rag, "_load_corpus", lambda col: _fake_corpus(metas, docs))
    out = rag._scan_untagged_notes(None)
    assert len(out) == 1
    # Uses the FIRST chunk's body.
    assert out[0][1] == "c1"


# ── CLI (integration) ────────────────────────────────────────────────────


def test_cli_empty_vocab_warns(fake_helper, tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "_load_corpus", lambda col: _fake_corpus([], []))
    monkeypatch.setattr(rag, "get_db", lambda: None)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["auto-tag"])
    assert result.exit_code == 0
    assert "vocabulary vacío" in result.output.lower()


def test_cli_no_untagged_notes(fake_helper, tmp_path, monkeypatch):
    """When everything is tagged, say so."""
    metas = [{"file": "a.md", "tags": "proyecto"}]
    corpus_with_vocab = {
        "docs": ["a"], "metas": metas, "count": 1,
        "collection_id": "t", "tags": {"proyecto"},
    }
    monkeypatch.setattr(rag, "_load_corpus", lambda col: corpus_with_vocab)
    monkeypatch.setattr(rag, "get_db", lambda: None)
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["auto-tag"])
    assert result.exit_code == 0
    assert "todo limpio" in result.output.lower()


def test_cli_dry_run_does_not_write(
    fake_helper, tmp_path, monkeypatch,
):
    metas = [{"file": "x.md", "tags": ""}]
    corpus = {"docs": ["body"], "metas": metas, "count": 1,
              "collection_id": "t", "tags": {"proyecto"}}
    monkeypatch.setattr(rag, "_load_corpus", lambda col: corpus)
    monkeypatch.setattr(rag, "get_db", lambda: None)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)

    # Create the note.
    note = tmp_path / "x.md"
    note.write_text("body content")
    original = note.read_text()

    fake_helper.set_next(json.dumps({"tags": ["proyecto"]}))

    runner = CliRunner()
    # No --apply → dry run
    result = runner.invoke(rag.cli, ["auto-tag", "--limit", "1"])
    assert result.exit_code == 0
    # File unchanged (no frontmatter written).
    assert note.read_text() == original


def test_cli_apply_with_yes_writes_frontmatter(
    fake_helper, tmp_path, monkeypatch,
):
    metas = [{"file": "y.md", "tags": ""}]
    corpus = {"docs": ["body"], "metas": metas, "count": 1,
              "collection_id": "t", "tags": {"proyecto", "urgente"}}
    monkeypatch.setattr(rag, "_load_corpus", lambda col: corpus)
    monkeypatch.setattr(rag, "get_db", lambda: None)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)

    note = tmp_path / "y.md"
    note.write_text("body")

    fake_helper.set_next(json.dumps({"tags": ["proyecto"]}))

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["auto-tag", "--limit", "1", "--apply", "--yes"],
    )
    assert result.exit_code == 0, result.output
    content = note.read_text()
    assert "tags:" in content
    assert "- proyecto" in content


def test_cli_as_json_outputs_shape(
    fake_helper, tmp_path, monkeypatch,
):
    metas = [{"file": "z.md", "tags": ""}]
    corpus = {"docs": ["body"], "metas": metas, "count": 1,
              "collection_id": "t", "tags": {"proyecto"}}
    monkeypatch.setattr(rag, "_load_corpus", lambda col: corpus)
    monkeypatch.setattr(rag, "get_db", lambda: None)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    (tmp_path / "z.md").write_text("body")

    fake_helper.set_next(json.dumps({"tags": ["proyecto"]}))

    runner = CliRunner()
    result = runner.invoke(
        rag.cli, ["auto-tag", "--limit", "1", "--as-json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.output.strip().splitlines()[-1])
    assert data["total"] == 1
    assert len(data["suggested"]) == 1
    assert data["suggested"][0]["path"] == "z.md"
    assert data["suggested"][0]["suggested"] == ["proyecto"]
