"""Tests para promote-on-cite (A): notas citadas en conversations recientes
no se archivan aunque el detector las marque como dead.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag
from rag import SqliteVecClient as _TestVecClient


@pytest.fixture
def tmp_vault_with_convs(tmp_path, monkeypatch, fake_embed):
    vault = tmp_path / "vault"
    for d in (
        "00-Inbox",
        "01-Projects",
        "02-Areas",
        "03-Resources",
        "04-Archive/99-obsidian-system/99-AI/conversations",
        "04-Archive/99-obsidian-system/99-AI/reviews",
    ):
        (vault / d).mkdir(parents=True)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    client = _TestVecClient(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="promote_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    monkeypatch.setattr(rag, "_index_single_file", lambda *a, **kw: "skipped")
    monkeypatch.setattr(rag, "FILING_BATCHES_DIR", tmp_path / "filing_batches")
    monkeypatch.setattr(rag, "ARCHIVE_LOG_PATH", tmp_path / "archive.jsonl")
    monkeypatch.setattr(rag, "_ambient_config", lambda: None)
    rag._invalidate_corpus_cache()
    return vault, col


def _write_conv(vault: Path, slug: str, sources: list[str], days_ago: int = 1):
    conv_dir = vault / "04-Archive/99-obsidian-system/99-AI/conversations"
    p = conv_dir / f"{slug}.md"
    created = (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")
    src_block = "\n".join(f"  - {s}" for s in sources)
    p.write_text(
        f"---\n"
        f"session_id: web:test{slug}\n"
        f"created: {created}\n"
        f"turns: 1\n"
        f"confidence_avg: 0.6\n"
        f"sources:\n{src_block}\n"
        f"tags:\n  - conversation\n"
        f"---\n\n## Turn 1\n\n> test query\n\nrespuesta del bot\n",
        encoding="utf-8",
    )
    return p


def test_load_citations_counts_recent_only(tmp_vault_with_convs):
    vault, _ = tmp_vault_with_convs
    _write_conv(vault, "recent", ["00-Inbox/foo.md"], days_ago=5)
    _write_conv(vault, "old", ["00-Inbox/foo.md"], days_ago=60)
    _write_conv(vault, "another", ["00-Inbox/foo.md", "00-Inbox/bar.md"], days_ago=1)
    counts = rag._load_conversation_citations(vault, window_days=30)
    assert counts.get("00-Inbox/foo.md") == 2  # recent + another
    assert counts.get("00-Inbox/bar.md") == 1
    assert "whatsapp://abc" not in counts  # excluded by prefix filter


def test_load_citations_skips_pseudo_uris(tmp_vault_with_convs):
    vault, _ = tmp_vault_with_convs
    _write_conv(
        vault, "wa",
        ["whatsapp://120363@g.us/abc", "https://example.com", "00-Inbox/real.md"],
    )
    counts = rag._load_conversation_citations(vault)
    assert "00-Inbox/real.md" in counts
    assert not any(k.startswith("whatsapp://") for k in counts)
    assert not any(k.startswith("http") for k in counts)


def test_archive_skips_cited_note(tmp_vault_with_convs, monkeypatch):
    vault, col = tmp_vault_with_convs
    note = vault / "00-Inbox/runbook.md"
    note.write_text("# Runbook\n\nstuff\n", encoding="utf-8")
    # 2 conversations citan esta nota → debería skipearse del archive.
    _write_conv(vault, "c1", ["00-Inbox/runbook.md"], days_ago=2)
    _write_conv(vault, "c2", ["00-Inbox/runbook.md"], days_ago=4)

    candidates = [{"path": "00-Inbox/runbook.md", "age_days": 400}]
    res = rag.archive_dead_notes(
        col, vault, candidates, apply=True, force=True,
    )
    assert res["plan"] == []
    assert any(
        s["path"] == "00-Inbox/runbook.md" and "cited-2x" in s["reason"]
        for s in res["skipped"]
    )
    # Nota sigue en su lugar original.
    assert note.is_file()


def test_archive_proceeds_below_threshold(tmp_vault_with_convs):
    vault, col = tmp_vault_with_convs
    note = vault / "00-Inbox/lonely.md"
    note.write_text("# Lonely\n", encoding="utf-8")
    # Sólo 1 cita → bajo el threshold de 2 → archive procede.
    _write_conv(vault, "single", ["00-Inbox/lonely.md"])

    candidates = [{"path": "00-Inbox/lonely.md", "age_days": 400}]
    res = rag.archive_dead_notes(
        col, vault, candidates, apply=True, force=True,
    )
    assert len(res["plan"]) == 1
    assert res["plan"][0]["src"] == "00-Inbox/lonely.md"
    assert (vault / "04-Archive/00-Inbox/lonely.md").is_file()


def test_archive_threshold_env_override(tmp_vault_with_convs, monkeypatch):
    vault, col = tmp_vault_with_convs
    note = vault / "00-Inbox/protected.md"
    note.write_text("# Protected\n", encoding="utf-8")
    _write_conv(vault, "once", ["00-Inbox/protected.md"])
    # Bajamos threshold a 1 — una sola cita debería frenar.
    # NB: el símbolo `rag.archive` se shadowea por el comando Click `archive`
    # vía el wildcard re-export. Llegamos al módulo via sys.modules.
    import sys
    arch_mod = sys.modules["rag.archive"]
    monkeypatch.setattr(arch_mod, "CITATION_PROMOTE_THRESHOLD", 1)

    candidates = [{"path": "00-Inbox/protected.md", "age_days": 400}]
    res = rag.archive_dead_notes(col, vault, candidates, apply=True, force=True)
    assert res["plan"] == []
    assert note.is_file()
