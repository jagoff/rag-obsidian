"""Tests for Phase 2 episodic-memory consolidation.

Covers the pure-Python pieces end-to-end without touching ollama:
  - scan_conversations parses frontmatter + extracts first-turn Q/A
  - Clustering (union-find) respects threshold + min_cluster
  - classify_target_folder distinguishes project vs resource
  - promote() + archive_originals() + render frontmatter + wikilinks
  - _unique_path collision handling
  - is_excluded gates 04-Archive/99-obsidian-system/99-AI/_archive/conversations/
    (path nuevo) y la ruta legacy 04-Archive/conversations/ (defense-in-depth)
  - run() end-to-end with monkeypatched embed + synthesise
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag
from scripts import consolidate_conversations as cc
from web import conversation_writer


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_vault(tmp_path: Path, monkeypatch):
    vault = tmp_path / "vault"
    # `04-Archive/...` se crea con parents=True abajo, no hace falta el
    # segundo mkdir() para `04-Archive` (FileExistsError tras la rename
    # del 2026-04-25 que movió las conversations bajo 99-obsidian-system/).
    (vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "conversations").mkdir(parents=True)
    (vault / "01-Projects").mkdir()
    (vault / "03-Resources").mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    return vault


def _seed_conversation(
    vault: Path, *, session_id: str, question: str, answer: str,
    sources: list[str] | None = None,
    created: datetime | None = None,
    turns: int = 1,
) -> Path:
    """Write a conversation note matching the Phase 1 writer schema."""
    created_dt = created or datetime.now()
    sources = sources or []
    folder = vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "conversations"
    folder.mkdir(parents=True, exist_ok=True)
    slug = conversation_writer.slugify(question)
    path = folder / f"{created_dt.strftime('%Y-%m-%d-%H%M')}-{slug}.md"
    iso = conversation_writer._iso_z(created_dt)
    meta = {
        "session_id": session_id,
        "created": iso,
        "updated": iso,
        "turns": turns,
        "confidence_avg": "0.500",
        "sources": sources,
        "tags": list(conversation_writer._TAGS),
    }
    body = (
        f"## Turn 1 — {created_dt.strftime('%H:%M')}\n\n"
        f"> {question}\n\n"
        f"{answer}\n\n"
        f"**Sources**: "
        + (
            " · ".join(
                f"[[{s[:-3] if s.endswith('.md') else s}]]" for s in sources
            )
            if sources else "—"
        )
        + "\n"
    )
    path.write_text(
        conversation_writer._render_frontmatter(meta) + "\n" + body,
        encoding="utf-8",
    )
    return path


# ── Scan ────────────────────────────────────────────────────────────────────

def test_scan_reads_first_turn_and_metadata(tmp_vault: Path):
    _seed_conversation(
        tmp_vault, session_id="web:1",
        question="¿qué es el Ikigai?",
        answer="El Ikigai es un concepto japonés.",
        sources=["02-Areas/Coaching.md"],
    )
    items = cc.scan_conversations(
        tmp_vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "conversations", window_days=14,
    )
    assert len(items) == 1
    it = items[0]
    assert it.first_question == "¿qué es el Ikigai?"
    assert it.first_answer == "El Ikigai es un concepto japonés."
    assert it.turns == 1
    assert "02-Areas/Coaching.md" in it.sources
    assert it.rel_path.startswith("04-Archive/99-obsidian-system/99-AI/conversations/")


def test_scan_respects_window_days(tmp_vault: Path, monkeypatch):
    old = _seed_conversation(
        tmp_vault, session_id="web:old",
        question="pregunta vieja",
        answer="respuesta vieja",
    )
    # Force mtime to 30 days ago — outside the 14-day window.
    old_ts = (datetime.now() - timedelta(days=30)).timestamp()
    import os
    os.utime(old, (old_ts, old_ts))
    _seed_conversation(
        tmp_vault, session_id="web:new",
        question="pregunta reciente", answer="respuesta reciente",
    )
    items = cc.scan_conversations(
        tmp_vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "conversations", window_days=14,
    )
    assert len(items) == 1
    assert items[0].first_question == "pregunta reciente"


def test_scan_skips_malformed(tmp_vault: Path):
    # Missing frontmatter closing
    bad = tmp_vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "conversations" / "broken.md"
    bad.write_text("---\nno-closing\n\n## Turn 1\n> q\n\na\n", encoding="utf-8")
    items = cc.scan_conversations(
        tmp_vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "conversations", window_days=14,
    )
    assert items == []


# ── Clustering ─────────────────────────────────────────────────────────────

def test_cluster_by_embedding_respects_threshold():
    # Two tight clusters (cos=1.0) + one isolate.
    u = cc._l2_normalize([1.0, 0.0, 0.0])
    v = cc._l2_normalize([0.0, 1.0, 0.0])
    w = cc._l2_normalize([0.0, 0.0, 1.0])
    embeds = [u, u, u, v, v, v, w]  # 3 of u, 3 of v, 1 of w
    clusters = cc.cluster_by_embedding(embeds, threshold=0.75, min_cluster=3)
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [3, 3]


def test_cluster_filters_below_min_cluster():
    u = cc._l2_normalize([1.0, 0.0])
    v = cc._l2_normalize([0.0, 1.0])
    embeds = [u, u, v, v]  # two pairs, neither reaches min=3
    clusters = cc.cluster_by_embedding(embeds, threshold=0.75, min_cluster=3)
    assert clusters == []


def test_cluster_empty_returns_empty():
    assert cc.cluster_by_embedding([], threshold=0.75, min_cluster=3) == []


# ── Classification ────────────────────────────────────────────────────────

def test_classify_project_when_action_signals_present():
    items = [
        cc.ConversationNote(
            path=Path("/x"), rel_path="x", created=datetime.now(),
            updated=datetime.now(), turns=1,
            first_question="q", first_answer="a",
            body="tengo que mandar el informe mañana",
            sources=[],
        ),
        cc.ConversationNote(
            path=Path("/y"), rel_path="y", created=datetime.now(),
            updated=datetime.now(), turns=1,
            first_question="q", first_answer="a",
            body="agendá reunión para el próximo lunes",
            sources=[],
        ),
    ]
    assert cc.classify_target_folder(items) == "01-Projects"


def test_classify_resource_when_no_action_signals():
    items = [
        cc.ConversationNote(
            path=Path("/x"), rel_path="x", created=datetime.now(),
            updated=datetime.now(), turns=1,
            first_question="q", first_answer="a",
            body="explicación sobre filosofía japonesa del Ikigai.",
            sources=[],
        ),
        cc.ConversationNote(
            path=Path("/y"), rel_path="y", created=datetime.now(),
            updated=datetime.now(), turns=1,
            first_question="q", first_answer="a",
            body="nota sobre la definición del concepto.",
            sources=[],
        ),
    ]
    assert cc.classify_target_folder(items) == "03-Resources"


def test_classify_requires_two_hits_for_project():
    # Single hit → still resource (conservative).
    items = [
        cc.ConversationNote(
            path=Path("/x"), rel_path="x", created=datetime.now(),
            updated=datetime.now(), turns=1,
            first_question="q", first_answer="a",
            body="agendá esta llamada",  # single hit
            sources=[],
        ),
        cc.ConversationNote(
            path=Path("/y"), rel_path="y", created=datetime.now(),
            updated=datetime.now(), turns=1,
            first_question="q", first_answer="a",
            body="contexto neutral sin acción",
            sources=[],
        ),
    ]
    assert cc.classify_target_folder(items) == "03-Resources"


# ── Promote + archive ───────────────────────────────────────────────────────

def test_promote_writes_frontmatter_and_origin_wikilinks(tmp_vault: Path):
    p1 = _seed_conversation(
        tmp_vault, session_id="web:1",
        question="q1", answer="a1",
        sources=["02-Areas/Foo.md"],
    )
    p2 = _seed_conversation(
        tmp_vault, session_id="web:2",
        question="q2", answer="a2",
        sources=["02-Areas/Bar.md"],
    )
    items = cc.scan_conversations(
        tmp_vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "conversations", window_days=14,
    )
    assert len(items) == 2
    written = cc.promote(
        tmp_vault, "03-Resources", items,
        title="Ikigai consolidado", body="Body de prueba con [[Coaching]].",
    )
    assert written.exists()
    text = written.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "type: consolidated-conversation" in text
    assert "source_conversations: 2" in text
    assert "Body de prueba con [[Coaching]]." in text
    # Archive wikilinks present
    assert "## Conversaciones originales" in text
    assert "[[04-Archive/99-obsidian-system/99-AI/_archive/conversations/" in text
    # Sources referenciadas
    assert "[[02-Areas/Foo]]" in text
    assert "[[02-Areas/Bar]]" in text


def test_archive_moves_originals_into_monthly_folder(tmp_vault: Path):
    p1 = _seed_conversation(tmp_vault, session_id="a", question="q1", answer="a1")
    p2 = _seed_conversation(tmp_vault, session_id="b", question="q2", answer="a2")
    items = cc.scan_conversations(
        tmp_vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "conversations", window_days=14,
    )
    moved = cc.archive_originals(tmp_vault, items)
    assert len(moved) == 2
    for m in moved:
        assert m.exists()
        # Under 04-Archive/99-obsidian-system/99-AI/_archive/conversations/YYYY-MM/
        assert "04-Archive/99-obsidian-system/99-AI/_archive/conversations" in str(m)
    # Originals gone
    assert not p1.exists()
    assert not p2.exists()


def test_unique_path_adds_suffix_on_collision(tmp_vault: Path):
    p = tmp_vault / "03-Resources" / "test.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("a", encoding="utf-8")
    alt = cc._unique_path(p)
    assert alt != p
    assert "(2)" in alt.name


def test_unique_path_returns_candidate_if_free(tmp_vault: Path):
    p = tmp_vault / "03-Resources" / "nonexistent.md"
    assert cc._unique_path(p) == p


# ── Index exclusion ───────────────────────────────────────────────────────

def test_is_excluded_covers_archived_conversations():
    # Nuevo path post-2026-04-30: archive vive bajo `99-obsidian-system/99-AI/_archive/`
    # — queda excluido por el prefix general `04-Archive/99-obsidian-system/`
    # (no por la rama defensiva legacy).
    assert rag.is_excluded(
        "04-Archive/99-obsidian-system/99-AI/_archive/conversations/2026-04/foo.md"
    ) is True
    assert rag.is_excluded(
        "04-Archive/99-obsidian-system/99-AI/_archive/conversations/2025-12/bar.md"
    ) is True
    # Path legacy pre-2026-04-30: rama defensiva específica para archivos
    # que el user no haya migrado todavía.
    assert rag.is_excluded("04-Archive/conversations/2026-04/foo.md") is True
    assert rag.is_excluded("04-Archive/conversations/2025-12/bar.md") is True
    # Sibling archive folders stay indexed
    assert rag.is_excluded("04-Archive/WhatsApp/chat.md") is False
    assert rag.is_excluded("04-Archive/OtherProject/note.md") is False
    # Inbox conversations still excluded too (regression guard)
    assert rag.is_excluded("04-Archive/99-obsidian-system/99-AI/conversations/foo.md") is True


def test_is_excluded_indexes_mem_vault_memories():
    """Las memorias persistentes del MCP `mem-vault` viven bajo
    `04-Archive/99-obsidian-system/99-AI/memory/` y son la única excepción
    (junto con `99-Mentions/`) al exclude del prefix de system folders.
    Las queremos indexadas para que `rag query` recupere bug patterns,
    decisiones y convenciones que el user acumuló entre sesiones.
    """
    # ✅ memory/ excepcionado
    assert rag.is_excluded("04-Archive/99-obsidian-system/99-AI/memory/foo.md") is False
    assert rag.is_excluded(
        "04-Archive/99-obsidian-system/99-AI/memory/bug_pattern_xyz.md"
    ) is False
    # ✅ 99-Mentions sigue indexado (regression guard, este caso es histórico)
    assert rag.is_excluded("04-Archive/99-obsidian-system/99-Mentions/Maria.md") is False
    # ❌ Otras subcarpetas de 99-AI/ siguen excluidas
    assert rag.is_excluded("04-Archive/99-obsidian-system/99-AI/reviews/2026-04-29.md") is True
    assert rag.is_excluded("04-Archive/99-obsidian-system/99-AI/conversations/x.md") is True
    assert rag.is_excluded("04-Archive/99-obsidian-system/99-AI/system/foo/plan.md") is True
    # ❌ Otras carpetas de system siguen excluidas
    assert rag.is_excluded("04-Archive/99-obsidian-system/99-Templates/x.md") is True
    assert rag.is_excluded("04-Archive/99-obsidian-system/99-Attachments/img.png") is True


# ── End-to-end run() ─────────────────────────────────────────────────────

def test_run_dry_run_does_not_write(tmp_vault: Path, monkeypatch):
    # Three Ikigai convs (cluster) + 1 n8n (singleton, below min).
    for i in range(3):
        _seed_conversation(
            tmp_vault, session_id=f"web:ikigai-{i}",
            question=f"qué es el Ikigai {i}",
            answer="El Ikigai es un concepto japonés.",
        )
    _seed_conversation(
        tmp_vault, session_id="web:n8n",
        question="cómo armar un workflow n8n",
        answer="n8n workflows se definen con nodos.",
    )

    def _fake_embed(texts):
        # 3 Ikigai docs → same vector; n8n → orthogonal.
        u = [1.0, 0.0, 0.0]
        v = [0.0, 1.0, 0.0]
        out = []
        for t in texts:
            out.append(u if "Ikigai" in t or "ikigai" in t else v)
        return out

    monkeypatch.setattr(rag, "embed", _fake_embed)
    summary = cc.run(
        vault_root=tmp_vault, window_days=14, dry_run=True,
    )
    assert summary["n_conversations"] == 4
    assert summary["n_clusters"] == 1
    assert summary["n_promoted"] == 0
    assert summary["n_archived"] == 0
    # No consolidated note written
    assert not (tmp_vault / "03-Resources").glob("*.md").__iter__().__next__() \
        if list((tmp_vault / "03-Resources").glob("*.md")) else True
    assert list((tmp_vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "conversations").glob("*.md"))


def test_run_real_promotes_and_archives(tmp_vault: Path, monkeypatch):
    for i in range(3):
        _seed_conversation(
            tmp_vault, session_id=f"web:ikigai-{i}",
            question=f"qué es el Ikigai {i}",
            answer="El Ikigai japonés.",
        )
    _seed_conversation(
        tmp_vault, session_id="web:solo",
        question="algo aislado",
        answer="respuesta aislada",
    )

    def _fake_embed(texts):
        u = [1.0, 0.0, 0.0]
        v = [0.0, 1.0, 0.0]
        return [u if "Ikigai" in t or "ikigai" in t else v for t in texts]

    def _fake_synth(cluster, model=None):
        return "Ikigai consolidated", "Nota consolidada sobre Ikigai."

    monkeypatch.setattr(rag, "embed", _fake_embed)
    monkeypatch.setattr(cc, "synthesize_cluster", _fake_synth)
    # Point the log at tmp so we don't pollute the real disk.
    monkeypatch.setattr(cc, "CONSOLIDATION_LOG", tmp_vault / "consolidation.log")

    summary = cc.run(
        vault_root=tmp_vault, window_days=14, dry_run=False,
    )
    assert summary["n_clusters"] == 1
    assert summary["n_promoted"] == 1
    assert summary["n_archived"] == 3
    # Consolidated note lives in 03-Resources (no action signals → resources).
    promoted = list((tmp_vault / "03-Resources").glob("Ikigai*.md"))
    assert len(promoted) == 1
    # Archive got the three originals
    archived = list((tmp_vault / "04-Archive" / "conversations").rglob("*.md"))
    assert len(archived) == 3
    # Singleton stayed in inbox
    remaining = list((tmp_vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "conversations").glob("*.md"))
    assert len(remaining) == 1
    # Log record written
    log = (tmp_vault / "consolidation.log").read_text(encoding="utf-8")
    assert log.strip()
    parsed = json.loads(log.strip())
    assert parsed["n_promoted"] == 1


def test_run_skips_cluster_on_synth_failure(tmp_vault: Path, monkeypatch):
    for i in range(3):
        _seed_conversation(
            tmp_vault, session_id=f"web:x-{i}",
            question=f"tema repetido {i}",
            answer="misma respuesta.",
        )

    def _fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    def _boom(cluster, model=None):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(rag, "embed", _fake_embed)
    monkeypatch.setattr(cc, "synthesize_cluster", _boom)
    monkeypatch.setattr(cc, "CONSOLIDATION_LOG", tmp_vault / "consolidation.log")

    summary = cc.run(vault_root=tmp_vault, window_days=14, dry_run=False)
    assert summary["n_clusters"] == 1
    assert summary["n_promoted"] == 0
    assert summary["n_archived"] == 0
    # Originals untouched
    remaining = list((tmp_vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "conversations").glob("*.md"))
    assert len(remaining) == 3
    # Cluster entry recorded the error
    assert any("error" in c for c in summary["clusters"])


def test_run_empty_inbox_is_noop(tmp_vault: Path, monkeypatch):
    monkeypatch.setattr(cc, "CONSOLIDATION_LOG", tmp_vault / "consolidation.log")
    summary = cc.run(vault_root=tmp_vault, window_days=14, dry_run=False)
    assert summary["n_conversations"] == 0
    assert summary["n_clusters"] == 0
    assert summary["n_promoted"] == 0


def test_run_bails_when_promoted_note_missing(tmp_vault: Path, monkeypatch):
    """Pre-check safety: if `promote()` returns but the file didn't land
    (or landed empty), originals must NOT be moved to 04-Archive/. They
    stay in 00-Inbox/ so the next run can retry. Without this guard we'd
    end up with orphan archives and no consolidated note to cite them."""
    for i in range(3):
        _seed_conversation(
            tmp_vault, session_id=f"web:x-{i}",
            question=f"tema repetido {i}",
            answer="misma respuesta.",
        )

    def _fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    # Fake promote: returns a path but doesn't write the file (simulates
    # fs full or permissions failure mid-write).
    def _fake_promote(vault_root, target_folder, cluster, title, body):
        return vault_root / target_folder / "ghost.md"

    monkeypatch.setattr(rag, "embed", _fake_embed)
    monkeypatch.setattr(cc, "synthesize_cluster",
                        lambda cluster, model=None: ("T", "B"))
    monkeypatch.setattr(cc, "promote", _fake_promote)
    monkeypatch.setattr(cc, "CONSOLIDATION_LOG", tmp_vault / "consolidation.log")

    summary = cc.run(vault_root=tmp_vault, window_days=14, dry_run=False)
    assert summary["n_clusters"] == 1
    # The pre-check fires — no promotion, no archive.
    assert summary["n_promoted"] == 0
    assert summary["n_archived"] == 0
    # Originals untouched in 00-Inbox (the key invariant).
    remaining = list((tmp_vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "conversations").glob("*.md"))
    assert len(remaining) == 3
    # Error is reported for observability.
    assert any(
        c.get("error") == "promoted_note_missing_or_empty"
        for c in summary["clusters"]
    )


def test_run_bails_when_promoted_note_is_empty(tmp_vault: Path, monkeypatch):
    """Same guard but for the case where the file exists but the write
    was truncated (tmp→replace gone wrong or zero-byte body)."""
    for i in range(3):
        _seed_conversation(
            tmp_vault, session_id=f"web:x-{i}",
            question=f"tema repetido {i}",
            answer="misma respuesta.",
        )

    def _fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    def _fake_promote_empty(vault_root, target_folder, cluster, title, body):
        target_dir = vault_root / target_folder
        target_dir.mkdir(parents=True, exist_ok=True)
        p = target_dir / "tiny.md"
        p.write_text("x\n", encoding="utf-8")  # 2 bytes, below 200 threshold
        return p

    monkeypatch.setattr(rag, "embed", _fake_embed)
    monkeypatch.setattr(cc, "synthesize_cluster",
                        lambda cluster, model=None: ("T", "B"))
    monkeypatch.setattr(cc, "promote", _fake_promote_empty)
    monkeypatch.setattr(cc, "CONSOLIDATION_LOG", tmp_vault / "consolidation.log")

    summary = cc.run(vault_root=tmp_vault, window_days=14, dry_run=False)
    assert summary["n_promoted"] == 0
    assert summary["n_archived"] == 0
    remaining = list((tmp_vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "conversations").glob("*.md"))
    assert len(remaining) == 3


# ── launchd wiring ────────────────────────────────────────────────────────

def test_consolidate_plist_renders_valid_schedule():
    p = rag._consolidate_plist("/usr/local/bin/rag")
    # Weekday=1 (Monday), Hour=6 — matches plan spec.
    assert "<key>Weekday</key><integer>1</integer>" in p
    assert "<key>Hour</key><integer>6</integer>" in p
    assert "<key>Minute</key><integer>0</integer>" in p
    assert "com.fer.obsidian-rag-consolidate" in p
    assert "<string>consolidate</string>" in p
    assert "<key>RunAtLoad</key><false/>" in p


def test_consolidate_service_registered_in_setup():
    spec = rag._services_spec("/usr/local/bin/rag")
    labels = [label for label, _, _ in spec]
    assert "com.fer.obsidian-rag-consolidate" in labels
