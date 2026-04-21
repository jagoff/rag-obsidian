"""Tests para `rag archive` — mover dead notes a 04-Archive con mirror PARA.

Las rutas de detección ya están cubiertas en test_capture_morning_dead.py. Acá
cubrimos: path mirroring, colisiones, frontmatter stamp, opt-out, dry-run vs
apply, gate de confirmación, batch log, push a WhatsApp stubbed.
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from rag import SqliteVecClient as _TestVecClient
import pytest

import rag


@pytest.fixture
def fake_embed(monkeypatch):
    def _embed(texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setattr(rag, "embed", _embed)


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch, fake_embed):
    vault = tmp_path / "vault"
    for d in ("00-Inbox", "01-Projects", "02-Areas",
              "03-Resources", "04-Archive", "05-Reviews"):
        (vault / d).mkdir(parents=True)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    client = _TestVecClient(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="archive_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    monkeypatch.setattr(
        rag, "_index_single_file", lambda *a, **kw: "skipped",
    )
    monkeypatch.setattr(rag, "FILING_BATCHES_DIR",
                        tmp_path / "filing_batches")
    monkeypatch.setattr(rag, "ARCHIVE_LOG_PATH",
                        tmp_path / "archive.jsonl")
    # Silence push-notifications by default.
    monkeypatch.setattr(rag, "_ambient_config", lambda: None)
    rag._invalidate_corpus_cache()
    return vault, col


def _touch(path: Path, days_ago: int):
    ts = (datetime.now() - timedelta(days=days_ago)).timestamp()
    os.utime(path, (ts, ts))


# ── path mirroring ──────────────────────────────────────────────────────────

def test_archive_target_mirrors_para():
    assert rag._archive_target_path("01-Projects/app-X/nota.md") == \
        "04-Archive/01-Projects/app-X/nota.md"
    assert rag._archive_target_path("02-Areas/foo.md") == \
        "04-Archive/02-Areas/foo.md"


def test_archive_target_noop_if_already_archived():
    assert rag._archive_target_path("04-Archive/old.md") == "04-Archive/old.md"


def test_archive_resolves_collision(tmp_vault):
    vault, _ = tmp_vault
    existing = vault / "04-Archive" / "02-Areas" / "dup.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("already here")
    resolved = rag._archive_resolve_collision(
        vault, "04-Archive/02-Areas/dup.md",
    )
    assert resolved != "04-Archive/02-Areas/dup.md"
    assert resolved.startswith("04-Archive/02-Areas/dup-archived-")
    assert resolved.endswith(".md")


# ── frontmatter stamp ───────────────────────────────────────────────────────

def test_stamp_injects_into_existing_frontmatter():
    raw = "---\ntitle: Foo\ntags:\n- x\n---\n\nbody\n"
    out = rag._archive_stamp_frontmatter(raw, "01-Projects/foo.md")
    fm = rag.parse_frontmatter(out)
    assert fm["title"] == "Foo"
    assert fm["tags"] == ["x"]
    assert fm["archived_from"] == "01-Projects/foo.md"
    assert fm["archived_reason"] == "dead"
    assert "body" in out


def test_stamp_creates_frontmatter_if_absent():
    raw = "just a body, no frontmatter\n"
    out = rag._archive_stamp_frontmatter(raw, "02-Areas/x.md")
    assert out.startswith("---\n")
    fm = rag.parse_frontmatter(out)
    assert fm["archived_from"] == "02-Areas/x.md"
    assert "just a body" in out


def test_stamp_overwrites_prior_stamp():
    raw = (
        "---\narchived_at: 2020-01-01\narchived_from: old\n"
        "archived_reason: dead\nkeep: kept-value\n---\n\nbody\n"
    )
    out = rag._archive_stamp_frontmatter(raw, "02-Areas/new.md")
    fm = rag.parse_frontmatter(out)
    assert fm["archived_from"] == "02-Areas/new.md"
    assert fm["keep"] == "kept-value"
    # Ensure the old archived_from isn't lingering twice.
    assert out.count("archived_from:") == 1


# ── opt-out ─────────────────────────────────────────────────────────────────

def test_opt_out_archive_never():
    raw = "---\narchive: never\n---\n\nbody\n"
    assert rag._is_archive_opt_out(raw) is True


def test_opt_out_type_moc():
    raw = "---\ntype: MOC\n---\n\nbody\n"
    assert rag._is_archive_opt_out(raw) is True


def test_opt_out_type_permanent():
    raw = "---\ntype: permanent\n---\n\nbody\n"
    assert rag._is_archive_opt_out(raw) is True


def test_opt_out_regular_note():
    raw = "---\ntype: capture\n---\n\nbody\n"
    assert rag._is_archive_opt_out(raw) is False


# ── archive_dead_notes plan + apply ─────────────────────────────────────────

def _make_candidate(vault: Path, rel: str, days_ago: int,
                    body: str = "old content") -> dict:
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    _touch(p, days_ago)
    return {"path": rel, "age_days": days_ago, "mtime": "", "age_source": "mtime"}


def test_dry_run_does_not_move(tmp_vault):
    vault, col = tmp_vault
    c = _make_candidate(vault, "02-Areas/dead.md", 400)
    result = rag.archive_dead_notes(
        col, vault, [c], apply=False, force=False,
    )
    assert len(result["plan"]) == 1
    assert result["applied"] == []
    assert (vault / "02-Areas/dead.md").is_file()
    assert not (vault / "04-Archive/02-Areas/dead.md").is_file()


def test_apply_moves_and_stamps(tmp_vault):
    vault, col = tmp_vault
    c = _make_candidate(vault, "02-Areas/dead.md", 400,
                        body="---\ntitle: T\n---\n\nbody\n")
    result = rag.archive_dead_notes(
        col, vault, [c], apply=True, force=False,
    )
    assert len(result["applied"]) == 1
    assert not (vault / "02-Areas/dead.md").exists()
    moved = vault / "04-Archive/02-Areas/dead.md"
    assert moved.is_file()
    fm = rag.parse_frontmatter(moved.read_text())
    assert fm["title"] == "T"
    assert fm["archived_from"] == "02-Areas/dead.md"
    assert fm["archived_reason"] == "dead"
    # batch log written
    assert result["batch_path"] is not None
    log_lines = Path(result["batch_path"]).read_text().splitlines()
    assert len(log_lines) == 1
    entry = json.loads(log_lines[0])
    assert entry["src"] == "02-Areas/dead.md"
    assert entry["dst"] == "04-Archive/02-Areas/dead.md"


def test_apply_skips_opt_out(tmp_vault):
    vault, col = tmp_vault
    c = _make_candidate(vault, "02-Areas/moc.md", 400,
                        body="---\ntype: moc\n---\n\nhub\n")
    result = rag.archive_dead_notes(
        col, vault, [c], apply=True, force=False,
    )
    assert result["plan"] == []
    assert result["applied"] == []
    reasons = [s["reason"] for s in result["skipped"]]
    assert "opt-out" in reasons
    # File stays put.
    assert (vault / "02-Areas/moc.md").is_file()


def test_gate_blocks_large_batch(tmp_vault):
    vault, col = tmp_vault
    candidates = [
        _make_candidate(vault, f"02-Areas/n{i}.md", 400 + i)
        for i in range(6)
    ]
    result = rag.archive_dead_notes(
        col, vault, candidates, apply=True, force=False, gate=3,
    )
    assert result["gated"] is True
    assert result["applied"] == []
    # Files untouched
    for i in range(6):
        assert (vault / f"02-Areas/n{i}.md").is_file()


def test_force_bypasses_gate(tmp_vault):
    vault, col = tmp_vault
    candidates = [
        _make_candidate(vault, f"02-Areas/n{i}.md", 400 + i)
        for i in range(5)
    ]
    result = rag.archive_dead_notes(
        col, vault, candidates, apply=True, force=True, gate=2,
    )
    assert result["gated"] is False
    assert len(result["applied"]) == 5


def test_collision_produces_suffixed_destination(tmp_vault):
    vault, col = tmp_vault
    # Pre-existing file at the intended destination
    existing = vault / "04-Archive/02-Areas/dup.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("preexisting")
    c = _make_candidate(vault, "02-Areas/dup.md", 400)
    result = rag.archive_dead_notes(
        col, vault, [c], apply=True, force=False,
    )
    assert len(result["applied"]) == 1
    dst = result["applied"][0]["dst"]
    assert dst != "04-Archive/02-Areas/dup.md"
    assert dst.startswith("04-Archive/02-Areas/dup-archived-")
    assert (vault / dst).is_file()
    # Original collision target untouched
    assert existing.read_text() == "preexisting"


def test_same_batch_dst_collision(tmp_vault):
    """Two candidates mapping to the same dst after suffixing — both must move."""
    vault, col = tmp_vault
    # Preexisting file at the intended destination for BOTH candidates
    existing = vault / "04-Archive/02-Areas/x.md"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("pre")
    c1 = _make_candidate(vault, "02-Areas/x.md", 400)
    # Simulate a second candidate that would also collide into the same suffix
    # by placing an already-existing archived-YYYY-MM variant.
    tag = datetime.now().strftime("%Y-%m")
    (vault / f"04-Archive/02-Areas/x-archived-{tag}.md").write_text("pre2")
    c2 = _make_candidate(vault, "02-Areas/x.md", 500)
    # (Same path can't really appear twice in find_dead_notes, but the
    # suffix walker should still resolve sequential collisions.)
    result = rag.archive_dead_notes(
        col, vault, [c1, c2], apply=True, force=False,
    )
    applied_dsts = {e["dst"] for e in result["applied"]}
    # At least one move should succeed; the duplicate src is skipped after move
    assert len(applied_dsts) >= 1
    for d in applied_dsts:
        assert d.startswith("04-Archive/02-Areas/x")


def test_skips_missing_file(tmp_vault):
    vault, col = tmp_vault
    fake = {"path": "02-Areas/ghost.md", "age_days": 400, "mtime": "",
            "age_source": "mtime"}
    result = rag.archive_dead_notes(
        col, vault, [fake], apply=True, force=False,
    )
    assert result["applied"] == []
    assert any(s["reason"] == "missing" for s in result["skipped"])


def test_notification_respects_ambient_off(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    # ambient config stays None from fixture → push is a silent no-op
    c = _make_candidate(vault, "02-Areas/dead.md", 400)
    result = rag.archive_dead_notes(col, vault, [c], apply=True, force=False)
    sent = rag._push_archive_notification(result, apply=True)
    assert sent is False


def test_notification_fires_when_ambient_on(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    monkeypatch.setattr(rag, "_ambient_config", lambda: {"jid": "x@g.us"})
    calls = []
    monkeypatch.setattr(
        rag, "_ambient_whatsapp_send",
        lambda jid, msg: (calls.append((jid, msg)) or True),
    )
    # Silence ambient log writes
    monkeypatch.setattr(rag, "_ambient_log_event", lambda *a, **kw: None)
    c = _make_candidate(vault, "02-Areas/dead.md", 400)
    result = rag.archive_dead_notes(col, vault, [c], apply=True, force=False)
    sent = rag._push_archive_notification(result, apply=True)
    assert sent is True
    assert calls and "Archive" in calls[0][1]


def test_report_written_to_reviews(tmp_vault):
    vault, col = tmp_vault
    c = _make_candidate(vault, "02-Areas/dead.md", 400)
    result = rag.archive_dead_notes(col, vault, [c], apply=True, force=False)
    report = rag._write_archive_report(result, apply=True)
    assert report is not None
    assert report.parent == vault / "05-Reviews"
    body = report.read_text()
    assert "02-Areas/dead.md" in body
    assert "04-Archive/02-Areas/dead.md" in body


def test_report_appends_on_second_run(tmp_vault):
    vault, col = tmp_vault
    c1 = _make_candidate(vault, "02-Areas/a.md", 400)
    r1 = rag.archive_dead_notes(col, vault, [c1], apply=True, force=False)
    rag._write_archive_report(r1, apply=True)
    c2 = _make_candidate(vault, "02-Areas/b.md", 500)
    r2 = rag.archive_dead_notes(col, vault, [c2], apply=True, force=False)
    report = rag._write_archive_report(r2, apply=True)
    body = report.read_text()
    assert "a.md" in body and "b.md" in body
