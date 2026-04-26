import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from rag import SqliteVecClient as _TestVecClient
import pytest
from click.testing import CliRunner

import rag


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch, fake_embed):
    vault = tmp_path / "vault"
    (vault / "00-Inbox").mkdir(parents=True)
    (vault / "02-Areas").mkdir(parents=True)
    (vault / "04-Archive/99-obsidian-system/99-Claude/reviews").mkdir(parents=True)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    client = _TestVecClient(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="followup_test", metadata={"hnsw:space": "cosine"},
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    monkeypatch.setattr(
        rag, "_index_single_file",
        lambda *a, **kw: "skipped",
    )
    # Stub osascript + helper LLM by default so no test accidentally hits
    # Reminders.app or Ollama. `find_followup_loops` auto-fetches completed
    # reminders when `completed_reminders is None`, and `_classify_followup_loop`
    # falls back to `_followup_judge` (Ollama) when `judge_fn is None`. Both
    # took 20-55s per hit in the full-suite audit (2026-04-16). Individual
    # tests that want real-ish judge behaviour still pass `judge_fn=...`.
    monkeypatch.setattr(rag, "_fetch_completed_reminders", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "_followup_judge", lambda *a, **kw: (False, ""))
    rag._invalidate_corpus_cache()
    return vault, col


def _touch_mtime(path: Path, days_ago: int):
    ts = (datetime.now() - timedelta(days=days_ago)).timestamp()
    os.utime(path, (ts, ts))


# ── _extract_followup_loops ─────────────────────────────────────────────────


def test_extract_frontmatter_todo_list():
    raw = "---\ntodo:\n- llamar a Juan\n- mandar email\n---\n\nbody"
    loops = rag._extract_followup_loops(raw, "02-Areas/x.md", 0)
    todo_texts = [l["loop_text"] for l in loops if l["kind"] == "todo"]
    assert "llamar a Juan" in todo_texts
    assert "mandar email" in todo_texts


def test_extract_frontmatter_due_scalar():
    raw = "---\ndue: 2026-04-20\n---\n\nbody"
    loops = rag._extract_followup_loops(raw, "02-Areas/x.md", 0)
    assert any(l["kind"] == "todo" and "2026-04-20" in l["loop_text"] for l in loops)


def test_extract_unchecked_checkboxes_only():
    raw = "body\n\n- [ ] revisar código\n- [x] ya hecho\n- [ ] mandar PR\n"
    loops = rag._extract_followup_loops(raw, "02-Areas/x.md", 0)
    texts = [l["loop_text"] for l in loops if l["kind"] == "checkbox"]
    assert "revisar código" in texts
    assert "mandar PR" in texts
    assert "ya hecho" not in texts


def test_extract_inline_imperatives():
    raw = (
        "notas sueltas\n\n"
        "tengo que llamar al médico mañana.\n"
        "pendiente: revisar el presupuesto\n"
        "algo random acá\n"
    )
    loops = rag._extract_followup_loops(raw, "02-Areas/x.md", 0)
    inline = [l["loop_text"] for l in loops if l["kind"] == "inline"]
    assert any("llamar al médico" in t for t in inline)
    assert any("revisar el presupuesto" in t for t in inline)


def test_extract_no_loops_in_plain_note():
    raw = "solo prosa sin acciones. todo tranquilo por acá."
    loops = rag._extract_followup_loops(raw, "02-Areas/x.md", 0)
    # "todo tranquilo" has no colon/verb→obj split that matches our regex.
    assert [l for l in loops if l["kind"] in ("todo", "checkbox")] == []


def test_extract_malformed_frontmatter_falls_back():
    raw = "---\ntodo: [broken\n---\n\n- [ ] checkbox ok\n"
    loops = rag._extract_followup_loops(raw, "02-Areas/x.md", 0)
    kinds = {l["kind"] for l in loops}
    assert "checkbox" in kinds


def test_extract_inline_skipped_inside_checkbox():
    raw = "- [ ] tengo que terminar esto\n"
    loops = rag._extract_followup_loops(raw, "02-Areas/x.md", 0)
    checkboxes = [l for l in loops if l["kind"] == "checkbox"]
    inlines = [l for l in loops if l["kind"] == "inline"]
    assert len(checkboxes) == 1
    assert len(inlines) == 0


def test_extract_carries_iso_extracted_at():
    ts = datetime(2026, 1, 15, 10, 30).timestamp()
    raw = "---\ntodo:\n- algo\n---\n"
    loops = rag._extract_followup_loops(raw, "02-Areas/x.md", ts)
    assert loops[0]["extracted_at"].startswith("2026-01-15")


# ── _classify_followup_loop ─────────────────────────────────────────────────


def test_classify_activo_when_recent_and_no_evidence(tmp_vault, monkeypatch):
    _, col = tmp_vault
    now = datetime.now()
    loop = {
        "source_note": "02-Areas/x.md",
        "loop_text": "llamar a Juan",
        "extracted_at": (now - timedelta(days=3)).isoformat(timespec="seconds"),
        "kind": "todo",
    }
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"docs": [], "metas": [], "scores": []},
    )
    out = rag._classify_followup_loop(col, loop, now)
    assert out["status"] == "activo"
    assert out["age_days"] == 3


def test_classify_stale_when_old_and_no_evidence(tmp_vault, monkeypatch):
    _, col = tmp_vault
    now = datetime.now()
    loop = {
        "source_note": "02-Areas/x.md",
        "loop_text": "llamar a Juan",
        "extracted_at": (now - timedelta(days=30)).isoformat(timespec="seconds"),
        "kind": "todo",
    }
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"docs": [], "metas": [], "scores": []},
    )
    out = rag._classify_followup_loop(col, loop, now)
    assert out["status"] == "stale"


def test_classify_resolved_when_judge_says_yes(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    now = datetime.now()
    loop = {
        "source_note": "02-Areas/x.md",
        "loop_text": "llamar a Juan",
        "extracted_at": (now - timedelta(days=5)).isoformat(timespec="seconds"),
        "kind": "todo",
    }
    later = vault / "02-Areas" / "resolution.md"
    later.write_text("hablé con Juan ayer, todo ok")
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {
            "docs": ["hablé con Juan ayer, todo ok"],
            "metas": [{"file": "02-Areas/resolution.md"}],
            "scores": [0.5],
        },
    )
    out = rag._classify_followup_loop(
        col, loop, now, judge_fn=lambda q, ctx: (True, "hablaron"),
    )
    assert out["status"] == "resolved"
    assert out["resolution_path"] == "02-Areas/resolution.md"
    assert out["reason"] == "hablaron"


def test_classify_not_resolved_when_judge_says_no(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    now = datetime.now()
    loop = {
        "source_note": "02-Areas/x.md",
        "loop_text": "llamar a Juan",
        "extracted_at": (now - timedelta(days=2)).isoformat(timespec="seconds"),
        "kind": "todo",
    }
    later = vault / "02-Areas" / "other.md"
    later.write_text("algo sin relación")
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {
            "docs": ["algo sin relación"],
            "metas": [{"file": "02-Areas/other.md"}],
            "scores": [0.5],
        },
    )
    out = rag._classify_followup_loop(
        col, loop, now, judge_fn=lambda q, ctx: (False, ""),
    )
    assert out["status"] == "activo"
    assert out["resolution_path"] is None


def test_classify_ignores_self_citation(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    now = datetime.now()
    source = vault / "02-Areas" / "x.md"
    source.write_text("tengo que llamar a Juan")
    loop = {
        "source_note": "02-Areas/x.md",
        "loop_text": "llamar a Juan",
        "extracted_at": (now - timedelta(days=3)).isoformat(timespec="seconds"),
        "kind": "inline",
    }
    judge_calls = []

    def _judge(q, ctx):
        judge_calls.append((q, ctx))
        return True, "sí"

    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {
            "docs": ["tengo que llamar a Juan"],
            "metas": [{"file": "02-Areas/x.md"}],
            "scores": [0.9],
        },
    )
    out = rag._classify_followup_loop(col, loop, now, judge_fn=_judge)
    assert out["status"] == "activo"
    assert judge_calls == []


def test_classify_ignores_older_than_extracted(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    now = datetime.now()
    loop = {
        "source_note": "02-Areas/x.md",
        "loop_text": "llamar a Juan",
        "extracted_at": (now - timedelta(days=5)).isoformat(timespec="seconds"),
        "kind": "todo",
    }
    older = vault / "02-Areas" / "older.md"
    older.write_text("algo viejo")
    _touch_mtime(older, 20)
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {
            "docs": ["algo viejo"],
            "metas": [{"file": "02-Areas/older.md"}],
            "scores": [0.9],
        },
    )
    called = []
    out = rag._classify_followup_loop(
        col, loop, now,
        judge_fn=lambda q, ctx: (called.append(1), (True, "x"))[1],
    )
    assert out["status"] == "activo"
    assert called == []


def test_classify_handles_tzaware_modified_metadata(tmp_vault, monkeypatch):
    """Regression: chunk metadata carries tz-aware ISO (e.g. '…-03:00') while
    `extracted_at` is naive — the comparison used to crash with TypeError."""
    vault, col = tmp_vault
    now = datetime.now()
    loop = {
        "source_note": "02-Areas/x.md",
        "loop_text": "llamar a Juan",
        "extracted_at": (now - timedelta(days=5)).isoformat(timespec="seconds"),
        "kind": "todo",
    }
    later = vault / "02-Areas" / "later.md"
    later.write_text("hablé con Juan")
    tz_aware_iso = (now - timedelta(days=1)).isoformat(timespec="seconds") + "-03:00"
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {
            "docs": ["hablé con Juan"],
            "metas": [{"file": "02-Areas/later.md", "modified": tz_aware_iso}],
            "scores": [0.5],
        },
    )
    out = rag._classify_followup_loop(
        col, loop, now, judge_fn=lambda q, ctx: (True, "ok"),
    )
    assert out["status"] == "resolved"
    assert out["resolution_path"] == "02-Areas/later.md"


def test_classify_below_min_score_unresolved(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    now = datetime.now()
    loop = {
        "source_note": "02-Areas/x.md",
        "loop_text": "llamar a Juan",
        "extracted_at": (now - timedelta(days=2)).isoformat(timespec="seconds"),
        "kind": "todo",
    }
    later = vault / "02-Areas" / "later.md"
    later.write_text("irrelevante")
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {
            "docs": ["irrelevante"],
            "metas": [{"file": "02-Areas/later.md"}],
            "scores": [0.001],
        },
    )
    called = []
    out = rag._classify_followup_loop(
        col, loop, now,
        judge_fn=lambda q, ctx: (called.append(1), (True, "x"))[1],
    )
    assert out["status"] == "activo"
    assert called == []


# ── find_followup_loops (end-to-end) ────────────────────────────────────────


def _write(vault: Path, rel: str, content: str, days_ago: int = 0):
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    if days_ago:
        _touch_mtime(p, days_ago)
    return p


def test_find_followup_extracts_all_three_kinds(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    _write(vault, "02-Areas/todo.md",
           "---\ntodo:\n- llamar médico\n---\n\nbody")
    _write(vault, "02-Areas/check.md", "- [ ] revisar PR 42\n")
    _write(vault, "02-Areas/inline.md",
           "blah\n\ntengo que mandar el informe hoy\n")
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"docs": [], "metas": [], "scores": []},
    )
    items = rag.find_followup_loops(col, vault, days=30)
    kinds = {it["kind"] for it in items}
    assert kinds == {"todo", "checkbox", "inline"}


def test_find_followup_days_filter(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    _write(vault, "02-Areas/fresh.md", "- [ ] fresh task\n")
    _write(vault, "02-Areas/old.md", "- [ ] ancient task\n", days_ago=120)
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"docs": [], "metas": [], "scores": []},
    )
    items = rag.find_followup_loops(col, vault, days=30)
    sources = {it["source_note"] for it in items}
    assert "02-Areas/fresh.md" in sources
    assert "02-Areas/old.md" not in sources


def test_find_followup_resolution_classifies_correctly(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    _write(vault, "02-Areas/todo.md",
           "---\ntodo:\n- llamar a Juan\n---\n")
    later = _write(vault, "02-Areas/later.md", "llamé a Juan, cerrado")

    def _retrieve_stub(col_, q, k, folder, **kw):
        return {
            "docs": ["llamé a Juan, cerrado"],
            "metas": [{"file": "02-Areas/later.md"}],
            "scores": [0.8],
        }
    monkeypatch.setattr(rag, "retrieve", _retrieve_stub)
    items = rag.find_followup_loops(
        col, vault, days=30,
        judge_fn=lambda q, ctx: (True, "hecho"),
    )
    resolved = [it for it in items if it["status"] == "resolved"]
    assert len(resolved) == 1
    assert resolved[0]["resolution_path"] == "02-Areas/later.md"


def test_find_followup_excludes_dot_dirs(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    _write(vault, ".trash/x.md", "- [ ] nope\n")
    _write(vault, "02-Areas/ok.md", "- [ ] yep\n")
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"docs": [], "metas": [], "scores": []},
    )
    items = rag.find_followup_loops(col, vault, days=30)
    assert [it["source_note"] for it in items] == ["02-Areas/ok.md"]


def test_find_followup_sort_stale_first(tmp_vault, monkeypatch):
    vault, col = tmp_vault
    # Extracted_at derived from note mtime since there's no frontmatter `created`.
    _write(vault, "02-Areas/stale.md", "- [ ] viejo\n", days_ago=25)
    _write(vault, "02-Areas/activo.md", "- [ ] nuevo\n", days_ago=2)
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"docs": [], "metas": [], "scores": []},
    )
    items = rag.find_followup_loops(col, vault, days=60)
    statuses = [it["status"] for it in items]
    assert statuses[0] == "stale"
    assert "activo" in statuses


# ── CLI ─────────────────────────────────────────────────────────────────────


def test_cli_json_output_shape(tmp_vault, monkeypatch, tmp_path):
    vault, col = tmp_vault
    _write(vault, "02-Areas/t.md", "- [ ] una tarea\n")
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"docs": [], "metas": [], "scores": []},
    )
    monkeypatch.setattr(rag, "LOG_PATH", tmp_path / "q.jsonl")
    runner = CliRunner()
    res = runner.invoke(rag.cli, ["followup", "--json", "--days", "30"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output.strip())
    assert isinstance(data, list)
    assert data and {
        "source_note", "loop_text", "extracted_at", "kind",
        "status", "age_days",
    }.issubset(data[0].keys())


def test_cli_status_filter(tmp_vault, monkeypatch, tmp_path):
    vault, col = tmp_vault
    _write(vault, "02-Areas/stale.md", "- [ ] viejo\n", days_ago=25)
    _write(vault, "02-Areas/activo.md", "- [ ] nuevo\n", days_ago=2)
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"docs": [], "metas": [], "scores": []},
    )
    monkeypatch.setattr(rag, "LOG_PATH", tmp_path / "q.jsonl")
    runner = CliRunner()
    res = runner.invoke(
        rag.cli,
        ["followup", "--json", "--days", "60", "--status", "stale"],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output.strip())
    assert all(it["status"] == "stale" for it in data)
    assert len(data) >= 1


def test_cli_empty_vault_prints_message(tmp_vault, monkeypatch, tmp_path):
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"docs": [], "metas": [], "scores": []},
    )
    monkeypatch.setattr(rag, "LOG_PATH", tmp_path / "q.jsonl")
    runner = CliRunner()
    res = runner.invoke(rag.cli, ["followup", "--plain"])
    assert res.exit_code == 0
    assert "Sin open loops" in res.output
