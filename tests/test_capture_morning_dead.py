import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import chromadb
import pytest

import rag


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeResponse:
    def __init__(self, content):
        self.message = _FakeMessage(content)


@pytest.fixture
def fake_embed(monkeypatch):
    def _embed(texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setattr(rag, "embed", _embed)


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch, fake_embed):
    vault = tmp_path / "vault"
    (vault / "00-Inbox").mkdir(parents=True)
    (vault / "05-Reviews").mkdir(parents=True)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="cmd_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    # Silence auto-index side effects when capture runs.
    monkeypatch.setattr(
        rag, "_index_single_file",
        lambda *a, **kw: "skipped",
    )
    rag._invalidate_corpus_cache()
    return vault, col


# ── capture_note ─────────────────────────────────────────────────────────────


def test_capture_writes_to_inbox(tmp_vault):
    vault, _ = tmp_vault
    path = rag.capture_note("idea suelta sobre ikigai")
    assert path.parent == vault / "00-Inbox"
    assert path.is_file()
    txt = path.read_text()
    assert "idea suelta sobre ikigai" in txt
    assert "type: capture" in txt
    assert "- capture" in txt


def test_capture_with_extra_tags_and_source(tmp_vault):
    path = rag.capture_note(
        "voice transcript",
        tags=["voice", "telegram"],
        source="tg:123",
    )
    txt = path.read_text()
    assert "- capture" in txt
    assert "- voice" in txt
    assert "- telegram" in txt
    assert "source: tg:123" in txt


def test_capture_dedups_on_filename_collision(tmp_vault):
    p1 = rag.capture_note("primera")
    p2 = rag.capture_note("primera")
    assert p1.name != p2.name
    assert p2.name.endswith("-2.md")


def test_capture_empty_raises(tmp_vault):
    with pytest.raises(ValueError):
        rag.capture_note("   \n   ")


def test_capture_custom_title_slug(tmp_vault):
    path = rag.capture_note("body text", title="Claude Code hooks")
    assert "claude-code-hooks" in path.name


def test_capture_slug_first_line(tmp_vault):
    path = rag.capture_note(
        "Llamada con María: temas\n\ndetalles largos abajo")
    assert "llamada-con-mar" in path.name.lower() or \
           "llamada-con-maria" in path.name.lower()


# ── _collect_morning_evidence ────────────────────────────────────────────────


def test_morning_evidence_empty_vault(tmp_vault, tmp_path):
    vault, _ = tmp_vault
    ev = rag._collect_morning_evidence(
        datetime.now(), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    assert ev["recent_notes"] == []
    assert ev["inbox_pending"] == []
    assert ev["todos"] == []
    assert ev["new_contradictions"] == []
    assert ev["low_conf_queries"] == []


def test_morning_picks_up_inbox_pending(tmp_vault, tmp_path):
    vault, _ = tmp_vault
    (vault / "00-Inbox" / "pend.md").write_text("pendiente raw body")
    ev = rag._collect_morning_evidence(
        datetime.now(), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    paths = [i["path"] for i in ev["inbox_pending"]]
    assert "00-Inbox/pend.md" in paths


def test_morning_respects_lookback_window(tmp_vault, tmp_path):
    vault, _ = tmp_vault
    fresh = vault / "02-Areas" / "fresh.md"
    fresh.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_text("modificada recientemente")
    # Mark another as "old" by setting mtime in the past (3 days).
    old = vault / "02-Areas" / "old.md"
    old.write_text("vieja")
    old_mtime = (datetime.now() - timedelta(days=3)).timestamp()
    import os
    os.utime(old, (old_mtime, old_mtime))

    ev = rag._collect_morning_evidence(
        datetime.now(), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
        lookback_hours=36,
    )
    recent_paths = {r["path"] for r in ev["recent_notes"]}
    assert "02-Areas/fresh.md" in recent_paths
    assert "02-Areas/old.md" not in recent_paths


def test_morning_picks_up_todos_in_frontmatter(tmp_vault, tmp_path):
    vault, _ = tmp_vault
    (vault / "02-Areas").mkdir(parents=True, exist_ok=True)
    (vault / "02-Areas" / "with-todo.md").write_text(
        "---\ntodo:\n- llamar a Juan\ndue: 2026-04-20\n---\n\nbody"
    )
    ev = rag._collect_morning_evidence(
        datetime.now(), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    paths = [t["path"] for t in ev["todos"]]
    assert "02-Areas/with-todo.md" in paths


def test_morning_reads_contradiction_sidecar(tmp_vault, tmp_path):
    vault, _ = tmp_vault
    contr = tmp_path / "c.jsonl"
    now = datetime.now()
    entry = {
        "ts": now.isoformat(timespec="seconds"),
        "cmd": "contradict_index",
        "subject_path": "02-Areas/x.md",
        "contradicts": [{"path": "02-Areas/y.md", "why": "X vs Y"}],
    }
    contr.write_text(json.dumps(entry) + "\n")
    ev = rag._collect_morning_evidence(
        now + timedelta(hours=1), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=contr,
    )
    assert len(ev["new_contradictions"]) == 1
    assert ev["new_contradictions"][0]["subject_path"] == "02-Areas/x.md"


def test_morning_low_conf_queries(tmp_vault, tmp_path):
    vault, _ = tmp_vault
    ql = tmp_path / "q.jsonl"
    now = datetime.now()
    entries = [
        {"ts": now.isoformat(timespec="seconds"), "cmd": "query",
         "q": "algo raro", "top_score": 0.005},
        {"ts": now.isoformat(timespec="seconds"), "cmd": "query",
         "q": "ok query", "top_score": 0.4},
    ]
    ql.write_text("\n".join(json.dumps(e) for e in entries))
    ev = rag._collect_morning_evidence(
        now + timedelta(hours=1), vault,
        query_log=ql,
        contradiction_log=tmp_path / "c.jsonl",
    )
    queries = [q["q"] for q in ev["low_conf_queries"]]
    assert "algo raro" in queries
    assert "ok query" not in queries


# ── find_dead_notes ──────────────────────────────────────────────────────────


def _add_chunk(col, path, note, outlinks="", backlink_target=None):
    col.add(
        ids=[f"{path}::0"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        documents=[f"chunk for {note}"],
        metadatas=[{
            "file": path, "note": note, "folder": str(Path(path).parent),
            "tags": "", "outlinks": outlinks, "hash": "x",
        }],
    )
    if backlink_target:
        # A second chunk from a different file pointing AT `backlink_target`
        col.add(
            ids=[f"linker_to_{backlink_target}::0"],
            embeddings=[[1.0, 0.0, 0.0, 0.0]],
            documents=["linker chunk"],
            metadatas=[{
                "file": f"02-Areas/linker-{backlink_target}.md",
                "note": f"linker-{backlink_target}",
                "folder": "02-Areas",
                "tags": "",
                "outlinks": backlink_target,
                "hash": "x",
            }],
        )


def _touch_with_mtime(path: Path, days_ago: int):
    import os
    ts = (datetime.now() - timedelta(days=days_ago)).timestamp()
    os.utime(path, (ts, ts))


def test_dead_finds_old_orphan_note(tmp_vault, tmp_path):
    vault, col = tmp_vault
    target_dir = vault / "02-Areas"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "orphan.md").write_text("old note")
    _touch_with_mtime(target_dir / "orphan.md", 400)
    _add_chunk(col, "02-Areas/orphan.md", "orphan")  # no outlinks, no backlinks
    rag._invalidate_corpus_cache()
    items = rag.find_dead_notes(
        col, vault, query_log=tmp_path / "q.jsonl",
        min_age_days=365,
    )
    paths = [it["path"] for it in items]
    assert "02-Areas/orphan.md" in paths


def test_dead_excludes_backlinked(tmp_vault, tmp_path):
    vault, col = tmp_vault
    target = vault / "02-Areas" / "referenced.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("referenced")
    _touch_with_mtime(target, 400)
    _add_chunk(col, "02-Areas/referenced.md", "referenced",
               backlink_target="referenced")
    rag._invalidate_corpus_cache()
    items = rag.find_dead_notes(col, vault, query_log=tmp_path / "q.jsonl")
    assert "02-Areas/referenced.md" not in [it["path"] for it in items]


def test_dead_excludes_recent_mtime(tmp_vault, tmp_path):
    vault, col = tmp_vault
    target = vault / "02-Areas" / "fresh.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("fresh")
    # mtime default is "now" — recent
    _add_chunk(col, "02-Areas/fresh.md", "fresh")
    rag._invalidate_corpus_cache()
    items = rag.find_dead_notes(col, vault, query_log=tmp_path / "q.jsonl")
    assert "02-Areas/fresh.md" not in [it["path"] for it in items]


def test_dead_excludes_notes_in_exclude_folders(tmp_vault, tmp_path):
    vault, col = tmp_vault
    (vault / "04-Archive").mkdir(parents=True, exist_ok=True)
    target = vault / "04-Archive" / "old.md"
    target.write_text("archived old")
    _touch_with_mtime(target, 800)
    _add_chunk(col, "04-Archive/old.md", "old")
    rag._invalidate_corpus_cache()
    items = rag.find_dead_notes(col, vault, query_log=tmp_path / "q.jsonl")
    assert "04-Archive/old.md" not in [it["path"] for it in items]


def test_dead_excludes_recently_retrieved(tmp_vault, tmp_path):
    vault, col = tmp_vault
    target = vault / "02-Areas" / "used.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("used in queries")
    _touch_with_mtime(target, 400)
    _add_chunk(col, "02-Areas/used.md", "used")
    ql = tmp_path / "q.jsonl"
    # Query that retrieved it recently (7 days ago)
    entry = {
        "ts": (datetime.now() - timedelta(days=7)).isoformat(timespec="seconds"),
        "cmd": "query", "q": "whatever",
        "paths": ["02-Areas/used.md"],
    }
    ql.write_text(json.dumps(entry) + "\n")
    rag._invalidate_corpus_cache()
    items = rag.find_dead_notes(col, vault, query_log=ql, query_window_days=30)
    assert "02-Areas/used.md" not in [it["path"] for it in items]


def test_dead_orders_by_age_desc(tmp_vault, tmp_path):
    vault, col = tmp_vault
    (vault / "02-Areas").mkdir(parents=True, exist_ok=True)
    for name, age in [("a.md", 400), ("b.md", 800), ("c.md", 600)]:
        p = vault / "02-Areas" / name
        p.write_text(name)
        _touch_with_mtime(p, age)
        _add_chunk(col, f"02-Areas/{name}", name.rstrip(".md"))
    rag._invalidate_corpus_cache()
    items = rag.find_dead_notes(col, vault, query_log=tmp_path / "q.jsonl")
    ages = [it["age_days"] for it in items]
    assert ages == sorted(ages, reverse=True)
