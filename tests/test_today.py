import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from rag import SqliteVecClient as _TestVecClient
import pytest
from click.testing import CliRunner

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
    (vault / "02-Areas").mkdir(parents=True)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    client = _TestVecClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(
        name="today_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)
    monkeypatch.setattr(
        rag, "_index_single_file",
        lambda *a, **kw: "skipped",
    )
    rag._invalidate_corpus_cache()
    return vault, col, tmp_path


def _set_mtime(path: Path, when: datetime):
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def _fake_chat(content: str):
    def _chat(*a, **kw):
        return _FakeResponse(content)
    return _chat


NARRATIVE_STUB = (
    "## 🪞 Lo que pasó hoy\ntexto de recap hoy\n\n"
    "## 📥 Sin procesar\nitem sin tags\n\n"
    "## 🔍 Preguntas abiertas\npregunta\n\n"
    "## 🌅 Para mañana\nseed 1\nseed 2\n"
)


# ── _collect_today_evidence ─────────────────────────────────────────────────


def test_today_evidence_empty_vault(tmp_vault):
    vault, _, tmp_path = tmp_vault
    ev = rag._collect_today_evidence(
        datetime.now(), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    assert ev["recent_notes"] == []
    assert ev["inbox_today"] == []
    assert ev["todos"] == []
    assert ev["new_contradictions"] == []
    assert ev["low_conf_queries"] == []


def test_today_picks_up_modified_today(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p = vault / "02-Areas" / "today.md"
    p.write_text("cuerpo de hoy")
    now = datetime.now().replace(hour=15, minute=0, second=0, microsecond=0)
    _set_mtime(p, now)
    ev = rag._collect_today_evidence(
        now + timedelta(minutes=5), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    paths = {r["path"] for r in ev["recent_notes"]}
    assert "02-Areas/today.md" in paths


def test_today_excludes_yesterday(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p = vault / "02-Areas" / "yesterday.md"
    p.write_text("vieja de ayer")
    yesterday = datetime.now().replace(hour=22, minute=0) - timedelta(days=1)
    _set_mtime(p, yesterday)
    ev = rag._collect_today_evidence(
        datetime.now(), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    paths = {r["path"] for r in ev["recent_notes"]}
    assert "02-Areas/yesterday.md" not in paths


def test_today_excludes_reviews_folder(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p = vault / "05-Reviews" / "2026-04-15.md"
    p.write_text("morning brief")
    now = datetime.now()
    _set_mtime(p, now - timedelta(minutes=10))
    ev = rag._collect_today_evidence(
        now, vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    paths = {r["path"] for r in ev["recent_notes"]}
    assert "05-Reviews/2026-04-15.md" not in paths


def test_today_inbox_capture_routed_to_inbox_bucket(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p = vault / "00-Inbox" / "cap.md"
    p.write_text("captura rápida")
    now = datetime.now().replace(hour=10, minute=30, second=0, microsecond=0)
    _set_mtime(p, now)
    ev = rag._collect_today_evidence(
        now + timedelta(minutes=5), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    inbox_paths = {r["path"] for r in ev["inbox_today"]}
    recent_paths = {r["path"] for r in ev["recent_notes"]}
    assert "00-Inbox/cap.md" in inbox_paths
    assert "00-Inbox/cap.md" not in recent_paths


def test_today_inbox_untagged_flag(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p1 = vault / "00-Inbox" / "untagged.md"
    p1.write_text("sin tags")
    p2 = vault / "00-Inbox" / "tagged.md"
    p2.write_text("---\ntags:\n- area/health\n---\nbody")
    now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    _set_mtime(p1, now)
    _set_mtime(p2, now)
    ev = rag._collect_today_evidence(
        now + timedelta(minutes=5), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    by_path = {r["path"]: r for r in ev["inbox_today"]}
    assert by_path["00-Inbox/untagged.md"]["tags"] == []
    assert "area/health" in by_path["00-Inbox/tagged.md"]["tags"]


def test_today_midnight_boundary(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p_today = vault / "02-Areas" / "just-after-midnight.md"
    p_today.write_text("justo después")
    p_yest = vault / "02-Areas" / "just-before.md"
    p_yest.write_text("justo antes")
    now = datetime.now().replace(hour=0, minute=5, second=0, microsecond=0)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    _set_mtime(p_today, today_start + timedelta(seconds=1))
    _set_mtime(p_yest, today_start - timedelta(seconds=1))
    ev = rag._collect_today_evidence(
        now, vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    paths = {r["path"] for r in ev["recent_notes"]}
    assert "02-Areas/just-after-midnight.md" in paths
    assert "02-Areas/just-before.md" not in paths


def test_today_low_conf_queries_today_only(tmp_vault):
    vault, _, tmp_path = tmp_vault
    ql = tmp_path / "q.jsonl"
    now = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
    today_ts = now.replace(hour=10, minute=0).isoformat(timespec="seconds")
    yest_ts = (now - timedelta(days=1)).isoformat(timespec="seconds")
    entries = [
        {"ts": today_ts, "cmd": "query", "q": "q de hoy", "top_score": 0.005},
        {"ts": yest_ts, "cmd": "query", "q": "q de ayer", "top_score": 0.003},
        {"ts": today_ts, "cmd": "query", "q": "q buena", "top_score": 0.4},
    ]
    ql.write_text("\n".join(json.dumps(e) for e in entries))
    ev = rag._collect_today_evidence(
        now, vault,
        query_log=ql,
        contradiction_log=tmp_path / "c.jsonl",
    )
    qs = [q["q"] for q in ev["low_conf_queries"]]
    assert "q de hoy" in qs
    assert "q de ayer" not in qs
    assert "q buena" not in qs


def test_today_contradictions_today_only(tmp_vault):
    vault, _, tmp_path = tmp_vault
    contr = tmp_path / "c.jsonl"
    now = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
    today_entry = {
        "ts": now.replace(hour=11, minute=0).isoformat(timespec="seconds"),
        "cmd": "contradict_index",
        "subject_path": "02-Areas/x.md",
        "contradicts": [{"path": "02-Areas/y.md", "why": "X vs Y"}],
    }
    yest_entry = {
        "ts": (now - timedelta(days=1)).isoformat(timespec="seconds"),
        "cmd": "contradict_index",
        "subject_path": "02-Areas/a.md",
        "contradicts": [{"path": "02-Areas/b.md", "why": "A vs B"}],
    }
    contr.write_text(
        json.dumps(today_entry) + "\n" + json.dumps(yest_entry) + "\n"
    )
    ev = rag._collect_today_evidence(
        now, vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=contr,
    )
    subjects = [c["subject_path"] for c in ev["new_contradictions"]]
    assert subjects == ["02-Areas/x.md"]


def test_today_todo_frontmatter_in_window(tmp_vault):
    vault, _, tmp_path = tmp_vault
    p = vault / "02-Areas" / "with-todo.md"
    p.write_text("---\ntodo:\n- algo\ndue: 2026-05-01\n---\nbody")
    now = datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)
    _set_mtime(p, now)
    ev = rag._collect_today_evidence(
        now + timedelta(minutes=1), vault,
        query_log=tmp_path / "q.jsonl",
        contradiction_log=tmp_path / "c.jsonl",
    )
    paths = [t["path"] for t in ev["todos"]]
    assert "02-Areas/with-todo.md" in paths


# ── CLI `rag today` ──────────────────────────────────────────────────────────


def test_today_cli_silent_no_op_when_empty(tmp_vault, monkeypatch):
    monkeypatch.setattr(rag, "LOG_PATH", tmp_vault[2] / "q.jsonl")
    monkeypatch.setattr(rag, "CONTRADICTION_LOG_PATH", tmp_vault[2] / "c.jsonl")
    called = []

    def _boom(*a, **kw):
        called.append(True)
        return _FakeResponse("must not run")
    monkeypatch.setattr(rag.ollama, "chat", _boom)

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["today", "--plain"])
    assert result.exit_code == 0
    assert "sin actividad hoy" in result.output
    assert called == []


def test_today_cli_dry_run_does_not_write(tmp_vault, monkeypatch):
    vault, _, tmp_path = tmp_vault
    monkeypatch.setattr(rag, "LOG_PATH", tmp_path / "q.jsonl")
    monkeypatch.setattr(rag, "CONTRADICTION_LOG_PATH", tmp_path / "c.jsonl")
    p = vault / "02-Areas" / "activity.md"
    p.write_text("algo hoy")
    _set_mtime(p, datetime.now() - timedelta(minutes=5))

    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "stub")
    monkeypatch.setattr(rag.ollama, "chat", _fake_chat(NARRATIVE_STUB))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["today", "--dry-run", "--plain"])
    assert result.exit_code == 0
    assert "type: evening-brief" in result.output
    # 4 expected headers
    for h in ("Lo que pasó hoy", "Sin procesar", "Preguntas abiertas", "Para mañana"):
        assert h in result.output
    # No file written to 05-Reviews
    files = list((vault / "05-Reviews").glob("*.md"))
    assert files == []


def test_today_cli_writes_evening_suffix_and_frontmatter(tmp_vault, monkeypatch):
    vault, _, tmp_path = tmp_vault
    monkeypatch.setattr(rag, "LOG_PATH", tmp_path / "q.jsonl")
    monkeypatch.setattr(rag, "CONTRADICTION_LOG_PATH", tmp_path / "c.jsonl")
    p = vault / "02-Areas" / "activity.md"
    p.write_text("algo hoy")
    _set_mtime(p, datetime.now() - timedelta(minutes=5))

    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "stub")
    monkeypatch.setattr(rag.ollama, "chat", _fake_chat(NARRATIVE_STUB))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["today", "--plain"])
    assert result.exit_code == 0
    date_label = datetime.now().strftime("%Y-%m-%d")
    expected = vault / "05-Reviews" / f"{date_label}-evening.md"
    assert expected.is_file(), result.output
    body = expected.read_text()
    assert "type: evening-brief" in body
    assert f"date: '{date_label}'" in body
    assert "- evening-brief" in body
    for h in ("Lo que pasó hoy", "Sin procesar", "Preguntas abiertas", "Para mañana"):
        assert h in body


def test_today_cli_does_not_collide_with_morning_file(tmp_vault, monkeypatch):
    vault, _, tmp_path = tmp_vault
    monkeypatch.setattr(rag, "LOG_PATH", tmp_path / "q.jsonl")
    monkeypatch.setattr(rag, "CONTRADICTION_LOG_PATH", tmp_path / "c.jsonl")
    date_label = datetime.now().strftime("%Y-%m-%d")
    morning_file = vault / "05-Reviews" / f"{date_label}.md"
    morning_file.write_text("morning brief existente")

    p = vault / "02-Areas" / "a.md"
    p.write_text("contenido")
    _set_mtime(p, datetime.now() - timedelta(minutes=5))

    monkeypatch.setattr(rag, "resolve_chat_model", lambda: "stub")
    monkeypatch.setattr(rag.ollama, "chat", _fake_chat(NARRATIVE_STUB))

    runner = CliRunner()
    result = runner.invoke(rag.cli, ["today", "--plain"])
    assert result.exit_code == 0
    # Morning file untouched
    assert morning_file.read_text() == "morning brief existente"
    # Evening file present separately
    assert (vault / "05-Reviews" / f"{date_label}-evening.md").is_file()


def test_today_plist_registered_in_services(tmp_path):
    spec = rag._services_spec("/tmp/fake-rag")
    labels = [s[0] for s in spec]
    assert "com.fer.obsidian-rag-today" in labels
    today_entry = next(s for s in spec if s[0] == "com.fer.obsidian-rag-today")
    plist_xml = today_entry[2]
    assert "<string>today</string>" in plist_xml
    for wd in (1, 2, 3, 4, 5):
        assert f"<integer>{wd}</integer>" in plist_xml
    assert "<key>Hour</key><integer>22</integer>" in plist_xml
    assert "today.log" in plist_xml
    assert "today.error.log" in plist_xml
