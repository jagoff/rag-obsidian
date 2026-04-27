import json
import os
from datetime import datetime

import click
import pytest
from click.testing import CliRunner

import rag


@pytest.fixture
def digest_env(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    qlog = tmp_path / "queries.jsonl"
    clog = tmp_path / "contradictions.jsonl"
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "LOG_PATH", qlog)
    monkeypatch.setattr(rag, "CONTRADICTION_LOG_PATH", clog)
    return vault, qlog, clog


def _write_note(vault, rel_path, body, mtime=None):
    p = vault / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    if mtime is not None:
        ts = mtime.timestamp()
        os.utime(p, (ts, ts))
    return p


def _append_jsonl(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def test_iso_week_label_known_date():
    assert rag._iso_week_label(datetime(2026, 4, 9)) == "2026-W15"
    assert rag._iso_week_label(datetime(2026, 4, 13)) == "2026-W16"


def test_parse_iso_week_returns_monday_to_monday():
    start, end = rag._parse_iso_week("2026-W15")
    assert start == datetime(2026, 4, 6)
    assert end == datetime(2026, 4, 13)


def test_parse_iso_week_rejects_bad_format():
    with pytest.raises(click.BadParameter):
        rag._parse_iso_week("no-such-week")


def test_collect_evidence_empty_sources(digest_env):
    vault, qlog, clog = digest_env
    ev = rag._collect_week_evidence(
        datetime(2026, 4, 6), datetime(2026, 4, 13), vault, qlog, clog,
    )
    assert ev == {
        "recent_notes": [],
        "fm_contradictions": [],
        "index_contradictions": [],
        "query_contradictions": [],
        "low_conf_queries": [],
    }


def test_collect_evidence_recent_notes_respects_window(digest_env):
    vault, qlog, clog = digest_env
    in_window = datetime(2026, 4, 9, 12, 0, 0)
    out_window = datetime(2026, 3, 1, 12, 0, 0)
    _write_note(vault, "foo.md", "---\ntags: []\n---\n# foo\nbody in window.",
                mtime=in_window)
    _write_note(vault, "bar.md", "---\ntags: []\n---\n# bar\nout of window.",
                mtime=out_window)
    ev = rag._collect_week_evidence(
        datetime(2026, 4, 6), datetime(2026, 4, 13), vault, qlog, clog,
    )
    paths = {n["path"] for n in ev["recent_notes"]}
    assert paths == {"foo.md"}
    assert ev["recent_notes"][0]["title"] == "foo"
    assert "body in window" in ev["recent_notes"][0]["snippet"]


def test_collect_evidence_reads_frontmatter_contradicts(digest_env):
    vault, qlog, clog = digest_env
    _write_note(
        vault, "alpha.md",
        "---\ncontradicts:\n- beta.md\n- gamma.md\n---\nbody",
        mtime=datetime(2026, 4, 9, 10, 0, 0),
    )
    _write_note(vault, "no-fm.md", "# plain note",
                mtime=datetime(2026, 4, 9, 10, 0, 0))
    ev = rag._collect_week_evidence(
        datetime(2026, 4, 6), datetime(2026, 4, 13), vault, qlog, clog,
    )
    assert len(ev["fm_contradictions"]) == 1
    fc = ev["fm_contradictions"][0]
    assert fc["path"] == "alpha.md"
    assert fc["targets"] == ["beta.md", "gamma.md"]


def test_collect_evidence_windows_contradiction_log(digest_env):
    vault, qlog, clog = digest_env
    _append_jsonl(clog, [
        {
            "ts": "2026-04-09T10:00:00",
            "cmd": "contradict_index",
            "subject_path": "new.md",
            "contradicts": [{"path": "old.md", "note": "old", "why": "tensión"}],
            "skipped": None,
        },
        {
            "ts": "2026-03-01T10:00:00",
            "cmd": "contradict_index",
            "subject_path": "old-note.md",
            "contradicts": [{"path": "other.md", "why": "viejo"}],
        },
        {
            "ts": "2026-04-10T10:00:00",
            "cmd": "contradict_index",
            "subject_path": "empty.md",
            "contradicts": [],
            "skipped": "too_short",
        },
    ])
    ev = rag._collect_week_evidence(
        datetime(2026, 4, 6), datetime(2026, 4, 13), vault, qlog, clog,
    )
    assert len(ev["index_contradictions"]) == 1
    ic = ev["index_contradictions"][0]
    assert ic["subject_path"] == "new.md"
    assert ic["targets"] == [{"path": "old.md", "why": "tensión"}]


def test_collect_evidence_reads_query_log(digest_env):
    vault, qlog, clog = digest_env
    _append_jsonl(qlog, [
        {
            "ts": "2026-04-09T10:00:00",
            "cmd": "query",
            "q": "¿qué es X?",
            "top_score": 0.35,
            "contradictions": [{"path": "a.md", "why": "X vs Y"}],
        },
        {
            "ts": "2026-04-10T10:00:00",
            "cmd": "query",
            "q": "algo oscuro",
            "top_score": 0.005,
            "contradictions": None,
        },
        {
            "ts": "2026-03-01T10:00:00",
            "cmd": "query",
            "q": "fuera de rango",
            "top_score": 0.001,
        },
    ])
    ev = rag._collect_week_evidence(
        datetime(2026, 4, 6), datetime(2026, 4, 13), vault, qlog, clog,
    )
    assert len(ev["query_contradictions"]) == 1
    assert ev["query_contradictions"][0]["path"] == "a.md"
    low_qs = {lq["q"] for lq in ev["low_conf_queries"]}
    assert low_qs == {"algo oscuro"}


def test_digest_dry_run_with_no_evidence(digest_env, monkeypatch):
    monkeypatch.setattr(
        rag, "_generate_digest_narrative", lambda p: "should-not-be-called",
    )
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["digest", "--week", "2026-W15", "--dry-run"])
    assert result.exit_code == 0
    assert "Sin evidencia" in result.output


def test_digest_dry_run_with_evidence(digest_env, monkeypatch):
    vault, qlog, clog = digest_env
    _write_note(
        vault, "tema.md",
        "---\ncontradicts:\n- viejo.md\n---\nhay un cambio de opinión.",
        mtime=datetime(2026, 4, 9, 10, 0, 0),
    )
    monkeypatch.setattr(
        rag, "_generate_digest_narrative",
        lambda p: "Esta semana [[tema]] reorganizó mi vista previa.",
    )
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["digest", "--week", "2026-W15", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Review 2026-W15" in result.output
    assert "[[tema]]" in result.output
    # dry-run must NOT write
    assert not (vault / "04-Archive/99-obsidian-system/99-AI/reviews" / "2026-W15.md").exists()


def test_digest_writes_and_indexes(digest_env, monkeypatch):
    vault, qlog, clog = digest_env
    _write_note(
        vault, "tema.md",
        "---\ntags: [reflexion]\n---\nbody útil.",
        mtime=datetime(2026, 4, 9, 10, 0, 0),
    )
    monkeypatch.setattr(
        rag, "_generate_digest_narrative",
        lambda p: "Prosa narrativa mock. [[tema]] apareció.",
    )
    indexed: list = []
    monkeypatch.setattr(rag, "get_db", lambda: "FAKE_COL")
    monkeypatch.setattr(
        rag, "_index_single_file", lambda col, path: indexed.append((col, path))
    )
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["digest", "--week", "2026-W15"])
    assert result.exit_code == 0, result.output
    out_path = vault / "04-Archive/99-obsidian-system/99-AI/reviews" / "2026-W15.md"
    assert out_path.is_file()
    written = out_path.read_text(encoding="utf-8")
    assert "week: '2026-W15'" in written
    assert "Prosa narrativa mock" in written
    assert indexed == [("FAKE_COL", out_path)]
