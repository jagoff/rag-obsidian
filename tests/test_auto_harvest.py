"""Unit tests for `rag feedback auto-harvest` (Feature #1 del 2026-04-23).

LLM-as-judge autónomo que labelea queries low-confidence sin intervención
del usuario. Desbloquea el gate de fine-tune del reranker (GC#2.C, 20
corrective_paths) sin que el user se siente a labelear manualmente.

Todos los tests mockean `rag._summary_client` y `rag._ragvec_state_conn`
para no tocar ollama ni la DB real.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3

import pytest
from click.testing import CliRunner

import rag


# ── fixtures ─────────────────────────────────────────────────────────────

_FEEDBACK_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_feedback ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ts TEXT NOT NULL,"
    " turn_id TEXT,"
    " rating INTEGER NOT NULL,"
    " q TEXT,"
    " scope TEXT,"
    " paths_json TEXT,"
    " extra_json TEXT,"
    " UNIQUE(turn_id, rating, ts)"
    ")"
)

_QUERIES_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_queries ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ts TEXT NOT NULL,"
    " cmd TEXT,"
    " q TEXT NOT NULL,"
    " session TEXT,"
    " paths_json TEXT,"
    " scores_json TEXT,"
    " top_score REAL,"
    " extra_json TEXT"
    ")"
)

_GOLDEN_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_feedback_golden ("
    " q TEXT, path TEXT, weight REAL)",
    "CREATE TABLE IF NOT EXISTS rag_feedback_golden_meta ("
    " k TEXT PRIMARY KEY, v TEXT)",
)


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeResponse:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeClient:
    """Stub for rag._summary_client() — stores the last call + returns
    whatever the test installed via `set_response`."""

    def __init__(self):
        self._next: list = []  # list of (content_or_exception)
        self.calls: list[dict] = []

    def set_response(self, content: str | Exception):
        self._next.append(content)

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self._next:
            return _FakeResponse('{"verdict": "none", "confidence": 0.0, "reason": "no-setup"}')
        nxt = self._next.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return _FakeResponse(nxt)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Temp telemetry.db + patched `_ragvec_state_conn`."""
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(_FEEDBACK_DDL)
    conn.execute(_QUERIES_DDL)
    for ddl in _GOLDEN_DDL:
        conn.execute(ddl)
    conn.commit()

    @contextlib.contextmanager
    def _fake_conn():
        c = sqlite3.connect(str(db_path))
        try:
            yield c
            c.commit()
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_conn)
    monkeypatch.setattr(rag, "_feedback_golden_memo", None, raising=False)
    monkeypatch.setattr(rag, "_feedback_golden_source_ts_sql", None, raising=False)
    try:
        yield conn, db_path
    finally:
        conn.close()


@pytest.fixture
def fake_ollama(monkeypatch):
    """Patch rag._summary_client() → returns a _FakeClient singleton."""
    client = _FakeClient()
    monkeypatch.setattr(rag, "_summary_client", lambda: client)
    return client


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    """Point _resolve_vault_path() at a temp dir with a couple of .md files."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    return vault


def _insert_q(conn, *, ts, q, top_score, paths, scores=None, cmd="query"):
    conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, paths_json, scores_json, top_score) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, cmd, q,
         json.dumps(paths),
         json.dumps(scores or [0.0] * len(paths)),
         top_score),
    )
    conn.commit()


# ── _auto_harvest_snippets ───────────────────────────────────────────────


def test_snippets_reads_file_and_returns_title_plus_body(tmp_vault):
    (tmp_vault / "a.md").write_text(
        "# Mi Título\n\nCuerpo del archivo con detalles.\n"
    )
    snippets = rag._auto_harvest_snippets(["a.md"], vault=tmp_vault)
    assert len(snippets) == 1
    path, body = snippets[0]
    assert path == "a.md"
    assert "Mi Título" in body
    assert "detalles" in body


def test_snippets_strips_yaml_frontmatter(tmp_vault):
    (tmp_vault / "b.md").write_text(
        "---\ntitle: Foo\ntag: bar\n---\n\n# Body heading\n\nContent.\n"
    )
    snippets = rag._auto_harvest_snippets(["b.md"], vault=tmp_vault)
    _, body = snippets[0]
    assert "title:" not in body
    assert "Body heading" in body


def test_snippets_missing_file_returns_empty(tmp_vault):
    snippets = rag._auto_harvest_snippets(["missing.md"], vault=tmp_vault)
    assert snippets == [("missing.md", "")]


def test_snippets_caps_at_5_paths(tmp_vault):
    for i in range(10):
        (tmp_vault / f"f{i}.md").write_text(f"# F{i}\n")
    snippets = rag._auto_harvest_snippets(
        [f"f{i}.md" for i in range(10)], vault=tmp_vault
    )
    assert len(snippets) == 5


# ── _auto_harvest_judge ──────────────────────────────────────────────────


def test_judge_returns_parsed_verdict(fake_ollama):
    fake_ollama.set_response(json.dumps({
        "verdict": "a.md", "confidence": 0.92, "reason": "match perfecto",
    }))
    result = rag._auto_harvest_judge(
        "qué es foo?", [("a.md", "título foo"), ("b.md", "otro")],
    )
    assert result is not None
    assert result["verdict"] == "a.md"
    assert result["confidence"] == pytest.approx(0.92)
    assert "match perfecto" in result["reason"]


def test_judge_treats_none_verdict_as_null(fake_ollama):
    fake_ollama.set_response(json.dumps({
        "verdict": "none", "confidence": 0.9, "reason": "ninguno aplica",
    }))
    result = rag._auto_harvest_judge(
        "x?", [("a.md", "algo")],
    )
    assert result is not None
    assert result["verdict"] is None
    assert result["confidence"] == pytest.approx(0.9)


def test_judge_ollama_exception_returns_none(fake_ollama):
    fake_ollama.set_response(RuntimeError("ollama down"))
    result = rag._auto_harvest_judge(
        "x?", [("a.md", "algo")],
    )
    assert result is None


def test_judge_invalid_json_returns_none(fake_ollama):
    fake_ollama.set_response("not valid JSON {{{")
    result = rag._auto_harvest_judge(
        "x?", [("a.md", "algo")],
    )
    assert result is None


def test_judge_clamps_confidence_to_0_1(fake_ollama):
    fake_ollama.set_response(json.dumps({
        "verdict": "a.md", "confidence": 1.7, "reason": "",
    }))
    result = rag._auto_harvest_judge("x?", [("a.md", "")])
    assert result["confidence"] == pytest.approx(1.0)

    fake_ollama.set_response(json.dumps({
        "verdict": "a.md", "confidence": -0.3, "reason": "",
    }))
    result = rag._auto_harvest_judge("x?", [("a.md", "")])
    assert result["confidence"] == pytest.approx(0.0)


def test_judge_empty_candidates_returns_none(fake_ollama):
    result = rag._auto_harvest_judge("x?", [])
    assert result is None


def test_judge_missing_confidence_returns_zero_conf(fake_ollama):
    fake_ollama.set_response(json.dumps({
        "verdict": "a.md", "reason": "sin confidence",
    }))
    result = rag._auto_harvest_judge("x?", [("a.md", "")])
    assert result is not None
    assert result["confidence"] == pytest.approx(0.0)


# ── auto_harvest ─────────────────────────────────────────────────────────


def test_auto_harvest_no_candidates_returns_empty_stats(temp_db, tmp_vault):
    stats = rag.auto_harvest(since_days=1, limit=10)
    assert stats["processed"] == 0
    assert stats["judged_positive"] == 0
    assert stats["judged_negative"] == 0


def test_auto_harvest_high_conf_positive_inserts_row(
    temp_db, tmp_vault, fake_ollama,
):
    conn, _ = temp_db
    (tmp_vault / "golden.md").write_text("# Golden\n\nLa respuesta correcta.\n")
    (tmp_vault / "other.md").write_text("# Other\n\nIrrelevante.\n")
    _insert_q(conn, ts="2026-04-23T10:00", q="qué es golden?",
              top_score=0.1, paths=["golden.md", "other.md"])

    fake_ollama.set_response(json.dumps({
        "verdict": "golden.md", "confidence": 0.95, "reason": "",
    }))

    stats = rag.auto_harvest(
        since_days=30, confidence_below=0.5, min_judge_conf=0.8, limit=5,
    )
    assert stats["processed"] == 1
    assert stats["judged_positive"] == 1
    assert stats["judged_negative"] == 0

    row = conn.execute(
        "SELECT rating, q, paths_json, extra_json FROM rag_feedback"
    ).fetchone()
    assert row is not None
    assert row[0] == 1
    assert row[1] == "qué es golden?"
    assert json.loads(row[2]) == ["golden.md"]
    extra = json.loads(row[3])
    assert extra["corrective_path"] == "golden.md"
    # Quick Win C.7 (2026-04-29): auto_harvest tags rows con
    # source='auto-harvester' (distinto del manual 'harvester' skill).
    # Pre-fix _feedback_insert_harvested hardcodeaba 'harvester' para
    # todo, lo que hacia indistinguible auto vs manual en auditoria.
    assert extra["source"] == "auto-harvester"


def test_auto_harvest_low_conf_skips_without_insert(
    temp_db, tmp_vault, fake_ollama,
):
    conn, _ = temp_db
    (tmp_vault / "a.md").write_text("# A\n\nAlgo.\n")
    _insert_q(conn, ts="2026-04-23T10:00", q="ambiguo query",
              top_score=0.05, paths=["a.md"])

    fake_ollama.set_response(json.dumps({
        "verdict": "a.md", "confidence": 0.6, "reason": "no seguro",
    }))

    stats = rag.auto_harvest(
        since_days=30, confidence_below=0.5, min_judge_conf=0.8, limit=5,
    )
    assert stats["processed"] == 1
    assert stats["judged_positive"] == 0
    assert stats["skipped_low_conf"] == 1

    rows = conn.execute("SELECT COUNT(*) FROM rag_feedback").fetchone()
    assert rows[0] == 0


def test_auto_harvest_none_verdict_high_conf_inserts_negative(
    temp_db, tmp_vault, fake_ollama,
):
    conn, _ = temp_db
    (tmp_vault / "a.md").write_text("# A\n\nIrrelevante 1.\n")
    (tmp_vault / "b.md").write_text("# B\n\nIrrelevante 2.\n")
    _insert_q(conn, ts="2026-04-23T10:00", q="nada relevante acá",
              top_score=0.05, paths=["a.md", "b.md"])

    fake_ollama.set_response(json.dumps({
        "verdict": "none", "confidence": 0.9, "reason": "todos irrelevantes",
    }))

    stats = rag.auto_harvest(
        since_days=30, confidence_below=0.5, min_judge_conf=0.8, limit=5,
    )
    assert stats["judged_negative"] == 1
    assert stats["judged_positive"] == 0

    row = conn.execute(
        "SELECT rating, paths_json FROM rag_feedback"
    ).fetchone()
    assert row[0] == -1
    assert json.loads(row[1]) == ["a.md", "b.md"]


def test_auto_harvest_hallucinated_path_is_skipped(
    temp_db, tmp_vault, fake_ollama,
):
    conn, _ = temp_db
    (tmp_vault / "a.md").write_text("# A\n\nAlgo.\n")
    _insert_q(conn, ts="2026-04-23T10:00", q="inventame un path",
              top_score=0.05, paths=["a.md"])

    fake_ollama.set_response(json.dumps({
        "verdict": "totally-invented.md", "confidence": 0.99, "reason": "",
    }))

    stats = rag.auto_harvest(
        since_days=30, confidence_below=0.5, min_judge_conf=0.8, limit=5,
    )
    assert stats["skipped_invalid_path"] == 1
    assert stats["judged_positive"] == 0

    rows = conn.execute("SELECT COUNT(*) FROM rag_feedback").fetchone()
    assert rows[0] == 0


def test_auto_harvest_judge_failure_counted_separately(
    temp_db, tmp_vault, fake_ollama,
):
    conn, _ = temp_db
    (tmp_vault / "a.md").write_text("# A\n\n.\n")
    _insert_q(conn, ts="2026-04-23T10:00", q="ollama falla",
              top_score=0.05, paths=["a.md"])

    fake_ollama.set_response(RuntimeError("ollama timeout"))

    stats = rag.auto_harvest(
        since_days=30, confidence_below=0.5, min_judge_conf=0.8, limit=5,
    )
    assert stats["skipped_judge_failed"] == 1
    assert stats["judged_positive"] == 0


def test_auto_harvest_dry_run_does_not_insert(
    temp_db, tmp_vault, fake_ollama,
):
    conn, _ = temp_db
    (tmp_vault / "a.md").write_text("# A\n\n.\n")
    _insert_q(conn, ts="2026-04-23T10:00", q="dry run",
              top_score=0.05, paths=["a.md"])

    fake_ollama.set_response(json.dumps({
        "verdict": "a.md", "confidence": 0.95, "reason": "",
    }))

    stats = rag.auto_harvest(
        since_days=30, confidence_below=0.5, min_judge_conf=0.8,
        limit=5, dry_run=True,
    )
    # Stats still reflect what would have happened.
    assert stats["judged_positive"] == 1
    # But no rows were inserted.
    rows = conn.execute("SELECT COUNT(*) FROM rag_feedback").fetchone()
    assert rows[0] == 0


def test_auto_harvest_candidates_with_empty_paths_are_skipped(
    temp_db, tmp_vault, fake_ollama,
):
    conn, _ = temp_db
    _insert_q(conn, ts="2026-04-23T10:00", q="sin paths",
              top_score=0.05, paths=[])

    stats = rag.auto_harvest(
        since_days=30, confidence_below=0.5, min_judge_conf=0.8, limit=5,
    )
    assert stats["skipped_empty_paths"] == 1
    assert stats["judged_positive"] == 0


# ── feedback auto-harvest CLI ────────────────────────────────────────────


def test_cli_auto_harvest_json_output(temp_db, tmp_vault, fake_ollama):
    conn, _ = temp_db
    (tmp_vault / "a.md").write_text("# A\n\nContenido.\n")
    _insert_q(conn, ts="2026-04-23T10:00", q="test cli",
              top_score=0.05, paths=["a.md"])

    fake_ollama.set_response(json.dumps({
        "verdict": "a.md", "confidence": 0.9, "reason": "",
    }))

    runner = CliRunner()
    result = runner.invoke(rag.cli, [
        "feedback", "auto-harvest",
        "--since", "30", "--limit", "5",
        "--confidence-below", "0.5", "--min-judge-conf", "0.8",
        "--json",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip().splitlines()[-1])
    assert data["judged_positive"] == 1
    assert data["processed"] == 1


def test_cli_auto_harvest_empty_renders_summary(temp_db, tmp_vault):
    runner = CliRunner()
    result = runner.invoke(rag.cli, [
        "feedback", "auto-harvest",
        "--since", "30", "--limit", "5",
    ])
    assert result.exit_code == 0, result.output
    assert "Processed" in result.output


def test_cli_auto_harvest_dry_run_flag(
    temp_db, tmp_vault, fake_ollama,
):
    conn, _ = temp_db
    (tmp_vault / "a.md").write_text("# A\n\n.\n")
    _insert_q(conn, ts="2026-04-23T10:00", q="dry run cli",
              top_score=0.05, paths=["a.md"])

    fake_ollama.set_response(json.dumps({
        "verdict": "a.md", "confidence": 0.95, "reason": "",
    }))

    runner = CliRunner()
    result = runner.invoke(rag.cli, [
        "feedback", "auto-harvest",
        "--since", "30", "--limit", "5",
        "--confidence-below", "0.5", "--min-judge-conf", "0.8",
        "--dry-run",
    ])
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output.lower() or "(dry-run)" in result.output
    # No rows inserted.
    rows = conn.execute("SELECT COUNT(*) FROM rag_feedback").fetchone()
    assert rows[0] == 0


# ── plist registration ───────────────────────────────────────────────────


def test_auto_harvest_plist_is_registered_in_services_spec():
    spec = rag._services_spec("/usr/local/bin/rag")
    labels = [label for label, _, _ in spec]
    assert "com.fer.obsidian-rag-auto-harvest" in labels


def test_auto_harvest_plist_has_valid_xml():
    from xml.etree import ElementTree as ET
    content = rag._auto_harvest_plist("/usr/local/bin/rag")
    # plist parses as valid XML
    ET.fromstring(content)
    assert "feedback" in content
    assert "auto-harvest" in content
    assert "<integer>3</integer>" in content  # Hour=3
