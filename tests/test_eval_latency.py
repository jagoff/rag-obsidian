"""Tests for `rag eval --latency` P50/P95 reporting + --max-p95-ms gate."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from click.testing import CliRunner


@pytest.fixture
def fake_queries_yaml(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(
        """queries:
  - question: one
    expected: [a.md]
  - question: two
    expected: [b.md]
chains:
  - id: c1
    turns:
      - question: t1
        expected: [a.md]
      - question: t2
        expected: [b.md]
""",
        encoding="utf-8",
    )
    return path


def _install_stubs(monkeypatch, retrieve_delay_s=0.0):
    """Stub out heavy deps so eval runs deterministically in-memory."""
    import rag
    import time as _time

    class _FakeCol:
        def count(self):
            return 10

    monkeypatch.setattr(rag, "get_db", lambda: _FakeCol())

    def _fake_retrieve(col, q, k, **kw):
        if retrieve_delay_s:
            _time.sleep(retrieve_delay_s)
        # Return expected paths so hit@k stays nonzero (not asserted here)
        return {
            "metas": [
                {"file": "a.md"},
                {"file": "b.md"},
            ],
            "docs": ["", ""],
            "scores": [0.5, 0.4],
        }

    monkeypatch.setattr(rag, "retrieve", _fake_retrieve)
    monkeypatch.setattr(rag, "reformulate_query",
                        lambda q, hist, summary=None, seen_titles=None: q)
    monkeypatch.setattr(rag, "session_summary", lambda s: "")


def test_latency_flag_adds_table_and_snapshot(fake_queries_yaml, tmp_path, monkeypatch):
    import rag
    import sqlite3
    _install_stubs(monkeypatch)
    eval_log = tmp_path / "eval.jsonl"
    monkeypatch.setattr(rag, "EVAL_LOG_PATH", eval_log)
    # Post-T10: eval log writes to rag_eval_runs (SQL).
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)

    runner = CliRunner()
    res = runner.invoke(rag.eval, ["--file", str(fake_queries_yaml), "--latency"])
    assert res.exit_code == 0, res.output
    assert "Latencia retrieve()" in res.output
    assert "singles" in res.output
    assert "chains" in res.output

    # Trend log should have an extra_json entry with `latency` inside.
    conn = sqlite3.connect(str(tmp_path / rag._TELEMETRY_DB_FILENAME))
    conn.row_factory = sqlite3.Row
    try:
        rows = list(conn.execute(
            "SELECT extra_json FROM rag_eval_runs ORDER BY id"
        ).fetchall())
    finally:
        conn.close()
    assert rows, "no row written to rag_eval_runs"
    extra = json.loads(rows[-1]["extra_json"] or "{}")
    assert "latency" in extra
    assert "singles" in extra["latency"]
    assert extra["latency"]["singles"]["n"] == 2
    for fld in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms"):
        assert fld in extra["latency"]["singles"]


def test_no_latency_flag_omits_table(fake_queries_yaml, tmp_path, monkeypatch):
    import rag
    _install_stubs(monkeypatch)
    monkeypatch.setattr(rag, "EVAL_LOG_PATH", tmp_path / "eval.jsonl")

    runner = CliRunner()
    res = runner.invoke(rag.eval, ["--file", str(fake_queries_yaml)])
    assert res.exit_code == 0, res.output
    assert "Latencia retrieve()" not in res.output


def test_gate_passes_when_under_threshold(fake_queries_yaml, tmp_path, monkeypatch):
    import rag
    _install_stubs(monkeypatch, retrieve_delay_s=0.001)
    monkeypatch.setattr(rag, "EVAL_LOG_PATH", tmp_path / "eval.jsonl")

    runner = CliRunner()
    res = runner.invoke(rag.eval, [
        "--file", str(fake_queries_yaml),
        "--max-p95-ms", "10000",
    ])
    assert res.exit_code == 0, res.output
    assert "GATE FAIL" not in res.output


def test_gate_fails_when_over_threshold(fake_queries_yaml, tmp_path, monkeypatch):
    import rag
    _install_stubs(monkeypatch, retrieve_delay_s=0.03)  # 30ms per call
    monkeypatch.setattr(rag, "EVAL_LOG_PATH", tmp_path / "eval.jsonl")

    runner = CliRunner()
    res = runner.invoke(rag.eval, [
        "--file", str(fake_queries_yaml),
        "--max-p95-ms", "5",  # very low → must fail
    ])
    assert res.exit_code == 1, res.output
    assert "GATE FAIL" in res.output


def test_max_p95_implies_latency(fake_queries_yaml, tmp_path, monkeypatch):
    """Passing --max-p95-ms alone should turn on --latency automatically."""
    import rag
    _install_stubs(monkeypatch)
    monkeypatch.setattr(rag, "EVAL_LOG_PATH", tmp_path / "eval.jsonl")

    runner = CliRunner()
    res = runner.invoke(rag.eval, [
        "--file", str(fake_queries_yaml),
        "--max-p95-ms", "99999",
    ])
    assert res.exit_code == 0, res.output
    assert "Latencia retrieve()" in res.output
