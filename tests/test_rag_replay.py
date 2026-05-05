"""Tests for rag replay command — Sprint 3 Tarea B.

Covers:
- _replay_load_row: SQL loader + JSON parsing
- _replay_cosine: edge cases
- _replay_query_row: shape, empty_q, corpus_drift gating, env isolation
- bulk mode: since/limit/filter_cmd, summary line
- failure modes: missing ID (exit 2), empty q (exit 2), corpus drift (exit 3)
- determinism: two runs of the same fixture give the same verdict
- --explain mode: exit 0, no comparison
- --skip-gen: path-only comparison
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import rag as _rag
from rag import _replay_cosine, _replay_load_row, _replay_query_row, cli


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_db_path(tmp_path: Path) -> Generator:
    """Redirect DB_PATH so replay touches a test DB, not production."""
    snap = _rag.DB_PATH
    _rag.DB_PATH = tmp_path / "ragvec"
    _rag.DB_PATH.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        _rag.DB_PATH = snap


@pytest.fixture()
def telemetry_db(tmp_path: Path) -> Path:
    """Create a minimal telemetry.db with rag_queries populated."""
    db_dir = tmp_path / "ragvec"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "telemetry.db"

    # Patch DB_PATH to point here
    _rag.DB_PATH = tmp_path / "ragvec"

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            cmd TEXT,
            q TEXT,
            session TEXT,
            mode TEXT,
            top_score REAL,
            t_retrieve REAL,
            t_gen REAL,
            answer_len INTEGER,
            citation_repaired INTEGER,
            critique_fired INTEGER,
            critique_changed INTEGER,
            variants_json TEXT,
            paths_json TEXT,
            scores_json TEXT,
            filters_json TEXT,
            bad_citations_json TEXT,
            extra_json TEXT,
            trace_id TEXT
        )
    """)
    conn.execute("""
        INSERT INTO rag_queries (
            ts, cmd, q, session, mode, top_score, t_retrieve, t_gen,
            answer_len, citation_repaired, paths_json, scores_json,
            filters_json, extra_json
        ) VALUES (
            '2026-05-01T10:00:00', 'cli', 'que es coaching', 'sess1',
            'query', 0.85, 1200, 900, 200, 0,
            '["01-Projects/Coaching/intro.md","02-Areas/Personal/reflexion.md"]',
            '[0.85,0.72]',
            '{}',
            '{"intent":"semantic","corpus_hash":"abc123def456"}'
        )
    """)
    # Row with empty q
    conn.execute("""
        INSERT INTO rag_queries (ts, cmd, q, paths_json, extra_json)
        VALUES ('2026-05-01T11:00:00', 'cli', '', '[]', '{}')
    """)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def sample_row(telemetry_db: Path) -> dict:
    """Load the first test row from telemetry_db."""
    conn = sqlite3.connect(str(telemetry_db))
    cur = conn.execute("SELECT * FROM rag_queries WHERE q != '' LIMIT 1")
    cols = [d[0] for d in cur.description]
    row = dict(zip(cols, cur.fetchone()))
    conn.close()
    row["paths_json"] = json.loads(row["paths_json"])
    row["extra_json"] = json.loads(row["extra_json"])
    row["filters_json"] = json.loads(row["filters_json"]) if row.get("filters_json") else {}
    return row


# ─── _replay_cosine ───────────────────────────────────────────────────────────


def test_replay_cosine_identical():
    v = [1.0, 0.0, 0.5]
    assert _replay_cosine(v, v) == pytest.approx(1.0, abs=1e-6)


def test_replay_cosine_orthogonal():
    assert _replay_cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)


def test_replay_cosine_empty_returns_zero():
    assert _replay_cosine([], [1.0]) == 0.0
    assert _replay_cosine([1.0], []) == 0.0


def test_replay_cosine_mismatched_len_returns_zero():
    assert _replay_cosine([1.0, 2.0], [1.0]) == 0.0


def test_replay_cosine_zero_vector_returns_zero():
    assert _replay_cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


# ─── _replay_load_row ─────────────────────────────────────────────────────────


def test_replay_load_row_returns_none_for_missing_id(telemetry_db: Path):
    result = _replay_load_row(99999)
    assert result is None


def test_replay_load_row_returns_dict_for_existing_id(telemetry_db: Path):
    result = _replay_load_row(1)
    assert result is not None
    assert result["q"] == "que es coaching"
    assert isinstance(result.get("paths_json"), list)
    assert isinstance(result.get("extra_json"), dict)


def test_replay_load_row_parses_filters_json(telemetry_db: Path):
    result = _replay_load_row(1)
    assert result is not None
    # filters_json was '{}', should be parsed to dict
    assert isinstance(result.get("filters_json"), dict)


# ─── _replay_query_row ────────────────────────────────────────────────────────


def test_replay_query_row_empty_q_returns_regression():
    row = {"id": 99, "ts": "2026-05-01", "q": "", "cmd": "cli",
           "paths_json": [], "extra_json": {}, "filters_json": {}}
    result = _replay_query_row(row, skip_gen=True)
    assert result["verdict"] == "regression"
    assert result["error"] == "empty_q"


def test_replay_query_row_corpus_drift_without_force_returns_exit(sample_row: dict):
    """When corpus_hash mismatches and --force is False, return corpus_drift error."""
    sample_row["extra_json"]["corpus_hash"] = "deadbeef00000000"  # wrong hash

    mock_col = MagicMock()
    with patch.object(_rag, "_compute_corpus_hash", return_value="different_hash"):
        with patch.object(_rag, "get_db", return_value=mock_col):
            result = _replay_query_row(sample_row, skip_gen=True, force=False)

    assert result["corpus_drift"] is True
    assert result["error"] == "corpus_drift"
    assert result["verdict"] == "regression"


def test_replay_query_row_corpus_drift_with_force_continues(sample_row: dict):
    """With --force, corpus drift is noted but retrieve still runs."""
    sample_row["extra_json"]["corpus_hash"] = "deadbeef00000000"

    mock_rr = MagicMock()
    mock_rr.metas = []
    mock_rr.docs = []
    mock_rr.top_score = None

    mock_col = MagicMock()
    with patch.object(_rag, "_compute_corpus_hash", return_value="different_hash"):
        with patch.object(_rag, "get_db", return_value=mock_col):
            with patch.object(_rag, "multi_retrieve", return_value=mock_rr):
                result = _replay_query_row(sample_row, skip_gen=True, force=True)

    assert result["corpus_drift"] is True
    assert result["error"] is None  # no error from corpus drift when force=True
    # verdict may be path_drift (empty new paths vs orig paths) or equivalent
    assert result["verdict"] in ("equivalent", "path_drift", "response_drift", "regression")


def test_replay_query_row_result_shape(sample_row: dict):
    """Result always has all required keys."""
    mock_rr = MagicMock()
    mock_rr.metas = [{"file": "01-Projects/Coaching/intro.md"}]
    mock_rr.docs = ["some doc"]
    mock_rr.top_score = 0.85

    sample_row["extra_json"].pop("corpus_hash", None)

    with patch.object(_rag, "multi_retrieve", return_value=mock_rr):
        result = _replay_query_row(sample_row, skip_gen=True)

    required_keys = {
        "query_id", "ts", "q", "cmd",
        "verdict", "path_jaccard", "top3_changed",
        "response_cosine", "response_hash_match",
        "corpus_drift", "new_paths", "new_top_score", "error",
    }
    assert required_keys == required_keys & set(result.keys())


def test_replay_query_row_env_isolation(sample_row: dict):
    """RAG_EXPLORE must be scrubbed; RAG_SKIP_BEHAVIOR_LOG must be set during run."""
    os.environ["RAG_EXPLORE"] = "1"
    seen_env: dict = {}

    def fake_multi_retrieve(*args, **kwargs):
        seen_env["RAG_EXPLORE"] = os.environ.get("RAG_EXPLORE", "ABSENT")
        seen_env["RAG_SKIP_BEHAVIOR_LOG"] = os.environ.get("RAG_SKIP_BEHAVIOR_LOG", "ABSENT")
        mock_rr = MagicMock()
        mock_rr.metas = []
        mock_rr.docs = []
        mock_rr.top_score = None
        return mock_rr

    sample_row["extra_json"].pop("corpus_hash", None)

    with patch.object(_rag, "multi_retrieve", side_effect=fake_multi_retrieve):
        _replay_query_row(sample_row, skip_gen=True)

    # During retrieve, RAG_EXPLORE must be absent
    assert seen_env.get("RAG_EXPLORE") == "ABSENT"
    # RAG_SKIP_BEHAVIOR_LOG must be set
    assert seen_env.get("RAG_SKIP_BEHAVIOR_LOG") == "1"
    # After return, RAG_EXPLORE must be restored
    assert os.environ.get("RAG_EXPLORE") == "1"
    del os.environ["RAG_EXPLORE"]


def test_replay_query_row_path_jaccard_identical_paths(sample_row: dict):
    """When new paths == original paths, jaccard = 1.0 and verdict = equivalent."""
    orig_paths = sample_row["paths_json"]

    mock_rr = MagicMock()
    mock_rr.metas = [{"file": p} for p in orig_paths]
    mock_rr.docs = []
    mock_rr.top_score = 0.8

    sample_row["extra_json"].pop("corpus_hash", None)

    with patch.object(_rag, "multi_retrieve", return_value=mock_rr):
        result = _replay_query_row(sample_row, skip_gen=True)

    assert result["path_jaccard"] == pytest.approx(1.0)
    assert result["verdict"] == "equivalent"


def test_replay_query_row_path_jaccard_divergent_paths(sample_row: dict):
    """When new paths are completely different, jaccard = 0.0, verdict = path_drift."""
    mock_rr = MagicMock()
    mock_rr.metas = [
        {"file": "99-Archive/old.md"},
        {"file": "03-Resources/tech/git.md"},
    ]
    mock_rr.docs = []
    mock_rr.top_score = 0.5

    sample_row["extra_json"].pop("corpus_hash", None)

    with patch.object(_rag, "multi_retrieve", return_value=mock_rr):
        result = _replay_query_row(sample_row, skip_gen=True)

    assert result["path_jaccard"] == pytest.approx(0.0)
    assert result["verdict"] == "path_drift"


def test_replay_query_row_determinism(sample_row: dict):
    """Two runs of the same row with identical mocks give the same verdict."""
    mock_rr = MagicMock()
    mock_rr.metas = [{"file": "01-Projects/Coaching/intro.md"}]
    mock_rr.docs = []
    mock_rr.top_score = 0.85

    sample_row["extra_json"].pop("corpus_hash", None)

    with patch.object(_rag, "multi_retrieve", return_value=mock_rr):
        r1 = _replay_query_row(sample_row, skip_gen=True)

    with patch.object(_rag, "multi_retrieve", return_value=mock_rr):
        r2 = _replay_query_row(sample_row, skip_gen=True)

    assert r1["verdict"] == r2["verdict"]
    assert r1["path_jaccard"] == r2["path_jaccard"]
    assert r1["top3_changed"] == r2["top3_changed"]


# ─── CLI: replay command ──────────────────────────────────────────────────────


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_replay_cli_missing_id_exits_2(runner: CliRunner, telemetry_db: Path):
    result = runner.invoke(cli, ["replay"])
    assert result.exit_code == 2


def test_replay_cli_nonexistent_id_exits_2(runner: CliRunner, telemetry_db: Path):
    result = runner.invoke(cli, ["replay", "99999", "--plain"])
    assert result.exit_code == 2
    assert "not_found" in result.output


def test_replay_cli_empty_q_row_exits_2(runner: CliRunner, telemetry_db: Path):
    # Row 2 has empty q
    result = runner.invoke(cli, ["replay", "2", "--plain"])
    assert result.exit_code == 2


def test_replay_cli_explain_exits_0(runner: CliRunner, telemetry_db: Path):
    mock_rr = MagicMock()
    mock_rr.metas = [{"file": "01-Projects/Coaching/intro.md"}]
    mock_rr.docs = []
    mock_rr.top_score = 0.85

    with patch.object(_rag, "multi_retrieve", return_value=mock_rr):
        with patch.object(_rag, "_compute_corpus_hash", return_value="abc123def456"):
            with patch.object(_rag, "get_db", return_value=MagicMock()):
                result = runner.invoke(cli, ["replay", "1", "--explain", "--plain"])
    assert result.exit_code == 0


def test_replay_cli_diff_equivalent_exits_0(runner: CliRunner, telemetry_db: Path):
    orig_paths = ["01-Projects/Coaching/intro.md", "02-Areas/Personal/reflexion.md"]
    mock_rr = MagicMock()
    mock_rr.metas = [{"file": p} for p in orig_paths]
    mock_rr.docs = []
    mock_rr.top_score = 0.85

    with patch.object(_rag, "multi_retrieve", return_value=mock_rr):
        with patch.object(_rag, "_compute_corpus_hash", return_value="abc123def456"):
            with patch.object(_rag, "get_db", return_value=MagicMock()):
                result = runner.invoke(
                    cli, ["replay", "1", "--skip-gen", "--plain"]
                )
    assert result.exit_code == 0
    assert "equivalent" in result.output


def test_replay_cli_diff_path_drift_exits_1(runner: CliRunner, telemetry_db: Path):
    mock_rr = MagicMock()
    mock_rr.metas = [{"file": "99-Archive/something_else.md"}]
    mock_rr.docs = []
    mock_rr.top_score = 0.3

    with patch.object(_rag, "multi_retrieve", return_value=mock_rr):
        with patch.object(_rag, "_compute_corpus_hash", return_value="abc123def456"):
            with patch.object(_rag, "get_db", return_value=MagicMock()):
                result = runner.invoke(
                    cli, ["replay", "1", "--skip-gen", "--plain"]
                )
    assert result.exit_code == 1
    assert "path_drift" in result.output


def test_replay_cli_corpus_drift_exits_3(runner: CliRunner, telemetry_db: Path):
    with patch.object(_rag, "_compute_corpus_hash", return_value="totally_different"):
        with patch.object(_rag, "get_db", return_value=MagicMock()):
            result = runner.invoke(
                cli, ["replay", "1", "--skip-gen", "--plain"]
            )
    assert result.exit_code == 3
    assert "corpus_drift" in result.output


def test_replay_cli_bulk_empty_window(runner: CliRunner, telemetry_db: Path):
    # Use a far-future window by passing an invalid since — exception path → no filter
    # but filter_cmd=nonexistent_cmd2 ensures zero rows
    result = runner.invoke(
        cli, ["replay", "--bulk", "--since", "30d",
              "--filter-cmd", "nonexistent_cmd_xyz", "--plain"]
    )
    # Should exit 0 with "no queries found"
    assert result.exit_code == 0
    assert "no queries found" in result.output


def test_replay_cli_bulk_runs_and_summarizes(runner: CliRunner, telemetry_db: Path):
    orig_paths = ["01-Projects/Coaching/intro.md", "02-Areas/Personal/reflexion.md"]
    mock_rr = MagicMock()
    mock_rr.metas = [{"file": p} for p in orig_paths]
    mock_rr.docs = []
    mock_rr.top_score = 0.85

    with patch.object(_rag, "multi_retrieve", return_value=mock_rr):
        with patch.object(_rag, "_compute_corpus_hash", return_value="abc123def456"):
            with patch.object(_rag, "get_db", return_value=MagicMock()):
                result = runner.invoke(
                    cli, ["replay", "--bulk", "--since", "30d",
                          "--skip-gen", "--plain"]
                )
    # Should emit summary line
    assert "total=" in result.output
    assert "regressions=" in result.output


def test_replay_cli_bulk_filter_cmd(runner: CliRunner, telemetry_db: Path):
    """--filter-cmd 'nonexistent' should return 0 rows."""
    result = runner.invoke(
        cli, ["replay", "--bulk", "--since", "30d",
              "--filter-cmd", "nonexistent_cmd", "--plain"]
    )
    assert result.exit_code == 0
    assert "no queries found" in result.output


def test_replay_cli_json_output(runner: CliRunner, telemetry_db: Path):
    orig_paths = ["01-Projects/Coaching/intro.md", "02-Areas/Personal/reflexion.md"]
    mock_rr = MagicMock()
    mock_rr.metas = [{"file": p} for p in orig_paths]
    mock_rr.docs = []
    mock_rr.top_score = 0.85

    with patch.object(_rag, "multi_retrieve", return_value=mock_rr):
        with patch.object(_rag, "_compute_corpus_hash", return_value="abc123def456"):
            with patch.object(_rag, "get_db", return_value=MagicMock()):
                result = runner.invoke(
                    cli, ["replay", "1", "--skip-gen", "--json"]
                )
    # Should be parseable JSON
    data = json.loads(result.output)
    assert "verdict" in data
    assert "path_jaccard" in data
    assert "new_paths" in data
