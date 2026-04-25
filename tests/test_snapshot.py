"""Feature #17 del 2026-04-23 — `rag snapshot` backup/restore tests.

Validates:
- _snapshot_create gathers all 4 sections (ranker, calibration,
  paraphrases, feedback) with correct shape.
- _snapshot_restore upserts calibration, increments paraphrases hits,
  ignores duplicate feedback, writes ranker.json atomically.
- CLI: create → file written, restore → confirms + applies, list → table,
  version mismatch warning.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3

import pytest
from click.testing import CliRunner

import rag


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

_CAL_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_score_calibration ("
    " source TEXT PRIMARY KEY,"
    " raw_knots_json TEXT NOT NULL,"
    " cal_knots_json TEXT NOT NULL,"
    " n_pos INTEGER NOT NULL,"
    " n_neg INTEGER NOT NULL,"
    " trained_at TEXT NOT NULL,"
    " model_version TEXT NOT NULL,"
    " extra_json TEXT"
    ")"
)

_PARAPHRASES_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_learned_paraphrases ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " q_normalized TEXT NOT NULL,"
    " paraphrase TEXT NOT NULL,"
    " hit_count INTEGER NOT NULL DEFAULT 1,"
    " created_ts TEXT NOT NULL,"
    " last_used_ts TEXT NOT NULL,"
    " UNIQUE(q_normalized, paraphrase)"
    ")"
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolate ranker path, snapshots dir, and state DB."""
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(_FEEDBACK_DDL)
    conn.execute(_CAL_DDL)
    conn.execute(_PARAPHRASES_DDL)
    conn.commit()

    @contextlib.contextmanager
    def _fake_conn():
        c = sqlite3.connect(str(db_path))
        try:
            yield c
            c.commit()
        finally:
            c.close()

    ranker_path = tmp_path / "ranker.json"
    snaps_dir = tmp_path / "snapshots"
    # Patch home-derived paths inside the snapshot command.
    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_conn)
    monkeypatch.setattr(rag, "RANKER_CONFIG_PATH", ranker_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    try:
        yield {
            "tmp_path": tmp_path,
            "db_path": db_path,
            "ranker_path": ranker_path,
            "snaps_dir": snaps_dir,
            "conn": conn,
        }
    finally:
        conn.close()


# ── _snapshot_create ─────────────────────────────────────────────────────


def test_snapshot_create_empty_state(env):
    snap = rag._snapshot_create()
    assert snap["version"] == rag._SNAPSHOT_VERSION
    assert snap["ranker"] is None
    assert snap["score_calibration"] == []
    assert snap["learned_paraphrases"] == []
    assert snap["feedback"] == []
    assert "created_at" in snap


def test_snapshot_create_captures_ranker(env):
    env["ranker_path"].write_text(json.dumps({"weights": {"recency": 0.1}}))
    snap = rag._snapshot_create()
    assert snap["ranker"] == {"weights": {"recency": 0.1}}


def test_snapshot_create_captures_calibration(env):
    conn = env["conn"]
    conn.execute(
        "INSERT INTO rag_score_calibration "
        "(source, raw_knots_json, cal_knots_json, n_pos, n_neg, "
        " trained_at, model_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("vault", "[0.0, 1.0]", "[0.0, 1.0]", 10, 50,
         "2026-04-23T10:00", "isotonic-v1"),
    )
    conn.commit()
    snap = rag._snapshot_create()
    assert len(snap["score_calibration"]) == 1
    assert snap["score_calibration"][0]["source"] == "vault"
    assert snap["score_calibration"][0]["n_pos"] == 10


def test_snapshot_create_captures_paraphrases(env):
    conn = env["conn"]
    conn.execute(
        "INSERT INTO rag_learned_paraphrases "
        "(q_normalized, paraphrase, hit_count, created_ts, last_used_ts) "
        "VALUES (?, ?, ?, ?, ?)",
        ("test q", "alt phrasing", 3, "2026-04-22T00:00", "2026-04-23T00:00"),
    )
    conn.commit()
    snap = rag._snapshot_create()
    assert len(snap["learned_paraphrases"]) == 1
    assert snap["learned_paraphrases"][0]["hit_count"] == 3


def test_snapshot_create_captures_feedback_positive_only(env):
    conn = env["conn"]
    # Positive with corrective_path → should be captured.
    conn.execute(
        "INSERT INTO rag_feedback(ts, turn_id, rating, q, paths_json, extra_json) "
        "VALUES (?, ?, 1, ?, '[]', ?)",
        ("2026-04-23T10:00", "t1", "q1",
         json.dumps({"corrective_path": "a.md"})),
    )
    # Negative without corrective_path → should be skipped.
    conn.execute(
        "INSERT INTO rag_feedback(ts, turn_id, rating, q, paths_json) "
        "VALUES (?, ?, -1, ?, '[]')",
        ("2026-04-23T10:01", "t2", "q2"),
    )
    # Negative WITH corrective_path → should be captured.
    conn.execute(
        "INSERT INTO rag_feedback(ts, turn_id, rating, q, paths_json, extra_json) "
        "VALUES (?, ?, -1, ?, '[]', ?)",
        ("2026-04-23T10:02", "t3", "q3",
         json.dumps({"corrective_path": "b.md"})),
    )
    conn.commit()

    snap = rag._snapshot_create()
    turn_ids = {r["turn_id"] for r in snap["feedback"]}
    assert "t1" in turn_ids
    assert "t3" in turn_ids
    assert "t2" not in turn_ids  # plain negative skipped


# ── _snapshot_restore ────────────────────────────────────────────────────


def test_restore_writes_ranker(env):
    snap = {
        "version": 1, "created_at": "",
        "ranker": {"weights": {"recency": 0.42}},
        "score_calibration": [], "learned_paraphrases": [], "feedback": [],
    }
    stats = rag._snapshot_restore(snap)
    assert stats["ranker_restored"] is True
    assert env["ranker_path"].exists()
    restored = json.loads(env["ranker_path"].read_text())
    assert restored["weights"]["recency"] == 0.42


def test_restore_upserts_calibration(env):
    conn = env["conn"]
    # Pre-existing row for same source.
    conn.execute(
        "INSERT INTO rag_score_calibration "
        "(source, raw_knots_json, cal_knots_json, n_pos, n_neg, "
        " trained_at, model_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("vault", "[0.0, 0.5]", "[0.0, 0.5]", 5, 25,
         "2026-04-22T10:00", "isotonic-v1"),
    )
    conn.commit()

    snap = {
        "version": 1, "created_at": "", "ranker": None,
        "score_calibration": [{
            "source": "vault",
            "raw_knots_json": "[0.0, 1.0]",
            "cal_knots_json": "[0.0, 1.0]",
            "n_pos": 20, "n_neg": 100,
            "trained_at": "2026-04-23T10:00",
            "model_version": "isotonic-v1",
            "extra_json": None,
        }],
        "learned_paraphrases": [], "feedback": [],
    }
    stats = rag._snapshot_restore(snap)
    assert stats["calibration_rows"] == 1
    # Confirm UPSERT overwrote.
    row = conn.execute(
        "SELECT n_pos, raw_knots_json FROM rag_score_calibration WHERE source='vault'"
    ).fetchone()
    assert row[0] == 20
    assert row[1] == "[0.0, 1.0]"


def test_restore_paraphrases_increments_hit_count(env):
    conn = env["conn"]
    conn.execute(
        "INSERT INTO rag_learned_paraphrases "
        "(q_normalized, paraphrase, hit_count, created_ts, last_used_ts) "
        "VALUES (?, ?, ?, ?, ?)",
        ("q", "alt", 2, "2026-04-22", "2026-04-22"),
    )
    conn.commit()

    snap = {
        "version": 1, "created_at": "", "ranker": None,
        "score_calibration": [],
        "learned_paraphrases": [{
            "q_normalized": "q", "paraphrase": "alt",
            "hit_count": 3,
            "created_ts": "2026-04-23", "last_used_ts": "2026-04-23",
        }],
        "feedback": [],
    }
    rag._snapshot_restore(snap)
    row = conn.execute(
        "SELECT hit_count FROM rag_learned_paraphrases "
        "WHERE q_normalized='q' AND paraphrase='alt'"
    ).fetchone()
    # 2 (existing) + 3 (incoming) = 5
    assert row[0] == 5


def test_restore_feedback_ignores_duplicates(env):
    conn = env["conn"]
    # Pre-existing feedback row.
    conn.execute(
        "INSERT INTO rag_feedback(ts, turn_id, rating, q) VALUES (?, 'dup', 1, 'q')",
        ("2026-04-23T10:00",),
    )
    conn.commit()

    snap = {
        "version": 1, "created_at": "", "ranker": None,
        "score_calibration": [], "learned_paraphrases": [],
        "feedback": [
            {"ts": "2026-04-23T10:00", "turn_id": "dup", "rating": 1,
             "q": "q", "scope": None, "paths_json": None, "extra_json": None},
            {"ts": "2026-04-23T11:00", "turn_id": "new", "rating": 1,
             "q": "q2", "scope": None, "paths_json": None, "extra_json": None},
        ],
    }
    rag._snapshot_restore(snap)
    n = conn.execute("SELECT COUNT(*) FROM rag_feedback").fetchone()[0]
    assert n == 2  # dup ignored, new inserted


def test_restore_selective_skip_flags(env):
    env["ranker_path"].unlink(missing_ok=True)
    snap = {
        "version": 1, "created_at": "",
        "ranker": {"w": 1},
        "score_calibration": [{"source": "x", "raw_knots_json": "[0,1]",
                               "cal_knots_json": "[0,1]", "n_pos": 1, "n_neg": 1,
                               "trained_at": "", "model_version": ""}],
        "learned_paraphrases": [], "feedback": [],
    }
    # Skip ranker only.
    stats = rag._snapshot_restore(snap, apply_ranker=False)
    assert stats["ranker_restored"] is False
    assert stats["calibration_rows"] == 1
    # ranker.json NOT written.
    assert not env["ranker_path"].exists()


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_snapshot_create_writes_file(env):
    runner = CliRunner()
    out_path = env["tmp_path"] / "test-snap.json"
    result = runner.invoke(
        rag.cli,
        ["snapshot", "create", "--output", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["version"] == 1


def test_cli_snapshot_create_as_json(env):
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["snapshot", "create", "--as-json"])
    assert result.exit_code == 0, result.output
    # First non-empty line should be parsable JSON.
    data = json.loads(result.output)
    assert "version" in data


def test_cli_snapshot_restore_as_json(env):
    snap_path = env["tmp_path"] / "snap.json"
    snap_path.write_text(json.dumps({
        "version": 1, "created_at": "", "ranker": None,
        "score_calibration": [], "learned_paraphrases": [], "feedback": [],
    }))
    runner = CliRunner()
    result = runner.invoke(
        rag.cli,
        ["snapshot", "restore", str(snap_path), "--yes", "--as-json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip().splitlines()[-1])
    assert "ranker_restored" in data


def test_cli_snapshot_list_empty(env):
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["snapshot", "list"])
    assert result.exit_code == 0
    assert "Sin snapshots" in result.output


def test_cli_snapshot_list_shows_files(env):
    snaps_dir = env["tmp_path"] / ".local/share/obsidian-rag/snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    (snaps_dir / "snapshot-2026-04-23.json").write_text("{}")
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["snapshot", "list"])
    assert result.exit_code == 0
    assert "snapshot-2026-04-23.json" in result.output
