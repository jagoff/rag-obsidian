"""Tests for scripts/migrate_state_to_sqlite.py.

Fixtures build tiny JSONL/JSON files in tmp_path, invoke the migrator's
`main()` with `--source-dir tmp_path`, and assert against the resulting
rows in `tmp_path/ragvec/ragvec.db`.
"""
from __future__ import annotations

import json
import sqlite3
import struct
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure the scripts/ module is importable
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import migrate_state_to_sqlite as mig  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def source_dir(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    return d


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _db_conn(source_dir: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(source_dir / "ragvec" / "ragvec.db"))


# ── Tests ────────────────────────────────────────────────────────────────────

def test_dry_run_no_writes(source_dir):
    q = source_dir / "queries.jsonl"
    _write_jsonl(q, [
        {"ts": "2026-04-19T12:00:00", "q": "hola", "cmd": "query"},
        {"ts": "2026-04-19T12:01:00", "q": "mundo", "cmd": "query"},
    ])
    rc = mig.main(["--dry-run", "--source-dir", str(source_dir), "--force"])
    assert rc == 0
    # Source still exists (no rename)
    assert q.is_file()
    # DB created but empty
    db = source_dir / "ragvec" / "ragvec.db"
    assert db.is_file()
    conn = sqlite3.connect(str(db))
    try:
        cnt = conn.execute("SELECT COUNT(*) FROM rag_queries").fetchone()[0]
        assert cnt == 0
    finally:
        conn.close()


def test_migrate_queries_jsonl(source_dir):
    q = source_dir / "queries.jsonl"
    _write_jsonl(q, [
        {
            "ts": "2026-04-19T12:00:00",
            "cmd": "query",
            "q": "qué dice el vault",
            "session": "wa:5493425153999",
            "mode": "strict",
            "top_score": 0.42,
            "t_retrieve": 0.15,
            "t_gen": 1.8,
            "answer_len": 420,
            "variants": ["v1", "v2"],
            "paths": ["00-Inbox/a.md", "01-Projects/b.md"],
            "scores": [0.42, 0.31],
            "filters": {"folder": "01-Projects"},
            "bad_citations": [],
        },
    ])
    rc = mig.main(["--source-dir", str(source_dir), "--only", "queries", "--force"])
    assert rc == 0
    # Source renamed to .bak.<ts>
    assert not q.is_file()
    baks = list(source_dir.glob("queries.jsonl.bak.*"))
    assert len(baks) == 1
    # Row inserted with correct mapping
    conn = _db_conn(source_dir)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM rag_queries WHERE q = ?", ("qué dice el vault",),
        ).fetchone()
        assert row is not None
        assert row["cmd"] == "query"
        assert row["session"] == "wa:5493425153999"
        assert abs(row["top_score"] - 0.42) < 1e-9
        assert row["answer_len"] == 420
        assert json.loads(row["variants_json"]) == ["v1", "v2"]
        assert json.loads(row["paths_json"]) == ["00-Inbox/a.md", "01-Projects/b.md"]
        assert json.loads(row["filters_json"]) == {"folder": "01-Projects"}
    finally:
        conn.close()


def test_migrate_feedback_idempotent(source_dir):
    f = source_dir / "feedback.jsonl"
    recs = [
        {"ts": "2026-04-19T12:00:00", "turn_id": "abc123",
         "rating": 1, "q": "hola", "paths": ["a.md"]},
        # Exact duplicate (UNIQUE on turn_id, rating, ts)
        {"ts": "2026-04-19T12:00:00", "turn_id": "abc123",
         "rating": 1, "q": "hola", "paths": ["a.md"]},
        # Different ts → allowed
        {"ts": "2026-04-19T13:00:00", "turn_id": "abc123",
         "rating": 1, "q": "hola", "paths": ["a.md"]},
    ]
    _write_jsonl(f, recs)
    rc = mig.main(["--source-dir", str(source_dir), "--only", "feedback", "--force"])
    assert rc == 0
    conn = _db_conn(source_dir)
    try:
        cnt = conn.execute("SELECT COUNT(*) FROM rag_feedback").fetchone()[0]
        # 3 records, 1 dup suppressed → 2 rows
        assert cnt == 2
    finally:
        conn.close()

    # Re-run (source renamed on first run, so second run is a no-op via skip)
    rc = mig.main(["--source-dir", str(source_dir), "--only", "feedback", "--force"])
    assert rc == 0
    conn = _db_conn(source_dir)
    try:
        cnt = conn.execute("SELECT COUNT(*) FROM rag_feedback").fetchone()[0]
        assert cnt == 2
    finally:
        conn.close()


def test_migrate_feedback_golden_embeddings(source_dir):
    # Need rag_feedback to exist so MAX(ts) returns something
    _write_jsonl(source_dir / "feedback.jsonl", [
        {"ts": "2026-04-19T10:00:00", "turn_id": "t1", "rating": 1,
         "q": "x", "paths": ["p.md"]},
    ])
    # feedback_golden with 1024-dim vectors on 2 paths
    import random
    rnd = random.Random(42)
    emb1 = [rnd.random() for _ in range(1024)]
    emb2 = [rnd.random() - 0.5 for _ in range(1024)]
    golden = {
        "positives": [{"q": "query one", "emb": emb1, "paths": ["pos1.md"]}],
        "negatives": [{"q": "query two", "emb": emb2, "paths": ["neg1.md"]}],
    }
    (source_dir / "feedback_golden.json").write_text(
        json.dumps(golden), encoding="utf-8",
    )
    rc = mig.main([
        "--source-dir", str(source_dir),
        "--only", "feedback,feedback_golden",
        "--force",
    ])
    assert rc == 0
    conn = _db_conn(source_dir)
    try:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute(
            "SELECT path, rating, embedding, dim, source_ts FROM rag_feedback_golden "
            "ORDER BY rating DESC"
        ))
        assert len(rows) == 2
        # positive row
        pos = [r for r in rows if r["rating"] == 1][0]
        assert pos["path"] == "pos1.md"
        assert pos["dim"] == 1024
        assert len(pos["embedding"]) == 4 * 1024
        unpacked = list(struct.unpack("<" + "f" * 1024, pos["embedding"]))
        assert unpacked[0] == pytest.approx(emb1[0], rel=1e-6)
        assert unpacked[-1] == pytest.approx(emb1[-1], rel=1e-6)
        # source_ts should come from rag_feedback max(ts)
        assert pos["source_ts"] == "2026-04-19T10:00:00"
        # Meta row
        meta = conn.execute(
            "SELECT v FROM rag_feedback_golden_meta WHERE k = 'last_built_source_ts'"
        ).fetchone()
        assert meta[0] == "2026-04-19T10:00:00"
    finally:
        conn.close()


def test_migrate_ambient_state_latest_wins(source_dir):
    f = source_dir / "ambient_state.jsonl"
    _write_jsonl(f, [
        {"path": "note.md", "hash": "h1", "analyzed_at": 1.0},
        {"path": "note.md", "hash": "h2", "analyzed_at": 2.0},
        {"path": "note.md", "hash": "h3", "analyzed_at": 3.0},
    ])
    rc = mig.main(["--source-dir", str(source_dir), "--only", "ambient_state", "--force"])
    assert rc == 0
    conn = _db_conn(source_dir)
    try:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute("SELECT * FROM rag_ambient_state"))
        assert len(rows) == 1
        assert rows[0]["hash"] == "h3"
        assert rows[0]["analyzed_at"] == 3.0
    finally:
        conn.close()


def test_round_trip_check_passes(source_dir):
    _write_jsonl(source_dir / "queries.jsonl", [
        {"ts": "2026-04-19T12:00:00", "q": "a", "cmd": "query",
         "paths": ["a.md"], "scores": [0.1]},
        {"ts": "2026-04-19T13:00:00", "q": "b", "cmd": "query",
         "paths": ["b.md"], "scores": [0.2]},
    ])
    rc = mig.main([
        "--source-dir", str(source_dir),
        "--only", "queries",
        "--round-trip-check",
        "--force",
    ])
    assert rc == 0


def test_round_trip_check_detects_drift(source_dir):
    _write_jsonl(source_dir / "queries.jsonl", [
        {"ts": "2026-04-19T12:00:00", "q": "a", "cmd": "query"},
        {"ts": "2026-04-19T13:00:00", "q": "b", "cmd": "query"},
    ])
    rc = mig.main([
        "--source-dir", str(source_dir),
        "--only", "queries",
        "--force",
    ])
    assert rc == 0
    # Delete a row post-migration → round-trip must fail
    conn = _db_conn(source_dir)
    try:
        conn.execute("DELETE FROM rag_queries WHERE q = 'a'")
        conn.commit()
    finally:
        conn.close()
    # Now run round-trip explicitly (reverse + compare)
    rc = mig.main([
        "--source-dir", str(source_dir),
        "--only", "queries",
        "--round-trip-check",
        "--force",
    ])
    # Main migration run already consumed the .jsonl (renamed to .bak), so the
    # second run skips the migration but still runs round-trip against the
    # existing .bak. We expect a non-zero exit from round-trip failures.
    assert rc != 0


def test_reverse_roundtrips(source_dir, tmp_path):
    original = [
        {"ts": "2026-04-19T12:00:00", "q": "a", "cmd": "query",
         "paths": ["a.md"], "scores": [0.1]},
        {"ts": "2026-04-19T13:00:00", "q": "b", "cmd": "query",
         "paths": ["b.md"], "scores": [0.2]},
    ]
    _write_jsonl(source_dir / "queries.jsonl", original)
    rc = mig.main(["--source-dir", str(source_dir), "--only", "queries", "--force"])
    assert rc == 0
    # Reverse into a fresh output dir
    out_dir = tmp_path / "reverse_out"
    out_dir.mkdir()
    # Reuse the same --source-dir for DB path; pass a different target by
    # pointing --source-dir at a dir where ragvec/ragvec.db lives but
    # overwriting is what we want. Simplest: run reverse in source_dir itself;
    # it overwrites queries.jsonl (the original was renamed to .bak).
    rc = mig.main(["--source-dir", str(source_dir), "--reverse", "--only", "queries", "--force"])
    assert rc == 0
    # The reversed file must be JSONL, parseable, and contain same records
    # modulo key-order.
    reversed_records = []
    for line in (source_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            reversed_records.append(json.loads(line))
    assert len(reversed_records) == 2
    got_q = sorted(r["q"] for r in reversed_records)
    want_q = sorted(r["q"] for r in original)
    assert got_q == want_q


def test_refuses_with_running_services(source_dir):
    _write_jsonl(source_dir / "queries.jsonl", [{"ts": "2026-04-19T12:00:00", "q": "x"}])
    with mock.patch.object(mig, "_check_running_services", return_value=["12345"]):
        rc = mig.main(["--source-dir", str(source_dir), "--only", "queries"])
    assert rc == 3
    # Source intact, not renamed.
    assert (source_dir / "queries.jsonl").is_file()


def test_force_bypasses_running_services(source_dir):
    _write_jsonl(source_dir / "queries.jsonl", [{"ts": "2026-04-19T12:00:00", "q": "x"}])
    with mock.patch.object(mig, "_check_running_services", return_value=["12345"]):
        rc = mig.main(["--source-dir", str(source_dir), "--only", "queries", "--force"])
    assert rc == 0
    assert not (source_dir / "queries.jsonl").is_file()


def test_source_missing_skipped_silently(source_dir, capsys):
    # No files created at all
    rc = mig.main(["--source-dir", str(source_dir), "--only", "queries", "--force"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "queries.jsonl" in out
    assert "skipped-missing" in out


def test_summary_json_format(source_dir):
    _write_jsonl(source_dir / "queries.jsonl", [
        {"ts": "2026-04-19T12:00:00", "q": "x", "cmd": "query"},
    ])
    # --summary is like dry-run but emits parseable JSON
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mig.main(["--source-dir", str(source_dir), "--summary", "--only", "queries", "--force"])
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert isinstance(data, list)
    entry = [d for d in data if d["name"] == "queries"][0]
    assert entry["source"] == "queries.jsonl"
    assert entry["rows_read"] == 1
    assert entry["status"] == "dry-run"
    # Source unchanged (summary mode doesn't rename)
    assert (source_dir / "queries.jsonl").is_file()


def test_conversations_index_migrates(source_dir):
    (source_dir / "conversations_index.json").write_text(
        json.dumps({
            "sess1": "04-Archive/99-obsidian-system/99-AI/conversations/a.md",
            "sess2": "04-Archive/99-obsidian-system/99-AI/conversations/b.md",
        }), encoding="utf-8",
    )
    rc = mig.main([
        "--source-dir", str(source_dir),
        "--only", "conversations_index",
        "--force",
    ])
    assert rc == 0
    conn = _db_conn(source_dir)
    try:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute(
            "SELECT session_id, relative_path FROM rag_conversations_index "
            "ORDER BY session_id"
        ))
        assert [(r["session_id"], r["relative_path"]) for r in rows] == [
            ("sess1", "04-Archive/99-obsidian-system/99-AI/conversations/a.md"),
            ("sess2", "04-Archive/99-obsidian-system/99-AI/conversations/b.md"),
        ]
    finally:
        conn.close()


def test_unknown_only_rejected(source_dir):
    rc = mig.main(["--source-dir", str(source_dir), "--only", "does_not_exist", "--force"])
    assert rc == 2
