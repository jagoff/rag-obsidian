"""Unit tests for scripts/finetune_reranker.py (GC#2.C).

No real training, no real retrieve. Tests focus on the pre-training data
pipeline: corrective_path extraction, positive/negative construction, and
the RAG_FINETUNE_MIN_CORRECTIVES gate in main().
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

import rag
from scripts import finetune_reranker as ft


_RAG_FEEDBACK_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_feedback ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ts TEXT NOT NULL,"
    " turn_id TEXT,"
    " rating INTEGER NOT NULL,"
    " q TEXT,"
    " scope TEXT,"
    " paths_json TEXT,"
    " extra_json TEXT"
    ")"
)

_RAG_QUERIES_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_queries ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ts TEXT NOT NULL,"
    " cmd TEXT,"
    " q TEXT NOT NULL,"
    " session TEXT,"
    " paths_json TEXT,"
    " scores_json TEXT,"
    " extra_json TEXT"
    ")"
)


def _mk_telemetry_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute(_RAG_FEEDBACK_DDL)
    conn.execute(_RAG_QUERIES_DDL)
    conn.commit()
    return conn


def _insert_feedback(conn, *, ts, turn_id, rating, q, paths, extra):
    conn.execute(
        "INSERT INTO rag_feedback(ts, turn_id, rating, q, paths_json, extra_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, turn_id, rating, q, json.dumps(paths), json.dumps(extra)),
    )


def _insert_query(conn, *, ts, turn_id, q, paths, scores=None):
    conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, paths_json, scores_json, extra_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, "query", q, json.dumps(paths),
         json.dumps(scores or [0.5] * len(paths)),
         json.dumps({"turn_id": turn_id})),
    )


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Temp telemetry DB with rag_feedback + rag_queries. Patches
    `rag._ragvec_state_conn` so `_fetch_feedback_pairs()` reads from it."""
    db_path = tmp_path / "telemetry.db"
    conn = _mk_telemetry_db(db_path)

    @contextlib.contextmanager
    def _fake_conn():
        c = sqlite3.connect(str(db_path))
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_conn)
    try:
        yield conn, db_path
    finally:
        conn.close()


# ── _fetch_feedback_pairs ────────────────────────────────────────────────

def test_fetch_pairs_extracts_corrective_path(temp_db):
    conn, _ = temp_db
    _insert_feedback(conn, ts="2026-04-22T10:00:00", turn_id="t1", rating=1,
                     q="q1", paths=["a.md"],
                     extra={"corrective_path": "vault/golden1.md"})
    _insert_feedback(conn, ts="2026-04-22T10:01:00", turn_id="t2", rating=1,
                     q="q2", paths=["b.md"],
                     extra={"corrective_path": "vault/golden2.md"})
    _insert_feedback(conn, ts="2026-04-22T10:02:00", turn_id="t3", rating=1,
                     q="q3", paths=["c.md"], extra={})
    _insert_query(conn, ts="2026-04-22T09:59:00", turn_id="t1", q="q1",
                  paths=["a.md"])
    _insert_query(conn, ts="2026-04-22T09:59:00", turn_id="t2", q="q2",
                  paths=["b.md"])
    _insert_query(conn, ts="2026-04-22T09:59:00", turn_id="t3", q="q3",
                  paths=["c.md"])
    conn.commit()

    rows = ft._fetch_feedback_pairs()
    assert len(rows) == 3
    by_turn = {r["turn_id"]: r for r in rows}
    assert by_turn["t1"]["corrective_path"] == "vault/golden1.md"
    assert by_turn["t2"]["corrective_path"] == "vault/golden2.md"
    assert by_turn["t3"]["corrective_path"] is None


# ── _build_training_pairs ────────────────────────────────────────────────

@pytest.fixture
def fake_vault(tmp_path):
    """tmp vault with readable files so `_path_to_doc` returns non-None."""
    root = tmp_path / "vault"
    root.mkdir()
    for name in ["golden.md", "other1.md", "other2.md", "a.md", "b.md",
                 "neg1.md", "neg2.md", "neg3.md"]:
        (root / name).write_text(f"content of {name}\n" * 5, encoding="utf-8")
    return root


def test_build_pairs_uses_corrective_as_only_positive(monkeypatch, fake_vault):
    captured_excludes: list[set[str]] = []

    def _fake_mine(query, positive_paths, col, *, k_pool=10):
        captured_excludes.append(set(positive_paths))
        return ["neg1.md", "neg2.md"]

    monkeypatch.setattr(ft, "_mine_hard_negatives", _fake_mine)

    rows = [{
        "q": "how to X",
        "paths": ["golden.md", "other1.md", "other2.md"],
        "corrective_path": "golden.md",
    }]
    pairs = ft._build_training_pairs(rows, col=None, vault_root=fake_vault,
                                     hard_neg_k=5)
    positives = [p for p in pairs if p["label"] == 1.0]
    negatives = [p for p in pairs if p["label"] == 0.0]
    assert len(positives) == 1
    assert "golden.md" in positives[0]["text2"]
    assert len(negatives) == 2
    assert captured_excludes == [{"golden.md"}]


def test_build_pairs_fallback_when_no_corrective(monkeypatch, fake_vault):
    captured_excludes: list[set[str]] = []

    def _fake_mine(query, positive_paths, col, *, k_pool=10):
        captured_excludes.append(set(positive_paths))
        return []

    monkeypatch.setattr(ft, "_mine_hard_negatives", _fake_mine)

    rows = [{
        "q": "fallback q",
        "paths": ["a.md", "b.md"],
        "corrective_path": None,
    }]
    pairs = ft._build_training_pairs(rows, col=None, vault_root=fake_vault,
                                     hard_neg_k=5)
    positives = [p for p in pairs if p["label"] == 1.0]
    assert len(positives) == 2
    assert {"a.md", "b.md"} <= captured_excludes[0]


# ── _mine_hard_negatives ─────────────────────────────────────────────────

def test_mine_hard_negatives_excludes_corrective(monkeypatch):
    def _fake_retrieve(col, question, k, folder, multi_query, auto_filter):
        return {
            "metas": [
                {"file": "golden.md"},
                {"file": "neg1.md"},
                {"file": "neg2.md"},
                {"file": "neg3.md"},
                {"file": "cross://whatsapp/1234"},
            ],
        }

    monkeypatch.setattr(rag, "retrieve", _fake_retrieve)

    negs = ft._mine_hard_negatives("q", {"golden.md"}, col=None, k_pool=5)
    assert "golden.md" not in negs
    assert "cross://whatsapp/1234" not in negs
    assert set(negs) == {"neg1.md", "neg2.md", "neg3.md"}


# ── Gate in main() ───────────────────────────────────────────────────────

def _mk_rows(with_cp: int, without_cp: int) -> list[dict]:
    out = []
    for i in range(with_cp):
        out.append({
            "rating": 1, "turn_id": f"cp{i}", "q": f"q{i}",
            "paths": ["a.md"], "scores": [0.5],
            "corrective_path": f"golden{i}.md",
        })
    for i in range(without_cp):
        out.append({
            "rating": 1, "turn_id": f"nocp{i}", "q": f"q_nocp{i}",
            "paths": ["a.md"], "scores": [0.5],
            "corrective_path": None,
        })
    return out


def test_gate_aborts_below_minimum_correctives(monkeypatch, capsys):
    monkeypatch.setattr(ft, "_fetch_feedback_pairs",
                        lambda: _mk_rows(with_cp=5, without_cp=60))
    monkeypatch.setenv("RAG_FINETUNE_MIN_CORRECTIVES", "20")
    monkeypatch.setattr(sys, "argv", ["finetune_reranker.py"])

    with pytest.raises(SystemExit) as ei:
        ft.main()
    assert ei.value.code == 5
    err = capsys.readouterr().err
    assert "Rows with corrective_path: 5" in err
    assert "need ≥20" in err


def test_gate_proceeds_above_minimum_correctives(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ft, "_fetch_feedback_pairs",
                        lambda: _mk_rows(with_cp=25, without_cp=0))
    monkeypatch.setattr(ft, "_build_training_pairs",
                        lambda rows, col, vault_root, hard_neg_k: [])
    monkeypatch.setattr(rag, "get_db", lambda: object())
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    monkeypatch.setenv("RAG_FINETUNE_MIN_CORRECTIVES", "20")
    monkeypatch.setattr(sys, "argv", ["finetune_reranker.py", "--dry-run"])

    ft.main()
    err = capsys.readouterr().err
    assert "Rows with corrective_path: 25" in err
    assert "[dry-run] exiting before training" in err


def test_gate_env_var_override(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ft, "_build_training_pairs",
                        lambda rows, col, vault_root, hard_neg_k: [])
    monkeypatch.setattr(rag, "get_db", lambda: object())
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    monkeypatch.setenv("RAG_FINETUNE_MIN_CORRECTIVES", "5")
    monkeypatch.setattr(sys, "argv", ["finetune_reranker.py", "--dry-run"])

    monkeypatch.setattr(ft, "_fetch_feedback_pairs",
                        lambda: _mk_rows(with_cp=7, without_cp=5))
    ft.main()
    err = capsys.readouterr().err
    assert "Rows with corrective_path: 7" in err
    assert "(min required: 5)" in err

    monkeypatch.setattr(ft, "_fetch_feedback_pairs",
                        lambda: _mk_rows(with_cp=3, without_cp=20))
    with pytest.raises(SystemExit) as ei:
        ft.main()
    assert ei.value.code == 5
    err = capsys.readouterr().err
    assert "need ≥5" in err


def test_dry_run_reports_corrective_counts(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ft, "_fetch_feedback_pairs",
                        lambda: _mk_rows(with_cp=22, without_cp=5))
    monkeypatch.setattr(ft, "_build_training_pairs",
                        lambda rows, col, vault_root, hard_neg_k: [])
    monkeypatch.setattr(rag, "get_db", lambda: object())
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    monkeypatch.setenv("RAG_FINETUNE_MIN_CORRECTIVES", "20")
    monkeypatch.setattr(sys, "argv", ["finetune_reranker.py", "--dry-run"])

    ft.main()
    err = capsys.readouterr().err
    assert "Rows with corrective_path: 22" in err
    assert "(min required: 20)" in err


# ── --pairs-from: consume JSONL exported by scripts/export_training_pairs.py ──
# (2026-04-22) Additive flag that skips _fetch_feedback_pairs() and reads
# pre-mined pairs from disk. The miner uses rag_behavior + impression
# history to mine richer signal than rag_feedback alone.


def _jsonl_row(query: str, positive: str, negatives: list[str], source: str = "behavior_copy") -> dict:
    """Match the shape exported by scripts/export_training_pairs.py."""
    return {
        "query": query, "positive": positive, "negatives": negatives,
        "source": source, "turn_id": None,
        "ts": "2026-04-22T20:00:00",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _make_vault(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a fake vault with the given path→content map so _path_to_doc
    can read the files back."""
    vault = tmp_path / "vault"
    vault.mkdir(exist_ok=True)
    for rel, body in files.items():
        full = vault / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(body, encoding="utf-8")
    return vault


def test_load_pairs_from_jsonl_happy_path(tmp_path):
    """Each JSONL row yields 1 positive + up to hard_neg_k negatives as
    {text1, text2, label} rows. text2 is the first 800 chars of the file."""
    vault = _make_vault(tmp_path, {
        "A.md": "body of A — positive document",
        "B.md": "body of B — negative document",
        "C.md": "body of C — another negative",
    })
    jsonl = tmp_path / "pairs.jsonl"
    _write_jsonl(jsonl, [
        _jsonl_row("query uno", "A.md", ["B.md", "C.md"]),
    ])
    pairs = ft._load_pairs_from_jsonl(jsonl, vault, hard_neg_k=5)
    # 1 positive + 2 negatives = 3 training pairs
    assert len(pairs) == 3
    pos_pairs = [p for p in pairs if p["label"] == 1.0]
    neg_pairs = [p for p in pairs if p["label"] == 0.0]
    assert len(pos_pairs) == 1
    assert len(neg_pairs) == 2
    # All share the same text1 (query).
    assert pos_pairs[0]["text1"] == "query uno"
    assert all(p["text1"] == "query uno" for p in neg_pairs)
    # text2 is the file content.
    assert "body of A" in pos_pairs[0]["text2"]


def test_load_pairs_caps_negatives_at_hard_neg_k(tmp_path):
    """JSONL may have 9 negs per row; --hard-neg-k caps them at 5 (default)."""
    files = {"pos.md": "pos"}
    files.update({f"n{i}.md": f"neg {i}" for i in range(9)})
    vault = _make_vault(tmp_path, files)

    jsonl = tmp_path / "pairs.jsonl"
    _write_jsonl(jsonl, [
        _jsonl_row("q", "pos.md", [f"n{i}.md" for i in range(9)]),
    ])
    pairs = ft._load_pairs_from_jsonl(jsonl, vault, hard_neg_k=3)
    neg_pairs = [p for p in pairs if p["label"] == 0.0]
    assert len(neg_pairs) == 3


def test_load_pairs_skips_unreadable_paths(tmp_path, capsys):
    """Paths that don't exist on disk are skipped with a tally in stderr."""
    vault = _make_vault(tmp_path, {"real.md": "exists"})
    jsonl = tmp_path / "pairs.jsonl"
    _write_jsonl(jsonl, [
        _jsonl_row("q", "real.md", ["ghost1.md", "ghost2.md"]),
    ])
    pairs = ft._load_pairs_from_jsonl(jsonl, vault, hard_neg_k=5)
    # Positive read; both negatives skipped → 1 pair.
    assert len(pairs) == 1
    err = capsys.readouterr().err
    assert "skipped unreadable paths: 2" in err


def test_load_pairs_skips_empty_or_bad_rows(tmp_path, capsys):
    """Empty lines, missing query, missing positive, bad JSON all skipped
    without raising. Bad JSON logs a [skip] line."""
    vault = _make_vault(tmp_path, {"A.md": "a", "B.md": "b"})
    jsonl = tmp_path / "pairs.jsonl"
    jsonl.write_text(
        # Valid row
        json.dumps(_jsonl_row("good q", "A.md", ["B.md"])) + "\n"
        # Empty line
        + "\n"
        # Missing query
        + json.dumps({"positive": "A.md", "negatives": []}) + "\n"
        # Missing positive
        + json.dumps({"query": "q", "negatives": []}) + "\n"
        # Broken JSON
        + "{not valid json]\n",
        encoding="utf-8",
    )
    pairs = ft._load_pairs_from_jsonl(jsonl, vault, hard_neg_k=5)
    # Only the one valid row contributes: 1 pos + 1 neg = 2 pairs.
    assert len(pairs) == 2
    err = capsys.readouterr().err
    assert "bad JSON" in err


def test_load_pairs_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        ft._load_pairs_from_jsonl(
            tmp_path / "does-not-exist.jsonl", tmp_path, hard_neg_k=5,
        )
    assert "does-not-exist.jsonl" in str(exc_info.value)


def test_main_pairs_from_bypasses_sql_fetch(monkeypatch, tmp_path, capsys):
    """With --pairs-from, main() must NOT call _fetch_feedback_pairs (the
    SQL-path data source). Instead it reads the JSONL and feeds the
    trainer directly."""
    vault = _make_vault(tmp_path, {
        "pos.md": "positive body",
        "neg1.md": "negative 1 body",
    })
    jsonl = tmp_path / "pairs.jsonl"
    _write_jsonl(jsonl, [
        # Need ≥20 training pairs to pass the gate. 10 rows × (1 pos + 1 neg) = 20.
        _jsonl_row(f"q-{i}", "pos.md", ["neg1.md"])
        for i in range(10)
    ])

    # If the SQL fetch is called, we explode — this asserts bypass.
    monkeypatch.setattr(ft, "_fetch_feedback_pairs",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("should not be called in --pairs-from mode")
                        ))
    monkeypatch.setattr(ft, "_build_training_pairs",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            AssertionError("should not be called either")
                        ))
    monkeypatch.setattr(rag, "get_db", lambda: object())
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    monkeypatch.setattr(sys, "argv", [
        "finetune_reranker.py",
        "--dry-run",
        "--pairs-from", str(jsonl),
    ])

    ft.main()
    err = capsys.readouterr().err
    # Stderr should mention the JSONL source and the pair count.
    assert "Loading pre-mined pairs from" in err
    assert "JSONL rows seen: 10" in err
    assert "Training pairs: total=20 pos=10 neg=10" in err
    assert "[dry-run] exiting before training" in err


def test_main_pairs_from_aborts_when_too_few(monkeypatch, tmp_path):
    """Gate at 20 training pairs from the JSONL path."""
    vault = _make_vault(tmp_path, {"pos.md": "p", "neg.md": "n"})
    jsonl = tmp_path / "pairs.jsonl"
    # Only 2 training pairs (1 pos + 1 neg) — below the 20 threshold.
    _write_jsonl(jsonl, [_jsonl_row("q", "pos.md", ["neg.md"])])

    monkeypatch.setattr(rag, "get_db", lambda: object())
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    monkeypatch.setattr(sys, "argv", [
        "finetune_reranker.py", "--pairs-from", str(jsonl),
    ])

    with pytest.raises(SystemExit) as ei:
        ft.main()
    assert ei.value.code == 2
