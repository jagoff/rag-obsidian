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
