"""Tests del cosine_to_positive en mine_hard_negatives_for_synthetic
(Quick Win #4, 2026-04-29).

Pre-fix: el campo `rag_synthetic_negatives.cosine_to_positive` siempre
era NULL (TODO en el código histórico). Post-fix: lo populamos con el
cosine query→positive_path, extraído del NN search results, para que el
calibrate cross-source pueda consumirlo como raw_score sintético del
lado positivo.
"""

from __future__ import annotations

import sqlite3

import pytest

from rag_ranker_lgbm.hard_negatives import mine_hard_negatives_for_synthetic


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.executescript(
        """
        CREATE TABLE rag_synthetic_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            note_path TEXT NOT NULL,
            note_hash TEXT NOT NULL,
            query TEXT NOT NULL,
            query_kind TEXT,
            gen_model TEXT,
            gen_meta_json TEXT,
            UNIQUE(note_path, query)
        );
        """
    )
    yield c
    c.close()


def _seed_synth(conn: sqlite3.Connection, *, query: str, positive_path: str) -> int:
    cur = conn.execute(
        "INSERT INTO rag_synthetic_queries (ts, note_path, note_hash, query) "
        "VALUES ('2026-04-29', ?, 'h', ?)",
        (positive_path, query),
    )
    return cur.lastrowid


def test_cosine_to_positive_populated_when_positive_in_neighbors(conn):
    """Si el positive_path aparece en neighbors, su cosine se persiste."""
    _seed_synth(conn, query="qué dijo Juan", positive_path="01-Projects/note.md")

    # NN devuelve el positive (cosine 0.92) + 3 negs.
    def fake_nn(_embedding, _k):
        return [
            {"path": "01-Projects/note.md", "cosine": 0.92},
            {"path": "01-Projects/other.md", "cosine": 0.78},
            {"path": "01-Projects/third.md", "cosine": 0.65},
            {"path": "01-Projects/fourth.md", "cosine": 0.55},
        ]

    result = mine_hard_negatives_for_synthetic(
        conn,
        embed_fn=lambda t: [0.1, 0.2, 0.3],
        nearest_neighbors_fn=fake_nn,
        negatives_per_query=3,
    )
    assert result["n_negatives_inserted"] == 3

    rows = conn.execute(
        "SELECT neg_path, cosine_to_query, cosine_to_positive "
        "FROM rag_synthetic_negatives ORDER BY id"
    ).fetchall()
    assert len(rows) == 3
    for _neg_path, _cos_q, cos_pos in rows:
        # cosine_to_positive debe ser 0.92 (mismo query→positive en todos).
        assert cos_pos is not None
        assert abs(cos_pos - 0.92) < 1e-6


def test_cosine_to_positive_null_when_positive_absent(conn):
    """Si el positive_path NO aparece en NN top-K, cosine_to_positive=NULL.

    Caso edge: el positive embedding cayó marginal vs query — los top-K
    no lo incluyen. Histórico, esto era el caso default (sin fix). Post
    Quick Win #4 sigue siendo NULL para preservar honestidad: NO
    inventamos un valor cuando no lo medimos.
    """
    _seed_synth(conn, query="q", positive_path="missing.md")

    def fake_nn(_embedding, _k):
        # Positive no está. Tres negs nada que ver.
        return [
            {"path": "01-Projects/A.md", "cosine": 0.5},
            {"path": "01-Projects/B.md", "cosine": 0.45},
            {"path": "01-Projects/C.md", "cosine": 0.4},
            {"path": "01-Projects/D.md", "cosine": 0.35},
        ]

    result = mine_hard_negatives_for_synthetic(
        conn,
        embed_fn=lambda t: [0.0],
        nearest_neighbors_fn=fake_nn,
        negatives_per_query=3,
    )
    assert result["n_negatives_inserted"] == 3

    rows = conn.execute(
        "SELECT cosine_to_positive FROM rag_synthetic_negatives"
    ).fetchall()
    for (cos_pos,) in rows:
        assert cos_pos is None


def test_cosine_to_positive_present_in_sample_pairs(conn):
    """El sample_pairs del result incluye `cosine_query_to_positive`."""
    _seed_synth(conn, query="q", positive_path="P.md")

    def fake_nn(_e, _k):
        return [
            {"path": "P.md", "cosine": 0.88},
            {"path": "N1.md", "cosine": 0.6},
        ]

    result = mine_hard_negatives_for_synthetic(
        conn,
        embed_fn=lambda t: [0.0],
        nearest_neighbors_fn=fake_nn,
        negatives_per_query=1,
    )
    assert len(result["sample_pairs"]) == 1
    pair = result["sample_pairs"][0]
    assert "cosine_query_to_positive" in pair
    assert abs(pair["cosine_query_to_positive"] - 0.88) < 1e-6


def test_cosine_to_positive_dry_run_no_persist(conn):
    """Dry run: el sample_pairs SÍ incluye cosine_query_to_positive,
    pero nada se persiste a la DB."""
    _seed_synth(conn, query="q", positive_path="P.md")

    def fake_nn(_e, _k):
        return [
            {"path": "P.md", "cosine": 0.91},
            {"path": "N1.md", "cosine": 0.6},
        ]

    result = mine_hard_negatives_for_synthetic(
        conn,
        embed_fn=lambda t: [0.0],
        nearest_neighbors_fn=fake_nn,
        negatives_per_query=1,
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert len(result["sample_pairs"]) == 1
    assert result["sample_pairs"][0]["cosine_query_to_positive"] is not None
    rows = conn.execute(
        "SELECT COUNT(*) FROM rag_synthetic_negatives"
    ).fetchone()
    assert rows[0] == 0
