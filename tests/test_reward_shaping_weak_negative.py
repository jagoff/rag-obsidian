"""Tests del weak-negative branch en reward_shaping (Quick Win #3).

Cubre:
- Sessions abandon con `top_score < WEAK_NEGATIVE_TOP_SCORE_THRESHOLD`
  generan un feedback con `rating=-1` y
  `extra_json.implicit_loss_source = 'session_outcome_weak_negative'`.
- Sessions abandon con `top_score >= threshold` siguen skipeando
  (ambiguous_outcome).
- Sessions abandon sin top_score registrado: skip.
- Boundary exacta: `top_score == 0.4` (igual al threshold) skip.
- Win/loss intactos: confidence gate, source tag legacy, sin weak.
- Outcome `partial`: skip independiente del top_score.
- Confidence override: weak_negative ignora `min_confidence` (el branch
  no aplica el gate por diseño).
- Idempotencia: re-run no duplica el feedback (mismo source).
- Atenuación en training: `feedback_to_training_data` setea `weight=0.3`
  para weak_negative y `weight=1.0` para el resto.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pytest

from rag_implicit_learning.reward_shaping import (
    WEAK_NEGATIVE_SOURCE,
    WEAK_NEGATIVE_TOP_SCORE_THRESHOLD,
    apply_reward_from_session_outcomes,
)


# ── Fixture: schema mínimo que matchea producción ──────────────────────────

@pytest.fixture
def conn() -> sqlite3.Connection:
    """Mismo schema que `tests/test_implicit_learning_session_outcome.py`
    pero con `top_score REAL` poblado para que la rama weak_negative
    tenga datos. `rag_queries.top_score` es la columna que el branch lee
    via `SELECT top_score FROM rag_queries WHERE id = ?`.
    """
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.executescript(
        """
        CREATE TABLE rag_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            cmd TEXT,
            q TEXT NOT NULL,
            session TEXT,
            mode TEXT,
            top_score REAL,
            t_retrieve REAL,
            t_gen REAL,
            answer_len INTEGER,
            paths_json TEXT,
            extra_json TEXT
        );
        CREATE TABLE rag_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            turn_id TEXT,
            rating INTEGER NOT NULL,
            q TEXT,
            scope TEXT,
            paths_json TEXT,
            extra_json TEXT,
            UNIQUE(turn_id, rating, ts)
        );
        """
    )
    yield c
    c.close()


def _insert(
    conn: sqlite3.Connection,
    *,
    ts: str,
    session: str,
    q: str,
    top_score: float | None = None,
    paths_json: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO rag_queries (ts, cmd, q, session, top_score, paths_json) "
        "VALUES (?, 'chat', ?, ?, ?, ?)",
        (ts, q, session, top_score, paths_json),
    )
    return cur.lastrowid


# ── Happy path: weak negative se inserta correctamente ─────────────────────

def test_abandon_with_low_top_score_inserts_weak_negative(conn):
    """Single-turn abandon con top_score=0.3 → rating=-1 + source=weak_negative."""
    _insert(
        conn,
        ts="2026-04-25T18:00:00",
        session="A",
        q="qué tengo de Grecia",
        top_score=0.3,  # below 0.4 threshold
    )

    result = apply_reward_from_session_outcomes(
        conn,
        days=30,
        # 1h después → silencio largo, single-turn → outcome=abandon (rule
        # `single_turn_then_silence`).
        now=datetime(2026, 4, 25, 19, 0, 0),
    )

    assert result["n_turns_inserted_neg"] == 1
    assert result["n_turns_inserted_pos"] == 0
    assert result["n_turns_inserted_weak_negative"] == 1
    # No se contó como ambiguous skip — entró por el branch nuevo.
    assert result["n_skip_ambiguous_outcome"] == 0

    rows = conn.execute(
        "SELECT rating, extra_json FROM rag_feedback ORDER BY id"
    ).fetchall()
    assert len(rows) == 1
    rating, extra_str = rows[0]
    assert rating == -1
    extra = json.loads(extra_str)
    assert extra["implicit_loss_source"] == WEAK_NEGATIVE_SOURCE
    # confidence se capa a 0.5 max para weak_negative.
    assert extra["session_confidence"] <= 0.5


def test_abandon_with_high_top_score_skips(conn):
    """Single-turn abandon con top_score=0.7 → skip ambiguous (encontró algo)."""
    _insert(
        conn,
        ts="2026-04-25T18:00:00",
        session="A",
        q="qué tengo de Grecia",
        top_score=0.7,
    )

    result = apply_reward_from_session_outcomes(
        conn,
        days=30,
        now=datetime(2026, 4, 25, 19, 0, 0),
    )

    assert result["n_turns_inserted_neg"] == 0
    assert result["n_turns_inserted_weak_negative"] == 0
    assert result["n_skip_ambiguous_outcome"] >= 1
    rows = conn.execute("SELECT COUNT(*) FROM rag_feedback").fetchone()
    assert rows[0] == 0


def test_abandon_without_top_score_skips(conn):
    """Abandon sin top_score (NULL en rag_queries) → skip, no podemos decidir."""
    _insert(
        conn,
        ts="2026-04-25T18:00:00",
        session="A",
        q="qué tengo de Grecia",
        top_score=None,
    )

    result = apply_reward_from_session_outcomes(
        conn,
        days=30,
        now=datetime(2026, 4, 25, 19, 0, 0),
    )

    assert result["n_turns_inserted_neg"] == 0
    assert result["n_turns_inserted_weak_negative"] == 0
    assert result["n_skip_ambiguous_outcome"] >= 1


def test_threshold_boundary_exact_value_skips(conn):
    """`top_score == WEAK_NEGATIVE_TOP_SCORE_THRESHOLD` → skip (>=)."""
    _insert(
        conn,
        ts="2026-04-25T18:00:00",
        session="A",
        q="boundary",
        top_score=WEAK_NEGATIVE_TOP_SCORE_THRESHOLD,  # = 0.4
    )

    result = apply_reward_from_session_outcomes(
        conn,
        days=30,
        now=datetime(2026, 4, 25, 19, 0, 0),
    )

    assert result["n_turns_inserted_weak_negative"] == 0
    assert result["n_skip_ambiguous_outcome"] >= 1


# ── Sanidad: win/loss y partial intactos ─────────────────────────────────

def test_win_session_with_low_top_score_still_wins(conn):
    """Session win (keyword `gracias`) con top_score=0.3 sigue rating=+1.

    El branch weak_negative aplica SOLO a outcome=abandon — wins se evalúan
    con la lógica clásica (positive_keyword_in_last_turn).
    """
    _insert(
        conn,
        ts="2026-04-25T18:00:00",
        session="A",
        q="qué tengo de Grecia",
        top_score=0.3,
    )
    _insert(
        conn,
        ts="2026-04-25T18:00:30",
        session="A",
        q="gracias, perfecto",
        top_score=0.2,  # weak match en el segundo turn, pero outcome=win
    )

    result = apply_reward_from_session_outcomes(
        conn,
        days=30,
        now=datetime(2026, 4, 25, 19, 0, 0),
    )

    assert result["n_turns_inserted_pos"] == 2
    assert result["n_turns_inserted_neg"] == 0
    assert result["n_turns_inserted_weak_negative"] == 0
    rows = conn.execute(
        "SELECT extra_json FROM rag_feedback ORDER BY id"
    ).fetchall()
    for (extra_str,) in rows:
        extra = json.loads(extra_str)
        assert extra["implicit_loss_source"] == "session_outcome_win"


def test_partial_outcome_skips_regardless_of_top_score(conn):
    """3+ turns sin keyword + silencio corto → partial, skip total
    independiente del top_score."""
    _insert(
        conn,
        ts="2026-04-25T17:00:00",
        session="A",
        q="qué tengo",
        top_score=0.1,  # bajo, pero outcome != abandon → no aplica weak
    )
    _insert(
        conn,
        ts="2026-04-25T17:00:45",
        session="A",
        q="y otra cosa",
        top_score=0.1,
    )
    _insert(
        conn,
        ts="2026-04-25T17:01:30",
        session="A",
        q="otra cosa más",
        top_score=0.1,
    )

    result = apply_reward_from_session_outcomes(
        conn,
        days=30,
        # silencio < 5min para que classify_session devuelva partial.
        now=datetime(2026, 4, 25, 17, 2, 0),
    )

    assert result["n_skip_ambiguous_outcome"] >= 1
    assert result["n_turns_inserted_neg"] == 0
    assert result["n_turns_inserted_weak_negative"] == 0
    assert result["n_turns_inserted_pos"] == 0


# ── Confidence override ────────────────────────────────────────────────────

def test_weak_negative_ignores_min_confidence_gate(conn):
    """Session abandon con confidence baja (~0.4) y top_score=0.2 sigue
    generando weak_negative aunque min_confidence=0.7 (default).

    El punto del feature es absorber data débil: el confidence gate no
    aplica al branch weak_negative.
    """
    # Single turn + 1h de silencio → outcome=abandon con confidence=0.4
    # (rule `single_turn_then_silence`).
    _insert(
        conn,
        ts="2026-04-25T18:00:00",
        session="A",
        q="qué tengo de Grecia",
        top_score=0.2,
    )

    result = apply_reward_from_session_outcomes(
        conn,
        days=30,
        min_confidence=0.7,  # default, pero gateado solo para win/loss
        now=datetime(2026, 4, 25, 19, 0, 0),
    )

    # Aunque session_confidence está debajo de 0.7, el branch inserta igual.
    assert result["n_turns_inserted_weak_negative"] == 1
    assert result["n_skip_low_confidence"] == 0


# ── Idempotencia ───────────────────────────────────────────────────────────

def test_weak_negative_idempotent_on_rerun(conn):
    """Re-run no duplica el feedback weak_negative (mismo source matcheado
    contra `existing['implicit_sources']`)."""
    _insert(
        conn,
        ts="2026-04-25T18:00:00",
        session="A",
        q="qué tengo",
        top_score=0.2,
    )

    first = apply_reward_from_session_outcomes(
        conn, days=30, now=datetime(2026, 4, 25, 19, 0, 0)
    )
    second = apply_reward_from_session_outcomes(
        conn, days=30, now=datetime(2026, 4, 25, 19, 0, 0)
    )

    assert first["n_turns_inserted_weak_negative"] == 1
    assert second["n_turns_inserted_weak_negative"] == 0
    assert second["n_turns_skip_already_shaped"] == 1
    rows = conn.execute("SELECT COUNT(*) FROM rag_feedback").fetchone()
    assert rows[0] == 1


def test_weak_negative_skipped_if_explicit_feedback_exists(conn):
    """Si el turn ya tiene feedback explícito (sin implicit_loss_source),
    no le pisamos con weak_negative.
    """
    turn_id_int = _insert(
        conn,
        ts="2026-04-25T18:00:00",
        session="A",
        q="qué tengo",
        top_score=0.2,
    )
    conn.execute(
        "INSERT INTO rag_feedback "
        "(ts, turn_id, rating, q, paths_json, extra_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "2026-04-25T18:01:00",
            f"A:{turn_id_int}",
            1,  # 👍 explícito del user
            "qué tengo",
            None,
            json.dumps({"reason": "manual rating from user"}),
        ),
    )

    result = apply_reward_from_session_outcomes(
        conn, days=30, now=datetime(2026, 4, 25, 19, 0, 0)
    )

    assert result["n_turns_inserted_weak_negative"] == 0
    assert result["n_turns_skip_explicit"] == 1
    # Sigue solo el feedback explícito original.
    rows = conn.execute(
        "SELECT rating, extra_json FROM rag_feedback ORDER BY id"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1


# ── Dry-run respeta el branch ──────────────────────────────────────────────

def test_weak_negative_dry_run_does_not_insert(conn):
    """Dry-run reporta el update pero no escribe."""
    _insert(
        conn,
        ts="2026-04-25T18:00:00",
        session="A",
        q="qué tengo",
        top_score=0.2,
    )

    result = apply_reward_from_session_outcomes(
        conn, days=30, dry_run=True, now=datetime(2026, 4, 25, 19, 0, 0)
    )

    assert result["n_turns_inserted_weak_negative"] == 1
    assert len(result["updates"]) == 1
    update = result["updates"][0]
    assert update["rating"] == -1
    assert update["implicit_loss_source"] == WEAK_NEGATIVE_SOURCE
    rows = conn.execute("SELECT COUNT(*) FROM rag_feedback").fetchone()
    assert rows[0] == 0


# ── Atenuación end-to-end en training ──────────────────────────────────────

def test_feedback_to_training_data_weights_weak_negative_attenuated():
    """`feedback_to_training_data` setea weight=0.3 para los rows con
    `implicit_loss_source = 'session_outcome_weak_negative'` y 1.0 para
    el resto. Verifica el contrato end-to-end del Quick Win #3.
    """
    from rag_ranker_lgbm.features import (
        WEAK_NEGATIVE_TRAINING_WEIGHT,
        feedback_to_training_data,
    )

    c = sqlite3.connect(":memory:", isolation_level=None)
    try:
        c.executescript(
            """
            CREATE TABLE rag_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                turn_id TEXT,
                rating INTEGER NOT NULL,
                q TEXT,
                scope TEXT,
                paths_json TEXT,
                extra_json TEXT
            );
            """
        )

        # Row 1: weak_negative — esperamos weight=0.3 en sus candidates.
        c.execute(
            "INSERT INTO rag_feedback (ts, turn_id, rating, q, paths_json, extra_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "2026-04-25T18:00:00",
                "A:1",
                -1,
                "weak query",
                json.dumps(["weak.md"]),
                json.dumps({
                    "implicit_loss_source": WEAK_NEGATIVE_SOURCE,
                    "session_confidence": 0.4,
                }),
            ),
        )
        # Row 2: loss "fuerte" (rating=-1 pero source=session_outcome_loss).
        c.execute(
            "INSERT INTO rag_feedback (ts, turn_id, rating, q, paths_json, extra_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "2026-04-25T19:00:00",
                "B:1",
                -1,
                "loss query",
                json.dumps(["loss.md"]),
                json.dumps({
                    "implicit_loss_source": "session_outcome_loss",
                    "session_confidence": 0.95,
                }),
            ),
        )

        # Replay fake: 1 candidate per query, mismo path que paths_json para
        # que `_label_for_candidate` les ponga label 0 (rating=-1, in paths).
        def _fake_replay(q: str) -> list[dict]:
            mapping = {"weak query": "weak.md", "loss query": "loss.md"}
            return [{
                "path": mapping[q],
                "rerank": 0.5,
                "recency_raw": 0.0,
                "tag_hits": 0,
                "fb_pos_cos": 0.0,
                "fb_neg_cos": 0.0,
                "graph_pagerank": 0.0,
                "click_prior": 0.0,
                "click_prior_folder": 0.0,
                "click_prior_hour": 0.0,
                "click_prior_dayofweek": 0.0,
                "dwell_score": 0.0,
                "contradiction_count": 0.0,
                "has_recency_cue": False,
            }]

        data = feedback_to_training_data(c, replay_features_fn=_fake_replay)

        weights = data["weight"]
        assert len(weights) == len(data["X"])
        assert len(weights) == 2  # 1 candidate per query, 2 queries
        # El orden de queries en feedback_to_training_data es por ts ASC,
        # así que weak (18:00) viene antes que loss (19:00).
        assert weights[0] == pytest.approx(WEAK_NEGATIVE_TRAINING_WEIGHT)
        assert weights[1] == pytest.approx(1.0)
        assert data["n_weak_negative_candidates"] == 1
    finally:
        c.close()
