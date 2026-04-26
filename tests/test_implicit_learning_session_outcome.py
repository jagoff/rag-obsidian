"""Tests para `rag_implicit_learning.session_outcome` y reward_shaping.

Cubre:
- Heurísticas de classify_session: keywords positivos/negativos, re-query,
  silencio largo, single-turn, abandono.
- Confidence levels acordes a la heurística.
- classify_recent_sessions agrupa correctamente por session.
- reward_shaping inserta feedback implícito + skipea explícitos previos.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

from rag_implicit_learning.reward_shaping import (
    apply_reward_from_session_outcomes,
)
from rag_implicit_learning.session_outcome import (
    classify_recent_sessions,
    classify_session,
    session_outcome_summary,
)


# ── classify_session() unit tests ───────────────────────────────────────────

class TestClassifySession:
    """Heurísticas puras — no DB, solo lógica."""

    def test_positive_keyword_in_last_turn_wins(self):
        a = classify_session(
            "s1",
            [
                ("2026-04-25T18:00:00", "qué tengo de Grecia"),
                ("2026-04-25T18:01:00", "perfecto, gracias"),
            ],
        )
        assert a.outcome == "win"
        assert a.confidence >= 0.9
        assert a.evidence["rule"] == "positive_keyword_in_last_turn"

    def test_negative_keyword_in_last_turn_loses(self):
        a = classify_session(
            "s1",
            [
                ("2026-04-25T18:00:00", "qué tengo de Grecia"),
                ("2026-04-25T18:00:30", "no es así, está mal"),
            ],
        )
        assert a.outcome == "loss"
        assert a.confidence >= 0.9
        assert a.evidence["rule"] == "negative_keyword_in_last_turn"

    def test_negative_keyword_wins_over_positive(self):
        """Si el último turn tiene "no, está mal pero gracias", priorizamos negativo."""
        a = classify_session(
            "s1",
            [
                ("2026-04-25T18:00:00", "ayuda con esto"),
                ("2026-04-25T18:00:30", "no, gracias, está mal"),
            ],
        )
        assert a.outcome == "loss"

    def test_internal_requery_marks_loss(self):
        """Re-query <30s dentro de la session → loss aunque sin keyword final."""
        a = classify_session(
            "s1",
            [
                ("2026-04-25T18:00:00", "qué sabes de Grecia"),
                ("2026-04-25T18:00:15", "dame info sobre Grecia"),
                ("2026-04-25T18:01:00", "ok ya entendí"),  # ni positivo ni negativo
            ],
        )
        assert a.outcome == "loss"
        assert a.evidence["rule"] == "internal_requery"

    def test_single_turn_session_is_abandon(self):
        """Una sola query y no hay nada más."""
        a = classify_session(
            "s1",
            [("2026-04-25T18:00:00", "qué tengo de Grecia")],
            now=datetime(2026, 4, 25, 18, 1, 0),  # 1 min después
        )
        # Todavía abierta (silence < session_close_after_seconds default 300).
        assert a.outcome == "abandon"

    def test_silence_after_chain_no_requery_is_win(self):
        """Chain de turns + silencio largo + sin re-query → win implícito."""
        # 3 turns en chain, último a las 18:00:00, "now" a las 18:30:00.
        # Sin re-query interno y sin keyword.
        a = classify_session(
            "s1",
            [
                ("2026-04-25T18:00:00", "qué tengo de Grecia"),
                ("2026-04-25T18:00:45", "y de Italia"),
                ("2026-04-25T18:01:30", "y de España"),
            ],
            now=datetime(2026, 4, 25, 18, 31, 0),  # 30 min de silencio post último
        )
        assert a.outcome == "win"
        assert a.evidence["rule"] == "silence_after_chain_no_requery"
        # confidence más baja que con keyword explícito.
        assert 0.4 <= a.confidence < 0.95

    def test_chain_engagement_no_clear_signal_is_partial(self):
        """3+ turns sin keyword, silencio corto → partial."""
        a = classify_session(
            "s1",
            [
                ("2026-04-25T18:00:00", "qué tengo de Grecia"),
                ("2026-04-25T18:00:45", "y de Italia"),
                ("2026-04-25T18:01:30", "y de España"),
            ],
            now=datetime(2026, 4, 25, 18, 2, 0),  # solo 30s de silencio
        )
        assert a.outcome == "partial"

    def test_empty_session(self):
        a = classify_session("s1", [])
        assert a.outcome == "abandon"
        assert a.confidence == 1.0
        assert a.n_turns == 0


# ── classify_recent_sessions() integration ──────────────────────────────────

@pytest.fixture
def conn() -> sqlite3.Connection:
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
            extra_json TEXT
        );
        """
    )
    yield c
    c.close()


def _insert(conn, *, ts, session, q, paths_json=None):
    cur = conn.execute(
        "INSERT INTO rag_queries (ts, cmd, q, session, paths_json) "
        "VALUES (?, 'chat', ?, ?, ?)",
        (ts, q, session, paths_json),
    )
    return cur.lastrowid


def test_classify_recent_sessions_groups_correctly(conn):
    """Sessions agrupadas, slash-commands filtrados."""
    # Session A: win (gracias al final).
    _insert(conn, ts="2026-04-25T18:00:00", session="A", q="qué tengo")
    _insert(conn, ts="2026-04-25T18:00:30", session="A", q="gracias, perfecto")
    # Session B: solo slash → vacía después del filtro.
    _insert(conn, ts="2026-04-25T18:00:00", session="B", q="/clear")
    # Session C: 1 turn (abandon-like, depende del now).
    _insert(conn, ts="2026-04-25T18:00:00", session="C", q="qué hora es")

    analyses = classify_recent_sessions(
        conn, days=30, now=datetime(2026, 4, 25, 19, 0, 0)
    )
    by_session = {a.session_id: a for a in analyses}

    assert "A" in by_session
    assert by_session["A"].outcome == "win"
    # B no aparece porque solo tenía slash-commands.
    assert "B" not in by_session
    # C: 1 turn + 1h de silencio → abandon (single_turn_then_silence rule).
    assert "C" in by_session
    assert by_session["C"].outcome == "abandon"


def test_session_outcome_summary(conn):
    """Resumen agrega correctamente."""
    _insert(conn, ts="2026-04-25T18:00:00", session="A", q="qué tengo")
    _insert(conn, ts="2026-04-25T18:00:30", session="A", q="gracias")
    _insert(conn, ts="2026-04-25T19:00:00", session="B", q="ayuda")
    _insert(conn, ts="2026-04-25T19:00:30", session="B", q="está mal")

    analyses = classify_recent_sessions(
        conn, days=30, now=datetime(2026, 4, 25, 20, 0, 0)
    )
    summary = session_outcome_summary(analyses)
    assert summary["n_sessions"] >= 2
    assert summary["by_outcome"]["win"] >= 1
    assert summary["by_outcome"]["loss"] >= 1


# ── apply_reward_from_session_outcomes() integration ────────────────────────

def test_reward_shaping_inserts_pos_for_win(conn):
    _insert(conn, ts="2026-04-25T18:00:00", session="A", q="qué tengo")
    _insert(conn, ts="2026-04-25T18:00:30", session="A", q="gracias, perfecto")

    result = apply_reward_from_session_outcomes(
        conn, days=30, now=datetime(2026, 4, 25, 19, 0, 0)
    )

    assert result["n_turns_inserted_pos"] == 2
    assert result["n_turns_inserted_neg"] == 0
    rows = conn.execute(
        "SELECT rating, extra_json FROM rag_feedback ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert all(r[0] == 1 for r in rows)
    for _, extra in rows:
        ej = json.loads(extra)
        assert ej["implicit_loss_source"] == "session_outcome_win"
        assert ej["session_confidence"] >= 0.9


def test_reward_shaping_inserts_neg_for_loss(conn):
    _insert(conn, ts="2026-04-25T18:00:00", session="A", q="qué tengo")
    _insert(conn, ts="2026-04-25T18:00:30", session="A", q="está mal, no es así")

    result = apply_reward_from_session_outcomes(
        conn, days=30, now=datetime(2026, 4, 25, 19, 0, 0)
    )

    assert result["n_turns_inserted_neg"] == 2
    assert result["n_turns_inserted_pos"] == 0


def test_reward_shaping_skips_partial_outcomes(conn):
    """Sessions con outcome `partial` no propagan reward (ambiguo)."""
    # Chain de 3 turns sin keyword y silencio corto → partial.
    _insert(conn, ts="2026-04-25T17:00:00", session="A", q="qué tengo")
    _insert(conn, ts="2026-04-25T17:00:45", session="A", q="y otra cosa")
    _insert(conn, ts="2026-04-25T17:01:30", session="A", q="otra cosa más")

    result = apply_reward_from_session_outcomes(
        conn, days=30, now=datetime(2026, 4, 25, 17, 2, 0)  # silencio < 5min
    )
    assert result["n_skip_ambiguous_outcome"] >= 1
    assert result["n_turns_inserted_pos"] == 0
    assert result["n_turns_inserted_neg"] == 0


def test_reward_shaping_skips_explicit_feedback(conn):
    """Si un turn ya tiene feedback explícito (sin implicit_loss_source en
    extra_json), NO le aplicamos reward shaping arriba."""
    turn_id_int = _insert(
        conn, ts="2026-04-25T18:00:00", session="A", q="qué tengo"
    )
    _insert(conn, ts="2026-04-25T18:00:30", session="A", q="gracias, perfecto")
    # Insertar feedback EXPLÍCITO previo para el primer turn.
    conn.execute(
        "INSERT INTO rag_feedback (ts, turn_id, rating, q, paths_json, extra_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "2026-04-25T18:01:00",
            f"A:{turn_id_int}",
            1,
            "qué tengo",
            None,
            json.dumps({"reason": "manual rating from user"}),  # sin implicit_loss_source
        ),
    )

    result = apply_reward_from_session_outcomes(
        conn, days=30, now=datetime(2026, 4, 25, 19, 0, 0)
    )
    # Solo el segundo turn debería recibir reward shaping; el primero ya tiene explícito.
    assert result["n_turns_skip_explicit"] == 1
    assert result["n_turns_inserted_pos"] == 1


def test_reward_shaping_idempotent(conn):
    _insert(conn, ts="2026-04-25T18:00:00", session="A", q="qué tengo")
    _insert(conn, ts="2026-04-25T18:00:30", session="A", q="perfecto")

    first = apply_reward_from_session_outcomes(
        conn, days=30, now=datetime(2026, 4, 25, 19, 0, 0)
    )
    second = apply_reward_from_session_outcomes(
        conn, days=30, now=datetime(2026, 4, 25, 19, 0, 0)
    )

    assert first["n_turns_inserted_pos"] == 2
    assert second["n_turns_inserted_pos"] == 0
    assert second["n_turns_skip_already_shaped"] == 2


def test_reward_shaping_dry_run(conn):
    _insert(conn, ts="2026-04-25T18:00:00", session="A", q="qué tengo")
    _insert(conn, ts="2026-04-25T18:00:30", session="A", q="gracias")

    result = apply_reward_from_session_outcomes(
        conn, days=30, dry_run=True, now=datetime(2026, 4, 25, 19, 0, 0)
    )

    assert len(result["updates"]) == 2
    rows = conn.execute("SELECT COUNT(*) FROM rag_feedback").fetchone()
    assert rows[0] == 0  # no inserts


def test_reward_shaping_min_confidence(conn):
    """Sessions con confidence < min_confidence no propagan reward."""
    # Chain de 2 turns + silencio largo → win con confidence baja (~0.5).
    _insert(conn, ts="2026-04-25T18:00:00", session="A", q="qué tengo")
    _insert(conn, ts="2026-04-25T18:00:45", session="A", q="y otra cosa más")

    result = apply_reward_from_session_outcomes(
        conn,
        days=30,
        min_confidence=0.9,  # filtra todo lo que no sea keyword explícito
        now=datetime(2026, 4, 25, 19, 0, 0),
    )
    assert result["n_skip_low_confidence"] >= 1
    assert result["n_turns_inserted_pos"] == 0
