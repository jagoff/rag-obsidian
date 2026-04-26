"""Tests para `rag_implicit_learning.requery_detection`.

Cubre:
- `is_paraphrase()` happy paths + edge cases.
- `detect_requery_loss_signal()` end-to-end con sqlite in-memory.
- Boundary de window_seconds.
- Filtro de slash-commands.
- Idempotency (re-run no duplica feedback).
- only_after_ts cursor.
"""

from __future__ import annotations

import sqlite3

import pytest

from rag_implicit_learning.requery_detection import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_WINDOW_SECONDS,
    detect_requery_loss_signal,
    is_paraphrase,
)


# ── is_paraphrase() unit tests ──────────────────────────────────────────────

class TestIsParaphrase:
    """Tests de la función pura `is_paraphrase()` — no DB."""

    def test_obvious_paraphrase_in_spanish(self):
        assert is_paraphrase(
            "qué sabés de Grecia",
            "decime qué sabés sobre Grecia",
        )

    def test_word_order_changed_with_shared_rare_token(self):
        """'sabes de grecia' vs 'grecia que sabes' — ratio bajo, overlap alto."""
        assert is_paraphrase("qué sabés de Grecia", "Grecia: qué sabés")

    def test_completely_different_topics(self):
        """No comparten tokens informativos → no paráfrasis."""
        assert not is_paraphrase("qué tengo de Grecia", "cómo estás hoy")

    def test_only_stopwords_in_common(self):
        """Comparten 'qué', 'sabes', 'tengo' (stopwords) → no paráfrasis."""
        assert not is_paraphrase("qué sabes de Grecia", "qué sabes de Italia")
        # Aclaración: si 'grecia' e 'italia' fueran muy similares por chars,
        # el ratio podría engañar — pero el filtro de tokens informativos lo
        # bloquea (no comparten ninguno > 3 chars no-stopword).

    def test_empty_strings(self):
        assert not is_paraphrase("", "qué sabés")
        assert not is_paraphrase("qué sabés", "")
        assert not is_paraphrase("", "")

    def test_identical_query(self):
        """Idéntica = paráfrasis trivial (ratio=1.0)."""
        assert is_paraphrase(
            "cuánto debe Alex de la macbook",
            "cuánto debe Alex de la macbook",
        )

    def test_threshold_makes_short_edits_paraphrases(self):
        """Con threshold default 0.5, edits chicos cuentan como paráfrasis."""
        assert is_paraphrase(
            "qué cuotas pagó Alex",
            "qué cuotas pago Alex",  # un acento de diferencia
        )

    def test_high_overlap_low_ratio_still_matches(self):
        """4 tokens informativos compartidos pero ratio bajo (orden distinto):
        debe matchear por la regla de overlap."""
        assert is_paraphrase(
            "ranker calibration tune feedback",
            "feedback tune ranker calibration",
        )

    def test_custom_threshold(self):
        """Threshold más alto → más estricto."""
        # Con threshold 0.95, este edit chico ya no cuenta como paráfrasis
        # por similarity, pero TODAVÍA matchea por overlap (1 rare token).
        assert is_paraphrase(
            "qué cuotas pagó Alex",
            "qué cuotas pago Alex",
            similarity_threshold=0.95,
        )

    def test_two_token_overlap_triggers_match(self):
        """2 rare tokens compartidos → match aunque ratio bajo y union grande."""
        assert is_paraphrase(
            "configurar ranker para producción",
            "tunear ranker en deployment producción",
        )


# ── detect_requery_loss_signal() integration ────────────────────────────────

@pytest.fixture
def conn() -> sqlite3.Connection:
    """In-memory sqlite con schema mínimo."""
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


def _insert_query(
    conn: sqlite3.Connection,
    *,
    ts: str,
    session: str,
    q: str,
    cmd: str = "chat",
) -> int:
    cur = conn.execute(
        "INSERT INTO rag_queries (ts, cmd, q, session) VALUES (?, ?, ?, ?)",
        (ts, cmd, q, session),
    )
    return cur.lastrowid


def _count_implicit_loss(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT COUNT(*) FROM rag_feedback
        WHERE rating = -1
          AND extra_json LIKE '%requery_detection%'
        """
    ).fetchone()
    return rows[0]


def test_detects_paraphrase_within_window(conn):
    """Caso canónico: 2 queries similares en la misma session, gap <30s."""
    _insert_query(
        conn, ts="2026-04-25T18:00:00", session="s1", q="qué sabes de Grecia"
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:15",
        session="s1",
        q="dame info sobre Grecia",
    )

    result = detect_requery_loss_signal(conn, dry_run=False)

    assert result["n_paraphrases_detected"] == 1
    assert result["n_inserted"] == 1
    assert _count_implicit_loss(conn) == 1
    assert result["detections"][0]["delta_seconds"] == pytest.approx(15.0, abs=0.1)


def test_skips_pair_outside_window(conn):
    """Gap >30s → no es re-query reactiva, se ignora."""
    _insert_query(
        conn, ts="2026-04-25T18:00:00", session="s1", q="qué sabes de Grecia"
    )
    _insert_query(
        conn, ts="2026-04-25T18:01:30", session="s1", q="dame info sobre Grecia"
    )  # +90s

    result = detect_requery_loss_signal(conn, dry_run=False)
    assert result["n_paraphrases_detected"] == 0
    assert result["n_skip_outside_window"] >= 1


def test_skips_different_sessions(conn):
    """Queries paráfrasis pero en sessions distintas → no es re-query."""
    _insert_query(
        conn, ts="2026-04-25T18:00:00", session="s1", q="qué sabes de Grecia"
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:10",
        session="s2",
        q="dame info sobre Grecia",
    )

    result = detect_requery_loss_signal(conn, dry_run=False)
    assert result["n_paraphrases_detected"] == 0


def test_skips_slash_commands(conn):
    """`/q`, `/clear`, etc. no son queries reales — se filtran."""
    _insert_query(conn, ts="2026-04-25T18:00:00", session="s1", q="/clear")
    _insert_query(
        conn, ts="2026-04-25T18:00:10", session="s1", q="qué sabes"
    )

    result = detect_requery_loss_signal(conn, dry_run=False)
    assert result["n_paraphrases_detected"] == 0


def test_dry_run_does_not_insert(conn):
    _insert_query(
        conn, ts="2026-04-25T18:00:00", session="s1", q="qué sabes de Grecia"
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:15",
        session="s1",
        q="dame info sobre Grecia",
    )

    result = detect_requery_loss_signal(conn, dry_run=True)
    assert result["n_paraphrases_detected"] == 1
    assert result["n_inserted"] == 0
    assert _count_implicit_loss(conn) == 0


def test_idempotent_rerun(conn):
    """Correr dos veces → no duplica el feedback."""
    _insert_query(
        conn, ts="2026-04-25T18:00:00", session="s1", q="qué sabes de Grecia"
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:15",
        session="s1",
        q="dame info sobre Grecia",
    )

    first = detect_requery_loss_signal(conn, dry_run=False)
    second = detect_requery_loss_signal(conn, dry_run=False)

    assert first["n_inserted"] == 1
    assert second["n_inserted"] == 0
    assert second["n_skip_already_marked"] >= 1
    assert _count_implicit_loss(conn) == 1


def test_unrelated_followup_does_not_trigger(conn):
    """Mismo session, dentro de la ventana, pero queries de temas distintos
    → no es paráfrasis."""
    _insert_query(
        conn, ts="2026-04-25T18:00:00", session="s1", q="qué sabes de Grecia"
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:10",
        session="s1",
        q="cuál es la receta de la milanesa",
    )

    result = detect_requery_loss_signal(conn, dry_run=False)
    assert result["n_paraphrases_detected"] == 0


def test_only_after_ts_cursor(conn):
    """`only_after_ts` filtra turns viejos."""
    _insert_query(
        conn,
        ts="2026-04-20T10:00:00",
        session="s1",
        q="qué tengo sobre Argentina",
    )
    _insert_query(
        conn,
        ts="2026-04-20T10:00:10",
        session="s1",
        q="dame info sobre Argentina",
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:00",
        session="s2",
        q="qué tengo sobre Brasil",
    )
    _insert_query(
        conn,
        ts="2026-04-25T18:00:10",
        session="s2",
        q="dame info sobre Brasil",
    )

    result = detect_requery_loss_signal(
        conn, dry_run=False, only_after_ts="2026-04-22T00:00:00"
    )
    # Solo el par de Brasil cae dentro del cursor.
    assert result["n_paraphrases_detected"] == 1


def test_default_constants():
    """Sanity de los defaults — si bajan, captura más; si suben, menos."""
    assert DEFAULT_WINDOW_SECONDS == 30
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.5
