"""Tests del synthetic fallback en train_calibration (Quick Win #4, 2026-04-29).

Cubre:
- `_gather_synthetic_calibration_pairs` lee POS desde cosine_to_positive
  y NEG desde cosine_to_query, filtrados por source via path classification.
- `train_calibration` con `use_synthetic_fallback=True` usa synthetic
  cuando feedback real < SYNTH_FALLBACK_THRESHOLD.
- Si feedback real ≥ threshold, NO se llama synthetic (real signal preferred).
- `model_version` queda como 'isotonic-v1-synth' cuando synthetic se usó,
  'isotonic-v1' cuando no.
- `use_synthetic_fallback=False` desactiva el fallback (legacy behavior).
- `n_synth_pairs` y `used_synthetic` en el output reportan correctamente.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

import rag


@pytest.fixture
def conn() -> sqlite3.Connection:
    """Schema mínimo para los tests del calibrate.

    Incluye:
    - rag_feedback / rag_queries (feedback real path)
    - rag_synthetic_queries / rag_synthetic_negatives (synthetic path)
    - rag_score_calibration (write target)
    """
    c = sqlite3.connect(":memory:", isolation_level=None)
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
            extra_json TEXT,
            UNIQUE(turn_id, rating, ts)
        );
        CREATE TABLE rag_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            cmd TEXT,
            q TEXT NOT NULL,
            session TEXT,
            mode TEXT,
            top_score REAL,
            paths_json TEXT,
            scores_json TEXT,
            extra_json TEXT
        );
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
        CREATE TABLE rag_synthetic_negatives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            synthetic_query_id INTEGER NOT NULL,
            query TEXT NOT NULL,
            positive_path TEXT NOT NULL,
            neg_path TEXT NOT NULL,
            cosine_to_query REAL,
            cosine_to_positive REAL,
            UNIQUE(synthetic_query_id, neg_path)
        );
        CREATE TABLE rag_score_calibration (
            source TEXT PRIMARY KEY,
            raw_knots_json TEXT NOT NULL,
            cal_knots_json TEXT NOT NULL,
            n_pos INTEGER NOT NULL,
            n_neg INTEGER NOT NULL,
            trained_at TEXT NOT NULL,
            model_version TEXT NOT NULL,
            extra_json TEXT
        );
        """
    )
    yield c
    c.close()


def _insert_synth(
    conn: sqlite3.Connection,
    *,
    synth_id: int,
    positive_path: str,
    neg_path: str,
    cosine_to_query: float,
    cosine_to_positive: float | None,
) -> None:
    """Helper: agrega 1 row de synthetic_queries + 1 row de negatives.

    Las dos tablas en conjunto representan: para esta synthetic query
    Q (con positive_path P), una de las negatives mineadas fue neg_path
    con cosine query→neg_path = `cosine_to_query`. El cosine query→P
    sale en `cosine_to_positive` (mismo valor para todas las rows del
    mismo synth_id).
    """
    conn.execute(
        "INSERT OR IGNORE INTO rag_synthetic_queries (id, ts, note_path, note_hash, query) "
        "VALUES (?, '2026-04-29', ?, 'h', ?)",
        (synth_id, positive_path, f"q-{synth_id}"),
    )
    conn.execute(
        "INSERT INTO rag_synthetic_negatives "
        "(ts, synthetic_query_id, query, positive_path, neg_path, "
        " cosine_to_query, cosine_to_positive) "
        "VALUES ('2026-04-29', ?, ?, ?, ?, ?, ?)",
        (synth_id, f"q-{synth_id}", positive_path, neg_path,
         cosine_to_query, cosine_to_positive),
    )


# ─── _gather_synthetic_calibration_pairs ──────────────────────────────


def test_gather_synthetic_pairs_filters_by_source(conn):
    """Pares cuyo positive/neg matchean source van; otros se descartan."""
    # Synth 1: positive en gmail, neg en gmail → ambos filtran "gmail".
    _insert_synth(conn, synth_id=1,
                  positive_path="gmail://thread/A",
                  neg_path="gmail://thread/B",
                  cosine_to_query=0.45,
                  cosine_to_positive=0.85)
    # Synth 2: positive en gmail, neg en vault → solo el positive matchea
    # "gmail" (el neg cuenta para "vault").
    _insert_synth(conn, synth_id=2,
                  positive_path="gmail://thread/C",
                  neg_path="01-Projects/foo.md",
                  cosine_to_query=0.30,
                  cosine_to_positive=0.78)

    pairs_gmail = rag._gather_synthetic_calibration_pairs(conn, "gmail")
    # Esperados: 2 positives (synth 1 + synth 2, distinct synth_ids), 1 neg
    # (synth 1 cuyo neg_path es gmail).
    n_pos = sum(1 for _, y in pairs_gmail if y == 1)
    n_neg = sum(1 for _, y in pairs_gmail if y == 0)
    assert n_pos == 2
    assert n_neg == 1

    pairs_vault = rag._gather_synthetic_calibration_pairs(conn, "vault")
    # Esperado: 0 positives (ninguno es vault), 1 neg (el del synth 2).
    n_pos_v = sum(1 for _, y in pairs_vault if y == 1)
    n_neg_v = sum(1 for _, y in pairs_vault if y == 0)
    assert n_pos_v == 0
    assert n_neg_v == 1


def test_gather_synthetic_pairs_skips_null_cosine(conn):
    """Rows con cosine NULL no contribuyen pares."""
    _insert_synth(conn, synth_id=1,
                  positive_path="gmail://A",
                  neg_path="gmail://B",
                  cosine_to_query=0.5,
                  cosine_to_positive=None)  # NULL → no positive
    _insert_synth(conn, synth_id=2,
                  positive_path="gmail://C",
                  neg_path="gmail://D",
                  cosine_to_query=0.3,
                  cosine_to_positive=None)
    # Inicialmente NULL en cosine_to_positive → 0 positives
    # (que es el caso pre-Quick Win #4 si re-corremos sobre data legacy).
    pairs = rag._gather_synthetic_calibration_pairs(conn, "gmail")
    n_pos = sum(1 for _, y in pairs if y == 1)
    assert n_pos == 0
    # Pero los neg sí van porque cosine_to_query está populated.
    n_neg = sum(1 for _, y in pairs if y == 0)
    assert n_neg == 2


def test_gather_synthetic_pairs_distinct_positives(conn):
    """Mismo synth_id con 5 hard-negs no inflar el positive count."""
    # 1 synth_id, 5 negatives → DISTINCT synth_id pero 1 positive.
    for i, neg in enumerate(["B", "C", "D", "E", "F"]):
        conn.execute(
            "INSERT OR IGNORE INTO rag_synthetic_queries (id, ts, note_path, "
            "note_hash, query) VALUES (1, '2026-04-29', 'gmail://A', 'h', 'q-1')"
        )
        conn.execute(
            "INSERT INTO rag_synthetic_negatives "
            "(ts, synthetic_query_id, query, positive_path, neg_path, "
            " cosine_to_query, cosine_to_positive) "
            "VALUES ('2026-04-29', 1, 'q-1', 'gmail://A', ?, ?, 0.85)",
            (f"gmail://{neg}", 0.4 + i * 0.05),
        )
    pairs = rag._gather_synthetic_calibration_pairs(conn, "gmail")
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = sum(1 for _, y in pairs if y == 0)
    assert n_pos == 1  # DISTINCT synth_id
    assert n_neg == 5


def test_gather_synthetic_pairs_no_table(conn):
    """Si las tablas synth no existen, devuelve [] sin raise."""
    c = sqlite3.connect(":memory:")
    pairs = rag._gather_synthetic_calibration_pairs(c, "gmail")
    assert pairs == []
    c.close()


# ─── train_calibration synthetic fallback ──────────────────────────────


def test_train_calibration_uses_synth_when_real_is_low(conn, monkeypatch):
    """Source con < SYNTH_FALLBACK_THRESHOLD reales + synthetic disponible
    → entra el fallback, model_version='isotonic-v1-synth'."""
    # Stub `_ragvec_state_conn` para que apunte a nuestra in-memory conn.
    # Devolvemos un context manager wrapper.
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )

    # Seed solo synthetic data para gmail (cero feedback real).
    # Aseguramos ≥ 20 pairs para pasar min_pairs_per_source.
    for i in range(15):
        _insert_synth(conn, synth_id=i + 1,
                      positive_path=f"gmail://thread/P{i}",
                      neg_path=f"gmail://thread/N{i}",
                      cosine_to_query=0.3 + (i % 5) * 0.05,
                      cosine_to_positive=0.75 + (i % 3) * 0.05)

    result = rag.train_calibration(
        since_days=90,
        min_pairs_per_source=10,  # Más bajo para el test
        dry_run=False,
        use_synthetic_fallback=True,
    )

    gmail_entry = result["sources"]["gmail"]
    assert gmail_entry["used_synthetic"] is True
    assert gmail_entry["n_synth_pairs"] > 0
    assert gmail_entry["n_real_pos"] == 0
    assert gmail_entry["n_real_neg"] == 0

    # Si el fit pasó, debería haber persistido con isotonic-v1-synth.
    if gmail_entry["status"] == "trained":
        row = conn.execute(
            "SELECT model_version FROM rag_score_calibration WHERE source = 'gmail'"
        ).fetchone()
        assert row is not None
        assert row[0] == "isotonic-v1-synth"


def test_train_calibration_skips_synth_when_real_is_enough(conn, monkeypatch):
    """Si feedback real ≥ threshold, synthetic NO se llama aunque exista."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )

    # Stub `_gather_calibration_pairs` para devolver feedback abundante.
    def fake_real_pairs(c, source, since_days=90):
        if source == "gmail":
            # 25 pares reales (>= SYNTH_FALLBACK_THRESHOLD=20).
            # Mix: 12 pos + 13 neg, scores en rango razonable.
            return [(0.7 + i * 0.01, 1) for i in range(12)] + \
                   [(0.3 + i * 0.01, 0) for i in range(13)]
        return []

    monkeypatch.setattr(rag, "_gather_calibration_pairs", fake_real_pairs)

    # Seed synthetic data — debería ignorarse.
    for i in range(10):
        _insert_synth(conn, synth_id=i + 1,
                      positive_path=f"gmail://X{i}",
                      neg_path=f"gmail://Y{i}",
                      cosine_to_query=0.4,
                      cosine_to_positive=0.8)

    result = rag.train_calibration(
        since_days=90,
        min_pairs_per_source=10,
        dry_run=False,
        use_synthetic_fallback=True,
    )

    gmail_entry = result["sources"]["gmail"]
    assert gmail_entry["used_synthetic"] is False
    assert gmail_entry["n_synth_pairs"] == 0
    assert gmail_entry["n_real_pos"] == 12
    assert gmail_entry["n_real_neg"] == 13

    if gmail_entry["status"] == "trained":
        row = conn.execute(
            "SELECT model_version FROM rag_score_calibration WHERE source = 'gmail'"
        ).fetchone()
        # Real-only → "isotonic-v1" classic.
        assert row[0] == "isotonic-v1"


def test_train_calibration_synth_fallback_disabled(conn, monkeypatch):
    """`use_synthetic_fallback=False` → no synth aunque real sea bajo."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )

    # Synthetic abundante.
    for i in range(15):
        _insert_synth(conn, synth_id=i + 1,
                      positive_path=f"gmail://P{i}",
                      neg_path=f"gmail://N{i}",
                      cosine_to_query=0.4,
                      cosine_to_positive=0.8)

    result = rag.train_calibration(
        since_days=90,
        min_pairs_per_source=10,
        dry_run=False,
        use_synthetic_fallback=False,
    )

    gmail_entry = result["sources"]["gmail"]
    assert gmail_entry["used_synthetic"] is False
    assert gmail_entry["n_synth_pairs"] == 0
    # Sin synth → status='insufficient' o 'no-positive' (cero reales)
    assert gmail_entry["status"] in ("insufficient", "no-positive", "no-negative")


def test_train_calibration_dry_run_still_reports_synth_use(conn, monkeypatch):
    """Dry-run: persiste nada pero el reporte sí refleja used_synthetic."""
    class CMWrap:
        def __init__(self, c):
            self.c = c
        def __enter__(self):
            return self.c
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        rag, "_ragvec_state_conn", lambda: CMWrap(conn),
    )
    for i in range(15):
        _insert_synth(conn, synth_id=i + 1,
                      positive_path=f"gmail://P{i}",
                      neg_path=f"gmail://N{i}",
                      cosine_to_query=0.3 + (i % 5) * 0.05,
                      cosine_to_positive=0.75 + (i % 3) * 0.05)

    result = rag.train_calibration(
        since_days=90,
        min_pairs_per_source=10,
        dry_run=True,
        use_synthetic_fallback=True,
    )
    gmail_entry = result["sources"]["gmail"]
    assert gmail_entry["used_synthetic"] is True
    # Nada escrito a rag_score_calibration por dry_run.
    rows = conn.execute("SELECT COUNT(*) FROM rag_score_calibration").fetchone()
    assert rows[0] == 0


# ─── path classification edge case ──────────────────────────────────


def test_classify_source_normalizes_gdrive_to_drive():
    """Quick Win #4: el scheme `gdrive://` (legacy del ingester de Drive)
    debe mapear a la source canonical 'drive' que usa _CALIBRATION_SOURCES."""
    assert rag._classify_source_from_path("gdrive://file/abc") == "drive"
    assert rag._classify_source_from_path("drive://x/y") == "drive"
    assert rag._classify_source_from_path("gmail://thread/abc") == "gmail"
    assert rag._classify_source_from_path("01-Projects/foo.md") == "vault"
    assert rag._classify_source_from_path("") == "vault"
