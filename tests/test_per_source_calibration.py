"""Feature #2 del 2026-04-23 — per-source score calibration via isotonic
regression.

Validates:
- rag_score_calibration DDL present (idempotent)
- _classify_source_from_path maps correctly
- _fit_isotonic_from_pairs handles small / degenerate inputs
- calibrate_score falls back to raw when flag is OFF / no model
- calibrate_score applies piecewise linear interp when model exists
- train_calibration extracts pairs from feedback + persists rows
- CLI `rag calibrate` renders + writes correctly
- Plist registered in _services_spec
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

_QUERIES_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_queries ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " ts TEXT NOT NULL,"
    " cmd TEXT,"
    " q TEXT NOT NULL,"
    " session TEXT,"
    " paths_json TEXT,"
    " scores_json TEXT,"
    " top_score REAL,"
    " extra_json TEXT"
    ")"
)

_CALIBRATION_DDL = (
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

_GOLDEN_DDL = (
    "CREATE TABLE IF NOT EXISTS rag_feedback_golden ("
    " q TEXT, path TEXT, weight REAL)",
    "CREATE TABLE IF NOT EXISTS rag_feedback_golden_meta ("
    " k TEXT PRIMARY KEY, v TEXT)",
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(_FEEDBACK_DDL)
    conn.execute(_QUERIES_DDL)
    conn.execute(_CALIBRATION_DDL)
    for ddl in _GOLDEN_DDL:
        conn.execute(ddl)
    conn.commit()

    @contextlib.contextmanager
    def _fake_conn():
        c = sqlite3.connect(str(db_path))
        try:
            yield c
            c.commit()
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_conn)
    rag._reset_calibration_cache()
    try:
        yield conn, db_path
    finally:
        conn.close()
        rag._reset_calibration_cache()


def _insert_fb(conn, *, ts, rating, q, paths=None, extra=None, turn_id="t"):
    conn.execute(
        "INSERT INTO rag_feedback(ts, turn_id, rating, q, paths_json, extra_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, turn_id, rating, q,
         json.dumps(paths or []),
         json.dumps(extra) if extra is not None else None),
    )
    conn.commit()


def _insert_q(conn, *, ts, q, paths, scores, top_score=0.1, cmd="query"):
    conn.execute(
        "INSERT INTO rag_queries(ts, cmd, q, paths_json, scores_json, top_score) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ts, cmd, q, json.dumps(paths), json.dumps(scores), top_score),
    )
    conn.commit()


# ── _classify_source_from_path ───────────────────────────────────────────

def test_classify_source_vault_default():
    assert rag._classify_source_from_path("01-Projects/foo.md") == "vault"


def test_classify_source_whatsapp_folder():
    assert rag._classify_source_from_path(
        "03-Resources/WhatsApp/chat/2026-04.md"
    ) == "whatsapp"


def test_classify_source_uri_scheme():
    assert rag._classify_source_from_path("whatsapp://jid/msg_id") == "whatsapp"
    assert rag._classify_source_from_path("gmail://msgid") == "gmail"
    assert rag._classify_source_from_path("calendar://event") == "calendar"


def test_classify_source_unknown_uri_falls_to_vault():
    assert rag._classify_source_from_path("unknown://foo") == "vault"


def test_classify_source_empty_string():
    assert rag._classify_source_from_path("") == "vault"


# ── _fit_isotonic_from_pairs ─────────────────────────────────────────────

def test_fit_isotonic_too_few_pairs_returns_none():
    pairs = [(0.1, 0), (0.5, 1)]
    assert rag._fit_isotonic_from_pairs(pairs) is None


def test_fit_isotonic_single_label_returns_none():
    pairs = [(0.1, 0), (0.2, 0), (0.3, 0), (0.4, 0), (0.5, 0), (0.6, 0)]
    assert rag._fit_isotonic_from_pairs(pairs) is None


def test_fit_isotonic_separable_gives_monotonic_knots():
    # Clear signal: low scores → 0, high scores → 1.
    pairs = (
        [(0.01 + i * 0.005, 0) for i in range(15)]
        + [(0.2 + i * 0.01, 1) for i in range(15)]
    )
    result = rag._fit_isotonic_from_pairs(pairs)
    assert result is not None
    raw_k, cal_k = result
    assert len(raw_k) >= 2
    assert len(raw_k) == len(cal_k)
    # Monotonic non-decreasing cal_knots (isotonic guarantee).
    for a, b in zip(cal_k, cal_k[1:]):
        assert a <= b + 1e-9
    # Calibration at the low end ≈ 0, at the high end ≈ 1.
    assert cal_k[0] < 0.5
    assert cal_k[-1] > 0.5


# ── calibrate_score ──────────────────────────────────────────────────────

def test_calibrate_flag_off_returns_raw(monkeypatch):
    monkeypatch.setattr(rag, "_SCORE_CALIBRATION_ENABLED", False)
    rag._reset_calibration_cache()
    assert rag.calibrate_score("vault", 0.42) == 0.42


def test_calibrate_no_model_returns_raw(monkeypatch, temp_db):
    monkeypatch.setattr(rag, "_SCORE_CALIBRATION_ENABLED", True)
    rag._reset_calibration_cache()
    assert rag.calibrate_score("vault", 0.42) == 0.42


def test_calibrate_applies_piecewise_linear(monkeypatch, temp_db):
    conn, _ = temp_db
    # Install a simple calibration: raw 0.0→0.0, 0.5→0.5, 1.0→1.0.
    conn.execute(
        "INSERT INTO rag_score_calibration "
        "(source, raw_knots_json, cal_knots_json, n_pos, n_neg, "
        " trained_at, model_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("vault", json.dumps([0.0, 0.5, 1.0]),
         json.dumps([0.0, 0.5, 1.0]),
         10, 10, "2026-04-23T00:00:00", "isotonic-v1"),
    )
    conn.commit()
    monkeypatch.setattr(rag, "_SCORE_CALIBRATION_ENABLED", True)
    rag._reset_calibration_cache()
    # Identity-like map on this mock.
    assert rag.calibrate_score("vault", 0.25) == pytest.approx(0.25, abs=1e-6)
    assert rag.calibrate_score("vault", 0.75) == pytest.approx(0.75, abs=1e-6)
    # Below lower knot clamps to first cal knot.
    assert rag.calibrate_score("vault", -0.5) == 0.0
    # Above upper knot clamps to last cal knot.
    assert rag.calibrate_score("vault", 2.0) == 1.0


def test_calibrate_wa_compresses_range(monkeypatch, temp_db):
    """Per-source calibration maps WA scores (0.02..0.1) to the full [0,1]."""
    conn, _ = temp_db
    conn.execute(
        "INSERT INTO rag_score_calibration "
        "(source, raw_knots_json, cal_knots_json, n_pos, n_neg, "
        " trained_at, model_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("whatsapp", json.dumps([0.01, 0.05, 0.1]),
         json.dumps([0.0, 0.5, 1.0]),
         10, 10, "2026-04-23T00:00:00", "isotonic-v1"),
    )
    conn.commit()
    monkeypatch.setattr(rag, "_SCORE_CALIBRATION_ENABLED", True)
    rag._reset_calibration_cache()
    # Raw 0.05 (WA perfect match) → calibrated 0.5 (middle).
    assert rag.calibrate_score("whatsapp", 0.05) == pytest.approx(0.5, abs=1e-6)
    # Raw 0.1 (WA excellent) → calibrated 1.0.
    assert rag.calibrate_score("whatsapp", 0.1) == pytest.approx(1.0, abs=1e-6)


def test_calibrate_unknown_source_falls_to_vault(monkeypatch, temp_db):
    conn, _ = temp_db
    conn.execute(
        "INSERT INTO rag_score_calibration "
        "(source, raw_knots_json, cal_knots_json, n_pos, n_neg, "
        " trained_at, model_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("vault", json.dumps([0.0, 1.0]), json.dumps([0.0, 1.0]),
         10, 10, "2026-04-23T00:00:00", "isotonic-v1"),
    )
    conn.commit()
    monkeypatch.setattr(rag, "_SCORE_CALIBRATION_ENABLED", True)
    rag._reset_calibration_cache()
    # No model for 'calendar' but vault exists — fall back.
    assert rag.calibrate_score("calendar", 0.5) == pytest.approx(0.5, abs=1e-6)


def test_calibrate_nan_raw_returns_zero(monkeypatch):
    monkeypatch.setattr(rag, "_SCORE_CALIBRATION_ENABLED", True)
    assert rag.calibrate_score("vault", "not-a-number") == 0.0


# ── train_calibration ────────────────────────────────────────────────────

def test_train_calibration_insufficient_pairs_skips(temp_db):
    conn, _ = temp_db
    _insert_q(conn, ts="2026-04-22T10:00", q="q1",
              paths=["a.md"], scores=[0.5])
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="q1",
               paths=["a.md"], extra={"corrective_path": "a.md"},
               turn_id="t1")

    result = rag.train_calibration(
        since_days=30, min_pairs_per_source=20, dry_run=True,
    )
    assert result["sources"]["vault"]["status"] == "insufficient"
    assert result["trained_sources"] == 0


def test_train_calibration_with_enough_data_persists(temp_db):
    conn, _ = temp_db
    # Build 20+ pairs for vault: one positive + several negatives per query.
    for i in range(25):
        q = f"query {i}"
        ts = f"2026-04-22T10:{i:02d}:00"
        paths = [f"good_{i}.md", f"bad_a_{i}.md", f"bad_b_{i}.md"]
        scores = [0.9 - i * 0.005, 0.4, 0.2]  # strong signal
        _insert_q(conn, ts=ts, q=q, paths=paths, scores=scores)
        _insert_fb(conn, ts=ts, rating=1, q=q,
                   paths=[paths[0]],
                   extra={"corrective_path": paths[0]},
                   turn_id=f"t{i}")

    result = rag.train_calibration(
        since_days=30, min_pairs_per_source=20, dry_run=False,
    )
    vault_entry = result["sources"]["vault"]
    assert vault_entry["status"] == "trained"
    assert vault_entry["n_pos"] == 25
    assert vault_entry["n_neg"] == 50
    assert result["trained_sources"] == 1

    # Verify persistence.
    row = conn.execute(
        "SELECT raw_knots_json, cal_knots_json, model_version "
        "FROM rag_score_calibration WHERE source='vault'"
    ).fetchone()
    assert row is not None
    raw_knots = json.loads(row[0])
    cal_knots = json.loads(row[1])
    assert len(raw_knots) >= 2
    assert len(raw_knots) == len(cal_knots)
    assert row[2] == "isotonic-v1"


def test_train_calibration_dry_run_does_not_persist(temp_db):
    conn, _ = temp_db
    for i in range(25):
        q = f"dry {i}"
        ts = f"2026-04-22T10:{i:02d}:00"
        paths = [f"good_{i}.md", f"bad_{i}.md"]
        scores = [0.8, 0.2]
        _insert_q(conn, ts=ts, q=q, paths=paths, scores=scores)
        _insert_fb(conn, ts=ts, rating=1, q=q,
                   paths=[paths[0]],
                   extra={"corrective_path": paths[0]},
                   turn_id=f"dt{i}")

    result = rag.train_calibration(
        since_days=30, min_pairs_per_source=10, dry_run=True,
    )
    assert result["sources"]["vault"]["status"] == "trained"
    rows = conn.execute("SELECT COUNT(*) FROM rag_score_calibration").fetchone()
    assert rows[0] == 0


# ── CLI ──────────────────────────────────────────────────────────────────

def test_cli_calibrate_json_output(temp_db):
    conn, _ = temp_db
    _insert_q(conn, ts="2026-04-22T10:00", q="cli",
              paths=["a.md"], scores=[0.5])
    _insert_fb(conn, ts="2026-04-22T10:00", rating=1, q="cli",
               paths=["a.md"], extra={"corrective_path": "a.md"},
               turn_id="tcli")

    runner = CliRunner()
    result = runner.invoke(rag.cli, [
        "calibrate", "--since", "30", "--min-pairs", "20",
        "--dry-run", "--as-json",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip().splitlines()[-1])
    assert "sources" in data
    assert "vault" in data["sources"]


def test_cli_calibrate_renders_summary_when_empty(temp_db):
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["calibrate", "--since", "30", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Calibration training" in result.output


# ── plist registration ───────────────────────────────────────────────────

def test_calibration_plist_registered_in_services_spec():
    spec = rag._services_spec("/usr/local/bin/rag")
    labels = [label for label, _, _ in spec]
    assert "com.fer.obsidian-rag-calibrate" in labels


def test_calibration_plist_valid_xml_and_schedule():
    from xml.etree import ElementTree as ET
    content = rag._calibration_plist("/usr/local/bin/rag")
    ET.fromstring(content)
    assert "<integer>4</integer>" in content  # Hour=4
    assert "<integer>30</integer>" in content  # Minute=30
    assert "calibrate" in content
    # 2026-04-30: rolleado de "0" → "1" — el daemon entrenaba pero
    # `calibrate_score()` bailea con el flag apagado, así que el
    # isotonic que producía nunca se aplicaba a queries nuevas. Con
    # "1" el flag se respeta y las nuevas queries del web/serve plists
    # usan los scores calibrados. Detalle en `_calibration_plist`
    # docstring.
    assert "<key>RAG_SCORE_CALIBRATION</key><string>1</string>" in content
