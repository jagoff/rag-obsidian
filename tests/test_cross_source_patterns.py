"""Tests para `rag.cross_source_patterns` — engine de Pearson + lag
analysis sobre métricas diarias del usuario.

Cubre:
1. _pearson: r perfect 1.0 / -1.0, r=0 con vectores ortogonales,
   handling de None, n<3 → 0.0, std=0 → 0.0.
2. _align_series: lag=0 mismo día, lag=1 desplaza A 1 día atrás,
   lag=7 desplaza 1 semana, intersect dates only.
3. compute_correlations: filtra por min_n + min_abs_r + max_p,
   ordena por |r| desc, severity bands correctas.
4. Collector registry: register_metric agrega al registry,
   known_metrics lista los registrados.
5. Cache: re-call con misma input no re-calcula (verificable via
   spy en _pearson).
6. Smoke con DB tmp para los collectors SQL-based.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import rag
from rag import cross_source_patterns as csp


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _date_offset(d: int) -> str:
    return (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")


# ── Pearson ───────────────────────────────────────────────────────────────


def test_pearson_perfect_positive():
    r, n, p = csp._pearson([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0])
    assert r == pytest.approx(1.0, abs=1e-6)
    assert n == 5
    # p para r=1.0 con n=5 es ~0
    assert p < 0.01


def test_pearson_perfect_negative():
    r, n, _ = csp._pearson([1.0, 2.0, 3.0, 4.0, 5.0], [50.0, 40.0, 30.0, 20.0, 10.0])
    assert r == pytest.approx(-1.0, abs=1e-6)


def test_pearson_zero_correlation():
    """Vectores con correlación cero (más o menos)."""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [3.0, 1.0, 4.0, 1.0, 5.0]  # noisy
    r, n, _ = csp._pearson(xs, ys)
    assert n == 5
    # r debe estar cerca de 0 (no exactamente porque hay 5 datos).
    assert abs(r) < 0.5


def test_pearson_too_few_points():
    """n < 3 devuelve r=0, p=1."""
    r, n, p = csp._pearson([1.0, 2.0], [3.0, 4.0])
    assert r == 0.0
    assert n == 2
    assert p == 1.0


def test_pearson_constant_series():
    """Std=0 (constante) devuelve r=0, p=1 sin division by zero."""
    r, n, p = csp._pearson([5.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert r == 0.0
    assert n == 5
    assert p == 1.0


def test_pearson_handles_none_values():
    """None values se filtran antes de computar."""
    r, n, _ = csp._pearson(
        [1.0, 2.0, None, 4.0, 5.0],
        [10.0, 20.0, 30.0, 40.0, 50.0],
    )
    # Solo 4 pares válidos, pero correlación perfecta.
    assert n == 4
    assert r == pytest.approx(1.0, abs=1e-6)


def test_pearson_mismatched_lengths_returns_zero():
    r, n, p = csp._pearson([1, 2, 3], [4, 5])
    assert r == 0.0 and n == 0 and p == 1.0


# ── _align_series ──────────────────────────────────────────────────────────


def test_align_series_same_day():
    a = {"2026-04-29": 1.0, "2026-04-30": 2.0}
    b = {"2026-04-29": 10.0, "2026-04-30": 20.0}
    xs, ys, dates = csp._align_series(a, b, lag=0)
    assert sorted(zip(xs, ys)) == sorted([(1.0, 10.0), (2.0, 20.0)])
    assert sorted(dates) == ["2026-04-29", "2026-04-30"]


def test_align_series_lag_1_shifts_a_back():
    """lag=1 → ys[date] vs xs[date - 1 day]."""
    a = {"2026-04-29": 5.0, "2026-04-30": 7.0}
    b = {"2026-04-29": 100.0, "2026-04-30": 200.0}
    xs, ys, _ = csp._align_series(a, b, lag=1)
    # b[2026-04-30] vs a[2026-04-29]. b[2026-04-29] no tiene a[2026-04-28].
    assert xs == [5.0]
    assert ys == [200.0]


def test_align_series_lag_7():
    a = {"2026-04-23": 1.0, "2026-04-30": 2.0}
    b = {"2026-04-30": 99.0}
    xs, ys, _ = csp._align_series(a, b, lag=7)
    # b[2026-04-30] vs a[2026-04-23].
    assert xs == [1.0]
    assert ys == [99.0]


def test_align_series_intersect_only():
    """Solo dates con valor en ambas series quedan."""
    a = {"2026-04-29": 1.0, "2026-04-30": 2.0}
    b = {"2026-04-30": 20.0}  # no tiene 2026-04-29
    xs, ys, _ = csp._align_series(a, b, lag=0)
    assert len(xs) == 1
    assert (xs[0], ys[0]) == (2.0, 20.0)


# ── compute_correlations ───────────────────────────────────────────────────


def test_compute_correlations_filters_by_min_n():
    """Pares con n < min_n no se reportan."""
    metrics = {
        "a": {f"2026-04-{i:02d}": float(i) for i in range(1, 5)},  # n=4
        "b": {f"2026-04-{i:02d}": float(i * 2) for i in range(1, 5)},  # n=4 (paired = 4)
    }
    findings = csp.compute_correlations(
        metrics, lags=(0,), min_n=21, min_abs_r=0.3, max_p=0.05,
    )
    assert findings == []


def test_compute_correlations_strong_pair():
    """Par con perfect correlation, n>=21, p<0.05 → finding strong."""
    n = 25
    metrics = {
        "a": {f"2026-04-{(i % 30) + 1:02d}": float(i) for i in range(n)},
        "b": {f"2026-04-{(i % 30) + 1:02d}": float(i * 2 + 1) for i in range(n)},
    }
    findings = csp.compute_correlations(
        metrics, lags=(0,), min_n=21, min_abs_r=0.4, max_p=0.05,
    )
    assert len(findings) >= 1
    strong = [f for f in findings if f["severity"] == "strong"]
    assert len(strong) >= 1
    assert strong[0]["pair"] == ("a", "b")
    assert strong[0]["r"] >= 0.6
    assert strong[0]["lag"] == 0


def test_compute_correlations_orders_by_abs_r():
    """Findings ordenadas por |r| descendente."""
    # Construyo 3 pares con correlaciones distintas.
    n = 25
    metrics = {
        "x": {f"2026-04-{(i % 30) + 1:02d}": float(i) for i in range(n)},
        "y_strong": {f"2026-04-{(i % 30) + 1:02d}": float(i * 1.5) for i in range(n)},
        "y_neg": {f"2026-04-{(i % 30) + 1:02d}": float(-i * 1.2) for i in range(n)},
    }
    findings = csp.compute_correlations(
        metrics, lags=(0,), min_n=21, min_abs_r=0.4, max_p=0.05,
    )
    assert len(findings) >= 2
    # |r| de cada finding debe estar en orden desc.
    abs_rs = [abs(f["r"]) for f in findings]
    assert abs_rs == sorted(abs_rs, reverse=True)


def test_compute_correlations_includes_lag():
    """Cada finding tiene `lag` entre los testeados."""
    n = 25
    metrics = {
        "a": {f"2026-04-{(i % 30) + 1:02d}": float(i) for i in range(n)},
        "b": {f"2026-04-{(i % 30) + 1:02d}": float(i * 2) for i in range(n)},
    }
    findings = csp.compute_correlations(
        metrics, lags=(0, 1), min_n=10, min_abs_r=0.4, max_p=0.05,
    )
    lags_seen = {f["lag"] for f in findings}
    assert lags_seen.issubset({0, 1})


def test_compute_correlations_caches():
    """Re-call con misma input es cache hit (sin re-cálculo)."""
    n = 25
    metrics = {
        "a": {f"2026-04-{(i % 30) + 1:02d}": float(i) for i in range(n)},
        "b": {f"2026-04-{(i % 30) + 1:02d}": float(i * 2) for i in range(n)},
    }
    # Limpiar cache primero (test isolation).
    csp._CACHE.clear()
    f1 = csp.compute_correlations(metrics, lags=(0,))
    assert len(csp._CACHE) == 1
    f2 = csp.compute_correlations(metrics, lags=(0,))
    assert f1 is f2  # mismo objeto = cache hit


# ── Collector registry ────────────────────────────────────────────────────


def test_known_metrics_includes_core():
    """Los 12 collectors core están registrados."""
    names = csp.known_metrics()
    expected = {
        "mood_score", "mood_self_report",
        "sleep_quality", "sleep_duration_h", "sleep_awakenings",
        "sleep_deep_pct", "wakeup_mood",
        "spotify_minutes", "spotify_distinct_tracks",
        "queries_total", "queries_existential",
        "wa_outbound_avg_chars",
    }
    missing = expected - set(names)
    assert not missing, f"missing collectors: {missing}"


def test_metric_label_humanizes():
    assert "mood" in csp.metric_label("mood_score").lower()
    assert csp.metric_label("unknown_metric") == "unknown_metric"


# ── Smoke contra DB tmp ───────────────────────────────────────────────────


_SCORE_DAILY_DDL = """
CREATE TABLE rag_mood_score_daily (
    date TEXT PRIMARY KEY,
    score REAL NOT NULL,
    n_signals INTEGER NOT NULL,
    sources_used TEXT,
    top_evidence TEXT,
    updated_at REAL NOT NULL
)
"""


_SLEEP_DDL = """
CREATE TABLE rag_sleep_sessions (
    pk INTEGER NOT NULL,
    uuid TEXT PRIMARY KEY,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    date TEXT NOT NULL,
    is_nap INTEGER NOT NULL DEFAULT 0,
    is_edited INTEGER NOT NULL DEFAULT 0,
    quality REAL,
    fatigue REAL,
    wakeup_mood INTEGER,
    awakenings INTEGER,
    snoozes INTEGER,
    time_awake_s REAL,
    deep_s REAL,
    light_s REAL,
    rem_s REAL,
    deep_pct REAL,
    rem_pct REAL,
    time_to_sleep_s REAL,
    device TEXT,
    used_watch INTEGER,
    tz TEXT,
    source_file TEXT,
    ingested_at REAL NOT NULL
)
"""


@pytest.fixture
def tmp_telemetry(tmp_path, monkeypatch):
    """Fixture con todas las tablas relevantes para el engine."""
    db = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_SCORE_DAILY_DDL)
    conn.execute(_SLEEP_DDL)
    conn.execute("""
        CREATE TABLE rag_mood_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL, date TEXT NOT NULL,
            source TEXT NOT NULL, signal_kind TEXT NOT NULL,
            value REAL NOT NULL, weight REAL NOT NULL DEFAULT 1.0,
            evidence TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE rag_spotify_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id TEXT NOT NULL, name TEXT, artist TEXT,
            album TEXT, state TEXT, duration_ms INTEGER,
            first_seen REAL NOT NULL, last_seen REAL NOT NULL,
            date TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE rag_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, cmd TEXT, q TEXT NOT NULL,
            session TEXT
        )
    """)
    conn.commit()
    conn.close()

    @contextlib.contextmanager
    def _fake_conn():
        c = sqlite3.connect(str(db))
        try:
            yield c
            c.commit()
        finally:
            c.close()

    monkeypatch.setattr(rag, "_ragvec_state_conn", _fake_conn)
    # Reset caches.
    csp._CACHE.clear()
    yield db


def test_collect_mood_score_reads_daily_table(tmp_telemetry):
    today = _today()
    yesterday = _date_offset(1)
    conn = sqlite3.connect(str(tmp_telemetry))
    conn.execute(
        "INSERT INTO rag_mood_score_daily(date, score, n_signals, updated_at) "
        "VALUES(?, ?, ?, ?), (?, ?, ?, ?)",
        (today, 0.5, 4, 0, yesterday, -0.3, 3, 0),
    )
    conn.commit()
    conn.close()
    result = csp._c_mood_score(yesterday, today)
    assert result == {today: 0.5, yesterday: -0.3}


def test_collect_skips_zero_signal_days(tmp_telemetry):
    """Días con n_signals=0 NO se incluyen (gap, no neutro)."""
    today = _today()
    conn = sqlite3.connect(str(tmp_telemetry))
    conn.execute(
        "INSERT INTO rag_mood_score_daily(date, score, n_signals, updated_at) "
        "VALUES(?, ?, ?, ?)",
        (today, 0.0, 0, 0),
    )
    conn.commit()
    conn.close()
    assert csp._c_mood_score(today, today) == {}


def test_collect_daily_metrics_returns_all(tmp_telemetry):
    """collect_daily_metrics devuelve dict con todas las keys conocidas
    (aunque algunas estén vacías)."""
    result = csp.collect_daily_metrics(days=7)
    assert set(result.keys()) >= set(csp.known_metrics())


def test_patterns_summary_with_real_data(tmp_telemetry):
    """Smoke end-to-end: insertar data correlacionada y ver findings."""
    conn = sqlite3.connect(str(tmp_telemetry))
    # 25 días con sleep_quality y mood_score correlacionados (r positivo).
    for offset in range(25):
        date = _date_offset(offset)
        sleep_q = 0.4 + (offset % 7) * 0.05  # variación
        mood_s = (sleep_q - 0.5) * 2  # correlación cercana a 1
        conn.execute(
            "INSERT INTO rag_mood_score_daily(date, score, n_signals, updated_at) "
            "VALUES(?, ?, ?, ?)",
            (date, mood_s, 3, 0),
        )
        conn.execute(
            "INSERT INTO rag_sleep_sessions(pk, uuid, start_ts, end_ts, date, "
            "quality, ingested_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (offset, f"u-{offset}", 0, 28800, date, sleep_q, 0),
        )
    conn.commit()
    conn.close()

    summary = csp.patterns_summary(days=30, top=5, lags=(0,))
    # Esperamos al menos un finding para sleep_quality ↔ mood_score.
    pairs = [f["pair"] for f in summary["top"]]
    has_sleep_mood = any(
        ("sleep_quality" in p and "mood_score" in p) for p in pairs
    )
    assert has_sleep_mood, f"esperaba sleep_quality↔mood_score en top, got {pairs}"
    assert summary["n_findings"] >= 1
    assert ("sleep_quality", 25) in summary["metrics_with_data"] or \
           ("mood_score", 25) in summary["metrics_with_data"]
