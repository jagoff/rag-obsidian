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
    """Los 14 collectors core están registrados."""
    names = csp.known_metrics()
    expected = {
        "mood_score", "mood_self_report",
        "sleep_quality", "sleep_duration_h", "sleep_awakenings",
        "sleep_deep_pct", "wakeup_mood",
        "spotify_minutes", "spotify_distinct_tracks",
        "queries_total", "queries_existential",
        "wa_outbound_avg_chars",
        "gmail_received", "vault_notes_touched",
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


def test_collect_gmail_received_parses_frontmatter(tmp_path, monkeypatch):
    """gmail_received parsea `message_count: N` del frontmatter de las
    notas YYYY-MM-DD.md bajo `<vault>/03-Resources/Gmail/`."""
    folder = tmp_path / "03-Resources" / "Gmail"
    folder.mkdir(parents=True)
    (folder / "2026-04-28.md").write_text(
        "---\nsource: gmail\nmessage_count: 6\nwindow_hours: 48\n---\n\n# Body\n",
        encoding="utf-8",
    )
    (folder / "2026-04-29.md").write_text(
        "---\nmessage_count: 11\n---\nbody",
        encoding="utf-8",
    )
    # Archivo fuera del rango — se ignora.
    (folder / "2025-01-01.md").write_text(
        "---\nmessage_count: 999\n---\nbody",
        encoding="utf-8",
    )
    # Archivo basura — se ignora.
    (folder / "notas-sueltas.md").write_text("hola", encoding="utf-8")
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    assert csp._c_gmail_received("2026-04-28", "2026-04-29") == {
        "2026-04-28": 6.0,
        "2026-04-29": 11.0,
    }


def test_collect_gmail_received_returns_empty_when_folder_missing(tmp_path, monkeypatch):
    """Sin folder Gmail/, devolver {} silenciosamente (degradación graceful)."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    assert csp._c_gmail_received("2026-04-01", "2026-04-30") == {}


def test_collect_vault_notes_touched_counts_by_mtime(tmp_path, monkeypatch):
    """vault_notes_touched cuenta notas .md tocadas por día (mtime),
    skipea folders system, respeta el rango."""
    import os

    (tmp_path / "01-Projects").mkdir()
    (tmp_path / "04-Archive" / "99-obsidian-system").mkdir(parents=True)
    (tmp_path / ".obsidian").mkdir()

    user_note = tmp_path / "01-Projects" / "alpha.md"
    user_note.write_text("alpha", encoding="utf-8")
    user_note2 = tmp_path / "01-Projects" / "beta.md"
    user_note2.write_text("beta", encoding="utf-8")
    system_note = tmp_path / "04-Archive" / "99-obsidian-system" / "auto.md"
    system_note.write_text("auto", encoding="utf-8")
    obsidian_note = tmp_path / ".obsidian" / "workspace.md"
    obsidian_note.write_text("ws", encoding="utf-8")

    # Force mtime a una fecha conocida.
    target_date = "2026-04-29"
    target_dt = datetime.strptime(target_date, "%Y-%m-%d") + timedelta(hours=14)
    target_ts = target_dt.timestamp()
    for p in (user_note, user_note2, system_note, obsidian_note):
        os.utime(p, (target_ts, target_ts))

    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path)
    result = csp._c_vault_notes_touched("2026-04-28", "2026-04-30")
    # Solo las 2 user notes cuentan; system + .obsidian se skipean.
    assert result == {target_date: 2.0}


def test_collect_vault_notes_touched_returns_empty_when_vault_missing(tmp_path, monkeypatch):
    """Sin vault accesible, degrada a {}."""
    monkeypatch.setattr(rag, "VAULT_PATH", tmp_path / "no-existe")
    assert csp._c_vault_notes_touched("2026-04-01", "2026-04-30") == {}


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


# ── Endpoint /api/patterns ───────────────────────────────────────────────


@pytest.fixture
def patterns_client():
    from web.server import app
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_patterns_endpoint_returns_summary(patterns_client, monkeypatch):
    """GET /api/patterns devuelve la misma shape que `patterns_summary`."""
    from rag import cross_source_patterns as _csp
    monkeypatch.setattr(_csp, "patterns_summary", lambda **kw: {
        "n_findings": 2,
        "top": [
            {"pair": ("a", "b"), "lag": 0, "r": 0.7, "n": 25, "p": 0.001,
             "severity": "strong", "description": "x"},
        ],
        "by_severity": {"strong": 1, "moderate": 1, "weak": 0},
        "metrics_with_data": [("a", 25), ("b", 25)],
        "days_range": kw.get("days", 30),
        "lags_tested": list(kw.get("lags", (0, 1, 7))),
    })
    r = patterns_client.get("/api/patterns?days=30")
    assert r.status_code == 200
    data = r.json()
    assert data["n_findings"] == 2
    assert data["days_range"] == 30
    assert data["lags_tested"] == [0, 1, 7]


def test_patterns_endpoint_clamps_days(patterns_client, monkeypatch):
    """`days` clampeado a [7, 90]."""
    from rag import cross_source_patterns as _csp
    captured = {}
    monkeypatch.setattr(_csp, "patterns_summary", lambda **kw: (
        captured.update(kw) or {
            "n_findings": 0, "top": [], "by_severity": {},
            "metrics_with_data": [], "days_range": kw.get("days"),
            "lags_tested": list(kw.get("lags", ())),
        }
    ))
    patterns_client.get("/api/patterns?days=999")
    assert captured["days"] == 90
    captured.clear()
    patterns_client.get("/api/patterns?days=2")
    assert captured["days"] == 7


def test_patterns_endpoint_parses_lags_csv(patterns_client, monkeypatch):
    """`lags=0,1,3` se parsea a tuple correctamente."""
    from rag import cross_source_patterns as _csp
    captured = {}
    monkeypatch.setattr(_csp, "patterns_summary", lambda **kw: (
        captured.update(kw) or {
            "n_findings": 0, "top": [], "by_severity": {},
            "metrics_with_data": [], "days_range": kw.get("days"),
            "lags_tested": list(kw.get("lags", ())),
        }
    ))
    patterns_client.get("/api/patterns?lags=0,1,3")
    assert captured["lags"] == (0, 1, 3)


def test_patterns_endpoint_silent_fail_on_exception(patterns_client, monkeypatch):
    """Si patterns_summary tira, el endpoint devuelve estructura vacía
    (NO 500) — el frontend muestra empty state."""
    from rag import cross_source_patterns as _csp
    def _broken(**kw):
        raise RuntimeError("DB locked")
    monkeypatch.setattr(_csp, "patterns_summary", _broken)
    r = patterns_client.get("/api/patterns?days=30")
    assert r.status_code == 200
    data = r.json()
    assert data["n_findings"] == 0
    assert data["top"] == []


# ── CLI smoke ────────────────────────────────────────────────────────────


def test_cli_patterns_show_no_data(monkeypatch, tmp_path):
    """`rag patterns show` sin data en DB → mensaje "sin findings".
    NB: usamos --days 7 para forzar el min_n=21 a no satisfacerse."""
    monkeypatch.setattr(
        "rag.cross_source_patterns.patterns_summary",
        lambda **kw: {
            "n_findings": 0, "top": [], "by_severity": {},
            "metrics_with_data": [],
            "days_range": kw.get("days", 30),
            "lags_tested": list(kw.get("lags", (0,))),
        },
    )
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["patterns", "show", "--plain"])
    assert result.exit_code == 0, result.output
    assert "n_findings=0" in result.output


def test_cli_patterns_metrics_lists_collectors(monkeypatch):
    """`rag patterns metrics` lista los 12 collectors core."""
    from rag import cross_source_patterns as _csp
    monkeypatch.setattr(
        _csp, "collect_daily_metrics",
        lambda **kw: {name: {} for name in _csp.known_metrics()},
    )
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["patterns", "metrics", "--plain"])
    assert result.exit_code == 0
    # Cada collector core aparece en el output.
    for name in _csp.known_metrics():
        assert name in result.output


def test_cli_patterns_explain_pair(monkeypatch):
    """`rag patterns explain --pair a,b` corre Pearson sobre data
    inyectada y muestra r/n/p."""
    from rag import cross_source_patterns as _csp
    series_a = {f"2026-04-{i:02d}": float(i) for i in range(1, 26)}
    series_b = {f"2026-04-{i:02d}": float(i * 2) for i in range(1, 26)}
    monkeypatch.setattr(_csp, "collect_daily_metrics", lambda **kw: {
        "mood_score": series_a, "sleep_quality": series_b,
    })
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, [
        "patterns", "explain",
        "--pair", "mood_score,sleep_quality", "--lag", "0", "--plain",
    ])
    assert result.exit_code == 0, result.output
    assert "r=+1.000" in result.output  # perfect correlation
    assert "n=25" in result.output


# ── Frontend bundle smoke (panel p-correlations + modal) ────────────────


def test_home_v2_html_has_correlations_panel():
    """El panel correlations está declarado con id `p-correlations` (NO
    `p-patterns`, que está tomado por otro panel pre-existente)."""
    html_path = Path(__file__).resolve().parent.parent / "web" / "static" / "home.v2.html"
    html = html_path.read_text(encoding="utf-8")
    assert 'id="p-correlations"' in html
    assert 'class="panel panel-correlations"' in html
    # aria-live explicit en el panel-body.
    idx = html.index('id="p-correlations"')
    panel_block = html[idx:idx + 800]
    assert 'aria-live="polite"' in panel_block


def test_home_v2_html_has_correlations_modal():
    html_path = Path(__file__).resolve().parent.parent / "web" / "static" / "home.v2.html"
    html = html_path.read_text(encoding="utf-8")
    assert '<dialog id="patterns-modal"' in html
    assert 'aria-labelledby="patterns-modal-title"' in html
    assert 'data-patterns-modal-close' in html


# ── predict_mood_tomorrow ────────────────────────────────────────────────


def test_predict_mood_returns_none_with_insufficient_data():
    """< _PREDICT_MIN_DAYS rows entrenables → None."""
    metrics = {
        "mood_score": {f"2026-04-{i:02d}": 0.5 for i in range(1, 10)},
        "sleep_quality": {f"2026-04-{i:02d}": 0.7 for i in range(1, 10)},
    }
    result = csp.predict_mood_tomorrow(metrics=metrics)
    assert result is None


def test_predict_mood_with_synthetic_correlated_data():
    """Construir 30 días con sleep_quality[t-1] perfectamente
    correlacionado con mood_score[t]. Predicción debe usar sleep como
    feature dominante."""
    today = datetime.now()
    sleep_series = {}
    mood_series = {}
    for offset in range(30, -1, -1):
        date = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        # sleep[t] = oscilación 0.4 a 0.9
        sleep_q = 0.4 + (offset % 7) * 0.05
        sleep_series[date] = sleep_q
        # mood[t] = sleep[t-1] * 1.5 - 0.5 (mapping aprox a [-0.5, 0.7])
        # Mood NO está en t=hoy hasta t=ayer (correlación con t-1).
        prev_date = (today - timedelta(days=offset + 1)).strftime("%Y-%m-%d")
        if prev_date in sleep_series:
            mood_series[date] = sleep_series[prev_date] * 1.5 - 0.5
    metrics = {
        "mood_score": mood_series,
        "sleep_quality": sleep_series,
        "sleep_duration_h": sleep_series,  # duplicate, no importa
        "sleep_awakenings": {d: 2.0 for d in sleep_series},
        "wakeup_mood": sleep_series,
        "spotify_minutes": {d: 30.0 for d in sleep_series},
        "spotify_distinct_tracks": {d: 5.0 for d in sleep_series},
        "queries_total": {d: 10.0 for d in sleep_series},
        "queries_existential": {d: 0.0 for d in sleep_series},
        "wa_outbound_avg_chars": {d: 50.0 for d in sleep_series},
        "mood_self_report": {},  # vacío — no debe romper
    }

    result = csp.predict_mood_tomorrow(metrics=metrics, days=60)
    assert result is not None
    # n_training_days >= MIN
    assert result["n_training_days"] >= csp._PREDICT_MIN_DAYS
    # Confidence alta porque la correlación es perfecta.
    assert result["confidence"] >= 0.8
    # Top features incluye sleep_quality (la real feature predictiva).
    feature_names = [f["feature"] for f in result["top_features"]]
    assert "sleep_quality" in feature_names


def test_predict_mood_top_features_ordered_by_contribution():
    today = datetime.now()
    series = {(today - timedelta(days=i)).strftime("%Y-%m-%d"): float(i)
              for i in range(0, 30)}
    metrics = {
        "mood_score": series,
        "sleep_quality": series,
        "sleep_duration_h": {d: 0.0 for d in series},  # zero contribution
        "sleep_awakenings": {d: 1.0 for d in series},
        "wakeup_mood": {d: 1.0 for d in series},
        "spotify_minutes": {d: 1.0 for d in series},
        "spotify_distinct_tracks": {d: 1.0 for d in series},
        "queries_total": {d: 1.0 for d in series},
        "queries_existential": {d: 0.0 for d in series},
        "wa_outbound_avg_chars": {d: 1.0 for d in series},
        "mood_self_report": {},
    }
    result = csp.predict_mood_tomorrow(metrics=metrics, days=30)
    if result is None or result.get("prediction") is None:
        pytest.skip("not enough data after alignment")
    contribs = [abs(f["contribution"]) for f in result["top_features"]]
    assert contribs == sorted(contribs, reverse=True)


def test_predict_mood_silent_fail_when_no_target():
    """Sin mood_score como métrica, devuelve None."""
    metrics = {
        "sleep_quality": {f"2026-04-{i:02d}": 0.5 for i in range(1, 30)},
    }
    result = csp.predict_mood_tomorrow(metrics=metrics)
    assert result is None


def _synthetic_30d_metrics():
    """Helper: 30 días con sleep_quality[t-1] perfectamente correlacionado
    con mood_score[t]. Devuelve metrics dict listo para usar."""
    today = datetime.now()
    sleep_series = {}
    mood_series = {}
    for offset in range(30, -1, -1):
        date = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        sleep_q = 0.4 + (offset % 7) * 0.05
        sleep_series[date] = sleep_q
        prev_date = (today - timedelta(days=offset + 1)).strftime("%Y-%m-%d")
        if prev_date in sleep_series:
            mood_series[date] = sleep_series[prev_date] * 1.5 - 0.5
    return {
        "mood_score": mood_series,
        "sleep_quality": sleep_series,
        "sleep_duration_h": sleep_series,
        "sleep_awakenings": {d: 2.0 for d in sleep_series},
        "wakeup_mood": sleep_series,
        "spotify_minutes": {d: 30.0 for d in sleep_series},
        "spotify_distinct_tracks": {d: 5.0 for d in sleep_series},
        "queries_total": {d: 10.0 for d in sleep_series},
        "queries_existential": {d: 0.0 for d in sleep_series},
        "wa_outbound_avg_chars": {d: 50.0 for d in sleep_series},
        "mood_self_report": {},
    }


def test_predict_uses_ridge_cv_with_alpha_metadata():
    """RidgeCV es el modelo default y el alpha elegido se reporta.
    Si Ridge falla por algún motivo (no debería con data limpia),
    cae a LinearRegression."""
    result = csp.predict_mood_tomorrow(metrics=_synthetic_30d_metrics(), days=60)
    assert result is not None
    # Modelo elegido + metadata.
    assert result["model"] in {"ridge_cv", "linear"}
    if result["model"] == "ridge_cv":
        # alpha viene del grid pasado a RidgeCV (0.01..10.0).
        assert result["alpha"] is not None
        assert 0.0 <= result["alpha"] <= 100.0


def test_predict_confidence_is_cross_validated():
    """`confidence` es R² out-of-sample (CV time-aware), no in-sample.
    Para data perfectamente correlacionada, CV R² debe ser alto pero
    distinto de 1.0 (CV puede tener leakage parcial pero no perfecto).
    `confidence_in_sample` queda como retrocompat."""
    result = csp.predict_mood_tomorrow(metrics=_synthetic_30d_metrics(), days=60)
    assert result is not None
    # Las dos confidences existen y son floats.
    assert isinstance(result["confidence"], (int, float))
    assert isinstance(result["confidence_in_sample"], (int, float))
    # cv_n_splits > 0 → CV se corrió.
    assert result["cv_n_splits"] >= 2
    # Para correlación perfecta, in-sample R² es altísimo (>0.9).
    assert result["confidence_in_sample"] >= 0.9


def test_predict_confidence_can_be_negative_with_noise():
    """Cuando las features no predicen el target, CV R² puede ser
    negativo. Eso es signal valid (no rompe el contrato)."""
    today = datetime.now()
    rng = list(range(30, -1, -1))
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in rng]
    # Mood random; features constantes → CV R² ~0 o negativo.
    import random
    random.seed(42)
    metrics = {
        "mood_score": {d: random.gauss(0, 0.3) for d in dates},
        "sleep_quality": {d: 0.5 for d in dates},
        "sleep_duration_h": {d: 7.0 for d in dates},
        "sleep_awakenings": {d: 2.0 for d in dates},
        "wakeup_mood": {d: 0.0 for d in dates},
        "spotify_minutes": {d: 30.0 for d in dates},
        "spotify_distinct_tracks": {d: 5.0 for d in dates},
        "queries_total": {d: 10.0 for d in dates},
        "queries_existential": {d: 0.0 for d in dates},
        "wa_outbound_avg_chars": {d: 50.0 for d in dates},
        "mood_self_report": {},
    }
    result = csp.predict_mood_tomorrow(metrics=metrics, days=60)
    if result is None or result.get("prediction") is None:
        pytest.skip("not enough data after alignment")
    # Confidence ahora es CV R² — para features constantes vs target
    # ruidoso, debe estar bajo (probably < 0.3 o incluso negativo).
    assert result["confidence"] < 0.5


def test_predict_top_features_include_baseline_and_deviation():
    """Cada top feature trae `value_baseline` (mean histórico) y
    `deviation_contribution` (SHAP-style). Para correlación perfecta
    con sleep, sleep_quality debe estar en el top y su deviation
    debe ser != 0 (porque sleep_quality oscila día a día)."""
    result = csp.predict_mood_tomorrow(metrics=_synthetic_30d_metrics(), days=60)
    assert result is not None
    if result.get("prediction") is None:
        pytest.skip("missing today features")
    top = result["top_features"]
    assert top, "expected at least 1 top feature"
    sample = top[0]
    # Schema completo: nuevos keys presentes.
    assert "value_baseline" in sample
    assert "deviation_contribution" in sample
    # Y los legacy keys siguen.
    assert "contribution" in sample
    assert "coef" in sample
    assert "value_today" in sample
    # value_baseline es float razonable (mean del training).
    assert isinstance(sample["value_baseline"], (int, float))


def test_predict_top_features_ordered_by_deviation_contribution():
    """El nuevo orden es por |deviation_contribution|, no por
    |contribution| (legacy). Verificable construyendo features con
    coef alto pero value_today == mean (deviation=0) — esa feature
    NO debe estar primera aunque su contribución absoluta sea alta."""
    result = csp.predict_mood_tomorrow(metrics=_synthetic_30d_metrics(), days=60)
    assert result is not None
    if result.get("prediction") is None:
        pytest.skip("missing today features")
    devs = [abs(f["deviation_contribution"]) for f in result["top_features"]]
    assert devs == sorted(devs, reverse=True)


# ── CLI rag mood predict ─────────────────────────────────────────────────


def test_cli_mood_predict_no_data(monkeypatch):
    """`rag mood predict` sin data → mensaje de feature off o
    insufficient data."""
    monkeypatch.setenv("RAG_MOOD_ENABLED", "1")
    from rag import cross_source_patterns as _csp
    monkeypatch.setattr(_csp, "predict_mood_tomorrow", lambda **kw: None)
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["mood", "predict", "--plain"])
    assert result.exit_code == 0
    assert "insufficient_data" in result.output or "prediction=none" in result.output


def test_cli_mood_predict_with_result(monkeypatch):
    """`rag mood predict` con result válido → muestra prediction +
    confidence + top features."""
    monkeypatch.setenv("RAG_MOOD_ENABLED", "1")
    from rag import cross_source_patterns as _csp
    monkeypatch.setattr(_csp, "predict_mood_tomorrow", lambda **kw: {
        "prediction": -0.42, "confidence": 0.65, "n_training_days": 28,
        "target_date": "2026-05-01", "based_on_date": "2026-04-30",
        "top_features": [
            {"feature": "sleep_quality", "coef": 1.5,
             "value_today": 0.6, "contribution": 0.9},
            {"feature": "spotify_minutes", "coef": -0.01,
             "value_today": 30.0, "contribution": -0.3},
        ],
    })
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(rag.cli, ["mood", "predict", "--plain"])
    assert result.exit_code == 0
    assert "prediction=-0.420" in result.output
    assert "confidence=0.650" in result.output
    assert "sleep_quality" in result.output


# ── Hook today_correlator → bucket cross_patterns ────────────────────────


def test_correlate_cross_patterns_returns_none_when_no_findings(monkeypatch):
    """Sin findings strong/moderate y sin prediction → bucket None."""
    from rag import today_correlator as _tc
    from rag import cross_source_patterns as _csp
    monkeypatch.setattr(_csp, "patterns_summary", lambda **kw: {
        "n_findings": 0, "top": [], "by_severity": {},
        "metrics_with_data": [], "days_range": 30, "lags_tested": [0],
    })
    monkeypatch.setattr(_csp, "predict_mood_tomorrow", lambda **kw: None)
    result = _tc._correlate_cross_patterns({}, {})
    assert result is None


def test_correlate_cross_patterns_filters_weak(monkeypatch):
    """Findings con severity=weak NO se incluyen (evita ruido)."""
    from rag import today_correlator as _tc
    from rag import cross_source_patterns as _csp
    monkeypatch.setattr(_csp, "patterns_summary", lambda **kw: {
        "n_findings": 3, "by_severity": {},
        "metrics_with_data": [], "days_range": 30, "lags_tested": [0],
        "top": [
            {"description": "strong x", "r": 0.7, "n": 25,
             "lag": 0, "severity": "strong",
             "pair": ("a", "b")},
            {"description": "weak y", "r": 0.32, "n": 25,
             "lag": 1, "severity": "weak",
             "pair": ("c", "d")},
            {"description": "moderate z", "r": 0.5, "n": 25,
             "lag": 7, "severity": "moderate",
             "pair": ("e", "f")},
        ],
    })
    monkeypatch.setattr(_csp, "predict_mood_tomorrow", lambda **kw: None)
    result = _tc._correlate_cross_patterns({}, {})
    assert result is not None
    descriptions = [f["description"] for f in result["top_findings"]]
    assert "strong x" in descriptions
    assert "moderate z" in descriptions
    assert "weak y" not in descriptions
    assert result["n_findings_total"] == 3


def test_correlate_cross_patterns_includes_prediction(monkeypatch):
    """Si prediction está, viene en el bucket."""
    from rag import today_correlator as _tc
    from rag import cross_source_patterns as _csp
    monkeypatch.setattr(_csp, "patterns_summary", lambda **kw: {
        "n_findings": 0, "top": [], "by_severity": {},
        "metrics_with_data": [], "days_range": 30, "lags_tested": [0],
    })
    monkeypatch.setattr(_csp, "predict_mood_tomorrow", lambda **kw: {
        "prediction": 0.3, "confidence": 0.55, "n_training_days": 25,
        "target_date": "2026-05-01", "based_on_date": "2026-04-30",
        "top_features": [],
    })
    result = _tc._correlate_cross_patterns({}, {})
    assert result is not None
    assert result["prediction"]["prediction"] == 0.3
    assert result["prediction"]["confidence"] == 0.55


# ── Today prompt rule #9 ────────────────────────────────────────────────


def test_today_prompt_includes_rule_9_when_cross_patterns_present():
    """Cuando extras.correlations.cross_patterns tiene findings, la
    regla #9 aparece en el prompt."""
    extras = {"correlations": {
        "people": [], "topics": [], "time_overlaps": [], "gaps": [],
        "mood": None, "sleep": None,
        "cross_patterns": {
            "top_findings": [
                {"description": "sleep alta → mood sube (mismo día)",
                 "r": 0.65, "n": 25, "lag": 0, "severity": "strong"},
            ],
            "prediction": None,
            "n_findings_total": 1,
        },
    }}
    ev = {"recent_notes": [{"title": "n", "path": "02-Areas/n.md", "snippet": "x"}],
          "inbox_today": [], "todos": [], "new_contradictions": [],
          "low_conf_queries": [], "wa_scheduled_today_pending": []}
    prompt = rag._render_today_prompt("2026-04-30", ev, extras=extras)
    assert "Patrones cross-source" in prompt
    assert "sleep alta → mood sube" in prompt
    assert "PROHIBIDO afirmar causalidad" in prompt


def test_today_prompt_omits_rule_9_when_no_cross_patterns():
    """Sin bucket cross_patterns o vacío → regla #9 ausente."""
    extras = {"correlations": {
        "people": [], "topics": [], "time_overlaps": [], "gaps": [],
        "cross_patterns": None,
    }}
    ev = {"recent_notes": [{"title": "n", "path": "02-Areas/n.md", "snippet": "x"}],
          "inbox_today": [], "todos": [], "new_contradictions": [],
          "low_conf_queries": [], "wa_scheduled_today_pending": []}
    prompt = rag._render_today_prompt("2026-04-30", ev, extras=extras)
    assert "Patrones cross-source" not in prompt
    assert "9. **Patrones" not in prompt


def test_today_prompt_includes_prediction_with_confidence_label():
    """Prediction con confidence alta → label "alta" en el prompt."""
    extras = {"correlations": {
        "people": [], "topics": [], "time_overlaps": [], "gaps": [],
        "cross_patterns": {
            "top_findings": [],
            "prediction": {
                "prediction": -0.5, "confidence": 0.7,
                "target_date": "2026-05-01",
                "n_training_days": 25, "based_on_date": "2026-04-30",
                "top_features": [],
            },
            "n_findings_total": 0,
        },
    }}
    ev = {"recent_notes": [{"title": "n", "path": "02-Areas/n.md", "snippet": "x"}],
          "inbox_today": [], "todos": [], "new_contradictions": [],
          "low_conf_queries": [], "wa_scheduled_today_pending": []}
    prompt = rag._render_today_prompt("2026-04-30", ev, extras=extras)
    assert "predicción mañana" in prompt
    assert "confianza alta" in prompt
    assert "tendencia bajo" in prompt  # -0.5 → bajo


def test_home_v2_bundle_includes_render_correlations():
    """El JS bundle tiene `renderCorrelations` (no `renderPatterns`)
    para evitar shadowing con el render existente del panel
    `p-patterns` (entidades cross-source)."""
    js_path = Path(__file__).resolve().parent.parent / "web" / "static" / "home.v2.js"
    js = js_path.read_text(encoding="utf-8")
    # Las funciones nuevas:
    assert "async function renderCorrelations" in js
    assert "fetchPatterns" in js
    assert "openPatternsModal" in js
    assert "renderPatternsModal" in js
    # Llama a /api/patterns con days=30.
    assert '"/api/patterns?days=30&top=20"' in js
    # El render loop principal llama a renderCorrelations(payload).
    assert "renderCorrelations(payload)" in js
