"""Tests para `web/server.py:_fetch_mood` — payload del panel `p-mood`.

Cubre:
1. Feature off → None (panel hidden client-side).
2. Sin row hoy → None (daemon no corrió aún).
3. Row con n_signals=0 → None (no contaminamos panel con neutro).
4. Row con data → dict completo {score, n_signals, sources_used,
   trend, week_avg, drift, top_evidence, spark_score_14d, spark_dates_14d}.
5. Sparkline: 14 fechas, fechas con n_signals=0 quedan como null
   (gap visual), fechas con data quedan con score.
6. trend "improving" / "declining" / "stable" según delta vs week_avg.
7. Excepción al leer DB → None (silent-fail, no rompe el dashboard).

NB: estos tests se enfocan en _fetch_mood. Tests del frontend (renderMood
en home.v2.js) se ejercitan via webapp-testing en E2E si lo requerimos
después; el unit del backend cubre el contrato del payload.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Asegurar que `web/` es importable (el repo lo agrega via PYTHONPATH).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "web"))


@pytest.fixture
def fetch_mood():
    """Importa _fetch_mood freshly. El módulo `web.server` es pesado
    (carga FastAPI app), así que lazy-import."""
    from web.server import _fetch_mood
    return _fetch_mood


@pytest.fixture
def mood_enabled(monkeypatch):
    monkeypatch.setenv("RAG_MOOD_ENABLED", "1")


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _date_offset(offset_days: int) -> str:
    return (datetime.now() - timedelta(days=offset_days)).strftime("%Y-%m-%d")


# ── Sin score hoy ────────────────────────────────────────────────────────
# NB: _fetch_mood es read-only — NO chequea RAG_MOOD_ENABLED env var.
# Si hay datos en rag_mood_score_daily los muestra. El gate del feature
# (env var + state file) vive en el daemon que es el único writer.


def test_fetch_mood_returns_none_when_no_score_today(mood_enabled, monkeypatch, fetch_mood):
    """Daemon nunca corrió → no hay row para hoy → None."""
    from rag import mood as _mood
    monkeypatch.setattr(_mood, "get_score_for_date", lambda _d: None)
    monkeypatch.setattr(_mood, "get_recent_scores", lambda days=14: [])
    monkeypatch.setattr(_mood, "recent_drift", lambda **kw: {
        "drifting": False, "n_consecutive": 0, "avg_score": 0.0,
    })
    assert fetch_mood() is None


def test_fetch_mood_returns_none_when_score_has_zero_signals(
    mood_enabled, monkeypatch, fetch_mood,
):
    """Row presente pero n_signals=0 → None (no contaminamos panel)."""
    from rag import mood as _mood
    monkeypatch.setattr(_mood, "get_score_for_date", lambda _d: {
        "date": _today_str(), "score": 0.0, "n_signals": 0,
        "sources_used": [], "top_evidence": [], "updated_at": 0,
    })
    monkeypatch.setattr(_mood, "get_recent_scores", lambda days=14: [])
    monkeypatch.setattr(_mood, "recent_drift", lambda **kw: {
        "drifting": False, "n_consecutive": 0, "avg_score": 0.0,
    })
    assert fetch_mood() is None


# ── Payload completo ─────────────────────────────────────────────────────


def test_fetch_mood_returns_full_payload(mood_enabled, monkeypatch, fetch_mood):
    """Score con data → payload con todas las keys que renderMood espera."""
    from rag import mood as _mood
    today = _today_str()
    monkeypatch.setattr(_mood, "get_score_for_date", lambda _d: {
        "date": today, "score": -0.5, "n_signals": 4,
        "sources_used": ["spotify", "journal"],
        "top_evidence": [
            {"source": "journal", "signal_kind": "keyword_negative",
             "value": -0.7, "weight": 1.0, "evidence": {"keywords": ["bajón"]}},
            {"source": "spotify", "signal_kind": "artist_mood_lookup",
             "value": -0.4, "weight": 1.0, "evidence": {}},
        ],
        "updated_at": 0,
    })
    monkeypatch.setattr(_mood, "get_recent_scores", lambda days=14: [
        {"date": today, "score": -0.5, "n_signals": 4},
        {"date": _date_offset(1), "score": -0.2, "n_signals": 3},
        {"date": _date_offset(2), "score": +0.1, "n_signals": 3},
    ])
    monkeypatch.setattr(_mood, "recent_drift", lambda **kw: {
        "drifting": False, "n_consecutive": 1,
        "avg_score": -0.5, "dates": [today], "reason": "only_1_days",
    })

    result = fetch_mood()
    assert result is not None
    # Required keys que renderMood lee.
    expected_keys = {
        "score", "n_signals", "sources_used", "trend", "week_avg",
        "drift", "top_evidence", "spark_score_14d", "spark_dates_14d",
    }
    assert expected_keys.issubset(result.keys()), (
        f"missing keys: {expected_keys - result.keys()}"
    )
    assert result["score"] == -0.5
    assert result["n_signals"] == 4
    assert "spotify" in result["sources_used"]
    # week_avg = mean(-0.5, -0.2, +0.1) = -0.2
    assert result["week_avg"] == pytest.approx(-0.2, abs=0.01)
    # delta = -0.5 - (-0.2) = -0.3 → declining (delta < -0.2)
    assert result["trend"] == "declining"
    assert result["drift"]["drifting"] is False
    assert len(result["top_evidence"]) == 2


def test_sparkline_has_14_entries_chronological(mood_enabled, monkeypatch, fetch_mood):
    """spark_score_14d devuelve siempre 14 entries (oldest → newest).
    Días sin n_signals → None (gap visual)."""
    from rag import mood as _mood
    today = _today_str()
    # Solo 3 días tienen data; los otros 11 deben ser null en el sparkline.
    recent = [
        {"date": today, "score": -0.3, "n_signals": 2},
        {"date": _date_offset(1), "score": -0.1, "n_signals": 1},
        {"date": _date_offset(5), "score": +0.4, "n_signals": 3},
    ]
    monkeypatch.setattr(_mood, "get_score_for_date", lambda _d: recent[0])
    monkeypatch.setattr(_mood, "get_recent_scores", lambda days=14: recent)
    monkeypatch.setattr(_mood, "recent_drift", lambda **kw: {
        "drifting": False, "n_consecutive": 0, "avg_score": 0.0,
    })

    result = fetch_mood()
    assert len(result["spark_score_14d"]) == 14
    assert len(result["spark_dates_14d"]) == 14
    # Último entry = today.
    assert result["spark_dates_14d"][-1] == today
    assert result["spark_score_14d"][-1] == -0.3
    # Día -1 (ayer).
    assert result["spark_dates_14d"][-2] == _date_offset(1)
    assert result["spark_score_14d"][-2] == -0.1
    # Día -5.
    assert result["spark_dates_14d"][8] == _date_offset(5)
    assert result["spark_score_14d"][8] == 0.4
    # Días sin data → null.
    assert result["spark_score_14d"][0] is None  # offset -13


def test_sparkline_skips_zero_signal_days(mood_enabled, monkeypatch, fetch_mood):
    """Día con n_signals=0 (daemon corrió pero nada matched) →
    aparece como null en sparkline (gap), NO como score=0 (que se
    confundiría con neutro)."""
    from rag import mood as _mood
    today = _today_str()
    yesterday = _date_offset(1)
    recent = [
        {"date": today, "score": -0.4, "n_signals": 3},
        {"date": yesterday, "score": 0.0, "n_signals": 0},  # gap
    ]
    monkeypatch.setattr(_mood, "get_score_for_date", lambda _d: recent[0])
    monkeypatch.setattr(_mood, "get_recent_scores", lambda days=14: recent)
    monkeypatch.setattr(_mood, "recent_drift", lambda **kw: {
        "drifting": False, "n_consecutive": 0, "avg_score": 0.0,
    })

    result = fetch_mood()
    # Today con score, yesterday con null (gap).
    assert result["spark_score_14d"][-1] == -0.4
    assert result["spark_score_14d"][-2] is None


# ── Trend ────────────────────────────────────────────────────────────────


def test_trend_improving_when_delta_positive(mood_enabled, monkeypatch, fetch_mood):
    from rag import mood as _mood
    today = _today_str()
    monkeypatch.setattr(_mood, "get_score_for_date", lambda _d: {
        "date": today, "score": +0.6, "n_signals": 3,
        "sources_used": ["spotify"], "top_evidence": [], "updated_at": 0,
    })
    monkeypatch.setattr(_mood, "get_recent_scores", lambda days=14: [
        {"date": today, "score": +0.6, "n_signals": 3},
        {"date": _date_offset(1), "score": -0.1, "n_signals": 2},
    ])
    monkeypatch.setattr(_mood, "recent_drift", lambda **kw: {
        "drifting": False, "n_consecutive": 0, "avg_score": 0.0,
    })
    result = fetch_mood()
    assert result["trend"] == "improving"


def test_trend_stable_when_delta_small(mood_enabled, monkeypatch, fetch_mood):
    from rag import mood as _mood
    today = _today_str()
    monkeypatch.setattr(_mood, "get_score_for_date", lambda _d: {
        "date": today, "score": -0.05, "n_signals": 3,
        "sources_used": ["spotify"], "top_evidence": [], "updated_at": 0,
    })
    monkeypatch.setattr(_mood, "get_recent_scores", lambda days=14: [
        {"date": today, "score": -0.05, "n_signals": 3},
        {"date": _date_offset(1), "score": +0.0, "n_signals": 2},
    ])
    monkeypatch.setattr(_mood, "recent_drift", lambda **kw: {
        "drifting": False, "n_consecutive": 0, "avg_score": 0.0,
    })
    result = fetch_mood()
    assert result["trend"] == "stable"


# ── Silent-fail ──────────────────────────────────────────────────────────


def test_silent_fail_when_get_score_raises(mood_enabled, monkeypatch, fetch_mood):
    """Si rag.mood tira excepción, _fetch_mood devuelve None en lugar
    de propagar. Crítico: el dashboard NO debería romperse si la DB
    está locked o el módulo mood se rompió temporalmente."""
    from rag import mood as _mood
    def _broken(*args, **kwargs):
        raise RuntimeError("DB locked")
    monkeypatch.setattr(_mood, "get_score_for_date", _broken)
    assert fetch_mood() is None


# ── Drift bucket shape ───────────────────────────────────────────────────


def test_drift_active_passes_through(mood_enabled, monkeypatch, fetch_mood):
    """Si recent_drift devuelve drifting=true, el bucket queda con la
    info para que el frontend muestre el warning ámbar."""
    from rag import mood as _mood
    today = _today_str()
    monkeypatch.setattr(_mood, "get_score_for_date", lambda _d: {
        "date": today, "score": -0.6, "n_signals": 4,
        "sources_used": ["journal", "spotify"],
        "top_evidence": [], "updated_at": 0,
    })
    monkeypatch.setattr(_mood, "get_recent_scores", lambda days=14: [
        {"date": today, "score": -0.6, "n_signals": 4},
        {"date": _date_offset(1), "score": -0.5, "n_signals": 3},
        {"date": _date_offset(2), "score": -0.55, "n_signals": 3},
    ])
    monkeypatch.setattr(_mood, "recent_drift", lambda **kw: {
        "drifting": True, "n_consecutive": 3, "avg_score": -0.55,
        "dates": [_date_offset(2), _date_offset(1), today],
        "reason": None,
    })

    result = fetch_mood()
    assert result["drift"]["drifting"] is True
    assert result["drift"]["n_consecutive"] == 3
    assert result["drift"]["avg_score"] == pytest.approx(-0.55, abs=0.01)


# ── Frontend bundle smoke ────────────────────────────────────────────────


def test_home_v2_bundle_includes_mood_render():
    """El bundle JS servido al browser debe contener los nuevos
    helpers que renderizan el panel `p-mood` con buttons + tooltips +
    placeholder. Smoke estático contra el archivo en disco — si alguien
    rompe el render, este test salta antes de que lo veamos en prod."""
    js_path = Path(__file__).resolve().parent.parent / "web" / "static" / "home.v2.js"
    assert js_path.exists(), f"missing {js_path}"
    js = js_path.read_text(encoding="utf-8")
    # Funciones nuevas del commit de UI mejorada.
    for sym in (
        "renderMoodSparkline",
        "MOOD_SELF_REPORT_OPTIONS",
        "mood-self-btn",
        "data-self-report",
        "spark-zero",
        "mood-spark-placeholder",
    ):
        assert sym in js, f"missing JS symbol: {sym}"
    # Renderer principal sigue cableado al render loop.
    assert "renderMood(payload)" in js
    # POST /api/mood reusado para el self-report.
    assert "/api/mood" in js


def test_home_v2_html_has_mood_panel_with_aria_live():
    """El panel `p-mood` debe tener `aria-live="polite"` para que screen
    readers anuncien updates (cumplir Web Interface Guidelines #async)."""
    html_path = Path(__file__).resolve().parent.parent / "web" / "static" / "home.v2.html"
    html = html_path.read_text(encoding="utf-8")
    assert 'id="p-mood"' in html
    # aria-live debe estar sobre el data-body del panel mood (no en otro).
    # Usamos un slice alrededor del id para no falsear el match.
    idx = html.index('id="p-mood"')
    panel_block = html[idx:idx + 1000]
    assert 'aria-live="polite"' in panel_block, (
        "p-mood data-body falta aria-live=\"polite\""
    )


def test_home_v2_css_honors_reduced_motion():
    """El CSS debe respetar `prefers-reduced-motion` para los buttons
    de self-report (cumplir Web Interface Guidelines #animation).

    El home.v2 ya tiene un `@media (prefers-reduced-motion: reduce)`
    global que aplica `transition-duration: 0.01ms !important` a `*` —
    eso de por sí cubre los buttons. Pero como buena práctica, el
    bloque CSS específico de `.mood-self-btn` también declara su
    propio media query. Buscamos el último para no chocar con el
    global."""
    css_path = Path(__file__).resolve().parent.parent / "web" / "static" / "home.v2.css"
    css = css_path.read_text(encoding="utf-8")
    assert "@media (prefers-reduced-motion: reduce)" in css
    # Hay 2+ media queries — buscar el ÚLTIMO que es el específico
    # del mood (el primero es global, el segundo el del panel).
    pos = css.rfind("@media (prefers-reduced-motion: reduce)")
    block = css[pos:pos + 600]
    assert ".mood-self-btn" in block, (
        "último @media reduced-motion no incluye .mood-self-btn"
    )
    # `transition: all` está prohibido por web guidelines — verificamos
    # que NO usemos esa shortcut en .mood-self-btn.
    btn_pos = css.index(".mood-self-btn {")
    btn_block = css[btn_pos:btn_pos + 800]
    assert "transition: all" not in btn_block, (
        ".mood-self-btn usa `transition: all` (prohibido por guidelines)"
    )
    # Y `touch-action: manipulation` está (mejora touch-screens).
    assert "touch-action: manipulation" in btn_block
