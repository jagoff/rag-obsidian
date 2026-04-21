"""Tests for `confidence_badge()` + `score_bar()` — UX helpers calibrados
2026-04-21 contra la distribución real de `rag_queries.top_score` (n=904).

Pre-calibración los thresholds asumían score range [-5, 10] y nunca
triggereaban "alta" (p99 real es 1.08, no llega al 3.0). Post-calibración:
  baja < 0.10 · media 0.10-0.49 · alta >= 0.50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

import rag  # noqa: E402


# ── confidence_badge thresholds ────────────────────────────────────────────


@pytest.mark.parametrize("score", [-1.0, -0.5, 0.0, 0.05, 0.09, 0.099])
def test_badge_baja_below_low_threshold(score):
    """Scores estrictamente < 0.10 → baja · rojo.

    Threshold alineado con `CONFIDENCE_DEEP_THRESHOLD` del backend
    (abajo de este valor auto-deep retrieval fires, vault shaky)."""
    emoji, label = rag.confidence_badge(score)
    assert emoji == "🔴"
    assert label.startswith("baja")


@pytest.mark.parametrize("score", [0.10, 0.14, 0.30, 0.48, 0.49, 0.499])
def test_badge_media_in_middle_band(score):
    """Scores en [0.10, 0.50) → media · amarillo. Cubre ~50% del tráfico
    (p25=0.09, p75=0.48 en la muestra observada)."""
    emoji, label = rag.confidence_badge(score)
    assert emoji == "🟡"
    assert label.startswith("media")


@pytest.mark.parametrize("score", [0.50, 0.70, 0.93, 1.0, 1.12, 5.0])
def test_badge_alta_above_mid_threshold(score):
    """Scores >= 0.50 → alta · verde. ~p75 y arriba en la distribución real."""
    emoji, label = rag.confidence_badge(score)
    assert emoji == "🟢"
    assert label.startswith("alta")


def test_badge_precision_is_2_decimals():
    """Label muestra 2 decimales (0.30 no 0.3) — suficiente precisión para
    distinguir scores cercanos al boundary sin abrumar visualmente."""
    _, label = rag.confidence_badge(0.5)
    assert "0.50" in label
    _, label = rag.confidence_badge(0.14)
    assert "0.14" in label


def test_badge_constants_are_exposed():
    """Los thresholds son constantes module-level — si alguien los cambia
    los tests fallan por los parametrize hardcoded, forzando re-bench."""
    assert rag.SCORE_BADGE_LOW_HIGH == 0.10
    assert rag.SCORE_BADGE_MID_HIGH == 0.50


# ── score_bar filling ──────────────────────────────────────────────────────


@pytest.mark.parametrize("score, expected_filled", [
    (-5.0, 0),   # negative saturates at 0 (sentinel / vault vacío)
    (0.0,  0),
    (0.05, 0),   # rounds down — Python banker's rounding `round(0.25) = 0`
    (0.10, 0),   # 0.5 cells rounds to 0 (banker) — OK, tone still "low"
    (0.12, 1),   # just past threshold where round hits 1
    (0.30, 2),   # p50 band → middle
    (0.50, 2),   # boundary alta — round(2.5) = 2 (banker)
    (0.51, 3),   # just past boundary
    (0.70, 4),   # p85 band → near full
    (0.93, 5),   # round(4.65) = 5
    (1.00, 5),   # full bar
    (1.20, 5),   # saturates above 1.0
    (5.0,  5),
])
def test_score_bar_filling(score, expected_filled):
    """Linear mapping [0, 1.0] → 5 cells con saturación arriba/abajo del range."""
    bar = rag.score_bar(score)
    filled = bar.count("■")
    assert filled == expected_filled, f"score={score}: expected {expected_filled} cells, got {filled}"


def test_score_bar_width_is_respected():
    """Custom width param sigue funcionando post-calibración (usado en CLI
    render_sources + web bar emission)."""
    bar = rag.score_bar(0.5, width=10)
    assert len(bar) == 10


def test_score_bar_always_5_cells_by_default():
    """Width default = 5 — UI existente asume esa longitud."""
    bar = rag.score_bar(0.5)
    assert len(bar) == 5
    # Total = filled + empty
    assert bar.count("■") + bar.count("□") == 5


def test_score_bar_max_constant_is_exposed():
    """Regresión: si alguien bumpea SCORE_BAR_MAX sin re-bench, los tests
    fallan por los expected_filled hardcoded arriba."""
    assert rag.SCORE_BAR_MAX == 1.0
