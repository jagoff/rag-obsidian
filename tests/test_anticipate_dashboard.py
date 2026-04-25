"""Tests for `rag_anticipate.dashboard` — analytics del Anticipatory Agent.

Cubre las 4 funciones públicas:
- `fetch_metrics(days)` — agregados sobre rag_anticipate_candidates.
- `render_dashboard(days)` — render texto para CLI/log.
- `top_reasons_skipped(days, limit)` — agrupado por reason de los skipped.
- `signal_health(days)` — health check per-signal con statuses
  silent/stale/noisy/healthy.

Todos los tests aíslan el telemetry DB en `tmp_path` vía monkeypatching de
`rag.DB_PATH` ANTES de instanciar SqliteVecClient + invocar
`_ragvec_state_conn()` (mismo patrón que `tests/test_anticipate_agent.py`).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import rag
from rag_anticipate.dashboard import (
    fetch_metrics,
    render_dashboard,
    signal_health,
    top_reasons_skipped,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Tmp telemetry DB con la tabla `rag_anticipate_candidates` creada
    pero vacía. Mismo patrón que `state_db` en test_anticipate_agent.py.
    """
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    from rag import SqliteVecClient as _VecClient
    _VecClient(path=str(db_path))
    # Trigger DDL de las telemetry tables (incluye rag_anticipate_candidates).
    with rag._ragvec_state_conn() as _conn:
        pass
    return tmp_path


def _insert(ts: str, kind: str, score: float, dedup_key: str,
            selected: int, sent: int, reason: str = "",
            message_preview: str = "") -> None:
    """Helper para insertar una row sintética en rag_anticipate_candidates."""
    with rag._ragvec_state_conn() as conn:
        conn.execute(
            "INSERT INTO rag_anticipate_candidates "
            "(ts, kind, score, dedup_key, selected, sent, reason, message_preview) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, kind, score, dedup_key, selected, sent, reason, message_preview),
        )
        conn.commit()


def _now_offset(hours: float = 0.0, days: float = 0.0) -> str:
    """ISO ts relativo a now (negativo = pasado)."""
    dt = datetime.now() - timedelta(hours=hours, days=days)
    return dt.isoformat(timespec="seconds")


# ── fetch_metrics ────────────────────────────────────────────────────────────


def test_fetch_metrics_empty_db_returns_zeros(state_db):
    """DB con la tabla creada pero sin rows → totales en 0, by_kind vacío."""
    m = fetch_metrics(days=7)
    assert m["window_days"] == 7
    assert m["total_evaluated"] == 0
    assert m["total_selected"] == 0
    assert m["total_sent"] == 0
    assert m["by_kind"] == {}
    assert m["send_rate"] == 0.0
    assert m["selection_rate"] == 0.0


def test_fetch_metrics_totals_correct_with_rows(state_db):
    """Insertar N rows mixtas y validar los totales."""
    # 3 evaluadas, 2 selected, 1 sent — todos en ventana
    _insert(_now_offset(hours=1), "anticipate-calendar", 0.8, "k1", 1, 1, "ok")
    _insert(_now_offset(hours=2), "anticipate-calendar", 0.7, "k2", 1, 0, "skip:cap")
    _insert(_now_offset(hours=3), "anticipate-echo", 0.4, "k3", 0, 0, "skip:score")
    m = fetch_metrics(days=7)
    assert m["total_evaluated"] == 3
    assert m["total_selected"] == 2
    assert m["total_sent"] == 1


def test_fetch_metrics_groups_by_kind(state_db):
    """by_kind debe tener una entry por cada `kind` distinto."""
    _insert(_now_offset(hours=1), "anticipate-calendar", 0.8, "k1", 1, 1, "")
    _insert(_now_offset(hours=2), "anticipate-calendar", 0.6, "k2", 0, 0, "")
    _insert(_now_offset(hours=3), "anticipate-echo", 0.5, "k3", 1, 1, "")
    m = fetch_metrics(days=7)
    assert set(m["by_kind"].keys()) == {"anticipate-calendar", "anticipate-echo"}
    assert m["by_kind"]["anticipate-calendar"]["evaluated"] == 2
    assert m["by_kind"]["anticipate-calendar"]["selected"] == 1
    assert m["by_kind"]["anticipate-calendar"]["sent"] == 1
    assert m["by_kind"]["anticipate-echo"]["evaluated"] == 1
    assert m["by_kind"]["anticipate-echo"]["sent"] == 1


def test_fetch_metrics_avg_score_correct(state_db):
    """avg_score = AVG(score) por kind."""
    _insert(_now_offset(hours=1), "anticipate-calendar", 0.8, "k1", 0, 0)
    _insert(_now_offset(hours=2), "anticipate-calendar", 0.6, "k2", 0, 0)
    _insert(_now_offset(hours=3), "anticipate-calendar", 0.4, "k3", 0, 0)
    m = fetch_metrics(days=7)
    # Promedio de 0.8, 0.6, 0.4 = 0.6
    assert m["by_kind"]["anticipate-calendar"]["avg_score"] == pytest.approx(0.6)


def test_fetch_metrics_send_rate_is_sent_over_evaluated(state_db):
    """send_rate = total_sent / total_evaluated. Selection_rate igual."""
    # 4 rows: 2 sent, 3 selected.
    _insert(_now_offset(hours=1), "anticipate-calendar", 0.9, "k1", 1, 1)
    _insert(_now_offset(hours=2), "anticipate-calendar", 0.8, "k2", 1, 1)
    _insert(_now_offset(hours=3), "anticipate-echo", 0.7, "k3", 1, 0)
    _insert(_now_offset(hours=4), "anticipate-echo", 0.2, "k4", 0, 0)
    m = fetch_metrics(days=7)
    assert m["total_evaluated"] == 4
    assert m["total_sent"] == 2
    assert m["total_selected"] == 3
    assert m["send_rate"] == pytest.approx(0.5)
    assert m["selection_rate"] == pytest.approx(0.75)


def test_fetch_metrics_window_days_param_filters_old_rows(state_db):
    """Rows fuera de la ventana `days` no deben contar."""
    # row reciente
    _insert(_now_offset(hours=1), "anticipate-calendar", 0.8, "k1", 1, 1)
    # row vieja (10 días atrás)
    _insert(_now_offset(days=10), "anticipate-calendar", 0.7, "k2", 1, 1)
    # ventana de 7 días → solo cuenta la reciente
    m7 = fetch_metrics(days=7)
    assert m7["total_evaluated"] == 1
    # ventana de 30 días → cuenta las 2
    m30 = fetch_metrics(days=30)
    assert m30["total_evaluated"] == 2


# ── render_dashboard ─────────────────────────────────────────────────────────


def test_render_dashboard_contains_expected_strings(state_db):
    """Render con datos debe incluir título + totales + send rate + por-kind."""
    _insert(_now_offset(hours=1), "anticipate-calendar", 0.8, "k1", 1, 1, "")
    _insert(_now_offset(hours=2), "anticipate-echo", 0.6, "k2", 0, 0, "")
    out = render_dashboard(days=7)
    assert "Anticipate Dashboard (7 days)" in out
    assert "Total evaluated: 2" in out
    assert "Total selected:" in out
    assert "Total sent:" in out
    assert "Send rate:" in out
    assert "Selection rate:" in out
    assert "By signal:" in out
    assert "anticipate-calendar" in out
    assert "anticipate-echo" in out
    assert "avg_score=" in out


def test_render_dashboard_empty_case_says_no_data(state_db):
    """DB vacío → el render dice 'no data' en lugar de imprimir ceros."""
    out = render_dashboard(days=7)
    assert "Anticipate Dashboard (7 days)" in out
    assert "no data" in out.lower()


# ── top_reasons_skipped ──────────────────────────────────────────────────────


def test_top_reasons_skipped_groups_and_orders_desc(state_db):
    """Agrupa por reason (solo sent=0), ordenado desc por count."""
    # 3 con reason="score_too_low"
    _insert(_now_offset(hours=1), "anticipate-calendar", 0.2, "k1", 0, 0,
            reason="score_too_low")
    _insert(_now_offset(hours=2), "anticipate-calendar", 0.2, "k2", 0, 0,
            reason="score_too_low")
    _insert(_now_offset(hours=3), "anticipate-echo", 0.1, "k3", 0, 0,
            reason="score_too_low")
    # 2 con reason="dedup_seen"
    _insert(_now_offset(hours=4), "anticipate-echo", 0.5, "k4", 1, 0,
            reason="dedup_seen")
    _insert(_now_offset(hours=5), "anticipate-echo", 0.5, "k5", 1, 0,
            reason="dedup_seen")
    # 1 sent (no debe contar)
    _insert(_now_offset(hours=6), "anticipate-calendar", 0.9, "k6", 1, 1,
            reason="ok")

    reasons = top_reasons_skipped(days=7)
    # Esperamos primer score_too_low (3), luego dedup_seen (2). El "ok" no
    # aparece porque es de un row con sent=1.
    assert reasons[0] == ("score_too_low", 3)
    assert reasons[1] == ("dedup_seen", 2)
    # "ok" no aparece (era sent=1)
    assert all(r != "ok" for r, _ in reasons)


def test_top_reasons_skipped_respects_limit_param(state_db):
    """`limit` recorta el número de buckets devueltos."""
    for i in range(5):
        _insert(_now_offset(hours=i + 1), "anticipate-calendar", 0.1, f"k{i}",
                0, 0, reason=f"reason_{i}")
    out = top_reasons_skipped(days=7, limit=3)
    assert len(out) == 3


def test_top_reasons_skipped_empty_db_returns_empty_list(state_db):
    """DB sin rows → lista vacía (no raise)."""
    assert top_reasons_skipped(days=7) == []


# ── signal_health ────────────────────────────────────────────────────────────


def test_signal_health_silent_when_no_emits_ever(state_db):
    """Sin rows en toda la tabla → los kinds MVP aparecen como 'silent'."""
    health = signal_health(days=7)
    # Los 3 MVP signals SIEMPRE están en el output (son hardcoded en el módulo).
    assert "anticipate-calendar" in health
    assert health["anticipate-calendar"]["status"] == "silent"
    assert health["anticipate-calendar"]["last_emit"] is None
    assert health["anticipate-calendar"]["avg_score_7d"] == 0.0
    assert health["anticipate-calendar"]["send_rate"] == 0.0


def test_signal_health_noisy_when_low_avg_and_many_emits(state_db):
    """≥10 emits en ventana con avg_score < 0.3 → noisy."""
    # 12 emits con score bajísimo (avg ~0.2)
    for i in range(12):
        _insert(_now_offset(hours=i + 1), "anticipate-echo", 0.2, f"e{i}",
                0, 0, reason="score_too_low")
    health = signal_health(days=7)
    assert health["anticipate-echo"]["status"] == "noisy"
    assert health["anticipate-echo"]["avg_score_7d"] == pytest.approx(0.2)


def test_signal_health_stale_when_last_emit_old(state_db):
    """last_emit existe pero es ≥ window_days viejo + 0 emits en ventana → stale."""
    _insert(_now_offset(days=10), "anticipate-calendar", 0.8, "k_old", 1, 1,
            reason="ok")
    health = signal_health(days=7)
    # 0 emits en ventana de 7d, pero last_emit existe → stale.
    assert health["anticipate-calendar"]["status"] == "stale"
    assert health["anticipate-calendar"]["last_emit"] is not None


def test_signal_health_healthy_normal_case(state_db):
    """Pocos emits en ventana con avg score normal → healthy."""
    # 3 emits con score ~0.6 (no noisy: ni count >=10 ni avg < 0.3).
    _insert(_now_offset(hours=1), "anticipate-calendar", 0.6, "k1", 1, 1)
    _insert(_now_offset(hours=2), "anticipate-calendar", 0.7, "k2", 1, 1)
    _insert(_now_offset(hours=3), "anticipate-calendar", 0.5, "k3", 0, 0)
    health = signal_health(days=7)
    assert health["anticipate-calendar"]["status"] == "healthy"
    assert health["anticipate-calendar"]["last_emit"] is not None
    # send_rate = 2/3
    assert health["anticipate-calendar"]["send_rate"] == pytest.approx(2.0 / 3.0)


def test_signal_health_unknown_kind_in_db_is_included(state_db):
    """Un kind nuevo (no registrado en MVP/registry) que tenga rows también
    debe aparecer en el output, con status calculado normalmente."""
    _insert(_now_offset(hours=1), "anticipate-someweirdkind", 0.7, "k1", 1, 1)
    health = signal_health(days=7)
    assert "anticipate-someweirdkind" in health
    assert health["anticipate-someweirdkind"]["status"] == "healthy"


def test_top_reasons_skipped_groups_null_reason_under_no_reason(state_db):
    """Rows con reason NULL/'' caen bajo el bucket '(no reason)'."""
    _insert(_now_offset(hours=1), "anticipate-calendar", 0.1, "k1", 0, 0,
            reason="")
    _insert(_now_offset(hours=2), "anticipate-calendar", 0.1, "k2", 0, 0,
            reason="")
    out = top_reasons_skipped(days=7)
    assert ("(no reason)", 2) in out
