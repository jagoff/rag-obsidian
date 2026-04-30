"""Tests for Phase 2 del Anticipatory Agent (2026-04-29).

Cubre las 3 sub-features que convierten el agent de "weights hardcoded"
a "aprende del user y se ajusta solo":

  - 2.A — feedback tuning (`rag_anticipate.feedback_tuning`):
      compute_kind_threshold_adjustment ajusta el threshold per-kind
      según el ratio mute/(mute+positive) acumulado en 30d.
  - 2.B — quiet hours contextuales (`rag_anticipate.quiet_hours`):
      is_in_quiet_hours retorna `(True, reason)` para nighttime /
      in_meeting / focus_code.
  - 2.D — user-configurable weights por kind (`rag_anticipate.kind_weights`):
      tabla SQL `rag_anticipate_kind_weights`, multiplicador del score
      base, CLI `rag anticipate weights set/list/reset`.
  - Wire-up integrado en `rag.anticipate_run_impl` + endpoint web no
    rompe backwards-compat.

Mín 10 casos requeridos; este archivo trae 11 + helpers compartidos.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag  # noqa: E402
from rag import SqliteVecClient as _TestVecClient  # noqa: E402
from rag_anticipate import feedback_tuning, kind_weights  # noqa: E402
from rag_anticipate.feedback import record_feedback  # noqa: E402
from rag_anticipate.quiet_hours import is_in_quiet_hours  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Aísla telemetry DB en tmp_path. Mismo patrón que
    `tests/test_anticipate_agent.py::state_db`. Crea las tablas estándar
    via `_ragvec_state_conn` (rag_anticipate_candidates incluida) y
    deja `rag_anticipate_feedback` + `rag_anticipate_kind_weights` que
    los módulos crean on-demand.
    """
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    client = _TestVecClient(path=str(db_path))
    client.get_or_create_collection(
        name="phase2_test", metadata={"hnsw:space": "cosine"},
    )
    with rag._ragvec_state_conn() as _conn:
        pass
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_feedback_tuning_cache():
    """El módulo feedback_tuning tiene una cache 1h en memoria. Entre
    tests la limpiamos para que el orden del test runner no influya."""
    feedback_tuning.reset_cache()
    yield
    feedback_tuning.reset_cache()


@pytest.fixture(autouse=True)
def _enable_feedback_tuning(monkeypatch):
    """El conftest global no setea esta env var; default = enabled.
    Pero un peer puede haber dejado `RAG_ANTICIPATE_FEEDBACK_TUNING=0`
    en el env de la sesión. Forzamos enabled para los tests acá."""
    monkeypatch.delenv("RAG_ANTICIPATE_FEEDBACK_TUNING", raising=False)
    yield


@pytest.fixture(autouse=True)
def _undo_quiet_hours_bypass(monkeypatch):
    """El conftest global setea `RAG_ANTICIPATE_BYPASS_QUIET=1` para
    desactivar el gate global durante la suite. Tests de Phase 2.B
    quieren ejercer la lógica real del gate; deshacemos el bypass acá.

    Tests específicos que quieran reactivar el bypass pueden hacer
    `monkeypatch.setenv("RAG_ANTICIPATE_BYPASS_QUIET", "1")` en su
    propio scope.
    """
    monkeypatch.delenv("RAG_ANTICIPATE_BYPASS_QUIET", raising=False)
    monkeypatch.delenv("RAG_QUIET_HOURS_NIGHTTIME", raising=False)
    monkeypatch.delenv("RAG_QUIET_HOURS_MEETINGS", raising=False)
    monkeypatch.delenv("RAG_QUIET_HOURS_FOCUS_CODE", raising=False)
    yield


def _seed_candidate_and_feedback(
    dedup_key: str, kind: str, ratings: list[str],
) -> None:
    """Helper: insertá una row en `rag_anticipate_candidates` con
    `kind` y `dedup_key`, después N rows en `rag_anticipate_feedback`
    con `dedup_key` y los ratings dados.

    Los tests usan esto para simular "el user reaccionó N veces a un
    push del kind X" de forma controlada.
    """
    # 1. La row del candidate (selected=1, sent=1 — fue pushed de verdad).
    cand = rag.AnticipatoryCandidate(
        kind=kind, score=0.7, message="seed",
        dedup_key=dedup_key, snooze_hours=2, reason="seed",
    )
    rag._anticipate_log_candidate(cand, selected=True, sent=True)
    # 2. Las rows de feedback. Cada llamada a `record_feedback` también
    # asegura la creación de la tabla `rag_anticipate_feedback`.
    for r in ratings:
        ok = record_feedback(dedup_key, r)
        assert ok is True, f"record_feedback({dedup_key}, {r}) returned False"


# ── 2.A — feedback tuning ───────────────────────────────────────────────────


def test_compute_kind_threshold_adjustment_zero_feedback(state_db):
    """T1: con 0 feedback, el delta es 0 (threshold base sin cambios)."""
    delta = feedback_tuning.compute_kind_threshold_adjustment(
        "anticipate-calendar", use_cache=False,
    )
    assert delta == 0.0


def test_compute_kind_threshold_adjustment_five_mutes(state_db):
    """T2: 5 mutes / 0 positives → delta positivo (más estricto).

    Con ratio=1.0 y mutes=5, la fórmula da min(0.2, 0.1 + 0.5*0.4) = 0.2.
    """
    _seed_candidate_and_feedback(
        "cal:silence-me", "anticipate-calendar",
        ["mute", "mute", "mute", "mute", "mute"],
    )
    delta = feedback_tuning.compute_kind_threshold_adjustment(
        "anticipate-calendar", use_cache=False,
    )
    assert delta > 0.0
    assert delta == pytest.approx(0.2)


def test_compute_kind_threshold_adjustment_five_positives(state_db):
    """T3: 5 positives / 0 mutes → delta negativo (más permisivo).

    Con ratio=0.0 y positives=5, la fórmula da
    max(-0.2, -0.1 - 0.2*0.5) = max(-0.2, -0.20) = -0.20.
    """
    _seed_candidate_and_feedback(
        "echo:resonant", "anticipate-echo",
        ["positive", "positive", "positive", "positive", "positive"],
    )
    delta = feedback_tuning.compute_kind_threshold_adjustment(
        "anticipate-echo", use_cache=False,
    )
    assert delta < 0.0
    assert delta == pytest.approx(-0.2)


def test_compute_kind_threshold_adjustment_capped_to_two_tenths(state_db):
    """T4: el delta NUNCA sobrepasa ±0.2 incluso con muestras extremas
    (ej. 50 mutes consecutivos). Cap defensivo del cálculo."""
    _seed_candidate_and_feedback(
        "cal:hate-it", "anticipate-calendar",
        ["mute"] * 50,
    )
    delta = feedback_tuning.compute_kind_threshold_adjustment(
        "anticipate-calendar", use_cache=False,
    )
    assert delta <= 0.2
    assert delta >= -0.2

    # También testeamos el otro extremo (50 positives).
    _seed_candidate_and_feedback(
        "echo:love-it", "anticipate-echo",
        ["positive"] * 50,
    )
    delta_pos = feedback_tuning.compute_kind_threshold_adjustment(
        "anticipate-echo", use_cache=False,
    )
    assert delta_pos >= -0.2
    assert delta_pos <= 0.2


def test_compute_kind_threshold_adjustment_disabled_via_env(state_db, monkeypatch):
    """Bonus: env var `RAG_ANTICIPATE_FEEDBACK_TUNING=0` → siempre 0."""
    _seed_candidate_and_feedback(
        "cal:would-tune", "anticipate-calendar",
        ["mute", "mute", "mute"],
    )
    monkeypatch.setenv("RAG_ANTICIPATE_FEEDBACK_TUNING", "0")
    delta = feedback_tuning.compute_kind_threshold_adjustment(
        "anticipate-calendar", use_cache=False,
    )
    assert delta == 0.0


# ── 2.B — quiet hours ───────────────────────────────────────────────────────


def test_is_in_quiet_hours_at_2am_returns_nighttime(monkeypatch):
    """T5: 02:00 cae dentro del default `23-7` (wrap-around) → True con
    reason 'nighttime'."""
    # Asegurar que `_fetch_calendar_today` no devuelva eventos (default ON).
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: [])
    quiet, reason = is_in_quiet_hours(datetime(2026, 4, 25, 2, 0))
    assert quiet is True
    assert reason == "nighttime"


def test_is_in_quiet_hours_at_2pm_no_meeting_no_focus(monkeypatch):
    """T6: 14:00 sin meeting + sin focus + nighttime fuera de ventana
    → (False, None)."""
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: [])
    # Asegurarse focus_code default OFF (nuestro default).
    quiet, reason = is_in_quiet_hours(datetime(2026, 4, 25, 14, 0))
    assert quiet is False
    assert reason is None


def test_is_in_quiet_hours_meeting_active(monkeypatch):
    """Bonus 2.B: meeting en curso (default RAG_QUIET_HOURS_MEETINGS=1)
    → (True, 'in_meeting'). Verifica que la nueva API consulta el
    calendar y respeta el toggle MEETINGS."""
    events = [{"title": "Standup", "start": "14:00", "end": "15:00"}]
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: events)
    quiet, reason = is_in_quiet_hours(datetime(2026, 4, 25, 14, 30))
    assert quiet is True
    assert reason == "in_meeting"


# ── 2.D — kind weights (SQL-backed) + CLI ──────────────────────────────────


def test_cli_weights_set_persists_to_sql(state_db):
    """T7: `rag anticipate weights set --kind echo --weight 0.5` deja
    una row en `rag_anticipate_kind_weights` y `list_kind_weights`
    la devuelve."""
    runner = CliRunner()
    result = runner.invoke(
        rag.anticipate, ["weights", "set", "--kind", "anticipate-echo", "--weight", "0.5"],
    )
    assert result.exit_code == 0, result.output
    assert "anticipate-echo" in result.output
    rows = kind_weights.list_kind_weights()
    assert any(r["kind"] == "anticipate-echo" and r["weight"] == 0.5 for r in rows)
    # Y un get_kind_weight directo también lee el override.
    assert kind_weights.get_kind_weight("anticipate-echo") == 0.5


def test_orchestrator_applies_kind_weight_to_score(state_db, monkeypatch):
    """T8: candidate con weight=2.0 y score base 0.4 → effective 0.8.

    Verificamos via output del orchestrator: el `selected.score` ya
    viene multiplicado por el weight. Como 0.8 > _ANTICIPATE_MIN_SCORE
    (0.35), el candidate pasa filter y queda seleccionado.
    """
    assert kind_weights.set_kind_weight("anticipate-test", 2.0)
    cand = rag.AnticipatoryCandidate(
        kind="anticipate-test", score=0.4, message="m",
        dedup_key="t:weight-test", snooze_hours=2, reason="r",
    )
    monkeypatch.setattr(
        rag, "_ANTICIPATE_SIGNALS",
        (("test_only", lambda now: [cand]),),
    )
    res = rag.anticipate_run_impl(dry_run=True)
    assert res["selected"] is not None
    assert res["selected"]["kind"] == "anticipate-test"
    # 0.4 * 2.0 = 0.8 (clamp a 1.0 no se aplica acá).
    assert res["selected"]["score"] == pytest.approx(0.8)


def test_orchestrator_skips_candidate_when_feedback_silenced_kind(state_db, monkeypatch):
    """T9: kind silenciado por feedback (5 mutes acumulados) → threshold
    sube de 0.35 a 0.55 → un candidate de score 0.40 queda por debajo
    y el orchestrator NO lo selecciona aunque pase el dedup.

    Esto valida el wire-up de Phase 2.A en el orchestrator: el threshold
    NO es global, se ajusta per-kind.
    """
    # Seed feedback para que el delta sea +0.2 (max).
    _seed_candidate_and_feedback(
        "cal:hated", "anticipate-muted",
        ["mute"] * 5,
    )
    feedback_tuning.reset_cache()  # forzar re-cómputo

    # Candidate del MISMO kind con score 0.40 — pasa el threshold base
    # (0.35) pero no el ajustado (0.55).
    cand = rag.AnticipatoryCandidate(
        kind="anticipate-muted", score=0.40, message="should not push",
        dedup_key="cal:fresh-but-mutedkind", snooze_hours=2, reason="r",
    )
    monkeypatch.setattr(
        rag, "_ANTICIPATE_SIGNALS",
        (("muted_kind", lambda now: [cand]),),
    )
    pushed = []
    monkeypatch.setattr(
        rag, "proactive_push",
        lambda kind, msg, **kw: (pushed.append(kind) or True, None),
    )

    res = rag.anticipate_run_impl(dry_run=False)
    assert res["selected"] is None, (
        "El kind silenciado por feedback NO debería pasar el threshold "
        "subido"
    )
    assert pushed == [], "proactive_push NO debería invocarse"


# ── Endpoint web no rompe backwards-compat ──────────────────────────────────


def test_api_anticipate_feedback_endpoint_still_works(state_db):
    """T10: POST /api/anticipate/feedback sigue funcionando con el
    payload Phase 1 (dedup_key + rating + reason). El endpoint delega
    en `record_feedback` que es el mismo helper que Phase 2.A consume
    para lectura — debe ser idempotente cross-fases.
    """
    _web_server = pytest.importorskip("web.server")
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient
    import rag_anticipate.feedback as _ant_fb

    client = TestClient(_web_server.app)
    with patch.object(_ant_fb, "record_feedback", return_value=True) as mock_rf:
        resp = client.post(
            "/api/anticipate/feedback",
            json={"dedup_key": "cal:phase2-bcompat", "rating": "positive"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True}
    args, kwargs = mock_rf.call_args
    assert args == ("cal:phase2-bcompat", "positive")
    assert kwargs["source"] == "wa"


# ── CLI extras ──────────────────────────────────────────────────────────────


def test_cli_quiet_hours_status_runs(state_db, monkeypatch):
    """`rag anticipate quiet-hours status` corre sin errores y
    el output describe el estado actual (open o quiet)."""
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda max_events=30: [])
    runner = CliRunner()
    result = runner.invoke(rag.anticipate, ["quiet-hours", "status"])
    assert result.exit_code == 0, result.output
    # Output incluye "open" o "quiet hours" — uno de los dos
    out_lc = result.output.lower()
    assert ("open" in out_lc) or ("quiet hours" in out_lc), result.output


def test_cli_weights_list_empty(state_db):
    """Con la tabla recién creada, `weights list` debe mostrar el
    mensaje vacío (no error)."""
    runner = CliRunner()
    result = runner.invoke(rag.anticipate, ["weights", "list"])
    assert result.exit_code == 0, result.output
    assert "sin overrides" in result.output


def test_cli_feedback_stats_empty(state_db):
    """Sin feedback registrado, `feedback stats` muestra el mensaje
    placeholder y exit 0."""
    runner = CliRunner()
    result = runner.invoke(rag.anticipate, ["feedback", "stats"])
    assert result.exit_code == 0, result.output
    assert "sin feedback" in result.output.lower()
