"""End-to-end integration test del Anticipatory Agent.

Pre-existing tests (test_anticipate_agent.py) testean el orchestrator con
mocked signals individualmente. Estos tests ejercitan el flujo COMPLETO:
- Vault filesystem real (tmp)
- Calendar mock con eventos plausibles
- SQL state real (tmp)
- Múltiples runs consecutivos (verifying dedup across runs)
- WhatsApp push mocked, but verificando shape del payload
"""

from datetime import datetime, timedelta

import pytest

import rag
from rag import SqliteVecClient


@pytest.fixture
def integrated_env(tmp_path, monkeypatch):
    """Setup completo: vault + state DB + ambient.json + proactive state."""
    # Vault
    vault = tmp_path / "vault"
    (vault / "00-Inbox").mkdir(parents=True)
    (vault / "01-Projects").mkdir(parents=True)
    (vault / "02-Areas").mkdir(parents=True)
    (vault / "04-Archive/99-obsidian-system/99-Claude/reviews").mkdir(parents=True)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)

    # State DB (telemetry sqlite — rag_anticipate_candidates lives here).
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    SqliteVecClient(path=str(db_path))
    # Trigger DDL via dummy conn (creates rag_anticipate_candidates et al).
    with rag._ragvec_state_conn() as _conn:
        pass

    # Mock proactive state path → tmp file. Sin esto, _proactive_load_state()
    # leería el archivo real de ~/.local/share/obsidian-rag/proactive.json,
    # contaminando el test con silenced/snooze/daily_count del usuario real.
    monkeypatch.setattr(
        rag, "PROACTIVE_STATE_PATH", tmp_path / "proactive.json",
    )

    # Mock ambient config (para que proactive_push no se queje).
    monkeypatch.setattr(rag, "_ambient_config",
                        lambda: {"jid": "test@s.whatsapp.net"})

    # Mock WA send → record en lista en lugar de http call.
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        rag, "_ambient_whatsapp_send",
        lambda jid, text: sent.append((jid, text)) or True,
    )

    # Mock retrieve para que las signals que usan retrieve no fallen.
    monkeypatch.setattr(
        rag, "retrieve",
        lambda col, q, k, **kw: {
            "metas": [], "scores": [], "ids": [], "documents": [],
        },
    )
    monkeypatch.setattr(rag, "get_db", lambda: object())

    # Mock find_followup_loops (no loops por default).
    monkeypatch.setattr(rag, "find_followup_loops", lambda *a, **kw: [])

    # Mock _fetch_calendar_today (no eventos default).
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda *a, **kw: [])

    # Aislar el agent del set extra de signals (rag_anticipate.signals.*).
    # Tests del flujo core trabajan con calendar/echo/commitment; los extras
    # tienen sus propios test_anticipate_<kind>.py. Algunos extras tocan SQL
    # tables (rag_queries, rag_wa_tasks) o filesystem en formas que pueden
    # emitir candidates espurios sobre vaults vacíos.
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS",
                        rag._ANTICIPATE_CORE_SIGNALS)

    return {
        "vault": vault,
        "db_path": db_path,
        "sent": sent,
        "monkeypatch": monkeypatch,
        "tmp_path": tmp_path,
    }


# ── Tests ────────────────────────────────────────────────────────────────────

def test_e2e_no_signals_active_no_push(integrated_env):
    """Vault vacío, no calendar events, no loops → no push."""
    result = rag.anticipate_run_impl(dry_run=False)
    assert result["selected"] is None
    assert result["sent"] is False
    assert integrated_env["sent"] == []


def test_e2e_calendar_event_with_context_pushes(integrated_env):
    """Evento próximo + retrieve devuelve match → push real."""
    monkeypatch = integrated_env["monkeypatch"]

    # Calculate now and event 30min ahead. Si caemos cerca de medianoche y el
    # evento "salta al día siguiente", igual el delta_min computado contra
    # `now.replace(hh, mm)` queda dentro de [15, 90] porque parsea como hoy.
    now = datetime.now()
    event_dt = now + timedelta(minutes=30)
    event_time = event_dt.strftime("%H:%M")

    monkeypatch.setattr(
        rag, "_fetch_calendar_today",
        lambda **kw: [{"title": "Coaching call", "start": event_time, "end": ""}],
    )
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {
            "metas": [{"file": "02-Areas/Coaching.md", "note": "Coaching",
                       "preview": "context"}],
            "scores": [0.85],
        },
    )

    result = rag.anticipate_run_impl(dry_run=False)

    assert result["selected"] is not None
    assert result["selected"]["kind"] == "anticipate-calendar"
    assert "Coaching call" in result["selected"]["message"]
    assert result["sent"] is True
    assert len(integrated_env["sent"]) == 1
    jid, text = integrated_env["sent"][0]
    assert jid == "test@s.whatsapp.net"
    assert "Coaching call" in text


def test_e2e_dedup_across_multiple_runs(integrated_env):
    """2 runs consecutivos del MISMO context: 1er run pushea, 2do no (dedup)."""
    monkeypatch = integrated_env["monkeypatch"]

    now = datetime.now()
    event_time = (now + timedelta(minutes=30)).strftime("%H:%M")
    monkeypatch.setattr(
        rag, "_fetch_calendar_today",
        lambda **kw: [{"title": "Standup", "start": event_time, "end": ""}],
    )
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"metas": [{"file": "Standup.md"}], "scores": [0.9]},
    )

    # 1er run.
    r1 = rag.anticipate_run_impl(dry_run=False)
    assert r1["selected"] is not None
    assert r1["sent"] is True

    # 2do run: dedup_key ya tiene sent=1 en SQL → skip.
    r2 = rag.anticipate_run_impl(dry_run=False)
    assert r2["selected"] is None  # dedup'd
    assert r2["sent"] is False
    assert len(integrated_env["sent"]) == 1  # solo 1 push total


def test_e2e_dry_run_does_not_push(integrated_env):
    """dry_run=True evalúa pero no manda."""
    monkeypatch = integrated_env["monkeypatch"]
    now = datetime.now()
    event_time = (now + timedelta(minutes=30)).strftime("%H:%M")
    monkeypatch.setattr(
        rag, "_fetch_calendar_today",
        lambda **kw: [{"title": "Test", "start": event_time, "end": ""}],
    )
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"metas": [{"file": "x.md"}], "scores": [0.9]},
    )

    result = rag.anticipate_run_impl(dry_run=True)

    # selected si pasa el threshold pero no envía.
    assert result["selected"] is not None
    assert result["selected"]["kind"] == "anticipate-calendar"
    assert result["sent"] is False
    assert integrated_env["sent"] == []


def test_e2e_force_bypasses_dedup(integrated_env):
    """force=True permite re-push aunque dedup_key ya esté."""
    monkeypatch = integrated_env["monkeypatch"]
    now = datetime.now()
    event_time = (now + timedelta(minutes=30)).strftime("%H:%M")
    monkeypatch.setattr(
        rag, "_fetch_calendar_today",
        lambda **kw: [{"title": "Standup", "start": event_time, "end": ""}],
    )
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"metas": [{"file": "x.md"}], "scores": [0.9]},
    )

    r1 = rag.anticipate_run_impl(dry_run=False)              # 1er push
    r2 = rag.anticipate_run_impl(dry_run=False, force=True)  # 2do con force

    # Con force, dedup se bypassa pero proactive_push sigue aplicando
    # silence/snooze/cap. En este test el state está limpio (sin silence,
    # daily_count<3) → ambos runs deberían tener selected != None. El push
    # del 2do run probablemente no sale (proactive_push setea snooze tras el
    # 1er send, y snooze_hours=2 para calendar ⇒ rebota), pero ese
    # comportamiento es de proactive_push, no del orchestrator. Lo que sí
    # afirmamos: el orchestrator NO filtra por dedup cuando force=True.
    assert r1["selected"] is not None
    assert r2["selected"] is not None
    assert r1["sent"] is True
    # Sin force, r2 hubiera tenido selected=None (dedup'd). El hecho de que
    # selected esté seteado prueba que force bypasseó dedup.


def test_e2e_disabled_env_short_circuits(integrated_env):
    """RAG_ANTICIPATE_DISABLED=1 → early return."""
    integrated_env["monkeypatch"].setenv("RAG_ANTICIPATE_DISABLED", "1")
    result = rag.anticipate_run_impl(dry_run=False)
    assert result.get("disabled") is True
    assert integrated_env["sent"] == []


def test_e2e_logs_all_candidates_to_sql(integrated_env):
    """Todos los candidates evaluados se loguean a rag_anticipate_candidates."""
    monkeypatch = integrated_env["monkeypatch"]
    now = datetime.now()
    event_time = (now + timedelta(minutes=20)).strftime("%H:%M")
    monkeypatch.setattr(
        rag, "_fetch_calendar_today",
        lambda **kw: [{"title": "Foo", "start": event_time, "end": ""}],
    )
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"metas": [{"file": "foo.md"}], "scores": [0.5]},
    )

    rag.anticipate_run_impl(dry_run=True)

    with rag._ragvec_state_conn() as conn:
        rows = conn.execute(
            "SELECT kind, score, sent FROM rag_anticipate_candidates"
        ).fetchall()

    assert len(rows) >= 1
    kinds = {r[0] for r in rows}
    assert "anticipate-calendar" in kinds
    # En dry_run, ningún row debería tener sent=1.
    assert all(r[2] == 0 for r in rows)


def test_e2e_signal_error_does_not_crash_orchestrator(integrated_env):
    """Una signal que crashea no debe tumbar al resto."""
    def crashing_signal(now):
        raise RuntimeError("kaboom")

    monkeypatch = integrated_env["monkeypatch"]
    monkeypatch.setattr(
        rag, "_ANTICIPATE_SIGNALS",
        (("crash", crashing_signal),) + tuple(rag._ANTICIPATE_SIGNALS),
    )

    # Should not raise.
    result = rag.anticipate_run_impl(dry_run=True)
    assert isinstance(result, dict)
    assert "selected" in result
    assert "all" in result


def test_e2e_multiple_signals_picks_highest(integrated_env):
    """Si calendar + commitment ambos emiten, el de mayor score gana."""
    monkeypatch = integrated_env["monkeypatch"]
    now = datetime.now()
    event_time = (now + timedelta(minutes=20)).strftime("%H:%M")

    # Calendar event → score = 1 - 20/90 ≈ 0.78.
    monkeypatch.setattr(
        rag, "_fetch_calendar_today",
        lambda **kw: [{"title": "X", "start": event_time, "end": ""}],
    )
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"metas": [{"file": "x.md"}], "scores": [0.8]},
    )

    # Commitment loop con score lower (8/30 ≈ 0.27, debajo del threshold).
    monkeypatch.setattr(
        rag, "find_followup_loops",
        lambda *a, **kw: [
            {"status": "stale", "age_days": 8, "loop_text": "x",
             "source_note": "n.md", "kind": "checkbox"},
        ],
    )

    result = rag.anticipate_run_impl(dry_run=True)
    assert result["selected"] is not None
    # Calendar score ~0.78 > commitment score (8/30) ~0.27.
    assert result["selected"]["kind"] == "anticipate-calendar"


def test_e2e_orchestrator_returns_all_candidates_for_explain(integrated_env):
    """El field 'all' incluye TODOS los candidates evaluados (no solo selected)."""
    monkeypatch = integrated_env["monkeypatch"]
    now = datetime.now()
    event_time = (now + timedelta(minutes=20)).strftime("%H:%M")
    monkeypatch.setattr(
        rag, "_fetch_calendar_today",
        lambda **kw: [{"title": "X", "start": event_time, "end": ""}],
    )
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"metas": [{"file": "x.md"}], "scores": [0.8]},
    )
    monkeypatch.setattr(
        rag, "find_followup_loops",
        lambda *a, **kw: [
            {"status": "stale", "age_days": 10, "loop_text": "x",
             "source_note": "n.md", "kind": "checkbox"},
        ],
    )

    result = rag.anticipate_run_impl(dry_run=True)
    # 'all' contiene TODOS los candidates (sin filtrar por threshold/dedup).
    # Calendar (0.78) + commitment (10/30 = 0.33) → 2 candidates al menos.
    assert len(result["all"]) >= 2
    kinds = {c["kind"] for c in result["all"]}
    assert "anticipate-calendar" in kinds
    assert "anticipate-commitment" in kinds
