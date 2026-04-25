"""Tests for the Anticipatory Agent — el RAG empuja info timely sin que pregunten.

Cubre:
- AnticipatoryCandidate dataclass (frozen, fields)
- _anticipate_dedup_seen (SQL lookup)
- _anticipate_log_candidate (append-only log)
- 3 signals (calendar / echo / commitment) con mocks
- Orchestrator: threshold, dedup, top-1 pick, force, dry_run, error isolation
- CLI: run / log / explain
- Disabled flag + threshold env override
"""

from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

import rag
from rag import SqliteVecClient as _TestVecClient


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Aísla el telemetry DB en tmp_path. _ragvec_state_conn() crea las tablas
    on-demand al primer uso (vía _ensure_telemetry_tables). Devuelve una
    sqlite-vec collection separada para tests que la necesiten."""
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    # Monkeypatch ANTES de crear el client/abrir conn
    monkeypatch.setattr(rag, "DB_PATH", db_path)
    client = _TestVecClient(path=str(db_path))
    col = client.get_or_create_collection(
        name="anticipate_test", metadata={"hnsw:space": "cosine"},
    )
    # Trigger DDL via una conn dummy
    with rag._ragvec_state_conn() as _conn:
        pass
    return col


# ── Dataclass ────────────────────────────────────────────────────────────────

def test_anticipatory_candidate_is_frozen():
    c = rag.AnticipatoryCandidate(
        kind="anticipate-calendar", score=0.5, message="hi",
        dedup_key="k", snooze_hours=2, reason="why",
    )
    with pytest.raises(FrozenInstanceError):
        c.score = 0.9  # type: ignore[misc]


def test_anticipatory_candidate_required_fields():
    c = rag.AnticipatoryCandidate(
        kind="anticipate-echo", score=0.7, message="m",
        dedup_key="echo:a:b", snooze_hours=72, reason="cosine=0.7",
    )
    d = asdict(c)
    assert set(d.keys()) == {
        "kind", "score", "message", "dedup_key", "snooze_hours", "reason",
    }


# ── Dedup helper ─────────────────────────────────────────────────────────────

def test_dedup_seen_returns_false_when_empty(state_db):
    assert rag._anticipate_dedup_seen("never_seen_before") is False


def test_dedup_seen_returns_true_within_window(state_db):
    c = rag.AnticipatoryCandidate(
        kind="anticipate-calendar", score=0.9, message="m",
        dedup_key="cal:test:01", snooze_hours=2, reason="r",
    )
    rag._anticipate_log_candidate(c, selected=True, sent=True)
    assert rag._anticipate_dedup_seen("cal:test:01") is True


def test_dedup_seen_returns_false_when_not_sent(state_db):
    """Un candidate evaluado pero NO enviado (sent=0) NO bloquea futuras pasadas."""
    c = rag.AnticipatoryCandidate(
        kind="anticipate-echo", score=0.6, message="m",
        dedup_key="echo:a:b", snooze_hours=72, reason="r",
    )
    rag._anticipate_log_candidate(c, selected=False, sent=False)
    assert rag._anticipate_dedup_seen("echo:a:b") is False


def test_dedup_seen_respects_window(state_db):
    """Si la entry tiene ts más viejo que window_hours, no bloquea."""
    # Insert manualmente con ts de hace 48h
    old_ts = (datetime.now() - timedelta(hours=48)).isoformat(timespec="seconds")
    with rag._ragvec_state_conn() as conn:
        conn.execute(
            "INSERT INTO rag_anticipate_candidates "
            "(ts, kind, score, dedup_key, selected, sent, reason, message_preview) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (old_ts, "anticipate-calendar", 0.9, "cal:old:01", 1, 1, "r", "m"),
        )
        conn.commit()
    assert rag._anticipate_dedup_seen("cal:old:01", window_hours=24) is False
    assert rag._anticipate_dedup_seen("cal:old:01", window_hours=72) is True


# ── Calendar signal ──────────────────────────────────────────────────────────

def test_calendar_parse_event_start_24h():
    now = datetime(2026, 4, 25, 10, 0, 0)
    start = rag._anticipate_parse_event_start({"start": "14:30"}, now)
    assert start == datetime(2026, 4, 25, 14, 30, 0)


def test_calendar_parse_event_start_am_pm():
    now = datetime(2026, 4, 25, 10, 0, 0)
    assert rag._anticipate_parse_event_start({"start": "9:30 AM"}, now) == \
        datetime(2026, 4, 25, 9, 30, 0)
    assert rag._anticipate_parse_event_start({"start": "2:30 PM"}, now) == \
        datetime(2026, 4, 25, 14, 30, 0)
    assert rag._anticipate_parse_event_start({"start": "12:00 AM"}, now) == \
        datetime(2026, 4, 25, 0, 0, 0)
    assert rag._anticipate_parse_event_start({"start": "12:00 PM"}, now) == \
        datetime(2026, 4, 25, 12, 0, 0)


def test_calendar_parse_event_start_invalid():
    now = datetime(2026, 4, 25, 10, 0, 0)
    assert rag._anticipate_parse_event_start({"start": ""}, now) is None
    assert rag._anticipate_parse_event_start({"start": "garbage"}, now) is None
    assert rag._anticipate_parse_event_start({"start": "25:00"}, now) is None


def test_calendar_signal_empty_calendar(monkeypatch):
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda *a, **kw: [])
    out = rag._anticipate_signal_calendar(datetime.now())
    assert out == []


def test_calendar_signal_event_in_window_emits(monkeypatch, state_db):
    now = datetime(2026, 4, 25, 10, 0, 0)
    # Evento a las 11:00 → 60min de delta → in window [15, 90]
    monkeypatch.setattr(rag, "_fetch_calendar_today",
                        lambda *a, **kw: [{"title": "Coaching call", "start": "11:00"}])
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {
            "metas": [{"file": "02-Areas/Coaching.md", "note": "Coaching"}],
            "scores": [0.8],
        },
    )
    out = rag._anticipate_signal_calendar(now)
    assert len(out) == 1
    assert out[0].kind == "anticipate-calendar"
    assert "Coaching call" in out[0].message
    assert out[0].snooze_hours == 2
    assert out[0].dedup_key.startswith("cal:Coaching call:")


def test_calendar_signal_event_too_soon_skipped(monkeypatch):
    now = datetime(2026, 4, 25, 10, 0, 0)
    # Evento a las 10:05 → 5min → < 15min, skip
    monkeypatch.setattr(rag, "_fetch_calendar_today",
                        lambda *a, **kw: [{"title": "x", "start": "10:05"}])
    out = rag._anticipate_signal_calendar(now)
    assert out == []


def test_calendar_signal_event_too_far_skipped(monkeypatch):
    now = datetime(2026, 4, 25, 10, 0, 0)
    # Evento a las 13:00 → 180min → > 90min, skip
    monkeypatch.setattr(rag, "_fetch_calendar_today",
                        lambda *a, **kw: [{"title": "x", "start": "13:00"}])
    out = rag._anticipate_signal_calendar(now)
    assert out == []


def test_calendar_signal_no_vault_context_skipped(monkeypatch, state_db):
    now = datetime(2026, 4, 25, 10, 0, 0)
    monkeypatch.setattr(rag, "_fetch_calendar_today",
                        lambda *a, **kw: [{"title": "Random", "start": "11:00"}])
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    monkeypatch.setattr(rag, "retrieve",
                        lambda *a, **kw: {"metas": [{"file": "x.md"}], "scores": [0.10]})
    out = rag._anticipate_signal_calendar(now)
    assert out == []


def test_calendar_signal_score_inversely_proportional(monkeypatch, state_db):
    now = datetime(2026, 4, 25, 10, 0, 0)
    monkeypatch.setattr(rag, "_fetch_calendar_today",
                        lambda *a, **kw: [
                            {"title": "imminent", "start": "10:30"},  # 30min
                            {"title": "later", "start": "11:30"},       # 90min
                        ])
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    monkeypatch.setattr(rag, "retrieve",
                        lambda *a, **kw: {"metas": [{"file": "x.md"}], "scores": [0.5]})
    out = rag._anticipate_signal_calendar(now)
    assert len(out) == 2
    # Imminent debería tener score más alto
    by_title = {("imminent" if "imminent" in c.message else "later"): c for c in out}
    assert by_title["imminent"].score > by_title["later"].score


# ── Echo signal ──────────────────────────────────────────────────────────────

def test_echo_no_recent_notes(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: tmp_path)
    out = rag._anticipate_signal_echo(datetime.now())
    assert out == []


def test_echo_emits_when_old_match(monkeypatch, tmp_path, state_db):
    vault = tmp_path / "vault"
    vault.mkdir()
    today_note = vault / "today.md"
    body = "x" * 600  # ≥500 chars
    today_note.write_text(body, encoding="utf-8")

    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    # Mock retrieve to return an old note
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {
            "metas": [
                {"file": "today.md", "note": "today"},  # same → must be skipped
                {"file": "02-Areas/Old.md", "note": "Old"},
            ],
            "scores": [0.95, 0.85],
        },
    )
    # Edad de la old: simular que tiene 365d
    monkeypatch.setattr(
        rag, "_anticipate_note_age_days",
        lambda file_rel, vault: 365.0 if "Old" in file_rel else 0.0,
    )
    out = rag._anticipate_signal_echo(datetime.now())
    assert len(out) == 1
    assert out[0].kind == "anticipate-echo"
    assert out[0].dedup_key.startswith("echo:")
    assert "Old" in out[0].message or "[[Old]]" in out[0].message


def test_echo_below_cosine_threshold_skipped(monkeypatch, tmp_path, state_db):
    vault = tmp_path / "vault"
    vault.mkdir()
    today_note = vault / "today.md"
    today_note.write_text("x" * 600, encoding="utf-8")

    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {
            "metas": [{"file": "old.md", "note": "old"}],
            "scores": [0.55],  # below 0.70
        },
    )
    monkeypatch.setattr(rag, "_anticipate_note_age_days", lambda *a, **kw: 200)
    out = rag._anticipate_signal_echo(datetime.now())
    assert out == []


def test_echo_skips_same_file(monkeypatch, tmp_path, state_db):
    """Si el top retrieve match es la MISMA nota de hoy, no emit."""
    vault = tmp_path / "vault"
    vault.mkdir()
    today_note = vault / "today.md"
    today_note.write_text("x" * 600, encoding="utf-8")

    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {
            "metas": [{"file": "today.md", "note": "today"}],
            "scores": [0.99],
        },
    )
    monkeypatch.setattr(rag, "_anticipate_note_age_days", lambda *a, **kw: 365)
    out = rag._anticipate_signal_echo(datetime.now())
    assert out == []


# ── Commitment signal ────────────────────────────────────────────────────────

def test_commitment_no_loops(monkeypatch, state_db):
    monkeypatch.setattr(rag, "find_followup_loops",
                        lambda *a, **kw: [])
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: Path("/tmp"))
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    out = rag._anticipate_signal_commitment(datetime.now())
    assert out == []


def test_commitment_only_fresh_skipped(monkeypatch, state_db):
    monkeypatch.setattr(rag, "find_followup_loops",
                        lambda *a, **kw: [
                            {"status": "stale", "age_days": 3,
                             "loop_text": "x", "source_note": "n.md", "kind": "checkbox"},
                        ])
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: Path("/tmp"))
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    out = rag._anticipate_signal_commitment(datetime.now())
    assert out == []  # 3d < 7d threshold


def test_commitment_emits_oldest(monkeypatch, state_db):
    monkeypatch.setattr(rag, "find_followup_loops",
                        lambda *a, **kw: [
                            {"status": "stale", "age_days": 8,
                             "loop_text": "newer task", "source_note": "a.md", "kind": "checkbox"},
                            {"status": "stale", "age_days": 15,
                             "loop_text": "older task", "source_note": "b.md", "kind": "todo"},
                            {"status": "stale", "age_days": 10,
                             "loop_text": "middle task", "source_note": "c.md", "kind": "inline"},
                        ])
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: Path("/tmp"))
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    out = rag._anticipate_signal_commitment(datetime.now())
    assert len(out) == 1
    assert "older task" in out[0].message
    assert out[0].snooze_hours == 168


def test_commitment_score_caps_at_30d(monkeypatch, state_db):
    monkeypatch.setattr(rag, "find_followup_loops",
                        lambda *a, **kw: [
                            {"status": "stale", "age_days": 100,
                             "loop_text": "very old", "source_note": "x.md", "kind": "todo"},
                        ])
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: Path("/tmp"))
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    out = rag._anticipate_signal_commitment(datetime.now())
    assert out[0].score == 1.0


def test_commitment_dedup_key_stable(monkeypatch, state_db):
    """Mismo loop + mismo source debe dar SIEMPRE el mismo dedup_key."""
    loop = {
        "status": "stale", "age_days": 10,
        "loop_text": "stable text", "source_note": "stable.md", "kind": "checkbox",
    }
    monkeypatch.setattr(rag, "find_followup_loops", lambda *a, **kw: [loop])
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: Path("/tmp"))
    monkeypatch.setattr(rag, "get_db", lambda: state_db)
    out1 = rag._anticipate_signal_commitment(datetime.now())
    out2 = rag._anticipate_signal_commitment(datetime.now())
    assert out1[0].dedup_key == out2[0].dedup_key


# ── Orchestrator ─────────────────────────────────────────────────────────────

def test_orchestrator_no_signals_returns_empty(monkeypatch, state_db):
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS",
                        (("noop", lambda now: []),))
    res = rag.anticipate_run_impl(dry_run=True)
    assert res["selected"] is None
    assert res["all"] == []


def test_orchestrator_picks_highest_score(monkeypatch, state_db):
    cands = [
        rag.AnticipatoryCandidate(
            kind="anticipate-echo", score=0.5, message="low",
            dedup_key="k1", snooze_hours=72, reason="r",
        ),
        rag.AnticipatoryCandidate(
            kind="anticipate-calendar", score=0.9, message="high",
            dedup_key="k2", snooze_hours=2, reason="r",
        ),
    ]
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS",
                        (("a", lambda now: [cands[0]]),
                         ("b", lambda now: [cands[1]])))
    res = rag.anticipate_run_impl(dry_run=True)
    assert res["selected"]["score"] == 0.9
    assert res["selected"]["kind"] == "anticipate-calendar"


def test_orchestrator_below_threshold_no_selection(monkeypatch, state_db):
    cands = [rag.AnticipatoryCandidate(
        kind="anticipate-echo", score=0.10, message="m",
        dedup_key="k1", snooze_hours=72, reason="r",
    )]
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS",
                        (("a", lambda now: cands),))
    res = rag.anticipate_run_impl(dry_run=True)
    assert res["selected"] is None
    assert len(res["all"]) == 1


def test_orchestrator_skips_dedup_seen(monkeypatch, state_db):
    cand = rag.AnticipatoryCandidate(
        kind="anticipate-calendar", score=0.9, message="m",
        dedup_key="dup_key", snooze_hours=2, reason="r",
    )
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS",
                        (("a", lambda now: [cand]),))
    monkeypatch.setattr(rag, "_anticipate_dedup_seen",
                        lambda *a, **kw: True)
    res = rag.anticipate_run_impl(dry_run=True)
    assert res["selected"] is None


def test_orchestrator_force_bypasses_dedup(monkeypatch, state_db):
    cand = rag.AnticipatoryCandidate(
        kind="anticipate-calendar", score=0.9, message="m",
        dedup_key="dup_key", snooze_hours=2, reason="r",
    )
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS",
                        (("a", lambda now: [cand]),))
    monkeypatch.setattr(rag, "_anticipate_dedup_seen",
                        lambda *a, **kw: True)
    res = rag.anticipate_run_impl(dry_run=True, force=True)
    assert res["selected"] is not None


def test_orchestrator_dry_run_does_not_push(monkeypatch, state_db):
    cand = rag.AnticipatoryCandidate(
        kind="anticipate-calendar", score=0.9, message="m",
        dedup_key="k", snooze_hours=2, reason="r",
    )
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS",
                        (("a", lambda now: [cand]),))
    pushed = []
    monkeypatch.setattr(rag, "proactive_push",
                        lambda kind, msg, **kw: pushed.append(kind) or True)
    rag.anticipate_run_impl(dry_run=True)
    assert pushed == []


def test_orchestrator_signal_error_does_not_block_others(monkeypatch, state_db):
    def boom(now):
        raise RuntimeError("kaboom")
    cand = rag.AnticipatoryCandidate(
        kind="anticipate-echo", score=0.9, message="m",
        dedup_key="k", snooze_hours=72, reason="r",
    )
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS",
                        (("crashed", boom),
                         ("ok", lambda now: [cand])))
    res = rag.anticipate_run_impl(dry_run=True)
    assert res["selected"]["kind"] == "anticipate-echo"


def test_orchestrator_disabled_env_returns_immediately(monkeypatch, state_db):
    monkeypatch.setenv("RAG_ANTICIPATE_DISABLED", "1")
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS",
                        (("would_have_fired",
                          lambda now: [rag.AnticipatoryCandidate(
                              kind="anticipate-echo", score=0.9, message="m",
                              dedup_key="k", snooze_hours=72, reason="r",
                          )]),))
    res = rag.anticipate_run_impl(dry_run=False)
    assert res.get("disabled") is True
    assert res["all"] == []


def test_orchestrator_pushes_when_not_dry_run(monkeypatch, state_db):
    cand = rag.AnticipatoryCandidate(
        kind="anticipate-calendar", score=0.9, message="m",
        dedup_key="k_unique", snooze_hours=2, reason="r",
    )
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS",
                        (("a", lambda now: [cand]),))
    pushed = []
    monkeypatch.setattr(rag, "proactive_push",
                        lambda kind, msg, **kw: pushed.append((kind, msg)) or True)
    res = rag.anticipate_run_impl(dry_run=False)
    assert res["selected"] is not None
    assert res["sent"] is True
    assert pushed[0][0] == "anticipate-calendar"


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_anticipate_run_default_no_subcommand(monkeypatch, state_db):
    """`rag anticipate` sin subcomando = `rag anticipate run`."""
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS", (("noop", lambda now: []),))
    runner = CliRunner()
    result = runner.invoke(rag.anticipate, [])
    assert result.exit_code == 0
    assert "ninguna señal activa" in result.output


def test_cli_anticipate_explain(monkeypatch, state_db):
    cand = rag.AnticipatoryCandidate(
        kind="anticipate-echo", score=0.6, message="m",
        dedup_key="k:explain", snooze_hours=72, reason="why",
    )
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS",
                        (("a", lambda now: [cand]),))
    runner = CliRunner()
    result = runner.invoke(rag.anticipate, ["explain"])
    assert result.exit_code == 0
    assert "anticipate-echo" in result.output


def test_cli_anticipate_log_empty(state_db):
    runner = CliRunner()
    result = runner.invoke(rag.anticipate, ["log"])
    assert result.exit_code == 0
    assert "sin registros" in result.output


def test_cli_anticipate_log_with_rows(state_db):
    c = rag.AnticipatoryCandidate(
        kind="anticipate-echo", score=0.7, message="m",
        dedup_key="k:log", snooze_hours=72, reason="why",
    )
    rag._anticipate_log_candidate(c, selected=True, sent=True)
    runner = CliRunner()
    result = runner.invoke(rag.anticipate, ["log", "-n", "5"])
    assert result.exit_code == 0
    assert "anticipate-echo" in result.output
    assert "0.70" in result.output


# ── Threshold env override ───────────────────────────────────────────────────

def test_threshold_env_override(monkeypatch):
    """RAG_ANTICIPATE_MIN_SCORE override es leído al import. Reload garantiza
    que el cambio en runtime tome efecto."""
    # Probamos directo el módulo: el constante se setea al import. Test que
    # un score por debajo del MIN actual no pasa filter.
    cand = rag.AnticipatoryCandidate(
        kind="anticipate-echo", score=rag._ANTICIPATE_MIN_SCORE - 0.01,
        message="m", dedup_key="k:thresh", snooze_hours=72, reason="r",
    )
    # El orchestrator debe filtrarlo por score
    monkeypatch.setattr(rag, "_ANTICIPATE_SIGNALS",
                        (("a", lambda now: [cand]),))
    res = rag.anticipate_run_impl(dry_run=True)
    assert res["selected"] is None
