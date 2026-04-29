"""Tests for brief schedule auto-tuning (`rag/brief_schedule.py`,
`_services_spec()` override path, `rag brief schedule` CLI subgroup).

Coverage:
  1. `analyze_brief_feedback` con DB vacía → `should_shift=False`.
  2. `analyze_brief_feedback` con 3+ mutes en hour=7 + ratio>0.5 → shift
     a hour=8 (dentro de la banda morning).
  3. Banda segura para morning respetada: nunca devolver hour<6:30 ni >9:00.
  4. `rag brief schedule status` corre sin DB poblada (sin crashear).
  5. `rag brief schedule auto-tune` (default = dry-run) no escribe a la DB.
  6. `rag brief schedule auto-tune --apply` escribe en
     `rag_brief_schedule_prefs`.
  7. `rag brief schedule reset --kind morning` borra el override.
  8. `_services_spec()` lee la pref si existe y la usa en el plist generado.
"""
from __future__ import annotations

import plistlib
import sqlite3
import sys
import sqlite3 as _sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag as rag_module  # noqa: E402
import rag.brief_schedule as bs  # noqa: E402  (submodule, NOT the click group)

RAG_BIN = "/usr/local/bin/rag"


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def state_db(tmp_path, monkeypatch):
    """Aísla telemetry DB en tmp_path. Mismo patrón que
    `tests/test_brief_feedback_endpoint.py::state_db`. Limpia el cache
    `_TELEMETRY_DDL_ENSURED_PATHS` para forzar re-ensure de las tablas
    (necesario para que la nueva tabla `rag_brief_schedule_prefs` se
    cree en el tmp DB)."""
    db_path = tmp_path / "ragvec"
    db_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rag_module, "DB_PATH", db_path)
    # Force DDL re-ensure on the next `_ragvec_state_conn()` open.
    rag_module._TELEMETRY_DDL_ENSURED_PATHS.clear()
    # Touch the conn once to materialise the schema.
    with rag_module._ragvec_state_conn() as _conn:
        pass
    return db_path


def _insert_feedback_row(state_db: Path, ts: str, dedup_key: str, rating: str):
    """Insert directly into rag_brief_feedback to simulate user reactions."""
    db_file = state_db / rag_module._TELEMETRY_DB_FILENAME
    conn = _sqlite3.connect(str(db_file))
    try:
        conn.execute(
            "INSERT INTO rag_brief_feedback (ts, dedup_key, rating, reason, source)"
            " VALUES (?, ?, ?, ?, ?)",
            (ts, dedup_key, rating, "", "wa"),
        )
        conn.commit()
    finally:
        conn.close()


def _morning_brief_path(date_str: str) -> str:
    """Mimic the actual vault layout used by `rag morning`."""
    return f"04-Archive/99-obsidian-system/99-AI/reviews/{date_str}.md"


# ── 1) analyze sin data ────────────────────────────────────────────────────


def test_analyze_no_data_returns_no_shift(state_db):
    """With an empty `rag_brief_feedback` table, the analyzer must not
    suggest any shift — there's nothing to optimise from."""
    out = bs.analyze_brief_feedback("morning", lookback_days=30)
    assert out["mute_count"] == 0
    assert out["positive_count"] == 0
    assert out["negative_count"] == 0
    assert out["recommendation"]["should_shift"] is False
    assert out["recommendation"]["suggested_hour"] is None
    # Default schedule surfaces in `current_hour`.
    assert out["current_hour"] == 7
    assert out["current_minute"] == 0


# ── 2) 3 mutes en hora 7 + ratio>0.5 → shift sugerido ──────────────────────


def test_analyze_three_mutes_at_hour_7_suggests_shift(state_db):
    """3 mutes a hour=7 con 0 positives → ratio=1.0 > 0.5 → debe sugerir
    shift dentro de la banda morning [06:30, 09:00]. El primer slot que
    tenga menos mutes que hour=7 (o sea cualquier hora sin mutes) debe
    salir como sugerencia. Como mutes_by_hour={7: 3} y la banda incluye
    7:30 (= hour 7) y 8:00 (= hour 8), el suggested debe ser 8:00 (la
    primera slot CON HORA distinta a 7 después de 7:00)."""
    today = datetime.now()
    # 3 mute reactions, all at hour 7 (within first hour of brief).
    for i in range(3):
        date = (today - timedelta(days=i + 1)).strftime("%Y-%m-%d")
        ts = (today - timedelta(days=i + 1)).replace(hour=7, minute=15).isoformat(timespec="seconds")
        _insert_feedback_row(state_db, ts, _morning_brief_path(date), "mute")

    out = bs.analyze_brief_feedback("morning", lookback_days=30)
    assert out["mute_count"] == 3
    assert out["positive_count"] == 0
    rec = out["recommendation"]
    assert rec["should_shift"] is True
    # The first slot strictly after (7,0) is (7,30) but its hour bucket
    # is still 7 (mute count = 3). The next is (8,0) which has 0 mutes
    # and IS strictly less than 3 → first improvement.
    assert (rec["suggested_hour"], rec["suggested_minute"]) == (8, 0)


# ── 3) Banda segura respetada ──────────────────────────────────────────────


def test_safe_band_morning_never_outside(state_db):
    """The morning band is [06:30, 09:00]. Even with all-mute saturation,
    the suggested slot must fall strictly inside the band — never <6:30
    nor >9:00. Manufacturing a worst-case: many mutes at every band
    hour (7,8,9) so every slot is "as bad" — `_suggest_shift` should
    return None (no strict improvement) and `should_shift` is False."""
    today = datetime.now()
    # 5 mutes at each hour 7, 8, 9 → no strict improvement available.
    for hour in (7, 8, 9):
        for i in range(5):
            ts = (today - timedelta(days=i + 1)).replace(hour=hour, minute=10).isoformat(timespec="seconds")
            _insert_feedback_row(
                state_db, ts,
                _morning_brief_path((today - timedelta(days=i + 1)).strftime("%Y-%m-%d")),
                "mute",
            )

    out = bs.analyze_brief_feedback("morning", lookback_days=30)
    rec = out["recommendation"]
    # Either no shift (saturated band, can't improve) OR shift inside band.
    if rec["should_shift"]:
        sh, sm = rec["suggested_hour"], rec["suggested_minute"]
        assert (6, 30) <= (sh, sm) <= (9, 0), (
            f"suggested {sh:02d}:{sm:02d} fuera de banda [06:30, 09:00]"
        )
    else:
        # No strict improvement available: should_shift False is fine.
        assert rec["suggested_hour"] is None

    # Double-check: `set_brief_schedule_pref` refuses out-of-band writes.
    assert bs.set_brief_schedule_pref("morning", 5, 0) is False
    assert bs.set_brief_schedule_pref("morning", 12, 0) is False


# ── 4) CLI status sin data ─────────────────────────────────────────────────


def test_cli_schedule_status_runs_without_data(state_db):
    """`rag brief schedule status` must produce zero-error output even
    against an empty feedback table — the CLI is a sanity-check tool and
    should never crash on cold install."""
    runner = CliRunner()
    result = runner.invoke(rag_module.cli, ["brief", "schedule", "status", "--plain"])
    assert result.exit_code == 0, result.output
    # Must emit a line per kind.
    assert "morning" in result.output
    assert "today" in result.output
    assert "digest" in result.output


# ── 5) auto-tune --dry-run no escribe ──────────────────────────────────────


def test_cli_auto_tune_dry_run_no_db_write(state_db):
    """Even with feedback that would suggest a shift, dry-run must NOT
    write to `rag_brief_schedule_prefs`. That table stays empty."""
    today = datetime.now()
    for i in range(4):
        ts = (today - timedelta(days=i + 1)).replace(hour=7, minute=20).isoformat(timespec="seconds")
        _insert_feedback_row(
            state_db, ts,
            _morning_brief_path((today - timedelta(days=i + 1)).strftime("%Y-%m-%d")),
            "mute",
        )

    runner = CliRunner()
    # Default (no --apply) is dry-run.
    result = runner.invoke(rag_module.cli, ["brief", "schedule", "auto-tune", "--plain"])
    assert result.exit_code == 0, result.output

    # Verify rag_brief_schedule_prefs is empty.
    db_file = state_db / rag_module._TELEMETRY_DB_FILENAME
    conn = sqlite3.connect(str(db_file))
    try:
        rows = conn.execute(
            "SELECT brief_kind, hour, minute FROM rag_brief_schedule_prefs"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [], f"dry-run wrote to prefs table: {rows}"


# ── 6) auto-tune --apply escribe ───────────────────────────────────────────


def test_cli_auto_tune_apply_writes_pref(state_db, monkeypatch):
    """`--apply` writes the suggested shift to `rag_brief_schedule_prefs`.
    We monkeypatch `_bootstrap_brief_plist` so the test doesn't try to
    talk to a real launchctl (it gracefully fails on Linux CI anyway).
    """
    today = datetime.now()
    for i in range(4):
        ts = (today - timedelta(days=i + 1)).replace(hour=7, minute=20).isoformat(timespec="seconds")
        _insert_feedback_row(
            state_db, ts,
            _morning_brief_path((today - timedelta(days=i + 1)).strftime("%Y-%m-%d")),
            "mute",
        )

    # Avoid touching launchctl. Returns True so the CLI logs success.
    monkeypatch.setattr(
        rag_module, "_bootstrap_brief_plist",
        lambda _kind: (True, "skipped (test)"),
    )

    runner = CliRunner()
    result = runner.invoke(
        rag_module.cli, ["brief", "schedule", "auto-tune", "--apply", "--plain"],
    )
    assert result.exit_code == 0, result.output

    db_file = state_db / rag_module._TELEMETRY_DB_FILENAME
    conn = sqlite3.connect(str(db_file))
    try:
        rows = conn.execute(
            "SELECT brief_kind, hour, minute FROM rag_brief_schedule_prefs"
            " WHERE brief_kind = 'morning'"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1, f"expected 1 morning override row, got {rows}"
    kind, hour, minute = rows[0]
    assert kind == "morning"
    # Inside the safe band [06:30, 09:00].
    assert (6, 30) <= (hour, minute) <= (9, 0)


# ── 7) reset borra override ────────────────────────────────────────────────


def test_cli_reset_morning_drops_override(state_db):
    """`reset --kind morning` removes the override row. The next
    `_services_spec()` call falls back to the hardcoded default."""
    # Seed an override directly.
    assert bs.set_brief_schedule_pref("morning", 8, 0, reason="manual seed") is True
    pref = bs.get_brief_schedule_pref("morning")
    assert pref is not None
    assert pref["hour"] == 8

    runner = CliRunner()
    result = runner.invoke(
        rag_module.cli, ["brief", "schedule", "reset", "--kind", "morning", "--plain"],
    )
    assert result.exit_code == 0, result.output

    pref_after = bs.get_brief_schedule_pref("morning")
    assert pref_after is None, f"override survived reset: {pref_after}"


# ── 8) _services_spec lee la pref ──────────────────────────────────────────


def test_services_spec_uses_pref_in_morning_plist(state_db):
    """When `rag_brief_schedule_prefs` has a morning row at 8:30, the
    `_services_spec()` output for `com.fer.obsidian-rag-morning` must
    embed Hour=8 and Minute=30 — not the hardcoded 7:00."""
    assert bs.set_brief_schedule_pref("morning", 8, 30, reason="test override") is True

    specs = rag_module._services_spec(RAG_BIN)
    morning_xml = next(
        xml for label, _fname, xml in specs
        if label == "com.fer.obsidian-rag-morning"
    )
    parsed = plistlib.loads(morning_xml.encode())
    intervals = parsed["StartCalendarInterval"]
    # Mon-Fri = 5 dicts, all with Hour=8, Minute=30 after the override.
    assert isinstance(intervals, list)
    assert len(intervals) == 5
    for d in intervals:
        assert d["Hour"] == 8, f"override no aplicado en {d}"
        assert d["Minute"] == 30, f"override no aplicado en {d}"


# ── Extra: classifier coverage ─────────────────────────────────────────────


def test_classify_brief_kind_handles_all_naming_styles():
    """The path classifier must understand both naming conventions
    observed in the wild: explicit `-morning.md` infix and the
    "date-only" convention used by the production daemons."""
    f = bs._classify_brief_kind
    assert f("04-Archive/.../reviews/2026-04-29.md") == "morning"
    assert f("04-Archive/.../reviews/2026-04-29-evening.md") == "today"
    assert f("04-Archive/.../reviews/2026-W17.md") == "digest"
    assert f("02-Areas/Briefs/2026-04-29-morning.md") == "morning"
    assert f("02-Areas/Briefs/2026-04-29-digest.md") == "digest"
    assert f("") is None
    assert f("random/path/notes.md") is None


# ── Extra: out-of-band writes refused ──────────────────────────────────────


def test_set_pref_refuses_out_of_band(state_db):
    """The writer enforces safe bands — passing an out-of-band slot must
    return False without touching the DB."""
    assert bs.set_brief_schedule_pref("morning", 5, 0) is False
    assert bs.set_brief_schedule_pref("today", 22, 0) is False  # outside [18,21]
    assert bs.set_brief_schedule_pref("digest", 20, 0) is False  # outside [21,23:30]
    # Inside-band must still work.
    assert bs.set_brief_schedule_pref("digest", 23, 30) is True


# ── Extra: brief-auto-tune plist registered ────────────────────────────────


def test_services_spec_includes_brief_auto_tune():
    """The new daemon must appear in `_services_spec()` so `rag setup`
    picks it up the next time it runs. NOT auto-installed during this
    session — the PM bootstraps after review."""
    specs = rag_module._services_spec(RAG_BIN)
    labels = {label for label, _, _ in specs}
    assert "com.fer.obsidian-rag-brief-auto-tune" in labels
    plist_xml = next(
        xml for label, _, xml in specs
        if label == "com.fer.obsidian-rag-brief-auto-tune"
    )
    parsed = plistlib.loads(plist_xml.encode())
    assert parsed["Label"] == "com.fer.obsidian-rag-brief-auto-tune"
    # Sunday (Weekday=0) at 03:00 — same window the user spec'd.
    cal = parsed["StartCalendarInterval"]
    assert cal["Weekday"] == 0
    assert cal["Hour"] == 3
    assert cal["Minute"] == 0
    # `--apply` must be in the args so the cron actually applies.
    args = parsed["ProgramArguments"]
    assert "brief" in args
    assert "schedule" in args
    assert "auto-tune" in args
    assert "--apply" in args
