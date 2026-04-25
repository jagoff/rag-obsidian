"""Tests for the deadline signal — `rag_anticipate/signals/deadline.py`.

Cubre:
- Empty vault → []
- Nota sin frontmatter `due:` → []
- Notas con due en la ventana [hoy, +3 días] → emite candidates con score calibrado
- Notas con due fuera de ventana (+5d, pasado) → skip
- MAX 2 candidates, ordenados por proximidad
- `due:` como lista YAML → parse primera parseable
- `due:` malformado → skip silencioso
- dedup_key estable
- Helper `_parse_due_value`: cubre str ISO, str DD/MM/YYYY, date scalar, lista
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

import rag
from rag_anticipate.signals import deadline as deadline_mod
from rag_anticipate.signals.deadline import (
    _parse_due_value,
    _walk_notes_with_due,
    deadline_signal,
)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_vault(tmp_path, monkeypatch):
    """Vault aislado en tmp_path; patchea `_resolve_vault_path` y `VAULT_PATH`
    para que la signal use este vault durante el test."""
    vault = tmp_path / "vault"
    (vault / "01-Projects").mkdir(parents=True)
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: vault)
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    return vault


def _write_note(vault, path, due):
    """Write a note at `vault/path` with a frontmatter `due:` field. The `due`
    value is inserted literally (so the caller controls YAML formatting)."""
    full = vault / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(f"---\ndue: {due}\n---\nbody\n", encoding="utf-8")


def _write_raw_note(vault, path, body):
    """Write a note with arbitrary content (escape hatch for weird FM shapes)."""
    full = vault / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(body, encoding="utf-8")


# ── _parse_due_value ─────────────────────────────────────────────────────────

def test_parse_due_value_iso_string():
    assert _parse_due_value("2026-04-28") == date(2026, 4, 28)


def test_parse_due_value_iso_with_time():
    assert _parse_due_value("2026-04-28T10:00") == date(2026, 4, 28)
    assert _parse_due_value("2026-04-28T10:00:00") == date(2026, 4, 28)


def test_parse_due_value_dd_mm_yyyy():
    assert _parse_due_value("28/04/2026") == date(2026, 4, 28)


def test_parse_due_value_date_scalar():
    assert _parse_due_value(date(2026, 4, 28)) == date(2026, 4, 28)


def test_parse_due_value_datetime_scalar():
    assert _parse_due_value(datetime(2026, 4, 28, 10, 0)) == date(2026, 4, 28)


def test_parse_due_value_list_picks_first_parseable():
    # Primera válida, segunda ignorada
    assert _parse_due_value(["2026-05-01", "garbage"]) == date(2026, 5, 1)
    # Primera garbage → cae a la segunda
    assert _parse_due_value(["garbage", "2026-05-02"]) == date(2026, 5, 2)


def test_parse_due_value_invalid():
    assert _parse_due_value(None) is None
    assert _parse_due_value("") is None
    assert _parse_due_value("garbage") is None
    assert _parse_due_value("2026-13-40") is None
    assert _parse_due_value(12345) is None
    assert _parse_due_value([]) is None
    assert _parse_due_value(["garbage", ""]) is None


# ── signal: empty + no-match cases ───────────────────────────────────────────

def test_empty_vault_returns_empty(mock_vault):
    out = deadline_signal(datetime(2026, 4, 25, 10, 0, 0))
    assert out == []


def test_note_without_due_frontmatter_skipped(mock_vault):
    # Nota sin frontmatter del todo
    _write_raw_note(mock_vault, "01-Projects/plain.md", "no frontmatter here")
    # Nota con frontmatter pero sin `due:`
    _write_raw_note(
        mock_vault, "01-Projects/other.md",
        "---\ntitle: foo\ntags: [a, b]\n---\nbody\n",
    )
    out = deadline_signal(datetime(2026, 4, 25, 10, 0, 0))
    assert out == []


def test_no_vault_returns_empty(monkeypatch, tmp_path):
    # Vault path que no existe en disco → generator yields nada
    ghost = tmp_path / "does_not_exist"
    monkeypatch.setattr(rag, "_resolve_vault_path", lambda: ghost)
    monkeypatch.setattr(rag, "VAULT_PATH", ghost)
    out = deadline_signal(datetime(2026, 4, 25, 10, 0, 0))
    assert out == []


# ── signal: score calibration ────────────────────────────────────────────────

def test_due_today_score_is_1(mock_vault):
    now = datetime(2026, 4, 25, 10, 0, 0)
    _write_note(mock_vault, "01-Projects/today.md", "2026-04-25")
    out = deadline_signal(now)
    assert len(out) == 1
    assert out[0].score == 1.0
    assert out[0].kind == "anticipate-deadline"
    assert "0 días" in out[0].message
    assert "[[today]]" in out[0].message
    assert out[0].snooze_hours == 24


def test_due_tomorrow_score_is_0_75(mock_vault):
    now = datetime(2026, 4, 25, 10, 0, 0)
    _write_note(mock_vault, "01-Projects/tomorrow.md", "2026-04-26")
    out = deadline_signal(now)
    assert len(out) == 1
    assert out[0].score == 0.75
    assert "1 días" in out[0].message


def test_due_plus_two_score_is_0_5(mock_vault):
    now = datetime(2026, 4, 25, 10, 0, 0)
    _write_note(mock_vault, "01-Projects/plus2.md", "2026-04-27")
    out = deadline_signal(now)
    assert len(out) == 1
    assert out[0].score == 0.5


def test_due_plus_three_score_is_0_25(mock_vault):
    now = datetime(2026, 4, 25, 10, 0, 0)
    _write_note(mock_vault, "01-Projects/plus3.md", "2026-04-28")
    out = deadline_signal(now)
    assert len(out) == 1
    assert out[0].score == 0.25


# ── signal: window boundaries ────────────────────────────────────────────────

def test_due_plus_five_out_of_window(mock_vault):
    now = datetime(2026, 4, 25, 10, 0, 0)
    _write_note(mock_vault, "01-Projects/plus5.md", "2026-04-30")
    out = deadline_signal(now)
    assert out == []


def test_due_in_past_skipped(mock_vault):
    now = datetime(2026, 4, 25, 10, 0, 0)
    _write_note(mock_vault, "01-Projects/past.md", "2026-04-20")
    _write_note(mock_vault, "01-Projects/yesterday.md", "2026-04-24")
    out = deadline_signal(now)
    assert out == []


# ── signal: multi-candidate ordering + cap ───────────────────────────────────

def test_multiple_dues_max_2_ordered_by_proximity(mock_vault):
    now = datetime(2026, 4, 25, 10, 0, 0)
    # 4 notas dentro de la ventana; esperamos sólo 2 (las más próximas)
    _write_note(mock_vault, "01-Projects/d0.md", "2026-04-25")  # hoy
    _write_note(mock_vault, "01-Projects/d1.md", "2026-04-26")  # +1
    _write_note(mock_vault, "01-Projects/d2.md", "2026-04-27")  # +2
    _write_note(mock_vault, "01-Projects/d3.md", "2026-04-28")  # +3
    out = deadline_signal(now)
    assert len(out) == 2
    # Los dos más próximos son hoy y mañana, en ese orden
    assert out[0].score == 1.0  # hoy primero
    assert out[1].score == 0.75  # mañana después
    assert "[[d0]]" in out[0].message
    assert "[[d1]]" in out[1].message


# ── signal: YAML list form ───────────────────────────────────────────────────

def test_due_as_yaml_list_picks_first_parseable(mock_vault):
    now = datetime(2026, 4, 25, 10, 0, 0)
    # YAML list form — PyYAML lo hidrata como lista de date/str
    _write_note(
        mock_vault, "01-Projects/multi.md",
        "[2026-04-26, 2026-04-28]",
    )
    out = deadline_signal(now)
    assert len(out) == 1
    # Primera parseable en la lista = 2026-04-26 (mañana) → score 0.75
    assert out[0].score == 0.75
    assert "2026-04-26" in out[0].message


def test_due_as_yaml_list_first_garbage_fallback(mock_vault):
    now = datetime(2026, 4, 25, 10, 0, 0)
    _write_note(
        mock_vault, "01-Projects/multi2.md",
        '["garbage", "2026-04-27"]',
    )
    out = deadline_signal(now)
    assert len(out) == 1
    # Primera parseable → 2026-04-27 → +2d → 0.5
    assert out[0].score == 0.5


# ── signal: malformed input ──────────────────────────────────────────────────

def test_due_malformed_silently_skipped(mock_vault):
    now = datetime(2026, 4, 25, 10, 0, 0)
    _write_note(mock_vault, "01-Projects/bad.md", "not-a-date")
    _write_note(mock_vault, "01-Projects/bad2.md", "2026-99-99")
    # Una nota válida al lado para confirmar que el walk no aborta
    _write_note(mock_vault, "01-Projects/good.md", "2026-04-25")
    out = deadline_signal(now)
    assert len(out) == 1
    assert "[[good]]" in out[0].message


def test_broken_frontmatter_does_not_abort_walk(mock_vault):
    now = datetime(2026, 4, 25, 10, 0, 0)
    # YAML roto (key sin valor, lista sin cerrar)
    _write_raw_note(
        mock_vault, "01-Projects/broken.md",
        "---\ntags: [unclosed\ndue: 2026-04-25\n---\nbody\n",
    )
    _write_note(mock_vault, "01-Projects/good.md", "2026-04-26")
    out = deadline_signal(now)
    # El broken se skipea (parse_frontmatter devuelve {}), el good emite 1
    assert len(out) == 1
    assert "[[good]]" in out[0].message


# ── signal: dedup_key stability ──────────────────────────────────────────────

def test_dedup_key_stable_across_runs(mock_vault):
    now = datetime(2026, 4, 25, 10, 0, 0)
    _write_note(mock_vault, "01-Projects/foo.md", "2026-04-26")
    out1 = deadline_signal(now)
    out2 = deadline_signal(now)
    assert len(out1) == 1 and len(out2) == 1
    assert out1[0].dedup_key == out2[0].dedup_key
    assert out1[0].dedup_key == "deadline:01-Projects/foo.md:2026-04-26"


def test_dedup_key_differs_per_due_date(mock_vault):
    now = datetime(2026, 4, 25, 10, 0, 0)
    _write_note(mock_vault, "01-Projects/a.md", "2026-04-26")
    _write_note(mock_vault, "01-Projects/b.md", "2026-04-27")
    out = deadline_signal(now)
    assert len(out) == 2
    assert out[0].dedup_key != out[1].dedup_key


# ── signal: registration + snooze default ────────────────────────────────────

def test_signal_registered_in_package():
    # El decorator debe haber metido "deadline" en la lista global
    from rag_anticipate.signals.base import SIGNALS
    names = [name for (name, _fn) in SIGNALS]
    assert "deadline" in names


def test_signal_visible_in_rag_anticipate_signals():
    # El orchestrator en rag.py lee rag_anticipate.SIGNALS y lo concatena
    assert "deadline" in [name for (name, _) in rag._ANTICIPATE_SIGNALS]


def test_signal_snooze_hours_default_24():
    # El metadata del decorator
    assert deadline_mod.deadline_signal.__anticipate_default_snooze__ == 24


# ── signal: exception safety ─────────────────────────────────────────────────

def test_signal_returns_empty_when_vault_resolve_raises(monkeypatch):
    def _boom():
        raise RuntimeError("vault broken")
    monkeypatch.setattr(rag, "_resolve_vault_path", _boom)
    out = deadline_signal(datetime(2026, 4, 25, 10, 0, 0))
    assert out == []


# ── helper: _walk_notes_with_due ─────────────────────────────────────────────

def test_walk_notes_yields_tuples(mock_vault):
    _write_note(mock_vault, "01-Projects/a.md", "2026-04-26")
    _write_note(mock_vault, "01-Projects/b.md", "2026-04-27")
    _write_raw_note(mock_vault, "01-Projects/no_due.md", "no frontmatter")
    items = list(_walk_notes_with_due(mock_vault))
    paths = sorted(rel for rel, _due in items)
    assert paths == ["01-Projects/a.md", "01-Projects/b.md"]
    for rel, due in items:
        assert isinstance(due, date)


def test_walk_notes_skips_excluded_folders(mock_vault):
    # .trash/ es excluído por is_excluded (dotfolder)
    _write_note(mock_vault, ".trash/deleted.md", "2026-04-26")
    _write_note(mock_vault, "01-Projects/real.md", "2026-04-26")
    items = list(_walk_notes_with_due(mock_vault))
    paths = [rel for rel, _ in items]
    assert paths == ["01-Projects/real.md"]


# ── integration: now as datetime with time (mid-day) ─────────────────────────

def test_now_with_time_uses_date_component(mock_vault):
    """Un `now` con hora debe quedarse sólo con la parte fecha (no que
    las 23:59 corran la cuenta de días)."""
    now_late = datetime(2026, 4, 25, 23, 59, 0)
    now_early = datetime(2026, 4, 25, 0, 1, 0)
    _write_note(mock_vault, "01-Projects/today.md", "2026-04-25")
    out_late = deadline_signal(now_late)
    out_early = deadline_signal(now_early)
    assert len(out_late) == 1 and len(out_early) == 1
    assert out_late[0].score == 1.0
    assert out_early[0].score == 1.0
    assert out_late[0].dedup_key == out_early[0].dedup_key
