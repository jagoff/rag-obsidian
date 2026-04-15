"""Tests para el refactor template-first del morning brief.

Cubre la parte determinística (agenda rendering, continuity footer, assembly)
y la mecánica de fallback cuando el LLM JSON falla. El LLM real no se toca —
monkeypatch del generador devuelve dicts conocidos.
"""
from datetime import datetime
from pathlib import Path

import pytest

import rag


# ── _render_morning_agenda_section ──────────────────────────────────────────


def test_agenda_empty_returns_empty_string():
    ev = {"calendar_today": [], "reminders_due": [], "weather_rain": None}
    assert rag._render_morning_agenda_section(ev) == ""


def test_agenda_calendar_only():
    ev = {
        "calendar_today": [
            {"title": "Reunión equipo", "start": "2026-04-16 10:00"},
            {"title": "Yoga", "start": "2026-04-16 19:00"},
        ],
        "reminders_due": [],
    }
    out = rag._render_morning_agenda_section(ev)
    assert "📅 Hoy en la agenda" in out
    assert "Reunión equipo" in out
    assert "Yoga" in out


def test_agenda_reminders_render_bucket_and_list():
    ev = {
        "reminders_due": [
            {"name": "comprar pan", "due": "2026-04-16 10:00",
             "bucket": "hoy", "list": "Recordatorios"},
        ],
    }
    out = rag._render_morning_agenda_section(ev)
    assert "comprar pan" in out
    assert "[Recordatorios]" in out
    assert "(hoy)" in out


def test_agenda_weather_emoji_appended():
    ev = {"weather_rain": {"summary": "100% entre 18-22h"}}
    out = rag._render_morning_agenda_section(ev)
    assert "🌧" in out
    assert "18-22h" in out


def test_agenda_caps_long_lists():
    ev = {
        "calendar_today": [{"title": f"e{i}", "start": "x"} for i in range(30)],
    }
    out = rag._render_morning_agenda_section(ev)
    # 10 events cap
    assert out.count("- ") <= 10


# ── _yesterday_evening_link ─────────────────────────────────────────────────


def test_continuity_link_present_when_file_exists(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "05-Reviews").mkdir(parents=True)
    (vault / "05-Reviews" / "2026-04-14-evening.md").write_text("x")
    target = datetime(2026, 4, 15)
    out = rag._yesterday_evening_link(target, vault)
    assert "2026-04-14-evening" in out
    assert "ayer cerraste con" in out


def test_continuity_link_empty_when_missing(tmp_path):
    vault = tmp_path / "vault"
    (vault / "05-Reviews").mkdir(parents=True)
    target = datetime(2026, 4, 15)
    assert rag._yesterday_evening_link(target, vault) == ""


# ── _assemble_morning_brief ─────────────────────────────────────────────────


def test_assemble_skips_empty_sections():
    parts = {
        "yesterday": "",        # empty → no 📬 section
        "focus": ["foo", "bar"],
        "pending": [],          # empty → no 🗂
        "attention": ["!"],
    }
    out = rag._assemble_morning_brief(
        "2026-04-15", agenda_md="", parts=parts, continuity="",
    )
    assert "📬 Ayer" not in out
    assert "🎯 Foco" in out
    assert "🗂 Pendientes" not in out
    assert "⚠ Atender" in out


def test_assemble_renders_bullets_correctly():
    parts = {
        "yesterday": "Tranquilo.",
        "focus": ["[[Nota1]] revisar", "[[Nota2]] avanzar"],
        "pending": ["inbox item"],
        "attention": [],
    }
    out = rag._assemble_morning_brief(
        "2026-04-15", agenda_md="", parts=parts, continuity="",
    )
    assert "- [[Nota1]] revisar" in out
    assert "- [[Nota2]] avanzar" in out
    assert "- inbox item" in out
    assert "# Morning brief — 2026-04-15" in out


def test_assemble_prepends_agenda_before_focus():
    parts = {
        "yesterday": "",
        "focus": ["x"], "pending": [], "attention": [],
    }
    agenda = "## 📅 Hoy en la agenda\n- evento 1"
    out = rag._assemble_morning_brief(
        "2026-04-15", agenda_md=agenda, parts=parts, continuity="",
    )
    idx_agenda = out.index("📅")
    idx_focus = out.index("🎯")
    assert idx_agenda < idx_focus


def test_assemble_appends_continuity_at_end():
    parts = {"yesterday": "x", "focus": [], "pending": [], "attention": []}
    out = rag._assemble_morning_brief(
        "2026-04-15", agenda_md="", parts=parts,
        continuity="\n\n---\n_ayer cerraste con:_ [[2026-04-14-evening]]",
    )
    assert out.endswith("[[2026-04-14-evening]]")


def test_assemble_handles_non_list_fields_gracefully():
    parts = {"yesterday": "x", "focus": None, "pending": "not-a-list", "attention": []}
    out = rag._assemble_morning_brief(
        "2026-04-15", agenda_md="", parts=parts, continuity="",
    )
    # focus=None and pending="not-a-list" should not blow up; they're ignored.
    assert "📬" in out
    assert "🎯" not in out
    assert "🗂" not in out


def test_assemble_filters_empty_string_bullets():
    parts = {
        "yesterday": "",
        "focus": ["real bullet", "", "   "],
        "pending": [], "attention": [],
    }
    out = rag._assemble_morning_brief(
        "2026-04-15", agenda_md="", parts=parts, continuity="",
    )
    # Only the real bullet should render
    assert out.count("- ") == 1


# ── _generate_morning_json fallback behavior ────────────────────────────────


def test_generate_morning_json_returns_none_on_exception(monkeypatch):
    def _raise(*a, **kw): raise RuntimeError("offline")
    monkeypatch.setattr(rag.ollama, "chat", _raise)
    assert rag._generate_morning_json("x") is None


def test_generate_morning_json_returns_none_on_invalid_json(monkeypatch):
    class _Resp:
        class message:
            content = "not a json"
    monkeypatch.setattr(rag.ollama, "chat", lambda *a, **kw: _Resp())
    assert rag._generate_morning_json("x") is None


def test_generate_morning_json_parses_valid(monkeypatch):
    class _Resp:
        class message:
            content = '{"yesterday":"x","focus":["a"],"pending":[],"attention":[]}'
    monkeypatch.setattr(rag.ollama, "chat", lambda *a, **kw: _Resp())
    result = rag._generate_morning_json("x")
    assert result is not None
    assert result["yesterday"] == "x"
    assert result["focus"] == ["a"]
