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


def test_agenda_heavy_rain_triggers_indoor_hint():
    ev = {"weather_rain": {"summary": "100% a las 18h", "max_chance": 100}}
    out = rag._render_morning_agenda_section(ev)
    assert "💡" in out
    assert "indoor" in out.lower()


def test_agenda_light_rain_no_hint():
    ev = {"weather_rain": {"summary": "40% posible", "max_chance": 40}}
    out = rag._render_morning_agenda_section(ev)
    assert "💡" not in out


def test_agenda_rain_no_max_chance_no_hint():
    # Some wttr.in payloads only give `summary`, no max_chance (or 0).
    ev = {"weather_rain": {"summary": "llovizna"}}
    out = rag._render_morning_agenda_section(ev)
    assert "💡" not in out


# ── _dedup_todos_vs_reminders ───────────────────────────────────────────────


def test_dedup_exact_match_drops_todo():
    todos = [{"title": "Shopping", "todo": "comprar pan"}]
    reminders = [{"name": "Comprar pan"}]
    out = rag._dedup_todos_vs_reminders(todos, reminders)
    assert out == []


def test_dedup_accent_fold():
    todos = [{"todo": "pagar alquiler"}]
    reminders = [{"name": "Pagar Alquiler"}]
    out = rag._dedup_todos_vs_reminders(todos, reminders)
    assert out == []


def test_dedup_substring_above_threshold():
    todos = [{"todo": "comprar pan"}]
    reminders = [{"name": "comprar pan en Coto"}]
    # tokens: {comprar, pan} vs {comprar, pan, coto} → 2/3 ≥ 0.6
    out = rag._dedup_todos_vs_reminders(todos, reminders)
    assert out == []


def test_dedup_preserves_unrelated():
    todos = [{"todo": "revisar PR"}, {"todo": "comprar pan"}]
    reminders = [{"name": "comprar pan"}]
    out = rag._dedup_todos_vs_reminders(todos, reminders)
    assert len(out) == 1
    assert out[0]["todo"] == "revisar PR"


def test_dedup_handles_list_todo_value():
    todos = [{"todo": ["comprar", "pan"]}]
    reminders = [{"name": "comprar pan"}]
    out = rag._dedup_todos_vs_reminders(todos, reminders)
    assert out == []


def test_dedup_empty_inputs_noop():
    assert rag._dedup_todos_vs_reminders([], [{"name": "x"}]) == []
    todos = [{"todo": "x"}]
    assert rag._dedup_todos_vs_reminders(todos, []) == todos


def test_dedup_short_tokens_keep_todo():
    # Very short words (<3 chars) don't contribute to tokens → treated as
    # non-comparable → todo kept.
    todos = [{"todo": "ir a X"}]
    reminders = [{"name": "Y"}]
    out = rag._dedup_todos_vs_reminders(todos, reminders)
    assert out == todos


def test_dedup_threshold_below_cut():
    # 1/3 tokens overlap = 0.33, below 0.6 → keep.
    todos = [{"todo": "comprar queso y vino"}]
    reminders = [{"name": "comprar pan"}]
    # tokens todo: {comprar, queso, vino}; rem: {comprar, pan}
    # jaccard = 1 / 4 = 0.25
    out = rag._dedup_todos_vs_reminders(todos, reminders)
    assert len(out) == 1


# ── _collect_morning_evidence wires dedup ────────────────────────────────────

def test_collect_morning_wires_dedup(monkeypatch, tmp_path):
    # Smoke-test: pass evidence through and verify dedup fires. We don't exercise
    # the full scanner — just stub the expensive fetches and inspect output.
    monkeypatch.setattr(rag, "_fetch_weather_rain", lambda *a, **kw: None)
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda *a, **kw: [])
    monkeypatch.setattr(
        rag, "_fetch_reminders_due",
        lambda *a, **kw: [{"name": "comprar pan", "bucket": "hoy", "due": "", "list": "R"}],
    )
    monkeypatch.setattr(rag, "_fetch_mail_unread", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "_fetch_whatsapp_unread", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "_fetch_recent_queries", lambda *a, **kw: [])
    # Prepare a vault with one note carrying a todo that matches the reminder.
    vault = tmp_path / "vault"
    (vault / "02-Areas").mkdir(parents=True)
    (vault / "02-Areas" / "shopping.md").write_text(
        "---\ntitle: Shopping\ntodo: comprar pan\n---\n\nbody\n"
    )
    ev = rag._collect_morning_evidence(
        rag.datetime.now(), vault, tmp_path / "q.jsonl",
        tmp_path / "c.jsonl", lookback_hours=48,
    )
    # The vault todo "comprar pan" should have been dropped because the
    # reminder has the same task text.
    texts = [(t.get("todo") or "") for t in ev["todos"]]
    assert "comprar pan" not in texts


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
