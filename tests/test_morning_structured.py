"""Tests para el refactor template-first del morning brief.

Cubre la parte determinística (agenda rendering, continuity footer, assembly)
y la mecánica de fallback cuando el LLM JSON falla. El LLM real no se toca —
monkeypatch del generador devuelve dicts conocidos.
"""
import json
import os
from datetime import datetime, timedelta

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
    (vault / "04-Archive/99-obsidian-system/99-Claude/reviews").mkdir(parents=True)
    (vault / "04-Archive/99-obsidian-system/99-Claude/reviews" / "2026-04-14-evening.md").write_text("x")
    target = datetime(2026, 4, 15)
    out = rag._yesterday_evening_link(target, vault)
    assert "2026-04-14-evening" in out
    assert "ayer cerraste con" in out


def test_continuity_link_empty_when_missing(tmp_path):
    vault = tmp_path / "vault"
    (vault / "04-Archive/99-obsidian-system/99-Claude/reviews").mkdir(parents=True)
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


@pytest.mark.requires_ollama
def test_generate_morning_json_parses_valid(monkeypatch):
    class _Resp:
        class message:
            content = '{"yesterday":"x","focus":["a"],"pending":[],"attention":[]}'
    monkeypatch.setattr(rag.ollama, "chat", lambda *a, **kw: _Resp())
    result = rag._generate_morning_json("x")
    assert result is not None
    assert result["yesterday"] == "x"
    assert result["focus"] == ["a"]


# ── _fetch_system_activity ───────────────────────────────────────────────────


def _redirect_system_paths(monkeypatch, tmp_path):
    """Apuntar las 3 rutas on-disk a tmp_path para aislar el test."""
    monkeypatch.setattr(rag, "AMBIENT_LOG_PATH", tmp_path / "ambient.jsonl")
    monkeypatch.setattr(rag, "FILING_BATCHES_DIR", tmp_path / "filing_batches")
    monkeypatch.setattr(rag, "TUNE_LOG_PATH", tmp_path / "tune.jsonl")


def test_fetch_system_activity_empty_when_files_missing(monkeypatch, tmp_path):
    _redirect_system_paths(monkeypatch, tmp_path)
    now = datetime.now()
    out = rag._fetch_system_activity(now, lookback_hours=36)
    assert out == {
        "ambient_events": 0, "ambient_wikilinks": 0,
        "filing_moves": 0, "archive_moves": 0,
        "tune_runs": 0, "tune_last_delta": None,
    }


def test_fetch_system_activity_counts_ambient_events(monkeypatch, tmp_path):
    _redirect_system_paths(monkeypatch, tmp_path)
    now = datetime(2026, 4, 15, 7, 0, 0)
    recent = (now - timedelta(hours=3)).isoformat(timespec="seconds")
    old = (now - timedelta(hours=50)).isoformat(timespec="seconds")
    lines = [
        # In-window: 3 wikilinks applied, 0 dupes, 0 related → counts
        json.dumps({"ts": recent, "wikilinks_applied": 3, "dupes": [], "related_count": 0}),
        # In-window: dupes present → counts as event, 0 wikilinks
        json.dumps({"ts": recent, "wikilinks_applied": 0,
                    "dupes": [{"path": "x", "sim": 0.9}], "related_count": 0}),
        # In-window: only related → counts
        json.dumps({"ts": recent, "wikilinks_applied": 0, "dupes": [], "related_count": 2}),
        # In-window but no signals → does NOT count
        json.dumps({"ts": recent, "wikilinks_applied": 0, "dupes": [], "related_count": 0}),
        # Out of window → does NOT count
        json.dumps({"ts": old, "wikilinks_applied": 99, "dupes": [], "related_count": 0}),
        # Broken json line → ignored
        "{not json",
    ]
    rag.AMBIENT_LOG_PATH.write_text("\n".join(lines) + "\n")
    out = rag._fetch_system_activity(now, lookback_hours=36)
    assert out["ambient_events"] == 3
    assert out["ambient_wikilinks"] == 3


def test_fetch_system_activity_filing_and_archive(monkeypatch, tmp_path):
    _redirect_system_paths(monkeypatch, tmp_path)
    now = datetime.now()
    rag.FILING_BATCHES_DIR.mkdir(parents=True)

    # Archive batch in-window, 2 rows
    archive_batch = rag.FILING_BATCHES_DIR / "archive-20260415-120000.jsonl"
    archive_batch.write_text(
        json.dumps({"src": "a.md", "dst": "04-Archive/a.md"}) + "\n"
        + json.dumps({"src": "b.md", "dst": "04-Archive/b.md"}) + "\n"
    )
    # Plain filing batch in-window, 1 row
    filing_batch = rag.FILING_BATCHES_DIR / "20260415-100000.jsonl"
    filing_batch.write_text(json.dumps({"src": "c.md", "dst": "02-Areas/c.md"}) + "\n")

    # Out-of-window batch (set old mtime) → should NOT count
    old_batch = rag.FILING_BATCHES_DIR / "archive-20200101-000000.jsonl"
    old_batch.write_text(json.dumps({"src": "d.md"}) + "\n")
    old_ts = (now - timedelta(hours=200)).timestamp()
    os.utime(old_batch, (old_ts, old_ts))

    out = rag._fetch_system_activity(now, lookback_hours=36)
    # Total moves = 3 (2 archive + 1 filing); archive_moves = 2
    assert out["filing_moves"] == 3
    assert out["archive_moves"] == 2


def test_fetch_system_activity_tune_runs_and_last_delta(monkeypatch, tmp_path):
    _redirect_system_paths(monkeypatch, tmp_path)
    now = datetime(2026, 4, 15, 9, 0, 0)
    t1 = (now - timedelta(hours=20)).isoformat(timespec="seconds")
    t2 = (now - timedelta(hours=2)).isoformat(timespec="seconds")   # más reciente
    t_old = (now - timedelta(hours=100)).isoformat(timespec="seconds")
    lines = [
        json.dumps({"ts": t1, "delta": 0.008}),
        json.dumps({"ts": t2, "delta": 0.012}),
        json.dumps({"ts": t_old, "delta": 0.999}),   # out of window, ignored
    ]
    rag.TUNE_LOG_PATH.write_text("\n".join(lines) + "\n")
    out = rag._fetch_system_activity(now, lookback_hours=36)
    assert out["tune_runs"] == 2
    assert out["tune_last_delta"] == pytest.approx(0.012)


def test_fetch_system_activity_respects_window(monkeypatch, tmp_path):
    _redirect_system_paths(monkeypatch, tmp_path)
    now = datetime(2026, 4, 15, 9, 0, 0)
    too_old = (now - timedelta(hours=48)).isoformat(timespec="seconds")
    rag.AMBIENT_LOG_PATH.write_text(
        json.dumps({"ts": too_old, "wikilinks_applied": 5,
                    "dupes": [], "related_count": 0}) + "\n"
    )
    rag.TUNE_LOG_PATH.write_text(
        json.dumps({"ts": too_old, "delta": 0.5}) + "\n"
    )
    out = rag._fetch_system_activity(now, lookback_hours=36)
    assert out["ambient_events"] == 0
    assert out["ambient_wikilinks"] == 0
    assert out["tune_runs"] == 0
    assert out["tune_last_delta"] is None


# ── _render_system_activity_section ─────────────────────────────────────────


def test_render_system_activity_empty_all_zeros():
    act = {
        "ambient_events": 0, "ambient_wikilinks": 0,
        "filing_moves": 0, "archive_moves": 0,
        "tune_runs": 0, "tune_last_delta": None,
    }
    assert rag._render_system_activity_section(act) == ""


def test_render_system_activity_ignores_tune_delta_alone():
    # Only tune_last_delta being non-None should NOT force the section.
    act = {
        "ambient_events": 0, "ambient_wikilinks": 0,
        "filing_moves": 0, "archive_moves": 0,
        "tune_runs": 0, "tune_last_delta": 0.05,
    }
    assert rag._render_system_activity_section(act) == ""


def test_render_system_activity_ambient_bullet():
    act = {
        "ambient_events": 3, "ambient_wikilinks": 7,
        "filing_moves": 0, "archive_moves": 0,
        "tune_runs": 0, "tune_last_delta": None,
    }
    out = rag._render_system_activity_section(act)
    assert "## ⚙️ Lo que el sistema hizo solo" in out
    assert "ambient aplicó 7" in out
    assert "3 capturas" in out


def test_render_system_activity_archive_bullet():
    act = {
        "ambient_events": 0, "ambient_wikilinks": 0,
        "filing_moves": 2, "archive_moves": 2,
        "tune_runs": 0, "tune_last_delta": None,
    }
    out = rag._render_system_activity_section(act)
    assert "archiver movió 2 dead notes a 04-Archive" in out
    # non-archive filing is 0 → no filing bullet
    assert "filing movió" not in out


def test_render_system_activity_mixed_archive_and_filing():
    act = {
        "ambient_events": 0, "ambient_wikilinks": 0,
        "filing_moves": 5, "archive_moves": 2,
        "tune_runs": 0, "tune_last_delta": None,
    }
    out = rag._render_system_activity_section(act)
    assert "archiver movió 2" in out
    assert "filing movió 3" in out


def test_render_system_activity_tune_bullet_with_delta():
    act = {
        "ambient_events": 0, "ambient_wikilinks": 0,
        "filing_moves": 0, "archive_moves": 0,
        "tune_runs": 1, "tune_last_delta": 0.012,
    }
    out = rag._render_system_activity_section(act)
    assert "rag tune recalibró pesos (+0.012 objetivo)" in out


def test_render_system_activity_tune_bullet_no_delta():
    act = {
        "ambient_events": 0, "ambient_wikilinks": 0,
        "filing_moves": 0, "archive_moves": 0,
        "tune_runs": 2, "tune_last_delta": None,
    }
    out = rag._render_system_activity_section(act)
    assert "rag tune corrió 2 veces" in out


def test_render_system_activity_combined():
    act = {
        "ambient_events": 2, "ambient_wikilinks": 5,
        "filing_moves": 3, "archive_moves": 3,
        "tune_runs": 1, "tune_last_delta": 0.008,
    }
    out = rag._render_system_activity_section(act)
    # All 3 bullets present
    assert "ambient aplicó 5" in out
    assert "archiver movió 3" in out
    assert "rag tune recalibró pesos (+0.008 objetivo)" in out


# ── _assemble_morning_brief with system_md ──────────────────────────────────


def test_assemble_includes_system_md_between_agenda_and_focus():
    parts = {"yesterday": "ayer", "focus": ["x"], "pending": [], "attention": []}
    agenda = "## 📅 Hoy en la agenda\n- reu 10h"
    system = "## ⚙️ Lo que el sistema hizo solo\n- ambient aplicó 3"
    out = rag._assemble_morning_brief(
        "2026-04-15", agenda_md=agenda, parts=parts, continuity="",
        system_md=system,
    )
    idx_agenda = out.index("📅")
    idx_system = out.index("⚙️")
    idx_focus = out.index("🎯")
    assert idx_agenda < idx_system < idx_focus


def test_assemble_skips_empty_system_md():
    parts = {"yesterday": "ayer", "focus": [], "pending": [], "attention": []}
    out = rag._assemble_morning_brief(
        "2026-04-15", agenda_md="", parts=parts, continuity="", system_md="",
    )
    assert "⚙️" not in out


def test_assemble_system_md_default_noop_preserves_legacy():
    # Old callers not passing system_md still work — no section rendered.
    parts = {"yesterday": "ayer", "focus": [], "pending": [], "attention": []}
    out = rag._assemble_morning_brief(
        "2026-04-15", agenda_md="", parts=parts, continuity="",
    )
    assert "⚙️" not in out


# ── _collect_morning_evidence wires system_activity ─────────────────────────


def test_collect_morning_wires_system_activity(monkeypatch, tmp_path):
    _redirect_system_paths(monkeypatch, tmp_path)
    now = datetime.now()
    recent = (now - timedelta(hours=3)).isoformat(timespec="seconds")
    rag.AMBIENT_LOG_PATH.write_text(
        json.dumps({"ts": recent, "wikilinks_applied": 4,
                    "dupes": [], "related_count": 0}) + "\n"
    )
    monkeypatch.setattr(rag, "_fetch_weather_rain", lambda *a, **kw: None)
    monkeypatch.setattr(rag, "_fetch_calendar_today", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "_fetch_reminders_due", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "_fetch_mail_unread", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "_fetch_whatsapp_unread", lambda *a, **kw: [])
    monkeypatch.setattr(rag, "_fetch_recent_queries", lambda *a, **kw: [])
    vault = tmp_path / "vault"
    vault.mkdir()
    ev = rag._collect_morning_evidence(
        now, vault, tmp_path / "q.jsonl", tmp_path / "c.jsonl",
        lookback_hours=36,
    )
    act = ev.get("system_activity")
    assert act is not None
    assert act["ambient_wikilinks"] == 4
    assert act["ambient_events"] == 1


# ── _fetch_mail_unread: body_preview + is_vip ───────────────────────────────


def _stub_mail_osascript(monkeypatch, payload: str):
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: payload)


def test_fetch_mail_unread_returns_body_preview_and_is_vip(monkeypatch, tmp_path):
    cfg = tmp_path / "mail-vip.json"
    cfg.write_text(json.dumps({"vips": ["boss@acme.com"]}))
    monkeypatch.setattr(rag, "MAIL_VIP_CONFIG_PATH", cfg)
    payload = (
        "Quarterly review|boss@acme.com|today|<p>Hola, <b>revisamos</b> Q1.</p>\n"
        "Newsletter|news@medium.com|today|Top stories of the week\n"
    )
    _stub_mail_osascript(monkeypatch, payload)
    items = rag._fetch_mail_unread()
    assert len(items) == 2
    # VIP sorted to top
    assert items[0]["sender"] == "boss@acme.com"
    assert items[0]["is_vip"] is True
    assert "body_preview" in items[0]
    assert items[0]["body_preview"] == "Hola, revisamos Q1."
    assert items[1]["is_vip"] is False
    assert items[1]["body_preview"] == "Top stories of the week"


def test_fetch_mail_unread_body_preview_capped_at_200(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "MAIL_VIP_CONFIG_PATH", tmp_path / "missing.json")
    long_body = "x" * 500
    payload = f"Subj|sender@x.com|today|{long_body}\n"
    _stub_mail_osascript(monkeypatch, payload)
    items = rag._fetch_mail_unread()
    assert len(items[0]["body_preview"]) == 200


def test_load_mail_vips_missing_file_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "MAIL_VIP_CONFIG_PATH", tmp_path / "nope.json")
    assert rag._load_mail_vips() == set()


def test_fetch_mail_unread_missing_vip_file_no_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "MAIL_VIP_CONFIG_PATH", tmp_path / "nope.json")
    payload = "Subj|x@y.com|today|body\n"
    _stub_mail_osascript(monkeypatch, payload)
    items = rag._fetch_mail_unread()
    assert items[0]["is_vip"] is False


def test_load_mail_vips_malformed_json_warns_returns_empty(
    monkeypatch, tmp_path, capsys,
):
    cfg = tmp_path / "mail-vip.json"
    cfg.write_text("{not json")
    monkeypatch.setattr(rag, "MAIL_VIP_CONFIG_PATH", cfg)
    out = rag._load_mail_vips()
    assert out == set()
    captured = capsys.readouterr()
    assert "mail-vip" in (captured.out + captured.err).lower()


def test_load_mail_vips_wrong_shape_returns_empty(monkeypatch, tmp_path):
    cfg = tmp_path / "mail-vip.json"
    cfg.write_text(json.dumps(["not", "a", "dict"]))
    monkeypatch.setattr(rag, "MAIL_VIP_CONFIG_PATH", cfg)
    assert rag._load_mail_vips() == set()


def test_is_vip_sender_substring_match():
    vips = {"boss@acme.com"}
    # Mail puts senders as `Name <email>` typically
    assert rag._is_vip_sender("Big Boss <boss@acme.com>", vips) is True
    assert rag._is_vip_sender("Newsletter <news@x.com>", vips) is False
    assert rag._is_vip_sender("", vips) is False
    assert rag._is_vip_sender("anyone@x.com", set()) is False


def test_strip_html_to_preview_basic():
    out = rag._strip_html_to_preview("<p>Hola <b>mundo</b></p>\n\n<br>line2", cap=200)
    assert out == "Hola mundo line2"
    assert rag._strip_html_to_preview("", cap=10) == ""
    assert rag._strip_html_to_preview("a" * 50, cap=10) == "a" * 10


def test_fetch_mail_unread_vips_survive_max_items_cap(monkeypatch, tmp_path):
    cfg = tmp_path / "mail-vip.json"
    cfg.write_text(json.dumps({"vips": ["vip@x.com"]}))
    monkeypatch.setattr(rag, "MAIL_VIP_CONFIG_PATH", cfg)
    # 12 non-vip first, 1 vip last → vip must survive max_items=10
    lines = [f"S{i}|noise{i}@x.com|today|b{i}" for i in range(12)]
    lines.append("Important|vip@x.com|today|critical")
    payload = "\n".join(lines) + "\n"
    _stub_mail_osascript(monkeypatch, payload)
    items = rag._fetch_mail_unread(max_items=10)
    assert len(items) == 10
    assert items[0]["is_vip"] is True
    assert items[0]["sender"] == "vip@x.com"


# ── Morning prompts: VIP marker + body preview ──────────────────────────────


def test_render_morning_structured_prompt_includes_vip_marker():
    ev = {
        "mail_unread": [
            {"subject": "Quarter", "sender": "boss@acme.com",
             "received": "today", "body_preview": "Revisemos Q1",
             "is_vip": True},
            {"subject": "Newsletter", "sender": "news@x.com",
             "received": "today", "body_preview": "weekly", "is_vip": False},
        ],
    }
    prompt = rag._render_morning_structured_prompt("2026-04-16", ev)
    assert "**VIP**" in prompt
    assert "Quarter" in prompt
    assert "Revisemos Q1" in prompt
    # Non-VIP must not get the marker on its own line
    lines = [l for l in prompt.splitlines() if "Newsletter" in l]
    assert lines and "**VIP**" not in lines[0]


def test_render_morning_prompt_legacy_includes_vip_marker():
    ev = {
        "recent_notes": [], "inbox_pending": [], "todos": [],
        "new_contradictions": [], "low_conf_queries": [],
        "mail_unread": [
            {"subject": "Urgent", "sender": "boss@acme.com",
             "received": "today", "body_preview": "Hoy a las 4",
             "is_vip": True},
        ],
    }
    prompt = rag._render_morning_prompt("2026-04-16", ev)
    assert "**VIP**" in prompt
    assert "Hoy a las 4" in prompt


def test_render_morning_structured_prompt_no_mail_section_when_empty():
    ev = {"mail_unread": []}
    prompt = rag._render_morning_structured_prompt("2026-04-16", ev)
    assert "Mail no leído" not in prompt
