"""Tests para `rag followup` cross-resolve contra Apple Reminders completadas
(T2B). Cubre:
  - `_fetch_completed_reminders`: parsing + filtro por ventana + sort desc.
  - `_match_loop_to_completed_reminder`: Jaccard fuzzy sobre tokens ≥3 chars.
  - `_classify_followup_loop` fast path (salta retrieve+LLM cuando hay match).
  - `find_followup_loops` integration: auto-fetch + classify end-to-end.

No tocamos osascript real — monkeypatch de `_osascript` + `retrieve` + judge.
"""
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import rag


# ── _fetch_completed_reminders ──────────────────────────────────────────────


def _mk_apple_date_str(dt: datetime) -> str:
    """Format a datetime like AppleScript's default date string (ISO-ish
    variant that `_parse_applescript_date` already accepts).
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def test_fetch_completed_parses_rows(monkeypatch):
    now = datetime(2026, 4, 15, 12, 0, 0)
    out = "\n".join([
        f"comprar pan|{_mk_apple_date_str(now - timedelta(days=1))}|Recordatorios",
        f"pagar alquiler|{_mk_apple_date_str(now - timedelta(days=3))}|Trabajo",
    ])
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: out)
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    items = rag._fetch_completed_reminders(now, days=30)
    assert len(items) == 2
    assert items[0]["name"] == "comprar pan"  # newer first
    assert items[0]["list"] == "Recordatorios"
    assert items[1]["name"] == "pagar alquiler"


def test_fetch_completed_window_filter(monkeypatch):
    now = datetime(2026, 4, 15, 12, 0, 0)
    out = "\n".join([
        f"reciente|{_mk_apple_date_str(now - timedelta(days=5))}|L",
        f"vieja|{_mk_apple_date_str(now - timedelta(days=100))}|L",
    ])
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: out)
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    items = rag._fetch_completed_reminders(now, days=30)
    names = {it["name"] for it in items}
    assert "reciente" in names
    assert "vieja" not in names


def test_fetch_completed_empty_when_apple_disabled(monkeypatch):
    monkeypatch.setattr(rag, "_apple_enabled", lambda: False)
    # Si apple está desactivado no debería ni tocar osascript.
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no tocar")))
    assert rag._fetch_completed_reminders(datetime.now()) == []


def test_fetch_completed_empty_on_osascript_fail(monkeypatch):
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: "")
    assert rag._fetch_completed_reminders(datetime.now()) == []


def test_fetch_completed_skips_unparseable_lines(monkeypatch):
    now = datetime(2026, 4, 15, 12, 0, 0)
    out = "\n".join([
        f"ok|{_mk_apple_date_str(now - timedelta(days=1))}|L",
        "malformed-line-no-pipes",
        "solo-dos|campos",           # falta date parseable
        f"|{_mk_apple_date_str(now)}|L",  # name vacío → skip
    ])
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: out)
    items = rag._fetch_completed_reminders(now)
    assert len(items) == 1
    assert items[0]["name"] == "ok"


def test_fetch_completed_max_items_cap(monkeypatch):
    now = datetime(2026, 4, 15, 12, 0, 0)
    many = "\n".join(
        f"tarea {i}|{_mk_apple_date_str(now - timedelta(hours=i))}|L"
        for i in range(10)
    )
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *a, **kw: many)
    items = rag._fetch_completed_reminders(now, max_items=3)
    assert len(items) == 3


# ── _match_loop_to_completed_reminder ───────────────────────────────────────


def test_match_exact_wins():
    completed = [
        {"name": "comprar pan", "completed_date": "2026-04-15T10:00", "list": "L"},
    ]
    m = rag._match_loop_to_completed_reminder("comprar pan", completed)
    assert m is not None
    assert m["name"] == "comprar pan"


def test_match_accent_insensitive():
    completed = [
        {"name": "pagar alquiler", "completed_date": "2026-04-15T10:00", "list": "L"},
    ]
    m = rag._match_loop_to_completed_reminder("Pagar Alquiler", completed)
    assert m is not None


def test_match_jaccard_above_threshold():
    # "comprar pan coto" vs "comprar pan" → tokens {comprar, pan, coto} vs
    # {comprar, pan} → Jaccard = 2/3 ≈ 0.67 → match.
    completed = [{"name": "comprar pan", "completed_date": "x", "list": "L"}]
    m = rag._match_loop_to_completed_reminder("comprar pan coto", completed)
    assert m is not None


def test_match_below_threshold_returns_none():
    # "revisar PR" vs "comprar pan" → 0 overlap.
    completed = [{"name": "comprar pan", "completed_date": "x", "list": "L"}]
    m = rag._match_loop_to_completed_reminder("revisar PR", completed)
    assert m is None


def test_match_picks_best_of_multiple():
    completed = [
        {"name": "comprar pan coto", "completed_date": "2026-04-10T10:00", "list": "L"},
        {"name": "comprar pan", "completed_date": "2026-04-15T10:00", "list": "L"},
    ]
    m = rag._match_loop_to_completed_reminder("comprar pan", completed)
    # "comprar pan" exacto (jaccard=1.0) gana al parcial (jaccard≈0.67).
    assert m["name"] == "comprar pan"


def test_match_empty_inputs():
    assert rag._match_loop_to_completed_reminder("", [{"name": "x", "completed_date": "x", "list": "L"}]) is None
    assert rag._match_loop_to_completed_reminder("algo", []) is None


def test_match_skips_short_token_only_loops():
    # Todos los tokens de loop son <3 chars → no se puede puntuar → None.
    completed = [{"name": "un x y z", "completed_date": "x", "list": "L"}]
    m = rag._match_loop_to_completed_reminder("X Y Z", completed)
    assert m is None


# ── _classify_followup_loop fast path ───────────────────────────────────────


def test_classify_fast_path_resolves_without_retrieve(monkeypatch):
    """Si hay match con completed reminder, NI retrieve NI judge son llamados."""
    now = datetime(2026, 4, 15, 12, 0, 0)
    loop = {
        "loop_text": "comprar pan",
        "source_note": "02-Areas/shopping.md",
        "extracted_at": (now - timedelta(days=3)).isoformat(),
        "kind": "checkbox",
    }
    completed = [
        {"name": "comprar pan", "completed_date": "2026-04-14T09:30", "list": "Recordatorios"},
    ]
    # Si alguno de estos se llama, el test falla — el fast path no debería tocarlos.
    sentinel = MagicMock(side_effect=AssertionError("retrieve NO debería llamarse"))
    monkeypatch.setattr(rag, "retrieve", sentinel)
    judge = MagicMock(side_effect=AssertionError("judge NO debería llamarse"))

    fake_col = MagicMock()
    out = rag._classify_followup_loop(
        fake_col, loop, now,
        completed_reminders=completed, judge_fn=judge,
    )
    assert out["status"] == "resolved"
    assert out["resolved_by"] == "reminder"
    assert "2026-04-14" in out["reason"]
    assert "Recordatorios" in out["resolution_path"]


def test_classify_no_completed_falls_to_retrieve(monkeypatch):
    """Sin matches de reminders, el camino viejo debe activarse."""
    now = datetime(2026, 4, 15, 12, 0, 0)
    loop = {
        "loop_text": "algo totalmente distinto",
        "source_note": "02-Areas/x.md",
        "extracted_at": (now - timedelta(days=2)).isoformat(),
        "kind": "inline",
    }
    completed = [{"name": "comprar pan", "completed_date": "x", "list": "L"}]
    # retrieve devuelve vacío → sin candidates → out queda activo.
    monkeypatch.setattr(
        rag, "retrieve",
        lambda *a, **kw: {"metas": [], "docs": [], "scores": []},
    )
    out = rag._classify_followup_loop(
        MagicMock(), loop, now, completed_reminders=completed,
    )
    assert out["status"] == "activo"
    assert out["resolved_by"] is None
    assert out["resolution_path"] is None


def test_classify_stale_threshold_respected_on_fast_path():
    """El fast path hace status=resolved independiente de stale_days."""
    now = datetime(2026, 4, 15, 12, 0, 0)
    # 30 días viejo → sería stale, pero tiene reminder match.
    loop = {
        "loop_text": "comprar pan",
        "source_note": "x.md",
        "extracted_at": (now - timedelta(days=30)).isoformat(),
        "kind": "checkbox",
    }
    completed = [{"name": "comprar pan", "completed_date": "2026-04-15T09:00", "list": "L"}]
    out = rag._classify_followup_loop(
        MagicMock(), loop, now,
        stale_days=14, completed_reminders=completed,
    )
    assert out["status"] == "resolved"  # resolved beats stale


# ── find_followup_loops integration ────────────────────────────────────────


def test_find_loops_auto_fetches_completed_reminders(monkeypatch, tmp_path):
    """Si no pasamos `completed_reminders`, la fn lo fetchea."""
    vault = tmp_path / "vault"
    (vault / "02-Areas").mkdir(parents=True)
    note = vault / "02-Areas" / "shopping.md"
    note.write_text("# Shopping\n\n- [ ] comprar pan\n")
    # Make the note recent so it's inside the window.
    import os, time
    os.utime(note, (time.time(), time.time()))

    calls = {"fetched": 0}
    def _fake_fetch(now, days=30, max_items=200):
        calls["fetched"] += 1
        return [{"name": "comprar pan", "completed_date": "2026-04-15T09:00", "list": "L"}]
    monkeypatch.setattr(rag, "_fetch_completed_reminders", _fake_fetch)

    fake_col = MagicMock()
    items = rag.find_followup_loops(fake_col, vault, days=7, judge_fn=None)
    assert calls["fetched"] == 1
    # El loop "comprar pan" debe aparecer como resolved-by-reminder.
    resolved = [it for it in items if it["status"] == "resolved"]
    assert any(it.get("resolved_by") == "reminder" for it in resolved)


def test_find_loops_uses_injected_reminders_no_fetch(monkeypatch, tmp_path):
    """Si pasamos `completed_reminders`, NO se fetchea."""
    vault = tmp_path / "vault"
    vault.mkdir()
    # No notes — no loops — no retrieve calls.
    monkeypatch.setattr(
        rag, "_fetch_completed_reminders",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no fetch")),
    )
    items = rag.find_followup_loops(
        MagicMock(), vault, days=7, completed_reminders=[],
    )
    assert items == []
