"""Tests para `rag.integrations.reminders` — leaf ETL de Apple Reminders.

Surfaces cubiertas:
- `_fetch_reminders_due(now, horizon_days, max_items)` — incomplete
  reminders within today+horizon. Parsea las 3 shapes que el AppleScript
  puede emitir (4-field nuevo, 3-field legacy, 2-field super-legacy),
  bucketea (overdue/today/upcoming/undated), ordena por bucket asc, y
  trunca a max_items.
- `_fetch_completed_reminders(now, days, max_items)` — completed
  reminders en últimos `days`, sorted newest-first, trunc a max_items.

Mocking strategy:
- `_apple_enabled`, `_osascript`, `_parse_applescript_date` viven en
  `rag.__init__`. Module-body `from rag import …` resuelve a runtime
  contra el package, así que mockeamos `rag.<func>` (no `reminders_mod.<func>`).
- El conftest autouse setea `OBSIDIAN_RAG_NO_APPLE=1`, por eso los tests
  que ejercen happy path explícitamente monkeypatchean
  `rag._apple_enabled` a `lambda: True`.
- Para `_parse_applescript_date` en happy paths usamos el real (formato
  ISO `2026-04-25 10:00:00`) — esto ejercita el path completo de parsing
  que un AppleScript real podría devolver.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from rag.integrations import reminders as rem_mod


# ── _fetch_reminders_due: short-circuit paths ───────────────────────────


def test_fetch_reminders_due_returns_empty_when_apple_disabled(monkeypatch):
    """`OBSIDIAN_RAG_NO_APPLE=1` (autouse) → []. Verificamos con sentinel
    que `_osascript` NO se llame."""
    import rag
    monkeypatch.setattr(rag, "_apple_enabled", lambda: False)
    monkeypatch.setattr(
        rag, "_osascript",
        lambda *_a, **_kw: pytest.fail("osascript no debió llamarse"),
    )
    out = rem_mod._fetch_reminders_due(datetime(2026, 4, 25, 12, 0))
    assert out == []


def test_fetch_reminders_due_returns_empty_on_blank_osascript_output(monkeypatch):
    """Output vacío del osascript (timeout, no Reminders configurados) → []."""
    import rag
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: "")
    out = rem_mod._fetch_reminders_due(datetime(2026, 4, 25, 12, 0))
    assert out == []


# ── Pipe-shape parsing ──────────────────────────────────────────────────


def test_fetch_reminders_due_parses_4_field_shape(monkeypatch):
    """Shape canónica `id|name|due|list` (post 2026-04 schema). Cada
    field strip()'eado y mapeado al dict del output."""
    import rag
    now = datetime(2026, 4, 25, 12, 0)
    osascript_out = (
        "X-1|Pagar luz|2026-04-25 18:00:00|Personal\n"
        "X-2|Llamar dentista||Personal\n"
    )
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: osascript_out)

    items = rem_mod._fetch_reminders_due(now, horizon_days=1, max_items=10)
    assert len(items) == 2

    luz = next(i for i in items if i["name"] == "Pagar luz")
    assert luz["id"] == "X-1"
    assert luz["list"] == "Personal"
    assert luz["bucket"] == "today"
    assert luz["due"].startswith("2026-04-25T18:00")

    dent = next(i for i in items if i["name"] == "Llamar dentista")
    assert dent["id"] == "X-2"
    assert dent["due"] == ""
    assert dent["bucket"] == "undated"


def test_fetch_reminders_due_parses_3_field_legacy_shape(monkeypatch):
    """Shape vieja `name|due|list` (pre-2026-04, sin id). El módulo NO
    debe drop'ear estos reminders durante un revert del script — defaults
    `id=""`."""
    import rag
    now = datetime(2026, 4, 25, 12, 0)
    osascript_out = (
        "Comprar pan|2026-04-25 09:00:00|Casa\n"
    )
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: osascript_out)

    items = rem_mod._fetch_reminders_due(now, horizon_days=1, max_items=10)
    assert len(items) == 1
    assert items[0]["name"] == "Comprar pan"
    assert items[0]["id"] == ""
    assert items[0]["list"] == "Casa"


def test_fetch_reminders_due_buckets_overdue_today_upcoming(monkeypatch):
    """3 reminders, uno overdue, otro hoy más tarde, otro mañana. Cada
    uno cae al bucket correcto y el sort respeta `overdue < today < upcoming`."""
    import rag
    now = datetime(2026, 4, 25, 12, 0)
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    today_evening = now.replace(hour=18).strftime("%Y-%m-%d %H:%M:%S")
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    osascript_out = (
        f"R-1|Tarde|{tomorrow}|L\n"
        f"R-2|Now|{today_evening}|L\n"
        f"R-3|Atrasado|{yesterday}|L\n"
    )
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: osascript_out)

    items = rem_mod._fetch_reminders_due(now, horizon_days=2, max_items=10)
    assert [i["name"] for i in items] == ["Atrasado", "Now", "Tarde"]
    assert items[0]["bucket"] == "overdue"
    assert items[1]["bucket"] == "today"
    assert items[2]["bucket"] == "upcoming"


def test_fetch_reminders_due_filters_outside_horizon(monkeypatch):
    """Reminder con due > now + horizon_days+1 → SE EXCLUYE (no entra al
    output)."""
    import rag
    now = datetime(2026, 4, 25, 12, 0)
    far_future = (now + timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
    near = now.strftime("%Y-%m-%d %H:%M:%S")
    osascript_out = (
        f"R-FAR|Lejano|{far_future}|L\n"
        f"R-NEAR|Cerca|{near}|L\n"
    )
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: osascript_out)

    items = rem_mod._fetch_reminders_due(now, horizon_days=1, max_items=10)
    names = [i["name"] for i in items]
    assert "Lejano" not in names
    assert "Cerca" in names


def test_fetch_reminders_due_skips_unparseable_due_date(monkeypatch):
    """Si `_parse_applescript_date(due)` devuelve None, el reminder se
    descarta (no entra como undated; el due RAW estaba presente pero no
    parseable, así que es un `continue`)."""
    import rag
    now = datetime(2026, 4, 25, 12, 0)
    osascript_out = (
        "R-1|OK|2026-04-25 10:00:00|L\n"
        "R-2|RotoDate|fecha-en-klingon|L\n"
    )
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: osascript_out)

    items = rem_mod._fetch_reminders_due(now, horizon_days=2, max_items=10)
    names = [i["name"] for i in items]
    assert "OK" in names
    assert "RotoDate" not in names


def test_fetch_reminders_due_skips_lines_with_empty_name(monkeypatch):
    """Lines con name vacío (raro pero ocurre con scripts mal-shaped) se
    saltan."""
    import rag
    now = datetime(2026, 4, 25, 12, 0)
    osascript_out = (
        "R-1||2026-04-25 10:00:00|L\n"
        "R-2|OK||L\n"
    )
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: osascript_out)

    items = rem_mod._fetch_reminders_due(now, horizon_days=1, max_items=10)
    assert len(items) == 1
    assert items[0]["name"] == "OK"


def test_fetch_reminders_due_caps_to_max_items(monkeypatch):
    """Cap a max_items DESPUÉS de bucket-sort. Con 30 items y cap=5
    devuelve los 5 más prioritarios (overdue primero)."""
    import rag
    now = datetime(2026, 4, 25, 12, 0)
    lines = []
    for i in range(10):
        yesterday = (now - timedelta(days=i + 1)).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"R-OVD-{i}|Overdue {i}|{yesterday}|L")
    for i in range(10):
        lines.append(f"R-UND-{i}|Undated {i}||L")
    osascript_out = "\n".join(lines) + "\n"
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: osascript_out)

    items = rem_mod._fetch_reminders_due(now, horizon_days=2, max_items=5)
    assert len(items) == 5
    for it in items:
        assert it["bucket"] == "overdue"


# ── _fetch_completed_reminders ──────────────────────────────────────────


def test_fetch_completed_reminders_returns_empty_when_apple_disabled(monkeypatch):
    import rag
    monkeypatch.setattr(rag, "_apple_enabled", lambda: False)
    monkeypatch.setattr(
        rag, "_osascript",
        lambda *_a, **_kw: pytest.fail("osascript no debió llamarse"),
    )
    out = rem_mod._fetch_completed_reminders(datetime(2026, 4, 25))
    assert out == []


def test_fetch_completed_reminders_parses_and_sorts_newest_first(monkeypatch):
    """Happy path: 3 completed con dates distintas; output sorted
    newest-first."""
    import rag
    now = datetime(2026, 4, 25, 12, 0)
    osascript_out = (
        "Tarea vieja|2026-04-15 08:00:00|Personal\n"
        "Tarea reciente|2026-04-24 18:00:00|Personal\n"
        "Tarea media|2026-04-20 12:00:00|Personal\n"
    )
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: osascript_out)

    items = rem_mod._fetch_completed_reminders(now, days=30, max_items=10)
    assert len(items) == 3
    assert items[0]["name"] == "Tarea reciente"
    assert items[1]["name"] == "Tarea media"
    assert items[2]["name"] == "Tarea vieja"


def test_fetch_completed_reminders_filters_older_than_cutoff(monkeypatch):
    """Reminders completados HACE MÁS de `days` días → excluidos."""
    import rag
    now = datetime(2026, 4, 25, 12, 0)
    inside = (now - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
    outside = (now - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
    osascript_out = (
        f"Adentro|{inside}|L\n"
        f"Afuera|{outside}|L\n"
    )
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: osascript_out)

    items = rem_mod._fetch_completed_reminders(now, days=30, max_items=10)
    names = [i["name"] for i in items]
    assert "Adentro" in names
    assert "Afuera" not in names


def test_fetch_completed_reminders_skips_unparseable_date(monkeypatch):
    """Date que no parsea → el item se salta (no aparece undated en este
    bucket — completed sin date = noise)."""
    import rag
    now = datetime(2026, 4, 25, 12, 0)
    osascript_out = (
        "OK|2026-04-24 10:00:00|L\n"
        "RotoDate|fecha-en-klingon|L\n"
    )
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: osascript_out)

    items = rem_mod._fetch_completed_reminders(now, days=30, max_items=10)
    assert len(items) == 1
    assert items[0]["name"] == "OK"


def test_fetch_completed_reminders_caps_to_max_items(monkeypatch):
    """Cap a max_items DESPUÉS del sort (los más recientes sobreviven)."""
    import rag
    now = datetime(2026, 4, 25, 12, 0)
    lines = []
    for i in range(20):
        comp = (now - timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"Tarea {i}|{comp}|L")
    osascript_out = "\n".join(lines) + "\n"
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: osascript_out)

    items = rem_mod._fetch_completed_reminders(now, days=30, max_items=5)
    assert len(items) == 5
    assert items[0]["name"] == "Tarea 0"
    assert items[4]["name"] == "Tarea 4"


def test_fetch_completed_reminders_handles_missing_list_field(monkeypatch):
    """Lines de 2 fields (`name|date`) sin list → list defaultea a ""."""
    import rag
    now = datetime(2026, 4, 25, 12, 0)
    osascript_out = "Tarea sin lista|2026-04-24 10:00:00\n"
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_osascript", lambda *_a, **_kw: osascript_out)
    items = rem_mod._fetch_completed_reminders(now, days=30, max_items=10)
    assert len(items) == 1
    assert items[0]["list"] == ""


# ── AppleScript contract ────────────────────────────────────────────────


def test_reminders_script_iterates_pending_with_id_field():
    """Sanity del AppleScript embedded: itera lists, filtra por
    `completed is false`, emite `id|name|due|list`. Si alguien refactorea
    sin actualizar el comentario del módulo, el test falla."""
    s = rem_mod._REMINDERS_SCRIPT
    assert "completed is false" in s
    assert "id of _r" in s
    assert "name of _list" in s


def test_completed_reminders_script_filters_completed_with_completion_date():
    """Hermana del de arriba — `_COMPLETED_REMINDERS_SCRIPT` filtra
    `completed is true` y emite la `completion date` (no la due)."""
    s = rem_mod._COMPLETED_REMINDERS_SCRIPT
    assert "completed is true" in s
    assert "completion date" in s
