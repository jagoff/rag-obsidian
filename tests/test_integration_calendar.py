"""Tests para `rag.integrations.calendar` — leaf ETL de Apple Calendar.

Surfaces cubiertas:
- `_fetch_calendar_today(max_events)` — events del día via icalBuddy.
  Parsea title-line + property-lines indented (datetime), ordena por start,
  trunca a `max_events`.
- `_fetch_calendar_ahead(days_ahead, max_events)` — ventana de N días
  con `date_label` + `time_range`. `days_ahead < 0` → []. `days_ahead == 0`
  usa `eventsToday`; `> 0` usa `eventsToday+N`.

Mocking strategy:
- `_apple_enabled` y `_icalbuddy_path` viven en `rag.__init__`. El módulo
  los importa con `from rag import …` dentro del cuerpo, así que mockeamos
  `rag.<func>` (no `calendar_mod.<func>`) — la lookup re-resuelve al
  attribute del package en cada call.
- `subprocess.run` se mockea inyectando un fake con `monkeypatch.setattr`
  sobre `subprocess.run` global; el módulo hace `import subprocess`
  inline así el mock toma efecto.
- El conftest autouse setea `OBSIDIAN_RAG_NO_APPLE=1`, así que tests
  que ejercen el happy path explícitamente monkeypatchean
  `rag._apple_enabled` a `lambda: True`.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.integrations import calendar as cal_mod


# ── _fetch_calendar_today: disabled / missing icalbuddy ──────────────────


def test_fetch_calendar_today_returns_empty_when_apple_disabled(monkeypatch):
    """Cuando `OBSIDIAN_RAG_NO_APPLE=1` (autouse), el helper short-circuitea
    SIN buscar icalbuddy ni invocar subprocess."""
    import rag
    monkeypatch.setattr(rag, "_apple_enabled", lambda: False)
    monkeypatch.setattr(
        rag, "_icalbuddy_path",
        lambda: pytest.fail("_icalbuddy_path no debió llamarse"),
    )
    out = cal_mod._fetch_calendar_today()
    assert out == []


def test_fetch_calendar_today_returns_empty_when_icalbuddy_missing(monkeypatch):
    """`_icalbuddy_path() → None` (binary not installed) → []."""
    import rag
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_icalbuddy_path", lambda: None)
    out = cal_mod._fetch_calendar_today()
    assert out == []


def test_fetch_calendar_today_returns_empty_on_subprocess_timeout(monkeypatch):
    """Si `subprocess.run` levanta `TimeoutExpired` / `FileNotFoundError` /
    `OSError`, devolvemos [] silenciosamente."""
    import rag
    import subprocess
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_icalbuddy_path", lambda: "/fake/icalBuddy")

    def _boom_run(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd="icalBuddy", timeout=10.0)

    monkeypatch.setattr(subprocess, "run", _boom_run)
    out = cal_mod._fetch_calendar_today()
    assert out == []


def test_fetch_calendar_today_returns_empty_on_nonzero_returncode(monkeypatch):
    """returncode != 0 → []. Pasa con icalBuddy quejándose de que no hay
    Calendar.app, sin permisos, etc."""
    import rag
    import subprocess
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_icalbuddy_path", lambda: "/fake/icalBuddy")

    fake_res = SimpleNamespace(returncode=1, stdout="", stderr="error")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: fake_res)
    assert cal_mod._fetch_calendar_today() == []


def test_fetch_calendar_today_returns_empty_on_blank_stdout(monkeypatch):
    """returncode 0 + stdout vacío (no hay events hoy) → []."""
    import rag
    import subprocess
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_icalbuddy_path", lambda: "/fake/icalBuddy")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout="   \n  "),
    )
    assert cal_mod._fetch_calendar_today() == []


# ── _fetch_calendar_today: happy path / parsing ─────────────────────────


def test_fetch_calendar_today_parses_title_and_time_range(monkeypatch):
    """Happy path: 2 events con time-range, sorted asc por start."""
    import rag
    import subprocess
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_icalbuddy_path", lambda: "/fake/icalBuddy")

    icb_output = (
        "Standup matutino\n"
        "    today at 09:30 - 10:00\n"
        "Reunión con equipo\n"
        "    today at 14:00 - 15:00\n"
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout=icb_output),
    )
    events = cal_mod._fetch_calendar_today()
    assert len(events) == 2
    assert events[0]["title"] == "Standup matutino"
    assert events[0]["start"] == "09:30"
    assert events[0]["end"] == "10:00"
    assert events[1]["title"] == "Reunión con equipo"
    assert events[1]["start"] == "14:00"


def test_fetch_calendar_today_caps_to_max_events(monkeypatch):
    """Más events que `max_events` → truncación al final."""
    import rag
    import subprocess
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_icalbuddy_path", lambda: "/fake/icalBuddy")

    lines = []
    for i in range(15):
        hh = f"{9 + i:02d}"
        lines.append(f"Event {i}")
        lines.append(f"    today at {hh}:00 - {hh}:30")
    icb_output = "\n".join(lines) + "\n"

    monkeypatch.setattr(
        subprocess, "run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout=icb_output),
    )
    events = cal_mod._fetch_calendar_today(max_events=5)
    assert len(events) == 5


def test_fetch_calendar_today_handles_event_without_time(monkeypatch):
    """All-day events (sin time-range parseable) quedan con start/end
    vacíos y aparecen al final del sort (`""` fallback compara como
    `99:99`)."""
    import rag
    import subprocess
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_icalbuddy_path", lambda: "/fake/icalBuddy")
    icb_output = (
        "Cumpleaños Maria\n"
        "    14/04/2026\n"
        "Standup\n"
        "    today at 09:30 - 10:00\n"
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout=icb_output),
    )
    events = cal_mod._fetch_calendar_today()
    assert len(events) == 2
    assert events[0]["title"] == "Standup"
    assert events[1]["title"] == "Cumpleaños Maria"
    assert events[1]["start"] == ""
    assert events[1]["end"] == ""


# ── _fetch_calendar_ahead ───────────────────────────────────────────────


def test_fetch_calendar_ahead_returns_empty_on_negative_days(monkeypatch):
    """`days_ahead < 0` → [] sin invocar icalbuddy."""
    import rag
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(
        rag, "_icalbuddy_path",
        lambda: pytest.fail("no debió llamarse"),
    )
    assert cal_mod._fetch_calendar_ahead(days_ahead=-1) == []


def test_fetch_calendar_ahead_returns_empty_when_apple_disabled(monkeypatch):
    """Apple disabled → [] sin invocar nada."""
    import rag
    monkeypatch.setattr(rag, "_apple_enabled", lambda: False)
    assert cal_mod._fetch_calendar_ahead(days_ahead=7) == []


def test_fetch_calendar_ahead_returns_empty_when_icalbuddy_missing(monkeypatch):
    import rag
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_icalbuddy_path", lambda: None)
    assert cal_mod._fetch_calendar_ahead(days_ahead=3) == []


def test_fetch_calendar_ahead_uses_eventstoday_query_for_zero_days(monkeypatch):
    """`days_ahead == 0` usa el query `eventsToday`. Verificamos que el
    arg correcto se pasa a `subprocess.run`."""
    import rag
    import subprocess
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_icalbuddy_path", lambda: "/fake/icb")

    captured: list[list[str]] = []

    def _fake_run(args, **_kw):
        captured.append(list(args))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    cal_mod._fetch_calendar_ahead(days_ahead=0)
    assert captured, "subprocess.run debió invocarse"
    assert "eventsToday" in captured[0]
    assert not any(arg.startswith("eventsToday+") for arg in captured[0])


def test_fetch_calendar_ahead_uses_eventstoday_plus_n_query(monkeypatch):
    """`days_ahead > 0` usa `eventsToday+N`."""
    import rag
    import subprocess
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_icalbuddy_path", lambda: "/fake/icb")

    captured: list[list[str]] = []

    def _fake_run(args, **_kw):
        captured.append(list(args))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    cal_mod._fetch_calendar_ahead(days_ahead=7)
    assert "eventsToday+7" in captured[0]


def test_fetch_calendar_ahead_parses_date_label_and_time_range(monkeypatch):
    """Happy path: parsea los 3 fields de la shape `_fetch_calendar_ahead`:
    `title`, `date_label`, `time_range`. El time-range usa em-dash (–)."""
    import rag
    import subprocess
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_icalbuddy_path", lambda: "/fake/icb")

    icb_output = (
        "Standup\n"
        "    today at 09:30 - 10:00\n"
        "Cumpleaños Maria\n"
        "    Sat 14 Jun 2026\n"
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout=icb_output),
    )
    events = cal_mod._fetch_calendar_ahead(days_ahead=10)
    assert len(events) == 2
    assert events[0]["title"] == "Standup"
    assert events[0]["time_range"] == "09:30–10:00"
    assert events[0]["date_label"] == "today"
    assert events[1]["title"] == "Cumpleaños Maria"
    assert events[1]["time_range"] == ""
    assert events[1]["date_label"] == "Sat 14 Jun 2026"


def test_fetch_calendar_ahead_caps_to_max_events(monkeypatch):
    """Cap a `max_events` items."""
    import rag
    import subprocess
    monkeypatch.setattr(rag, "_apple_enabled", lambda: True)
    monkeypatch.setattr(rag, "_icalbuddy_path", lambda: "/fake/icb")

    lines = []
    for i in range(50):
        lines.append(f"Event {i}")
        lines.append("    today at 09:00 - 10:00")
    icb_output = "\n".join(lines) + "\n"
    monkeypatch.setattr(
        subprocess, "run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout=icb_output),
    )
    events = cal_mod._fetch_calendar_ahead(days_ahead=7, max_events=10)
    assert len(events) == 10
