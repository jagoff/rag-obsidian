"""Tests for scripts/ingest_calendar.py + `rag index --source calendar`.

No live Google Calendar OAuth — tests inject a mock service via `svc=`
and assert parsing / chunking / upsert semantics. Covers:
  - Event dict parsing (all-day, timed, cancelled, missing fields)
  - Datetime parsing (Z suffix, tz-aware, date-only)
  - Body formatting + char cap
  - upsert_events writes source=calendar + all expected meta fields
  - delete_cancelled removes rows by calendar + event_id
  - run() orchestration: bootstrap vs incremental, sync_token cycle
  - CLI `rag index --source calendar` routes correctly
"""
from __future__ import annotations


import pytest

import rag
from scripts import ingest_calendar as ic


# ── Fixtures ────────────────────────────────────────────────────────────

class _FakeCalendarService:
    """Minimal stand-in for googleapiclient Calendar v3 service. Supports
    .calendarList().list() and .events().list() per Google's fluent API
    style."""

    def __init__(self, calendars: list[dict], events_by_cal: dict[str, list[dict]],
                 bootstrap_sync: str = "sync-after-bootstrap",
                 incremental_responses: dict | None = None):
        self._calendars = calendars
        self._events = events_by_cal
        self._bootstrap_sync = bootstrap_sync
        self._incremental = incremental_responses or {}

    def calendarList(self):
        return self

    def events(self):
        return self

    def list(self, **kw):
        # Route by what kwargs were passed.
        if "calendarId" in kw:
            return _ExecProxy(self._events_payload(kw))
        return _ExecProxy({"items": self._calendars})

    def _events_payload(self, kw: dict) -> dict:
        cal_id = kw["calendarId"]
        st = kw.get("syncToken")
        if st:
            inc = self._incremental.get(st)
            if inc == "GONE":
                raise RuntimeError("410 Gone: syncToken expired")
            # Return the explicit incremental payload or an empty delta.
            return inc or {"items": [], "nextSyncToken": st}
        return {
            "items": self._events.get(cal_id, []),
            "nextSyncToken": self._bootstrap_sync,
        }


class _ExecProxy:
    def __init__(self, payload):
        self._p = payload

    def execute(self):
        return self._p


@pytest.fixture
def tmp_vault_col(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(rag, "VAULT_PATH", vault)
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    (tmp_path / "ragvec").mkdir()

    from rag import SqliteVecClient as _C
    client = _C(path=str(tmp_path / "ragvec"))
    col = client.get_or_create_collection(
        name="cal_test", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(rag, "get_db", lambda: col)

    def _fake_embed(texts):
        return [[float(len(t) % 10) / 10] + [0.0] * 7 for t in texts]
    monkeypatch.setattr(rag, "embed", _fake_embed)
    return col


# ── Helpers ────────────────────────────────────────────────────────────

def _mk_event(eid, summary, start_iso=None, start_date=None,
              end_iso=None, end_date=None,
              description="", location="", attendees=None, status="confirmed"):
    d: dict = {"id": eid, "summary": summary, "status": status}
    if start_iso:
        d["start"] = {"dateTime": start_iso}
    elif start_date:
        d["start"] = {"date": start_date}
    if end_iso:
        d["end"] = {"dateTime": end_iso}
    elif end_date:
        d["end"] = {"date": end_date}
    if description:
        d["description"] = description
    if location:
        d["location"] = location
    if attendees:
        d["attendees"] = [{"email": a} for a in attendees]
    return d


# ── Datetime parsing ───────────────────────────────────────────────────

def test_parse_cal_dt_all_day():
    ts, is_all_day = ic._parse_cal_dt({"date": "2026-04-20"})
    assert is_all_day is True
    assert ts > 0


def test_parse_cal_dt_timed_z():
    ts, is_all_day = ic._parse_cal_dt({"dateTime": "2026-04-20T10:00:00Z"})
    assert is_all_day is False
    assert ts > 0


def test_parse_cal_dt_timed_with_offset():
    ts, is_all_day = ic._parse_cal_dt({"dateTime": "2026-04-20T10:00:00-03:00"})
    assert is_all_day is False
    assert ts > 0


def test_parse_cal_dt_missing_returns_zero():
    assert ic._parse_cal_dt({}) == (0.0, False)
    assert ic._parse_cal_dt(None) == (0.0, False)


def test_parse_cal_dt_bad_date_returns_zero():
    ts, is_all_day = ic._parse_cal_dt({"date": "not-a-date"})
    assert ts == 0.0 and is_all_day is True


# ── Event parsing ──────────────────────────────────────────────────────

def test_parse_event_happy_path():
    raw = _mk_event("e1", "Standup",
                     start_iso="2026-04-20T10:00:00Z",
                     end_iso="2026-04-20T10:30:00Z",
                     description="Sync semanal",
                     location="Meet",
                     attendees=["a@x.com", "b@x.com"])
    ev = ic._parse_event(raw, "work@example.com", "Work")
    assert ev is not None
    assert ev.id == "e1"
    assert ev.title == "Standup"
    assert ev.location == "Meet"
    assert ev.attendees == ["a@x.com", "b@x.com"]
    assert ev.is_all_day is False
    assert ev.status == "confirmed"
    assert ev.calendar_name == "Work"


def test_parse_event_all_day():
    raw = _mk_event("e2", "Vacation",
                     start_date="2026-04-20", end_date="2026-04-27")
    ev = ic._parse_event(raw, "x", "x")
    assert ev and ev.is_all_day is True


def test_parse_event_defaults_for_missing_fields():
    raw = {"id": "e3", "status": "confirmed"}
    ev = ic._parse_event(raw, "x", "x")
    assert ev is not None
    assert ev.title == "(sin título)"
    assert ev.location == ""
    assert ev.attendees == []


def test_parse_event_drops_missing_id():
    assert ic._parse_event({"summary": "x"}, "x", "x") is None


# ── Body formatting ────────────────────────────────────────────────────

def test_format_event_body_includes_all_sections():
    ev = ic._parse_event(
        _mk_event("e1", "Demo",
                   start_iso="2026-04-20T10:00:00Z",
                   end_iso="2026-04-20T11:00:00Z",
                   description="Mostrar feature X",
                   location="Oficina",
                   attendees=["a@x.com"]),
        "cal1", "Cal",
    )
    body = ic._format_event_body(ev)
    assert "Título: Demo" in body
    assert "Cuándo:" in body
    assert "Dónde: Oficina" in body
    assert "Con: a@x.com" in body
    assert "Mostrar feature X" in body


def test_format_event_body_all_day_marker():
    ev = ic._parse_event(
        _mk_event("e2", "Holiday", start_date="2026-12-25"),
        "cal1", "Cal",
    )
    body = ic._format_event_body(ev)
    assert "todo el día" in body


def test_format_event_body_truncates_long_description():
    long_desc = "x" * 2000
    ev = ic._parse_event(
        _mk_event("e3", "Big", start_iso="2026-04-20T10:00:00Z",
                   description=long_desc),
        "cal1", "Cal",
    )
    body = ic._format_event_body(ev)
    assert len(body) <= ic.CHUNK_MAX_CHARS


def test_format_event_body_truncates_attendees_list():
    ev = ic._parse_event(
        _mk_event("e4", "Big", start_iso="2026-04-20T10:00:00Z",
                   attendees=[f"a{i}@x.com" for i in range(10)]),
        "cal1", "Cal",
    )
    body = ic._format_event_body(ev)
    # Should only show first 5 + "+N más" indicator.
    assert "(+5 más)" in body


# ── Writer ─────────────────────────────────────────────────────────────

def test_upsert_events_writes_source_calendar(tmp_vault_col):
    raw = _mk_event("e1", "Standup",
                     start_iso="2026-04-20T10:00:00Z",
                     description="Sync semanal",
                     attendees=["a@x.com"])
    ev = ic._parse_event(raw, "work@example.com", "Work")
    n = ic.upsert_events(tmp_vault_col, [ev])
    assert n == 1

    got = tmp_vault_col.get(where={"source": "calendar"}, include=["metadatas"])
    assert len(got["ids"]) == 1
    meta = got["metadatas"][0]
    assert meta["source"] == "calendar"
    assert meta["event_id"] == "e1"
    assert meta["calendar_id"] == "work@example.com"
    assert meta["calendar_name"] == "Work"
    assert meta["title"] == "Standup"
    assert meta["start_ts"] > 0
    assert meta["file"].startswith("calendar://work@example.com/")


def test_upsert_events_idempotent(tmp_vault_col):
    ev = ic._parse_event(_mk_event("e1", "x", start_iso="2026-04-20T10:00:00Z"),
                           "c1", "C")
    ic.upsert_events(tmp_vault_col, [ev])
    before = len(tmp_vault_col.get(where={"source": "calendar"}, include=[])["ids"])
    ic.upsert_events(tmp_vault_col, [ev])
    after = len(tmp_vault_col.get(where={"source": "calendar"}, include=[])["ids"])
    assert before == after


def test_delete_cancelled_removes_rows(tmp_vault_col):
    ev1 = ic._parse_event(_mk_event("e1", "x", start_iso="2026-04-20T10:00:00Z"),
                            "c1", "C")
    ev2 = ic._parse_event(_mk_event("e2", "y", start_iso="2026-04-21T10:00:00Z"),
                            "c1", "C")
    ic.upsert_events(tmp_vault_col, [ev1, ev2])
    assert len(tmp_vault_col.get(where={"source": "calendar"}, include=[])["ids"]) == 2

    deleted = ic.delete_cancelled(tmp_vault_col, "c1", ["e1"])
    assert deleted == 1
    remaining = tmp_vault_col.get(where={"source": "calendar"}, include=["metadatas"])
    assert len(remaining["ids"]) == 1
    assert remaining["metadatas"][0]["event_id"] == "e2"


# ── Orchestration ──────────────────────────────────────────────────────

def test_run_bootstrap_indexes_calendars_and_events(tmp_vault_col):
    svc = _FakeCalendarService(
        calendars=[{"id": "cal1@x", "summary": "Work"}],
        events_by_cal={
            "cal1@x": [
                _mk_event("e1", "Meeting", start_iso="2026-04-20T10:00:00Z"),
                _mk_event("e2", "1-on-1",  start_iso="2026-04-21T15:00:00Z"),
            ],
        },
    )
    summary = ic.run(svc=svc, vault_col=tmp_vault_col)
    assert "error" not in summary
    assert summary["calendars_scanned"] == 1
    assert summary["events_indexed"] == 2
    assert summary["bootstrapped"] == 1
    assert summary["incremental"] == 0


def test_run_incremental_after_bootstrap(tmp_vault_col):
    """Second run with the bootstrap sync_token → incremental path."""
    svc = _FakeCalendarService(
        calendars=[{"id": "cal1@x", "summary": "Work"}],
        events_by_cal={
            "cal1@x": [_mk_event("e1", "M", start_iso="2026-04-20T10:00:00Z")],
        },
        incremental_responses={
            "sync-after-bootstrap": {
                "items": [_mk_event("e2", "New", start_iso="2026-04-22T10:00:00Z")],
                "nextSyncToken": "sync-next",
            },
        },
    )
    ic.run(svc=svc, vault_col=tmp_vault_col)
    # Second call reuses the sync_token.
    summary = ic.run(svc=svc, vault_col=tmp_vault_col)
    assert summary["incremental"] == 1
    assert summary["bootstrapped"] == 0
    assert summary["events_indexed"] == 1  # just the new one


def test_run_handles_cancelled_events(tmp_vault_col):
    svc = _FakeCalendarService(
        calendars=[{"id": "cal1@x", "summary": "Work"}],
        events_by_cal={
            "cal1@x": [
                _mk_event("e1", "Kept", start_iso="2026-04-20T10:00:00Z"),
                _mk_event("e2", "Gone", start_iso="2026-04-21T10:00:00Z",
                           status="cancelled"),
            ],
        },
    )
    summary = ic.run(svc=svc, vault_col=tmp_vault_col)
    assert summary["events_indexed"] == 1
    assert summary["events_cancelled"] == 1


def test_run_dry_run_writes_nothing(tmp_vault_col):
    svc = _FakeCalendarService(
        calendars=[{"id": "cal1@x", "summary": "Work"}],
        events_by_cal={
            "cal1@x": [_mk_event("e1", "M", start_iso="2026-04-20T10:00:00Z")],
        },
    )
    summary = ic.run(svc=svc, vault_col=tmp_vault_col, dry_run=True)
    assert summary["events_indexed"] == 1  # counted but not written
    got = tmp_vault_col.get(where={"source": "calendar"}, include=[])
    assert got["ids"] == []


def test_run_no_service_reports_error(tmp_vault_col, monkeypatch):
    monkeypatch.setattr(ic, "_calendar_service", lambda: None)
    summary = ic.run(vault_col=tmp_vault_col)
    assert "error" in summary
    assert summary["events_indexed"] == 0


def test_run_single_calendar_filter(tmp_vault_col):
    svc = _FakeCalendarService(
        calendars=[{"id": "cal1@x", "summary": "Work"},
                   {"id": "cal2@x", "summary": "Personal"}],
        events_by_cal={
            "cal1@x": [_mk_event("e1", "W", start_iso="2026-04-20T10:00:00Z")],
            "cal2@x": [_mk_event("e2", "P", start_iso="2026-04-20T15:00:00Z")],
        },
    )
    summary = ic.run(svc=svc, vault_col=tmp_vault_col, calendar_id="cal1@x")
    assert summary["calendars_scanned"] == 1
    assert summary["events_indexed"] == 1


# ── CLI routing ────────────────────────────────────────────────────────

def test_cli_index_source_calendar_routes(monkeypatch):
    called = {}
    def _fake_run(**kw):
        called.update(kw)
        return {
            "calendars_scanned": 2, "events_indexed": 15,
            "events_cancelled": 0, "bootstrapped": 2, "incremental": 0,
            "duration_s": 0.1,
        }
    from scripts import ingest_calendar as ic_mod
    monkeypatch.setattr(ic_mod, "run", _fake_run)

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.index, ["--source", "calendar"])
    assert result.exit_code == 0, result.output
    assert "Calendar" in result.output
    assert "15 eventos" in result.output
    assert called["reset"] is False
    assert called["dry_run"] is False


def test_cli_index_source_calendar_dry_run(monkeypatch):
    called = {}
    def _fake_run(**kw):
        called.update(kw)
        return {
            "calendars_scanned": 1, "events_indexed": 3,
            "events_cancelled": 0, "bootstrapped": 1, "incremental": 0,
            "duration_s": 0.0,
        }
    from scripts import ingest_calendar as ic_mod
    monkeypatch.setattr(ic_mod, "run", _fake_run)

    from click.testing import CliRunner
    result = CliRunner().invoke(rag.index, ["--source", "calendar", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert called["dry_run"] is True
    assert "[dry-run]" in result.output


def test_cli_index_source_calendar_reports_error(monkeypatch):
    from scripts import ingest_calendar as ic_mod
    monkeypatch.setattr(
        ic_mod, "run",
        lambda **kw: {"error": "calendar service unavailable — configure ..."},
    )
    from click.testing import CliRunner
    result = CliRunner().invoke(rag.index, ["--source", "calendar"])
    assert "calendar service unavailable" in result.output
