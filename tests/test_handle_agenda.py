"""Tests for `handle_agenda` — the calendar/reminders-filtered lookup
handler introduced 2026-04-21 evening to fix the bug where queries like
"qué tengo esta semana" fell through to `handle_recent` and listed
vault notes instead of calendar events.

Unit tests monkeypatch `_load_corpus` with a synthetic metas payload so
the handler logic can be exercised without a real sqlite-vec backend.
"""
from __future__ import annotations

import pytest

import rag


def _fake_meta(
    *,
    file: str,
    source: str,
    created_ts: float | str | None = None,
    modified: str | None = None,
    note: str = "",
    tags: str = "",
    folder: str = "",
) -> dict:
    """Shape-compatible meta dict matching what `_load_corpus` emits."""
    return {
        "file": file,
        "source": source,
        "created_ts": created_ts,
        "modified": modified,
        "note": note,
        "tags": tags,
        "folder": folder,
    }


@pytest.fixture
def mixed_corpus(monkeypatch):
    """Install a 7-row synthetic corpus spanning all 5 sources so the
    agenda filter has both keeps and drops to decide on."""
    metas = [
        _fake_meta(
            file="calendar://fernando@gmail.com/event-newest",
            source="calendar",
            created_ts=1_800_000_000.0,   # future-ish
            note="Reunión con Seba",
        ),
        _fake_meta(
            file="calendar://fernando@gmail.com/event-oldest",
            source="calendar",
            created_ts=1_700_000_000.0,   # 2023-ish
            note="Almuerzo viejo",
        ),
        _fake_meta(
            file="reminders://x-apple-reminder://AAA-111",
            source="reminders",
            created_ts=1_770_000_000.0,   # 2026-ish
            note="Comprar pan",
        ),
        _fake_meta(
            file="gmail://thread/12345",
            source="gmail",
            created_ts=1_780_000_000.0,
            note="Email de Juan",
        ),
        _fake_meta(
            file="whatsapp://120363@g.us/msg-99",
            source="whatsapp",
            created_ts=1_790_000_000.0,
            note="WA chat X",
        ),
        _fake_meta(
            file="02-Areas/Trabajo/proyecto-rag.md",
            source="vault",
            modified="2026-04-21T12:00",
            note="Proyecto RAG",
        ),
        # Legacy row with source=None (pre-cross-source data) — should be
        # treated as "vault" by normalize_source and stay out of agenda.
        _fake_meta(
            file="01-Projects/Old/legacy.md",
            source=None,
            modified="2026-04-20T10:00",
            note="Legacy",
        ),
    ]
    monkeypatch.setattr(rag, "_load_corpus", lambda col: {"metas": metas})
    return metas


def test_agenda_returns_only_calendar_and_reminders(mixed_corpus):
    files = rag.handle_agenda(col=None, params={})
    sources = [m["source"] for m in files]
    assert set(sources) == {"calendar", "reminders"}, (
        f"agenda should only surface calendar+reminders, got {set(sources)}"
    )
    # Three items total: 2 calendar + 1 reminders.
    assert len(files) == 3


def test_agenda_excludes_gmail_whatsapp_vault(mixed_corpus):
    files = rag.handle_agenda(col=None, params={})
    file_paths = [m["file"] for m in files]
    # None of the non-agenda-source paths surface.
    assert not any("gmail://" in p for p in file_paths)
    assert not any("whatsapp://" in p for p in file_paths)
    assert not any(p.endswith(".md") for p in file_paths)


def test_agenda_sorts_by_created_ts_desc(mixed_corpus):
    files = rag.handle_agenda(col=None, params={})
    # event-newest (1_800_000_000) > reminders-AAA (1_770_000_000) > event-oldest (1_700_000_000)
    assert files[0]["file"] == "calendar://fernando@gmail.com/event-newest"
    assert files[1]["file"] == "reminders://x-apple-reminder://AAA-111"
    assert files[2]["file"] == "calendar://fernando@gmail.com/event-oldest"


def test_agenda_limit_param(mixed_corpus):
    files = rag.handle_agenda(col=None, params={}, limit=2)
    assert len(files) == 2
    # Sort stays desc — the 2 newest survive.
    assert files[0]["file"] == "calendar://fernando@gmail.com/event-newest"
    assert files[1]["file"] == "reminders://x-apple-reminder://AAA-111"


def test_agenda_handles_missing_created_ts(monkeypatch):
    """Metas without `created_ts` drift to the bottom (ts=0.0) but still
    surface — they aren't silently dropped. Belt-and-suspenders for
    ingesters that might omit the field."""
    metas = [
        _fake_meta(
            file="calendar://a/1",
            source="calendar",
            created_ts=1_800_000_000.0,
            note="Has ts",
        ),
        _fake_meta(
            file="calendar://a/2",
            source="calendar",
            created_ts=None,  # missing
            note="No ts",
        ),
    ]
    monkeypatch.setattr(rag, "_load_corpus", lambda col: {"metas": metas})
    files = rag.handle_agenda(col=None, params={})
    assert len(files) == 2
    assert files[0]["file"] == "calendar://a/1"
    # The None-ts row drifts to last, but is not dropped.
    assert files[1]["file"] == "calendar://a/2"


def test_agenda_coerces_string_created_ts(monkeypatch):
    """Some sources (reminders via AppleScript pipe) might serialize
    created_ts as a string. The handler should coerce gracefully."""
    metas = [
        _fake_meta(file="calendar://a/1", source="calendar",
                   created_ts="1800000000.0", note="string ts"),
        _fake_meta(file="calendar://a/2", source="calendar",
                   created_ts=1_700_000_000.0, note="float ts"),
    ]
    monkeypatch.setattr(rag, "_load_corpus", lambda col: {"metas": metas})
    files = rag.handle_agenda(col=None, params={})
    # string coerces to the larger ts → wins sort.
    assert files[0]["file"] == "calendar://a/1"


def test_agenda_malformed_ts_drifts_to_bottom_not_raised(monkeypatch):
    """Malformed ts (non-numeric string) should fall back to 0.0 via the
    try/except in the sort key, not raise."""
    metas = [
        _fake_meta(file="calendar://a/1", source="calendar",
                   created_ts=1_800_000_000.0, note="ok"),
        _fake_meta(file="calendar://a/2", source="calendar",
                   created_ts="garbage", note="bad"),
    ]
    monkeypatch.setattr(rag, "_load_corpus", lambda col: {"metas": metas})
    # Should not raise.
    files = rag.handle_agenda(col=None, params={})
    assert len(files) == 2
    # Bad ts = 0.0, gets outranked by the valid one.
    assert files[0]["file"] == "calendar://a/1"


def test_agenda_dedup_by_file_key(monkeypatch):
    """`_filter_files` dedupes by `file` key — agenda inherits that so
    multiple chunks of the same calendar event collapse to one entry."""
    metas = [
        _fake_meta(file="calendar://a/1", source="calendar",
                   created_ts=1_800_000_000.0, note="chunk 0"),
        _fake_meta(file="calendar://a/1", source="calendar",
                   created_ts=1_800_000_000.0, note="chunk 1"),
        _fake_meta(file="calendar://a/2", source="calendar",
                   created_ts=1_700_000_000.0, note="another event"),
    ]
    monkeypatch.setattr(rag, "_load_corpus", lambda col: {"metas": metas})
    files = rag.handle_agenda(col=None, params={})
    assert len(files) == 2
    assert {m["file"] for m in files} == {
        "calendar://a/1", "calendar://a/2",
    }


def test_agenda_sources_constant_is_locked():
    """Sanity guard: if someone adds WA/gmail to `_AGENDA_SOURCES` the
    semantics of this intent change materially. Force a review via
    test rather than silent drift."""
    assert rag._AGENDA_SOURCES == frozenset({"calendar", "reminders"})


# ── Window filter (agenda v2, 2026-04-21 evening) ─────────────────────

from datetime import datetime, timedelta


def _windowed_corpus(monkeypatch, anchor: datetime):
    """Install a 5-row corpus with calendar events at spread-out
    timestamps so window filters have something to accept/reject."""
    metas = [
        _fake_meta(
            file="calendar://a/today",
            source="calendar",
            created_ts=anchor.replace(hour=10).timestamp(),
            note="hoy 10hs",
        ),
        _fake_meta(
            file="calendar://a/tomorrow",
            source="calendar",
            created_ts=(anchor + timedelta(days=1)).replace(hour=14).timestamp(),
            note="mañana 14hs",
        ),
        _fake_meta(
            file="calendar://a/in-3-days",
            source="calendar",
            created_ts=(anchor + timedelta(days=3)).replace(hour=9).timestamp(),
            note="+3 días",
        ),
        _fake_meta(
            file="calendar://a/next-month",
            source="calendar",
            created_ts=(anchor + timedelta(days=35)).replace(hour=9).timestamp(),
            note="mes que viene",
        ),
        _fake_meta(
            file="reminders://x-apple-reminder://TODAY",
            source="reminders",
            created_ts=anchor.replace(hour=18).timestamp(),
            note="reminder hoy",
        ),
    ]
    monkeypatch.setattr(rag, "_load_corpus", lambda col: {"metas": metas})
    return metas


def test_handle_agenda_window_hoy_filters_to_today(monkeypatch):
    """With question='qué tengo hoy' and 5 events spread 0-35 days, only
    the 2 same-day events survive the window filter."""
    anchor = datetime(2026, 4, 22, 10, 30)  # Wed
    _windowed_corpus(monkeypatch, anchor)
    # Patch datetime.now() in the window parser's scope so the anchor
    # is deterministic. The parser already accepts an explicit `now`,
    # but the handler doesn't forward it (reasonable since production
    # calls don't pin time). We rely on monkeypatching datetime.now
    # instead. Simpler: call handle_agenda with an explicit anchor via
    # the module's datetime patch.
    monkeypatch.setattr(
        rag, "datetime",
        type("D", (), {"now": staticmethod(lambda: anchor)}),
    )
    files = rag.handle_agenda(col=None, params={}, question="qué tengo hoy")
    # Exactly 2 items today: event @10hs + reminder @18hs.
    assert len(files) == 2
    paths = {m["file"] for m in files}
    assert paths == {
        "calendar://a/today",
        "reminders://x-apple-reminder://TODAY",
    }


def test_handle_agenda_window_esta_semana(monkeypatch):
    """Anchor Wed → ISO week Mon-Sun: days 0 (today), +1 (tomorrow),
    +3 (Saturday) are all within-week; +35 is out."""
    anchor = datetime(2026, 4, 22, 10, 30)  # Wed, ISO week 17 (Mon 20 - Sun 26)
    _windowed_corpus(monkeypatch, anchor)
    monkeypatch.setattr(
        rag, "datetime",
        type("D", (), {"now": staticmethod(lambda: anchor)}),
    )
    files = rag.handle_agenda(col=None, params={}, question="qué tengo esta semana")
    paths = {m["file"] for m in files}
    # 4 items in-week: hoy event, hoy reminder, mañana, +3 days (Sat).
    # "next-month" is out.
    assert "calendar://a/next-month" not in paths
    assert paths == {
        "calendar://a/today",
        "calendar://a/tomorrow",
        "calendar://a/in-3-days",
        "reminders://x-apple-reminder://TODAY",
    }


def test_handle_agenda_window_no_question_falls_back(monkeypatch):
    """question=None → no window filter; all 5 items pass (source filter
    still applies, but 5 are all calendar/reminders here)."""
    anchor = datetime(2026, 4, 22, 10, 30)
    _windowed_corpus(monkeypatch, anchor)
    # No monkeypatch on datetime.now needed — with question=None the
    # parser is never called.
    files = rag.handle_agenda(col=None, params={}, question=None)
    assert len(files) == 5


def test_handle_agenda_window_unparseable_question_falls_back(monkeypatch):
    """question present but no temporal anchor → parser returns None,
    handler falls back to top-N by ts desc (all pass)."""
    anchor = datetime(2026, 4, 22, 10, 30)
    _windowed_corpus(monkeypatch, anchor)
    # "mi agenda" is an agenda-intent trigger but has no temporal cue —
    # so window filter is skipped and every source-matching item passes.
    files = rag.handle_agenda(col=None, params={}, question="mi agenda")
    assert len(files) == 5
