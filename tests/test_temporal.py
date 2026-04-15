"""Temporal retrieval — detect intent, parse --since, filter on created_ts.

Clock is frozen via monkeypatch of `rag._now_dt` so range assertions are
stable. No Chroma in these tests; we only exercise the pure-Python helpers
and the where-clause builder.
"""
import pytest
from datetime import datetime

import rag


FROZEN = datetime(2026, 4, 14, 12, 0, 0)  # Tuesday, mid-day


@pytest.fixture(autouse=True)
def freeze_time(monkeypatch):
    monkeypatch.setattr(rag, "_now_dt", lambda: FROZEN)


# ── detect_temporal_intent ────────────────────────────────────────────────────


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def test_last_n_days():
    rng, cleaned = rag.detect_temporal_intent("últimos 10 días sobre música")
    assert rng is not None
    assert _fmt(rng[0]) == "2026-04-04"
    assert _fmt(rng[1]) == "2026-04-14"
    assert cleaned == "sobre música"


def test_hace_n_meses():
    rng, cleaned = rag.detect_temporal_intent("hace 3 meses pensaba en X")
    assert rng is not None
    assert _fmt(rng[0]) == "2026-01-14"
    assert cleaned == "pensaba en X"


def test_ultima_semana_implicit_n1():
    rng, cleaned = rag.detect_temporal_intent("qué escribí la última semana sobre RAG")
    assert rng is not None
    assert _fmt(rng[0]) == "2026-04-07"
    assert cleaned == "qué escribí sobre RAG"


def test_mes_pasado_strips_del_prefix():
    # Regression: "del mes pasado" previously left "notas d" in the residue.
    rng, cleaned = rag.detect_temporal_intent("notas del mes pasado")
    assert rng is not None
    assert cleaned == "notas"


def test_esta_semana_iso_boundary():
    # 2026-04-14 is a Tuesday → start of week should be Monday 2026-04-13.
    rng, _ = rag.detect_temporal_intent("esta semana")
    assert _fmt(rng[0]) == "2026-04-13"


def test_este_anio_starts_january_first():
    rng, _ = rag.detect_temporal_intent("este año")
    assert _fmt(rng[0]) == "2026-01-01"


def test_ayer_is_yesterday_at_midnight():
    rng, cleaned = rag.detect_temporal_intent("ayer puse algo sobre prompts")
    assert rng is not None
    assert _fmt(rng[0]) == "2026-04-13"
    assert cleaned == "puse algo sobre prompts"


def test_spanish_month_future_wraps_to_last_year():
    # Asking for "noviembre" in April → last year's November.
    rng, _ = rag.detect_temporal_intent("qué pensaba en noviembre")
    assert _fmt(rng[0]) == "2025-11-01"
    assert _fmt(rng[1]) == "2025-12-01"


def test_spanish_month_past_is_this_year():
    rng, cleaned = rag.detect_temporal_intent("mis ideas de enero")
    assert _fmt(rng[0]) == "2026-01-01"
    assert _fmt(rng[1]) == "2026-02-01"
    assert cleaned == "mis ideas"


def test_no_match_generic_query():
    rng, cleaned = rag.detect_temporal_intent("qué dice X sobre Y")
    assert rng is None
    assert cleaned == "qué dice X sobre Y"


def test_no_match_idiom():
    # Idiom "del año de la pera" should not trip the year pattern.
    # ("año pasado" / "este año" / "último año" / "hace N años" are the
    # triggers; "año de la pera" is not one of them.)
    rng, _ = rag.detect_temporal_intent("eso es del año de la pera")
    assert rng is None


def test_hoy_collapses_to_today_window():
    rng, _ = rag.detect_temporal_intent("hoy escribí sobre coaching")
    assert _fmt(rng[0]) == "2026-04-14"
    assert _fmt(rng[1]) == "2026-04-14"


# ── parse_since ───────────────────────────────────────────────────────────────


def test_parse_since_relative_days():
    ts = rag.parse_since("7d")
    assert _fmt(ts) == "2026-04-07"


def test_parse_since_relative_weeks():
    ts = rag.parse_since("2w")
    assert _fmt(ts) == "2026-03-31"


def test_parse_since_relative_months():
    ts = rag.parse_since("3m")
    assert _fmt(ts) == "2026-01-14"


def test_parse_since_relative_years():
    ts = rag.parse_since("1y")
    assert _fmt(ts) == "2025-04-14"


def test_parse_since_iso_date():
    ts = rag.parse_since("2026-01-01")
    assert _fmt(ts) == "2026-01-01"


def test_parse_since_iso_datetime():
    ts = rag.parse_since("2026-01-01T09:30:00")
    dt = datetime.fromtimestamp(ts)
    assert dt.hour == 9 and dt.minute == 30


def test_parse_since_invalid_raises():
    import click
    with pytest.raises(click.BadParameter):
        rag.parse_since("banana")


# ── build_where ───────────────────────────────────────────────────────────────


def test_build_where_date_range_only():
    w = rag.build_where(folder=None, tag=None, date_range=(1.0, 2.0))
    assert w == {
        "$and": [
            {"created_ts": {"$gte": 1.0}},
            {"created_ts": {"$lte": 2.0}},
        ]
    }


def test_build_where_date_range_plus_folder_combines():
    w = rag.build_where(folder="02-Areas", tag=None, date_range=(1.0, 2.0))
    assert "$and" in w
    # One clause is the folder $or, one is the date $and.
    kinds = [list(c.keys())[0] for c in w["$and"]]
    assert kinds.count("$or") == 1
    assert kinds.count("$and") == 1


def test_build_where_no_date_range_preserves_old_shape():
    # Regression: existing callers that pass only folder/tag must get the
    # same structure as before (single clause collapses out of $and).
    w = rag.build_where(folder=None, tag="ai", date_range=None)
    assert w == {"tags": {"$contains": "ai"}}


def test_build_where_all_three():
    w = rag.build_where(folder="02-Areas", tag="ai", date_range=(1.0, 2.0))
    assert "$and" in w
    assert len(w["$and"]) == 3


# ── bm25_search date_range filter ─────────────────────────────────────────────


def test_bm25_search_filters_by_date_range(monkeypatch):
    """bm25_search must drop chunks outside the date window and drop chunks
    missing `created_ts` (old schema)."""
    class FakeBM25:
        def get_scores(self, _tokens):
            return [1.0, 0.5, 0.2, 0.8]

    fake_corpus = {
        "bm25": FakeBM25(),
        "ids": ["id_new", "id_old_ts", "id_missing_ts", "id_mid"],
        "metas": [
            {"file": "a.md", "tags": "", "created_ts": 1000.0},   # in range
            {"file": "b.md", "tags": "", "created_ts": 10.0},     # too old
            {"file": "c.md", "tags": ""},                         # missing field
            {"file": "d.md", "tags": "", "created_ts": 500.0},    # in range
        ],
        "tags": set(),
        "folders": set(),
        "vocab": set(),
    }
    monkeypatch.setattr(rag, "_load_corpus", lambda _col: fake_corpus)

    got = rag.bm25_search(
        col=None, query="anything", k=10,
        folder=None, tag=None, date_range=(100.0, 2000.0),
    )
    assert "id_new" in got
    assert "id_mid" in got
    assert "id_old_ts" not in got
    assert "id_missing_ts" not in got


def test_bm25_search_no_date_range_unchanged(monkeypatch):
    """No date_range → pre-existing behavior (all chunks ranked)."""
    class FakeBM25:
        def get_scores(self, _tokens):
            return [1.0, 0.5]

    fake_corpus = {
        "bm25": FakeBM25(),
        "ids": ["id1", "id2"],
        "metas": [
            {"file": "a.md", "tags": ""},
            {"file": "b.md", "tags": ""},
        ],
        "tags": set(), "folders": set(), "vocab": set(),
    }
    monkeypatch.setattr(rag, "_load_corpus", lambda _col: fake_corpus)

    got = rag.bm25_search(
        col=None, query="anything", k=10,
        folder=None, tag=None, date_range=None,
    )
    assert set(got) == {"id1", "id2"}
