"""Tests for `rag insights` primitives: normalization, log loading, gap/hot
detection, orphan scanning. Pure functions — no LLM, no sqlite-vec."""

import json
from datetime import datetime, timedelta

import pytest

import rag


# ── _normalize_query_for_grouping ─────────────────────────────────────────────


def test_normalize_lowercases():
    assert rag._normalize_query_for_grouping("HOLA Mundo") == "hola mundo"


def test_normalize_strips_accents():
    assert rag._normalize_query_for_grouping("qué es RAG") == "que es rag"


def test_normalize_strips_punctuation():
    assert rag._normalize_query_for_grouping("¿Qué es RAG?") == "que es rag"


def test_normalize_collapses_whitespace():
    assert rag._normalize_query_for_grouping("a   b\t\tc\nd") == "a b c d"


def test_normalize_empty_returns_empty():
    assert rag._normalize_query_for_grouping("") == ""
    assert rag._normalize_query_for_grouping("   ") == ""
    # None-safe via the `q or ""` guard
    assert rag._normalize_query_for_grouping(None) == ""  # type: ignore[arg-type]


def test_normalize_equivalence_across_punctuation_and_accents():
    a = rag._normalize_query_for_grouping("¿Qué es RAG?")
    b = rag._normalize_query_for_grouping("que es rag")
    c = rag._normalize_query_for_grouping("  QUE   es Rag!! ")
    assert a == b == c


# ── _load_query_entries ───────────────────────────────────────────────────────


@pytest.fixture
def tmp_log(tmp_path):
    p = tmp_path / "queries.jsonl"
    return p


def _write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_filters_by_since(tmp_log):
    now = datetime(2026, 4, 15, 12, 0, 0)
    old = now - timedelta(days=10)
    recent = now - timedelta(hours=1)
    _write_lines(tmp_log, [
        json.dumps({"ts": old.isoformat(), "q": "viejo", "cmd": "query"}),
        json.dumps({"ts": recent.isoformat(), "q": "nuevo", "cmd": "query"}),
    ])
    out = rag._load_query_entries(now - timedelta(days=1), log_path=tmp_log)
    assert [e["q"] for e in out] == ["nuevo"]


def test_load_skips_malformed_json(tmp_log):
    now = datetime(2026, 4, 15, 12, 0, 0)
    _write_lines(tmp_log, [
        "{not valid json",
        json.dumps({"ts": now.isoformat(), "q": "ok", "cmd": "query"}),
        "",
    ])
    out = rag._load_query_entries(now - timedelta(days=1), log_path=tmp_log)
    assert [e["q"] for e in out] == ["ok"]


def test_load_skips_entries_with_unparseable_ts(tmp_log):
    now = datetime(2026, 4, 15, 12, 0, 0)
    _write_lines(tmp_log, [
        json.dumps({"ts": "not-a-date", "q": "bad_ts"}),
        json.dumps({"q": "missing_ts"}),
        json.dumps({"ts": now.isoformat(), "q": "ok"}),
    ])
    out = rag._load_query_entries(now - timedelta(days=1), log_path=tmp_log)
    assert [e["q"] for e in out] == ["ok"]


def test_load_empty_file(tmp_log):
    tmp_log.write_text("", encoding="utf-8")
    assert rag._load_query_entries(datetime(2026, 1, 1), log_path=tmp_log) == []


def test_load_nonexistent_path(tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"
    assert rag._load_query_entries(datetime(2026, 1, 1), log_path=missing) == []


# ── detect_gap_queries ────────────────────────────────────────────────────────


def _gap_entry(q, score, ts="2026-04-15T10:00:00", cmd="query"):
    return {"cmd": cmd, "q": q, "top_score": score, "ts": ts}


def test_gap_ignores_non_query_cmds():
    entries = [
        _gap_entry("qué es rag", 0.005, cmd="read"),
        _gap_entry("qué es rag", 0.005, cmd="dead"),
        _gap_entry("qué es rag", 0.005, cmd="query"),
    ]
    # Only 1 "query" → below min_occurrences=2 default → empty.
    assert rag.detect_gap_queries(entries) == []


def test_gap_only_low_confidence_below_threshold():
    # threshold = INSIGHTS_GAP_THRESHOLD (0.015). top > threshold is excluded.
    entries = [
        _gap_entry("alta", 0.5),
        _gap_entry("alta", 0.5),
        _gap_entry("baja", 0.01),
        _gap_entry("baja", 0.005),
    ]
    out = rag.detect_gap_queries(entries)
    assert [g["query"] for g in out] == ["baja"]


def test_gap_boundary_equal_to_threshold_included():
    threshold = rag.INSIGHTS_GAP_THRESHOLD
    entries = [
        _gap_entry("borde", threshold),
        _gap_entry("borde", threshold),
    ]
    out = rag.detect_gap_queries(entries, threshold=threshold, min_occurrences=2)
    assert len(out) == 1 and out[0]["query"] == "borde"


def test_gap_respects_min_occurrences():
    entries = [
        _gap_entry("una", 0.01),
        _gap_entry("dos", 0.01),
        _gap_entry("dos", 0.01),
    ]
    out = rag.detect_gap_queries(entries, min_occurrences=2)
    assert [g["query"] for g in out] == ["dos"]
    assert out[0]["count"] == 2


def test_gap_groups_by_normalized_query_and_keeps_original_label():
    entries = [
        _gap_entry("¿Qué es RAG?", 0.005),
        _gap_entry("que es rag", 0.01),
        _gap_entry("QUE ES RAG", 0.008),
    ]
    out = rag.detect_gap_queries(entries, min_occurrences=2)
    assert len(out) == 1
    g = out[0]
    assert g["count"] == 3
    # The original query string of the FIRST entry that seeded the group is kept.
    assert g["query"] == "¿Qué es RAG?"
    assert g["max_score"] == pytest.approx(0.01)


def test_gap_sorts_by_count_desc_then_max_score_desc():
    entries = [
        # Two groups with count=2; "bbb" has higher max_score → should come first.
        _gap_entry("aaa", 0.002),
        _gap_entry("aaa", 0.003),
        _gap_entry("bbb", 0.005),
        _gap_entry("bbb", 0.014),
        # "ccc" has count=3 → first overall.
        _gap_entry("ccc", 0.001),
        _gap_entry("ccc", 0.001),
        _gap_entry("ccc", 0.001),
    ]
    out = rag.detect_gap_queries(entries, min_occurrences=2)
    assert [g["query"] for g in out] == ["ccc", "bbb", "aaa"]


def test_gap_last_ts_tracks_maximum():
    entries = [
        _gap_entry("q", 0.005, ts="2026-04-10T09:00:00"),
        _gap_entry("q", 0.005, ts="2026-04-15T14:30:00"),
        _gap_entry("q", 0.005, ts="2026-04-12T11:00:00"),
    ]
    out = rag.detect_gap_queries(entries, min_occurrences=2)
    assert out[0]["last_ts"] == "2026-04-15T14:30:00"


def test_gap_skips_non_numeric_top_score():
    entries = [
        _gap_entry("q", None),
        _gap_entry("q", "low"),
        _gap_entry("q", 0.005),
        _gap_entry("q", 0.003),
    ]
    out = rag.detect_gap_queries(entries, min_occurrences=2)
    assert len(out) == 1 and out[0]["count"] == 2


# ── detect_hot_queries ────────────────────────────────────────────────────────


def _hot_entry(q, score, paths=None, ts="2026-04-15T10:00:00", cmd="query"):
    return {"cmd": cmd, "q": q, "top_score": score, "ts": ts, "paths": paths or []}


def test_hot_ignores_non_query_cmds():
    entries = [_hot_entry("q", 0.5, cmd="read")] * 5
    assert rag.detect_hot_queries(entries) == []


def test_hot_only_high_confidence_above_threshold():
    entries = [
        _hot_entry("alta", 0.5),
        _hot_entry("alta", 0.5),
        _hot_entry("alta", 0.5),
        _hot_entry("baja", 0.01),
        _hot_entry("baja", 0.005),
        _hot_entry("baja", 0.01),
    ]
    out = rag.detect_hot_queries(entries, min_occurrences=3)
    assert [g["query"] for g in out] == ["alta"]


def test_hot_respects_min_occurrences():
    entries = [
        _hot_entry("q1", 0.5),
        _hot_entry("q1", 0.5),  # only 2 — below default min_hot=3
        _hot_entry("q2", 0.5),
        _hot_entry("q2", 0.5),
        _hot_entry("q2", 0.5),
    ]
    out = rag.detect_hot_queries(entries)
    assert [g["query"] for g in out] == ["q2"]


def test_hot_avg_score_rounded_to_3dp():
    entries = [
        _hot_entry("q", 0.1),
        _hot_entry("q", 0.2),
        _hot_entry("q", 0.3),
    ]
    out = rag.detect_hot_queries(entries, min_occurrences=3)
    assert out[0]["avg_score"] == pytest.approx(0.2)
    assert out[0]["count"] == 3


def test_hot_top_paths_aggregate_across_entries_and_cap_at_3():
    entries = [
        _hot_entry("q", 0.5, paths=["a.md", "b.md"]),
        _hot_entry("q", 0.5, paths=["a.md", "c.md"]),
        _hot_entry("q", 0.5, paths=["a.md", "b.md", "d.md"]),
    ]
    out = rag.detect_hot_queries(entries, min_occurrences=3)
    assert out[0]["top_paths"][0] == "a.md"  # most frequent
    assert len(out[0]["top_paths"]) == 3


def test_hot_top_paths_per_entry_capped_at_first_3():
    # Each entry contributes only its first 3 paths. "z.md" at pos 4 → never counted.
    entries = [
        _hot_entry("q", 0.5, paths=["a.md", "b.md", "c.md", "z.md"]),
        _hot_entry("q", 0.5, paths=["a.md", "b.md", "c.md", "z.md"]),
        _hot_entry("q", 0.5, paths=["a.md", "b.md", "c.md", "z.md"]),
    ]
    out = rag.detect_hot_queries(entries, min_occurrences=3)
    assert "z.md" not in out[0]["top_paths"]


def test_hot_sorted_by_count_desc():
    entries = (
        [_hot_entry("bajo", 0.5)] * 3
        + [_hot_entry("medio", 0.5)] * 5
        + [_hot_entry("alto", 0.5)] * 7
    )
    out = rag.detect_hot_queries(entries, min_occurrences=3)
    assert [g["query"] for g in out] == ["alto", "medio", "bajo"]


# ── detect_orphan_notes ───────────────────────────────────────────────────────


def _make_md(vault, rel, body=""):
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body or "#\n", encoding="utf-8")
    return p


def test_orphan_empty_when_vault_missing(tmp_path):
    missing = tmp_path / "nope"
    assert rag.detect_orphan_notes([], missing) == []


def test_orphan_returns_notes_never_retrieved(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _make_md(vault, "02-Areas/seen.md")
    _make_md(vault, "02-Areas/unseen.md")
    entries = [{"paths": ["02-Areas/seen.md"]}]
    out = rag.detect_orphan_notes(entries, vault)
    assert out == ["02-Areas/unseen.md"]


def test_orphan_excludes_default_prefixes(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _make_md(vault, "00-Inbox/new.md")
    _make_md(vault, "04-Archive/99-obsidian-system/99-AI/reviews/2026-04-15.md")
    _make_md(vault, "04-Archive/old.md")
    _make_md(vault, "02-Areas/real.md")
    out = rag.detect_orphan_notes([], vault)
    assert out == ["02-Areas/real.md"]


def test_orphan_respects_is_excluded_dotfolders(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _make_md(vault, ".trash/gone.md")
    _make_md(vault, ".obsidian/templates/t.md")
    _make_md(vault, "02-Areas/keep.md")
    out = rag.detect_orphan_notes([], vault)
    assert out == ["02-Areas/keep.md"]


def test_orphan_sorted_alphabetically(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _make_md(vault, "03-Resources/zeta.md")
    _make_md(vault, "01-Projects/alpha.md")
    _make_md(vault, "02-Areas/mu.md")
    out = rag.detect_orphan_notes([], vault)
    assert out == [
        "01-Projects/alpha.md",
        "02-Areas/mu.md",
        "03-Resources/zeta.md",
    ]


def test_orphan_custom_excluded_prefixes(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _make_md(vault, "00-Inbox/new.md")
    _make_md(vault, "02-Areas/real.md")
    # Override: exclude only 02-Areas/
    out = rag.detect_orphan_notes([], vault, excluded_prefixes=("02-Areas/",))
    assert out == ["00-Inbox/new.md"]


def test_orphan_ignores_non_string_paths_in_entries(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _make_md(vault, "02-Areas/a.md")
    _make_md(vault, "02-Areas/b.md")
    entries = [{"paths": [None, 42, "02-Areas/a.md", {"not": "a path"}]}]
    out = rag.detect_orphan_notes(entries, vault)
    assert out == ["02-Areas/b.md"]
