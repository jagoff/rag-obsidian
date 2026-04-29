"""Tests for scripts/export_training_pairs.py (2026-04-22).

Miner that extracts (query, positive, hard_negs) pairs from all telemetry
sources — rag_feedback AND rag_behavior — complementing
scripts/finetune_reranker.py which today only uses rag_feedback.

The tests seed a tmp DB with synthetic telemetry, then call
`export_pairs()` and assert on:
  - classification of positives by source (corrective > rating > behavior_*)
  - history-based hard-neg mining (uses impressions, not re-retrieve)
  - cross-source path filtering (calendar://, whatsapp://)
  - dedup of (query, path, event) triples
  - cutoff window respect

End-to-end the script runs as CLI — smoke test below invokes `main()`
with --stats-only to verify the plumbing.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rag  # noqa: E402
import export_training_pairs as etp  # noqa: E402


# ── Fixture: isolated DB ────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DB_PATH", tmp_path)
    with rag._ragvec_state_conn() as conn:
        rag._ensure_telemetry_tables(conn)
    return tmp_path


def _seed_feedback(ev: dict) -> None:
    ev = {"ts": datetime.now().isoformat(timespec="seconds"), **ev}
    with rag._ragvec_state_conn() as conn:
        rag._sql_append_event(conn, "rag_feedback", rag._map_feedback_row(ev))


def _seed_behavior(ev: dict) -> None:
    ev = {"ts": datetime.now().isoformat(timespec="seconds"), **ev}
    with rag._ragvec_state_conn() as conn:
        rag._sql_append_event(conn, "rag_behavior", rag._map_behavior_row(ev))


def _seed_query(ev: dict) -> None:
    ev = {"ts": datetime.now().isoformat(timespec="seconds"), **ev}
    with rag._ragvec_state_conn() as conn:
        rag._sql_append_event(conn, "rag_queries", rag._map_queries_row(ev))


# ── Behavior source filter (cerrado 2026-04-28) ────────────────────────────


def test_extract_behavior_rows_filters_bot_initiated_sources(tmp_db):
    """rag_behavior rows con source bot-initiated (anticipate-calendar,
    anticipate-echo, followup, eval) NO entran al training set. Solo
    `cli`, `web`, `whatsapp` cuentan como user signal.

    Cerrado 2026-04-28: pre-fix `retrieve()` siempre logueaba con
    source="cli" hardcoded → bot-initiated impressions contaminaban el
    training. El caller-aware logging ya distinguía; este test verifica
    que el miner respeta el split."""
    now = datetime.now()
    iso = lambda d: d.isoformat(timespec="seconds")  # noqa: E731

    # User signal — debe pasar.
    _seed_behavior({
        "ts": iso(now), "source": "cli", "event": "impression",
        "query": "user query 1", "path": "user-a.md", "rank": 1,
    })
    _seed_behavior({
        "ts": iso(now - timedelta(minutes=1)), "source": "web",
        "event": "open", "query": "user query 2", "path": "user-b.md",
    })
    _seed_behavior({
        "ts": iso(now - timedelta(minutes=2)), "source": "whatsapp",
        "event": "kept", "query": "user query 3", "path": "user-c.md",
    })

    # Bot signal — debe filtrarse.
    _seed_behavior({
        "ts": iso(now - timedelta(minutes=3)), "source": "anticipate-calendar",
        "event": "impression", "query": "calendar event title",
        "path": "bot-a.md", "rank": 1,
    })
    _seed_behavior({
        "ts": iso(now - timedelta(minutes=4)), "source": "anticipate-echo",
        "event": "impression", "query": "echo snippet",
        "path": "bot-b.md", "rank": 1,
    })
    _seed_behavior({
        "ts": iso(now - timedelta(minutes=5)), "source": "followup",
        "event": "impression", "query": "follow-up loop",
        "path": "bot-c.md", "rank": 1,
    })
    _seed_behavior({
        "ts": iso(now - timedelta(minutes=6)), "source": "eval",
        "event": "impression", "query": "eval set query",
        "path": "bot-d.md", "rank": 1,
    })

    cutoff_iso = iso(now - timedelta(hours=1))
    rows = etp._extract_behavior_rows(cutoff_iso)
    paths = sorted(r["path"] for r in rows)
    sources = sorted(set(r["source"] for r in rows))

    assert paths == ["user-a.md", "user-b.md", "user-c.md"], (
        f"esperado solo user paths, got: {paths}"
    )
    assert sources == ["cli", "web", "whatsapp"], (
        f"sources should only contain user-driven channels, got: {sources}"
    )


def test_extract_behavior_rows_includes_only_whitelist():
    """Sanity: el frozenset de allowed sources es el contract. Si alguien
    agrega una source que no debería contar como user signal, este test
    rompe y obliga a actualizar el whitelist conscientemente."""
    assert etp._USER_BEHAVIOR_SOURCES == frozenset({"cli", "web", "whatsapp"})


# ── Positive sources: priority ordering ─────────────────────────────────────


def test_corrective_path_emits_pair_with_single_positive(tmp_db):
    """A feedback row with corrective_path = X must produce exactly one pair
    (query, X, [other retrieved paths]) with source='corrective'."""
    _seed_query({
        "cmd": "web", "turn_id": "t1", "session": "web:s1",
        "q": "¿qué tengo sobre postura?",
        "paths": ["A.md", "B.md", "C.md", "D.md"],
        "top_score": 0.65,
    })
    _seed_feedback({
        "rating": -1, "turn_id": "t1",
        "q": "¿qué tengo sobre postura?",
        "paths": ["A.md", "B.md", "C.md", "D.md"],
        "corrective_path": "C.md",
    })

    pairs, stats = etp.export_pairs(days=7, min_negatives=1)
    assert len(pairs) == 1
    p = pairs[0]
    assert p["positive"] == "C.md"
    assert p["source"] == "corrective"
    # Corrective excluded from negs; retrieved paths preserved.
    assert "C.md" not in p["negatives"]
    assert set(p["negatives"]) == {"A.md", "B.md", "D.md"}
    assert p["turn_id"] == "t1"


def test_rating_positive_emits_one_pair_per_retrieved_path(tmp_db):
    """A +1 feedback without corrective → one pair per path, because we don't
    know which path the user liked specifically (noisy positive fallback).
    Each pair mines its own hard negs."""
    _seed_query({
        "cmd": "web", "turn_id": "t2", "session": "web:s1",
        "q": "tengo algo de MOZE",
        "paths": ["moze.md", "other.md"],
    })
    # For `rating_pos`, the miner pulls hard negs from behavior impressions,
    # not from the feedback's own paths — so seed impressions for the query.
    _seed_behavior({"source": "cli", "event": "impression",
                    "query": "tengo algo de MOZE", "path": "moze.md", "rank": 1})
    _seed_behavior({"source": "cli", "event": "impression",
                    "query": "tengo algo de MOZE", "path": "decoy1.md", "rank": 2})
    _seed_behavior({"source": "cli", "event": "impression",
                    "query": "tengo algo de MOZE", "path": "decoy2.md", "rank": 3})
    _seed_feedback({
        "rating": 1, "turn_id": "t2",
        "q": "tengo algo de MOZE",
        "paths": ["moze.md", "other.md"],
    })
    # Also seed an impression so hard-neg mining can work.

    pairs, stats = etp.export_pairs(days=7, min_negatives=1)
    # One pair per path in the turn → 2 pairs from rating_pos.
    rating_pairs = [p for p in pairs if p["source"] == "rating_pos"]
    assert len(rating_pairs) == 2
    positives = {p["positive"] for p in rating_pairs}
    assert positives == {"moze.md", "other.md"}
    # Each has hard negs mined from the impression history.
    for p in rating_pairs:
        # Mined hard negs include decoys that weren't clicked/copied.
        assert any(n.startswith("decoy") for n in p["negatives"])


def test_behavior_copy_emits_pair_with_query(tmp_db):
    """A copy event on a path, with query, becomes a training pair with
    source='behavior_copy'."""
    _seed_behavior({
        "source": "web", "event": "copy",
        "query": "cómo uso rag tune",
        "path": "docs/rag-tune.md", "rank": 1,
    })
    _seed_behavior({
        "source": "web", "event": "impression",
        "query": "cómo uso rag tune",
        "path": "docs/rag-tune.md", "rank": 1,
    })
    _seed_behavior({
        "source": "web", "event": "impression",
        "query": "cómo uso rag tune",
        "path": "docs/other-unrelated.md", "rank": 2,
    })

    pairs, stats = etp.export_pairs(days=7, min_negatives=1)
    copy_pairs = [p for p in pairs if p["source"] == "behavior_copy"]
    assert len(copy_pairs) == 1
    p = copy_pairs[0]
    assert p["positive"] == "docs/rag-tune.md"
    assert "docs/other-unrelated.md" in p["negatives"]


def test_behavior_open_and_save_are_positives(tmp_db):
    """open / save / kept all count as positives — synced with
    rag._BEHAVIOR_POSITIVE."""
    for event in ("open", "save", "kept"):
        _seed_behavior({
            "source": "web", "event": event,
            "query": f"q-{event}", "path": f"{event}.md", "rank": 1,
        })
        _seed_behavior({
            "source": "web", "event": "impression",
            "query": f"q-{event}", "path": f"{event}.md", "rank": 1,
        })
        _seed_behavior({
            "source": "web", "event": "impression",
            "query": f"q-{event}", "path": f"{event}-decoy.md", "rank": 2,
        })

    pairs, stats = etp.export_pairs(days=7, min_negatives=1)
    sources = {p["source"] for p in pairs}
    assert "behavior_open" in sources
    assert "behavior_save" in sources
    assert "behavior_kept" in sources


# ── Hard-neg mining: history-based ──────────────────────────────────────────


def test_hard_negs_mined_from_impression_history(tmp_db):
    """For a positive pair, hard negs come from paths SURFACED (impression)
    for the same query but NOT interacted with."""
    q = "qué tengo sobre Mayra"
    # Impressions: top-4 paths. Only rank-1 gets clicked; ranks 2-4 are
    # hard negs.
    for i, path in enumerate(["win.md", "neg1.md", "neg2.md", "neg3.md"], start=1):
        _seed_behavior({
            "source": "web", "event": "impression",
            "query": q, "path": path, "rank": i,
        })
    _seed_behavior({
        "source": "web", "event": "copy",
        "query": q, "path": "win.md", "rank": 1,
    })

    pairs, stats = etp.export_pairs(days=7, min_negatives=1, max_hard_negs=3)
    copy_pairs = [p for p in pairs if p["source"] == "behavior_copy"]
    assert len(copy_pairs) == 1
    p = copy_pairs[0]
    assert set(p["negatives"]) == {"neg1.md", "neg2.md", "neg3.md"}
    # The positive ("win.md") is NOT in negatives.
    assert "win.md" not in p["negatives"]


def test_hard_negs_preserve_rank_order(tmp_db):
    """When capping with max_hard_negs, keep lower-rank (more promising
    as hard neg) paths first."""
    q = "rank order test"
    # Impressions with explicit ranks.
    for i, path in enumerate([
        "pos.md", "neg_r2.md", "neg_r5.md", "neg_r3.md", "neg_r4.md",
    ], start=1):
        _seed_behavior({
            "source": "web", "event": "impression",
            "query": q, "path": path,
            "rank": {"pos.md": 1, "neg_r2.md": 2, "neg_r5.md": 5,
                     "neg_r3.md": 3, "neg_r4.md": 4}[path],
        })
    _seed_behavior({
        "source": "web", "event": "copy",
        "query": q, "path": "pos.md", "rank": 1,
    })

    pairs, stats = etp.export_pairs(days=7, min_negatives=1, max_hard_negs=2)
    p = next(p for p in pairs if p["source"] == "behavior_copy")
    # With max=2, expect the two lowest-rank negs (rank 2 and 3).
    assert p["negatives"] == ["neg_r2.md", "neg_r3.md"]


def test_interacted_paths_excluded_from_negatives(tmp_db):
    """If the same query got multiple positive actions on different paths,
    none of those paths can be a hard neg for each other."""
    q = "multi-positive"
    for i, path in enumerate(["a.md", "b.md", "c.md", "d.md"], start=1):
        _seed_behavior({
            "source": "web", "event": "impression",
            "query": q, "path": path, "rank": i,
        })
    _seed_behavior({"source": "web", "event": "copy",
                    "query": q, "path": "a.md", "rank": 1})
    _seed_behavior({"source": "web", "event": "save",
                    "query": q, "path": "b.md", "rank": 2})

    pairs, stats = etp.export_pairs(days=7, min_negatives=1)
    # For the 'copy' on a.md, b.md should NOT be a neg (user saved it).
    copy_pair = next(p for p in pairs if p["source"] == "behavior_copy")
    assert "b.md" not in copy_pair["negatives"]
    # For the 'save' on b.md, a.md should NOT be a neg either.
    save_pair = next(p for p in pairs if p["source"] == "behavior_save")
    assert "a.md" not in save_pair["negatives"]


# ── Filtering ────────────────────────────────────────────────────────────────


def test_cross_source_paths_excluded(tmp_db):
    """Paths with :// (calendar://, whatsapp://, gmail://) aren't vault-
    relative — exclude from both positives and negatives."""
    _seed_behavior({
        "source": "cli", "event": "impression",
        "query": "q", "path": "valid.md", "rank": 1,
    })
    _seed_behavior({
        "source": "cli", "event": "impression",
        "query": "q", "path": "calendar://xyz", "rank": 2,
    })
    _seed_behavior({
        "source": "cli", "event": "copy",
        "query": "q", "path": "valid.md", "rank": 1,
    })
    # Also a cross-source positive — should be skipped entirely.
    _seed_behavior({
        "source": "cli", "event": "copy",
        "query": "q", "path": "whatsapp://chat-123", "rank": 2,
    })

    pairs, stats = etp.export_pairs(days=7, min_negatives=0)
    # Only the valid.md pair survives.
    assert len(pairs) == 1
    assert pairs[0]["positive"] == "valid.md"
    # Negs do NOT include cross-source paths.
    assert not any("://" in n for n in pairs[0]["negatives"])


def test_min_negatives_gate(tmp_db):
    """When min_negatives > available, the pair is dropped."""
    q = "gated"
    # One impression, one positive copy — no hard negs available.
    _seed_behavior({"source": "web", "event": "impression",
                    "query": q, "path": "solo.md", "rank": 1})
    _seed_behavior({"source": "web", "event": "copy",
                    "query": q, "path": "solo.md", "rank": 1})

    # min_negatives=0 → pair survives (no neg requirement).
    pairs, _ = etp.export_pairs(days=7, min_negatives=0)
    assert len(pairs) == 1

    # min_negatives=1 → pair dropped.
    pairs, _ = etp.export_pairs(days=7, min_negatives=1)
    assert len(pairs) == 0


def test_window_cutoff_excludes_old_rows(tmp_db):
    """Events older than `days` don't appear in the export."""
    old_ts = (datetime.now() - timedelta(days=100)).isoformat(timespec="seconds")
    new_ts = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
    # Seed both old and new (raw insert, bypassing helper so we control ts).
    for ts, path in [(old_ts, "old.md"), (new_ts, "new.md")]:
        with rag._ragvec_state_conn() as conn:
            rag._sql_append_event(conn, "rag_behavior", rag._map_behavior_row({
                "ts": ts, "source": "cli", "event": "copy",
                "query": "windowed", "path": path, "rank": 1,
            }))
            rag._sql_append_event(conn, "rag_behavior", rag._map_behavior_row({
                "ts": ts, "source": "cli", "event": "impression",
                "query": "windowed", "path": "decoy.md", "rank": 2,
            }))

    pairs, stats = etp.export_pairs(days=30, min_negatives=0)
    positives = [p["positive"] for p in pairs]
    assert "new.md" in positives
    assert "old.md" not in positives


# ── Dedup ───────────────────────────────────────────────────────────────────


def test_same_query_path_event_deduplicated(tmp_db):
    """A user who copies the same chunk 5 times generates 1 training row,
    not 5 redundant ones."""
    q = "dupe"
    _seed_behavior({"source": "web", "event": "impression",
                    "query": q, "path": "pos.md", "rank": 1})
    _seed_behavior({"source": "web", "event": "impression",
                    "query": q, "path": "neg.md", "rank": 2})
    for _ in range(5):
        _seed_behavior({"source": "web", "event": "copy",
                        "query": q, "path": "pos.md", "rank": 1})

    pairs, stats = etp.export_pairs(days=7, min_negatives=1)
    copy_pairs = [p for p in pairs if p["source"] == "behavior_copy"]
    assert len(copy_pairs) == 1, (
        f"expected 1 deduped pair, got {len(copy_pairs)}"
    )


# ── Stats shape ─────────────────────────────────────────────────────────────


def test_stats_shape_and_counts(tmp_db):
    """Stats dict contract."""
    _seed_behavior({"source": "web", "event": "impression",
                    "query": "qx", "path": "a.md", "rank": 1})
    _seed_behavior({"source": "web", "event": "impression",
                    "query": "qx", "path": "b.md", "rank": 2})
    _seed_behavior({"source": "web", "event": "copy",
                    "query": "qx", "path": "a.md", "rank": 1})

    pairs, stats = etp.export_pairs(days=7, min_negatives=1)
    assert stats["total_pairs"] == len(pairs)
    assert stats["unique_queries"] >= 1
    assert stats["behavior_rows"] >= 3
    assert "behavior_copy" in stats["by_source"]
    assert stats["by_source"]["behavior_copy"] == 1


# ── Smoke: CLI entrypoint ───────────────────────────────────────────────────


def test_cli_stats_only_smoke(tmp_db, monkeypatch, capsys):
    """`main()` with --stats-only shouldn't print JSONL nor raise."""
    _seed_behavior({"source": "web", "event": "impression",
                    "query": "qs", "path": "p.md", "rank": 1})
    monkeypatch.setattr(sys, "argv", [
        "export_training_pairs.py", "--stats-only", "--days", "7",
    ])
    rc = etp.main()
    assert rc == 0
    # stats go to stderr; stdout should be empty.
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Training-pair export" in captured.err
