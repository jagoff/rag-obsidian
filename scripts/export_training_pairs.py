"""Export training pairs from all telemetry sources for ranker fine-tuning.

Complements `scripts/finetune_reranker.py` (2026-04-22, peer commit eca8c6f)
which consumes `rag_feedback` only. This script widens the net:

  Positives (in decreasing priority):
    1. rag_feedback.corrective_path      (clean, explicit — strongest signal)
    2. rag_feedback rating=+1 paths      (noisy, user liked *something* in the turn)
    3. rag_behavior event in {copy, save, kept, open}  (implicit, from use)
    4. rag_behavior event='positive_implicit'          (ambient agent signals)

  Hard negatives (mined from history, no re-retrieve):
    a. rag_queries.paths NOT clicked within N seconds of the impression
       (i.e. surfaced by retrieve but user ignored → classic hard neg)
    b. rag_feedback rating=-1 paths (when corrective_path is absent → all
       retrieved paths were unhelpful)
    c. rag_feedback rating=-1 paths EXCEPT the corrective_path (when
       corrective_path is present → those were surfaced but wrong)
    d. rag_behavior event in {deleted, negative_implicit}

  Output (JSONL, one line per training pair):
    {"query": str, "positive": str, "negatives": [str, ...],
     "source": "corrective"|"rating"|"behavior_copy"|...,
     "turn_id": str|None, "ts": iso8601}

Usage:
  # Export all pairs from the last 60 days (default):
  python scripts/export_training_pairs.py > pairs.jsonl

  # Narrower window + minimum 5 negatives per pair:
  python scripts/export_training_pairs.py --days 30 --min-negatives 5 \\
    --output pairs.jsonl

  # Stats-only (no file output), grouped by source type:
  python scripts/export_training_pairs.py --stats-only

Rationale for history-based hard negs (vs re-retrieve like the peer script):
  - Re-retrieve is O(1-3s per query) × N queries = slow on large history
  - History-based uses the ACTUAL retrieve results the user saw, which are
    more representative of real rerank difficulty than today's retrieve
    (the corpus may have drifted — new notes that didn't exist at turn time
    would muddy the signal)
  - Captures cases the peer script misses: turns where the user went
    straight to Obsidian without clicking anything in our UI

Standalone utility: does NOT trigger fine-tuning or consume compute beyond
the SQL reads. Safe to run on the live DB at any time.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Run with the venv python so `rag._ragvec_state_conn` resolves. The import
# is heavy (torch, sqlite-vec) but we only use the connection helper — the
# rest of rag.py is untouched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402


# ── Signal classification ───────────────────────────────────────────────────

# Rows in rag_behavior that count as implicit positives. Synced with
# rag._BEHAVIOR_POSITIVE so the training set agrees with the CTR aggregator.
_BEHAVIOR_POSITIVE_EVENTS = frozenset({
    "open", "copy", "save", "kept", "positive_implicit",
})

# Rows in rag_behavior that count as explicit negatives (user actively
# rejected — not just "didn't click" which is weaker).
_BEHAVIOR_NEGATIVE_EVENTS = frozenset({"deleted", "negative_implicit"})

# Sentinel source tags for the exported JSONL — stable, greppable.
_SOURCE_CORRECTIVE = "corrective"
_SOURCE_RATING_POS = "rating_pos"
_SOURCE_RATING_NEG_NO_CORR = "rating_neg_no_corrective"
_SOURCE_BEHAVIOR_PREFIX = "behavior_"  # + event type, e.g. behavior_copy


# ── Extractors — one per positive signal source ─────────────────────────────


def _extract_feedback_rows(cutoff_iso: str) -> list[dict]:
    """Fetch rag_feedback rows joined with their originating rag_queries row
    (so we have the full top-K paths at impression time, not just the ones
    the user volunteered in the feedback payload).

    Shape per row:
      {rating, turn_id, q, paths: [...], corrective_path, ts}
    """
    rows: list[dict] = []
    sql = (
        "SELECT f.ts, f.rating, f.turn_id, f.q, f.paths_json, "
        "       json_extract(f.extra_json, '$.corrective_path') AS cp, "
        "       q.paths_json AS q_paths_json "
        "FROM rag_feedback f "
        "LEFT JOIN rag_queries q "
        "  ON json_extract(q.extra_json, '$.turn_id') = f.turn_id "
        "WHERE f.ts >= ? "
        "ORDER BY f.ts ASC"
    )
    with rag._ragvec_state_conn() as conn:
        cur = conn.execute(sql, (cutoff_iso,))
        for ts, rating, turn_id, q, paths_json, cp, q_paths_json in cur.fetchall():
            if not q:
                continue
            # Prefer paths from the query-join (full top-K) over feedback's
            # truncated list. The feedback payload sometimes caps at top-3
            # for display; we want the real top-K for negative mining.
            try:
                paths = json.loads(q_paths_json) if q_paths_json else None
            except Exception:
                paths = None
            if not paths:
                try:
                    paths = json.loads(paths_json) if paths_json else []
                except Exception:
                    paths = []
            # Strip cross-source URIs (calendar://, whatsapp://, gmail://) —
            # not vault-relative, useless for a vault-only reranker.
            vault_paths = [p for p in paths if p and "://" not in p]
            corrective = cp.strip() if isinstance(cp, str) and cp.strip() else None
            rows.append({
                "ts": ts,
                "rating": int(rating) if rating is not None else 0,
                "turn_id": turn_id or None,
                "q": q,
                "paths": vault_paths,
                "corrective_path": corrective,
            })
    return rows


def _extract_behavior_rows(cutoff_iso: str) -> list[dict]:
    """Fetch rag_behavior rows within the window. We don't join to queries
    here — the behavior row already carries `query` (populated by every
    emitter: web Cmd+C listener, CLI /copy handler, retrieval impression
    logger).

    Shape per row:
      {ts, source, event, path, query, rank}
    """
    rows: list[dict] = []
    sql = (
        "SELECT ts, source, event, path, query, rank "
        "FROM rag_behavior "
        "WHERE ts >= ? "
        "  AND path IS NOT NULL AND path != '' "
        "  AND query IS NOT NULL AND query != '' "
        "ORDER BY ts ASC"
    )
    with rag._ragvec_state_conn() as conn:
        cur = conn.execute(sql, (cutoff_iso,))
        for ts, source, event, path, query, rank in cur.fetchall():
            # Skip cross-source paths — same rationale as feedback.
            if not path or "://" in path:
                continue
            rows.append({
                "ts": ts,
                "source": source or "",
                "event": event or "",
                "path": path,
                "q": query,
                "rank": int(rank) if rank is not None else None,
            })
    return rows


# ── Hard-negative mining (history-based) ────────────────────────────────────


def _index_impressions_by_query(
    behavior_rows: list[dict],
) -> dict[str, list[tuple[str, int | None]]]:
    """Group `impression` events by normalised query → list of (path, rank).
    These are the paths the system surfaced at retrieve time; whichever
    wasn't later `open`/`copy`/`save`d within the same query's window is
    a hard negative.
    """
    out: dict[str, list[tuple[str, int | None]]] = defaultdict(list)
    for r in behavior_rows:
        if r["event"] == "impression":
            out[r["q"].strip().lower()].append((r["path"], r["rank"]))
    return out


def _index_positives_by_query(
    behavior_rows: list[dict],
) -> dict[str, set[str]]:
    """Group positive-action events (open, copy, save, kept, positive_implicit)
    by normalised query → set of paths. These are the ones the user DID
    interact with.
    """
    out: dict[str, set[str]] = defaultdict(set)
    for r in behavior_rows:
        if r["event"] in _BEHAVIOR_POSITIVE_EVENTS:
            out[r["q"].strip().lower()].add(r["path"])
    return out


def _mine_hard_negatives_from_history(
    query: str,
    positive_path: str,
    impressions: dict[str, list[tuple[str, int | None]]],
    interacted: dict[str, set[str]],
    max_negs: int,
) -> list[str]:
    """For a (query, positive_path) pair, return paths that were surfaced
    (impression) for the same query but were NOT clicked/copied/saved, up
    to max_negs. Excludes the positive_path itself.

    Falls back to empty list if the query has no impression history (e.g.
    the `log_impressions()` call was added after this data was collected).
    The caller can still export the pair with zero negatives — downstream
    tuning decides whether to skip or use in-batch negatives.
    """
    key = query.strip().lower()
    surfaced = impressions.get(key, [])
    clicked = interacted.get(key, set()) | {positive_path}
    # Preserve rank order (lower rank = more promising hard neg).
    seen: set[str] = set()
    ordered: list[tuple[str, int]] = []
    for path, rank in surfaced:
        if path in clicked or path in seen:
            continue
        seen.add(path)
        ordered.append((path, rank if rank is not None else 999))
    ordered.sort(key=lambda pr: pr[1])
    return [p for p, _ in ordered[:max_negs]]


def _explicit_negatives_from_feedback(
    row: dict,
) -> list[str]:
    """For a feedback row with rating=-1, the retrieved paths are negatives.
    If corrective_path is present, it's the "right" answer among what was
    surfaced → exclude it from the neg list.
    """
    paths = row.get("paths", []) or []
    corrective = row.get("corrective_path")
    return [p for p in paths if p and p != corrective]


# ── Pair builders ───────────────────────────────────────────────────────────


def _build_pair(
    *,
    query: str,
    positive: str,
    negatives: list[str],
    source: str,
    turn_id: str | None,
    ts: str,
    min_negatives: int,
) -> dict | None:
    """Emit a standardised pair dict, or None when negatives fall below
    the threshold. Deduplicates negatives and caps implicitly via the
    min_negatives gate."""
    if not query or not positive:
        return None
    negs = []
    seen: set[str] = set()
    for n in negatives:
        if not n or n == positive or n in seen:
            continue
        seen.add(n)
        negs.append(n)
    if len(negs) < min_negatives:
        return None
    return {
        "query": query,
        "positive": positive,
        "negatives": negs,
        "source": source,
        "turn_id": turn_id,
        "ts": ts,
    }


def _yield_pairs_from_feedback(
    feedback_rows: list[dict],
    impressions: dict[str, list[tuple[str, int | None]]],
    interacted: dict[str, set[str]],
    *,
    min_negatives: int,
    max_hard_negs: int,
):
    """Emit pairs sourced from rag_feedback. Priority order:
      1. corrective_path  → _SOURCE_CORRECTIVE (strongest: explicit gold)
      2. rating=+1 paths  → _SOURCE_RATING_POS (noisy: any path in turn)
      3. rating=-1 without corrective → negative-only row, returned as
         pure hard-neg seed (no positive — caller discards if no positive
         for the query elsewhere)
    """
    for row in feedback_rows:
        q = row["q"]
        # Case 1: corrective — single gold positive.
        if row.get("corrective_path"):
            positive = row["corrective_path"]
            # Negs: other retrieved paths (except the corrective) + history
            # hard negs from other surfaces.
            explicit = [p for p in row["paths"] if p != positive]
            history = _mine_hard_negatives_from_history(
                q, positive, impressions, interacted, max_hard_negs,
            )
            # Dedup while preserving order (explicit first — they were
            # literally in the same turn, highest confidence as negs).
            merged: list[str] = []
            seen: set[str] = set()
            for n in explicit + history:
                if n not in seen and n != positive:
                    seen.add(n)
                    merged.append(n)
            pair = _build_pair(
                query=q, positive=positive, negatives=merged,
                source=_SOURCE_CORRECTIVE,
                turn_id=row["turn_id"], ts=row["ts"],
                min_negatives=min_negatives,
            )
            if pair:
                yield pair
            continue

        # Case 2: rating=+1, no corrective. Every path in the turn is a
        # noisy positive (we don't know WHICH of the top-K helped). Emit
        # one pair per path — common approach in implicit feedback mining.
        if row["rating"] > 0 and row["paths"]:
            for positive in row["paths"]:
                history = _mine_hard_negatives_from_history(
                    q, positive, impressions, interacted, max_hard_negs,
                )
                pair = _build_pair(
                    query=q, positive=positive, negatives=history,
                    source=_SOURCE_RATING_POS,
                    turn_id=row["turn_id"], ts=row["ts"],
                    min_negatives=min_negatives,
                )
                if pair:
                    yield pair


def _yield_pairs_from_behavior(
    behavior_rows: list[dict],
    impressions: dict[str, list[tuple[str, int | None]]],
    interacted: dict[str, set[str]],
    *,
    min_negatives: int,
    max_hard_negs: int,
):
    """Emit pairs sourced from rag_behavior positive events. Each (query,
    path) positive gets hard negs mined from its own impression history.
    """
    # Dedup: emit only one pair per (query, path, event_type). Without
    # this, a user who copies the same chunk 5 times generates 5
    # redundant training rows.
    seen_pairs: set[tuple[str, str, str]] = set()
    for row in behavior_rows:
        if row["event"] not in _BEHAVIOR_POSITIVE_EVENTS:
            continue
        key = (row["q"].strip().lower(), row["path"], row["event"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        negs = _mine_hard_negatives_from_history(
            row["q"], row["path"], impressions, interacted, max_hard_negs,
        )
        pair = _build_pair(
            query=row["q"], positive=row["path"], negatives=negs,
            source=f"{_SOURCE_BEHAVIOR_PREFIX}{row['event']}",
            turn_id=None, ts=row["ts"],
            min_negatives=min_negatives,
        )
        if pair:
            yield pair


# ── Orchestration ───────────────────────────────────────────────────────────


def export_pairs(
    *,
    days: int = 60,
    min_negatives: int = 1,
    max_hard_negs: int = 9,
) -> tuple[list[dict], dict]:
    """Fetch all telemetry sources, mine pairs, return (pairs, stats).

    Stats shape:
      {
        "days_window": int,
        "total_pairs": int,
        "by_source": {source_name: int},
        "unique_queries": int,
        "feedback_rows": int,
        "behavior_rows": int,
        "impression_queries": int,
      }
    """
    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

    feedback_rows = _extract_feedback_rows(cutoff_iso)
    behavior_rows = _extract_behavior_rows(cutoff_iso)

    impressions = _index_impressions_by_query(behavior_rows)
    interacted = _index_positives_by_query(behavior_rows)

    pairs: list[dict] = []
    for p in _yield_pairs_from_feedback(
        feedback_rows, impressions, interacted,
        min_negatives=min_negatives, max_hard_negs=max_hard_negs,
    ):
        pairs.append(p)
    for p in _yield_pairs_from_behavior(
        behavior_rows, impressions, interacted,
        min_negatives=min_negatives, max_hard_negs=max_hard_negs,
    ):
        pairs.append(p)

    # Stats
    by_source: dict[str, int] = defaultdict(int)
    unique_qs: set[str] = set()
    for p in pairs:
        by_source[p["source"]] += 1
        unique_qs.add(p["query"].strip().lower())
    stats = {
        "days_window": days,
        "total_pairs": len(pairs),
        "by_source": dict(by_source),
        "unique_queries": len(unique_qs),
        "feedback_rows": len(feedback_rows),
        "behavior_rows": len(behavior_rows),
        "impression_queries": len(impressions),
    }
    return pairs, stats


# ── CLI ────────────────────────────────────────────────────────────────────


def _print_stats(stats: dict, file=None) -> None:
    # Resolve sys.stderr at call time so capsys/capfd in tests can
    # intercept (a default of `sys.stderr` in the signature binds the
    # original fd at import time and bypasses fixture patching).
    if file is None:
        file = sys.stderr
    print(f"== Training-pair export — {stats['days_window']}d window ==", file=file)
    print(f"  Pairs total: {stats['total_pairs']}", file=file)
    print(f"  Unique queries: {stats['unique_queries']}", file=file)
    print(f"  Source breakdown:", file=file)
    for src, n in sorted(stats["by_source"].items(), key=lambda kv: -kv[1]):
        print(f"    {src:30s} {n}", file=file)
    print(f"  Underlying rows: "
          f"feedback={stats['feedback_rows']} "
          f"behavior={stats['behavior_rows']} "
          f"impression_queries={stats['impression_queries']}", file=file)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=60,
                    help="Lookback window in days (default 60)")
    ap.add_argument("--min-negatives", type=int, default=1,
                    help="Skip pairs with fewer hard negatives than this (default 1)")
    ap.add_argument("--max-hard-negs", type=int, default=9,
                    help="Cap negatives per pair (default 9, for 1:9 pos:neg ratio)")
    ap.add_argument("--output", "-o", type=str, default="-",
                    help="Output path (default stdout)")
    ap.add_argument("--stats-only", action="store_true",
                    help="Print stats to stderr; don't emit JSONL")
    args = ap.parse_args()

    pairs, stats = export_pairs(
        days=args.days,
        min_negatives=args.min_negatives,
        max_hard_negs=args.max_hard_negs,
    )

    _print_stats(stats)

    if args.stats_only:
        return 0

    # Write JSONL
    if args.output == "-":
        for p in pairs:
            sys.stdout.write(json.dumps(p, ensure_ascii=False) + "\n")
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(pairs)} pairs → {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
