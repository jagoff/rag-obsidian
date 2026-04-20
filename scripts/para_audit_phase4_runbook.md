# Phase 4: Post-Move Repair — PARA Audit Runbook

**When to run**: immediately after `para_audit_apply.py --apply` reports success and its own summary.
**Goal**: restore wikilink density, synchronize the corpus index, verify no eval regression.

---

## Prerequisites

- `para_audit_apply.py --apply` completed with exit 0.
- `scripts/moves_applied.jsonl` exists (audit trail — do not delete).
- `com.fer.obsidian-rag-watch` is running (`launchctl list | grep obsidian-rag-watch`). The watcher's debounce has been running during the bulk move; by the time you reach this runbook, most file-system events have already fired.

---

## Step 1 — Re-densify wikilinks

```
rag wikilinks suggest --apply
```

**Why.** Obsidian resolves `[[Title]]` by title, not path, so folder moves preserve existing links. This step is not about repairing broken links — it is about densifying the graph. Phase 3b moved notes into new PARA buckets, which changes which titles are now co-located and which paths are freshly visible in `title_to_paths`. Running the densifier after the move ensures every note gets the maximum number of wikilinks the current vault title-map supports.

The densifier skips frontmatter, code blocks, existing links, ambiguous titles, titles shorter than 4 chars, and self-links. It applies suggestions high-to-low by offset so character positions stay stable.

**Expected.** Terminal shows per-note suggestion counts and a total. A non-zero count is normal; zero is also fine if the vault was already dense. No files are deleted.

---

## Step 2 — Incremental reindex

```
rag index
```

**Why (and why NOT `--reset`).** The `watch` service auto-reindexes on file changes, so most path metadata is already updated. Running `rag index` manually ensures the sqlite-vec corpus is fully consistent before the eval gate runs. It is hash-based: chunks whose content is unchanged are not re-embedded. The cost is metadata sync only — roughly one pass through file mtimes.

The one exception: Phase 3b stamps each moved note with `para_reclassified_*` frontmatter keys. That changes the file hash, so those notes re-embed. On a typical vault this is hundreds of notes, not thousands.

Do NOT use `--reset`. Full re-embed is 15–30 minutes and serves no purpose when file contents are largely unchanged.

**Expected.** Output like `X chunks added, Y removed, Z updated`. Counts depend on how many notes were moved and had frontmatter stamped. The command exits 0.

---

## Step 3 — Eval gate

### 3a. Unset RAG_EXPLORE in your shell

```
unset RAG_EXPLORE
```

`RAG_EXPLORE=1` is set on the `morning`/`today` plists to generate counterfactuals via ε-exploration in `retrieve()`. `rag eval` actively pops it as a belt-and-suspenders guard, but unset it manually first so the subprocesses inherit a clean environment.

### 3b. Run eval

```
rag eval
```

**Why.** The bulk move changes the document distribution seen by retrieval. A path that previously matched a query chunk may now be indexed under a different folder prefix. Eval catches this before it silently degrades daily queries.

**What it reports.** For each query group:

- `singles hit@5` — fraction of single-turn queries where the target note appears in top 5
- `MRR` — mean reciprocal rank
- `chains hit@5 / MRR / chain_success` — multi-turn chain metrics
- Bootstrap 95% CI for each (1000 resamples, seed=42)

**Gate (hard stop if either fails):**

| Metric | Floor |
|--------|-------|
| singles hit@5 | ≥ 76.19% |
| chains hit@5 | ≥ 63.64% |

These are the lower CI bounds from the 2026-04-17 expanded eval floor (42 singles, 12 chains). If either number drops below its floor, something broke — most likely a wikilink that was silently used as a retrieval signal is now dangling, or the moved note's embed_text prefix changed enough to shift cosine similarity below the rerank threshold.

**If both pass**: continue to Step 4.
**If either fails**: go to the Rollback section.

---

## Step 4 — Latency check (optional but recommended)

```
rag eval --latency --max-p95-ms 4000
```

**Why.** After 1000+ note moves, sqlite-vec may have fragmented pages. The latency flag reports P50/P95/P99 of `retrieve()` per bucket (singles/chains) and fails if `--max-p95-ms` is exceeded.

**Current floor (2026-04-17):**

| Bucket | P95 floor |
|--------|-----------|
| singles | 2447 ms |
| chains | 3003 ms |

The `--max-p95-ms 4000` gate gives ~37% headroom over the chains floor — enough to flag real fragmentation without false positives from load variance.

If latency fails: run `rag maintenance` (rotates SQL tables, WAL checkpoint, VACUUM gated). Then re-run `rag eval --latency`.

---

## What NOT to do after Phase 3b

- **Do not run `rag index --reset`** — full re-embed is unnecessary and takes 15–30 minutes.
- **Do not run `rag tune --apply` immediately** — behavior priors in `rag_behavior` are keyed by old paths and are briefly stale. The nightly `com.fer.obsidian-rag-online-tune` at 03:30 will re-accumulate signal naturally. Let it run for 2–3 nights before tuning again.
- **Do not delete `filing_batches/archive-*.jsonl` or `moves_applied.jsonl`** — these are the audit trail and rollback source. They are pruned automatically by `rag maintenance` after 60 days.
- **Do not move notes manually in Finder during this sequence** — the watcher will fire mid-reindex and produce a race. Finish all four steps, then make manual edits.

---

## Rollback — if eval fails

### 1. Locate the audit trail

```
~/.local/share/obsidian-rag/para_audit/moves_applied.jsonl
```

Each line is a JSON object with at minimum `from_path` and `to_path` (absolute paths as written by `para_audit_apply.py`).

### 2. Reverse all moves (jq one-liner)

```bash
jq -r '"\(.to_path)\t\(.from_path)"' \
  ~/.local/share/obsidian-rag/para_audit/moves_applied.jsonl \
  | while IFS=$'\t' read -r src dst; do
      mkdir -p "$(dirname "$dst")"
      mv "$src" "$dst"
    done
```

This reverses each move in the order the lines appear. If a line was already reversed or the file is missing, `mv` prints an error and continues — check the output for unexpected failures.

### 3. Reindex after reversal

```
rag index
```

Picks up the reversed paths automatically. Same incremental rules apply.

### 4. Confirm restoration

```
rag eval
```

Both gates should return to their pre-audit values. If they do not, the issue predates Phase 3b — compare against the eval baseline in `CLAUDE.md` and investigate separately.

---

## Summary of commands in order

```
rag wikilinks suggest --apply
rag index
unset RAG_EXPLORE
rag eval
rag eval --latency --max-p95-ms 4000   # optional
```

Total estimated time: 5–15 minutes depending on vault size and how many notes were moved.
