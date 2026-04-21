# `scripts/` — developer utilities and one-shot tools

Scripts that don't belong in `rag.py` because they're diagnostic,
one-shot, or off-path. Run from the repo root with
`.venv/bin/python scripts/<name>.py --help`.

## Benchmarks & measurement

### `bench_chat.py`
Latency benchmark for the `rag chat` path. Replicates the retrieve +
LLM-stream flow without a TTY and reports per-stage timing.

  When to run: after touching the retrieval pipeline, the reranker
  config, or the ollama model stack, to confirm no regression in
  ttft / total latency. Supports `--runs 3` for warm-vs-cold comparison.

  Output: per-query `retrieve_ms / ttft_ms / llm_ms / total_ms /
  confidence / n_docs` plus P50/P95 tail + slowest-query ranking.

### `measure_wikilink_context.py`
Measures how many wikilinks in the vault have explicit relationship
keywords within ±80 chars. Decision gate for the "typed-edges"
feature proposal: if <30% of edges carry a verb like `implementa`,
`contradice`, `mejora`, typed-edges aren't worth the implementation.

  When to run: before committing to any graph-edge typing work.
  Sampling + `--window N` lets you probe different context widths.

### `calibrate_thresholds.py`
Empirical percentile distribution of `top_score` over `rag_queries`
for the last N days (default 30). Reports where each hardcoded gate
(`CONFIDENCE_RERANK_MIN`, `CONFIDENCE_DEEP_THRESHOLD`,
`GRAPH_EXPANSION_GATE`) sits in the distribution, plus the observed
rates of `gated_low_confidence`, `bad_citations`, `critique_changed`.

  When to run: before tweaking any confidence gate in `rag.py`, to
  verify the proposed value isn't too aggressive on the current
  corpus. Read-only; does not modify anything.

## Vault structural audits (PARA flow)

Four interlocking scripts that audit vault organisation against the
PARA method (Projects / Areas / Resources / Archive). Run in order:

  1. `para_audit_inventory.py`
     Phase 0 — scan the vault, emit one JSONL line per note with
     its current PARA bucket, modified date, wikilink density.
     Output: `~/.local/share/obsidian-rag/para_audit/inventory.jsonl`

  2. `para_audit_classify.py`
     Phase 1 — LLM batch classifier (qwen2.5:3b) proposes a
     `proposed_bucket` for each note. Only emits lines where the
     proposal differs from the current bucket — review material.
     Output: `para_audit/proposals.jsonl` (NOT applied).

  3. `para_audit_review.py`
     Phase 2 — interactive review. Walks each proposal, shows the
     note's title + first paragraph + current/proposed bucket,
     accepts y/n/skip. Accepted lines go to `proposals_approved.jsonl`.

  4. `para_audit_apply.py`
     Phase 3 — bulk mover. Reads `proposals_approved.jsonl` and
     physically relocates notes. **Dry-run by default**; pass
     `--apply` to mutate. Creates parent folders as needed; updates
     wikilink references to the moved note (phase 4 handles
     wikilink repair — see runbook).

Ancillary reports: `para_audit_report.py` aggregates the inventory +
proposals into a markdown summary.

### `para_audit_phase4_runbook.md`
Runbook for post-move repair. Run **after** `para_audit_apply.py --apply`
reports success:

  - Restore wikilink density for moved notes
  - Synchronise the corpus index (`rag index`)
  - Verify no eval regression against the floor CIs (singles hit@5
    ≥ 76.19%, chains ≥ 63.64%)
  - Rollback procedure if eval breaks

## Migration

### `migrate_state_to_sqlite.py`
One-shot migration of the pre-2026-04-19 JSONL telemetry files
into the `rag_*` SQL tables created by T1 (`_ensure_telemetry_tables`).
Renames each source to `<name>.bak.<unix_ts>` on successful import
so re-runs are naturally idempotent.

**Already run on this machine (2026-04-19 cutover).** Kept in the
repo for:
  - Fresh installs (if someone clones + restores a pre-cutover
    backup).
  - Reference when implementing future migrations in the same shape
    (BEGIN IMMEDIATE + stream-parse + per-source commit + rename).
  - The --round-trip-check mode, used in
    `rag maintenance --validate-cutover` wiring.

Refuses to run while `com.fer.obsidian-rag-*` launchd services are
up (concurrent append to the source files would corrupt the
migration). Pass `--force` to override.

Docs: `--help` is authoritative (full option list, dry-run modes,
round-trip check, summary output, reverse mode).

## Why these live here, not in `rag.py`

- **Diagnostic**: `bench_chat.py`, `measure_wikilink_context.py` — run
  on-demand from the command line, never imported by the runtime.

- **One-shot**: `migrate_state_to_sqlite.py` is a 7,946-record
  import that only makes sense once per install.

- **Multi-phase orchestration**: the PARA audit flow is four
  scripts + a runbook. Bundling into `rag.py` subcommands would
  hide the phased nature (inventory → classify → review → apply)
  that forces the user to review proposals before mutation.

Any script here can be deleted if it stops being useful — they're
not load-bearing for the main install. `uv tool install --editable .`
installs `rag` and `obsidian-rag-mcp` entry points; nothing in
scripts/ is exposed through those.
