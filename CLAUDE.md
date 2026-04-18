# CLAUDE.md

Local RAG over an Obsidian vault. Single-file: `rag.py` (~21k lines) + `mcp_server.py` (thin wrapper) + `tests/` (883 tests, 44 files). Resist package-split until real friction shows up.

Entry points (both installed via `uv tool install --editable .`):
- `rag` — CLI for indexing, querying, chat, productivity, automation
- `obsidian-rag-mcp` — MCP server (`rag_query`, `rag_read_note`, `rag_list_notes`, `rag_links`, `rag_stats`)

Fully local: ChromaDB + Ollama + sentence-transformers. No cloud calls.

## Agent dispatch rule

**Any task that will edit ≥3 files MUST go through `pm` first.** No preguntas — invocar directamente:

```
Agent(subagent_type: "pm", prompt: "<goal + context + ruled-out + invariants at risk>")
```

The PM returns a dispatch plan (tasks, dependencies, parallel-safe flags, risks, validation). The main session executes the plan by spawning the named agents in the prescribed order — never silently skips PM and improvises.

Tasks that touch ≤2 files go directly to the owning agent (`rag-retrieval`, `rag-llm`, `rag-brief-curator`, `rag-ingestion`, `rag-vault-health`, `rag-integrations`, `developer-{1,2,3}`). Roster + ownership lives in `.claude/agents/README.md`.

When peers are active (`mcp__claude-peers__list_peers(scope: "repo")` returns >1), even ≤2-file tasks may need PM coordination — flag overlapping zones before editing.

## Commands

```bash
uv tool install --reinstall --editable .   # reinstall after code changes

# Core
rag index [--reset] [--no-contradict]      # incremental hash-based; --reset rebuilds
rag watch                                  # watchdog auto-reindex (debounce 3s)
rag query "text" [--hyde --no-multi --raw --loose --force --counter --no-deep --session ID --continue --plain]
rag chat [--counter --no-deep --session ID --resume] # /save /reindex (or NL) work mid-conversation
rag do "instrucción" [--yes --max-iterations 8]  # tool-calling agent loop
rag stats                                  # models + index status
rag session list|show|clear|cleanup

# Productivity
rag capture "texto" [--tag X --source Y --stdin --title T --plain]
rag inbox [--apply]                        # triage 00-Inbox: folder + tags + wikilinks + dupes
rag prep "tema" [--save]                   # context brief → optionally 00-Inbox/
rag read <url> [--save --plain]            # ingest article → 00-Inbox/ w/ auto-wikilinks
rag dupes [--threshold 0.85 --folder X]
rag links "query" [--open N --rebuild]     # semantic URL finder, no LLM
rag wikilinks suggest [--folder X --apply] # graph densifier, no LLM
rag followup [--days 30 --status stale|activo|resolved --json]
rag dead [--min-age-days 365]              # candidates to archive (read-only)
rag archive [--apply --force --gate 20]    # move dead → 04-Archive/ (dry-run default)

# Daily automation
rag morning [--dry-run]                    # daily brief → 05-Reviews/YYYY-MM-DD.md
rag today [--dry-run]                      # EOD closure → 05-Reviews/YYYY-MM-DD-evening.md
rag digest [--week YYYY-WNN --days N]      # weekly narrative → 05-Reviews/YYYY-WNN.md

# Ambient agent
rag ambient status|disable|test [path]|log [-n N]
rag ambient folders list|add <F>|remove <F>

# Quality
rag eval [--latency --max-p95-ms N]        # queries.yaml → hit@k, MRR, recall@k (+ bootstrap CI); gate on P95
rag tune [--samples 500] [--apply] [--online --days 14] [--rollback]  # offline + online ranker-vivo loop
rag log [-n 20] [--low-confidence]
rag dashboard [--days 30]                  # analytics: scores, latency, topics, PageRank
rag open <path> [--query Q --rank N --source cli]  # emits behavior event + `open` path (ranker-vivo click tracking)

# Maintenance
rag maintenance [--dry-run --skip-reindex --skip-logs --json]  # all-in-one housekeeping

# Automation
rag setup [--remove]                       # install/remove 9 launchd services

# Tests
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_foo.py::test_bar -q   # single test
```

Python 3.13, `uv`. Runtime venv: `.venv/bin/python`. Global tool: `~/.local/share/uv/tools/obsidian-rag/`.

### Env vars

- `OBSIDIAN_RAG_VAULT` — override default vault path. Collections are namespaced per resolved path (sha256[:8]).
- `RAG_TRACK_OPENS=1` — switches OSC 8 link scheme from `file://` to `x-rag-open://` so CLI clicks route through `rag open` (ranker-vivo signal capture). Absent = no behavior change.
- `RAG_EXPLORE=1` — enable ε-exploration in `retrieve()` (10% chance to swap a top-3 result with a rank-4..7 candidate). Set on `morning`/`today` plists to generate counterfactuals. MUST be unset during `rag eval` — the command actively `os.environ.pop`s it and asserts, as a belt-and-suspenders guard.
- `RAG_RERANKER_IDLE_TTL` — seconds the cross-encoder stays resident before idle-unload (default 900).
- `RAG_RERANKER_NEVER_UNLOAD` — set to `1` in the web launchd plist to pin the reranker in MPS VRAM permanently; sweeper loop still runs but skips `maybe_unload_reranker()`. Eliminates the 9s cold-reload hit after idle eviction. Cost: ~2-3 GB unified memory pinned. Safe on 36 GB with command-r + qwen3:8b resident.
- `RAG_LOCAL_EMBED` — set to `1` in the web launchd plist to use in-process `SentenceTransformer("BAAI/bge-m3")` for query embedding instead of ollama HTTP (~10-30ms vs ~140ms). Requires BAAI/bge-m3 cached in `~/.cache/huggingface/hub/` — download once with `python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"` before enabling. Verify cosine >0.999 vs ollama embeddings of same text before enabling in production. Do NOT set for indexing/watch/ingest processes — bulk chunk embedding stays on ollama. Uses CLS pooling (same as ollama gguf).

## Architecture — invariants

### Retrieval pipeline (`retrieve()`)

```
query → classify_intent → infer_filters [auto]
      → expand_queries (3 paraphrases, ONE qwen2.5:3b call)
      → embed(variants) batched bge-m3
      → per variant: ChromaDB sem + BM25 (accent-normalised, GIL-serialised — do NOT parallelise)
      → RRF merge → dedup → expand to parent section (O(1) metadata)
      → cross-encoder rerank (bge-reranker-v2-m3, MPS+fp16)
      → graph expansion (1-hop wikilink neighbors, always on)
      → [auto-deep: if confidence < 0.10, iterative sub-query retrieval]
      → top-k → LLM (streamed)
```

**Graph expansion** (always on): after rerank, top-3 results expand via 1-hop wikilink neighbors (`_build_graph_adj` + `_hop_set`). Up to 3 graph neighbors added as supplementary LLM context marked `[nota relacionada (grafo)]`. Cost: in-memory graph lookups, negligible.

**Auto-deep retrieval**: when top rerank score < `CONFIDENCE_DEEP_THRESHOLD` (0.10), `deep_retrieve()` auto-triggers: helper model judges sufficiency → generates focused sub-query → second retrieve pass → merge results. Max 3 iterations. Disable with `--no-deep`.

**Corpus cache** (`_load_corpus`): BM25 + vocab built once, invalidated by `col.count()` delta. Cold 341ms → warm 2ms. Do not touch without re-measuring.

### Indexing

Chunks 150–800 chars, split on headers + blank lines, merged if < MIN_CHUNK. Each chunk: `embed_text` (prefixed `[folder|title|area|#tags]` + contextual summary), `display_text` (raw), `parent` metadata (enclosing section, ≤1200 chars). Hash per file → re-embed only on change. `is_excluded()` skips `.`-prefixed segments.

**Contextual embeddings** (v8→v9): `get_context_summary()` generates a 1-2 sentence document-level summary per note via qwen2.5:3b, prepended to each chunk's `embed_text` as `Contexto: ...`. Cached by file hash in `~/.local/share/obsidian-rag/context_summaries.json`. Notes < 300 chars skip summarization. Improves multi-hop and chain retrieval significantly (+11% chain_success).

**Temporal tokens** (v9): `build_prefix()` appends `[recent]`/`[this-month]`/`[this-quarter]`/`[older]` based on `modified`/`created` frontmatter. Shifts embedding space so "current work" queries prefer recent notes without post-hoc boosts.

**Graph PageRank**: `_graph_pagerank()` computes authority scores over the wikilink adjacency graph (power iteration, <10ms). Cached per corpus. Used as a tuneable ranking signal (`graph_pagerank` weight) and to sort graph expansion neighbors.

**Schema changes**: bump `_COLLECTION_BASE` (currently `obsidian_notes_v9`). Per-vault suffix = sha256[:8] of resolved path.

### Model stack

| Role | Model | Notes |
|------|-------|-------|
| Chat | `resolve_chat_model()`: command-r > qwen2.5:14b > phi4 | RAG-trained, citation-native |
| Helper | `qwen2.5:3b` | paraphrase/HyDE/reformulation |
| Embed | `bge-m3` | 1024-dim multilingual |
| Reranker | `BAAI/bge-reranker-v2-m3` | `device="mps"` + `float16` forced — do NOT remove (CPU fallback = 3× slower) |

All ollama calls: `keep_alive=-1` (VRAM resident). `CHAT_OPTIONS`: `num_ctx=4096, num_predict=768` — don't bump unless prompts grow.

**Pattern**: helper for cheap rewrites, chat model for judgment. Contradiction detector MUST use chat model (qwen2.5:3b proved non-deterministic + malformed JSON on this task).

### Confidence gate

`top_score < 0.015` (CONFIDENCE_RERANK_MIN) + no `--force` → refuse without LLM call. Calibrated for bge-reranker-v2-m3 on this corpus. Re-calibrate if reranker changes.

### Generation prompts

- `SYSTEM_RULES_STRICT` (default `rag query`): forbids external prose.
- `SYSTEM_RULES` (`--loose`, always in chat): allows `<<ext>>...<</ext>>` rendered dim yellow + ⚠.

### Rendering

OSC 8 `file://` hyperlinks for both `[Label](path.md)` and `[path.md]` formats. `NOTE_LINK_RE` handles single-level balanced parens. `verify_citations()` flags unknown paths post-response.

### Scoring formula (post-rerank)

```
score = rerank_logit
      + w.recency_cue        * recency_raw      [if has_recency_cue]
      + w.recency_always     * recency_raw      [always]
      + w.tag_literal        * n_tag_matches
      + w.graph_pagerank     * (pr/max_pr)      [wikilink authority signal]
      + w.click_prior        * ctr_path         [behavior: path CTR, Laplace-smoothed]
      + w.click_prior_folder * ctr_folder       [behavior: top-level folder CTR]
      + w.click_prior_hour   * ctr_path_hour    [behavior: path × current-hour CTR]
      + w.dwell_score        * log1p(dwell_s)   [behavior: mean dwell time per path]
      + w.feedback_pos                          [if path in feedback+ cosine≥0.80]
      - w.feedback_neg                          [if path in feedback- cosine≥0.80]
```

Weights in `~/.local/share/obsidian-rag/ranker.json` (written by `rag tune --apply`). Defaults `recency_always=0, tag_literal=0, click_*=0, dwell_score=0` preserve pre-tune behavior. Behavior knobs are inert until `behavior.jsonl` accumulates signal and `rag tune` finds non-zero weights.

Behavior priors (`_load_behavior_priors()`): read from `behavior.jsonl`, cached per mtime/size. Positive events: `open`, `positive_implicit`, `save`, `kept`. Negative: `negative_implicit`, `deleted`. CTR uses Laplace smoothing `(clicks+1)/(impressions+10)`.

## Key subsystems — contracts only

Subsystems have autodescriptive docstrings in `rag.py` and dedicated test files. Only contracts/invariants here.

**Sessions**: JSON per session in `sessions/<id>.json`. TTL 30d, cap 50 turns, history window 6. IDs validated `^[A-Za-z0-9_.:-]{1,64}$`; invalid → mint fresh. WhatsApp passes `wa:<jid>`.

**Ambient agent**: hook in `_index_single_file` on saves within `allowed_folders` (default `["00-Inbox"]`). Config: `~/.local/share/obsidian-rag/ambient.json` (`{jid, enabled, allowed_folders?}`). Skip rules: outside allowed_folders, no config, frontmatter `ambient: skip`, `type: morning-brief|weekly-digest|prep`, dedup 5min. Sends via `whatsapp-bridge` POST (`http://localhost:8080/api/send`). Bridge down = message lost but analysis persists in `ambient.jsonl`.

**Contradiction radar**: Phase 1 (query-time `--counter`), Phase 2 (index-time frontmatter `contradicts:` + `contradictions.jsonl`), Phase 3 (`rag digest` weekly). Skipped on `--reset` (O(n²)) and `note_body < 200 chars`.

**URL sub-index**: `obsidian_urls_v1` collection embeds **prose context** (±240 chars) not URL strings. `PER_FILE_CAP=2`. Auto-backfill on first `find_urls()` if collection empty.

**Wikilinks**: regex scan against `title_to_paths`. Skips: frontmatter, code, existing links, ambiguous titles, short titles (min-len 4), self-links. First occurrence only. Apply iterates high→low offset.

**Archive**: reuses `find_dead_notes`, maps to `04-Archive/<original-path>` (PARA mirror), stamps frontmatter `archived_at/archived_from/archived_reason`. Opt-out: `archive: never` or `type: moc|index|permanent`. Gate: >20 candidates without `--force` → dry-run. Batch log in `filing_batches/archive-*.jsonl`.

**Morning**: collects 36h window (modified notes, inbox, todos, contradictions, low-conf queries, Apple Reminders, calendar, weather, screentime). Weather hint only if rain ≥70%. Dedup vault-todos vs reminders (Jaccard ≥0.6). System-activity + Screen Time sections are deterministic (no LLM).

**Screen Time**: `_collect_screentime(start, end)` reads `~/Library/Application Support/Knowledge/knowledgeC.db` (`/app/usage` stream, read-only via `immutable=1` URI). Sessions <5s filtered. Bundle→label map + category rollup (code/notas/comms/browser/media/otros). Renders only if ≥5min of activity. Section omitted silently if db missing. Dashboard `/api/dashboard` exposes 7d aggregate + daily series (capped at 7 — CoreDuet aggregates older data away).

**Today**: `[00:00, now)` window, 4 fixed sections, writes `-evening.md` suffix. Feeds next morning organically.

**Followup**: extracts loops (frontmatter todo/due, unchecked `- [ ]`, imperative regex), classifies via qwen2.5:3b judge (temp=0, seed=42, conservative). One embed + one LLM call per loop.

**Read**: fetch URL → readability strip → gate (< 500 chars = error) → command-r summary → two-pass related lookup → tags from existing vocab (never invents) → `00-Inbox/`. Dry-run default, `--save` to write.

**Ranker-vivo (closed-loop ranker)**: implicit feedback from daily use re-tunes `ranker.json` nightly without manual intervention. Four signal sources append to `behavior.jsonl`: (1) CLI `rag open` wrapper (opt-in via `RAG_TRACK_OPENS=1` + user-registered `x-rag-open://` handler); (2) WhatsApp listener classifying follow-up turns (`/save`, quoted reply → positive; "no"/"la otra"/rephrase → negative; 120s silence → weak positive); (3) web `/api/behavior` POST from home dashboard `sendBeacon` clicks; (4) morning/today brief diff (`_diff_brief_signal` compares yesterday's written brief vs current on-disk — wikilinks that survived = `kept`, missing = `deleted`, dedup via `brief_state.jsonl`). Nightly `com.fer.obsidian-rag-online-tune` at 03:30 runs `rag tune --online --days 14 --apply --yes`, which calls `_behavior_augmented_cases` (weight=0.5, drops conflicts), backs up current `ranker.json` → `ranker.{ts}.json` (keeps 3 newest), re-tunes, runs the bootstrap-CI gate (`_run_eval_gate`: scrubs `RAG_EXPLORE`, subprocess `rag eval`, 10min cap, regex parses hit@5). If singles < 76.19% OR chains < 63.64% (lower CI bounds of the 2026-04-17 expanded floor) → auto-rollback + exit 1 + log to `tune.jsonl`. `rag tune --rollback` restores the most recent backup manually.

## Eval baselines

**Floor (2026-04-17, post-golden-expansion + bootstrap CI)** — queries.yaml doubled (21→42 singles, 9→12 chains; +15 singles in under-represented folders 03-Resources/Agile+Tech, 02-Areas/Personal, 01-Projects/obsidian-rag, 04-Archive memory). `rag eval` now reports percentile bootstrap 95% CI (1000 resamples, seed=42) alongside each metric + `rag eval --latency` reports P50/P95/P99 of retrieve() per bucket and accepts `--max-p95-ms` as a CI gate.
- Singles: `hit@5 88.10% [76.19, 97.62] · MRR 0.772 [0.651, 0.873] · n=42`
- Chains: `hit@5 78.79% [63.64, 90.91] · MRR 0.629 [0.490, 0.768] · chain_success 50.00% [25.00, 75.00] · turns=33 chains=12`
- Latency: singles p95 2447ms · chains p95 3003ms

Every post-expansion metric sits inside the prior floor's CI on the smaller set — expansion surfaced the noise band (~21pp singles hit, ~50pp chain_success) that previously masqueraded as drift.

**Prior floor (2026-04-17, post-title-in-rerank, n=21 singles / 9 chains):** Singles `hit@5 90.48% · MRR 0.821`; Chains `hit@5 80.00% · MRR 0.627 · chain_success 55.56%`. Kept for historical trend, but do not compare new numbers against it without overlapping CIs.

**Even-earlier floor (2026-04-16, post-quick-wins, n=21/9):** Singles `hit@5 90.48% · MRR 0.786`; Chains `hit@5 76.00% · MRR 0.580 · chain_success 55.56%`.

The 2026-04-15 floor (`95.24/0.802` singles, `72.00/0.557/44.44` chains, see `docs/eval-tune-2026-04-15.md`) pre-dates both the expansion and the CI tooling — treat as a qualitative reference only.

Never claim improvement without re-running `rag eval`. Helper LLM calls (`expand_queries`, `reformulate_query`, `_judge_sufficiency`) are already deterministic via `HELPER_OPTIONS = {temperature: 0, seed: 42}`.

**HyDE with qwen2.5:3b drops singles hit@5 ~5pp**. HyDE is opt-in (`--hyde`); re-measure if helper model changes.

**`seen_titles` in `reformulate_query` regressed chains** (2026-04-17). Injecting "notas ya consultadas: [...]" into the helper prompt as a diversity nudge dropped chains hit@5 −16pp (80→64) and chain_success −33pp (55.56→22.22). The helper treats the list as "avoid these" and drifts off-topic. The kwarg remains on the signature (callers in eval pass it via `_titles_from_paths`) but is intentionally unused in the prompt. Future: try as a soft *reranker* hint (penalty on already-seen chunks post-rerank) instead of an LLM instruction.

## On-disk state (`~/.local/share/obsidian-rag/`)

- `chroma/` — ChromaDB collections (per-vault suffixed). Orphan HNSW segment dirs (not referenced in `segments` table) + stale WAL are cleaned by `rag maintenance` via `_prune_orphan_segment_dirs` + `_chroma_wal_checkpoint`. One 33GB orphan cleanup shipped 2026-04-17.
- `queries.jsonl` — query log (q, variants, paths, scores, timings, mode, cmd)
- `feedback.jsonl` / `feedback_golden.json` (cache, rebuilt lazy on mtime gap)
- `behavior.jsonl` — ranker-vivo event log (4 sources: cli, whatsapp, web, brief). Rotated by `rag maintenance`.
- `brief_written.jsonl` / `brief_state.jsonl` — brief diff sidecars (what was cited vs dedup of already-emitted kept/deleted pairs).
- `ranker.json` — tuned weights. Delete = reset to hardcoded defaults.
- `ranker.{unix_ts}.json` — 3 most recent backups, written on every `rag tune --apply`. Consumed by `rag tune --rollback` + auto-rollback CI gate.
- `sessions/*.json` + `last_session` — multi-turn state
- `contradictions.jsonl` — radar sidecar
- `ambient.json` / `ambient.jsonl` / `ambient_state.jsonl` — config + log + dedup
- `filing_batches/*.jsonl` — audit log (prefix `archive-*` for archiver)
- `ignored_notes.json`, `tune.jsonl` (online-tune regressions also land here)
- `online-tune.{log,error.log}` — nightly tune launchd output
- `*.{log,error.log}` — other launchd service logs

**Reset learned state**: `rm ranker.json feedback_golden.json`. Dimension mismatch in feedback golden → silently ignored. Full re-embed: `rag index --reset`.

## Vault path

Default: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes`. Override: `OBSIDIAN_RAG_VAULT` env var. Collections namespaced per vault (sha256[:8]).

Claude Code memory (`~/.claude/projects/-Users-fer/memory/`) is symlinked into vault at `04-Archive/99-obsidian-system/99-Claude/memory/`.
