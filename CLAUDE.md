# CLAUDE.md

Local RAG over an Obsidian vault. Single-file: `rag.py` (~16k lines) + `mcp_server.py` (thin wrapper) + `tests/` (715 tests, 33 files). Resist package-split until real friction shows up.

Entry points (both installed via `uv tool install --editable .`):
- `rag` — CLI for indexing, querying, chat, productivity, automation
- `obsidian-rag-mcp` — MCP server (`rag_query`, `rag_read_note`, `rag_list_notes`, `rag_links`, `rag_stats`)

Fully local: ChromaDB + Ollama + sentence-transformers. No cloud calls.

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
rag eval                                   # queries.yaml → hit@k, MRR, recall@k
rag tune [--samples 500] [--apply]         # auto-calibrate ranker weights
rag log [-n 20] [--low-confidence]
rag dashboard [--days 30]                  # analytics: scores, latency, topics, PageRank

# Maintenance
rag maintenance [--dry-run --skip-reindex --skip-logs --json]  # all-in-one housekeeping

# Automation
rag setup [--remove]                       # install/remove 7 launchd services

# Tests
.venv/bin/python -m pytest tests/ -q
```

Python 3.13, `uv`. Runtime venv: `.venv/bin/python`. Global tool: `~/.local/share/uv/tools/obsidian-rag/`.

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
      + w.recency_cue    * recency_raw   [if has_recency_cue]
      + w.recency_always * recency_raw   [always]
      + w.tag_literal    * n_tag_matches
      + w.graph_pagerank * (pr/max_pr)   [wikilink authority signal]
      + w.feedback_pos                   [if path in feedback+ cosine≥0.80]
      - w.feedback_neg                   [if path in feedback- cosine≥0.80]
```

Weights in `~/.local/share/obsidian-rag/ranker.json` (written by `rag tune --apply`). Defaults `recency_always=0, tag_literal=0` preserve pre-tune behavior.

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

## Eval baselines

**Floor (2026-04-16, post-quick-wins)** — measured after `rag tune` confirmed no better weights vs current ranker.json
- Singles: `hit@5 90.48% · MRR 0.786 · n=21`
- Chains: `hit@5 76.00% · MRR 0.580 · chain_success 55.56% · turns=25 chains=9`

The prior floor (`95.24/0.802` singles, `72.00/0.557/44.44` chains from 2026-04-15, see `docs/eval-tune-2026-04-15.md`) drifted on the singles side due to vault content changes (queries.yaml golden is fixed but the vault gains/modifies notes daily). Chains improved +4pp hit and +11pp chain_success over the same window, confirming the fixed ranker weights are not the bottleneck — drift is natural, not regression.

Never claim improvement without re-running `rag eval`. Helper LLM calls (`expand_queries`, `reformulate_query`, `_judge_sufficiency`) are already deterministic via `HELPER_OPTIONS = {temperature: 0, seed: 42}`.

**HyDE with qwen2.5:3b drops singles hit@5 ~5pp**. HyDE is opt-in (`--hyde`); re-measure if helper model changes.

## On-disk state (`~/.local/share/obsidian-rag/`)

- `chroma/` — ChromaDB collections (per-vault suffixed)
- `queries.jsonl` — query log (q, variants, paths, scores, timings, mode, cmd)
- `feedback.jsonl` / `feedback_golden.json` (cache, rebuilt lazy on mtime gap)
- `ranker.json` — tuned weights. Delete = reset to hardcoded defaults
- `sessions/*.json` + `last_session` — multi-turn state
- `contradictions.jsonl` — radar sidecar
- `ambient.json` / `ambient.jsonl` / `ambient_state.jsonl` — config + log + dedup
- `filing_batches/*.jsonl` — audit log (prefix `archive-*` for archiver)
- `ignored_notes.json`, `tune.jsonl`
- `*.{log,error.log}` — launchd service logs

**Reset learned state**: `rm ranker.json feedback_golden.json`. Dimension mismatch in feedback golden → silently ignored. Full re-embed: `rag index --reset`.

## Vault path

Default: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes`. Override: `OBSIDIAN_RAG_VAULT` env var. Collections namespaced per vault (sha256[:8]).

Claude Code memory (`~/.claude/projects/-Users-fer/memory/`) is symlinked into vault at `04-Archive/99-obsidian-system/99-Claude/memory/`.
