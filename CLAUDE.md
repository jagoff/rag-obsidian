# CLAUDE.md

Local RAG over an Obsidian vault. Single-file: `rag.py` (~27k lines) + `mcp_server.py` (thin wrapper) + `web/` (FastAPI server, 5.4k lines + ~5.9k JS/HTML/CSS) + `tests/` (1,173 tests, 66 files). Resist package-split until real friction shows up.

Entry points (both installed via `uv tool install --editable .`):
- `rag` — CLI for indexing, querying, chat, productivity, automation
- `obsidian-rag-mcp` — MCP server (`rag_query`, `rag_read_note`, `rag_list_notes`, `rag_links`, `rag_stats`)

Fully local: Sqlite-vec + Ollama + sentence-transformers. No cloud calls.

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
- `OLLAMA_KEEP_ALIVE` — passed to every ollama chat/embed call. Code default `"20m"` (`rag.py:1114`); every launchd plist overrides to `-1` so models stay VRAM-resident for the daemon lifetime. Accepts int seconds or duration string.
- `RAG_STATE_SQL=1` — enables the SQL telemetry store (20 `rag_*` tables in `ragvec/ragvec.db`). Set on every launchd plist since the 2026-04-19 cutover. If unset → code falls back to legacy JSONL writes/reads. Will become default-on + JSONL stripped at T10 (~2026-04-26).
- `RAG_TRACK_OPENS=1` — switches OSC 8 link scheme from `file://` to `x-rag-open://` so CLI clicks route through `rag open` (ranker-vivo signal capture). Absent = no behavior change.
- `RAG_EXPLORE=1` — enable ε-exploration in `retrieve()` (10% chance to swap a top-3 result with a rank-4..7 candidate). Set on `morning`/`today` plists to generate counterfactuals. MUST be unset during `rag eval` — the command actively `os.environ.pop`s it and asserts, as a belt-and-suspenders guard.
- `RAG_RERANKER_IDLE_TTL` — seconds the cross-encoder stays resident before idle-unload (default 900).
- `RAG_RERANKER_NEVER_UNLOAD` — set to `1` in the web launchd plist to pin the reranker in MPS VRAM permanently; sweeper loop still runs but skips `maybe_unload_reranker()`. Eliminates the 9s cold-reload hit after idle eviction. Cost: ~2-3 GB unified memory pinned. Safe on 36 GB with command-r + qwen3:8b resident.
- `RAG_LOCAL_EMBED` — set to `1` in the web launchd plist to use in-process `SentenceTransformer("BAAI/bge-m3")` for query embedding instead of ollama HTTP (~10-30ms vs ~140ms). Requires BAAI/bge-m3 cached in `~/.cache/huggingface/hub/` — download once with `python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"` before enabling. Verify cosine >0.999 vs ollama embeddings of same text before enabling in production. Do NOT set for indexing/watch/ingest processes — bulk chunk embedding stays on ollama. Uses CLS pooling (same as ollama gguf).
- `OBSIDIAN_RAG_NO_APPLE=1` — disables Apple integrations (Calendar, Reminders, Mail, Screen Time) entirely. Useful on non-macOS hosts or when Full Disk Access is not granted.
- `OBSIDIAN_RAG_MOZE_FOLDER` — override MOZE ETL target folder inside the vault (default `02-Areas/Personal/Finanzas/MOZE`).
- `OBSIDIAN_RAG_WATCH_EXCLUDE_FOLDERS` — comma-separated vault-relative folders `rag watch` must ignore. Default `"03-Resources/WhatsApp"` (WA dumps re-fire the handler dozens of times per minute via periodic ETL; they're picked up by manual/periodic `rag index` instead).

Dev/debug toggles (not set in production):

- `RAG_DEBUG=1` — verbose stderr in the local embed path (`rag.py:6593`).
- `RAG_RETRIEVE_TIMING=1` — per-stage timing breakdown printed to stderr at the end of `retrieve()`.
- `RAG_NO_WARMUP=1` — skip the background reranker + bge-m3 + corpus warmup (shaves startup for lightweight commands like `rag stats`, `rag session list`; first query pays the cold-load cost).
- `OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY=1` — short-circuit `get_context_summary()` to empty string. Used by tests + emergency fallback if qwen2.5:3b is unavailable; leaves embeddings without contextual prefix.
- `OBSIDIAN_RAG_SKIP_SYNTHETIC_Q=1` — same, for `get_synthetic_questions()`.

## Architecture — invariants

### Retrieval pipeline (`retrieve()`)

```
query → classify_intent → infer_filters [auto]
      → expand_queries (3 paraphrases, ONE qwen2.5:3b call)
      → embed(variants) batched bge-m3
      → per variant: sqlite-vec sem + BM25 (accent-normalised, GIL-serialised — do NOT parallelise)
      → RRF merge → dedup → expand to parent section (O(1) metadata)
      → cross-encoder rerank (bge-reranker-v2-m3, MPS+fp32)
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
| Chat | `resolve_chat_model()`: qwen2.5:7b > qwen3:30b-a3b > command-r > qwen2.5:14b > phi4 | qwen2.5:7b default tras bench 2026-04-18 (total P50 5.9s vs 37s de command-r); fallbacks high-quality disponibles. |
| Helper | `qwen2.5:3b` | paraphrase/HyDE/reformulation |
| Embed | `bge-m3` | 1024-dim multilingual |
| Reranker | `BAAI/bge-reranker-v2-m3` | `device="mps"` + `float32` forced — do NOT switch to fp16 on MPS (score collapse to ~0.001, verified 2026-04-13); CPU fallback = 3× slower. |

All ollama calls use `keep_alive=OLLAMA_KEEP_ALIVE` — default `"20m"` in code (`rag.py:1114`), overridden to `-1` (forever) in every launchd plist so deployed models stay VRAM-resident. `CHAT_OPTIONS`: `num_ctx=4096, num_predict=768` — don't bump unless prompts grow.

**Pattern**: helper for cheap rewrites, chat model for judgment. Contradiction detector MUST use chat model (qwen2.5:3b proved non-deterministic + malformed JSON on this task).

### Confidence gate

`top_score < 0.015` (CONFIDENCE_RERANK_MIN) + no `--force` → refuse without LLM call. Calibrated for bge-reranker-v2-m3 on this corpus. Re-calibrate if reranker changes.

### Generation prompts

- `SYSTEM_RULES_STRICT` (default `rag query`, `semantic` intent): forbids external prose.
- `SYSTEM_RULES` (`--loose`, always in chat): allows `<<ext>>...<</ext>>` rendered dim yellow + ⚠.
- `SYSTEM_RULES_LOOKUP` (intent `count`/`list`/`recent`): terse 1-2 sentences, exact "No encontré esto en el vault." refusal.
- `SYSTEM_RULES_SYNTHESIS` (intent `synthesis`, extension point — not yet emitted by `classify_intent`): cross-reference ≥2 overlapping sources, must surface tension.
- `SYSTEM_RULES_COMPARISON` (intent `comparison`, extension point): explicit `X dice A / Y dice B / Diferencia clave: …` structure.
- Routed through `system_prompt_for_intent(intent, loose)` at generation time (both `query()` and `chat()` paths). `--loose` always maps to `SYSTEM_RULES` for every intent.

### Response-quality post-pipeline

**Citation-repair** (always-on): after generation, `verify_citations(full, metas)` flags invented paths. If non-empty, ONE repair call runs (`resolve_chat_model()` + `CHAT_OPTIONS`, non-streaming, `keep_alive=-1`) with system prompt `"Solo puedes citar las siguientes rutas: [...]. ... No inventes otras."` If repair output also has bad citations or is empty → keep original. On success → replace `full` silently (interactive: reprints via `render_response`; plain: single `click.echo` deferred until AFTER repair + critique). Logs `citation_repaired: bool` to `queries.jsonl`.

**`--critique` flag** (opt-in, both `rag query` and `rag chat`, plus `/critique` chat toggle): after citation-repair, second non-streaming chat-model call evaluates + regenerates if needed. Whitespace-normalized diff vs original → replace + `critique_changed=True`. Logs `critique_fired` (always equals flag state) + `critique_changed` to `queries.jsonl`. Adds one extra ollama round-trip only when flag is set — off by default so no latency cost.

### Rendering

OSC 8 `file://` hyperlinks for both `[Label](path.md)` and `[path.md]` formats. `NOTE_LINK_RE` handles single-level balanced parens. `verify_citations()` flags unknown paths post-response (feeds citation-repair loop).

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

**Episodic memory** (`web/conversation_writer.py`, silent write): after every `/api/chat` `done` event, `web/server.py` spawns a daemon thread that appends the turn to `00-Inbox/conversations/YYYY-MM-DD-HHMM-<slug>.md`. One note per `session_id`, multi-turn. Hand-rolled YAML frontmatter (`session_id`, `created`, `updated`, `turns`, `confidence_avg`, `sources`, `tags`). Session → relative path mapping lives in the `rag_conversations_index` SQL table (post-cutover; legacy sidecar `conversations_index.json` remains as read fallback until T10 strip). Atomic write via `os.replace` + `fcntl.flock`. Errors land on `LOG_PATH` as `conversation_turn_error` — never raised, never SSE-emitted. Indexed automatically by `com.fer.obsidian-rag-watch` (no exclusion rule matches), so conversations become retrievable within one debounce cycle (~3s). Do NOT edit these notes manually — system artifacts. Curate by moving to PARA.

**Web chat tool-calling** (`web/tools.py`, 9 tools): `search_vault`, `read_note`, `reminders_due`, `gmail_recent`, `finance_summary`, `calendar_ahead`, `weather` (read-only) + `propose_reminder`, `propose_calendar_event` (create-intent). `/api/chat` runs a 2-phase tool loop: pre-router (`_detect_tool_intent`, keyword → forced read tool) + optional LLM tool-decide round (gated by `RAG_WEB_TOOL_LLM_DECIDE`, default OFF). Create intent ("recordame", "creá un evento", ...) is detected by `_detect_propose_intent` which FORCES the LLM decide round ON for that query — propose tools need LLM arg extraction, can't run from pre-router. Create tools return JSON with `proposal_id` + `fields` + optional `needs_clarification`; `_maybe_emit_proposal` parses that output and fires a secondary SSE event `proposal` which the UI renders as an inline confirmation card with ✓ Crear / ✗ Descartar buttons. Nothing lands in Calendar.app / Reminders.app until the user clicks ✓ → POST to `/api/reminders/create` or `/api/calendar/create` with the parsed fields. Low-level helpers `_parse_natural_datetime` (dateparser + qwen2.5:3b fallback), `_parse_natural_recurrence` (regex over ES/EN patterns), `_create_reminder` (supports `due_dt`, `priority`, `notes`, `recurrence`), `_create_calendar_event` (via Calendar.app AppleScript — iCloud writable, unlike the JXA read path) all in `rag.py`. Recurrence on Reminders is best-effort (inner try/on error) since the property is macOS-version-dependent; on Calendar it's stable.

**Rioplatense datetime normalization** (`_preprocess_rioplatense_datetime`, runs before `dateparser` inside `_parse_natural_datetime`): dateparser 1.4 handles maybe 30% of AR-idiom inputs correctly and silently echoes the anchor time for another 30% (e.g. "a las 10 de la mañana" → anchor time). We hand-roll regex rewrites that normalize to forms dateparser CAN parse — mostly English equivalents with `PREFER_DATES_FROM=future`. Covers: `18hs` → `18:00`; `al mediodía` → `12:00`; `X que viene` → bare weekday/`next week`/`next month`; `el|este|próximo <weekday>` → bare English weekday (because dateparser 1.4 rejects `next <weekday>` silently but accepts bare `thursday` with future-prefer); `pasado mañana` → `day after tomorrow`; `a las N de la mañana|tarde|noche` → `N:00 am`/`(N+12):00`; `a la mañana|tarde|noche|tardecita` → default hour (09/16/20/17); `tipo N` / `a eso de las N` → `N:00` (rioplatense approximations); diminutives (`horitas` → `horas`); `el finde` → `saturday`. Anchor-echo guard after dateparser: if the input carries a time marker but dateparser returned exactly the anchor time, discard and fall through to LLM. LLM fallback prompt (qwen2.5:3b, `HELPER_OPTIONS` deterministic) explicitly flags rioplatense, passes both raw text and normalized hint, and instructs rollforward for bare weekdays + 09:00 default for missing times.

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

**Post prompt-per-intent + citation-repair (2026-04-19):** Singles `hit@5 88.10% [76.19, 97.62] · MRR 0.767 [0.643, 0.869]` — identical hit@5, MRR within CI. Chains `hit@5 81.82% [66.67, 93.94] · MRR 0.636 [0.505, 0.773] · chain_success 58.33% [33.33, 83.33]` — +3pp hit@5, +8pp chain_success, both inside prior CI so treat as noise until replicated. Floor unchanged for auto-rollback gate (still 76.19% / 63.64%).

**Post golden-set re-mapping (2026-04-20):** vault reorg (PARA moves: many notes `02-Areas/Coaching/*` → `03-Resources/Coaching/*`, `03-Resources/{Agile,Tech}/*` → `04-Archive/*`, etc.) left 33 of 65 `expected` paths in `queries.yaml` pointing at dead files, artificially cratering eval to singles hit@5 26% / chains 33%. Golden rebuilt by auto-mapping 31 unique paths via filename-stem lookup to the closest surviving note (prefer non-archive, bias `01→02→03→04` for tie-breaks) and dropping one chain whose source notes (`reference_{claude,ollama}_telegram_bot.md`) no longer exist. Post-rebuild eval: Singles `hit@5 78.57% [64.29, 90.48] · MRR 0.696 [0.554, 0.810]`; Chains `hit@5 75.76% [60.61, 90.91] · MRR 0.641 [0.510, 0.788]`. Both CIs overlap the 2026-04-19 run — within noise band. Floor unchanged (76.19% / 63.64%); current singles 78.57% and chains 75.76% pass the auto-rollback gate.

**Prior floor (2026-04-17, post-title-in-rerank, n=21 singles / 9 chains):** Singles `hit@5 90.48% · MRR 0.821`; Chains `hit@5 80.00% · MRR 0.627 · chain_success 55.56%`. Kept for historical trend, but do not compare new numbers against it without overlapping CIs.

**Even-earlier floor (2026-04-16, post-quick-wins, n=21/9):** Singles `hit@5 90.48% · MRR 0.786`; Chains `hit@5 76.00% · MRR 0.580 · chain_success 55.56%`.

The 2026-04-15 floor (`95.24/0.802` singles, `72.00/0.557/44.44` chains, see `docs/eval-tune-2026-04-15.md`) pre-dates both the expansion and the CI tooling — treat as a qualitative reference only.

Never claim improvement without re-running `rag eval`. Helper LLM calls (`expand_queries`, `reformulate_query`, `_judge_sufficiency`) are already deterministic via `HELPER_OPTIONS = {temperature: 0, seed: 42}`.

**HyDE with qwen2.5:3b drops singles hit@5 ~5pp**. HyDE is opt-in (`--hyde`); re-measure if helper model changes.

**`seen_titles` in `reformulate_query` regressed chains** (2026-04-17). Injecting "notas ya consultadas: [...]" into the helper prompt as a diversity nudge dropped chains hit@5 −16pp (80→64) and chain_success −33pp (55.56→22.22). The helper treats the list as "avoid these" and drifts off-topic. The kwarg remains on the signature (callers in eval pass it via `_titles_from_paths`) but is intentionally unused in the prompt. Future: try as a soft *reranker* hint (penalty on already-seen chunks post-rerank) instead of an LLM instruction.

## On-disk state (`~/.local/share/obsidian-rag/`)

### Telemetry — SQL tables (post-cutover 2026-04-19)

All operational telemetry + learning state now lives in `ragvec/ragvec.db` alongside the sqlite-vec meta/vec tables. 20 `rag_*` prefixed tables (no collision with `meta_*`/`vec_*`). Gated by `RAG_STATE_SQL=1`, set on every launchd plist at cutover. Legacy JSONL writes/reads still live in `rag.py` as fallback — T10 (~2026-04-26, +7d observation) strips them.

Log-style tables (`id INTEGER PK AUTOINCREMENT`, `ts TEXT` ISO-8601, indexed):
- `rag_queries` — query log (q, variants_json, paths_json, scores_json, top_score, t_retrieve, t_gen, cmd, session, mode, citation_repaired, critique_fired/changed, extra_json). Retention 90d via `rag maintenance`.
- `rag_behavior` — ranker-vivo events (source: cli/whatsapp/web/brief × event: open/kept/deleted/positive_implicit/negative_implicit/save). Retention 90d.
- `rag_feedback` — explicit +1/-1 + optional corrective_path (UNIQUE(turn_id,rating,ts)). Keep all.
- `rag_tune` — offline + online tune history (cmd, baseline/best_json, delta, eval_hit5_*, rolled_back). Keep all.
- `rag_contradictions` — radar Phase 2 (UNIQUE(ts,subject_path)). Keep all.
- `rag_ambient` / `rag_ambient_state` — ambient agent log (retention 60d) + dedup state (upsert by path).
- `rag_brief_written` — morning/today brief citation manifest (retention 60d).
- `rag_brief_state` — kept/deleted dedup (upsert by pair_key = hash(brief_type, kind, path)).
- `rag_wa_tasks`, `rag_archive_log`, `rag_filing_log`, `rag_eval_runs`, `rag_surface_log`, `rag_proactive_log` — 60d retention.
- `rag_cpu_metrics`, `rag_memory_metrics`, `system_memory_metrics` — per-minute samplers, 30d retention.

State-style tables:
- `rag_conversations_index` — episodic session_id → relative_path (web/conversation_writer.py upsert; replaces the old conversations_index.json + fcntl dance).
- `rag_feedback_golden` (pk=path,rating, `embedding BLOB` float32 little-endian, `source_ts`) + `rag_feedback_golden_meta` (k/v) — cache rebuilt when `rag_feedback.max(ts) > meta.last_built_source_ts`.

Primitives in `rag.py` (`# ── SQL state store (T1: foundation) ──` section):
- `_ensure_telemetry_tables(conn)` — idempotent DDL
- `_ragvec_state_conn()` — short-lived WAL conn with `synchronous=NORMAL` + `busy_timeout=10000`
- `_sql_append_event(conn, table, row)`, `_sql_upsert(conn, table, row, pk_cols)`, `_sql_query_window(conn, table, since_ts, ...)`, `_sql_max_ts(conn, table)`

Writer contract: if flag ON, single-row BEGIN/COMMIT into SQL; on exception, log to `sql_state_errors.jsonl` and fall through to JSONL (cutover fail-safe). Reader contract: if flag ON + SQL has data, return SQL; if SQL empty or raises, fall through to JSONL (historical data bridge).

Migration one-shot: `scripts/migrate_state_to_sqlite.py --source-dir ~/.local/share/obsidian-rag [--dry-run] [--round-trip-check] [--reverse] [--summary]`. Refuses to run while `com.fer.obsidian-rag-*` services are up (preflight `pgrep`; `--force` to override). Renames each source → `<name>.bak.<unix_ts>` on successful commit. Cutover of 2026-04-19 imported 7,946 records across 19 sources; 43 malformed pre-existing records dropped (missing NOT NULL fields).

Rollback escape hatch: `rag maintenance --rollback-state-migration [--force]`. Restores the newest `.bak.<ts>` per source, drops the 20 `rag_*` tables, WAL checkpoint + VACUUM. Refuses if services up + no bak files found. `rag maintenance` also prunes `.bak.*` older than 30d.

### Other state (unchanged; still on disk)

- `ranker.json` — tuned weights. Delete = reset to hardcoded defaults.
- `ranker.{unix_ts}.json` — 3 most recent backups, written on every `rag tune --apply`. Consumed by `rag tune --rollback` + auto-rollback CI gate.
- `sessions/*.json` + `last_session` — multi-turn state (TTL 30d, cap 50 turns).
- `ambient.json` — ambient agent config (jid, enabled, allowed_folders).
- `filing_batches/*.jsonl` — audit log (prefix `archive-*` for archiver).
- `ignored_notes.json`, `home_cache.json`, `context_summaries.json`, `auto_index_state.json`, `coach_state.json`, `synthetic_questions.json`, `wa_tasks_state.json` — app state + caches.
- `online-tune.{log,error.log}`, `*.{log,error.log}` — launchd service logs.
- `sql_state_errors.jsonl` — diagnostic sink for SQL-path write failures (only populated if the SQL branch raises and JSONL fallback fires).

**Reset learned state**: `rm ranker.json` + `DELETE FROM rag_feedback_golden; DELETE FROM rag_feedback_golden_meta;` inside ragvec.db. Full re-embed: `rag index --reset`.

## Vault path

Default: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes`. Override: `OBSIDIAN_RAG_VAULT` env var. Collections namespaced per vault (sha256[:8]).

Claude Code memory (`~/.claude/projects/-Users-fer/memory/`) is symlinked into vault at `04-Archive/99-obsidian-system/99-Claude/memory/`.
