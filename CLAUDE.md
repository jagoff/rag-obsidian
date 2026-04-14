# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Local RAG over an Obsidian vault. Two CLI entry points — both installed via `uv tool install --editable .` — plus an MCP server:

- `rag` — indexing, querying, chat, watch, eval, log
- `obsidian-rag-mcp` — exposes `rag_query`, `rag_read_note`, `rag_list_notes`, `rag_stats` to Claude Code over stdio

Fully local: ChromaDB + Ollama + sentence-transformers. No cloud calls.

## Commands

```bash
# Dev install (or reinstall after pyproject.toml / code changes)
uv tool install --reinstall --editable .

# Index ops
rag index              # incremental (hash-based). --reset rebuilds. --no-contradict to skip phase-2 check
rag watch              # watchdog auto-reindex on vault changes (debounce 3s)

# Query paths
rag query "text"       # one-shot; flags: --hyde --no-multi --raw --loose --force
                       #             --counter --session <id> --continue --plain
rag chat               # interactive; /save and /reindex (or NL equivalents) work mid-conversation
                       # flags: --counter --session <id> --resume
rag stats              # models + index status

# Sessions (multi-turn state)
rag session list | show <id> | clear <id> | cleanup [--days N]

# Agent mode + retrospective
rag do "instrucción"   # tool-calling agent: search/read/list + propose_write with confirm
rag digest             # weekly narrative review of vault evolution. --week YYYY-WNN, --days N, --dry-run

# Quality + observability
rag eval               # run queries.yaml → hit@k, MRR, recall@k (singles + chains)
rag log [-n 20] [--low-confidence]

# Tests
.venv/bin/python -m pytest tests/ -q   # pytest in [project.optional-dependencies].dev
.venv/bin/python -m pytest tests/test_sessions.py::test_round_trip   # single test

# Run the MCP server manually (Claude Code launches it on demand)
obsidian-rag-mcp
```

Python 3.13, `uv` for deps. Runtime: `.venv/bin/python` is the local venv; the global tool lives at `~/.local/share/uv/tools/obsidian-rag/`.

## Architecture

Everything is in `rag.py` (~3500 lines) + `mcp_server.py` (thin wrapper) + `tests/`. Single-file by design — no framework abstractions between the caller and the pipeline. The size grew with knowledge graph, agent loop, sessions, contradiction radar, digest — the call graph stays flat and readable; resist the urge to package-split until a real friction shows up.

### Retrieval pipeline (`retrieve()` in rag.py)

```
query
 → classify_intent()            # count / list / recent / semantic — non-semantic short-circuits to metadata scan
 → infer_filters() [auto]       # conservative: explicit #tag or folder leaf ≥5 chars
 → expand_queries() via helper  # 3 paraphrases, ONE ollama call to qwen2.5:3b
 → embed(variants)              # batched in one bge-m3 call
 → per variant: ChromaDB sem + BM25 (accent-normalised, GIL-serialised — do NOT parallelise)
 → RRF merge → dedup union
 → expand to parent section     # pre-computed at index time, O(1) metadata lookup
 → cross-encoder rerank         # bge-reranker-v2-m3 on MPS+fp16, scored on parent-expanded text
 → top-k → LLM (streamed)
```

Corpus cache (`_load_corpus`) builds BM25 + vocabulary once per process, invalidated by `col.count()` delta. Cold→warm: 341ms → 2ms. Do not touch without re-measuring.

### Indexing (`_index_single_file`, `index` command)

Chunks 150–800 chars, split on headers + blank lines, merged if < MIN_CHUNK. Each chunk carries:
- `embed_text` (prefix `[folder | title | area=... | #tags]` + chunk body) — what bge-m3 sees
- `display_text` — raw chunk, what the reranker/LLM sees
- `parent` metadata — enclosing section between `^#` headers, capped at 1200 chars

Hash per file triggers re-embed only when content changes. Orphan cleanup on full index. `is_excluded()` skips any path whose segment starts with `.` (.trash, .obsidian, .claude, .git…).

Breaking schema changes must bump `_COLLECTION_BASE` (currently `obsidian_notes_v7`) so `get_or_create_collection` produces a fresh space; old collections become orphans and can be deleted. Per-vault suffix is appended automatically (sha256[:8] of `VAULT_PATH`) when running against a non-default vault, so multi-vault setups don't share an index.

### Model stack (two tiers, with fallback resolver)

- `CHAT_MODEL_PREFERENCE = ("command-r:latest", "qwen2.5:14b", "phi4:latest")` — `resolve_chat_model()` picks the first installed. Command-r is RAG-trained and citation-native; the two fallbacks are only for dev on machines without the 20GB pull.
- `HELPER_MODEL = "qwen2.5:3b"` — paraphrase/HyDE/history reformulation. Fast, cheap, runs everywhere.
- `EMBED_MODEL = "bge-m3"` — multilingual 1024-dim.
- `RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"` — `get_reranker()` forces `device="mps"` + `torch_dtype=float16` on Apple Silicon. The sentence-transformers auto-detect falls back to CPU in uv venvs and costs ~3×. Do not remove the explicit device.

All `ollama.chat/embed` calls pass `keep_alive=OLLAMA_KEEP_ALIVE` (default `-1`) to keep models resident in VRAM. `CHAT_OPTIONS` sets `num_ctx=4096, num_predict=768` — generation window is sized to the real prompt (≈2.5k tokens) plus a capped answer; don't bump unless prompts grow.

### Generation prompts

Two system prompts: `SYSTEM_RULES_STRICT` (default for `rag query`) forbids external prose entirely. `SYSTEM_RULES` (opt-in `--loose`) allows prose wrapped in `<<ext>>...<</ext>>` which the renderer highlights in dim yellow with a ⚠ icon. Chat always uses `SYSTEM_RULES` because conversational follow-ups sometimes need a little non-literal glue.

### Confidence gate

After rerank, if `top_score < CONFIDENCE_RERANK_MIN` (0.015) and `--force` is absent, the query refuses without calling the LLM. The threshold is calibrated for bge-reranker-v2-m3 on this corpus: irrelevant queries hit ~0.005–0.015, legit queries ≥0.02. Re-calibrate if the reranker model changes.

### Rendering (`render_response`)

Two link formats recognised and both styled with OSC 8 `file://` hyperlinks (Ctrl+Click opens in Obsidian):
- `[Label](path.md)` → label bold cyan, path cyan dim
- `[path.md]` (command-r's default) → path bold magenta

`NOTE_LINK_RE` accepts single-level balanced parens inside the path so `Explorando (otras)/…md` renders correctly. `verify_citations()` checks both formats against the retrieved metas and flags unknown paths after the response.

### In-chat intents (`detect_save_intent`, `detect_reindex_intent`)

`rag chat` parses two intents off the user's input before treating it as a question. Reindex is checked first so "reindexá las notas" doesn't trip the save heuristic via the word "notas".

- **Save**: `/save [título]` or NL saves. Strong save verbs (`guardá|salvá|agendá`) trigger alone; neutral verbs (`creá|agregá|añadí|escribí|armá|generá`) require "nota/notas" within ~5 tokens. Saved notes land in `00-Inbox/` with frontmatter carrying union-of-source tags, `related:` wikilinks to retrieved notes, and the original `source_query`. The new note is auto-indexed so it's searchable immediately.
- **Reindex**: `/reindex [reset]` or NL ("reindexá", "actualizá el vault", "reescaneá las notas", "refrescá el índice"). Strong verbs (reindex/reescan/refresc) trigger alone; weak `actualiz` requires an object (vault/índice/notas/todo). Reset markers (`desde cero|completo|reset|rebuild|scratch`) flip to full rebuild. Calls `_run_index()` — the same helper the CLI command wraps — so behaviour matches `rag index` exactly.

### Agent mode (`rag do`)

`rag do "instrucción"` runs a tool-calling loop with command-r. Tools exposed: `_agent_tool_search` (calls `retrieve()`), `_agent_tool_read_note`, `_agent_tool_list_notes`, `_agent_tool_propose_write`. Writes are NOT applied during the loop — they accumulate in `_AGENT_PENDING_WRITES` and the user confirms each at the end (skip with `--yes`, cap iterations with `--max-iterations`, default 8). No delete/move tools by design — first version is conservative.

### Contradiction Radar

Three-phase feature that turns the vault into an interlocutor instead of an archive: it surfaces tensions between notes instead of just retrieving agreement.

- **Phase 1 — query-time counter-evidence** (`find_contradictions`, `render_contradictions`). Opt-in `--counter` on `rag query` and `rag chat`. After the LLM answer, embed it, pull nearest chunks, drop already-cited paths, rerank vs the original question, ask the chat model for real contradictions only (strict JSON prompt, conservative bias). Renders a "⚡ Counter-evidence" block under the answer; persists `contradictions: [{path, why}]` to the query log + session turn.
- **Phase 2 — index-time flag** (`find_contradictions_for_note`, `_check_and_flag_contradictions`). On every new/modified note in incremental indexing, the same detector runs against the rest of the vault. Hits land in **two sinks**: `contradicts: [path, ...]` in the note's YAML frontmatter (so Obsidian/dataview can surface it inline) AND a sidecar log at `~/.local/share/obsidian-rag/contradictions.jsonl`. Skipped on `--reset` (would be O(n²)); also skipped if `note_body < 200 chars` or via `--no-contradict`.
- **Phase 3 — weekly narrative digest** (`rag digest`). Reads recent notes (vault mtime), `contradicts:` frontmatter, the sidecar log, the query log's `contradictions` field, and low-confidence queries. command-r drafts a first-person review (~400-600 words, with `[[wikilinks]]`) into `05-Reviews/YYYY-WNN.md`, auto-indexed.

**Detector model — empirical**: both detectors use `resolve_chat_model()` (command-r), NOT the helper. qwen2.5:3b proved non-deterministic at temp=0 seed=42 on this corpus (same case yields FP first run, empty next) and emits malformed JSON often. command-r hugs source text and returns parseable output. Cost ~5-10s per check — bounded by guardrails (opt-in for query; only changed notes for index). The pattern of "use the helper for cheap rewrites, escalate to chat model for judgment" is worth preserving.

Tests: `tests/test_contradictions.py` (16 cases — fixtures monkeypatch `embed`/`get_reranker`/`ollama.chat` so model-agnostic), `tests/test_digest.py` (11 cases). `tests/contradiction_cases.yaml` is a golden set ready for a future eval harness when the detector prompt gets tuned.

## Eval harness (`rag eval` + `queries.yaml`)

`queries.yaml` is the golden set. Two axes:
- **singles**: 21 queries across RAG/coaching/música/tech, mixing easy keyword matches with harder cases (accents stripped, typos, content-about queries, metaphorical). Baseline on v7: `hit@5 90.48% · MRR 0.786 · recall@5 90.48%`.
- **chains**: 6 multi-turn chains (16 turns total) exercising follow-ups with pronouns/demonstratives — each turn after the first is reformulated via `reformulate_query` against the running history. Baseline on v7: `hit@5 75.00% · MRR 0.656 · recall@5 75.00% · chain_success 50.00%`.

The v6→v7 drop in singles (95.24 → 90.48) is the schema bump (outlinks + re-chunking), not a retrieval regression — confirmed by `--no-multi` showing the same numbers. Use these baselines to measure any change to chunking, prompts, models, or retrieval — don't ship blind.

Empirical finding that informed defaults: **HyDE with qwen2.5:3b drops hit@5 from 95 → 90%**. Small models drift the hypothetical from real note phrasing. HyDE is opt-in (`--hyde`); re-measure if the helper model changes size class.

## Observability

`rag query` and `rag chat` log to `~/.local/share/obsidian-rag/queries.jsonl` — one line per turn with `q`, `variants`, `paths`, `scores`, `top_score`, `t_retrieve`, `t_gen`, `bad_citations`, `mode`. `rag log` tails it. Use `--low-confidence` to surface queries the reranker wasn't sure about — those are usually prompts to write a new note in the vault.

## Conversational sessions

Multi-turn state shared across `rag chat`, `rag query`, the MCP server, and the Telegram bots — follow-ups like "profundizá" or "y el otro" resolve against prior turns.

Storage: one JSON per session at `~/.local/share/obsidian-rag/sessions/<id>.json`, plus a `last_session` pointer for `--continue` / `--resume`. Schema: `{id, created_at, updated_at, mode, turns: [{ts, q, q_reformulated?, a, paths, top_score}]}`. Writes are atomic (tmp + replace). TTL 30 days (`SESSION_TTL_DAYS`), cap 50 turns per session (`SESSION_MAX_TURNS`, oldest dropped), history window 6 messages (`SESSION_HISTORY_WINDOW`) fed to the helper.

Ids are opaque strings; callers can supply their own — the Telegram bots pass `tg:<chat_id>` so each chat keeps its thread. Validated against `SESSION_ID_RE = ^[A-Za-z0-9_.:-]{1,64}$`; an invalid id silently mints a fresh one rather than raising.

Surfaces:
- `rag chat --session <id> | --resume`
- `rag query --session <id> | --continue` (plus `--plain` for bot-friendly output)
- MCP `rag_query(session_id=...)`
- Telegram `/rag <query>` in both bots passes `tg:<chat_id>` via `--session`

Semantics: when `session_history()` returns anything, the helper rewrites the incoming query (`reformulate_query`) to absorb antecedents before retrieval — orthogonal to `--precise`. The rewrite is stored as `q_reformulated` on the turn.

Admin: `rag session list | show <id> | clear <id> | cleanup` (cleanup drops files older than TTL by mtime).

Tests: `tests/test_sessions.py` covers the module end-to-end — monkeypatches `SESSIONS_DIR` / `LAST_SESSION_FILE` to `tmp_path`. Run with `.venv/bin/python -m pytest` (pytest is in `[project.optional-dependencies].dev`).

## Vault path

Defaults to `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes` (iCloud). Override at runtime with the `OBSIDIAN_RAG_VAULT` env var — collections are namespaced per resolved vault path (sha256[:8] suffix on `_COLLECTION_BASE`) so switching doesn't pollute the index. Fresh vault = fresh collection automatically; just `rag index` after pointing at it.

The `memory` directory Claude Code reads at `~/.claude/projects/-Users-fer/memory/` is a symlink into the vault (`04-Archive/99-obsidian-system/99-Claude/memory/`). Renaming that folder in the vault requires re-pointing the symlink.
