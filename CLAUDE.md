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

# URL finder (semantic-by-context, no LLM)
rag links "documentación de X"     # ranked URLs from the vault, OSC 8 clickable
rag links "X" --open 1             # open rank-1 URL in default browser
rag links --rebuild                # backfill the URL sub-index from existing notes

# Wikilink density (graph builder, no LLM)
rag wikilinks suggest                       # dry-run for whole vault
rag wikilinks suggest --folder X --apply    # write [[wikilinks]] for unlinked title mentions

# Daily productivity
rag dupes [--threshold 0.85] [--folder X]   # pairs of notes with similar centroids
rag inbox [--apply]                         # triage 00-Inbox: folder + tags + wikilinks + dupes
rag prep "tema o persona" [--save]          # context brief from vault, optionally saved to 00-Inbox/
rag capture "texto" [--tag X --source Y]    # quick note to 00-Inbox/ (auto-indexed)
rag read <url> [--save --plain]             # ingest external article → 00-Inbox/ w/ auto-wikilinks
rag morning [--dry-run]                      # daily brief → 05-Reviews/YYYY-MM-DD.md
rag today [--dry-run]                        # end-of-day closure → 05-Reviews/YYYY-MM-DD-evening.md
rag followup [--days 30 --status stale|activo|resolved --json]   # open-loop scanner
rag dead [--min-age-days 365]                # candidates to archive: 0 edges + not retrieved + old

# Automation (launchd)
rag setup            # install com.fer.obsidian-rag-{watch,digest,morning,today} launchd services
rag setup --remove   # uninstall all

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

### Automation (launchd)

Four services keep the RAG running without manual rituals — installed by `rag setup`, removed by `rag setup --remove`. All idempotent on reinstall (unload then load). Logs land in `~/.local/share/obsidian-rag/{watch,digest,morning,today}.{log,error.log}`.

- **`com.fer.obsidian-rag-watch`** — runs `rag watch` continuously (`RunAtLoad + KeepAlive`). Re-indexes on every vault save (debounce 3s). Removes the manual "did I `rag index` after that batch of edits?" friction.
- **`com.fer.obsidian-rag-digest`** — runs `rag digest` every Sunday 22:00 local. Generates `05-Reviews/YYYY-WNN.md` automatically. Honours `NO_COLOR=1` and `TERM=dumb` so logs stay readable.
- **`com.fer.obsidian-rag-morning`** — runs `rag morning` every weekday (Mon-Fri) at 7:00 local. Generates `05-Reviews/YYYY-MM-DD.md`. Reads the vault + queries.jsonl + contradictions.jsonl, drafts a 120-280-word brief with command-r: 1-line recap of yesterday, 3 focus items for today, pending triage if any, contradictions/gaps if any. Silently skips when there's zero evidence.
- **`com.fer.obsidian-rag-today`** — runs `rag today` every weekday (Mon-Fri) at 22:00 local. Generates `05-Reviews/YYYY-MM-DD-evening.md`. Cierre simétrico al morning: mira hacia atrás el día que terminó y prepara hand-off a mañana. Silent no-op cuando no hay evidencia.

Plist generation uses absolute paths to the `rag` binary resolved at install time (`_rag_binary()` checks `~/.local/bin/rag`, `/usr/local/bin/rag`, `/opt/homebrew/bin/rag`, then `shutil.which`). The launchd PATH includes Homebrew + uv tool dirs so subprocess hops (e.g., `ollama`) resolve too.

Auto-backfill: `find_urls()` calls `_maybe_backfill_urls()` once per process — if the URL collection is empty but the main collection isn't, the URL sub-index rebuilds itself silently (~1 min). No more "did I run `rag links --rebuild` after upgrading?" tax.

### Ambient Agent — co-autor reactivo del Inbox

Hook en `_index_single_file` que dispara sobre saves en `00-Inbox/` cuando el hash cambió. **Composición pura de primitivas existentes, sin LLM extra** (el hook cuesta ~0 además del indexing ya pagado). Activable por usuario vía `/enable_ambient` en el bot de WhatsApp (`~/whatsapp-listener/listener.ts`).

- **Auto-aplica**: `find_wikilink_suggestions` + `apply_wikilink_suggestions`. Regex determinística (el suggester ya es conservador: skip ambigüos, short titles, self-links).
- **Notifica vía WhatsApp**: near-duplicates (cosine ≥0.85), related notes (graph/tags), wikilinks aplicados. Mensaje compacto con `[[wikilinks]]` para que sea clickeable en Obsidian.
- **Silent por default**: si no hay findings interesantes, no manda nada (evita ruido por cada save).
- **Skip rules**:
  - Fuera de `00-Inbox/` → no-op.
  - Sin config (no se hizo `/enable_ambient`) → no-op.
  - Frontmatter `ambient: skip` → opt-out por nota.
  - `type: morning-brief | weekly-digest | prep` → skip (system-generated).
  - Dedup 5min: `{path, hash}` analizado recientemente → skip (evita doble ping si el usuario guarda dos veces).

**Config** en `~/.local/share/obsidian-rag/ambient.json` con `{jid, enabled}`. Escrito por el listener (`/enable_ambient` toma el JID del grupo del config del listener — no hay token porque la entrega va por el bridge local). Leído por rag.py vía `_ambient_config()`.

**Contract con el bridge**: rag.py SOLO lee la config y POST al `whatsapp-bridge` local (`http://localhost:8080/api/send`, urllib) con `{recipient: jid, message}`. No depende del listener estando up — si el listener muere el análisis sigue y queda en `ambient.jsonl`; si el bridge muere se pierde el ping. Desacoplado.

**Backwards-compat**: configs viejas con `{chat_id, bot_token}` y log lines con `telegram_sent` quedan inertes — el lector de config requiere `jid` y el sender escribe `whatsapp_sent`. Borrar `~/.local/share/obsidian-rag/ambient.json` y re-`/enable_ambient` desde WhatsApp si veías ambient corriendo bajo Telegram.

**CLI admin**: `rag ambient status | disable | test [path] | log [-n N]`. El comando `test` dispara el hook contra una nota existente para debugging sin tener que guardar en Obsidian.

**Storage**:
- `ambient.jsonl` → log append-only de eventos con wikilinks/dupes/related counts + `whatsapp_sent` bool (líneas pre-migración llevan `telegram_sent`).
- `ambient_state.jsonl` → dedup state (path + hash + timestamp), scan últimas 500 líneas para decidir skip.

Tests: `tests/test_ambient.py` — 18 casos cubriendo config gate, frontmatter opt-out, dedup window, auto-apply wikilinks, whatsapp send stubbed (el `_ambient_whatsapp_send` se monkeypatchea para no tocar el bridge).

### Capture / morning / dead-notes trilogy

**`rag capture "<text>"`** — atomic building block for quick capture. Writes `<vault>/00-Inbox/YYYY-MM-DD-HHMM-<slug>.md` with frontmatter `{type: capture, tags: [capture, ...], source?}`. Supports `--stdin` (voice transcripts piped in), `--tag` (repeatable), `--source`, `--title`, `--plain`. Auto-indexes so the capture is retrievable immediately. Used by the user directly AND by the WhatsApp listener's `/note` command + voice-message capture path (`~/whatsapp-listener/listener.ts`).

**`rag morning [--dry-run]`** — proactive daily brief. Collects, for the last 36h: notes modified (excluding `05-Reviews/`), current `00-Inbox/` contents, frontmatter `todo:`/`due:` signals, new entries in `contradictions.jsonl`, low-confidence queries (`top_score ≤ CONFIDENCE_RERANK_MIN`). command-r drafts 120-280 words in 1ra persona with a fixed Markdown structure (`## 📬 Ayer en una línea` / `## 🎯 Foco sugerido para hoy` / `## 🗂 Pendientes que asoman` / `## ⚠ Atender`). Writes to `05-Reviews/YYYY-MM-DD.md` and auto-indexes. Auto-fires via launchd Mon-Fri 7:00. Silent no-op when evidence is zero.

**`rag today [--dry-run]`** — end-of-day closure, simétrico a `morning`. Ventana: `[hoy 00:00, ahora)`. Recolecta notas modificadas hoy (excluyendo `05-Reviews/` y `00-Inbox/`), capturas del Inbox de hoy con flag `[sin-tags]` para las sin taxonomía, `todo:`/`due:` tocados hoy, entradas nuevas en `contradictions.jsonl`, queries low-confidence de hoy. command-r draftea 150-250 palabras en 1ra persona mirando hacia atrás, con 4 secciones fijas: `## 🪞 Lo que pasó hoy` / `## 📥 Sin procesar` / `## 🔍 Preguntas abiertas` / `## 🌅 Para mañana`. Escribe a `05-Reviews/YYYY-MM-DD-evening.md` (sufijo `-evening` para no colisionar con morning), auto-indexa. Auto-dispara via launchd Mon-Fri 22:00 (`com.fer.obsidian-rag-today`). Silent no-op cuando no hay evidencia. El output alimenta organicamente el `morning` del día siguiente — cierra el loop diario.

**`rag followup [--days 30]`** — scanner de loops abiertos. Extrae 3 tipos de "promesas" de notas modificadas en ventana `--days`:
1. Frontmatter `todo:` / `due:` (str/list/YAML date, normalizado por `_coerce_loop_items`)
2. Checkboxes `- [ ]` sin marcar
3. Imperativos inline: regex `tengo que / pendiente / revisar / explorar / profundizar / chequear`

Por cada loop: `retrieve(loop_text, k=5, multi_query=False)` filtrando a notas *posteriores* al extraction date (no self), gate en `rerank_score ≥ 0.03`, delegado a juez LLM cheap (qwen2.5:3b, temp=0 seed=42, JSON strict, sesgo conservador). Clasifica: `resuelto` (juez dijo yes), `stale` (no resuelto + >14 días), `activo` (no resuelto + ≤14 días). Output grouped (stale → activo → resolved summary) con OSC 8 hyperlinks. Flags: `--status`, `--json`, `--plain`, `--stale-days`. Reusa `retrieve()`, frontmatter parser, `_note_created_ts` (frontmatter-first, mtime fallback). Un embed + un call LLM cheap por loop — bounded cost.

**`rag read <url>`** — ingesta externa al vault con auto-linking al grafo personal. Pipeline:
1. `urllib.request` con UA custom + timeout 20s → readability por regex (strip `<script>/<style>/<nav>/<header>/<footer>/<aside>/<form>/svg/iframe` + HTML entities + collapse whitespace, cap 16k chars). Gate: texto extraído < 500 chars → error, no escribe.
2. command-r summariza en 150-300 palabras, 1ra persona, mismo voice que `morning`/`prep`. Extrae `<title>` para slug.
3. **Two-pass related lookup**: seed-embed del texto crudo → `_find_related_by_embedding` top-5 → feed de titles a command-r → refine con embed del summary. El LLM teje `[[wikilinks]]` inline al draftear (le pasamos la lista de titles relacionados en el prompt).
4. Tags de vocab existente via `_suggest_tags_for_note` — NEVER invents.
5. Frontmatter `{type: read, source, title, tags: [read, ...], related: [[[N1]], ...], created}` → escribe a `00-Inbox/YYYY-MM-DD-HHMM-read-<slug>.md` → `_index_single_file(skip_contradict=True)`.

Flags: dry-run default (fetch+summarize+preview, no write), `--save` (escribe + indexa), `--plain` (bot-friendly). Logs a `queries.jsonl` como `cmd: read`. Integración con WhatsApp listener (`~/whatsapp-listener/listener.ts`): mensaje que contiene SOLO URL(s) → auto-dispara `ragRead`; URL dentro de pregunta → cae al RAG normal.

**`rag dead`** — dead-note detector. Criterion AND: 0 outlinks + 0 backlinks + not retrieved in `--query-window-days` + age > `--min-age-days` + outside `00-Inbox/04-Archive/05-Reviews`. **Age source**: frontmatter `created:` parsed via ISO/strftime fallbacks (preferred — iCloud sync constantly bumps mtime), else mtime. Set `use_frontmatter_date=False` to force mtime. Pure Python; logs the run to `queries.jsonl` as `cmd: "dead"`. Never moves or deletes — only surfaces candidates.

### Daily-productivity layer

Three commands that compose the lower-level primitives into routines that have an obvious "moment" in the user's day.

**`rag dupes`** — surfaces pairs of notes with high centroid cosine similarity (mean of chunk embeddings). Pure local: numpy pairwise over the main collection (`_note_centroids` builds the matrix, `find_duplicate_notes` runs `arr @ arr.T` and masks the upper triangle). No LLM. Real-vault baseline: 521 notes finishes in <1s. Threshold 0.85 default; song-draft iterations and old-notes refactors surface immediately. The lower-threshold cousin `find_near_duplicates_for(col, path)` is what the inbox triage uses to flag "this incoming note may already exist."

**`rag inbox`** — triage every note in `00-Inbox/` (configurable). Per note, composes:
- `_suggest_folder_for_note` — semantic neighbours of the note's body, mode of their folders (excluding `00-` Inbox-style prefixes so we recommend a real home).
- `_suggest_tags_for_note` — the helper LLM picks tags from the existing vault vocabulary (extracted from `rag autotag`'s logic into a shared helper). NEVER invents new tags.
- `find_wikilink_suggestions` — graph densification on the incoming note.
- `find_near_duplicates_for` — flags possible duplicates above 0.85 cosine.

Without `--apply` it prints the plan; with `--apply` it moves the file (when folder confidence ≥ `--folder-min-conf`, default 0.4), rewrites the frontmatter `tags:` block via `_apply_frontmatter_tags` (preserves the rest of the YAML), applies the wikilinks, and re-indexes the new path. Skipped: confidence below threshold = no move; existing destination = no overwrite. The triage helpers are individually testable; tests cover folder/tag suggestion, frontmatter rewrite (including malformed YAML rejection), and end-to-end composition.

**`rag prep`** — context brief for an upcoming meeting / topic / project. Pipeline:
1. `retrieve(topic, k=8)` — main pool.
2. `find_related(top_metas)` — graph neighbours.
3. `find_urls(topic, k=5)` — bookmarks/refs scoped to the topic.
4. command-r drafts a Markdown brief in 1ra persona with fixed sections (Resumen ejecutivo / Background / Threads abiertos / Preguntas para explorar / URLs y fuentes), 350-550 words, citing notes with `[[wikilinks]]`.

`--save` writes the brief to `00-Inbox/YYYY-MM-DD-prep-<slug>.md` with frontmatter `{type: prep, topic, tags: [prep], sources: [[[note], ...]]}` and re-indexes so the brief itself becomes part of the corpus. v1 is vault-only; Gmail/Calendar integration is out of scope until the Python side gets Google API credentials.

### Wikilink density (`rag wikilinks suggest`)

The Obsidian graph is the system's foundation: `find_related`, the contradiction radar's neighbour pool, and chain reformulation all benefit from denser wikilinks. But manually adding `[[Title]]` is friction users skip. This command densifies the graph in a single sweep — no LLM, just a regex scan against the corpus' `title_to_paths` index.

For each note: walk the body, look for word-boundary matches of any *other* note's title, propose wrapping with `[[ ]]`. Skips frontmatter, fenced + inline code, existing wikilinks, markdown links, and HTML tags via `_wikilink_skip_spans`. Skips ambiguous titles (same string maps to multiple paths). Skips short titles by default (`--min-len 4`) — "TDD"/"AI"/"X" hit too widely. Only the first occurrence per title per note is proposed (one wikilink per page is enough for graph purposes).

CLI:
- `rag wikilinks suggest` — dry-run, all notes. `--folder X` or `--note <path>` to scope. `--show N` controls per-note display.
- `rag wikilinks suggest --apply` — actually wraps with `[[ ]]`. Each modified note is re-indexed via `_index_single_file(skip_contradict=True)` so retrieval picks up the new outlinks immediately.

`apply_wikilink_suggestions` iterates suggestions from highest char-offset to lowest so earlier offsets stay valid; defensive re-check at each offset silently skips stale suggestions if the file changed mid-flight.

Real-vault baseline: 132 suggestions across 98/521 notes — ~25% of the vault gets denser in one apply.

Tests: `tests/test_wikilinks.py` — 23 cases (skip-mask building, finder behavior incl. self-link suppression / one-per-title / ambiguous-title skip / short-title filter / word-boundary, `apply_wikilink_suggestions` round-trip + stale-offset defense).

### URL finder (`rag links`, `rag_links` MCP tool)

URLs are second-class citizens in the main pipeline — bge-m3 embeds whole chunks, the reranker scores topical relevance, and the LLM tends to paraphrase or omit literal URLs in prose. The URL sub-index fixes this for "where is the link to X" type questions.

At index time `_index_urls()` extracts every URL from each note (`extract_urls`: markdown `[anchor](url)` first, then bare `https?://...`, deduped per file). Image/media URLs (`png|jpg|svg|webp|mp4|pdf|...`) are filtered — they're noise for "find the doc". Each URL row in the `obsidian_urls_v1` Chroma collection embeds the **prose around the URL** (±240 chars), not the URL string itself, plus metadata `{file, note, folder, tags, url, anchor, line}`.

`find_urls(query, k, folder, tag)` embeds the query, semantic-matches against contexts, reranks with the cross-encoder, dedups by URL, and caps at 2 per source file (`PER_FILE_CAP`) so one note doesn't dominate. Bypasses the chat LLM entirely — output is the literal URLs with rich-link-styled OSC 8 hyperlinks (Cmd/Ctrl-click opens in browser).

Surfaces:
- `rag links "<query>"` — CLI. Flags: `-k`, `--folder`, `--tag`, `--open N` (opens rank N via macOS `open`), `--plain`, `--rebuild`.
- `rag chat` natural-language intent (`detect_link_intent`) — patterns like "donde está el link a X", "dame la url de Y", "documentación de Z" route to `find_urls` instead of the LLM. Tight regex; explicitly does NOT match generic "qué dice X sobre Y" queries.
- MCP `rag_links(query, k, folder, tag)` — for Claude Code to fetch URLs from the vault directly.

`--rebuild` re-extracts URLs from every note without re-embedding chunks (no LLM, ~1 min for 521 notes). Use after upgrades that change extraction logic, or to backfill into a vault that was indexed before this feature landed.

Tests: `tests/test_urls.py` — 30 cases covering extraction edge cases (markdown vs bare, dedup, line numbers, multiple per line, image filtering, trailing punctuation), `find_urls` behavior (empty, dedup-by-URL, k cap, folder filter), `_index_urls` idempotent replace, and `detect_link_intent` matches/non-matches.

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

Multi-turn state shared across `rag chat`, `rag query`, the MCP server, and the WhatsApp listener — follow-ups like "profundizá" or "y el otro" resolve against prior turns.

Storage: one JSON per session at `~/.local/share/obsidian-rag/sessions/<id>.json`, plus a `last_session` pointer for `--continue` / `--resume`. Schema: `{id, created_at, updated_at, mode, turns: [{ts, q, q_reformulated?, a, paths, top_score}]}`. Writes are atomic (tmp + replace). TTL 30 days (`SESSION_TTL_DAYS`), cap 50 turns per session (`SESSION_MAX_TURNS`, oldest dropped), history window 6 messages (`SESSION_HISTORY_WINDOW`) fed to the helper.

Ids are opaque strings; callers can supply their own — the WhatsApp listener passes `wa:<jid>` so each chat keeps its thread. Validated against `SESSION_ID_RE = ^[A-Za-z0-9_.:-]{1,64}$`; an invalid id silently mints a fresh one rather than raising. Pre-migration sessions stored under `tg:<chat_id>` keys remain readable (TTL 30d will sweep them).

Surfaces:
- `rag chat --session <id> | --resume`
- `rag query --session <id> | --continue` (plus `--plain` for bot-friendly output)
- MCP `rag_query(session_id=...)`
- WhatsApp `/rag <query>` in the listener passes `wa:<jid>` via `--session`

Semantics: when `session_history()` returns anything, the helper rewrites the incoming query (`reformulate_query`) to absorb antecedents before retrieval — orthogonal to `--precise`. The rewrite is stored as `q_reformulated` on the turn.

Admin: `rag session list | show <id> | clear <id> | cleanup` (cleanup drops files older than TTL by mtime).

Tests: `tests/test_sessions.py` covers the module end-to-end — monkeypatches `SESSIONS_DIR` / `LAST_SESSION_FILE` to `tmp_path`. Run with `.venv/bin/python -m pytest` (pytest is in `[project.optional-dependencies].dev`).

## Vault path

Defaults to `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes` (iCloud). Override at runtime with the `OBSIDIAN_RAG_VAULT` env var — collections are namespaced per resolved vault path (sha256[:8] suffix on `_COLLECTION_BASE`) so switching doesn't pollute the index. Fresh vault = fresh collection automatically; just `rag index` after pointing at it.

The `memory` directory Claude Code reads at `~/.claude/projects/-Users-fer/memory/` is a symlink into the vault (`04-Archive/99-obsidian-system/99-Claude/memory/`). Renaming that folder in the vault requires re-pointing the symlink.
