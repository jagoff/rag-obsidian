# Retrieval internals

Detalle del pipeline de recuperación, indexing, model stack, scoring, prompts y cross-source corpus. Resumen + invariantes en [`CLAUDE.md`](../CLAUDE.md).

## Retrieval pipeline (`retrieve()`)

```
query → typo correct → anaphora resolve → classify_intent → infer_filters [auto]
      → [adaptive routing: skip reformulate si metadata-only intent]
      → [decomposition gate: regex + LLM fallback → N sub-retrieves + RRF (k=60) si multi-aspecto]
      → expand_queries (3 paraphrases, ONE qwen2.5:3b call)
      → embed(variants) batched bge-m3
      → per variant: sqlite-vec sem + BM25 (accent-normalised, GIL-serialised)
      → RRF merge → dedup → expand to parent section
      → cross-encoder rerank (bge-reranker-v2-m3, MPS+fp32)
      → [LLM judge: si top_score < 0.5 AND len ≥5 AND RAG_LLM_JUDGE=1, qwen2.5:3b score 0-10 → blend α·ce + (1-α)·llm]
      → score loop (recency multiplier per source/intent + behavior priors + contradiction penalty + feedback golden)
      → [MMR diversification gate: si RAG_MMR=1 y no counter]
      → [contradiction penalty post-rerank: demote chunks en rag_contradictions]
      → [seen_titles soft penalty: -0.1 si meta.note coincide]
      → graph expansion (1-hop wikilink, top-3, 3 neighbors max)
      → [auto-deep: si confidence < 0.10, iterative sub-query, 3 iters max, 30s wall-time cap]
      → top-k → LLM (streamed)
      → citation-repair [si bad citations + score<threshold + n_bad ≤ 2]
      → [NLI grounding: si RAG_NLI_GROUNDING=1, skip count/list/recent/agenda]
      → [citation NLI verifier: si RAG_NLI_MODE != off, mark/strip por sentence]

Intent dispatch: semantic | synthesis | comparison | count | list | recent | agenda | entity_lookup
```

**Graph expansion**: always on, 1-hop wikilink neighbors, top-3 → 3 neighbors max marked `[nota relacionada (grafo)]`.

**Auto-deep**: cuando `top_score < CONFIDENCE_DEEP_THRESHOLD=0.10`, helper judge sufficiency → sub-query → segundo retrieve → merge. Max 3 iters + 30s wall-time. Disable: `--no-deep`.

**Rerank pool** (`RERANK_POOL_MAX = 25`, bumpeado de 15 el 2026-04-25): el set golden creció a n=60 con queries cross-source y pool=15 expulsaba candidatos correctos. Historia: 30→15 (2026-04-21) — pool=15 dominó vs 30: hit@5 idéntico, MRR chains +5pp, P95 singles -66%. Path `retrieve_only` usa `RERANK_POOL_RETRIEVE_ONLY=10`. Web `/api/chat` no-list-intent usa `RAG_WEB_RERANK_POOL=3` (default).

**Corpus cache** (`_load_corpus`): BM25 + vocab built once, invalidated by `col.count()` delta. Cold 341ms → warm 2ms.

**Cache locks** (concurrency invariants para writers desde múltiples threads):

| Cache | Lock |
|---|---|
| `_context_cache` | `_context_cache_lock` (Lock) |
| `_synthetic_q_cache` | `_synthetic_q_cache_lock` (Lock) |
| `_mentions_cache` | `_mentions_cache_lock` (Lock) |
| `_embed_cache` | `_embed_cache_lock` (Lock) |
| `_corpus_cache` + `_pagerank_cache*` | `_corpus_cache_lock` (RLock) |
| `_contacts_cache` | `_contacts_cache_lock` (Lock) |

LLM calls corren **outside** del lock para no serializar concurrent requests. Tests: `tests/test_cache_concurrency.py`.

## Indexing

Chunks 150-800 chars, split on headers + blank lines, merged si <MIN_CHUNK. Hash per file → re-embed only on change. `is_excluded()` skips `.`-prefixed segments.

**Contextual embeddings** (v9 actual): `get_context_summary()` genera 1-2 sentences per note via qwen2.5:3b, prepended a cada chunk's `embed_text` como `Contexto: ...`. Cached por file hash.

**`created_ts` backfill marker**: persistido en `rag_schema_version` (sentinel `_created_ts_backfill_complete`). Pre-fix re-escaneaba 3600+ chunks por restart del web daemon (149 restarts en ~3 días).

**Schema changes**: bump `_COLLECTION_BASE` (currently `obsidian_notes_v11`). Per-vault suffix = sha256[:8] of resolved path.

## Model stack

| Role | Model | Notes |
|---|---|---|
| Chat | `resolve_chat_model()`: qwen2.5:7b > qwen3:30b-a3b > command-r > qwen2.5:14b > phi4 | qwen2.5:7b default tras bench 2026-04-18 (P50 5.9s vs 37s command-r). |
| Helper | `qwen2.5:3b` | paraphrase/HyDE/reformulation; deterministic via `HELPER_OPTIONS = {temperature: 0, seed: 42}` |
| Embed | `bge-m3` | 1024-dim multilingual |
| Reranker | `BAAI/bge-reranker-v2-m3` | `device="mps"` + `float32` forced. **NO switch fp16** — 2 A/Bs failed (collapse 2026-04-13, overhead 2x con calidad equivalente 2026-04-22). |
| NLI grounding (opt-in) | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | ~400 MB MPS fp32, idle-unload via `RAG_NLI_IDLE_TTL`. |
| Citation NLI verifier (opt-in) | `cross-encoder/nli-deberta-v3-small` | ~80 MB, lazy-load sticky-fail. |

`CHAT_OPTIONS`: `num_ctx=4096, num_predict=384`. Don't bump unless prompts grow.

## Confidence gate

`top_score < 0.015` (CONFIDENCE_RERANK_MIN) + no `--force` → refuse sin LLM call. Per-source override scaffolding: `CONFIDENCE_RERANK_MIN_PER_SOURCE` dict (todos =baseline 0.015 hoy) + helper `confidence_threshold_for_source(source)`. Re-calibrate cuando ingesters tengan ≥1 semana de feedback.

## Generation prompts

- `SYSTEM_RULES_STRICT` (default `rag query` semantic): forbids external prose.
- `SYSTEM_RULES` (`--loose`, always en chat): allows `<<ext>>...<</ext>>` rendered dim yellow + ⚠.
- `SYSTEM_RULES_LOOKUP` (intent count/list/recent/agenda): terse 1-2 sentences, exact "No encontré esto en el vault." refusal.
- `SYSTEM_RULES_SYNTHESIS` (intent synthesis): cross-reference ≥2 sources, surface tension. Fires via `_INTENT_SYNTHESIS_RE`.
- `SYSTEM_RULES_COMPARISON` (intent comparison): `X dice A / Y dice B / Diferencia clave`. Fires via `_INTENT_COMPARISON_RE`. Checked BEFORE synthesis.
- Routed via `system_prompt_for_intent(intent, loose)`.

## Agenda intent

Fired by `_INTENT_AGENDA_RE`, checked **before `recent`** (compartían tokens temporales). `handle_agenda(col, params, limit=20, *, question=None)` filtra por `source ∈ _AGENDA_SOURCES = {"calendar", "reminders"}`, sort por `created_ts` desc.

**Window filter** `_parse_agenda_window(question, *, now=None) → (ts_start, ts_end) | None`: dispatch order narrowest first (day anchors → weekend → week → month → year → weekday-specific). Half-open [start, end). Snap a 00:00 local.

## Prompt-injection defence (passive)

Dos layers en `rag/__init__.py` (sobre `SYSTEM_RULES`):

- **Redaction** `_redact_sensitive(text)` — strip OTPs/tokens/passwords/CBU/cards antes de chunk → LLM. Cue-gated (value next to `code|token|password|cbu|cvv`) con digit-presence lookahead.
- **Context isolation** `_format_chunk_for_llm(doc, meta, role)` — wrappea body en `<<<CHUNK>>>...<<<END_CHUNK>>>`. Paired with `_CHUNK_AS_DATA_RULE` (REGLA 0) en cada `SYSTEM_RULES*`.

NOT a barrier vs motivated attacker con vault write access — hint a la classifier.

## Name-preservation guardrail

`_NAME_PRESERVATION_RULE` (después de `_CHUNK_AS_DATA_RULE`): bloquea LLM "corrigiendo" proper nouns que no reconoce. Regression seed: "Bizarrap" → "Bizarra". Verify: `python -c "import rag; print(rag._NAME_PRESERVATION_RULE[:80])"`.

## Response-quality post-pipeline

- **Citation-repair** (always-on): `verify_citations(full, metas)` flags invented paths. ONE repair call si non-empty + n_bad ≤ 2. Logs `citation_repaired: bool`.
- **`--critique` flag** (opt-in): segundo non-streaming chat-model call evalúa + regenera. Logs `critique_fired/changed`.
- **Citation NLI verifier** (`RAG_NLI_MODE`): sentence-level entailment via `cross-encoder/nli-deberta-v3-small`. Modes: off | mark (`(?)` suffix) | strip. Implementado en `rag/postprocess.py`.

## Scoring formula (post-rerank)

```
score = rerank_logit
      + w.recency_cue          * recency_raw  [if has_recency_cue]
      + w.recency_always       * recency_raw  [always]
      + w.tag_literal          * n_tag_matches
      + w.graph_pagerank       * (pr/max_pr)
      + w.click_prior          * ctr_path
      + w.click_prior_folder   * ctr_folder
      + w.click_prior_hour     * ctr_path_hour
      + w.click_prior_dayofweek* ctr_path_weekday
      + w.dwell_score          * log1p(dwell_s)
      - w.contradiction_penalty* log1p(n_contrad_ts)  [90d window]
      + w.feedback_pos                          [if path en feedback+ cosine≥0.80]
      - w.feedback_neg                          [if path en feedback- cosine≥0.80]
```

Weights en `~/.local/share/obsidian-rag/ranker.json`. Defaults `recency_always=0, tag_literal=0, click_*=0, dwell_score=0, contradiction_penalty=0` preservan pre-tune behavior.

## GC#2.C — Reranker fine-tune (gated on data)

Infra completa + gate E2E validado, esperando ≥20 rows con `corrective_path` en `rag_feedback`. Runs anteriores fallaron (-3.3pp chains hit@5) por señal positiva ruidosa. Fix: [`scripts/finetune_reranker.py`](../scripts/finetune_reranker.py) lee `corrective_path` del `extra_json` y lo usa como único positivo; fallback a todos paths cuando no.

Generar data: `rag chat` thumbs-down con prompt path correcto, `rag feedback backfill`, `rag feedback harvest`, `rag feedback infer-implicit` (rama opens 600s window, rama paráfrasis fallback). Status: `rag feedback status`. Re-trigger: `python scripts/finetune_reranker.py --epochs 2`.

## Cross-source corpus (Phase 1)

`retrieve()` source-aware. Collection `obsidian_notes_v11`, legacy rows sin `source` → `"vault"` via `normalize_source()`. Constants en `rag/__init__.py`: `VALID_SOURCES` (frozenset 11), `SOURCE_WEIGHTS` (vault 1.00 → WA 0.75), `SOURCE_RECENCY_HALFLIFE_DAYS` (None vault/calendar, 30d WA/messages, 90d reminders, 180d gmail), `SOURCE_RETENTION_DAYS` (None vault/calendar/reminders, 180 WA/messages, 365 gmail).

Helpers: `normalize_source(v, default="vault")`, `source_weight(src)`, `source_recency_multiplier(src, created_ts, now)`, `source_recency_multiplier_with_intent(src, ts, intent, *, now)`.

**Filter**: `--source S[,S2]` o kwarg `source` en retrieve/deep_retrieve/multi_retrieve. Unknown sources → error.

**Conversational dedup** (`_conv_dedup_window`): collapse WA/messages chunks misma `chat_jid` ±30min, keep highest-scored.

### Ingesters

Cada ingester: chunk-per-record (parent=body), state table cursor + content hash diffing, idempotent upsert, `--reset --dry-run --since ISO --json` flags. Tests dedicados en `tests/test_ingest_*.py`.

| Source | Cursor | Fuente | Notes |
|---|---|---|---|
| **whatsapp** (Phase 1.a) | `rag_whatsapp_state(chat_jid)` | `~/repositories/whatsapp-mcp/whatsapp-bridge/store/messages.db` (immutable RO) | Conversational chunking 5min/800char/speaker-change. doc_id `whatsapp://{jid}/{first_msg}::{idx}`. **Excluye RagNet (`WHATSAPP_BOT_JID`) + content U+200B prefix** (defense in depth, 2026-04-28). |
| **calendar** (Phase 1.b) | `rag_calendar_state(calendar_id, sync_token)` | Google Calendar OAuth (`~/.calendar-mcp/`) | Window `[now-2y, now+180d]`, `singleEvents=True`. Hardcoded exclude addressbook+holidays. doc_id `calendar://<calendar_id>/<event_id>`. |
| **gmail** (Phase 1.c) | `rag_gmail_state(history_id)` | Gmail OAuth (`~/.gmail-mcp/`) | Thread-level chunking. Strip quotes + signatures. Bootstrap `q=newer_than:365d`. |
| **reminders** (Phase 1.d) | `rag_reminders_state(reminder_id, content_hash)` | AppleScript local | Re-fetch full catalogue, upsert changed/new, delete stale. Field separator chr(31). |
| **contacts** (Phase 1.e) | `rag_contacts_state(contact_uid, content_hash)` | `~/Library/Application Support/AddressBook/Sources/*/AddressBook-v22.abcddb` (SQLite directo) | Two-pass phone index: pass 1 dedupe canonical, pass 2 fan out suffix keys (full/last-10/last-8/last-7), drop cross-UID collisions. Used by ingest_calls + futuros iMessage/WA enrichment. doc_id `contacts://<UID>::0`. |
| **calls** (Phase 1.f) | `rag_calls_state(call_uid, content_hash)` | `~/Library/Application Support/CallHistoryDB/CallHistory.storedata` | Enriched via `resolve_phone()`. Headlines BM25-friendly: "Llamada perdida de Juli". Retention 180d, halflife 30d. doc_id `calls://<UID>::0`. |
| **safari** (Phase 2) | `rag_safari_history_state(history_item_id)` + `rag_safari_bookmark_state(bookmark_uuid)` | `~/Library/Safari/History.db` + `Bookmarks.plist` (incl. ReadingList) | Aggregate por URL no por visita. doc_ids: `safari://history/<id>::0`, `safari://bm/<uuid>::0`, `safari://rl/<uuid>::0`. Source weight 0.80, halflife 90d. SQLite contention con web running: retry. |
| **drive** | `rag_drive_state` | Google Drive OAuth (`~/.gdrive-mcp/`) | Search DAO + shared docs. |
| **pillow** | `rag_sleep_sessions` | `~/Library/Mobile Documents/com~apple~CloudDocs/Sueño/PillowData.txt` | Local-only, NO al corpus vectorial. CLI: `rag sleep show/patterns/ingest`. |

Phase 1.g (apagar workaround `/note` `/ob` del WA listener) + Phase 2 OCR adjuntos: deferred hasta que ingesters cumplan ≥1 semana en prod con data activa.

## Contextual Retrieval prototype (gated, default OFF)

[Anthropic technique](https://www.anthropic.com/news/contextual-retrieval): qwen2.5:3b genera summary corto (≤100 tokens) que ubica el chunk en su documento, prepended al embed_text antes de embed. Módulo: [`rag/contextual_retrieval.py`](../rag/contextual_retrieval.py).

Wire-up en `_index_single_file` + `_run_index`. CLI: `rag index --contextual` setea env por invocación. Cache `rag_chunk_contexts` PK `(doc_id, chunk_idx, chunk_hash)`, sobrevive `--reset`. Display_texts NO se mutan.

Promote checklist: (1) `RAG_CONTEXTUAL_RETRIEVAL=1 rag index --reset --contextual`, (2) `rag eval` con CI bootstrap, (3) si singles+chains hit@5 mejoran fuera del CI noise → promote default ON. Diferente de `get_context_summary` (per-doc summary compartido a TODOS los chunks de la nota); este es per-chunk.
