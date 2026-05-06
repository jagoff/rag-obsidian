# Web chat features

Tool-calling, scoping (Feature H), inline reminders (Feature K), sessions, episodic memory, Quick Wins. Resumen + invariantes en [`CLAUDE.md`](../CLAUDE.md).

## Web chat tool-calling

[`web/tools.py`](../web/tools.py) — 9 tools: `search_vault`, `read_note`, `reminders_due`, `gmail_recent`, `finance_summary`, `calendar_ahead`, `weather` (read-only) + `propose_reminder`, `propose_calendar_event` (create-intent, impl en `rag/__init__.py`).

`/api/chat` corre 2-phase loop: pre-router (`_detect_tool_intent`, keyword → forced read tool) + optional LLM tool-decide round (gated por `RAG_WEB_TOOL_LLM_DECIDE`, default OFF). Create intent ("recordame", "creá evento") detectado por `_detect_propose_intent` (shared web + CLI) FORZA LLM decide round.

Create tools auto-crean si datetime es unambiguo (SSE `created` event → chip `╌ ✓ agregado...`, reminders con inline `deshacer` link via `DELETE /api/reminders/{id}`, eventos NO porque Calendar.app AppleScript delete es unreliable). Si parser flagea `needs_clarification` → `proposal` card con ✓ Crear / ✗ Descartar.

Helpers en `rag/__init__.py`: `_parse_natural_datetime` (dateparser + qwen2.5:3b fallback + `_preprocess_rioplatense_datetime`), `_parse_natural_recurrence`, `_create_reminder` (due_dt/priority/notes/recurrence), `_create_calendar_event` (Calendar.app AppleScript), `_has_explicit_time` (auto all-day), `_delete_reminder`, `_delete_calendar_event`.

**Rioplatense datetime normalization** (`_preprocess_rioplatense_datetime`): regex rewrites pre-dateparser. `18hs` → `18:00`, `al mediodía` → `12:00`, `X que viene` → bare weekday/`next week`, `el|este|próximo <weekday>` → bare English (dateparser 1.4 rechaza `next <weekday>` pero acepta bare con `PREFER_DATES_FROM=future`), `pasado mañana` → `day after tomorrow`, `tipo N`/`a eso de las N` → `N:00`, `el finde` → `saturday`. Anchor-echo guard post-dateparser: si input tenía time marker pero output = anchor time, fall through a LLM.

**CLI chat create-intent** (`rag chat`): mismo `_detect_propose_intent` ported a terminal via `_handle_chat_create_intent`. Single-round ollama tool-decide con `_CHAT_CREATE_OVERRIDE` prompt + tools=[propose_reminder, propose_calendar_event] only. command-r `{parameters: {...}}` arg wrapping unwrappeado. Returns `(handled, created_info)` con `{kind, reminder_id, title}` (None para events). Stash en `last_created` + `/undo` slash command dispatcha `_delete_reminder(reminder_id)`. Tests: [`tests/test_chat_create_handler.py`](../tests/test_chat_create_handler.py) (8 cases) + [`tests/test_chat_undo.py`](../tests/test_chat_undo.py) (5 cases).

## Sessions

JSON per session en `sessions/<id>.json`. TTL 30d, cap 50 turns, history window 6. IDs validados `^[A-Za-z0-9_.:-]{1,64}$`; invalid → mint fresh. WhatsApp pasa `wa:<jid>`.

## Quick Win #5 — Selective history summarisation

`_summarize_conversation_history`, 2026-05-04: cuando `len(history) > 2` AND `RAG_HISTORY_SUMMARY != "0"`, `/api/chat` comprime turnos previos (N-1) via qwen2.5:3b en 2-3 sentences; último turno keep verbatim. Cache `rag_conversation_summaries(session_id, history_hash, summary, ts)` retention 30d. Silent-fail a raw concat. Tests: [`tests/test_history_summary.py`](../tests/test_history_summary.py).

## Quick Win #4 — LLM typo correction

`_correct_typos_llm`, 2026-05-04: pre-embed step en `expand_queries()`. qwen2.5:3b con HELPER_OPTIONS deterministic. LRU 256, sanity check len>1.5× → preserve original. Telemetry: `llm_typo_*` en `rag_queries.extra_json`. Tests: [`tests/test_typo_correction.py`](../tests/test_typo_correction.py).

Default ON con Ollama / OFF con MLX (resolved en `_resolve_typo_correction_default()`). Bug encontrado 2026-05-05: bajo MLX backend qwen2.5:3b parafrasea agresivamente (`charla→chatea`, `fantastical→fantastic`, introduce typos). Drop singles 54.72% Ollama → 5.66% MLX cuando typo corrector ON. Override `RAG_TYPO_CORRECTION=1` siempre gana.

Dos sanity checks: (1) `len(corrected) > 1.5 * len(query)` → reject; (2) **`RAG_TYPO_JACCARD_MIN=0.7`** token-set Jaccard accent-insensitive **solo para multi-token** → reject paraphrases (`charla con juan`→`chatea con juan` jaccard=0.5; `whatsapp con mama`→`whatsapp com mama` PT-leak jaccard=0.5). 1-token queries saltan Jaccard porque `asor`→`Astor` daría jaccard=0 incluso siendo typo fix válido — ahí solo rige el length cap. Silent-fail.

## Quick Win #1 — Anaphora resolver

`RAG_ANAPHORA_RESOLVER` default ON (2026-05-04). Detector regex (microsegundos): True cuando `len(history) >= 1` AND query <8 tokens OR empieza con conector. Resolver qwen2.5:3b con LRU 128. Clamp si helper devuelve >3× input. Telemetría: `anaphora_*` en `rag_queries.extra_json`.

## Episodic memory ([`web/conversation_writer.py`](../web/conversation_writer.py), silent write)

Post `/api/chat` `done` event, daemon thread append a `04-Archive/99-obsidian-system/99-AI/conversations/YYYY-MM-DD-HHMM-<slug>.md`. One note per `session_id`, multi-turn. Index `session_id → relative_path` en `rag_conversations_index` (SQL upsert). Atomic .md write via `os.replace`. **Excluido del index** (`is_excluded`: prefix `04-Archive/99-obsidian-system/` + legacy `00-Inbox/conversations/` + `04-Archive/conversations/`) — leak hallucinations back si indexado.

**Shutdown drain** (`_CONV_WRITERS` + `@app.on_event("shutdown")`): cada writer in-flight registra. `_drain_conversation_writers` joins con 5s budget. Stragglers → `_CONV_PENDING_PATH` (`conversation_turn_pending.jsonl`) → re-aplicado en próximo startup por `_retry_pending_conversation_turns`. Threads daemon=True por design (wedged write no debe bloquear exit).

**Phase 2 consolidation** ([`scripts/consolidate_conversations.py`](../scripts/consolidate_conversations.py), `rag consolidate`, weekly launchd): scan rolling window (default 14d), embed cada conv como `first_question + first_answer` via bge-m3, group por connected components cosine ≥0.75, promote clusters ≥3 a PARA. Target folder: ≥2 matches `_PROJECT_PATTERNS` → `01-Projects/`, else `03-Resources/`. Synthesis via `resolve_chat_model()` + CHAT_OPTIONS (un call por cluster ~6s). Originals move a `04-Archive/conversations/YYYY-MM/`. Errores per-cluster swallowed.

## Feature H — Chat scoped a nota/folder

Selector compacto en composer del chat web ([`web/static/index.html`](../web/static/index.html) + [`app.js`](../web/static/app.js)): botón ◉ → popover con autocomplete → click setea scope en `sessionStorage` + chip "🎯 Limitado a: `<path>` ×". JS monkey-patchea `fetch` para inyectar `path` o `folder` solo en POST `/api/chat`.

Backend ([`web/server.py`](../web/server.py) `# ── Feature H`): `ChatRequest` acepta `folder` + `path` (validators rechazan URI schemes y `..`). `multi_retrieve(...)` recibe `folder` como 4to posicional. Si viene `path`, filtro post-retrieve exact-match contra `meta.file`. Short-circuit cuando no hay matches: SSE `sources(confidence=0)` + canned token + `done(scope_no_match=True)`. NO 404.

Endpoint nuevo `GET /api/notes/autocomplete?q=&limit=20`: substring matching case-insensitive contra `meta.file`/`meta.note`/`meta.folder`. Sortea exact → startswith → contains-path → contains-title → folder. Limit clamped 50. Empty corpus → `{items: [], reason: "empty_index"}`. Rate-limit reusa `_BEHAVIOR_BUCKETS` (120 req/60s).

Telemetry: `result["filters_applied"]["path_scope"]` distingue scope explícito de auto-filter. Bucket nuevo: `web.chat.scope_no_match`. Tests: [`tests/test_chat_scoped.py`](../tests/test_chat_scoped.py) (11 cases).

## Feature K — "Recordame X" inline en chat

Detecta comandos tipo "recordame llamar a Juan mañana 9am" → crea reminder Apple Reminders **sin LLM**, devuelve SSE `created` event en <100ms vs 5-15s del flow LLM+tools.

Detector ([`rag/__init__.py`](../rag/__init__.py) `# ══ Feature K`): `parse_remind_intent(text) → dict | None`. Pattern strict-leading: `^(recordame|recuerdame|acordate|hacéme acordar|reminder|remember me|remind me) [de/que] <rest>$`. Sobre `<rest>`, primer marker temporal con `_REMIND_TIME_MARKERS_RE` parte título/cuándo. Reusa `_parse_natural_datetime`. Anchor-echo guard.

Wire-up ([`web/server.py`](../web/server.py) `# ══ Feature K`): ANTES del flow normal, llama `parse_remind_intent(question)`. Match → `_create_reminder(title, due_dt=...)` directo + SSE `sources(confidence=1, intent=remind_inline)` + `created(kind=reminder, remind_inline=True)` + canned token + `done(mode=remind_inline)`. Si `_create_reminder` falla → `proposal(needs_clarification=True)`. NO match → fall-through al flow normal (donde `propose_reminder` tool sigue funcionando).

Telemetry bucket: `web.chat.remind_inline`. Tests: [`tests/test_chat_remind_inline.py`](../tests/test_chat_remind_inline.py) (10 cases).
