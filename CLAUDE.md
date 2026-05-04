idian-rag/ranker.json` (written by `rag tune --apply`). Defaults `recency_always=0, tag_literal=0, click_*=0, dwell_score=0, contradiction_penalty=0` preserve pre-tune behavior. Behavior + contradiction knobs son inert until `rag_behavior` / `rag_contradictions` accumulate signal y `rag tune` finds non-zero weights.

Behavior priors (`_load_behavior_priors()`): read from `rag_behavior` (SQL), cached per MAX(ts). Positive events: `open`, `positive_implicit`, `save`, `kept`. Negative: `negative_implicit`, `deleted`. CTR usa Laplace smoothing `(clicks+1)/(impressions+10)`.

### GC#2.C — Reranker fine-tune (infra ready, gated on data)

- **Estado**: infra completa + gate E2E validado, esperando ≥20 rows con `corrective_path` en `rag_feedback` antes de re-correr con chances de promover.
- **Run anterior fallido** (`~/.cache/obsidian-rag/reranker-ft-20260422-124112/`, 2.1 GB, cleanup manual pendiente con `rm -rf` por ask-rule): −3.3pp chains hit@5 vs baseline. Causa: 1 epoch undertraining + señal positiva ruidosa (55 turns positivos × ~4 chunks cada uno, todos label=1.0 aunque sólo uno era golden).
- **Run 2 noisy** (`~/.cache/obsidian-rag/reranker-ft-20260422-182127/`, 2.1 GB, gate=0 override, 3 epochs, ver [`docs/finetune-run-2026-04-22.md`](docs/finetune-run-2026-04-22.md)): **mismo −3.3pp chains**. Loss convergió de 0.96 a 0.13 (overfitting claro en epoch 3); val margin +0.455 (pos 0.515 vs neg 0.060). Modelo aprendió muy bien la data ruidosa — por eso regresionó chains. **Gate E2E validado**: detectó regresión y NO promovió. Conclusión firme: sin `corrective_path` limpios, fine-tune no supera baseline con esta config — hace falta señal limpia, no más epochs.
- **Fix aplicado**: [`scripts/finetune_reranker.py`](scripts/finetune_reranker.py) ahora lee `corrective_path` de `rag_feedback.extra_json` y lo usa como único positivo cuando está presente. Fallback a todos los paths cuando no.
- **Gate pre-training**: `RAG_FINETUNE_MIN_CORRECTIVES` (default 20). Aborta con exit 5 si señal limpia es insuficiente.
- **Cómo generar data**:
  - `rag chat` + thumbs-down en turnos malos — prompt pide path correcto (commit `23f2899`). Web UI tiene mismo picker (commit `33ed3f0`).
  - `rag feedback backfill` — rescata corrective_path de turns ya en `rag_feedback` que no lo tienen (aplica a los 55 positivos del run 2026-04-22 que no recibieron prompt — commit 23f2899 es posterior). Muestra query + top-5 del turn, aceptás [1-5]/texto libre/skip/quit. Update in-place via `json_set()`, nunca duplica rows.
  - `rag feedback harvest` — equivalente CLI del skill `rag-feedback-harvester` de Claude Code. Lista queries recent low-confidence sin thumbs y pide [+N/-/c/s/q]. Tagged `source='harvester'` + `original_query_id` en `extra_json` para trazabilidad.
  - `rag feedback status` — progress hacia los 20 del gate + breakdown por bucket (pos_no_cp / neg_no_cp) + comando exacto para re-disparar fine-tune cuando gate está open.
  - `rag feedback infer-implicit [--window-seconds N --dry-run --json]` — derivador batch: para cada `rating=-1` que no tenga `corrective_path` todavía, prueba dos ramas (en orden): (1) **opens-based** — busca `open` en `rag_behavior` dentro del window, mismo `session_id`, path distinto al top-1 que ranker eligió → ESE es el corrective; `corrective_source = "implicit_behavior_inference"`. (2) **paráfrasis fallback** (post-2026-04-29) — si no hubo opens, busca en `rag_queries` follow-up query en misma session que sea paráfrasis del original (`requery_detection.is_paraphrase`) Y cuyo `top_score >= 0.5` Y top-1 distinto del original → ESE top-1 es el corrective; `corrective_source = "implicit_paraphrase_inference"`. **Window default 600s (10 min)** — pre-2026-04-29 era 60s, lo subimos porque con 60s cerrábamos sólo 1 corrective_path en 6 días (user lee nota abierta antes de actuar, no abre otra al toque). Idempotente: skipea feedbacks que ya tienen `corrective_path`. **Valores posibles de `corrective_source`**: `implicit_behavior_inference` (rama 1, señal fuerte), `implicit_paraphrase_inference` (rama 2, señal más débil pero útil para destrabar gate de 20 que dispara LoRA fine-tune del reranker). Tests: [`tests/test_implicit_learning_corrective.py`](tests/test_implicit_learning_corrective.py) (rama opens) + [`tests/test_corrective_paraphrase_fallback.py`](tests/test_corrective_paraphrase_fallback.py) (rama paráfrasis + backwards-compat).
- **Miner JSONL como data alternativa** (commit `5f33d44`): [`scripts/export_training_pairs.py`](scripts/export_training_pairs.py) complementa `rag_feedback` directo con signal implícita de `rag_behavior` (`copy`/`open`/`save`/`kept`/`positive_implicit`) + hard-negs mined de `impression` events reales del historial (no re-retrieve). Análisis del JSONL actual: [`docs/training-pairs-miner-analysis-2026-04-22.md`](docs/training-pairs-miner-analysis-2026-04-22.md) — 176 pairs, ratio neg:pos 6.1:1 (vs 2.4:1 del run previo), 74% con ≥5 hard-negs. Calidad superior al run noisy → próximo intento con estos pairs + `--epochs 2` tiene chances reales de pasar el eval gate. Integración al finetune pendiente (zona de abp2vvvw actual).
- **Monitoreo**: `sqlite3 ~/.local/share/obsidian-rag/ragvec/telemetry.db "SELECT COUNT(*) FROM rag_feedback WHERE json_extract(extra_json, '\$.corrective_path') IS NOT NULL AND json_extract(extra_json, '\$.corrective_path') <> ''"` — conteo directo de corrective_paths disponibles. (Post split 2026-04-21 `rag_feedback` vive en `telemetry.db`, no `ragvec.db`.)
- **Re-trigger**: `python scripts/finetune_reranker.py --epochs 2` una vez que gate lo permita (2 epochs, no 3 — loss convergió a 0.22 en epoch 2 en run noisy; epoch 3 es overfit puro). Gate de `rag eval` decide promoción via symlink `~/.cache/obsidian-rag/reranker-ft-current`.

## Key subsystems — contracts only

Subsystems tienen autodescriptive docstrings en `rag/__init__.py` y dedicated test files. Sólo contracts/invariants acá.

**Sessions**: JSON per session in `sessions/<id>.json`. TTL 30d, cap 50 turns, history window 6. IDs validated `^[A-Za-z0-9_.:-]{1,64}$`; invalid → mint fresh. WhatsApp passes `wa:<jid>`.

**Episodic memory** (`web/conversation_writer.py`, silent write): post each `/api/chat` `done` event, `web/server.py` spawnea daemon thread via `_spawn_conversation_writer` que appendea turn a `04-Archive/99-obsidian-system/99-AI/conversations/YYYY-MM-DD-HHMM-<slug>.md` (pre-2026-04-25: `00-Inbox/conversations/` — user pidió que "carpetas de sistema" vivan bajo `99-obsidian-system/99-AI/` para que su PARA quede limpio). Una nota por `session_id`, multi-turn. Hand-rolled YAML frontmatter (`session_id`, `created`, `updated`, `turns`, `confidence_avg`, `sources`, `tags`). Index session_id → relative_path vive en `rag_conversations_index` (SQL, upsert). Atomic .md write via `os.replace`; concurrent writes para mismo session no son escenario producción (un /api/chat por session a la vez) así que pre-T10 whole-body fcntl lock se fue — SQL upsert dentro `BEGIN IMMEDIATE` handles index serialisation. Errors landean en `LOG_PATH` como `conversation_turn_error` — never raised, never SSE-emitted. Raw conversations **excluidas del search index** (`is_excluded`: prefix general `04-Archive/99-obsidian-system/` + legacy `00-Inbox/conversations/` por compat + `04-Archive/conversations/` para archivados post-consolidación) — leak LLM hallucinations back as ground truth si indexed (T6 regression). Curation via `rag consolidate` (Phase 2, abajo), no manual editing.

**Conversation writer shutdown drain** (`_CONV_WRITERS` + `@app.on_event("shutdown")`): cada in-flight writer registra en `_CONV_WRITERS` y se remueve cuando `_persist_conversation_turn` returns (success or exception). On server stop `_drain_conversation_writers` hook joinea cada pending thread con combined 5s budget. Anything still running cae en exception path normal, landea en `_CONV_PENDING_PATH` (`conversation_turn_pending.jsonl`), y se re-aplica at next startup by `_retry_pending_conversation_turns`. Threads quedan `daemon=True` por diseño — wedged SQL/disk write no debe bloquear process exit. Stragglers past cap se loggean once como `conversation_writer_shutdown_timeout` to `LOG_PATH`. Tests: `tests/test_web_conv_shutdown.py` (6 cases covering self-remove, empty drain no-op, quick-writer wait, 5s cap with wedged writer, spawn tracking, exception-path release).

**Episodic memory — Phase 2 consolidation** (`scripts/consolidate_conversations.py`, `rag consolidate`, weekly launchd): scans `04-Archive/99-obsidian-system/99-AI/conversations/` en rolling window (default 14d), embeds cada como `first_question + first_answer` via bge-m3, agrupa by connected components on cosine ≥ 0.75, promueve clusters ≥ 3 a PARA. Target folder picked by regex over cluster bodies: ≥2 matches contra `_PROJECT_PATTERNS` (ES+EN action verbs / future-tense / dates) → `01-Projects/`, else `03-Resources/` (conservative default). Synthesis via `resolve_chat_model()` + `CHAT_OPTIONS` — una non-streaming call por cluster (~6s). Consolidated note gets frontmatter `type: consolidated-conversation`, wikilink section to originals (now under `04-Archive/conversations/YYYY-MM/`), and wikilinks to every source note union'd across turns. Originals move via `shutil.move`; archive folder is also excluded del index so archived raws don't compete con curated synthesis. Errors per cluster swallowed (cluster entry gets `error` key; other clusters proceed). Log schema en `~/.local/share/obsidian-rag/consolidation.log` (JSONL: `{run_at, window_days, n_conversations, n_clusters, n_promoted, n_archived, duration_s, dry_run, clusters: [...]}`). CLI flags: `--window-days`, `--threshold`, `--min-cluster`, `--dry-run`, `--json`. Launchd: `com.fer.obsidian-rag-consolidate` (Mondays 06:00 local), registered en `_services_spec()`, installable via `rag setup`.

**Web chat tool-calling** (`web/tools.py`, 9 tools): `search_vault`, `read_note`, `reminders_due`, `gmail_recent`, `finance_summary`, `calendar_ahead`, `weather` (read-only) + `propose_reminder`, `propose_calendar_event` (create-intent, implementations live in `rag/__init__.py` — `web/tools.py` re-exports). `/api/chat` corre 2-phase tool loop: pre-router (`_detect_tool_intent`, keyword → forced read tool) + optional LLM tool-decide round (gated by `RAG_WEB_TOOL_LLM_DECIDE`, default OFF). Create intent ("recordame", "creá un evento", ...) detected by `_detect_propose_intent` (defined in `rag/__init__.py`, shared between web + CLI) which FORCES the LLM decide round ON for that query — propose tools necesitan LLM arg extraction, no pueden correr from pre-router. Create tools auto-create reminder/event si datetime es unambiguous (SSE `created` event → inline `╌ ✓ agregado...` chip, reminders get inline `deshacer` link backed by `DELETE /api/reminders/{id}`, events don't since Calendar.app AppleScript delete is unreliable) o caen back a `proposal` card con ✓ Crear / ✗ Descartar cuando parser flagueó `needs_clarification`. Low-level helpers `_parse_natural_datetime` (dateparser + qwen2.5:3b fallback, `_preprocess_rioplatense_datetime` para `18hs`/`al mediodía`/`X que viene`), `_parse_natural_recurrence` (regex over ES/EN patterns), `_create_reminder` (supports `due_dt`, `priority`, `notes`, `recurrence`), `_create_calendar_event` (via Calendar.app AppleScript — iCloud writable, unlike JXA read path), `_has_explicit_time` (auto all-day detection), `_delete_reminder`, `_delete_calendar_event` all in `rag/__init__.py`. Recurrence on Reminders is best-effort (inner try/on error) since property is macOS-version-dependent; on Calendar es stable.

**CLI chat create-intent** (`rag chat`): mismo `_detect_propose_intent` + mismas propose tools, pero ported to terminal via `_handle_chat_create_intent` at top de cada turn's input. Single-round ollama tool-decide con `_CHAT_CREATE_OVERRIDE` prompt + `tools=[propose_reminder, propose_calendar_event]` only; on tool_call → dispatches + renders Rich chip `╌ ✓ agregado...` en mismo `sáb 25-04 (todo el día)` / `lun 20-04 22:27` shape como web UI (hard-coded `es-AR` weekdays porque `%a` is locale-dependent). command-r's `{parameters: {...}}` arg wrapping unwrapped same way as `rag do`. Returns `(handled, created_info)` donde `created_info` carries `{kind, reminder_id, title}` on successful reminder create (None for events — Calendar.app AppleScript delete is unreliable, matches web UX which shows no undo for events). Chat loop stashes `created_info` en `last_created` (session-local, not persisted) y `/undo` slash command dispatchea `_delete_reminder(last_created["reminder_id"])` para reversar most recent create; `last_created` clears on success so a second `/undo` returns "nothing to undo". Tests: `tests/test_chat_create_handler.py` (8 cases) + `tests/test_chat_undo.py` (5 cases) — all monkeypatched, no live ollama.

**Rioplatense datetime normalization** (`_preprocess_rioplatense_datetime`, runs before `dateparser` inside `_parse_natural_datetime`): dateparser 1.4 maneja maybe 30% de AR-idiom inputs correctamente y silenciosamente echoes anchor time para otro 30% (e.g. "a las 10 de la mañana" → anchor time). Hand-rolleamos regex rewrites que normalizan a forms que dateparser PUEDE parsear — mostly English equivalents con `PREFER_DATES_FROM=future`. Cubre: `18hs` → `18:00`; `al mediodía` → `12:00`; `X que viene` → bare weekday/`next week`/`next month`; `el|este|próximo <weekday>` → bare English weekday (porque dateparser 1.4 rejects `next <weekday>` silently pero acepta bare `thursday` con future-prefer); `pasado mañana` → `day after tomorrow`; `a las N de la mañana|tarde|noche` → `N:00 am`/`(N+12):00`; `a la mañana|tarde|noche|tardecita` → default hour (09/16/20/17); `tipo N` / `a eso de las N` → `N:00` (rioplatense approximations); diminutives (`horitas` → `horas`); `el finde` → `saturday`. Anchor-echo guard tras dateparser: si input lleva time marker pero dateparser devolvió exact anchor time, descartar y caer al LLM. LLM fallback prompt (qwen2.5:3b, `HELPER_OPTIONS` deterministic) explícitamente flagea rioplatense, pasa both raw text and normalized hint, e instruye rollforward para bare weekdays + 09:00 default for missing times.

**Ambient agent**: hook en `_index_single_file` on saves dentro `allowed_folders` (default `["00-Inbox"]`). Config: `~/.local/share/obsidian-rag/ambient.json` (`{jid, enabled, allowed_folders?}`). Skip rules: outside allowed_folders, no config, frontmatter `ambient: skip`, `type: morning-brief|weekly-digest|prep`, dedup 5min (upsert on `rag_ambient_state.path`). Sends via `whatsapp-bridge` POST (`http://localhost:8080/api/send`). Bridge down = message lost pero analysis persiste en `rag_ambient`.

**Contradiction radar**: Phase 1 (query-time `--counter`), Phase 2 (index-time frontmatter `contradicts:` + `rag_contradictions`), Phase 3 (`rag digest` weekly). Skipped on `--reset` (O(n²)) y `note_body < 200 chars`.

**Contradicciones → ranker penalty (loop cerrado, default OFF)**: `_load_contradiction_priors()` lee `rag_contradictions` (window 90d, cap 5000 paths) y devuelve `{subject_path: log1p(count_distinct_ts)}` — penaliza más fuerte notas con detecciones SEPARADAS en el tiempo (señal robusta de contradicción persistente, no falso positivo único). Consumido por (a) `retrieve()` como debuff `final -= weights.contradiction_penalty * priors.get(path, 0.0)` después del bloque behavior priors, (b) `collect_ranker_features()` como feature `contradiction_count` (14ta del LightGBM ranker, última posición). Default `weights.contradiction_penalty = 0.0` → loop OFF: hasta que `rag tune` no lo levante (range `(0.0, 0.30)` en `_TUNE_SPACE`), comportamiento bit-idéntico al pre-feature. Silent-fail en read SQL → `{}` (retrieve sigue funcionando sin priors). Schema `rag_contradictions` ya existía (populado por `_log_contradictions` cuando `rag contradictions <path>` corre); este es primer consumer que lo realimenta al ranker. Tests: `tests/test_contradiction_priors.py`, `tests/test_retrieve_contradiction_penalty.py`, `tests/test_ranker_lgbm_contradiction_feature.py`.

**URL sub-index**: `obsidian_urls_v1` collection embeds **prose context** (±240 chars) no URL strings. `PER_FILE_CAP=2`. Auto-backfill on first `find_urls()` if collection empty.

**Wikilinks**: regex scan against `title_to_paths`. Skips: frontmatter, code, existing links, ambiguous titles, short titles (min-len 4), self-links. First occurrence only. Apply iterates high→low offset.

**Archive**: reuses `find_dead_notes`, maps to `04-Archive/<original-path>` (PARA mirror), stamps frontmatter `archived_at/archived_from/archived_reason`. Opt-out: `archive: never` o `type: moc|index|permanent`. Gate: >20 candidates without `--force` → dry-run. Batch log en `filing_batches/archive-*.jsonl`.

**Morning**: collects 36h window (modified notes, inbox, todos, contradictions, low-conf queries, Apple Reminders, calendar, weather, screentime). Weather hint sólo si rain ≥70%. Dedup vault-todos vs reminders (Jaccard ≥0.6). System-activity + Screen Time sections son deterministic (no LLM).

**Screen Time**: `_collect_screentime(start, end)` lee `~/Library/Application Support/Knowledge/knowledgeC.db` (`/app/usage` stream, read-only via `immutable=1` URI). Sessions <5s filtered. Bundle→label map + category rollup (code/notas/comms/browser/media/otros). Renders sólo si ≥5min de activity. Section omitted silently si db missing. Dashboard `/api/dashboard` expone 7d aggregate + daily series (capped at 7 — CoreDuet aggregates older data away).

**Today**: `[00:00, now)` window, 4 fixed sections, writes `-evening.md` suffix. Feeds next morning organically.

**Followup**: extracts loops (frontmatter todo/due, unchecked `- [ ]`, imperative regex), classifies via qwen2.5:3b judge (temp=0, seed=42, conservative). Un embed + un LLM call por loop.

**Read**: fetch URL → readability strip → gate (< 500 chars = error) → summary via `resolve_chat_model()` (default qwen2.5:7b) → two-pass related lookup → tags from existing vocab (never invents) → `00-Inbox/`. Dry-run default, `--save` to write.

**Ranker-vivo (closed-loop ranker)**: implicit feedback from daily use re-tunes `ranker.json` nightly without manual intervention. Cuatro signal sources insertan en `rag_behavior`: (1) CLI `rag open` wrapper (opt-in via `RAG_TRACK_OPENS=1` + user-registered `x-rag-open://` handler); (2) WhatsApp listener classifying follow-up turns (`/save`, quoted reply → positive; "no"/"la otra"/rephrase → negative; 120s silence → weak positive); (3) web `/api/behavior` POST from home dashboard `sendBeacon` clicks; (4) morning/today brief diff (`_diff_brief_signal` compara yesterday's written brief vs current on-disk — wikilinks que survived = `kept`, missing = `deleted`, dedup via `rag_brief_state`). Nightly `com.fer.obsidian-rag-online-tune` at 03:30 corre `rag tune --online --days 14 --apply --yes`, que llama `_behavior_augmented_cases` (weight=0.5, drops conflicts), backs up current `ranker.json` → `ranker.{ts}.json` (keeps 3 newest), re-tunes, corre bootstrap-CI gate (`_run_eval_gate`: scrubs `RAG_EXPLORE`, subprocess `rag eval`, 10min cap, regex parses hit@5). Si singles < `GATE_SINGLES_HIT5_MIN` (default 0.60, override via `RAG_EVAL_GATE_SINGLES_MIN`) OR chains < `GATE_CHAINS_HIT5_MIN` (default 0.73, override via `RAG_EVAL_GATE_CHAINS_MIN`) → auto-rollback + exit 1 + log to `rag_tune`. `rag tune --rollback` restaura most recent backup manualmente. **Floor recalibrados 2026-04-23** desde originales 0.7619 / 0.6364: con expansión de `queries.yaml` (42→60 singles post-2026-04-21, +cross-source/synthesis/comparison goldens deliberadamente más duros), baseline estable cayó a 71.67% / 86.67% y floors fueron rebajados a nuevos CI lower bounds (mismo criterio metodológico: "95% confianza de que corridas bajo el floor son regresión real, no noise"). Ver bloque comentarios sobre `GATE_SINGLES_HIT5_MIN` en `rag/__init__.py` (~línea 23121) para timeline completa.

## Eval baselines

**Floor (2026-04-27, post-golden-remap vault reorg, commit 6f8994f)** — vault reorg eliminó paths que ya no existen; golden remap redujo set de n=60→54 singles / n=12→9 chains. Dos corridas reproducibles (bit-idénticas en hit@5 + chain_success; MRR chains dentro de CI):
- Singles: `hit@5 53.70% [40.74, 66.67] · MRR 0.528 [0.407, 0.657] · n=54`
- Chains: `hit@5 72.00% [52.00, 88.00] · MRR 0.633–0.653 [0.460, 0.820] · chain_success 33.33% [11.11, 66.67] · turns=25 chains=9`
- Lower-CI-bound gate (nightly online-tune auto-rollback): singles < 40.74% OR chains < 52.00%
- Nota: caída vs floor previo (singles 71.67% → 53.70%, chains 86.67% → 72.00%) NO es regresión del pipeline — es reducción del n y remoción de goldens fáciles que ya no existen en vault post-reorg. Queries removidas pertenecían mayoritariamente a `01-Projects/RAG-Local/*` (notas movidas al .trash/ por user). Floors nuevos codifican mismo criterio: "95% confianza de que corrida bajo el floor es regresión real, no noise".

**Floor (2026-04-17, post-golden-expansion + bootstrap CI)** — queries.yaml doubled (21→42 singles, 9→12 chains; +15 singles in under-represented folders 03-Resources/Agile+Tech, 02-Areas/Personal, 01-Projects/obsidian-rag, 04-Archive memory). `rag eval` ahora reporta percentile bootstrap 95% CI (1000 resamples, seed=42) alongside each metric + `rag eval --latency` reporta P50/P95/P99 de retrieve() per bucket y acepta `--max-p95-ms` como CI gate.
- Singles: `hit@5 88.10% [76.19, 97.62] · MRR 0.772 [0.651, 0.873] · n=42`
- Chains: `hit@5 78.79% [63.64, 90.91] · MRR 0.629 [0.490, 0.768] · chain_success 50.00% [25.00, 75.00] · turns=33 chains=12`
- Latency: singles p95 2447ms · chains p95 3003ms

Cada post-expansion metric sits dentro del prior floor's CI on smaller set — expansion surfaced the noise band (~21pp singles hit, ~50pp chain_success) que previamente masqueraba como drift.

**Post prompt-per-intent + citation-repair (2026-04-19):** Singles `hit@5 88.10% [76.19, 97.62] · MRR 0.767 [0.643, 0.869]` — identical hit@5, MRR within CI. Chains `hit@5 81.82% [66.67, 93.94] · MRR 0.636 [0.505, 0.773] · chain_success 58.33% [33.33, 83.33]` — +3pp hit@5, +8pp chain_success, both inside prior CI so treat as noise until replicated. Floor unchanged for auto-rollback gate (still 76.19% / 63.64%).

**Post golden-set re-mapping (2026-04-20):** vault reorg (PARA moves: many notes `02-Areas/Coaching/*` → `03-Resources/Coaching/*`, `03-Resources/{Agile,Tech}/*` → `04-Archive/*`, etc.) left 33 of 65 `expected` paths in `queries.yaml` pointing at dead files, artificially cratering eval to singles hit@5 26% / chains 33%. Golden rebuilt by auto-mapping 31 unique paths via filename-stem lookup to closest surviving note (prefer non-archive, bias `01→02→03→04` for tie-breaks) and dropping one chain whose source notes (`reference_{claude,ollama}_telegram_bot.md`) no longer exist. Post-rebuild eval: Singles `hit@5 78.57% [64.29, 90.48] · MRR 0.696 [0.554, 0.810]`; Chains `hit@5 75.76% [60.61, 90.91] · MRR 0.641 [0.510, 0.788]`. Both CIs overlap the 2026-04-19 run — within noise band. Floor unchanged (76.19% / 63.64%); current singles 78.57% and chains 75.76% pass the auto-rollback gate.

**Post-T10 (2026-04-20, after JSONL-fallback strip, commit `81e32b4`):** Singles `hit@5 78.57% [64.29, 90.48] · MRR 0.696 [0.554, 0.810] · recall@5 76.19% · n=42`; Chains `hit@5 86.67% [73.33, 96.67] · MRR 0.728 [0.594, 0.850] · chain_success 63.64% [36.36, 90.91] · turns=30 chains=11`. Singles **bit-identical** vs pre-T10 (expected — T10 is pure storage refactor, retrieval pipeline untouched); chains drifted +11pp inside prior CI (same noise band). Latency: singles p95 2797ms, chains p95 3406ms — slight uptick vs pre-T10 (2447/3003ms) attributable to SQL being only write path (no JSONL-queue offload anymore). Still ×5 below any action threshold. Floor gate passed at exact chain_success boundary (63.64%) — fine this run but worth re-measuring next tune cycle.

**Post cross-source corpus (2026-04-21, n=55 singles / 11 chains):** Primer eval con corpus mixto activo — 20 chunks gmail + 36 chunks reminders + 4071 chunks whatsapp + vault (ingesters Phase 1.a-1.d corridos por primera vez en prod). `queries.yaml` extendido con 6 queries synthesis/comparison (Fase 2) + 7 queries cross-source (Fase 1.f — 4 reminders + 3 gmail). Singles `hit@5 80.00% [69.09, 90.91] · MRR 0.714 [0.609, 0.818]`; Chains `hit@5 83.33% [70.00, 96.67] · MRR 0.706 [0.567, 0.833] · chain_success 54.55% [27.27, 81.82]`. **Todos los metrics overlapean CI del baseline anterior** — singles +1.4pp noise, chains −3.3pp noise, chain_success −9pp dentro del ±CI. Auto-rollback gate pasa por doble margen (floor 76.19% / 63.64% vs actual 80.00% / 83.33%). 6 de las 7 queries cross-source hitearon — la que falla ("resumen bancario BICA enero 2026") no es issue de threshold sino retrieval específico del corpus (thread Gmail está más oculto de lo que query esperaba). **Decisión Phase 1.f tuning**: `CONFIDENCE_RERANK_MIN_PER_SOURCE` queda todo en global 0.015 — no hay regresión medible que justifique bajarlo per-source. Si aparecen false-refuse cross-source en logs producción, re-evaluar.

**Post Calendar API enable (2026-04-21 evening, n=60 singles / 11 chains):** Calendar API del proyecto GCP `701505738025` activada por user → `rag index --source calendar --reset` ingestó 368 eventos (2y history + 180d future del calendar `fernandoferrari@gmail.com`). Corpus: 8231 chunks (calendar 4.5% · WA 49.5% · vault 45.3% · reminders 0.4% · gmail 0.2%). `queries.yaml` +5 queries calendar (turno psicólogo / workshop AI Engineer / despedida de jardín de astor / reunión con Max ops / turno erica franzen). Singles `hit@5 81.67% [71.67, 91.67] · MRR 0.735 [0.639, 0.831] · recall@5 76.94% · n=60`; Chains sin cambio (no se agregaron chains calendar): `hit@5 83.33% · MRR 0.706 · chain_success 54.55%`. **5/5 queries calendar hit con MRR promedio 0.87** — confirmación directa de que pipeline existente resuelve events.list()-ingested calendar sin tocar scoring/threshold. Smoke test previo sobre query real log: "qué hago el viernes 20hs" pasó de score 0.04 (refuse) → 0.46 (respuesta útil) post-ingesta, validando hallazgo de producción. **Bug arquitectónico residual detectado**: `classify_intent` devuelve `recent` para "qué tengo esta semana" → itera notas del vault en vez de filtrar `source=calendar` con ventana temporal. Antes era inerte porque calendar estaba vacío; ahora merece ticket como Phase 1.g (intent-source routing) — **addressed later in same session, see next entry**. Auto-rollback gate sigue pasando holgado. **Calendar path format drift fix**: `tests/test_eval_bootstrap.py::test_golden_cross_source_paths_have_native_id_format` tenía regex `calendar://(event:)?<id>` basado en design doc §2.7, pero implementación real del ingester (`scripts/ingest_calendar.py::_event_file_key`) usa `calendar://<calendar_id>/<event_id>` (two-segment, paralelo a WhatsApp). Test updated para seguir implementación como ground truth.

**Post session close (2026-04-21 evening, same n=60 singles / 11 chains):** Final eval tras `agenda` intent (v1 + v2 window filter), cold-embed non-blocking fix (`_local_embedder_ready` Event), + feedback harvest de 8 negativos adicionales al `rag_feedback` SQL (ratio pos:neg pasó de 55:2 → 55:10). Singles `hit@5 81.67% [71.67, 91.67] · MRR 0.779 [0.683, 0.875] · recall@5 76.94%`; Chains `hit@5 80.00% [66.67, 93.33] · MRR 0.733 [0.583, 0.867] · chain_success 54.55% [27.27, 81.82]`. **Singles MRR +0.044 vs baseline de la mañana** (dentro del CI anterior pero consistent uplift — probablemente absorción de los 8 negatives al golden cache). Chains MRR +0.027 idem. Hit@5 bit-identical en singles, −3.3pp en chains (dentro del CI, noise). Floor gate pasa por doble margen. Calendar queries siguen 5/5 hit (idéntico al baseline anterior — fix del intent router no las rompió, sólo redirigió queries browsing que antes caían en `recent`). Auto-rollback gate holgado.

**Fase D closeout — 3 mejoras validadas (2026-04-21, n=60 singles / 11 chains):** Dos runs de `rag eval` con flags todas-OFF y todas-ON (RAG_ADAPTIVE_ROUTING=1 RAG_ENTITY_LOOKUP=1 RAG_NLI_GROUNDING=1). Resultados **bit-idénticos** en ambas corridas — las 3 mejoras no introducen regresión retrieval medible. **Flags-OFF**: Singles `hit@5 71.67% [60.00%, 83.33%] · MRR 0.681 [0.567, 0.794]`; Chains `hit@5 86.67% [73.33%, 96.67%] · MRR 0.807 [0.680, 0.923] · chain_success 72.73% [45.45%, 100.00%]`. **Flags-ON**: Singles `hit@5 71.67% [60.00%, 83.33%] · MRR 0.681 [0.567, 0.794]`; Chains `hit@5 86.67% [73.33%, 96.67%] · MRR 0.790 [0.657, 0.917] · chain_success 72.73% [45.45%, 100.00%]`.

**Post perf-audit close (2026-04-22 evening, n=60 singles / 11 chains, 8311 chunks):** Sesión extensa de audit + fixes (deep_retrieve 30s guard, semantic_cache background write + refusal gating, fast-path num_ctx 2048→4096, WhatsApp corpus reindexado 0→4134 chunks, gliner instalado en tool install, backfill_entities sobre todo el corpus cross-source +356 entidades / +5046 mentions, test_tag isolation, 3462 legacy rows source-backfilled a vault). Corpus post-reindex ahora con whatsapp 49.7% / vault 45.8% / calendar 4.4% / gmail+reminders <1%. **Transformers downgrade 5.6→5.1** forzado por constraint de gliner. Eval final: Singles `hit@5 71.67% [60.00%, 83.33%] · MRR 0.678 [0.561, 0.794] · recall@5 69.17%`; Chains `hit@5 86.67% [73.33%, 96.67%] · MRR 0.740 [0.603, 0.867] · recall@5 77.22% · chain_success 72.73% [45.45%, 100.00%]`. **Chain success +9pp vs baseline de la mañana pre-sesión** (63.64% → 72.73%); singles bit-idéntico. Latencia retrieve(): singles p50 1302ms / p95 1661ms; chains p50 1564ms / **p95 3307ms** (+49% vs pre-sesión — explicado por corpus 2x post-WA-reindex: más candidatos al BM25 + semantic → más pares al cross-encoder). Floor gate chains pasa por doble margen; singles en banda de noise LLM pre-existing que ranker-vivo ya intentó mover y auto-rollback rechazó. **Transformers 5.1 no introdujo regresión medible**. `semantic_cache_store_failed` se congeló en 314 (0 nuevos desde fix `background=True` + `_is_refusal()` gating), `gliner_import_failed` se congeló en 49 (0 nuevos desde `uv tool install --with gliner`). El system_memory_metrics watchdog no disparó escalación durante sesión (peak wired+active+compressed ~60%). **Nota de variabilidad**: singles hit@5 bajó de 81.67% (baseline previo) a 71.67% en esta sesión — CIs solapan [71.67%, 83.33%] — causa probable: LLM non-determinism (qwen2.5:7b stochastic) + posible drift de paths en queries.yaml. Caída es pre-existente, no causada por flags (flags-OFF muestra igual caída). **Decisión flags**: (a) `RAG_ADAPTIVE_ROUTING` → **stays OFF por default** — criterios floor (lower CI ≥ 76.19%) no se cumplen en esta sesión eval (lower CI 60.00% < 76.19%); aunque caída es pre-existente y no causada por flag, instrucciones son conservadoras. Flipear a ON cuando `rag eval` vuelva a superar floor de forma estable. (b) `RAG_ENTITY_LOOKUP` → **flipped a ON tras backfill** (same-day evening commit) — `scripts/backfill_entities.py` corrió sobre corpus (2022 entities / 6520 mentions / 71% chunk coverage), smoke-test directo validó `rag query "todo lo que tengo de Astor"` retornando síntesis cross-source agregada. Ver §Env vars para detalle. (c) `RAG_NLI_GROUNDING` → **stays OFF** — requiere validación de latencia P95 post-50 queries reales.

**Rerank fp16 A/B 2026-04-22 — NO PROMOTED:** probado si degradar cross-encoder `bge-reranker-v2-m3` de fp32 a fp16 en MPS ahorra latencia sin romper calidad. Baseline fp32 vs candidato fp16 (via `CrossEncoder(model_kwargs={"torch_dtype": torch.float16})` — alternativa `model.half()` post-load crashea en predict con `AttributeError: 'ne'` en transformers 5.6 / ST 5.4.1, y de hecho en A/B previa 2026-04-13 colapsaba todos scores a ~0.001).

| métrica                    | fp32                         | fp16 (model_kwargs)          | delta                 |
|---------------------------|------------------------------|------------------------------|-----------------------|
| singles hit@5 (n=60)      | 71.67% [60.00%, 83.33%]      | 71.67% [60.00%, 83.33%]      | **0 pp**              |
| singles MRR               | 0.669 [0.553, 0.783]         | 0.678 [0.561, 0.794]         | +0.009 (dentro CI)    |
| singles recall@5          | 68.33%                       | 69.17%                       | +0.84 pp              |
| singles wall-time 60 Q    | **63 s**                     | **121 s**                    | **+58 s (~2× slower)**|
| chains                    | n/a (ollama ReadTimeout)     | n/a (ollama ReadTimeout)     | n/a                   |
| retrieve P50/P95          | no reportado (crash)         | no reportado (crash)         | n/a                   |

Ollama helper (qwen2.5:3b para `reformulate_query`) colapsó en etapa de chains en ambas runs incluso tras retry — bug infra ortogonal al A/B, documentado como ruido. Suficiente signal en singles: calidad **igual** (hit@5 bit-idéntico, MRR dentro del CI), latencia **peor 2×** (~+970ms/query en wall-clock del stage). Explicación: MPS no tiene kernels fp16 optimizados para arquitectura xlm-roberta de `bge-reranker-v2-m3`; overhead de casting dtype supera cualquier win teórico de throughput. Criterio "retrieve P95 baja >200ms" **violado** (fue al revés). Decisión: **NO PROMOTE**, revertido patch de `RAG_RERANKER_FP16`, código queda en fp32. Nueva NOTE en `get_reranker()` documenta ambos A/B (2026-04-13 y 2026-04-22) para evitar re-intentar.

**Prior floor (2026-04-17, post-title-in-rerank, n=21 singles / 9 chains):** Singles `hit@5 90.48% · MRR 0.821`; Chains `hit@5 80.00% · MRR 0.627 · chain_success 55.56%`. Kept for historical trend, but do not compare new numbers against it without overlapping CIs.

**Even-earlier floor (2026-04-16, post-quick-wins, n=21/9):** Singles `hit@5 90.48% · MRR 0.786`; Chains `hit@5 76.00% · MRR 0.580 · chain_success 55.56%`.

El 2026-04-15 floor (`95.24/0.802` singles, `72.00/0.557/44.44` chains, ver `docs/eval-tune-2026-04-15.md`) pre-data tanto la expansion como CI tooling — treat as qualitative reference only.

Never claim improvement without re-running `rag eval`. Helper LLM calls (`expand_queries`, `reformulate_query`, `_judge_sufficiency`) are already deterministic via `HELPER_OPTIONS = {temperature: 0, seed: 42}`.

**HyDE with qwen2.5:3b drops singles hit@5 ~5pp**. HyDE is opt-in (`--hyde`); re-measure if helper model changes.

**`seen_titles` as post-rerank penalty** (2026-04-20, `SEEN_TITLE_PENALTY = 0.1` in `retrieve()`). LLM-instruction path regressed chains (chains hit@5 −16pp, chain_success −33pp — helper treats list as "avoid these" and drifts off-topic); post-rerank soft penalty es shipped replacement. Candidates whose `meta.note` (case-insensitive) matches any `seen_titles` entry get final score docked by 0.1 — diversity nudge, not filter (strong rerank leads still win). En `reformulate_query` el kwarg permanece on signature but is intentionally unused en prompt (dead per design). Tests in `tests/test_seen_titles_penalty.py`. Empirical lift on queries.yaml chains hit@5 83.33% → 90.00% (both inside CI — re-measure on next tune cycle before claiming stable gain).

## Cross-source corpus (Phase 1, in progress — 2026-04-20 decisions)

Corpus ya no es vault-only. Per `docs/design-cross-source-corpus.md` + §10 user decisions, `retrieve()` ahora es source-aware y la sqlite-vec collection holds chunks de múltiples sources via discriminator metadata `source`. Collection stays at `obsidian_notes_v11` (no rename / no re-embed) — legacy vault rows sin `source` se leen como `"vault"` via `normalize_source()`.

**Constants** (`rag/__init__.py`): `VALID_SOURCES` (frozenset of 11 — vault + calendar + gmail + whatsapp + reminders + messages + contacts + calls + safari + drive + pillow), `SOURCE_WEIGHTS` (vault 1.00 → WA 0.75), `SOURCE_RECENCY_HALFLIFE_DAYS` (None for vault/calendar, 30d for WA/messages, 90d for reminders, 180d for gmail), `SOURCE_RETENTION_DAYS` (None for vault/calendar/reminders, 180 for WA/messages, 365 for gmail). `pillow` es source local-only (no entra al corpus vectorial — datos viven en `rag_sleep_sessions` y se consumen via panel home + brief, no via retrieve).

**Helpers**: `normalize_source(v, default="vault")` → safe legacy-row read; `source_weight(src)` → lookup + 0.50 fallback; `source_recency_multiplier(src, created_ts, now)` → exponential decay `2**-(age/halflife)` in [0,1], accepts epoch float or ISO-8601 string (Zulu Z), clamps future-ts at 1.0, None-halflife short-circuits to 1.0.

**Scoring** (inside `retrieve()` post-rerank loop + en `apply_weighted_scores()` para eval parity): después que la formula de scoring existente produce `final`, multiplicar por `source_weight(src) * source_recency_multiplier(src, created_ts)`. Vault default → `1.0 * 1.0` = no-op. Old vault data completamente untouched.

**Filter** (retrieve/deep_retrieve/multi_retrieve `source` kwarg + `rag query --source S[,S2]`): string o iterable de strings; restricts candidate pool post-rerank. Unknown sources from CLI are rejected upfront with helpful error. Legacy vault path: `source=None` o `source="vault"` → identical to pre-Phase-1 behavior.

**Conversational dedup** (`_conv_dedup_window`, applied post-scoring pre top-k slice): collapses WhatsApp/messages chunks from same `chat_jid` within ±30min window — keeps only highest-scored. Non-WA sources pass through unchanged. Intentionally simple O(n²) — pool capped at `RERANK_POOL_MAX`, constant factor negligible.

### WhatsApp ingester — Phase 1.a (`scripts/ingest_whatsapp.py`, `rag index --source whatsapp`)

Reads from `~/repositories/whatsapp-mcp/whatsapp-bridge/store/messages.db` in read-only immutable mode. Filters empty content, `status@broadcast` pseudo-chat, **`WHATSAPP_BOT_JID` (RagNet group)**, **mensajes con prefix U+200B (output del bot)**, y anything older than 180d. Timestamps (Go RFC3339 with nanoseconds / Z suffix / numeric) parsed defensively. Conversational chunking (§2.6 option A): groups same-sender contiguous messages within 5min windows; splits on speaker change OR >=5min gap OR >800 chars; merges undersized groups (<150 chars) into temporally-nearest neighbor in same chat. Parent window ±10 messages, 1200 char cap. Embed prefix `[source=whatsapp | chat=X | from=Y] {body}`; display text stays raw. doc_ids son `whatsapp://{chat_jid}/{first_msg_id}::{idx}` — stable across bridge DB compactions. Idempotent upsert (delete prior by `file` key + add). Incremental cursor en `rag_whatsapp_state(chat_jid, last_ts, last_msg_id)`; `--reset` wipes, `--since ISO` overrides uniformly. CLI flags: `--bridge-db`, `--since`, `--reset`, `--max-chats`, `--max-messages`, `--dry-run`, `--json`.

**Anti-feedback-loop guards** (closed 2026-04-28): RagNet (`120363426178035051@g.us`) es UI del bot — recibe morning briefs, archive pushes, reminder pushes, anticipatory prompts, draft cards via `RAG_DRAFT_VIA_RAGNET`, slash commands del user (`/help`, `/note`, `/cap`), y respuestas del bot a esos comandos. **Nada de eso es contenido conversacional**. Pre-fix indexer chunkeaba todo eso → corpus se llenaba de output del propio bot → retrieve devolvía briefs viejos como "evidencia" → siguiente brief incluía self-references. Fetchers usados por brief mismo (`_fetch_whatsapp_unread`, `_fetch_whatsapp_today`, `_fetch_whatsapp_window` en [`rag/integrations/whatsapp.py`](rag/integrations/whatsapp.py)) ya filtraban RagNet a nivel SQL desde Phase 1.a; indexer era último path abierto. **Fix 2026-04-28**: `HARDCODED_EXCLUDE_JIDS` agrega `WHATSAPP_BOT_JID` + content-level filter `content.startswith('\u200B')` (defense in depth — cualquier mensaje en cualquier chat con prefix U+200B es output del bot por contrato del listener, no se indexa). Aplica también a [`_read_recent_image_messages`](scripts/ingest_whatsapp.py) (mismo `exclude_jids` default). **Implicación operativa**: si corpus tenía chunks de RagNet de runs anteriores, siguen ahí — `rag index --source whatsapp --reset` los limpia. Tests: [`tests/test_ingest_whatsapp.py`](tests/test_ingest_whatsapp.py) (3 casos nuevos: RagNet exclusion, U+200B content filter, frozenset sanity).

### Calendar ingester — Phase 1.b (`scripts/ingest_calendar.py`, `rag index --source calendar`)

Google Calendar via OAuth (§10.6 user override — rompe local-first). Creds bajo `~/.calendar-mcp/{gcp-oauth.keys.json, credentials.json}`. Window `[now − 2y, now + 180d]` on bootstrap, `syncToken` for incremental. `singleEvents=True` (expands RRULE instances). Chunk-per-event, `parent=body`, body cap 800 chars. Cancelled events → delete. State en `rag_calendar_state(calendar_id, sync_token, last_updated, updated_at)`. Hardcoded exclude list filtra `addressbook#contacts` + `en.usa#holiday` noise calendars.

### Gmail ingester — Phase 1.c (`scripts/ingest_gmail.py`, `rag index --source gmail`)

Gmail via OAuth (same cred dir convention, `~/.gmail-mcp/`). Thread-level chunking (§2.6 — un chunk per thread, no per message — empíricamente matchea user "cuándo hablamos de X" granularity better than message-level). Quoted replies + signatures stripped via regex before chunking. `parent = subject + first 1200 chars of thread`. Incremental via Gmail's `historyId` cursor en `rag_gmail_state(history_id, updated_at)`. Bootstrap usa `q=newer_than:365d` per §10.2 retention. Deleted threads removed from index on incremental pass.

### Reminders ingester — Phase 1.d (`scripts/ingest_reminders.py`, `rag index --source reminders`)

Apple Reminders via AppleScript (local, same trust boundary as morning brief's `_fetch_reminders_due`). Pulls every reminder (pending + completed) with id, list, due/completion/creation/modification dates, name, body, priority, flagged state. Chunk-per-reminder, body cap 800 chars. `created_ts` anchor preference: creation → due → modified → completion. Content-hash diffing en `rag_reminders_state(reminder_id, content_hash, last_seen_ts, updated_at)` — on each run, re-fetch full catalogue, upsert changed/new, delete stale (id no longer present). No cursor / modification-date polling — Reminders.app's `modification date` is unreliable via AppleScript. Field separator en AS → Python pipe es chr(31) (Unit Separator) para evitar collisions con body content. CLI flags: `--reset`, `--dry-run`, `--only-pending` (default indexes both). Retention None (§10.2); source weight 0.90, recency halflife 90d (§10.3).

### Contacts ingester — Phase 1.e (`scripts/ingest_contacts.py`, `rag index --source contacts`)

Apple Contacts via direct SQLite read on `~/Library/Application Support/AddressBook/Sources/*/AddressBook-v22.abcddb` (one DB per account: local, iCloud, Google, etc.). Cero pyobjc, cero AppleScript — el .abcddb es plain SQLite + standard Apple Core Data schema (`ZABCDRECORD` + `ZABCDPHONENUMBER` + `ZABCDEMAILADDRESS` + `ZABCDNOTE`). Chunk-per-contact (atomic, <800 chars), body cap 800. Merges first+last+org+phones+emails+note+birthday. Timestamps converted from Cocoa epoch (2001-01-01) to Unix via `_cocoa_to_unix()`. Phone `ZLABEL`s like `_$!<Mobile>!$_` → clean `mobile`. Empty records (no name + no phone + no email + no org) filtered out as Core Data internals / groups.

**Dual role**: además de corpus ingestion, módulo expone `resolve_phone(raw_number) -> Contact | None` used by `scripts/ingest_calls.py` (and future iMessage/WhatsApp enrichment) to map phone digits back to human name. Two-pass phone index: Pass 1 dedupes `(digits → UID)` canonically (iCloud linked cards across sources share same phone but have DIFFERENT `ZUNIQUEID`s → pick contact con longer `display_name`, tiebreak by lexicographic UID for determinism), Pass 2 fans out to suffix keys (full, last-10, last-8, last-7) dropping genuine cross-UID collisions (AR mobile y US landline sharing last-7 digits → ambiguous, return None rather than mis-attribute). Measured on dev host: naive-UID-conflict detection dropped 90 of 580 keys and resolved only 3% of call numbers; two-pass fix raised it to 39% (which is plausible upper bound since ~60% of incoming calls are spam/telemarketers not in contacts).

State: `rag_contacts_state(contact_uid, content_hash, last_seen_ts, updated_at)`. Content hash excludes `modified_ts` (iCloud bumps it on idle sync). Stale deletion on each run (UIDs in state DB but missing from live AddressBook → delete from corpus). `invalidate_phone_index()` called at end of every successful `run()` so downstream in-process callers (same process indexing calls after contacts) see fresh data. doc_ids son `contacts://<ZUNIQUEID>::0`. Retention None (§10.2 — contacts don't age), source weight 0.95, recency halflife None (no decay). CLI flags: `--reset`, `--dry-run`, `--root` (override AddressBook root, useful for tests), `--json`. Tests: [`tests/test_ingest_contacts.py`](tests/test_ingest_contacts.py) (40 cases).

### Calls ingester — Phase 1.f (`scripts/ingest_calls.py`, `rag index --source calls`)

macOS/iOS CallHistory via direct SQLite read on `~/Library/Application Support/CallHistoryDB/CallHistory.storedata` (Core Data). Pulls every call (phone, FaceTime audio/video, incoming / outgoing / missed / unanswered) from `ZCALLRECORD` within retention window. Chunk-per-call (atomic, <800 chars). Timestamps converted from Cocoa epoch (`ZDATE + 978307200`). Direction matrix from `(ZORIGINATED, ZANSWERED)`: outgoing+answered → "saliente · atendida", outgoing+unanswered → "saliente · sin respuesta", incoming+answered → "entrante · atendida", incoming+unanswered → "perdida" (missed — `is_missed=True`). Service provider mapping: `com.apple.Telephony` → "Teléfono", `com.apple.FaceTime` → "FaceTime", unknown passed through raw.

**Enrichment via Contacts**: chunk body headline usa `ingest_contacts.resolve_phone(ZADDRESS)` para mostrar human names en vez de raw digits. Fallback chain: resolved Contact → `ZNAME` cached by Apple at call time → raw address → "(desconocido)". Headline phrasing es BM25-friendly: "Llamada perdida de Juli" / "Llamada saliente a Astor" para que queries como "llamadas perdidas de Juli" hiteen sin depender de embeddings sólo.

State: `rag_calls_state(call_uid, content_hash, last_seen_ts, updated_at)`. Calls son efectivamente immutable once logged, así que hash mostly guards against Apple's rare retroactive edits. Stale deletion when `ZUNIQUE_ID` rolls off Apple's retention window (macOS keeps a few months). doc_ids son `calls://<ZUNIQUE_ID>::0`. Retention 180d (matches WhatsApp/messages — equally ephemeral), source weight 0.80 (between gmail 0.85 y whatsapp 0.75 — log entries son factual pero semánticamente thin), recency halflife 30d (who-called-yesterday es crítico, who-called-six-months-ago es trivia). CLI flags: `--reset`, `--dry-run`, `--since ISO` (hard floor, intersects con retention), `--retention-days N` (override default 180; 0 disables), `--db` (override path), `--json`. Tests: [`tests/test_ingest_calls.py`](tests/test_ingest_calls.py) (34 cases).

### Safari ingester — Phase 2 (`scripts/ingest_safari.py`, `rag index --source safari`)

Dos fuentes, un solo ingester: `~/Library/Safari/History.db` (SQLite) + `~/Library/Safari/Bookmarks.plist` (binary plist que incluye Reading List como subtree `com.apple.ReadingList`). Complementa el inline Chrome ETL existente (`rag/__init__.py` (función `_sync_chrome_history`) escribe `.md` al vault) — Safari va por arquitectura source-prefixed (chunks en DB vectorial directo, sin polución al vault).

**Historia**: SQL JOIN `history_items ← history_visits`. Filtra `load_successful=1` (drop 404s + red fails) + `redirect_source IS NULL` (drop 301/302 intermediate hops), agrega por URL (no por visita — 7407 visitas aplastadas a ~3800 URLs únicas), agarra título más reciente non-empty como display. Retention 180d sobre `last_visit`. Cap `max_urls=5000` por run (configurable) para evitar runs multi-minute en historial largo. doc_ids: `safari://history/<history_item_id>::0`.

**Bookmarks**: recursive walk del plist tree, skippeando folders internos (`BookmarksBar`, `BookmarksMenu`) y nodos Proxy (Historial placeholder, Reading List shortcut). URIDictionary.title gana sobre Title directo para leaves. Reading List entries cargan `ReadingList.PreviewText` que se concatena al título con ` — ` (rico para BM25 en "artículo que guardé sobre X"). doc_ids: `safari://bm/<UUID>::0` para bookmarks, `safari://rl/<UUID>::0` para Reading List — prefix distinto porque UUID podría moverse entre bookmarks ↔ RL y necesitamos delete surgical por el otro prefix al migrar.

**State**: dos tablas separadas — `rag_safari_history_state(history_item_id INTEGER PK, content_hash, last_seen_ts, updated_at)` + `rag_safari_bookmark_state(bookmark_uuid TEXT PK, content_hash, ...)`. Diff + stale delete por cada una. Content hash del history excluye `first_visit_ts` (estable por definición) pero incluye `last_visit_ts` + `visit_count` para detectar nuevas visitas. Bookmark hash incluye `is_reading_list` flag para detectar movimiento RL↔bookmarks.

Source weight 0.80 (mismo banda que calls — signal factual rico pero no curado por usuario como Contacts). Halflife 90d (browsing context ages mid-term; no es conversacional como WA pero tampoco permanente como Calendar). Retention 180d. CLI flags: `--reset`, `--dry-run`, `--since ISO`, `--retention-days N`, `--max-urls N` (default 5000), `--skip-bookmarks`, `--history-db`, `--bookmarks-plist`, `--json`. Tests: [`tests/test_ingest_safari.py`](tests/test_ingest_safari.py) (37 cases).

**Note on SQLite contention**: cuando `rag serve` / `web/server.py` está corriendo, primer `rag index --source safari` puede pegar `database is locked` en bookmarks bulk-add (1000+ rows en una transacción + GLiNER entity extraction concurrente). Reintentá — state de history ya commiteó en primera tanda, así que retry sólo procesa bookmarks. Long-term fix: serializar con `vault_write_lock` el branch de safari (TODO, no-blocker por ahora).

### Remaining (Phase 1.g, 1.h + 2)

- **Phase 1.g — apagar workaround** (gated on 1.c stable in prod ≥1 week): deprecar `/note` + `/ob` del whatsapp-listener ahora que corpus captura WA por barrido. ~100 LOC, mostly external repo. (Renamed from 1.e; letter freed for Contacts ingester.)
- **Phase 2 — OCR pipeline para adjuntos** (deferred 2026-04-21, no shipped): design doc §8 flagea "no indexa adjuntos binarios (imágenes WA, PDFs en Gmail). Eso es fase 2". Evaluado en tanda 2026-04-21 y **skipped** porque sistema actual tiene 16 imágenes (todas en `04-Archive/99-obsidian-system/99-Attachments/`, screenshots archived) + **0 PDFs** en vault, y corpus cross-source (donde estarían adjuntos reales de Gmail/WA) tiene 0 chunks — ingesters Phase 1.a-1.d nunca corrieron en prod. Sin data activa, implementar OCR sería scaffolding + agregar dep (`pyobjc-framework-Vision` ~20 MB, o tesseract via brew) sin beneficio medible actual. **Trigger activación**: cuando ingesters hayan corrido ≥1 semana y haya ≥20 adjuntos referenciados en corpus cross-source, implementar usando Apple Vision (`VNRecognizeTextRequest`, local) para imágenes + `pdftotext` (poppler) para PDFs con fallback Vision para scans. Chunk OCR como prose con metadata `attachment_of: <parent doc_id>` + `media_type: "ocr"`.
- **Phase 1.h — re-calibración eval** *(infra shipped 2026-04-21, tuning pending real data)*:  (Renamed from 1.f; letter freed for Calls ingester.)
  - **Infra shipped**: `CONFIDENCE_RERANK_MIN_PER_SOURCE` dict en `rag/__init__.py` (scaffolding — todos los valores = baseline 0.015 hoy) + helper `confidence_threshold_for_source(source)` con fallback al global. Invocado en `query()` y `rag serve` sobre `source` del top-result meta. Tests: [`tests/test_confidence_threshold_per_source.py`](tests/test_confidence_threshold_per_source.py) (9 casos). Test `tests/test_eval_bootstrap.py::test_queries_yaml_all_paths_exist_or_placeholder` ahora acepta paths con prefijos `gmail://` / `whatsapp://` / `calendar://` / `reminders://` / `messages://` como placeholders válidos; sanity-test aparte `test_queries_yaml_cross_source_prefixes_cover_all_valid_sources` detecta drift contra `VALID_SOURCES`. Template de queries cross-source está comentado en [`queries.yaml`](queries.yaml) listo para un-commentar cuando ingesters populen corpus.
  - **Tuning pending**: re-correr `rag eval` + bajar per-source thresholds empíricamente (expected: WA 0.008-0.010, Calendar 0.012, Gmail 0.010-0.012, Reminders 0.012) una vez que ingesters hayan corrido ≥1 semana + haya feedback data. Validar `SOURCE_WEIGHTS` hardcoded (vault 1.00 / calendar 0.95 / reminders 0.90 / gmail 0.85 / WA 0.75 / messages 0.75) contra queries reales. Deferred per §10.8.

## On-disk state (`~/.local/share/obsidian-rag/`)

### Telemetry — SQL tables (post-T10 2026-04-19, post-split 2026-04-21)

Telemetry + learning state vive en **dos** databases bajo `~/.local/share/obsidian-rag/ragvec/`:

- **`ragvec.db`** (~104M) — sqlite-vec `meta_*`/`vec_*` tables del corpus + **10 state tables**: `rag_whatsapp_state`, `rag_calendar_state`, `rag_gmail_state`, `rag_reminders_state`, `rag_contacts_state`, `rag_calls_state`, `rag_safari_history_state`, `rag_safari_bookmark_state`, `rag_wa_media_state`, `rag_schema_version`. Sólo cursors + dedup keys de ingesters.
- **`telemetry.db`** (~36M) — **45+ tablas** operativas: `rag_queries`, `rag_behavior`, `rag_feedback`, `rag_feedback_golden*`, `rag_tune`, `rag_contradictions`, `rag_ambient*`, `rag_brief_*`, `rag_wa_tasks`, `rag_archive_log`, `rag_filing_log`, `rag_eval_runs`, `rag_surface_log`, `rag_proactive_log`, `rag_cpu_metrics`, `rag_memory_metrics`, `system_memory_metrics`, `rag_conversations_index`, `rag_response_cache`, `rag_entities`, `rag_entity_mentions`, `rag_ocr_cache`, `rag_vlm_captions`, `rag_audio_transcripts`, `rag_learned_paraphrases`, `rag_cita_detections`, `rag_score_calibration`, `rag_schema_version`.

**Split rationale** (`scripts/migrate_ragvec_split.py`, 2026-04-21): cada DB comparte único WAL entre todos sus writers. Mezclar chunks + telemetría en WAL único causaba bursts de lock contention — indexer escribiendo 100 chunks interfería con write sync de cada query log. Separar en 2 DBs permite que cada WAL tenga su propio pattern de writes (indexer bulk vs telemetry append) sin bloquearse entre sí. `_ragvec_state_conn()` resuelve a `telemetry.db` (post-split); ingesters siguen abriendo directamente `ragvec.db` para su state cursor (ver `rag.DB_PATH / "ragvec.db"` en `scripts/ingest_*.py`).

**Reset total**: `rm ragvec/ragvec.db ragvec/telemetry.db` + `rag index --reset`. Para reset sólo telemetría preservando corpus: `rm ragvec/telemetry.db` (se recrea vacía en próximo open).

SQL es único storage path — T10 (2026-04-19) stripped JSONL writers + readers. `RAG_STATE_SQL` fue removida del código el 2026-05-04; plists de launchd siguen seteándola como trail de deployment para rollback más rápido vía git-revert.

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
- `rag_draft_decisions` (2026-04-29) — decisiones del user sobre drafts del bot WA (`approved_si | approved_editar | rejected | expired`) + bot_draft + sent_text + original_msgs. **Keep all forever** — dataset histórico es gold humano para fine-tunear modelo de drafts (ver "Bot WA draft loop" más arriba). Populated via `POST /api/draft/decision` desde listener TS.
- `rag_brief_feedback` (2026-04-29) — reactions del user sobre briefs (morning / evening / digest) pusheados por daemon (`positive | negative | mute`) + `dedup_key=vault_relpath`. **Keep all forever** — input de tuning para horario / cadencia / contenido de briefs (ver "Brief feedback loop" más arriba). Populated via `POST /api/brief/feedback` desde listener TS.

State-style tables:
- `rag_conversations_index` — episodic session_id → relative_path (web/conversation_writer.py upsert; replaces old conversations_index.json + fcntl dance).
- `rag_feedback_golden` (pk=path,rating, `embedding BLOB` float32 little-endian, `source_ts`) + `rag_feedback_golden_meta` (k/v) — cache rebuilt cuando `rag_feedback.max(ts) > meta.last_built_source_ts`. `record_feedback` clears both tables synchronously so next `load_feedback_golden()` call always rebuilds (sidesteps a same-second MAX(ts) collision que could leave stale cache).
- `rag_response_cache` — semantic response cache (GC#1, 2026-04-22, durability + wiring fixes 2026-04-23). Key shape: `(id, ts, question, q_embedding BLOB, dim, corpus_hash, intent, ttl_seconds, response, paths_json, scores_json, top_score, hit_count, last_hit_ts, extra_json)`. Lookup: cosine ≥ `_SEMANTIC_CACHE_COSINE` (default 0.93 vía `RAG_CACHE_COSINE`) contra últimas `_SEMANTIC_CACHE_MAX_ROWS` entradas del mismo `corpus_hash` dentro de `ttl_seconds`; hits bump `hit_count` + `last_hit_ts`. **Gates de store** (aplican igual en sync y background): (a) cache disabled, (b) corpus_hash vacío, (c) response vacío, (d) `top_score < 0.015` (refuse por gate de confianza), (e) `_is_refusal(response)` matchea (refuse conceptual del LLM — patrón añadido 2026-04-22 tras observar cache poisoning: una query con top_score 1.18 cacheó "No tengo esa información" y envenenó queries similares permanentemente). Helpers: `semantic_cache_lookup()` / `semantic_cache_store()` / `semantic_cache_clear(corpus_hash?)` / `semantic_cache_stats()`. **2026-04-23 audit + fix** (cache tenía 0 hits reales con 2,335 queries y 14 queries repeated ≥10×): tres changes concurrentes.
    1. **`corpus_hash` simplificado a count-only** (era count + top-10 mtimes). Cada edit a nota individual no invalida más el cache global. Sólo add/remove de notas (chunk-count delta) dispara invalidación coarse. **2026-04-24 follow-up** (commit `09f00bd`): count exacto seguía cambiando con cada chunk que ingester agregaba/removía — audit del web.log mostró 30 SEMANTIC PUTs con 24 corpus_hashes DISTINTOS → 0 hits ever porque cada query mintea hash nuevo. Ahora bucketea por `_CORPUS_HASH_BUCKET = 100`: `_compute_corpus_hash(col)` = `sha256(f"count_bucket:{col.count() // 100}")[:16]`. Hash sólo cambia cuando count cruza múltiplo de 100 — sobrevive rotación normal de WhatsApp/Calendar/Gmail incrementales (typically <50 chunks/run). Tradeoff: bulk +/-100 chunks netos invalidan cache (correcto: corpus cambió suficiente como para esperar retrieval diferente). Per-entry staleness check (paso 2) ya cubre edits a paths citadas dentro del bucket.
    2. **Per-entry freshness check** (`_cached_entry_is_stale(paths, cached_ts)`) en `semantic_cache_lookup`: si cualquiera de paths cacheadas tiene `mtime > cached_ts`, fila se skippea con `probe.reason="stale_source"` sin tumbar resto del cache. File missing / vault-path unresolvable se tratan como fresh (no blow-up por infra issues; invalidación global ya atrapa deletes).
    3. **Durabilidad del store en `query()` CLI**: `background=True → background=False`. Store del `rag query` one-shot se perdía en atexit drain (2s cap) porque worker daemon se estaba todavía reintentando bajo contention (telemetry.db recibe 2k+ writes/hour entre queries/behavior/cpu/memory metrics). Synchronous store bloquea return por ≤1.3s pero user ya vio respuesta — no hay regresión de latencia percibida, y cache queda realmente persistente. Background mode sigue disponible para long-running processes (web server, serve.chat) que no tienen el problema del atexit.
    4. **Wiring extendido a `run_chat_turn()`** (helper compartido — cubre `chat()` CLI + futuros callers del unified pipeline). Eligibility: single-vault, no history, no source/folder/tag/date_range filter, no critique/counter/precise. Nuevo field en `ChatTurnRequest`: `cache_lookup: bool = True`, `cache_store: bool = True`, `cache_background: bool = True` (opt-out per-caller). Hit path sintetiza `RetrieveResult` mínimo desde paths cacheados para que `to_log_event` loguee normal.
    4.b **Wiring en `/api/chat` (web server, 2026-04-23)** — cubre caller más grande (856/2,335 queries = 37% del tráfico). Web ya tenía LRU exact-string (`_CHAT_CACHE`, 100 entries × TTL 5min, in-memory) en `web/server.py`. Semantic cache SQL se agrega como *segundo layer* POST-LRU-miss: exact → semantic → pipeline, con hit del semantic hidratando LRU así próxima query exact-string pega instantánea. Gates: no history, single-vault, no propose_intent. Sintetiza `sources_items` desde paths+scores del hit (minimal meta: file/note/folder/score/bar). `done` event trae `cache_layer="semantic"` para que UI lo distinga del LRU (UI-key ya existente — mismo stage=`cached`). Store post-pipeline con `background=True` (web server es long-running, no sufre atexit drop del CLI).
    4.c **Wiring en `rag serve` /query (WhatsApp listener + bots, 2026-04-23)** — cierra último caller. Serve ya tenía su LRU propio (`_serve_cache`, 64 entries × TTL 5min) keyed en `(sid|folder|tag|loose|question)`. Mismo patrón: lookup semantic post LRU miss (después del weather + tasks short-circuits para no cachear time-sensitive), store semantic pegado al `_cache_put(cache_key, payload)` dentro del mismo `if not force and not qfolder and not qtag:` guard. Hit path sintetiza sources en shape del listener (`{note, path, score}`, no `{file, note, folder, score, bar}` como web). Log event: `cmd="serve.cached_semantic"` — bucket propio para distinguirlo de `serve`/`serve.tasks`/`serve.chat` en analytics. Tests: [`tests/test_serve_semantic_cache.py`](tests/test_serve_semantic_cache.py) (12 casos — source-grep contract por consistencia con test_serve_fast_path_consumption.py + test_serve_short_circuits.py existentes).
    5. **`cache_probe` instrumentation** en `rag_queries.extra_json`: `{result: hit|miss|skipped|disabled|error, reason: match|below_threshold|ttl_expired|stale_source|corpus_mismatch|flags_skip|cache_disabled|no_corpus_hash|db_error, top_cosine: float|null, candidates: int, skipped_stale: int, skipped_ttl: int}`. `semantic_cache_lookup(..., return_probe=True)` devuelve tupla `(hit_or_None, probe_dict)` — backward-compat preservada (default `return_probe=False` devuelve sólo hit/None).
    6. **`rag cache stats --days N`** extendido: hit rate real del período leyendo `extra_json.cache_probe` + distribución de miss reasons + ahorro estimado (avg `t_gen_ms` de misses × hits) + top 10 queries cacheadas por `hit_count`. Nuevo helper `_cache_telemetry_stats(days=7)` cross-referencea `rag_queries` con `rag_response_cache`.
    Tests: [`tests/test_semantic_cache.py`](tests/test_semantic_cache.py) (22 casos base), [`tests/test_semantic_cache_probe.py`](tests/test_semantic_cache_probe.py) (8 casos — probe shape por cada `reason`), [`tests/test_semantic_cache_freshness.py`](tests/test_semantic_cache_freshness.py) (8 casos — `_cached_entry_is_stale` unit + integration lookup skip), [`tests/test_semantic_cache_run_chat_turn.py`](tests/test_semantic_cache_run_chat_turn.py) (9 casos — hit short-circuits LLM, miss corre pipeline, skip por history/source/critique/multi-vault, cache_lookup=False opt-out, `to_log_event` emite cache fields), [`tests/test_cache_stats_telemetry.py`](tests/test_cache_stats_telemetry.py) (6 casos — eligible/hits/reasons/top_queries/CLI smoke), [`tests/test_web_chat_semantic_cache.py`](tests/test_web_chat_semantic_cache.py) (9 casos — SSE replay shape, ollama.chat no llamado en hit, sources sintetizados, LRU hit beats semantic, store post-pipeline, gates history/propose/multi-vault, lookup-exception fallback).

**Nota sobre cobertura de policy**: Las ~20 tablas restantes (rag_status_samples, rag_home_compute_metrics, rag_synthetic_negatives, rag_synthetic_queries, rag_audio_corrections, rag_anticipate_candidates, rag_behavior_priors_wa, rag_error_queue, rag_learned_paraphrases, rag_llm_captions, rag_negotiation_*, rag_promises, rag_reminder_wa_pushed, rag_routing_decisions, rag_routing_rules, rag_score_calibration, rag_style_fingerprints, rag_whatsapp_scheduled, rag_cita_detections, etc.) tienen policies implícitas en sus ingesters/writers pero no documentadas acá — consultar código fuente en `rag/__init__.py` (`_TELEMETRY_DDL` y callers de `_sql_append_event`) para determinar retention/schema actuales.

Primitives en `rag/__init__.py` (`# ── SQL state store (T1: foundation) ──` section):
- `_ensure_telemetry_tables(conn)` — idempotent DDL, **ensure-once por (proceso, db_path)** desde commit `09f00bd` (2026-04-24). Set keyed `_TELEMETRY_DDL_ENSURED_PATHS` skip-ea las ~32 CREATE TABLE IF NOT EXISTS + ALTER tras primera invocación contra path. Cuts ~17K DDL stmts/hr × schema-lock contention (medido: avg conn-open 1.5ms first → 0.64ms next, ~5-8x speedup). **Si agregás entry nueva a `_TELEMETRY_DDL` y querés que aparezca en procesos already-running, hay que reiniciarlos** (launchctl bootout/bootstrap los daemons `com.fer.obsidian-rag-*`); no hay hot-reload. Tests con tmp DB siguen funcionando porque set es path-keyed, no proceso-global.
- `_ragvec_state_conn()` — short-lived WAL conn with `synchronous=NORMAL` + `busy_timeout=10000`
- `_sql_append_event(conn, table, row)`, `_sql_upsert(conn, table, row, pk_cols)`, `_sql_query_window(conn, table, since_ts, ...)`, `_sql_max_ts(conn, table)`

Writer contract (post-T10): single-row BEGIN/COMMIT into SQL. On exception, log error to `sql_state_errors.jsonl` and **silently drop event** — no JSONL fallback. Callers never see raised exception. Reader contract: SQL-only. Readers return empty snapshots (behavior priors, feedback golden, behavior-augmented cases, contradictions) o False/None (brief_state, ambient_state lookups) on SQL error; retrieval pipeline stays functional sin priors hasta que DB sea readable de nuevo.

#### Invariantes del telemetry stack (audit 2026-04-24 + extensión 2026-04-25)

Cuatro reglas que código tiene que respetar — violar cualquiera deja bugs latentes. Las tres primeras salieron del audit del 2026-04-24 tras 6 días de degradación silenciosa; cuarta del audit 2026-04-25 tras encontrar tests escribiendo a la prod telemetry.db en 3 clases distintas.

1. **Todo silent-error sink llama `_bump_silent_log_counter()`**. Cualquier función nueva tipo `_log_X_error(...)` que escribe a `.jsonl` y devuelve sin raisear DEBE invocar helper en `rag/__init__.py` post-write. Sin esto, alerting a stderr (threshold `RAG_SILENT_LOG_ALERT_THRESHOLD=20/h`) queda parcial — exactamente cómo 1756 errores SQL en 6 días no dispararon un solo alert. Pre-fix `_silent_log` lo bumpeba pero `_log_sql_state_error` no. Tests: `tests/test_silent_log_alerting.py`.

2. **Async writer = paquete completo de 4 cambios**. Cuando writer pasa a usar `_enqueue_background_sql`: (a) helper de gate per-writer (`_log_X_event_background_default()`), (b) caller con branch sync/async, (c) autouse fixture en conftest que setea `RAG_LOG_X_ASYNC=0`, (d) doc del env var en este CLAUDE.md. Tocar sólo (a)+(b) deja tests rotos en producción y la próxima persona descubre el override por accidente. Tests: `tests/test_sql_async_writers.py`.

3. **Readers SQL: retry + stale-cache fallback, nunca empty default que sobrescriba memo**. `_load_behavior_priors` y `load_feedback_golden` son los modelos. En error path, devolver cache previo SIN tocar `_X_memo`. Bug clásico que esto previene: `default=("error", {empty}, None)` del retry era asignado al memo, envenenando cache hasta que `source_ts` cambiara. Tests: `tests/test_sql_reader_retry.py`.

4. **Tests con TestClient o writers SQL aíslan `DB_PATH` per-file**. Conftest autouse `_isolate_vault_path` cubre `VAULT_PATH` global pero NO hay equivalente para `DB_PATH` — intentos conftest-wide reverteados (sesión 2026-04-25, dos veces) porque disparan warning falso de `_stabilize_rag_state` cuando un test sub-fixture redirige a sub-tmp. Cualquier test que use `fastapi.testclient.TestClient(app)`, llame `log_query_event` / `log_behavior_event` / `semantic_cache_*` / `record_feedback` directamente, o ejercite endpoints `/api/chat` / `/api/behavior`, DEBE redirigir `rag.DB_PATH` con autouse fixture **snap+restore manual** (no `monkeypatch.setattr`):

   ```python
   @pytest.fixture(autouse=True)
   def _isolate_db_path(tmp_path):
       import rag as _rag
       snap = _rag.DB_PATH
       _rag.DB_PATH = tmp_path / "ragvec"
       try: yield
       finally: _rag.DB_PATH = snap
   ```

   Razón del manual snap+restore: `monkeypatch.setattr` revierte en su propio teardown que corre DESPUÉS del teardown de `_stabilize_rag_state` → la stabilizer ve el tmp todavía aplicado y warning. Mismo patrón que `tests/test_rag_log_sql_read.py::sql_env`. Tests con isolation aplicada (al 2026-04-25): `test_degenerate_query`, `test_semantic_cache*` (5 archivos), `test_rag_log_sql_read`, `test_post_t10_sql_readers`, `test_followup`, `test_read`. **Pendiente** (gap conocido): `test_web_{cors,pwa,chat_low_conf_bypass,sessions_sidebar,static_cache,chat_tools,propose_endpoints,chat_mode}`, `test_propose_mail_send`, `test_drive_search_tool`. Pollution medida 2026-04-25: 161 entries `event=test_tag` en `sql_state_errors.jsonl`, 5 rows `question='test'` en `rag_response_cache`, 57 rows `cmd='web.chat.degenerate'` con `q='?¡@#'` en `rag_queries`. Memoria: [feedback_test_db_path_isolation.md](.claude/projects/-Users-fer-repositories-obsidian-rag/memory/feedback_test_db_path_isolation.md) si existe symlink en tu workspace.

Diagnóstico data-first: correr `python scripts/audit_telemetry_health.py --days 7` antes de cualquier "auditá el sistema" — agrega los 5 queries que reprodujeron audit 2026-04-24 en 1 segundo (errores SQL/silent, latency outliers, cache probe distribution, DB sizes). Primer comando del workflow.

rag implicit-feedback [--days 14 --json]  # recolecta feedback implícito de interacciones
rag routing-rules [--reset --debug --json]  # descriptor de rutas + patterns detectados
rag whisper-vocab [--refresh --show --source X --limit N]  # manejo de vocabulario de transcripción WhatsApp
rag vault-cleanup [--dry-run --apply --force]  # limpia carpetas transitorias del vault
rag ingest-drive [--reset --dry-run --json]  # Google Drive ingester — busca DAO + documentos compartidos


**Drift fixes (2026-04-21 evening)** — cuatro CLI readers seguían tail-readeando JSONL files que post-T10 ya no reciben los expected events o se repurposaron para otro log stream. Todos migrated a SQL:

| Reader | Pre-fix behaviour | Fix |
|---|---|---|
| `rag log` | Renderizaba todas columnas vacías — `queries.jsonl` ahora recibe `conversation_turn_written` observability events con schema diferente | `_read_queries_for_log(n, *, low_confidence)` + `_read_feedback_map_for_log()` en `rag/__init__.py` — SELECT desde `rag_queries` + `rag_feedback`, hoist `turn_id` de `extra_json`, filter admin rows con `q` vacío. Renderer null-safety on `t_retrieve` / `t_gen` / `ts` / `mode` cuando SQL rows return `None` (metachat / create-intent turns). Tests: `tests/test_rag_log_sql_read.py` (13 cases). |
| `rag emergent` + `rag dashboard` | `_scan_queries_log(days)` leía `LOG_PATH` → empty events list → "sin queries en ventana", dashboard mostraba `n=0` | Misma `_scan_queries_log(days)` signature, SELECT desde `rag_queries WHERE ts >= ?` (chronological ASC), hoists **all** `extra_json` keys to top-level (`q_reformulated`, `answered`, `gated_low_confidence`, `turn_id`) so callers don't re-parse JSON. Admin rows (`q=""`) excluded. Tests: `tests/test_post_t10_sql_readers.py` (5 cases para este reader). |
| `feedback_counts()` / `rag insights` | `FEEDBACK_PATH.is_file()` → False (sólo `.bak` queda post-cutover) → returned `(0, 0)` silently; `rag insights` showed vacío | Single SQL `SELECT SUM(CASE WHEN rating > 0 ...)` aggregate. Tests: 3 cases. |
| `_feedback_augmented_cases()` / `rag tune` | `FEEDBACK_PATH` empty → returned `[]` → **rag tune silently lost every corrective-path signal** the user ever gave. Era el highest-impact drift — tune no podía aprender de "este path era correcto" feedback. | SQL con `json_extract(extra_json, '$.corrective_path')` since `corrective_path` no es first-class column. Filter `scope != 'session'` + `len(q) >= min_len` + dedup por normalised query. Tests: 5 cases. |
| `rag patterns` | Mismo shape que feedback_counts — read empty `feedback.jsonl`, always "sin feedback log todavía" | Inline SQL query con `json_extract(extra_json, '$.reason')` para `reason` field (también dentro de `extra_json`). |

También: deleted dead code `_iter_behavior_jsonl()` (definido post-T10 pero nunca llamado).

Pattern para nuevos SQL readers: usar `_ragvec_state_conn()` context manager; wrap SELECT en `try/except` + `_silent_log` + return empty value matching old JSONL-reader shape. Never raise from reader — calling CLI command keeps working con degraded signal.

Migration one-shot: `scripts/migrate_state_to_sqlite.py --source-dir ~/.local/share/obsidian-rag [--dry-run] [--round-trip-check] [--reverse] [--summary]`. Refuses to run while `com.fer.obsidian-rag-*` services are up (preflight `pgrep`; `--force` to override). Renames each source → `<name>.bak.<unix_ts>` on successful commit. Cutover de 2026-04-19 imported 7,946 records across 19 sources; 43 malformed pre-existing records dropped (missing NOT NULL fields).

Rollback procedure (post-T10): **escape hatch ahora requiere code revert, no sólo CLI invocation.** `rag maintenance --rollback-state-migration [--force]` still restaura newest `.bak.<ts>` per source y dropea las 20 `rag_*` tables + VACUUM — pero in-code readers/writers sólo conocen SQL path después de T10. Para revertir totalmente:

1. `git revert <T10-commit-sha>` (o `git reset --hard <pre-T10-sha>` si T10 commits son tip). Esto trae back JSONL fallback code.
2. Restart launchd services so reverted `rag.py` is loaded in-process.
3. Run `rag maintenance --rollback-state-migration` — esto restaura JSONL .bak files que reverted code now reads.

Los `.bak.<ts>` files bajo `~/.local/share/obsidian-rag/` siguen ahí (kept for 30-day window) so data-loss is bounded, pero sin code revert restored files son ignored. `rag maintenance` continúa pruning `.bak.*` older than 30d.

### Other state (unchanged; still on disk)

- `ranker.json` — tuned weights. Delete = reset to hardcoded defaults.
- `ranker.{unix_ts}.json` — 3 most recent backups, escritos en cada `rag tune --apply`. Consumidos por `rag tune --rollback` + auto-rollback CI gate.
- `sessions/*.json` + `last_session` — multi-turn state (TTL 30d, cap 50 turns).
- `ambient.json` — ambient agent config (jid, enabled, allowed_folders).
- `filing_batches/*.jsonl` — audit log (prefix `archive-*` para archiver).
- `ignored_notes.json`, `home_cache.json`, `context_summaries.json`, `auto_index_state.json`, `coach_state.json`, `synthetic_questions.json`, `wa_tasks_state.json` — app state + caches.
- `online-tune.{log,error.log}`, `*.{log,error.log}` — launchd service logs.
- `sql_state_errors.jsonl` — diagnostic sink para SQL-path write/read failures. Post-T10 esto es único signal visible cuando SQL errors ocurren, since JSONL fallback is gone y event is dropped post-logging here.

**Reset learned state**: `rm ranker.json` + `DELETE FROM rag_feedback_golden; DELETE FROM rag_feedback_golden_meta;` dentro de **`telemetry.db`** (post-split 2026-04-21 esas tablas se movieron de `ragvec.db` → `telemetry.db`). Full re-embed: `rag index --reset`.

## Vault path

Default: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes`. Override: `OBSIDIAN_RAG_VAULT` env var. Collections namespaced per vault (sha256[:8]).

**Persistent memories del MCP [`mem-vault`](https://github.com/jagoff/mem-vault)** viven en `04-Archive/99-obsidian-system/99-AI/memory/` (folder real, no symlink — comentario antiguo sobre Claude Code era obsoleto). Configurado via env vars del web server plist:
- `MEM_VAULT_PATH=Notes/`
- `MEM_VAULT_MEMORY_SUBDIR=04-Archive/99-obsidian-system/99-AI/memory`

A diferencia del resto de `99-obsidian-system/`, este folder **NO está excluido por `is_excluded()`** (junto con `99-Mentions/`) — `rag index` lo scanea y los `.md` de memorias entran al index como notas más del vault `home`. Eso permite que `rag query "..."` recupere bug patterns, decisiones y convenciones acumuladas entre sesiones (66 memorias / 398 chunks al 2026-04-29). El MCP `mem-vault` sigue teniendo su propio Qdrant local con las mismas memorias — los dos sistemas coexisten: el MCP es writer canónico, `rag` es reader adicional via embedding pipeline normal.

## Features que dependen de launchd: dejá daemon ACTIVO al cerrar el commit

**Regla universal del repo**: cuando feature nueva se completa con plist `com.fer.obsidian-rag-*`, daemon tiene que estar **cargado y verificado** al cierre del turno. NO dejar como TODO "corré `rag setup` cuando puedas". Feature con cron-dependent behavior **no está completa** hasta que se demuestra que ejecuta sola.

Aprendido 2026-04-25 con `com.fer.obsidian-rag-wa-scheduled-send` (worker de mensajes WhatsApp programados): código + plist factory se shippearon en commit `9740fa1`, pero archivo nunca se copió a `~/Library/LaunchAgents/`. User programó mensaje, esperó la hora, y nada — worker no existía como proceso. Feature parecía rota cuando en realidad sólo faltaba último paso operativo.

### Checklist al agregar plist nuevo

1. **Código**: factory `_<nombre>_plist(rag_bin: str)` + tuple en lista de [`rag/__init__.py:39190+`](rag/__init__.py) que `rag setup` consume.
2. **Click subcommand**: el comando que el plist ejecuta (`@cli.command("...")`).
3. **Smoke del comando manual**: `rag <subcomando> --dry-run` corre sin error y reporta.
4. **Generar plist y copiarlo**:
   ```bash
   .venv/bin/python -c "import rag; print(rag._<nombre>_plist('/Users/fer/.local/bin/rag'))" \
     > ~/Library/LaunchAgents/com.fer.obsidian-rag-<nombre>.plist
   ```
5. **Cargar con launchctl**:
   ```bash
   launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.fer.obsidian-rag-<nombre>.plist
   ```
6. **Verificar que está vivo**:
   ```bash
   launchctl list | grep obsidian-rag-<nombre>          # debe aparecer con state 0 o running
   launchctl print gui/$UID/com.fer.obsidian-rag-<nombre>  # ver run interval, last exit, runs
   ```
7. **Esperar al menos un tick** (o `launchctl kickstart -k gui/$UID/com.fer.obsidian-rag-<nombre>` para forzar primer run) y verificar que log se generó:
   ```bash
   tail -20 ~/.local/share/obsidian-rag/<nombre>.log
   ```
8. **Sólo después** del paso 7, marcar feature como completa.

### Daemons activos del proyecto (referencia)

Lista de plists registrados (cualquier `obsidian-rag-*` que `launchctl list` muestre):

| Plist | Cadencia | Comando | Propósito |
|---|---|---|---|
| `com.fer.obsidian-rag-watch` | filesystem watcher | `rag watch` | Auto-reindex del vault |
| `com.fer.obsidian-rag-serve` | KeepAlive | `rag serve` | MCP server |
| `com.fer.obsidian-rag-web` | KeepAlive | `web/server.py` | Web UI + chat |
| `com.fer.obsidian-rag-digest` | semanal | `rag digest` | Brief semanal |
| `com.fer.obsidian-rag-morning` | calendar 7am L-V | `rag morning` | Brief matinal |
| `com.fer.obsidian-rag-today` | calendar 22hs L-V | `rag today` | Brief vespertino |
| `com.fer.obsidian-rag-wake-up` | calendar | `rag wake-up` | Setup post-sleep |
| `com.fer.obsidian-rag-emergent` | viernes 10am | `rag emergent` | Detector de temas emergentes |
| `com.fer.obsidian-rag-patterns` | domingo 20:00 | `rag patterns` | Alertas de feedback |
| `com.fer.obsidian-rag-archive` | weekly | `rag archive` | Auto-archivo de notas muertas |
| `com.fer.obsidian-rag-wa-tasks` | 30min | `rag wa-tasks` | Extracción de tareas WhatsApp |
| `com.fer.obsidian-rag-reminder-wa-push` | 5min | `rag remind-wa` | Push de Reminders al WA |
| `com.fer.obsidian-rag-wa-scheduled-send` | 5min | `rag wa-scheduled-send` | **(nuevo 2026-04-25)** Worker de mensajes WA programados |
| `com.fer.obsidian-rag-anticipate` | 10min | `rag anticipate` | Anticipatory agent |
| `com.fer.obsidian-rag-auto-harvest` | weekly | `rag auto-harvest` | Auto-tune feedback |
| `com.fer.obsidian-rag-online-tune` | nightly 03:30 | `rag tune --apply` | Ranker-vivo nightly |
| `com.fer.obsidian-rag-calibrate` | nightly | `rag calibrate` | Score calibration |
| `com.fer.obsidian-rag-maintenance` | weekly | `rag maintenance` | Vacuum + WAL checkpoint + log rotation |
| `com.fer.obsidian-rag-consolidate` | nightly | `rag consolidate` | Memory consolidation |
| `com.fer.obsidian-rag-ingest-whatsapp` | horaria | `rag index --source whatsapp` | WhatsApp ingester |
| `com.fer.obsidian-rag-ingest-gmail` | horaria | `rag index --source gmail` | Gmail ingester |
| `com.fer.obsidian-rag-ingest-calendar` | cada 6h | `rag index --source calendar` | Google Calendar ingester |
| `com.fer.obsidian-rag-ingest-reminders` | cada 6h | `rag index --source reminders` | Apple Reminders ingester |
| `com.fer.obsidian-rag-ingest-drive` | horaria | `rag index --source drive` | Google Drive ingester |
| `com.fer.obsidian-rag-implicit-feedback` | 15min | `rag implicit-feedback` | Auto-harvest de feedback implícito |
| `com.fer.obsidian-rag-routing-rules` | 5min | `rag routing-rules` | Detector de patrones de ruteo |
| `com.fer.obsidian-rag-cloudflare-tunnel` | KeepAlive | `cloudflared tunnel` | HTTPS público via Cloudflare Quick Tunnel |
| `com.fer.obsidian-rag-cloudflare-tunnel-watcher` | daemon | `scripts/cloudflared_watcher.sh` | Monitor de túnel + notificación de URL |
| `com.fer.obsidian-rag-serve-watchdog` | daemon | Monitor de `com.fer.obsidian-rag-serve` | Watchdog para reiniciar serve si cae |
| `com.fer.obsidian-rag-active-learning-nudge` | lunes 10:00 | `rag active-learning nudge --json` | Envía nudge WA cuando hay queries low-confidence acumuladas para labeling activo |
| `com.fer.obsidian-rag-brief-auto-tune` | Domingo 03:00 | `rag brief schedule auto-tune --apply` | **(nuevo 2026-04-29)** Auto-tune del horario de los briefs (morning/today/digest) basado en `rag_brief_feedback` |
| `com.fer.obsidian-rag-lgbm-train` | Domingo 02:30 | `rag tune-lambdarank --apply` | Entrena el ranker LightGBM (distinto del nightly online-tune lineal) |
| `com.fer.obsidian-rag-paraphrases-train` | Domingo 04:30 | `rag paraphrases train --since 90` | Lee `rag_feedback rating=1` y upsertea `rag_learned_paraphrases`; acelera `expand_queries()` |
| `com.fer.obsidian-rag-spotify-poll` | cada 60s + RunAtLoad | `scripts/spotify_poll.py` | Registra track actual de Spotify en `rag_spotify_log`; reemplaza la API HTTP cerrada 2026-04-30 |
| `com.fer.obsidian-rag-synth-refresh` | Sábado 22:00 | `rag synth-queries generate --apply && rag synth-queries mine-negatives --apply` | Feeder del lgbm-train: genera queries sintéticas + mina hard-negatives |
| `com.fer.obsidian-rag-vault-cleanup` | nightly | `rag vault-cleanup` | Limpieza de carpetas transitorias |
| `com.fer.obsidian-rag-whisper-vocab` | 03:15 | `rag whisper-vocab refresh` | Extracción nightly de vocab WhatsApp |
| `com.fer.obsidian-rag-ingest-calls` | cada 6h | `rag index --source calls` | Apple CallHistory ingester — llamadas perdidas/entrantes/salientes |
| `com.fer.obsidian-rag-ingest-safari` | cada 6h 15min | `rag index --source safari` | Safari History + Bookmarks + Reading List ingester |
| `com.fer.obsidian-rag-ingest-pillow` | 1×/día 09:30 | `rag index --source pillow` | Pillow ingester — sleep tracker iOS, lee `~/Library/Mobile Documents/com~apple~CloudDocs/Sueño/PillowData.txt` (Core Data dump sync iCloud) → `rag_sleep_sessions`. Silent-fail si Pillow no está instalado / sync roto. CLI: `rag sleep show/patterns/ingest` |
| `com.fer.obsidian-rag-mood-poll` | cada 30min | `rag mood-poll` | Mood poll daemon — **UI no cableada** (mood signals NO se renderizan en home.v2 actualmente) |
| `com.fer.obsidian-rag-daemon-watchdog` | 5min | `rag daemons reconcile --apply --gentle` | **(nuevo 2026-05-01)** Control plane watchdog — retry de daemons en exit≠0 + kickstart-overdue. Reemplaza el catchup post-sleep que tenía el difunto `serve-watchdog`. |

**Nota 2026-05-01**: daemons listados con `(manual)` en tabla arriba están instalados a mano y NO son regenerados por `rag setup` (no figuran en `_services_spec()`). Quedan trackeados por control plane vía `_services_spec_manual()` — `rag daemons status` los muestra con `category=manual_keep`. Lista actual: `cloudflare-tunnel`, `cloudflare-tunnel-watcher`, `lgbm-train`, `paraphrases-train`, `synth-refresh`, `spotify-poll`, `log-rotate`.

Si listado anterior queda desactualizado, source de verdad es lista de tuplas en [`rag/__init__.py`](rag/__init__.py) función `_services_spec()` — `grep -n "_services_spec\|com.fer.obsidian-rag-" rag/__init__.py | head -80`.

### Bypass: `rag setup` también funciona

Si feature shippea junto con cambios al `rag setup` (o si user prefiere reinstalar todo en bloque), `rag setup` instala/recarga TODOS los plists de tabla anterior. Es más invasivo (puede recargar daemons que ya estaban corriendo bien) pero menos manual. Como compromiso: para plists nuevos individuales → recipe del checklist. Para refactors masivos → `rag setup`.

### Control plane: `rag daemons`

Control plane unificado para visibilidad + reconciliación + self-healing del stack launchd. Reemplaza ritual manual de `launchctl list | grep obsidian-rag` + `launchctl print` + `tail` de logs cuando algo no está corriendo bien.

Subcomandos:

- `rag daemons status [--json --unhealthy-only]` — tabla de estado actual (loaded? running? last_exit? overdue? category).
- `rag daemons reconcile [--apply --dry-run --gentle]` — converge drift entre `_services_spec()` y lo que realmente está cargado (default dry-run; `--apply` hace cambios; `--gentle` evita acciones destructivas).
- `rag daemons doctor` — diagnóstico humano + remediation sugerida por daemon problemático (lee logs, parsea exit codes, propone fix).
- `rag daemons retry <label>` — kickstart -k de daemon puntual (acepta slug corto tipo `web` o label completo `com.fer.obsidian-rag-web`).
- `rag daemons kickstart-overdue` — kickstart de daemons marcados `overdue=true` (catchup post-sleep manual cuando Mac estuvo dormido y se saltearon `StartCalendarInterval` schedules).

Daemon `com.fer.obsidian-rag-daemon-watchdog` corre `reconcile --apply --gentle` cada 5min automático. `--gentle` sólo retry-ea daemons en exit≠0 + kickstart-ea overdues, NO bootea huérfanos ni regenera plists — watchdog corre desatendido y no debe tomar decisiones destructivas. Para reconciliación agresiva (incluye bootout de huérfanos + regeneración de plists drifteados) hay que correr `rag daemons reconcile --apply` a mano.

Acciones del control plane (retry, kickstart, bootout, bootstrap) loggean a `rag_daemon_runs` en `telemetry.db` con retention 90d. Útil para audit post-mortem cuando algo se cae a las 4am.

Workflow típico: `rag daemons status` para ver qué hay; `rag daemons doctor` para diagnosticar; `rag daemons reconcile --apply` para corregir drift agresivamente; watchdog corre solo cada 5min sin que tengas que llamarlo.

### Cuándo NO instalar plist en el commit

Excepción legítima: si feature requiere config previo del user (ej. OAuth de Gmail, ambient.json, etc.) y plist crashea sin eso. En ese caso, commit msg debe decir explícito "el plist NO se instala automáticamente porque requiere `<X>` primero" — no "corré `rag setup` cuando puedas" sin más contexto. [`com.fer.obsidian-rag-ingest-calendar.plist`](rag/__init__.py) es ejemplo: `rag setup` lo skipea si `~/.calendar-mcp/credentials.json` no existe.

## Feature H — Chat scoped a nota / folder (2026-04-29)

Selector compacto en composer del chat web (`/chat` → [`web/static/index.html`](web/static/index.html)) que limita retrieval a **una nota específica o folder** en lugar de buscar en todo el vault.

**Flujo end-to-end**:

1. **UI** (`web/static/index.html` + `web/static/app.js`):
   - Botón target (◉) al lado del `+` en composer (`#composer-scope-btn`).
   - Click → abre `#scope-popover` con input de filtro y lista de matches del autocomplete.
   - Click en item → `window.setActiveScope(kind, path)` setea scope en `sessionStorage` y muestra chip "🎯 Limitado a: `<path>` ×" arriba del `#messages`.
   - `×` del chip llama `window.clearActiveScope()` y vuelve a vault entero.
   - JS monkey-patchea `fetch` para inyectar `path` o `folder` SÓLO en POST `/api/chat`, sin tocar `reqBody` literal sepultado a 4500 lines arriba.

2. **Backend** ([`web/server.py`](web/server.py) — buscar `# ── Feature H`):
   - `ChatRequest` ahora acepta `folder: str | None` y `path: str | None`. Validators rechazan URI schemes y traversal (`..`).
   - `multi_retrieve(...)` se llama pasando `folder` como 4to arg posicional (query queda acotada al subset).
   - Si viene `path`, **filtro post-retrieve** exact-match contra `meta.file`. Mantenemos call signature de `multi_retrieve` intacto para no tocar `rag/__init__.py`.
   - **Short-circuit cuando no hay matches**: emit SSE `sources(confidence=0)` + token canned ("No encontré contenido en `<path>`...") + `done(scope_no_match=True, source_specific=True)`. NO 404 — frontend igual quiere SSE stream completo para liberar spinner.

3. **Endpoint nuevo** `GET /api/notes/autocomplete?q=&limit=20`:
   - Substring matching case-insensitive contra `meta.file` + `meta.note` + `meta.folder` desde `_load_corpus(get_db())`.
   - Sortea por: exact-match → startswith → contains-en-path → contains-en-title → folder.
   - `limit` clamped a 50. Empty corpus → `{items: [], reason: "empty_index"}`.
   - Rate-limit reusa `_BEHAVIOR_BUCKETS` (120 req/60s).

**Telemetría**: cuando viene `path` el `result["filters_applied"]["path_scope"]` queda seteado para que `rag_queries` distinga "user pidió scope=path" de "auto-filter encontró un folder". Bucket de log_query_event nuevo: `web.chat.scope_no_match`.

**Tests**: [`tests/test_chat_scoped.py`](tests/test_chat_scoped.py) — 11 casos cubriendo path/folder happy path, no-match short-circuit, autocomplete (matches + clamp + empty index), validators (URI / traversal), HTML smoke test del composer.

## Feature K — "Recordame X" inline en chat (2026-04-29)

Detecta comandos tipo "recordame llamar a Juan mañana 9am" en textarea del chat web y crea reminder de Apple Reminders **automáticamente sin pasar por LLM**, devolviendo SSE `created` event en <100ms vs 5-15s del flow LLM + tools.

**Flujo end-to-end**:

1. **Detector** ([`rag/__init__.py`](rag/__init__.py) — buscar `# ══ Feature K`):
   - `parse_remind_intent(text) → dict | None`.
   - Pattern strict-leading: `^(recordame|recuerdame|acordate|hacéme acordar|reminder|remember me|remind me) [de/que] <rest>$`.
   - Sobre `<rest>`, busca primer marker temporal con `_REMIND_TIME_MARKERS_RE` (mañana, lunes, "en 2 horas", "a las 9", "9am", "18hs", etc.) y parte título/cuándo ahí.
   - Reusa `_parse_natural_datetime` (mismo parser que `_validate_scheduled_for` y `propose_reminder`) — NO duplica lógica de fecha.
   - Anchor-echo guard: si `_parse_natural_datetime` devuelve ~now, descarta (false positive).
   - Devuelve `{title, due_iso, original_text}` o `None` si ambiguo (sin tiempo claro).

2. **Wire-up** ([`web/server.py`](web/server.py) — buscar `# ══ Feature K`):
   - **Antes** del flow normal (`gen()`, después de yield `session`), llamamos a `parse_remind_intent(question)`.
   - Si match → `_create_reminder(title, due_dt=...)` directo (sin tools, sin LLM, sin retrieval).
   - Emit SSE: `sources(confidence=1, intent=remind_inline)` + `created(kind=reminder, created=True, reminder_id, fields, remind_inline=True)` + token canned "✓ Reminder creado: «...» para `<iso>`" + `done(mode=remind_inline, source_specific=True)`.
   - Si `_create_reminder` falla → `proposal(needs_clarification=True, error=...)` para que user reintente desde UI.
   - Si NO match → fall-through al flow normal (LLM + tools, donde `propose_reminder` sigue funcionando).

3. **UI** ([`web/static/app.js`](web/static/app.js)):
   - `event === "created"` con `kind=reminder` ya estaba manejado por `appendCreatedChip()`. Reusamos. Sin cambios al JS específico para Feature K.

**Telemetría**: bucket nuevo `web.chat.remind_inline` en `log_query_event`. Turn persiste con `outcome="reminder_created"` + `reminder_id`.

**Cuándo NO dispara** (por diseño — fallback al LLM):

- "recordame algo" → sin tiempo claro → `None` → flow normal donde `propose_reminder` hace clarificación.
- "qué tengo mañana?" → no hay trigger → flow normal.
- Trigger ambiguo: "recordame llamar a Juan" → sin marker temporal → flow normal.

**Tests**: [`tests/test_chat_remind_inline.py`](tests/test_chat_remind_inline.py) — 10 casos cubriendo detector standalone (happy / ambiguo / sin-trigger / empty / question-with-temporal-word) + wire-up end-to-end (`/api/chat` emite `created` SSE event con shape correcto, `_create_reminder` se llama con args parseados, queries normales NO disparan).

## Query decomposition + RRF fusion (2026-05-04, prototype, default OFF)

Módulo nuevo [`rag/query_decompose.py`](rag/query_decompose.py): cuando `RAG_QUERY_DECOMPOSE=1`, `retrieve()` detecta queries multi-aspecto ("compará X vs Y", "tanto X como Y", "diferencia entre P y Q"), descompone en N sub-queries y ejecuta los retrieves en paralelo. Resultados se fusionan con [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (k=60, formula `score(d) = Σ 1/(k + rank_i(d))`). Apunta específicamente a la métrica más débil del eval: chains `chain_success`.

**Detector híbrido** (regex + LLM fallback):
  - **Regex** (sin costo LLM, ~70% de casos obvios): patterns para `X vs Y`, `compará X con Y`, `diferencia entre P y Q`, `tanto X como Y`, `X y también Y`, `X así como Y`, `qué tengo sobre X y Y`. Detecta + extrae los aspectos en una sola pasada.
  - **LLM fallback** (qwen2.5:3b vía `_helper_client`, deterministic via `HELPER_OPTIONS`): cuando el regex no matchea, una llamada JSON-only `{"is_multi_aspect": bool, "sub_queries": [...]}`. Silent-fail en timeout / parse-error → tratar como single-aspect.
  - Cache LRU 256 entries (positive + negative), thread-safe. Cleared con `clear_cache()`.

**Pre-gates** (`should_consider_decomposition`):
  - Token floor: `<6 tokens` → no descomponer (singles-fact típicos), salvo que regex explícita matchee (bypass).
  - Single-fact: `cuándo / dónde / qué hora / cuánto / quién` interrogativos cortos.
  - Conjunción dentro de nombre propio compuesto: heurística para "Juan y María".
  - Scope explícito (`folder` / `tag` / `path`): la decomposition no aporta valor.

**Wire-up** ([`rag/__init__.py`](rag/__init__.py) `retrieve()`):
  - Insertado después del paso 2 (auto-filter), antes del paso 3 (multi-query expansion).
  - Sólo activa cuando `history is None` y `variants is None` — multi-turn caller decide su propia descomposición.
  - Las N sub-retrieves corren con `concurrent.futures.ThreadPoolExecutor(max_workers=min(N, RAG_QUERY_DECOMPOSE_MAX_WORKERS=3))`. Cada worker hace `os.environ.pop("RAG_QUERY_DECOMPOSE")` antes de la llamada recursiva (evita recursión infinita).
  - RRF fuse sobre la unión de paths devueltos por cada sub-retrieve (k=60, top_k=k del caller). Tiebreak determinístico por path lex.
  - Telemetría: `extra_json.decomposed`, `n_sub_queries`, `decompose_ms` (a `to_log_event` desde `rr.timing` + `rr.filters_applied`).

**Default OFF** (env var no seteada): retrieve hot-path tiene 0% overhead — single `os.environ.get` check al inicio del bloque. Cuando ON, regex + cache lookup ~10µs en miss; LLM fallback ~500ms-1s por query con regex-miss.

**Env vars**:
  - `RAG_QUERY_DECOMPOSE=1` — activa el feature.
  - `RAG_QUERY_DECOMPOSE_LLM_FALLBACK=0` — limita a regex-only (no LLM fallback).
  - `RAG_QUERY_DECOMPOSE_MAX_WORKERS=3` — cap del threadpool.

**Tests**: [`tests/test_query_decompose.py`](tests/test_query_decompose.py) — 66 casos: regex match per pattern, LLM fallback con JSON malformado / excepción / `is_multi_aspect=false`, cache hit/eviction, RRF formula + tiebreak determinístico, pre-gates por scope, integration end-to-end con `retrieve()` mockeado.

**Eval impact** (medido 2026-05-04, queries.yaml current set, n=54 singles / 9 chains, baseline post-vault-reorg):
  - **Cero efecto en queries normales**: el detector salta tempranamente vía pre-gate (singles típicos son <6 tokens y/o single-fact interrogatives). Resultados bit-idénticos a flag-OFF en singles set.
  - **Sub-set de queries multi-aspecto**: las chains de queries.yaml NO usan patrones decomposition-friendly (mayoría son follow-ups con pronombre tipo "y la otra", "y los acordes"). Por construcción del prototype el feature no se dispara en chains existentes — es una capa para queries adicionales que el user pueda escribir tipo "compará X con Y".
  - **Regla pragmática**: feature ON queda como infra disponible, NO promovido a default hasta que (a) tengamos chains golden con patrón multi-aspecto (TODO: extender `queries.yaml`), o (b) se observe lift en producción real.

**Cuándo NO descompone** (gates duros):
  - Query corta (<6 tokens) sin regex match.
  - Filtro path/folder/tag explícito → la fusion no aporta sobre un scope ya estrecho.
  - History-aware retrieve (multi-turn): el caller decide.
  - Single-fact interrogativos.

**Sub-retrieve workers**: cada uno corre `auto_filter=False`, `multi_query=True` (paráfrasis del expand_queries habituales), `precise=False`, `hyde=False`. La intención es que cada sub-aspecto se beneficie del recall normal pero NO re-trigger decomposition (env pop scoped al worker).

**Fall-through robusto**: si todas las sub-retrieves devuelven 0 candidates, o cualquier excepción rompe el bloque, caemos al pipeline normal con la query original — silent-log a `_silent_log("query_decompose.outer", exc)` para diagnóstico sin tumbar el retrieve.

## Wave-8 gotchas — pipeline de filtros + carry-over (2026-04-28)

Tres patrones que se mordieron durante wave-8 de eval Playwright. Documentados acá porque cualquiera de los tres puede repetirse silencioso y costar sesión entera de debug.

### Filtros definidos pero no cableados

**Síntoma**: codebase tiene clase `_XxxFilter` o función `_strip_*`/`_redact_*`/`_normalize_*` con regex completo + docstring + comentario explicando bug que arregla, pero **ningún call site la invoca**. Intención es real, alguien la dejó "lista para conectar" y nunca la conectó. Bug que se suponía fixeada sigue ahí.

**Caso real**: `_strip_foreign_scripts` (`web/server.py:1504-1531`) existía con docstring "Remove characters from non-allowed scripts (CJK, Cyrillic, Hebrew, Arabic…)". Nunca se llamaba. CJK leak en respuestas de weather siguió en producción hasta wave-8.

**Cómo evitarlo en futuro**:

1. Cuando agregues filtro nuevo, también editá `_emit()` helper dentro de `gen()` (línea ~11631 de `web/server.py`) Y pipeline de cache replay (línea ~9887 — `_redact_pii(_sem_text)`).
2. Antes de "ya está, queda para wirear después", ya escribí call site. Si lo dejás para "después" no llega.
3. Hay test de regresión [`tests/test_filter_wiring.py`](tests/test_filter_wiring.py) que falla si clase `_*Filter` o función `_strip_*`/`_redact_*` está definida sin call site. Si alguna vez te marca false-positive (filter intencionalmente no usado), agregalo a allowlist del test, no lo borres.

### Carry-over del pre-router silenciosamente sobrescrito por el fast-path

**Síntoma**: agregaste lógica al inicio de `gen()` que computa `_forced_tool_pairs` (lo que pre-router decidió disparar). Log dice que se computó. Pero en respuesta tool nunca corre. Causa es que **otro branch downstream** dentro de la misma `gen()` está re-llamando `_detect_tool_intent(question)` y descartando tu `_forced_tool_pairs`.

**Caso real**: wave-8 carry-over anafórico. Pre-router setear `_forced_tool_pairs = [('weather', {'location': 'Barcelona'})]`. Línea 10996 hacía `_forced_tools = [] if _propose_intent else _detect_tool_intent(question)` que retornaba `[]` (sin la query "y en Barcelona?" no matchea ningún keyword). Tool nunca corría aunque log decía que se había decidido. Fix: esa línea ahora hace `_forced_tools = list(_forced_tool_pairs)`.

**Cómo evitarlo en futuro**:

```bash
# Antes de cerrar un fix que toque _forced_tool_pairs, grep por re-detección:
grep -n '_detect_tool_intent\|_forced_tools\s*=' web/server.py
```

Si aparece más de una asignación a `_forced_tools` o más de una llamada a `_detect_tool_intent`, **leé contexto de cada una**. Regla es: pre-router corre UNA vez al inicio de `gen()`, todo el resto del flow debe LEER de `_forced_tool_pairs`, no recomputar.

### Bumpeo de `_FILTER_VERSION` es parte del fix, no un extra

**Síntoma**: arreglaste filtro / system prompt / regex que cambia el output user-facing. Validás vía Playwright. Test reporta que bug sigue. Te volvés loco buscando bug en tu código. Causa es que **semantic cache sigue sirviendo respuestas pre-fix** porque cache key no incluye nada que tu fix haya cambiado.

**Mecanismo**: `_FILTER_VERSION` (`rag/__init__.py:4656`) está horneado dentro de `_hash_chunk_count` (línea 4659+) y usado como parte del corpus_hash que entra en cache key del semantic cache. Bumpear string invalida TODAS las entries del cache pre-fix de un saque.

**Cuándo bumpear**:

- Cambia regex que afecta tools_fired (PII redact, raw tool stripper, iberian leaks, foreign scripts, lo que sea).
- Cambia `_WEB_SYSTEM_PROMPT` o cualquier de las REGLA N.
- Cambia traducción de descriptions (weather, etc.) inyectada al CONTEXTO.
- Cualquier cambio que user con cache hit verá como "no se aplicó tu fix".

**Cuándo NO bumpear**:

- Cambios performance / refactors sin output change.
- Cambios features off-by-default (gated por env var).
- Cambios herramientas administrativas (CLI flags, scripts).

**Convención naming**: `wave<N>-<YYYY-MM-DD>` ej. `wave8-2026-04-28`. Greppable + cronológico.