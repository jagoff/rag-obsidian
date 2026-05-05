# CLAUDE.md

Local RAG sobre vault Obsidian. Layout post-split (2026-05-04): `rag/` paquete (`__init__.py` 60.2k LOC core + sub-modules `plists.py`, `cross_source_etls.py`, `postprocess.py`, `archive.py`, `anticipatory.py`, `brief_schedule.py`, `contradictions_penalty.py`, `voice_brief.py`, `whisper.py`, `wa_scheduled.py`, `wa_tasks.py`, `mmr_diversification.py`, `today_correlator.py`, `vault_health.py`, etc) + `mcp_server.py` (thin wrapper) + `web/` (FastAPI server.py 20.6k LOC + static) + `tests/` (6,031 tests, 395 archivos). Re-export pattern: `__init__.py` hace `from rag.X import *  # noqa: F401, F403` con `__all__` explícito en cada sub-módulo para preservar 100% compat (`from rag import _watch_plist`, etc).

Entry points (instalados via `uv tool install --reinstall --editable '.[entities,stt,spotify,mlx]'`):
- `rag` — CLI indexing/querying/chat/productivity/automation
- `obsidian-rag-mcp` — MCP server (`rag_query`, `rag_read_note`, `rag_list_notes`, `rag_links`, `rag_stats`)

Local-first sobre VAULT + corpus locales (sqlite-vec + Ollama + sentence-transformers). Cross-source ingesters cloud (Gmail/Calendar/Drive) requieren creds OAuth en `~/.{gmail,calendar,gdrive}-mcp/`; sin esas creds silent-fail y corpus local sigue funcionando. WhatsApp + Reminders stay local.

Python 3.13, `uv`. Runtime venv: `.venv/bin/python`. Global tool: `~/.local/share/uv/tools/obsidian-rag/`.

## MLX migration (Ola 2 completa — 2026-05-05)

Migración Ollama → MLX para los 4 LLMs locales. **Estado**: Olas 0+1+2+3 completas (cutover 2026-05-05 — default `mlx`). Pendiente: Ola 4 (eval gate validation + rollback automático si regresiona). Dispatch + estado en [vault](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fmlx-migration%2Fdispatch).

**Mapping** (todos smoke-tested OK en Apple Silicon, 2026-05-05):
- `qwen2.5:3b` (HELPER) → [`mlx-community/Qwen2.5-3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit)
- `qwen2.5:7b` (CHAT default) → [`mlx-community/Qwen2.5-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit)
- `command-r` / `qwen2.5:14b` (HQ tier) → [`mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit) — contradiction detector, brief JSON, `rag do`, HyDE
- experimental → [`mlx-community/Qwen3-4B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) — A/B vs el 3B helper

**Switch runtime**: env var `RAG_LLM_BACKEND={ollama,mlx}`. **Default `mlx` post-cutover 2026-05-05**. Rollback: `RAG_LLM_BACKEND=ollama` exportado en shell o agregado al plist en cuestión; sin cambios de código.

**Backend abstraction** en [`rag/llm_backend.py`](rag/llm_backend.py): `OllamaBackend` (legacy passthrough) + `MLXBackend` con `chat()`, `chat_stream()`, `generate()`, `list_available()` — todos funcionales. Extra opcional `mlx` en `pyproject.toml` (Apple Silicon only, marker `requires_mlx`).

**4 dispatch points en `rag/__init__.py`** (Ola 2):

| Punto | Función | Rol |
|---|---|---|
| `_TimedOllamaProxy.chat()` | proxy-wrapper | ~29 call sites (helper/summary/chat-capped); rutea a MLXBackend cuando `RAG_LLM_BACKEND=mlx`, fallback Ollama si `tools=` o `stream=` |
| `_mlx_chat_via_backend()` | adapter | Traduce kwargs shape ollama → MLXBackend |
| `_mlx_or_ollama_chat()` | dispatcher no-streaming | Exportada para call sites raw de `web/server.py` |
| `_chat_stream_dispatch()` | dispatcher streaming | Para `for chunk in ollama.chat(stream=True, ...)` — usa `MLXBackend.chat_stream()` bajo MLX |

`resolve_chat_model()` es backend-aware: consulta `MLXBackend.list_available()` cuando `RAG_LLM_BACKEND=mlx`.

**Fallbacks silenciosos bajo MLX**:
- Tool-calling (`_handle_chat_create_intent` ~línea 30140, `do()` ~37486) — MLX no tiene formato nativo de tools; fallback a Ollama automático.
- `ollama.generate(prompt='', keep_alive=0)` — unload calls; fallback a Ollama.

**Tests**: `tests/conftest.py` tiene autouse fixture `_force_ollama_backend_for_tests` que fuerza `RAG_LLM_BACKEND=ollama` por test (evita leak de shell env). Marker `requires_mlx` registrado en `pyproject.toml`.

**Embeddings (bge-m3) NO entran en este scope** — migración separada en [`99-AI/system/embedding-swap-qwen3-8b/`](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fembedding-swap-qwen3-8b%2Fplan).

**Gate de no-regresión** (Ola 4): `rag eval` con bootstrap CIs vs floor actual (singles `hit@5 53.70% [40.74, 66.67]`, chains `hit@5 72.00% [52.00, 88.00]`). Correr con ambos backends antes de flipear default en plists. CIs no-overlapping abajo del floor → NO flipear.

Doc técnica completa en [`docs/mlx-migration.md`](docs/mlx-migration.md).

## Idioma

Español rioplatense Argentina por default (voseo, vocabulario rioplatense). Regla completa en [`~/.claude/CLAUDE.md`](file:///Users/fer/.claude/CLAUDE.md). Detector pre-emit: si el output contiene `você` / `obrigad` / `essa` / `isso` / CJK / `tú` formal → bug, corregir antes de mandar.

## Agent dispatch rule

Invocar `pm` ANTES de empezar cuando se cumple AL MENOS UNO:

1. Cruza ≥2 agent domains (retrieval + brief, llm + ingestion, integrations + vault-health).
2. Toca un invariant listado en [`pm.md`](.claude/agents/pm.md): schema version `_COLLECTION_BASE`, eval floor (singles/chains CI), reranker `device="mps"` + `float32`, HELPER model binding (`reformulate_query` + `qwen2.5:3b`), confidence gates (`CONFIDENCE_RERANK_MIN`, `CONFIDENCE_DEEP_THRESHOLD`), Ollama `keep_alive=-1`, session-id regex, local-first rule.
3. Hay peers activos (`mcp__claude-peers__list_peers(scope: "repo")` > 1) Y su `set_summary` se solapa con la zona.
4. No sabés qué agent owns la work.

Skip PM cuando: edits mecánicos (rename, ruff, bump versión, typo fix), single-domain con N archivos, exploración / Q&A / review de diffs, fix trivial obvio.

Roster + ownership en [`.claude/agents/README.md`](.claude/agents/README.md).

### Custom agent profiles requieren reload de la sesión

Profiles en `.claude/agents/*.md` se cargan **una sola vez al iniciar la sesión**. Si creás un agent nuevo durante una sesión activa, esa sesión NO lo ve. Workaround: reabrir sesión, o inyectar el system prompt inline en `subagent_explore` / `subagent_general`. Mismo gotcha aplica a skills custom. Hooks en `.devin/config.json` SÍ se refrescan en runtime.

## Auto-save a `mem-vault` al cerrar tarea

Regla universal en [`~/.claude/CLAUDE.md`](file:///Users/fer/.claude/CLAUDE.md). Trigger: bug fix con root cause no obvio, decisión arquitectónica, refactor con invariantes, performance findings con números, workflow operativo nuevo, gotchas reproducibles. Tool: `mcp_call_tool(server_name="mem-vault", tool_name="memory_save", ...)` con markdown enriquecido (Contexto / Problema / Solución / Tests / Aprendido el YYYY-MM-DD + commit SHA).

## Auto-pull + commit + push rule

Cuando termino algo: `git pull → git commit → git push origin master`. Sin preguntar. Mensaje completo en español rioplatense (qué cambié, por qué, cómo medí si aplica, cómo revertir si rompe). Trailer estándar Devin al final. Si tests fallan o build rompe → NO commiteás. Excepciones: tareas exploratorias, cambios que el user pidió no commitear, trabajo a medio camino.

### Gotcha: commits locales en `master` se pushean solos

Cualquier commit en `master` aparece en `origin/master` en segundos por **otra sesión paralela** (claude-peers MCP). Implicaciones:

1. `git commit` en master = `git push` casi inmediato. No hay ventana para "commit experimental + reset si no me gusta".
2. Para experimentar sin pushear → branch dedicada (`git checkout -b experimental/<slug>`).
3. Si pushiaste algo malo: `git revert <sha>` (force-push está en deny-list). Commit malo + revert quedan ambos en log.
4. No se puede desactivar el auto-pusher desde esta sesión; coordinarlo via `mcp__claude-peers__send_message`.

## Autonomous mode

Devin tiene 4 [permission modes](https://docs.devin.ai/reference/permissions). Este proyecto está configurado para minimizar interrupciones:

1. `.devin/config.json` — permissions pre-aprobadas: allow-list (~80 reglas: git, rag, uv, pytest, sqlite3, launchctl, observabilidad, writes en repo); deny-list (6 reglas: sudo, `git reset --hard`, `git push --force`, `git branch -D`); ask-list (.env, ~/.ssh, ~/.aws, writes al vault iCloud, fetch a OpenAI/Anthropic). `rm -rf` en allow desde 2026-04-28 (repo recuperable via clone, vault protegido por ask).
2. Bypass mode (`devin --permission-mode bypass` o Shift+Tab): cero prompts salvo `deny`. Ideal para "empezá feature, tests, commit, push, siguiente".

Precedencia: org → session-grants → `.devin/config.local.json` → `.devin/config.json` → `~/.config/devin/config.json`.

Rollback: `mv .devin/config.json{,.disabled}`.

## Zsh tab-completion

Hand-written en [`completions/_rag`](completions/_rag) con descriptions + sub-grupos + helpers dinámicos. Startup nativo zsh ~10-50ms vs ~350ms del autocompletion de Click.

Instalación: `cp completions/_rag ~/.oh-my-zsh/custom/completions/_rag && rm -f ~/.zcompdump* && exec zsh`. Regenerar tras cambios al árbol Click: `.venv/bin/python scripts/gen_zsh_completion.py > completions/_rag`.

## PWA + LAN/HTTPS exposure

PWA instalable en iOS Safari → home screen, fullscreen standalone con splash custom. Wiring: [`web/static/manifest.webmanifest`](web/static/manifest.webmanifest), [`sw.js`](web/static/sw.js) (stale-while-revalidate shell, cache-first /static, network-only /api), [`pwa/register-sw.js`](web/static/pwa/register-sw.js) + [`scripts/gen_pwa_assets.py`](scripts/gen_pwa_assets.py).

**Exponer al LAN**: dos env vars emparejadas en [`com.fer.obsidian-rag-web.plist`](~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist):

- `OBSIDIAN_RAG_BIND_HOST=0.0.0.0` — uvicorn bindea a todas las interfaces (default `127.0.0.1`).
- `OBSIDIAN_RAG_ALLOW_LAN=1` — extiende CORS regex a [RFC1918](https://datatracker.ietf.org/doc/html/rfc1918) (10/8, 172.16/12, 192.168/16).

**Tradeoff iOS**: SW solo registra en secure context (HTTPS o localhost). HTTP LAN da fullscreen + icon + splash, pero NO offline cache. Para SW completo via LAN: Caddy con `tls internal` + cert root al iPhone (AirDrop + Trust en Settings).

**Seguridad**: server NO tiene auth. Solo activar en WiFi privado.

### HTTPS público vía Cloudflare Quick Tunnel

[`cloudflared tunnel --url http://localhost:8765`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/) genera URL random tipo `https://word-word-random.trycloudflare.com`. Cero cuenta, HTTPS válido. CORS adicional: `OBSIDIAN_RAG_ALLOW_TUNNEL=1`.

**Trade-off clave**: URL cambia cada restart de cloudflared. PWA guardada en iPhone se rompe. Para URL estable: named tunnel + dominio propio.

Dos servicios launchd: `com.fer.obsidian-rag-cloudflare-tunnel` (corre cloudflared) + `com.fer.obsidian-rag-cloudflare-tunnel-watcher` (tail-F del log → escribe URL a `~/.local/share/obsidian-rag/cloudflared-url.txt` + pbcopy + macOS notification). Helpers: `alias rag-url`, `alias rag-url-c` en `~/.zshrc`.

## Anticipatory Agent

Daemon `com.fer.obsidian-rag-anticipate` (10min) push proactivo a WA. 3 señales: calendar proximity ([15,90]min), temporal echo (cosine ≥0.70 vs nota >60d), stale commitment (≥7d, reusa `find_followup_loops`). Doc completo: [`docs/anticipatory-agent.md`](docs/anticipatory-agent.md).

CLI: `rag anticipate [run|explain|log] [-n N --only-sent]`, `rag silence anticipate-{calendar,echo,commitment}`. Kill switches: `RAG_ANTICIPATE_DISABLED=1`. Tabla nueva: `rag_anticipate_candidates` (analytics, todos candidates loggean).

**Footer pattern**: `proactive_push(dedup_key=<key>)` agrega `_anticipate:<key>_` al body. El listener TS parsea ese footer cuando user reacciona 👍/👎/🔇 y postea a `POST /api/anticipate/feedback` → `rag_anticipate_feedback`.

## Bot WA draft loop

Auto-aprendizaje del modelo de respuestas. Listener TS ([`/Users/fer/whatsapp-listener`](file:///Users/fer/whatsapp-listener)) genera `bot_draft` por LLM y postea al RagNet group. User responde `/si` / `/no` / `/editar <texto>`. Listener postea a `POST /api/draft/decision` → tabla `rag_draft_decisions` (append-only, retention infinita, **gold humano para fine-tunes**).

CLI: `rag draft stats [--plain]`. Activación: `WA_DRAFT_ALL_CONTACTS=1` en plist del listener.

**Bug pattern lección (2026-04-29)**: helper público (`isXEnabled()`) respetaba flag pero call site real (SQL builder de `processDraftIncoming`) la ignoraba — siempre filtraba por whitelist. Lección: cuando agregás flag de comportamiento, auditar TODOS los call sites donde el feature decide qué procesar; no alcanza con que el helper la respete.

**Latencia retrieve_only**: `loadVaultContextForDraft` timeout subido 8s→12s (commit `c160079`) por p50 ~9s del rerank cross-encoder + BM25 + embed secuenciales.

### Fine-tune drafts — DPO + LoRA

Cierra el loop: pares `(bot_draft, sent_text)` cuando user hace `/editar` se entrenan via [DPO](https://arxiv.org/abs/2305.18290) sobre Qwen2.5-7B-Instruct con LoRA r=8/alpha=16 sobre q+v projections. NO sustituye el modelo del listener — solo accesible via `POST /api/draft/preview` con `RAG_DRAFTS_FT=1`.

CLI: `rag drafts finetune [--dry-run --epochs 1 --lr 5e-6]`. Requiere ≥100 GOLD pairs (rows `decision='approved_editar'` con `sent_text != bot_draft`). Adapter en `~/.local/share/obsidian-rag/drafts_ft/`. Métricas held-out 80/20 split por draft_id: BLEU-1, similarity char-level, **preference win rate** (% donde sim(pred, chosen) > sim(pred, rejected)). Setup deps: `uv tool install --reinstall --editable '.[finetune]'`.

## Brief feedback loop

Reactions del user a briefs morning/evening/digest. Body lleva footer `_brief:<vault_relpath>_`. Listener TS detecta reaction 👍/👎/🔇 dentro de 30min → postea a `POST /api/brief/feedback` → `rag_brief_feedback`. CLI: `rag brief stats [--plain]`.

### Brief schedule auto-tuning

Si user mutea consistentemente el morning en primera hora, sistema mueve el plist a horario más tarde automáticamente. Lógica en [`rag/brief_schedule.py`](rag/brief_schedule.py): `analyze_brief_feedback(brief_kind, lookback_days=30)` lee `rag_brief_feedback`, decision rule `mutes_first_hour ≥ 3 AND mute/(mute+positive) > 0.5` → shift `+30min` iterativo dentro de bandas seguras (`morning ∈ [06:30, 09:00]`, `today ∈ [18:00, 21:00]`, `digest ∈ [21:00, 23:30]`). Tabla `rag_brief_schedule_prefs` (single-row-per-kind, upsert). `_services_spec()` lee la pref antes de generar cada plist.

CLI: `rag brief schedule [status|reset|auto-tune] [--apply --kind morning|today|digest|all]`. Daemon: `com.fer.obsidian-rag-brief-auto-tune` Domingo 03:00.

## Voice brief

Phase 2.C: morning brief sintetiza voice note OGG/Opus + manda al WA antes del texto. Pipeline ([`rag/voice_brief.py`](rag/voice_brief.py)): strip markdown → `say -v Mónica --file-format=AIFF` → `ffmpeg libopus 24k 16kHz mono` → cache `~/.local/share/obsidian-rag/voice_briefs/YYYY-MM-DD-morning.ogg` (idempotente).

Caps: texto >4000 chars → trim, audio >5MB → fallback text-only, sin `say` o `ffmpeg` → degrade graceful. Footer `_brief:<path>_` queda intacto en el texto.

CLI: `rag voice-brief generate --date YYYY-MM-DD [--apply --voice "Diego" --text "..."]`, `rag morning --voice`. Activar daemon: `RAG_MORNING_VOICE=1` en `com.fer.obsidian-rag-morning.plist`. Cleanup auto: `rag maintenance` borra audios >30d.

## Whisper learning loop

Sistema transcripción audios WA aprende del corpus + correcciones. Plan completo: [vault](obsidian://open?vault=Notes&file=04-Archive%2F99-obsidian-system%2F99-AI%2Fsystem%2Fwhatsapp-whisper-learning%2Fplan).

3 surfaces: (1) **Pasivo** — daemon `com.fer.obsidian-rag-whisper-vocab` (03:15) → `rag_whisper_vocab` (caps por source: 100 corrections, 100 contacts, 200 notes, 100 chats). (2) **Explícito** — `/fix <texto>` por WA marca última transcripción gold. (3) **Confidence-gated LLM correct** — si `avg_logprob < -0.8`, listener pasa output por `qwen2.5:7b` con sysprompt + few-shot + vocab hints.

3 tablas SQL: `rag_audio_transcripts` (con `corrected_text`, `correction_source`, `avg_logprob`, `audio_hash`), `rag_audio_corrections` (append-only), `rag_whisper_vocab` (refresh full nightly).

CLI: `rag whisper [stats|vocab refresh|vocab show|patterns|export|import]`. WA cmds: `/fix`, `/whisper [stats|recent N]`. Dashboard: [/transcripts](https://ra.ai/transcripts) (server-rendered, dark mode default).

Env tuning en plist listener: `WHISPER_LLM_CORRECT_THRESHOLD=-0.8`, `WHISPER_LLM_CORRECT_MODEL=qwen2.5:7b`. Kill switches: `WHISPER_LLM_CORRECT_DISABLE=1`, `WHISPER_TELEMETRY_DISABLE=1`.

## Web chat tool-calling

[`web/tools.py`](web/tools.py) — 9 tools: `search_vault`, `read_note`, `reminders_due`, `gmail_recent`, `finance_summary`, `calendar_ahead`, `weather` (read-only) + `propose_reminder`, `propose_calendar_event` (create-intent, impl en `rag/__init__.py`).

`/api/chat` corre 2-phase loop: pre-router (`_detect_tool_intent`, keyword → forced read tool) + optional LLM tool-decide round (gated por `RAG_WEB_TOOL_LLM_DECIDE`, default OFF). Create intent ("recordame", "creá evento") detectado por `_detect_propose_intent` (shared web + CLI) FORZA LLM decide round.

Create tools auto-crean si datetime es unambiguo (SSE `created` event → chip `╌ ✓ agregado...`, reminders con inline `deshacer` link via `DELETE /api/reminders/{id}`, eventos NO porque Calendar.app AppleScript delete es unreliable). Si parser flagea `needs_clarification` → `proposal` card con ✓ Crear / ✗ Descartar.

Helpers en `rag/__init__.py`: `_parse_natural_datetime` (dateparser + qwen2.5:3b fallback + `_preprocess_rioplatense_datetime`), `_parse_natural_recurrence`, `_create_reminder` (due_dt/priority/notes/recurrence), `_create_calendar_event` (Calendar.app AppleScript), `_has_explicit_time` (auto all-day), `_delete_reminder`, `_delete_calendar_event`.

**Rioplatense datetime normalization** (`_preprocess_rioplatense_datetime`): regex rewrites pre-dateparser. `18hs` → `18:00`, `al mediodía` → `12:00`, `X que viene` → bare weekday/`next week`, `el|este|próximo <weekday>` → bare English (dateparser 1.4 rechaza `next <weekday>` pero acepta bare con `PREFER_DATES_FROM=future`), `pasado mañana` → `day after tomorrow`, `tipo N`/`a eso de las N` → `N:00`, `el finde` → `saturday`. Anchor-echo guard post-dateparser: si input tenía time marker pero output = anchor time, fall through a LLM.

**CLI chat create-intent** (`rag chat`): mismo `_detect_propose_intent` ported a terminal via `_handle_chat_create_intent`. Single-round ollama tool-decide con `_CHAT_CREATE_OVERRIDE` prompt + tools=[propose_reminder, propose_calendar_event] only. command-r `{parameters: {...}}` arg wrapping unwrappeado. Returns `(handled, created_info)` con `{kind, reminder_id, title}` (None para events). Stash en `last_created` + `/undo` slash command dispatcha `_delete_reminder(reminder_id)`. Tests: [`tests/test_chat_create_handler.py`](tests/test_chat_create_handler.py) (8 cases) + [`tests/test_chat_undo.py`](tests/test_chat_undo.py) (5 cases).

**Sessions**: JSON per session en `sessions/<id>.json`. TTL 30d, cap 50 turns, history window 6. IDs validados `^[A-Za-z0-9_.:-]{1,64}$`; invalid → mint fresh. WhatsApp pasa `wa:<jid>`.

**Quick Win #5 — Selective history summarisation** (`_summarize_conversation_history`, 2026-05-04): cuando `len(history) > 2` AND `RAG_HISTORY_SUMMARY != "0"`, `/api/chat` comprime turnos previos (N-1) via qwen2.5:3b en 2-3 sentences; último turno keep verbatim. Cache `rag_conversation_summaries(session_id, history_hash, summary, ts)` retention 30d. Silent-fail a raw concat. Tests: [`tests/test_history_summary.py`](tests/test_history_summary.py).

**Quick Win #4 — LLM typo correction** (`_correct_typos_llm`, 2026-05-04): pre-embed step en `expand_queries()`. qwen2.5:3b con HELPER_OPTIONS deterministic. LRU 256, sanity check len>1.5× → preserve original. Telemetry: `llm_typo_*` en `rag_queries.extra_json`. Tests: [`tests/test_typo_correction.py`](tests/test_typo_correction.py).

**Episodic memory** ([`web/conversation_writer.py`](web/conversation_writer.py), silent write): post `/api/chat` `done` event, daemon thread append a `04-Archive/99-obsidian-system/99-AI/conversations/YYYY-MM-DD-HHMM-<slug>.md`. One note per `session_id`, multi-turn. Index `session_id → relative_path` en `rag_conversations_index` (SQL upsert). Atomic .md write via `os.replace`. **Excluido del index** (`is_excluded`: prefix `04-Archive/99-obsidian-system/` + legacy `00-Inbox/conversations/` + `04-Archive/conversations/`) — leak hallucinations back si indexado.

**Shutdown drain** (`_CONV_WRITERS` + `@app.on_event("shutdown")`): cada writer in-flight registra. `_drain_conversation_writers` joins con 5s budget. Stragglers → `_CONV_PENDING_PATH` (`conversation_turn_pending.jsonl`) → re-aplicado en próximo startup por `_retry_pending_conversation_turns`. Threads daemon=True por design (wedged write no debe bloquear exit).

**Phase 2 consolidation** (`scripts/consolidate_conversations.py`, `rag consolidate`, weekly launchd): scan rolling window (default 14d), embed cada conv como `first_question + first_answer` via bge-m3, group por connected components cosine ≥0.75, promote clusters ≥3 a PARA. Target folder: ≥2 matches `_PROJECT_PATTERNS` → `01-Projects/`, else `03-Resources/`. Synthesis via `resolve_chat_model()` + CHAT_OPTIONS (un call por cluster ~6s). Originals move a `04-Archive/conversations/YYYY-MM/`. Errores per-cluster swallowed.

## Feature H — Chat scoped a nota/folder

Selector compacto en composer del chat web ([`web/static/index.html`](web/static/index.html) + [`app.js`](web/static/app.js)): botón ◉ → popover con autocomplete → click setea scope en `sessionStorage` + chip "🎯 Limitado a: `<path>` ×". JS monkey-patchea `fetch` para inyectar `path` o `folder` solo en POST `/api/chat`.

Backend ([`web/server.py`](web/server.py) `# ── Feature H`): `ChatRequest` acepta `folder` + `path` (validators rechazan URI schemes y `..`). `multi_retrieve(...)` recibe `folder` como 4to posicional. Si viene `path`, filtro post-retrieve exact-match contra `meta.file`. Short-circuit cuando no hay matches: SSE `sources(confidence=0)` + canned token + `done(scope_no_match=True)`. NO 404.

Endpoint nuevo `GET /api/notes/autocomplete?q=&limit=20`: substring matching case-insensitive contra `meta.file`/`meta.note`/`meta.folder`. Sortea exact → startswith → contains-path → contains-title → folder. Limit clamped 50. Empty corpus → `{items: [], reason: "empty_index"}`. Rate-limit reusa `_BEHAVIOR_BUCKETS` (120 req/60s).

Telemetry: `result["filters_applied"]["path_scope"]` distingue scope explícito de auto-filter. Bucket nuevo: `web.chat.scope_no_match`. Tests: [`tests/test_chat_scoped.py`](tests/test_chat_scoped.py) (11 cases).

## Feature K — "Recordame X" inline en chat

Detecta comandos tipo "recordame llamar a Juan mañana 9am" → crea reminder Apple Reminders **sin LLM**, devuelve SSE `created` event en <100ms vs 5-15s del flow LLM+tools.

Detector ([`rag/__init__.py`](rag/__init__.py) `# ══ Feature K`): `parse_remind_intent(text) → dict | None`. Pattern strict-leading: `^(recordame|recuerdame|acordate|hacéme acordar|reminder|remember me|remind me) [de/que] <rest>$`. Sobre `<rest>`, primer marker temporal con `_REMIND_TIME_MARKERS_RE` parte título/cuándo. Reusa `_parse_natural_datetime`. Anchor-echo guard.

Wire-up ([`web/server.py`](web/server.py) `# ══ Feature K`): ANTES del flow normal, llama `parse_remind_intent(question)`. Match → `_create_reminder(title, due_dt=...)` directo + SSE `sources(confidence=1, intent=remind_inline)` + `created(kind=reminder, remind_inline=True)` + canned token + `done(mode=remind_inline)`. Si `_create_reminder` falla → `proposal(needs_clarification=True)`. NO match → fall-through al flow normal (donde `propose_reminder` tool sigue funcionando).

Telemetry bucket: `web.chat.remind_inline`. Tests: [`tests/test_chat_remind_inline.py`](tests/test_chat_remind_inline.py) (10 cases).

## Implicit feedback (reward shaping con negativos débiles)

`rag feedback classify-sessions` backpropaga outcome de session → cada turn. Branches: `win` → `rating=+1`, `loss` → `-1`, `partial` → skip, `abandon` con `top_score < 0.4` → `rating=-1` source `session_outcome_weak_negative`. Treatment training: weight=0.3 (constante `WEAK_NEGATIVE_TRAINING_WEIGHT`) → lambdarank gradient penaliza la mitad. Pre-fix había 542 abandons / 18 losses (asymetría 30:1); post-fix absorbe ~50-100 negativos débiles/semana sin contaminar positivos.

Configs in-code (no env var): `WEAK_NEGATIVE_TOP_SCORE_THRESHOLD=0.4`, `WEAK_NEGATIVE_TRAINING_WEIGHT=0.3`.

## Commands

```bash
uv tool install --reinstall --editable '.[entities,stt,spotify,mlx]'

# Core
rag index [--reset] [--no-contradict] [--vault NAME]
rag index --source whatsapp|contacts|calls|safari|reminders|gmail|calendar|drive|pillow [--reset --since ISO --dry-run --max-N N]
rag watch                                                        # filesystem watcher
rag query "text" [--hyde --no-multi --raw --loose --force --counter --no-deep --session ID --plain --source S[,S2] --vault NAME]
rag chat [--counter --no-deep --session ID --resume]            # /save /reindex /undo
rag do "instrucción" [--yes --max-iterations 8]                 # tool-calling agent loop
rag stats
rag session list|show|clear|cleanup

# Productivity
rag capture "texto" [--tag X --source Y --stdin --title T --plain]
rag inbox [--apply]
rag prep "tema" [--save]
rag read <url> [--save --plain]
rag dupes [--threshold 0.85 --folder X]
rag links "query" [--open N --rebuild]
rag wikilinks suggest [--folder X --apply]
rag followup [--days 30 --status stale|activo|resolved --json]
rag dead [--min-age-days 365]
rag archive [--apply --force --gate 20]

# Daily automation
rag morning [--dry-run --voice]
rag today [--dry-run]
rag digest [--week YYYY-WNN --days N]
rag consolidate [--window-days 14 --threshold 0.75 --min-cluster 3 --dry-run --json]

# Ambient + Anticipatory
rag ambient status|disable|test [path]|log [-n N]
rag ambient folders list|add <F>|remove <F>
rag anticipate [run|explain|log] [-n 20 --only-sent --dry-run --force]
rag silence anticipate-{calendar,echo,commitment} [--off]

# Quality
rag eval [--latency --max-p95-ms N]
rag tune [--samples 500 --apply --online --days 14 --rollback]
rag log [-n 20 --low-confidence]
rag dashboard [--days 30]
rag feedback status|backfill|infer-implicit|harvest [...]
rag behavior backfill [--dry-run --window-minutes N --limit N]
rag open <path> [--query Q --rank N --source cli] | rag open --nth N [--session ID]
rag replay <query_id> [--diff|--explain] [--skip-gen] [--no-cache] [--force] [--plain|--json]
rag replay --bulk [--since 7d] [--limit 20] [--filter-cmd CMD] [--skip-gen] [--plain|--json]

# Maintenance
rag maintenance [--dry-run --skip-reindex --skip-logs --json]
rag free [--apply --yes --force --json --min-age-days N --ranker-keep N --skip-...]
python scripts/backfill_entities.py [--dry-run --limit N --vault NAME]
python scripts/audit_telemetry_health.py [--days 7 --json]      # PRIMER comando antes de "auditá el sistema"

rag implicit-feedback [--days 14 --json]
rag routing-rules [--reset --debug --json]
rag whisper-vocab [--refresh --show --source X --limit N]
rag vault-cleanup [--dry-run --apply --force]
rag health [--since HOURS --as-json]
rag trends [--days N --top N --as-json]
rag hygiene [--empty-threshold N --stale-days N --sample N --as-json]
rag state [texto] [--clear --plain]
rag config [--only-set --filter PATTERN --as-json]
rag pendientes [--days N --plain]
rag contact-note NOMBRE OBSERVACION [--category X --source-kind X]

# Automation + Tests
rag setup [--remove]
rag stop [--all] [--with-rag-net/--without-rag-net] [--with-ollama] [--with-qdrant] [-y --dry-run]
rag daemons [status|reconcile|doctor|retry <label>|kickstart-overdue] [--apply --dry-run --gentle --json --unhealthy-only]

.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_foo.py::test_bar -q
```

## Env vars críticas

Catálogo completo (47+ vars adicionales): [`docs/env-vars-catalog.md`](docs/env-vars-catalog.md). Esta sección cubre las críticas con rollback paths.

**Vault + ingest**:
- `OBSIDIAN_RAG_VAULT` — override vault path. Multi-vault: el `current` del registry pierde contra esta env. Cross-source ETLs gated por `_is_cross_source_target(vault_path)` — solo `_DEFAULT_VAULT` recibe los 11 ETLs salvo opt-in en `~/.config/obsidian-rag/vaults.json`.
- `RAG_OCR=0` — desactiva OCR de imágenes embebidas (default ON cuando `ocrmac` disponible). Cache en `rag_ocr_cache`. Soft-dep macOS-only.
- `OBSIDIAN_RAG_MOZE_DIR` — iCloud source dir CSV exports MOZE. Default Tally4 4.x usa `.realm` zip → bridge en [`rag/integrations/tally4_realm.py`](rag/integrations/tally4_realm.py) extrae a CSV vía Node + `realm` npm.
- `OBSIDIAN_RAG_FINANCE_DIR` — credit-card xlsx + bank PDFs.
- `OBSIDIAN_RAG_INDEX_WA_MONTHLY=1` — opt-in al double-indexing de WA monthly rollups (default OFF post-2026-04-22).

**Performance + memoria**:
- `OLLAMA_KEEP_ALIVE` — default `-1` (forever). Auto-clamp via `chat_keep_alive()` a `_LARGE_KEEP_ALIVE="20m"` cuando chat model está en `_LARGE_CHAT_MODELS` (command-r, qwen3:30b-a3b). Override: `RAG_KEEP_ALIVE_LARGE_MODEL`.
- `RAG_MEMORY_PRESSURE_DISABLE=1` — desactiva watchdog (default ON). Threshold 85% (override `RAG_MEMORY_PRESSURE_THRESHOLD`), interval 60s (override `RAG_MEMORY_PRESSURE_INTERVAL`). Bajo pressure: unload chat (`keep_alive=0`) + force-unload reranker (bypassa TTL idle + `RAG_RERANKER_NEVER_UNLOAD`).
- `RAG_RERANKER_NEVER_UNLOAD=1` — pina reranker en MPS VRAM permanente (set en plists web + serve). Cost ~2-3 GB pinned.
- `RAG_RERANKER_IDLE_TTL` — segundos antes de idle-unload (default 900).
- `RAG_LOCAL_EMBED=1` — in-process `SentenceTransformer('BAAI/bge-m3')` para queries (set en plists web + serve, auto-set en CLI query-like). Pre-condition: model cacheado en `~/.cache/huggingface/hub/`. NO setear en indexing/watch.
- `RAG_LOCAL_EMBED_WAIT_MS` — budget wait Event ready antes de fallback ollama (default 6000ms tras 2026-04-23, antes 4000ms causaba timeout exacto en cold load).

**Async writers** (default ON desde audit 2026-04-24):
- `RAG_LOG_QUERY_ASYNC=0`, `RAG_LOG_BEHAVIOR_ASYNC=0`, `RAG_METRICS_ASYNC=0`, `RAG_LOG_FT_RATING_ASYNC=0`, `RAG_LOG_AMBIENT_ASYNC=0`, `RAG_LOG_CONTRADICTIONS_ASYNC=0`, `RAG_LOG_ARCHIVE_ASYNC=0`, `RAG_LOG_TUNE_ASYNC=0`, `RAG_LOG_SURFACE_ASYNC=0` — opt-out al sync (tests setean `0` en conftest).

**Retrieval pipeline**:
- `RAG_ADAPTIVE_ROUTING` — default ON tras 2026-04-22. Skip helper reformulate para intents metadata-only + fast-path con `qwen2.5:3b num_ctx=4096` cuando `top_score >= 0.6`. Rollback: `=0`.
- `RAG_LOOKUP_NUM_CTX` — fast-path ctx (default 4096 tras 2026-04-22; pre-fix 2048 causaba refuses falsos por truncation).
- `RAG_FAST_PATH_KEEP_WITH_TOOLS=1` — rollback del downgrade cuando fast-path matchea tools deterministas (default OFF / downgrade activo). Marker telemetry `fast_path_downgraded=True`.
- `RAG_ENTITY_LOOKUP` — default ON tras backfill 2026-04-21. Dispatch a `handle_entity_lookup()` para intent `entity_lookup`. Rollback: `=0`.
- `RAG_EXTRACT_ENTITIES` — default ON tras 2026-04-21. Popula `rag_entities` + `rag_entity_mentions` durante indexing. Costo ~0.16s/chunk + cold-load GLiNER ~5s. Sticky-fail si gliner missing.
- `RAG_EXPLORE=1` — ε-exploration (10% swap top-3 ↔ rank 4-7). Set en morning/today plists. MUST be unset durante `rag eval` (comando lo `os.environ.pop`s).
- `RAG_EXPAND_MIN_TOKENS` — threshold short-query gate (default 4). Queries más cortas skipean qwen2.5:3b paraphrase.
- `RAG_CITATION_REPAIR_MAX_BAD` — threshold repair gate (default 2). Set `0` para disable.
- `RAG_DEEP_MAX_SECONDS` — wall-time cap auto-deep (default 30s).
- `RAG_NLI_GROUNDING` — default OFF. Carga mDeBERTa, verifica claims post-citation-repair, emite SSE `nli_grounding`. Idle-unload TTL `RAG_NLI_IDLE_TTL=900`. NO activar en producción aún.
- `RAG_NLI_MODE` — citation NLI sentence verifier: `off|mark|strip` (default `off`). `RAG_NLI_THRESHOLD=0.5`. Pre-condition: `huggingface-cli download cross-encoder/nli-deberta-v3-small`.
- `RAG_CONTRADICTION_PENALTY` — default ON. Demote post-rerank a chunks en `rag_contradictions` (skip si `--counter`). Magnitude `RAG_CONTRADICTION_PENALTY_MAGNITUDE=0.05`. Cache 5min.
- `RAG_MMR` — default OFF. MMR diversification embedding-based post-rerank. `RAG_MMR_LAMBDA=0.7`, `RAG_MMR_TOP_K=10`. Skipea si `counter=True`. Variante cheap: `RAG_MMR_FOLDER_PENALTY=1` (mutuamente exclusiva).
- `RAG_LLM_JUDGE` — default OFF (prototipo 2026-05-04). Cuando `top_score < RAG_LLM_JUDGE_THRESHOLD=0.5` AND `len(candidates) >= RAG_LLM_JUDGE_MIN_CANDIDATES=5`, qwen2.5:3b scorea top-20 0-10 y blendea con CE vía `α * ce + (1-α) * (llm/10)`, α=`RAG_LLM_JUDGE_ALPHA=0.5`. Skip implícito en `fast_path_taken`, `propose_intent`, scope estrecho. Telemetría: `llm_judge_*` en `rag_queries.extra_json`. Eval validation pendiente antes de promover.
- `RAG_QUERY_DECOMPOSE=1` — default OFF (prototipo 2026-05-04). Decompone queries multi-aspecto + RRF fusion. Detector híbrido (regex + LLM fallback en miss). Pre-gates: <6 tokens, single-fact, scope explícito → no descompone. `RAG_QUERY_DECOMPOSE_LLM_FALLBACK=0`, `RAG_QUERY_DECOMPOSE_MAX_WORKERS=3`. Eval lift no medible en chains golden actuales (no tienen pattern multi-aspecto).
- `RAG_INTENT_RECENCY` — default ON (Quick Win #3, 2026-05-04). Halflife multipliers per intent: recent ×0.3, historical ×3.0, neutral ×1.0. Vault/Calendar/Contacts always None (atemporales). Detector regex puro ES+EN. Telemetría: `temporal_intent` en `rag_queries.extra_json`. Rollback: `=0`.
- `RAG_TYPO_CORRECTION` — default ON (Quick Win #4, 2026-05-04). qwen2.5:3b pre-embed para typos ("asor" → "Astor"). LRU 256, sanity check len>1.5× → preserve original. Silent-fail.
- `RAG_HISTORY_SUMMARY` — default ON (Quick Win #5, 2026-05-04). Cuando `len(history) > 2`, qwen2.5:3b resume turnos previos en 2-3 sentences. Cache `rag_conversation_summaries` PK `(session_id, history_hash)`, retention 30d.
- `RAG_ANAPHORA_RESOLVER` — default ON (Quick Win #1, 2026-05-04). Detector regex (microsegundos): True cuando `len(history) >= 1` AND query <8 tokens OR empieza con conector. Resolver qwen2.5:3b con LRU 128. Clamp si helper devuelve >3× input. Telemetría: `anaphora_*` en `rag_queries.extra_json`.
- `RAG_CONTEXTUAL_RETRIEVAL=1` — default OFF (prototipo 2026-05-04, [Anthropic technique](https://www.anthropic.com/news/contextual-retrieval)). Re-embed full requiere ~25-40min. Cache `rag_chunk_contexts` PK `(doc_id, chunk_idx, chunk_hash)` sobrevive a `--reset`. CLI: `rag index --contextual`. Validation pendiente vs `rag eval`.
- `RAG_WA_FAST_PATH` / `RAG_WA_FAST_PATH_THRESHOLD=0.05` — fast-path WhatsApp (default ON). Branch 1: caller explícito `source="whatsapp"` → bypass score gate. Branch 2: ≥2 de top-3 metas WA AND top-1 > threshold.
- `RAG_WA_SKIP_PARAPHRASE` — default ON. Skip `expand_queries()` cuando único source es WhatsApp.

**Ranker + cache**:
- `RAG_TRACK_OPENS=1` — switches OSC 8 a `x-rag-open://` para ranker-vivo signal capture.
- `RAG_RERANKER_FT=1` — opt-in LoRA adapter cross-encoder en `~/.local/share/obsidian-rag/reranker_ft/`. Default OFF. Failure modes silent_fail con fallback a base. Distinto de `RAG_RERANKER_FT_PATH` (full FT via symlink).
- `RAG_FINETUNE_MIN_CORRECTIVES` — default 20. Aborta `scripts/finetune_reranker.py` con exit 5 si la señal limpia es insuficiente.
- `RAG_DRAFT_VIA_RAGNET=1` — legacy override para redirigir TODOS los ambient sends a RagNet en lugar de destinatarios reales. Útil para testing/debugging acotado.

**Replay + privacy**:
- `RAG_LOG_REPLAY_PAYLOAD=1` — default OFF. Cuando ON, persiste `response_text` (cap 8 KB) + `history_snapshot` (cap 4 KB) en `rag_queries.extra_json`. Contiene PII del vault — opt-in explícito requerido. Los hashes (`corpus_hash`, `response_hash`, `prompt_hash`, `history_hash`) SIEMPRE se persisten (16 chars hex, ~48 bytes/row overhead) independientemente de este flag.
- `RAG_LOG_RERANK_RAW=1` — default OFF. Cuando ON, persiste `rerank_logits_raw: list[float]` en `rag_queries.extra_json`. Útil sólo para debugging de regresiones del ranker. Combinado con `RAG_LOG_REPLAY_PAYLOAD=1` el storage estimado es ~3.5 MB/sem (50 queries/día × 2.3 KB/row). Storage con ambas flags OFF: ~30 KB/sem (sólo los 3 hashes).
- Helpers en `rag/__init__.py`: `_replay_hash(text)`, `_truncate_for_replay(text, max_bytes)`, `_build_replay_fields(*, response, history, prompt, corpus_hash, rerank_logits)`. Callers: CLI `query()` (usa `_cache_hash`), CLI `chat()` (usa `_corpus_hash_cached(col)` + `history[:-2]`), `rag serve` main path (usa `_serve_sem_hash`), `web/server.py gen()` (usa `_semantic_cache_hash`).
- Tests: `tests/test_replay_payload.py` (28 casos: shape con/sin flags, truncation UTF-8, hashes determinísticos, back-compat rows legacy, truthy/falsy variants).

**Misc**:
- `OBSIDIAN_RAG_NO_APPLE=1` — desactiva integraciones Apple (Calendar/Reminders/Mail/ScreenTime).
- `RAG_TIMEZONE` — IANA tz para `_parse_natural_datetime` con tzinfo (default `America/Argentina/Buenos_Aires`).
- `OBSIDIAN_RAG_WATCH_EXCLUDE_FOLDERS` — comma-separated folders ignorados por `rag watch` (default `"04-Archive/99-obsidian-system/99-AI/external-ingest/WhatsApp"`).

**Dev/debug** (NO en producción): `RAG_DEBUG=1`, `RAG_RETRIEVE_TIMING=1`, `RAG_NO_WARMUP=1`, `OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY=1`, `OBSIDIAN_RAG_SKIP_SYNTHETIC_Q=1`.

## Architecture — invariants

### Retrieval pipeline (`retrieve()`)

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

**Rerank pool** (`RERANK_POOL_MAX = 15`, dropped from 30 on 2026-04-21): pool=15 domina vs 30 — hit@5 idéntico, MRR chains +5pp, P95 singles -66%. Web `/api/chat` pasa `rerank_pool=5` explícito.

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

### Indexing

Chunks 150-800 chars, split on headers + blank lines, merged si <MIN_CHUNK. Hash per file → re-embed only on change. `is_excluded()` skips `.`-prefixed segments.

**Contextual embeddings** (v9 actual): `get_context_summary()` genera 1-2 sentences per note via qwen2.5:3b, prepended a cada chunk's `embed_text` como `Contexto: ...`. Cached por file hash.

**`created_ts` backfill marker**: persistido en `rag_schema_version` (sentinel `_created_ts_backfill_complete`). Pre-fix re-escaneaba 3600+ chunks por restart del web daemon (149 restarts en ~3 días).

**Schema changes**: bump `_COLLECTION_BASE` (currently `obsidian_notes_v11`). Per-vault suffix = sha256[:8] of resolved path.

### Model stack

| Role | Model | Notes |
|---|---|---|
| Chat | `resolve_chat_model()`: qwen2.5:7b > qwen3:30b-a3b > command-r > qwen2.5:14b > phi4 | qwen2.5:7b default tras bench 2026-04-18 (P50 5.9s vs 37s command-r). |
| Helper | `qwen2.5:3b` | paraphrase/HyDE/reformulation; deterministic via `HELPER_OPTIONS = {temperature: 0, seed: 42}` |
| Embed | `bge-m3` | 1024-dim multilingual |
| Reranker | `BAAI/bge-reranker-v2-m3` | `device="mps"` + `float32` forced. **NO switch fp16** — 2 A/Bs failed (collapse 2026-04-13, overhead 2x con calidad equivalente 2026-04-22). |
| NLI grounding (opt-in) | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | ~400 MB MPS fp32, idle-unload via `RAG_NLI_IDLE_TTL`. |
| Citation NLI verifier (opt-in) | `cross-encoder/nli-deberta-v3-small` | ~80 MB, lazy-load sticky-fail. |

`CHAT_OPTIONS`: `num_ctx=4096, num_predict=384`. Don't bump unless prompts grow.

### Confidence gate

`top_score < 0.015` (CONFIDENCE_RERANK_MIN) + no `--force` → refuse sin LLM call. Per-source override scaffolding: `CONFIDENCE_RERANK_MIN_PER_SOURCE` dict (todos =baseline 0.015 hoy) + helper `confidence_threshold_for_source(source)`. Re-calibrate cuando ingesters tengan ≥1 semana de feedback.

### Generation prompts

- `SYSTEM_RULES_STRICT` (default `rag query` semantic): forbids external prose.
- `SYSTEM_RULES` (`--loose`, always en chat): allows `<<ext>>...<</ext>>` rendered dim yellow + ⚠.
- `SYSTEM_RULES_LOOKUP` (intent count/list/recent/agenda): terse 1-2 sentences, exact "No encontré esto en el vault." refusal.
- `SYSTEM_RULES_SYNTHESIS` (intent synthesis): cross-reference ≥2 sources, surface tension. Fires via `_INTENT_SYNTHESIS_RE`.
- `SYSTEM_RULES_COMPARISON` (intent comparison): `X dice A / Y dice B / Diferencia clave`. Fires via `_INTENT_COMPARISON_RE`. Checked BEFORE synthesis.
- Routed via `system_prompt_for_intent(intent, loose)`.

### Agenda intent

Fired by `_INTENT_AGENDA_RE`, checked **before `recent`** (compartían tokens temporales). `handle_agenda(col, params, limit=20, *, question=None)` filtra por `source ∈ _AGENDA_SOURCES = {"calendar", "reminders"}`, sort por `created_ts` desc.

**Window filter** `_parse_agenda_window(question, *, now=None) → (ts_start, ts_end) | None`: dispatch order narrowest first (day anchors → weekend → week → month → year → weekday-specific). Half-open [start, end). Snap a 00:00 local.

### Prompt-injection defence (passive)

Dos layers en `rag/__init__.py` (sobre `SYSTEM_RULES`):

- **Redaction** `_redact_sensitive(text)` — strip OTPs/tokens/passwords/CBU/cards antes de chunk → LLM. Cue-gated (value next to `code|token|password|cbu|cvv`) con digit-presence lookahead.
- **Context isolation** `_format_chunk_for_llm(doc, meta, role)` — wrappea body en `<<<CHUNK>>>...<<<END_CHUNK>>>`. Paired with `_CHUNK_AS_DATA_RULE` (REGLA 0) en cada `SYSTEM_RULES*`.

NOT a barrier vs motivated attacker con vault write access — hint a la classifier.

### Name-preservation guardrail

`_NAME_PRESERVATION_RULE` (después de `_CHUNK_AS_DATA_RULE`): bloquea LLM "corrigiendo" proper nouns que no reconoce. Regression seed: "Bizarrap" → "Bizarra". Verify: `python -c "import rag; print(rag._NAME_PRESERVATION_RULE[:80])"`.

### Response-quality post-pipeline

- **Citation-repair** (always-on): `verify_citations(full, metas)` flags invented paths. ONE repair call si non-empty + n_bad ≤ 2. Logs `citation_repaired: bool`.
- **`--critique` flag** (opt-in): segundo non-streaming chat-model call evalúa + regenera. Logs `critique_fired/changed`.
- **Citation NLI verifier** (`RAG_NLI_MODE`): sentence-level entailment via `cross-encoder/nli-deberta-v3-small`. Modes: off | mark (`(?)` suffix) | strip. Implementado en `rag/postprocess.py`.

### Scoring formula (post-rerank)

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

### GC#2.C — Reranker fine-tune (gated on data)

Infra completa + gate E2E validado, esperando ≥20 rows con `corrective_path` en `rag_feedback`. Runs anteriores fallaron (-3.3pp chains hit@5) por señal positiva ruidosa. Fix: [`scripts/finetune_reranker.py`](scripts/finetune_reranker.py) lee `corrective_path` del `extra_json` y lo usa como único positivo; fallback a todos paths cuando no.

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

[Anthropic technique](https://www.anthropic.com/news/contextual-retrieval): qwen2.5:3b genera summary corto (≤100 tokens) que ubica el chunk en su documento, prepended al embed_text antes de embed. Módulo: [`rag/contextual_retrieval.py`](rag/contextual_retrieval.py).

Wire-up en `_index_single_file` + `_run_index`. CLI: `rag index --contextual` setea env por invocación. Cache `rag_chunk_contexts` PK `(doc_id, chunk_idx, chunk_hash)`, sobrevive `--reset`. Display_texts NO se mutan.

Promote checklist: (1) `RAG_CONTEXTUAL_RETRIEVAL=1 rag index --reset --contextual`, (2) `rag eval` con CI bootstrap, (3) si singles+chains hit@5 mejoran fuera del CI noise → promote default ON. Diferente de `get_context_summary` (per-doc summary compartido a TODOS los chunks de la nota); este es per-chunk.

## Eval baselines

**Floor actual (2026-04-27, post-golden-remap vault reorg, commit `6f8994f`)**:

- Singles: `hit@5 53.70% [40.74, 66.67] · MRR 0.528 [0.407, 0.657] · n=54`
- Chains: `hit@5 72.00% [52.00, 88.00] · MRR 0.633-0.653 · chain_success 33.33% [11.11, 66.67] · turns=25 chains=9`
- **Lower-CI-bound gate** (nightly online-tune auto-rollback): singles < 40.74% OR chains < 52.00%

La caída vs floor previo (singles 71.67% → 53.70%, chains 86.67% → 72.00%) NO es regresión del pipeline — es reducción del n + remoción de goldens fáciles que ya no existen post-vault-reorg.

`rag eval --latency --max-p95-ms N` agrega P50/P95/P99 + CI gate. Bootstrap: 1000 resamples, seed=42.

Helper LLM calls (`expand_queries`, `reformulate_query`, `_judge_sufficiency`) ya son determinísticos via `HELPER_OPTIONS`. **HyDE con qwen2.5:3b drops singles hit@5 ~5pp** — opt-in via `--hyde`.

**`seen_titles` post-rerank penalty** (`SEEN_TITLE_PENALTY=0.1`): candidates cuya `meta.note` (case-insensitive) matchea cualquier `seen_titles` entry pierden 0.1. Diversity nudge, no filter.

Historia detallada de baselines (eval timeline 2026-04-15 → 2026-04-29 + comparaciones por feature) en git log. Nunca claim improvement sin re-correr `rag eval`.

## Query replay (Sprint 3 Tarea B, 2026-05-04)

`rag replay` rerunea queries históricas de `rag_queries` y diffa el resultado nuevo contra el output original.

**Modos**:
- `rag replay <id>` — diff de un query puntual (exit 0 = sin regresión, 1 = drift, 2 = id no found / q vacío, 3 = corpus drift sin --force)
- `rag replay <id> --explain` — muestra los paths nuevos sin comparar (exit 0 siempre)
- `rag replay --bulk [--since 7d] [--limit 20] [--filter-cmd CMD]` — batch sobre el historial

**Flags**:
- `--skip-gen` — solo comparar paths, sin LLM gen (más rápido, útil para CI)
- `--no-cache` — disable semantic cache durante el replay (default ON — replay debe ser reproducible)
- `--force` — continuar aunque haya corpus drift
- `--json` / `--plain` — output alternativo al Rich default

**Métricas de diff**:
- `path_jaccard` — Jaccard@5 entre paths originales y nuevos (1.0 = idénticos)
- `top3_changed` — cambió alguno de los 3 primeros paths
- `response_cosine` — cosine entre respuesta cacheada y nueva (cuando `response_text` disponible via Sprint 3 Tarea A)
- `response_hash_match` — hash comparison cuando solo hay `response_hash` en `extra_json`
- `corpus_drift` — flag cuando `corpus_hash` del row no matchea el corpus actual

**Verdicts**:
- `equivalent` — paths y respuesta dentro de los umbrales (jaccard ≥ 0.4 O top3 igual, cosine ≥ 0.85 si hay texto)
- `path_drift` — jaccard < 0.4 O top3 cambió, respuesta no comparada
- `response_drift` — respuesta divergente (cosine < 0.85 o hash mismatch)
- `regression` — error durante el replay o q vacío

**Invariantes**:
- `RAG_EXPLORE` scrubbed durante replay (misma invariante que `rag eval`)
- `RAG_SKIP_BEHAVIOR_LOG=1` durante replay — no contamina telemetría
- `auto_filter=False` — usa filtros del `filters_json` log, no re-infiere
- Forward-compatible con Sprint 3 Tarea A: funciona sin `response_text`/`response_hash` (solo compara paths en ese caso)

**Implementación**: `_replay_load_row`, `_replay_cosine`, `_replay_query_row`, `_replay_render_single` en `rag/__init__.py`. Tests: `tests/test_rag_replay.py` (27 casos).

## Tabla telemetry (post-T10 2026-04-19, post-split 2026-04-21)

Dos databases en `~/.local/share/obsidian-rag/ragvec/`:

- **`ragvec.db`** (~104M) — sqlite-vec corpus + 10 state tables (cursors ingesters): `rag_whatsapp_state`, `rag_calendar_state`, `rag_gmail_state`, `rag_reminders_state`, `rag_contacts_state`, `rag_calls_state`, `rag_safari_history_state`, `rag_safari_bookmark_state`, `rag_wa_media_state`, `rag_schema_version`.
- **`telemetry.db`** (~36M) — 45+ tablas operativas: `rag_queries`, `rag_behavior`, `rag_feedback`, `rag_feedback_golden*`, `rag_tune`, `rag_contradictions`, `rag_ambient*`, `rag_brief_*`, `rag_wa_tasks`, `rag_archive_log`, `rag_filing_log`, `rag_eval_runs`, `rag_surface_log`, `rag_proactive_log`, `rag_cpu_metrics`, `rag_memory_metrics`, `system_memory_metrics`, `rag_conversations_index`, `rag_conversation_summaries`, `rag_response_cache`, `rag_entities`, `rag_entity_mentions`, `rag_ocr_cache`, `rag_vlm_captions`, `rag_audio_transcripts`, `rag_learned_paraphrases`, `rag_cita_detections`, `rag_score_calibration`, `rag_chunk_contexts`, `rag_schema_version`.

**Reset total**: `rm ragvec/{ragvec,telemetry}.db && rag index --reset`. Solo telemetría: `rm ragvec/telemetry.db`.

SQL es único storage path (T10 stripped JSONL writers + readers). `RAG_STATE_SQL` removida del código 2026-05-04; los plists la siguen seteando como deployment trail.

**Retention** (via `rag maintenance`):
- 90d: `rag_queries`, `rag_behavior`, `rag_cpu_metrics`, `rag_memory_metrics` (30d), `rag_conversation_summaries` (30d), `system_memory_metrics` (30d).
- 60d: `rag_ambient`, `rag_brief_written`, `rag_wa_tasks`, `rag_archive_log`, `rag_filing_log`, `rag_eval_runs`, `rag_surface_log`, `rag_proactive_log`.
- Keep all forever: `rag_feedback`, `rag_tune`, `rag_contradictions`, `rag_draft_decisions`, `rag_brief_feedback`.

**Primitives** (`rag/__init__.py` `# ── SQL state store ──`):
- `_ensure_telemetry_tables(conn)` — idempotent DDL, **ensure-once por (proceso, db_path)** desde commit `09f00bd` (5-8x speedup; agregar entry nueva a `_TELEMETRY_DDL` requiere reiniciar daemons).
- `_ragvec_state_conn()` — short-lived WAL conn `synchronous=NORMAL` + `busy_timeout=10000`.
- `_sql_append_event`, `_sql_upsert`, `_sql_query_window`, `_sql_max_ts`.

**Writer contract**: single-row BEGIN/COMMIT. On exception → log a `sql_state_errors.jsonl` y silently drop. Callers nunca ven exception.

**Reader contract**: SQL-only. Empty snapshots / False / None on error. Retrieval pipeline keeps working.

### Invariantes telemetry stack (audit 2026-04-24 + 2026-04-25)

Cuatro reglas que el código debe respetar:

1. **Todo silent-error sink llama `_bump_silent_log_counter()`** post-write. Sin esto, alerting a stderr (threshold `RAG_SILENT_LOG_ALERT_THRESHOLD=20/h`) queda parcial. Tests: `tests/test_silent_log_alerting.py`.

2. **Async writer = paquete completo de 4 cambios**: (a) helper gate per-writer `_log_X_event_background_default()`, (b) caller con branch sync/async, (c) autouse fixture en conftest setea `RAG_LOG_X_ASYNC=0`, (d) doc del env var. Tests: `tests/test_sql_async_writers.py`.

3. **Readers SQL: retry + stale-cache fallback, nunca empty default que sobrescriba memo**. Modelo: `_load_behavior_priors`, `load_feedback_golden`. Tests: `tests/test_sql_reader_retry.py`.

4. **Tests con TestClient o writers SQL aíslan `DB_PATH` per-file**. NO hay autouse global (intentos conftest-wide reverteados). Pattern obligatorio (snap+restore manual, no `monkeypatch.setattr`):

   ```python
   @pytest.fixture(autouse=True)
   def _isolate_db_path(tmp_path):
       import rag as _rag
       snap = _rag.DB_PATH
       _rag.DB_PATH = tmp_path / "ragvec"
       try: yield
       finally: _rag.DB_PATH = snap
   ```

   Razón snap+restore: `monkeypatch.setattr` revierte en su propio teardown DESPUÉS del teardown de `_stabilize_rag_state` → warning falso. Pollution medida 2026-04-25: 161 test_tag entries en `sql_state_errors.jsonl`, 5 rows `question='test'` en `rag_response_cache`, 57 rows `cmd='web.chat.degenerate'` en `rag_queries`.

**Diagnóstico data-first**: `python scripts/audit_telemetry_health.py --days 7` — primer comando antes de cualquier "auditá el sistema". Agrega los 5 queries que reprodujeron audit 2026-04-24 en 1 segundo.

### Other state (on-disk, no DB)

- `ranker.json` + `ranker.{ts}.json` (3 más recientes) — tuned weights + backups por `rag tune --apply`. Reset: borrar.
- `sessions/*.json` + `last_session` — multi-turn (TTL 30d, cap 50 turns).
- `ambient.json`, `filing_batches/*.jsonl`, `ignored_notes.json`, `home_cache.json`, `context_summaries.json`, `auto_index_state.json`, `coach_state.json`, `synthetic_questions.json`, `wa_tasks_state.json`.
- `*.{log,error.log}` — launchd service logs.
- `sql_state_errors.jsonl` — diagnostic sink SQL failures.

**Reset learned state**: `rm ranker.json` + `DELETE FROM rag_feedback_golden*` en **`telemetry.db`** (post-split). Full re-embed: `rag index --reset`.

## Vault path

Default: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes`. Override: `OBSIDIAN_RAG_VAULT`. Collections namespaced per vault (sha256[:8]).

**Memorias del MCP [`mem-vault`](https://github.com/jagoff/mem-vault)** viven en `04-Archive/99-obsidian-system/99-AI/memory/`. Configurado via env vars del web server plist: `MEM_VAULT_PATH=Notes/`, `MEM_VAULT_MEMORY_SUBDIR=04-Archive/99-obsidian-system/99-AI/memory`. **NO está excluido por `is_excluded()`** (junto con `99-Mentions/`) — `rag index` lo scanea, los `.md` entran al index del vault `home`. MCP `mem-vault` es writer canónico, `rag` reader adicional.

## Daemons del proyecto

Source de verdad: lista de tuplas en [`rag/__init__.py`](rag/__init__.py) función `_services_spec()`. Manuales (no en `_services_spec`): `cloudflare-tunnel`, `cloudflare-tunnel-watcher`, `lgbm-train`, `paraphrases-train`, `synth-refresh`, `spotify-poll`, `log-rotate` — trackeados por control plane via `_services_spec_manual()`.

Listado actual (`launchctl list | grep obsidian-rag`):

| Plist | Cadencia | Comando | Propósito |
|---|---|---|---|
| `watch` | watcher | `rag watch` | Auto-reindex |
| `serve` | KeepAlive | `rag serve` | MCP server |
| `web` | KeepAlive | `web/server.py` | Web UI + chat |
| `morning` / `today` / `digest` | cal 7am / 22hs / weekly | `rag morning|today|digest` | Briefs |
| `wake-up` | calendar | `rag wake-up` | Setup post-sleep |
| `emergent` / `patterns` | weekly | | Detectores |
| `archive` | weekly | `rag archive` | Auto-archivo dead notes |
| `wa-tasks` / `reminder-wa-push` / `wa-scheduled-send` | 30min / 5min / 5min | | WhatsApp loops |
| `anticipate` | 10min | `rag anticipate` | Anticipatory agent |
| `auto-harvest` / `online-tune` / `calibrate` | weekly / nightly 03:30 / nightly | | Ranker-vivo |
| `maintenance` | weekly | | Vacuum + WAL checkpoint + log rotation |
| `consolidate` | nightly | `rag consolidate` | Memory consolidation Phase 2 |
| `ingest-{whatsapp,gmail,calendar,reminders,drive,calls,safari,pillow}` | varias | `rag index --source X` | |
| `implicit-feedback` / `routing-rules` | 15min / 5min | | |
| `cloudflare-tunnel` + `tunnel-watcher` (manual) | KeepAlive | | HTTPS público |
| `serve-watchdog` | daemon | | (deprecated 2026-05-01, reemplazado por daemon-watchdog) |
| `daemon-watchdog` | 5min | `rag daemons reconcile --apply --gentle` | Control plane retry + kickstart-overdue |
| `active-learning-nudge` | lunes 10am | | WA nudge para labeling |
| `brief-auto-tune` | Domingo 03:00 | `rag brief schedule auto-tune --apply` | |
| `lgbm-train` (manual) | Domingo 02:30 | `rag tune-lambdarank --apply` | |
| `paraphrases-train` (manual) | Domingo 04:30 | | `rag_learned_paraphrases` |
| `spotify-poll` (manual) | 60s | | `rag_spotify_log` |
| `synth-refresh` (manual) | Sábado 22:00 | | Feeder lgbm-train |
| `vault-cleanup` | nightly | | |
| `whisper-vocab` | 03:15 | | Vocab WA refresh |
| `mood-poll` | 30min | | UI no cableada (signals NO en home.v2) |

**Control plane** `rag daemons`:
- `status [--json --unhealthy-only]` — tabla loaded/running/last_exit/overdue/category.
- `reconcile [--apply --dry-run --gentle]` — converge drift entre `_services_spec()` + lo cargado. `--gentle` = retry exit≠0 + kickstart overdues, NO bootea huérfanos ni regenera plists. Para reconciliación agresiva: `--apply` sin `--gentle`.
- `doctor` — diagnóstico humano + remediation sugerida.
- `retry <label>` — kickstart -k puntual.
- `kickstart-overdue` — catchup post-sleep manual.

Las acciones se loggean a `rag_daemon_runs` (telemetry.db, retention 90d).

### Checklist al agregar plist nuevo

1. Factory `_<nombre>_plist(rag_bin)` + tuple en `_services_spec()`.
2. CLI subcommand.
3. Smoke `rag <subcomando> --dry-run`.
4. Generar plist + copiar a `~/Library/LaunchAgents/`.
5. `launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.fer.obsidian-rag-<nombre>.plist`.
6. Verificar: `launchctl list | grep` + `launchctl print gui/$UID/com.fer.obsidian-rag-<nombre>`.
7. Esperar tick (o `launchctl kickstart -k`) + verificar log generado.
8. Solo entonces marcar feature como completa.

Anti-patrón: dejar TODO "corré `rag setup`" en commit msg. Aprendido 2026-04-25 con `wa-scheduled-send` (commit `9740fa1` — plist nunca se copió, user programó msg, no llegó).

## Wave-8 gotchas (2026-04-28) — pipeline de filtros + carry-over

### Filtros definidos pero no cableados

Síntoma: `_XxxFilter` o función `_strip_*` / `_redact_*` con docstring + regex completo, ningún call site la invoca. Bug que se suponía fixeada sigue ahí.

Caso real: `_strip_foreign_scripts` ([`web/server.py:1504-1531`](web/server.py)) existía con docstring "Remove CJK/Cyrillic/Hebrew/Arabic". Nunca se llamaba. CJK leak en weather siguió hasta wave-8.

Cómo evitarlo:
1. Cuando agregás filtro, también editá `_emit()` helper en `gen()` (~línea 11631) Y pipeline de cache replay (~línea 9887, `_redact_pii(_sem_text)`).
2. Antes de "queda para wirear después", ya escribí el call site.
3. Test de regresión [`tests/test_filter_wiring.py`](tests/test_filter_wiring.py) falla si clase `_*Filter` o función `_strip_*`/`_redact_*` está sin call site. Si false-positive intencional → allowlist, no borrar.

### Carry-over del pre-router silenciosamente sobrescrito por fast-path

Síntoma: lógica al inicio de `gen()` computa `_forced_tool_pairs`. Log dice que se computó. Tool nunca corre porque otro branch downstream re-llama `_detect_tool_intent(question)` y descarta tu carry-over.

Caso real: pre-router seteaba `_forced_tool_pairs = [('weather', {'location': 'Barcelona'})]` por carry-over de "y en Barcelona?". Línea 10996 hacía `_forced_tools = [] if _propose_intent else _detect_tool_intent(question)` — la query sola no matchea keyword, retornaba `[]`. Fix: `_forced_tools = list(_forced_tool_pairs)`.

Cómo evitarlo:

```bash
grep -n '_detect_tool_intent\|_forced_tools\s*=' web/server.py
```

Regla: pre-router corre UNA vez al inicio de `gen()`, todo el resto del flow LEE de `_forced_tool_pairs`.

### Bumpear `_FILTER_VERSION` es parte del fix

Síntoma: arreglaste filtro / system prompt / regex. Validás Playwright. Test reporta bug sigue. La causa: semantic cache sirviendo respuestas pre-fix porque cache key no incluye nada que tu fix haya cambiado.

Mecanismo: `_FILTER_VERSION` (`rag/__init__.py:4656`) está horneado en `_hash_chunk_count` y usado en corpus_hash → cache key. Bumpear la string invalida TODAS las entries pre-fix.

Cuándo bumpear:
- Cambia regex que afecta tools_fired (PII redact, raw tool stripper, iberian leaks, foreign scripts).
- Cambia `_WEB_SYSTEM_PROMPT` o cualquier REGLA N.
- Cambia traducción de descriptions inyectada al CONTEXTO.

Cuándo NO: perf/refactors sin output change, features off-by-default, herramientas administrativas.

Naming: `wave<N>-<YYYY-MM-DD>` ej. `wave8-2026-04-28`.

## Notification al cerrar tarea

Hook `Stop` en [`~/.config/devin/config.json`](file:///Users/fer/.config/devin/config.json) dispara `osascript -e 'display notification ...'` (banner macOS, sin sonido desde 2026-04-27). NO ejecutar `afplay`/`osascript` manual — el hook se encarga. Permisos: System Settings > Notifications > "Script Editor". Ajustar texto: editar string en `command` del hook block.

## Referencias

- Vault path: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes`
- Memory dir: `04-Archive/99-obsidian-system/99-AI/memory/`
- Bug listener TS: [`/Users/fer/whatsapp-listener`](file:///Users/fer/whatsapp-listener)
- WhatsApp bridge: `~/repositories/whatsapp-mcp/whatsapp-bridge/store/messages.db`
- Cloudflared URL: `~/.local/share/obsidian-rag/cloudflared-url.txt`
- Auditoría salud sistema: `python scripts/audit_telemetry_health.py --days 7 --json`
- Eval baselines + perf history detallada: git log
