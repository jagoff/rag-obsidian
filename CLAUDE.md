# CLAUDE.md

Local RAG sobre vault Obsidian. Layout post-split (2026-05-04): `rag/` paquete (`__init__.py` 60.2k LOC core + sub-modules `plists.py`, `cross_source_etls.py`, `postprocess.py`, `archive.py`, `anticipatory.py`, `brief_schedule.py`, `contradictions_penalty.py`, `voice_brief.py`, `whisper.py`, `wa_scheduled.py`, `wa_tasks.py`, `mmr_diversification.py`, `today_correlator.py`, `vault_health.py`, etc) + `mcp_server.py` (thin wrapper) + `web/` (FastAPI server.py 20.6k LOC + static) + `tests/` (6,031 tests, 395 archivos). Re-export pattern: `__init__.py` hace `from rag.X import *  # noqa: F401, F403` con `__all__` explícito en cada sub-módulo (preserva 100% compat).

Entry points (instalados via `uv tool install --reinstall --editable '.[entities,stt,spotify,mlx]'`):
- `rag` — CLI indexing/querying/chat/productivity/automation
- `obsidian-rag-mcp` — MCP server (`rag_query`, `rag_read_note`, `rag_list_notes`, `rag_links`, `rag_stats`)

Local-first sobre VAULT + corpus locales (sqlite-vec + Ollama/MLX + sentence-transformers). Cross-source ingesters cloud (Gmail/Calendar/Drive) requieren creds OAuth en `~/.{gmail,calendar,gdrive}-mcp/`; sin esas creds silent-fail y corpus local sigue funcionando. WhatsApp + Reminders stay local.

Python 3.13, `uv`. Runtime venv: `.venv/bin/python`. Global tool: `~/.local/share/uv/tools/obsidian-rag/`.

## Docs detalle (cargar bajo demanda)

| Tema | Doc |
|---|---|
| MLX migration + LLM backend | [`docs/mlx-migration.md`](docs/mlx-migration.md) |
| Retrieval pipeline + scoring + intents + prompts + ingesters | [`docs/retrieval-internals.md`](docs/retrieval-internals.md) |
| Telemetry SQL + invariantes audit + retention | [`docs/telemetry-stack.md`](docs/telemetry-stack.md) |
| Daemons + control plane + checklist | [`docs/daemons.md`](docs/daemons.md) |
| Web chat tool-calling + Feature H/K + sessions + Quick Wins | [`docs/web-chat-features.md`](docs/web-chat-features.md) |
| Bot WA draft + Brief feedback + Voice + Whisper + Implicit | [`docs/feedback-loops.md`](docs/feedback-loops.md) |
| Wave-8 filter pipeline gotchas | [`docs/wave-8-gotchas.md`](docs/wave-8-gotchas.md) |
| Query replay (`rag replay`) | [`docs/query-replay.md`](docs/query-replay.md) |
| Anticipatory agent | [`docs/anticipatory-agent.md`](docs/anticipatory-agent.md) |
| Env vars completas (47+) | [`docs/env-vars-catalog.md`](docs/env-vars-catalog.md) |
| Comandos completos | [`docs/comandos.md`](docs/comandos.md) |
| Cómo funciona end-to-end | [`docs/como-funciona.md`](docs/como-funciona.md) |
| Recovery + problemas | [`docs/recovery.md`](docs/recovery.md), [`docs/problemas-comunes.md`](docs/problemas-comunes.md) |

## MLX migration (Ola 5 hard-cutover — 2026-05-06)

**Estado actual: 100% MLX, sin fallback Ollama disponible.** Modelos chat Ollama purgados del disco (decisión user 2026-05-06). Default `RAG_LLM_BACKEND=mlx`. Detalle completo en [`docs/mlx-migration.md`](docs/mlx-migration.md).

**Mapping**:
- `qwen2.5:3b` (HELPER) → [`mlx-community/Qwen2.5-3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit)
- `qwen2.5:7b` (CHAT default) → [`mlx-community/Qwen2.5-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit)
- `command-r` / `qwen2.5:14b` (HQ tier) → [`mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit)

`qwen3-embedding:0.6b` sigue corriendo via Ollama (embedder activo, NO migrado a MLX). Es el único modelo Ollama que queda en disco.

**Tool-calling**: nativo MLX via [`rag/mlx_tool_calls.py`](rag/mlx_tool_calls.py) (Ola 5, commit `82d27d5`). Parser Qwen `<tool_call>{...}</tool_call>` → `Message.ToolCall` ollama-shape. Wireado en [`rag/llm_backend.py:591`](rag/llm_backend.py).

**Rollback emergencia**: requiere re-pull de los 3 modelos chat (`ollama pull qwen2.5:3b qwen2.5:7b qwen3:30b-a3b`, ~24 GB) ANTES de exportar `RAG_LLM_BACKEND=ollama`. Sin eso, el rollback falla con `model 'X' not found`.

**Idle-unload watchdog** ([`rag/llm_backend.py`](rag/llm_backend.py)): evicta modelos con `now - last_used > RAG_MLX_IDLE_TTL` (default 1800s). Disable: `RAG_MLX_IDLE_TTL=0` o `RAG_MLX_IDLE_DISABLE=1`.

**Tests**: `tests/conftest.py` autouse fixture `_force_ollama_backend_for_tests` fuerza `RAG_LLM_BACKEND=ollama` por test. Marker `requires_mlx` registrado. Como Ollama-chat no está en disco, los tests que asumen el backend fake-Ollama deben monkeypatchear `ollama.chat` directamente, no apuntar a un daemon real.

**Embeddings (bge-m3) NO entran en este scope**.

## Idioma

Español rioplatense (voseo) por default. Regla universal en [`~/.claude/CLAUDE.md`](file:///Users/fer/.claude/CLAUDE.md). Detector pre-emit: `você` / `obrigad` / `essa` / `isso` / CJK / `tú` formal → bug, corregir antes de mandar.

## Agent dispatch rule

Invocar `pm` ANTES de empezar cuando AL MENOS UNO:

1. Cruza ≥2 agent domains (retrieval + brief, llm + ingestion, integrations + vault-health).
2. Toca un invariant listado en [`pm.md`](.claude/agents/pm.md): schema version `_COLLECTION_BASE`, eval floor (singles/chains CI), reranker `device="mps"` + `float32`, HELPER model binding (`reformulate_query` + `qwen2.5:3b`), confidence gates (`CONFIDENCE_RERANK_MIN`, `CONFIDENCE_DEEP_THRESHOLD`), Ollama `keep_alive=-1`, session-id regex, local-first.
3. Hay peers activos (`mcp__claude-peers__list_peers(scope: "repo")` > 1) Y su `set_summary` se solapa.
4. No sabés qué agent owns la work.

Skip PM cuando: edits mecánicos (rename, ruff, bump versión, typo fix), single-domain con N archivos, exploración / Q&A / review de diffs, fix trivial obvio.

Roster + ownership en [`.claude/agents/README.md`](.claude/agents/README.md).

### Custom agent profiles requieren reload de la sesión

Profiles en `.claude/agents/*.md` se cargan **una sola vez al iniciar la sesión**. Si creás un agent nuevo durante una sesión activa, esa sesión NO lo ve. Workaround: reabrir sesión, o inyectar el system prompt inline en `subagent_explore` / `subagent_general`. Mismo gotcha aplica a skills custom. Hooks en `.devin/config.json` SÍ se refrescan en runtime.

## Auto-save a `mem-vault` al cerrar tarea

Regla universal en [`~/.claude/CLAUDE.md`](file:///Users/fer/.claude/CLAUDE.md). Trigger: bug fix con root cause no obvio, decisión arquitectónica, refactor con invariantes, performance findings con números, workflow operativo nuevo, gotchas reproducibles. Tool: `mcp_call_tool(server_name="mem-vault", tool_name="memory_save", ...)` con markdown enriquecido (Contexto / Problema / Solución / Tests / Aprendido el YYYY-MM-DD + commit SHA).

## Auto-pull + commit + push rule

Cuando termino algo: `git pull → git commit → git push origin master`. Sin preguntar. Mensaje completo en español rioplatense (qué cambié, por qué, cómo medí si aplica, cómo revertir si rompe). Trailer estándar Devin al final. Si tests fallan o build rompe → NO commiteás. Excepciones: tareas exploratorias, cambios pedidos no commitear, trabajo a medio camino.

### Gotcha: commits locales en `master` se pushean solos

Cualquier commit en `master` aparece en `origin/master` en segundos por **otra sesión paralela** (claude-peers MCP). Implicaciones:

1. `git commit` master = `git push` casi inmediato. Sin ventana para "commit experimental + reset si no me gusta".
2. Para experimentar sin pushear → branch dedicada (`git checkout -b experimental/<slug>`).
3. Si pushiaste algo malo: `git revert <sha>` (force-push está en deny-list). Commit malo + revert quedan ambos en log.
4. No se puede desactivar el auto-pusher desde esta sesión; coordinarlo via `mcp__claude-peers__send_message`.

## Autonomous mode

Devin tiene 4 [permission modes](https://docs.devin.ai/reference/permissions). Config minimiza interrupciones:

1. `.devin/config.json` — ~80 allow rules (git, rag, uv, pytest, sqlite3, launchctl, observabilidad, writes en repo); 6 deny (sudo, `git reset --hard`, `git push --force`, `git branch -D`); ask (.env, ~/.ssh, ~/.aws, writes al vault iCloud, fetch a OpenAI/Anthropic). `rm -rf` allow desde 2026-04-28.
2. Bypass mode (`devin --permission-mode bypass` o Shift+Tab): cero prompts salvo `deny`.

Precedencia: org → session-grants → `.devin/config.local.json` → `.devin/config.json` → `~/.config/devin/config.json`.

Rollback: `mv .devin/config.json{,.disabled}`.

## Zsh tab-completion

Hand-written en [`completions/_rag`](completions/_rag) con descriptions + sub-grupos + helpers dinámicos. Startup nativo zsh ~10-50ms vs ~350ms del autocompletion de Click.

Instalación: `cp completions/_rag ~/.oh-my-zsh/custom/completions/_rag && rm -f ~/.zcompdump* && exec zsh`. Regenerar: `.venv/bin/python scripts/gen_zsh_completion.py > completions/_rag`.

## PWA + LAN/HTTPS exposure

PWA instalable iOS Safari → home screen. Wiring: [`web/static/manifest.webmanifest`](web/static/manifest.webmanifest), [`sw.js`](web/static/sw.js), [`pwa/register-sw.js`](web/static/pwa/register-sw.js) + [`scripts/gen_pwa_assets.py`](scripts/gen_pwa_assets.py).

**LAN exposure**: dos env vars en [`com.fer.obsidian-rag-web.plist`](~/Library/LaunchAgents/com.fer.obsidian-rag-web.plist):
- `OBSIDIAN_RAG_BIND_HOST=0.0.0.0` — uvicorn bindea a todas las interfaces (default `127.0.0.1`).
- `OBSIDIAN_RAG_ALLOW_LAN=1` — extiende CORS regex a [RFC1918](https://datatracker.ietf.org/doc/html/rfc1918) (10/8, 172.16/12, 192.168/16).

**Tradeoff iOS**: SW solo registra en secure context (HTTPS o localhost). HTTP LAN da fullscreen + icon + splash, NO offline cache. Para SW completo via LAN: Caddy con `tls internal` + cert root al iPhone.

**HTTPS público**: [`cloudflared tunnel --url http://localhost:8765`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/) genera URL random. CORS: `OBSIDIAN_RAG_ALLOW_TUNNEL=1`. URL cambia cada restart cloudflared — para estable: named tunnel + dominio. Dos plists: `cloudflare-tunnel` + `cloudflare-tunnel-watcher` (escribe URL a `~/.local/share/obsidian-rag/cloudflared-url.txt` + pbcopy + macOS notification). Aliases: `rag-url`, `rag-url-c`.

**Seguridad**: server NO tiene auth. Solo activar en WiFi privado.

## Commands (canonical subset)

```bash
uv tool install --reinstall --editable '.[entities,stt,spotify,mlx]'

# Core
rag index [--reset] [--no-contradict] [--vault NAME]
rag index --source whatsapp|contacts|calls|safari|reminders|gmail|calendar|drive|pillow [--reset --since ISO --dry-run]
rag query "text" [--hyde --no-multi --raw --loose --force --counter --no-deep --plain --source S]
rag chat [--counter --no-deep --session ID --resume]            # /save /reindex /undo
rag do "instrucción" [--yes --max-iterations 8]
rag stats

# Daily automation
rag morning [--voice], rag today, rag digest, rag consolidate

# Productivity
rag capture, rag inbox, rag prep, rag read, rag dupes, rag links, rag wikilinks, rag followup, rag dead, rag archive

# Quality
rag eval [--latency --max-p95-ms N]
rag tune [--samples 500 --apply --online --days 14 --rollback]
rag replay <id> [--diff|--explain] [--bulk --since 7d]
rag log, rag dashboard, rag feedback {status|backfill|infer-implicit|harvest}
rag open <path>, rag pendientes, rag contact-note

# Anticipatory + Bot WA
rag anticipate [run|explain|log] [-n 20 --only-sent --dry-run --force]
rag silence anticipate-{calendar,echo,commitment} [--off]
rag draft stats, rag drafts finetune
rag brief stats, rag brief schedule [status|reset|auto-tune]
rag voice-brief generate, rag whisper {stats|vocab|patterns|export|import}

# Maintenance
rag maintenance, rag free, rag setup, rag stop
rag daemons {status|reconcile|doctor|retry|kickstart-overdue}
python scripts/audit_telemetry_health.py --days 7  # PRIMER comando antes de "auditá el sistema"

# Tests
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_foo.py::test_bar -q
```

Set completo en [`docs/comandos.md`](docs/comandos.md).

## Env vars críticas (rollback paths)

Catálogo completo (47+ vars adicionales) en [`docs/env-vars-catalog.md`](docs/env-vars-catalog.md). Esta sección cubre las críticas con rollback.

**Vault + ingest**:
- `OBSIDIAN_RAG_VAULT` — override vault path. Cross-source ETLs gated por `_is_cross_source_target(vault_path)` — solo `_DEFAULT_VAULT` recibe los 11 ETLs salvo opt-in en `~/.config/obsidian-rag/vaults.json`.
- `RAG_OCR=0` — desactiva OCR (default ON cuando `ocrmac` disponible).
- `OBSIDIAN_RAG_MOZE_DIR`, `OBSIDIAN_RAG_FINANCE_DIR` — iCloud sources MOZE + xlsx/PDFs.
- `OBSIDIAN_RAG_INDEX_WA_MONTHLY=1` — opt-in al double-indexing WA monthly rollups (default OFF post-2026-04-22).

**Backend LLM** (ver MLX section arriba):
- `RAG_LLM_BACKEND={ollama,mlx}` — default `mlx`.
- `RAG_MLX_IDLE_TTL` (default 1800s), `RAG_MLX_IDLE_DISABLE=1`.

**Performance + memoria**:
- `OLLAMA_KEEP_ALIVE=-1` (default forever). Auto-clamp a `_LARGE_KEEP_ALIVE="20m"` para `_LARGE_CHAT_MODELS` (command-r, qwen3:30b-a3b). Override: `RAG_KEEP_ALIVE_LARGE_MODEL`.
- `RAG_MEMORY_PRESSURE_DISABLE=1` — desactiva watchdog (default ON, threshold 85%, interval 60s). Bajo pressure: unload chat + force-unload reranker (bypassa `RAG_RERANKER_NEVER_UNLOAD`).
- `RAG_RERANKER_NEVER_UNLOAD=1` — pina reranker en MPS VRAM. Cost ~2-3 GB.
- `RAG_RERANKER_IDLE_TTL=900` — segundos idle-unload.
- `RAG_LOCAL_EMBED=1` — in-process bge-m3 (set en plists web + serve, auto-set en CLI query-like). NO en indexing/watch.
- `RAG_LOCAL_EMBED_WAIT_MS=6000` — budget Event ready antes fallback Ollama.

**Async writers** (default ON desde audit 2026-04-24):
- Set `RAG_LOG_{QUERY,BEHAVIOR,FT_RATING,AMBIENT,CONTRADICTIONS,ARCHIVE,TUNE,SURFACE}_ASYNC=0` + `RAG_METRICS_ASYNC=0` para opt-out.

**Retrieval**:
- `RAG_ADAPTIVE_ROUTING` (default ON) — skip helper reformulate intents metadata-only + fast-path.
- `RAG_LOOKUP_NUM_CTX=4096` — fast-path ctx.
- `RAG_FAST_PATH_KEEP_WITH_TOOLS=1` — rollback del downgrade fast-path con tools (default OFF).
- `RAG_ENTITY_LOOKUP` / `RAG_EXTRACT_ENTITIES` (default ON post-2026-04-21).
- `RAG_EXPLORE=1` — ε-exploration. **MUST unset durante `rag eval`** (comando lo `os.environ.pop`s).
- `RAG_EXPAND_MIN_TOKENS=4` — threshold short-query gate.
- `RAG_CITATION_REPAIR_MAX_BAD=2` (set 0 para disable).
- `RAG_DEEP_MAX_SECONDS=30` — wall-time cap auto-deep.
- `RAG_NLI_GROUNDING` (default OFF) — mDeBERTa post-citation-repair. `RAG_NLI_IDLE_TTL=900`.
- `RAG_NLI_MODE={off,mark,strip}` (default off) — citation NLI verifier. `RAG_NLI_THRESHOLD=0.5`.
- `RAG_CONTRADICTION_PENALTY` (default ON, magnitude `RAG_CONTRADICTION_PENALTY_MAGNITUDE=0.05`).
- `RAG_MMR` (default OFF, `RAG_MMR_LAMBDA=0.7`, `RAG_MMR_TOP_K=10`). Variante: `RAG_MMR_FOLDER_PENALTY=1` (mutex).
- `RAG_LLM_JUDGE` (default OFF, prototipo) — score blend cuando top<0.5 AND len≥5. `RAG_LLM_JUDGE_THRESHOLD=0.5`, `RAG_LLM_JUDGE_MIN_CANDIDATES=5`, `RAG_LLM_JUDGE_ALPHA=0.5`.
- `RAG_QUERY_DECOMPOSE` (default OFF, prototipo) — sub-retrieves + RRF. `RAG_QUERY_DECOMPOSE_LLM_FALLBACK=0`, `RAG_QUERY_DECOMPOSE_MAX_WORKERS=3`.
- `RAG_INTENT_RECENCY` (default ON, Quick Win #3) — halflife per intent (recent ×0.3, historical ×3.0, neutral ×1.0).
- `RAG_TYPO_CORRECTION` — default ON con Ollama / **OFF con MLX** (resolved `_resolve_typo_correction_default()` por bug 2026-05-05: qwen2.5:3b parafrasea agresivo bajo MLX). Override `=1` siempre gana. `RAG_TYPO_JACCARD_MIN=0.7` solo multi-token.
- `RAG_HISTORY_SUMMARY` (default ON, Quick Win #5).
- `RAG_ANAPHORA_RESOLVER` (default ON, Quick Win #1).
- `RAG_CONTEXTUAL_RETRIEVAL=1` (default OFF, prototipo Anthropic).
- `RAG_WA_FAST_PATH` / `RAG_WA_FAST_PATH_THRESHOLD=0.05` / `RAG_WA_SKIP_PARAPHRASE` (default ON).

**Ranker + cache**:
- `RAG_TRACK_OPENS=1` — switches OSC 8 a `x-rag-open://` para ranker-vivo signal.
- `RAG_RERANKER_FT=1` — opt-in LoRA adapter (default OFF). Failure modes silent_fail con fallback a base.
- `RAG_FINETUNE_MIN_CORRECTIVES=20` — abort `scripts/finetune_reranker.py` si insuficiente.
- `RAG_DRAFT_VIA_RAGNET=1` — legacy override redirige ambient sends a RagNet.

**Replay + privacy**:
- `RAG_LOG_REPLAY_PAYLOAD=1` (default OFF) — opt-in PII (`response_text` cap 8KB, `history_snapshot` cap 4KB).
- `RAG_LOG_RERANK_RAW=1` (default OFF) — opt-in `rerank_logits_raw`.
- Hashes (corpus/response/prompt/history) **siempre** persistidos (16 chars hex).

**Misc**:
- `OBSIDIAN_RAG_NO_APPLE=1` — desactiva Apple integrations.
- `RAG_TIMEZONE` — IANA tz (default `America/Argentina/Buenos_Aires`).
- `OBSIDIAN_RAG_WATCH_EXCLUDE_FOLDERS` — comma-separated.

**Dev/debug** (NO producción): `RAG_DEBUG=1`, `RAG_RETRIEVE_TIMING=1`, `RAG_NO_WARMUP=1`, `OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY=1`, `OBSIDIAN_RAG_SKIP_SYNTHETIC_Q=1`.

## Architecture invariants

Detalle completo del pipeline en [`docs/retrieval-internals.md`](docs/retrieval-internals.md). Resumen invariantes críticos:

**Schema collection**: bump `_COLLECTION_BASE` (currently `obsidian_notes_v11`). Per-vault suffix sha256[:8] of resolved path.

**Reranker**: `BAAI/bge-reranker-v2-m3` con `device="mps"` + `float32` forced. **NO switch fp16** — 2 A/Bs failed (collapse 2026-04-13, overhead 2x con calidad equivalente 2026-04-22).

**HELPER**: `qwen2.5:3b` con `HELPER_OPTIONS = {temperature: 0, seed: 42}` deterministic. Bound to `reformulate_query`, `expand_queries`, `_judge_sufficiency`. command-r como helper regresiona −11pp chains + 5× latencia.

**Confidence gate**: `top_score < 0.015` (CONFIDENCE_RERANK_MIN) + no `--force` → refuse sin LLM call. Per-source override scaffolding existe (`CONFIDENCE_RERANK_MIN_PER_SOURCE`).

**`RERANK_POOL_MAX = 15`** (dropped from 30 on 2026-04-21): pool=15 domina vs 30 — hit@5 idéntico, MRR chains +5pp, P95 singles -66%.

**Cache locks**: `_context_cache_lock`, `_synthetic_q_cache_lock`, `_mentions_cache_lock`, `_embed_cache_lock`, `_corpus_cache_lock` (RLock), `_contacts_cache_lock`. LLM calls **outside** lock.

**Pipeline resumen**:
```
query → typo correct → anaphora resolve → classify_intent → infer_filters
      → adaptive routing → decomposition gate → expand_queries (qwen2.5:3b)
      → embed bge-m3 → sqlite-vec sem + BM25 → RRF + dedup → expand to parent
      → rerank (bge-reranker-v2-m3, MPS+fp32) → LLM judge gate
      → score loop (recency/intent/behavior/contradiction/feedback)
      → MMR diversification → contradiction penalty → seen_titles soft penalty (-0.1)
      → graph expansion (1-hop, top-3, 3 neighbors) → auto-deep (≤3 iters, 30s cap)
      → top-k → LLM streamed → citation-repair → NLI verifier
```

**Generation prompts**: `SYSTEM_RULES_STRICT` (default semantic), `SYSTEM_RULES` (`--loose`), `SYSTEM_RULES_LOOKUP` (count/list/recent/agenda), `SYSTEM_RULES_SYNTHESIS`, `SYSTEM_RULES_COMPARISON`. Routed via `system_prompt_for_intent(intent, loose)`. `_CHUNK_AS_DATA_RULE` (REGLA 0) + `_NAME_PRESERVATION_RULE` previenen prompt injection + name corruption.

**`_FILTER_VERSION`** ([`rag/__init__.py:6017`](rag/__init__.py)): bumpear cuando cambia regex que afecta tools_fired, `_WEB_SYSTEM_PROMPT`/REGLA N, traducción descriptions inyectada. Naming: `wave<N>-<YYYY-MM-DD>`. Detalle en [`docs/wave-8-gotchas.md`](docs/wave-8-gotchas.md).

## Eval baselines (floor MLX 2026-05-05)

Floor actual (post-Ola 3 cutover, post-typo-corrector-fix `48ababf`):
- Singles: `hit@5 56.60% [43.40, 69.81] · MRR 0.535 [0.403, 0.667] · n=53`
- Chains: `hit@5 72.00% [56.00, 88.00] · MRR 0.617 [0.447, 0.773]`
- **Lower-CI-bound gate** (nightly online-tune auto-rollback): singles < 43.40% OR chains < 56.00%

Floor PRE-MLX (archivado): singles `53.70% [40.74, 66.67]`, chains `72.00% [52.00, 88.00]`. Post-cutover MLX supera ambos (+2.9pp singles, chains match con CI más estrecho).

`rag eval --latency --max-p95-ms N` agrega P50/P95/P99 + CI gate. Bootstrap 1000 resamples seed=42.

Helper LLM calls determinísticos via `HELPER_OPTIONS`. **HyDE drops singles ~5pp** — opt-in via `--hyde`. `seen_titles` post-rerank penalty `0.1`.

**Nunca claim improvement sin re-correr `rag eval`**.

## Telemetry stack

Detalle completo: [`docs/telemetry-stack.md`](docs/telemetry-stack.md).

Dos databases en `~/.local/share/obsidian-rag/ragvec/`:
- `ragvec.db` (~104M) — sqlite-vec corpus + 10 state tables.
- `telemetry.db` (~36M) — 45+ tablas operativas.

**Reset**: `rm ragvec/{ragvec,telemetry}.db && rag index --reset`. Solo telemetría: `rm ragvec/telemetry.db`.

**Invariantes** (audit 2026-04-24+25):
1. Todo silent-error sink llama `_bump_silent_log_counter()` post-write.
2. Async writer = paquete completo de 4 cambios (helper, branch, conftest, doc).
3. Readers SQL: retry + stale-cache fallback, nunca empty default que sobrescriba memo.
4. Tests con TestClient o writers SQL aíslan `DB_PATH` per-file (snap+restore manual, no `monkeypatch.setattr`).

**Diagnóstico data-first**: `python scripts/audit_telemetry_health.py --days 7` — PRIMER comando antes de "auditá el sistema". Agrega los 5 queries que reprodujeron audit 2026-04-24 en 1 segundo.

## Daemons

Detalle completo + tabla plists + checklist: [`docs/daemons.md`](docs/daemons.md).

Source de verdad: `_services_spec()` en [`rag/__init__.py`](rag/__init__.py). Manuales en `_services_spec_manual()`.

Control plane `rag daemons {status|reconcile|doctor|retry|kickstart-overdue}` — acciones loggeadas a `rag_daemon_runs` (telemetry.db, retention 90d).

**Anti-patrón** al cerrar feature con plist nuevo: dejar TODO "corré `rag setup`". Aprendido 2026-04-25 con `wa-scheduled-send` (commit `9740fa1` — plist nunca se copió, user programó msg, no llegó).

## Feedback loops

Detalle completo: [`docs/feedback-loops.md`](docs/feedback-loops.md).

- **Anticipatory Agent** — daemon 10min, 3 señales (calendar/echo/commitment).
- **Bot WA draft loop** — listener TS genera bot_draft → user `/si`/`/no`/`/editar` → `rag_draft_decisions` (gold humano para fine-tunes).
- **Brief feedback + auto-tuning** — reactions 👍/👎/🔇 → `rag_brief_feedback`. Mute consistent → shift schedule +30min iterativo.
- **Voice brief** — Phase 2.C: morning OGG/Opus via `say -v Mónica` + ffmpeg libopus.
- **Whisper learning** — daemon vocab refresh + `/fix` corrections + confidence-gated LLM correct.
- **Implicit feedback** — `rag feedback classify-sessions` backpropaga outcome session → turn (weak negatives en abandons low-score, weight 0.3).

## Wave-8 gotchas

Detalle: [`docs/wave-8-gotchas.md`](docs/wave-8-gotchas.md).

Tres patrones: (1) filtros definidos sin call site, (2) carry-over pre-router sobrescrito por `_detect_tool_intent` downstream, (3) bumpear `_FILTER_VERSION` es parte del fix cuando cambia filtro/prompt/regex (cache key invalidation).

## Vault path

Default: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes`. Override: `OBSIDIAN_RAG_VAULT`. Collections namespaced per vault (sha256[:8]).

**Memorias del MCP [`mem-vault`](https://github.com/jagoff/mem-vault)** viven en `04-Archive/99-obsidian-system/99-AI/memory/`. Configurado via env vars del web server plist: `MEM_VAULT_PATH=Notes/`, `MEM_VAULT_MEMORY_SUBDIR=04-Archive/99-obsidian-system/99-AI/memory`. NO está excluido por `is_excluded()` (junto con `99-Mentions/`) — `rag index` lo scanea, los `.md` entran al index del vault `home`. MCP `mem-vault` es writer canónico, `rag` reader adicional.

## Notification al cerrar tarea

Hook `Stop` en [`~/.config/devin/config.json`](file:///Users/fer/.config/devin/config.json) dispara `osascript -e 'display notification ...'` (banner macOS sin sonido desde 2026-04-27). NO ejecutar `afplay`/`osascript` manual — el hook se encarga. Permisos: System Settings > Notifications > "Script Editor".

## Referencias

- Vault path: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes`
- Memory dir: `04-Archive/99-obsidian-system/99-AI/memory/`
- Listener TS: [`/Users/fer/whatsapp-listener`](file:///Users/fer/whatsapp-listener)
- WhatsApp bridge: `~/repositories/whatsapp-mcp/whatsapp-bridge/store/messages.db`
- Cloudflared URL: `~/.local/share/obsidian-rag/cloudflared-url.txt`
- Auditoría salud sistema: `python scripts/audit_telemetry_health.py --days 7 --json`
- Eval baselines + perf history detallada: git log
