# CLAUDE.md

Local RAG over an Obsidian vault. Single-file: `rag.py` (~32.7k lines) + `mcp_server.py` (thin wrapper, 283 lines) + `web/` (FastAPI server, 6.1k lines + ~7.7k JS/HTML/CSS) + `tests/` (2,247 tests, 125 files). Resist package-split until real friction shows up.

Entry points (both installed via `uv tool install --editable .`):
- `rag` — CLI for indexing, querying, chat, productivity, automation
- `obsidian-rag-mcp` — MCP server (`rag_query`, `rag_read_note`, `rag_list_notes`, `rag_links`, `rag_stats`)

Fully local: Sqlite-vec + Ollama + sentence-transformers. **Exception**: Gmail + Calendar cross-source ingesters (Phase 1.b/c, pending) use OAuth Google via the Claude harness MCP — user override 2026-04-20, see `docs/design-cross-source-corpus.md §10.6`. WhatsApp + Reminders stay local (bridge SQLite + EventKit).

## Agent dispatch rule

**Any task that will edit ≥3 files MUST go through `pm` first.** No preguntas — invocar directamente:

```
Agent(subagent_type: "pm", prompt: "<goal + context + ruled-out + invariants at risk>")
```

The PM returns a dispatch plan (tasks, dependencies, parallel-safe flags, risks, validation). The main session executes the plan by spawning the named agents in the prescribed order — never silently skips PM and improvises.

Tasks that touch ≤2 files go directly to the owning agent (`rag-retrieval`, `rag-llm`, `rag-brief-curator`, `rag-ingestion`, `rag-vault-health`, `rag-integrations`, `developer-{1,2,3}`). Roster + ownership lives in `.claude/agents/README.md`.

When peers are active (`mcp__claude-peers__list_peers(scope: "repo")` returns >1), even ≤2-file tasks may need PM coordination — flag overlapping zones before editing.

## Auto-pull + commit + push rule

**Regla universal: cuando termino ALGO — feature, fix, refactor, limpieza, ajuste — el ciclo es siempre `git pull → git commit → git push origin master`. Sin preguntar.**

1. **Pull primero** (`git pull --rebase origin master` si hay cambios locales sin push, o `git pull` si el working tree está limpio). Evita que otro proceso — el user en su Mac, un cron, un Devin en paralelo — genere merge conflicts silenciosos.
2. **Commit** con un mensaje **completo y en palabras simples** de lo que hice, no jerga técnica seca. Formato esperado:
   - **Subject line**: `tipo(scope): resumen corto en español rioplatense (1 línea, ~70 chars)`
   - **Cuerpo**: explicación en párrafos cortos que cualquier humano (no solo yo en 6 meses) pueda leer: **qué cambié**, **por qué** lo cambié, **cómo lo medí** si aplica, y **cómo revertir** si rompe algo. Evitar tecnicismos gratuitos — si uso un término especializado (LoRA, RRF, idle-unload), que quede claro qué es en contexto.
   - **Trailer estándar Devin** al final:
     ```
     Generated with [Devin](https://cli.devin.ai/docs)
     Co-Authored-By: Devin <158243242+devin-ai-integration[bot]@users.noreply.github.com>
     ```
3. **Push** (`git push origin master`) inmediatamente después del commit. Si el pull del paso 1 hizo rebase y hay conflicts, los resolvés vos sin preguntar — nunca push con conflicts sin resolver.

Si los tests fallan o el build rompe, **NO commiteás** — arreglás primero.

Excepciones obvias: tareas exploratorias (investigar, responder preguntas, revisar diffs), cambios que el usuario explicitamente pidió no commitear, trabajo a medio camino (fix parcial + deferred-follow-up). Ante duda genuina, commit pero no push.

Esta regla NO cambia — es el comportamiento default para siempre. Si el user dice "hace el commit", "commit + push", "cerrá esto", o cualquier cosa que signifique "terminaste → guardá", ejecutás el ciclo completo pull → commit → push sin confirmar cada paso.

## Autonomous mode — empezar y terminar una feature sin interrupciones

Devin for Terminal tiene 4 [permission modes](https://docs.devin.ai/reference/permissions): **Normal** (default, pide permiso para writes + exec), **Accept Edits** (auto-aprueba edits dentro del workspace), **Bypass** (auto-aprueba TODO sin prompts), y **Autonomous** (Bypass + sandbox OS). Este proyecto está configurado para minimizar interrupciones aun en modo Normal — y cuando querés zero prompts absolutos, hay un interruptor global.

**Dos niveles de autonomía disponibles:**

1. **`.devin/config.json` — permissions pre-aprobadas por default** (siempre activas)
   - Allow-list (~80 reglas): todo el workflow normal del RAG auto-aprobado — `git *`, `rag *`, `uv *`, `pytest`, `sqlite3`, `launchctl` (menos remove/unload), `tail/head/cat/ls/find/grep/rg/awk/sed` (observabilidad), `.venv/bin/python`, `ollama ls/list/ps/show` (solo read), `curl localhost`, writes dentro del repo (`Write(**)` relativo al cwd).
   - Deny-list (10 reglas): cosas irreversibles que NUNCA queremos — `rm -rf`, `sudo`, `git reset --hard`, `git push --force`, `git branch -D`, `launchctl remove/unload`.
   - Ask-list (8 reglas): operaciones sensibles que SIEMPRE preguntan aunque estemos en Bypass mode — `rm` (sin `-rf`), `.env*`, `~/.ssh/`, `~/.aws/`, `~/.config/devin/`, writes al vault iCloud real, fetch a OpenAI/Anthropic APIs (el proyecto es local-first; si alguna vez tocamos esas URLs es un bug).
   - Resultado: en modo Normal, ~todo el flujo habitual corre sin pedir permiso. Las pausas que quedan son el handful de ops sensibles de la ask-list.

2. **Bypass mode — cero prompts absolutos** (explícito, por sesión)
   - Arrancar Devin con la flag: `devin --permission-mode bypass`
   - O durante una sesión: presionar **Shift+Tab** para alternar entre Normal → Accept Edits → Bypass → Plan (la barra de estado del terminal muestra el modo activo).
   - En Bypass mode todos los tool calls se auto-aprueban sin excepción — **salvo las reglas `deny` del `.devin/config.json`, que siempre ganan**. Por eso la deny-list está pensada como el safety net no-negociable: incluso en Bypass, no vas a `rm -rf` el repo por accidente.
   - Las reglas `ask` también siguen preguntando en Bypass (por diseño del sistema de precedencia: deny > ask > allow). Si querés que también se auto-aprueben en Bypass, moverlas a `allow` en el config.
   - Ideal para: "empezá esta feature, corré tests, commit + push, seguí con la siguiente" sin babysitting.

**Precedencia de permissions** (de mayor a menor prioridad, del doc oficial):
1. Org/team settings (si hay enterprise).
2. Session-level grants (lo que aprobaste interactivamente).
3. `.devin/config.local.json` (override local no commiteado).
4. `.devin/config.json` (este archivo, committeado al repo).
5. `~/.config/devin/config.json` (user-level, aplicado a todos los proyectos).

**Agregar overrides sin commitear** (ej. permisos extra que solo valen en tu Mac):
```bash
# ~/.config/devin/config.json o .devin/config.local.json (ignorado por git)
{
  "permissions": {
    "allow": ["Exec(docker)", "Fetch(https://*.my-company.com/*)"]
  }
}
```

**Rollback rápido si el config rompe algo**:
```bash
# Desactivar todas las permissions del proyecto:
mv .devin/config.json .devin/config.json.disabled
# O solo una regla específica: editar y quitar la línea.
```

## Commands

```bash
uv tool install --reinstall --editable .   # reinstall after code changes

# Core
rag index [--reset] [--no-contradict] [--vault NAME]  # incremental hash-based; --reset rebuilds; --vault override
rag index --source whatsapp [--reset] [--since ISO] [--dry-run] [--max-chats N]  # WA ingester (Phase 1.a)
rag watch                                  # watchdog auto-reindex (debounce 3s)
rag query "text" [--hyde --no-multi --raw --loose --force --counter --no-deep --session ID --continue --plain --source S[,S2] --vault NAME]
rag chat [--counter --no-deep --session ID --resume] # /save /reindex (or NL) work; create-intent tool-calling (`recordame X`, `cumple de Y el viernes`)
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
rag consolidate [--window-days 14 --threshold 0.75 --min-cluster 3 --dry-run --json]  # episodic memory Phase 2 → PARA

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
python scripts/backfill_entities.py [--dry-run --limit N --vault NAME]  # one-shot GLiNER entity extraction

# Automation
rag setup [--remove]                       # install/remove 11 launchd services

# Tests
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_foo.py::test_bar -q   # single test
```

Python 3.13, `uv`. Runtime venv: `.venv/bin/python`. Global tool: `~/.local/share/uv/tools/obsidian-rag/`.

### Env vars

- `OBSIDIAN_RAG_VAULT` — override default vault path. Collections are namespaced per resolved path (sha256[:8]). En la precedencia multi-vault, gana sobre el `current` del registry. `rag query --vault NAME` y `rag index --vault NAME` son equivalentes por-invocación sin mutar el env. Single-vault only en ambos comandos; para cross-vault query usar `rag chat --vault a,b`. Los cross-source ETLs (MOZE, WhatsApp, Gmail, Reminders, Calendar, Chrome, Drive, GitHub, Claude, YouTube, Spotify) se gatean por `_is_cross_source_target(vault_path)` — por default solo el `_DEFAULT_VAULT` (iCloud Notes) los recibe. Para opt-inear a otro vault agregar `"cross_source_target": "<name>"` al `~/.config/obsidian-rag/vaults.json`. Sin opt-in, `rag index --vault work` skippea los 11 ETLs con un log `[dim]Cross-source syncs: skip[/dim]` y solo indexa las `.md` reales del vault — evita la contaminación medida 2026-04-21 en que los ETLs copiaron 19 archivos MOZE al vault `work`. Tests: `tests/test_vaults.py` (10 casos sobre guard + flag).
- `RAG_OCR=0` — desactiva OCR en imágenes embebidas durante el indexing (default ON cuando `ocrmac` está disponible). El indexer extrae `![[img.png]]` + `![alt](img.png)` de cada nota (`_extract_embedded_images` en `rag.py`), llama `_ocr_image` vía Apple Vision (es-ES + en-US), y concatena el texto extraído al body antes de chunkear con un marker `<!-- OCR: path/to/img -->`. Cache persistente en `rag_ocr_cache` SQL table (key = abs path, invalidación por mtime — sin TTL). El hash del chunk (`_file_hash_with_images`) suma los mtimes de las imágenes embebidas, así que una screenshot actualizada fuerza reindex aunque el .md no haya cambiado. Soft-dep: ocrmac + pyobjc son macOS-only — en Linux el `try: import ocrmac` falla y el OCR skippea silently. Motivación (2026-04-21): notas tipo link-hub (`dev cycles.md` = un link + un `![[captura.png]]` con la tabla informativa en la PNG) eran invisibles al retrieval porque el body textual tenía ~0 signal; post-OCR el rerank score pasó de +0.0 a +1.1 y el LLM responde queries específicas ("cuándo termina el cycle 10.59"). Tests: `tests/test_ocr_embedded_images.py` (26 casos).
- `OLLAMA_KEEP_ALIVE` — passed to every ollama chat/embed call. Code default `-1` (forever, `rag.py:1608`); launchd plists set the same `-1` for symmetry. Accepts int seconds or duration string. **Historia**: default fue `"20m"` entre 2026-04-17 y 2026-04-21 por un Mac-freeze bug cuando `-1` pineaba command-r (~19 GB) como wired memory en 36 GB unificados. Vuelto a `-1` el 2026-04-21 tras el cambio del chat model default a qwen2.5:7b (~4.7 GB, bench 2026-04-18) — stack pinned ahora ≈ 8 GB (qwen2.5:7b + qwen2.5:3b helper + bge-m3), holgado en 36 GB. **Guard automático** (`chat_keep_alive()` en `rag.py:1626`): si el chat model efectivo está en `_LARGE_CHAT_MODELS` (command-r, qwen3:30b-a3b), el keep_alive se clampea a `_LARGE_KEEP_ALIVE="20m"` automáticamente — no más Mac freeze aunque `resolve_chat_model()` caiga a command-r por falta de qwen2.5:7b. **Rollback**: si vuelven los beachballs, exportá `OLLAMA_KEEP_ALIVE=20m` en tu shell antes de la primera invocación de `rag` — override sigue funcionando en ambas direcciones. Tests: [`tests/test_chat_keep_alive_guard.py`](tests/test_chat_keep_alive_guard.py) (10 casos cubren clamp, passthrough, override por env, degradación graciosa).
- `RAG_KEEP_ALIVE_LARGE_MODEL` — opt-out del auto-clamp de `chat_keep_alive()` para modelos grandes. Útil si tenés >64 GB unified memory y querés pinear command-r "forever" igualmente. Setear a una duración (`"4h"`) o entero segundos (`"14400"`). Sin esta variable, los modelos en `_LARGE_CHAT_MODELS` usan `"20m"`.
- `RAG_MEMORY_PRESSURE_DISABLE=1` — desactiva el memory-pressure watchdog (`start_memory_pressure_watchdog()` en `rag.py:8816`). El watchdog es un daemon thread que arranca desde `rag serve` y desde el startup del web server; cada `RAG_MEMORY_PRESSURE_INTERVAL` segundos muestrea `vm_stat` + `sysctl hw.memsize` vía `_system_memory_used_pct()` y si `(wired + active + compressed) / total >= RAG_MEMORY_PRESSURE_THRESHOLD` (default 85%) descarga proactivamente el chat model vía `ollama.chat(keep_alive=0)` y — si la presión sigue ≥ threshold tras eso — fuerza el unload del cross-encoder con `maybe_unload_reranker(force=True)` (bypasseando tanto el TTL idle check como el `RAG_RERANKER_NEVER_UNLOAD=1` del operador: bajo presión real de memoria preferimos el 5s cold-reload antes que un Mac freeze). **Motivación**: post `chat_keep_alive()` auto-clamp, queda un edge case — usuario corriendo `rag do` + web server + Claude Code + Chrome pesado puede saturar los 36 GB igualmente, porque `OLLAMA_MAX_LOADED_MODELS=2` NO es VRAM-aware y helper + embed + chat + reranker ya ocupan ~8 GB pinned. El watchdog es la red de seguridad activa. **No-op en Linux** (vm_stat es macOS-only) y en CLI one-shot (no se arranca en `rag query/chat/do`). Setear `RAG_MEMORY_PRESSURE_DISABLE=1` en tests/CI. Tests: [`tests/test_memory_pressure_watchdog.py`](tests/test_memory_pressure_watchdog.py) (17 casos: parse de vm_stat con 16 KB pages Apple Silicon, zero-total defensive, escalación chat→reranker, idempotencia, env overrides, `maybe_unload_reranker(force=True)` bypass del TTL).
- `RAG_MEMORY_PRESSURE_THRESHOLD` — umbral % para disparar el watchdog (default 85). Valores razonables: 80 (más agresivo, previene antes), 90 (más tolerante, menos unloads inútiles). Bajarlo si tenés <32 GB; subirlo si tenés >64 GB.
- `RAG_MEMORY_PRESSURE_INTERVAL` — intervalo de sampling en segundos (default 60). Menos = detecta pressure más rápido pero gasta más CPU en `vm_stat` forks. Más = reacción más lenta pero menos overhead.
- `RAG_STATE_SQL=1` — historically enabled the SQL telemetry store (20 `rag_*` tables in `ragvec/ragvec.db`). Post-T10 (2026-04-20) the JSONL fallback is gone and the flag is a **no-op** — neither writers nor readers consult it, SQL is the only path. Still set on every launchd plist for deployment-config symmetry / faster rollback if needed.
- `RAG_TRACK_OPENS=1` — switches OSC 8 link scheme from `file://` to `x-rag-open://` so CLI clicks route through `rag open` (ranker-vivo signal capture). Absent = no behavior change.
- `RAG_EXPLORE=1` — enable ε-exploration in `retrieve()` (10% chance to swap a top-3 result with a rank-4..7 candidate). Set on `morning`/`today` plists to generate counterfactuals. MUST be unset during `rag eval` — the command actively `os.environ.pop`s it and asserts, as a belt-and-suspenders guard.
- `RAG_RERANKER_IDLE_TTL` — seconds the cross-encoder stays resident before idle-unload (default 900).
- `RAG_RERANKER_NEVER_UNLOAD` — set to `1` in the web + serve launchd plists to pin the reranker in MPS VRAM permanently; sweeper loop still runs but skips `maybe_unload_reranker()`. Eliminates the 9s cold-reload hit after idle eviction. Cost: ~2-3 GB unified memory pinned. Safe on 36 GB with command-r + qwen3:8b resident.
- `RAG_LOCAL_EMBED` — set to `1` in the web + serve launchd plists to use in-process `SentenceTransformer("BAAI/bge-m3")` for query embedding instead of ollama HTTP (~10-30ms vs ~140ms). Requires BAAI/bge-m3 cached in `~/.cache/huggingface/hub/` — download once with `python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"` before enabling. Verify cosine >0.999 vs ollama embeddings of same text before enabling in production. Do NOT set for indexing/watch/ingest processes — bulk chunk embedding stays on ollama. Uses CLS pooling (same as ollama gguf). Post 2026-04-21 the CLI group (`cli()` in `rag.py:11894`) auto-sets this to `1` when invoking query-like subcommands (set in `_LOCAL_EMBED_AUTO_CMDS`: `query`, `chat`, `do`, `pendientes`, `prep`, `links`, `dupes`) unless the user already set it explicitly (both truthy + falsy overrides respected). Bulk paths (`index`, `watch`, ingesters) stay off the allow-list per the same invariant. **Cold-load warmup**: loading `SentenceTransformer` on MPS takes ~5.6s end-to-end (imports + weights + first encode JIT). `_warmup_local_embedder()` (rag.py, next to `query_embed_local`) centralises the preload and is invoked from `warmup_async()` (background daemon thread for CLI query-like subcommands) and from `rag serve`'s eager warmup. Before this, only `web/server.py:_do_warmup` preloaded it — `rag serve` and one-shot CLI invocations paid the 5.6s on the critical path of the first retrieve (confirmed 2026-04-21 in `rag_queries.extra_json`: embed_ms 3455/4137/4898 on the first few serve turns, dropping to 46ms post-warmup). Helper self-gates on `_local_embed_enabled()`, swallows exceptions, and is a no-op when the flag is falsy — safe to call unconditionally. Tests: `tests/test_warmup_local_embed.py` (16 cases). **Non-blocking gate (2026-04-21 evening)**: `_local_embedder_ready: threading.Event` fires *only* after `_warmup_local_embedder()` completes load + first successful encode. `query_embed_local()` checks `Event.is_set()` **before** entering the lock — if clear, returns None immediately so the caller falls back to ollama embed (~150ms). Pre-fix the main thread blocked on the lock for 5-12s on one-shot CLI when warmup_async raced the query (measured embed_ms up to 12014 in production). Long-running processes warm up at startup so the Event is set before the first user query, keeping the fast in-process path. Tests: `tests/test_local_embed_nonblocking.py` (9 cases: Event-clear bail timing, slow-loader doesn't block, post-warmup path works, concurrent warmup+5 queries stay <100ms).
- `RAG_EXPAND_MIN_TOKENS` — threshold for the `expand_queries()` short-query perf gate (default `4`, `rag.py:7821`). Queries shorter than this token count (split by whitespace) skip the qwen2.5:3b paraphrase call (~1-3s saved). Raise to be more aggressive about skipping paraphrase; lower to restore pre-2026-04-21 behaviour (`<= 2` tokens skipped). Web server already forced `multi_query=False` globally (`web/server.py:3648`), this makes CLI caller-by-caller.
- `RAG_CITATION_REPAIR_MAX_BAD` — threshold for the citation-repair perf gate (default `3`, `rag.py:109`). When `verify_citations()` returns more than this many invented paths, the repair round-trip is skipped entirely (rationale: heavily hallucinated answers rarely recover under a single-shot repair and the 5-8s non-streaming call dominates latency). Set to `0` to disable citation-repair completely. Applies to both `rag query` (`rag.py:13924`) and `rag chat` (`rag.py:15043`) paths.
- `OBSIDIAN_RAG_NO_APPLE=1` — disables Apple integrations (Calendar, Reminders, Mail, Screen Time) entirely. Useful on non-macOS hosts or when Full Disk Access is not granted.
- `RAG_TIMEZONE` — IANA tz string used by `_parse_natural_datetime` for ISO-with-tzinfo inputs (ISO strings with `Z` / offset). Default `America/Argentina/Buenos_Aires` (UTC-3 / UTC-2 depending on DST, but AR stays UTC-3 year-round as of 2019). Naive datetimes (user typing "mañana 10am") are interpreted relative to anchor and don't hit the TZ conversion path; only IS0-8601 inputs with tzinfo do. Week-start follows dateparser's ISO default (Monday).
- `OBSIDIAN_RAG_MOZE_FOLDER` — override MOZE ETL target folder inside the vault (default `02-Areas/Personal/Finanzas/MOZE`).
- `OBSIDIAN_RAG_WATCH_EXCLUDE_FOLDERS` — comma-separated vault-relative folders `rag watch` must ignore. Default `"03-Resources/WhatsApp"` (WA dumps re-fire the handler dozens of times per minute via periodic ETL; they're picked up by manual/periodic `rag index` instead).

- `RAG_ADAPTIVE_ROUTING` — activa el pipeline adaptativo (Mejora #3). Cuando está ON: (a) `_should_skip_reformulate()` saltea el helper call de reformulación para intents metadata-only (`count`, `list`, `recent`, `agenda`, `entity_lookup`) — ahorra ~1-2s; (b) fast-path dispatch cuando `top_score >= _LOOKUP_THRESHOLD` (0.6) — usa `qwen2.5:3b` con `num_ctx=2048` y saltea citation-repair, pure perf win para queries simples. **Default OFF** (`os.environ.get("RAG_ADAPTIVE_ROUTING") == "1"`). Sin regresión eval: ambas runs ON y OFF producen resultados bit-idénticos en `rag eval` (validado 2026-04-21, ver §Eval baselines). Override de debug: `RAG_FORCE_FULL_PIPELINE=1` fuerza todos los stages aun con adaptive ON. Ver `docs/improvement-3-adaptive-routing-design.md`. **Plan de activación**: flipear a ON por default (cambiar a `os.environ.get("RAG_ADAPTIVE_ROUTING", "1") == "1"`) una vez que `rag eval` vuelva a superar el floor de singles 76.19% de forma estable (LLM non-determinism causó variabilidad en la sesión de cierre 2026-04-21).
- `RAG_ENTITY_LOOKUP` — activa el dispatch de `handle_entity_lookup()` para el intent `entity_lookup` en `query()` (Mejora #2). **Default ON tras 2026-04-21** (`os.environ.get("RAG_ENTITY_LOOKUP", "").strip().lower() not in ("0","false","no")`). Motivación del flip: `scripts/backfill_entities.py` corrió sobre el corpus activo (2026-04-21 evening, 4150 chunks procesados en ~6 min) y pobló `rag_entities` con 2022 entidades + `rag_entity_mentions` con 6520 mentions (cobertura 71% = 2953/4150 chunks con al menos una entidad). Top persons: Astor (129 mentions), Juli (113), Maria (75), Fernando (51). Top orgs: Moka (94), BBI (35). Sources: vault (5784 mentions) + calendar (672) + gmail (47) + whatsapp (17). Smoke-test validado: `rag query "todo lo que tengo de Astor"` devuelve síntesis cross-source agregada (biografía + conversaciones María + Flor + WhatsApp + Calendar cumpleaños). Handler tiene fall-through a semantic si la entidad no resuelve o la tabla está vacía → sin riesgo de regresión (peor caso: +1 SQL query ~10 ms antes de caer al pipeline legacy). Para rollback: `export RAG_ENTITY_LOOKUP=0`. Instalar la dep: `uv pip install obsidian-rag[entities]` (GLiNER + transitivos, ~500 MB modelo cacheado en `~/.cache/huggingface/hub/`). Ver `docs/improvement-2-entity-retrieval-plan.md`.
- `RAG_EXTRACT_ENTITIES` — **default ON tras 2026-04-21** (permissive: `val not in ("0","false","no")`). Popula `rag_entities` + `rag_entity_mentions` automáticamente durante indexing — evita que la tabla quede stale cuando el corpus crece. Wired en 5 sitios: `rag.py:_index_single_file` (incremental vault + `rag watch`), `rag.py` bulk vault path (`rag index --reset`), y los 4 ingesters cross-source (`scripts/ingest_whatsapp.py`, `ingest_reminders.py`, `ingest_gmail.py`, `ingest_calendar.py`). Cada caller pasa el `source` correspondiente (`"vault"` / `"whatsapp"` / etc.) así las mentions quedan atribuidas. Silent-fail si `gliner` no está instalado (sticky-fail via `_gliner_load_failed`: primer ImportError/load-fail marca el flag y ningún retry en la vida del proceso — sin log spam). Costo: ~0.16s/chunk cuando gliner está disponible + cold-load GLiNER ~5s una vez por proceso; para `rag index --reset` sobre 4150 chunks eso son ~11 min extras. Para skipear durante un reindex grande y correr el backfill después: `RAG_EXTRACT_ENTITIES=0 rag index --reset && python scripts/backfill_entities.py`. El one-shot [`scripts/backfill_entities.py`](scripts/backfill_entities.py) queda útil para re-popular tras cambios de schema (bump `_COLLECTION_BASE`) o para llenar la tabla en hosts donde `gliner` se instaló después del indexing inicial.
- `RAG_NLI_GROUNDING` — activa NLI grounding post-citation-repair (Mejora #1). Carga `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (~400 MB en MPS fp32), extrae claims del LLM response, y agrega un bloque NLI-grounded al final con porcentaje de claims soportados/neutros/contradictorios. En web: emite SSE event `nli_grounding` + panel UI coloreado. **Default OFF** (truthy check: `os.environ.get("RAG_NLI_GROUNDING", "").strip() not in ("", "0", "false", "no")`). No aplica para intents `count`/`list`/`recent`/`agenda` (controlado por `RAG_NLI_SKIP_INTENTS`). **No activar en producción aún**: cold-load mDeBERTa ~2s + inference ~300-1000ms por query impactan UX antes de medir en ≥50 queries reales. Activar con `RAG_NLI_IDLE_TTL` (default 900s para idle-unload). Ver `docs/improvement-1-nli-integration-plan.md`.

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
      → [RAG_ADAPTIVE_ROUTING: skip reformulate_query if metadata-only intent]
      → expand_queries (3 paraphrases, ONE qwen2.5:3b call)
      → embed(variants) batched bge-m3
      → per variant: sqlite-vec sem + BM25 (accent-normalised, GIL-serialised — do NOT parallelise)
      → RRF merge → dedup → expand to parent section (O(1) metadata)
      → cross-encoder rerank (bge-reranker-v2-m3, MPS+fp32)
      → [RAG_ADAPTIVE_ROUTING + score≥0.6: fast-path → qwen2.5:3b num_ctx=2048, skip citation-repair]
      → graph expansion (1-hop wikilink neighbors, always on)
      → [auto-deep: if confidence < 0.10, iterative sub-query retrieval]
      → top-k → LLM (streamed)
      → citation-repair [if needed + score<threshold]
      → [RAG_NLI_GROUNDING: NLI claim verification, skip count/list/recent/agenda]

Intent dispatch: semantic | synthesis | comparison | count | list | recent | agenda | entity_lookup
```

**Graph expansion** (always on): after rerank, top-3 results expand via 1-hop wikilink neighbors (`_build_graph_adj` + `_hop_set`). Up to 3 graph neighbors added as supplementary LLM context marked `[nota relacionada (grafo)]`. Cost: in-memory graph lookups, negligible.

**Auto-deep retrieval**: when top rerank score < `CONFIDENCE_DEEP_THRESHOLD` (0.10), `deep_retrieve()` auto-triggers: helper model judges sufficiency → generates focused sub-query → second retrieve pass → merge results. Max 3 iterations. Disable with `--no-deep`.

**Rerank pool** (`RERANK_POOL_MAX = 15`, dropped from 30 on 2026-04-21): cap on candidates scored by the cross-encoder. `rag eval --latency` bench sweep (pool=30/25/20/15 × 60 singles + 30 chains):

| pool | singles hit@5 | singles MRR | chains MRR | singles P50 | singles P95 |
|---|---|---|---|---|---|
| 30 | 71.67% | 0.679 | 0.740 | 2157 ms | 4704 ms |
| 25 | 71.67% | 0.679 | 0.740 | 1899 ms | 2333 ms |
| 20 | 71.67% | 0.681 | 0.757 | 1458 ms | 1914 ms |
| **15** | **71.67%** | **0.681** | **0.790** | **1163 ms** | **1577 ms** |

Pool=15 domina: hit@5 bit-identical, MRR chains **+5pp** (less noise in the rerank head), P95 singles **−66%** (−3.1s). The pre-existing comment claiming "pool=25 pierde 1 hit (90.48% vs 95.24%)" was measured against a smaller/older `queries.yaml` and does not reproduce on the current corpus. Web `/api/chat` still passes `rerank_pool=5` explicitly; CLI inherits the default. `rag tune` now fits weights on pool=15 feature vectors (old weights fit on pool=30 should be re-tuned — default weights are unaffected).

**Corpus cache** (`_load_corpus`): BM25 + vocab built once, invalidated by `col.count()` delta. Cold 341ms → warm 2ms. Do not touch without re-measuring.

**Cache locks (concurrency invariants)** — the web server hits rag from multiple threads concurrently, so every module-level cache that gets written more than once is protected:

| Cache | Lock | Guards against |
|---|---|---|
| `_context_cache` / `_context_cache_dirty` | `_context_cache_lock` (Lock) | Double lazy-init + `json.dumps` during mutation |
| `_synthetic_q_cache` / `_synthetic_q_cache_dirty` | `_synthetic_q_cache_lock` (Lock) | Same |
| `_mentions_cache` | `_mentions_cache_lock` (Lock) | Parallel folder re-scan + overwrite race |
| `_embed_cache` | `_embed_cache_lock` (Lock) | LRU eviction race |
| `_corpus_cache` + `_pagerank_cache` + `_pagerank_cache_cid` | `_corpus_cache_lock` (RLock) | Partial-build reads from watchdog invalidation |
| `_contacts_cache` | `_contacts_cache_lock` (Lock) | Apple Contacts lookup race |

LLM calls (`_generate_context_summary`, `_generate_synthetic_questions`) run **outside** the lock so concurrent requests don't serialise on helper-model latency. Tests: `tests/test_cache_concurrency.py` (8 cases — presence, lazy-init uniqueness, save-during-mutation safety for each cache).

### Indexing

Chunks 150–800 chars, split on headers + blank lines, merged if < MIN_CHUNK. Each chunk: `embed_text` (prefixed `[folder|title|area|#tags]` + contextual summary), `display_text` (raw), `parent` metadata (enclosing section, ≤1200 chars). Hash per file → re-embed only on change. `is_excluded()` skips `.`-prefixed segments.

**Contextual embeddings** (v8→v9): `get_context_summary()` generates a 1-2 sentence document-level summary per note via qwen2.5:3b, prepended to each chunk's `embed_text` as `Contexto: ...`. Cached by file hash in `~/.local/share/obsidian-rag/context_summaries.json`. Notes < 300 chars skip summarization. The original commit claimed "+11% chain_success" but that figure was never replicated against the current queries.yaml — treat as unverified.

**Temporal tokens** (removed 2026-04-20, v10→v11): `temporal_token()` was defined in commit d6e1073 to append `[recent]`/`[this-month]`/`[this-quarter]`/`[older]` to the embedding prefix but was never actually wired into `build_prefix()` (dead code). The 2026-04-20 A/B wired it in (v10) + reindexed + re-ran `rag eval`: singles hit@5 / MRR / chains hit@5 / chain_success all within noise vs the v9 baseline (singles MRR −0.011, others bit-identical). Feature removed (v11) along with the `temporal_token()` function. If recency ever matters empirically, resurrect from git history rather than reintroducing the dead code.

**`created_ts` backfill — SQL-persisted marker** (2026-04-21): `_maybe_backfill_created_ts()` populates `created_ts` metadata en chunks que no la tenían (pre-temporal-feature indexes). Antes: el guard `_CREATED_TS_BACKFILL_DONE` era solo in-process — cada restart del web daemon (medido en `web.log`: **149 restarts** en ~3 días por `[idle-sweep] ... disabled` prints) re-escaneaba 3600+ chunks + 580 file reads en la primera query con `date_range`. Ahora el marker se persiste en `rag_schema_version` (table_name sentinel `_created_ts_backfill_complete`, version=1) tras un `col.update()` exitoso o cuando el escaneo determina que todos los chunks ya tienen el campo. `_created_ts_backfill_persisted()` lee la fila antes de escanear → skip instantáneo tras el primer proceso exitoso. Relevancia perf: el pre-fix del `col.update` (commit `d41bfe7`) dejaba el `AttributeError` causar **queries de hasta 200s** (retrieve=202652ms medido en web.log) — tras el fix el escaneo vale ~1s pero era gratis en restarts. Ahora es ~0s en restarts. Tests: [`tests/test_created_ts_backfill_persist.py`](tests/test_created_ts_backfill_persist.py) (9 casos: marker presence/absence, round-trip, idempotencia, skip-si-persistido, empty vault no-mark, mark-on-no-missing, defensivos contra DB broken, sentinel name no colide con tablas reales).

**Graph PageRank**: `_graph_pagerank()` computes authority scores over the wikilink adjacency graph (power iteration, <10ms). Cached per corpus. Used as a tuneable ranking signal (`graph_pagerank` weight) and to sort graph expansion neighbors.

**Schema changes**: bump `_COLLECTION_BASE` (currently `obsidian_notes_v11`). Per-vault suffix = sha256[:8] of resolved path.

### Model stack

| Role | Model | Notes |
|------|-------|-------|
| Chat | `resolve_chat_model()`: qwen2.5:7b > qwen3:30b-a3b > command-r > qwen2.5:14b > phi4 | qwen2.5:7b default tras bench 2026-04-18 (total P50 5.9s vs 37s de command-r); fallbacks high-quality disponibles. |
| Helper | `qwen2.5:3b` | paraphrase/HyDE/reformulation |
| Embed | `bge-m3` | 1024-dim multilingual |
| Reranker | `BAAI/bge-reranker-v2-m3` | `device="mps"` + `float32` forced — do NOT switch to fp16 on MPS (score collapse to ~0.001, verified 2026-04-13); CPU fallback = 3× slower. |
| NLI (opt-in) | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | ~400 MB MPS fp32, idle-unload via `RAG_NLI_IDLE_TTL` (default 900s). Gate: `RAG_NLI_GROUNDING`. |

All ollama calls use `keep_alive=OLLAMA_KEEP_ALIVE` — default `-1` (forever) in code (`rag.py:1608`) since 2026-04-21; launchd plists use the same value for symmetry. `CHAT_OPTIONS`: `num_ctx=4096, num_predict=384` — don't bump unless prompts grow.

**Pattern**: helper for cheap rewrites, chat model for judgment. Contradiction detector MUST use chat model (qwen2.5:3b proved non-deterministic + malformed JSON on this task).

### Confidence gate

`top_score < 0.015` (CONFIDENCE_RERANK_MIN) + no `--force` → refuse without LLM call. Calibrated for bge-reranker-v2-m3 on this corpus. Re-calibrate if reranker changes.

### Generation prompts

- `SYSTEM_RULES_STRICT` (default `rag query`, `semantic` intent): forbids external prose.
- `SYSTEM_RULES` (`--loose`, always in chat): allows `<<ext>>...<</ext>>` rendered dim yellow + ⚠.
- `SYSTEM_RULES_LOOKUP` (intent `count`/`list`/`recent`/`agenda`): terse 1-2 sentences, exact "No encontré esto en el vault." refusal.
- `SYSTEM_RULES_SYNTHESIS` (intent `synthesis`): cross-reference ≥2 overlapping sources, must surface tension. Fires via `_INTENT_SYNTHESIS_RE` — triggers on `resumí/resumime/síntesis/sintetizame/integrame todo lo que hay sobre X`, `qué dice el vault sobre X`, `summary of|synthesis of|synthesize X`. Plain `qué es X` stays `semantic`.
- `SYSTEM_RULES_COMPARISON` (intent `comparison`): explicit `X dice A / Y dice B / Diferencia clave: …` structure. Fires via `_INTENT_COMPARISON_RE` — triggers on `diferencia(s)? entre X y Y`, `comparame X con Y`, `X vs/versus Y`, `en qué se diferencian X y Y`, `qué distingue X de Y`, `contraste entre X y Y`. Checked BEFORE synthesis (precedence) because `X vs Y` is inherently comparative. 49 tests in [`tests/test_classify_intent.py`](tests/test_classify_intent.py); golden queries in [`queries.yaml`](queries.yaml) at the "Comparison intent" + "Synthesis intent" sections.
- Routed through `system_prompt_for_intent(intent, loose)` at generation time (both `query()` and `chat()` paths). `--loose` always maps to `SYSTEM_RULES` for every intent.

### Agenda intent (2026-04-21 evening)

Fired by `_INTENT_AGENDA_RE` in `classify_intent`, checked **before `recent`** because both share temporal tokens ("hoy" / "esta semana" / "este mes"). Pre-fix, queries like `"qué tengo esta semana"` fell into `handle_recent` → listed vault notes sorted by `modified` desc. Post-calendar-ingest (368 events + 37 reminders), that was a user-facing bug — calendar was invisible to browsing queries.

Regex patterns (positive fires):
- Possessives: `mi agenda` / `mis eventos|calendario|reuniones|turnos|citas`
- `qué tengo|hay|tenemos` + temporal anchor (hoy/mañana/esta semana/etc)
- `qué eventos|reuniones|citas|turnos tengo|hay|tenemos` (event noun carries agenda semantic even without temporal)
- `agenda de|del|para X`

Negative guards (stays semantic/recent): specific lookups like `"cuándo es el workshop de X"`, `"turno con Y"` (source-weighted retrieve already resolves) + `"notas modificadas hoy"` / `"últimas notas"` stay on `recent`.

`handle_agenda(col, params, limit=20, *, question=None)` in `rag.py` filters metas by `source ∈ _AGENDA_SOURCES = frozenset({"calendar", "reminders"})` via `normalize_source`, sorts by `created_ts` desc (start_ts for calendar events, creation anchor for reminders). Gracefully handles missing/string/malformed `created_ts` (coerce + try/except → 0.0).

**Temporal window filter (v2)**: `_parse_agenda_window(question, *, now=None) → (ts_start, ts_end) | None` parses the anchor and returns a half-open epoch interval. Dispatch order (narrowest first, critical for regex overlaps):
1. Day anchors (`pasado mañana` before `mañana`, then `hoy`)
2. Weekend (`el finde` / `este finde` / `próximo finde` — via dateparser for "finde" = saturday)
3. Week (`esta semana` / `la semana que viene` / `la próxima semana`)
4. Month (`este mes` / `el próximo mes` / `el mes que viene`)
5. Year (`este año` / `el próximo año` / `el año que viene`)
6. Weekday-specific (`el viernes` / `el próximo lunes` — delegated to `_parse_natural_datetime` so rollforward math matches the propose tool)

All windows snap to 00:00 local (naive datetime, `RAG_TIMEZONE` default `America/Argentina/Buenos_Aires`). December wraparound (`el próximo mes` from Dec → Jan next year) tested. Half-open semantics: event at exactly `end_ts` belongs to the NEXT window.

When `question` is passed to `handle_agenda` and the parser returns a window, post-source-filter metas are narrowed to `ts_start <= created_ts < ts_end`. No anchor (e.g. plain `"mi agenda"`) → fallback to top-20 by ts desc.

Tests: `tests/test_classify_intent.py` (+25 agenda cases — positives across possessives/tengo/hay/eventos/agenda shapes, negatives to guard against over-matching), `tests/test_handle_agenda.py` (13 cases: source filter + sort + limit + null ts + dedup + `_AGENDA_SOURCES` lock + 4 window-filter integration cases), `tests/test_agenda_window.py` (25 cases: every anchor type + ordering guards + half-open + December wraparound).

### Prompt-injection defence (passive, 2026-04-21)

Cross-source corpus (Gmail / WhatsApp, user override §10.5 = indexá-todo) means a hostile email or WhatsApp can land in the index and reach the LLM context through a legitimate semantic match. Two passive layers in `rag.py` (right above `SYSTEM_RULES`):

- **Redaction** — `_redact_sensitive(text)` strips OTPs, tokens, passwords, CBU / card / account numbers, CVV/CVC/CCV *before* the chunk body hits the LLM. Cue-gated (value must sit next to `code|token|password|cbu|cvv|...` within ~20 chars) with a digit-presence lookahead to avoid false positives (the regex `cue="code"` alone tripped on prose like "the code base is large"; the `(?=[A-Z0-9]*[0-9])` lookahead drops those matches). Embeddings stay indexed with the raw value — this defence only hides values from the LLM at generation time. NOT a barrier against a motivated attacker with vault write access.

- **Context isolation** — `_format_chunk_for_llm(doc, meta, role)` centralises the per-chunk wrapping: header `[{role}: {note}] [ruta: {file}]` stays identical (citation-repair + path-extraction rules in every `SYSTEM_RULES*` keep working unchanged), body goes between `<<<CHUNK>>>...<<<END_CHUNK>>>` fences after redaction. Paired with `_CHUNK_AS_DATA_RULE` — a prepended `REGLA 0` in every `SYSTEM_RULES*` variant that tells the model content between fences is DATA to cite, NEVER instructions. Hint to the classifier, not a cryptographic barrier; a sufficiently capable model may still follow injected instructions.

Callers of `_format_chunk_for_llm`: `build_progressive_context` (primary + multi-vault), `query()` graph section, `chat()` graph section, `rag serve` generation block. All four legacy inline formats `f"[nota: {m['note']}] [ruta: {m['file']}]\n{d}"` were migrated in the 2026-04-21 pass — any new caller assembling chunks for an LLM prompt MUST go through the helper so redaction + fencing stay centralised.

Tests: [`tests/test_prompt_injection_defence.py`](tests/test_prompt_injection_defence.py) (61 cases: OTP positives en ES + EN + unaccented, bank secret positives, negative cases for version strings / dates / commit SHAs / prose like "code base", chunk wrapper contract, `REGLA 0` presence + ordering in every `SYSTEM_RULES*`).

### Name-preservation guardrail (2026-04-21)

Separate from prompt-injection, a second always-on clause `_NAME_PRESERVATION_RULE` in `rag.py` (right below `_CHUNK_AS_DATA_RULE`) blocks a distinct failure mode: the LLM "correcting" proper nouns it doesn't recognise. Regression seed: user asked about "Bizarrap" (Argentine producer), the vault had no musical info, the model answered refusing about **"Bizarra"** — silently swapping a rare proper noun for a commoner dictionary word. Rule is prepended right after `_CHUNK_AS_DATA_RULE` in every `SYSTEM_RULES*` (ordering: chunks-as-data → names → variant body) so the model copies user-supplied names TEXTUAL and treats unknown terms as valid proper nouns it doesn't know. Verify with `python -c "import rag; print(rag._NAME_PRESERVATION_RULE[:80])"`. Tests: [`tests/test_name_preservation_rule.py`](tests/test_name_preservation_rule.py) (46 cases: constant presence, per-variant inclusion + ordering + no-duplication, `system_prompt_for_intent()` coverage for every intent incl. `loose=True`).

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

Weights in `~/.local/share/obsidian-rag/ranker.json` (written by `rag tune --apply`). Defaults `recency_always=0, tag_literal=0, click_*=0, dwell_score=0` preserve pre-tune behavior. Behavior knobs are inert until `rag_behavior` accumulates signal and `rag tune` finds non-zero weights.

Behavior priors (`_load_behavior_priors()`): read from `rag_behavior` (SQL), cached per MAX(ts). Positive events: `open`, `positive_implicit`, `save`, `kept`. Negative: `negative_implicit`, `deleted`. CTR uses Laplace smoothing `(clicks+1)/(impressions+10)`.

### GC#2.C — Reranker fine-tune (infra ready, gated on data)

- **Estado**: infra completa + gate E2E validado, esperando ≥20 rows con `corrective_path` en `rag_feedback` antes de re-correr con chances de promover.
- **Run anterior fallido** (`~/.cache/obsidian-rag/reranker-ft-20260422-124112/`, 2.1 GB, cleanup manual pendiente con `rm -rf` por ask-rule): −3.3pp chains hit@5 vs baseline. Causa: 1 epoch undertraining + señal positiva ruidosa (55 turns positivos × ~4 chunks cada uno, todos label=1.0 aunque solo uno era golden).
- **Run 2 noisy** (`~/.cache/obsidian-rag/reranker-ft-20260422-182127/`, 2.1 GB, gate=0 override, 3 epochs, ver [`docs/finetune-run-2026-04-22.md`](docs/finetune-run-2026-04-22.md)): **mismo −3.3pp chains**. Loss convergió de 0.96 a 0.13 (overfitting claro en epoch 3); val margin +0.455 (pos 0.515 vs neg 0.060). El modelo aprendió muy bien la data ruidosa — por eso regresionó chains. **Gate E2E validado**: detectó la regresión y NO promovió. Conclusión firme: sin `corrective_path` limpios, el fine-tune no supera baseline con esta config — hace falta la señal limpia, no más epochs.
- **Fix aplicado**: [`scripts/finetune_reranker.py`](scripts/finetune_reranker.py) ahora lee `corrective_path` de `rag_feedback.extra_json` y lo usa como único positivo cuando está presente. Fallback a todos los paths cuando no.
- **Gate pre-training**: `RAG_FINETUNE_MIN_CORRECTIVES` (default 20). Aborta con exit 5 si la señal limpia es insuficiente.
- **Cómo generar data**: `rag chat` + thumbs-down en turnos malos — el prompt pide el path correcto (commit `23f2899`). La web UI tiene el mismo picker (commit `33ed3f0`). O el skill `rag-feedback-harvester` para labeleo batch.
- **Monitoreo**: `sqlite3 ~/.local/share/obsidian-rag/ragvec/ragvec.db "SELECT COUNT(*) FROM rag_feedback WHERE json_extract(extra_json, '\$.corrective_path') IS NOT NULL AND json_extract(extra_json, '\$.corrective_path') <> ''"` — conteo directo de corrective_paths disponibles.
- **Re-trigger**: `python scripts/finetune_reranker.py --epochs 2` una vez que el gate lo permita (2 epochs, no 3 — la loss convergió a 0.22 en epoch 2 en el run noisy; epoch 3 es overfit puro). El gate de `rag eval` decide promoción via symlink `~/.cache/obsidian-rag/reranker-ft-current`.

## Key subsystems — contracts only

Subsystems have autodescriptive docstrings in `rag.py` and dedicated test files. Only contracts/invariants here.

**Sessions**: JSON per session in `sessions/<id>.json`. TTL 30d, cap 50 turns, history window 6. IDs validated `^[A-Za-z0-9_.:-]{1,64}$`; invalid → mint fresh. WhatsApp passes `wa:<jid>`.

**Episodic memory** (`web/conversation_writer.py`, silent write): after every `/api/chat` `done` event, `web/server.py` spawns a daemon thread via `_spawn_conversation_writer` that appends the turn to `00-Inbox/conversations/YYYY-MM-DD-HHMM-<slug>.md`. One note per `session_id`, multi-turn. Hand-rolled YAML frontmatter (`session_id`, `created`, `updated`, `turns`, `confidence_avg`, `sources`, `tags`). The session_id → relative_path index lives in `rag_conversations_index` (SQL, upsert). Atomic .md write via `os.replace`; concurrent writes for the same session are not a production scenario (one /api/chat per session at a time) so the pre-T10 whole-body fcntl lock is gone — SQL upsert inside `BEGIN IMMEDIATE` handles index serialisation. Errors land on `LOG_PATH` as `conversation_turn_error` — never raised, never SSE-emitted. Raw conversations are **excluded from the search index** (`is_excluded`: `00-Inbox/conversations/` + `04-Archive/conversations/`) — they leak LLM hallucinations back as ground truth if indexed (T6 regression). Curation happens via `rag consolidate` (Phase 2, below), not by manual editing.

**Conversation writer shutdown drain** (`_CONV_WRITERS` + `@app.on_event("shutdown")`): every in-flight writer registers in `_CONV_WRITERS` and removes itself when `_persist_conversation_turn` returns (success or exception). On server stop the `_drain_conversation_writers` hook joins each pending thread with a combined 5s budget. Anything still running falls through the normal exception path, lands in `_CONV_PENDING_PATH` (`conversation_turn_pending.jsonl`), and gets re-applied at next startup by `_retry_pending_conversation_turns`. Threads stay `daemon=True` on purpose — a wedged SQL/disk write must not block process exit. Stragglers past the cap are logged once as `conversation_writer_shutdown_timeout` to `LOG_PATH`. Tests: `tests/test_web_conv_shutdown.py` (6 cases covering self-remove, empty drain no-op, quick-writer wait, 5s cap with wedged writer, spawn tracking, exception-path release).

**Episodic memory — Phase 2 consolidation** (`scripts/consolidate_conversations.py`, `rag consolidate`, weekly launchd): scans `00-Inbox/conversations/` in a rolling window (default 14d), embeds each as `first_question + first_answer` via bge-m3, groups by connected components on cosine ≥ 0.75, promotes clusters ≥ 3 to PARA. Target folder picked by regex over cluster bodies: ≥2 matches against `_PROJECT_PATTERNS` (ES+EN action verbs / future-tense / dates) → `01-Projects/`, else `03-Resources/` (conservative default). Synthesis via `resolve_chat_model()` + `CHAT_OPTIONS` — one non-streaming call per cluster (~6s). Consolidated note gets frontmatter `type: consolidated-conversation`, wikilink section to originals (now under `04-Archive/conversations/YYYY-MM/`), and wikilinks to every source note union'd across turns. Originals move via `shutil.move`; archive folder is also excluded from the index so archived raws don't compete with the curated synthesis. Errors per cluster are swallowed (cluster entry gets `error` key; other clusters proceed). Log schema at `~/.local/share/obsidian-rag/consolidation.log` (JSONL: `{run_at, window_days, n_conversations, n_clusters, n_promoted, n_archived, duration_s, dry_run, clusters: [...]}`). CLI flags: `--window-days`, `--threshold`, `--min-cluster`, `--dry-run`, `--json`. Launchd: `com.fer.obsidian-rag-consolidate` (Mondays 06:00 local), registered in `_services_spec()`, installable via `rag setup`.

**Web chat tool-calling** (`web/tools.py`, 9 tools): `search_vault`, `read_note`, `reminders_due`, `gmail_recent`, `finance_summary`, `calendar_ahead`, `weather` (read-only) + `propose_reminder`, `propose_calendar_event` (create-intent, implementations live in `rag.py` — `web/tools.py` re-exports). `/api/chat` runs a 2-phase tool loop: pre-router (`_detect_tool_intent`, keyword → forced read tool) + optional LLM tool-decide round (gated by `RAG_WEB_TOOL_LLM_DECIDE`, default OFF). Create intent ("recordame", "creá un evento", ...) is detected by `_detect_propose_intent` (defined in `rag.py`, shared between web + CLI) which FORCES the LLM decide round ON for that query — propose tools need LLM arg extraction, can't run from pre-router. Create tools auto-create the reminder/event if the datetime is unambiguous (SSE `created` event → inline `╌ ✓ agregado...` chip, reminders get an inline `deshacer` link backed by `DELETE /api/reminders/{id}`, events don't since Calendar.app AppleScript delete is unreliable) or fall back to a `proposal` card with ✓ Crear / ✗ Descartar when the parser flagged `needs_clarification`. Low-level helpers `_parse_natural_datetime` (dateparser + qwen2.5:3b fallback, `_preprocess_rioplatense_datetime` for `18hs`/`al mediodía`/`X que viene`), `_parse_natural_recurrence` (regex over ES/EN patterns), `_create_reminder` (supports `due_dt`, `priority`, `notes`, `recurrence`), `_create_calendar_event` (via Calendar.app AppleScript — iCloud writable, unlike the JXA read path), `_has_explicit_time` (auto all-day detection), `_delete_reminder`, `_delete_calendar_event` all in `rag.py`. Recurrence on Reminders is best-effort (inner try/on error) since the property is macOS-version-dependent; on Calendar it's stable.

**CLI chat create-intent** (`rag chat`): same `_detect_propose_intent` + same propose tools, but ported to terminal via `_handle_chat_create_intent` at the top of every turn's input. Single-round ollama tool-decide with `_CHAT_CREATE_OVERRIDE` prompt + `tools=[propose_reminder, propose_calendar_event]` only; on tool_call → dispatches + renders a Rich chip `╌ ✓ agregado...` in the same `sáb 25-04 (todo el día)` / `lun 20-04 22:27` shape as the web UI (hard-coded `es-AR` weekdays because `%a` is locale-dependent). command-r's `{parameters: {...}}` arg wrapping is unwrapped the same way as `rag do`. Returns `(handled, created_info)` where `created_info` carries `{kind, reminder_id, title}` on a successful reminder create (None for events — Calendar.app AppleScript delete is unreliable, matches web UX which shows no undo for events). The chat loop stashes `created_info` in `last_created` (session-local, not persisted) and the `/undo` slash command dispatches `_delete_reminder(last_created["reminder_id"])` to reverse the most recent create; `last_created` clears on success so a second `/undo` returns "nothing to undo". Tests: `tests/test_chat_create_handler.py` (8 cases) + `tests/test_chat_undo.py` (5 cases) — all monkeypatched, no live ollama.

**Rioplatense datetime normalization** (`_preprocess_rioplatense_datetime`, runs before `dateparser` inside `_parse_natural_datetime`): dateparser 1.4 handles maybe 30% of AR-idiom inputs correctly and silently echoes the anchor time for another 30% (e.g. "a las 10 de la mañana" → anchor time). We hand-roll regex rewrites that normalize to forms dateparser CAN parse — mostly English equivalents with `PREFER_DATES_FROM=future`. Covers: `18hs` → `18:00`; `al mediodía` → `12:00`; `X que viene` → bare weekday/`next week`/`next month`; `el|este|próximo <weekday>` → bare English weekday (because dateparser 1.4 rejects `next <weekday>` silently but accepts bare `thursday` with future-prefer); `pasado mañana` → `day after tomorrow`; `a las N de la mañana|tarde|noche` → `N:00 am`/`(N+12):00`; `a la mañana|tarde|noche|tardecita` → default hour (09/16/20/17); `tipo N` / `a eso de las N` → `N:00` (rioplatense approximations); diminutives (`horitas` → `horas`); `el finde` → `saturday`. Anchor-echo guard after dateparser: if the input carries a time marker but dateparser returned exactly the anchor time, discard and fall through to LLM. LLM fallback prompt (qwen2.5:3b, `HELPER_OPTIONS` deterministic) explicitly flags rioplatense, passes both raw text and normalized hint, and instructs rollforward for bare weekdays + 09:00 default for missing times.

**Ambient agent**: hook in `_index_single_file` on saves within `allowed_folders` (default `["00-Inbox"]`). Config: `~/.local/share/obsidian-rag/ambient.json` (`{jid, enabled, allowed_folders?}`). Skip rules: outside allowed_folders, no config, frontmatter `ambient: skip`, `type: morning-brief|weekly-digest|prep`, dedup 5min (upsert on `rag_ambient_state.path`). Sends via `whatsapp-bridge` POST (`http://localhost:8080/api/send`). Bridge down = message lost but analysis persists in `rag_ambient`.

**Contradiction radar**: Phase 1 (query-time `--counter`), Phase 2 (index-time frontmatter `contradicts:` + `rag_contradictions`), Phase 3 (`rag digest` weekly). Skipped on `--reset` (O(n²)) and `note_body < 200 chars`.

**URL sub-index**: `obsidian_urls_v1` collection embeds **prose context** (±240 chars) not URL strings. `PER_FILE_CAP=2`. Auto-backfill on first `find_urls()` if collection empty.

**Wikilinks**: regex scan against `title_to_paths`. Skips: frontmatter, code, existing links, ambiguous titles, short titles (min-len 4), self-links. First occurrence only. Apply iterates high→low offset.

**Archive**: reuses `find_dead_notes`, maps to `04-Archive/<original-path>` (PARA mirror), stamps frontmatter `archived_at/archived_from/archived_reason`. Opt-out: `archive: never` or `type: moc|index|permanent`. Gate: >20 candidates without `--force` → dry-run. Batch log in `filing_batches/archive-*.jsonl`.

**Morning**: collects 36h window (modified notes, inbox, todos, contradictions, low-conf queries, Apple Reminders, calendar, weather, screentime). Weather hint only if rain ≥70%. Dedup vault-todos vs reminders (Jaccard ≥0.6). System-activity + Screen Time sections are deterministic (no LLM).

**Screen Time**: `_collect_screentime(start, end)` reads `~/Library/Application Support/Knowledge/knowledgeC.db` (`/app/usage` stream, read-only via `immutable=1` URI). Sessions <5s filtered. Bundle→label map + category rollup (code/notas/comms/browser/media/otros). Renders only if ≥5min of activity. Section omitted silently if db missing. Dashboard `/api/dashboard` exposes 7d aggregate + daily series (capped at 7 — CoreDuet aggregates older data away).

**Today**: `[00:00, now)` window, 4 fixed sections, writes `-evening.md` suffix. Feeds next morning organically.

**Followup**: extracts loops (frontmatter todo/due, unchecked `- [ ]`, imperative regex), classifies via qwen2.5:3b judge (temp=0, seed=42, conservative). One embed + one LLM call per loop.

**Read**: fetch URL → readability strip → gate (< 500 chars = error) → command-r summary → two-pass related lookup → tags from existing vocab (never invents) → `00-Inbox/`. Dry-run default, `--save` to write.

**Ranker-vivo (closed-loop ranker)**: implicit feedback from daily use re-tunes `ranker.json` nightly without manual intervention. Four signal sources insert into `rag_behavior`: (1) CLI `rag open` wrapper (opt-in via `RAG_TRACK_OPENS=1` + user-registered `x-rag-open://` handler); (2) WhatsApp listener classifying follow-up turns (`/save`, quoted reply → positive; "no"/"la otra"/rephrase → negative; 120s silence → weak positive); (3) web `/api/behavior` POST from home dashboard `sendBeacon` clicks; (4) morning/today brief diff (`_diff_brief_signal` compares yesterday's written brief vs current on-disk — wikilinks that survived = `kept`, missing = `deleted`, dedup via `rag_brief_state`). Nightly `com.fer.obsidian-rag-online-tune` at 03:30 runs `rag tune --online --days 14 --apply --yes`, which calls `_behavior_augmented_cases` (weight=0.5, drops conflicts), backs up current `ranker.json` → `ranker.{ts}.json` (keeps 3 newest), re-tunes, runs the bootstrap-CI gate (`_run_eval_gate`: scrubs `RAG_EXPLORE`, subprocess `rag eval`, 10min cap, regex parses hit@5). If singles < 76.19% OR chains < 63.64% (lower CI bounds of the 2026-04-17 expanded floor) → auto-rollback + exit 1 + log to `rag_tune`. `rag tune --rollback` restores the most recent backup manually.

## Eval baselines

**Floor (2026-04-17, post-golden-expansion + bootstrap CI)** — queries.yaml doubled (21→42 singles, 9→12 chains; +15 singles in under-represented folders 03-Resources/Agile+Tech, 02-Areas/Personal, 01-Projects/obsidian-rag, 04-Archive memory). `rag eval` now reports percentile bootstrap 95% CI (1000 resamples, seed=42) alongside each metric + `rag eval --latency` reports P50/P95/P99 of retrieve() per bucket and accepts `--max-p95-ms` as a CI gate.
- Singles: `hit@5 88.10% [76.19, 97.62] · MRR 0.772 [0.651, 0.873] · n=42`
- Chains: `hit@5 78.79% [63.64, 90.91] · MRR 0.629 [0.490, 0.768] · chain_success 50.00% [25.00, 75.00] · turns=33 chains=12`
- Latency: singles p95 2447ms · chains p95 3003ms

Every post-expansion metric sits inside the prior floor's CI on the smaller set — expansion surfaced the noise band (~21pp singles hit, ~50pp chain_success) that previously masqueraded as drift.

**Post prompt-per-intent + citation-repair (2026-04-19):** Singles `hit@5 88.10% [76.19, 97.62] · MRR 0.767 [0.643, 0.869]` — identical hit@5, MRR within CI. Chains `hit@5 81.82% [66.67, 93.94] · MRR 0.636 [0.505, 0.773] · chain_success 58.33% [33.33, 83.33]` — +3pp hit@5, +8pp chain_success, both inside prior CI so treat as noise until replicated. Floor unchanged for auto-rollback gate (still 76.19% / 63.64%).

**Post golden-set re-mapping (2026-04-20):** vault reorg (PARA moves: many notes `02-Areas/Coaching/*` → `03-Resources/Coaching/*`, `03-Resources/{Agile,Tech}/*` → `04-Archive/*`, etc.) left 33 of 65 `expected` paths in `queries.yaml` pointing at dead files, artificially cratering eval to singles hit@5 26% / chains 33%. Golden rebuilt by auto-mapping 31 unique paths via filename-stem lookup to the closest surviving note (prefer non-archive, bias `01→02→03→04` for tie-breaks) and dropping one chain whose source notes (`reference_{claude,ollama}_telegram_bot.md`) no longer exist. Post-rebuild eval: Singles `hit@5 78.57% [64.29, 90.48] · MRR 0.696 [0.554, 0.810]`; Chains `hit@5 75.76% [60.61, 90.91] · MRR 0.641 [0.510, 0.788]`. Both CIs overlap the 2026-04-19 run — within noise band. Floor unchanged (76.19% / 63.64%); current singles 78.57% and chains 75.76% pass the auto-rollback gate.

**Post-T10 (2026-04-20, after JSONL-fallback strip, commit `81e32b4`):** Singles `hit@5 78.57% [64.29, 90.48] · MRR 0.696 [0.554, 0.810] · recall@5 76.19% · n=42`; Chains `hit@5 86.67% [73.33, 96.67] · MRR 0.728 [0.594, 0.850] · chain_success 63.64% [36.36, 90.91] · turns=30 chains=11`. Singles **bit-identical** vs pre-T10 (expected — T10 is pure storage refactor, retrieval pipeline untouched); chains drifted +11pp inside prior CI (same noise band). Latency: singles p95 2797ms, chains p95 3406ms — slight uptick vs pre-T10 (2447/3003ms) attributable to SQL being the only write path (no JSONL-queue offload anymore). Still ×5 below any action threshold. Floor gate passed at the exact chain_success boundary (63.64%) — fine this run but worth re-measuring next tune cycle.

**Post cross-source corpus (2026-04-21, n=55 singles / 11 chains):** Primer eval con el corpus mixto activo — 20 chunks gmail + 36 chunks reminders + 4071 chunks whatsapp + vault (ingesters Phase 1.a-1.d corridos por primera vez en prod). `queries.yaml` extendido con 6 queries synthesis/comparison (Fase 2) + 7 queries cross-source (Fase 1.f — 4 reminders + 3 gmail). Singles `hit@5 80.00% [69.09, 90.91] · MRR 0.714 [0.609, 0.818]`; Chains `hit@5 83.33% [70.00, 96.67] · MRR 0.706 [0.567, 0.833] · chain_success 54.55% [27.27, 81.82]`. **Todos los metrics overlapean el CI del baseline anterior** — singles +1.4pp noise, chains −3.3pp noise, chain_success −9pp dentro del ±CI. Auto-rollback gate pasa por doble margen (floor 76.19% / 63.64% vs actual 80.00% / 83.33%). 6 de las 7 queries cross-source hitearon — la que falla ("resumen bancario BICA enero 2026") no es un issue de threshold sino retrieval específico del corpus (el thread Gmail está más oculto de lo que el query esperaba). **Decisión Phase 1.f tuning**: `CONFIDENCE_RERANK_MIN_PER_SOURCE` queda todo en el global 0.015 — no hay regresión medible que justifique bajarlo per-source. Si aparecen false-refuse cross-source en logs de producción, re-evaluar.

**Post Calendar API enable (2026-04-21 evening, n=60 singles / 11 chains):** Calendar API del proyecto GCP `701505738025` activada por el user → `rag index --source calendar --reset` ingestó 368 eventos (2y history + 180d future del calendar `fernandoferrari@gmail.com`). Corpus: 8231 chunks (calendar 4.5% · WA 49.5% · vault 45.3% · reminders 0.4% · gmail 0.2%). `queries.yaml` +5 queries calendar (turno psicólogo / workshop AI Engineer / despedida de jardín de astor / reunión con Max ops / turno erica franzen). Singles `hit@5 81.67% [71.67, 91.67] · MRR 0.735 [0.639, 0.831] · recall@5 76.94% · n=60`; Chains sin cambio (no se agregaron chains calendar): `hit@5 83.33% · MRR 0.706 · chain_success 54.55%`. **5/5 queries calendar hit con MRR promedio 0.87** — confirmación directa de que el pipeline existente resuelve events.list()-ingested calendar sin tocar scoring/threshold. Smoke test previo sobre query real log: "qué hago el viernes 20hs" pasó de score 0.04 (refuse) → 0.46 (respuesta útil) post-ingesta, validando el hallazgo de producción. **Bug arquitectónico residual detectado**: `classify_intent` devuelve `recent` para "qué tengo esta semana" → itera notas del vault en vez de filtrar `source=calendar` con ventana temporal. Antes era inerte porque calendar estaba vacío; ahora merece ticket como Phase 1.g (intent-source routing) — **addressed later in the same session, see the next entry**. Auto-rollback gate sigue pasando holgado. **Calendar path format drift fix**: `tests/test_eval_bootstrap.py::test_golden_cross_source_paths_have_native_id_format` tenía regex `calendar://(event:)?<id>` basado en design doc §2.7, pero la implementación real del ingester (`scripts/ingest_calendar.py::_event_file_key`) usa `calendar://<calendar_id>/<event_id>` (two-segment, paralelo a WhatsApp). Test updated para seguir implementación como ground truth.

**Post session close (2026-04-21 evening, same n=60 singles / 11 chains):** Final eval tras el `agenda` intent (v1 + v2 window filter), cold-embed non-blocking fix (`_local_embedder_ready` Event), + feedback harvest de 8 negativos adicionales al `rag_feedback` SQL (ratio pos:neg pasó de 55:2 → 55:10). Singles `hit@5 81.67% [71.67, 91.67] · MRR 0.779 [0.683, 0.875] · recall@5 76.94%`; Chains `hit@5 80.00% [66.67, 93.33] · MRR 0.733 [0.583, 0.867] · chain_success 54.55% [27.27, 81.82]`. **Singles MRR +0.044 vs el baseline de la mañana** (dentro del CI anterior pero consistent uplift — probablemente absorción de los 8 negatives al golden cache). Chains MRR +0.027 idem. Hit@5 bit-identical en singles, −3.3pp en chains (dentro del CI, noise). Floor gate pasa por doble margen. Calendar queries siguen 5/5 hit (idéntico al baseline anterior — el fix del intent router no las rompió, solo redirigió queries browsing que antes caían en `recent`). Auto-rollback gate holgado.

**Fase D closeout — 3 mejoras validadas (2026-04-21, n=60 singles / 11 chains):** Dos runs de `rag eval` con flags todas-OFF y todas-ON (RAG_ADAPTIVE_ROUTING=1 RAG_ENTITY_LOOKUP=1 RAG_NLI_GROUNDING=1). Resultados **bit-idénticos** en ambas corridas — las 3 mejoras no introducen regresión retrieval medible. **Flags-OFF**: Singles `hit@5 71.67% [60.00%, 83.33%] · MRR 0.681 [0.567, 0.794]`; Chains `hit@5 86.67% [73.33%, 96.67%] · MRR 0.807 [0.680, 0.923] · chain_success 72.73% [45.45%, 100.00%]`. **Flags-ON**: Singles `hit@5 71.67% [60.00%, 83.33%] · MRR 0.681 [0.567, 0.794]`; Chains `hit@5 86.67% [73.33%, 96.67%] · MRR 0.790 [0.657, 0.917] · chain_success 72.73% [45.45%, 100.00%]`. **Nota de variabilidad**: singles hit@5 bajó de 81.67% (baseline previo) a 71.67% en esta sesión — CIs solapan [71.67%, 83.33%] — causa probable: LLM non-determinism (qwen2.5:7b stochastic) + posible drift de paths en queries.yaml. La caída es pre-existente, no causada por los flags (flags-OFF muestra igual caída). **Decisión de flags**: (a) `RAG_ADAPTIVE_ROUTING` → **stays OFF por default** — criterios de floor (lower CI ≥ 76.19%) no se cumplen en esta sesión eval (lower CI 60.00% < 76.19%); aunque la caída es pre-existente y no causada por el flag, las instrucciones son conservadoras. Flipear a ON cuando `rag eval` vuelva a superar el floor de forma estable. (b) `RAG_ENTITY_LOOKUP` → **flipped a ON tras backfill** (same-day evening commit) — `scripts/backfill_entities.py` corrió sobre el corpus (2022 entities / 6520 mentions / 71% chunk coverage), smoke-test directo validó `rag query "todo lo que tengo de Astor"` retornando síntesis cross-source agregada. Ver §Env vars para detalle. (c) `RAG_NLI_GROUNDING` → **stays OFF** — requiere validación de latencia P95 post-50 queries reales.

**Prior floor (2026-04-17, post-title-in-rerank, n=21 singles / 9 chains):** Singles `hit@5 90.48% · MRR 0.821`; Chains `hit@5 80.00% · MRR 0.627 · chain_success 55.56%`. Kept for historical trend, but do not compare new numbers against it without overlapping CIs.

**Even-earlier floor (2026-04-16, post-quick-wins, n=21/9):** Singles `hit@5 90.48% · MRR 0.786`; Chains `hit@5 76.00% · MRR 0.580 · chain_success 55.56%`.

The 2026-04-15 floor (`95.24/0.802` singles, `72.00/0.557/44.44` chains, see `docs/eval-tune-2026-04-15.md`) pre-dates both the expansion and the CI tooling — treat as a qualitative reference only.

Never claim improvement without re-running `rag eval`. Helper LLM calls (`expand_queries`, `reformulate_query`, `_judge_sufficiency`) are already deterministic via `HELPER_OPTIONS = {temperature: 0, seed: 42}`.

**HyDE with qwen2.5:3b drops singles hit@5 ~5pp**. HyDE is opt-in (`--hyde`); re-measure if helper model changes.

**`seen_titles` as post-rerank penalty** (2026-04-20, `SEEN_TITLE_PENALTY = 0.1` in `retrieve()`). The LLM-instruction path regressed chains (chains hit@5 −16pp, chain_success −33pp — helper treats the list as "avoid these" and drifts off-topic); the post-rerank soft penalty is the shipped replacement. Candidates whose `meta.note` (case-insensitive) matches any `seen_titles` entry get their final score docked by 0.1 — a diversity nudge, not a filter (strong rerank leads still win). In `reformulate_query` the kwarg remains on the signature but is intentionally unused in the prompt (dead per design). Tests in `tests/test_seen_titles_penalty.py`. Empirical lift on queries.yaml chains hit@5 83.33% → 90.00% (both inside CI — re-measure on next tune cycle before claiming stable gain).

## Cross-source corpus (Phase 1, in progress — 2026-04-20 decisions)

The corpus is no longer vault-only. Per `docs/design-cross-source-corpus.md` + §10 user decisions, `retrieve()` is now source-aware and the sqlite-vec collection holds chunks from multiple sources via a `source` metadata discriminator. Collection stays at `obsidian_notes_v11` (no rename / no re-embed) — legacy vault rows without `source` are read as `"vault"` via `normalize_source()`.

**Constants** (`rag.py:~1288`): `VALID_SOURCES` (frozenset of 6), `SOURCE_WEIGHTS` (vault 1.00 → WA 0.75), `SOURCE_RECENCY_HALFLIFE_DAYS` (None for vault/calendar, 30d for WA/messages, 90d for reminders, 180d for gmail), `SOURCE_RETENTION_DAYS` (None for vault/calendar/reminders, 180 for WA/messages, 365 for gmail).

**Helpers**: `normalize_source(v, default="vault")` → safe legacy-row read; `source_weight(src)` → lookup + 0.50 fallback; `source_recency_multiplier(src, created_ts, now)` → exponential decay `2**-(age/halflife)` in [0,1], accepts epoch float or ISO-8601 string (Zulu Z), clamps future-ts at 1.0, None-halflife short-circuits to 1.0.

**Scoring** (inside `retrieve()` post-rerank loop + in `apply_weighted_scores()` for eval parity): after the existing scoring formula produces `final`, multiply by `source_weight(src) * source_recency_multiplier(src, created_ts)`. Vault default → `1.0 * 1.0` = no-op. Old vault data completely untouched.

**Filter** (retrieve/deep_retrieve/multi_retrieve `source` kwarg + `rag query --source S[,S2]`): string or iterable of strings; restricts candidate pool post-rerank. Unknown sources from the CLI are rejected upfront with a helpful error. Legacy vault path: `source=None` or `source="vault"` → identical to pre-Phase-1 behavior.

**Conversational dedup** (`_conv_dedup_window`, applied post-scoring pre top-k slice): collapses WhatsApp/messages chunks from the same `chat_jid` within a ±30min window — keeps only the highest-scored. Non-WA sources pass through unchanged. Intentionally simple O(n²) — pool is capped at `RERANK_POOL_MAX`, constant factor negligible.

### WhatsApp ingester — Phase 1.a (`scripts/ingest_whatsapp.py`, `rag index --source whatsapp`)

Reads from `~/repositories/whatsapp-mcp/whatsapp-bridge/store/messages.db` in read-only immutable mode. Filters empty content, `status@broadcast` pseudo-chat, and anything older than 180d. Timestamps (Go RFC3339 with nanoseconds / Z suffix / numeric) parsed defensively. Conversational chunking (§2.6 option A): groups same-sender contiguous messages within 5min windows; splits on speaker change OR >=5min gap OR >800 chars; merges undersized groups (<150 chars) into temporally-nearest neighbor in the same chat. Parent window ±10 messages, 1200 char cap. Embed prefix `[source=whatsapp | chat=X | from=Y] {body}`; display text stays raw. doc_ids are `whatsapp://{chat_jid}/{first_msg_id}::{idx}` — stable across bridge DB compactions. Idempotent upsert (delete prior by `file` key + add). Incremental cursor in `rag_whatsapp_state(chat_jid, last_ts, last_msg_id)`; `--reset` wipes, `--since ISO` overrides uniformly. CLI flags: `--bridge-db`, `--since`, `--reset`, `--max-chats`, `--max-messages`, `--dry-run`, `--json`.

### Calendar ingester — Phase 1.b (`scripts/ingest_calendar.py`, `rag index --source calendar`)

Google Calendar via OAuth (§10.6 user override — rompe local-first). Creds under `~/.calendar-mcp/{gcp-oauth.keys.json, credentials.json}`. Window `[now − 2y, now + 180d]` on bootstrap, `syncToken` for incremental. `singleEvents=True` (expands RRULE instances). Chunk-per-event, `parent=body`, body cap 800 chars. Cancelled events → delete. State in `rag_calendar_state(calendar_id, sync_token, last_updated, updated_at)`. Hardcoded exclude list filters `addressbook#contacts` + `en.usa#holiday` noise calendars.

### Gmail ingester — Phase 1.c (`scripts/ingest_gmail.py`, `rag index --source gmail`)

Gmail via OAuth (same cred dir convention, `~/.gmail-mcp/`). Thread-level chunking (§2.6 — one chunk per thread, not per message — empirically matches user "cuándo hablamos de X" granularity better than message-level). Quoted replies + signatures stripped via regex before chunking. `parent = subject + first 1200 chars of thread`. Incremental via Gmail's `historyId` cursor in `rag_gmail_state(history_id, updated_at)`. Bootstrap uses `q=newer_than:365d` per §10.2 retention. Deleted threads removed from index on incremental pass.

### Reminders ingester — Phase 1.d (`scripts/ingest_reminders.py`, `rag index --source reminders`)

Apple Reminders via AppleScript (local, same trust boundary as the morning brief's `_fetch_reminders_due`). Pulls every reminder (pending + completed) with id, list, due/completion/creation/modification dates, name, body, priority, flagged state. Chunk-per-reminder, body cap 800 chars. `created_ts` anchor preference: creation → due → modified → completion. Content-hash diffing in `rag_reminders_state(reminder_id, content_hash, last_seen_ts, updated_at)` — on each run, re-fetch the full catalogue, upsert changed/new, delete stale (id no longer present). No cursor / modification-date polling — Reminders.app's `modification date` is unreliable via AppleScript. Field separator in the AS → Python pipe is chr(31) (Unit Separator) to avoid collisions with body content. CLI flags: `--reset`, `--dry-run`, `--only-pending` (default indexes both). Retention None (§10.2); source weight 0.90, recency halflife 90d (§10.3).

### Remaining (Phase 1.e + 1.f + 2)

- **Phase 1.e — apagar workaround** (gated on 1.c stable in prod ≥1 week): deprecar `/note` + `/ob` del whatsapp-listener ahora que el corpus captura WA por barrido. ~100 LOC, mostly external repo.
- **Phase 2 — OCR pipeline para adjuntos** (deferred 2026-04-21, no shipped): el design doc §8 flagea "no indexa adjuntos binarios (imágenes WA, PDFs en Gmail). Eso es fase 2". Evaluado en la tanda 2026-04-21 y **skipped** porque el sistema actual tiene 16 imágenes (todas en `04-Archive/99-obsidian-system/99-Attachments/`, screenshots archived) + **0 PDFs** en el vault, y el corpus cross-source (donde estarían los adjuntos reales de Gmail/WA) tiene 0 chunks — los ingesters Phase 1.a-1.d nunca corrieron en prod. Sin data activa, implementar OCR sería scaffolding + agregar dep (`pyobjc-framework-Vision` ~20 MB, o tesseract via brew) sin beneficio medible actual. **Trigger de activación**: cuando los ingesters hayan corrido ≥1 semana y haya ≥20 adjuntos referenciados en el corpus cross-source, implementar usando Apple Vision (`VNRecognizeTextRequest`, local) para imágenes + `pdftotext` (poppler) para PDFs con fallback Vision para scans. Chunk OCR como prose con metadata `attachment_of: <parent doc_id>` + `media_type: "ocr"`.
- **Phase 1.f — re-calibración eval** *(infra shipped 2026-04-21, tuning pending real data)*:
  - **Infra shipped**: `CONFIDENCE_RERANK_MIN_PER_SOURCE` dict en `rag.py` (scaffolding — todos los valores = baseline 0.015 hoy) + helper `confidence_threshold_for_source(source)` con fallback al global. Invocado en `query()` y `rag serve` sobre `source` del top-result meta. Tests: [`tests/test_confidence_threshold_per_source.py`](tests/test_confidence_threshold_per_source.py) (9 casos). El test `tests/test_eval_bootstrap.py::test_queries_yaml_all_paths_exist_or_placeholder` ahora acepta paths con prefijos `gmail://` / `whatsapp://` / `calendar://` / `reminders://` / `messages://` como placeholders válidos; sanity-test aparte `test_queries_yaml_cross_source_prefixes_cover_all_valid_sources` detecta drift contra `VALID_SOURCES`. Template de queries cross-source está comentado en [`queries.yaml`](queries.yaml) listo para un-commentar cuando los ingesters populen el corpus.
  - **Tuning pending**: re-correr `rag eval` + bajar per-source thresholds empíricamente (expected: WA 0.008-0.010, Calendar 0.012, Gmail 0.010-0.012, Reminders 0.012) una vez que los ingesters hayan corrido ≥1 semana + haya feedback data. Validar los `SOURCE_WEIGHTS` hardcoded (vault 1.00 / calendar 0.95 / reminders 0.90 / gmail 0.85 / WA 0.75 / messages 0.75) contra queries reales. Deferred per §10.8.

## On-disk state (`~/.local/share/obsidian-rag/`)

### Telemetry — SQL tables (post-T10 2026-04-19)

All operational telemetry + learning state lives in `ragvec/ragvec.db` alongside the sqlite-vec meta/vec tables. 20 `rag_*` prefixed tables (no collision with `meta_*`/`vec_*`). SQL is now the only storage path — T10 (2026-04-19, same day as cutover) stripped the JSONL writers + readers so there is no 7-day observation window. `RAG_STATE_SQL=1` is still set on every launchd plist for trail/future-proofing, but it is now a no-op toggle: the flag is not consulted anywhere in the code (neither writers nor readers branch on it). Leaving it set costs nothing and keeps the deployment config ready for rollback.

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
- `rag_feedback_golden` (pk=path,rating, `embedding BLOB` float32 little-endian, `source_ts`) + `rag_feedback_golden_meta` (k/v) — cache rebuilt when `rag_feedback.max(ts) > meta.last_built_source_ts`. `record_feedback` clears both tables synchronously so the next `load_feedback_golden()` call always rebuilds (sidesteps a same-second MAX(ts) collision that could leave a stale cache).

Primitives in `rag.py` (`# ── SQL state store (T1: foundation) ──` section):
- `_ensure_telemetry_tables(conn)` — idempotent DDL
- `_ragvec_state_conn()` — short-lived WAL conn with `synchronous=NORMAL` + `busy_timeout=10000`
- `_sql_append_event(conn, table, row)`, `_sql_upsert(conn, table, row, pk_cols)`, `_sql_query_window(conn, table, since_ts, ...)`, `_sql_max_ts(conn, table)`

Writer contract (post-T10): single-row BEGIN/COMMIT into SQL. On exception, log the error to `sql_state_errors.jsonl` and **silently drop the event** — no JSONL fallback. Callers never see a raised exception. Reader contract: SQL-only. Readers return empty snapshots (behavior priors, feedback golden, behavior-augmented cases, contradictions) or False/None (brief_state, ambient_state lookups) on SQL error; retrieval pipeline stays functional without priors until the DB is readable again.

**Drift fixes (2026-04-21 evening)** — four CLI readers still tail-read JSONL files that post-T10 either no longer receive the expected events or got repurposed for another log stream. All migrated to SQL:

| Reader | Pre-fix behaviour | Fix |
|---|---|---|
| `rag log` | Rendered all columns empty — `queries.jsonl` now receives `conversation_turn_written` observability events with a different schema | `_read_queries_for_log(n, *, low_confidence)` + `_read_feedback_map_for_log()` in `rag.py` — SELECT from `rag_queries` + `rag_feedback`, hoist `turn_id` from `extra_json`, filter admin rows with empty `q`. Renderer null-safety on `t_retrieve` / `t_gen` / `ts` / `mode` when SQL rows return `None` (metachat / create-intent turns). Tests: `tests/test_rag_log_sql_read.py` (13 cases). |
| `rag emergent` + `rag dashboard` | `_scan_queries_log(days)` read `LOG_PATH` → empty events list → "sin queries en ventana", dashboard showed `n=0` | Same `_scan_queries_log(days)` signature, SELECT from `rag_queries WHERE ts >= ?` (chronological ASC), hoists **all** `extra_json` keys to top-level (`q_reformulated`, `answered`, `gated_low_confidence`, `turn_id`) so callers don't re-parse JSON. Admin rows (`q=""`) excluded. Tests: `tests/test_post_t10_sql_readers.py` (5 cases for this reader). |
| `feedback_counts()` / `rag insights` | `FEEDBACK_PATH.is_file()` → False (only `.bak` remains post-cutover) → returned `(0, 0)` silently; `rag insights` showed vacío | Single SQL `SELECT SUM(CASE WHEN rating > 0 ...)` aggregate. Tests: 3 cases. |
| `_feedback_augmented_cases()` / `rag tune` | `FEEDBACK_PATH` empty → returned `[]` → **rag tune silently lost every corrective-path signal** the user ever gave. This was the highest-impact drift — tune couldn't learn from "this path was actually correct" feedback. | SQL with `json_extract(extra_json, '$.corrective_path')` since `corrective_path` isn't a first-class column. Filter `scope != 'session'` + `len(q) >= min_len` + dedup by normalised query. Tests: 5 cases. |
| `rag patterns` | Same shape as feedback_counts — read empty `feedback.jsonl`, always "sin feedback log todavía" | Inline SQL query with `json_extract(extra_json, '$.reason')` for the `reason` field (also inside `extra_json`). |

Also: deleted dead code `_iter_behavior_jsonl()` (defined post-T10 but never called).

Pattern for new SQL readers: use `_ragvec_state_conn()` context manager; wrap the SELECT in `try/except` + `_silent_log` + return empty value matching the old JSONL-reader shape. Never raise from a reader — the calling CLI command keeps working with degraded signal.

Migration one-shot: `scripts/migrate_state_to_sqlite.py --source-dir ~/.local/share/obsidian-rag [--dry-run] [--round-trip-check] [--reverse] [--summary]`. Refuses to run while `com.fer.obsidian-rag-*` services are up (preflight `pgrep`; `--force` to override). Renames each source → `<name>.bak.<unix_ts>` on successful commit. Cutover of 2026-04-19 imported 7,946 records across 19 sources; 43 malformed pre-existing records dropped (missing NOT NULL fields).

Rollback procedure (post-T10): **the escape hatch now requires a code revert, not just a CLI invocation.** `rag maintenance --rollback-state-migration [--force]` still restores the newest `.bak.<ts>` per source and drops the 20 `rag_*` tables + VACUUM — but the in-code readers/writers only know the SQL path after T10. To fully revert:

1. `git revert <T10-commit-sha>` (or `git reset --hard <pre-T10-sha>` if the T10 commits are the tip). This brings back the JSONL fallback code.
2. Restart launchd services so the reverted `rag.py` is loaded in-process.
3. Run `rag maintenance --rollback-state-migration` — this restores the JSONL .bak files that the reverted code now reads.

The `.bak.<ts>` files under `~/.local/share/obsidian-rag/` are still there (kept for the 30-day window) so data-loss is bounded, but without the code revert the restored files are ignored. `rag maintenance` continues to prune `.bak.*` older than 30d.

### Other state (unchanged; still on disk)

- `ranker.json` — tuned weights. Delete = reset to hardcoded defaults.
- `ranker.{unix_ts}.json` — 3 most recent backups, written on every `rag tune --apply`. Consumed by `rag tune --rollback` + auto-rollback CI gate.
- `sessions/*.json` + `last_session` — multi-turn state (TTL 30d, cap 50 turns).
- `ambient.json` — ambient agent config (jid, enabled, allowed_folders).
- `filing_batches/*.jsonl` — audit log (prefix `archive-*` for archiver).
- `ignored_notes.json`, `home_cache.json`, `context_summaries.json`, `auto_index_state.json`, `coach_state.json`, `synthetic_questions.json`, `wa_tasks_state.json` — app state + caches.
- `online-tune.{log,error.log}`, `*.{log,error.log}` — launchd service logs.
- `sql_state_errors.jsonl` — diagnostic sink for SQL-path write/read failures. Post-T10 this is the only visible signal when SQL errors happen, since the JSONL fallback is gone and the event is dropped after logging here.

**Reset learned state**: `rm ranker.json` + `DELETE FROM rag_feedback_golden; DELETE FROM rag_feedback_golden_meta;` inside ragvec.db. Full re-embed: `rag index --reset`.

## Vault path

Default: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes`. Override: `OBSIDIAN_RAG_VAULT` env var. Collections namespaced per vault (sha256[:8]).

Claude Code memory (`~/.claude/projects/-Users-fer/memory/`) is symlinked into vault at `04-Archive/99-obsidian-system/99-Claude/memory/`.
