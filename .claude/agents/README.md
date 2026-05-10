# Agents del repo obsidian-rag

Agents especializados por dominio. Claude Code los detecta automáticamente al bootear en este repo y pueden ser invocados vía `Agent(subagent_type: <name>, ...)`.

## Roster

| Agent | Slug | Responsabilidad |
|-------|------|-----------------|
| Project manager | `pm` | Planifica y rutea — descompone tareas grandes, asigna a specialists, detecta overlaps con peers. No edita código. |
| Generalista #1 | `developer-1` | Refactors cross-cutting, nuevos subcommands CLI, tests, `mcp_server.py`, `pyproject.toml`, launchd plists, bug fixes que no caen en un dominio. |
| Generalista #2 | `developer-2` | Idéntico a `developer-1`. Existe para que un segundo peer Claude reclame su propio slug sin colisión. |
| Generalista #3 | `developer-3` | Idéntico a `developer-1`/`developer-2`. Tercer slot paralelo. |
| Retrieval | `rag-retrieval` | `retrieve()`, HyDE on/off, rerank, BM25, corpus cache, graph expansion, deep retrieve, scoring formula, `ranker.json`, behavior priors, ranker-vivo nightly online-tune + rollback gate. |
| LLM expert | `rag-llm` | Toda prompt en `rag/__init__.py`, model resolution chain, `HELPER_OPTIONS`/`CHAT_OPTIONS`, JSON schema + parsers, citation verifier, contextual summary cache, HyDE prompt body, `rag do` agent loop, STT (whisper-cli) + TTS (`say` Mónica) contracts, MLX backend en `rag/llm_backend.py` + `rag/mlx_tool_calls.py`. Consultivo cross-domain — coordina con quien dispara cada prompt. |
| Briefs | `rag-brief-curator` | `rag morning` / `rag today` / `rag digest` / `rag pendientes`, evidence rendering, secciones determinísticas (Agenda/Gmail/System/Screen Time/Drive), brief diff signal (kept/deleted → behavior.jsonl), WhatsApp push. |
| Ingestion | `rag-ingestion` | `rag read` (incl. YouTube), `rag capture`, `rag inbox`, `rag prep`, wikilinks densifier, `rag links` semantic URL finder. |
| Vault health | `rag-vault-health` | `rag archive`, `rag dead`, `rag followup`, `rag dupes`, contradiction radar (Phase 1+2+3), `rag maintenance` (incl. orphan HNSW segment cleanup, WAL checkpoint, log + behavior rotation). |
| Integraciones | `rag-integrations` | Todos los `_fetch_*` (Apple Mail/Reminders/Calendar, Gmail API, WhatsApp bridge SQLite + listener, weather, Drive activity, screen time `knowledgeC.db`), ambient agent, `wa-tasks` extractor. |
| Eval harness | `rag-eval` | `rag eval`, `rag tune` (offline sweep), `queries.yaml` golden set, `feedback_golden.json` labelling, `behavior.jsonl` curation as eval input, bootstrap CI methodology, baselines floor, latency gate (`--max-p95-ms`). Owner de `tests/test_eval*.py`. |
| Infra / launchd | `rag-infra` | Plists en `~/Library/LaunchAgents/com.fer.obsidian-rag-*.plist`, Caddy + `tls internal` para `ra.ai`, Cloudflare Quick Tunnel, Devin permissions (`.devin/config.json`, `~/.config/devin/config.json`), `pyproject.toml` entry points + `uv tool install --reinstall --editable .`, `launchctl bootstrap`/`bootout`/`kickstart`/`print`. |
| Perf auditor | `rag-perf-auditor` | Read-only auditor de hot paths en `rag/__init__.py` + `web/server.py`: N+1 sobre sqlite-vec, locking redundante en WAL, blocking I/O en handlers async, sentence-transformers sin batch, LRU caches missing/oversized, `fetchall()` en tablas grandes. NO edita — devuelve reporte priorizado. |
| Doc curator | `rag-doc-curator` | Detecta drift entre `CLAUDE.md` / `AGENTS.md` / `README.md` y el código real (`rag/`, `web/server.py`). Read-only — devuelve diff estructurado (commands no expuestos, surface no documentado, invariantes contradichos). |
| Telemetry | `rag-telemetry` | SQL state telemetry (`rag_queries` / `rag_behavior` / `rag_feedback` / `system_memory_metrics` y demás tablas de `telemetry.db`), DDL ensure-once, `corpus_hash` bucketing, query layer del `/dashboard`, rotation lifecycle (SQL + `behavior.jsonl`). |
| Web | `rag-web` | `web/server.py` (FastAPI: chat, dashboard, SSE, `/api/*`), static frontend (`web/static/*.{js,html,css}`), PWA wiring (manifest + service worker + iOS splash), LAN-exposure env vars (`OBSIDIAN_RAG_BIND_HOST`/`OBSIDIAN_RAG_ALLOW_LAN`), Cloudflare Quick Tunnel publishing. |
| Anticipatory | `rag-anticipatory` | `rag_anticipate/signals/`, threshold tuning, feedback tuning, kind weights, quiet hours, lockfile. Proactive push system "talks to you first". |
| Cross-source ETL | `rag-cross-source-etl` | Coordinación ETL externos — MOZE, credit cards, WhatsApp, Gmail, Calendar, Chrome, YouTube, GitHub, Claude, Spotify, Drive. |
| Mood + wellness | `rag-mood-wellness` | `rag/mood.py`, `rag/integrations/pillow_sleep.py`, `rag/integrations/screentime.py`, cross-source correlations. |
| Negotiations | `rag-negotiations` | `rag_negotiations` package, state transitions, CRUD, WhatsApp follow-ups. |
| Error analyst | `rag-error-analyst` | Read-only forensics — race conditions, deadlocks, memory leaks, silent-fail patterns, lock ordering. |
| **MLX lifecycle** ⭐ | `rag-mlx-lifecycle` | **NUEVO 2026-05-10**: idle-unload watchdog, memory-pressure handler, `_MLX_FORWARD_LOCK` global, reranker pin/unpin, OOM debugging, command-buffer interactions. |
| **Entities + overrides** ⭐ | `rag-entities` | **NUEVO 2026-05-10**: `_extract_entities_*`, 3 JSON files canonical (`known_places.json`, `entity_overrides.json`, `known_places_extra.json`), atlas display layer. |
| **Test harness** ⭐ | `rag-test-harness` | **NUEVO 2026-05-10**: `tests/conftest.py` autouse fixtures, pytest markers, `_mlx_chat` auto-stub, mock isolation, xdist races, DB_PATH per-file isolation. |

Para tareas ambiciosas o cross-dominio, arrancar por `pm`: devuelve un plan de dispatch que el caller ejecuta.

## Por qué hay 3 slots de developer idénticos

Layout post-split 2026-05-04 (LOC re-medido audit 2026-05-10): `rag/` paquete (`__init__.py` **~52.8k LOC** core + sub-modules como `plists/` package, `cross_source_etls.py`, `cross_source_collectors.py`, `postprocess.py`, `archive.py`, `anticipatory.py`, `brief_schedule.py`, `voice_brief.py`, `whisper.py`, `wa_scheduled.py` (shim → `rag/integrations/whatsapp/scheduled.py`), `_memory_pressure_watchdog.py`, `mlx_embed.py`, etc.) + `mcp_server.py` thin wrapper + `web/` (`server.py` ~23.1k LOC) + `tests/` (8,103 tests, 453 archivos). Las escrituras siguen serializándose sobre `rag/__init__.py` 52.8k LOC (es el core que casi todos los specialists tocan), así que la presión de paralelismo es la misma que con el single-file viejo. Tres slots `developer-{1,2,3}` con cuerpo idéntico permiten:

1. **Claim por slot**: cada peer Claude reclama el slug libre más bajo (`developer-1`, después `-2`, después `-3`). Slugs distintos = no shadowing en el registry de subagents.
2. **Paralelismo real**: 3 generalistas pueden trabajar en sub-zonas distintas simultáneamente, coordinados por `mcp__claude-peers__set_summary`.
3. **Sin proliferación temática**: en lugar de inventar agents por feature (`developer-mcp`, `developer-tests`...) que rotan rápido, mantenemos un único contrato de generalista replicado.

Si necesitás más de 3 generalistas en paralelo, agregá `developer04.md` con `name: developer-4` (mismo cuerpo). Si la presión cede, podés borrar `developer-3` sin afectar a los otros.

## Por qué los demás existen

1. **Documentan ownership** — cada uno declara su zona y qué NO debe tocar (ver tabla en `developer*.md` y en cada `rag-*.md`).
2. **Enrutan tareas** — `Agent(subagent_type: rag-retrieval, ...)` para que la sesión principal delegue en contexto fresco sin pisar edits en curso.
3. **Preservan invariantes** — cada agent lista los contratos que no se pueden romper (eval baselines con CIs, silent-fail, layout del brief, schema version, MLX defaults, etc.).

## Protocolo cuando hay peers activos

Si `mcp__claude-peers__list_peers(scope: "repo")` devuelve >1 instancia Claude Code en el repo:

1. **Announce**: `mcp__claude-peers__set_summary` declarando dominio + función (ej. `"developer-1: editando _wa_extract_actions en rag/__init__.py:18420"`).
2. **Check peers**: `list_peers(scope: "repo")` antes de cada edit grande en `rag/__init__.py`.
3. **Rebase mental**: si otro peer tiene summary overlapping con tu zona, **pausar y coordinar** vía `send_message` antes de escribir.
4. **Commit frecuente**: commits chicos por feature → git es el source of truth ante conflictos.
5. **Trabajo paralelo ambicioso**: usar `EnterWorktree` antes de editar la misma rama en simultáneo.

## Cuándo NO usar un agent

- Tarea trivial de 1 archivo fuera de `rag/__init__.py` (ej. editar un test): trabajar directo.
- Pregunta conversacional: respondé vos.
- Tareas cross-dominio (ej. "agregá Gmail al morning brief"): arrancar por `pm` para que rutee — típicamente termina en `rag-brief-curator` (que decide layout) coordinando con `rag-integrations` (que aporta el fetcher).

## Cuándo agregar un nuevo agent de dominio

Si aparece una zona nueva que se edita frecuentemente (ej. dashboard web en `web/`, eval harness en `queries.yaml`), crear un agent dedicado con zona clara y "don't touch" explícito. No crear agents por feature efímera — el costo de mantenerlos sincronizados con el repo CLAUDE.md no se paga si la zona se va a tocar dos veces y nunca más.
