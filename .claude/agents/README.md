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
| LLM expert | `rag-llm` | Toda prompt en `rag.py`, model resolution chain, `HELPER_OPTIONS`/`CHAT_OPTIONS`, JSON schema + parsers, citation verifier, contextual summary cache, HyDE prompt body, `rag do` agent loop, STT (whisper-cli) + TTS (`say` Mónica) contracts. Consultivo cross-domain — coordina con quien dispara cada prompt. |
| Briefs | `rag-brief-curator` | `rag morning` / `rag today` / `rag digest` / `rag pendientes`, evidence rendering, secciones determinísticas (Agenda/Gmail/System/Screen Time/Drive), brief diff signal (kept/deleted → behavior.jsonl), WhatsApp push. |
| Ingestion | `rag-ingestion` | `rag read` (incl. YouTube), `rag capture`, `rag inbox`, `rag prep`, wikilinks densifier, `rag links` semantic URL finder. |
| Vault health | `rag-vault-health` | `rag archive`, `rag dead`, `rag followup`, `rag dupes`, contradiction radar (Phase 1+2+3), `rag maintenance` (incl. orphan HNSW segment cleanup, WAL checkpoint, log + behavior rotation). |
| Integraciones | `rag-integrations` | Todos los `_fetch_*` (Apple Mail/Reminders/Calendar, Gmail API, WhatsApp bridge SQLite + listener, weather, Drive activity, screen time `knowledgeC.db`), ambient agent, `wa-tasks` extractor. |

Para tareas ambiciosas o cross-dominio, arrancar por `pm`: devuelve un plan de dispatch que el caller ejecuta.

## Por qué hay 3 slots de developer idénticos

`rag.py` es single-file (~21k líneas, 883 tests, 44 archivos de test). Cuando dos sesiones de Claude Code editan el archivo en paralelo, la última `Write` pisa los cambios intermedios. Tres slots `developer-{1,2,3}` con cuerpo idéntico permiten:

1. **Claim por slot**: cada peer Claude reclama el slug libre más bajo (`developer-1`, después `-2`, después `-3`). Slugs distintos = no shadowing en el registry de subagents.
2. **Paralelismo real**: 3 generalistas pueden trabajar en sub-zonas distintas simultáneamente, coordinados por `mcp__claude-peers__set_summary`.
3. **Sin proliferación temática**: en lugar de inventar agents por feature (`developer-mcp`, `developer-tests`...) que rotan rápido, mantenemos un único contrato de generalista replicado.

Si necesitás más de 3 generalistas en paralelo, agregá `developer04.md` con `name: developer-4` (mismo cuerpo). Si la presión cede, podés borrar `developer-3` sin afectar a los otros.

## Por qué los demás existen

1. **Documentan ownership** — cada uno declara su zona y qué NO debe tocar (ver tabla en `developer*.md`).
2. **Enrutan tareas** — `Agent(subagent_type: rag-retrieval, ...)` para que la sesión principal delegue en contexto fresco sin pisar edits en curso.
3. **Preservan invariantes** — cada agent lista los contratos que no se pueden romper (eval baselines con CIs, silent-fail, layout del brief, etc.).

## Protocolo cuando hay peers activos

Si `mcp__claude-peers__list_peers(scope: "repo")` devuelve >1 instancia Claude Code en el repo:

1. **Announce**: `mcp__claude-peers__set_summary` declarando dominio + función (ej. `"developer-1: editando _wa_extract_actions en rag.py:18420"`).
2. **Check peers**: `list_peers(scope: "repo")` antes de cada edit grande en `rag.py`.
3. **Rebase mental**: si otro peer tiene summary overlapping con tu zona, **pausar y coordinar** vía `send_message` antes de escribir.
4. **Commit frecuente**: commits chicos por feature → git es el source of truth ante conflictos.
5. **Trabajo paralelo ambicioso**: usar `EnterWorktree` antes de editar la misma rama en simultáneo.

## Cuándo NO usar un agent

- Tarea trivial de 1 archivo fuera de `rag.py` (ej. editar un test): trabajar directo.
- Pregunta conversacional: respondé vos.
- Tareas cross-dominio (ej. "agregá Gmail al morning brief"): arrancar por `pm` para que rutee — típicamente termina en `rag-brief-curator` (que decide layout) coordinando con `rag-integrations` (que aporta el fetcher).

## Cuándo agregar un nuevo agent de dominio

Si aparece una zona nueva que se edita frecuentemente (ej. dashboard web en `web/`, eval harness en `queries.yaml`), crear un agent dedicado con zona clara y "don't touch" explícito. No crear agents por feature efímera — el costo de mantenerlos sincronizados con el repo CLAUDE.md no se paga si la zona se va a tocar dos veces y nunca más.
