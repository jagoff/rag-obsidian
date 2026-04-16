# Agents del repo obsidian-rag

Agents especializados por dominio. Claude Code los detecta automáticamente al bootear en este repo y pueden ser invocados vía `Agent(subagent_type: <name>, ...)`.

## Dominios

| Agent | Responsabilidad | Zona en `rag.py` (aprox) |
|-------|-----------------|--------------------------|
| `pm` | Planifica y rutea — descompone tareas grandes, asigna a specialists, detecta overlaps con peers. No edita código. | N/A |
| `developer` | Generalista — refactors cross-cutting, nuevos subcommands CLI, tests, `mcp_server.py`, `pyproject.toml`, launchd plists | spread |
| `rag-retrieval` | Retrieve pipeline, HyDE, rerank, corpus cache, scoring | `retrieve()` + helpers |
| `rag-brief-curator` | Morning / today / digest briefs, evidence rendering | ~12900–13950 |
| `rag-ingestion` | `rag read` (incl. YouTube), capture, wikilinks, inbox triage, prep | ~11665–12100 |
| `rag-vault-health` | Archive, dead, followup, contradictions, dupes, maintenance | spread |
| `rag-integrations` | Apple Mail/Reminders/Calendar, Gmail API, WhatsApp, weather, ambient | ~12100–12700 |

Para tareas ambiciosas o cross-dominio, arrancar por `pm`: devuelve un plan de dispatch que el caller ejecuta.

## Por qué existen

`rag.py` es single-file (~16k líneas por decisión explícita — ver `CLAUDE.md`). Cuando múltiples sesiones de Claude Code editan el archivo en paralelo, la última `Write` pisa los cambios intermedios. Los agents:

1. **Documentan ownership** — cada uno declara su zona y qué NO debe tocar.
2. **Enrutan tareas** — `Agent(subagent_type: rag-retrieval, ...)` para que la sesión principal delegue en contexto fresco sin pisar edits en curso.
3. **Preservan invariantes** — cada agent lista los contratos que no se pueden romper (eval baselines, silent-fail, layout del brief, etc.).

## Protocolo cuando hay peers activos

Si `list_peers` devuelve >1 instancia Claude Code en el repo:

1. **Announce**: usar `mcp__claude-peers__set_summary` para declarar el dominio que vas a tocar (ej. `"editando rag-retrieval: rerank weights"`).
2. **Check peers**: `list_peers(scope: "repo")` antes de empezar edit grande en `rag.py`.
3. **Rebase mental**: si otro peer tiene summary overlapping con tu zona, **pausar y coordinar** vía `send_message` antes de escribir.
4. **Commit frecuente**: commits chicos por feature → git es el source of truth ante conflictos. En trabajo paralelo real, preferir worktrees (`EnterWorktree`) antes que editar main branch en simultáneo.

## Cuándo NO usar un agent

- Tarea trivial de 1 archivo fuera de `rag.py` (ej. editar un test): trabajar directo.
- Pregunta conversacional: respondé vos.
- Tareas cross-dominio (ej. "agrega Gmail a morning brief"): agent de `rag-brief-curator` (es quien decide layout) que coordine con `rag-integrations` si necesita el fetcher.

## Cuándo agregar un nuevo agent

Si aparece un dominio nuevo que se edita frecuentemente (ej. web dashboard en `web/`, MCP server en `mcp_server.py`, eval harness en `queries.yaml`), crear un agent dedicado. Mantener la regla: cada agent con zona clara y "don't touch" explícito.
