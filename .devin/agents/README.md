# Devin agents — obsidian-rag

Este proyecto usa **custom subagents** de [Devin CLI](https://cli.devin.ai/docs) para rutear trabajo especializado (retrieval, LLM prompts, briefs, ingestion, vault health, integraciones).

## TL;DR — la fuente canónica son los `.md` en `.claude/agents/`

Devin **importa automáticamente** agents definidos en `.claude/agents/*.md` ([docs](https://cli.devin.ai/docs/subagents#importing-from-other-tools)), así que los 10 agents standard de este repo ya están disponibles desde Devin sin duplicación. Editá `.claude/agents/<name>.md` y los cambios se reflejan en ambos tools (Claude Code + Devin). No dupliques archivos en `.devin/agents/<name>/AGENT.md` salvo que quieras un agent **Devin-only**.

## Roster

| Slug | Responsabilidad | Archivo |
|------|-----------------|---------|
| `pm` | Planifica y rutea — descompone tareas grandes, asigna a specialists, detecta overlaps con peers. No edita código. | [`.claude/agents/pm.md`](../../.claude/agents/pm.md) |
| `developer-1` | Refactors cross-cutting, nuevos subcommands CLI, tests, `mcp_server.py`, `pyproject.toml`, launchd plists, bug fixes que no caen en un dominio. | [`.claude/agents/developer01.md`](../../.claude/agents/developer01.md) |
| `developer-2` | Idéntico a `developer-1`. Segundo slot paralelo para peers concurrentes. | [`.claude/agents/developer02.md`](../../.claude/agents/developer02.md) |
| `developer-3` | Idéntico a `developer-1`/`developer-2`. Tercer slot paralelo. | [`.claude/agents/developer03.md`](../../.claude/agents/developer03.md) |
| `rag-retrieval` | `retrieve()`, HyDE, rerank, BM25, corpus cache, graph expansion, deep retrieve, scoring, `ranker.json`, ranker-vivo nightly online-tune + rollback. | [`.claude/agents/rag-retrieval.md`](../../.claude/agents/rag-retrieval.md) |
| `rag-llm` | Toda prompt en `rag.py`, model resolution chain, `HELPER_OPTIONS`/`CHAT_OPTIONS`, JSON schema + parsers, citation verifier, contextual summary cache, HyDE prompt body, `rag do` agent loop, STT/TTS. | [`.claude/agents/rag-llm.md`](../../.claude/agents/rag-llm.md) |
| `rag-brief-curator` | `rag morning` / `rag today` / `rag digest` / `rag pendientes`, evidence rendering, secciones determinísticas, brief diff signal, WhatsApp push. | [`.claude/agents/rag-brief-curator.md`](../../.claude/agents/rag-brief-curator.md) |
| `rag-ingestion` | `rag read` (incl. YouTube), `rag capture`, `rag inbox`, `rag prep`, wikilinks densifier, `rag links` semantic URL finder. | [`.claude/agents/rag-ingestion.md`](../../.claude/agents/rag-ingestion.md) |
| `rag-vault-health` | `rag archive`, `rag dead`, `rag followup`, `rag dupes`, contradiction radar (Phase 1+2+3), `rag maintenance`. | [`.claude/agents/rag-vault-health.md`](../../.claude/agents/rag-vault-health.md) |
| `rag-integrations` | Todos los `_fetch_*` (Apple Mail/Reminders/Calendar, Gmail, WhatsApp bridge, weather, Drive, screen time, ambient agent, `wa-tasks`. | [`.claude/agents/rag-integrations.md`](../../.claude/agents/rag-integrations.md) |

Reglas + doctrina completas en [`.claude/agents/README.md`](../../.claude/agents/README.md) (incluye "por qué hay 3 slots de developer idénticos", protocolo con peers, cuándo NO usar un agent, cuándo agregar uno nuevo).

## Cómo invoca Devin estos agents

Usá la tool [`run_subagent`](https://cli.devin.ai/docs/subagents) con el slug como `profile`:

```
run_subagent(
  title: "Fix BM25 accent folding",
  task: "<goal + context + ruled-out + invariants at risk>",
  profile: "rag-retrieval",
  is_background: false
)
```

- `profile` debe ser el `name` del frontmatter (`pm`, `rag-retrieval`, etc.), no el nombre del archivo.
- `is_background: false` = foreground (pausa el root agent hasta terminar, permite aprobar tool calls).
- `is_background: true` = paralelo (ideal para tasks independientes, el root sigue trabajando).
- Los subagents no heredan el historial del root — frontloadeá todo el contexto que necesiten en `task`.

La regla "cualquier tarea que edita ≥3 archivos pasa primero por `pm`" del [`CLAUDE.md`](../../CLAUDE.md) aplica acá igual:

```
run_subagent(
  title: "Plan: agregar gmail al morning brief",
  task: "<goal + context + ...>",
  profile: "pm",
  is_background: false
)
```

El PM devuelve un dispatch plan; el root lo ejecuta lanzando los agents nombrados en el orden prescripto.

## Sintaxis nativa Devin (para agents Devin-only)

Si alguna vez necesitás un agent que **solo** deba aparecer en Devin (y no en Claude Code), usá la sintaxis nativa — directorio + `AGENT.md`:

```
.devin/agents/
└── <slug>/
    └── AGENT.md
```

Formato del `AGENT.md` (YAML frontmatter + system prompt):

```markdown
---
name: <slug>                    # debe matchear el directorio; no puede ser subagent_explore/subagent_general
description: <cuándo usarlo>    # visible al root agent cuando elige profile
model: sonnet                   # opcional — override del modelo del subagent
allowed-tools:                  # opcional — whitelist de tools. Si se omite, acceso total
  - read
  - grep
  - glob
  - exec
permissions:                    # opcional — overrides de permisos
  allow:
    - Exec(git diff)
  deny:
    - write
    - edit
---

You are a <slug> subagent. <system prompt>…
```

Diferencias vs `.claude/agents/*.md`:

| Campo | Claude Code | Devin nativo |
|-------|-------------|--------------|
| Ubicación | `.claude/agents/<slug>.md` (archivo suelto) | `.devin/agents/<slug>/AGENT.md` (directorio + file) |
| Tools | `tools: Read, Edit, Grep, Glob, Bash` | `allowed-tools: [read, edit, grep, glob, exec]` (lista YAML, lowercase) |
| Modelo | `model: sonnet` | `model: sonnet` (idem) |
| Permisos | no soporta granularidad | `permissions: { allow: [...], deny: [...], ask: [...] }` |

Devin entiende **ambos** formatos automáticamente — si un agent vive en `.claude/agents/<slug>.md` con `tools:` frontmatter, lo carga igual. Solo usá el formato Devin nativo cuando necesites `permissions:` granulares o cuando quieras ocultar el agent de Claude Code.

## Cuándo agregar un agent nuevo

Guía en [`.claude/agents/README.md` §"Cuándo agregar un nuevo agent de dominio"](../../.claude/agents/README.md). En resumen: si aparece una zona nueva que se edita frecuentemente, creá un agent dedicado con zona clara y "don't touch" explícito. No crear agents por feature efímera.

Ubicación recomendada por default: **`.claude/agents/<slug>.md`** (compatible con ambos tools). Solo caer en `.devin/agents/<slug>/AGENT.md` si hay razón específica (permisos granulares, Devin-only).
