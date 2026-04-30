# Package split de `rag/__init__.py` — Plan ejecutable

**Status**: Plan documentado, NO ejecutado. Trabajo multi-sesión (estimado 1-2 semanas focused).

**Motivación**: el monolito creció 32k → ~60k líneas en pocas semanas (drift +87%). Hoy `rag/__init__.py` tiene 752 funciones top-level + 12 grupos Click + 27 plists generators + 56 CREATE TABLE + helpers de retrieval/rerank/learning/ambient/anticipatory/etc. Cada feature nueva (voice brief, MMR, vault health, contradiction penalty) ya se shipea como módulo separado por convención — pero el core sigue concentrado.

## Diagnóstico actual (al 2026-04-29)

```
59852 rag/__init__.py              ← MONOLITO
 1003 rag/anticipatory.py           ← extracted previously
  511 rag/brief_schedule.py         ← extracted 2026-04-29
  369 rag/voice_brief.py            ← extracted 2026-04-29
  259 rag/contradictions_penalty.py ← extracted 2026-04-29
  426 rag/mmr_diversification.py    ← extracted 2026-04-29
  595 rag/vault_health.py           ← extracted 2026-04-29
 ~4400 rag/integrations/*.py        ← already split
```

**Métricas problemáticas**:
- 752 funciones top-level (`def name(...)` o `@click.command/group`).
- Tiempo de import del módulo: ~3-5s warm (entrar a un comando CLI cualquiera implica cargar TODO).
- Tests que mockean `rag.X` tienen que reimplementar mocks porque cada función referencia 10+ siblings.
- Imports circulares forzaron el patrón "lazy import dentro de la función" en 30+ lugares.

## Estrategia — extract en chunks, NO big-bang

Hard constraint: el repo está en producción 24/7 (web server + 26 daemons). Cualquier refactor que rompa imports tira todo abajo. **Cero downtime regla**.

Patrón de cada chunk:
1. **Crear módulo nuevo** `rag/<topic>.py` con las funciones extraídas.
2. **Mantener re-exports en `rag/__init__.py`** durante el transition: `from rag.<topic> import *` o nombres explícitos. El módulo viejo sigue funcionando para callers externos.
3. **Internal callers** del rag/__init__.py se updean para usar `from rag.<topic> import name` directo.
4. **Tests propios** del módulo nuevo + suite full sin regresiones.
5. **Eval gate** (`rag eval` ≥ baseline) entre chunks que tocan retrieval/rerank.
6. **Deprecation warning** opcional en los re-exports si querés forzar la migración eventual.
7. **Commit + push** después de cada chunk extraído. **NO mergear todo al final** — cada chunk es independiente y rollbackeable.

## Chunks identificados (orden de prioridad)

### Chunk 1 — `rag/cli/` (~15k líneas estimadas)

Todos los `@click.command` y `@click.group` decorators. Es el más grande pero también el más ortogonal — los CLI subcomandos llaman a helpers core, no se llaman entre sí.

- `rag/cli/__init__.py` — agrega `cli` group raíz + registers todos los subcomandos.
- `rag/cli/index.py` — `index`, `watch`.
- `rag/cli/query.py` — `query`, `chat`, `links`.
- `rag/cli/session.py` — `session list/show/clear/cleanup/export`.
- `rag/cli/productivity.py` — `capture`, `morning`, `today`, `dead`, `dupes`, `inbox`, `prep`, `consolidate`.
- `rag/cli/agent.py` — `do`, `surface`, `file`, `autotag`.
- `rag/cli/vault.py` — `vault list/add/use/current/remove`.
- `rag/cli/ambient.py` — `ambient status/disable/test/log/folders`.
- `rag/cli/health.py` — `health`, `dashboard`, `stats`, `eval`, `log`, `gaps`, `timeline`, `graph`, `digest`, `maintenance`, `insights`.
- `rag/cli/feedback.py` — `feedback`, `rating`, `path`, `draft`, `brief`, `active-learning`, `routing`.
- `rag/cli/automation.py` — `setup`, `wa-scheduled-send`, `remind-wa`, `scheduler`, `serve`, `web`.
- `rag/cli/misc.py` — `weather`, `state`, `transcribe`, `open`, `followup`, `spotify-auth`, `silence`, `wikilinks`, `anticipate`, `voice-brief`, `migrations`.

**Riesgo**: medio. Click resolution depende del orden de imports — el `cli` group raíz tiene que existir antes de que cada `cli.add_command(...)` se ejecute. Patrón validado en otros monolitos refactoreados.

**Eval gate**: NO necesario (CLI es UI thin sobre helpers core).

**Estimado LOC tras extract**: rag/__init__.py baja ~25%.

### Chunk 2 — `rag/services.py` (~3k líneas)

Todo el roster de plists + `_services_spec()` + el bootstrap launchd. Auto-contenido excepto algunos helpers que llama (PATH del rag binary, env vars).

- `_services_spec()` y los 27 `_*_plist()` functions.
- `setup()` y la lógica de install/remove launchd.
- Helpers de plist generation.

**Riesgo**: bajo. Toca infra, no hot-path. Pero el test `test_services_spec_total_count` ya tuvo bugs varias veces — agregar tests más robustos antes/durante.

**Eval gate**: NO necesario.

### Chunk 3 — `rag/learning/` (~10k líneas)

Subsystem completo de feedback + calibration + online-tune + active learning + draft fine-tune.

- `rag/learning/feedback.py` — `record_feedback`, `harvest`, backfill.
- `rag/learning/calibration.py` — score calibration, paraphrase learning.
- `rag/learning/tune.py` — online tune nightly, eval-driven gates.
- `rag/learning/active.py` — re-query detection, session classification, implicit feedback inference.
- `rag/learning/draft_fine_tune.py` — wire-up del fine-tune (helper en `scripts/finetune_drafts.py`).
- `rag/learning/reranker_fine_tune.py` — wire-up del reranker LoRA (helper en `scripts/finetune_reranker.py`).

**Riesgo**: medio-alto. Tablas SQL compartidas, varios módulos se llaman entre sí. Necesita mapping cuidadoso de internal calls.

**Eval gate**: SÍ. Cualquier regresión ≥0.5pp → rollback.

### Chunk 4 — `rag/ambient.py` (~3k líneas)

Ambient agent: hook on save → wikilinks/dupes/whatsapp_ping. Auto-contenido.

- `_ambient_*` functions.
- `rag ambient` CLI subgrupo (movido al chunk 1).

**Riesgo**: bajo.

**Eval gate**: NO necesario.

### Chunk 5 — `rag/dashboards.py` (~4k líneas)

Endpoints SQL para `/dashboard`, `/learning`, `/status`, `/api/insights`, `/api/health`. Helpers que generan los JSON consumidos por las pages HTML.

- Funciones `_compute_dashboard_*`, `_compute_learning_*`, etc.
- Helpers de query a `rag_queries` / `rag_behavior` / `rag_feedback` para presentación.

**Riesgo**: bajo (es read-only, no escribe).

**Eval gate**: NO necesario.

### Chunk 6 — `rag/morning.py` + `rag/today.py` + `rag/digest.py` (~4k líneas)

Generación de los briefs (morning/today/digest narrative). Auto-contenido excepto retrieve() y embed_query.

**Riesgo**: bajo. Tests existen. Voice brief ya extraído.

**Eval gate**: NO necesario.

### Chunk 7 — `rag/proactive.py` (~3k líneas)

Subsystem proactivo: emergent themes, patterns, archive, surface bridges, dead notes detection.

**Riesgo**: bajo. Cada feature ya está medio aislada.

### Chunk 8 — `rag/retrieve.py` + `rag/rerank.py` (~6k líneas)

**El más sensible**. Hot-path de query/chat. Toca BM25, vector search, hybrid, rerank, MMR (ya extraído), contradiction penalty (ya extraído).

- `rag/retrieve.py` — `retrieve()`, `multi_retrieve()`, `deep_retrieve()`.
- `rag/rerank.py` — reranker loader, fine-tune wire-up, batch logic.
- `rag/embeddings.py` — embed_query, embed_texts, model loader.

**Riesgo**: ALTO. Cualquier cambio impacta calidad.

**Eval gate**: SÍ obligatorio. Tres runs de `rag eval` después de cada extract. Post-merge debe estar ≥ baseline.

**Estrategia especial**: extraer en sub-chunks pequeños (BM25 first, luego vector, luego hybrid, luego rerank). Eval gate entre cada uno.

### Chunk final — `rag/__init__.py` queda como API público

Tras los 8 chunks, `rag/__init__.py` debería bajar a ~5-10k líneas (constantes, schema DDL, init/setup).

Final state esperado:
```
~5000 rag/__init__.py            ← solo API pública + constantes
 ~15000 rag/cli/                  ← split en ~12 archivos
 ~10000 rag/learning/             ← 6 archivos
  ~6000 rag/retrieve.py + rerank.py + embeddings.py
  ~4000 rag/dashboards.py
  ~4000 rag/morning.py + today.py + digest.py
  ~3000 rag/services.py
  ~3000 rag/ambient.py
  ~3000 rag/proactive.py
  ~XXXX módulos ya extraídos (anticipatory, brief_schedule, voice_brief, contradictions_penalty, mmr_diversification, vault_health)
```

## Validación entre chunks

Antes de cada commit:

```bash
# 1. Tests de la zona afectada
pytest tests/test_<topic>*.py -q

# 2. Suite completa sin pre-existing failures
pytest tests/ -q -k 'not slow' --maxfail=15

# 3. Eval gate (solo chunks 3 y 8)
rag eval

# 4. Smoke CLI: cada subcomando que querés preservar
rag --help
rag query --help
rag stats
rag health --as-json
```

## Risk register

| Riesgo | Mitigación |
|--------|-----------|
| Imports circulares post-split | Lazy imports dentro de funciones, igual que ya hace el monolito en 30+ lugares. |
| Click subcomando rompe registro | Test `rag --help` después de cada chunk extract de CLI. |
| Tests existentes referencian `rag.foo` que se movió | Mantener re-export en `rag/__init__.py` durante el transition. |
| Eval gate regresa | Reduce el chunk a sub-chunks más chicos. |
| Daemon launchd se cae con import path nuevo | `rag setup` regenera plists. Plist usa `rag` como entry point. Si rompemos eso, nuevo plist queda mal. Verificar `launchctl list \| grep obsidian-rag` después de cada chunk. |
| Consumers externos del package | NO hay (es uso interno). El MCP server (`mcp_server.py`) usa `import rag` — preservar API pública. |

## Cómo arrancar el primer chunk

```bash
git checkout -b experimental/package-split-cli
mkdir -p rag/cli
# Mover decoradores Click. Empezar por el grupo más chico (e.g. `vault`).
# 1 PR por chunk, no agrupados.
```

**Tiempo estimado**: 1-2 semanas de trabajo focused (no part-time).

## NO en scope de este plan

- No es responsabilidad del split: refactorear el código que se mueve. Solo se mueve, no se mejora.
- No es responsabilidad del split: agregar tests nuevos. Si los tests existentes pasan, OK.
- No es responsabilidad del split: cambiar APIs públicas. Re-export everything.

## Aprendido del proceso de extracción 2026-04-29 (mini-extracts)

Durante el PM-mode del 2026-04-29 se extrajeron 6 módulos chicos como práctica del patrón:
- `rag/anticipatory.py` (1003 líneas, pre-existente)
- `rag/brief_schedule.py` (511, este día)
- `rag/voice_brief.py` (369, este día)
- `rag/contradictions_penalty.py` (259, este día)
- `rag/mmr_diversification.py` (426, este día)
- `rag/vault_health.py` (595, este día)

**Lecciones operativas**:
1. **Nombres exportados deben coordinarse**: el `voice_brief.py` exportó `synthesize_brief_audio` y `send_audio_to_whatsapp` que fueron llamadas desde `rag/__init__.py:_brief_push_to_whatsapp`. Si los nombres no matchean, ImportError silencioso (logged a `silent_errors.jsonl`) y feature OFF sin warning user-visible. Caso real: contradiction penalty estuvo wired a master 1+ semana antes de tener helper, con feature efectivamente OFF (verificado con eval pre/post).
2. **Tests del módulo nuevo cubren la API pública**: cada `apply_*`, `compute_*`, `is_*` documentado.
3. **`silent_errors.jsonl` es el oráculo**: si un import falla durante un extract, ahí queda registro. Greppear ese archivo después de cada deploy.
4. **Funciones pequeñas (<500 LOC) se extraen en <1 día**. Funciones grandes (>5k LOC, e.g. `retrieve()`) requieren split incremental por sub-chunks.

---

**Recomendación al user**: ejecutar este plan en una sesión dedicada, no en PM-mode mixto. El roadmap del PM-mode 2026-04-29 priorizó cerrar 13 features pequeñas; el package split es trabajo de infrastructure que merece foco aparte.
