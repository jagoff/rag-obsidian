---
name: perf-budget
description: Get the empirical performance budget for any code path in obsidian-rag BEFORE deciding to optimize. Queries telemetry SQL (rag_log_sql, rag_behavior, rag_anticipate_candidates) for p50/p95/p99 + count/hr in a 7-day window, compares against eval baseline if relevant, and returns a clear verdict — "this path is OK / worth optimizing / actual bottleneck". Triggers on `/perf-budget <function>`, `cuál es el budget de X`, `dame números reales de X`, `cuánto tarda X en producción`.
allowed-tools:
  - read
  - grep
  - exec
permissions:
  allow:
    - Exec(sqlite3)
    - Exec(rg)
    - Exec(.venv/bin/python)
---

# /perf-budget — telemetry-first before optimizing

Esta skill materializa la regla establecida en la memoria
`feedback_telemetry_first_audit.md`:

> Antes de leer código, queries directos a telemetry.db + sql_state_errors.jsonl;
> data-driven encuentra bugs en 5s que grep no.

Y la lección del 2026-04-25 sobre el audit fantasma del rag-perf-auditor:
nunca optimices basado en code-reading sin antes validar con telemetría
real.

**Diferencia con `/profile`**: `/profile` corre cProfile sintético sobre
una invocación; `/perf-budget` lee la telemetría histórica de
producción. Complementarios — para una optimización seria, usar ambas.

## Inputs

- `/perf-budget retrieve` — function name
- `/perf-budget --component brief` — component-level (mapea a múltiples ops)
- `perf-budget de _load_behavior_priors` — lenguaje natural

## Workflow

### Paso 1 — Identificar el "op" name en telemetría

Las funciones del repo se loguean con un `op` string en `rag_log_sql`.
Ejemplos:

| Función | op pattern |
|---|---|
| `retrieve()` | `retrieve`, `retrieve.embed`, `retrieve.rerank`, `retrieve.bm25` |
| `_load_behavior_priors` | `behavior_priors_sql_read` |
| `_load_corpus` | (no logueado, monkey-patch needed) |
| LLM calls | `llm.{model}` |
| Brief sections | `brief.{section}` |

Buscá en rag.py para confirmar:

```bash
rg -n "rag_log_sql|_log_sql|_record_sql_op" rag.py | head -20
```

### Paso 2 — Query base contra telemetry.db

```bash
sqlite3 ~/.local/share/obsidian-rag/ragvec/ragvec.db "
  WITH window_data AS (
    SELECT op, duration_ms,
           ts, datetime(ts, 'unixepoch') as ts_iso
    FROM rag_log_sql
    WHERE op LIKE '%{op_pattern}%'
      AND ts > strftime('%s','now')-604800   -- últimos 7 días
  )
  SELECT
    op,
    COUNT(*) as n_calls,
    ROUND(COUNT(*) * 1.0 / 7, 1) as calls_per_day,
    ROUND(AVG(duration_ms), 2) as avg_ms,
    ROUND(MIN(duration_ms), 2) as min_ms,
    -- Aprox p50/p95/p99 (sqlite no tiene percentile_cont nativo;
    -- aproximamos con NTILE)
    ROUND((SELECT duration_ms FROM window_data w2 WHERE w2.op = w1.op
           ORDER BY duration_ms LIMIT 1
           OFFSET (COUNT(*) / 2)), 2) as p50_ms,
    ROUND(MAX(duration_ms), 2) as max_ms
  FROM window_data w1
  GROUP BY op
  ORDER BY n_calls DESC
  LIMIT 20;
"
```

Si la función NO sale en `rag_log_sql`, decirlo explícitamente y
sugerir profilear con `/profile` o instrumentar con `_log_sql_op(...)`.

### Paso 3 — Cruce contra eval baseline (si aplica)

Si la función está en el path crítico de retrieve:

```bash
sqlite3 ~/.local/share/obsidian-rag/ragvec/ragvec.db "
  SELECT
    DATE(ts, 'unixepoch') as day,
    AVG(p95_latency_ms) as avg_p95,
    AVG(hit_at_5) as hit5,
    AVG(mrr) as mrr
  FROM rag_eval_runs
  WHERE ts > strftime('%s','now')-2592000   -- 30 días
  GROUP BY day
  ORDER BY day DESC LIMIT 10;
"
```

Buscar el threshold en `queries.yaml` o `--max-p95-ms` flag default.

### Paso 4 — Devolveme reporte estructurado

```markdown
# Perf budget de `{function_name}` — {fecha}

## Producción (últimos 7 días)

| Op | Calls/día | Avg | p50 | p95 | Max |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

## Eval baseline (últimos 30 días)

- p95 retrieve actual: NN ms
- Threshold (queries.yaml --max-p95-ms): NN ms
- Trend: estable / degradando / mejorando
- Hit@5 actual: 0.NN (vs baseline 0.NN)

## Análisis

- **Costo total estimado**: NN_calls/día × NN_ms = X minutos/día de wall time
- **% del hot path crítico**: si la función forma parte de `retrieve`,
  cuál es su contribución (cumtime fraction) — cross-reference con
  `/profile` si disponible
- **Trend**: ¿el avg está subiendo en las últimas 2 semanas? (cambio reciente
  introdujo regresión?)

## Veredicto

[claro y accionable]:

- ✅ **OK como está**: <5% del hot path, p95 dentro de threshold, trend
  estable. No optimizar.
- 🟡 **Vale la pena**: 5-15% del hot path, p95 cercano a threshold, trend
  degradando. Investigar fix con número estimado de mejora.
- 🔴 **Bottleneck real**: >15% del hot path Y/O p95 supera threshold.
  Prioridad alta — implementar fix con `/profile` para guiar.
- ❓ **Sin telemetría**: no está logueado. Sugerir instrumentar primero
  o usar `/profile` para análisis sintético.

## Sugerencia de próximo paso

Concrete next action — ej. "instrumentar con `_log_sql_op` línea NNNN
antes de seguir; reunir 3 días de data; re-correr `/perf-budget`".
```

## Anti-patterns

- ❌ NO sugerir fix basado solo en código sin validar con telemetría.
- ❌ NO comparar avg sin p95/p99 — outliers cuentan.
- ❌ NO ignorar el trend; un path "estable lento" es distinto de uno
  "estaba OK pero degradó la última semana" (segundo es regresión
  reciente, prioridad alta).
- ❌ NO afirmar "X es bottleneck" sin cross-reference contra el rest del
  hot path (un 50ms en una función que se llama 1×/día es <5min wall
  time anual; en una función que se llama 10k×/día es 8min/día).
- ✅ SÍ proponer instrumentar si la función crítica no está logueada.
- ✅ SÍ distinguir warm-path vs cold-path (primer query del día vs
  query con cache warm).

## Cross-reference con `/profile`

Para optimizaciones serias, **invocar las dos**:

1. `/perf-budget <function>` → "este path es realmente lento en
   producción? cuán seguido se invoca?"
2. Si veredicto = vale la pena: `/profile <function>` → "qué línea
   exacta consume el cumtime? hay micro-optimizations o requiere
   refactor estructural?"

Ambas dan números reales. Audit estático (rag-perf-auditor) sirve para
**generar candidatos**, no para validar wins. La validación va por las
dos skills empíricas.
