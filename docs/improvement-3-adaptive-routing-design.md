# Improvement #3 — Adaptive routing design

> Query-aware pipeline routing: cada intent paga SOLO los stages que necesita.
> Precedente empírico: el bench `rerank_pool 30 → 15` dropeó P95 singles −66%
> (4704→1577 ms) sin perder hit@5 ([`rag.py:1755-1775`](../rag.py#L1755) +
> tabla en [CLAUDE.md "Rerank pool"](../CLAUDE.md) §Retrieval pipeline).
> Misma filosofía: menos trabajo por query cuando el stage no aporta.

---

## 1. Motivation + baseline

### Baseline actual (full pipeline, siempre)

Hoy `retrieve()` ([`rag.py:12250`](../rag.py#L12250)) corre sin cortocircuitar para
TODA query no-intent-shortcut:

```
query → classify_intent → infer_filters
      → expand_queries (qwen2.5:3b, ~1-3s, gated por _EXPAND_MIN_TOKENS=4)
      → embed(variants) batched bge-m3
      → per variant: sqlite-vec sem + BM25
      → RRF merge → dedup → expand to parent section
      → rerank pool=15 (bge-reranker-v2-m3 MPS fp32, ~500-800ms)
      → graph expansion (gated por GRAPH_EXPANSION_GATE=0.5)
      → [auto-deep si confidence < 0.10]
      → top-5 → LLM streamed (qwen2.5:7b, CHAT_OPTIONS num_ctx=4096)
      → citation-repair si ≤ 3 bad citations (~5-8s non-streaming)
```

Breakdown típico:

| Stage                       | P50 (ms)    | P95 (ms)    |
|-----------------------------|-------------|-------------|
| embed (local bge-m3)        | 30          | 60          |
| expand_queries              | 1200        | 2800        |
| sem + bm25 (3 variants)     | 80          | 180         |
| rerank pool=15              | 500         | 800         |
| graph expansion             | 20          | 50          |
| **retrieve() total**        | **1163**    | **1577**    |
| LLM generation              | 3000        | 5500        |
| citation-repair (cuando corre) | 5000     | 8000        |
| **query total**             | **~5-9s**   | **~10-15s** |

Los intent shortcuts (`count` / `list` / `recent` / `agenda`) ya evitan retrieve +
LLM ([`rag.py:15859-15920`](../rag.py#L15859)), pero **antes** pagaron
`reformulate_and_expand()` helper call (~1-2s) cuando hay history → overhead que
NO se usa porque el intent no va a retrieve.

### Observación clave

- **Lookup 1-fact** (top-1 rerank ≫ 0.3): con qwen2.5:3b + num_ctx=2048 alcanza.
- **Metadata-only** (`count`/`list`/`recent`/`agenda`): ya skipea retrieve, pero paga expand con history.
- **Synthesis/comparison**: se beneficia de **más** pool (30) y **más** num_ctx (8192). Pool=15 global es sub-óptimo ahí.
- **Semantic default**: pipeline actual está bien calibrado.

### Meta

Agregar **per-intent config** que controle qué stages corren / con qué parámetros. Feature-flagged (OFF por default). Validar en bucketed eval que singles hit@5 no regresa ±0.5pp y P95 por bucket baja en los buckets targeted.

---

## 2. Per-intent pipeline matrix

Precedencia de `classify_intent` ([`rag.py:9291-9307`](../rag.py#L9291)):
`count → list → agenda → recent → comparison → synthesis → semantic`.

| Intent       | expand_queries       | rerank pool | graph expansion | citation-repair | chat model    | num_ctx | auto-deep | SLA target   |
|--------------|----------------------|-------------|-----------------|-----------------|---------------|---------|-----------|--------------|
| `count`      | skip                 | skip        | skip            | skip            | none (metadata) | —     | skip      | <200ms       |
| `list`       | skip                 | skip        | skip            | skip            | none          | —       | skip      | <200ms       |
| `recent`     | skip                 | skip        | skip            | skip            | none          | —       | skip      | <200ms       |
| `agenda`     | skip                 | skip        | skip            | skip            | none (formateador) | —  | skip      | <500ms       |
| `semantic` (lookup fast-path) | skip (short) / default | 5   | skip            | skip            | qwen2.5:3b    | 2048    | skip      | <2s          |
| `semantic` (default)          | default (3 paraphrases) | 15   | default (gated) | on (≤3 bad)     | qwen2.5:7b    | 4096    | on (<0.10) | <6s          |
| `comparison` | skip (ruido dos-términos) | 30    | on (always)     | on              | qwen2.5:7b    | 8192    | on         | <12s         |
| `synthesis`  | skip (ruido cross-source) | 30      | on (always)     | on              | qwen2.5:7b    | 8192    | on         | <15s         |

### Justificación por celda

**`count`/`list`/`recent`/`agenda` — stages todos OFF**

Handlers devuelven lista de metas sin LLM. El único overhead evitable hoy: `reformulate_and_expand()` cuando hay history. **Fix Fase B**: mover `classify_intent` **antes** del bloque reform+expand cuando el intent resultante es metadata-only. Alternativa: classify dos veces (pre-reform + post-reform) y skipear expand si algún lado es metadata.

**`semantic` — fast-path lookup (new)**

Cuándo disparar (early exit, no intent nuevo):
1. Query corta (tokens < 4) → ya skipea expand
2. Top-1 rerank score > `RAG_LOOKUP_THRESHOLD` (default 0.6) después de pool=5 pass
3. No múltiples notas con score > 0.3 en top-3 (no cross-reference)

Por qué rerank pool=5: top-1 muy fuerte → ampliar pool solo gasta reranker latency. Si falla threshold → fall-through a semantic default.

Por qué qwen2.5:3b: lookup 1-fact es "citar un chunk". Produce citas literales cuando el chunk es corto y evidente. Caveat: prompt **tiene** que ser `SYSTEM_RULES_LOOKUP` ([`rag.py:9925`](../rag.py#L9925)) para que el modelo chico no divague.

Por qué num_ctx=2048: top-5 chunks ≈ 1500 tok + SYSTEM_RULES_LOOKUP + query + answer cabe sobrado. **OJO**: el invariant de prefix-cache ([`rag.py:1535-1541`](../rag.py#L1535)) dice "num_ctx must match every caller". El fast-path va a un modelo DISTINTO (qwen2.5:3b, no qwen2.5:7b), por lo tanto no comparte prefix-cache con semantic default. Primera query fast-path paga ~2-3s de KV re-init; cálido después.

**`comparison`/`synthesis` — pool=30 + num_ctx=8192**

El invariante del rerank pool ("pool=15 domina") se midió sobre mix completo. Para queries que piden **cross-reference ≥2 fuentes**, recall en top-5 NO es suficiente — querés ≥2 notas independientes dentro del top-5, que a pool=15 puede colapsar a una sola nota con 3 chunks.

num_ctx=8192 porque: rules (~500 tok) + top-5 full chunks (~2500 tok) + extras progressive (~500 tok) + graph (~1000 tok) + history (si chat) + answer padding llega a 4k sin margen. La regla "no bumpees num_ctx sin medir" aplica a default path; synthesis/comparison son casos nuevos donde el prompt real genuinamente crece.

expand_queries skipped: paraphrases de "X vs Y" introducen tokens que no están en ninguna nota (el helper parafrasea el conector), empeorando BM25. Precedente: `--no-multi` flag + `web/server.py:3953` ya hardcodea `multi_query=False`.

graph expansion ALWAYS on: para synthesis/comparison el 1-hop wikilink es la señal más importante después del rerank. Override `GRAPH_EXPANSION_GATE=0.5` ([`rag.py:12675`](../rag.py#L12675)).

### ¿`lookup` amerita intent explícito?

**Recomendación: NO.** Usar early-exit dentro de `semantic`.

Razones:
1. No hay regex confiable que distinga single-fact lookup de semantic general. "Qué usa adam jones?" (lookup) vs "Qué sabés de adam jones?" (semantic) son léxicamente idénticos.
2. Los intents existentes se detectan por patrón léxico (verbo + estructura). Lookup no tiene uno — es propiedad del resultado (top-1 score).
3. Agregar `lookup` al enum rompe invariant "`system_prompt_for_intent()` cubre todos los intents" sin beneficio.

**Implementación del early-exit (Fase C)**:

```python
# dentro de retrieve(), post-rerank, pre-graph-expansion:
if (_adaptive_routing_enabled()
        and intent == "semantic"
        and _effective_pool == 5
        and final_scores and final_scores[0] > _LOOKUP_THRESHOLD):
    return {..., "fast_path": True, ...}
```

Y en el caller:
```python
if result.get("fast_path"):
    chat_model = "qwen2.5:3b"
    options = {**CHAT_OPTIONS, "num_ctx": 2048}
    repair_max = 0
```

Decisión pool: empezar conservador — correr siempre `pool=15` default + gate post-rerank. Si luego queremos −40% extra de latencia, activar pool=5 en sub-fase C.2.

---

## 3. Archivos y líneas

Todo en [`rag.py`](../rag.py). Cero cambios en otros archivos salvo tests.

| Archivo | Línea | Cambio |
|---|---|---|
| [`rag.py`](../rag.py) | ~110 | Agregar `_LOOKUP_THRESHOLD`, `_LOOKUP_MODEL`, `_adaptive_routing()` helper |
| [`rag.py`](../rag.py) | 1533-1544 | `CHAT_OPTIONS` default queda. Nuevos dicts derivados `_CHAT_OPTIONS_SYNTHESIS` (num_ctx=8192) y `_CHAT_OPTIONS_LOOKUP` (num_ctx=2048). NO editar el base (prefix-cache invariant) |
| [`rag.py`](../rag.py) | 1670-1690 | `resolve_chat_model(intent_hint=None)` — cuando hint="lookup" y HELPER_MODEL disponible, devolver helper. Dict separado `_CHAT_MODEL_RESOLVED_BY_HINT` |
| [`rag.py`](../rag.py) | 1775 | `RERANK_POOL_MAX=15` queda + `_RERANK_POOL_BY_INTENT = {"comparison": 30, "synthesis": 30}` |
| [`rag.py`](../rag.py) | 8957 | `_EXPAND_SKIP_INTENTS = frozenset({"comparison", "synthesis"})` |
| [`rag.py`](../rag.py) | 8974-9043 | `expand_queries()` — param `intent=None`. Si `intent in _EXPAND_SKIP_INTENTS` → `[question]` sin helper. Guard pre-cache check |
| [`rag.py`](../rag.py) | 12250-12268 | `retrieve()` — params `intent=None, fast_path=False`. Cuando adaptive activo propagar a expand + pickear rerank_pool de `_RERANK_POOL_BY_INTENT` si no pasado explícito |
| [`rag.py`](../rag.py) | 12344-12351 | `expand_queries(search_query)` → `expand_queries(search_query, intent=intent)` |
| [`rag.py`](../rag.py) | 12441 | `_effective_pool` — prioridad: caller override > intent default > `RERANK_POOL_MAX` |
| [`rag.py`](../rag.py) | 12663-12717 | Graph expansion — override `GRAPH_EXPANSION_GATE=0.5` cuando `intent in _GRAPH_ALWAYS_INTENTS` |
| [`rag.py`](../rag.py) | 12804-12821 | Retorno retrieve() — agregar `"intent": intent, "fast_path": fast_path_fired` |
| [`rag.py`](../rag.py) | 12866-12897 | `deep_retrieve()` — propagar `intent`. Sub-query generado por `_judge_sufficiency()` sigue siendo semantic (intent=None en iter profundas) |
| [`rag.py`](../rag.py) | 15858-15920 | Intent routing en `query()` — mover `classify_intent` ANTES del reformulate bloque. Si metadata-only → handler directo |
| [`rag.py`](../rag.py) | 15936-15953 | `_retrieve_kwargs` + `intent=intent`. Si `result.get("fast_path")` → skip auto-deep aun si confidence<0.10 |
| [`rag.py`](../rag.py) | 16070-16089 | Prompt assembly — seleccionar chat_model + options según intent + fast_path |
| [`rag.py`](../rag.py) | 16097-16120 | Stream call `ollama.chat()` — variables en vez de literales `resolve_chat_model()` + `CHAT_OPTIONS` |
| [`rag.py`](../rag.py) | 16134-16168 | Citation-repair gate — skip si `result.get("fast_path")` o `intent == "lookup"` |
| [`rag.py`](../rag.py) | 17130-17140 | Mismo patrón en `chat()` — auto-deep + fast-path |
| [`rag.py`](../rag.py) | 17247-17284 | Skip citation-repair en chat() |
| [`rag.py`](../rag.py) | 17636-17646 | `rag eval --buckets` flag (ver §5) |
| [`rag.py`](../rag.py) | 30667-30668 | `rag do` tool-calling retrieve — propagar intent (no fast-path) |
| [`web/server.py`](../web/server.py) | 3953 | Override `rerank_pool=5` + `multi_query=False` queda. Caller-knows-best — respetado siempre |

---

## 4. Feature flags

| Env var | Default | Efecto |
|---|---|---|
| `RAG_ADAPTIVE_ROUTING` | `0` | Master switch. `0` → `retrieve()` ignora `intent` kwarg (no-op). `1` → toda §2 activa |
| `RAG_FORCE_FULL_PIPELINE` | `0` | Debug override. `1` → todos los stages siempre, aun con routing activo |
| `RAG_LOOKUP_THRESHOLD` | `0.6` | Top-1 rerank score mínimo para fast-path semantic. Calibrado: bge-reranker 0.3-0.5 = match bueno, 0.6-0.9 = hit fuerte |
| `RAG_LOOKUP_MODEL` | `qwen2.5:3b` | Modelo del fast-path. Aislado del helper |
| `RAG_SYNTHESIS_POOL` | `30` | Rerank pool para synthesis. Expone para `rag eval --buckets` sweep |
| `RAG_COMPARISON_POOL` | `30` | Idem comparison |
| `RAG_SYNTHESIS_NUM_CTX` | `8192` | num_ctx synthesis/comparison. Bajar a 6144 si empeora latencia |

```python
_LOOKUP_THRESHOLD = float(os.environ.get("RAG_LOOKUP_THRESHOLD", "0.6"))
_LOOKUP_MODEL = os.environ.get("RAG_LOOKUP_MODEL", "qwen2.5:3b")

def _adaptive_routing() -> bool:
    if os.environ.get("RAG_FORCE_FULL_PIPELINE") == "1":
        return False
    return os.environ.get("RAG_ADAPTIVE_ROUTING") == "1"
```

---

## 5. Bucketed eval extension

### Estado actual

`rag eval` ([`rag.py:17636-18009`](../rag.py#L17636)) reporta 2 buckets: `singles` y `chains`. No discrimina por intent — un regression en solo `synthesis` queda escondido detrás del promedio.

### Propuesta: `rag eval --buckets`

```bash
rag eval                     # como hoy
rag eval --latency           # como hoy
rag eval --buckets           # nueva
rag eval --buckets --latency # todas las métricas
```

Output tentativo:

```
Per-intent bucket (k=5)
┌────────────────┬─────┬────────┬───────┬──────────┬────────┬────────┐
│ Intent         │ n   │ hit@5  │ MRR   │ recall@5 │ P50 ms │ P95 ms │
├────────────────┼─────┼────────┼───────┼──────────┼────────┼────────┤
│ count          │ 4   │ 100.0% │ 1.000 │  1.00    │    45  │    80  │
│ list           │ 3   │ 100.0% │ 1.000 │  1.00    │    52  │    75  │
│ recent         │ 5   │ 100.0% │ 1.000 │  1.00    │    60  │   110  │
│ agenda         │ 8   │  87.5% │ 0.812 │  0.88    │    85  │   180  │
│ semantic       │ 27  │  74.1% │ 0.693 │  0.78    │  1250  │  1600  │
│ semantic/fast  │ 9   │  88.9% │ 0.870 │  0.92    │   450  │   700  │  ← nuevo
│ comparison     │ 8   │  75.0% │ 0.704 │  0.71    │  1420  │  1950  │
│ synthesis      │ 7   │  71.4% │ 0.682 │  0.69    │  1680  │  2200  │
├────────────────┼─────┼────────┼───────┼──────────┼────────┼────────┤
│ ALL singles    │ 60  │  78.3% │ 0.739 │  0.82    │  1163  │  1577  │
└────────────────┴─────┴────────┴───────┴──────────┴────────┴────────┘
```

### Implementación

1. **Clasificar singles**: `classify_intent(q, tags, folders)` per single, cache por string. Bucket label = intent. Fast-path etiquetado retrospectivo desde `result.get("fast_path")`.
2. **Clasificar chains**: agregar campo opcional `category: comparison|synthesis|cross-source` al spec de chain en `queries.yaml`. Default "semantic".
3. **Preservar output existente**. `--buckets` AGREGA tabla per-intent al final.
4. **Bootstrap CI per-bucket**: `_bootstrap_ci()` ([`rag.py:17694`](../rag.py#L17694)) filtered por bucket. Skip buckets con `n < 3`.
5. **Snapshot SQL**: `rag_eval_runs.extra_json` agrega `buckets` key.
6. **Gate `--max-p95-ms-bucket BUCKET=VALOR`** (repetible, fase E+).

---

## 6. Fases

### Fase A — infra sin side-effects

1. Agregar constantes + helper `_adaptive_routing()` en [`rag.py`](../rag.py)
2. Extender firmas `expand_queries(question, intent=None)` y `retrieve(..., intent=None, fast_path=False)`. Cuerpo ignora kwargs cuando flag OFF
3. Propagar `intent=intent` en callers de `retrieve()` (query, chat, rag do, morning/today/digest, multi_retrieve)
4. Tests: 2247 existentes deben pasar sin cambios

**Entregable**: commit "infra: per-intent routing scaffolding (no-op)". `rag eval --latency` numbers idénticos.

### Fase B — skip reformulate en metadata-only

1. Mover `classify_intent` en `query()` ([`rag.py:15857`](../rag.py#L15857)) ANTES del bloque reform+expand. Intent crudo metadata → handler directo
2. Idem chat()
3. Gated por `_adaptive_routing()` — OFF = reformulate corre como siempre
4. Nuevo test `tests/test_adaptive_metadata_skip.py`: mock helper, asertar que con count/list/recent/agenda + history, helper chat no se llama

**Entregable**: `rag query --session continue "cuántas notas tengo sobre AI"` baja ~1-2s post-B.

### Fase C — fast-path lookup

1. En `retrieve()` post-rerank: si adaptive ON + intent=semantic + top_score > threshold → `fast_path=True`
2. En `query()`: si `result.get("fast_path")` → `chat_model=_LOOKUP_MODEL`, `options=_CHAT_OPTIONS_LOOKUP`, skip repair + auto-deep
3. chat() — mismo dispatch
4. Empezar conservador: pool=15 default + gate post-rerank. C.2 opcional: activar pool=5 para doble-pasada
5. Tests: `tests/test_adaptive_fast_path.py`

**Entregable**: ~25% singles (top-1 > 0.6) bajan P50 de ~5s a ~1.5-2s.

### Fase D — synthesis/comparison pool=30 + num_ctx=8192

1. Wire `_RERANK_POOL_BY_INTENT` en `retrieve()`
2. Wire `_GRAPH_ALWAYS_INTENTS` override
3. Wire `_CHAT_OPTIONS_SYNTHESIS` en query/chat
4. expand_queries skip para synthesis/comparison
5. Tests: `tests/test_adaptive_synthesis_pipeline.py`

**Entregable**: chain_success +5-10pp en synthesis/comparison. Latencia sube ~2-3s en esos intents.

### Fase E — bucketed eval + default-on

1. Implementar `rag eval --buckets`
2. Baseline snapshot adaptive OFF → `docs/improvement-3-baseline.md`
3. Corrida con adaptive ON
4. Gates:
   - Merge: ningún intent regresa hit@5 más de −1pp
   - Default-on: synthesis+comparison chain_success +3pp; semantic P50 −20%; metadata P50 −50%
5. Flip default `RAG_ADAPTIVE_ROUTING=1` en commit separado, `RAG_ADAPTIVE_ROUTING=0` en launchd plists como rollback ready
6. Update CLAUDE.md §Retrieval pipeline sección "Adaptive routing"

---

## 7. Tests a agregar

| Path | Cobertura | Fase |
|---|---|---|
| `tests/test_adaptive_routing_flag.py` | env vars, `_adaptive_routing()` helper, ON/OFF/FORCE_FULL idempotency | A |
| `tests/test_adaptive_metadata_skip.py` | count/list/recent/agenda + history NO llama helper chat | B |
| `tests/test_adaptive_fast_path.py` | top_score > threshold + semantic → qwen2.5:3b + num_ctx=2048; no repair; no deep_retrieve | C |
| `tests/test_adaptive_synthesis_pipeline.py` | synthesis/comparison: pool=30 efectivo, no expand_queries, num_ctx=8192, graph siempre | D |
| `tests/test_expand_queries_intent.py` | `expand_queries("X", intent="synthesis")` → `["X"]` sin helper | A + D |
| `tests/test_retrieve_intent_propagation.py` | `retrieve(..., intent="synthesis")` propaga correctamente | A |
| `tests/test_eval_buckets.py` | `rag eval --buckets` imprime tabla per-intent; JSON buckets key | E |

Tests existentes a REVISAR (no modificar):
- `tests/test_classify_intent.py` — adaptive no cambia clasificación. Agregar ~10 casos ON/OFF guard
- `tests/test_eval_bootstrap.py`, `test_eval_latency.py` — pasar sin `--buckets`
- `tests/test_retrieve_source_filter.py`, `test_tasks_intent.py`, `test_timing_breakdown.py` — intactos

---

## 8. Métricas esperadas

### Latencia (post fase E default-on)

| Bucket (% tráfico est.) | P50 baseline | P50 adaptive | P95 baseline | P95 adaptive |
|---|---|---|---|---|
| count/list/recent/agenda (25%) | 150ms (no-hist) / 1500ms (hist) | 100ms (any) | 200/2500 | 200 |
| semantic fast-path (15%) | 5500ms | ~1800ms | 9000ms | ~3000ms |
| semantic default (40%) | 5500ms | 5500ms | 9000ms | 9000ms |
| comparison (8%) | 6500ms | 8000ms (−20% prefill extra) | 11000 | 12000 |
| synthesis (12%) | 7500ms | 10500ms | 14000 | 15500 |

Agregado: **−50% P50 en ~40% de queries** (metadata + fast-path), sin regresión en semantic default, +30-40% en synthesis/comparison (aceptable por calidad).

### Calidad

- **Singles hit@5**: estable 71-72% agregado (CI 68-75%). Gate: ningún bucket −1pp vs baseline
- **Synthesis/Comparison chain_success**: +5-10pp por pool=30 + num_ctx=8192 + siempre-graph. Decomposition ([`improvement-3-decomposition-plan.md`](./improvement-3-decomposition-plan.md)) multiplica esta ganancia
- **Fast-path hit@5**: debe SUBIR (no bajar) respecto a semantic baseline porque threshold 0.6 selecciona donde retriever ya es confiable

### Costo

- qwen2.5:3b ya siempre cargado (`OLLAMA_KEEP_ALIVE=-1`). Fast-path **no agrega** load nuevo
- num_ctx=8192 agrega ~1 GB KV cache efímero por synthesis/comparison query. Stack actual: ~8 GB + 1 = 9 GB, sobra en 36 GB
- Prefix-cache: synthesis/comparison cold reinit primera vez ~4s; warm 20ms después

---

## 9. Riesgos + rollback

### Riesgo 1 — classify_intent mal clasifica → pipeline wrong

Ejemplo: lookup mal clasificado como `count` → handler devuelve lista vacía.

**Mitigación**:
- Regexes `_INTENT_*_RE` ya con 49+ casos en `test_classify_intent.py`
- **Shadow mode** (D.5): log `intent` a `rag_queries` durante 1 semana con `RAG_ADAPTIVE_ROUTING=0`. Comparar fast-path decisions vs 👍/👎 del user

### Riesgo 2 — qwen2.5:3b fast-path produce respuesta pobre

- SYSTEM_RULES_LOOKUP prompt terse 1-2 oraciones. Helper ya usa este patrón
- `_NAME_PRESERVATION_RULE` aplica
- Shadow mode + behavior-event: si 👎 3× más frecuente sobre fast-path, desactivar

### Riesgo 3 — num_ctx=8192 memory pressure

qwen2.5:7b num_ctx=4096 ≈ 4.7 GB; num_ctx=8192 + ~1 GB. 3 sessions concurrentes synthesis podrían picar 11 GB.

**Mitigación**:
- `start_memory_pressure_watchdog()` + 85% threshold ya existe (17 test cases)
- `RAG_SYNTHESIS_NUM_CTX` permite bajar a 6144 sin recompilar

### Riesgo 4 — caller override no honrado

`web/server.py:3953` pasa `rerank_pool=5` explícito. Si adaptive decide pool=30 por intent=synthesis y pisa → web rompe.

**Mitigación**: invariante **caller override SIEMPRE gana**:

```python
if rerank_pool is not None:
    _effective_pool = rerank_pool                       # caller wins
elif _adaptive_routing() and intent in _RERANK_POOL_BY_INTENT:
    _effective_pool = _RERANK_POOL_BY_INTENT[intent]
else:
    _effective_pool = RERANK_POOL_MAX
```

Test incluye `rerank_pool=5 + intent="synthesis"` → asertar pool efectivo = 5.

### Rollback

- **Runtime inmediato**: `RAG_ADAPTIVE_ROUTING=0` → bit-identical al pipeline actual
- **Rollback parcial (mantener fase B)**: fase B es refactor estructural, no depende del flag; equivalencia verificada por test
- **Emergency**: `RAG_FORCE_FULL_PIPELINE=1` corre todo aun con adaptive ON
- **Git-level**: commits separados por fase → `git revert` granular

---

## 10. Out of scope

- **Sub-query decomposition** para synthesis/comparison: vive en [`improvement-3-decomposition-plan.md`](./improvement-3-decomposition-plan.md) (doc paralelo). Complementa pool=30 + num_ctx=8192 agregando fase de decomposición LLM previa al retrieve. Composable.
- **Tuning `_LOOKUP_THRESHOLD`**: valor inicial; derivar empíricamente de `rag_queries.extra_json.fast_path_fired` + 👍/👎 acumulados (ROC curve). No bloquea implementación.
- **Caching retrieve entre pool=5 y pool=15**: si doble-pasada de C.2 se materializa, evaluar amortización vía SQL embed cache + BM25 corpus cache.
- **NLI grounding** ([`improvement-1-nli-integration-plan.md`](./improvement-1-nli-integration-plan.md)): independiente. Post-generation, corre en fast-path o default igual.
