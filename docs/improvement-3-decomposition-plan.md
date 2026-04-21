# Improvement #3 — Decomposition plan

## Executive summary

Agregar **sub-query decomposition proactiva** al pipeline de [`retrieve()`](../rag.py#L12250) para queries tipo `synthesis` y `comparison`. Hoy esos dos intents disparan un solo retrieve + un system prompt que fuerza estructura ([`SYSTEM_RULES_SYNTHESIS`](../rag.py#L9938) / [`SYSTEM_RULES_COMPARISON`](../rag.py#L9952)), pero la evidencia sigue siendo un único pool centrado alrededor de los términos literales de la pregunta. El resultado, medido en los post-T10 baselines, es `chain_success 54.55-63.64%` — el LLM tiene que hacer síntesis sobre contexto shallow.

Paper base: [self-ask (Press et al., 2022)](https://arxiv.org/abs/2210.03350) + [Plan-and-Solve (Wang et al., 2023)](https://arxiv.org/abs/2305.04091). Elegimos la variante **plan-first** (un helper call genera todas las sub-questions en JSON, N retrieves corren en paralelo) por latencia — self-ask es secuencial (sub-question N depende de la respuesta a N-1) y acumula 2-3× la latencia en un stack local.

La interacción con el auto-deep actual ([`CONFIDENCE_DEEP_THRESHOLD = 0.10`](../rag.py#L1833), [`deep_retrieve()`](../rag.py#L12866)) es **complementaria**: decomposition es *proactiva* (cobertura antes de saber si falta), deep_retrieve es *reactiva* (iteración cuando el rerank confidence quedó bajo). Se quedan ambas, con decomposition corriendo primero.

---

## 1. Trigger

### Recomendación: **(A) Solo en `intent ∈ {synthesis, comparison}`**, detrás de dos flags env.

```python
_DECOMPOSITION_ENABLED = os.environ.get("RAG_DECOMPOSITION", "0") == "1"
_ADAPTIVE_ROUTING = os.environ.get("RAG_ADAPTIVE_ROUTING", "0") == "1"

if (_ADAPTIVE_ROUTING and _DECOMPOSITION_ENABLED
        and intent in ("synthesis", "comparison")):
    result = decompose_and_retrieve(col, question, intent, ...)
else:
    result = retrieve(col, question, ...)
```

### Por qué (A) y no (B) ni (C)

- **(B) threshold de complejidad (>8 tokens)** suena razonable pero mete decomposition en queries semantic legítimas ("qué equipo usa adam jones en el último disco") donde el pipeline actual ya resuelve fine. Además, token-count como proxy de "complejidad" es ruidoso — `diferencia entre X y Y` tiene 5 tokens pero sí necesita decomposition, y `cómo configuré el launchd plist del web server en mi macbook` tiene 12 tokens y NO la necesita. Los intents existentes ya son un clasificador regex-based (99%+ precisión per los 49 tests de [`tests/test_classify_intent.py`](../tests/test_classify_intent.py)) afinado exactamente a las formas lingüísticas que necesitan cross-reference.
- **(C) flag `--decompose` on-demand** desperdicia calidad — los usuarios que más lo necesitan no saben a priori que querrían el flag.
- **(A) via regex intent es el menor trade-off**: solo paga los +1-3s de decomposition cuando la intent explícitamente detectó un cue de aggregation/contraste. Semantic queries siguen con pipeline actual (overhead cero); si la regex falla abierta, el auto-deep actual sigue actuando como red de seguridad.

### Costo agregado

| Componente | Costo |
|---|---|
| Plan-first helper call (qwen2.5:3b, JSON output) | ~0.4-0.8s |
| N × retrieve (paralelo, N ≤ 3) | ~0.8-1.5s extra vs 1 retrieve |
| Re-rerank sobre pool unido (query original) | ~0.2-0.4s extra (pool hasta 3× más grande) |
| **Total overhead esperado** | **~1.5-2.7s por query** |

Baseline synthesis queries hoy ~6s end-to-end. Post-decomposition proyección ~8-9s con quality uplift justificable. Sin env flag, path viejo intacto.

### Dos env flags, no uno

`RAG_ADAPTIVE_ROUTING=1` habilita la **familia** de features adaptativos (futuro: HyDE condicional, contrast-amplification, query rewriting intent-aware). `RAG_DECOMPOSITION=1` habilita *esta* feature específica. Permite rollout gradual — si decomposition regresa chain_success, rollback es `export RAG_DECOMPOSITION=0` sin apagar el resto del paraguas.

---

## 2. Decomposition algorithm

### Plan-first (NO self-ask)

Un helper call a [`qwen2.5:3b`](../rag.py#L181) genera TODAS las sub-questions en un JSON en la primera pasada. Paralelizamos los N retrieves via `ThreadPoolExecutor(max_workers=N)` — cada sub-retrieve es I/O-bound (sqlite-vec + ollama embed) así que el GIL no estorba.

### Por qué plan-first gana a self-ask

Self-ask (sub-question N depende de respuesta a N-1) requiere N helper calls secuenciales + N retrieves intercalados → latencia ~N×1s helper + N×1s retrieve = 4-6s para N=3. Plan-first: 1 helper (0.6s) + max(retrieves) paralelo (1.5s) = ~2.1s total. Para queries tipo "diferencia entre X y Y" donde las sub-questions son ortogonales, self-ask NO agrega valor — solo paga latencia.

### Pseudocódigo

```python
# rag.py, junto a deep_retrieve() / _judge_sufficiency (~line 12832)

_DECOMPOSE_MAX_SUBQ = int(os.environ.get("RAG_DECOMPOSITION_MAX_SUBQ", "3"))

def _plan_sub_questions(question: str, intent: str) -> list[str]:
    """Genera hasta N sub-questions ortogonales vía qwen2.5:3b + JSON output.
    Retorna [question] como fallback si parsing falla o helper crashea.

    NO es expand_queries() — aquí las sub-questions descomponen la intent
    en preguntas que cubren territorios distintos, no paraphrases del mismo.
    """
    prompt = _DECOMPOSE_PROMPT_COMPARISON if intent == "comparison" else _DECOMPOSE_PROMPT_SYNTHESIS
    try:
        resp = _helper_client().chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt.format(query=question)}],
            options={**HELPER_OPTIONS, "num_predict": 200, "format": "json"},
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        payload = json.loads(resp.message.content)
        subs = payload.get("sub_questions", [])
        if not isinstance(subs, list) or not subs:
            return [question]
        out: list[str] = []
        seen_lower: set[str] = set()
        for s in subs:
            if not isinstance(s, str):
                continue
            s = s.strip()
            if len(s) < 4 or s.lower() in seen_lower:
                continue
            seen_lower.add(s.lower())
            out.append(s)
            if len(out) >= _DECOMPOSE_MAX_SUBQ:
                break
        return out or [question]
    except (json.JSONDecodeError, Exception):
        return [question]  # fail-safe


def decompose_and_retrieve(col, question, intent, k, folder, ...) -> dict:
    """Plan-first + parallel retrieve + rerank con QUERY ORIGINAL sobre pool unido."""
    sub_questions = _plan_sub_questions(question, intent)
    if len(sub_questions) <= 1:
        return retrieve(col, question, k, folder, ...)  # no ganancia

    # N × retrieve en paralelo (I/O-bound)
    with ThreadPoolExecutor(max_workers=len(sub_questions)) as ex:
        futures = [
            ex.submit(retrieve, col, sq, k, folder, ...,
                      multi_query=False)  # sub-question ya focused
            for sq in sub_questions
        ]
        sub_results = [f.result() for f in futures]

    # Union + dedup por (file, doc[:50])
    all_docs, all_metas = [], []
    seen = set()
    for r in sub_results:
        for d, m in zip(r["docs"], r["metas"]):
            key = (m.get("file", "") + "::" + d[:50])
            if key not in seen:
                seen.add(key)
                all_docs.append(d)
                all_metas.append(m)

    # Re-rerank con QUERY ORIGINAL sobre pool unido (cap 2×)
    rerank_cap = min((rerank_pool or RERANK_POOL_MAX) * 2, len(all_docs))
    scores = rerank(question, all_docs[:rerank_cap])
    top = sorted(zip(all_docs, all_metas, scores), key=lambda t: t[2], reverse=True)[:k]

    return {
        "docs": [d for d, _, _ in top],
        "metas": [m for _, m, _ in top],
        "scores": [s for _, _, s in top],
        "confidence": top[0][2] if top else float("-inf"),
        "query_variants": sub_questions,
        "decomposed": True,
        "timing": {"decomposition_plan_ms": ..., "parallel_retrieve_ms": ...},
    }
```

### Prompt templates

```python
_DECOMPOSE_PROMPT_COMPARISON = """Descomponé esta consulta comparativa en sub-preguntas para buscar por separado.

Regla clave: las dos entidades comparadas NECESITAN búsquedas independientes —
si buscás "diferencia entre A y B" directamente, el retrieval sesga a la intersección.
Cada entidad merece su propia búsqueda, + una búsqueda de relación.

Consulta: "{query}"

Devolvé JSON EXACTO:
{{
  "sub_questions": [
    "qué es <entidad A>",
    "qué es <entidad B>",
    "cómo se relacionan <A> y <B>"
  ]
}}

Ejemplos:

Consulta: "diferencia entre ranker-vivo y feedback loop"
{{
  "sub_questions": [
    "qué es ranker-vivo",
    "qué es feedback loop",
    "cómo se relacionan ranker-vivo y feedback loop"
  ]
}}

SIN explicar, SIN markdown, SOLO el JSON.
"""

_DECOMPOSE_PROMPT_SYNTHESIS = """Descomponé esta consulta de síntesis en sub-preguntas ortogonales
(que cubran aspectos DISTINTOS del tópico, no paraphrases).

Consulta: "{query}"

Devolvé JSON EXACTO:
{{
  "sub_questions": [
    "<aspecto 1 del tópico>",
    "<aspecto 2 del tópico>",
    "<aspecto 3 del tópico>"
  ]
}}

Ejemplos:

Consulta: "resumime todo lo que hay sobre obsidian-rag"
{{
  "sub_questions": [
    "arquitectura obsidian-rag",
    "pipelines obsidian-rag",
    "decisiones de diseño obsidian-rag"
  ]
}}

Máximo 3 sub-questions. SIN explicar, SIN markdown, SOLO el JSON.
"""
```

### Validación del output

1. `ollama.chat(format="json")` fuerza JSON válido (qwen2.5 ≥v0.2).
2. `json.loads()` defensivo → fallback a `[question]`.
3. Guard: `len > 4` chars + unique case-insensitive. Caída común de qwen2.5:3b: `["qué es X", "qué es X", "qué es X"]` → dedupe colapsa → fallback.
4. Cap a `_DECOMPOSE_MAX_SUBQ` (default 3, env-tunable).

---

## 3. Gating (max sub-questions)

### `RAG_DECOMPOSITION_MAX_SUBQ=3` (default, env-tunable)

- 2 es mínimo útil
- 3 cubre patrones canónicos (comparison: A + B + relación; synthesis: 3 aspectos)
- 4+ no aporta cobertura marginal proporcional: rerank pool post-union 3×15=45, +150ms rerank + pressure en cross-encoder

### Skip si query < 6 tokens

Queries `"X vs Y"` (3 tokens) tienen cue regex hit pero NO valen decomposition:

```python
if len(question.strip().split()) < 6:
    return retrieve(col, question, k, folder, ...)  # skip
```

Paralelo al [`_EXPAND_MIN_TOKENS = 4`](../rag.py#L8988) gate en `expand_queries()`.

---

## 4. Merge strategy

### Recomendación: **(C) Union + re-rerank con QUERY ORIGINAL sobre pool unido**

### Por qué NO (A) ni (B)

- **(A) Union + todos los chunks al LLM**: rompe [`RERANK_TOP = 5`](../rag.py#L1776) invariant (latencia + coherencia). Con N=3 × top-5 = 15 chunks sin re-rankear, el LLM se diluye.
- **(B) RRF merge**: RRF se calibra asumiendo listas = paraphrases del MISMO intent (scores comparables). Sub-questions ortogonales violan eso: rank-1 de "qué es ranker-vivo" no es equivalente a rank-1 de "qué es feedback loop" para la query original.

### Por qué (C)

Cross-encoder `bge-reranker-v2-m3` hace pairwise scoring `(query, chunk) → relevance`. Corriendo con `query=question` (original) sobre pool unido da ranking único, calibrado, reflejando la intent del usuario.

**Limitación**: pool unido puede ser 3× el normal (45 vs 15). Rerank es O(pool) → ~3× latencia del rerank stage. Bench 2026-04-21: pool=30 P95=4704ms vs pool=15 P95=1577ms. Mitigación: cap `min(rerank_pool * 2, len(pool))`.

### Dedup

Key: `(file_path, doc[:50])` — idéntico a [`deep_retrieve`](../rag.py#L12909). File + primeros 50 chars resuelve 99% de colisiones.

---

## 5. Final prompt al chat model

### Recomendación: **(A) un único prompt con TODA la evidencia + template por intent**

Top-k chunks del pool unido re-rankeado se pasa al LLM con el system prompt existente — [`SYSTEM_RULES_SYNTHESIS`](../rag.py#L9938) / [`SYSTEM_RULES_COMPARISON`](../rag.py#L9952) routed por [`system_prompt_for_intent()`](../rag.py#L9968).

### Por qué NO (B) "LLM call por sub-question + síntesis final"

- **Cost**: 1 chat call (5-6s) → N+1 chat calls (15-18s). 3× latencia.
- **Coherencia**: la estructura forzada de `SYSTEM_RULES_COMPARISON` se rompe cuando las respuestas de sub-questions se pasan al prompt de síntesis — se pierde rutas citables.
- **Citation repair**: asume que citaciones vienen de chunks reales. Con B, citaciones serían a "respuestas de sub-question" → repair marca todo inventado.

### Por qué (A) funciona

- `SYSTEM_RULES_COMPARISON` YA fuerza estructura `[Fuente A] / [Fuente B] / Diferencia`. Con mejor evidencia (pool decompose), más material sobre CADA entidad.
- `SYSTEM_RULES_SYNTHESIS` YA exige cross-reference + tension. Con top-k abarcando N aspectos ortogonales, más overlap.
- Cero cambios en `build_progressive_context`, `_format_chunk_for_llm`, `verify_citations`. Upgrade 100% upstream del LLM call.

### Invariants respetados

- `CHAT_OPTIONS`: `num_ctx=4096, num_predict=384` — intocados
- [`RERANK_TOP = 5`](../rag.py#L1776) final chunks — intocados (solo cambia pool previo)
- Citation-repair, critique, confidence gate, prompt-injection defence, `_NAME_PRESERVATION_RULE` — todos intactos

---

## 6. Integration

### Ubicación

Nueva función `decompose_and_retrieve` en [`rag.py`](../rag.py) junto a [`deep_retrieve()`](../rag.py#L12866). Misma firma shape que `retrieve()` / `deep_retrieve()` (drop-in).

### Call-sites

1. **CLI query path** ([`rag.py:15944-15954`](../rag.py#L15944)):
```python
def _do_retrieve():
    if (_decomposition_enabled()
            and intent in ("synthesis", "comparison")
            and not raw):
        result = decompose_and_retrieve(**_retrieve_kwargs, intent=intent)
    else:
        result = retrieve(**_retrieve_kwargs)
    # Auto-deep reactive: SOLO si decomposition NO corrió
    if (not no_deep and not raw and result["docs"]
            and result["confidence"] < CONFIDENCE_DEEP_THRESHOLD
            and not result.get("decomposed", False)):
        result = deep_retrieve(**_retrieve_kwargs)
    return result
```

2. **CLI chat path** ([`rag.py:17123-17140`](../rag.py#L17123)): análogo con wrapper `multi_retrieve`. Decomposition corre por-vault; multi-vault merge se aplica post-decomposition.

3. **Web `/api/chat` path** ([`rag.py:30662-30671`](../rag.py#L30662)): análogo al CLI query path.

### Helper

```python
def _decomposition_enabled() -> bool:
    return (os.environ.get("RAG_ADAPTIVE_ROUTING", "0") == "1"
            and os.environ.get("RAG_DECOMPOSITION", "0") == "1")
```

### Logging

En `log_query_event` añadir:
- `decomposed: bool`
- `sub_questions: list[str]` (null si no decomposed)
- `decomposition_plan_ms: int`
- `parallel_retrieve_ms: int`

---

## 7. Interacción con auto-deep actual

### Decomposition corre PRIMERO (proactiva). Deep_retrieve queda como red de seguridad.

```python
if intent in ("synthesis", "comparison") and _decomposition_enabled():
    result = decompose_and_retrieve(...)
else:
    result = retrieve(...)

if (not no_deep and result["docs"]
        and result["confidence"] < CONFIDENCE_DEEP_THRESHOLD):
    if result.get("decomposed", False):
        pass  # decomposition ya trajo contexto rico; deep no va a inventar
    else:
        result = deep_retrieve(...)  # reactive fallback for non-decomposed
```

### Por qué NO stackear por defecto

- **Costo**: decompose (1.5-2.7s) + deep (12s) = >20s. Synthesis usuario espera medida, no 30s.
- **Redundancia**: decomposition ya cubre "top-K shallow → buscar más amplio". Si rerank confidence sigue <0.10 post-decomposition, el problema es ausencia real en corpus, no cobertura.

### Opt-in agresivo

`RAG_DECOMPOSITION_STACK_DEEP=1` (default OFF) corre decompose → retrieve → deep_retrieve. Útil en `rag eval` para medir ceiling de cobertura.

### Ablation matrix

| Config | Intent | Flow |
|---|---|---|
| Flags OFF (default) | cualquiera | `retrieve()` → auto-deep si confidence<0.10 |
| `RAG_DECOMPOSITION=1` | semantic/count/list/recent/agenda | `retrieve()` → auto-deep (sin cambio) |
| `RAG_DECOMPOSITION=1` | synthesis/comparison | `decompose_and_retrieve()` → **no deep** |
| `RAG_DECOMPOSITION_STACK_DEEP=1` | synthesis/comparison | `decompose_and_retrieve()` → auto-deep |

---

## 8. Tests

### Unit: `tests/test_decomposition.py` (~20 cases)

Mockean `_helper_client()` y `retrieve()`. Modelo: [`tests/test_deep_retrieve.py`](../tests/test_deep_retrieve.py).

```
# Plan parsing
test_plan_parses_valid_json
test_plan_returns_original_on_empty_list
test_plan_returns_original_on_malformed_json
test_plan_deduplicates_case_insensitive
test_plan_drops_short_sub_questions
test_plan_caps_at_max_subq
test_plan_respects_RAG_DECOMPOSITION_MAX_SUBQ
test_plan_uses_comparison_prompt_for_comparison_intent
test_plan_uses_synthesis_prompt_for_synthesis_intent
test_plan_returns_original_on_helper_exception

# decompose_and_retrieve orchestration
test_decompose_falls_back_when_subq_is_one
test_decompose_runs_N_retrieves_in_parallel
test_decompose_dedups_chunks
test_decompose_reranks_pool_against_original_query
test_decompose_caps_rerank_pool_at_2x
test_decompose_marks_result_with_decomposed_True
test_decompose_passes_through_source_filter
test_decompose_skips_when_query_under_6_tokens
test_decompose_propagates_filters_applied
test_decompose_handles_sub_retrieve_exception_without_crash
```

### Integration: `tests/test_decomposition_integration.py` (~10 cases)

```
test_decomposition_disabled_by_default_envs
test_decomposition_enabled_requires_BOTH_env_flags
test_cli_query_path_invokes_decompose_for_synthesis_intent
test_cli_query_path_invokes_retrieve_for_semantic_intent
test_cli_chat_path_invokes_decompose_multi_vault
test_web_api_chat_invokes_decompose
test_auto_deep_skipped_when_decomposed_True_and_low_confidence
test_auto_deep_runs_when_decomposed_False_and_low_confidence
test_log_query_event_records_sub_questions_and_timing
test_decomposition_stack_deep_env_enables_both
```

### Golden set: [`queries.yaml`](../queries.yaml)

Agregar 5 queries difíciles:

```yaml
- question: "integrame todo sobre ranker-vivo, desde tuning offline hasta feedback loop"
  expected:
    - 01-Projects/RAG-Local/obsidian-rag - ranker-vivo.md
    - 01-Projects/RAG-Local/obsidian-rag - Ranker Tuning.md
    - 01-Projects/RAG-Local/obsidian-rag - Feedback Loop.md
  note: "synthesis 3-aspecto del mismo concepto"

- question: "diferencia entre el flow de rag chat y el de rag do"
  expected:
    - 01-Projects/RAG-Local/obsidian-rag - Chat Flow.md
    - 01-Projects/RAG-Local/obsidian-rag - Agent Loop.md
  note: "comparison: single-pass sesga a una entidad"

- question: "contraste entre indexado incremental y flujo reset"
  expected:
    - 01-Projects/RAG-Local/obsidian-rag - Index Incremental.md
    - 01-Projects/RAG-Local/obsidian-rag - Reset Flow.md
  note: "comparison ambas entidades en top-k crítico"
```

### Bench target: `rag eval --latency`

Comparar:
- Baseline: `RAG_DECOMPOSITION=0`
- Treatment A: `RAG_DECOMPOSITION=1 RAG_DECOMPOSITION_MAX_SUBQ=2`
- Treatment B: `RAG_DECOMPOSITION=1 RAG_DECOMPOSITION_MAX_SUBQ=3`
- Treatment C: `RAG_DECOMPOSITION_STACK_DEEP=1` (ceiling)

Métricas por treatment: `hit@5`, `MRR`, `chain_success`, P50/P95. Decidir default MAX_SUBQ basado en Pareto.

---

## 9. Métricas esperadas

### Baseline (post-T10)

| Métrica | Valor |
|---|---|
| chain_success | 54.55-63.64% |
| comparison MRR | ~0.706 |
| Latencia synthesis retrieve P50 | ~4.5s |
| Latencia synthesis total P50 | ~6s |

### Target post-decomposition

| Métrica | Target | Rationale |
|---|---|---|
| **Synthesis chain_success** | **+10pp** (54% → 64%) | Cross-reference aggregation multi-aspecto |
| **Comparison MRR** | **+5pp** (0.706 → 0.756) | Retrieve separado por entidad |
| **Synthesis hit@5** | **+5pp** | Union + re-rerank amplía pool |
| **Semantic hit@5** | **0** (neutral) | Decomposition NO corre en semantic |
| **Latencia synthesis retrieve P50** | **+1.5-2.7s** (4.5s → 6-7s) | Aceptable dado uplift |
| **Latencia semantic total P50** | **0 delta** | Path intocado |

### Qualitative uplift (ejemplos)

**Query**: `"diferencia entre ranker-vivo y feedback loop"`
- **Baseline**: top-5 todos de `Feedback Loop.md` (embed sesga al término más específico). LLM: "feedback loop hace X, ranker-vivo mencionado once".
- **Post-decomposition**: 3 sub-retrieves → 5 chunks de cada entidad + 5 de relación. Top-5 post-union tiene ≥2 de CADA lado. LLM grounding real en ambas.

**Query**: `"sintetizame todo sobre obsidian-rag"`
- **Baseline**: top-5 todos del index `Obsidian RAG Local.md`. LLM parafrasea el index.
- **Post-decomposition**: 3 aspectos (arquitectura, pipelines, decisiones). Top-5 cubre las 3 áreas. LLM surface tensions.

---

## 10. Riesgos + rollback

### Riesgos

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| **Plan degeneración** (qwen2.5:3b devuelve 3× la misma query) | Media | Dedupe case-insensitive + guard `<4 chars`. Fallback a `[question]` → retrieve() single-pass. Test `test_plan_deduplicates`. |
| **Plan JSON malformed** | Media | `format="json"` ollama option + try/except `json.loads` + fallback |
| **Prompt injection via sub-question** | Baja | `_plan_sub_questions` opera sobre QUERY, no sobre chunks. Chunks nunca entran al decomposer. `_CHUNK_AS_DATA_RULE` sigue aplicando al LLM final. |
| **Latencia P95 >3× baseline** | Baja | Cap MAX_SUBQ=3, paralelismo. Si bench muestra >3× degradación, bajar a MAX_SUBQ=2 via env override |
| **Regresión auto-deep coverage** | Nula | Decomposition NO corre en semantic. Auto-deep intacto en ese path. |
| **Rerank explosion** (pool >45) | Baja | Cap `min(rerank_pool * 2, len(pool))` |
| **N sub-questions idénticas a query** | Baja | Regex guard: drop si coincide con `question.lower()` |
| **Intent miscalificación** | Media (49 tests medidos) | Auto-deep reactivo sigue en semantic intent |
| **Memory pressure** | Baja | `start_memory_pressure_watchdog()` observa 85% threshold |

### Rollback (3 niveles)

1. **Soft** (inmediato): `unset RAG_DECOMPOSITION` en shells + launchd plists. Función sigue existiendo pero nunca se invoca.
2. **Partial** (inmediato): `export RAG_DECOMPOSITION_MAX_SUBQ=2` baja agresividad sin apagar.
3. **Hard**: `git revert` del commit + call-sites. Tests siguen funcionando (mockean retrieve).

### Gate de deployment

NO activar `RAG_DECOMPOSITION=1` en prod hasta:
1. Tests de `test_decomposition.py` + `test_decomposition_integration.py` pasan ≥50 cases.
2. `rag eval --latency` muestra `chain_success` +≥5pp Y P95 synthesis ≤10s.
3. Shadow-mode 24h (CLI ON, web OFF): si ≥20% de plans degeneran, NO avanzar.

### Re-tune post-rollout

`rag tune` fitea weights sobre `retrieve()` single-pass. Post-decomposition, features cambian (pool más grande, distribución rerank scores distinta). Re-correr `rag tune --apply --samples 500` tras activar stable.

---

## Referencias

- Paper: [Self-ask (Press et al., 2022)](https://arxiv.org/abs/2210.03350)
- Paper: [Plan-and-Solve (Wang et al., 2023)](https://arxiv.org/abs/2305.04091)
- Código:
  - [`retrieve()`](../rag.py#L12250)
  - [`deep_retrieve()`](../rag.py#L12866)
  - [`classify_intent()`](../rag.py#L9255)
  - [`_INTENT_SYNTHESIS_RE`](../rag.py#L9240), [`_INTENT_COMPARISON_RE`](../rag.py#L9224)
  - [`SYSTEM_RULES_SYNTHESIS`](../rag.py#L9938), [`SYSTEM_RULES_COMPARISON`](../rag.py#L9952)
  - [`system_prompt_for_intent()`](../rag.py#L9968)
  - [`expand_queries()`](../rag.py#L8974)
  - [`CONFIDENCE_DEEP_THRESHOLD`](../rag.py#L1833), [`RERANK_POOL_MAX`](../rag.py#L1775)
  - [`rrf_merge()`](../rag.py#L8434)
- Tests: [`tests/test_deep_retrieve.py`](../tests/test_deep_retrieve.py), [`tests/test_classify_intent.py`](../tests/test_classify_intent.py)
- Golden: [`queries.yaml`](../queries.yaml) líneas 259-302
- Docs hermanas: [`improvement-1-nli-integration-plan.md`](./improvement-1-nli-integration-plan.md), [`improvement-3-adaptive-routing-design.md`](./improvement-3-adaptive-routing-design.md)
