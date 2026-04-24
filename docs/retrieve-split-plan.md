# Plan de split de `retrieve()` — 2026-04-24

**Estado**: DEFERIDO. Plan listo; ejecución requiere sesión dedicada con capacidad
para 2 corridas de `rag eval` (~20 min ollama + potencial memory pressure).

## Motivación

`retrieve()` en [`rag.py:16338`](../rag.py) tiene **882 líneas** y es la función
más crítica del pipeline. Mezcla 4 fases lógicas cuyo cambio individual
es imposible sin navegar todo el bloque:

1. **Input preprocessing**: `reformulate_query`, `infer_filters`, `detect_temporal_intent`.
2. **Embed + retrieval**: per-variant `sqlite-vec` + BM25 + RRF merge, feedback path injection.
3. **Rerank**: cross-encoder scoring con `_effective_pool` cap.
4. **Postprocess**: scoring formula con pesos, graph expansion, `seen_titles` penalty, WA fast-path, confidence gate.

## Shape del split propuesto

Tres helpers privados **dentro del mismo `rag.py`** (NO módulo separado —
evita circular imports y mantiene la public API intacta):

```python
def _retrieve_prepare_query(
    col, question, history, folder, tag, precise, multi_query, auto_filter,
    date_range, summary, variants, source,
) -> tuple[str, list[str], list[list[float]], dict, str | None, str | None, tuple | None]:
    """Returns (search_query, variants, variant_embeds, filters_applied,
    folder, tag, date_range). Encapsula steps 1-3 + embed."""

def _retrieve_candidate_pool(
    col, variants, variant_embeds, where, folder, tag, date_range,
    rerank_pool,
) -> tuple[list[str], list[str], dict]:
    """Returns (seen_ids, merged_ordered, timing). Encapsula la loop
    per-variant sem+BM25+RRF + feedback path injection + pool cap."""

def _retrieve_rerank_and_score(
    col, question, search_query, candidate_ids, seen_titles, source, intent,
) -> tuple[list[str], list[dict], list[float], bool]:
    """Returns (docs, metas, scores, fast_path). Encapsula rerank +
    scoring formula + WA fast-path + confidence gate + graph expansion."""
```

`retrieve()` queda como thin wrapper (~50 líneas):

```python
def retrieve(col, question, k, ...):
    # Empty col short-circuit
    if col.count() == 0:
        return RetrieveResult(docs=[], ...)

    search_query, variants, variant_embeds, filters_applied, \
        folder, tag, date_range = _retrieve_prepare_query(...)

    seen_ids, merged_ordered, timing_pool = _retrieve_candidate_pool(...)
    if not merged_ordered:
        return {...empty with timing...}

    docs, metas, scores, fast_path = _retrieve_rerank_and_score(...)
    return {docs, metas, scores, confidence, search_query, filters_applied,
            query_variants, timing, fast_path}
```

## Invariants a preservar (no-regresión)

1. **Bit-identical output**: mismo dict shape + misma order de `docs`/`metas`/`scores`
   para toda query en `queries.yaml`. Validado con `rag eval` pre/post.

2. **Reranker input shape**: el cross-encoder ve `{title}\n({folder})\n\n{parent_body}`
   — probado `+8pp chains`. El stage 3 DEBE preservar el assembly exacto.

3. **Timing keys preservados**: `embed_ms`, `sem_ms`, `bm25_ms`, `rrf_ms`,
   `rerank_ms`, `total_ms` siguen en el `timing` dict. Los dashboards + alertas
   ya consumen estos keys.

4. **Fast-path marker**: `fast_path: bool` en el return. Web lo usa para downgrade
   a qwen2.5:3b + num_ctx=4096 ([`web/server.py:3648`](../web/server.py)).

5. **Feedback signals + path injection**: orden de fetch relativo al RRF merge
   preservado (path injection POST merge, PRE rerank).

## Procedimiento de ejecución

```bash
# 1. Baseline (NO correr si algún otro proceso está generando carga)
RAG_EXPLORE=0 rag eval 2>&1 | tee /tmp/eval_pre_split.txt

# 2. Implementar split (3 funciones nuevas, retrieve() shrink).
# 3. Pytest completo.
.venv/bin/python -m pytest tests/ -q -x --ignore=tests/test_e2e_live.py

# 4. Eval post-split.
RAG_EXPLORE=0 rag eval 2>&1 | tee /tmp/eval_post_split.txt

# 5. Comparar CIs. Criterio de aceptación:
#    - singles hit@5 >= 71.67% (lower CI del floor actual)
#    - chains hit@5 >= 86.67%
#    - MRR dentro del CI previo
# 6. Si verde: segundo run de eval para doblar confianza.
RAG_EXPLORE=0 rag eval 2>&1 | tee /tmp/eval_post_split_2.txt

# 7. Commit + push solo si AMBOS eval runs pasan.
```

## Riesgos específicos detectados

- **Stage 2 tiene una early-return**: cuando `merged_ordered` está vacío devuelve
  un dict con `confidence=-inf` + `timing` parcial. El split debe preservar esa
  shape — el caller (`query()`, `chat()`) checkea `confidence` explícitamente.

- **`_timing` mutado in-situ**: el dict se acumula across stages. En el split,
  cada helper debe devolver su timing parcial y el wrapper los mergea.

- **`filters_applied` también mutado**: se agrega `wa_skip_paraphrase` flag en
  stage 1 + `since/until` en stage 1. El wrapper recoge el dict completo.

- **Ranker-vivo auto-rollback** (nightly 03:30 via `com.fer.obsidian-rag-online-tune`):
  si el split introduce un cambio sutil que regresiona eval, el gate rollbackeará
  `ranker.json`. Ejecutar el split antes de las 22:00 local deja 5+ horas para
  detectar y revertir manualmente si hace falta.

## Valor esperado post-split

- `retrieve()` pasa de 882 → ~50 líneas (thin wrapper).
- Tres helpers testeables en aislamiento con fixtures mock más pequeños.
- Cambios futuros al rerank (p.ej. extract `_rerank_stage` a una clase con
  cache) localizados al stage 3 sin leer los 800+ líneas de contexto.

## Por qué no ahora

1. **Eval cycles son caros**: ~10 min cada uno; 2 runs + verificación manual = 30+ min de wall time durante el cual el user probablemente tiene ollama ocupado con otras cosas (`com.fer.obsidian-rag-web` + `com.fer.obsidian-rag-serve` pinnean VRAM).
2. **No hay un trigger concreto hoy**: no hay bug en retrieve(), no hay performance regression, no hay feature que necesite los stages separados. Refactor preventivo.
3. **Riesgo asimétrico**: win = "código más limpio"; loss = "ranker auto-rollback + debug sesión completa".

Ejecutar cuando:
- (a) Hay un feature concreto que necesita un stage separado (ej. cache del rerank pool entre queries consecutivas del chat).
- (b) Hay una sesión dedicada con ollama disponible exclusivamente para evals.
- (c) El user acepta el riesgo del ranker-vivo auto-rollback esa noche.
