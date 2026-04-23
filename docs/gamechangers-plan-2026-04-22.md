# 3 Game-changers — plan de ejecución

Fecha: 2026-04-22
Evidencia del análisis: `docs/` (este doc) + telemetría real de `telemetry.db`
Orden: #3 → #1 → #2 (de menos a más riesgoso, de self-contained a cross-cutting).

## Invariantes (no romper)

- `rag eval` gate: singles ≥60%, chains ≥73% (auto-rollback). **Nota
  2026-04-23**: estos floors fueron recalibrados desde los originales
  0.7619 / 0.6364 tras la expansión de `queries.yaml` (42→60 singles +
  +7 cross-source + +5 calendar goldens el 2026-04-21). Con el golden
  set actual el baseline estable es singles 71.67% [60.00, 83.33] /
  chains 86.67% [73.33, 96.67] — los floors matchean los CI lower
  bounds del baseline nuevo. Ver el bloque de comentarios sobre
  `GATE_SINGLES_HIT5_MIN` en `rag.py` para la timeline completa.
- 2,247 tests existentes en verde.
- `ranker.json` retrocompatible (no cambios de schema; solo pesos).
- SQL-only writers (post-T10) — sin reintroducir JSONL.
- Auto-commit + push por feature cuando los tests + eval pasan.

---

## Game-changer #3 — Streaming + modelo chico para post-processing

**Why**: Citation-repair + critique usan `resolve_chat_model()` (qwen2.5:7b / command-r) non-streaming. En ~15% de queries el usuario ve stream terminar + 5-13s de spinner antes del re-render. La reparación es una re-escritura estructurada (re-citar paths existentes); qwen2.5:3b alcanza y con stream el feedback es visible en tiempo real.

**Cambios**:
1. `_repair_with_model()` helper nuevo en `rag.py`: encapsula el repair call — usa qwen2.5:3b con `HELPER_OPTIONS` (temp=0, seed=42) + `stream=True` + Live() re-render progresivo.
2. `_critique_with_model()` helper nuevo: mismo patrón.
3. `query()` (rag.py:16702-16793) y `chat()` (rag.py:17880-17931) invocan los helpers.
4. `ThreadPoolExecutor` cuando `--critique` está activo: repair + critique en paralelo sobre `full` original; si ambos cambian, preferir repair (conserva prosa, solo toca citations).
5. Bajar default `_CITATION_REPAIR_MAX_BAD` de 3 a 2. Override env sigue funcionando.

**Tests**:
- `tests/test_citation_repair_streaming.py`: 10 casos — helper streamea, modelo = qwen2.5:3b, Live() se actualiza, fallback a original si repair da empty, telemetría `citation_repaired=True` correcta.
- `tests/test_critique_small_model.py`: 6 casos — critique usa HELPER_OPTIONS, whitespace normalization sigue funcionando, `critique_changed` correcto.
- `tests/test_post_processing_parallel.py`: 5 casos — repair + critique paralelos, merge priority, exceptions en uno no tumban el otro.

**Eval gate**: `rag eval` post-cambio, hit@5 + MRR within CI del baseline previo.

**Rollback**: env `RAG_POSTPROCESS_LEGACY=1` fuerza el modelo grande + non-streaming.

---

## Game-changer #1 — Cache semántico + typo normalization

**Why**: 14 queries con ≥10 repeticiones cada una en `rag_queries` (ej. "mis proyectos actuales" 29x, "llueve hoy?" 18x). ~20 min/día de latencia desperdiciada. Además typos sobre el mismo concepto ("cycle"/"clycle"/"diclo") dan scores 0.0-0.95 porque bge-m3 es char-sensitive.

**Cambios**:
1. Nueva tabla `rag_response_cache`: `(id, ts, q_embedding BLOB, question, response_json, corpus_hash, intent, ttl_seconds, hit_count)`. DDL idempotente en `_ensure_telemetry_tables`.
2. `semantic_cache_lookup(q_embedding, now)` en `rag.py`: SELECT entries dentro del corpus_hash actual + ts + ttl_seconds válidos; cosine >0.95 contra embeddings cargados en memoria (LRU 1000). Return dict o None.
3. `semantic_cache_store(q_embedding, question, response, metas, scores, intent, corpus_hash)`: INSERT con TTL dinámico por intent.
4. Hook en `query()` y `chat()` al principio, después del embed: si hit → render cached + return. `extra_json` loggea `cache_hit=True`.
5. `corpus_hash()`: sha256(col.count() + top-10 file mtimes). Invalida cuando vault cambia.
6. Typo normalization (`maybe_normalize_query`): solo si first-pass top_score <0.1. Usa `_corpus_cache.vocab` + Levenshtein threshold ≤2 sobre palabras >4 chars. Retorna versión corregida O None.
7. Re-retrieve con query normalizado; si mejora top_score → usar la versión corregida.
8. CLI flags: `--no-cache` para debugging, `rag cache clear` para invalidar manualmente.

**Tests**:
- `tests/test_semantic_cache.py`: 15 casos — lookup/store/invalidation por corpus_hash, TTL por intent, cosine threshold, LRU eviction, concurrent access.
- `tests/test_typo_normalization.py`: 12 casos — corrige typos reales ("clycle"→"cycle"), no corrige nombres propios ("Bizarrap" stays), solo dispara si score <0.1, vocab-bounded.

**Eval gate**: `rag eval` sin regresión (cache OFF en eval, opt-in check sí).

---

## Game-changer #2 — Fine-tune del reranker + telemetría honesta

**Why**: 5 de 13 pesos del ranker en 0.0 (behavior priors dormidos). Los 2 pesos dominantes (`click_prior`, `click_prior_hour`) se entrenaron con 20 clicks reales → overfit. bge-reranker-v2-m3 nunca vio datos del dominio del usuario (coaching, dev cycles, personas específicas). 65 feedback pairs son suficientes para un LoRA fine-tune que personalice.

**Cambios** (en orden de implementación):

### 2.A Telemetría honesta (pre-requisito)
1. Loguear `intent` en `extra_json` de `rag_queries` al final de `query()`/`chat()`. Hoy 486/500 últimas tienen `intent=NULL`.
2. Loguear `query` en `rag_behavior` events (hoy solo tienen path).
3. Cache hit rates en `extra_json`: `embed_cache_hit`, `corpus_cache_hit`, `feedback_golden_hit`.
4. `rag insights --telemetry-health`: command nuevo que reporta % de queries con intent populado, CTR por source, gaps de feedback.

### 2.B A/B de context_summary (decidir si vale)
1. Reindex completo con `OBSIDIAN_RAG_SKIP_CONTEXT_SUMMARY=1`.
2. `rag eval` 3 veces en cada arm (con/sin summary). Promedio + CI.
3. Si delta singles hit@5 <0.5pp → **remover** `get_context_summary()`. Ahorra ~11 min/reindex + 1-3s/query.
4. Si delta ≥1pp → mantener + documentar el número real en CLAUDE.md.

### 2.C Fine-tune del reranker
1. `scripts/finetune_reranker.py` nuevo. Inputs:
   - Positive pairs: `rag_feedback` con `rating=1` + `corrective_path` events.
   - Hard negatives: top-10 rerank candidates del retrieve() sobre cada query que NO son el positivo, cosine 0.5-0.85 al pos (semánticamente cercanos pero incorrectos).
   - Held-out: 20% stratified por intent/source.
2. [sentence-transformers CrossEncoder training](https://www.sbert.net/docs/cross_encoder/training/overview.html) con LoRA (r=8, alpha=16) sobre bge-reranker-v2-m3. 3 epochs, lr=2e-5, batch=8 en MPS fp32.
3. Output: `~/.cache/obsidian-rag/reranker-ft-{ts}/` + symlink `~/.cache/obsidian-rag/reranker-ft-current`.
4. `rag.py` carga el reranker fine-tuned si existe el symlink + `RAG_RERANKER_FT=1` (default OFF durante rollout).
5. Gate: `rag eval` post-fine-tune ≥ baseline (hit@5 + MRR). Si no, el script no promueve el symlink.
6. Métricas: hit@5 + MRR sobre held-out + queries.yaml, antes/después.

**Tests**:
- `tests/test_telemetry_intent_logged.py`: 5 casos — query/chat populan intent en extra_json.
- `tests/test_rag_behavior_query_field.py`: 4 casos — behavior events incluyen query field.
- `tests/test_finetune_reranker_gate.py`: 6 casos — gate rechaza fine-tunes que bajan eval, symlink solo se promueve si ≥baseline.

**Rollback**: el reranker fine-tuned es opt-in (RAG_RERANKER_FT=1). Borrar symlink vuelve al base.

---

## Ejecución

Cada game-changer tiene su commit + push propio. Si `rag eval` regresa → rollback + retry.
