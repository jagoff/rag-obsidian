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

### 2.C Fine-tune del reranker — ✅ COMPLETADO 2026-04-23

**Estado**: infra LoRA shipped. Loader runtime + script de entrenamiento + tests + docs completos. Pendiente solo correr el training real cuando haya ≥10 pares positivos (gate del script) — ortogonal a la infra que ya quedó verificada.

**Cambios shippeados**:

1. **[`scripts/finetune_reranker.py`](../scripts/finetune_reranker.py)** ahora soporta `--mode {lora,full}` (default `lora`). El modo nuevo:
   - Lee positivos de `rag_feedback` (rating=+1, con `corrective_path` cuando está) **y** de `rag_behavior` (`event='positive_implicit'` con query field).
   - Mina hard negatives de `rag_behavior`: chunks que aparecieron en top-5 (`event='impression'` con rank ≥3) cuando el user clickeó OTRO path en una ventana de ±10 min.
   - Held-out 20% stratified por query (mismo split que el modo full).
   - Entrena LoRA con [peft](https://github.com/huggingface/peft) — `r=8`, `alpha=16`, `dropout=0.1`, target `query`/`value` projections de XLM-RoBERTa. 3 epochs default, lr=2e-5, batch=8 en CPU (override con `RAG_FT_DEVICE=mps`).
   - Output: `~/.local/share/obsidian-rag/reranker_ft/` (PEFT adapter dir, ~5 MB) + `ft_meta.json` con métricas del run.
   - Print metrics: nDCG@5 antes/después en held-out, mean pos/neg score, margin, AUC pair-ranking.
2. **Loader en [`rag/__init__.py`](../rag/__init__.py)** (sección "GC#2.C — LoRA adapter loader"):
   - `RERANKER_FT_ADAPTER_DIR` constante apunta al path canónico.
   - `_reranker_ft_enabled()` parsea `RAG_RERANKER_FT` con la misma tabla de truthy strings que el resto del codebase (`1`/`true`/`yes`/`on`).
   - `_apply_reranker_lora_adapter(model, dir)` aplica el adapter on top del CrossEncoder. NUNCA raisea: dir missing, peft no instalado, adapter_config inválido, splice failure → log a `silent_errors.jsonl` + return False, caller queda en base model.
   - `get_reranker()` invoca `_apply_reranker_lora_adapter` solo cuando la flag está prendida (default OFF, no cambia comportamiento pre-GC#2.C).
3. **Tests**: [`tests/test_finetune_reranker_gate.py`](../tests/test_finetune_reranker_gate.py) — 19 casos (los 6 del plan + 13 de invariantes adicionales del helper). Cubren: env-unset → no peft import, dir vacío → fallback + warning, adapter válido → load OK, scores stay en [0,1], smoke con flag ON sin adapter, helpers de nDCG@5 + AUC, todos los failure modes del `_apply_reranker_lora_adapter`. Mocks pesados — los tests no requieren peft instalado.
4. **Pyproject**: nuevo extra `[finetune]` con `peft + transformers + datasets + accelerate`. Soft dep — el loader y los tests degradan a base+warning si falta. Install: `uv tool install --reinstall --editable '.[finetune]'`.
5. **Env var doc**: `CLAUDE.md` actualizado con `RAG_RERANKER_FT` (default OFF) — distinto de `RAG_RERANKER_FT_PATH` que sigue siendo el switch del modo `full` (GC#2.B).

**Coexistencia con GC#2.B**: ambos pueden estar prendidos simultáneamente. El path resuelto por `_resolve_reranker_model_path()` (base o full-FT-via-symlink) es el modelo sobre el que se aplica el LoRA. Default operacional: GC#2.B desactivado, GC#2.C desactivado → bge-reranker-v2-m3 plano.

**Próximo paso**: cuando `rag feedback status` muestre ≥10 positivos limpios (rating=+1 con o sin `corrective_path`), correr:

```bash
uv tool install --reinstall --editable '.[finetune]'
python scripts/finetune_reranker.py --mode lora --epochs 3 --lr 2e-5
export RAG_RERANKER_FT=1   # opt-in al adapter
```

El script printea las métricas before/after; el operador decide si setear la env var en los plists. Para revertir: `unset RAG_RERANKER_FT` (o borrar la línea del plist) + `launchctl bootout/bootstrap` para que tome efecto.

**Rollback**: el reranker fine-tuned es opt-in (`RAG_RERANKER_FT=1`). `unset RAG_RERANKER_FT` o `rm -rf ~/.local/share/obsidian-rag/reranker_ft/` lo desactiva.

---

## Ejecución

Cada game-changer tiene su commit + push propio. Si `rag eval` regresa → rollback + retry.
