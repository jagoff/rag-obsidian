# Project Status Review — obsidian-rag (2026-05-15)

## 🟢 Salud General: ESTABLE

El proyecto está en **buen estado operativo** pero con varios bugs documentados pendientes de resolver. Últimas 6 commits enfocadas en **refactoring de excepciones específicas** y **optimizaciones de performance**.

---

## 📊 Snapshot Actual

| Métrica | Valor |
|---|---|
| **Branch** | `kimi-new-features` (no en master) |
| **Test Suite** | 501 test files, 8,103 tests |
| **Codebase** | ~80k LOC (rag/__init__.py: 53.8k, web/server.py: 26k) |
| **Telemetry Tables** | 55 (comprehensive tracking) |
| **Ingesters** | 11 (+ WhatsApp sub-package con 12 módulos) |
| **Daemons** | 8+ servicios (watch, web, morning brief, digest, wa-scheduled-send, etc.) |
| **MLX Models** | 4 defaults (Qwen2.5-7B chat, Qwen2.5-3B helper, Qwen3-Embedding-0.6B, Whisper-small) |

---

## 🏗️ Arquitectura

### Core Components

1. **RAG Engine** (`rag/` package)
   - Multi-stage retrieval: query decomposition → hybrid search (BM25 + vector) → MLX reranking → LLM inference
   - 11 ingesters (Gmail, Calendar, Drive, WhatsApp, Reminders, Spotify, etc.)
   - **MLX-first design**: todos los inferences (LLM, embeddings, STT, VLM, NLI) corren localmente en Apple Silicon
   - Telemetry stack: 55-table SQLite database tracking every query/interaction for closed-loop learning

2. **Web Layer** (`web/server.py`)
   - FastAPI daemon con PWA dashboards (Chat UI, Mission Control, WhatsApp bridge)
   - Real-time SSE streaming para chat responses
   - Tool-calling integration con MLX backend

3. **Storage**
   - `ragvec.db`: sqlite-vec para semantic search (high-dimensional vectors)
   - `state.db` + `telemetry.db`: 55-table telemetry schema para learning loops

---

## 🐛 Bugs Abiertos (255 TODOs/FIXMEs encontrados)

### ✅ Críticos FIXED (Documentados en ADRs)

| ADR | Fecha | Problema | Solución |
|---|---|---|---|
| **001** | 2026-04-26 | Loky semaphore leak en daemons | Pre-set tqdm lock a threading.RLock antes de heavy imports |
| **002** | 2026-05-08 | MLX-LM race condition (non-main thread) | Force mlx_lm import en main thread antes de prewarm threads |
| **003** | 2026-04-20 | HuggingFace offline + fastembed cache GC | Set HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, FASTEMBED_CACHE_PATH antes de imports |

### 🔴 Abiertos (Audit 2026-04-26)

| Bug | Ubicación | Impacto | Estado |
|---|---|---|---|
| **#31** | web/server.py:14373 | top_score=0.0 dispara fallback cluster | Documentado |
| **#33** | web/server.py:7088 | Rate limit per-IP NO implementado | **BLOCKER CRÍTICO** |
| **#6** | web/server.py:10657 | TOCTOU race en web thread writes | Documentado |
| **#9** | web/server.py:22914 | Null byte (\\x00) en telemetry | Documentado |
| **#5** | web/server.py:24065 | Slot cap faltante en cache | Documentado |
| **#1** | web/server.py:24103 | Event loop bloqueado pre-fix | Documentado |
| **#16** | rag/__init__.py:910 | 10K item cap + drops counter | Documentado |
| **#18** | rag/__init__.py:4288 | cache_key_sql invalidation | Documentado |

### Wave-8 Gotchas (2026-04-28)

1. **Filtros sin wirear**
   - Síntoma: `_strip_foreign_scripts` existía con docstring pero nunca se llamaba → CJK leak en weather
   - Solución: `test_filter_wiring.py` valida que todos los `_*Filter` tengan call sites
   - Prevención: cuando agregás filtro, también editá `_emit()` helper + pipeline de cache replay

2. **Carry-over sobrescrito**
   - Síntoma: pre-router computa `_forced_tool_pairs` pero downstream lo descarta
   - Caso real: pre-router seteaba `_forced_tool_pairs = [('weather', {'location': 'Barcelona'})]` pero línea 10996 hacía `_forced_tools = [] if _propose_intent else _detect_tool_intent(question)` → retornaba `[]`
   - Solución: Regla: pre-router corre UNA vez al inicio de `gen()`, todo el resto del flow LEE de `_forced_tool_pairs`

3. **Cache no invalidado**
   - Síntoma: arreglaste filtro/system prompt/regex, validás Playwright, test reporta bug sigue
   - Causa: semantic cache sirve respuestas pre-fix porque cache key no incluye nada que tu fix haya cambiado
   - Solución: Bumpear `_FILTER_VERSION` invalida TODAS las entries pre-fix
   - Naming: `wave<N>-<YYYY-MM-DD>` ej. `wave8-2026-04-28`

---

## 🔧 Reranker Backend — Decisión Documentada

**Default**: PyTorch+MPS+fp32 (NO MLX)

**Razón**: 2 A/Bs fallidos
- Collapse 2026-04-13 con fp16
- 2x overhead 2026-04-22 con calidad equivalente

**MLX reranker existe** pero no es default:
- Activable via `RAG_RERANKER_BACKEND=mlx`
- Documentado en `rag/mlx_reranker.py:285-292`
- Invariante: `device="mps"+float32` forced

**Próximo paso**: Diseñar nuevo A/B test con mejor métrica antes de hacer MLX default

---

## 🔄 Learning Loops (Cerrados)

### Mecanismos Activos

1. **Telemetry-driven learning**
   - `rag_queries`: Log de todas las queries (3 meses si corre `rag maintenance` semanal)
   - `rag_behavior`: Opens/clicks/copies de la UI web
   - `rag_feedback`: 👍/👎 del usuario

2. **Confidence gates**
   - `CONFIDENCE_RERANK_MIN`: Threshold mínimo para responder
   - `CONFIDENCE_DEEP_THRESHOLD`: Trigger para búsqueda profunda

3. **Fine-tuning gate (GC#2.C)**
   - Requiere 20+ corrective feedback entries para disparar reranker fine-tune

4. **Semantic cache**
   - `rag_response_cache` con invalidación basada en `_FILTER_VERSION`

5. **Anticipatory agent**
   - Briefs diarios, streak-break signals, mood drift detection

### Loops Implícitos

- Implicit learning from user behavior (copy/click patterns)
- Mood tracking con agregación daily/weekly
- Cross-source pattern detection (Calendar + WhatsApp + Reminders)
- Contradiction detection y penalty scoring

---

## 📈 Trabajo Reciente (Últimos 20 commits)

### Refactoring & Stabilization

- ✅ Exception handling specificity (debecf1, 61073ad, 6b9c0d0)
  - Replaced broad `except Exception` with specific exception types
  - Added missing sqlite3 import for except clauses

- ✅ Settings extraction (94b01d9)
  - Centralized environment variables into `rag/settings.py` dataclass

- ✅ Cache refactoring (d37c3fd)
  - Extracted `ThreadSafeCache`/`ThreadSafeCacheMultiKey` to `rag/cache.py`
  - Reused in `web/server.py`

### Performance Audit (2a95016, 2026-05-14)

- ✅ Feedback window optimization
- ✅ LRU cache para contactos
- ✅ WAL connection persistence
- ✅ Voice threadpool improvements

### Feature Additions

- ✅ Finances ingester (.xls support via xlrd + HTML fallback)
- ✅ Home v2 dashboard (screen captures, Apple Health, Spotify hero block)
- ✅ sqlite-vec-gui Streamlit integration

---

## ⚠️ Blockers & Gaps

| Blocker | Impacto | Prioridad | Acción |
|---|---|---|---|
| **Rate limiting (BUG #33)** | Sin límite per-IP en web | 🔴 CRÍTICO | Implementar en web/server.py |
| **Reranker A/B testing** | MLX reranker no es default (2 A/Bs fallidos) | 🟡 ALTO | Diseñar nuevo A/B con mejor métrica |
| **Filter wiring CI** | Riesgo de filtros sin wirear | 🟡 ALTO | Ejecutar test_filter_wiring.py en CI |
| **Memory watchdog tuning** | Necesita ajuste por modelo Mac | 🟢 MEDIO | Perfilar en diferentes Macs |
| **Telemetry backup** | No hay backup automático | 🟢 BAJO | Documentar rsync strategy |

---

## 🎯 Recomendaciones para Próxima Fase

### Corto Plazo (Esta semana)

1. ✅ **Stabilization**: Correr full test suite, validar no hay regressions
2. 🔴 **BUG #33**: Implementar rate limit per-IP en `web/server.py`
3. 🟡 **Filter audit**: Ejecutar `test_filter_wiring.py` en CI

### Mediano Plazo (2-3 semanas)

1. **Performance profiling**: Usar `/profile` skill en top 5 slow paths
2. **Reranker A/B**: Diseñar nuevo A/B test para MLX reranker
3. **Documentation**: Crear docs similares a `wave-8-gotchas.md` para otros subsistemas

### Largo Plazo

1. **MLX reranker default**: Cuando A/B test gane
2. **Telemetry backup**: Documentar rsync strategy
3. **Memory watchdog**: Tuning per Mac model

---

## 📚 Documentación Clave

| Doc | Propósito |
|---|---|
| [`CLAUDE.md`](../CLAUDE.md) | Guía completa para Claude Code (MLX-first, agent dispatch, auto-pull/push) |
| [`docs/mlx-migration.md`](./mlx-migration.md) | Migración MLX completada (10 olas, 2026-04 → 2026-05-07) |
| [`docs/retrieval-internals.md`](./retrieval-internals.md) | Pipeline detallado + scoring + intents |
| [`docs/telemetry-stack.md`](./telemetry-stack.md) | 55-table schema + invariantes audit |
| [`docs/wave-8-gotchas.md`](./wave-8-gotchas.md) | **Excelente**: gotchas de filter wiring + carry-over |
| [`docs/recovery.md`](./recovery.md) | Runbooks para 7 modos de falla |
| [`docs/problemas-comunes.md`](./problemas-comunes.md) | Troubleshooting por síntoma |
| [`docs/adr/`](./adr/) | Architecture Decision Records (3 ADRs documentados) |

---

## 📝 Notas de Implementación

### MLX-First Invariant

**Todo el sistema es MLX-first.** Para inferencia / embedding / STT / VLM / NLI / reranking / tool-calling, la opción MLX-nativa es default y requisito.

Alternativas non-MLX (PyTorch / `sentence-transformers` / `faster-whisper` / Ollama / CrossEncoder) solo se aceptan como:

1. **Rollback path explícito** detrás de env var (ej. `RAG_EMBED_BACKEND=pytorch`, `RAG_NLI_BACKEND=mdeberta`) cuando MLX tiene bug abierto reproducible.
2. **Path opt-in NO-MLX-compat por design** (ej. `gliner` NER → CPU only, gated por `RAG_EXTRACT_ENTITIES`).
3. **Dependency externa de un MCP / integración no-RAG** que el user usa por separado.

**Antes de agregar dep nueva**: ¿Hay versión `mlx-community/...`? ¿Hay equivalente MLX de la librería? Si NO hay MLX viable → flaggearlo en el plan, proponer fallback con env-var de rollback al path MLX.

### Auto-Pull + Commit + Push Rule

Cuando termino algo: `git pull → git commit → git push origin master`. Sin preguntar. Mensaje completo en español rioplatense (qué cambié, por qué, cómo medí si aplica, cómo revertir si rompe).

**Gotcha**: commits locales en `master` se pushean solos por otra sesión paralela (claude-peers MCP). Para experimentar sin pushear → branch dedicada (`git checkout -b experimental/<slug>`).

---

**Revisión completada**: 2026-05-15 15:48 -03:00
