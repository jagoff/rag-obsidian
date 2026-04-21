# Improvement #2 — Entity extraction design

## 1. Modelo elegido: GLiNER + qwen2.5:3b fallback

### Decisión principal

**[GLiNER multi-v2.1](https://huggingface.co/urchade/gliner_multi-v2.1)** (~200M params, zero-shot multilingüe NER) como extractor primario.

**Razones:**
- **Velocidad**: <50ms por chunk en MPS (vs 800ms–1.5s para qwen2.5:3b con structured output)
- **Zero-shot**: detecta entidades sin entrenamiento previo en el corpus específico
- **Multilingüe**: soporta español + inglés nativamente (importante para corpus con emails en ambos idiomas)
- **Bajo VRAM**: ~400MB residente, compatible con stack actual (qwen2.5:7b + bge-m3 + reranker)
- **Backfill cost**: 8231 chunks × 50ms = ~7 minutos (vs 2.3h con qwen2.5:3b)
- **Incremental cost**: negligible para batches diarios de WhatsApp (20–50 chunks)

### Fallback

Si GLiNER falla a cargar (no instalado, OOM, etc.):
- **qwen2.5:3b con structured output** (patrón `HELPER_OPTIONS` existente)
- Prompt-engineered con `temp=0, seed=42` para determinismo
- Costo: ~1s/chunk, pero solo en error path

### Entidades a extraer

Labels GLiNER:
1. **person** — nombres de personas (Juan, Ana, Fer, etc.)
2. **organization** — empresas, proyectos, equipos (Max ops, RagNet, etc.)
3. **location** — lugares (Buenos Aires, Oficina, etc.)
4. **event** — eventos nombrados (workshop, standup, etc.)
5. **date** — fechas explícitas (20 de marzo, viernes próximo) — *opt-in, ver §8*

**Confianza mínima**: 0.70 (GLiNER devuelve scores en [0, 1]; filtramos <0.70)

---

## 2. Schema SQL

### Tabla `rag_entities` (canonical entity store)

```sql
CREATE TABLE IF NOT EXISTS rag_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Identidad
    canonical_name TEXT NOT NULL,           -- "Juan Pérez", "Max ops"
    normalized TEXT NOT NULL,               -- "juan perez", "max ops" (lower + accent-strip)
    entity_type TEXT NOT NULL,              -- person | organization | location | event | date

    -- Aliases (JSON array)
    aliases TEXT,                           -- ["Juan", "JP", "Juancito"]

    -- Temporal
    first_seen_ts REAL,                     -- epoch; primer chunk donde aparece
    last_seen_ts REAL,                      -- epoch; último chunk donde aparece
    mention_count INTEGER DEFAULT 0,        -- N veces mencionada en corpus

    -- Calidad
    confidence REAL,                        -- promedio scores GLiNER (0.70–1.0)

    -- Extensibilidad
    extra_json TEXT,                        -- {"source_types": [...], "dossier_path": "..."}

    UNIQUE(normalized, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_entities_normalized ON rag_entities(normalized);
CREATE INDEX IF NOT EXISTS idx_entities_type ON rag_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON rag_entities(canonical_name);
```

**Rationale:**
- `UNIQUE(normalized, entity_type)`: "juan" como person ≠ "juan" como organization
- `normalized` permite búsqueda case-insensitive + accent-insensitive
- `aliases` captura variaciones sin duplicar filas
- `extra_json` para metadata futuro (dossier path en vault, contacto Apple, etc.)

### Tabla `rag_entity_mentions` (relación chunk → entity)

```sql
CREATE TABLE IF NOT EXISTS rag_entity_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    entity_id INTEGER NOT NULL REFERENCES rag_entities(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL,                 -- match con collection ID (ej. "whatsapp://...")

    -- Contexto
    source TEXT,                            -- vault | whatsapp | gmail | calendar | reminders
    ts REAL,                                -- epoch; timestamp del chunk
    snippet TEXT,                           -- ±80 chars alrededor de la mención

    -- Calidad
    confidence REAL,                        -- score GLiNER para esta mención específica

    UNIQUE(entity_id, chunk_id)             -- no duplicar mención en el mismo chunk
);

CREATE INDEX IF NOT EXISTS idx_mentions_entity ON rag_entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON rag_entity_mentions(chunk_id);
CREATE INDEX IF NOT EXISTS idx_mentions_source ON rag_entity_mentions(source);
CREATE INDEX IF NOT EXISTS idx_mentions_ts ON rag_entity_mentions(ts);
CREATE INDEX IF NOT EXISTS idx_mentions_entity_ts ON rag_entity_mentions(entity_id, ts DESC);
```

**Rationale:**
- Separación 1:N permite queries "todos los chunks donde aparece Juan" sin denormalizar
- `snippet` para debug + UI (mostrar contexto de la mención)
- `UNIQUE(entity_id, chunk_id)` previene duplicados si chunk menciona a Juan 2×
- Índice compuesto `(entity_id, ts DESC)` es el que usa `handle_entity_lookup()` (ver improvement-2-entity-retrieval-plan.md §3)

### Integración con `_TELEMETRY_DDL` (rag.py:3273)

Agregar dos tuplas al `_TELEMETRY_DDL` siguiendo el patrón existente de `rag_feedback`, `rag_behavior`, `rag_conversations_index`, `rag_ocr_cache`. Schema changes NO bumpean `_COLLECTION_BASE` (SQL tables son side-tables, no afectan embedding schema).

---

## 3. Normalización + alias resolution

### Algoritmo de deduplicación

```
1. Extracción bruta (GLiNER)
   Input: chunk text
   Output: [(entity_text, type, confidence), ...]

2. Normalización
   - lowercase + accent-strip (unicodedata.NFD + combining filter)
   - Reutiliza `_fold()` existente en rag.py (~7813)

3. Clustering por normalized + type
   - Candidatos con (normalized, type) idéntico → misma entidad
   - NO merge por cosine si normalized difiere ("Juan" ≠ "Juana", aunque cosine ~0.85)

4. Merge de aliases
   - Canonical = forma más frecuente (más larga si empate)
   - Aliases = todas las otras formas vistas
   - JSON array: ["Juan", "JP", "Juancito"]

5. Upsert a rag_entities
   - INSERT OR REPLACE by (normalized, entity_type)
   - Actualiza mention_count, last_seen_ts, confidence (promedio)
```

### Cuidado con falsos positivos

**Problema**: "Juan" y "Juana" tienen cosine ~0.85 pero son personas distintas.

**Solución**:
- NO merge automático por cosine si `normalized` difiere
- Solo merge si `normalized` idéntico (ej. "juan perez" = "Juan Pérez" = "JUAN PEREZ")
- Cosine >0.85 se usa SOLO para **alias detection dentro de un cluster ya formado** (ej. detectar "JP" como alias de "Juan Pérez" post-extracción vía co-ocurrencia en el mismo chunk)

---

## 4. Integración en indexing

### Cambios en `_index_single_file` (rag.py ~14855-14977)

Después de `col.add()` (línea ~14977), agregar:

```python
# NUEVO: entity extraction
if os.environ.get("RAG_EXTRACT_ENTITIES", "1") == "1":
    try:
        _extract_and_index_entities_for_chunks(
            chunks=chunks,
            ids=ids,
            metadatas=metadatas,
            source="vault",
        )
    except Exception as e:
        _log_entity_extraction_error({"path": doc_id_prefix, "error": str(e)[:200]})
```

### Helper nueva: `_extract_and_index_entities_for_chunks`

Firma:

```python
def _extract_and_index_entities_for_chunks(
    chunks: list[tuple[str, str, str]],  # (embed_text, display_text, parent)
    ids: list[str],
    metadatas: list[dict],
    source: str,
) -> None:
    """
    Extrae entidades de chunks y las upserta a rag_entities + rag_entity_mentions.
    Silent-fail: cualquier error se loguea a LOG_PATH como 'entity_extraction_error',
    no bloquea indexing.
    """
```

### Helper GLiNER batch

```python
_gliner_model = None
_gliner_lock = threading.Lock()

def _get_gliner_model():
    global _gliner_model
    with _gliner_lock:
        if _gliner_model is None:
            from gliner import GLiNER
            _gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        return _gliner_model

def _extract_entities_batch(texts: list[str]) -> list[dict]:
    """
    Batch extraction: returns list-of-dicts, uno por texto.
    Each dict: {normalized: {canonical, aliases, type, confidence}}
    """
    try:
        model = _get_gliner_model()
    except ImportError:
        return _extract_entities_fallback_qwen(texts)

    results = []
    for text in texts:
        raw = model.predict_entities(
            text,
            labels=["person", "organization", "location", "event"],
        )
        candidates = [
            (e["text"], e["label"].lower(), e["score"])
            for e in raw
            if e["score"] >= 0.70
        ]
        entities = _cluster_entities(candidates)
        results.append(entities)
    return results
```

### Para ingesters externos (scripts/ingest_*.py)

Patrón compartido — cada ingester después de `col.add()` llama:

```python
if os.environ.get("RAG_EXTRACT_ENTITIES", "1") == "1":
    try:
        rag._extract_and_index_entities_for_chunks(
            chunks=chunks,
            ids=ids,
            metadatas=metadatas,
            source=source,  # "whatsapp" | "gmail" | "calendar" | "reminders"
        )
    except Exception as e:
        logger.warning(f"Entity extraction failed for {source}: {e}")
```

Tocar: `scripts/ingest_whatsapp.py`, `scripts/ingest_gmail.py`, `scripts/ingest_calendar.py`, `scripts/ingest_reminders.py`.

### Gate por env var

```python
RAG_EXTRACT_ENTITIES = os.environ.get("RAG_EXTRACT_ENTITIES", "1") == "1"
```

Disable fácil si hay problemas: `RAG_EXTRACT_ENTITIES=0 rag index`.

---

## 5. Backfill script: `scripts/backfill_entities.py`

### Especificación

- Itera todos los chunks en la colección actual
- Procesa en batches (default 50)
- Progress bar con Rich (patrón resto del codebase)
- Idempotente: re-runnable sin duplicar filas
- Flag `--dry-run`: no escribe a SQL
- Flag `--batch-size N`: override default

### Invocación

```bash
# Dry run
python scripts/backfill_entities.py --dry-run

# Real (7 min para 8231 chunks)
python scripts/backfill_entities.py

# Batch size custom
python scripts/backfill_entities.py --batch-size 100
```

### Idempotencia

`INSERT OR REPLACE` en `rag_entities` + `INSERT OR IGNORE` en `rag_entity_mentions` hacen safe re-run. El `mention_count` se incrementa correctamente vía `COALESCE((SELECT mention_count FROM ... WHERE ... ), 0) + 1` pero hay que tener cuidado: re-correr backfill aumenta count. **Fix**: reset explícito en backfill mode vía flag `--reset` (borra filas antes de repoblar).

---

## 6. Tests

| Path | Cases |
|---|---|
| `tests/test_entity_extraction.py` | 15: persona/org/alias/normalización/confianza/clustering/edge |
| `tests/test_entity_schema.py` | 10: UNIQUE constraint, FK cascade, índices, upsert idempotent |
| `tests/test_entity_backfill.py` | 8: procesa all chunks, dry-run no writes, idempotent |

### Casos críticos

```python
def test_gliner_extracts_person():
    text = "Juan Pérez y Ana García hablaron sobre el proyecto."
    entities = rag._extract_entities_batch([text])[0]
    assert "juan perez" in entities or "juan" in entities
    assert entities.get("juan perez", {}).get("type") == "person"

def test_normalize_entity_name():
    assert rag._normalize_entity_name("Juan Pérez") == "juan perez"
    assert rag._normalize_entity_name("JUAN PÉREZ") == "juan perez"

def test_cluster_entities_merges_duplicates():
    candidates = [
        ("Juan Pérez", "person", 0.95),
        ("Juan", "person", 0.88),
        ("JP", "person", 0.80),
    ]
    clusters = rag._cluster_entities(candidates)
    assert clusters["juan perez"]["canonical"] == "Juan Pérez"
    assert set(clusters["juan perez"]["aliases"]) == {"Juan", "JP"}

def test_cluster_entities_respects_type():
    candidates = [
        ("Juan", "person", 0.95),
        ("Juan", "organization", 0.85),
    ]
    clusters = rag._cluster_entities(candidates)
    assert len(clusters) == 2  # distintos tipos → distintos clusters

def test_upsert_entities_idempotent():
    # Re-upsert misma entity NO duplica filas en rag_entities
    # Sí duplica en rag_entity_mentions si distinto chunk_id (correcto)

def test_foreign_key_cascade():
    # DELETE entity → mentions borradas automáticamente

def test_backfill_dry_run_no_writes():
    # --dry-run no toca SQL
```

---

## 7. Performance budget

### Indexing cost per nota (5 chunks típico)

| Fase | Sin extraction | Con extraction |
|---|---|---|
| Embed (bge-m3) | 200ms | 200ms |
| col.add() | 50ms | 50ms |
| Entity extraction (GLiNER) | — | 250ms (5 × 50ms) |
| Upsert SQL | — | 10ms |
| **Total** | **250ms** | **510ms (+104%)** |

Overhead aceptable para `watch` y manual `rag index`. Para bulk re-index (`rag index --reset`), considerar `RAG_EXTRACT_ENTITIES=0` + backfill post-facto.

### Backfill corpus actual

- 8231 chunks × 50ms = ~7 minutos (one-time)
- Memory peak: ~1GB (GLiNER + batch processing)

### Storage estimado

| Tabla | Filas | Tamaño |
|---|---|---|
| `rag_entities` | ~500 uniques | ~100 KB |
| `rag_entity_mentions` | ~8231 (1:1 con chunks worst case) | ~1 MB |
| **Total** | | **~1.1 MB** |

Negligible vs los ~50 MB de `ragvec.db` actual.

### VRAM budget

Stack con entity extraction running:
- bge-m3: ~1.2 GB
- bge-reranker-v2-m3: ~2-3 GB (pinned)
- qwen2.5:3b: ~2.5 GB
- qwen2.5:7b: ~4.7 GB
- **GLiNER: ~400 MB** (NUEVO)
- **Total: ~12 GB** en 36 GB unified memory — holgado (33%)

---

## 8. Open questions

### 1. ¿Extraer fechas como pseudo-entities?

**Pro**: queries "mi agenda del 20 de marzo" podrían beneficiarse
**Con**: duplica info con `created_ts` metadata; GLiNER date detection débil en ES; `handle_agenda` ya parsea con `_parse_agenda_window()`

**Decisión**: **diferir a v2**. Hoy intent `agenda` ya cubre el caso.

### 2. ¿Usar `_mentions_cache` como warm-start?

**Contexto**: vault tiene `_mentions_cache` (rag.py ~7801) para dossiers de personas (scans frontmatter, lock-protected).

**Pro**: evitar re-extraer personas ya documentadas
**Con**: solo cubre vault (no cross-source); requiere parsing dossier format

**Decisión**: **v2**. Para v1 GLiNER es suficiente.

### 3. ¿Integración con entity_lookup intent?

Ver `docs/improvement-2-entity-retrieval-plan.md` (documento hermano, escrito en paralelo). El extraction design asume que handle_entity_lookup consume los datos producidos acá.

---

## 9. Checklist de implementación

- [ ] Agregar `rag_entities` + `rag_entity_mentions` a `_TELEMETRY_DDL` (rag.py:3273)
- [ ] Implementar `_normalize_entity_name()`, `_cluster_entities()`, `_upsert_entities()`
- [ ] Implementar `_extract_entities_batch()` con GLiNER + fallback qwen2.5:3b
- [ ] Implementar `_extract_and_index_entities_for_chunks()` helper
- [ ] Integrar en `_index_single_file()` (línea ~14977)
- [ ] Integrar en 4 ingesters: `scripts/ingest_{whatsapp,gmail,calendar,reminders}.py`
- [ ] Crear `scripts/backfill_entities.py` con `--dry-run` + `--batch-size` + `--reset`
- [ ] Tests: `tests/test_entity_extraction.py` (15), `test_entity_schema.py` (10), `test_entity_backfill.py` (8)
- [ ] Documentar en CLAUDE.md: env var `RAG_EXTRACT_ENTITIES`, performance notes
- [ ] Bench: medir latency `_index_single_file` con/sin extraction antes de merge
- [ ] Backfill corpus existente (7 min)
- [ ] Agregar `gliner` a `pyproject.toml` dependencies

---

## 10. Referencias

- [GLiNER multi-v2.1](https://huggingface.co/urchade/gliner_multi-v2.1) — zero-shot NER multilingüe
- [GLiNER paper (Zaratiana et al.)](https://arxiv.org/abs/2311.08526)
- Patrón SQL existente: `rag.py:3273` (`_TELEMETRY_DDL`)
- Patrón normalización: `rag.py` `_fold()` (~7813)
- Patrón ingesters: `scripts/ingest_whatsapp.py`, `scripts/ingest_gmail.py`
- Tests existentes SQL: `tests/test_rag_writers_sql.py`
- Design doc cross-source: `docs/design-cross-source-corpus.md` (§2.3–2.7)
- Retrieval plan hermano: `docs/improvement-2-entity-retrieval-plan.md`
