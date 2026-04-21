# Improvement #2 — Entity-aware hybrid retrieval plan

## Overview

Extensión del pipeline de retrieval para explotar entity-aware filtering, siguiendo el patrón establecido por el intent `agenda` (2026-04-21 evening). El sistema detecta referencias a entidades en queries, las resuelve contra `rag_entities` (poblada por el agente hermano, ver `improvement-2-entity-extraction-design.md`), y filtra resultados a chunks que mencionan esa entidad vía `rag_entity_mentions`.

**Asunción**: `rag_entities` (canonical_name, aliases JSON, entity_type) y `rag_entity_mentions` (entity_id, chunk_id, ts) existen y están pobladas.

---

## 1. Intent regex + positives/negatives

### `_INTENT_ENTITY_LOOKUP_RE`

Siguiendo el patrón de `_INTENT_AGENDA_RE` (rag.py ~9188-9220) y `_INTENT_COMPARISON_RE` (rag.py ~9224-9235):

```python
_INTENT_ENTITY_LOOKUP_RE = re.compile(
    r"\b("
    # "con quién hablé de X" / "a quién le mandé X"
    r"con\s+qu[ié]n\s+habl[eé]|"
    r"a\s+qu[ié]n\s+le\s+mand[eé]|"
    r"a\s+qu[ié]n\s+(?:le\s+)?dije|"
    # "qué dice <Persona> sobre X" / "qué me dijo <Persona>"
    r"qu[eé]\s+(?:dice|dijo|me\s+dijo)\s+\w+\s+(?:sobre|de|acerca\s+de)|"
    # "todo lo de <Persona>" / "todos los mensajes de <Persona>"
    r"todo\s+lo\s+(?:de|sobre|que\s+tengo\s+de)\s+\w+|"
    r"todos?\s+los?\s+(?:mensajes?|mails?|notas?|chats?|correos?)\s+(?:de|con)\s+\w+|"
    # "con <Persona> cuándo"
    r"con\s+\w+\s+(?:cu[aá]ndo|qu[eé]|d[oó]nde)|"
    # "mensajes de <Persona>" + temporal opcional
    r"(?:mensajes?|mails?|notas?|chats?|correos?|conversaciones?)\s+(?:de|con)\s+\w+\s+"
    r"(?:esta\s+semana|hoy|ma[ñn]ana|este\s+mes|el\s+(?:viernes|lunes|martes|mi[eé]rcoles|jueves|s[aá]bado|domingo))?"
    r")",
    re.IGNORECASE,
)
```

### Positives (20+)

- "con quién hablé de ops"
- "con quién hablé sobre el proyecto"
- "a quién le mandé el documento"
- "qué dice max sobre ops"
- "qué me dijo fernando sobre el rag"
- "todo lo de juan"
- "todos los mensajes de max"
- "todos los mails de erica"
- "todas las notas de seba"
- "con max cuándo"
- "con fernando cuándo fue"
- "mensajes de juan esta semana"
- "mails de fernando hoy"
- "notas de erica el viernes"
- "conversaciones con max este mes"

### Negative guards (10+)

- "qué es una entidad" → semantic
- "cuándo es mi cumple" → agenda
- "qué tengo esta semana" → agenda
- "notas recientes de juan" → recent (signal `recientes` más fuerte)
- "últimas notas modificadas" → recent
- "juan vs fernando" → comparison (inherente)
- "síntesis de todo lo que hay sobre RAG" → synthesis (topic, no person)

### Precedencia de intents

**Entity lookup se chequea DESPUÉS de `agenda` pero ANTES de `recent`**:

```
count → list → agenda → entity_lookup → recent → comparison → synthesis → semantic
```

Razones:
- Agenda tiene semántica temporal más estrecha → gana (ej. "qué reuniones tengo con Max esta semana" queda agenda por el noun "reuniones" + temporal)
- Entity lookup gana a `recent` porque "notas recientes de juan" es ambiguo; si user dice "recientes" explícitamente el regex de entity no matchea (guarda)
- Entity lookup pierde a `comparison` — "juan vs fernando" es inherentemente comparativo

---

## 2. Resolver entity en query

### `resolve_entity_from_query(question, sql_conn) → tuple[str, int] | None`

**Algoritmo**:

1. **Extraer candidate string** via regex capture rightmost-match:
   - Pattern: `(?:con|a|de|sobre)\s+(\w+)(?:\s+(?:sobre|de|cuándo|qué|dónde))?`
   - Ejemplo: "qué dice max sobre ops" → candidate = "max"

2. **Normalizar**: lowercase + accent-strip

3. **Lookup contra rag_entities**:
   - **Exact match**: `SELECT id, canonical_name FROM rag_entities WHERE normalized = ?`
   - **Fuzzy fallback** si no exact: [rapidfuzz](https://github.com/maxbachmann/RapidFuzz) `fuzz.token_set_ratio` sobre canonical + aliases, threshold ≥ 85
   - **Multiple matches** (ambigüedad): devolver top-3 por score → UX handler

4. **Fallback**: no match → return `None` → caller cae a semantic retrieve

### Implementación sketch

```python
def resolve_entity_from_query(question: str, sql_conn) -> tuple[str, int] | None:
    m = re.search(
        r"(?:con|a|de|sobre)\s+(\w+)(?:\s+(?:sobre|de|acerca|cuándo|qu[eé]|d[oó]nde))?",
        question.lower()
    )
    if not m:
        return None
    candidate = m.group(1).strip().lower()
    if len(candidate) < 2:
        return None

    # Exact
    row = sql_conn.execute(
        "SELECT id, canonical_name FROM rag_entities WHERE normalized = ? LIMIT 1",
        (candidate,)
    ).fetchone()
    if row:
        return (row[1], row[0])

    # Fuzzy
    from rapidfuzz import fuzz
    all_entities = sql_conn.execute(
        "SELECT id, canonical_name, aliases FROM rag_entities"
    ).fetchall()
    scored = []
    for eid, canonical, aliases_json in all_entities:
        score = fuzz.token_set_ratio(candidate, canonical.lower())
        if aliases_json:
            try:
                for alias in json.loads(aliases_json):
                    score = max(score, fuzz.token_set_ratio(candidate, alias.lower()))
            except (json.JSONDecodeError, TypeError):
                pass
        if score >= 85:
            scored.append((score, eid, canonical))

    if not scored:
        return None
    scored.sort(reverse=True)
    if len(scored) > 1:
        _log_ambiguity(candidate, scored[:3])  # log, MVP keeps top
    return (scored[0][2], scored[0][1])
```

### Ambigüedad UX

**MVP (Option A)**: top fuzzy match, log ambiguity a `rag_queries.extra_json`
**Phase 2 (Option B)**: UX prompt "¿Te referís a Juan Pérez, Juan García, o Juan López?" con inline selection

---

## 3. `handle_entity_lookup()` contract

### Firma (siguiendo patrón `handle_agenda`)

```python
def handle_entity_lookup(
    col: SqliteVecCollection,
    params: dict,
    limit: int = 20,
    *,
    question: str | None = None,
) -> list[dict]:
    """ENTITY_LOOKUP intent — chunks mencionando entity resuelta,
    opcionalmente filtrados por ventana temporal."""
```

### Workflow

```
1. resolve_entity_from_query(question, sql_conn) → (canonical, entity_id) | None
2. Si entity_id is None → return [] (caller fallback semantic)
3. Query rag_entity_mentions:
     SELECT chunk_id, ts FROM rag_entity_mentions
     WHERE entity_id = ? ORDER BY ts DESC LIMIT 200
4. Hydrate metas desde collection
5. Apply tag/folder filters (params, mismo patrón handle_agenda)
6. Optional: _parse_agenda_window(question) → narrow [ts_start, ts_end)
7. Optional: cross-encoder rerank (pool=50 con query original)
8. Apply scoring formula (behavior priors, PageRank, dwell)
9. Return top-K
```

### Temporal window

**Decisión**: reutilizar `_parse_agenda_window()` (rag.py ~9396-9480). Mismos patrones:
- "qué hablé con juan esta semana" → week
- "qué dijo max el viernes" → ese viernes
- "mensajes de fernando hoy" → today
- "con erica cuándo" → sin window (browse all)

No necesitamos `_parse_entity_window()` separado.

### Rerank interaction

Para MVP: **skip rerank**, sortear por ts desc dentro del pool de mentions. Razón: pool de 200 chunks ya está pre-filtrado por entity — rerankearlos con query tipo "todos los mails de juan" no aporta (la query no es semántica).

**Phase 2**: rerank SOLO si la query tiene semantic content secundario ("qué dice max **sobre ops**") — el "sobre ops" requiere rerank para ordenar los chunks de Max por relevancia a ops.

---

## 4. Interacción con scoring formula

Formula actual (CLAUDE.md §256):
```
score = rerank_logit + w.recency_cue*... + w.graph_pagerank*... + w.click_prior*... + ...
```

### Ajustes entity-aware

1. **NO live ranker tune para entity_lookup**: los pesos actuales fueron tuned en semantic/agenda/comparison/synthesis. Entity es clase nueva → puede desplazar pesos existentes. **Acción**: tag intent en `rag_queries`, exclude de `rag tune` hasta tener 50+ entity queries en golden.

2. **Nuevo `w.entity_prior`** (default 0.0):
   - `entity_prior = frequency(user clicked chunks of this entity before)` — Laplace-smoothed CTR per entity_id
   - Computado desde `rag_behavior` agregando clicks sobre chunks que mencionan la entity
   - Activado solo cuando `intent == "entity_lookup"` en scoring loop

3. **Graph PageRank boost** (existente): chunks mencionando entity de alto PageRank ya se boostean via `w.graph_pagerank`, sin cambio.

4. **Recency dentro de entity**: temporal window (§3) maneja el filtro duro. Dentro de la ventana, sort ts desc.

### Extension RankerWeights

```python
class RankerWeights:
    __slots__ = (
        "recency_cue", "recency_always", "tag_literal", "title_match",
        "feedback_pos", "feedback_neg", "feedback_match_floor", "graph_pagerank",
        "click_prior", "click_prior_folder", "click_prior_hour", "dwell_score",
        "entity_prior",  # NEW
    )
```

---

## 5. Fallback cuando entities table empty

### Detección

```python
def handle_entity_lookup(...) -> list[dict]:
    with _ragvec_state_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM rag_entities").fetchone()[0]
        if count == 0:
            return []  # caller fallback
```

### Comportamiento

- Empty tables (sistema nuevo, backfill no corrió) → `[]` → caller hace semantic retrieve
- No user-facing error, degradación seamless

### Flag `--no-entity` / `RAG_NO_ENTITY=1`

CLI + env override para forzar skip. Útil para debug (comparar entity-filtered vs semantic) o disable temporal si entity es ruidoso.

---

## 6. Integración multi-source filter

Entity lookup combina con `--source`:

```bash
rag query "todos los mails de juan" --source gmail
rag query "reuniones con erica" --source calendar
rag query "mensajes de max" --source whatsapp
```

Workflow:
1. Resolve entity → entity_id
2. Fetch mentions WHERE entity_id=?
3. Hydrate metas, filter por source (mismo patrón que retrieve's post-rerank filter)
4. Apply temporal window
5. Return

Implementation: en `handle_entity_lookup()` post-hydrate, aplicar source filter idéntico al de `retrieve()` (normalize_source + frozenset).

---

## 7. Golden queries en queries.yaml

Agregar 10-15 queries al final de la sección "cross-source":

```yaml
  # Entity lookup — person-centric retrieval
  # ─────────────────────────────────────────────────────────────────

  - question: "qué hablé con fernando esta semana"
    expected:
      - whatsapp://120363.../msg-fernando-1
      - whatsapp://120363.../msg-fernando-2
    note: "entity + temporal window (week)"

  - question: "qué me dijo max sobre ops"
    expected:
      - whatsapp://120363.../msg-max-ops
      - gmail://thread/19db04ea6ec97f1f
    note: "entity + semantic filter (ops)"

  - question: "todas las reuniones con erica"
    expected:
      - calendar://fernandoferrari@gmail.com/erica-turno-1
      - calendar://fernandoferrari@gmail.com/erica-turno-2
    note: "entity + source=calendar (implicit noun 'reuniones')"

  - question: "todos los mails de juan"
    expected:
      - gmail://thread/...
    note: "entity + source=gmail (implicit noun 'mails')"

  - question: "mensajes de fernando hoy"
    expected: [whatsapp://...]
    note: "entity + temporal (today)"

  - question: "qué dice seba sobre el rag"
    expected:
      - whatsapp://...
      - gmail://...
    note: "entity + semantic (rag)"

  # Negative cases (must NOT fire entity_lookup)
  - question: "qué es una entidad"
    expected: []
    note: "NOT entity_lookup: semantic concept question"

  - question: "cuándo es mi cumple"
    expected: []
    note: "NOT entity_lookup: agenda temporal"

  - question: "notas recientes de juan"
    expected: [vault://...]
    note: "NOT entity_lookup: 'recientes' → recent intent wins"
```

---

## 8. Métricas esperadas

### Baseline actual (CLAUDE.md post 2026-04-21)

| Métrica | Valor |
|---|---|
| Singles hit@5 | 71.67-81.67% |
| Singles MRR | 0.68-0.78 |
| Chains MRR | 0.73-0.79 |
| Chain success | 54.55% |

### Post-entity-lookup (predicho)

- **Singles hit@5**: +2-5pp (golden actual tiene pocas entity-centric; agregar 10-15 empuja baseline)
- **Chains MRR**: +5-10pp (chains "juan → mail → event → decision" se benefician de entity filtering)
- **Latency**: +5-15ms por query entity (resolve ~3ms + mentions lookup ~5ms, indexado)

### Bucketed eval (Fase 3)

Una vez 20+ entity queries en golden, `rag eval --buckets` (ver `improvement-3-adaptive-routing-design.md`) reporta por intent.

---

## 9. Archivos y funciones a tocar

| File | Line | Change |
|---|---|---|
| `rag.py` | ~9188 | Agregar `_INTENT_ENTITY_LOOKUP_RE` |
| `rag.py` | ~9255 | Branch entity_lookup en `classify_intent()` (entre agenda y recent) |
| `rag.py` | ~9300 | `resolve_entity_from_query()` |
| `rag.py` | ~9500 | `handle_entity_lookup()` |
| `rag.py` | ~12250 | Dispatch en `retrieve()` |
| `rag.py` | ~9968 | `system_prompt_for_intent()` — decidir si reutiliza LOOKUP o nuevo prompt |
| `rag.py` | RankerWeights | Agregar slot `entity_prior` |
| `queries.yaml` | ~390 | +10-15 queries |
| `tests/test_classify_intent.py` | +25 casos | entity positives + negatives |
| `tests/test_handle_entity_lookup.py` | NEW | 15 cases |
| `tests/test_entity_resolution.py` | NEW | 10 cases fuzzy + ambiguity |

---

## 10. Riesgos + rollback

### Risk 1: Ambigüedad nombres comunes

"Juan" → 5 Juanes. MVP devuelve top fuzzy; log ambiguity. Phase 2 UX prompt.

### Risk 2: Regresión eval

Regex roba queries de semantic intent, hit@5 cae.

**Mitigación**:
- **Shadow mode** pre-rollout: log "would-route to entity" sin ejecutar, comparar vs actual semantic por 1 semana
- **Staged rollout**: habilitar web (rag chat) primero, CLI (rag query) después

### Risk 3: Entity table sparse

Backfill incompleto → entity lookup miss.

**Mitigación**:
- `handle_entity_lookup()` detecta empty/sparse + fallback graceful
- Daily dashboard: alert si `rag_entities.COUNT() < 100`

### Risk 4: Performance mentions lookup

Si `rag_entity_mentions` llega a 100k+ rows, query lento.

**Mitigación**:
- Index `(entity_id, ts DESC)` (ver extraction doc §2)
- Cap pre-rerank pool = 200
- Monitor query latency; si >50ms archivar mentions viejos o partition by entity_id

### Rollback procedure

1. **Inmediato**: `RAG_NO_ENTITY=1` env var (disables all paths)
2. **Short-term**: comment branch en `classify_intent()` (1-line change)
3. **Long-term**: revert commits + re-tune ranker

---

## 11. Orden de implementación (fases)

1. **Week 1 — Design + regex + tests**
   - `_INTENT_ENTITY_LOOKUP_RE` + 25 test cases
   - `resolve_entity_from_query()` + 10 test cases (mock SQL)
   - Verificar no hay regressions en `test_classify_intent.py`

2. **Week 2 — Handler + integration**
   - `handle_entity_lookup()` + 15 test cases
   - Integrar en `retrieve()` dispatch
   - +10-15 golden queries
   - `rag eval` — esperar +2-5pp singles hit@5

3. **Week 3 — Scoring + tuning**
   - `entity_prior` weight
   - Bootstrap desde `rag_behavior` (separate tune)
   - Monitor latency (+5-15ms target)

4. **Week 4 — Shadow + rollout**
   - Shadow mode 1 semana
   - Staged rollout web → CLI
   - Daily dashboard entity health

---

## 12. SQL schema assumptions (cross-ref extraction doc)

Ver `improvement-2-entity-extraction-design.md §2` para schema completo. Resumen:

```sql
rag_entities(id, canonical_name, normalized, entity_type, aliases JSON, ...)
rag_entity_mentions(id, entity_id FK, chunk_id, source, ts, snippet, confidence,
                    UNIQUE(entity_id, chunk_id))
-- índices: idx_mentions_entity_ts (entity_id, ts DESC) ← crítico para handle_entity_lookup
```

Phase 3: agregar `entity_id INTEGER` a `rag_behavior` para tunear `entity_prior`:
```sql
ALTER TABLE rag_behavior ADD COLUMN entity_id INTEGER DEFAULT NULL;
CREATE INDEX idx_rag_behavior_entity ON rag_behavior(entity_id);
```

---

## Referencias

- CLAUDE.md §120-147 (Retrieval pipeline), §202-228 (Agenda intent), §256-274 (Scoring)
- `rag.py:9188-9220` `_INTENT_AGENDA_RE`
- `rag.py:9255-9307` `classify_intent()`
- `rag.py:9483-9541` `handle_agenda()`
- `rag.py:9396-9480` `_parse_agenda_window()`
- `rag.py:12250-12630` `retrieve()` + scoring
- `tests/test_classify_intent.py`, `tests/test_handle_agenda.py`
- [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) — fuzzy string matching
- Design doc hermano: `docs/improvement-2-entity-extraction-design.md`
