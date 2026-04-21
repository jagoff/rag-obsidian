# Improvement #1 — NLI grounding integration plan

## Executive Summary

Implementar claim-level Natural Language Inference (NLI) grounding para validar que las respuestas generadas estén soportadas por los chunks retrieved. Cada claim de la respuesta se clasifica como `entails`, `neutral` o `contradicts` respecto a la evidencia. Claims no soportados se marcan en UI; claims contradichos opcionalmente disparan regeneración. Mejora la calidad sobre `verify_citations()` (que solo valida existencia de paths) sin romper el pipeline actual.

---

## 1. Claim splitting

### Recomendación: **Opción D (híbrida con fallback)**

Nuevo método `split_claims(text: str) -> list[Claim]` en `rag.py` que:
1. **Primary**: spacy-es sentence splitter (~50ms una sola carga)
2. **Fallback**: regex sobre `[.!?]\s+` cuando spacy no disponible
3. **Edge-cases**: preservar listas markdown, tablas y code blocks como claims atómicos únicos

### Justificación

- **Opción A (regex puro)** falla en listas/tablas/markdown — común en este vault (calendario, bullets agenda)
- **Opción B (qwen2.5:3b)** suma 300-600ms por query (latencia inaceptable en todas las respuestas)
- **Opción C (syntok)** unmaintained; spacy es estándar ES+EN
- **Opción D** balancea calidad + degradación graciosa

### Edge cases en respuestas típicas

```python
# Prosa corta — splits limpios
"El proyecto comenzó en 2024. Tiene tres fases. La primera termina en junio."
# → 3 claims

# Lista con bullets — preservar como claim único
"Las tareas:\n- Implementar NLI\n- Escribir tests\n- Integrar"
# → 1 claim

# Tabla markdown — preservar como claim único
"| Fecha | Evento |\n|---|---|\n| 2024-05-01 | Kickoff |"
# → 1 claim

# Refusal — single claim marked
"No encontré esto en el vault."
# → 1 claim (is_refusal=True, skip NLI)
```

### Implementation sketch

```python
@dataclass
class Claim:
    text: str                    # Atomic claim (1-300 chars)
    start_char: int              # Offset en respuesta original (UI highlighting)
    end_char: int
    is_refusal: bool = False     # True para "No encontré..." patterns

def split_claims(text: str) -> list[Claim]:
    """
    1. Detectar refusals tempranos
    2. Extraer code fences + tables (preservar as-is)
    3. Split resto via spacy o regex
    4. Return con char offsets para UI
    """
```

---

## 2. Claim → evidence matching

### Recomendación: **Strategy C (cosine prefilter + top-3 por claim)**

```
Por cada claim:
  1. Embed claim via bge-m3 (reuse embedder existente)
  2. Cosine similarity contra TODOS los chunks retrieved
  3. Top-3 candidatos (threshold > 0.5)
  4. NLI contra esos 3 (NO contra los K)

Costo: N claims × 3 chunks × NLI = N*3 calls (vs N*K en Strategy A)
```

### Por qué NO A ni B

- **Strategy A (todos los K × N)**: para K=10, N=5 → 50 NLI calls = 5-10s. Inaceptable.
- **Strategy B (solo paths citados)**: pierde contradicciones en chunks retrieved pero no citados. Si un chunk contradice la respuesta pero el LLM no lo cita, queremos flagearlo igual.
- **Strategy C**: balance óptimo. Chunks con cosine < 0.5 son semánticamente irrelevantes — NLI sobre ellos sería ruido.

### Integración con flujo existente

Flujo actual (`rag.py:16135`):
```python
bad = verify_citations(full, result["metas"])  # Chequea paths
if bad and len(bad) <= _CITATION_REPAIR_MAX_BAD:
    # repair loop
```

Flujo nuevo (post-repair):
```python
bad = verify_citations(full, result["metas"])
# ... repair ...

# NEW: NLI grounding (corre después de repair)
if RAG_NLI_GROUNDING:
    claims = split_claims(full)
    groundings = ground_claims_nli(claims, result["docs"], result["metas"])
    if groundings.claims_contradicted > 0:
        full = apply_grounding_markup(full, groundings.claims)
```

---

## 3. API shape

### Dataclasses (rag.py ~line 150)

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ClaimGrounding:
    text: str
    verdict: Literal["entails", "neutral", "contradicts"]
    evidence_chunk_id: str | None    # índice a result["metas"]
    evidence_span: str | None        # ≤200 chars del chunk
    score: float                      # 0.0..1.0
    start_char: int                   # UI highlighting
    end_char: int

@dataclass
class GroundingResult:
    claims: list[ClaimGrounding]
    claims_total: int
    claims_supported: int
    claims_contradicted: int
    claims_neutral: int
    nli_ms: int
```

### Log shape

```python
log_query_event({
    "cmd": "query",
    "q": question,
    # ... existing ...
    "grounding": {
        "claims_total": 5,
        "claims_supported": 4,
        "claims_contradicted": 0,
        "claims_neutral": 1,
        "nli_ms": 2340,
    } if RAG_NLI_GROUNDING else None,
})
```

### SSE event shape (web)

```python
yield _sse("grounding", {
    "claims": [
        {
            "text": "El proyecto termina en junio.",
            "verdict": "entails",
            "evidence_span": "...termina el 30 de junio...",
            "score": 0.92,
            "start_char": 45,
            "end_char": 75,
        },
    ],
    "summary": {"total": 5, "supported": 4, "contradicted": 0, "neutral": 1}
})
```

---

## 4. Integración pipeline

### Archivos + líneas

| File | Line | Change |
|---|---|---|
| `rag.py` | ~150 | `@dataclass Claim`, `ClaimGrounding`, `GroundingResult` |
| `rag.py` | ~200-300 | `split_claims()`, `ground_claims_nli()`, `apply_grounding_markup()` |
| `rag.py` | ~9735 | `_format_chunk_for_llm()` — opcional return chunk_id |
| `rag.py` | 16135-16170 | Post-citation-repair en `query()` — NLI pass |
| `rag.py` | 17253-17285 | Post-citation-repair en `chat()` — NLI pass |
| `rag.py` | 16250+ | `log_query_event()` — grounding fields |
| `rag.py` | 17355+ | `log_query_event()` chat path |
| `web/server.py` | 3737+ | `_emit_grounding()` + SSE event |
| `web/static/app.js` | ~500 | `renderGrounding()` + CSS classes |
| `web/static/index.html` | ~200 | Grounding panel HTML |

### Flow detallado en `query()` (16135-16170)

```python
citation_repaired = False
bad = verify_citations(full, result["metas"])
if bad and len(bad) <= _CITATION_REPAIR_MAX_BAD:
    # ... existing repair ...

# NEW: NLI grounding (independiente del repair outcome)
grounding_result = None
if os.environ.get("RAG_NLI_GROUNDING"):
    try:
        claims = split_claims(full)
        grounding_result = ground_claims_nli(
            claims, result["docs"], result["metas"],
            threshold_contradicts=float(os.environ.get("RAG_NLI_CONTRADICTS_THRESHOLD", "0.7"))
        )
        if grounding_result.claims_contradicted > 0:
            full = apply_grounding_markup(full, grounding_result.claims)
    except Exception as exc:
        _silent_log("nli_grounding_failed", exc)
```

---

## 5. Feature flags

```bash
RAG_NLI_GROUNDING=0               # default OFF hasta validar
RAG_NLI_CONTRADICTS_THRESHOLD=0.7 # threshold contradict marking
RAG_NLI_SKIP_INTENTS=count,list,recent,agenda  # skip trivial intents
RAG_NLI_MODEL=qwen2.5:3b          # ollama NLI model (o bge-nli cross-encoder)
RAG_NLI_COSINE_THRESHOLD=0.5      # prefilter cosine min
RAG_NLI_MAX_CLAIMS=20             # safety gate para respuestas largas
```

### Entrada CLAUDE.md

```markdown
- `RAG_NLI_GROUNDING=1` — enable claim-level NLI grounding (default OFF). When ON,
  cada claim de la respuesta se valida contra chunks retrieved via NLI. Claims no
  soportados se marcan en UI. Costo: +200-400ms por query (N claims × 3 chunks ×
  NLI model). Skippea intents triviales via `RAG_NLI_SKIP_INTENTS`.
- `RAG_NLI_CONTRADICTS_THRESHOLD=0.7` — umbral NLI score para marcar "contradicts".
  Scores ≥ threshold → entails; [threshold-0.2, threshold) → neutral; 
  < threshold-0.2 → contradicts. Subir para conservador (menos false positives).
- `RAG_NLI_SKIP_INTENTS=count,list,recent,agenda` — intents donde NLI se skippea
  (validación claim-level sin sentido en metadata lookups).
- `RAG_NLI_MODEL=qwen2.5:3b` — modelo NLI (default qwen2.5:3b fallback o mDeBERTa-v3 
  base via sentence-transformers, ver improvement-1-nli-model-selection.md).
- `RAG_NLI_COSINE_THRESHOLD=0.5` — cosine mínimo para prefilter chunks antes de NLI.
- `RAG_NLI_MAX_CLAIMS=20` — gate máximo claims por respuesta (explosión cost).
```

---

## 6. Tests

| Path | Cases | Coverage |
|---|---|---|
| `tests/test_nli_grounding.py` | 35 | `split_claims()` edge cases, `ground_claims_nli()` verdict logic, threshold tuning |
| `tests/test_nli_integration.py` | 20 | query + chat con flag ON/OFF, graceful fail cuando modelo NLI indispone, grounding en log |
| `tests/test_nli_regression.py` | 10 | test_response_quality.py pasa con flag OFF, citation-repair intacto, latencia <5s |

### Casos mínimos — test_nli_grounding.py

```python
def test_split_claims_simple_prose():
    text = "El proyecto comenzó en 2024. Tiene tres fases."
    claims = rag.split_claims(text)
    assert len(claims) == 2

def test_split_claims_preserves_lists():
    text = "Las tareas:\n- Implementar\n- Testing"
    claims = rag.split_claims(text)
    assert len(claims) == 1
    assert "- Implementar" in claims[0].text

def test_split_claims_refusal():
    claims = rag.split_claims("No encontré esto en el vault.")
    assert len(claims) == 1
    assert claims[0].is_refusal is True

def test_ground_claims_entails():
    claims = [rag.Claim(text="El proyecto termina en junio.", ...)]
    docs = ["El proyecto termina el 30 de junio."]
    metas = [{"file": "nota.md", "note": "nota"}]
    result = rag.ground_claims_nli(claims, docs, metas)
    assert result.claims[0].verdict == "entails"

def test_ground_claims_contradicts():
    claims = [rag.Claim(text="El proyecto termina en marzo.", ...)]
    docs = ["El proyecto termina el 30 de junio."]
    result = rag.ground_claims_nli(claims, docs, [...], threshold_contradicts=0.7)
    assert result.claims[0].verdict == "contradicts"

def test_ground_claims_neutral():
    claims = [rag.Claim(text="El cielo es azul.", ...)]
    docs = ["El proyecto termina en junio."]
    result = rag.ground_claims_nli(claims, docs, [...])
    assert result.claims[0].verdict == "neutral"

def test_nli_model_unavailable_graceful():
    # monkeypatch ollama.chat/CrossEncoder to raise
    # Should not crash; return None or empty
```

### Regresión existente

- `tests/test_response_quality.py` — correr con flag OFF, bit-identical
- `tests/test_prompt_injection_defence.py` — unaffected (NLI post-generation)
- `tests/test_name_preservation_rule.py` — unaffected

---

## 7. Dashboard + métricas

### Nuevos campos en `queries.jsonl` / `rag_queries` SQL

```json
{
  "cmd": "query",
  "q": "qué hago este fin de semana",
  "grounding": {
    "claims_total": 5,
    "claims_supported": 4,
    "claims_contradicted": 0,
    "claims_neutral": 1,
    "nli_ms": 2340
  }
}
```

### Dashboard integration

Nueva sección "Answer Quality" en `web/static/dashboard.html`:

1. **Time-series chart**: % claims supported sobre 30d
2. **Histograma**: distribución de `nli_ms` (buckets <500ms, 500-1000ms, 1-2s, >2s)
3. **Heatmap**: intent × verdict (qué intents tienen más contradicciones)

### Backend endpoint

```python
@app.get("/api/dashboard/grounding")
def get_grounding_stats(days: int = 30) -> dict:
    """Agrega grounding stats desde rag_queries SQL para últimos N días."""
    # group by date, compute sum(claims_supported) / sum(claims_total),
    # percentiles nli_ms, cross-tab intent × verdict
```

---

## 8. Riesgos + rollback

| Risk | Mitigation |
|---|---|
| **NLI hallucination** — modelo marca supported claims como contradicts | Conservative threshold 0.7, monitor dashboard, tune `RAG_NLI_CONTRADICTS_THRESHOLD` |
| **Latency explosion** — NLI suma 2-4s | Gate `RAG_NLI_MAX_CLAIMS=20`, skip trivial intents, modelo rápido (bge-nli ~45ms o qwen2.5:3b ~150ms) |
| **Ollama overload** — NLI concurrent starva chat model | `_chat_capped_client()` rate limit, NLI serial not parallel |
| **False positives UI** — usuarios confundidos por rojo en claims OK | Flag OFF default, validación manual 20 queries antes de flip |
| **Regresión citation-repair** — NLI interfiere con repair | NLI corre AFTER repair, lógica de repair intacta. Test con flag OFF para verificar |

### Rollback procedure

1. **Inmediato**: `RAG_NLI_GROUNDING=0` (disables all paths)
2. **Si regresión código**: revert commits en `rag.py` + `web/`
3. **Cleanup data**: `grounding: null` rows son inocuas, sin migración

---

## 9. Fases de implementación

### Fase A: módulo standalone + unit tests (1-2 days)

- Dataclasses + `split_claims()` + `ground_claims_nli()` en `rag.py`
- `tests/test_nli_grounding.py` (35 cases)
- **Deliverable**: Tests pasando, sin dependencias pipeline

### Fase B: integración query()/chat() con flag OFF (2-3 days)

- NLI call post-citation-repair en `query()` + `chat()`
- try/except wrapping, silent-log
- Conditional env var + skip-intents logic
- `tests/test_nli_integration.py` (20 cases)
- Regression: `test_response_quality.py` con flag OFF
- **Deliverable**: Integrado pero OFF, tests + regresión limpia

### Fase C: integración `/api/chat` + UI (2-3 days)

- `_emit_grounding()` SSE event en web/server.py
- `renderGrounding()` en app.js con highlights
- Grounding panel HTML + CSS
- **Deliverable**: UI muestra grounding, SSE bien formado

### Fase D: validación manual + flip flag ON (1-2 days)

- 20 manual queries con flag ON
- Revisar dashboard metrics (claims_supported %, nli_ms dist)
- Tune threshold si false-positive rate alto
- Flip flag o mantener OFF si issues
- **Deliverable**: Validation report `docs/nli-grounding-validation.md`

---

## 10. Orden de commits

1. `feat(nli): claim splitting + grounding module + unit tests`
2. `feat(nli): integrate into query() + chat() pipeline (flag OFF)`
3. `feat(nli): web UI + SSE events`
4. `feat(nli): dashboard grounding metrics`
5. `docs(nli): validation report + threshold tuning + flip flag`

---

## 11. Performance

### Latency budget

- Claim splitting: ~50ms (spacy) o ~5ms (regex)
- Embedding claims: ~100ms (bge-m3 batched, reuse embedder)
- Cosine prefilter: ~10ms (numpy)
- NLI inference: ~45-150ms/claim (depende del modelo elegido, ver `improvement-1-nli-model-selection.md`)
- **Total típico**: 300-1000ms para respuesta 3-5 claims

Gate: skip si claims > 20 o intent ∈ skip-list.

### Memory

- bge-m3 embedder: ya cargado (reuse retrieval)
- NLI model: ~900 MB (mDeBERTa) o ~3.5 GB (qwen2.5:3b fallback, ya residente)
- No memory pressure nuevo — reuse modelos existentes

### Concurrency

- NLI calls serial (no parallel, no starva chat)
- `_chat_capped_client()` rate limiting aplica
- Web /api/chat concurrent safe

---

## 12. Validación de contratos CLAUDE.md

✅ **Prompt-injection defence**: NLI post-generation, no afecta `_format_chunk_for_llm()` / `_CHUNK_AS_DATA_RULE`
✅ **Name-preservation guardrail**: NLI no modifica response text (solo marca), proper nouns intactos
✅ **Response-quality post-pipeline**: NLI es stage NUEVO después de repair, antes de render
✅ **Citation-repair**: runs BEFORE NLI. Si repair éxito, NLI valida reparada. Si falla, NLI valida original. Paths independientes.

### Contratos nuevos NLI

- **Verdict advisory, no prescriptive**: UI marca pero no bloquea
- **Best-effort**: modelo unavailable → `grounding_result = None`, silent fail
- **Intent-aware**: skip en count/list/recent/agenda
- **Determinístico**: mismo claim + chunks + modelo = mismo verdict (HELPER_OPTIONS temp=0, seed=42)

---

## 13. Próximos pasos post-Fase D

1. Análisis hallucination rate sobre 30d con grounding data
2. Fine-tuning NLI sobre (claim, chunk, verdict) labeled — vault-specific
3. Integrar con feedback loop (+/- buttons → grounding priors)
4. Cross-vault grounding multi-vault queries
5. Temporal grounding (past vs future claims)

---

## Appendix A: NLI model prompt (fallback ollama)

```
SYSTEM:
Sos un evaluador de inferencia natural. Dada CLAIM y EVIDENCE, clasificá:
- "entails": la evidencia apoya completamente la afirmación
- "neutral": la evidencia no dice nada (irrelevante)
- "contradicts": la evidencia contradice

Respondé SOLO con JSON: {"verdict": "entails|neutral|contradicts", "score": 0.0..1.0, "reason": "..."}

USER:
CLAIM: El proyecto termina en junio.
EVIDENCE: El proyecto termina el 30 de junio de 2024.

RESPONSE:
{"verdict": "entails", "score": 0.95, "reason": "Confirmada por evidencia explícita"}
```

## Appendix B: UI example

```
[Respuesta generada]

El proyecto comenzó en 2024.     ✓ (entails, 92%)
Tiene tres fases planificadas.   ⚠ (neutral, 45%)
La primera termina en junio.     ✓ (entails, 88%)

[Grounding summary]
3 claims | 2 supported | 0 contradicted | 1 neutral
NLI latency: 1.2s
```
