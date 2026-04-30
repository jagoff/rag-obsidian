"""Extracción de patrones repetidos de ``rag_routing_decisions``.

El listener escribe una fila por cada audio actionable que clasifica:
``(transcript, bucket_llm, bucket_final, user_response, ...)``. Después
de N decisiones acumuladas, queremos detectar:

> ¿hay frases que el user clasifica consistentemente al mismo bucket?

Si "tengo que <X>" termina en ``reminder`` el 95% de 23 casos, eso es una
regla aprendida — la promovemos al sysprompt como sesgo. El classifier
entonces clasifica más rápido y con más confianza esos casos.

## Algoritmo

Para cada decisión con ``bucket_final IS NOT NULL`` (descartamos las que
el user dijo "no"):

1. Tokenize el transcript en palabras (lower, sin puntuación, sin tildes
   removidas — preservamos el español rioplatense con sus acentos).
2. Generar n-grams de 2 y 3 palabras consecutivas.
3. Para cada n-gram: contar (n-gram, bucket) → frecuencia.

Después de procesar todas las decisiones, para cada n-gram con
``total_count ≥ min_count``:

- Bucket dominante = ``argmax(bucket_counts)``.
- Ratio = ``max_count / total_count``.
- Si ``ratio >= min_ratio`` → es un patrón candidato.

## Por qué n-grams literales (no regex)

Las regex tipo ``"tengo que <verb>"`` son más expresivas pero requieren un
parser semántico para llenar el placeholder. Para MVP usamos el bigram
literal ``"tengo que"`` — cuando aparece en un transcript nuevo, le
agrega sesgo al bucket aprendido, sin importar qué venga después. El
LLM se encarga del resto del contexto.

Si en el futuro queremos placeholders, este módulo es el lugar — la
estructura ``RoutingPattern`` ya tiene un campo ``pattern: str`` libre.

## Cost

Para 1000 decisiones × ~50 tokens/transcript = 50k tokens, ~100k n-grams.
Counter + iteración: <100ms en una laptop. No requiere LLM ni red.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# Stopwords mínimas — sólo las muy frecuentes que generan ruido en n-grams.
# Mantenemos lista corta porque palabras como "tengo", "que", "voy" son
# precisamente las que arman patrones útiles ("tengo que ...", "voy a ...").
# Filtramos solo las que aparecen muchas veces sin aportar (artículos,
# preposiciones cortas).
_NGRAM_STOPWORDS = frozenset({
    "a", "ante", "con", "de", "del", "el", "en", "es", "este", "esto", "esta",
    "la", "las", "lo", "los", "para", "por", "que", "qué", "se", "si", "sí",
    "sin", "su", "tu", "un", "una", "y", "o", "u", "al",
})

# Tokenizer simple — split por whitespace + puntuación, lowercase, conserva
# tildes/eñes/acentos. Mismo enfoque que rag_whisper_learning/vocab.py
# (no importamos directo para mantener el subpackage aislado).
_TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass
class RoutingPattern:
    """Un patrón candidato extraído del histórico de decisiones."""
    pattern: str                       # n-gram literal, ej. "tengo que"
    bucket: str                        # bucket dominante
    count: int                         # # decisiones que matchean este n-gram
    ratio: float                       # bucket_count / total_count, 0.0-1.0
    bucket_breakdown: dict[str, int] = field(default_factory=dict)
    # Ejemplos representativos (hasta 3) — útil para que el user revise
    # el patrón y decida si lo promueve. Cada uno es un transcript real.
    examples: list[str] = field(default_factory=list)


# ── Public API ───────────────────────────────────────────────────────────────


def extract_pivot_phrases(
    *,
    min_count: int = 5,
    min_ratio: float = 0.90,
    days: int = 60,
    ngram_sizes: tuple[int, ...] = (2, 3),
    max_examples: int = 3,
) -> list[RoutingPattern]:
    """Encuentra n-grams consistentemente asociados a un mismo bucket.

    Args:
        min_count: Total ocurrencias mínimas del n-gram para ser considerado.
            Default 5 — un n-gram aislado es ruido. 5 ya es señal estable.
        min_ratio: Fracción mínima de decisiones del n-gram que deben caer en
            el bucket dominante. Default 0.90 (90%).
        days: Ventana de retención (decisiones más viejas se ignoran). Los
            hábitos viejos no necesariamente reflejan los actuales.
        ngram_sizes: Qué tamaños de n-gram considerar. Default (2, 3) — los
            unigrams son demasiado ruidosos, los 4-grams demasiado raros.
        max_examples: Cuántos transcripts representativos guardar por
            patrón. Útil para que el user revise antes de promover.

    Returns:
        Lista de RoutingPattern ordenada por count DESC. Vacía si la DB
        no está, no hay decisiones suficientes, o ningún patrón califica.
    """
    import rag

    cutoff_ts = int(time.time()) - days * 86400

    # Counter: (ngram, bucket) → count
    pair_counts: Counter[tuple[str, str]] = Counter()
    # Examples por (ngram, bucket): hasta max_examples transcripts
    pair_examples: dict[tuple[str, str], list[str]] = {}

    try:
        with rag._ragvec_state_conn() as conn:
            # Bug fix 2026-04-30: el listener TS escribía `bucket_llm` y
            # `confidence_llm` pero NUNCA actualizaba `bucket_final` después
            # de la ejecución (gap del wiring entre TS y RAG). Resultado:
            # 17/17 rows con `bucket_final IS NULL` → 0 patterns extraíbles
            # → 0 reglas promovidas → loop muerto desde día 1.
            #
            # Workaround acá: usar `bucket_llm` como fallback cuando hay
            # `confidence_llm='high'`. La premisa: si el LLM clasificó con
            # alta confianza y nadie lo corrigió, asumimos que el bucket
            # final fue ese (caso común — el user solo interviene cuando
            # hay error visible). Riesgo controlado: requerimos
            # min_count=5 + min_ratio=0.90 para promover, así que un
            # outlier ocasional no genera regla mala.
            #
            # Forma final del bucket: COALESCE(bucket_final,
            #   bucket_llm WHERE confidence_llm='high', NULL).
            rows = conn.execute(
                "SELECT transcript, COALESCE(NULLIF(bucket_final, ''), "
                "                            CASE WHEN confidence_llm = 'high' "
                "                                 AND bucket_llm IS NOT NULL "
                "                                 AND bucket_llm != '_failed' "
                "                                 THEN bucket_llm END) "
                "       AS bucket_effective "
                "FROM rag_routing_decisions "
                "WHERE ts >= ? "
                "ORDER BY ts DESC",
                (cutoff_ts,),
            ).fetchall()
    except Exception:
        # DB ausente o tabla no creada todavía. Silent fail — el cron
        # diario reintenta mañana y eventualmente la DB existirá.
        return []

    for transcript, bucket_final in rows:
        if not transcript or not bucket_final:
            continue
        tokens = _tokenize(transcript)
        if len(tokens) < 2:
            continue
        seen_in_this_row: set[str] = set()
        for n in ngram_sizes:
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                ngram_tokens = tokens[i:i + n]
                # Skip n-grams compuestos enteramente por stopwords —
                # generan ruido sin aportar discriminación de bucket.
                if all(t in _NGRAM_STOPWORDS for t in ngram_tokens):
                    continue
                ngram = " ".join(ngram_tokens)
                if ngram in seen_in_this_row:
                    # Contamos el n-gram una vez por transcript, no una vez
                    # por aparición. Sino "tengo que tengo que" infla 2x.
                    continue
                seen_in_this_row.add(ngram)
                key = (ngram, bucket_final)
                pair_counts[key] += 1
                examples_list = pair_examples.setdefault(key, [])
                if len(examples_list) < max_examples:
                    examples_list.append(transcript)

    # Reagrupar por n-gram y calcular bucket dominante + ratio.
    by_ngram: dict[str, dict[str, int]] = {}
    for (ngram, bucket), c in pair_counts.items():
        by_ngram.setdefault(ngram, {})[bucket] = c

    out: list[RoutingPattern] = []
    for ngram, bucket_breakdown in by_ngram.items():
        total = sum(bucket_breakdown.values())
        if total < min_count:
            continue
        top_bucket = max(bucket_breakdown.items(), key=lambda kv: kv[1])
        top_count = top_bucket[1]
        ratio = top_count / total
        if ratio < min_ratio:
            continue
        out.append(RoutingPattern(
            pattern=ngram,
            bucket=top_bucket[0],
            count=total,
            ratio=ratio,
            bucket_breakdown=dict(bucket_breakdown),
            examples=pair_examples.get((ngram, top_bucket[0]), []),
        ))

    # Ordenar por count DESC: los patrones con más evidencia primero.
    # Tie-break por ratio DESC, luego pattern ASC (estable).
    out.sort(key=lambda p: (-p.count, -p.ratio, p.pattern))
    return out
