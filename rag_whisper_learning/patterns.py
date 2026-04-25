"""Detect patrones repetidos en correcciones — ej. "samando" → "fernando" 3 veces.

Útil para identificar:
1. Errores sistemáticos del modelo whisper en palabras específicas (nombres
   propios, voseo fragmentado, slang del usuario).
2. Casos donde el LLM auto-correct está fallando consistentemente y conviene
   un /fix manual con el patrón correcto.

## Algoritmo

Para cada par `(original, corrected)`:
1. Tokenize ambos preservando acentos.
2. Computa `tokens_added = corrected_tokens - original_tokens` (palabras
   que el corrector agregó).
3. Computa `tokens_removed = original_tokens - corrected_tokens` (palabras
   que el corrector quitó).
4. Caso fuerte: `len(added) == 1 and len(removed) == 1` → es un single-word
   swap. El par `(removed[0], added[0])` es un PATTERN claro.
5. Casos más complejos (multi-word changes) se descartan por ahora — pueden
   ser reformulaciones que no aportan signal limpio.

Estrategia conservativa: un single-word swap repetido N veces es un signal
muy fuerte. Multi-word puede haber 100 razones (paráfrasis, estructura,
correcciones gramaticales).

## Cost

Para 1000 correcciones acumuladas: ~50ms (in-memory). No requiere
conexión a Ollama, ni embeddings, ni nada externo.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rag_whisper_learning.vocab import _tokenize


@dataclass
class CorrectionPattern:
    """Un patrón repetido en las correcciones del usuario."""
    original: str        # palabra que whisper transcribe mal
    corrected: str       # palabra que el user manda como fix
    count: int           # cuántas veces apareció este pattern
    sources: dict[str, int]  # breakdown por source (explicit/llm/vault_diff)


def find_correction_patterns(min_count: int = 2) -> list[CorrectionPattern]:
    """Encuentra single-word swaps repetidos ≥min_count veces.

    Args:
        min_count: filtro mínimo. Default 2 — un solo swap es ruido (puede
            ser typo fortuito); 2 ya es signal de error sistemático.

    Returns:
        Lista de `CorrectionPattern` ordenada por count DESC.
        Vacía si no hay patrones repetidos o la DB no está disponible.
    """
    import rag
    pattern_counter: Counter[tuple[str, str]] = Counter()
    pattern_sources: dict[tuple[str, str], dict[str, int]] = {}

    try:
        with rag._ragvec_state_conn() as conn:
            for row in conn.execute(
                "SELECT original, corrected, source FROM rag_audio_corrections "
                "ORDER BY ts DESC"
            ):
                orig_text, corr_text, source = row
                # Tokenize y comparar
                orig_tokens = set(_tokenize(orig_text or ""))
                corr_tokens = set(_tokenize(corr_text or ""))
                added = corr_tokens - orig_tokens
                removed = orig_tokens - corr_tokens
                # Solo considerar single-word swaps — multi-word es noisy
                if len(added) != 1 or len(removed) != 1:
                    continue
                added_word = next(iter(added))
                removed_word = next(iter(removed))
                # Skip si las palabras son muy similares (probablemente typo
                # del propio user en el /fix, no error de whisper).
                if _are_similar(removed_word, added_word):
                    continue
                pattern = (removed_word, added_word)
                pattern_counter[pattern] += 1
                if pattern not in pattern_sources:
                    pattern_sources[pattern] = {"explicit": 0, "llm": 0, "vault_diff": 0}
                if source in pattern_sources[pattern]:
                    pattern_sources[pattern][source] += 1
    except Exception:
        return []

    out: list[CorrectionPattern] = []
    for (orig, corr), count in pattern_counter.most_common():
        if count < min_count:
            continue
        out.append(CorrectionPattern(
            original=orig,
            corrected=corr,
            count=count,
            sources=pattern_sources[(orig, corr)],
        ))
    return out


def _are_similar(a: str, b: str, threshold: float = 0.92) -> bool:
    """True si las palabras son tan parecidas que probablemente sean el
    mismo término con un typo del user (no un error real de whisper).

    Usa `SequenceMatcher` ratio del stdlib. Threshold default 0.92 — bastante
    conservativo para que solo descarte casos muy obvios de typo en el `/fix`.
    Ratios típicos:
    - `calendar` vs `calendar.` → 0.94 → similar (probablemente typo).
    - `fernando` vs `fernandó` → 0.875 → distinto (signal legítimo).
    - `samando` vs `fernando` → 0.53 → distinto (signal fuerte).
    - `calendar` vs `calendarizá` → 0.84 → distinto (voseo legítimo).
    """
    from difflib import SequenceMatcher
    if not a or not b:
        return False
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold
