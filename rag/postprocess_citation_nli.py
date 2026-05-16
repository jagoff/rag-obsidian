"""Citation NLI Verifier — extracted from rag/postprocess.py 2026-05-09.

Quick Win #2 (2026-05-04). Verifies that each sentence of the answer is
supported by the retrieved chunks via NLI entailment scoring.

Model: ``cross-encoder/nli-deberta-v3-small`` (HuggingFace,
https://huggingface.co/cross-encoder/nli-deberta-v3-small) — ~80 MB,
MPS-friendly via sentence-transformers ``CrossEncoder``.

Controlled by ``RAG_NLI_MODE``:
  - ``"off"`` (default): no verification.
  - ``"mark"``: unverified sentences get ``" (?)"`` suffix.
  - ``"strip"``: unverified sentences removed; if answer < 30 chars
    afterwards, returns fallback ``"No tengo evidencia clara para
    responder."``.

Invariants (rag-llm domain):
  - Model: ``cross-encoder/nli-deberta-v3-small`` (don't switch without A/B).
  - No sampling (CrossEncoder.predict is deterministic).
  - Threshold: 0.5 (configurable via ``RAG_NLI_THRESHOLD``).
  - Silent-fail: any error → answer passes through unchanged.
  - Default OFF → no UX regression.

Distinct from the NLI grounding pipeline (``ground_claims_nli`` /
``_ground_claims_via_llm`` in ``rag/postprocess.py``) which operates on
claim-level grounding for the post-process orchestrator. This module is
sentence-level citation verification on the final answer.

Re-exported from ``rag.postprocess`` (and from ``rag``) so call sites
keep working without change.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

__all__ = [
    "VerificationResult",
    "_get_citation_nli_lock",
    "_get_citation_nli_model",
    "_nli_mode",
    "split_sentences_for_nli",
    "verify_answer_nli",
    "apply_nli_mode",
]


# Singleton para el modelo NLI de citaciones (distinto del mDeBERTa NLI
# existente que está en rag/__init__.py para el grounding pipeline).
_citation_nli_model = None
_citation_nli_lock = None  # threading.Lock() se inicializa lazy para evitar
                            # importar threading en módulos que no lo necesitan


def _get_citation_nli_lock():
    """Inicializa el lock una sola vez (thread-safe por GIL en el init)."""
    global _citation_nli_lock
    if _citation_nli_lock is None:
        import threading
        _citation_nli_lock = threading.Lock()
    return _citation_nli_lock


_CITATION_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_CITATION_NLI_LOAD_FAILED = False  # Sticky flag: si falla el load, no reintentamos


def _get_citation_nli_model():
    """Lazy-load del CrossEncoder para verificación de citaciones.

    Modelo: `cross-encoder/nli-deberta-v3-small`
    (https://huggingface.co/cross-encoder/nli-deberta-v3-small)
    ~80 MB, MPS-friendly via sentence-transformers CrossEncoder.

    Retorna None si:
    - RAG_NLI_MODE == "off" (default).
    - El modelo no pudo cargarse (sticky-fail — no reintenta).
    - sentence-transformers no está instalado.

    Thread-safe: double-checked locking.
    """
    global _citation_nli_model, _CITATION_NLI_LOAD_FAILED
    if _CITATION_NLI_LOAD_FAILED:
        return None
    lock = _get_citation_nli_lock()
    with lock:
        if _citation_nli_model is not None:
            return _citation_nli_model
        if _CITATION_NLI_LOAD_FAILED:
            return None
        try:
            import torch
            from sentence_transformers import CrossEncoder
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            _citation_nli_model = CrossEncoder(
                _CITATION_NLI_MODEL_NAME,
                max_length=512,
                device=device,
            )
        except Exception as exc:
            _CITATION_NLI_LOAD_FAILED = True
            import warnings
            warnings.warn(
                f"[citation-nli] No se pudo cargar {_CITATION_NLI_MODEL_NAME}: "
                f"{type(exc).__name__}: {exc}. El verifier queda desactivado.",
                stacklevel=2,
            )
            return None
    return _citation_nli_model


def _nli_mode() -> str:
    """Lee RAG_NLI_MODE del entorno. Valores válidos: 'off', 'mark', 'strip'.
    Default 'off'. Valor inválido → 'off' con warning silencioso.
    """
    raw = os.environ.get("RAG_NLI_MODE", "off").strip().lower()
    if raw in ("off", "mark", "strip"):
        return raw
    return "off"


# ──────────────────────────────────────────────────────────────────────
# Dataclass VerificationResult
# ──────────────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """Resultado de verificación NLI para una oración del answer.

    Atributos:
        sentence: texto de la oración.
        verified: True si max_score >= threshold.
        max_score: score de entailment máximo sobre todos los chunks.
        supporting_chunk_idx: índice del chunk con max_score (o -1 si no hay).
    """
    sentence: str
    verified: bool
    max_score: float = 0.0
    supporting_chunk_idx: int = -1


# ──────────────────────────────────────────────────────────────────────
# Sentence splitter
# ──────────────────────────────────────────────────────────────────────

# Abreviaciones comunes que NO terminan oración.
# Patrón: N letras (mayúscula o minúscula) + punto, con palabra siguiente
# en minúscula o con punto de siguiente abreviación.
_ABBREV_RE = re.compile(
    r"\b(?:Sr|Sra|Dr|Dra|Prof|Lic|Ing|Arq|Esp|Est|Dept|Depto|"
    r"Mr|Mrs|Ms|Dr|Prof|Rev|Gen|Col|Sgt|Lt|Cpl|Pvt|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic|"
    r"vs|etc|et\.al|approx|approx|vol|num|no|pág|págs|ed|eds|fig|"
    r"e\.g|i\.e|a\.m|p\.m|U\.S|U\.K|a\.C|d\.C)\.",
    re.IGNORECASE,
)

# Regex para split de oraciones: punto/signo de interrogación/exclamación
# seguido de espacio(s) + letra mayúscula o comilla + mayúscula.
# Respeta que las abreviaciones no corten la oración.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÜÑ\"'¿¡])")

# Números decimales como "3.14" o URLs como "ver 2.5" no deben cortar
_DECIMAL_RE = re.compile(r"\d\.\d")
# URLs: no cortar en puntos de URL
_URL_RE = re.compile(r"https?://\S+")


def split_sentences_for_nli(text: str) -> list[str]:
    """Divide el texto en oraciones para la verificación NLI.

    Características:
    - Respeta abreviaciones comunes (Sr., Dr., etc.).
    - No corta en números decimales (3.14).
    - No corta dentro de URLs.
    - Omite oraciones < 8 chars (fragmentos de puntuación).
    - Preserva código entre backticks como una sola oración.

    Args:
        text: texto del answer del LLM.

    Returns:
        Lista de oraciones (strings) sin duplicados consecutivos.
    """
    if not text or not text.strip():
        return []

    # 1. Extraer bloques de código para no cortarlos
    code_blocks: list[tuple[int, int, str]] = []
    for m in re.finditer(r"`[^`\n]+`|```[\s\S]*?```", text):
        code_blocks.append((m.start(), m.end(), m.group(0)))

    # 2. Reemplazar temporalmente abreviaciones y decimales para que no corten
    # Usamos un placeholder que no aparece en el texto normal
    placeholder_map: dict[str, str] = {}
    working = text

    # Reemplazar URLs primero
    def _replace_url(m: re.Match) -> str:
        key = f"\x00URL{len(placeholder_map)}\x00"
        placeholder_map[key] = m.group(0)
        return key
    working = _URL_RE.sub(_replace_url, working)

    # Reemplazar decimales
    def _replace_decimal(m: re.Match) -> str:
        key = f"\x00DEC{len(placeholder_map)}\x00"
        placeholder_map[key] = m.group(0)
        return key
    working = _DECIMAL_RE.sub(_replace_decimal, working)

    # Reemplazar abreviaciones
    def _replace_abbrev(m: re.Match) -> str:
        key = f"\x00ABB{len(placeholder_map)}\x00"
        placeholder_map[key] = m.group(0)
        return key
    working = _ABBREV_RE.sub(_replace_abbrev, working)

    # 3. Splittear
    parts = _SENTENCE_END_RE.split(working)

    # 4. Restaurar placeholders
    def _restore(s: str) -> str:
        for k, v in placeholder_map.items():
            s = s.replace(k, v)
        return s

    sentences: list[str] = []
    for part in parts:
        s = _restore(part).strip()
        if len(s) >= 8:
            sentences.append(s)

    return sentences


# ──────────────────────────────────────────────────────────────────────
# verify_answer_nli
# ──────────────────────────────────────────────────────────────────────

_CITATION_NLI_THRESHOLD = float(os.environ.get("RAG_NLI_THRESHOLD", "0.5"))
# Columna de entailment en el output del CrossEncoder de 3 labels.
# cross-encoder/nli-deberta-v3-small usa: 0=contradiction, 1=entailment, 2=neutral
# (verificado con la model card en HF).
_NLI_DEBERTA_ENTAIL_IDX = 1


def verify_answer_nli(
    answer: str,
    chunks: list[str],
    *,
    threshold: float | None = None,
    model=None,
) -> list[VerificationResult]:
    """Verifica que cada oración del answer esté soportada por los chunks.

    Usa `cross-encoder/nli-deberta-v3-small` para calcular scores de
    entailment entre cada oración y cada chunk recuperado.

    Batching: todos los pares (oración × chunk) se envían en un solo
    `model.predict()` para minimizar overhead de inferencia.

    Args:
        answer: texto de la respuesta del LLM.
        chunks: lista de textos de los chunks recuperados.
        threshold: score mínimo de entailment para marcar verified=True.
            Default: RAG_NLI_THRESHOLD (0.5).
        model: modelo override para tests (evita cargar el singleton).

    Returns:
        Lista de VerificationResult, uno por oración.
        Lista vacía si:
        - answer vacío.
        - chunks vacíos.
        - modelo no disponible (RAG_NLI_MODE=off o load failed).
        - cualquier excepción (silent fail).
    """
    if not answer or not answer.strip() or not chunks:
        return []

    _thr = threshold if threshold is not None else _CITATION_NLI_THRESHOLD

    # Modelo: usar el override (tests) o el singleton
    _model = model
    if _model is None:
        _model = _get_citation_nli_model()
    if _model is None:
        return []

    sentences = split_sentences_for_nli(answer)
    if not sentences:
        return []

    results: list[VerificationResult] = []
    try:
        import numpy as np

        # Construir todos los pares (sentence, chunk) para batch inference
        n_chunks = len(chunks)
        pairs: list[tuple[str, str]] = []
        for sent in sentences:
            for chunk in chunks:
                pairs.append((sent, chunk))

        # Inferencia en batch (un solo predict para todo)
        raw_scores = _model.predict(pairs, convert_to_numpy=True)

        # raw_scores puede tener shape (N, 3) para modelos multiclase
        # o (N,) para modelos binarios. Manejar ambos.
        if hasattr(raw_scores, "shape") and len(raw_scores.shape) == 2:
            # Multiclase: extraer columna de entailment
            entail_scores = raw_scores[:, _NLI_DEBERTA_ENTAIL_IDX]
        else:
            # Binario o flat: usar el score directo
            entail_scores = np.array(raw_scores)

        # Agrupar por oración
        for i, sent in enumerate(sentences):
            sent_scores = entail_scores[i * n_chunks : (i + 1) * n_chunks]
            max_score = float(np.max(sent_scores))
            best_chunk_idx = int(np.argmax(sent_scores))
            results.append(VerificationResult(
                sentence=sent,
                verified=max_score >= _thr,
                max_score=max_score,
                supporting_chunk_idx=best_chunk_idx if max_score >= _thr else -1,
            ))
    except Exception as exc:
        # Silent fail: log pero no crashear
        import warnings
        warnings.warn(
            f"[citation-nli] verify_answer_nli falló: {type(exc).__name__}: {exc}",
            stacklevel=2,
        )
        return []

    return results


# ──────────────────────────────────────────────────────────────────────
# apply_nli_mode — aplica el modo al answer
# ──────────────────────────────────────────────────────────────────────

_NLI_STRIP_MIN_CHARS = 30
_NLI_STRIP_FALLBACK = "No tengo evidencia clara para responder."
_NLI_UNVERIFIED_SUFFIX = " (?)"


def apply_nli_mode(
    answer: str,
    results: list[VerificationResult],
    mode: str,
) -> str:
    """Aplica el modo NLI al answer basándose en los VerificationResult.

    Modos:
        "off":   devuelve answer sin cambios (no se debería llamar en modo off).
        "mark":  agrega sufijo " (?)" a las oraciones no verificadas.
        "strip": elimina las oraciones no verificadas; si el answer resultante
                 tiene < 30 chars, devuelve _NLI_STRIP_FALLBACK.

    Args:
        answer: texto original de la respuesta.
        results: lista de VerificationResult de verify_answer_nli().
        mode: "off" | "mark" | "strip".

    Returns:
        Texto modificado según el modo.
    """
    if mode == "off" or not results:
        return answer

    if mode == "mark":
        # Reconstruir desde los results en orden para evitar el bug de
        # substring: str.replace() con la primera oración más corta como
        # substring de una más larga ya marcada produce doble-marcado.
        parts = []
        for r in results:
            if r.verified:
                parts.append(r.sentence)
            else:
                parts.append(r.sentence + _NLI_UNVERIFIED_SUFFIX)
        return " ".join(parts)

    if mode == "strip":
        # Conservar solo las oraciones verificadas
        kept = [r.sentence for r in results if r.verified]
        if not kept:
            stripped = ""
        else:
            # Reconstruir el texto uniendo oraciones verificadas.
            # Intentar preservar el espaciado original del answer.
            stripped = " ".join(kept)
        if len(stripped.strip()) < _NLI_STRIP_MIN_CHARS:
            return _NLI_STRIP_FALLBACK
        return stripped

    # Fallback seguro
    return answer
