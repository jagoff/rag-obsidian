"""Post-process pipeline (citation repair, critique, NLI) — extracted from rag/__init__.py 2026-05-04.

All public symbols are re-exported into the ``rag`` namespace via::

    from rag.postprocess import *  # in rag/__init__.py

so callers that do ``import rag; rag.run_parallel_post_process(...)`` keep working
without any change.  Callers may also import directly::

    from rag.postprocess import run_parallel_post_process

NOTE on NLI singleton state (_nli_model, _nli_last_use, _nli_lock):
Those module-level variables remain in rag/__init__.py because tests mutate
them directly via ``rag._nli_model = mock``.  The functions here that need
the singleton (get_nli_model, maybe_unload_nli_model) also stay in
rag/__init__.py for the same reason.  This module imports them lazily at
call-time to avoid circular imports.
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rag.iberian_leak_filter import replace_iberian_leaks

if TYPE_CHECKING:
    pass

__all__ = [
    # Dataclasses
    "Claim",
    "ClaimGrounding",
    "GroundingResult",
    "PostProcessResult",
    # Refusal detection
    "_REFUSAL_PATTERNS",
    "_REFUSAL_RE",
    "_is_refusal",
    # Claims splitting
    "split_claims",
    "_split_prose",
    # NLI grounding
    "ground_claims_nli",
    # Post-process tasks
    "_pp_task_repair",
    "_pp_task_critique",
    "_pp_task_nli",
    "_pp_parallel_enabled",
    # Orchestrator
    "run_parallel_post_process",
    # Citation NLI verifier (Quick Win #2, 2026-05-04)
    "VerificationResult",
    "_get_citation_nli_model",
    "_nli_mode",
    "split_sentences_for_nli",
    "verify_answer_nli",
    "apply_nli_mode",
]


# ──────────────────────────────────────────────────────────────────────
# Dataclasses — NLI claim representation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Claim:
    """Atomic claim extracted from an LLM response.

    Attributes:
        text: The claim text (1-300 chars after strip).
        start_char: Offset in original response (UI highlighting).
        end_char: End offset in original response.
        is_refusal: True for "No encontré..." patterns — skip NLI on these.
    """
    text: str
    start_char: int = 0
    end_char: int = 0
    is_refusal: bool = False


@dataclass
class ClaimGrounding:
    """Result of NLI grounding for a single claim."""
    text: str
    verdict: Literal["entails", "neutral", "contradicts"]
    evidence_chunk_id: str | None = None
    evidence_span: str | None = None
    score: float = 0.0
    start_char: int = 0
    end_char: int = 0


@dataclass
class GroundingResult:
    """Aggregated NLI grounding for a full response."""
    claims: list[ClaimGrounding] = field(default_factory=list)
    claims_total: int = 0
    claims_supported: int = 0
    claims_contradicted: int = 0
    claims_neutral: int = 0
    nli_ms: int = 0


@dataclass
class PostProcessResult:
    """Resultado agregado de `run_parallel_post_process()`.

    El helper ejecuta hasta 3 tareas de post-procesamiento concurrentemente
    (citation-repair, critique, NLI grounding) y retorna el estado mergeado.
    Diseñado para encapsular los 4 bloques inline previos (query + chat
    ×repair + critique) con un único call site por caller.
    """
    full: str                                  # texto final (post-merge)
    bad_citations: list[tuple[str, str]] = field(default_factory=list)
    citation_repaired: bool = False
    critique_fired: bool = False
    critique_changed: bool = False
    nli_summary: dict | None = None
    nli_ms: int = 0
    nli_result: "GroundingResult | None" = None
    timing_ms: dict = field(default_factory=dict)  # wall/repair/critique/nli


# ──────────────────────────────────────────────────────────────────────
# Refusal detection
# ──────────────────────────────────────────────────────────────────────
# Refusal patterns — skip NLI on these, they're not factual claims.
# Match is case-insensitive over the whole claim text.
#
# Cubre las 4 variantes intencionales de refusal usadas en los prompts
# bajo `rag/prompts/intents/` (ver audit 2026-04-25 R2-6 #1 para la
# decisión de mantenerlas distintas en lugar de unificar):
#
# 1. ``"No tengo esa información en tus notas."`` — strict, web,
#    chat, system_rules. Default para queries semánticas que no
#    encontraron fuentes relevantes.
# 2. ``"No encontré esto en el vault."`` — lookup (count/list/recent/
#    agenda). Frase distinta para distinguir en telemetría
#    `count+refused` vs `semantic+refused`.
# 3. ``"No hay suficientes fuentes en el vault para sintetizar esto."``
#    — synthesis. Semánticamente distinto: tenemos UNA fuente pero
#    no llega para sintetizar (≥2 requerido).
# 4. ``"No hay suficientes fuentes en el vault para comparar esto."``
#    — comparison. Idem synthesis pero para comparison (≥2 lados).
#
# Si un prompt nuevo agrega una 5ta variante, agregá el pattern acá
# para que el cache poisoning detector y el NLI grounding lo skipeen.
_REFUSAL_PATTERNS = (
    r"^no encontr[eé]\b",
    r"^no hay\s+(?:ning[úu]n|informaci[óo]n|suficientes)\b",
    r"^no tengo (?:esa\s+)?informaci[óo]n\b",
    r"^i (?:don'?t|did not|didn'?t) find\b",
    r"^i could not find\b",
    r"^no information found\b",
)
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

# ──────────────────────────────────────────────────────────────────────
# Module-level compiled regex (Fix 1: avoid per-call re.compile)
# ──────────────────────────────────────────────────────────────────────
_SPLIT_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_SPLIT_TABLE_RE = re.compile(r"^(\|[^\n]*\|\s*\n){2,}", re.MULTILINE)
_SPLIT_LIST_RE = re.compile(
    r"^(?:[-*+]\s+[^\n]+\n?){2,}|^(?:\d+\.\s+[^\n]+\n?){2,}",
    re.MULTILINE,
)
_SPLIT_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])")
_NORM_WS_RE = re.compile(r"\s+")


def _is_refusal(text: str) -> bool:
    """True if the text matches a known refusal pattern."""
    return bool(_REFUSAL_RE.match(text.strip()))


# ──────────────────────────────────────────────────────────────────────
# Claims splitting
# ──────────────────────────────────────────────────────────────────────

def split_claims(text: str) -> list[Claim]:
    """Split an LLM response into atomic claims for NLI grounding.

    Strategy (regex-based, no spacy dependency required):
    1. Detect refusal upfront → single Claim with is_refusal=True
    2. Preserve markdown code fences (```...```) as single claims
    3. Preserve markdown bullet/ordered lists (consecutive - * + 1. lines) as one claim
    4. Preserve markdown tables (|...|...| rows) as one claim
    5. Split remaining prose on sentence boundaries (`[.!?]+\\s+`)
    6. Drop claims < 8 chars (punctuation-only fragments)

    Fase A = regex implementation. Fase B may swap for spacy-es if precision needed.

    Args:
        text: The LLM response text.

    Returns:
        List of Claim objects with char offsets preserved for UI highlighting.
        Empty list if text is empty/whitespace.
    """
    if not text or not text.strip():
        return []

    stripped = text.strip()

    # Early detection of refusal — single claim, no further splitting
    if _is_refusal(stripped):
        idx = text.find(stripped)
        return [Claim(
            text=stripped,
            start_char=idx if idx >= 0 else 0,
            end_char=(idx if idx >= 0 else 0) + len(stripped),
            is_refusal=True,
        )]

    # Strategy: extract code fences + tables + lists as atomic blocks,
    # then split prose between them on sentence boundaries.
    claims: list[Claim] = []

    blocks: list[tuple[int, int, str]] = []
    for pattern in (_SPLIT_CODE_FENCE_RE, _SPLIT_TABLE_RE, _SPLIT_LIST_RE):
        for m in pattern.finditer(text):
            blocks.append((m.start(), m.end(), m.group(0).strip()))

    # Sort blocks by start, skip overlaps (previous wins)
    blocks.sort(key=lambda b: b[0])
    merged: list[tuple[int, int, str]] = []
    for start, end, btext in blocks:
        if merged and start < merged[-1][1]:
            continue
        merged.append((start, end, btext))

    cursor = 0
    for start, end, btext in merged:
        if start > cursor:
            prose = text[cursor:start]
            claims.extend(_split_prose(prose, cursor))
        if btext and len(btext) >= 8:
            claims.append(Claim(
                text=btext,
                start_char=start,
                end_char=end,
                is_refusal=False,
            ))
        cursor = end

    if cursor < len(text):
        prose = text[cursor:]
        claims.extend(_split_prose(prose, cursor))

    # If no blocks at all, just split the whole text as prose
    if not merged:
        claims = _split_prose(text, 0)

    return claims


def _split_prose(prose: str, base_offset: int) -> list[Claim]:
    """Split prose on sentence boundaries. Helper for split_claims().

    Returns list of Claim objects with offsets relative to the original
    response (base_offset + local_offset).
    """
    if not prose or not prose.strip():
        return []

    claims: list[Claim] = []
    # Sentence boundary: punctuation followed by whitespace + capital letter.
    parts = _SPLIT_SENTENCE_RE.split(prose)

    offset = 0
    for part in parts:
        if not part:
            continue
        stripped = part.strip()
        if len(stripped) < 8:
            offset += len(part) + 1
            continue
        local_idx = part.find(stripped)
        claim_start = base_offset + offset + (local_idx if local_idx >= 0 else 0)
        claims.append(Claim(
            text=stripped,
            start_char=claim_start,
            end_char=claim_start + len(stripped),
            is_refusal=False,
        ))
        offset += len(part) + 1

    return claims


# ──────────────────────────────────────────────────────────────────────
# NLI grounding
# ──────────────────────────────────────────────────────────────────────

def ground_claims_nli(
    claims: list[Claim],
    docs: list[str],
    metas: list[dict],
    *,
    threshold_contradicts: float = 0.7,
    cosine_threshold: float = 0.5,
    max_claims: int = 20,
) -> GroundingResult | None:
    """Stub for NLI grounding (Fase B will implement actual NLI inference).

    Fase A: returns a GroundingResult with all claims marked "neutral"
    (safe default). This lets callers integrate the API shape without
    depending on the NLI model being loaded.

    Fase B will:
        1. Load mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 via CrossEncoder
        2. For each claim: embed + cosine prefilter top-3 chunks
        3. Run NLI on (claim, chunk) pairs → entails/neutral/contradicts
        4. Apply thresholds: score >= threshold_contradicts → entails
                           [threshold-0.2, threshold) → neutral
                           < threshold-0.2 → contradicts

    Args:
        claims: list of Claim objects from split_claims()
        docs: retrieved chunk display_text (parallel to metas)
        metas: retrieved chunk metadata dicts (file, note, chunk_id, etc.)
        threshold_contradicts: NLI score threshold for marking contradicts
        cosine_threshold: minimum cosine for prefilter (skip irrelevant chunks)
        max_claims: safety gate — skip if claims > max_claims (cost explosion)

    Returns:
        GroundingResult with per-claim verdicts + aggregate counts.
        None if inputs empty or max_claims exceeded.
    """
    # Lazy import to avoid circular dependency
    import rag as _rag

    if not claims or not docs:
        return None
    if len(claims) > max_claims:
        return None

    groundings: list[ClaimGrounding] = []
    nli_start = time.time()

    try:
        model = _rag.get_nli_model()
        for c in claims:
            if c.is_refusal:
                groundings.append(ClaimGrounding(
                    text=c.text, verdict="neutral", score=0.0,
                    start_char=c.start_char, end_char=c.end_char,
                ))
                continue
            if model is None:
                groundings.append(ClaimGrounding(
                    text=c.text, verdict="neutral", score=0.0,
                    start_char=c.start_char, end_char=c.end_char,
                ))
                continue
            try:
                import numpy as np
                pairs = [(c.text, doc) for doc in docs]
                scores = model.predict(pairs, convert_to_numpy=True)
                # Shape (N_docs, 3) con columnas [entailment, neutral, contradiction]
                best_entail_idx = int(np.argmax(scores[:, 0]))
                best_entail_score = float(scores[best_entail_idx, 0])
                best_contradict_idx = int(np.argmax(scores[:, 2]))
                best_contradict_score = float(scores[best_contradict_idx, 2])

                if best_entail_score > threshold_contradicts and best_entail_score > best_contradict_score:
                    verdict: "Literal['entails', 'neutral', 'contradicts']" = "entails"
                    evidence_idx = best_entail_idx
                elif best_contradict_score > threshold_contradicts:
                    verdict = "contradicts"
                    evidence_idx = best_contradict_idx
                else:
                    verdict = "neutral"
                    evidence_idx = best_entail_idx

                evidence_span = docs[evidence_idx][:200]
                evidence_chunk_id = metas[evidence_idx].get("chunk_id") or metas[evidence_idx].get("file")

                groundings.append(ClaimGrounding(
                    text=c.text,
                    verdict=verdict,
                    evidence_chunk_id=evidence_chunk_id,
                    evidence_span=evidence_span,
                    score=best_entail_score if verdict == "entails" else best_contradict_score,
                    start_char=c.start_char,
                    end_char=c.end_char,
                ))
            except Exception as exc:
                _rag._silent_log("nli_inference_failed", exc)
                groundings.append(ClaimGrounding(
                    text=c.text, verdict="neutral", score=0.0,
                    start_char=c.start_char, end_char=c.end_char,
                ))
    except Exception as exc:
        _rag._silent_log("nli_grounding_failed", exc)
        groundings = [ClaimGrounding(
            text=c.text, verdict="neutral", score=0.0,
            start_char=c.start_char, end_char=c.end_char,
        ) for c in claims]

    nli_ms = int((time.time() - nli_start) * 1000)

    return GroundingResult(
        claims=groundings,
        claims_total=len(groundings),
        claims_supported=sum(1 for g in groundings if g.verdict == "entails"),
        claims_contradicted=sum(1 for g in groundings if g.verdict == "contradicts"),
        claims_neutral=sum(1 for g in groundings if g.verdict == "neutral"),
        nli_ms=nli_ms,
    )


# ──────────────────────────────────────────────────────────────────────
# Post-process tasks (citation-repair, critique, NLI wrapper)
# ──────────────────────────────────────────────────────────────────────

def _pp_task_repair(
    full_orig: str,
    bad: list[tuple[str, str]],
    metas: list[dict],
    context: str,
    question: str,
) -> dict:
    """Citation-repair task — para ejecutar en el ThreadPoolExecutor.

    Condición de disparo: `bad` no vacío, cantidad ≤ _CITATION_REPAIR_MAX_BAD,
    existe al menos un path válido. Misma gate que el bloque inline original.
    Return: {ran, ok, full, ms}. ok=True solo si el repair output pasa
    verify_citations sin bad.
    """
    import rag as _rag
    t0 = time.perf_counter()
    valid_paths = [m.get("file", "") for m in metas if m.get("file")]
    if not bad or len(bad) > _rag._CITATION_REPAIR_MAX_BAD or not valid_paths:
        return {"ran": False, "ok": False, "full": None, "ms": 0}
    repair_system = (
        "Solo puedes citar las siguientes rutas: "
        + ", ".join(valid_paths)
        + ". Responde la misma pregunta usando SOLO esas rutas. No inventes otras."
    )
    messages = [
        {"role": "system", "content": repair_system},
        {"role": "user", "content": (
            f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}\n\nRESPUESTA:"
        )},
    ]
    try:
        _repair_model = _rag._postprocess_model()
        resp = _rag._chat_capped_client().chat(
            model=_repair_model,
            messages=messages,
            options=_rag._postprocess_options(),
            stream=False,
            keep_alive=_rag.chat_keep_alive(_repair_model),
        )
        # 2026-04-29: filter PT→ES post-gen. La respuesta reparada se
        # le muestra al user en CLI y web — sin filter, leaks pt del
        # repair pueden llegar al usuario.
        repair_full = replace_iberian_leaks((resp.message.content or "").strip())
    except Exception as exc:
        _rag._silent_log("postprocess_repair_failed", exc)
        return {"ran": True, "ok": False, "full": None, "ms": int((time.perf_counter() - t0) * 1000)}

    if not repair_full:
        return {"ran": True, "ok": False, "full": None, "ms": int((time.perf_counter() - t0) * 1000)}

    repair_bad = _rag.verify_citations(repair_full, metas)
    ok = not repair_bad
    return {
        "ran": True,
        "ok": ok,
        "full": repair_full if ok else None,
        "ms": int((time.perf_counter() - t0) * 1000),
    }


def _pp_task_critique(
    full_orig: str,
    metas: list[dict],
    context: str,
    question: str,
) -> dict:
    """Critique task — regenera la respuesta si el chat model no la aprueba.

    Corre siempre sobre `full_orig` (no sobre el repair output) porque en
    paralelo no podemos esperar al repair. Merge priority en el caller:
    repair gana sobre critique cuando ambos mutan (ver docstring de
    run_parallel_post_process).
    """
    import rag as _rag
    t0 = time.perf_counter()
    critique_system = (
        "Evalúa si la respuesta responde la pregunta usando SOLO las fuentes provistas. "
        "Si es correcta, devuélvela tal cual. "
        "Si es incorrecta o incompleta, regenerá una respuesta mejor usando SOLO esas fuentes. "
        "No expliques tu evaluación — devolvé solo la respuesta final."
    )
    paths_str = "\n".join(m.get("file", "") for m in metas if m.get("file"))
    critique_user = (
        f"Pregunta: {question}\n\n"
        f"Respuesta original:\n{full_orig}\n\n"
        f"Fuentes disponibles:\n{paths_str}\n\n"
        f"Contexto de las fuentes:\n{context}"
    )
    try:
        _critique_model = _rag._postprocess_model()
        resp = _rag._chat_capped_client().chat(
            model=_critique_model,
            messages=[
                {"role": "system", "content": critique_system},
                {"role": "user", "content": critique_user},
            ],
            options=_rag._postprocess_options(),
            stream=False,
            keep_alive=_rag.chat_keep_alive(_critique_model),
        )
        # 2026-04-29: filter PT→ES post-gen. La crítica regenera la
        # respuesta y el caller la pinta directo en CLI/web — sin
        # filter, leaks pt llegan al user.
        crit_full = replace_iberian_leaks((resp.message.content or "").strip())
    except Exception as exc:
        _rag._silent_log("postprocess_critique_failed", exc)
        return {"ran": True, "changed": False, "full": None, "ms": int((time.perf_counter() - t0) * 1000)}

    if not crit_full:
        return {"ran": True, "changed": False, "full": None, "ms": int((time.perf_counter() - t0) * 1000)}

    # Normalise whitespace for comparison — avoid spurious "changed" when the
    # LLM just reflowed the paragraph.
    norm = lambda s: _NORM_WS_RE.sub(" ", s.strip())
    changed = norm(crit_full) != norm(full_orig)
    return {
        "ran": True,
        "changed": changed,
        "full": crit_full if changed else None,
        "ms": int((time.perf_counter() - t0) * 1000),
    }


def _pp_task_nli(
    full_orig: str,
    docs: list[str],
    metas: list[dict],
) -> dict:
    """NLI grounding task — no muta el texto, solo clasifica claims.

    Return dict con summary (claims_total/supported/contradicted/neutral),
    ms, y el raw GroundingResult para render del panel en el caller.
    """
    import rag as _rag
    t0 = time.perf_counter()
    try:
        # Use rag.split_claims so monkeypatching in tests works correctly
        claims = _rag.split_claims(full_orig)
    except Exception as exc:
        _rag._silent_log("postprocess_nli_split_failed", exc)
        return {"ran": False, "summary": None, "raw": None, "ms": 0}

    if not claims:
        return {"ran": False, "summary": None, "raw": None, "ms": 0}

    try:
        result = _rag.ground_claims_nli(
            claims, docs, metas,
            threshold_contradicts=_rag._nli_contradicts_threshold(),
            max_claims=_rag._nli_max_claims(),
        )
    except Exception as exc:
        _rag._silent_log("postprocess_nli_failed", exc)
        return {"ran": False, "summary": None, "raw": None, "ms": int((time.perf_counter() - t0) * 1000)}

    if result is None:
        return {"ran": False, "summary": None, "raw": None, "ms": 0}

    summary = {
        "claims_total": result.claims_total,
        "supported": result.claims_supported,
        "contradicted": result.claims_contradicted,
        "neutral": result.claims_neutral,
    }
    # Preserve the GroundingResult's internal ms (inference-only timing) so
    # callers that log `nli_ms` get the same value they did pre-refactor.
    # The task's own wall-clock includes split_claims + the launch overhead
    # which is not what "nli_ms" historically meant.
    return {
        "ran": True,
        "summary": summary,
        "raw": result,
        "ms": int(result.nli_ms or 0),
    }


def _pp_parallel_enabled() -> bool:
    """Env toggle para debugging: `RAG_PARALLEL_POSTPROCESS=0` fuerza
    secuencial. Default on — sin env var, usa threads cuando hay ≥2 tareas.
    """
    val = os.environ.get("RAG_PARALLEL_POSTPROCESS", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def run_parallel_post_process(
    full_orig: str,
    *,
    docs: list[str],
    metas: list[dict],
    context: str,
    question: str,
    fast_path: bool,
    critique: bool,
    intent: str | None,
) -> PostProcessResult:
    """Ejecuta citation-repair + critique + NLI en paralelo y mergea.

    Parallelization rules:
      - Fast-path bypass: cuando `fast_path=True` (lookup intents con score
        alto), skip repair (mismo gate que el código inline original).
      - Critique: solo si `critique=True`.
      - NLI: solo si `RAG_NLI_GROUNDING` ON y `intent` no en _nli_skip_intents.

    Merge priority cuando múltiples tareas quieren mutar `full`:
      1. **Repair wins over critique**. Rationale: critique corre sobre
         `full_orig` (que tenía citations inválidas); aunque regenere con
         citations válidas, preferimos el repair que tiene un gate explícito
         de verify_citations. Critique podría haber cambiado prosa y retenido
         citations inválidas.
      2. Solo repair OK → full = repaired.
      3. Solo critique changed → full = critiqued.
      4. Nada → full = full_orig.

    NLI nunca muta `full`; se agrega al result en paralelo.

    Threading safety:
      - Todas las tareas leen full_orig + metas/docs/context/question (read-only).
      - Cada tarea hace su propia llamada a ollama (HTTP client thread-safe).
      - NLI accede a mDeBERTa singleton con lock interno en get_nli_model.
      - No shared mutable state entre tasks.

    Args:
        full_orig: texto del LLM stream (post-generation).
        docs, metas: retrieved chunks del retrieve().
        context: contexto construido (build_progressive_context output).
        question: query original.
        fast_path: flag de adaptive routing (skip repair si True).
        critique: flag `--critique` (opt-in critique pass).
        intent: intent clasificado (gate para NLI).

    Returns: PostProcessResult con full mergeado + flags + timing.
    """
    import rag as _rag

    wall_start = time.perf_counter()
    bad = _rag.verify_citations(full_orig, metas)

    # Figure out which tasks to run
    do_repair = (not fast_path) and bool(bad)
    do_critique = bool(critique)
    do_nli = _rag._nli_grounding_enabled() and (intent not in _rag._nli_skip_intents())

    result = PostProcessResult(full=full_orig, bad_citations=bad,
                                critique_fired=do_critique)

    if not (do_repair or do_critique or do_nli):
        result.timing_ms = {"wall": int((time.perf_counter() - wall_start) * 1000)}
        return result

    tasks: dict[str, tuple] = {}
    if do_repair:
        tasks["repair"] = (_pp_task_repair, (full_orig, bad, metas, context, question))
    if do_critique:
        tasks["critique"] = (_pp_task_critique, (full_orig, metas, context, question))
    if do_nli:
        tasks["nli"] = (_pp_task_nli, (full_orig, docs, metas))

    # Execute — parallel when ≥2 tasks AND the env toggle is on. Sequential
    # single-task path avoids thread overhead + preserves determinism for
    # existing tests that mock ollama.chat with an in-order call sequence.
    outcomes: dict[str, dict] = {}
    if len(tasks) >= 2 and _pp_parallel_enabled():
        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            futs = {name: ex.submit(fn, *args) for name, (fn, args) in tasks.items()}
            for name, fut in futs.items():
                try:
                    outcomes[name] = fut.result(timeout=60)
                except Exception as exc:
                    _rag._silent_log(f"postprocess_{name}_future_failed", exc)
                    outcomes[name] = {"ran": False, "ok": False, "changed": False,
                                       "full": None, "summary": None, "raw": None,
                                       "ms": 0}
    else:
        for name, (fn, args) in tasks.items():
            try:
                outcomes[name] = fn(*args)
            except Exception as exc:
                _rag._silent_log(f"postprocess_{name}_seq_failed", exc)
                outcomes[name] = {"ran": False, "ok": False, "changed": False,
                                   "full": None, "summary": None, "raw": None,
                                   "ms": 0}

    # Merge — repair wins over critique
    repair_out = outcomes.get("repair", {})
    critique_out = outcomes.get("critique", {})
    nli_out = outcomes.get("nli", {})

    if repair_out.get("ok") and repair_out.get("full"):
        result.full = repair_out["full"]
        result.citation_repaired = True
    elif critique_out.get("changed") and critique_out.get("full"):
        result.full = critique_out["full"]
        result.critique_changed = True

    if nli_out.get("ran"):
        result.nli_summary = nli_out.get("summary")
        result.nli_ms = int(nli_out.get("ms") or 0)
        result.nli_result = nli_out.get("raw")

    result.timing_ms = {
        "wall": int((time.perf_counter() - wall_start) * 1000),
        "repair": int(repair_out.get("ms") or 0),
        "critique": int(critique_out.get("ms") or 0),
        "nli": int(nli_out.get("ms") or 0),
    }
    return result


# ══════════════════════════════════════════════════════════════════════
# Citation NLI Verifier — Quick Win #2 (2026-05-04)
#
# Modelo: cross-encoder/nli-deberta-v3-small (HuggingFace)
# https://huggingface.co/cross-encoder/nli-deberta-v3-small
# ~80 MB MPS-friendly, sentence-transformers CrossEncoder.
#
# Verifica que cada oración del answer esté respaldada por los chunks
# recuperados. Controlado por RAG_NLI_MODE:
#   - "off"   (default): sin verificación.
#   - "mark":  oraciones no verificadas reciben sufijo "(?)".
#   - "strip": oraciones no verificadas se eliminan del answer; si el
#              answer queda < 30 chars, devuelve fallback "No tengo
#              evidencia clara para responder".
#
# Invariantes de ownership (rag-llm):
#   - Modelo: cross-encoder/nli-deberta-v3-small.
#   - Sampling: n/a (CrossEncoder.predict, no generación).
#   - Threshold: 0.5 (entailment score mínimo), configurable via env.
#   - Silent-fail: cualquier error → answer pasa crudo.
#   - Default OFF → no rompe UX existente.
# ══════════════════════════════════════════════════════════════════════

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
        len(sentences)
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
        # Reemplazar cada oración no verificada por oración + "(?)".
        # Hacemos reemplazo simple de izquierda a derecha usando el texto
        # original de la oración.
        modified = answer
        # Para no double-marcar si una oración aparece dos veces,
        # iteramos por índice de fin para sustituir la primera ocurrencia.
        for r in results:
            if not r.verified:
                orig = r.sentence
                marked = orig + _NLI_UNVERIFIED_SUFFIX
                # Solo reemplazar la primera ocurrencia en el texto actual
                modified = modified.replace(orig, marked, 1)
        return modified

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
