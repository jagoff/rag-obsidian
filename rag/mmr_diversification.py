"""MMR (Maximal Marginal Relevance) diversification post-rerank.

Contrato (post-rerank, pre cap top-k):

    reordered = apply_mmr(scored, lambda_=0.7, top_k=10)
    # …o más cheap, sin embeddings:
    reordered = apply_mmr_with_folder_penalty(scored, top_k=10, folder_penalty=0.1)

Idea (Carbonell & Goldstein 1998): cuando los top-k vienen todos del mismo
cluster semántico (mismo folder, misma temática, misma nota fragmentada
en chunks parecidos), el LLM recibe contexto redundante y respuestas
sesgadas. MMR re-ordena balanceando relevance vs novelty:

    MMR(d) = λ · rel(d) − (1−λ) · max_{d' ∈ S} sim(d, d')

donde S es el set ya seleccionado y `rel(d)` es el score post-rerank.
λ=1.0 → pure relevance (MMR no-op). λ=0.0 → pure diversity (top-1 fijo,
después maximize distance a lo seleccionado). Default 0.7 (sesgo a
relevance, leve nudge a diversity).

Diferencia con `_apply_mmr_reorder` (rag/__init__.py, gated por
RAG_MMR_DIVERSITY): aquel usa Jaccard de tokens — barato y sin
embeddings, captura near-duplicates léxicos. Este módulo usa cosine
sobre embeddings bge-m3 — más caro (1 embed batch por retrieve()) pero
captura similaridad semántica que Jaccard pierde (sinónimos, paráfrasis,
chunks que dicen lo mismo con palabras distintas). Los dos pueden
coexistir; gating independiente vía env (`RAG_MMR_DIVERSITY` vs
`RAG_MMR`).

Wire-up en `retrieve()`:
  - DESPUÉS del rerank + dedup + contradiction_penalty
  - ANTES del cap top-k (como contradiction_penalty)
  - Skip cuando `counter=True` (el user quiere ver TODO)
  - Gating opcional vía env:
      RAG_MMR=1                  → aplicá MMR pleno (embedding-based)
      RAG_MMR_FOLDER_PENALTY=1   → fallback cheap (sin embed) por folder
      RAG_MMR_LAMBDA=0.7         → tradeoff relevance/diversity
      RAG_MMR_TOP_K=10           → cuántos slots ordena MMR (resto preservado)

Performance + fallback silencioso:
  - `apply_mmr` precomputa embeddings de TODOS los candidates en UN solo
    batch a `embed()`. Costo medido: ~5-15ms para 15-30 chunks (bge-m3
    via local SentenceTransformer cuando RAG_LOCAL_EMBED=1).
  - Si el batch tarda >`MMR_BATCH_BUDGET_MS` (default 500ms): skip
    silencioso → return scored sin tocar. Logging via callback opcional.
  - `apply_mmr_with_folder_penalty` no necesita embeddings: penaliza por
    folder repetido. Útil cuando MMR pleno es muy caro (eg. corpus
    grande, retrieval con k=20, vault sin RAG_LOCAL_EMBED).

NO toca: scoring, BM25, vector search, contradicciones, graph_expand.
Sólo re-ordena la lista post-rerank.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Callable

# ── Tunables ─────────────────────────────────────────────────────────────────
# Budget ms para el batch embed: si tarda más, fallback silencioso a
# "no MMR" (devuelve scored sin tocar). 500ms es ~10× el costo medio
# medido (bge-m3 local sobre 15-30 chunks, primer-call cold ~80ms,
# warm ~10-30ms). Si embed() está degradado (ollama saturado), preferimos
# devolver el orden original que clavar el retrieve.
MMR_BATCH_BUDGET_MS = 500.0


# ── Helpers numéricos ────────────────────────────────────────────────────────


def _cosine(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    """Cosine similarity entre 2 vectores. Devuelve 0.0 si alguno es vacío
    o tiene norma 0 — evita NaN downstream.

    No normalizamos los vectores fuera (bge-m3 NO los entrega normalizados
    por default), así que computamos magnitudes inline. Para batches de
    ~15-30 vectores la sobrecarga vs precomputar normas es despreciable.
    """
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 0.0:
        return 0.0
    return dot / denom


def _doc_text(doc: Any) -> str:
    """Extrae el texto a embedear de una doc.

    Soporta:
      - str directo (legacy / tests sintéticos).
      - dict con clave `text`, `expanded`, `body`, o `content` (en ese orden
        de preferencia). Caemos a "" si ninguna existe.
      - tuple/list con un string adentro.

    Si no encontramos texto, devolvemos "" — el embed cae a un vector cero
    consistente y el cosine ulterior será 0.0 contra todo (skip implicito).
    """
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for k in ("text", "expanded", "body", "content", "snippet"):
            v = doc.get(k)
            if isinstance(v, str) and v:
                return v
        return ""
    if isinstance(doc, (tuple, list)):
        for item in doc:
            if isinstance(item, str) and item:
                return item
    return ""


def _doc_folder(doc: Any) -> str:
    """Extrae el folder (parent dir) del path de la doc.

    Mismas convenciones que `_doc_text` para localizar el path:
      - dict con `path` o `file` (preferencia: `path`, después `file`).
      - tuple/list cuyo primer dict-like tenga path/file.

    Devuelve "" si no podemos inferir folder. Eso degrada graceful: la
    folder-penalty no demote a chunks sin folder conocido (no podemos
    decir si comparten origen).
    """
    path = ""
    if isinstance(doc, dict):
        for k in ("path", "file"):
            v = doc.get(k)
            if isinstance(v, str) and v:
                path = v
                break
    elif isinstance(doc, (tuple, list)):
        for item in doc:
            if isinstance(item, dict):
                for k in ("path", "file"):
                    v = item.get(k)
                    if isinstance(v, str) and v:
                        path = v
                        break
                if path:
                    break
    if not path:
        return ""
    if "/" in path:
        return path.rsplit("/", 1)[0]
    return ""


# ── Embedding-based MMR (full) ───────────────────────────────────────────────


def apply_mmr(
    scored: list[tuple[Any, float]],
    *,
    lambda_: float = 0.7,
    top_k: int = 10,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    budget_ms: float | None = None,
    on_skip: Callable[[str], None] | None = None,
) -> list[tuple[Any, float]]:
    """Re-ordena `scored` con MMR (Carbonell & Goldstein 1998).

    Args:
        scored: lista de tuplas `(doc, score)` ordenada DESC por score
            (output post-rerank). `doc` puede ser str, dict, o tuple
            con texto adentro — ver `_doc_text` para el contrato exacto.
        lambda_: tradeoff relevance/diversity ∈ [0, 1]. 1.0 = pure
            relevance (MMR no-op, output == input). 0.0 = pure diversity
            (top-1 fijo, resto greedy max-distance).
        top_k: cuántos slots reordenar. Resto de la lista se preserva
            tal cual (sin tocar). Defaults a 10.
        embed_fn: function `list[str] -> list[list[float]]`. Default
            `rag.embed`. Inyectable para tests.
        budget_ms: budget total para el batch embed; si lo excede,
            fallback silencioso → return scored sin tocar. Default
            `MMR_BATCH_BUDGET_MS` (500ms).
        on_skip: callback opcional invocado cuando hacemos fallback,
            con un string descriptivo. Default None (silencioso).

    Returns:
        Lista nueva del mismo largo que `scored`. Los primeros
        `min(top_k, len(scored))` slots están reordenados según MMR;
        el resto va tal cual al final.

    Notas operativas:
      - Empezamos siempre con el top-1 (highest relevance) en slot 0.
      - Para slot 2..top_k: pickeamos el candidato i que maximice
        `λ·score_i − (1−λ)·max_{j ∈ S} sim(emb_i, emb_j)`.
      - Idempotencia: aplicar dos veces con mismos params devuelve el
        mismo orden (greedy es determinístico).
      - Robusto a docs sin texto: el embed de "" da un vector que el
        cosine retorna 0 contra cualquier otra cosa → no afecta el
        max_sim inicial pero sí participa de la selección por relevance.
    """
    if not scored:
        return scored
    if len(scored) <= 1:
        return scored
    lambda_ = max(0.0, min(1.0, float(lambda_)))
    # Pure relevance shortcut: no need to embed nor compute MMR.
    if lambda_ >= 1.0:
        return list(scored)

    pool_size = min(int(top_k), len(scored))
    if pool_size <= 1:
        return scored

    pool = list(scored[:pool_size])
    tail = list(scored[pool_size:])

    if budget_ms is None:
        budget_ms = MMR_BATCH_BUDGET_MS

    # Resolve embed function. Lazy import — avoids circular import en module
    # init time si alguna función arriba de `rag/__init__.py` importa este
    # módulo antes de que `embed` esté definido.
    if embed_fn is None:
        try:
            from rag import embed as _embed
        except Exception:
            if on_skip:
                on_skip("embed_import_failed")
            return scored
        embed_fn = _embed

    # Batch embed de los textos del pool. Si toma más del budget, fallback.
    texts = [_doc_text(doc) for doc, _ in pool]
    t0 = time.perf_counter()
    try:
        embeddings = embed_fn(texts)
    except Exception:
        if on_skip:
            on_skip("embed_call_failed")
        return scored
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if elapsed_ms > budget_ms:
        if on_skip:
            on_skip(f"embed_over_budget:{elapsed_ms:.0f}ms")
        return scored
    if not embeddings or len(embeddings) != len(pool):
        if on_skip:
            on_skip("embed_shape_mismatch")
        return scored

    # ── Greedy selection ────────────────────────────────────────────────
    # Slot 0: top-1 fijo.
    selected_idx: list[int] = [0]
    remaining: list[int] = list(range(1, pool_size))

    while remaining:
        best_idx: int | None = None
        best_mmr = -math.inf
        for i in remaining:
            rel = float(pool[i][1])
            max_sim = 0.0
            for j in selected_idx:
                sim = _cosine(embeddings[i], embeddings[j])
                if sim > max_sim:
                    max_sim = sim
            mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        if best_idx is None:
            break
        selected_idx.append(best_idx)
        remaining.remove(best_idx)

    reordered_pool = [pool[i] for i in selected_idx]
    return reordered_pool + tail


# ── Folder penalty (cheap MMR alternative) ───────────────────────────────────


def apply_mmr_with_folder_penalty(
    scored: list[tuple[Any, float]],
    *,
    top_k: int = 10,
    folder_penalty: float = 0.1,
) -> list[tuple[Any, float]]:
    """Re-ordena `scored` penalizando candidatos del mismo folder.

    Variante "cheap" de MMR: en vez de embedear y computar cosines, sólo
    fijamos top-1 y penalizamos a los siguientes que comparten folder
    con alguno ya seleccionado. Idea: si los 5 primeros son del mismo
    folder es señal fuerte de monocultivo, demote linear según cuántos
    "compañeros" de folder ya están en la lista.

    Args:
        scored: lista `(doc, score)` ordenada DESC.
        top_k: cuántos slots reordenar (rest preservado).
        folder_penalty: magnitud de la penalización por folder repetido.
            La penalty se SUMA por cada compañero ya seleccionado del
            mismo folder. Default 0.1 — leve, ~tamaño de un gap típico
            entre rerank scores consecutivos.

    Returns:
        Lista nueva. Los primeros `min(top_k, len(scored))` slots
        re-ordenados; resto preservado.

    Notas:
      - Top-1 SIEMPRE fijo (igual que MMR pleno).
      - Empata por relevance original cuando el penalty no rompe el
        orden (preserva determinismo).
      - Docs sin folder identificable (`_doc_folder` devuelve "")
        no reciben penalty NI lo aplican a otros — quedan sólo bajo
        relevance pura. Es el comportamiento conservador esperado.
    """
    if not scored:
        return scored
    if len(scored) <= 1:
        return scored
    if folder_penalty <= 0.0:
        return list(scored)

    pool_size = min(int(top_k), len(scored))
    if pool_size <= 1:
        return scored

    pool = list(scored[:pool_size])
    tail = list(scored[pool_size:])
    folders = [_doc_folder(doc) for doc, _ in pool]

    selected_idx: list[int] = [0]
    remaining: list[int] = list(range(1, pool_size))

    while remaining:
        best_idx: int | None = None
        best_score = -math.inf
        # Stable secondary key: rank original (índice). Si dos candidatos
        # quedan empatados después del penalty, gana el que venía antes.
        for i in remaining:
            rel = float(pool[i][1])
            f_i = folders[i]
            collisions = 0
            if f_i:  # docs sin folder no penalizan
                for j in selected_idx:
                    if folders[j] == f_i:
                        collisions += 1
            adjusted = rel - folder_penalty * collisions
            if adjusted > best_score:
                best_score = adjusted
                best_idx = i
        if best_idx is None:
            break
        selected_idx.append(best_idx)
        remaining.remove(best_idx)

    reordered_pool = [pool[i] for i in selected_idx]
    return reordered_pool + tail


# ── Telemetría helper ────────────────────────────────────────────────────────


def count_reordered(
    before: list[tuple[Any, float]],
    after: list[tuple[Any, float]],
) -> int:
    """Cuántos slots cambiaron de identidad después del re-orden.

    Compara `id(doc)` posición a posición. Si son idénticas las
    primeras N posiciones, devuelve 0 (MMR fue no-op). Si todo cambió,
    devuelve N. Útil para telemetry — distinto de `len(after)` porque
    sólo cuenta los slots que efectivamente movió el helper.

    Diseñado para slottado defensivo: si el caller pasa listas de
    distinto largo, comparamos sólo el prefijo común.
    """
    n = min(len(before), len(after))
    moved = 0
    for i in range(n):
        if before[i][0] is not after[i][0]:
            moved += 1
    return moved


# ── Env-var helpers (caller-friendly) ────────────────────────────────────────


def env_enabled(name: str) -> bool:
    """`RAG_MMR=1` style flag — truthy values: 1/true/yes/on (case insensitive)."""
    val = os.environ.get(name, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def env_lambda(default: float = 0.7) -> float:
    """RAG_MMR_LAMBDA env, clamped a [0, 1]. Default 0.7."""
    try:
        v = float(os.environ.get("RAG_MMR_LAMBDA", str(default)))
    except ValueError:
        v = default
    return max(0.0, min(1.0, v))


def env_top_k(default: int = 10) -> int:
    """RAG_MMR_TOP_K env, clamped a [1, 100]. Default 10."""
    try:
        v = int(os.environ.get("RAG_MMR_TOP_K", str(default)))
    except ValueError:
        v = default
    return max(1, min(100, v))


def env_folder_penalty(default: float = 0.1) -> float:
    """RAG_MMR_FOLDER_PENALTY_MAGNITUDE env, clamped a [0, 1]. Default 0.1."""
    try:
        v = float(os.environ.get("RAG_MMR_FOLDER_PENALTY_MAGNITUDE", str(default)))
    except ValueError:
        v = default
    return max(0.0, min(1.0, v))
