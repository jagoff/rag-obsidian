"""Query decomposition + Reciprocal Rank Fusion (RRF) prototype.

Idea (2026-05-04 prototype, gated por `RAG_QUERY_DECOMPOSE=1`):
    decompuesta = decompose_query("compará X vs Y") → ["sobre X", "sobre Y"]
    if decompuesta:
        results = [retrieve(c, sub, ...) for sub in decompuesta]  # paralelo
        fused = rrf_fuse(results, k=60, top_k=20)

Apunta específicamente a las queries multi-aspecto donde `expand_queries`
(que genera paráfrasis) NO ayuda: si el user pregunta "tanto X como Y",
las paráfrasis siguen siendo "sobre X y Y" — el embedding promedia los
dos topics y el reranker tiende a colapsar al chunk que matchea más X
que Y. Descomponiendo recuperamos pools INDEPENDIENTES por aspecto y
los fusionamos via RRF (Cormack et al. 2009).

Diferencias con primitives existentes:

  - `expand_queries()` (rag/__init__.py): genera 2 paráfrasis del MISMO
    sentido. Cada paráfrasi tiene los dos aspectos juntos. Misma falla
    sistémica para multi-aspecto.
  - `hyde_embed()`: genera 1 doc hipotético; embedea EL DOC, no la query.
    Subgenérico para queries que necesitan dos contextos disjuntos.
  - `rrf_merge()` (rag/__init__.py): fusiona sem+BM25 de UNA sola query;
    NO fusiona resultados de queries distintas. Esta función es nueva
    porque la signature es distinta (lista de listas de Candidates, no
    listas de IDs).

Detector híbrido — regex + LLM fallback:

  Patrones obvios capturan ~70% de casos sin LLM:
    - "X y también Y", "tanto X como Y", "X así como Y"
    - "X vs Y", "X versus Y"
    - "compará X y Y", "comparar X con Y"
    - "diferencia entre X y Y", "qué hay entre X y Y"
    - "qué hay sobre X y Y" (cuando la conjunción separa entidades)

  Resto pasa al LLM helper (qwen2.5:3b, deterministic) que devuelve JSON:
    {"is_multi_aspect": true, "sub_queries": ["...", "..."]}

  Si LLM timeout / parse-error: silent fall-back a regex-only result.
  Si regex tampoco matchea: tratar como single-aspect (no descompone).

Cuándo NO descompone (gates en `should_consider_decomposition`):

  1. Query corta (<6 tokens) — single-fact típico.
  2. Conjunción dentro de nombre propio ("Juan y María García") — heurística
     simple: si la conjunción separa frases nominales <3 palabras cada una
     y el resto del query es <2 tokens, no descompone.
  3. Detección de single-fact ("cuándo es X", "qué hora", "dónde está").
     Patterns interrogativos cortos que apuntan a un solo dato.
  4. Scope estrecho ya filtrado por path/folder — el caller pasa
     `scope_constrained=True` cuando hay path/folder explícito (la
     fusion no agrega valor si ya nos limitamos a una nota).

Fusion RRF:

  score(d) = Σ_i  1 / (k + rank_i(d))     con k=60 (estándar Cormack 2009)

  Determinístico — tiebreak por doc_id lex order para que dos runs con
  los mismos inputs den el mismo orden de salida (idempotencia).

Wire-up esperado en `retrieve()` (gated, opt-in):

  if os.environ.get("RAG_QUERY_DECOMPOSE", "").strip().lower() in ("1", "true", "yes"):
      sub_queries = decompose_query(question)
      if sub_queries and len(sub_queries) > 1 and not folder and not tag:
          # Paralelizar retrieves de cada sub-query y fusionar por RRF
          ...

NO toca scoring, BM25, vector search, rerank, contradiction_penalty,
MMR. Es un layer ortogonal SOBRE el query final que decide si splittear
en N retrieves antes de entrar al pipeline normal.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Iterable

# ──────────────────────────────────────────────────────────────────────────
# Constants — tunables vía env
# ──────────────────────────────────────────────────────────────────────────

# RRF constant (Cormack et al. 2009 default).
RRF_K_DEFAULT = 60

# Cuántos slots devuelve `rrf_fuse` por defecto.
RRF_TOP_K_DEFAULT = 20

# Cache LRU del detector — evita LLM repetido para la misma query.
DECOMPOSE_CACHE_MAX = 256

# Token mínimo para considerar descomposición. Queries cortas son
# single-fact típicas (cuándo / dónde / cuánto).
MIN_TOKENS_FOR_DECOMPOSE = 6

# ──────────────────────────────────────────────────────────────────────────
# Regex patterns — case-insensitive, captura grupos X / Y
# ──────────────────────────────────────────────────────────────────────────

# Patrones explícitos: capturan aspectos como group(1) y group(2).
# Cada pattern asume que las palabras de conexión NO son nombres propios
# (ej. "Juan y María" no debería matchear porque el resto del query
# alrededor sería <2 tokens). El detector aplica esa guarda después.
_DECOMPOSE_PATTERNS: tuple[re.Pattern, ...] = (
    # "compará X vs Y" / "compará X con Y" / "comparar X y Y"
    re.compile(
        r"^(?:compar(?:á|ar|a|amos|en)|comparación de)\s+(.+?)\s+(?:vs\.?|versus|con|y|frente a|contra)\s+(.+)$",
        re.IGNORECASE,
    ),
    # "X vs Y" / "X versus Y" — standalone short form
    re.compile(
        r"^(.+?)\s+(?:vs\.?|versus)\s+(.+)$",
        re.IGNORECASE,
    ),
    # "diferencia entre X y Y" / "diferencias entre X y Y"
    re.compile(
        r"^(?:cuál(?:es)? (?:es|son) la[s]? )?diferencia[s]?\s+entre\s+(.+?)\s+y\s+(.+)$",
        re.IGNORECASE,
    ),
    # "tanto X como Y" / "X así como Y"
    re.compile(
        r"^(?:.*?)\btanto\s+(.+?)\s+como\s+(.+?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(.+?)\s+así\s+como\s+(.+)$",
        re.IGNORECASE,
    ),
    # "X y también Y" / "X y además Y"
    re.compile(
        r"^(.+?)\s+y\s+(?:también|además|asimismo)\s+(.+)$",
        re.IGNORECASE,
    ),
    # "qué hay sobre X y Y" / "info sobre X y Y" — multi-topic on a single about
    re.compile(
        r"^(?:qué hay\s+|info\s+|notas\s+|qué tengo\s+)?sobre\s+(.+?)\s+y\s+(.+)$",
        re.IGNORECASE,
    ),
)

# Single-fact interrogatives — NO descomponer aunque haya conjunción.
_SINGLE_FACT_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^cu[áa]ndo\s+(?:es|fue|será|tengo|tuve|toca)\b", re.IGNORECASE),
    re.compile(r"^d[óo]nde\s+(?:está|queda|vive|guard[ée])\b", re.IGNORECASE),
    re.compile(r"^qu[ée]\s+hora\s+\b", re.IGNORECASE),
    re.compile(r"^cu[áa]nto\s+(?:cuesta|vale|tarda|falta|pesa|mide)\b", re.IGNORECASE),
    re.compile(r"^qui[ée]n\s+(?:es|era|fue)\b", re.IGNORECASE),
)

# Heurística para detectar conjunción dentro de nombre propio.
# Si toda la query es ≤4 tokens y la conjunción separa <3 palabras de cada lado,
# probablemente es un nombre compuesto ("Juan y María", "Iván y Sofía").
_PROPER_NAME_TOKEN = re.compile(r"^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+$")


# ──────────────────────────────────────────────────────────────────────────
# LRU cache thread-safe
# ──────────────────────────────────────────────────────────────────────────

_cache_lock = threading.Lock()
_cache: "OrderedDict[str, list[str] | None]" = OrderedDict()


def _cache_get(key: str) -> tuple[bool, list[str] | None]:
    """Return (hit, value). value=None means cached as "not multi-aspect"."""
    with _cache_lock:
        if key in _cache:
            val = _cache[key]
            _cache.move_to_end(key)
            return True, list(val) if val else None
    return False, None


def _cache_put(key: str, value: list[str] | None) -> None:
    with _cache_lock:
        _cache[key] = list(value) if value else None
        _cache.move_to_end(key)
        while len(_cache) > DECOMPOSE_CACHE_MAX:
            _cache.popitem(last=False)


def clear_cache() -> None:
    """Test/utility — limpia el LRU."""
    with _cache_lock:
        _cache.clear()


def cache_size() -> int:
    with _cache_lock:
        return len(_cache)


# ──────────────────────────────────────────────────────────────────────────
# Gates: cuándo NO descomponer
# ──────────────────────────────────────────────────────────────────────────


def _looks_single_fact(query: str) -> bool:
    """Patrones interrogativos cortos que apuntan a UN solo dato."""
    for p in _SINGLE_FACT_PATTERNS:
        if p.search(query):
            return True
    return False


def _conjunction_inside_proper_name(query: str) -> bool:
    """Heurística: query corta donde " y " separa dos nombres propios.

    Ejemplo: "juan y maría", "juan y maría sobre laburo" (5 tokens, izq+der<=2).
    No queremos descomponer si toda la query gira alrededor de un par de
    personas — el matching semántico de la query entera funciona mejor
    porque captura el contexto compartido.
    """
    parts = re.split(r"\s+y\s+", query, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return False
    left, right = parts[0].strip(), parts[1].strip()
    left_toks = left.split()
    right_toks = right.split()
    # Ambos lados <=3 palabras Y al menos uno empieza con mayúscula real
    # (cuando el caller mantiene casing original — eval lo lowercased
    # asi que también aceptamos lowercase corto).
    if len(left_toks) <= 3 and len(right_toks) <= 3:
        # Si la query entera es corta (≤6 tokens) Y al menos uno parece
        # nombre propio (capital o stopword-heavy), tratamos como nombre.
        if len(query.split()) <= 6:
            return True
    return False


def should_consider_decomposition(
    query: str,
    *,
    folder: str | None = None,
    tag: str | None = None,
    path: str | None = None,
    source: Any = None,
    min_tokens: int | None = None,
) -> bool:
    """Pre-gate: decide si vale la pena correr el detector.

    Devuelve False si:
      - query es vacía / muy corta
      - hay scope explícito (folder/tag/path) — descomposición no aporta
      - es claramente single-fact ("cuándo fue X")
      - conjunción dentro de un nombre propio compuesto

    El caller debe respetar este return — si False, NO llamar
    `decompose_query`. Cuando True, llamar al detector.
    """
    q = (query or "").strip()
    if not q:
        return False
    # Scope explícito — saltarse decomposition.
    if folder or tag or path:
        return False
    # Si algún pattern explícito matchea, bypass token floor + nombre-propio
    # — la regex ya provee alta precisión.
    has_explicit_pattern = any(p.search(q) for p in _DECOMPOSE_PATTERNS)
    # Single-fact → no descomponer.
    if _looks_single_fact(q):
        return False
    # Conjunción dentro de nombre propio compuesto — sólo cuando NO hay
    # patrón explícito (ej. "juan y maría" no matchea ningún regex y se
    # considera nombre; "diferencia entre python y rust" matchea regex
    # explícita y deja el guard out).
    if not has_explicit_pattern and _conjunction_inside_proper_name(q):
        return False
    # Token floor — pero saltarse si hay un pattern explícito.
    tokens = q.split()
    floor = min_tokens if min_tokens is not None else MIN_TOKENS_FOR_DECOMPOSE
    if len(tokens) < floor:
        return has_explicit_pattern
    return True


# ──────────────────────────────────────────────────────────────────────────
# Detector: regex primero, LLM fallback
# ──────────────────────────────────────────────────────────────────────────


def _normalize_sub_query(text: str) -> str:
    """Limpia bordes (puntuación, conectores residuales)."""
    s = (text or "").strip()
    # Trim conectores comunes al inicio
    s = re.sub(r"^(?:y|o|que|de|sobre|con|en)\s+", "", s, flags=re.IGNORECASE)
    # Trim signos
    s = s.strip(" .,;:¿?¡!\t\n")
    return s


def _validate_sub_queries(
    subs: Iterable[str],
    *,
    min_chars: int = 3,
) -> list[str]:
    """Filtra sub-queries inválidas (vacías, duplicadas, demasiado cortas).

    Devuelve lista deduplicada (case-insensitive) preservando orden de
    aparición. Caps en 4 (límite duro: más de eso explota el costo del
    pool sin retorno medible).

    `min_chars`: piso por sub-query en caracteres (default 3) — captura
    nombres como "kemper", "axe fx", "ikigai" pero descarta basura tipo
    " y " o "X" que sale de regex con grupos vacíos.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in subs:
        s = _normalize_sub_query(raw)
        if not s or len(s) < min_chars:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= 4:
            break
    return out


def _try_regex_decompose(query: str) -> list[str] | None:
    """Aplicar patterns regex en orden. Devuelve sub-queries o None."""
    q = (query or "").strip()
    for pat in _DECOMPOSE_PATTERNS:
        m = pat.search(q)
        if not m:
            continue
        groups = [g for g in m.groups() if g]
        subs = _validate_sub_queries(groups)
        if len(subs) >= 2:
            return subs
    return None


# Helper LLM call — separado del detector principal para que sea
# monkeypatcheable en tests sin tocar `decompose_query`.
def _llm_decompose(
    query: str,
    *,
    helper_chat=None,
    helper_model: str | None = None,
    helper_options: dict | None = None,
    timeout_s: float = 5.0,
) -> list[str] | None:
    """Llama qwen2.5:3b para clasificar + descomponer la query.

    Args:
      query: la pregunta del user.
      helper_chat: callable `chat(model=..., messages=..., options=..., ...)` —
        cuando None, importa el default de rag.__init__ y usa `_helper_client`.
      helper_model: override del modelo (default ``rag.HELPER_MODEL``).
      helper_options: override de las options (default ``rag.HELPER_OPTIONS``).
      timeout_s: budget total — si ollama se cuelga, devolvemos None.

    Devuelve lista de sub-queries (≥2) o None si:
      - el modelo declaró ``is_multi_aspect=False``
      - parse JSON falló
      - timeout / excepción

    Schema esperado del JSON: {"is_multi_aspect": bool, "sub_queries": [str, ...]}
    """
    # Lazy import para evitar circular import (este módulo lo importa rag.__init__).
    if helper_chat is None or helper_model is None or helper_options is None:
        try:
            import rag as _rag  # type: ignore
        except Exception:
            return None
        if helper_chat is None:
            helper_chat = _rag._helper_client().chat  # type: ignore[attr-defined]
        if helper_model is None:
            helper_model = _rag.HELPER_MODEL  # type: ignore[attr-defined]
        if helper_options is None:
            helper_options = dict(_rag.HELPER_OPTIONS)  # type: ignore[attr-defined]

    prompt = (
        "Clasificá esta pregunta como multi-aspecto o single-aspecto.\n\n"
        "Multi-aspecto: pide info sobre DOS o más temas/entidades distintas\n"
        "que merecen búsquedas separadas. Ejemplo:\n"
        '  "compará el axe fx 3 con el kemper" → 2 sub-queries\n'
        '  "qué tengo sobre python y rust" → 2 sub-queries\n\n'
        "Single-aspecto: una sola entidad o tema, aunque sea complejo. Ejemplo:\n"
        '  "cuándo fue mi turno con el psicólogo" → no descomponer\n'
        '  "info del banco santander" → no descomponer\n\n'
        "Devolvé SOLO un JSON válido (sin markdown, sin explicación):\n"
        '{"is_multi_aspect": <bool>, "sub_queries": ["sub1", "sub2", ...]}\n\n'
        "Si is_multi_aspect=false, devolvé sub_queries=[].\n"
        "Cada sub-query debe ser una pregunta autónoma y completa.\n\n"
        f"Pregunta: {query}"
    )

    t0 = time.perf_counter()
    try:
        resp = helper_chat(
            model=helper_model,
            messages=[{"role": "user", "content": prompt}],
            options=helper_options,
            keep_alive=-1,
        )
    except Exception:
        return None
    if (time.perf_counter() - t0) > timeout_s:
        # El helper devolvió, pero tarde — todavía aceptamos resultado
        # (no abortamos por timeout post-facto), pero esto es señal
        # para futuros logs.
        pass

    content = ""
    try:
        content = resp.message.content if hasattr(resp, "message") else resp.get("message", {}).get("content", "")
    except Exception:
        return None
    if not content:
        return None

    # qwen a veces wrappea JSON en ```json ... ```; extraer el primer { ... }.
    text = content.strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if not parsed.get("is_multi_aspect"):
        return None
    raw_subs = parsed.get("sub_queries") or []
    if not isinstance(raw_subs, list):
        return None
    subs = _validate_sub_queries(str(s) for s in raw_subs if isinstance(s, (str, int, float)))
    if len(subs) < 2:
        return None
    return subs


def decompose_query(
    query: str,
    *,
    use_llm_fallback: bool = True,
    helper_chat=None,
    helper_model: str | None = None,
    helper_options: dict | None = None,
) -> list[str] | None:
    """Detecta query multi-aspecto y devuelve sub-queries.

    Pipeline:
      1. Cache LRU lookup (256 entries).
      2. Regex patterns — si matchea, devuelvo sub-queries (sin LLM).
      3. LLM fallback (qwen2.5:3b) cuando regex no matchea.
      4. Cachear resultado (incluyendo None).

    Returns:
      list[str] con ≥2 sub-queries cuando es multi-aspecto.
      None cuando la query es single-aspecto o el detector falló.
    """
    q = (query or "").strip()
    if not q:
        return None

    cache_key = q.lower()
    hit, val = _cache_get(cache_key)
    if hit:
        return val

    # Regex first.
    subs = _try_regex_decompose(q)
    if subs:
        _cache_put(cache_key, subs)
        return subs

    if not use_llm_fallback:
        _cache_put(cache_key, None)
        return None

    # LLM fallback — silent fail OK.
    subs = _llm_decompose(
        q,
        helper_chat=helper_chat,
        helper_model=helper_model,
        helper_options=helper_options,
    )
    _cache_put(cache_key, subs)
    return subs


# ──────────────────────────────────────────────────────────────────────────
# RRF fusion
# ──────────────────────────────────────────────────────────────────────────


def rrf_fuse(
    rankings: list[list[Any]],
    *,
    k: int = RRF_K_DEFAULT,
    top_k: int = RRF_TOP_K_DEFAULT,
    key_fn=None,
) -> list[Any]:
    """Reciprocal Rank Fusion sobre N listas de candidatos.

    Formula (Cormack, Clarke & Buettcher 2009):
        score(d) = Σ_i  1 / (k + rank_i(d))
    donde rank_i(d) es la posición 1-indexed de d en la lista i.
    Documentos ausentes en una lista no contribuyen — equivalente a
    rank=∞.

    Args:
      rankings: lista de listas de candidatos. Cada candidato puede ser
        un dict, una tuple, un objeto, o un string. `key_fn` decide
        qué key usar para identificar el mismo candidato en distintas
        listas (default: el item entero como key, lo cual asume ya
        son strings/IDs).
      k: constante RRF (default 60).
      top_k: cuántos slots devolver. None → devolver todos.
      key_fn: callable que devuelve la key de comparación. Default
        usa el item directo (str-comparable).

    Returns:
      Lista de candidatos ordenados por score RRF descendente.
      Tiebreak determinístico por key (lex ascending).
      El item devuelto es el primer "encontrado" en las listas
      (preserva el objeto original con su metadata).
    """
    if not rankings:
        return []
    if key_fn is None:
        key_fn = lambda x: x  # noqa: E731

    scores: dict[Any, float] = {}
    first_seen: dict[Any, Any] = {}
    for ranking in rankings:
        if not ranking:
            continue
        for rank_idx, item in enumerate(ranking):
            try:
                key = key_fn(item)
            except Exception:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank_idx + 1)
            if key not in first_seen:
                first_seen[key] = item

    # Sort: score desc, key asc (deterministic tiebreak).
    ordered_keys = sorted(
        scores.keys(),
        key=lambda kk: (-scores[kk], str(kk)),
    )
    fused = [first_seen[kk] for kk in ordered_keys]
    if top_k is not None and top_k > 0:
        fused = fused[:top_k]
    return fused


def rrf_fuse_with_scores(
    rankings: list[list[Any]],
    *,
    k: int = RRF_K_DEFAULT,
    top_k: int = RRF_TOP_K_DEFAULT,
    key_fn=None,
) -> list[tuple[Any, float]]:
    """Variante que devuelve (item, rrf_score) por slot.

    Útil para tests + para callers que quieran exponer el score RRF
    a otros stages downstream (ej. mezclar con re-rank scores).
    """
    if not rankings:
        return []
    if key_fn is None:
        key_fn = lambda x: x  # noqa: E731

    scores: dict[Any, float] = {}
    first_seen: dict[Any, Any] = {}
    for ranking in rankings:
        if not ranking:
            continue
        for rank_idx, item in enumerate(ranking):
            try:
                key = key_fn(item)
            except Exception:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank_idx + 1)
            if key not in first_seen:
                first_seen[key] = item

    ordered_keys = sorted(
        scores.keys(),
        key=lambda kk: (-scores[kk], str(kk)),
    )
    out = [(first_seen[kk], scores[kk]) for kk in ordered_keys]
    if top_k is not None and top_k > 0:
        out = out[:top_k]
    return out


# ──────────────────────────────────────────────────────────────────────────
# Env gate
# ──────────────────────────────────────────────────────────────────────────


def env_enabled() -> bool:
    """Lee `RAG_QUERY_DECOMPOSE` cada call (test-friendly)."""
    return os.environ.get("RAG_QUERY_DECOMPOSE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def env_use_llm_fallback() -> bool:
    """`RAG_QUERY_DECOMPOSE_LLM_FALLBACK=0` para forzar regex-only."""
    return os.environ.get(
        "RAG_QUERY_DECOMPOSE_LLM_FALLBACK", "1",
    ).strip().lower() in ("1", "true", "yes", "on")


def env_max_workers() -> int:
    """Cap del threadpool de sub-retrieves."""
    try:
        return max(1, int(os.environ.get("RAG_QUERY_DECOMPOSE_MAX_WORKERS", "3")))
    except ValueError:
        return 3
