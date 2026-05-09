"""Anaphora resolver — Quick Win #1, 2026-05-04.

Phase 5 cont de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer el resolver de anáforas desde `rag/__init__.py`.

## Diseño

Quick-win dedicated upstream del reformulate. Detecta queries cortas
tipo "y en Madrid?", "y para mañana?", "y la otra opción?" y las
re-escribe con contexto del turno previo ANTES de llegar a `retrieve()`.

- **Detector regex-only** (microsegundos): decide si vale gastar 1
  helper LLM call. False cuando la query es self-contained
  (≥8 tokens y no empieza con conector). 0% overhead cuando no aplica.
- **Resolver** llama qwen2.5:3b con HELPER_OPTIONS (deterministic,
  seed=42). Cache LRU 128 entries por (history_hash, query).
- **Silent-fail** → return original query (mismo contract que
  `reformulate_query`).
- **Gate**: `RAG_ANAPHORA_RESOLVER` (default ON).

## Lazy imports

`_helper_client`, `HELPER_MODEL`, `HELPER_OPTIONS`, `LLM_KEEP_ALIVE`,
`_silent_log` viven en `rag/__init__.py`. Lazy adentro de
`_cached_anaphora_resolution` para evitar circular import.

## Re-export

`rag/__init__.py` hace `from rag._anaphora import *  # noqa`.
Preserva 100% compat con `rag._is_anaphoric_query`,
`rag._resolve_anaphora`, `rag._anaphora_resolver_enabled`.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re

__all__ = [
    "_ANAPHORA_CONNECTORS_RE",
    "_ANAPHORA_SHORT_TOKEN_THRESHOLD",
    "_is_anaphoric_query",
    "_cached_anaphora_resolution",
    "_resolve_anaphora",
    "_anaphora_resolver_enabled",
]


# Conectores típicos en español rioplatense que indican follow-up con
# referencia anafórica al turno anterior.
_ANAPHORA_CONNECTORS_RE = re.compile(
    r"^\s*(?:"
    r"y\s+para\b"
    r"|y\s+en\b"
    r"|y\s+de\b"
    r"|y\s+la\b"
    r"|y\s+el\b"
    r"|y\s+los\b"
    r"|y\s+las\b"
    r"|y\s+eso\b"
    r"|y\s+otro\b"
    r"|y\s+otra\b"
    r"|también\b"
    r"|tambien\b"
    r"|ahora\b"
    r"|pero\b"
    r"|y\b"
    r")",
    re.IGNORECASE,
)

# Token threshold debajo del cual la query es "corta" y candidata a
# anaphora resolution incluso sin un conector explícito.
_ANAPHORA_SHORT_TOKEN_THRESHOLD = 8


def _is_anaphoric_query(query: str, history: list[dict] | None) -> bool:
    """True si la query parece referirse al turno previo y necesita
    expansión usando contexto.

    Reglas:
      - Si no hay historial → False (no hay nada a qué referirse).
      - Si la query empieza con un conector (`y`, `pero`, `también`,
        `ahora`, `y para`, `y en`, etc.) → True.
      - Si la query tiene <8 tokens → True (queries cortas en chats
        casi siempre son follow-ups).
      - En cualquier otro caso → False (self-contained).

    No llama al LLM. Microsegundos. Seguro para usar inline en el hot path.
    """
    if not query or not history or len(history) < 1:
        return False
    q_stripped = query.strip()
    if not q_stripped:
        return False
    if _ANAPHORA_CONNECTORS_RE.match(q_stripped):
        return True
    tokens = q_stripped.split()
    if len(tokens) < _ANAPHORA_SHORT_TOKEN_THRESHOLD:
        return True
    return False


# LRU cache 128 entries por (history_hash, query).
@functools.lru_cache(maxsize=128)
def _cached_anaphora_resolution(
    history_hash: str, query: str, history_blob: str,
) -> str:
    """Backing store del LRU cache de `_resolve_anaphora`."""
    from rag import (  # noqa: PLC0415
        HELPER_MODEL,
        HELPER_OPTIONS,
        LLM_KEEP_ALIVE,
        _helper_client,
        _silent_log,
    )

    prompt = (
        "Tu tarea: reescribir la query actual del usuario expandiendo "
        "cualquier referencia anafórica (pronombres, demostrativos, "
        "elipsis) usando el historial. Reglas:\n"
        "1. Si la query ya es self-contained (todas las entidades "
        "explícitas, sin conectores), devolvela tal cual.\n"
        "2. Si la query empieza con un conector ('y', 'pero', 'también', "
        "'ahora', 'y para', 'y en', etc.), expandí la referencia: tomá "
        "la entidad o tema del último turno y completala.\n"
        "3. NO inventes entidades que no aparezcan en el historial o la "
        "query.\n"
        "4. Mantené el registro y la longitud aproximada.\n\n"
        f"Historial:\n{history_blob}\n\n"
        f"Query actual: \"{query}\"\n\n"
        "Respondé SOLO con la query reescrita, sin explicación ni comillas."
    )
    try:
        resp = _helper_client().chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={**HELPER_OPTIONS, "num_ctx": 1024, "num_predict": 96},
            keep_alive=LLM_KEEP_ALIVE,
        )
        raw = (resp.message.content or "").strip().strip('"').strip("'")
        if not raw:
            return query
        if len(raw) > 3 * max(len(query), 30):
            return query
        return raw
    except Exception as exc:
        try:
            _silent_log("anaphora_resolver_failed", exc)
        except Exception:
            pass
        return query


def _resolve_anaphora(query: str, history: list[dict]) -> str:
    """Reescribe `query` expandiendo referencias anafóricas usando los
    últimos turnos de `history`. Cache LRU 128 entries."""
    if not query or not history:
        return query
    recent = history[-4:]
    history_blob = "\n".join(
        f"{'Usuario' if m.get('role') == 'user' else 'Asistente'}: "
        f"{(m.get('content') or '')[:240]}"
        for m in recent
    )
    if not history_blob.strip():
        return query
    history_hash = hashlib.sha256(
        history_blob.encode("utf-8", errors="replace"),
    ).hexdigest()[:16]
    return _cached_anaphora_resolution(history_hash, query, history_blob)


def _anaphora_resolver_enabled() -> bool:
    """Gate: `RAG_ANAPHORA_RESOLVER` env var. Default ON."""
    val = os.environ.get("RAG_ANAPHORA_RESOLVER", "").strip().lower()
    return val not in ("0", "false", "no")
