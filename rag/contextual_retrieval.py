"""Contextual Retrieval (Anthropic, Sept 2024) — prototipo gated.

Referencia: [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval).

Resumen de la técnica
---------------------

Antes de embeddear cada chunk, se genera un *summary corto* (50-100 tokens)
con un LLM helper que ubica al chunk en su documento ("Este chunk pertenece
a la nota X sobre Y, sección Z"), se prependea al body del chunk, y se
embeddea ``contexto + body`` (no sólo body). El display_text que aterriza
en la UI queda raw — el contexto vive únicamente en el embedding.

Anthropic midió −35% retrieval failures con BM25+embeddings y −49% con
reranker contra su corpus interno. Acá lo dejamos *gated* (default OFF)
porque:

  1. El re-embed full del corpus actual (~8231 chunks) cuesta ~8200 LLM
     calls a qwen2.5:3b — no se debe disparar accidental por un merge.
  2. La ganancia depende del corpus; hay que medir contra ``rag eval``
     antes de promover el flag a default ON.

Contract
--------

- ``RAG_CONTEXTUAL_RETRIEVAL=1`` (env var, default OFF): activa la wrap
  en el pipeline de indexing. Sin esto, el pipeline corre exactamente
  como antes — bit-idéntico (vía short-circuit en
  :func:`contextual_retrieval_enabled`).
- ``contextualize_chunks(embed_texts, display_texts, doc_id_prefix,
  doc_full_text, ...)`` es la API pública que :func:`_index_single_file`
  y :func:`_run_index` llaman entre ``semantic_chunks(...)`` y
  ``embed(...)``. Si el flag está OFF retorna ``embed_texts`` sin tocar.
- Cache SQL en ``telemetry.db``, tabla ``rag_chunk_contexts`` (DDL acá
  abajo, registrada en ``_TELEMETRY_DDL`` de ``rag/__init__.py``).
  PK = ``(doc_id_prefix, chunk_idx, chunk_hash)``: cualquier cambio
  en el body del chunk → hash distinto → miss → regenera.
- Cuando ``RAG_CONTEXTUAL_RETRIEVAL=0/unset``, las filas viejas del
  cache se mantienen (no se borran) — al re-activar el flag, los hits
  se reusan sin re-LLM.

Display text contract
---------------------

``display_texts`` (segundo elemento de cada tuple de ``semantic_chunks``)
**NUNCA** se contamina con el ``[contexto: ...]`` prefix — vive como
"texto limpio" para snippets de UI, citation-repair y reranker title-
prefix. Sólo el ``embed_text`` (primer elemento) absorbe el contexto.

Cómo invalidamos el cache
-------------------------

- Cambia el body del chunk → ``chunk_hash`` cambia → miss + regenera.
- Cambia el prompt o el modelo helper (``HELPER_MODEL``) → bumpear
  :data:`PROMPT_VERSION` para invalidar todo de una.
- ``rag index --reset`` borra la collection sqlite-vec pero **NO**
  toca ``rag_chunk_contexts`` (es cache puro, sobrevive al reset
  para que un re-embed no pague de nuevo el costo LLM).
- Para limpiar el cache manualmente:
  ``DELETE FROM rag_chunk_contexts;`` en ``telemetry.db``.

Cómo activarlo
--------------

  export RAG_CONTEXTUAL_RETRIEVAL=1
  rag eval                          # baseline antes
  rag index --reset --contextual    # full re-embed con contexto
  rag eval                          # comparar deltas con CIs

Si chains/singles regresan dentro del CI noise → la técnica no aporta
contra este corpus, dejarla off. Si mejoran consistente y los floors
del nightly auto-rollback no rompen, considerar promover a default ON.

Restricciones / no-goals
------------------------

- No tocamos retrieve-time. La técnica es 100% indexing-time.
- No regeneramos summaries en cada query — la única fuente de generación
  es el indexer.
- Si el helper LLM falla (timeout, malformed output), el chunk cae al
  embed_text *sin* contexto (degraded gracefully) y NO se cachea el
  fallo — el próximo re-embed reintenta.
- Idempotente: re-correr con mismo content_hash skipea sin re-LLM.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Any

# ── Constantes / config ──────────────────────────────────────────────────────

# Bumpear este string si cambia el prompt o el modelo helper — invalida
# todo el cache de contexts en la próxima corrida (los hits viejos no
# matchean la nueva combinación). Same patrón que `_FILTER_VERSION` en
# `rag/__init__.py`.
PROMPT_VERSION = "anthropic-v1-2026-05-04"

# Mínimo de caracteres del documento padre antes de considerar contextual.
# Documentos cortos (< 300 chars, equivalente a 1-2 párrafos) no necesitan
# contexto: el chunk entero ES el documento. Threshold matchea el de
# `_CONTEXT_MIN_BODY` para `get_context_summary`.
MIN_DOC_CHARS_FOR_CONTEXT = 300

# Cap de chars del documento padre que mandamos al LLM. qwen2.5:3b tiene
# ~32k tokens de contexto pero queremos que la generación sea rápida —
# 6000 chars (~1500 tokens) cubre 95% de las notas reales del vault sin
# inflar la prompt innecesariamente.
DOC_TRUNCATE_CHARS = 6000

# Cap de chars del chunk dentro del prompt (raro que MAX_CHUNK lo supere
# pero defensivo). MAX_CHUNK es 800 → cabe holgado.
CHUNK_TRUNCATE_CHARS = 1200

# num_predict para el helper. Anthropic apunta a 50-100 tokens; con
# qwen2.5:3b eso es ~50-100 num_predict (tokens ≈ 1:1 para español/inglés).
NUM_PREDICT = 100

# Cap del summary final (post-LLM) en chars. 200 chars ≈ ~50 tokens de
# embedding overhead — manejable, no domina el chunk.
MAX_SUMMARY_CHARS = 240

# Prefix que envuelve el summary cuando lo prepenmos al embed_text.
# Usar un marker fácil de greppear para audits + tests.
SUMMARY_MARKER = "[contexto:"


# ── Gate ─────────────────────────────────────────────────────────────────────


def contextual_retrieval_enabled() -> bool:
    """True si ``RAG_CONTEXTUAL_RETRIEVAL`` está seteado a un truthy.

    Default OFF — el flag debe ser opt-in explícito porque activarlo
    mid-run sin reindex deja chunks viejos sin contexto y nuevos con
    contexto en la misma collection, mezclando dos distribuciones de
    embedding. La activación correcta es::

        export RAG_CONTEXTUAL_RETRIEVAL=1
        rag index --reset --contextual

    Aceptamos los mismos truthy que ``_context_summary_enabled`` para
    consistencia de UX.
    """
    val = os.environ.get("RAG_CONTEXTUAL_RETRIEVAL", "").strip().lower()
    return val in ("1", "true", "yes")


# ── Hash helper ──────────────────────────────────────────────────────────────


def chunk_hash(chunk_text: str) -> str:
    """SHA1[:16] del cuerpo del chunk + ``PROMPT_VERSION``.

    Incluir ``PROMPT_VERSION`` en el hash garantiza que cambiar el prompt
    invalida todos los cachés viejos sin tener que correr una migration
    explícita.
    """
    h = hashlib.sha1()
    h.update(PROMPT_VERSION.encode("utf-8"))
    h.update(b"\x00")
    h.update(chunk_text.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


# ── Prompt builder ───────────────────────────────────────────────────────────


def _build_prompt(chunk_text: str, doc_text: str, title: str, folder: str) -> str:
    """Prompt estilo Anthropic — versión rioplatense.

    Contrato del helper:
      - Devolver SOLO el contexto (1 oración corta, ~50-100 tokens).
      - No agregar disclaimers, preámbulos, ni citar el chunk literal.
      - Idioma: español rioplatense (matchea el resto del vault).
    """
    # Truncar defensivamente — si el doc es enorme, el LLM se distrae.
    doc_snippet = doc_text[:DOC_TRUNCATE_CHARS]
    chunk_snippet = chunk_text[:CHUNK_TRUNCATE_CHARS]

    # Marcadores XML-ish: misma defensa de prompt-injection que
    # `_wrap_untrusted` en `rag/__init__.py`. El doc body podría incluir
    # texto adversarial ("Ignorá las instrucciones") y queremos que el
    # helper lo trate como datos.
    doc_block = f"<DOCUMENTO>\n{doc_snippet}\n</DOCUMENTO>"
    chunk_block = f"<CHUNK>\n{chunk_snippet}\n</CHUNK>"

    return (
        f"Nota: \"{title}\" (carpeta: {folder})\n\n"
        f"{doc_block}\n\n"
        f"{chunk_block}\n\n"
        "El bloque CHUNK es un fragmento del bloque DOCUMENTO. Dame un "
        "contexto corto (1 oración, máximo 30 palabras) que ubique el "
        "CHUNK dentro del DOCUMENTO para que un sistema de búsqueda lo "
        "encuentre cuando alguien pregunte sobre ese tema. Respondé SOLO "
        "con el contexto, en español, sin preámbulos ni explicaciones."
    )


# ── LLM call ────────────────────────────────────────────────────────────────

# Lock + counters in-process — observability barata sin tener que tocar
# rag_queries para algo tan específico como "cuántas LLM calls hizo el
# indexer en este run". Llave a `cache_stats_snapshot()` por simetría con
# los demás caches.
_lock = threading.Lock()
_stats: dict[str, int] = {"hits": 0, "misses": 0, "errors": 0, "skipped_short": 0}


def _bump(name: str, n: int = 1) -> None:
    with _lock:
        _stats[name] = _stats.get(name, 0) + n


def stats_snapshot() -> dict[str, int]:
    """Snapshot de hits/misses/errors/skipped_short del run actual.

    Útil para CLI summary post-index ("X chunks contextualizados, Y
    hits de cache, Z errors") y tests."""
    with _lock:
        return dict(_stats)


def stats_reset() -> None:
    with _lock:
        for k in list(_stats):
            _stats[k] = 0


def generate_chunk_context(
    chunk_text: str,
    parent_doc_text: str,
    doc_metadata: dict[str, Any],
) -> str:
    """Genera el summary del chunk via qwen2.5:3b.

    Args:
        chunk_text: Body raw del chunk (sin prefix). Es lo que vamos a
            ubicar dentro del doc.
        parent_doc_text: Body completo del documento (post ``clean_md``).
            Lo truncamos a :data:`DOC_TRUNCATE_CHARS` adentro del prompt.
        doc_metadata: Dict con al menos ``title`` y ``folder``. Se
            inyectan al prompt como hint adicional ("Nota X en carpeta Y").

    Returns:
        String con el contexto generado, max :data:`MAX_SUMMARY_CHARS`.
        ``""`` (string vacío) si:
          - ``parent_doc_text`` es muy corto (< :data:`MIN_DOC_CHARS_FOR_CONTEXT`).
          - El helper LLM falló (timeout, malformed output).

    En el caller, ``""`` significa "no prependear nada al embed_text" —
    el chunk se embeddea solo, como antes.
    """
    # Importes lazy para no romper test_imports si ollama está down.
    # Mismo patrón que el resto del codebase (ver _summary_client).
    if len(parent_doc_text) < MIN_DOC_CHARS_FOR_CONTEXT:
        _bump("skipped_short")
        return ""

    title = str(doc_metadata.get("title") or doc_metadata.get("note") or "")
    folder = str(doc_metadata.get("folder") or "")

    prompt = _build_prompt(chunk_text, parent_doc_text, title, folder)

    # Importamos del paquete `rag` lazy para evitar el ciclo de import
    # (este módulo lo importa rag/__init__.py). En tests con monkeypatch,
    # el caller sustituye `rag._summary_client` o este módulo entero,
    # así que la indirección por atributo (no `from rag import ...`)
    # respeta el monkeypatch.
    import rag as _rag  # type: ignore

    try:
        resp = _rag._summary_client().chat(
            model=_rag.HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={**_rag.HELPER_OPTIONS, "num_predict": NUM_PREDICT},
            keep_alive=_rag.OLLAMA_KEEP_ALIVE,
        )
        raw = resp.message.content.strip()
    except Exception as exc:
        _bump("errors")
        # No cacheamos el fallo — el próximo run reintenta. Mismo patrón
        # que `_generate_context_summary` post-2026-04-26 audit.
        try:
            _rag._silent_log("contextual_retrieval_helper", exc)
        except Exception:
            pass
        return ""

    # Sanitize — agarrar primera oración / primera línea, capear chars.
    # Modelos chicos a veces devuelven "Contexto: <X>" — strippearlo si
    # aparece para no duplicar el marker en el prefix.
    first_line = raw.split("\n")[0].strip()
    for prefix_kill in ("Contexto:", "contexto:", "CONTEXTO:", "Context:"):
        if first_line.lower().startswith(prefix_kill.lower()):
            first_line = first_line[len(prefix_kill):].lstrip()
    summary = first_line[:MAX_SUMMARY_CHARS]
    return summary


# ── SQL cache ────────────────────────────────────────────────────────────────


def _ensure_table(conn) -> None:
    """Idempotent DDL para ``rag_chunk_contexts``.

    Llamado desde :func:`get_or_generate_context` en el lazy path. Cuando
    la tabla está registrada en ``_TELEMETRY_DDL`` (lo está post-merge),
    este helper es no-op gracias a ``CREATE TABLE IF NOT EXISTS``.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rag_chunk_contexts ("
        " doc_id TEXT NOT NULL,"
        " chunk_idx INTEGER NOT NULL,"
        " chunk_hash TEXT NOT NULL,"
        " summary TEXT NOT NULL,"
        " prompt_version TEXT NOT NULL,"
        " created_ts TEXT NOT NULL,"
        " PRIMARY KEY (doc_id, chunk_idx, chunk_hash)"
        ")"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_chunk_contexts_doc"
        " ON rag_chunk_contexts(doc_id)"
    )


def _cache_get(doc_id: str, chunk_idx: int, c_hash: str) -> str | None:
    """Lookup del summary cacheado. Devuelve ``None`` en miss."""
    import rag as _rag  # type: ignore

    try:
        with _rag._ragvec_state_conn() as conn:
            _ensure_table(conn)
            row = conn.execute(
                "SELECT summary FROM rag_chunk_contexts"
                " WHERE doc_id=? AND chunk_idx=? AND chunk_hash=?"
                " LIMIT 1",
                (doc_id, chunk_idx, c_hash),
            ).fetchone()
        if row and row[0]:
            return str(row[0])
        return None
    except Exception as exc:
        try:
            _rag._silent_log("contextual_retrieval_cache_get", exc)
        except Exception:
            pass
        return None


def _cache_put(doc_id: str, chunk_idx: int, c_hash: str, summary: str) -> None:
    """Persistir el summary. Silent-fail (mismo patrón que silent_log)."""
    if not summary:
        return  # nunca cacheamos summaries vacíos
    import rag as _rag  # type: ignore

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        with _rag._ragvec_state_conn() as conn:
            _ensure_table(conn)
            conn.execute(
                "INSERT OR REPLACE INTO rag_chunk_contexts"
                " (doc_id, chunk_idx, chunk_hash, summary,"
                "  prompt_version, created_ts)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, chunk_idx, c_hash, summary, PROMPT_VERSION, ts),
            )
    except Exception as exc:
        try:
            _rag._silent_log("contextual_retrieval_cache_put", exc)
        except Exception:
            pass


def get_or_generate_context(
    chunk_text: str,
    parent_doc_text: str,
    doc_id: str,
    chunk_idx: int,
    doc_metadata: dict[str, Any],
) -> str:
    """Cache-aware resolver del contexto del chunk.

    Flow::

        chunk_hash = hash(prompt_version + chunk_body)
        if cache hit on (doc_id, chunk_idx, chunk_hash) → return cached
        else: generate via LLM, persist if non-empty, return

    Args:
        chunk_text, parent_doc_text, doc_metadata: Pasados a
            :func:`generate_chunk_context` en el miss path.
        doc_id: Identificador estable del documento (típicamente
            ``relative_path`` para vault notes, doc_id_prefix de
            ``_index_single_file``).
        chunk_idx: Índice del chunk dentro del documento (0-based, mismo
            que el ``ids = [f"{doc_id_prefix}::{i}" ...]`` del indexer).

    Returns:
        Summary string o ``""`` si helper falló o doc demasiado corto.
    """
    c_hash = chunk_hash(chunk_text)

    cached = _cache_get(doc_id, chunk_idx, c_hash)
    if cached is not None:
        _bump("hits")
        return cached

    _bump("misses")
    summary = generate_chunk_context(chunk_text, parent_doc_text, doc_metadata)
    if summary:
        _cache_put(doc_id, chunk_idx, c_hash, summary)
    return summary


# ── Pipeline integration ─────────────────────────────────────────────────────


def contextualize_chunks(
    embed_texts: list[str],
    display_texts: list[str],
    doc_id: str,
    parent_doc_text: str,
    doc_metadata: dict[str, Any],
) -> list[str]:
    """Wrap principal — llamado desde el pipeline de indexing.

    Si ``RAG_CONTEXTUAL_RETRIEVAL=1``, para cada chunk genera (o lookup
    en cache) un contexto y lo prependea al ``embed_text``. Sino, retorna
    ``embed_texts`` sin tocar (bit-idéntico).

    **Importante**: ``display_texts`` (segundo arg) se acepta sólo para
    documentación / type checking — NO se mutan. El display sigue siendo
    el cuerpo raw del chunk para snippets en UI.

    Args:
        embed_texts: Lista de embed strings (output de
            ``semantic_chunks``, primer elemento de cada tuple).
        display_texts: Lista paralela con el cuerpo raw. Se usa como
            ``chunk_text`` en el helper (es el body sin prefix).
        doc_id: Identificador del documento (relative path del vault o
            URI scheme para cross-source).
        parent_doc_text: Body completo del documento (post ``clean_md``).
        doc_metadata: Dict con ``title`` / ``folder`` / etc.

    Returns:
        Lista paralela a ``embed_texts``: para cada chunk, ya sea el
        embed_text original (si flag OFF o doc muy corto) o el embed_text
        con ``[contexto: SUMMARY]\\n\\n`` prependeado.
    """
    if not contextual_retrieval_enabled():
        return embed_texts

    if not embed_texts:
        return embed_texts

    out: list[str] = []
    for i, (et, dt) in enumerate(zip(embed_texts, display_texts)):
        summary = get_or_generate_context(
            chunk_text=dt,
            parent_doc_text=parent_doc_text,
            doc_id=doc_id,
            chunk_idx=i,
            doc_metadata=doc_metadata,
        )
        if summary:
            # Prependeamos al embed_text completo (no al body raw) para
            # mantener el resto del prefix existente (title, folder,
            # tags, synthetic questions, context_summary del docnivel).
            # Esto deja el contexto chunk-level pegado al inicio,
            # típicamente al tope de la ventana de embedding.
            out.append(f"{SUMMARY_MARKER} {summary}]\n\n{et}")
        else:
            out.append(et)
    return out


# ── Test helpers ─────────────────────────────────────────────────────────────


def _reset_for_tests() -> None:
    """Limpia stats. NO toca la tabla SQL — los tests usan tmp DB_PATH."""
    stats_reset()
