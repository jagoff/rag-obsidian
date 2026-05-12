"""Semantic search cross-chat sobre WhatsApp messages (sqlite-vec + MLX).

Complementa a `search.py` (FTS5) con búsqueda por embedding — permite
queries como "menciones a lavarropas en todos los chats" matcheando
por significado en vez de wording exacto. Reusa:

- El mismo `rag_wa_messages_mirror` de telemetry.db (single source of
  truth de contenido).
- El embedder MLX in-process (`rag.embed` → Qwen3-Embedding-0.6B,
  1024 dim, L2-normalizado).
- sqlite-vec extension cargada on-demand sobre la conexión de telemetry
  (no hace falta segunda DB).

Storage:

- `rag_wa_messages_vec` (vec0 virtual table, `embedding float[1024]`).
  El `rowid` matches el `rowid` del mirror — JOIN directo sin column
  extra.

Indexación:

- `index_pending(batch_size)` busca rows del mirror que aún no están
  en vec (`rowid NOT IN (SELECT rowid FROM rag_wa_messages_vec)`),
  embeda en batch y los inserta. Idempotente.
- `index_all_pending()` itera hasta drenar la pending list.

Búsqueda:

- `search(q, jid=None, limit=20)` embeda la query, hace k-NN
  vía `MATCH` + ORDER BY distance, JOIN al mirror para metadata.
- Distance es cosine (vec0 default), 0 = idéntico, 2 = opuesto.
- Filter opcional `jid` para acotar a un chat.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import sqlite_vec  # type: ignore[import-not-found]

logger = logging.getLogger("rag.wa.search_semantic")

_INDEX_BATCH = 64
_VEC_DIM = 1024


_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS rag_wa_messages_vec USING vec0(
    embedding float[1024]
);
"""


def _open_vec_conn() -> sqlite3.Connection:
    """Telemetry DB con sqlite-vec extension cargada.

    `_db_local._connect()` no carga la extensión por design (el resto
    del paquete no la necesita) — la cargamos acá explícitamente. WAL
    + busy_timeout heredados del PRAGMA del connect.
    """
    from . import _db_local

    _db_local.ensure_schema()
    path = _db_local._telemetry_db_path()
    # uri=True habilita ATTACH 'file:...?mode=ro' (lo necesitamos
    # para joinear al bridge messages.db RO).
    con = sqlite3.connect(
        f"file:{path}", uri=True, timeout=15.0, isolation_level=None,
    )
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=15000")
    con.row_factory = sqlite3.Row
    # Idempotente. Si ya existe, no-op.
    con.execute(_SCHEMA_SQL)
    return con


def _attach_bridge(con: sqlite3.Connection) -> bool:
    import rag as _rag

    db_path = _rag.WHATSAPP_DB_PATH
    if not db_path.is_file():
        return False
    try:
        con.execute(f"ATTACH DATABASE 'file:{db_path}?mode=ro' AS br")
        return True
    except sqlite3.Error as e:
        logger.warning("attach bridge failed: %s", e)
        return False


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Wrapper sobre `rag.embed()` con clamp defensivo (cap 4096 chars
    por texto para no saturar el embedder con voice transcripts huge).
    """
    if not texts:
        return []
    import rag as _rag

    capped = [t[:4096] for t in texts]
    return _rag.embed(capped)


def index_pending(batch_size: int = _INDEX_BATCH) -> int:
    """Embed + INSERT la próxima ventana de rows del mirror que aún no
    estén en vec. Devuelve la cantidad insertada.

    Usado por:
    - Backfill inicial (correr en loop hasta que devuelva 0).
    - Daemon de sync incremental (1 call por tick).

    Excluye rows con `content` vacío (vec0 no soporta NULL en MATCH).
    """
    con = _open_vec_conn()
    inserted = 0
    try:
        rows = con.execute(
            """
            SELECT m.rowid AS r, m.content AS content
            FROM rag_wa_messages_mirror m
            WHERE m.content IS NOT NULL
              AND length(trim(m.content)) > 0
              AND m.rowid NOT IN (SELECT rowid FROM rag_wa_messages_vec)
            ORDER BY m.ts DESC
            LIMIT ?
            """,
            (max(1, int(batch_size)),),
        ).fetchall()
        if not rows:
            return 0
        texts = [r["content"] for r in rows]
        vectors = _embed_batch(texts)
        if len(vectors) != len(rows):
            logger.warning(
                "embed mismatch: %d rows vs %d vectors", len(rows), len(vectors),
            )
            return 0
        con.executemany(
            "INSERT INTO rag_wa_messages_vec(rowid, embedding) VALUES (?, ?)",
            [
                (r["r"], sqlite_vec.serialize_float32(v))
                for r, v in zip(rows, vectors)
            ],
        )
        con.commit()
        inserted = len(rows)
    except sqlite3.Error as e:
        logger.warning("index_pending failed: %s", e)
    except Exception as e:  # embedder errors, etc.
        logger.warning("index_pending unexpected: %s", e)
    finally:
        con.close()
    return inserted


def index_all_pending(max_batches: int = 1000) -> int:
    """Drena el pending pool. Devuelve la cantidad total insertada.

    `max_batches` previene loop infinito si el embedder se cuelga; con
    `batch=64` cubre hasta 64k rows. El listener WA actual tiene ~25k
    msgs, así que default cubre el corpus completo con margen.
    """
    total = 0
    for _ in range(max_batches):
        n = index_pending(_INDEX_BATCH)
        if n == 0:
            break
        total += n
    return total


def pending_count() -> int:
    """Cantidad de rows del mirror que aún no están embeddedas.

    Útil para monitoring y CLI. Hot path ~1ms (un EXCEPT count).
    """
    con = _open_vec_conn()
    try:
        row = con.execute(
            """
            SELECT count(*) AS n
            FROM rag_wa_messages_mirror m
            WHERE m.content IS NOT NULL
              AND length(trim(m.content)) > 0
              AND m.rowid NOT IN (SELECT rowid FROM rag_wa_messages_vec)
            """
        ).fetchone()
        return int(row["n"]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        con.close()


def search(
    q: str, jid: str | None = None, limit: int = 20,
) -> list[dict[str, Any]]:
    """k-NN search del corpus de mensajes WA. Devuelve top `limit` hits
    ordenados por similarity (más cerca primero).

    `jid` opcional acota a un chat. Para hybrid (FTS + semantic + RRF)
    usar `search_hybrid`.
    """
    q = (q or "").strip()
    if not q:
        return []
    try:
        vec = _embed_batch([q])
    except Exception as e:
        logger.warning("embed query failed: %s", e)
        return []
    if not vec:
        return []
    cap = max(1, min(int(limit or 20), 100))

    con = _open_vec_conn()
    bridge_ok = _attach_bridge(con)
    try:
        where_extra = ""
        params: list[Any] = [sqlite_vec.serialize_float32(vec[0]), cap]
        if jid:
            where_extra = "AND m.chat_jid = ?"
            params.append(jid)
        chat_join = (
            "LEFT JOIN br.chats c ON c.jid = m.chat_jid" if bridge_ok else ""
        )
        chat_col = "c.name AS chat_name," if bridge_ok else "'' AS chat_name,"
        rows = con.execute(
            f"""
            SELECT
              m.id,
              m.chat_jid,
              m.ts,
              m.sender,
              m.content,
              {chat_col}
              v.distance
            FROM rag_wa_messages_vec v
            JOIN rag_wa_messages_mirror m ON m.rowid = v.rowid
            {chat_join}
            WHERE v.embedding MATCH ?
              AND k = ?
              {where_extra}
            ORDER BY v.distance ASC
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r["id"],
                "chat_jid": r["chat_jid"],
                "chat_name": r["chat_name"] or "",
                "ts": (r["ts"] or "").replace(" ", "T", 1),
                "sender": r["sender"] or "",
                "content": r["content"] or "",
                "distance": float(r["distance"] or 0.0),
            }
            for r in rows
        ]
    except sqlite3.Error as e:
        logger.warning("search_semantic failed: %s", e)
        return []
    finally:
        if bridge_ok:
            try:
                con.execute("DETACH DATABASE br")
            except sqlite3.Error:
                pass
        con.close()


def search_hybrid(
    q: str, jid: str | None = None, limit: int = 20, k: int = 60,
) -> list[dict[str, Any]]:
    """Hybrid search FTS + semantic con RRF (Reciprocal Rank Fusion).

    Combina resultados de FTS (búsqueda exacta) y semantic (búsqueda por
    embeddings) usando RRF para fusionar rankings. RRF es robusto a
    diferencias de escala entre sistemas y funciona bien sin tuning.

    Fórmula RRF: score(d) = Σ 1 / (k + rank_i(d))
    Donde k=60 es una constante estándar que suaviza el impacto de rankings extremos.

    Args:
        q: query string
        jid: opcional, acota a un chat específico
        limit: cantidad de resultados a devolver
        k: constante RRF (default 60, valor estándar en la literatura)

    Returns:
        Lista de hits ordenados por score RRF descendente, con metadata
        de ambos sistemas (distance para semantic, snippet para FTS).
    """
    q = (q or "").strip()
    if not q:
        return []

    # Ejecutar ambas búsquedas con pool más grande para RRF
    pool_size = limit * 3  # Pool más grande para tener suficientes candidatos

    # FTS search
    from . import search as _wa_search
    fts_hits = _wa_search.search(q, jid=jid, limit=pool_size)

    # Semantic search
    semantic_hits = search(q, jid=jid, limit=pool_size)

    # RRF fusion
    scores: dict[str, float] = {}
    fts_by_id: dict[str, dict] = {}
    semantic_by_id: dict[str, dict] = {}

    # Score FTS results
    for rank, hit in enumerate(fts_hits, 1):
        msg_id = hit["id"]
        scores[msg_id] = scores.get(msg_id, 0) + 1 / (k + rank)
        fts_by_id[msg_id] = hit

    # Score semantic results
    for rank, hit in enumerate(semantic_hits, 1):
        msg_id = hit["id"]
        scores[msg_id] = scores.get(msg_id, 0) + 1 / (k + rank)
        semantic_by_id[msg_id] = hit

    # Merge metadata y ordenar por score RRF
    merged: list[dict[str, Any]] = []
    for msg_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        fts_hit = fts_by_id.get(msg_id, {})
        semantic_hit = semantic_by_id.get(msg_id, {})

        # Preferir metadata de semantic (tiene más campos), fallback a FTS
        base = semantic_hit if semantic_hit else fts_hit
        if not base:
            continue

        merged.append({
            "id": base["id"],
            "chat_jid": base["chat_jid"],
            "chat_name": base.get("chat_name", ""),
            "ts": base["ts"],
            "sender": base.get("sender", ""),
            "content": base.get("content", ""),
            "snippet": fts_hit.get("snippet", ""),  # Snippet solo de FTS
            "distance": semantic_hit.get("distance"),  # Distance solo de semantic
            "rrf_score": rrf_score,
        })

    return merged[:limit]


__all__ = [
    "index_pending",
    "index_all_pending",
    "pending_count",
    "search",
    "search_hybrid",
]
