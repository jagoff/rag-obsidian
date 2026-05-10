"""Semantic response cache — extracted from `rag/__init__.py` (Wave 4 split, 2026-05-10).

Caches full responses keyed by query embedding + corpus_hash, so repeated or
near-duplicate queries against an unchanged vault skip the full retrieve +
generate pipeline (1-15s) and return in <100ms.

## Lookup contract

`semantic_cache_lookup(q_embedding, corpus_hash, *, cosine_threshold, now,
return_probe=False)` — find row with cosine ≥ threshold within same
`corpus_hash` AND within TTL. Returns dict with `{question, response,
paths, scores, top_score, intent, cosine, cached_ts, age_seconds}` or
`None`. With `return_probe=True` returns `(hit_or_None, probe_dict)`.

Probe shape:

```
{
  "result":      "hit" | "miss" | "disabled" | "empty_corpus_hash",
  "reason":      "match" | "below_threshold" | "ttl_expired"
               | "corpus_mismatch" | "stale_source" | "dim_mismatch"
               | "empty_cache" | "cache_disabled" | "no_corpus_hash"
               | "db_error",
  "top_cosine":  float | None,
  "candidates":  int,
  "skipped_stale": int,
  "skipped_ttl":   int,
}
```

## Store contract

`semantic_cache_store(q_embedding, *, question, response, paths, scores,
top_score, intent, corpus_hash, ttl_seconds, extra, background)` — persist
row. Skips silently if cache disabled / corpus_hash empty / response empty
/ refusal / `top_score < 0.015`. `background=True` queues on SQL writer
worker thread; `background=False` (default) blocks until commit.

## Invariants

- TTL by intent: `recent`/`agenda` → `_SEMANTIC_CACHE_RECENT_TTL` (default
  600s = 10 min, vault churn matters); else
  `_SEMANTIC_CACHE_DEFAULT_TTL` (default 86400s = 24h).
- Per-entry staleness: `_cached_entry_is_stale` checks each cited path's
  mtime > cached_ts. Catches individual note edits without forcing
  corpus-level invalidation.
- Cosine threshold default 0.93 (override `RAG_CACHE_COSINE`). Below 0.97
  (where bge-m3 puts paraphrases) and above 0.80 (unrelated query
  cosine) — sweet spot empíricamente.
- Refusal poisoning prevention: `_is_refusal()` check on response BEFORE
  store. Regression 2026-04-22: top_score 1.18 + LLM refuse cached
  perpetuamente → cache poisoning.
- Bucket size `_CORPUS_HASH_BUCKET=500` chunks. Below 500 the count
  fluctuates with each ingester run and creates 0% hit rate (medido
  audit 2026-04-30).
- `_FILTER_VERSION` lives in `rag/__init__.py` (cross-cutting Wave-8
  invariant); imported lazily acá. Bumpear allá invalida automáticamente
  todas las entries del cache (los rows quedan con bucket viejo y
  `corpus_hash` no matchea).

## Cross-module deps (lazy imports)

- `_ragvec_state_conn` (DB conn manager).
- `_log_sql_state_error` (silent log writer).
- `_sql_write_with_retry`, `_enqueue_background_sql` (write infra).
- `record_cache_event` (cache hit/miss telemetry).
- `_is_refusal` (refusal detector).
- `_resolve_vault_path` (file mtime checks).
- `_FILTER_VERSION` (filter generation tag).

Imports are inside functions to evitar circular import — `rag/__init__.py`
re-exports response_cache via wildcard.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone

__all__ = [
    "_CORPUS_HASH_BUCKET",
    "_SEMANTIC_CACHE_COSINE",
    "_SEMANTIC_CACHE_DEFAULT_TTL",
    "_SEMANTIC_CACHE_ENABLED",
    "_SEMANTIC_CACHE_MAX_ROWS",
    "_SEMANTIC_CACHE_RECENT_TTL",
    "_blob_to_embedding",
    "_cached_entry_is_stale",
    "_compute_corpus_hash",
    "_corpus_hash_cached",
    "_corpus_hash_lock",
    "_corpus_hash_memo",
    "_embedding_to_blob",
    "_hash_chunk_count",
    "_semantic_cache_enabled",
    "_ttl_for_intent",
    "semantic_cache_clear",
    "semantic_cache_lookup",
    "semantic_cache_stats",
    "semantic_cache_store",
]


# ── Tuning knobs (env override) ─────────────────────────────────────────────
#
# Defaults balance correctness vs hit rate. Tuning history:
#
# **2026-04-22**: threshold 0.97 → 0.93 después del audit que encontró 0.00%
# hit rate sobre 1056 queries en 7 días, con 14 queries repeated ≥10×.
# bge-m3 puts same-meaning paraphrases at ~0.93-0.96; at 0.97 those miss.
# 0.93 is sweet spot — false positives stay rare (unrelated queries
# typically cosine ~0.80). Override `RAG_CACHE_COSINE`.

_SEMANTIC_CACHE_COSINE = float(os.environ.get("RAG_CACHE_COSINE", "0.93"))
_SEMANTIC_CACHE_DEFAULT_TTL = int(os.environ.get("RAG_CACHE_TTL_DEFAULT", "86400"))  # 24h
_SEMANTIC_CACHE_RECENT_TTL = int(os.environ.get("RAG_CACHE_TTL_RECENT", "600"))  # 10 min
_SEMANTIC_CACHE_MAX_ROWS = int(os.environ.get("RAG_CACHE_MAX_ROWS", "2000"))
_SEMANTIC_CACHE_ENABLED = os.environ.get("RAG_CACHE_ENABLED", "1").strip() not in ("0", "false", "no", "")


def _semantic_cache_enabled() -> bool:
    """Check at call-time so tests can flip the env without re-importing."""
    val = os.environ.get("RAG_CACHE_ENABLED", "1").strip().lower()
    return val not in ("0", "false", "no")


# In-memory memo of corpus_hash so we don't re-stat files on every query.
# Invalidated by chunk-count delta — same invariant `_load_corpus()` already
# uses so the cache follows the same life-cycle as the BM25/vocab cache.
_corpus_hash_memo: dict = {"hash": None, "chunk_count": None, "last_ts": 0.0}
_corpus_hash_lock = threading.Lock()


# Bucket size para el corpus_hash — fingerprint cambia cuando count cruza
# un múltiplo de _CORPUS_HASH_BUCKET. Pre-2026-04-24 era count exact, lo
# que hacía que ingesters continuos (whatsapp every 30min, calendar incremental)
# fluctuaran el count constantemente y cada query veía un hash diferente:
# 30 SEMANTIC PUT events en web.log con 24 hashes distintos → cache nunca
# hiteaba porque cada lookup venía con un corpus_hash recién minteado y no
# había rows con ese hash. Con bucket=100 y ~8K chunks, agregar/borrar 50
# notas no invalida el cache; agregar 100 sí.
#
# 2026-04-30 audit: bucket=100 seguía causando 0% hit-rate. El corpus
# tiene 3595 chunks y los ingesters (WA hourly, Calendar 6h, Reminders 6h)
# agregan ~30-80 chunks por run → 2-3 cruzadas de bucket por día → 3
# hashes distintos en 24h (medido: rag_response_cache muestra c6c1ba/
# 98a50f/2907da en el mismo día). Ninguna query lookup-ea con el mismo hash
# que el que se usó para el store. Fix: subir a 500 chunks por bucket.
# Con corpus de ~3600 chunks, un bucket cubre el 14% del corpus — cruzar
# requiere agregar 500 chunks netos (~1 `rag index --reset` o ~1 semana
# de ingesters incrementales). Tradeoff aceptable: el per-entry staleness
# check (ver `_cached_entry_is_stale`) ya cubre edits a paths citadas.
# Override: RAG_CORPUS_HASH_BUCKET=N env var.
_CORPUS_HASH_BUCKET: int = int(os.environ.get("RAG_CORPUS_HASH_BUCKET", "500"))


def _hash_chunk_count(chunk_count: int) -> str:
    """sha256 de `chunk_count // _CORPUS_HASH_BUCKET` con prefix de
    `_FILTER_VERSION` (importado lazily para evitar circular import).
    Bumpear `_FILTER_VERSION` en `rag/__init__.py` invalida automáticamente
    todas las entries del cache.
    """
    import hashlib
    from rag import _FILTER_VERSION  # noqa: PLC0415  (lazy)
    bucket = chunk_count // _CORPUS_HASH_BUCKET
    return hashlib.sha256(
        f"count_bucket:{bucket}|filter:{_FILTER_VERSION}".encode()
    ).hexdigest()[:16]


def _compute_corpus_hash(col) -> str:
    """Cheap fingerprint of the current corpus. sha256 of
    `chunk_count // _CORPUS_HASH_BUCKET`.

    Returns "" if col is unavailable. Bucket-bounded so ingesters
    incrementales (WA hourly etc) NO mueven el hash diariamente.

    Per-entry freshness se enforce en `semantic_cache_lookup` comparando
    mtime de cada path citado contra cached_ts.
    """
    try:
        chunk_count = int(col.count())
    except Exception:
        return ""
    return _hash_chunk_count(chunk_count)


def _corpus_hash_cached(col, *, hint_count: int | None = None) -> str:
    """Cached version de `_compute_corpus_hash` — re-computa solo cuando
    chunk count cruza bucket (mismo trigger que invalidation de
    `_load_corpus`).

    Una sola llamada a `col.count()`: cache caliente devuelve directo;
    miss computa hash desde `cnt` ya leído (sin re-llamar `col.count()`
    dentro de `_compute_corpus_hash`).

    `hint_count`: cuando el caller ya tiene `col.count()` (ej. `retrieve()`
    lo lee al inicio para `_load_corpus(hint_count=...)`) puede pasarlo
    para evitar el round-trip SQL redundante (~1-2ms ahorrados/retrieve).
    """
    if hint_count is not None:
        cnt = int(hint_count)
    else:
        try:
            cnt = int(col.count())
        except Exception:
            return ""
    with _corpus_hash_lock:
        if _corpus_hash_memo["hash"] and _corpus_hash_memo["chunk_count"] == cnt:
            return _corpus_hash_memo["hash"]
    # Resolve via `rag` module so tests que monkeypatchean
    # `rag._hash_chunk_count` propaguen al call site (CLAUDE.md split shim
    # pattern). Si rag re-export no funciona (early init), fallback al
    # local.
    try:
        import rag as _rag  # noqa: PLC0415
        h = _rag._hash_chunk_count(cnt)
    except Exception:
        h = _hash_chunk_count(cnt)
    with _corpus_hash_lock:
        _corpus_hash_memo["hash"] = h
        _corpus_hash_memo["chunk_count"] = cnt
        _corpus_hash_memo["last_ts"] = time.time()
    return h


def _embedding_to_blob(emb) -> tuple[bytes, int]:
    """Serialise embedding → float32 LE blob. Same format que
    `rag_feedback_golden`."""
    import numpy as np
    arr = np.asarray(emb, dtype="<f4")
    return arr.tobytes(), int(arr.shape[0])


def _blob_to_embedding(blob: bytes, dim: int):
    import numpy as np
    return np.frombuffer(blob, dtype="<f4", count=dim)


def _ttl_for_intent(intent: str | None) -> int:
    """TTL seconds by intent. recent/agenda → short (vault churn matters
    for "what's new" queries); semantic/synthesis/comparison/count/list →
    long.

    Env overrides: `RAG_CACHE_TTL_RECENT` (default 600),
    `RAG_CACHE_TTL_DEFAULT` (default 86400).
    """
    if intent in ("recent", "agenda"):
        return _SEMANTIC_CACHE_RECENT_TTL
    return _SEMANTIC_CACHE_DEFAULT_TTL


def _cached_entry_is_stale(
    paths: list[str],
    cached_ts: float,
    *,
    mtime_cache: dict[str, float | None] | None = None,
) -> bool:
    """Return True si alguno de los paths cited tiene mtime > cached_ts.

    Post 2026-04-23 per-entry freshness check. El corpus_hash ya no scanea
    mtimes (ver `_compute_corpus_hash`), individual note edits se cachan
    acá: si una respuesta cited cita `03-Resources/ikigai.md` y ese file
    se editó después del store, skip — servir contenido stale es peor que
    cache miss.

    Cheap: ≤5 paths/entry × ~100µs/stat ≈ <1ms en APFS. Política:

    - Empty paths → never stale (entry sin source refs; refusals ya
      rejected at store time).
    - File exists AND mtime > cached_ts → stale.
    - File exists AND mtime ≤ cached_ts → fresh.
    - File missing (deleted/renamed) → NO stale. Razón: corpus-level
      invalidation (chunk_count delta via `_compute_corpus_hash`) ya
      catchea note deletions a coarser granularidad. Missing files
      acá son típicamente vault-path mock de tests o relative-path
      mismatch. Punishing both con blow-the-cache stale-vote crea
      false negatives.
    - Unresolvable vault_path / permission error → asume fresh (mismo
      reasoning).

    `mtime_cache` (perf 2026-05-04): si caller itera sobre N cached rows
    en el mismo lookup pass, pasarle `{}` permite memoizar
    `(path → mtime|None)` — mismo path repetido en N entries paga 1
    `stat()` en vez de N. None preserva legacy behavior.
    """
    if not paths:
        return False
    try:
        from rag import _resolve_vault_path  # noqa: PLC0415  (lazy)
        vault = _resolve_vault_path()
    except Exception:
        return False
    for p in paths:
        try:
            full = vault / p
            full_key = str(full)
            if mtime_cache is not None and full_key in mtime_cache:
                mt = mtime_cache[full_key]
            else:
                if not full.exists():
                    mt = None
                else:
                    mt = full.stat().st_mtime
                if mtime_cache is not None:
                    mtime_cache[full_key] = mt
            if mt is None:
                continue
            if mt > cached_ts:
                return True
        except Exception:
            continue
    return False


def semantic_cache_lookup(
    q_embedding,
    corpus_hash: str,
    *,
    cosine_threshold: float | None = None,
    now: float | None = None,
    return_probe: bool = False,
):
    """Encuentra cached response con cosine ≥ threshold vs el embedding,
    dentro del mismo `corpus_hash` y dentro del TTL window.

    Returns row dict (question, response, paths, scores, top_score, intent,
    hit_id, cosine, cached_ts, age_seconds) o None on miss.

    Con `return_probe=True` returns tuple `(hit_or_None, probe_dict)`.
    Probe se emite a `rag_queries.extra_json.cache_probe` para telemetría.

    Linear scan sobre rows matching corpus_hash — aceptable para n hasta
    pocos miles (cache capped en `_SEMANTIC_CACHE_MAX_ROWS`). Errores en
    read degradan a "no hit".
    """
    from rag import _ragvec_state_conn, _log_sql_state_error, record_cache_event  # noqa: PLC0415

    probe: dict = {
        "result": "miss",
        "reason": "empty_cache",
        "top_cosine": None,
        "candidates": 0,
        "skipped_stale": 0,
        "skipped_ttl": 0,
    }
    if not _semantic_cache_enabled():
        probe["result"] = "disabled"
        probe["reason"] = "cache_disabled"
        return (None, probe) if return_probe else None
    if not corpus_hash:
        probe["result"] = "empty_corpus_hash"
        probe["reason"] = "no_corpus_hash"
        return (None, probe) if return_probe else None
    # Resolve threshold via `rag` so tests patching `rag._SEMANTIC_CACHE_COSINE`
    # propaguen al call site.
    import rag as _rag  # noqa: PLC0415
    default_cos = getattr(_rag, "_SEMANTIC_CACHE_COSINE", _SEMANTIC_CACHE_COSINE)
    default_ttl = getattr(_rag, "_SEMANTIC_CACHE_DEFAULT_TTL", _SEMANTIC_CACHE_DEFAULT_TTL)
    max_rows = getattr(_rag, "_SEMANTIC_CACHE_MAX_ROWS", _SEMANTIC_CACHE_MAX_ROWS)
    threshold = cosine_threshold if cosine_threshold is not None else default_cos
    now = now if now is not None else time.time()
    try:
        import numpy as np
        q = np.asarray(q_embedding, dtype="<f4")
        qn = float(np.linalg.norm(q))
        if qn == 0:
            probe["reason"] = "zero_norm_query"
            return (None, probe) if return_probe else None
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT id, ts, question, q_embedding, dim, intent, ttl_seconds,"
                " response, paths_json, scores_json, top_score"
                " FROM rag_response_cache WHERE corpus_hash = ?"
                " ORDER BY ts DESC LIMIT ?",
                (corpus_hash, max_rows),
            ).fetchall()
    except Exception as exc:
        _log_sql_state_error("semantic_cache_lookup_failed", err=repr(exc))
        probe["reason"] = "db_error"
        return (None, probe) if return_probe else None

    probe["candidates"] = len(rows)
    if not rows:
        probe["reason"] = "corpus_mismatch" if probe["candidates"] == 0 else "empty_cache"

    best_cos = -1.0
    best = None
    # Memoization local: `(full_path → mtime|None)` reusable entre entries
    # del mismo lookup. Si N entries citan el mismo set de notas (caso
    # típico — corpus no es disjoint), reduce stat() calls de
    # O(entries × paths) a O(unique paths). Win medido: 80% reducción
    # con 5 entries × 2 paths solapados.
    _mtime_cache: dict[str, float | None] = {}
    for row in rows:
        rid, ts_str, question, blob, dim, intent, ttl_seconds, response, paths_json, scores_json, top_score = row
        try:
            cached_ts = datetime.fromisoformat(ts_str).timestamp()
        except Exception:
            continue
        age = now - cached_ts
        if age > (ttl_seconds or default_ttl):
            probe["skipped_ttl"] += 1
            continue
        try:
            e = _blob_to_embedding(blob, int(dim))
            if e.shape[0] != q.shape[0]:
                continue
            en = float(np.linalg.norm(e))
            if en == 0:
                continue
            cos = float(np.dot(q, e)) / (qn * en)
        except Exception:
            continue
        if cos > best_cos:
            best_cos = cos
        if cos >= threshold:
            paths = json.loads(paths_json) if paths_json else []
            if _cached_entry_is_stale(paths, cached_ts, mtime_cache=_mtime_cache):
                probe["skipped_stale"] += 1
                continue
            if cos > best_cos or best is None:
                best_cos = cos
                best = {
                    "id": int(rid),
                    "question": question,
                    "response": response,
                    "paths": paths,
                    "scores": json.loads(scores_json) if scores_json else [],
                    "top_score": float(top_score) if top_score is not None else None,
                    "intent": intent,
                    "cosine": cos,
                    "cached_ts": cached_ts,
                    "age_seconds": age,
                }
    probe["top_cosine"] = round(best_cos, 4) if best_cos > -1.0 else None

    # Async hit-count bump — best-effort, no bloquea cache hit en SQL write.
    if best is not None:
        record_cache_event("semantic", hits=1)
        probe["result"] = "hit"
        probe["reason"] = "match"
        try:
            with _ragvec_state_conn() as conn:
                conn.execute(
                    "UPDATE rag_response_cache SET hit_count = hit_count + 1,"
                    " last_hit_ts = ? WHERE id = ?",
                    (datetime.now(timezone.utc).isoformat(), best["id"]),
                )
                conn.commit()
        except Exception as exc:
            _log_sql_state_error("semantic_cache_hit_bump_failed", err=repr(exc))
    else:
        record_cache_event("semantic", misses=1)
        # Reason ranking: stale > ttl > below_threshold > corpus_mismatch
        if probe["skipped_stale"] > 0 and probe["skipped_stale"] >= probe["skipped_ttl"]:
            probe["reason"] = "stale_source"
        elif probe["skipped_ttl"] > 0:
            probe["reason"] = "ttl_expired"
        elif probe["candidates"] > 0 and best_cos > -1.0:
            probe["reason"] = "below_threshold"
        elif probe["candidates"] == 0:
            probe["reason"] = "corpus_mismatch"
    return (best, probe) if return_probe else best


def semantic_cache_store(
    q_embedding,
    *,
    question: str,
    response: str,
    paths: list[str],
    scores: list[float],
    top_score: float | None,
    intent: str | None,
    corpus_hash: str,
    ttl_seconds: int | None = None,
    extra: dict | None = None,
    background: bool = False,
) -> bool:
    """Persiste response al cache. Returns True on success.

    Skipped silently si:
      - cache disabled (`RAG_CACHE_ENABLED=0`)
      - corpus_hash empty
      - response empty / refusal marker (`top_score < 0.015`)
      - response es refusal del LLM (cache poisoning prevention)

    `background=True` queues INSERT en SQL writer worker thread y returns
    inmediatamente. `background=False` (default) bloquea hasta commit.
    """
    from rag import (  # noqa: PLC0415
        _enqueue_background_sql,
        _is_refusal,
        _log_sql_state_error,
        _ragvec_state_conn,
        _sql_write_with_retry,
    )

    if not _semantic_cache_enabled() or not corpus_hash:
        return False
    if not response or not response.strip():
        return False
    # Skip storing low-confidence refusals — servirían stale nothing-answers
    # y poisonarían el cache para la misma query post-vault-grew.
    if top_score is not None and top_score < 0.015:
        return False
    # Skip refusals incluso con top_score HIGH. Regression 2026-04-22:
    # query con top_score 1.18 (alta confianza retrieve) cayó en refuse
    # falso del LLM (fast-path qwen2.5:3b con num_ctx truncado) y se
    # cacheó "No tengo esa información" perpetuamente.
    if _is_refusal(response):
        return False
    try:
        blob, dim = _embedding_to_blob(q_embedding)
        ttl = int(ttl_seconds) if ttl_seconds is not None else _ttl_for_intent(intent)
        now_iso = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        _log_sql_state_error("semantic_cache_store_failed", err=repr(exc))
        return False

    def _do_insert():
        with _ragvec_state_conn() as conn:
            conn.execute(
                "INSERT INTO rag_response_cache"
                " (ts, question, q_embedding, dim, corpus_hash, intent,"
                "  ttl_seconds, response, paths_json, scores_json,"
                "  top_score, hit_count, last_hit_ts, extra_json)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    now_iso, question, blob, dim, corpus_hash, intent,
                    ttl, response,
                    json.dumps(paths, ensure_ascii=False),
                    json.dumps(scores),
                    top_score, 0, None,
                    json.dumps(extra, ensure_ascii=False) if extra else None,
                ),
            )
            conn.commit()

    if background:
        _enqueue_background_sql(_do_insert, "semantic_cache_store_failed")
        return True

    _sql_write_with_retry(_do_insert, "semantic_cache_store_failed")
    return True


def semantic_cache_clear(corpus_hash: str | None = None) -> int:
    """DELETE cached rows (todas, o solo las que matchean corpus_hash).

    Returns number of rows deleted. Errores degradan a 0.
    """
    from rag import _log_sql_state_error, _ragvec_state_conn  # noqa: PLC0415

    try:
        with _ragvec_state_conn() as conn:
            if corpus_hash:
                cur = conn.execute(
                    "DELETE FROM rag_response_cache WHERE corpus_hash = ?",
                    (corpus_hash,),
                )
            else:
                cur = conn.execute("DELETE FROM rag_response_cache")
            conn.commit()
            return int(cur.rowcount or 0)
    except Exception as exc:
        _log_sql_state_error("semantic_cache_clear_failed", err=repr(exc))
        return 0


def semantic_cache_stats() -> dict:
    """Stats para debugging: total rows, distinct corpus_hashes, total
    hits, age snapshot. Used by `rag cache stats`.
    """
    from rag import _log_sql_state_error, _ragvec_state_conn  # noqa: PLC0415

    try:
        with _ragvec_state_conn() as conn:
            n_rows = int(conn.execute(
                "SELECT COUNT(*) FROM rag_response_cache"
            ).fetchone()[0] or 0)
            n_hashes = int(conn.execute(
                "SELECT COUNT(DISTINCT corpus_hash) FROM rag_response_cache"
            ).fetchone()[0] or 0)
            total_hits = int(conn.execute(
                "SELECT COALESCE(SUM(hit_count), 0) FROM rag_response_cache"
            ).fetchone()[0] or 0)
            rows = conn.execute(
                "SELECT MIN(ts), MAX(ts) FROM rag_response_cache"
            ).fetchone()
            oldest, newest = (rows or (None, None))
        # Resolve via `rag` for monkeypatch propagation (tests patch
        # `rag._SEMANTIC_CACHE_COSINE`).
        import rag as _rag  # noqa: PLC0415
        cos_value = getattr(_rag, "_SEMANTIC_CACHE_COSINE", _SEMANTIC_CACHE_COSINE)
        ttl_value = getattr(_rag, "_SEMANTIC_CACHE_DEFAULT_TTL", _SEMANTIC_CACHE_DEFAULT_TTL)
        return {
            "rows": n_rows,
            "corpus_hashes": n_hashes,
            "hits": total_hits,
            "oldest_ts": oldest,
            "newest_ts": newest,
            "enabled": _semantic_cache_enabled(),
            "cosine_threshold": cos_value,
            "default_ttl_s": ttl_value,
        }
    except Exception as exc:
        _log_sql_state_error("semantic_cache_stats_failed", err=repr(exc))
        return {"error": repr(exc)}
