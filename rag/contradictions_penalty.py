"""Contradiction penalty para el retrieval pipeline.

Contrato (post-rerank, pre cap top-k):

    paths = load_contradiction_paths(conn)
    apply_contradiction_penalty(results, paths, penalty=-0.05)

Idea: las notas que figuran en `rag_contradictions` (detectadas al indexar
y todavía sin resolver) reciben un demote leve en su score. NO se filtran
ni se eliminan — solo se baja el score y se re-ordena. El usuario sigue
viéndolas si son lo único relevante; pero si hay otra nota razonable en
top-k, esa pasa adelante.

Schema soportado (uno de los dos):
  - rag_contradictions(subject_path, contradicts_path, resolved_at, …)
    → unresolved = `WHERE resolved_at IS NULL`
    → paths = subject_path ∪ contradicts_path
  - rag_contradictions(subject_path, contradicts_json, skipped, …)  [actual]
    → unresolved = `WHERE skipped IS NULL OR skipped = ''`
    → paths = subject_path ∪ json_extract(contradicts_json[*].path)

Cuando triage marca una contradicción como resuelta (vía skill
`rag-contradict-triage`), su row queda con `skipped` no-null o
`resolved_at` no-null, y deja de bajar el score.

Cache: TTL 5min in-process. Las contradicciones cambian cuando el indexer
detecta una nueva o cuando el user la resuelve — ninguno de los dos
flujos es hot-path. La query pasa por `_ragvec_state_conn()` cada vez
SOLO en miss; los hits son lookup O(1) sobre un dict.

NO toca `rag/anticipatory.py` ni el detector existente (este módulo es
sólo CONSUMER del estado, no PRODUCER).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

# ── Cache ────────────────────────────────────────────────────────────────────
# 5-min TTL; las contradicciones no cambian rápido (las populan los
# ingesters al indexar, y se resuelven por triage explícito del user).
# La query suma ~50-200 filas filtradas por `WHERE skipped IS NULL` con
# índice por `ts` — no es cara, pero corre en hot-path por cada retrieve()
# así que vale el cache. Lock para evitar carrera en multi-thread (web/SSE).

_CACHE_TTL_SECS = 300.0  # 5 min
_cache_lock = threading.Lock()
_cache_state: dict[str, Any] = {
    "ts": 0.0,             # monotonic seconds del último fill
    "paths": set(),        # set[str] de paths actualmente sin resolver
    "queries": 0,          # contador de SQL fetches (testabilidad)
}


def reset_cache() -> None:
    """Limpia el cache. Para tests + para invalidación manual post-triage.

    No es necesario llamar esto en producción — el TTL natural de 5min
    cubre el delay aceptable entre que el user resuelve una
    contradicción y que el penalty deja de aplicar.
    """
    with _cache_lock:
        _cache_state["ts"] = 0.0
        _cache_state["paths"] = set()
        _cache_state["queries"] = 0


def _table_columns(conn) -> set[str]:
    """Devuelve los nombres de columnas de `rag_contradictions`.

    Si la tabla no existe (DB fresca, vault sin indexar) devuelve set vacío
    y el caller arriba interpreta como "sin contradicciones".
    """
    try:
        rows = list(conn.execute("PRAGMA table_info(rag_contradictions)"))
    except Exception:
        return set()
    return {r[1] for r in rows}


def _query_paths(conn) -> set[str]:
    """Ejecuta el SELECT real sin pasar por el cache.

    Tolerante a schemas diferentes:
      - Si existe `resolved_at`, filtra `resolved_at IS NULL`.
      - Si no, asume el schema actual con `skipped` y filtra
        `skipped IS NULL OR skipped = ''`.
      - Si tampoco existe `skipped`, todos los rows cuentan como
        unresolved (fallback conservador).

    Para columnas de `contradicts_path` (string) las agrega directo.
    Para `contradicts_json` (JSON list de `{path, why}` o de strings)
    parsea y extrae cada `path`. JSON malformado → skip silencioso.
    """
    cols = _table_columns(conn)
    if not cols:
        return set()

    if "resolved_at" in cols:
        where = "WHERE resolved_at IS NULL"
    elif "skipped" in cols:
        where = "WHERE skipped IS NULL OR skipped = ''"
    else:
        where = ""

    paths: set[str] = set()

    # Subject path siempre existe en cualquiera de los schemas.
    if "subject_path" not in cols:
        return paths

    # Caso A: columna escalar `contradicts_path`
    if "contradicts_path" in cols:
        sql = f"SELECT subject_path, contradicts_path FROM rag_contradictions {where}"
        for subj, contr in conn.execute(sql):
            if subj:
                paths.add(subj)
            if contr:
                paths.add(contr)
        return paths

    # Caso B: columna JSON `contradicts_json`
    if "contradicts_json" in cols:
        sql = f"SELECT subject_path, contradicts_json FROM rag_contradictions {where}"
        for subj, raw in conn.execute(sql):
            if subj:
                paths.add(subj)
            if not raw:
                continue
            try:
                items = json.loads(raw)
            except Exception:
                continue
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict):
                    p = it.get("path")
                    if isinstance(p, str) and p:
                        paths.add(p)
                elif isinstance(it, str) and it:
                    paths.add(it)
        return paths

    # Caso C: solo `subject_path` está en el schema
    sql = f"SELECT subject_path FROM rag_contradictions {where}"
    for (subj,) in conn.execute(sql):
        if subj:
            paths.add(subj)
    return paths


def load_contradiction_paths(conn, *, _now: float | None = None) -> set[str]:
    """Devuelve los paths con contradicción no resuelta. Cached 5min.

    `_now` es un parámetro de testing — permite simular el avance del
    reloj sin patch a `time.monotonic`. En production siempre se usa el
    reloj real.

    Returns:
        Un set NUEVO (copia del cache) para que callers no muten el
        estado interno por accidente.
    """
    now = time.monotonic() if _now is None else _now
    with _cache_lock:
        last_ts = _cache_state["ts"]
        if last_ts > 0 and (now - last_ts) < _CACHE_TTL_SECS:
            return set(_cache_state["paths"])

    # Cache miss: consultamos fuera del lock para no bloquear lecturas
    # concurrentes durante una query lenta. Race condition aceptable:
    # 2 threads simultáneos pueden duplicar la query inicial; converge
    # a 1 sola query por TTL al cabo de la primera población.
    paths = _query_paths(conn)

    with _cache_lock:
        _cache_state["ts"] = now
        _cache_state["paths"] = paths
        _cache_state["queries"] += 1
        return set(paths)


def cache_query_count() -> int:
    """Cuántos SELECTs reales hizo el cache desde el último reset.

    Sólo para tests — no exponer en métricas de producción (el counter
    no se persiste y arranca en 0 cada vez que se reimporta el módulo).
    """
    with _cache_lock:
        return int(_cache_state["queries"])


# ── Aplicación del penalty ───────────────────────────────────────────────────


def apply_contradiction_penalty(
    results: list[dict],
    paths_with_contradictions: set[str],
    penalty: float = -0.05,
) -> list[dict]:
    """Demote results cuyo `path` está en el set; re-ordena DESC por `score`.

    Args:
        results: lista de dicts con al menos las claves `path` y `score`.
            Se muta en place (también se re-ordena la misma list).
        paths_with_contradictions: set[str] de paths a penalizar.
        penalty: float que se SUMA al score (típicamente negativo).
            Default `-0.05` (5% de demote sobre el rango típico de
            rerank_logit ~ [0, 1]).

    Returns:
        La misma lista (post-mutación + sort), por conveniencia para
        usar en pipelines tipo `apply(...)[:k]`.

    Notas:
      - NO drop. Si todos los results están en el set, todos quedan
        con su score - magnitude y la lista se ordena entre ellos.
      - NO toca results sin la clave `path` (silently skip).
      - Idempotencia: si paths_with_contradictions está vacío, retorna
        la lista sin tocar (no sort, no mut) — preserva el ordering
        original del caller. Útil para el caso "DB sin contradicciones".
    """
    if not paths_with_contradictions:
        return results
    if not results:
        return results

    for r in results:
        try:
            p = r.get("path")
        except AttributeError:
            continue
        if isinstance(p, str) and p in paths_with_contradictions:
            r["score"] = float(r.get("score", 0.0)) + float(penalty)

    results.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    return results


def count_penalized(
    results: list[dict],
    paths_with_contradictions: set[str],
) -> int:
    """Cuántos results MATCH el set de paths penalizables.

    Helper para telemetría — el caller llama a esto ANTES de aplicar
    para loggear el counter sin tener que diff-ear scores antes/después.
    """
    if not paths_with_contradictions or not results:
        return 0
    n = 0
    for r in results:
        p = r.get("path") if isinstance(r, dict) else None
        if isinstance(p, str) and p in paths_with_contradictions:
            n += 1
    return n
