"""Backend de las features nuevas de memo v0.6.0 surfaces en `/memo`:

1. **Time-machine** (`memo as-of` / `memo diff`) — reconstruct corpus snapshot
   at past date by replaying `history.db` events in reverse. Devuelve
   list de records al T target + diff added/removed/updated entre dos T.

2. **Knowledge graph** — leer `entities` + `entity_memoria` de `graph.db`.
   Nodos = entidades extraídas (person/project/technology/file/org/concept).
   Edges = co-ocurrencia (2 entidades aparecen en la misma memoria).
   Pesos = #memorias compartidas. Centralidad ~ mention_count.

3. **Temporal** — saves/updates/deletes per día (30d), stale memorias
   (>N días sin update, sin recall), contradictions (cached, NO LLM live
   porque cuesta 200ms+/par).

Pattern: usa la API Python de `memo` como library (instalada en el venv
del rag con `uv pip install -e /Users/fer/repos/memo --no-deps`). NO usa
subprocess al binario `memo` (overhead 200-500ms cold + spinup MLX).

Cache TTL 60s in-process. Source-of-truth los 3 sqlite que memo escribe;
nunca tocamos esos files (read-only).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

# Lazy global Memory singleton — el primer endpoint que lo necesite paga
# el costo de cargar config + abrir 4 sqlite (~200ms). Las siguientes
# requests reusan el mismo handle. Thread-safe via lock.
_mem_lock = threading.Lock()
_mem_instance: Any = None

# In-process snapshot/diff cache. Snapshots son caros (1-2s en corpus
# grande porque hay que reconstruir + leer todo `events`); TTL 60s
# evita recomputar entre clicks del scrubber del frontend.
_cache: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 60.0


def _get_memory() -> Any:
    """Lazy singleton para `memo.memory.Memory`. None si memo no está
    instalado / no se puede inicializar (sin MEMO_ env, etc.).

    No prewarm MLX — los endpoints sólo necesitan store/history/graph
    sqlite, no embedder ni reranker. Override env var por las dudas.
    """
    global _mem_instance
    with _mem_lock:
        if _mem_instance is not None:
            return _mem_instance
        try:
            import os
            os.environ.setdefault("MEMO_PREWARM_DISABLE", "1")
            from memo.config import Config
            from memo.memory import Memory
            cfg = Config.from_env()
            _mem_instance = Memory(cfg)
        except Exception as e:
            _mem_instance = e  # marker — keeps us from retrying forever
        return _mem_instance


def _cached(key: str, fn):
    """Caché simple por (key, TTL)."""
    now = time.time()
    entry = _cache.get(key)
    if entry is not None and (now - entry[0]) < _CACHE_TTL:
        return entry[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def _err_if_no_memo(mem: Any) -> dict | None:
    if mem is None:
        return {"ok": False, "error": "memo not installed in this venv"}
    if isinstance(mem, Exception):
        return {"ok": False, "error": f"memo init failed: {type(mem).__name__}: {mem}"}
    return None


# ──────────────── Time-machine ────────────────────────────────────────

def _parse_date_input(s: str | None) -> datetime:
    """Acepta YYYY-MM-DD o ISO datetime. Default = hoy 23:59:59 UTC.
    Date-only se interpreta como fin del día (snapshot incluye todo lo
    que pasó *ese* día)."""
    if not s:
        return datetime.now(UTC)
    s = s.strip()
    if len(s) == 10:  # YYYY-MM-DD
        d = datetime.fromisoformat(s).replace(tzinfo=UTC)
        return d.replace(hour=23, minute=59, second=59)
    s = s.rstrip("Z")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def snapshot(date: str | None = None, type_: str | None = None,
             limit: int = 200) -> dict:
    """Corpus snapshot al `date` (YYYY-MM-DD o ISO). Devuelve hasta
    `limit` records ordenados por updated desc.

    Cada record: id (8-char prefix), title, type, tags, created, updated,
    body_unavailable (True si fue borrado y no tenemos versions row).
    """
    mem = _get_memory()
    if (e := _err_if_no_memo(mem)) is not None:
        return e
    target = _parse_date_input(date)
    cache_key = f"snap:{target.date().isoformat()}:{type_ or ''}"

    def _build():
        from memo.time_machine import reconstruct
        snap = reconstruct(mem, as_of=target)
        rows = []
        for r in snap.list(type_=type_)[:limit]:
            rows.append({
                "id": r.id[:8],
                "id_full": r.id,
                "title": r.title,
                "type": r.type,
                "tags": r.tags,
                "created": r.created,
                "updated": r.updated,
                "body_unavailable": r.body_unavailable,
            })
        # Type breakdown del snapshot completo (no del limit)
        type_counts: dict[str, int] = {}
        for r in snap.records.values():
            if r.body_unavailable:
                continue
            type_counts[r.type] = type_counts.get(r.type, 0) + 1
        return {
            "ok": True,
            "as_of": target.isoformat(),
            "as_of_date": target.date().isoformat(),
            "total": len(snap),
            "type_counts": type_counts,
            "rows": rows,
            "is_current": (datetime.now(UTC) - target).total_seconds() < 3600,
        }

    return _cached(cache_key, _build)


def timemachine_diff(from_date: str, to_date: str | None = None) -> dict:
    """Diff entre dos snapshots. `to_date` default = hoy."""
    mem = _get_memory()
    if (e := _err_if_no_memo(mem)) is not None:
        return e
    f_ts = _parse_date_input(from_date)
    t_ts = _parse_date_input(to_date)
    cache_key = f"diff:{f_ts.date().isoformat()}:{t_ts.date().isoformat()}"

    def _build():
        from memo.time_machine import diff
        d = diff(mem, from_ts=f_ts, to_ts=t_ts)
        return {
            "ok": True,
            "from": f_ts.isoformat(),
            "from_date": f_ts.date().isoformat(),
            "to": t_ts.isoformat(),
            "to_date": t_ts.date().isoformat(),
            "summary": d.summary(),
            "added": [
                {"id": r.id[:8], "id_full": r.id, "title": r.title,
                 "type": r.type, "tags": r.tags, "updated": r.updated}
                for r in d.added
            ],
            "removed": [
                {"id": r.id[:8], "id_full": r.id, "title": r.title,
                 "type": r.type, "tags": r.tags, "updated": r.updated}
                for r in d.removed
            ],
            "updated": [
                {"id": u["id"][:8], "id_full": u["id"], "title": u["title"],
                 "changed_fields": u["changed_fields"],
                 "before": u["before"], "after": u["after"]}
                for u in d.updated
            ],
        }

    return _cached(cache_key, _build)


# ──────────────── Knowledge graph ─────────────────────────────────────

def graph(limit_nodes: int = 80, min_count: int = 2,
          type_filter: str | None = None) -> dict:
    """Lee `graph.db` y arma nodos + edges para force-directed viz.

    - **Nodos**: top `limit_nodes` entidades por mention_count (filtradas
      opcionalmente a un type).
    - **Edges**: co-ocurrencia. Dos entidades comparten edge si aparecen
      ≥1 vez juntas en la misma memoria. Peso = #memorias compartidas.

    Falla suave si `graph.db` no existe / está vacío.
    """
    mem = _get_memory()
    if (e := _err_if_no_memo(mem)) is not None:
        return e

    cache_key = f"graph:{limit_nodes}:{min_count}:{type_filter or ''}"

    def _build():
        try:
            graph_store = mem.graph
            stats = graph_store.stats()
        except Exception as exc:
            return {"ok": False, "error": f"graph not available: {exc}"}

        if stats.get("entities", 0) == 0:
            return {
                "ok": True,
                "nodes": [],
                "edges": [],
                "stats": stats,
                "hint": "graph empty — run `memo extract-entities --all` to populate",
            }

        # Top-N entities by mention_count
        top = graph_store.top_entities(limit=limit_nodes, type_=type_filter)
        top = [e for e in top if e.get("mention_count", 0) >= min_count]
        names = {(e["name"], e["type"]) for e in top}

        # Mapping (name, type) → idx para edges
        idx_of: dict[tuple[str, str], int] = {}
        nodes = []
        for i, e in enumerate(top):
            key = (e["name"], e["type"])
            idx_of[key] = i
            nodes.append({
                "id": i,
                "name": e["name"],
                "type": e["type"],
                "count": e.get("mention_count", 0),
                "first_seen": e.get("first_seen"),
                "last_seen": e.get("last_seen"),
            })

        # Edges: co-ocurrencia. Por cada memoria que mencione ≥2 entidades
        # del top, generar todos los pares (sin doble-conteo).
        # Una sola query JOIN: por memoria, listar sus entities.
        # Path: db_path está en mem.graph.db_path
        try:
            cx = sqlite3.connect(str(graph_store.db_path), timeout=5.0)
            cx.row_factory = sqlite3.Row
            rows = cx.execute(
                "SELECT em.memoria_id, e.name, e.type "
                "FROM entity_memoria em JOIN entities e ON e.id = em.entity_id"
            ).fetchall()
            cx.close()
        except Exception as exc:
            return {"ok": False, "error": f"graph read failed: {exc}"}

        by_memoria: dict[str, list[tuple[str, str]]] = {}
        for r in rows:
            key = (r["name"], r["type"])
            if key not in idx_of:
                continue
            by_memoria.setdefault(r["memoria_id"], []).append(key)

        edge_weights: dict[tuple[int, int], int] = {}
        for mid, ents in by_memoria.items():
            uniq = list(set(ents))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = idx_of[uniq[i]], idx_of[uniq[j]]
                    if a > b:
                        a, b = b, a
                    edge_weights[(a, b)] = edge_weights.get((a, b), 0) + 1

        edges = [
            {"source": s, "target": t, "weight": w}
            for (s, t), w in edge_weights.items()
        ]
        # Sort heaviest first — frontend draws strong edges visible primero
        edges.sort(key=lambda e: e["weight"], reverse=True)

        return {
            "ok": True,
            "stats": stats,
            "nodes": nodes,
            "edges": edges,
            "isolated": sum(1 for n in nodes
                            if not any(e["source"] == n["id"] or e["target"] == n["id"]
                                       for e in edges)),
        }

    return _cached(cache_key, _build)


# ──────────────── Temporal ────────────────────────────────────────────

def temporal_timeline(days: int = 30) -> dict:
    """Saves / updates / deletes per día sobre los últimos `days`.
    Lee `events` table directo. Cap days=365.
    """
    mem = _get_memory()
    if (e := _err_if_no_memo(mem)) is not None:
        return e
    days = max(1, min(365, int(days)))
    cache_key = f"timeline:{days}"

    def _build():
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        events = mem.history.list_recent(limit=100_000)
        bucket_saves: dict[str, int] = {}
        bucket_updates: dict[str, int] = {}
        bucket_deletes: dict[str, int] = {}
        for ev in events:
            ts = ev.get("ts") or ""
            if ts < cutoff:
                continue
            day = ts[:10]
            op = ev.get("op")
            if op == "save":
                bucket_saves[day] = bucket_saves.get(day, 0) + 1
            elif op == "update":
                bucket_updates[day] = bucket_updates.get(day, 0) + 1
            elif op == "delete":
                bucket_deletes[day] = bucket_deletes.get(day, 0) + 1
        # Series ordenadas (relleno días vacíos con 0)
        series: list[dict[str, Any]] = []
        today = datetime.now(UTC).date()
        for i in range(days):
            d = (today - timedelta(days=days - 1 - i)).isoformat()
            series.append({
                "date": d,
                "saves": bucket_saves.get(d, 0),
                "updates": bucket_updates.get(d, 0),
                "deletes": bucket_deletes.get(d, 0),
            })
        total = {
            "saves": sum(s["saves"] for s in series),
            "updates": sum(s["updates"] for s in series),
            "deletes": sum(s["deletes"] for s in series),
        }
        return {"ok": True, "days": days, "series": series, "total": total}

    return _cached(cache_key, _build)


def temporal_stale(days_threshold: int = 90, limit: int = 30) -> dict:
    """Top memorias sin update >N días Y sin recalls recientes — candidatas
    a archive/delete. Lee meta directo (más rápido que TemporalAnalyzer
    que también arma access_count que NO está en history.db).
    """
    mem = _get_memory()
    if (e := _err_if_no_memo(mem)) is not None:
        return e
    days_threshold = max(7, min(720, int(days_threshold)))
    cache_key = f"stale:{days_threshold}:{limit}"

    def _build():
        cutoff = (datetime.now(UTC) - timedelta(days=days_threshold))
        cutoff_iso = cutoff.isoformat()
        rows = mem.list(limit=10_000)
        stale: list[dict[str, Any]] = []
        for r in rows:
            upd = r.updated or ""
            if upd < cutoff_iso:
                try:
                    dt = datetime.fromisoformat(upd.replace("Z", "+00:00"))
                    days_old = (datetime.now(UTC) - dt).days
                except Exception:
                    days_old = days_threshold
                stale.append({
                    "id": r.id[:8],
                    "id_full": r.id,
                    "title": r.title,
                    "type": r.type,
                    "tags": list(r.tags or []),
                    "updated": upd,
                    "days_old": days_old,
                })
        stale.sort(key=lambda x: x["days_old"], reverse=True)
        return {
            "ok": True,
            "threshold_days": days_threshold,
            "total_stale": len(stale),
            "rows": stale[:limit],
        }

    return _cached(cache_key, _build)


def cache_invalidate() -> None:
    """Drop el cache. Endpoint privado para tests / cuando memo escribe."""
    _cache.clear()


__all__ = [
    "snapshot",
    "timemachine_diff",
    "graph",
    "temporal_timeline",
    "temporal_stale",
    "cache_invalidate",
]
