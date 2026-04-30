"""Vault health score — métrica agregada 0-100 sobre la salud del vault.

Combina seis señales que ya viven en SQL (ragvec.db + telemetry.db) en un
score ponderado:

  - tags_pct        (peso 20%) — % notas con ≥1 tag
  - backlinks_pct   (peso 25%) — % notas con ≥1 backlink (otra nota la linkea)
  - orphans         (peso 15%) — # notas sin in/outlinks (menos = mejor)
  - contradictions  (peso 15%) — # contradicciones unresolved (menos = mejor)
  - dupes           (peso 10%) — # títulos duplicados en >1 file (menos = mejor)
  - dead_notes      (peso 15%) — # notas >365d sin tocar Y sin queries en 180d

Cada señal se mapea a un sub-score 0-100 mediante una función monotónica
(lineal-clipped). El score final es ``sum(weight × sub_score)`` con los
pesos arriba (suman 1.0).

Cache: el cálculo es relativamente caro (lee toda la tabla `meta_*` +
agrega contradicciones + cruza paths con `rag_queries`), pero los signals
no cambian rápido. Cacheamos el resultado en memoria con TTL 5min — el
endpoint web y el card del dashboard hablan contra el cache, no contra
la DB. Sin TTL real-time (no es crítico que el score esté al segundo).

Robustez: cualquier query SQL que falle (DB locked, schema mismatch,
permission error) hace que su componente individual valga 0 y se loguea
silencioso vía `logger.debug`. NUNCA raisea hacia el caller — el endpoint
se asegura de que el dashboard reciba un JSON parseable aunque el vault
esté en estado degradado.

Off switch: setear ``OBSIDIAN_RAG_VAULT_HEALTH=0`` deshabilita el cálculo
(``compute_vault_health()`` retorna un dict con ``score=None`` y
``error="disabled"``). Útil si en algún momento esto pesa lo suficiente
como para querer apagarlo sin tirar el endpoint.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Pesos ─────────────────────────────────────────────────────────────────
WEIGHTS: dict[str, float] = {
    "tags_pct":       0.20,
    "backlinks_pct":  0.25,
    "orphans":        0.15,
    "contradictions": 0.15,
    "dupes":          0.10,
    "dead_notes":     0.15,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1.0"

# Descripciones para tooltips del front (en español rioplatense, contando
# qué mide cada señal en una línea).
DESCRIPTIONS: dict[str, str] = {
    "tags_pct":
        "% notas con al menos 1 tag — taxonomía mínima para retrieval por filtros.",
    "backlinks_pct":
        "% notas que reciben al menos 1 wikilink desde otra nota — conectividad del vault.",
    "orphans":
        "Notas sin links entrantes ni salientes — tienden a perderse del retrieval.",
    "contradictions":
        "Contradicciones detectadas pero todavía sin resolver (`skipped` vacío en `rag_contradictions`).",
    "dupes":
        "Títulos de notas que aparecen en ≥2 archivos distintos — proxy de re-imports / duplicados.",
    "dead_notes":
        "Notas viejas (>365d sin modificar) y sin queries en los últimos 180 días — candidatas a archivar.",
}

# ── Cache (TTL 5min) ──────────────────────────────────────────────────────
_CACHE_TTL_S: float = 300.0
_cache_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_cache_ts: float = 0.0


def _now() -> float:
    """Indirection so tests pueden monkeypatchear sin tocar el time global."""
    return time.monotonic()


def invalidate_cache() -> None:
    """Limpia el cache. Llamado por tests + manualmente cuando hace falta."""
    global _cache, _cache_ts
    with _cache_lock:
        _cache = None
        _cache_ts = 0.0


# ── Helpers de acceso a DB ────────────────────────────────────────────────


def _ragvec_db_path() -> Path:
    """Path absoluto a ragvec.db. Lazy import para honrar monkeypatch en tests."""
    from rag import DB_PATH  # noqa: PLC0415

    return DB_PATH / "ragvec.db"


def _telemetry_db_path() -> Path:
    """Path absoluto a telemetry.db."""
    from rag import DB_PATH, _TELEMETRY_DB_FILENAME  # noqa: PLC0415

    return DB_PATH / _TELEMETRY_DB_FILENAME


def _meta_table_name() -> str:
    """Nombre de la tabla meta para el vault activo. ``meta_<COLLECTION_NAME>``."""
    from rag import COLLECTION_NAME  # noqa: PLC0415

    return f"meta_{COLLECTION_NAME}"


def _open_ro(path: Path) -> sqlite3.Connection | None:
    """Abre una conexión RO. Devuelve None si el archivo no existe o falla."""
    if not path.exists():
        return None
    try:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0, check_same_thread=False)
        conn.execute("PRAGMA busy_timeout=2000")
        return conn
    except Exception as exc:
        logger.debug("vault_health: cannot open %s ro: %s", path, exc)
        return None


# ── Componentes individuales ──────────────────────────────────────────────
#
# Cada función devuelve un dict parcial. Si la query falla, devuelve los
# defaults (counts en 0, listas vacías) sin raisear.


def _read_notes_meta() -> dict[str, Any]:
    """Lee metadata distinct-por-nota desde la tabla meta_<vault>.

    Devuelve:
      {
        "total":          int,        # # notas distinct (por path)
        "with_tags":      int,        # # notas con tags no vacíos
        "outlinks_by_path": {path: [titles...]},
        "titles_by_path": {path: title},
        "modified_by_path": {path: datetime or None},
      }

    Si la DB no existe o la query falla, devuelve totals en 0.
    """
    out: dict[str, Any] = {
        "total": 0,
        "with_tags": 0,
        "outlinks_by_path": {},
        "titles_by_path": {},
        "modified_by_path": {},
    }
    db = _ragvec_db_path()
    conn = _open_ro(db)
    if conn is None:
        return out
    try:
        meta = _meta_table_name()
        # Verificamos que la tabla existe — el vault podría no estar indexado.
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (meta,),
            ).fetchone()
            if not row:
                # Fallback: a veces el vault default usa la tabla sin sufijo.
                # Si COLLECTION_NAME apunta a una tabla que no existe pero la
                # base sí, la usamos como mejor esfuerzo (mismo schema).
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    ("meta_obsidian_notes_v11",),
                ).fetchone()
                if not row:
                    return out
                meta = "meta_obsidian_notes_v11"
        except Exception as exc:
            logger.debug("vault_health: sqlite_master probe failed: %s", exc)
            return out

        # 1 row por chunk → agregamos por `file` (el path real del .md).
        # Para tags y outlinks tomamos cualquier valor no vacío del path
        # (todos los chunks del mismo archivo comparten metadata por
        # construcción del indexer). MIN/MAX da resultados deterministas.
        sql = (
            f'SELECT file, MAX(note) AS note, MAX(tags) AS tags, '
            f'MAX(outlinks) AS outlinks, MAX(extra_json) AS extra_json '
            f'FROM "{meta}" '
            f'WHERE file IS NOT NULL AND file != "" '
            f'GROUP BY file'
        )
        try:
            rows = conn.execute(sql).fetchall()
        except Exception as exc:
            logger.debug("vault_health: read %s failed: %s", meta, exc)
            return out

        total = 0
        with_tags = 0
        outlinks_by_path: dict[str, list[str]] = {}
        titles_by_path: dict[str, str] = {}
        modified_by_path: dict[str, datetime | None] = {}
        for r in rows:
            path = r[0] or ""
            if not path:
                continue
            total += 1
            tags_raw = (r[2] or "").strip()
            if tags_raw:
                # Tags es CSV; algunos vacíos contaban como "" sólo
                # consideramos "tiene tags" si hay ≥1 token no vacío.
                if any(t.strip() for t in tags_raw.split(",")):
                    with_tags += 1

            outlinks_raw = (r[3] or "").strip()
            if outlinks_raw:
                links = [t.strip() for t in outlinks_raw.split(",") if t.strip()]
            else:
                links = []
            outlinks_by_path[path] = links

            title = (r[1] or "").strip()
            if title:
                titles_by_path[path] = title

            mod_dt: datetime | None = None
            extra_raw = r[4]
            if extra_raw:
                try:
                    extra = json.loads(extra_raw)
                    mod_str = extra.get("modified") or extra.get("created")
                    if mod_str:
                        try:
                            # ISO with TZ, ej "2026-04-13T16:59:19-03:00"
                            mod_dt = datetime.fromisoformat(mod_str)
                        except Exception:
                            mod_dt = None
                except Exception:
                    mod_dt = None
            modified_by_path[path] = mod_dt

        out.update({
            "total": total,
            "with_tags": with_tags,
            "outlinks_by_path": outlinks_by_path,
            "titles_by_path": titles_by_path,
            "modified_by_path": modified_by_path,
        })
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return out


def _compute_backlinks(
    outlinks_by_path: dict[str, list[str]],
    titles_by_path: dict[str, str],
) -> dict[str, set[str]]:
    """Resuelve backlinks: title → set de paths que la linkean."""
    title_to_paths: dict[str, set[str]] = {}
    for path, title in titles_by_path.items():
        if title:
            title_to_paths.setdefault(title, set()).add(path)

    backlinks_by_path: dict[str, set[str]] = {}
    for src_path, links in outlinks_by_path.items():
        for title in links:
            for target_path in title_to_paths.get(title, ()):
                if target_path != src_path:
                    backlinks_by_path.setdefault(target_path, set()).add(src_path)
    return backlinks_by_path


def _count_unresolved_contradictions() -> int:
    """Cuenta filas de rag_contradictions con `skipped` IS NULL OR ''.

    `skipped` es el "resolved" flag — cuando el user dismiss-ea la
    contradicción (o el helper la skip-ea) se setea. Si está vacío
    significa que la contradicción está pendiente de revisión.
    """
    db = _telemetry_db_path()
    conn = _open_ro(db)
    if conn is None:
        return 0
    try:
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM rag_contradictions "
                "WHERE skipped IS NULL OR skipped = ''"
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        except Exception as exc:
            logger.debug("vault_health: contradictions query failed: %s", exc)
            return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _count_dupe_titles(titles_by_path: dict[str, str]) -> int:
    """# títulos que aparecen en ≥2 archivos distintos.

    Proxy razonable para "dupes con sim ≥0.85 sin merge": cuando dos
    notas comparten título suelen ser re-imports, daily-notes mal
    movidos, o copies. Sin un tracking real de merge usamos esto.
    """
    title_to_paths: dict[str, set[str]] = {}
    for path, title in titles_by_path.items():
        if title:
            title_to_paths.setdefault(title, set()).add(path)
    return sum(1 for paths in title_to_paths.values() if len(paths) >= 2)


def _query_paths_recent(days: int = 180) -> set[str]:
    """Set de paths de notas referenciadas en `rag_queries.paths_json` en los
    últimos ``days``. Usado para detectar dead notes."""
    db = _telemetry_db_path()
    conn = _open_ro(db)
    if conn is None:
        return set()
    try:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        try:
            rows = conn.execute(
                "SELECT paths_json FROM rag_queries "
                "WHERE ts >= ? AND paths_json IS NOT NULL AND paths_json != ''",
                (cutoff,),
            ).fetchall()
        except Exception as exc:
            logger.debug("vault_health: rag_queries query failed: %s", exc)
            return set()

        seen: set[str] = set()
        for r in rows:
            raw = r[0]
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str) and item:
                        seen.add(item)
            elif isinstance(data, dict):
                # paths_json a veces es {path: score}
                for k in data.keys():
                    if isinstance(k, str) and k:
                        seen.add(k)
        return seen
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _count_dead_notes(
    modified_by_path: dict[str, datetime | None],
    recent_paths: set[str],
    age_days: int = 365,
) -> int:
    """# notas con modified > age_days Y no queryeadas en `recent_paths`."""
    if not modified_by_path:
        return 0
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=age_days)
    dead = 0
    for path, mod_dt in modified_by_path.items():
        if mod_dt is None:
            # Sin fecha de modified — la consideramos "vieja sin metadata".
            # Siendo conservadores no la contamos como dead (dato faltante,
            # no señal de muerte).
            continue
        # Normalizar TZ — algunos ISO traen offset, otros no.
        if mod_dt.tzinfo is None:
            mod_dt_aware = mod_dt.replace(tzinfo=timezone.utc)
        else:
            mod_dt_aware = mod_dt.astimezone(timezone.utc)
        if mod_dt_aware > cutoff:
            continue
        if path in recent_paths:
            continue
        dead += 1
    return dead


# ── Sub-scores ────────────────────────────────────────────────────────────
#
# Cada función mapea su input a [0, 100], monotónica:
#   - higher input → higher score (tags_pct, backlinks_pct)
#   - lower input → higher score (orphans, contradictions, dupes, dead_notes)


def _score_pct(pct: float) -> float:
    """Identidad clamped — input ya viene como %."""
    return max(0.0, min(100.0, float(pct)))


def _score_orphans(orphans: int, total: int) -> float:
    """% no-orphans clipeado. total == 0 → 0 (vault vacío no tiene salud)."""
    if total <= 0:
        return 0.0
    pct_orphans = 100.0 * orphans / total
    return max(0.0, 100.0 - pct_orphans)


def _score_contradictions(count: int) -> float:
    """Lineal clipeado: 0 contras = 100, ≥50 contras = 0."""
    if count <= 0:
        return 100.0
    return max(0.0, 100.0 - 100.0 * count / 50.0)


def _score_dupes(dupes: int, total: int) -> float:
    """Lineal clipeado: 0 dupes = 100. Saturación a total/4 dupes = 0."""
    if total <= 0:
        return 0.0
    threshold = max(1.0, total / 4.0)
    return max(0.0, 100.0 - 100.0 * dupes / threshold)


def _score_dead(dead: int, total: int) -> float:
    """Lineal clipeado: 0 dead = 100. Saturación a 50% dead = 0."""
    if total <= 0:
        return 0.0
    threshold = max(1.0, total / 2.0)
    return max(0.0, 100.0 - 100.0 * dead / threshold)


# ── Entry points ──────────────────────────────────────────────────────────


def get_components(force: bool = False) -> dict[str, Any]:
    """Devuelve solo el dict de componentes (sin el score agregado)."""
    return _compute(force=force)["components"]


def compute_vault_health(force: bool = False) -> dict[str, Any]:
    """Calcula y devuelve el health score + componentes + metadata.

    Cacheado por 5 minutos. Pasar ``force=True`` ignora el cache.

    Output:
        {
          "score": int,                 # 0-100 (None si feature off / error fatal)
          "components": {
              "tags_pct":       int,
              "backlinks_pct":  int,
              "orphans":        int,
              "contradictions": int,
              "dupes":          int,
              "dead_notes":     int,
              "total_notes":    int,
              "sub_scores": {
                  "tags_pct":       float,
                  "backlinks_pct":  float,
                  "orphans":        float,
                  "contradictions": float,
                  "dupes":          float,
                  "dead_notes":     float,
              },
              "descriptions": {key: str, ...},
          },
          "weights":         {key: float, ...},
          "last_calculated": "ISO ts",
          "ttl_seconds":     300,
          "error":           None | str,
        }
    """
    if os.environ.get("OBSIDIAN_RAG_VAULT_HEALTH", "1").strip() == "0":
        return {
            "score": None,
            "components": {},
            "weights": dict(WEIGHTS),
            "last_calculated": datetime.now().isoformat(timespec="seconds"),
            "ttl_seconds": int(_CACHE_TTL_S),
            "error": "disabled",
        }
    return _compute(force=force)


def _compute(force: bool = False) -> dict[str, Any]:
    """Hot path con cache. NUNCA raisea — cualquier excepción → score=None."""
    global _cache, _cache_ts
    with _cache_lock:
        now = _now()
        if (
            not force
            and _cache is not None
            and (now - _cache_ts) < _CACHE_TTL_S
        ):
            return _cache

    try:
        notes = _read_notes_meta()
        total = int(notes["total"])
        with_tags = int(notes["with_tags"])
        outlinks_by_path = notes["outlinks_by_path"]
        titles_by_path = notes["titles_by_path"]
        modified_by_path = notes["modified_by_path"]

        backlinks_by_path = _compute_backlinks(outlinks_by_path, titles_by_path)

        with_backlinks = sum(
            1 for p in outlinks_by_path
            if backlinks_by_path.get(p)
        )
        # Orphans: nota sin in/outlinks. Solo cuenta si conocemos la nota
        # (está en outlinks_by_path); las que no tienen file no entran.
        orphans = 0
        for path, links in outlinks_by_path.items():
            has_out = bool(links)
            has_in = bool(backlinks_by_path.get(path))
            if not has_out and not has_in:
                orphans += 1

        contradictions = _count_unresolved_contradictions()
        dupes = _count_dupe_titles(titles_by_path)
        recent_paths = _query_paths_recent(days=180)
        dead_notes = _count_dead_notes(modified_by_path, recent_paths, age_days=365)

        tags_pct = (100.0 * with_tags / total) if total > 0 else 0.0
        backlinks_pct = (100.0 * with_backlinks / total) if total > 0 else 0.0

        sub_scores = {
            "tags_pct":       _score_pct(tags_pct),
            "backlinks_pct":  _score_pct(backlinks_pct),
            "orphans":        _score_orphans(orphans, total),
            "contradictions": _score_contradictions(contradictions),
            "dupes":          _score_dupes(dupes, total),
            "dead_notes":     _score_dead(dead_notes, total),
        }
        weighted = sum(WEIGHTS[k] * sub_scores[k] for k in WEIGHTS)
        score = int(round(weighted))

        result = {
            "score": score if total > 0 else 0,
            "components": {
                "tags_pct":       int(round(tags_pct)),
                "backlinks_pct":  int(round(backlinks_pct)),
                "orphans":        int(orphans),
                "contradictions": int(contradictions),
                "dupes":          int(dupes),
                "dead_notes":     int(dead_notes),
                "total_notes":    int(total),
                "sub_scores":     {k: round(v, 2) for k, v in sub_scores.items()},
                "descriptions":   dict(DESCRIPTIONS),
            },
            "weights":         dict(WEIGHTS),
            "last_calculated": datetime.now().isoformat(timespec="seconds"),
            "ttl_seconds":     int(_CACHE_TTL_S),
            "error":           None,
        }
    except Exception as exc:
        # Defensive: no debería llegar acá (cada componente cacha sus
        # propias excepciones), pero garantizamos que el endpoint
        # nunca recibe un None inexpected.
        logger.debug("vault_health: unexpected failure in _compute: %s", exc)
        result = {
            "score":           None,
            "components":      {},
            "weights":         dict(WEIGHTS),
            "last_calculated": datetime.now().isoformat(timespec="seconds"),
            "ttl_seconds":     int(_CACHE_TTL_S),
            "error":           f"{type(exc).__name__}: {exc}",
        }

    with _cache_lock:
        _cache = result
        _cache_ts = _now()
    return result


__all__ = [
    "WEIGHTS",
    "DESCRIPTIONS",
    "compute_vault_health",
    "get_components",
    "invalidate_cache",
]
