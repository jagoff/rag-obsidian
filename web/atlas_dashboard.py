"""Backend del dashboard `/atlas` — atlas semántico del vault.

Combina dos fuentes de telemetría que ya existen en disco pero hasta hoy no
tenían vista web dedicada:

1. **Entidades** (telemetry.db: `rag_entities` + `rag_entity_mentions`).
   Pobladas durante indexing por GLiNER (ver
   ``rag/__init__.py:6080`` y siguientes). Discriminador
   ``(normalized, entity_type)`` con aliases en JSON. ~3.7k entities y
   ~19k mentions a 2026-04-29.

2. **Wikilinks entre notas** (ragvec.db: `meta_obsidian_notes_v11_*`).
   Cada chunk tiene una columna ``outlinks`` (CSV de targets de wikilinks
   resueltos a paths de notas). Esto es lo que reconstruye el "graph view"
   estilo Obsidian — nodos = notas, edges = wikilinks 1-hop.

Funciones públicas:
    snapshot(window_days, top_entities, graph_top_notes) → dict con shape
        estable consumido por ``GET /api/atlas``.

Performance budget: cold ≈ 200-500ms (depende de cuántas entities/notas
hay), warm < 5ms. El endpoint le pone TTL de 60s encima.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Cache compartido — clave incluye (window_days, top_entities, graph_top_notes,
# vault_path, telemetry_mtime, ragvec_mtime). Re-index del vault o nuevo
# tick del scheduler invalida automáticamente.
_DASHBOARD_CACHE: dict = {"key": None, "payload": None}
_DASHBOARD_CACHE_LOCK = threading.Lock()


# ── Junk entity filter ────────────────────────────────────────────────────────
# GLiNER (el extractor que puebla rag_entities) a veces clasifica como
# `person` strings que en realidad son números de teléfono internacionales
# o IDs puramente numéricos del WhatsApp bridge (ej. `5493424303891`,
# `34084894028025`). En la lista del atlas eso es ruido — bajan al user
# entidades reales del top y desordenan los rankings de hot/stale.
#
# Heurística: descartar nombres sin al menos UNA letra unicode. Cubre
# números puros, números con separadores (`+54 9 342 ...`), strings vacías,
# y strings de solo símbolos. Mantenemos legítimos como "5to grado",
# "C3PO", "Av. 9 de Julio" porque tienen al menos una letra.
#
# Aplicado tanto al query SELECT (con buffer 3x para no quedar por debajo
# del top_per_type post-filter) como a hot/stale (que iteran sobre `full`
# que ya viene filtrado).
_HAS_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def _is_junk_entity_name(name: str | None) -> bool:
    """True si el nombre es ruido del extractor (número, ID, solo símbolos).

    Usado para filtrar entidades GLiNER que no deberían aparecer en el
    top de personas/lugares/orgs/eventos del atlas. Mantiene cualquier
    string que contenga al menos una letra (incluye unicode — para que
    "Á", "Ñ", "ç" no se descarten).
    """
    if not name:
        return True
    s = str(name).strip()
    if len(s) < 2:  # "yo" pasa, "x" no
        return True
    if not _HAS_LETTER_RE.search(s):
        return True
    return False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _empty_payload(now: datetime, reason: str) -> dict:
    """Shape válido pero vacío para que el frontend pinte 'sin datos' sin
    checks defensivos."""
    return {
        "meta": {
            "generated_at": now.isoformat(timespec="seconds"),
            "reason": reason,
        },
        "kpis": {
            "n_entities": 0,
            "n_mentions": 0,
            "n_notes": 0,
            "n_edges": 0,
        },
        "entities_by_type": {"person": [], "location": [], "organization": [], "event": []},
        "hot": [],
        "stale": [],
        "cooccurrence": [],
        "graph": {"nodes": [], "links": []},
    }


# ── 1. Entidades ────────────────────────────────────────────────────────────


def _query_entities_by_type(
    conn: sqlite3.Connection, top_per_type: int, window_days: int
) -> tuple[dict[str, list[dict]], dict[int, dict]]:
    """Devuelve top N entities por tipo + dict de id → entity full.

    Cada entity incluye un sparkline de menciones por día en la ventana
    `window_days` (lista de int de longitud `window_days`, oldest-first)
    para que el frontend pinte mini-trends sin queries adicionales.
    """
    types = ("person", "location", "organization", "event")
    result: dict[str, list[dict]] = {t: [] for t in types}
    full: dict[int, dict] = {}

    now = _utc_now()
    window_start_ts = now.timestamp() - window_days * 86400
    # Hot vs stale split — comparamos last `window_days` vs los `window_days`
    # previos para detectar entidades que están subiendo (hot) vs hundiéndose.
    prev_window_start_ts = now.timestamp() - 2 * window_days * 86400

    for entity_type in types:
        # Buffer 3x: el filter de junk (`_is_junk_entity_name`) descarta
        # ~10% de los hits en el top de `person` (números de teléfono
        # del WA bridge). Pedimos 3x para que post-filter SIEMPRE alcance
        # `top_per_type` salvo en vaults pequeñísimos.
        rows = conn.execute(
            """
            SELECT id, canonical_name, entity_type, mention_count,
                   first_seen_ts, last_seen_ts, aliases
              FROM rag_entities
             WHERE entity_type = ?
             ORDER BY mention_count DESC
             LIMIT ?
            """,
            (entity_type, top_per_type * 3),
        ).fetchall()

        kept = 0
        for row in rows:
            if kept >= top_per_type:
                break
            ent_id, name, etype, mc, first_ts, last_ts, aliases_json = row
            if _is_junk_entity_name(name):
                continue
            kept += 1
            aliases: list[str] = []
            if aliases_json:
                try:
                    aliases = list(json.loads(aliases_json))[:5]
                except (ValueError, TypeError):
                    aliases = []

            # Sparkline: mentions por día en la ventana actual.
            spark_rows = conn.execute(
                """
                SELECT CAST(ts AS INTEGER) AS bucket, COUNT(*) AS n
                  FROM rag_entity_mentions
                 WHERE entity_id = ? AND ts >= ?
              GROUP BY CAST((? - ts) / 86400.0 AS INTEGER)
                """,
                (ent_id, window_start_ts, now.timestamp()),
            ).fetchall()
            # Bucket por día relativo a "ahora" (0 = hoy, window_days-1 = oldest).
            buckets = [0] * window_days
            for ts_int, _n in spark_rows:
                # Recompute bucket properly — la query de arriba agrupa pero
                # no ordena. Hacemos una segunda pasada barata por mention.
                pass
            # Más simple: hacer la cuenta en Python (window_days es chico, ~30).
            mention_ts_rows = conn.execute(
                "SELECT ts FROM rag_entity_mentions WHERE entity_id = ? AND ts >= ?",
                (ent_id, window_start_ts),
            ).fetchall()
            buckets = [0] * window_days
            for (ts,) in mention_ts_rows:
                if ts is None:
                    continue
                age_days = int((now.timestamp() - ts) / 86400.0)
                idx = window_days - 1 - age_days
                if 0 <= idx < window_days:
                    buckets[idx] += 1
            recent_mentions = sum(buckets)

            # Prev window count — para hot/stale ranking.
            prev_count = conn.execute(
                """
                SELECT COUNT(*)
                  FROM rag_entity_mentions
                 WHERE entity_id = ? AND ts >= ? AND ts < ?
                """,
                (ent_id, prev_window_start_ts, window_start_ts),
            ).fetchone()[0]

            ent = {
                "id": ent_id,
                "name": name,
                "type": etype,
                "mention_count": int(mc or 0),
                "recent_mentions": int(recent_mentions),
                "prev_mentions": int(prev_count or 0),
                "first_seen_ts": first_ts,
                "last_seen_ts": last_ts,
                "aliases": aliases,
                "sparkline": buckets,
            }
            result[entity_type].append(ent)
            full[ent_id] = ent

    return result, full


def _hot_and_stale(full: dict[int, dict], window_days: int, top: int = 10) -> tuple[list[dict], list[dict]]:
    """Hot = mayor crecimiento relativo último window vs prev window.
    Stale = mention_count alto pero 0 menciones recientes (último window).
    """
    now = _utc_now().timestamp()
    stale_cutoff = now - window_days * 86400

    hot: list[tuple[float, dict]] = []
    stale: list[tuple[float, dict]] = []

    for ent in full.values():
        recent = ent["recent_mentions"]
        prev = ent["prev_mentions"]
        # Hot score: crecimiento relativo, suavizado para evitar div-by-zero.
        # Solo entidades con al menos 3 menciones recientes para evitar ruido.
        if recent >= 3:
            growth = (recent - prev) / max(prev, 1)
            hot.append((growth, ent))

        # Stale: importante históricamente (mention_count >= 30), pero
        # sin menciones recientes en la ventana.
        last_ts = ent.get("last_seen_ts") or 0
        if ent["mention_count"] >= 30 and recent == 0 and last_ts < stale_cutoff:
            days_since = int((now - last_ts) / 86400.0) if last_ts else 9999
            ent_copy = dict(ent)
            ent_copy["days_since_last"] = days_since
            # Score: cuánto importaba × cuánto hace. Más viejo + más importante = más arriba.
            stale.append((ent["mention_count"] * days_since, ent_copy))

    hot.sort(key=lambda x: x[0], reverse=True)
    stale.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in hot[:top]], [e for _, e in stale[:top]]


def _cooccurrence(conn: sqlite3.Connection, full: dict[int, dict], top: int = 30) -> list[dict]:
    """Top pares de entidades que aparecen juntas en el mismo chunk.

    O(K²) por chunk donde K = entidades por chunk; en la práctica K es
    chico (~5-10). Se restringe a los entity_ids de `full` (top entidades)
    para no explotar.
    """
    if not full:
        return []
    ids = tuple(full.keys())
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    # Para cada chunk_id, traemos los entity_ids que están en `full`.
    # Después contamos pares en Python — más simple y rápido que self-join.
    rows = conn.execute(
        f"""
        SELECT chunk_id, entity_id
          FROM rag_entity_mentions
         WHERE entity_id IN ({placeholders})
        """,
        ids,
    ).fetchall()

    by_chunk: dict[str, list[int]] = defaultdict(list)
    for chunk_id, ent_id in rows:
        by_chunk[chunk_id].append(ent_id)

    pairs: Counter = Counter()
    for ent_list in by_chunk.values():
        if len(ent_list) < 2:
            continue
        unique_ids = sorted(set(ent_list))
        # All combinations — limit per chunk to avoid blowup if a chunk
        # has 50+ entities. Top 10 per chunk is plenty.
        if len(unique_ids) > 10:
            # Pick the top 10 by mention_count globally so the sample is
            # consistent across chunks.
            unique_ids.sort(
                key=lambda i: full.get(i, {}).get("mention_count", 0),
                reverse=True,
            )
            unique_ids = sorted(unique_ids[:10])
        for i in range(len(unique_ids)):
            for j in range(i + 1, len(unique_ids)):
                pairs[(unique_ids[i], unique_ids[j])] += 1

    out: list[dict] = []
    for (a, b), cnt in pairs.most_common(top):
        if a not in full or b not in full:
            continue
        out.append(
            {
                "a_id": a,
                "a_name": full[a]["name"],
                "a_type": full[a]["type"],
                "b_id": b,
                "b_name": full[b]["name"],
                "b_type": full[b]["type"],
                "count": int(cnt),
            }
        )
    return out


# ── 2. Graph estilo Obsidian (notas + wikilinks) ────────────────────────────


def _build_graph(meta_table: str, ragvec_db: Path, top_notes: int) -> dict:
    """Lee `meta_<collection>` y reconstruye nodos (notas) + edges (wikilinks).

    Cada nota tiene una columna `outlinks` con los TARGETS de los
    wikilinks (uno por entry; en la DB están como CSV resueltos a
    paths). Reconstruimos el grafo dirigido note→target, después
    colapsamos a edges no-dirigidos para layout (Obsidian también
    pinta el graph como undirected en su default view).

    Limitamos a `top_notes` por degree para que el frontend no
    explote — un vault de 5k notas con 20k edges es inviable
    renderizar en SVG sin paginar.
    """
    if not ragvec_db.exists():
        return {"nodes": [], "links": [], "truncated": False}

    conn = sqlite3.connect(f"file:{ragvec_db}?mode=ro", uri=True, timeout=30.0)
    try:
        # 1 row por chunk → necesitamos GROUP BY file para que cada nota
        # cuente una sola vez. Tomamos el primer outlinks no vacío de
        # cualquiera de los chunks de esa nota (todos los chunks de una
        # misma nota comparten outlinks porque vienen del mismo frontmatter
        # parse — ver _extract_wikilinks_from_markdown).
        rows = conn.execute(
            f"""
            SELECT file,
                   MAX(folder) AS folder,
                   GROUP_CONCAT(outlinks, ',') AS all_outlinks,
                   MAX(title) AS title,
                   MAX(area) AS area,
                   MAX(type) AS type,
                   MAX(created_ts) AS created_ts,
                   COUNT(*) AS n_chunks
              FROM {meta_table}
             WHERE file IS NOT NULL AND file != ''
          GROUP BY file
            """,
        ).fetchall()
    except sqlite3.OperationalError:
        # Tabla no existe (vault sin indexar). Devolvemos vacío.
        conn.close()
        return {"nodes": [], "links": [], "truncated": False}
    conn.close()

    if not rows:
        return {"nodes": [], "links": [], "truncated": False}

    # path → metadata
    notes: dict[str, dict] = {}
    # path → set(target paths) — los targets de outlinks ya vienen como
    # vault-relative paths (con .md), porque _resolve_wikilinks_to_paths los
    # convierte durante indexing.
    outlinks_by_path: dict[str, set[str]] = {}

    for row in rows:
        file_path, folder, all_outlinks, title, area, ntype, created_ts, n_chunks = row
        notes[file_path] = {
            "id": file_path,
            "label": title or Path(file_path).stem,
            "folder": folder or "",
            "area": area or "",
            "type": ntype or "",
            "created_ts": created_ts,
            "n_chunks": int(n_chunks or 0),
        }
        # GROUP_CONCAT puede traer duplicados; dedupe acá.
        targets = set()
        if all_outlinks:
            for tok in all_outlinks.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                # outlinks puede tener forma "path.md" o "title". Si no
                # tiene extensión, asumimos title y NO lo emitimos como
                # edge (no podemos resolver a path acá sin re-correr la
                # resolución del indexer). Solo emitimos edges cuya
                # destino sea un path válido que también existe en `notes`.
                targets.add(tok)
        outlinks_by_path[file_path] = targets

    # Edges: pares (src, dst) donde dst también es una nota indexada.
    # Colapsamos a undirected (a, b) ordenado.
    edges: Counter = Counter()
    for src, targets in outlinks_by_path.items():
        for dst in targets:
            # Match exacto primero (el indexer suele resolver a path completo
            # con .md). Si no matchea, intentamos heurística por basename.
            if dst not in notes:
                # Buscar por basename (case-insensitive). Es O(N) pero
                # sólo para los misses.
                dst_lower = dst.lower().rstrip(".md")
                for cand in notes:
                    if Path(cand).stem.lower() == dst_lower:
                        dst = cand
                        break
                else:
                    continue
            if dst == src:
                continue
            a, b = (src, dst) if src < dst else (dst, src)
            edges[(a, b)] += 1

    # Compute degree por nota.
    degree: Counter = Counter()
    for (a, b), w in edges.items():
        degree[a] += w
        degree[b] += w

    # Slice top notas por degree (las más conectadas — equivalente a lo
    # que Obsidian highlightea en su graph view).
    truncated = len(notes) > top_notes
    if truncated:
        keep_ids = {nid for nid, _ in degree.most_common(top_notes)}
        # Si quedan slots libres, llenamos con las notas con más chunks
        # (por si hay notas grandes pero sin links).
        if len(keep_ids) < top_notes:
            extras = sorted(
                (n for n in notes.values() if n["id"] not in keep_ids),
                key=lambda n: n["n_chunks"],
                reverse=True,
            )
            for n in extras[: top_notes - len(keep_ids)]:
                keep_ids.add(n["id"])
    else:
        keep_ids = set(notes.keys())

    nodes = []
    for nid in keep_ids:
        n = notes[nid]
        n["degree"] = int(degree.get(nid, 0))
        nodes.append(n)

    # Sort por degree desc para que el frontend pinte primero los más conectados.
    nodes.sort(key=lambda n: n["degree"], reverse=True)

    links = [
        {"source": a, "target": b, "weight": int(w)}
        for (a, b), w in edges.items()
        if a in keep_ids and b in keep_ids
    ]

    return {
        "nodes": nodes,
        "links": links,
        "truncated": truncated,
        "total_notes": len(notes),
        "total_edges": len(edges),
    }


# ── note_detail (side-panel del frontend al click en un nodo del grafo) ─────


def note_detail(
    *,
    path: str,
    vault_path: Path | None = None,
    preview_chars: int = 600,
    max_entities: int = 20,
    max_neighbors: int = 30,
) -> dict:
    """Devuelve datos detallados de UNA nota para el side-panel del atlas.

    Shape:
        { meta, preview, entities, neighbors, vault_uri }

    - `preview`: primeros ~600 chars del cuerpo (sin frontmatter).
    - `entities`: top entidades mencionadas en chunks de esta nota.
    - `neighbors`: notas conectadas 1-hop (in + out wikilinks).
    - `vault_uri`: deep link `obsidian://open?vault=<name>&file=<path>`
      para abrir la nota en Obsidian con un click.

    Fail-safe: cualquier error → payload vacío con `error` poblado, sin
    excepciones para evitar 500s en el frontend.
    """
    now = _utc_now()
    out: dict = {
        "meta": {
            "generated_at": now.isoformat(timespec="seconds"),
            "path": path,
        },
        "preview": "",
        "entities": [],
        "neighbors": [],
        "vault_uri": None,
        "error": None,
    }

    if not path or ".." in path or path.startswith("/"):
        out["error"] = "invalid_path"
        return out

    try:
        from rag import (  # type: ignore
            DB_PATH,
            VAULT_PATH,
            _TELEMETRY_DB_FILENAME,
            get_db,
            get_db_for,
        )
    except Exception as e:
        out["error"] = f"rag_import_failed:{e}"
        return out

    if vault_path is None:
        vault_path = VAULT_PATH
    note_full = Path(vault_path) / path

    # 1) Preview del cuerpo (skip frontmatter YAML).
    try:
        if note_full.exists() and note_full.is_file():
            text = note_full.read_text(encoding="utf-8", errors="replace")
            # Skip YAML frontmatter si está al inicio.
            if text.startswith("---"):
                end = text.find("---", 3)
                if end >= 0:
                    text = text[end + 3:].lstrip()
            out["preview"] = text[:preview_chars]
            if len(text) > preview_chars:
                out["preview"] += "…"
    except OSError as e:
        out["preview"] = f"(no se pudo leer la nota: {e})"

    # 2) Vault URI — el name del vault es el último segmento del path.
    try:
        vault_name = Path(vault_path).name
        # URL-encode mínimo: solo lo crítico que rompe el URI scheme.
        from urllib.parse import quote
        out["vault_uri"] = (
            f"obsidian://open?vault={quote(vault_name, safe='')}"
            f"&file={quote(path, safe='')}"
        )
    except Exception:
        pass

    # 3) Entities mencionadas en chunks de esta nota.
    telemetry_db = Path(DB_PATH) / _TELEMETRY_DB_FILENAME
    if telemetry_db.exists():
        try:
            tconn = sqlite3.connect(
                f"file:{telemetry_db}?mode=ro", uri=True, timeout=30.0
            )
            try:
                # En el schema actual, `m.source` es el bucket cross-source
                # (`vault`, `gmail`, `calendar`, etc.), NO el path. El path
                # vive en `chunk_id` con formato `<path>::<chunk_idx>`.
                # Filtramos con `chunk_id LIKE 'path::%'` para traer solo
                # las menciones que viven en chunks de ESTA nota específica.
                rows = tconn.execute(
                    """
                    SELECT e.id, e.canonical_name, e.entity_type, e.mention_count,
                           COUNT(*) AS chunks_in_note
                      FROM rag_entity_mentions m
                      JOIN rag_entities e ON e.id = m.entity_id
                     WHERE m.chunk_id LIKE ?
                  GROUP BY e.id
                  ORDER BY chunks_in_note DESC, e.mention_count DESC
                     LIMIT ?
                    """,
                    (f"{path}::%", max_entities),
                ).fetchall()
                for r in rows:
                    if _is_junk_entity_name(r[1]):
                        continue
                    out["entities"].append({
                        "id": r[0],
                        "name": r[1],
                        "type": r[2],
                        "mention_count": int(r[3] or 0),
                        "chunks_in_note": int(r[4] or 0),
                    })
            finally:
                tconn.close()
        except sqlite3.OperationalError as e:
            out["error"] = f"entities_query_failed:{e}"

    # 4) Vecinos 1-hop (in + out wikilinks). Reusamos el mismo path que
    # `_build_graph` pero filtrado al `path` solicitado, así no inflamos
    # el payload con todo el grafo.
    try:
        col = get_db_for(vault_path) if vault_path else get_db()
        meta_table = col._meta  # type: ignore[attr-defined]
        ragvec_db = Path(DB_PATH) / "ragvec.db"
        rconn = sqlite3.connect(
            f"file:{ragvec_db}?mode=ro", uri=True, timeout=30.0
        )
        try:
            # Out: outlinks de esta nota.
            out_rows = rconn.execute(
                f"""
                SELECT GROUP_CONCAT(outlinks, ',') AS all_outlinks,
                       MAX(folder) AS folder,
                       MAX(title) AS title
                  FROM {meta_table}
                 WHERE file = ?
                """,
                (path,),
            ).fetchone()
            outlinks_csv = (out_rows[0] if out_rows else "") or ""
            out_targets = {tok.strip() for tok in outlinks_csv.split(",") if tok.strip()}

            # In: notas que mencionan a esta en su outlinks.
            # Match por path exacto Y por basename (sin .md) para cubrir
            # ambas formas que produce `_resolve_wikilinks_to_paths`.
            note_basename = Path(path).stem
            in_rows = rconn.execute(
                f"""
                SELECT DISTINCT file, MAX(folder) AS folder, MAX(title) AS title
                  FROM {meta_table}
                 WHERE file != ?
                   AND outlinks IS NOT NULL AND outlinks != ''
                   AND (outlinks LIKE ? OR outlinks LIKE ? OR outlinks LIKE ?)
              GROUP BY file
                 LIMIT ?
                """,
                (
                    path,
                    f"%{path}%",
                    f"%{note_basename}.md%",
                    f"%{note_basename}%",
                    max_neighbors * 2,  # buffer porque LIKE puede traer false positives
                ),
            ).fetchall()

            seen = set()
            # Out neighbors first (notas que esta nota linkea).
            for tgt in list(out_targets)[:max_neighbors]:
                if tgt in seen:
                    continue
                seen.add(tgt)
                # Resolve target a path real si es solo un title.
                resolved = tgt if tgt.endswith(".md") else None
                row = rconn.execute(
                    f"SELECT MAX(folder), MAX(title) FROM {meta_table} WHERE file = ? OR file LIKE ? LIMIT 1",
                    (resolved or "", f"%{tgt}%"),
                ).fetchone()
                folder = (row[0] if row else "") or ""
                title = (row[1] if row else None) or Path(tgt).stem
                out["neighbors"].append({
                    "path": resolved or tgt,
                    "label": title,
                    "folder": folder,
                    "direction": "out",
                })

            # In neighbors (notas que linkean a esta).
            for r in in_rows:
                if len(out["neighbors"]) >= max_neighbors:
                    break
                file_p, folder, title = r
                if file_p in seen:
                    continue
                seen.add(file_p)
                out["neighbors"].append({
                    "path": file_p,
                    "label": title or Path(file_p).stem,
                    "folder": folder or "",
                    "direction": "in",
                })

            # Si no encontramos ni meta del file, igual seteamos label
            if out_rows and out_rows[2]:
                out["meta"]["title"] = out_rows[2]
                out["meta"]["folder"] = out_rows[1] or ""
        finally:
            rconn.close()
    except sqlite3.OperationalError as e:
        if not out["error"]:
            out["error"] = f"neighbors_query_failed:{e}"
    except Exception as e:
        if not out["error"]:
            out["error"] = f"neighbors_unexpected:{e}"

    return out


# ── snapshot ────────────────────────────────────────────────────────────────


def snapshot(
    *,
    window_days: int = 30,
    top_entities: int = 50,
    graph_top_notes: int = 250,
    vault_path: Path | None = None,
) -> dict:
    """Snapshot completo para `GET /api/atlas`.

    Args:
        window_days: ventana temporal para sparkline + hot/stale.
        top_entities: top N por tipo (4 tipos × N → max 4N entidades).
        graph_top_notes: cap del grafo de notas (top por degree).
        vault_path: override del vault activo (para tests / multi-vault).

    Returns:
        Dict con shape estable. Nunca lanza — silent-fail per convention.
    """
    now = _utc_now()
    window_days = max(7, min(int(window_days or 30), 365))
    top_entities = max(5, min(int(top_entities or 50), 200))
    graph_top_notes = max(50, min(int(graph_top_notes or 250), 2000))

    # Resolve DB paths via rag.* — respeta env overrides + multi-vault.
    try:
        from rag import DB_PATH, _TELEMETRY_DB_FILENAME, get_db, get_db_for, VAULT_PATH  # type: ignore
    except Exception as e:
        return _empty_payload(now, f"rag_import_failed:{e}")

    telemetry_db = Path(DB_PATH) / _TELEMETRY_DB_FILENAME
    ragvec_db = Path(DB_PATH) / "ragvec.db"

    if not telemetry_db.exists() or not ragvec_db.exists():
        return _empty_payload(now, "dbs_missing")

    # Cache key: incluye mtimes de ambas DBs + parámetros + vault path.
    if vault_path is None:
        vault_path = VAULT_PATH
    cache_key = (
        str(vault_path),
        _safe_mtime(telemetry_db),
        _safe_mtime(ragvec_db),
        window_days,
        top_entities,
        graph_top_notes,
    )
    with _DASHBOARD_CACHE_LOCK:
        if _DASHBOARD_CACHE.get("key") == cache_key and _DASHBOARD_CACHE.get("payload"):
            return _DASHBOARD_CACHE["payload"]

    # 1) Entities + sparklines + hot/stale
    try:
        tconn = sqlite3.connect(f"file:{telemetry_db}?mode=ro", uri=True, timeout=30.0)
    except sqlite3.OperationalError as e:
        return _empty_payload(now, f"telemetry_open_failed:{e}")

    try:
        # Sanity: tabla existe?
        tconn.execute("SELECT 1 FROM rag_entities LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        tconn.close()
        return _empty_payload(now, "rag_entities_missing")

    try:
        entities_by_type, full = _query_entities_by_type(tconn, top_entities, window_days)
        hot, stale = _hot_and_stale(full, window_days)
        cooc = _cooccurrence(tconn, full)
    finally:
        tconn.close()

    # Total counts para KPIs.
    try:
        tconn = sqlite3.connect(f"file:{telemetry_db}?mode=ro", uri=True, timeout=30.0)
        n_entities = tconn.execute("SELECT COUNT(*) FROM rag_entities").fetchone()[0]
        n_mentions = tconn.execute("SELECT COUNT(*) FROM rag_entity_mentions").fetchone()[0]
        tconn.close()
    except sqlite3.OperationalError:
        n_entities = sum(len(v) for v in entities_by_type.values())
        n_mentions = 0

    # 2) Graph de notas
    try:
        col = get_db_for(vault_path) if vault_path else get_db()
        meta_table = col._meta  # type: ignore[attr-defined]
    except Exception:
        meta_table = "meta_obsidian_notes_v11"

    graph = _build_graph(meta_table, ragvec_db, graph_top_notes)

    payload = {
        "meta": {
            "generated_at": now.isoformat(timespec="seconds"),
            "vault_path": str(vault_path),
            "meta_table": meta_table,
            "window_days": window_days,
            "top_entities": top_entities,
            "graph_top_notes": graph_top_notes,
        },
        "kpis": {
            "n_entities": int(n_entities or 0),
            "n_mentions": int(n_mentions or 0),
            "n_notes": int(graph.get("total_notes") or len(graph.get("nodes", []))),
            "n_edges": int(graph.get("total_edges") or len(graph.get("links", []))),
            "hot_count": len(hot),
            "stale_count": len(stale),
        },
        "entities_by_type": entities_by_type,
        "hot": hot,
        "stale": stale,
        "cooccurrence": cooc,
        "graph": graph,
    }

    with _DASHBOARD_CACHE_LOCK:
        _DASHBOARD_CACHE["key"] = cache_key
        _DASHBOARD_CACHE["payload"] = payload

    return payload
