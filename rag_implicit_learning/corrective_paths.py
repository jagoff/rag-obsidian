"""Inferir `corrective_path` desde behavior implícito.

Cuando el user da 👎 a una respuesta del chat web pero no le marca al
sistema CUÁL era la nota correcta, esa señal queda incompleta — el
`online-tune` nightly necesita `corrective_path` para mover los pesos
del ranker (gate de 20 corrective_paths para destrabar el fine-tune).

La idea: si dentro de los siguientes N segundos al 👎, el user abre
una source distinta a la #1 que se rankeó, ESA es la nota correcta
implícita. Recuperamos signal histórica sin que el user tenga que
tocar nada.

Ejemplo:
    Turn @ 18:00:00:
        Q: "cuánto debe Alex de la macbook"
        Top-5 ranking: [moka-foda.md, alex-pago.md, ...]
        User: 👎 a la respuesta (porque cita la nota equivocada).
    Turn @ 18:00:23 (23 segundos después):
        rag_behavior: open path=alex-pago.md
    →  Inferimos corrective_path = "alex-pago.md" para el 👎 anterior.

Reglas:
- Solo procesamos feedback con rating=-1 que NO tiene ya corrective_path
  (idempotente — reruns no rompen nada).
- Match por session_id (en `extra_json` de feedback es `session_id`, en
  behavior es `session` — discrepancia histórica del schema, lo manejamos
  acá sin migración).
- Ventana temporal default 60s. Si el user no abrió nada en ese rango,
  el feedback queda sin corrective.
- Si el path opened == top_path (el user abrió la #1 igual, después de
  haber dado 👎), NO inferimos corrective — no hay disconfirmación clara,
  podría ser que clickeó por curiosidad.
- Confidence baja si el opened path no estaba en los top-k mostrados al
  user (paths_json). Lo registramos pero marcamos `in_top_k=False`.

Operativo:
- `infer_corrective_paths_from_behavior(conn, dry_run=True)` reporta sin
  escribir.
- `infer_corrective_paths_from_behavior(conn, dry_run=False)` aplica los
  updates a `rag_feedback.extra_json`.

Schema de los updates al `extra_json`:
    {
        ...,
        "corrective_path": "<vault-relative path>",
        "corrective_source": "implicit_behavior_inference",
        "corrective_inferred_at": "<iso datetime>",
        "corrective_in_top_k": true,  # o false si era una nav externa
    }
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Default 60s — basado en heurística común de search UX: si el user no
# clickeó en 1 minuto, ya cambió de tarea / cerró el tab / decidió que
# nada le sirvió. Valores típicos en evaluación de rankers (Microsoft
# Learning to Rank, MSLR-WEB30k) son 30-90s. Tomamos el medio.
DEFAULT_WINDOW_SECONDS = 60


def _extract_session(extra_json_str: str | None) -> str | None:
    """Devuelve el session id desde el extra_json del feedback.

    Histórica: el schema de `rag_feedback.extra_json` usa `session_id`,
    pero `rag_behavior.extra_json` usa `session` (sin _id). Esta función
    se usa solo para el lado de feedback. La lectura del lado behavior
    se hace inline donde se necesita.
    """
    if not extra_json_str:
        return None
    try:
        data = json.loads(extra_json_str)
    except (json.JSONDecodeError, TypeError):
        return None
    return data.get("session_id") or data.get("session")


def _opened_paths_in_window(
    conn: sqlite3.Connection,
    *,
    after_ts: str,
    window_seconds: int,
    session: str,
) -> list[tuple[str, str]]:
    """Lista de (path, ts) de eventos `open` en `rag_behavior` posteriores
    a `after_ts` dentro de la ventana, filtrados por session match.

    La columna session vive en `extra_json.session` — no se puede filtrar
    en SQL puro sin un JSON_EXTRACT (sqlite tiene `json_extract` pero su
    perf depende del índice). Filtramos en Python después del fetch para
    mantener compatibilidad con sqlite sin JSON1 (caso edge).
    """
    # `datetime(ts)` en ambos lados normaliza el formato (`"YYYY-MM-DD HH:MM:SS"`,
    # con espacio) — los `ts` en disco usan ISO8601 con `T` (`"YYYY-MM-DDTHH:MM:SS"`)
    # y string-comparing `T` (0x54) vs ` ` (0x20) rompe el orden temporal.
    # Wrappear ambos lados con `datetime()` lo evita sin migrar schema.
    rows = conn.execute(
        """
        SELECT path, ts, extra_json
        FROM rag_behavior
        WHERE event = 'open'
          AND path IS NOT NULL
          AND path != ''
          AND datetime(ts) > datetime(?)
          AND datetime(ts) < datetime(?, '+' || ? || ' seconds')
        ORDER BY ts ASC
        """,
        (after_ts, after_ts, int(window_seconds)),
    ).fetchall()
    matching: list[tuple[str, str]] = []
    for path, ts, extra_json_str in rows:
        try:
            extra = json.loads(extra_json_str or "{}")
        except (json.JSONDecodeError, TypeError):
            extra = {}
        if extra.get("session") == session:
            matching.append((path, ts))
    return matching


def infer_corrective_paths_from_behavior(
    conn: sqlite3.Connection,
    *,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    dry_run: bool = False,
    only_feedback_id: int | None = None,
) -> dict[str, Any]:
    """Inferir `corrective_path` para feedback negativo desde behavior.

    Args:
        conn: Connection a `telemetry.db` (autocommit OK).
        window_seconds: cuántos segundos después del feedback considerar
            como "el user respondió abriendo otra source". Default 60.
        dry_run: si True, no escribe; solo reporta los updates que haría.
        only_feedback_id: si está, procesar solo ese feedback (útil para
            tests y para reprocesar uno específico tras un cambio).

    Returns:
        dict con métricas de la corrida + lista de updates inferidos:
            n_candidates: feedbacks negativos sin corrective_path
            n_inferred: cuántos updates se hicieron (o harían en dry-run)
            n_skip_*: razones por las que un candidato NO disparó update
            updates: lista de dicts con detalle por feedback inferido
            dry_run: bool, refleja el flag
    """
    where_clauses = ["rating = -1"]
    params: list[Any] = []
    if only_feedback_id is not None:
        where_clauses.append("id = ?")
        params.append(only_feedback_id)
    where_sql = " AND ".join(where_clauses)

    candidates = conn.execute(
        f"""
        SELECT id, ts, turn_id, q, paths_json, extra_json
        FROM rag_feedback
        WHERE {where_sql}
        ORDER BY ts ASC
        """,
        tuple(params),
    ).fetchall()

    metrics: dict[str, int] = {
        "n_candidates": len(candidates),
        "n_inferred": 0,
        "n_skip_already_corrective": 0,
        "n_skip_no_session": 0,
        "n_skip_no_paths": 0,
        "n_skip_no_open": 0,
        "n_skip_opened_top": 0,
    }
    updates: list[dict[str, Any]] = []
    now_iso = datetime.now().isoformat(timespec="seconds")

    for fb_id, ts, turn_id, q, paths_json_str, extra_json_str in candidates:
        try:
            extra = json.loads(extra_json_str or "{}")
        except (json.JSONDecodeError, TypeError):
            extra = {}

        if extra.get("corrective_path"):
            metrics["n_skip_already_corrective"] += 1
            continue

        session = _extract_session(extra_json_str)
        if not session:
            metrics["n_skip_no_session"] += 1
            continue

        try:
            paths = json.loads(paths_json_str or "[]")
        except (json.JSONDecodeError, TypeError):
            paths = []
        if not paths:
            metrics["n_skip_no_paths"] += 1
            continue

        top_path = paths[0]

        opens = _opened_paths_in_window(
            conn, after_ts=ts, window_seconds=window_seconds, session=session
        )

        if not opens:
            metrics["n_skip_no_open"] += 1
            continue

        # Buscar el primer open que sea distinto al top — ese es el
        # corrective. Si todos los opens son del top, el user no
        # contradijo el ranking → no inferimos.
        corrective_path: str | None = None
        for path, _open_ts in opens:
            if path != top_path:
                corrective_path = path
                break

        if corrective_path is None:
            metrics["n_skip_opened_top"] += 1
            continue

        in_top_k = corrective_path in paths

        update_record = {
            "feedback_id": fb_id,
            "ts": ts,
            "turn_id": turn_id,
            "session": session,
            "query": q,
            "top_path": top_path,
            "corrective_path": corrective_path,
            "in_top_k": in_top_k,
        }
        updates.append(update_record)

        if not dry_run:
            extra["corrective_path"] = corrective_path
            extra["corrective_source"] = "implicit_behavior_inference"
            extra["corrective_inferred_at"] = now_iso
            extra["corrective_in_top_k"] = in_top_k
            conn.execute(
                "UPDATE rag_feedback SET extra_json = ? WHERE id = ?",
                (json.dumps(extra), fb_id),
            )
        metrics["n_inferred"] += 1

    return {
        **metrics,
        "updates": updates,
        "dry_run": dry_run,
        "window_seconds": window_seconds,
    }
