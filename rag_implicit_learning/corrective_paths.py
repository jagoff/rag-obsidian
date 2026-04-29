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
- Ventana temporal default 600s (10 min). Si el user no abrió nada en
  ese rango Y tampoco re-preguntó con una paráfrasis útil, el feedback
  queda sin corrective. Pre-2026-04-29 era 60s — el bump destrabó casos
  donde el user lee la nota abierta antes de actuar (ver CLAUDE.md
  "loops de aprendizaje").
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
        "corrective_source": "implicit_behavior_inference"
                             | "implicit_paraphrase_inference",
        "corrective_inferred_at": "<iso datetime>",
        "corrective_in_top_k": true,  # o false si era una nav externa
    }

Sources posibles:
- `implicit_behavior_inference` — el user abrió OTRA nota dentro de la
  ventana; el ranker se equivocó eligiendo la #1.
- `implicit_paraphrase_inference` — el user no abrió nada pero
  reformuló la query (paráfrasis vía `requery_detection.is_paraphrase`)
  y la segunda corrida devolvió un top-1 distinto con `top_score >= 0.5`.
  Heurística más débil que el open (no sabemos si esa segunda respuesta
  fue útil), pero captura signal que de otro modo se perdería.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Default 600s (10 min) — pre-2026-04-29 era 60s. El bump capturó la
# realidad observada: el user lee la nota abierta antes de actuar (no
# necesariamente abre otra al toque). Con 60s solo cerrábamos 1
# corrective_path en 6 días; el gate de 20 (que dispara el LoRA fine-tune
# del reranker, ver CLAUDE.md §"Ranker-vivo") quedaba imposible de
# alcanzar pasivamente. La idempotencia está garantizada (skipea feedbacks
# que ya tienen `corrective_path` — línea ~205) así que un re-run con la
# nueva ventana no duplica trabajo. Override per-corrida via CLI
# `rag feedback infer-implicit --window-seconds N`.
DEFAULT_WINDOW_SECONDS = 600

# Threshold de top_score (cosine del cross-encoder reranker) para confiar
# en la paráfrasis follow-up: si el user reformuló y la nueva corrida
# devolvió un top-1 con score < 0.5, asumimos que tampoco encontró nada
# útil — no usamos ese path como corrective. Conservador (preferimos
# perder signal antes que envenenar el dataset del fine-tune con
# falsos positivos). Calibrado vs `_LOOKUP_THRESHOLD=0.6` (gate de
# confianza global) — mantenemos 0.1 de margen para captar paráfrasis
# semi-buenas que no llegaron al gate principal pero son útiles como
# negative-of-positive signal contra el top_path original.
DEFAULT_PARAPHRASE_TOP_SCORE_MIN = 0.5


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


def _recover_paths_from_behavior(
    conn: sqlite3.Connection,
    *,
    session: str,
    before_ts: str,
    window_seconds: int,
) -> list[str] | None:
    """Recuperar `paths_json` desde el último evento `query_response` (o `query`)
    en `rag_behavior` que precede al feedback dentro de la ventana.

    Quick Win #2 (2026-04-29): cuando el feedback row no tiene
    `paths_json` propio (228/270 = 84% de los feedbacks negativos del
    listener WA, donde la captura del bot no incluye sources), antes de
    skipear con `n_skip_no_paths` chequeamos si hay un evento de
    behavior reciente que tenga los paths citados. El listener TS
    postea a `/api/behavior` con `event=query_response` + `paths_json`
    inmediatamente después de mostrarle al user la respuesta del RAG.

    Devuelve la lista de paths del MÁS RECIENTE evento que matchea
    session, o None si no hay ninguno. Filtramos por session en Python
    (mismo patrón que `_opened_paths_in_window` arriba) — la session
    vive dentro del `extra_json`, no en columna nativa.

    Window semantics: buscamos eventos en `[before_ts - window_seconds,
    before_ts]`. El feedback ocurre DESPUÉS del query_response, así que
    miramos hacia atrás. Comparamos `<= before_ts` (no `<`) para tolerar
    el caso edge donde el listener postea el behavior y el feedback
    en el mismo segundo de wallclock.
    """
    rows = conn.execute(
        """
        SELECT extra_json, ts
        FROM rag_behavior
        WHERE event IN ('query_response', 'query')
          AND datetime(ts) <= datetime(?)
          AND datetime(ts) > datetime(?, '-' || ? || ' seconds')
        ORDER BY ts DESC
        """,
        (before_ts, before_ts, int(window_seconds)),
    ).fetchall()
    for extra_json_str, _ts in rows:
        try:
            extra = json.loads(extra_json_str or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if extra.get("session") != session:
            continue
        raw = extra.get("paths_json")
        # `paths_json` viene como lista nativa post-Quick Win #2 (porque
        # `_sql_serialise_row` JSON-encodea TODA la dict de extra). Pero
        # toleramos un string-of-array nested por si una versión vieja
        # (o un cliente custom) lo guardó así — defensivo.
        if isinstance(raw, list):
            paths = [p for p in raw if isinstance(p, str) and p]
            if paths:
                return paths
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, list):
                paths = [p for p in parsed if isinstance(p, str) and p]
                if paths:
                    return paths
    return None


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


def _infer_corrective_via_paraphrase(
    conn: sqlite3.Connection,
    *,
    after_ts: str,
    window_seconds: int,
    session: str,
    orig_q: str | None,
    top_path: str,
    paths: list[str],
    paraphrase_top_score_min: float = DEFAULT_PARAPHRASE_TOP_SCORE_MIN,
) -> str | None:
    """Fallback: si el user no abrió nada pero re-preguntó, el top-1 de la
    paráfrasis (con cosine ≥ threshold y distinto del original) es el
    corrective_path implícito.

    Heurística más débil que `_opened_paths_in_window` — el user puede
    haber re-preguntado y NO leído la nueva respuesta tampoco. Pero a
    nivel agregado, paráfrasis con top-1 distinto + score decente es
    signal de que el ranker original estaba en el camino equivocado.

    Args:
        conn: connection a `telemetry.db`.
        after_ts: timestamp del feedback negativo — buscamos queries
            posteriores a este ts.
        window_seconds: ventana temporal (igual que la de behavior).
        session: session id del feedback (debe matchear `rag_queries.session`).
        orig_q: la query original del feedback (para detectar paráfrasis
            con `is_paraphrase`).
        top_path: el #1 que el ranker eligió originalmente — si la
            paráfrasis devuelve el mismo top, no hay corrección.
        paths: paths_json del feedback original (para flag `in_top_k`,
            que el caller calcula después).
        paraphrase_top_score_min: cosine mínimo del cross-encoder en la
            corrida follow-up para confiar en su top-1.

    Returns:
        El path inferido como corrective, o `None` si no se encontró
        ninguna paráfrasis útil dentro del window.
    """
    if not orig_q:
        # Sin la query original no podemos comparar paráfrasis — silent skip.
        return None

    # Lazy import para evitar circular: requery_detection no importa nada
    # de este módulo, pero por consistencia con el resto del paquete lo
    # mantenemos lazy (el helper solo se llama en la branch fallback).
    from rag_implicit_learning.requery_detection import is_paraphrase

    try:
        rows = conn.execute(
            """
            SELECT q, paths_json, top_score, ts
            FROM rag_queries
            WHERE session = ?
              AND datetime(ts) > datetime(?)
              AND datetime(ts) < datetime(?, '+' || ? || ' seconds')
              AND q IS NOT NULL
              AND q != ''
              AND paths_json IS NOT NULL
              AND paths_json != ''
            ORDER BY datetime(ts) ASC
            LIMIT 5
            """,
            (session, after_ts, after_ts, int(window_seconds)),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        # `rag_queries` puede no existir en DBs minimal-schema (algunos
        # tests + bootstrap pre-`_ensure_telemetry_tables`). Silent-fail
        # alineado con el contrato del módulo: el read-side no rompe
        # la corrida — el feedback queda sin corrective y volvemos en
        # el próximo run.
        logger.debug(
            "paraphrase fallback skipped: rag_queries unavailable (%s)",
            exc,
        )
        return None

    for follow_q, follow_paths_json, follow_top_score, _follow_ts in rows:
        if not follow_q or follow_q == orig_q:
            continue
        if not is_paraphrase(orig_q, follow_q, similarity_threshold=0.5):
            continue
        # `top_score` es el cross-encoder cosine — silent-fail si vino
        # NULL o un string raro (no abortamos la corrida entera por una
        # row corrupta).
        try:
            score = (
                float(follow_top_score)
                if follow_top_score is not None
                else 0.0
            )
        except (TypeError, ValueError):
            score = 0.0
        if score < paraphrase_top_score_min:
            continue
        try:
            follow_paths = json.loads(follow_paths_json or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        if not follow_paths:
            continue
        candidate = follow_paths[0]
        # Si la paráfrasis cae en el MISMO top que el original, no hubo
        # cambio de ranking → no es corrective real (el user reformuló
        # pero el ranker insistió en lo mismo, no aprendemos nada nuevo).
        if candidate == top_path:
            continue
        return candidate  # type: ignore[no-any-return]

    return None


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
            como "el user respondió abriendo otra source o re-preguntó".
            Default 600 (post-2026-04-29; pre era 60).
        dry_run: si True, no escribe; solo reporta los updates que haría.
        only_feedback_id: si está, procesar solo ese feedback (útil para
            tests y para reprocesar uno específico tras un cambio).

    Returns:
        dict con métricas de la corrida + lista de updates inferidos:
            n_candidates: feedbacks negativos sin corrective_path
            n_inferred: cuántos updates se hicieron (o harían en dry-run)
            n_inferred_via_paraphrase: subset de `n_inferred` que vino
                de la rama paráfrasis (resto: rama opens-based)
            n_paths_recovered: feedbacks cuyo `paths_json` propio estaba
                vacío pero recuperamos los paths desde un evento
                `query_response` reciente del `rag_behavior` (Quick
                Win #2). Subset previo a la decisión final — un feedback
                que infiere usando paths recuperados cuenta tanto acá
                como en `n_inferred`.
            n_skip_*: razones por las que un candidato NO disparó update
            updates: lista de dicts con detalle por feedback inferido
                (incluye `corrective_source` por record)
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
        # Subset de `n_inferred` que vino de la rama paráfrasis (fallback).
        # El resto vino de la rama de opens-based (señal más fuerte).
        "n_inferred_via_paraphrase": 0,
        "n_skip_already_corrective": 0,
        "n_skip_no_session": 0,
        "n_skip_no_paths": 0,
        "n_skip_no_open": 0,
        "n_skip_opened_top": 0,
        # Quick Win #2 (2026-04-29): cuántos feedbacks no tenían
        # `paths_json` propio pero los recuperamos desde un evento
        # `query_response` reciente del `rag_behavior`. Subset previo a
        # la decisión final — un feedback que terminó inferiendo via
        # paths recuperados cuenta acá Y en `n_inferred`. Métrica de
        # impacto del wiring listener → /api/behavior. Pre-fix los 228
        # feedbacks negativos del WA caían en `n_skip_no_paths`; este
        # contador debería absorber la mayoría.
        "n_paths_recovered": 0,
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
            # Quick Win #2 (2026-04-29): el feedback no tiene paths_json
            # propio (caso típico WhatsApp, donde el bot no captura las
            # sources al row). Antes de skipear, intentamos recuperar
            # los paths desde el último `query_response` que el listener
            # TS posteó al `rag_behavior` para esta session, dentro de
            # la ventana hacia atrás. Si encontramos algo, seguimos el
            # flow normal (la rama de opens / paraphrase decide después
            # cuál es el `corrective_path`).
            recovered = _recover_paths_from_behavior(
                conn,
                session=session,
                before_ts=ts,
                window_seconds=window_seconds,
            )
            if recovered:
                paths = recovered
                metrics["n_paths_recovered"] += 1
        if not paths:
            metrics["n_skip_no_paths"] += 1
            continue

        top_path = paths[0]

        opens = _opened_paths_in_window(
            conn, after_ts=ts, window_seconds=window_seconds, session=session
        )

        corrective_path: str | None = None
        # Default source — se sobreescribe abajo si la rama paráfrasis matchea.
        corrective_source_local = "implicit_behavior_inference"

        if opens:
            # Rama 1 (señal fuerte): el user abrió OTRA nota dentro del
            # window. Buscar el primer open distinto al top — ese es el
            # corrective. Si todos los opens son del top, no inferimos
            # (el user no contradijo el ranking).
            for path, _open_ts in opens:
                if path != top_path:
                    corrective_path = path
                    break
            if corrective_path is None:
                metrics["n_skip_opened_top"] += 1
                continue
        else:
            # Rama 2 (fallback, señal más débil): no hubo opens, pero
            # tal vez el user reformuló la pregunta. Si la paráfrasis
            # devolvió un top-1 distinto con score decente, ESE es el
            # corrective implícito. Backwards-compat: si hay opens, la
            # rama 1 gana (no llegamos acá). Ver doc del helper para el
            # razonamiento del threshold.
            corrective_path = _infer_corrective_via_paraphrase(
                conn,
                after_ts=ts,
                window_seconds=window_seconds,
                session=session,
                orig_q=q,
                top_path=top_path,
                paths=paths,
            )
            if corrective_path is None:
                # Ni opens ni paráfrasis útil → bucket "no_open" sigue
                # siendo el contador apropiado (semánticamente "no
                # encontramos signal post-feedback en la ventana").
                metrics["n_skip_no_open"] += 1
                continue
            corrective_source_local = "implicit_paraphrase_inference"

        in_top_k = corrective_path in paths

        update_record = {
            "feedback_id": fb_id,
            "ts": ts,
            "turn_id": turn_id,
            "session": session,
            "query": q,
            "top_path": top_path,
            "corrective_path": corrective_path,
            "corrective_source": corrective_source_local,
            "in_top_k": in_top_k,
        }
        updates.append(update_record)

        if not dry_run:
            extra["corrective_path"] = corrective_path
            extra["corrective_source"] = corrective_source_local
            extra["corrective_inferred_at"] = now_iso
            extra["corrective_in_top_k"] = in_top_k
            conn.execute(
                "UPDATE rag_feedback SET extra_json = ? WHERE id = ?",
                (json.dumps(extra), fb_id),
            )
        metrics["n_inferred"] += 1
        if corrective_source_local == "implicit_paraphrase_inference":
            metrics["n_inferred_via_paraphrase"] += 1

    return {
        **metrics,
        "updates": updates,
        "dry_run": dry_run,
        "window_seconds": window_seconds,
    }
