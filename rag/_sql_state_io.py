"""SQL state readers + recorders + daemon log writer.

Phase 1b de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer los readers/recorders del SQL state stack desde
`rag/__init__.py` (que ya bajó de 64.5k → 64.2k LOC en Phase 1a).

## Qué vive acá

- `_read_queries_for_log(n, *, low_confidence=False)` — tail N events
  de `rag_queries` para el renderer del CLI `rag log`.
- `_read_feedback_map_for_log(turn_ids)` — mapping `{turn_id: ±1}`
  para la columna thumb-emoji del CLI.
- `_record_draft_decision(...)` — persiste decisión del user sobre
  draft WhatsApp (loop `/si`, `/no`, `/editar`).
- `_record_brief_feedback(...)` — persiste rating de brief
  (`positive`/`negative`/`mute`) — loop reactions 👍/👎/🔇.
- `_log_daemon_run_event(label, action, ...)` — append a
  `rag_daemon_runs` (control plane lifecycle).
- Constants: `_VALID_DRAFT_DECISIONS`, `_VALID_BRIEF_FEEDBACK_RATINGS`.

## Por qué lazy imports

Las funciones acá tienen deps en symbols definidos LATER en
`rag/__init__.py`:
  `_ragvec_state_conn`, `_sql_append_event`, `_sql_write_with_retry`,
  `_silent_log`, `CONFIDENCE_RERANK_MIN`.

Si las importáramos al top-level del módulo, habría circular import:
  `rag.__init__` → `from rag._sql_state_io import *`
  → `rag._sql_state_io` → `from rag import _silent_log`
  → AttributeError porque `rag.__init__` no terminó de ejecutarse.

Solución estándar Python: lazy import dentro de cada función. Costo:
~microsegundos por call (Python cachea el import). Acceptable para
funciones que hacen SQL (latencia >> import overhead).

`_map_draft_decision_row` / `_map_brief_feedback_row` SÍ se importan
al top: viven en `rag/_row_mappers.py` que solo depende de `datetime`
— sin circular import.

## Re-export

`rag/__init__.py` hace `from rag._sql_state_io import *  # noqa: F401, F403`.
Preserva 100% compat con call sites históricos (`from rag import
_record_draft_decision`, `monkeypatch.setattr(rag, "X", ...)`).
"""

from __future__ import annotations

import json
from datetime import datetime

# OK importar acá: `_row_mappers` solo depende de `datetime`, sin circular.
from rag._row_mappers import _map_brief_feedback_row, _map_draft_decision_row

__all__ = [
    "_VALID_DRAFT_DECISIONS",
    "_VALID_BRIEF_FEEDBACK_RATINGS",
    "_read_queries_for_log",
    "_read_feedback_map_for_log",
    "_record_draft_decision",
    "_record_brief_feedback",
    "_log_daemon_run_event",
]


# Decisión válida del user en RagNet: aprueba el draft tal cual ('si'),
# aprueba con edits ('editar'), lo descarta ('rejected'), o el draft
# venció sin respuesta del user ('expired'). El CHECK constraint de la
# tabla rag_draft_decisions enforcea el mismo set; lo replicamos en
# Python para fallar temprano sin tocar SQL.
_VALID_DRAFT_DECISIONS: frozenset[str] = frozenset({
    "approved_si", "approved_editar", "rejected", "expired",
})


# Set válido de ratings para `rag_brief_feedback`. Mismo dominio que el
# `rag_anticipate_feedback` (mute > negative > positive de precedencia
# en el parser TS); replicamos la frozenset acá para validación rápida
# sin tocar SQL. El CHECK de la tabla enforcea lo mismo a nivel DB.
_VALID_BRIEF_FEEDBACK_RATINGS: frozenset[str] = frozenset({
    "positive", "negative", "mute",
})


def _read_queries_for_log(
    n: int, *, low_confidence: bool = False,
) -> list[dict]:
    """Read the last N query events from `rag_queries` SQL for the CLI
    renderer. Returns event-shaped dicts in chronological order (newest
    last) to match the historical JSONL tail-read shape.

    Post-T10 (2026-04-19) this is the ONLY source — the JSONL
    `queries.jsonl` path got repurposed for conversation-writer
    observability events and no longer receives query events, which is
    why the pre-fix CLI rendered empty rows.

    `low_confidence=True` narrows the SELECT to `top_score <
    CONFIDENCE_RERANK_MIN` (excludes NULL scores to avoid matching
    metachat / create-intent turns that never scored anything).

    `turn_id` is hoisted from `extra_json` so the feedback join in the
    renderer can match without re-parsing the JSON blob for every row.

    Errors swallow + log via `_silent_log` + return [] — matches the
    degradation semantics of other SQL readers in the module; the CLI
    prints an empty table rather than blowing up.
    """
    # Lazy import — _ragvec_state_conn + _silent_log + CONFIDENCE_RERANK_MIN
    # viven en `rag/__init__.py`. Top-level import causaría circular import.
    from rag import (  # noqa: PLC0415
        CONFIDENCE_RERANK_MIN,
        _ragvec_state_conn,
        _silent_log,
    )

    if n is None or n <= 0:
        return []
    # Always exclude rows with empty `q` — these come from admin-style
    # events (followup, read, etc) that co-opt `rag_queries` to log their
    # run metadata but don't represent actual search queries. Pre-T10 the
    # JSONL tail didn't mix them because the older `log_query_event` path
    # didn't insert rows with `q=""`; post-T10 `_map_queries_row` hardens
    # against NOT NULL constraint with `setdefault("q", "")`, so those
    # admin rows now pile up. Filter them here rather than reshaping the
    # writers — the table still exposes them for downstream analytics.
    filters = ["q IS NOT NULL", "q != ''"]
    params: list = []
    if low_confidence:
        filters.append("top_score IS NOT NULL")
        filters.append("top_score < ?")
        params.append(CONFIDENCE_RERANK_MIN)
    where_clause = "WHERE " + " AND ".join(filters)
    sql = (
        "SELECT ts, cmd, q, session, mode, top_score, t_retrieve, t_gen,"
        " answer_len, citation_repaired, critique_fired, critique_changed,"
        " extra_json"
        f" FROM rag_queries {where_clause}"
        " ORDER BY ts DESC LIMIT ?"
    )
    params.append(int(n))
    try:
        with _ragvec_state_conn() as conn:
            cursor = conn.execute(sql, tuple(params))
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
    except Exception as exc:
        _silent_log("rag_log_sql_read", exc)
        return []
    # SQL ORDER BY ts DESC → flip to chronological so the renderer scrolls
    # like tail -f (newest at the bottom).
    events: list[dict] = []
    for row in reversed(rows):
        d = dict(zip(cols, row))
        extra = d.pop("extra_json", None)
        if extra:
            try:
                ej = json.loads(extra) if isinstance(extra, str) else extra
            except Exception:
                ej = None
            if isinstance(ej, dict):
                # Only hoist `turn_id` — other extras stay in the blob.
                # Preserves the minimal surface the renderer needs.
                tid = ej.get("turn_id")
                if tid:
                    d["turn_id"] = tid
        events.append(d)
    return events


def _read_feedback_map_for_log(
    turn_ids: frozenset[str] | None = None,
) -> dict[str, int]:
    """Read `{turn_id: +1|-1}` from `rag_feedback` for the CLI renderer's
    thumb-emoji column. Replaces the pre-T10 tail-read of `FEEDBACK_PATH`
    (`feedback.jsonl`).

    Normalisation: raw rating is any non-zero int; map collapses to +1/-1
    so the renderer's emoji logic stays unchanged. On duplicate turn_ids
    the LATEST by `ts` wins — matches user intent when someone flips
    their thumbs (down → up).

    Rows without `turn_id` (global scope feedback) are skipped — they
    can't attach to a specific query row in the rendered table.

    `turn_ids` — when provided, restricts the SELECT to only those ids via
    an IN clause. The caller always knows which turn_ids it fetched from
    `rag_queries`, so passing them avoids a full-table scan when feedback
    has accumulated thousands of rows.

    Audit perf 2026-05-08: si `turn_ids` es `None` o vacío, devolvemos
    `{}` directo. El renderer de `rag log` solo necesita los ratings de
    los turn_ids que ya fetcheó de `rag_queries`; no tiene caso pagar
    un full-table scan a `rag_feedback` (45k+ rows en instancias activas)
    para data que el caller no va a usar.
    """
    if not turn_ids:
        return {}
    from rag import _ragvec_state_conn, _silent_log  # noqa: PLC0415

    try:
        with _ragvec_state_conn() as conn:
            placeholders = ",".join("?" * len(turn_ids))
            cursor = conn.execute(
                "SELECT turn_id, rating FROM rag_feedback "
                f"WHERE turn_id IN ({placeholders}) "
                "ORDER BY ts ASC",
                tuple(turn_ids),
            )
            out: dict[str, int] = {}
            for tid, rating in cursor.fetchall():
                try:
                    r = int(rating)
                except (TypeError, ValueError):
                    continue
                if r == 0:
                    continue
                out[tid] = 1 if r > 0 else -1
            return out
    except Exception as exc:
        _silent_log("rag_log_feedback_sql_read", exc)
        return {}


def _record_draft_decision(
    *,
    draft_id: str,
    contact_jid: str,
    contact_name: str | None,
    original_msgs: list[dict],
    bot_draft: str,
    decision: str,
    sent_text: str | None = None,
    extra: dict | None = None,
) -> int | None:
    """Persiste una decisión de draft a `rag_draft_decisions`.

    Cierra el loop de auto-aprendizaje del bot WhatsApp: cuando el user
    puntúa un draft via `/si`, `/no`, o `/editar` en el RagNet group, el
    listener postea acá la decisión + lo que se mandó al contacto. Pares
    (bot_draft, sent_text) cuando `decision='approved_editar'` son gold
    humano para futuro fine-tune del modelo de drafts.

    Args:
        draft_id: short hex del listener TS, estable across updates del
            mismo draft (si el bot regenera el draft N veces antes de la
            decisión final, todas las rows comparten draft_id).
        contact_jid: jid de WhatsApp del contacto que escribió.
        contact_name: display name (opcional).
        original_msgs: list de `{id, text, ts}` con los mensajes que
            dispararon el draft.
        bot_draft: texto que el LLM generó.
        decision: uno de `approved_si`, `approved_editar`, `rejected`,
            `expired`. Cualquier otro valor → return None.
        sent_text: lo que finalmente se mandó al contacto. NULL si
            rejected/expired.
        extra: dict libre para metadatos future (ej. score del LLM,
            latency, model tag).

    Returns:
        rowid del INSERT, o None si la decisión es inválida o el write
        falló (silent-fail para no romper el listener si el DB está
        inaccesible — la UX del user nunca depende de telemetría).
    """
    if decision not in _VALID_DRAFT_DECISIONS:
        return None
    from rag import (  # noqa: PLC0415
        _ragvec_state_conn,
        _silent_log,
        _sql_append_event,
        _sql_write_with_retry,
    )

    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "draft_id": draft_id,
        "contact_jid": contact_jid,
        "contact_name": contact_name,
        "original_msgs": original_msgs or [],
        "bot_draft": bot_draft or "",
        "decision": decision,
        "sent_text": sent_text,
        "extra": extra,
    }
    # `_sql_write_with_retry` traga errors via _silent_log; usamos un
    # holder mutable para devolver el rowid al caller (las write_fn no
    # aceptan return value en su contract).
    holder: dict = {"id": None}

    def _do() -> None:
        with _ragvec_state_conn() as conn:
            holder["id"] = _sql_append_event(
                conn, "rag_draft_decisions", _map_draft_decision_row(row),
            )
    try:
        _sql_write_with_retry(_do, "draft_decision_sql_write_failed")
    except Exception as exc:  # pragma: no cover - retry helper ya silencia
        _silent_log("draft_decision_helper", exc)
        return None
    return holder["id"]


def _record_brief_feedback(
    *,
    dedup_key: str,
    rating: str,
    reason: str = "",
    source: str = "wa",
) -> int | None:
    """Persiste un feedback del user sobre un brief (morning/evening/digest)
    a `rag_brief_feedback`. Cierra el loop de los briefs:

    - El daemon de briefs escribe el archivo + lo pushea a WhatsApp
      con un footer `_brief:<vault_relpath>_` (ver
      `_brief_push_to_whatsapp`).
    - El listener TS detecta la reacción del user (👍/👎/🔇 o tokens
      "ok"/"no"/"basta") dentro de los 30min siguientes y postea acá
      via `POST /api/brief/feedback`.
    - Esta función inserta la row.

    Args:
        dedup_key: vault_relpath del brief al que apunta el feedback.
            Único por brief (cada brief lleva la fecha en el nombre).
        rating: uno de `positive | negative | mute`. Cualquier otro
            valor → return None.
        reason: texto libre opcional (ej. el body completo del reply
            para debug).
        source: origen del feedback (default 'wa' — WhatsApp). Forward-
            compatible con otros canales (PWA, CLI, etc.).

    Returns:
        rowid del INSERT, o None si la rating es inválida o el write
        falla (silent-fail — la UX del listener no debe romperse por
        un fallo de telemetría). Mismo contract que
        `_record_draft_decision` y `record_feedback` del anticipatory
        agent.
    """
    if not dedup_key:
        return None
    if rating not in _VALID_BRIEF_FEEDBACK_RATINGS:
        return None
    from rag import (  # noqa: PLC0415
        _ragvec_state_conn,
        _silent_log,
        _sql_append_event,
        _sql_write_with_retry,
    )

    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "dedup_key": dedup_key,
        "rating": rating,
        "reason": reason or "",
        "source": source or "wa",
    }
    holder: dict = {"id": None}

    def _do() -> None:
        with _ragvec_state_conn() as conn:
            holder["id"] = _sql_append_event(
                conn, "rag_brief_feedback", _map_brief_feedback_row(row),
            )
    try:
        _sql_write_with_retry(_do, "brief_feedback_sql_write_failed")
    except Exception as exc:  # pragma: no cover - retry helper ya silencia
        _silent_log("brief_feedback_helper", exc)
        return None
    return holder["id"]


def _log_daemon_run_event(
    label: str,
    action: str,
    *,
    prev_state: str | None = None,
    new_state: str | None = None,
    exit_code: int | None = None,
    reason: str | None = None,
) -> None:
    """Append a daemon-run event to rag_daemon_runs. Silent-fail on SQL error.

    Args:
        label:      Nombre del servicio launchd (ej.
                    "com.fer.obsidian-rag-web").
        action:     Tipo de acción; uno de "status_check", "bootstrap",
                    "bootout", "kickstart", "reconcile_dry_run",
                    "reconcile_apply", "retry".
        prev_state: Estado previo del daemon antes de la acción (ej.
                    "running", "not running", "missing", "error").
        new_state:  Estado posterior a la acción.
        exit_code:  Último exit code de launchctl (None si no aplica).
        reason:     Texto libre con el motivo de la acción (ej.
                    "last_exit=1 runs<3", "overdue 2x cadence").

    Contract: nunca raisea. Los errores SQL van a sql_state_errors.jsonl
    vía _log_sql_state_error (que ya llama a _bump_silent_log_counter
    internamente — no se duplica el call).
    """
    from rag import (  # noqa: PLC0415
        _ragvec_state_conn,
        _silent_log,
        _sql_append_event,
        _sql_write_with_retry,
    )

    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "action": action,
        "prev_state": prev_state,
        "new_state": new_state,
        "exit_code": exit_code,
        "reason": reason,
    }

    def _do() -> None:
        with _ragvec_state_conn() as conn:
            _sql_append_event(conn, "rag_daemon_runs", row)

    try:
        _sql_write_with_retry(_do, "daemon_runs_sql_write_failed")
    except Exception as exc:  # pragma: no cover - retry helper ya silencia
        _silent_log("daemon_runs_writer", exc)
