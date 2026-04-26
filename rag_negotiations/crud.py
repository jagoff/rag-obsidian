"""CRUD layer del módulo wa-negotiation-autopilot — Fase 0 (data only).

Helpers thin sobre las 5 tablas que la Fase 0 introduce:

- `rag_negotiations`              — 1 fila por negociación.
- `rag_negotiation_turns`         — 1 fila por mensaje (in/out).
- `rag_negotiation_pending_sends` — cola de envíos pendientes
                                   (pause simulator se enchufa acá en F3).
- `rag_style_fingerprints`        — 1 fila por contacto, refrescada por
                                   el watcher de Fase 1.
- `rag_behavior_priors_wa`        — 1 fila por contacto con priors de
                                   comportamiento (response lag, etc.).

Reglas de diseño Fase 0:

1. **No tocamos la state machine acá** — los helpers `update_status()`
   aceptan cualquier string. La validación de transiciones legales
   vive en `rag_negotiations.state_machine.transition()`. El
   orchestrator (Fase 3) primero valida, luego persiste.
2. **`extra_json` es para futuro** — los DDLs no tienen ese campo en
   `rag_negotiation_turns` / `rag_negotiation_pending_sends` pero todas
   las tablas similares del repo lo usan. Lo agregamos en F3 cuando
   surja necesidad de campos no previstos por el spec.
3. **Retornos seguros** — funciones que crean filas devuelven el id;
   las que update/upsert devuelven bool. Errores se loggean vía
   `rag._silent_log()` y la función devuelve `None` / `False`. El
   patrón es el mismo que `rag_routing_learning.promote.upsert_rule()`.
4. **Conexión vía `rag._ragvec_state_conn()`** — sigue el contract
   de los otros subpackages. Tests mockean ese context manager con
   un `tmp_path / "telemetry.db"` (ver `tests/test_rag_negotiations_*.py`).

Fases siguientes que se enchufan acá:

- **F1 (real-time learning)**: el watcher de fsevents llama
  `upsert_style_fingerprint()` + `upsert_behavior_priors()` cuando
  detecta cambios en `messages.db` del bridge.
- **F3 (orchestrator daemon)**: lee `rag_negotiation_pending_sends`
  con `dequeue_due()` cada N segundos, dispara los sends, y marca
  status según el resultado.
- **F5 (dashboard panel)**: usa `list_negotiations()` con filtros
  para renderizar las 3 secciones (activas / recientes / stats).
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any


# ── Helpers internos ─────────────────────────────────────────────────────────

def _silent_log(where: str, exc: Exception) -> None:
    """Wrapper que delega al `_silent_log` global del paquete `rag`.

    Import lazy para que tests que NO importan `rag` (ej. unit tests
    de `state_machine.py` puro) no carguen el monstruo de 51k líneas.
    """
    try:
        import rag  # type: ignore[import-not-found]
        rag._silent_log(where, exc)
    except Exception:
        # Último resorte: si rag no carga (test entorno raro), tragamos.
        # Sigue siendo idempotente — el caller no se rompe.
        pass


def _conn():
    """Context manager que devuelve una conexión a `ragvec.db`.

    Wrapper sobre `rag._ragvec_state_conn()` con import lazy para
    consistency con el resto de los subpackages.
    """
    import rag  # type: ignore[import-not-found]
    return rag._ragvec_state_conn()


def _now_iso() -> str:
    """ISO timestamp UTC con sufijo Z. Mismo formato que usan otras
    tablas del repo (ver `_now_iso` privados en `rag/__init__.py`)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row_to_dict(cur: sqlite3.Cursor, row: tuple | None) -> dict[str, Any] | None:
    """Convierte una row tuple a dict usando `cur.description` para los
    nombres de columna. Devuelve `None` si `row` es `None`. Útil para
    devolver shapes user-friendly desde los `get_*` helpers."""
    if row is None:
        return None
    return {col[0]: val for col, val in zip(cur.description, row)}


# ── rag_negotiations ─────────────────────────────────────────────────────────

def create_negotiation(
    *,
    user_intent: str,
    target_jid: str,
    target_name: str | None = None,
    perimeter: dict[str, Any] | None = None,
    confidence_threshold: float = 0.85,
    max_messages: int | None = None,
    style_seed_jid: str | None = None,
    trace_id: str | None = None,
) -> int | None:
    """Crea una negociación nueva en estado `draft`.

    El primer mensaje TODAVÍA NO se mandó — la PWA va a renderizar la
    pre-launch screen con el draft, y al confirmar el user la
    transition es `draft → launched` (vía `update_status`).

    Args:
        user_intent: prompt original del user al sistema. Persistido
            literal para audit ("¿qué le dije al bot que haga?").
        target_jid: JID del contacto en WA (ej. `5491112345678@s.whatsapp.net`).
        target_name: nombre humano leíble si está disponible. NO es
            usado por el classifier — solo para UI.
        perimeter: dict serializable con el scope de la negociación
            (ej. `{"topic": "scheduling", "items": ["café", "lugar"],
            "max_offers": 3}`). Se serializa a JSON; el classifier lo
            lee en F2 para detectar `perimeter_violation`.
        confidence_threshold: floor que el classifier va a respetar.
            Default 0.85; subir per-contacto sensible (jefe → 0.95).
        max_messages: hard cap de turnos antes de timeout. None = sin cap.
        style_seed_jid: si querés clonar el estilo desde otro JID
            (ej. mismo contacto en otro chat), pasalo. Default = None
            usa `target_jid`.
        trace_id: 8-char hex generado por `rag.generate_trace_id()`.
            Linkea esta negociación con queries / behavior / silent_errors.

    Returns:
        id de la fila insertada, o None si el INSERT falló.
    """
    perimeter_json = json.dumps(perimeter or {}, ensure_ascii=False)
    now = _now_iso()
    try:
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO rag_negotiations ("
                " trace_id, user_intent, target_jid, target_name,"
                " status, created_at, updated_at, perimeter_json,"
                " confidence_threshold, max_messages, style_seed_jid"
                ") VALUES (?, ?, ?, ?, 'draft', ?, ?, ?, ?, ?, ?)",
                (
                    trace_id, user_intent, target_jid, target_name,
                    now, now, perimeter_json,
                    confidence_threshold, max_messages,
                    style_seed_jid or target_jid,
                ),
            )
            return int(cur.lastrowid) if cur.lastrowid else None
    except Exception as exc:
        _silent_log("negotiations_create", exc)
        return None


def get_negotiation(neg_id: int) -> dict[str, Any] | None:
    """Devuelve la fila como dict o None si no existe.

    Incluye `perimeter` deserializado a dict (el JSON crudo no es
    útil para callers — todos lo van a parsear igual).
    """
    try:
        with _conn() as conn:
            cur = conn.execute(
                "SELECT * FROM rag_negotiations WHERE id = ?", (neg_id,)
            )
            row = cur.fetchone()
            d = _row_to_dict(cur, row)
            if d and d.get("perimeter_json"):
                try:
                    d["perimeter"] = json.loads(d["perimeter_json"])
                except json.JSONDecodeError:
                    d["perimeter"] = {}
            return d
    except Exception as exc:
        _silent_log("negotiations_get", exc)
        return None


def update_status(
    neg_id: int,
    new_status: str,
    *,
    closure_type: str | None = None,
    closure_summary: str | None = None,
    side_effect: dict[str, Any] | None = None,
) -> bool:
    """Actualiza el status. NO valida transition legal — para eso usar
    `state_machine.transition()` antes y pasar el `to_state` resultante.

    Si `new_status` es uno de los estados terminales y no se le pasa
    `closure_type`, asumimos `closure_type = new_status` (heurística
    razonable: `closed_ok` → closure_type=`closed_ok`).

    Args:
        neg_id: id de la negociación.
        new_status: estado destino. NO se valida acá.
        closure_type: tipo de cierre (agreement / rejection / timeout
            / user_cancel / interference). Solo guardamos si es
            terminal.
        closure_summary: resumen humano-leíble del cierre. Para audit.
        side_effect: dict serializable con el side effect que se
            disparó (calendar event id, reminder id, etc.). UI muestra
            como link al evento.

    Returns:
        True si la fila se actualizó, False si falló o no existe.
    """
    from rag_negotiations.state_machine import TERMINAL_STATES

    is_terminal = new_status in TERMINAL_STATES
    closed_at = _now_iso() if is_terminal else None
    if is_terminal and closure_type is None:
        closure_type = new_status

    try:
        with _conn() as conn:
            cur = conn.execute(
                "UPDATE rag_negotiations SET"
                " status = ?,"
                " updated_at = ?,"
                " closed_at = COALESCE(?, closed_at),"
                " closure_type = COALESCE(?, closure_type),"
                " closure_summary = COALESCE(?, closure_summary),"
                " side_effect_json = COALESCE(?, side_effect_json)"
                " WHERE id = ?",
                (
                    new_status,
                    _now_iso(),
                    closed_at,
                    closure_type,
                    closure_summary,
                    json.dumps(side_effect, ensure_ascii=False) if side_effect else None,
                    neg_id,
                ),
            )
            return cur.rowcount > 0
    except Exception as exc:
        _silent_log("negotiations_update_status", exc)
        return False


def increment_message_count(neg_id: int, *, sent: bool) -> bool:
    """Incrementa `messages_sent` o `messages_received` en 1.

    Args:
        neg_id: id de la negociación.
        sent: True para `messages_sent`, False para `messages_received`.

    Returns:
        True si la fila se actualizó.
    """
    column = "messages_sent" if sent else "messages_received"
    try:
        with _conn() as conn:
            cur = conn.execute(
                f"UPDATE rag_negotiations SET"
                f" {column} = {column} + 1,"
                f" updated_at = ?"
                f" WHERE id = ?",
                (_now_iso(), neg_id),
            )
            return cur.rowcount > 0
    except Exception as exc:
        _silent_log("negotiations_increment", exc)
        return False


def list_negotiations(
    *,
    status: str | tuple[str, ...] | None = None,
    target_jid: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Lista negociaciones, ordenadas por `updated_at` DESC.

    Args:
        status: filtro por status. Single string ("in_flight") o tupla
            ("in_flight", "escalated"). None = sin filtro.
        target_jid: filtro por contacto. None = todos.
        limit: max filas a devolver. Default 100.
    """
    where_clauses: list[str] = []
    params: list[Any] = []
    if status is not None:
        if isinstance(status, str):
            where_clauses.append("status = ?")
            params.append(status)
        else:
            placeholders = ",".join("?" for _ in status)
            where_clauses.append(f"status IN ({placeholders})")
            params.extend(status)
    if target_jid is not None:
        where_clauses.append("target_jid = ?")
        params.append(target_jid)
    where = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    params.append(int(limit))
    try:
        with _conn() as conn:
            cur = conn.execute(
                f"SELECT * FROM rag_negotiations{where}"
                " ORDER BY updated_at DESC LIMIT ?",
                params,
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        _silent_log("negotiations_list", exc)
        return []


# ── rag_negotiation_turns ────────────────────────────────────────────────────

def append_turn(
    *,
    negotiation_id: int,
    direction: str,
    content: str,
    classifier_confidence: float | None = None,
    classifier_reasoning: str | None = None,
    pause_simulated_ms: int | None = None,
    bridge_message_id: str | None = None,
    escalated: bool = False,
    user_response_text: str | None = None,
    user_overrode: bool | None = None,
) -> int | None:
    """Agrega un turno a la conversación.

    Args:
        negotiation_id: FK a `rag_negotiations.id`. Si la negociación
            no existe, el insert falla por FK constraint (`ON DELETE
            CASCADE` cubre el caso opuesto).
        direction: `"in"` o `"out"`. `"in"` = la otra parte; `"out"` =
            mandado por el bot (o por el user si fue user_resumes).
        content: el texto del mensaje. Persistir literal — el classifier
            lo va a procesar.
        classifier_confidence: 0.0-1.0 si direction=="in" (clasificamos
            el inbound). None para outbound (no clasificamos lo nuestro).
        classifier_reasoning: razonamiento del LLM (1-2 frases). Audit only.
        pause_simulated_ms: cuánto esperamos antes de mandar (solo
            outbound). None para inbound.
        bridge_message_id: id que devolvió el WA bridge tras el send.
            Permite cruzar contra el bridge SQLite si hace falta.
        escalated: True si este turno disparó un escalation (solo inbound).
        user_response_text: respuesta que el user dio al escalation
            (cuando aplica). Permite ver "qué dijo el user vs qué
            sugería el bot" en audit.
        user_overrode: True si el user respondió DISTINTO a lo que el
            bot sugería (señal para fine-tuning).

    Returns:
        id del turno insertado, o None si falló.
    """
    if direction not in {"in", "out"}:
        _silent_log(
            "negotiations_append_turn",
            ValueError(f"direction must be 'in' or 'out', got {direction!r}"),
        )
        return None
    now = _now_iso()
    escalated_at = now if escalated else None
    user_response_at = now if user_response_text else None
    try:
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO rag_negotiation_turns ("
                " negotiation_id, ts, direction, content,"
                " classifier_confidence, classifier_reasoning,"
                " pause_simulated_ms, bridge_message_id,"
                " escalated_at, user_response_text, user_response_at,"
                " user_overrode"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    negotiation_id, now, direction, content,
                    classifier_confidence, classifier_reasoning,
                    pause_simulated_ms, bridge_message_id,
                    escalated_at, user_response_text, user_response_at,
                    int(user_overrode) if user_overrode is not None else None,
                ),
            )
            return int(cur.lastrowid) if cur.lastrowid else None
    except Exception as exc:
        _silent_log("negotiations_append_turn", exc)
        return None


def list_turns(negotiation_id: int) -> list[dict[str, Any]]:
    """Devuelve todos los turnos de una negociación, ordenados por ts ASC."""
    try:
        with _conn() as conn:
            cur = conn.execute(
                "SELECT * FROM rag_negotiation_turns"
                " WHERE negotiation_id = ?"
                " ORDER BY ts ASC, id ASC",
                (negotiation_id,),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        _silent_log("negotiations_list_turns", exc)
        return []


# ── rag_negotiation_pending_sends ────────────────────────────────────────────

def enqueue_send(
    *,
    negotiation_id: int,
    content: str,
    typing_simulation_ms: int | None = None,
    send_after_ts: float | None = None,
) -> int | None:
    """Encola un send para que el orchestrator lo dispare cuando
    corresponda.

    Args:
        negotiation_id: FK.
        content: texto a mandar.
        typing_simulation_ms: cuánto debería el orchestrator emitir
            "typing..." antes del send. None = no emitir.
        send_after_ts: epoch-seconds en el que se debe mandar. None =
            mandar ASAP (el orchestrator usa `time.time()` como floor).

    Returns:
        id de la fila encolada, o None si falló.
    """
    if send_after_ts is None:
        send_after_ts = time.time()
    try:
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO rag_negotiation_pending_sends ("
                " negotiation_id, content, typing_simulation_ms,"
                " send_after_ts, queued_at, status"
                ") VALUES (?, ?, ?, ?, ?, 'pending')",
                (
                    negotiation_id, content, typing_simulation_ms,
                    float(send_after_ts), _now_iso(),
                ),
            )
            return int(cur.lastrowid) if cur.lastrowid else None
    except Exception as exc:
        _silent_log("negotiations_enqueue_send", exc)
        return None


def dequeue_due(now_ts: float | None = None, *, limit: int = 10) -> list[dict[str, Any]]:
    """Devuelve los sends listos para disparar (`status='pending'` y
    `send_after_ts <= now_ts`).

    No actualiza el status — el orchestrator es responsable de llamar
    `mark_send()` después de intentar el send (con success / fail).

    Args:
        now_ts: epoch-seconds. None = usa `time.time()`.
        limit: max filas a devolver.
    """
    if now_ts is None:
        now_ts = time.time()
    try:
        with _conn() as conn:
            cur = conn.execute(
                "SELECT * FROM rag_negotiation_pending_sends"
                " WHERE status = 'pending' AND send_after_ts <= ?"
                " ORDER BY send_after_ts ASC LIMIT ?",
                (float(now_ts), int(limit)),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        _silent_log("negotiations_dequeue_due", exc)
        return []


def mark_send(send_id: int, *, status: str, increment_attempts: bool = True) -> bool:
    """Marca el resultado de un send.

    Args:
        send_id: id de la fila en `rag_negotiation_pending_sends`.
        status: `'sent'`, `'failed'`, o `'cancelled'`. Cualquier otro
            es aceptado por el DDL pero no es estándar — el orchestrator
            debe quedarse en estos 3.
        increment_attempts: si True (default), suma 1 al contador.
    """
    try:
        with _conn() as conn:
            if increment_attempts:
                cur = conn.execute(
                    "UPDATE rag_negotiation_pending_sends SET"
                    " status = ?,"
                    " attempts = attempts + 1,"
                    " last_attempt_ts = ?"
                    " WHERE id = ?",
                    (status, _now_iso(), send_id),
                )
            else:
                cur = conn.execute(
                    "UPDATE rag_negotiation_pending_sends SET"
                    " status = ?,"
                    " last_attempt_ts = ?"
                    " WHERE id = ?",
                    (status, _now_iso(), send_id),
                )
            return cur.rowcount > 0
    except Exception as exc:
        _silent_log("negotiations_mark_send", exc)
        return False


# ── rag_style_fingerprints ───────────────────────────────────────────────────

def upsert_style_fingerprint(
    *,
    target_jid: str,
    fingerprint: dict[str, Any],
    messages_analyzed: int,
) -> bool:
    """REPLACE el fingerprint estilístico de un contacto.

    Llamado por el watcher de Fase 1 cuando detecta cambios en el
    bridge SQLite. El fingerprint es un dict con shape libre que el
    classifier (Fase 2) sabe interpretar — Fase 0 lo trata como blob
    opaco.

    Args:
        target_jid: PK natural — un JID, un fingerprint.
        fingerprint: dict serializable con tone / vocabulary /
            structural / temporal features.
        messages_analyzed: cuántos mensajes outbound del user al
            target se analizaron para construir este fingerprint. UI
            puede mostrarlo como confianza ("basado en N msgs").

    Returns:
        True si el upsert tuvo éxito.
    """
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO rag_style_fingerprints ("
                " target_jid, fingerprint_json, messages_analyzed, computed_at"
                ") VALUES (?, ?, ?, ?)"
                " ON CONFLICT(target_jid) DO UPDATE SET"
                " fingerprint_json = excluded.fingerprint_json,"
                " messages_analyzed = excluded.messages_analyzed,"
                " computed_at = excluded.computed_at",
                (
                    target_jid,
                    json.dumps(fingerprint, ensure_ascii=False),
                    int(messages_analyzed),
                    _now_iso(),
                ),
            )
            return True
    except Exception as exc:
        _silent_log("negotiations_upsert_fingerprint", exc)
        return False


def get_style_fingerprint(target_jid: str) -> dict[str, Any] | None:
    """Devuelve `{fingerprint: dict, messages_analyzed: int, computed_at: str}`
    o None si no existe."""
    try:
        with _conn() as conn:
            cur = conn.execute(
                "SELECT fingerprint_json, messages_analyzed, computed_at"
                " FROM rag_style_fingerprints WHERE target_jid = ?",
                (target_jid,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            try:
                fingerprint = json.loads(row[0]) if row[0] else {}
            except json.JSONDecodeError:
                fingerprint = {}
            return {
                "fingerprint": fingerprint,
                "messages_analyzed": int(row[1] or 0),
                "computed_at": row[2],
            }
    except Exception as exc:
        _silent_log("negotiations_get_fingerprint", exc)
        return None


# ── rag_behavior_priors_wa ───────────────────────────────────────────────────

def upsert_behavior_priors(
    *,
    target_jid: str,
    response_lag_mu: float | None = None,
    response_lag_sigma: float | None = None,
    avg_msg_length_words: float | None = None,
    msg_per_response: float | None = None,
    emoji_freq: float | None = None,
    samples_n: int = 0,
) -> bool:
    """REPLACE los priors de comportamiento de un contacto.

    Los priors alimentan al pause simulator (Fase 3). Específicamente
    `response_lag_mu` + `response_lag_sigma` son los parámetros del
    lognormal del histograma de delay entre mensajes recibidos del
    target_jid.

    Argumentos en None se persisten como NULL en la DB — el simulator
    falla open a defaults globales si una columna está NULL.

    Returns:
        True si el upsert tuvo éxito.
    """
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO rag_behavior_priors_wa ("
                " target_jid, response_lag_mu, response_lag_sigma,"
                " avg_msg_length_words, msg_per_response, emoji_freq,"
                " samples_n, computed_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(target_jid) DO UPDATE SET"
                " response_lag_mu = excluded.response_lag_mu,"
                " response_lag_sigma = excluded.response_lag_sigma,"
                " avg_msg_length_words = excluded.avg_msg_length_words,"
                " msg_per_response = excluded.msg_per_response,"
                " emoji_freq = excluded.emoji_freq,"
                " samples_n = excluded.samples_n,"
                " computed_at = excluded.computed_at",
                (
                    target_jid, response_lag_mu, response_lag_sigma,
                    avg_msg_length_words, msg_per_response, emoji_freq,
                    int(samples_n), _now_iso(),
                ),
            )
            return True
    except Exception as exc:
        _silent_log("negotiations_upsert_priors", exc)
        return False


def get_behavior_priors(target_jid: str) -> dict[str, Any] | None:
    """Devuelve la fila como dict o None si no existe."""
    try:
        with _conn() as conn:
            cur = conn.execute(
                "SELECT * FROM rag_behavior_priors_wa WHERE target_jid = ?",
                (target_jid,),
            )
            row = cur.fetchone()
            return _row_to_dict(cur, row)
    except Exception as exc:
        _silent_log("negotiations_get_priors", exc)
        return None


__all__ = [
    "append_turn",
    "create_negotiation",
    "dequeue_due",
    "enqueue_send",
    "get_behavior_priors",
    "get_negotiation",
    "get_style_fingerprint",
    "increment_message_count",
    "list_negotiations",
    "list_turns",
    "mark_send",
    "update_status",
    "upsert_behavior_priors",
    "upsert_style_fingerprint",
]
