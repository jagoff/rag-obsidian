"""Signal — Preguntas abiertas en WhatsApp sin respuesta del user.

Detecta el patrón clásico de "me preguntaste algo, no te contesté, y pasaron
días". Fuente primaria: la SQL table `rag_wa_tasks` poblada por el comando
`rag wa-tasks` (que clasifica mensajes de los chats con kind=question /
commitment / fact / action). Este signal mira solo `kind='question'`.

## Heurística

1. SELECT de `rag_wa_tasks` donde `kind='question'` y `ts` en los últimos 14
   días — ventana más allá de la cual "contestar una pregunta vieja" ya no
   tiene sentido (el contexto se perdió, mejor reabrir el tema a mano).
2. Por cada pregunta, chequear si en el MISMO `source_chat` hay alguna fila
   posterior del `user='me'` dentro de los 3 días siguientes a la pregunta.
   Si la hay → el user ya respondió (aunque sea con un "ok", no juzgamos
   calidad). Si NO hay → la pregunta queda "awaiting".
3. Filter final: solo preguntas con ≥3 días de edad. Antes de 3 días es
   razonable no haber contestado (horario laboral, weekend, etc.).
4. Ordenar "awaiting" por edad descendente (la más vieja primero) y emitir
   MÁXIMO 2 candidates — el feed del anticipatory agent se satura rápido.

## Score

    score = min(1.0, days_since_asked / 14.0)

Calibración:
- 3 días (min filter): 0.21
- 7 días: 0.50
- 14+ días: 1.00

## dedup_key

    f"awaiting:{chat_id}:{question_ts_iso_date}"

Estable cross-runs mientras la misma pregunta siga en la tabla. El isodate
de `ts` (no el timestamp completo) hace que re-ingesta del mismo día no
genere un key nuevo.

## Silent-fail

Cualquier excepción (tabla no existe, schema distinto al esperado, DB
locked, SQL malformado) → `[]`. El contrato del framework es "nunca tirar
desde una signal"; el orchestrator tiene un outer try/except pero este
doble cinturón es lo que se espera. Los `_silent_log` aseguran que si algo
falla silenciosamente quede un rastro en `silent_errors.jsonl`.

## Nota sobre el schema de producción

La tabla `rag_wa_tasks` en producción hoy (schema ~2026-04-21) registra
INVOCACIONES del CLI `rag wa-tasks` (columnas: id, ts, since, chats, items,
path, extra_json) — NO filas por-pregunta. Este signal asume que el schema
evolucionó / se extendió para incluir columnas `kind`, `source_chat`,
`message_preview`, `user`. Si esas columnas no existen, el `SELECT`
fallará con OperationalError → silent-fail → `[]`. Si/cuando el schema se
extienda el signal empieza a emitir solo. Los tests crean la tabla con las
columnas extendidas para ejercitar la lógica happy-path.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from rag_anticipate.signals.base import register_signal


# Ventana máxima de edad de la pregunta (días). Más allá el contexto está
# perdido y re-abrirlo automáticamente es contraproducente.
_QA_LOOKBACK_DAYS = 14

# Edad mínima antes de considerar "awaiting". <3d puede ser simplemente
# horario laboral, weekend, o tiempo razonable de respuesta.
_QA_MIN_AGE_DAYS = 3

# Ventana de "respuesta posterior" — si el user mandó algo en este chat
# dentro de los 3 días siguientes a la pregunta, contamos como respondido.
_QA_REPLY_WINDOW_DAYS = 3

# Máximo de candidates emitidos por pasada (los N más viejos).
_QA_MAX_CANDIDATES = 2


def _parse_ts(raw) -> datetime | None:
    """Normaliza un `ts` de la tabla a `datetime` naive local.

    Acepta ISO 8601 (con o sin timezone, con o sin microsegundos). Devuelve
    `None` si no parsea — el caller descarta esas filas.
    """
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=None) if raw.tzinfo else raw
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    # `Z` sufijo no lo acepta fromisoformat < 3.11 en algunos edge cases.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Fallback: intentar solo la parte de fecha.
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None
    # Convertir a naive (dropear tz) para comparar con `now` del signal,
    # que también es naive (viene de `datetime.now()`).
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


@register_signal(name="question_awaiting", snooze_hours=168)
def question_awaiting_signal(now: datetime) -> list:
    """Emite hasta 2 candidates por preguntas WA ≥3d sin respuesta del user.

    Silent-fail total: tabla inexistente, schema roto, DB locked, lo que
    sea → `[]`. Ver docstring del módulo para detalle del algoritmo.
    """
    from rag import AnticipatoryCandidate, _ragvec_state_conn, _silent_log

    try:
        lookback_cutoff = (now - timedelta(days=_QA_LOOKBACK_DAYS)).isoformat(
            timespec="seconds"
        )

        try:
            with _ragvec_state_conn() as conn:
                # Preguntas en la ventana de 14 días. Si la tabla no existe o
                # el schema no tiene estas columnas, esto tira OperationalError
                # y lo capturamos abajo.
                rows = conn.execute(
                    "SELECT ts, source_chat, message_preview "
                    "FROM rag_wa_tasks "
                    "WHERE kind = 'question' AND ts >= ? "
                    "ORDER BY ts ASC",
                    (lookback_cutoff,),
                ).fetchall()
        except Exception as exc:
            # Tabla inexistente / schema sin las columnas / DB locked —
            # silent-fail al contrato.
            _silent_log("question_awaiting.select_questions", exc)
            return []

        if not rows:
            return []

        # Para cada pregunta, chequear respuesta posterior del user en el
        # mismo chat dentro de la ventana de reply.
        awaiting: list[tuple[int, str, str, datetime]] = []
        # tuple: (days_since, chat_id, preview, question_dt)

        for row in rows:
            try:
                ts_raw, source_chat, preview = row[0], row[1], row[2]
            except (IndexError, TypeError):
                continue

            q_dt = _parse_ts(ts_raw)
            if q_dt is None:
                continue
            if not source_chat:
                continue

            days_since = (now - q_dt).total_seconds() / 86400.0
            if days_since < _QA_MIN_AGE_DAYS:
                continue

            reply_cutoff_start = q_dt.isoformat(timespec="seconds")
            reply_cutoff_end = (
                q_dt + timedelta(days=_QA_REPLY_WINDOW_DAYS)
            ).isoformat(timespec="seconds")

            try:
                with _ragvec_state_conn() as conn:
                    reply_row = conn.execute(
                        "SELECT 1 FROM rag_wa_tasks "
                        "WHERE source_chat = ? "
                        "AND user = 'me' "
                        "AND ts > ? AND ts <= ? "
                        "LIMIT 1",
                        (source_chat, reply_cutoff_start, reply_cutoff_end),
                    ).fetchone()
            except Exception as exc:
                _silent_log("question_awaiting.check_reply", exc)
                # Si no podemos chequear reply, conservador: asumir que
                # SÍ hubo respuesta (no spammear al user con falso awaiting).
                continue

            if reply_row is not None:
                continue  # ya respondió

            awaiting.append((
                int(days_since),
                str(source_chat),
                str(preview or ""),
                q_dt,
            ))

        if not awaiting:
            return []

        # Ordenar por edad descendente — las más viejas primero. Desempate
        # estable por chat_id para orden determinista en tests.
        awaiting.sort(key=lambda t: (-t[0], t[1]))

        candidates = []
        for days_since, chat_id, preview, q_dt in awaiting[:_QA_MAX_CANDIDATES]:
            score = min(1.0, days_since / 14.0)
            score = round(score, 4)

            # Preview truncado para no estirar el body de WA. El CLI
            # `wa-tasks` ya deja previews cortos pero defendemos igual.
            preview_short = preview[:120] if preview else "(sin preview)"

            message = (
                f"💬 {preview_short} — pregunta sin respuesta hace "
                f"{days_since} días en WhatsApp. ¿Responder?"
            )

            dedup_key = f"awaiting:{chat_id}:{q_dt.date().isoformat()}"

            reason = (
                f"chat={chat_id} days_since={days_since} "
                f"ts={q_dt.isoformat(timespec='seconds')}"
            )

            candidates.append(AnticipatoryCandidate(
                kind="anticipate-question_awaiting",
                score=score,
                message=message,
                dedup_key=dedup_key,
                snooze_hours=168,
                reason=reason,
            ))

        return candidates
    except Exception:
        # Outer belt-and-suspenders. El contrato es "nunca tirar".
        return []
