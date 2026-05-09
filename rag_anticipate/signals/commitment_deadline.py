"""Signal — Commitment deadline.

Detecta promesas/compromisos con due_date inminente (overdue, hoy, próximas
72h) leyendo `rag_promises` SQL. A diferencia del signal `deadline` (que
mira frontmatter `due:` + Apple Reminders genéricos), este signal cubre
la dimensión SOCIAL: "le prometiste a Juan que X" / "Juan te prometió
que Y".

Usa la tabla scaffolded `rag_promises` (DDL existente en `rag/__init__.py`
desde 2026-04-30, sin populator activo en master todavía). Cuando el
extractor de promesas llegue a producción, esta signal se activa
automáticamente sin más cambios. Mientras tanto: silent-fail, []. Esto
es deliberado — preferimos signal stub listo para activar a tener que
codear bajo presión cuando el feature aterriza.

Diseño:

- Lee `rag_promises` con `status='pending'` AND `due_ts` parseable
  AND `due_ts` en [now-7d, now+3d] (overdue 1 semana hasta +3 días).
- Direction-aware: el mensaje cambia según `direction`:
    - `outgoing` (user prometió) → "Le prometiste a X: '...' — vence hoy"
    - `incoming` (otro le prometió al user) → "X te prometió: '...' — vence hoy"
- Score escala con proximidad: overdue=1.0, hoy=1.0, +1d=0.75, +2d=0.5, +3d=0.25.
- Max 2 candidates por run (overdue + closest upcoming).
- Snooze 24h: re-emit cada día hasta que el user marque la promesa
  closed (o el due_ts pasa la ventana).
- Dedup_key con `promise_id + due_iso`: si una promesa se reagenda
  (`due_ts` editado), genera nuevo dedup_key → re-pusheable.

Silent-fail total: tabla vacía / DDL no aplicada / SQL error → [].
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime

from rag_anticipate.signals.base import register_signal


# Ventana de búsqueda. Aceptamos overdue hasta -7d (más viejo es ruido —
# user ya descartó mentalmente) y +3d hacia adelante.
_OVERDUE_MIN_DAYS = -7
_UPCOMING_MAX_DAYS = 3

# Máximo candidates por run.
_MAX_EMIT = 2


def _parse_due_ts(raw: str | None) -> date | None:
    """Normaliza `due_ts` (TEXT en SQL) a `date`. Acepta ISO datetime
    completo o solo fecha. None / vacío / malformado → None."""
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        # ISO datetime con o sin offset.
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _score_for_proximity(days_until: int) -> float:
    """Score [0.25, 1.0] basado en días hasta el due. Overdue clamp a 1.0."""
    if days_until <= 0:
        return 1.0  # overdue o hoy
    if days_until == 1:
        return 0.75
    if days_until == 2:
        return 0.5
    return 0.25  # 3 días


def _format_message(
    promise_text: str,
    contact_name: str,
    direction: str,
    days_until: int,
) -> str:
    """Genera el mensaje WhatsApp-friendly."""
    if days_until < 0:
        when = f"overdue ({-days_until}d)"
    elif days_until == 0:
        when = "vence hoy"
    elif days_until == 1:
        when = "vence mañana"
    else:
        when = f"vence en {days_until}d"

    name = (contact_name or "alguien").strip() or "alguien"
    txt = (promise_text or "").strip()
    if len(txt) > 120:
        txt = txt[:117] + "…"

    if direction == "outgoing":
        emoji = "🤝"
        verb = f"Le prometiste a *{name}*"
    else:  # incoming
        emoji = "📌"
        verb = f"*{name}* te prometió"

    return (
        f"{emoji} {verb} ({when}):\n"
        f"  '{txt}'\n"
        f"  ¿La cerrás, la movés, o la cumpliste?"
    )


def _fetch_pending_promises(now: datetime) -> list[dict]:
    """Lee `rag_promises` filtrando por status pending + due_ts en ventana.

    Silent-fail: tabla no existe / SQL error / DB lockada → [].
    """
    try:
        from rag import _ragvec_state_conn  # noqa: PLC0415
    except Exception:
        return []
    try:
        with _ragvec_state_conn() as conn:
            try:
                rows = conn.execute(
                    "SELECT id, contact_jid, contact_name, promise_text, "
                    "direction, due_ts "
                    "FROM rag_promises "
                    "WHERE status = 'pending' "
                    "  AND due_ts IS NOT NULL "
                    "  AND due_ts != '' "
                    "ORDER BY due_ts ASC"
                ).fetchall()
            except sqlite3.OperationalError:
                return []
    except Exception:
        return []

    out = []
    for row in rows or ():
        try:
            out.append({
                "id": int(row[0]),
                "contact_jid": row[1] or "",
                "contact_name": row[2] or "",
                "promise_text": row[3] or "",
                "direction": (row[4] or "").strip().lower(),
                "due_ts": row[5] or "",
            })
        except Exception:
            continue
    return out


@register_signal(name="commitment_deadline", snooze_hours=24)
def commitment_deadline_signal(now: datetime) -> list:
    """Emite hasta 2 candidates — promesa overdue más vieja + closest upcoming.

    Pasos:
    1. SELECT pendientes con due_ts.
    2. Filtrar por ventana [now-7d, now+3d].
    3. Sort: overdue first (más viejo arriba), después upcoming asc.
    4. Tomar top 2 + emitir.

    Silent-fail si:
      - rag_promises tabla vacía (feature scaffold sin populator).
      - DDL no aplicada (instalación nueva sin telemetry tables).
      - SQL error / DB lock.
    """
    try:
        from rag import AnticipatoryCandidate  # noqa: PLC0415

        promises = _fetch_pending_promises(now)
        if not promises:
            return []

        today = now.date() if isinstance(now, datetime) else now

        # (days_until, source_id, dict)
        in_window: list[tuple[int, int, dict]] = []
        for p in promises:
            due = _parse_due_ts(p["due_ts"])
            if due is None:
                continue
            days_until = (due - today).days
            if days_until < _OVERDUE_MIN_DAYS:
                continue
            if days_until > _UPCOMING_MAX_DAYS:
                continue
            # promise_text vacío = scaffold row sin contenido — skip.
            if not (p["promise_text"] or "").strip():
                continue
            # direction inválido = skip (defensivo, debería ser
            # outgoing/incoming).
            if p["direction"] not in ("outgoing", "incoming"):
                continue
            in_window.append((days_until, p["id"], p))

        if not in_window:
            return []

        # Sort: overdue first (más viejo / días_until más negativo arriba),
        # después por proximidad asc. Tiebreak por id determinístico.
        in_window.sort(key=lambda t: (t[0], t[1]))

        candidates = []
        for days_until, pid, p in in_window[:_MAX_EMIT]:
            score = _score_for_proximity(days_until)
            message = _format_message(
                promise_text=p["promise_text"],
                contact_name=p["contact_name"],
                direction=p["direction"],
                days_until=days_until,
            )
            due_iso = (_parse_due_ts(p["due_ts"]) or today).isoformat()
            dedup_key = f"commitment:{pid}:{due_iso}"
            reason = (
                f"id={pid} direction={p['direction']} "
                f"days_until={days_until} due={due_iso}"
            )
            candidates.append(AnticipatoryCandidate(
                kind="anticipate-commitment_deadline",
                score=score,
                message=message,
                dedup_key=dedup_key,
                snooze_hours=24,
                reason=reason,
            ))

        return candidates
    except Exception:
        return []
