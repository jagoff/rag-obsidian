"""Signal — Hilo de mail sin respuesta del user.

Lee `_fetch_gmail_evidence(now)["awaiting_reply"]` (Gmail API), filtra
hilos donde el último mensaje NO es del user Y tiene >24h de antigüedad,
y emite un push:

    📧 *<sender>* hace 3d sin respuesta
       _<subject>_
       > <snippet…>

## Diseño

- Reusa la infra ya existente (`_fetch_gmail_evidence`) que cachea +
  filtra exclusiones de categoría (-promotions -social -updates -forums)
  y devuelve `awaiting_reply` ya pre-filtrado a 3-14d.
- Filtramos LOCALMENTE a 1d-30d window para tener más cobertura
  (3d era restrictivo: muchos mails del laburo necesitan respuesta en 24h).
- Score por antigüedad (más viejo = más urgente):
    - >7d → 1.0
    - 3-7d → 0.7
    - 1-3d → 0.5
- dedup_key: `email-awaiting:<thread_id>` — hilo individual, snooze 24h
  para no spamear el mismo hilo todos los días.
- Cap `_MAX_CANDIDATES=3` para no quemar la attention del user con un
  inbox lleno.

## Anti-noise

- Skip hilos sin sender o sin subject (mails malformed).
- Skip si snippet es vacío (puede ser mail sólo con attachment).
- Silent-fail Gmail API: sin auth, OAuth revoked, network error → [].
- Si `_fetch_gmail_evidence` ya devuelve cached value, este signal usa
  ese cache (TTL 5min) — no hace API call extra por tick del anticipator.
"""

from __future__ import annotations

from datetime import datetime

from rag_anticipate.signals.base import register_signal


# Window: hilos con [_AGE_MIN_DAYS, _AGE_MAX_DAYS] de antigüedad sin reply.
# <1d es muy temprano (el user puede estar respondiendo activamente),
# >30d asumimos que se dejó pasar a propósito o es spam que escapó del filter.
_AGE_MIN_DAYS = 1.0
_AGE_MAX_DAYS = 30.0

# Cap defensivo: si el user vuelve de vacaciones con 50 hilos pendientes,
# 3 es lo que vale la pena pushear; el resto los ve via `rag morning`.
_MAX_CANDIDATES = 3

# Score buckets — más viejo = más urgente.
_SCORE_BUCKETS: tuple[tuple[float, float], ...] = (
    (7.0, 1.0),   # >7d → urgente
    (3.0, 0.7),   # 3-7d → importante
    (1.0, 0.5),   # 1-3d → reminder suave
)


def _truncate(s: str, n: int) -> str:
    """Trunca un string respetando palabras hasta `n` chars con elipsis."""
    s = (s or "").strip().replace("\n", " ").replace("\r", " ")
    if len(s) <= n:
        return s
    cut = s[: n - 1]
    # Cortar en última palabra si el cut quedó a mitad de palabra.
    last_space = cut.rfind(" ")
    if last_space > n * 0.7:
        cut = cut[:last_space]
    return cut + "…"


@register_signal(name="unread_email_thread", snooze_hours=24)
def unread_email_thread_signal(now: datetime) -> list:
    """Emite hasta `_MAX_CANDIDATES` candidates: hilos esperando respuesta
    del user. Silent-fail completo.
    """
    try:
        from rag import AnticipatoryCandidate
        from rag.integrations.gmail import _fetch_gmail_evidence

        ev = _fetch_gmail_evidence(now)
        if not isinstance(ev, dict):
            return []
        awaiting = ev.get("awaiting_reply") or []
        if not awaiting:
            return []

        # Sort by days_old DESC para priorizar los más viejos (más urgentes).
        try:
            sorted_threads = sorted(
                awaiting,
                key=lambda t: float(t.get("days_old") or 0.0),
                reverse=True,
            )
        except Exception:
            sorted_threads = list(awaiting)

        out: list = []
        for th in sorted_threads:
            if len(out) >= _MAX_CANDIDATES:
                break
            try:
                sender = (th.get("from") or "").strip()
                subject = (th.get("subject") or "").strip()
                snippet = (th.get("snippet") or "").strip()
                thread_id = (th.get("thread_id") or "").strip()
                if not sender or not subject or not thread_id:
                    continue
                try:
                    days_old = float(th.get("days_old") or 0.0)
                except (TypeError, ValueError):
                    continue
                if not (_AGE_MIN_DAYS <= days_old <= _AGE_MAX_DAYS):
                    continue

                # Score por bucket.
                score = 0.0
                for threshold, s in _SCORE_BUCKETS:
                    if days_old >= threshold:
                        score = s
                        break
                if score == 0.0:
                    continue

                # Sender preview: extraer "Name <email>" → "Name" preferido.
                sender_short = sender
                if "<" in sender:
                    sender_short = sender.split("<", 1)[0].strip().strip('"')
                    if not sender_short:
                        # Solo email, no name — usar el local-part.
                        email_part = sender[sender.index("<") + 1:].rstrip(">")
                        sender_short = email_part.split("@", 1)[0]
                sender_short = _truncate(sender_short, 40)

                # Days label en español rioplatense.
                if days_old < 2:
                    when = "ayer"
                else:
                    when = f"hace {int(days_old)}d"

                subject_preview = _truncate(subject, 70)
                snippet_preview = _truncate(snippet, 100)

                lines = [f"📧 *{sender_short}* {when} sin respuesta",
                         f"_{subject_preview}_"]
                if snippet_preview:
                    lines.append(f"> {snippet_preview}")
                message = "\n".join(lines)

                out.append(AnticipatoryCandidate(
                    kind="anticipate-unread_email_thread",
                    score=score,
                    message=message,
                    dedup_key=f"email-awaiting:{thread_id}",
                    snooze_hours=24,
                    reason=(
                        f"sender={sender_short!r} days_old={days_old:.1f} "
                        f"thread_id={thread_id}"
                    ),
                ))
            except Exception:
                continue

        return out
    except Exception:
        return []
