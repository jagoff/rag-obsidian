"""Deadline signal — empuja avisos proactivos de deadlines inminentes.

Dos sources combinadas, mismo formato de candidate output:

1. **Vault frontmatter `due:`** — scan `.md` files con `due:` en el YAML
   header. Default histórico del signal.
2. **Apple Reminders incompletos con `due_date`** (agregado 2026-05-09 P4):
   audit del audit reveló vault con 0 notas con `due:` — el user gestiona
   deadlines via Reminders + tasks WA. El signal leía el lugar equivocado.
   `_fetch_reminders_due(now, horizon_days=3)` retorna overdue/today/
   upcoming buckets; tomamos overdue + today + upcoming en [+0, +3 días].

Ambos sources alimentan la misma lista `found`; ordenamos por proximidad
ascendente (más inminente primero) y emitimos máximo 2 candidates totales
para no saturar el feed.

Score calibration (mismo para ambos sources):

Score calibration:
    días hasta el due → score
    0 (hoy)          → 1.00
    1 (mañana)       → 0.75
    2 (pasado)       → 0.50
    3 (+3 días)      → 0.25

Accepta varios formatos en el frontmatter:
    due: 2026-04-28                # ISO date (YAML parsea como datetime.date)
    due: "2026-04-28"              # string ISO
    due: "2026-04-28T10:00"        # ISO con hora
    due: "28/04/2026"              # DD/MM/YYYY
    due: [2026-05-01, 2026-04-25]  # lista YAML — usa la primera parseable

Fail silencioso: cualquier excepción interna (vault inaccesible, frontmatter
roto, fecha malformada) degrada a `return []` — el orchestrator tiene un
outer try/except como safety net, pero esta señal honra el contrato.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterator

from rag_anticipate.signals.base import register_signal


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_due_value(raw: object) -> date | None:
    """Normaliza un valor de frontmatter `due` a `datetime.date` o `None`.

    Acepta:
      - `datetime.date` / `datetime.datetime` (lo que PyYAML hidrata de un
        scalar tipo `2026-04-28`)
      - `str` en formato ISO ("2026-04-28" o "2026-04-28T10:00")
      - `str` en formato DD/MM/YYYY ("28/04/2026")
      - `list` / `tuple`: recursa sobre cada elemento y devuelve el primero
        parseable (orden de aparición, no de cercanía).

    Devuelve `None` si el input es `None`, vacío, o no matchea ningún formato.
    """
    if raw is None:
        return None

    # datetime.datetime primero porque es subclase de datetime.date
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw

    if isinstance(raw, (list, tuple)):
        for item in raw:
            parsed = _parse_due_value(item)
            if parsed is not None:
                return parsed
        return None

    if not isinstance(raw, str):
        # Cualquier otro tipo (int, dict, etc.) — no parseable
        return None

    s = raw.strip()
    if not s:
        return None

    # ISO con hora: "2026-04-28T10:00" → quedarse con la parte de fecha
    # fromisoformat acepta tanto "YYYY-MM-DD" como "YYYY-MM-DDTHH:MM[:SS]".
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        pass

    try:
        return date.fromisoformat(s)
    except ValueError:
        pass

    # DD/MM/YYYY (estilo rioplatense)
    try:
        return datetime.strptime(s, "%d/%m/%Y").date()
    except ValueError:
        pass

    return None


def _walk_notes_with_due(vault: Path) -> Iterator[tuple[str, date]]:
    """Generator: yield `(rel_path, due_date)` para cada nota con frontmatter
    `due:` parseable. Silencia errores por-archivo (nota con FM roto no
    aborta el walk)."""
    from rag import is_excluded, parse_frontmatter

    if not isinstance(vault, Path):
        vault = Path(str(vault))
    if not vault.is_dir():
        return

    for p in vault.rglob("*.md"):
        try:
            rel = str(p.relative_to(vault))
        except ValueError:
            continue
        try:
            if is_excluded(rel):
                continue
        except Exception:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        except UnicodeDecodeError:
            continue
        try:
            fm = parse_frontmatter(text)
        except Exception:
            continue
        if not fm or "due" not in fm:
            continue
        due = _parse_due_value(fm.get("due"))
        if due is None:
            continue
        yield rel, due


# ── reminders source (P4 audit 2026-05-09) ─────────────────────────────────


def _walk_reminders_with_due(now: datetime) -> Iterator[tuple[str, str, date]]:
    """Yield `(reminder_id, name, due_date)` para Apple Reminders incompletos
    con `due_date` ≤ hoy + 3 días.

    Silent-fail total: si Apple integration está off (`OBSIDIAN_RAG_NO_APPLE
    =1`), o el osascript fallа, o el módulo no está disponible, devolvemos
    iterator vacío sin propagar la excepción — el signal sigue funcionando
    contra el source de frontmatter solo.

    El `reminder_id` es el `id` del Reminder de Apple (cuando viene del
    AppleScript con la shape nueva `id|name|due|list`), o un fallback
    sintético `_no_id_<idx>` si el script revertió a la shape vieja sin id.
    Lo usamos para el dedup_key — `deadline:reminder:<id>:<due>` evita re-
    push del mismo Reminder al día siguiente si el user no lo completó.
    """
    try:
        from rag.integrations.reminders import _fetch_reminders_due
    except Exception:
        return

    try:
        # horizon_days=3 alinea con la ventana del signal (días_until ≤ 3
        # en el caller). El fetch retorna también `overdue` (días_until
        # negativo) — los filtramos abajo.
        reminders = _fetch_reminders_due(now, horizon_days=3, max_items=20)
    except Exception:
        return

    today = now.date() if isinstance(now, datetime) else now
    for idx, item in enumerate(reminders or ()):
        try:
            due_raw = (item or {}).get("due") or ""
            if not due_raw:
                continue  # `undated` bucket — no aplica
            # `due_raw` es ISO string del fetcher (`isoformat(timespec="minutes")`)
            try:
                due_dt = datetime.fromisoformat(due_raw)
            except ValueError:
                continue
            due = due_dt.date()
            days_until = (due - today).days
            # Aceptamos overdue (negativo) hasta -7 días — más viejo es ruido
            # de Reminders olvidados que el user ya descartó mentalmente.
            if days_until < -7 or days_until > 3:
                continue
            rid = (item.get("id") or "").strip() or f"_no_id_{idx}"
            name = (item.get("name") or "").strip()
            if not name:
                continue
            yield rid, name, due
        except Exception:
            continue


# ── signal ──────────────────────────────────────────────────────────────────

@register_signal(name="deadline", snooze_hours=24)
def deadline_signal(now: datetime) -> list:
    """Emite hasta 2 candidates para deadlines en [hoy-7d (overdue), +3 días].

    Sources combinadas: frontmatter `due:` en notas + Apple Reminders
    incompletos con `due_date`. Los candidates se ordenan por proximidad
    ascendente (más inminente / más overdue primero) y se trunca a 2 para
    no saturar el feed.
    """
    try:
        from rag import AnticipatoryCandidate, _resolve_vault_path

        try:
            vault = _resolve_vault_path()
        except Exception:
            vault = None

        today = now.date() if isinstance(now, datetime) else now

        # `found` items: (days_until, source_kind, identifier, display, due)
        # source_kind: "note" → identifier=rel_path, display=note_stem
        # source_kind: "reminder" → identifier=reminder_id, display=name
        found: list[tuple[int, str, str, str, date]] = []

        # Source 1: vault frontmatter due:
        if vault is not None:
            try:
                for rel, due in _walk_notes_with_due(vault):
                    days_until = (due - today).days
                    if days_until < 0 or days_until > 3:
                        continue
                    found.append((days_until, "note", rel, Path(rel).stem, due))
            except Exception:
                pass

        # Source 2: Apple Reminders due_date
        try:
            for rid, name, due in _walk_reminders_with_due(now):
                days_until = (due - today).days
                # Reminders source acepta overdue (≥-7d) — frontmatter no
                # porque el user ya tendría visibilidad en la nota misma
                if days_until > 3:
                    continue
                found.append((days_until, "reminder", rid, name, due))
        except Exception:
            pass

        if not found:
            return []

        # Sort por proximidad ascendente con tiebreak determinístico (kind
        # alfabético, después identifier — "note" antes que "reminder" cuando
        # empatan en days_until).
        found.sort(key=lambda t: (t[0], t[1], t[2]))

        candidates = []
        for days_until, source_kind, identifier, display, due in found[:2]:
            # Score: 1.0 hoy/overdue, 0.75 mañana, 0.5 +2d, 0.25 +3d.
            # Para overdue (negativo) clamp a 1.0.
            score = round(max(0.0, min(1.0, 1.0 - (max(0, days_until) / 4.0))), 4)
            due_iso = due.isoformat()
            if source_kind == "note":
                if days_until == 0:
                    when = "hoy"
                else:
                    when = f"en {days_until} días"
                message = (
                    f"📌 Deadline {when}: [[{display}]]\n"
                    f"  Due: {due_iso}\n"
                    f"  ¿Avance? Si ya está hecho, marcarlo."
                )
                dedup_key = f"deadline:{identifier}:{due_iso}"
            else:  # reminder
                if days_until < 0:
                    when = f"overdue ({-days_until}d)"
                elif days_until == 0:
                    when = "hoy"
                else:
                    when = f"en {days_until}d"
                message = (
                    f"📌 Reminder {when}: {display}\n"
                    f"  Due: {due_iso}\n"
                    f"  ¿Lo cerrás o lo movés?"
                )
                dedup_key = f"deadline:reminder:{identifier}:{due_iso}"
            candidates.append(AnticipatoryCandidate(
                kind="anticipate-deadline",
                score=score,
                message=message,
                dedup_key=dedup_key,
                snooze_hours=24,
                reason=f"days_until={days_until}, source={source_kind}, due={due_iso}",
            ))

        return candidates
    except Exception:
        return []
