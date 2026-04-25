"""Deadline signal — empuja avisos proactivos de notas con `due:` inminente.

Scan del vault buscando frontmatter `due:` y emite candidates para notas cuyo
deadline cae en la ventana [hoy, +3 días]. El snooze default es 24h para que
el mismo deadline se pushee como mucho 1×/día (no spam).

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


# ── signal ──────────────────────────────────────────────────────────────────

@register_signal(name="deadline", snooze_hours=24)
def deadline_signal(now: datetime) -> list:
    """Emite hasta 2 candidates para notas cuyo `due` cae en [hoy, +3 días].

    El orden del resultado es por proximidad ascendente (más inminente
    primero). Dos candidates máximo para no saturar el feed — si hay 5
    deadlines mañana, el user ve los 2 más próximos y el resto espera.
    """
    try:
        from rag import AnticipatoryCandidate, _resolve_vault_path

        try:
            vault = _resolve_vault_path()
        except Exception:
            return []

        today = now.date() if isinstance(now, datetime) else now

        found: list[tuple[int, str, date]] = []
        for rel, due in _walk_notes_with_due(vault):
            days_until = (due - today).days
            if days_until < 0 or days_until > 3:
                continue
            found.append((days_until, rel, due))

        if not found:
            return []

        # Ordenar por proximidad ascendente (más inminente primero), con el
        # rel_path como desempate estable para hacer el orden determinista.
        found.sort(key=lambda t: (t[0], t[1]))

        candidates = []
        for days_until, rel, due in found[:2]:
            # Título de la nota = filename sin extensión
            note_title = Path(rel).stem
            score = round(1.0 - (days_until / 4.0), 4)
            due_iso = due.isoformat()
            message = (
                f"📌 Deadline en {days_until} días: [[{note_title}]]\n"
                f"  Due: {due_iso}\n"
                f"  ¿Avance? Si ya está hecho, marcarlo."
            )
            candidates.append(AnticipatoryCandidate(
                kind="anticipate-deadline",
                score=score,
                message=message,
                dedup_key=f"deadline:{rel}:{due_iso}",
                snooze_hours=24,
                reason=f"days_until={days_until}, due={due_iso}",
            ))

        return candidates
    except Exception:
        return []
