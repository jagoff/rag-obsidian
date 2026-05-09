"""Signal — Cumpleaños del día.

Escanea `99-obsidian/99-Contacts/*.md` (las contact notes del user),
parsea la línea `- **Cumpleaños**: dd/mm/yyyy` (o variantes con `:` y
formato dd-mm-yyyy), y si el mes-día matchea HOY emite un push:

    🎂 *hoy es el cumple de Maria* (35 años) — ¿le saludas?

## Diseño

- File-system only: no llama a `retrieve()` ni toca embeddings. Camina
  exclusivamente la carpeta `99-obsidian/99-Contacts/` con `glob("*.md")`.
- Skip al user mismo (notas con `type: self` en frontmatter o nombres
  conocidos del user — `Fer F.md`, `Yo.md`).
- Skip `_template.md` (placeholder con valores genéricos).
- Multiple candidates si hay varios cumples el mismo día — cada uno con
  su propio dedup_key + snooze 24h.
- dedup_key incluye el AÑO actual: `birthday:<basename>:<year>`. Esto
  permite que el mismo cumple re-aparezca el año siguiente sin que el
  dedup lo bloquee.
- Score: 1.0 (todos los cumples merecen el mismo prioritario push).
- Skip si la fecha del frontmatter no parsea o si el año es claramente
  inválido (>= year actual, fechas tipo 1700 etc.).

## Edge cases manejados

- 29 de febrero: en años no-bisiestos, mostramos el push el 28-feb +
  info "(cumple 29-feb)".
- Año de nacimiento opcional: si parsea solo dd/mm sin año, mostramos
  push sin edad ("hoy es el cumple de X — ¿le saludas?" sin años).
- Múltiples cumples mismo día: emitimos uno por persona, cap 5
  defensivo para no spamear si una weird sync mete 50 contactos.

## Anti-noise

- Snooze 24h: si el cron corre cada 10min, solo emitimos una vez por
  día por persona. Si el user dismisses, vuelve a aparecer al día
  siguiente (no en el mismo día).
- Window aware: emitimos en cualquier hora del día (anticipator
  framework decide cuándo enviar via quiet-hours config).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from rag_anticipate.signals.base import register_signal

# Cap defensivo. Si un día caen 10 cumples (familia grande, equipo de
# laburo, fechas duplicadas por mistake), 5 es más que suficiente — el
# user puede ver el resto via `rag pendientes` después.
_MAX_CANDIDATES = 5

# Pattern que matchea la línea de cumpleaños en el body. Tolera:
#   - "**Cumpleaños**: 19/07/1981"
#   - "Cumpleanos: 19-07-1981"  (sin tilde, separador -)
#   - "**Cumpleaños:** 19/7/81"  (separadores y year cortos)
# Captura: día, mes, año (año opcional).
_BIRTHDAY_RX = re.compile(
    r"\*?\*?Cumplea[ñn]os\*?\*?\s*:?\s*\*?\*?\s*"
    r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?",
    re.IGNORECASE,
)

# Names del user mismo — nunca pushear "es tu propio cumple".
_USER_SELF_NAMES = frozenset({"Fer F", "Yo", "Fer", "Fernando"})

# Files que NO son contactos reales (templates, helpers).
_SKIP_FILENAMES = frozenset({"_template.md", "README.md", ".DS_Store"})


def _parse_birthday(text: str) -> tuple[int, int, int | None] | None:
    """Extrae (day, month, year_or_None) del body de la nota.

    Returns None si no hay match o el match no es plausible (mes >12,
    día >31, año-nacimiento futuro, año demasiado viejo).
    """
    m = _BIRTHDAY_RX.search(text)
    if not m:
        return None
    try:
        day = int(m.group(1))
        month = int(m.group(2))
    except (ValueError, TypeError):
        return None
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    year_raw = m.group(3)
    year: int | None = None
    if year_raw:
        try:
            year = int(year_raw)
        except (ValueError, TypeError):
            year = None
        else:
            # Year cortos tipo "81" → 1981, "05" → 2005 heurística.
            if year < 100:
                year = 1900 + year if year >= 30 else 2000 + year
            # Sanity: no aceptar futuras o pre-1900.
            now_year = datetime.now().year
            if year > now_year or year < 1900:
                year = None
    return (day, month, year)


def _matches_today(day: int, month: int, now: datetime) -> bool:
    """¿La fecha del cumple matchea hoy?

    Edge case: 29-feb en año no-bisiesto → matcheamos contra 28-feb
    para no perder el push del año.
    """
    if day == now.day and month == now.month:
        return True
    # 29-feb fallback en año no-bisiesto.
    if day == 29 and month == 2 and now.month == 2 and now.day == 28:
        # Verificar que este año NO sea bisiesto.
        is_leap = (now.year % 4 == 0 and now.year % 100 != 0) or (now.year % 400 == 0)
        if not is_leap:
            return True
    return False


def _is_user_self(stem: str, text: str) -> bool:
    """¿La nota es del user mismo (no un contacto)?

    Detecta por:
      1. Filename matchea _USER_SELF_NAMES.
      2. Frontmatter tiene `type: self` o `relacion: yo`.
    """
    if stem in _USER_SELF_NAMES:
        return True
    # Quick frontmatter peek (no usamos parse_frontmatter para evitar import circular).
    if text.startswith("---"):
        m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if m:
            fm = m.group(1).lower()
            if re.search(r"^type:\s*self\s*$", fm, re.MULTILINE):
                return True
            if re.search(r"^relaci[oó]n:\s*(yo|self|propio)\s*$", fm, re.MULTILINE):
                return True
    return False


@register_signal(name="birthday_today", snooze_hours=24)
def birthday_today_signal(now: datetime) -> list:
    """Emite hasta `_MAX_CANDIDATES` candidates: contactos cuyo cumple es hoy.

    Silent-fail completo. Vault no accesible / ningún contacto / errores
    individuales por archivo → `[]` o lista parcial.
    """
    try:
        from rag import AnticipatoryCandidate, _resolve_vault_path

        vault = _resolve_vault_path()
        if not isinstance(vault, Path) or not vault.exists():
            return []

        # 99-obsidian/99-Contacts es la convention del user (regla del
        # CLAUDE.md global: TODO sistema vive bajo 99-obsidian/).
        contacts_dir = vault / "99-obsidian" / "99-Contacts"
        if not contacts_dir.is_dir():
            return []

        out: list = []
        for md_path in sorted(contacts_dir.glob("*.md")):
            if len(out) >= _MAX_CANDIDATES:
                break
            try:
                if md_path.name in _SKIP_FILENAMES:
                    continue
                text = md_path.read_text(encoding="utf-8", errors="replace")
                if _is_user_self(md_path.stem, text):
                    continue
                parsed = _parse_birthday(text)
                if parsed is None:
                    continue
                day, month, year = parsed
                if not _matches_today(day, month, now):
                    continue

                name = md_path.stem
                age_str = ""
                if year is not None:
                    age = now.year - year
                    if 0 < age < 150:  # plausibility guard
                        age_str = f" ({age} años)"

                # 29-feb gets a hint when shown on 28-feb non-leap.
                date_hint = ""
                if day == 29 and month == 2 and now.day == 28 and now.month == 2:
                    date_hint = " _[cumple 29-feb, año no bisiesto]_"

                message = (
                    f"🎂 *hoy es el cumple de {name}*{age_str} — "
                    f"¿le saludas?{date_hint}"
                )

                out.append(AnticipatoryCandidate(
                    kind="anticipate-birthday_today",
                    score=1.0,
                    message=message,
                    dedup_key=f"birthday:{name}:{now.year}",
                    snooze_hours=24,
                    reason=f"name={name} dob={day:02d}/{month:02d}"
                           + (f"/{year}" if year else ""),
                ))
            except Exception:
                # Per-file failure: skip, sigue con los otros.
                continue

        return out
    except Exception:
        return []
