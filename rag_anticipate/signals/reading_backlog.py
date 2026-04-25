"""Reading backlog signal — detecta acumulación de notas marcadas como
"to-read" / "unread" que llevan ≥7 días sin tocar.

Idea: el user captura artículos / papers / blogposts con frontmatter
`status: to-read` o tag `#to-read` (o `#unread`) y la intención es leerlos
después. Si la pila supera ~10 ítems "olvidados" hace más de una semana,
es momento de bloquear una sesión de lectura o de archivar lo que ya no
interesa. Sin esta señal, los backlogs de lectura crecen monotónicamente
hasta que el user los abandona en silencio.

Algoritmo (silent-fail end-to-end):

1. Walk el vault buscando `.md` con `mtime ≥ 7 días atrás` (capturas de
   esta semana NO cuentan — el user todavía está procesando "lo nuevo").
2. Por cada nota vieja, verificar si califica como "to-read":
   - frontmatter `status: to-read` o `status: unread`
   - frontmatter `tags:` que contenga `to-read` o `unread`
   - inline `#to-read` o `#unread` en el body
   - O bien la nota vive en `03-Resources/Reading/` (folder convention)
3. Contar el total. Si `count >= 10` → emit UN candidate.

Score: `min(1.0, (count - 10) / 30.0 + 0.5)`
- count=10  → 0.5  (justo arriba del threshold del orchestrator 0.35)
- count=25  → 1.0  (saturado: ya es un problema serio)
- count=40+ → 1.0

dedup_key: `reading_backlog:{ISO_year}-W{ISO_week:02d}` — 1 push máximo
por semana ISO. Combinado con `snooze_hours=168` (1 semana), el user
recibe a lo sumo 1 recordatorio semanal aunque el cron corra cada 10 min.

Silent-fail total: cualquier error (vault no accesible, permission,
parse_frontmatter throws en yaml malformado, etc.) → `[]`. El
orchestrator tiene su propio outer try/except pero acá cumplimos el
contrato del framework.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from rag_anticipate.signals.base import register_signal


# Edad mínima (días) para que una nota cuente como "stale" en el backlog.
# Capturas de esta semana NO cuentan — el user todavía las puede procesar
# en su flujo normal. 7 días = "ya pasó un ciclo y sigue sin tocarse".
_BACKLOG_MIN_AGE_DAYS = 7

# Threshold para emitir. <10 notas en backlog: no molestamos. El user
# percibe ~10 como "ya se acumuló demasiado" según la heurística general
# del agente (mismo orden de magnitud que `inbox_pressure`).
_BACKLOG_EMIT_THRESHOLD = 10

# Score ramp: (count - threshold) / _BACKLOG_SCORE_RAMP + base.
# count=10 → 0.5; count=40 → 1.0 (saturado a partir de count=40).
_BACKLOG_SCORE_BASE = 0.5
_BACKLOG_SCORE_RAMP = 30.0

# Folders convention para reading list. Las notas que vivan acá cuentan
# como to-read aunque no tengan frontmatter explícito (la ubicación es
# señal suficiente).
_READING_FOLDERS = (
    "03-resources/reading/",
    "reading/",
)

# Valores que marcan una nota como to-read en `status:` o en tags.
_TO_READ_VALUES = frozenset({"to-read", "unread"})

# Inline hashtag matcher — `#to-read` o `#unread` como token completo.
# Word boundaries: precedido por inicio-de-string o whitespace, seguido
# por end-of-string, whitespace o puntuación común. Case-insensitive.
_INLINE_TAG_RE = re.compile(
    r"(?:^|\s)#(to-read|unread)(?=$|\s|[.,;:!?\)\]])",
    re.IGNORECASE,
)

# Strip frontmatter del body antes de buscar inline hashtags — un
# `tags: [to-read]` como YAML list NO debería matchear el regex inline,
# y si la string `#to-read` aparece dentro del bloque YAML por accidente
# tampoco queremos contarla 2 veces.
_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def _normalize_tag(tag: object) -> str:
    """Lowercase + strip leading `#`. Devuelve "" si el input es vacío/None."""
    s = str(tag or "").strip()
    if s.startswith("#"):
        s = s[1:]
    return s.lower()


def _is_to_read(text_or_fm) -> bool:
    """¿La nota está marcada como `to-read` / `unread`?

    Acepta tanto el texto completo (`str`) como un frontmatter ya parseado
    (`dict`). En el caso `str` se chequean además los inline hashtags
    (`#to-read`, `#unread`) en el body. En el caso `dict` solo se mira el
    frontmatter (no hay body disponible).

    Reconoce:
    - `status: to-read` / `status: unread` (case-insensitive)
    - `tags:` (list o scalar) que contenga `to-read` o `unread`
    - inline `#to-read` / `#unread` (solo si pasamos el texto completo)

    Silent-fail: si algo raro pasa al parsear el frontmatter, devuelve False
    (la nota simplemente no cuenta).
    """
    from rag import parse_frontmatter

    if isinstance(text_or_fm, dict):
        fm = text_or_fm
        body = ""
    else:
        text = str(text_or_fm or "")
        try:
            fm = parse_frontmatter(text)
        except Exception:
            fm = {}
        # Strip frontmatter para que los inline hashtags solo cuenten en el body real.
        body = _FRONTMATTER_RE.sub("", text, count=1)

    # 1. status field
    status = _normalize_tag(fm.get("status"))
    if status in _TO_READ_VALUES:
        return True

    # 2. tags field — Obsidian acepta list, scalar string ("a, b") o set.
    raw_tags = fm.get("tags")
    if raw_tags:
        if isinstance(raw_tags, str):
            parts = [p for p in re.split(r"[,;\s]+", raw_tags) if p]
        elif isinstance(raw_tags, (list, tuple, set)):
            parts = list(raw_tags)
        else:
            parts = [raw_tags]
        for t in parts:
            if _normalize_tag(t) in _TO_READ_VALUES:
                return True

    # 3. inline hashtags en el body (solo si tenemos texto)
    if body and _INLINE_TAG_RE.search(body):
        return True

    return False


def _in_reading_folder(rel_path: str) -> bool:
    """¿La nota vive en una carpeta convencional de reading list?

    Match case-insensitive contra los prefijos en `_READING_FOLDERS`.
    """
    rl = rel_path.lower().lstrip("/")
    return any(rl.startswith(prefix) for prefix in _READING_FOLDERS)


def _count_reading_backlog(vault: Path, min_age_days: int = _BACKLOG_MIN_AGE_DAYS) -> int:
    """Cuenta notas to-read con `mtime ≥ min_age_days` días atrás.

    Recorre todo el vault recursivamente respetando `is_excluded()`. Una
    nota cuenta si:
    - `_is_to_read(text)` devuelve True, OR
    - vive en una carpeta de reading list (`03-Resources/Reading/`, etc.)

    Silent-fail por nota: archivos que no se puedan leer / statear se
    skipean, no rompen el conteo.
    """
    from rag import is_excluded

    try:
        if not vault.exists() or not vault.is_dir():
            return 0
    except Exception:
        return 0

    cutoff_ts = datetime.now().timestamp() - (min_age_days * 86400.0)
    count = 0

    try:
        iterator = vault.rglob("*.md")
    except Exception:
        return 0

    for p in iterator:
        try:
            rel = str(p.relative_to(vault))
        except ValueError:
            continue
        try:
            if is_excluded(rel):
                continue
        except Exception:
            # is_excluded no debería tirar, pero si lo hace tratamos como
            # "no excluido" y seguimos para no silenciar señal.
            pass
        try:
            st = p.stat()
        except OSError:
            continue
        # Solo notas viejas (>= min_age_days). Capturas recientes no cuentan.
        if st.st_mtime > cutoff_ts:
            continue

        in_folder = _in_reading_folder(rel)
        if in_folder:
            count += 1
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _is_to_read(text):
            count += 1

    return count


@register_signal(name="reading_backlog", snooze_hours=168)
def reading_backlog_signal(now: datetime) -> list:
    """Emite MÁXIMO 1 candidate cuando el backlog de reading list ≥10 notas
    con mtime ≥ 7 días atrás.

    Silent-fail total: cualquier error → `[]`. Ver docstring del módulo.
    """
    from rag import AnticipatoryCandidate

    try:
        from rag import _resolve_vault_path
        try:
            vault = _resolve_vault_path()
        except Exception:
            return []
        if not isinstance(vault, Path) or not vault.exists():
            return []

        count = _count_reading_backlog(vault, min_age_days=_BACKLOG_MIN_AGE_DAYS)

        if count < _BACKLOG_EMIT_THRESHOLD:
            return []

        # Score: 0.5 en el threshold (10), saturado a 1.0 con count=40+.
        score = min(
            1.0,
            (count - _BACKLOG_EMIT_THRESHOLD) / _BACKLOG_SCORE_RAMP + _BACKLOG_SCORE_BASE,
        )

        message = (
            f"📚 {count} notas en backlog de lectura (≥7d). "
            f"¿Clear session de 1h o archivar?"
        )

        # dedup_key por semana ISO — 1 emisión por semana máx. snooze_hours=168
        # (1 semana) es el doble cinturón: si el push falla o el user no lo
        # confirma, no retry hasta la semana siguiente.
        iso = now.isocalendar()
        # Compatibilidad: en 3.9+ es IsoCalendarDate (named tuple), en 3.8 era tuple.
        try:
            iso_year = iso.year
            iso_week = iso.week
        except AttributeError:
            iso_year, iso_week, _ = iso
        week_label = f"{iso_year}-W{iso_week:02d}"
        dedup_key = f"reading_backlog:{week_label}"

        reason = f"backlog_count={count} threshold={_BACKLOG_EMIT_THRESHOLD} week={week_label}"

        return [AnticipatoryCandidate(
            kind="anticipate-reading_backlog",
            score=score,
            message=message,
            dedup_key=dedup_key,
            snooze_hours=168,
            reason=reason,
        )]
    except Exception:
        return []
