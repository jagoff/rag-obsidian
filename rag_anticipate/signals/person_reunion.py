"""Person reunion signal — detecta cuando una persona (wikilink estilo
nombre propio) reaparece en una nota de hoy después de un gap largo sin
ser mencionada en ningún otro lado del vault.

Algoritmo (silent-fail end-to-end):

1. Buscar notas modificadas en las últimas 6h con tamaño ≥300 chars.
2. Extraer wikilinks que parezcan nombres propios (mayúscula, 1-4
   palabras, sin barras ni chars raros).
3. Por cada persona, walk el vault buscando la nota MÁS RECIENTE previa
   a la `today_note` que también la mencione.
4. Si el gap > 30 días → reunion detectada.
5. Emit un `AnticipatoryCandidate` con score escalado por gap
   (`min(1.0, gap_days / 180.0)`). Máximo 2 candidates por run — las 2
   personas con mayor gap ganan.

Como cualquier signal del Anticipatory Agent, el contrato es:
- signature `(now: datetime) -> list[AnticipatoryCandidate]`
- excepciones internas devuelven `[]`, no propagan.
- `dedup_key` estable cross-runs: `reunion:{person}:{today_file_rel}`.
- `snooze_hours=72` — una vez empujada la reunion, no repetir 3 días.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

from rag_anticipate.signals.base import register_signal


# Regex local — formato `[[Target]]` con alias opcional `[[Target|alias]]`
# y anchor opcional `[[Target#section]]`. Captura solo el target.
_WIKILINK_RE_LOCAL = re.compile(r"\[\[([^\]|#\n]+?)(?:[\|#][^\]]*)?\]\]")

# Un nombre propio: 1-4 palabras, cada una empezando con mayúscula
# (incluye acentos del español). Permite apóstrofes, puntos y guiones
# dentro de una palabra (D'Angelo, Jr., Smith-Jones) pero no barras,
# dígitos o guiones bajos (filtra slugs, timestamps, MOCs con fechas).
_PERSON_NAME_RE = re.compile(
    r"^[A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü.'\-]*"
    r"(?:\s[A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü.'\-]*){0,3}$"
)


def _extract_capitalized_wikilinks(text: str) -> list[str]:
    """Return wikilinks que *parecen* nombres propios, deduplicated, en
    orden de aparición.

    Filtros:
    - empieza con mayúscula
    - 1-4 palabras
    - sin `/` (excluye paths del tipo `[[folder/note]]`)
    - sin dígitos (excluye `[[2024-05-01]]`, `[[Log 42]]`)
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in _WIKILINK_RE_LOCAL.findall(text):
        target = raw.strip()
        if not target or "/" in target:
            continue
        if target in seen:
            continue
        if not _PERSON_NAME_RE.match(target):
            continue
        seen.add(target)
        out.append(target)
    return out


def _find_last_mention_before(
    vault: Path, person: str, cutoff_mtime: float
) -> tuple[str, float] | None:
    """Walk el vault buscando la nota con mayor `mtime < cutoff_mtime`
    que contenga `[[person]]` (o la variante con alias/anchor).

    Devuelve `(rel_path, mtime)` o `None` si no hay mención previa.
    Silent-fail: cualquier error por nota se skipea, no propaga.
    """
    from rag import is_excluded

    if not vault.is_dir():
        return None

    escaped = re.escape(person)
    # Matches [[person]], [[person|alias]], [[person#section]]
    search_re = re.compile(r"\[\[" + escaped + r"(?:[\|#][^\]]*)?\]\]")

    best: tuple[float, str] | None = None
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
            st = p.stat()
        except OSError:
            continue
        if st.st_mtime >= cutoff_mtime:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not search_re.search(text):
            continue
        if best is None or st.st_mtime > best[0]:
            best = (st.st_mtime, rel)

    if best is None:
        return None
    return (best[1], best[0])


@register_signal(name="person_reunion", snooze_hours=72)
def person_reunion_signal(now: datetime) -> list:
    """Detecta personas que reaparecen en el vault tras >30 días de
    silencio. Ver docstring del módulo para el algoritmo completo.
    """
    from rag import AnticipatoryCandidate, _resolve_vault_path, is_excluded

    try:
        try:
            vault = _resolve_vault_path()
        except Exception:
            return []
        if not vault or not vault.is_dir():
            return []

        gap_threshold_days = 30.0
        within_hours = 6
        min_chars = 300
        max_candidates = 2
        score_full_gap_days = 180.0

        cutoff = time.time() - within_hours * 3600

        # 1. Notas recientes que califican.
        recent: list[tuple[float, Path, str]] = []
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
                st = p.stat()
            except OSError:
                continue
            if st.st_mtime < cutoff:
                continue
            if st.st_size < min_chars:
                continue
            recent.append((st.st_mtime, p, rel))

        if not recent:
            return []

        recent.sort(reverse=True)

        # 2-4. Por cada nota reciente, extraer personas, buscar reunions.
        # Dedup por persona — nos quedamos con el MAYOR gap si la misma
        # persona aparece en varias notas recientes.
        reunions: dict[str, dict] = {}
        for today_mtime, today_path, today_rel in recent:
            try:
                text = today_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            wikilinks = _extract_capitalized_wikilinks(text)
            if not wikilinks:
                continue
            today_title = today_path.stem
            for person in wikilinks:
                prev = _find_last_mention_before(vault, person, today_mtime)
                if prev is None:
                    continue
                old_rel, old_mtime = prev
                gap_days = (today_mtime - old_mtime) / 86400.0
                if gap_days < gap_threshold_days:
                    continue
                existing = reunions.get(person)
                if existing is None or gap_days > existing["gap_days"]:
                    reunions[person] = {
                        "gap_days": gap_days,
                        "today_rel": today_rel,
                        "today_title": today_title,
                        "old_rel": old_rel,
                        "old_mtime": old_mtime,
                    }

        if not reunions:
            return []

        # 5. Top-N por gap descendente.
        sorted_reunions = sorted(
            reunions.items(),
            key=lambda kv: kv[1]["gap_days"],
            reverse=True,
        )[:max_candidates]

        out: list = []
        for person, info in sorted_reunions:
            gap_days = info["gap_days"]
            score = min(1.0, gap_days / score_full_gap_days)
            old_date = datetime.fromtimestamp(info["old_mtime"]).strftime(
                "%Y-%m-%d"
            )
            old_title = Path(info["old_rel"]).stem
            msg = (
                f"👋 Después de {int(gap_days)} días mencionás a "
                f"[[{person}]] de nuevo en [[{info['today_title']}]]. "
                f"Última nota donde apareció: [[{old_title}]] "
                f"({old_date})."
            )
            dedup_key = f"reunion:{person}:{info['today_rel']}"
            out.append(
                AnticipatoryCandidate(
                    kind="anticipate-person_reunion",
                    score=score,
                    message=msg,
                    dedup_key=dedup_key,
                    snooze_hours=72,
                    reason=f"gap={int(gap_days)}d, person={person}",
                )
            )
        return out
    except Exception:
        return []
