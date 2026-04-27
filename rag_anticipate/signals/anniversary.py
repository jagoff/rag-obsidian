"""Signal — Aniversarios de notas del vault.

Detecta notas que cumplen ~1 año hoy (mtime o frontmatter `created:` en una
ventana de 360-370 días atrás, tolerancia ±5d para no perder semanas por
drift de scheduler). Emite un único candidate — el más cercano a 365d
exactos — para invitar al user a releer / actualizar / archivar.

Diseño:
- File-system only: no llama a `retrieve()` ni toca embeddings. Camina
  el vault con `rglob("*.md")` filtrando por PARA buckets (02-Areas,
  01-Projects, 03-Resources) para evitar ruido de Inbox/Archive.
- Limita a notas ≥500 chars (las triviales no justifican ping proactivo).
- Un solo candidate por pasada (el más cercano a 365d), dedup por
  `anniv:<file_rel>:<year_created>` estable cross-runs, snooze 30 días.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from rag_anticipate.signals.base import register_signal


# Buckets PARA de donde SÍ queremos empujar aniversarios — son notas
# "canónicas" (áreas permanentes, proyectos activos, recursos curados).
# Explícitamente NO incluye 00-Inbox (ruido) ni 04-Archive (ya archivadas
# por el user, el ping sería contraproducente).
_ANNIV_ALLOWED_PREFIXES: tuple[str, ...] = (
    "02-Areas/",
    "01-Projects/",
    "03-Resources/",
)

# Ventana de ±5 días alrededor de 365. Si el cron se saltea días por
# sleep/boot, no perdemos la nota — el mismo dedup_key re-aparece y el
# snooze de 30d lo ahoga igual.
_ANNIV_WINDOW_MIN_DAYS = 360
_ANNIV_WINDOW_MAX_DAYS = 370
_ANNIV_TARGET_DAYS = 365

# Mínimo de caracteres del body para considerar "nota sustantiva". Notas
# de <500 chars (stub, capturas de un link, TODOs sueltos) no merecen
# push proactivo de aniversario.
_ANNIV_MIN_CHARS = 500


def _anniv_parse_created_frontmatter(text: str) -> datetime | None:
    """Extrae `created:` del frontmatter YAML si parsea a datetime.

    Acepta formatos comunes que aparecen en Obsidian:
    - `created: 2024-05-13` → datetime al mediodía local
    - `created: 2024-05-13 14:30`
    - `created: 2024-05-13T14:30:00`
    - `created: "2024-05-13"`

    Devuelve None si no hay frontmatter, no hay campo `created`, o no
    parsea. Deliberadamente NO usa `rag.parse_frontmatter` para evitar
    un import circular temprano; re-implementa el match mínimo.
    """
    if not text.startswith("---"):
        return None
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return None
    fm_text = m.group(1)
    # Busqueda simple de `created: <valor>` en una línea (no anidado).
    cm = re.search(r"^created:\s*(.+?)\s*$", fm_text, re.MULTILINE)
    if not cm:
        return None
    raw = cm.group(1).strip().strip("'\"")
    # Intentar varios formatos en orden.
    fmts = (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    )
    for fmt in fmts:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _anniv_first_body_line(text: str) -> str:
    """Primera línea no-vacía del body (post-frontmatter, post-título H1).

    Usada para dar preview en el message. Trunca a 120 chars para que
    el WA body no se estire. Si no hay body (nota sólo con frontmatter),
    devuelve "".
    """
    # Strip frontmatter si existe.
    body = text
    if body.startswith("---"):
        m = re.match(r"^---\n.*?\n---\n?", body, re.DOTALL)
        if m:
            body = body[m.end():]
    for line in body.split("\n"):
        s = line.strip()
        if not s:
            continue
        # Saltear títulos H1 "# Title" — típicamente redundante con [[title]].
        if s.startswith("#"):
            continue
        # Truncar para preview.
        return s[:120]
    return ""


@register_signal(name="anniversary", snooze_hours=720)
def anniversary_signal(now: datetime) -> list:
    """Emite MÁXIMO 1 candidate: la nota más cercana a cumplir 1 año hoy.

    Silent-fail total: cualquier error (vault no accesible, permission,
    encoding, lo que sea) → `[]`. El orchestrator tiene su propio outer
    try/except pero este doble cinturón es el contrato del framework.
    """
    try:
        from rag import (
            AnticipatoryCandidate,
            _resolve_vault_path,
            is_excluded,
        )

        vault = _resolve_vault_path()
        if not isinstance(vault, Path) or not vault.exists():
            return []

        best: tuple[float, str, str, datetime] | None = None
        # Tuple: (score, file_rel, preview, created_dt)

        for md_path in vault.rglob("*.md"):
            try:
                if not md_path.is_file():
                    continue
                rel = md_path.relative_to(vault).as_posix()
                # Filtrar a los 3 buckets canónicos.
                if not any(rel.startswith(p) for p in _ANNIV_ALLOWED_PREFIXES):
                    continue
                # Respeta la exclusion list global (defensive — los prefixes
                # allowed ya descartan 00-Inbox/99-AI/etc., pero por si
                # hay subrutas aún excluidas como WhatsApp/Claude bajo 03-Resources/).
                if is_excluded(rel):
                    continue

                # Decidir el created_dt: preferencia frontmatter, fallback mtime.
                text: str | None = None
                try:
                    text = md_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                if len(text) < _ANNIV_MIN_CHARS:
                    continue

                created_dt: datetime | None = _anniv_parse_created_frontmatter(text)
                if created_dt is None:
                    try:
                        mtime = md_path.stat().st_mtime
                        created_dt = datetime.fromtimestamp(mtime)
                    except Exception:
                        continue

                age = now - created_dt
                age_days = age.total_seconds() / 86400.0
                if age_days < _ANNIV_WINDOW_MIN_DAYS or age_days > _ANNIV_WINDOW_MAX_DAYS:
                    continue

                # Score lineal: 1.0 en 365 exacto, 0.5 en ±5d extremos.
                # |365-age|=0 → 1.0; |365-age|=5 → 0.5. Clamp a [0, 1].
                distance = abs(_ANNIV_TARGET_DAYS - age_days)
                score = 1.0 - (distance / 10.0)
                score = max(0.0, min(1.0, score))

                preview = _anniv_first_body_line(text)

                if best is None or score > best[0]:
                    best = (score, rel, preview, created_dt)
            except Exception:
                # Nota individual malformada: saltar, seguir con las otras.
                continue

        if best is None:
            return []

        score, file_rel, preview, created_dt = best

        # Title: nombre del archivo sin extensión y sin path. Obsidian
        # wikilinks usan el basename.
        title = Path(file_rel).stem

        msg_lines = [f"🎂 Hace 1 año escribiste: [[{title}]]"]
        if preview:
            msg_lines.append(f"  > {preview}")
        msg_lines.append("")
        msg_lines.append("¿Releer, actualizar o archivar?")
        message = "\n".join(msg_lines)

        year_created = created_dt.year
        dedup_key = f"anniv:{file_rel}:{year_created}"

        reason = (
            f"age={(now - created_dt).total_seconds() / 86400.0:.1f}d "
            f"path={file_rel}"
        )

        return [AnticipatoryCandidate(
            kind="anticipate-anniversary",
            score=score,
            message=message,
            dedup_key=dedup_key,
            snooze_hours=720,
            reason=reason,
        )]
    except Exception:
        return []
