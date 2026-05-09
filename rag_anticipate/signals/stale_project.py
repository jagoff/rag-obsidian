"""Signal — Stale project nudge.

Detecta proyectos en `01-Projects/<sub>/` que no se tocaron en ≥7 días y
sugiere al user cerrarlos / archivarlos / continuarlos. La idea es romper
el patrón típico de "carpeta de proyecto eternamente abierta" donde el
user empezó algo, lo dejó a medias, y la carpeta queda años en
01-Projects/ acumulando ruido y compitiendo en el retrieval contra
proyectos vivos.

Diseño:

- File-system only: walks `01-Projects/<sub>/` immediate subdirs (NO
  recursión profunda). Para cada subdir, mtime más reciente entre sus
  `.md` files = "última actividad real" (mtime del directorio mismo es
  ruidoso por updates de Spotlight / Finder).
- Min notes ≥3: proyecto con 1-2 notas es un stub fresh, no nudge.
- Min staleness ≥7d: <7d es uso normal (semana de pausa por vacación,
  weekend, etc.).
- Max emit 1 por run: pushear 5 stale projects simultáneos es overwhelming.
  Sort por staleness desc, tomar el más viejo.
- Snooze 168h (1 semana): nudge mensual es suficiente, weekly es spam.
- Dedup bucketed: `stale_project:<sub>:<bucket>` donde bucket es
  `7d`/`14d`/`30d`/`60d`/`90d+`. Mismo proyecto re-pusheable cuando
  cruza al siguiente bucket — evita re-push diario sobre la misma
  carpeta sin escalar el mensaje.

Score calibration (escala con staleness, no con tamaño del proyecto):

    días desde última actividad → score
    7-13d   → 0.4 (apenas sobre threshold default 0.35)
    14-29d  → 0.6 (medio mes — empieza a oler a abandonado)
    30-59d  → 0.8 (un mes — high signal)
    60-89d  → 0.9
    90d+    → 1.0 (saturado — proyecto fantasma, considerar archive)

Excluye sub-prefixes ETL/auto-gen (típicamente NO están en 01-Projects/
pero si el user creó un proyecto que linka a ese path, defensivo). Y
excluye sub-prefix `_*` o `.*` (carpetas hidden/internal).

Silent-fail total: cualquier excepción interna → `[]`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from rag_anticipate.signals.base import register_signal


# Bucket donde viven los proyectos del user en el PARA del vault.
_PROJECTS_PREFIX = "01-Projects"

# Mínimo de notas .md en el subdir para considerarlo un proyecto real.
# Proyecto con 1-2 notas es un stub recién creado; no aplica el nudge.
_STALE_MIN_NOTES = 3

# Mínimo de días sin actividad para emitir. <7d es uso normal.
_STALE_MIN_DAYS = 7

# Buckets de staleness para dedup_key. Si el proyecto cruza al próximo
# bucket re-emite (escalation), si no, snooze normal.
_STALE_BUCKETS: tuple[tuple[int, str], ...] = (
    (90, "90d+"),
    (60, "60d"),
    (30, "30d"),
    (14, "14d"),
    (7, "7d"),
)


def _staleness_bucket(days: int) -> str:
    """Mapea días-desde-actividad al label de bucket."""
    for threshold, label in _STALE_BUCKETS:
        if days >= threshold:
            return label
    return "fresh"


def _score_for_staleness(days: int) -> float:
    """Mapea días-desde-actividad a score [0.4, 1.0]."""
    if days >= 90:
        return 1.0
    if days >= 60:
        return 0.9
    if days >= 30:
        return 0.8
    if days >= 14:
        return 0.6
    return 0.4


def _newest_mtime_in_dir(d: Path) -> tuple[float, int] | None:
    """Devuelve `(newest_mtime, num_md_files)` o `None` si vacío / no accesible.

    Walk recursivo limitado a `.md` (otras extensiones — imagenes, audio —
    no cuentan como "actividad" del user). Recursivo porque proyectos suelen
    tener subcarpetas (`Letras/`, `Reuniones/`, etc.) y queremos detectar
    actividad en cualquier nivel del proyecto.
    """
    newest = 0.0
    count = 0
    try:
        for p in d.rglob("*.md"):
            try:
                if not p.is_file():
                    continue
                # Skip hidden / system subpaths
                rel = p.relative_to(d).as_posix()
                if any(part.startswith(".") or part.startswith("_")
                       for part in rel.split("/")[:-1]):
                    continue
                m = p.stat().st_mtime
                if m > newest:
                    newest = m
                count += 1
            except Exception:
                continue
    except Exception:
        return None
    if count == 0:
        return None
    return newest, count


@register_signal(name="stale_project", snooze_hours=168)
def stale_project_signal(now: datetime) -> list:
    """Emite máximo 1 candidate — el proyecto MÁS stale que cumpla los gates.

    Pasos:
    1. Resuelve vault. Si no accesible → [].
    2. List `01-Projects/<sub>/` immediate subdirs.
    3. Para cada subdir: contar `.md` recursivos + mtime más reciente.
    4. Filtrar por (notes ≥ _STALE_MIN_NOTES, staleness ≥ _STALE_MIN_DAYS).
    5. Sort por staleness desc.
    6. Emit el top con score + message + dedup bucket.
    """
    try:
        from rag import AnticipatoryCandidate, _resolve_vault_path  # noqa: PLC0415

        vault = _resolve_vault_path()
        if not isinstance(vault, Path) or not vault.exists():
            return []

        projects_root = vault / _PROJECTS_PREFIX
        if not projects_root.is_dir():
            return []

        cutoff = now - timedelta(days=_STALE_MIN_DAYS)
        cutoff_ts = cutoff.timestamp()

        # (staleness_days, sub_name, num_notes, last_activity)
        stale: list[tuple[int, str, int, datetime]] = []

        try:
            entries = list(projects_root.iterdir())
        except Exception:
            return []

        for sub in entries:
            try:
                if not sub.is_dir():
                    continue
                name = sub.name
                # Skip hidden / internal folders.
                if name.startswith(".") or name.startswith("_"):
                    continue

                result = _newest_mtime_in_dir(sub)
                if result is None:
                    continue
                newest, count = result

                if count < _STALE_MIN_NOTES:
                    continue
                if newest >= cutoff_ts:
                    continue  # Activity within window — fresh.

                last_dt = datetime.fromtimestamp(newest)
                days = max(0, (now - last_dt).days)
                stale.append((days, name, count, last_dt))
            except Exception:
                continue

        if not stale:
            return []

        # Más stale primero (días desc). Tiebreak: alfabético del nombre.
        stale.sort(key=lambda t: (-t[0], t[1]))

        days, name, count, last_dt = stale[0]
        bucket = _staleness_bucket(days)
        score = _score_for_staleness(days)

        last_iso = last_dt.date().isoformat()
        message = (
            f"📁 Proyecto stale: [[{name}]]\n"
            f"  {count} notas · última actividad hace {days}d ({last_iso}).\n"
            f"  ¿Cerrás, archivás, o seguís?"
        )
        dedup_key = f"stale_project:{name}:{bucket}"
        reason = (
            f"sub={name} days={days} notes={count} "
            f"last={last_iso} bucket={bucket}"
        )

        return [AnticipatoryCandidate(
            kind="anticipate-stale_project",
            score=score,
            message=message,
            dedup_key=dedup_key,
            snooze_hours=168,
            reason=reason,
        )]
    except Exception:
        return []
