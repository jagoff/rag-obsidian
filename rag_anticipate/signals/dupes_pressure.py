"""Signal — Presión de duplicados (notas con títulos parecidos acumulados).

Detecta cuando hay un pile-up de pares candidatos a duplicado en el vault,
a partir de una heurística LIGHT sobre los títulos de las notas: sin embed,
sin correr el `rag dupes` completo (que es caro). La idea es que si ya
acumulaste ≥5 pares fuertes por similitud de nombre, vale la pena disparar
una invitación proactiva a revisar/mergear con `rag dupes --threshold 0.85`.

Diseño:

- File-system only: camina `vault.rglob("*.md")`, normaliza cada stem
  (lowercase + strip de todo lo que no sea alfanumérico) y busca pares
  con similitud ≥0.85 usando `difflib.SequenceMatcher` sobre la forma
  normalizada. El normalization hace que "ikigai.md" vs "Ikigai.md" (case),
  "coaching-notes.md" vs "coaching_notes.md" vs "coaching notes.md"
  (separadores) colapsen al mismo bucket y cuenten como pares exactos;
  y "proyecto-x-v1.md" vs "proyecto-x-v2.md" queden con ratio ~0.91,
  arriba del threshold.
- Respeta `is_excluded(rel)` del vault layer: dotdirs (`.obsidian/`,
  `.trash/`, ...), `00-Inbox/conversations/` (episodic), `04-Archive/99-*`,
  `03-Resources/Claude/<slug>/`, etc. quedan fuera del scan — son
  artefactos auto-generados que no son "dupes" del user.
- Cap defensivo de 2000 archivos escaneados — O(N²) sobre stems cortos es
  barato pero queremos evitar pathological cases donde un vault con 20k
  notas duplique la CPU del signal loop.
- Threshold de emisión: 5 pares. <5 pares → no emitimos (no vale la pena
  molestar al user por un par de dupes sueltos; el cron va a volver a
  checkear en 2 semanas vía snooze).
- Score: 0.5 en el threshold (5 pares), escala hasta 1.0 en 20+ pares.
- dedup_key por semana ISO → máximo 1 emisión por semana del año.
  Combinado con `snooze_hours=336` (2 semanas), la señal no spamea:
  aunque el cron corra cada 10 min, sólo empujamos 1×/cada 2 semanas
  efectivas.
"""

from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from rag_anticipate.signals.base import register_signal


# Mínimo de pares para disparar. <5 no emitimos — no vale la pena molestar
# por un par de dupes sueltos; el loop re-checkea en 2 semanas.
_DUPES_EMIT_THRESHOLD = 5

# Score ramp: score = (count - threshold) / ramp + base, clamped a [0, 1].
# count=5 → 0.5; count=20 → ~1.17 clamp → 1.0. count=12 → ~0.97.
_DUPES_SCORE_BASE = 0.5
_DUPES_SCORE_RAMP = 15.0

# Similarity threshold sobre stems normalizados (post strip de separadores
# y lowercase). 0.85 matchea el default de `rag dupes` y corresponde
# aproximadamente a 1-2 caracteres de diferencia en stems ~10-12 chars
# (ej. "proyectoxv1" vs "proyectoxv2" → 0.909).
_DUPES_RATIO_THRESHOLD = 0.85

# Cap defensivo de archivos escaneados. O(N²) es barato sobre strings
# cortos, pero si un vault crece a 20k notas queremos que el signal loop
# NO se convierta en un CPU hog. A 2000 archivos: ~2M comparaciones ×
# microsegundo = pocos segundos worst case.
_DUPES_MAX_FILES = 2000


def _normalize_stem(stem: str) -> str:
    """Lowercase + strip de todo lo que no sea alfanumérico.

    Colapsa separadores heterogéneos (`-`, `_`, espacio, `.`) al mismo
    string, así "coaching-notes", "coaching_notes" y "coaching notes"
    terminan todos en `"coachingnotes"` y matchean exacto.
    """
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def _find_title_similar_pairs(
    vault: Path,
    ratio_threshold: float = _DUPES_RATIO_THRESHOLD,
) -> list[tuple[Path, Path]]:
    """Devuelve pares `(path_a, path_b)` con stems "similares".

    "Similares" = `SequenceMatcher.ratio()` sobre los stems normalizados
    ≥ `ratio_threshold`. Los stems iguales post-normalization (ratio=1.0)
    obviamente también caen.

    - Respeta `is_excluded(rel)` del vault layer.
    - Skipea stems normalizados de <2 chars (poco significativos).
    - Cap defensivo a `_DUPES_MAX_FILES` archivos.
    - Silent-fail per file: si un path individual tira al stat/rglob,
      lo saltea sin tumbar el scan entero.
    """
    # Deferred import para evitar circular import durante el autodiscover
    # de señales (rag_anticipate se importa antes de que rag termine de
    # inicializar VAULT_PATH y compañía en algunos entrypoints).
    try:
        from rag import is_excluded
    except Exception:
        def is_excluded(_rel: str) -> bool:  # type: ignore[misc]
            return False

    files: list[tuple[Path, str]] = []
    try:
        iterator = vault.rglob("*.md")
    except Exception:
        return []

    for p in iterator:
        if len(files) >= _DUPES_MAX_FILES:
            break
        try:
            if not p.is_file():
                continue
            rel = p.relative_to(vault).as_posix()
            if is_excluded(rel):
                continue
        except Exception:
            continue
        norm = _normalize_stem(p.stem)
        # Stems de 0-1 chars post-normalization no son comparables con
        # ratio de forma útil — ej. "1.md" → "1", "A.md" → "a". Los
        # saltamos para evitar falsos positivos triviales.
        if len(norm) < 2:
            continue
        files.append((p, norm))

    pairs: list[tuple[Path, Path]] = []
    n = len(files)
    for i in range(n):
        pi, normi = files[i]
        li = len(normi)
        for j in range(i + 1, n):
            pj, normj = files[j]
            if normi == normj:
                pairs.append((pi, pj))
                continue
            # Prune rápido: si la diferencia de longitud es muy grande el
            # ratio no puede alcanzar 0.85 (upper bound = 2*min/(li+lj)).
            # Barato y evita entrar a SequenceMatcher para la mayoría de
            # pares obviamente disímiles.
            lj = len(normj)
            if 2 * min(li, lj) < ratio_threshold * (li + lj):
                continue
            try:
                ratio = SequenceMatcher(None, normi, normj).ratio()
            except Exception:
                continue
            if ratio >= ratio_threshold:
                pairs.append((pi, pj))
    return pairs


@register_signal(name="dupes_pressure", snooze_hours=336)
def dupes_pressure_signal(now: datetime) -> list:
    """Emite MÁXIMO 1 candidate cuando hay ≥5 pares candidatos a dupe.

    Silent-fail total: cualquier error (vault no accesible, rglob rompe,
    rag no importable) → `[]`. El orchestrator tiene su propio outer
    try/except como safety net, pero este doble cinturón es el contrato
    del framework.
    """
    try:
        from rag import AnticipatoryCandidate, _resolve_vault_path

        vault = _resolve_vault_path()
        if not isinstance(vault, Path) or not vault.exists():
            return []

        pairs = _find_title_similar_pairs(vault)
        count = len(pairs)

        if count < _DUPES_EMIT_THRESHOLD:
            return []

        # Score: 0.5 en threshold (5 pares), sube lineal hasta 1.0 en 20+.
        score = min(
            1.0,
            (count - _DUPES_EMIT_THRESHOLD) / _DUPES_SCORE_RAMP + _DUPES_SCORE_BASE,
        )

        message = (
            f"👥 {count} pares de posibles duplicados acumulados. "
            f"¿`rag dupes --threshold 0.85` para revisar y mergear?"
        )

        # dedup_key por semana ISO → una sola emisión por semana del año
        # incluso si el cron corre cada 10 min. Combinado con snooze_hours
        # =336 (2 semanas), nos da ~1×/cada 2 semanas efectivas.
        iso_year, iso_week, _ = now.isocalendar()
        dedup_key = f"dupes_pressure:{iso_year}-W{iso_week:02d}"

        reason = f"pairs={count} threshold={_DUPES_EMIT_THRESHOLD}"

        return [AnticipatoryCandidate(
            kind="anticipate-dupes_pressure",
            score=score,
            message=message,
            dedup_key=dedup_key,
            snooze_hours=336,
            reason=reason,
        )]
    except Exception:
        return []
