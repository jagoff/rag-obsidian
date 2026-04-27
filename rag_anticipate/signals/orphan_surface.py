"""Signal — Orphan surface.

Detecta notas recientes (creadas/modificadas en las últimas 2–24h) SIN
wikilinks outgoing. Las marca como huérfanas candidatas a densificar con
`rag wikilinks suggest` — es cuando "el costo" de linkear está más fresco
en la cabeza del user y el valor marginal es alto (red aún no construida).

Diseño:
- File-system only: camina el vault con `rglob("*.md")`, filtra por PARA
  buckets 01-Projects / 02-Areas / 03-Resources. 00-Inbox queda afuera
  porque igual se triga ahí (no tiene sentido pushear orphan de una nota
  que mañana va a otro bucket o al archive).
- Ventana de 2 a 24h: los 2h son "grace period" para que el user no sienta
  que lo interrumpimos apenas guardó; las 24h cortan lo suficientemente
  lejos como para no spammear notas viejas que ya se olvidó.
- Filtro ≥200 chars: fragments triviales (stubs, TODOs sueltos, captures
  rápidas) no ameritan el ping.
- Score escala con tamaño: una nota de 3000 chars sin UN solo wikilink es
  mucho más raro — y más valiosa de densificar — que una de 250 chars.
- Return MÁXIMO 2 candidates ordenados por tamaño descendente: orphan a
  granel es sintomático pero pushear 10 a la vez es ruido; 2 es el
  trade-off entre "no perder señal" y "no inundar WA".
- Silent-fail total: cualquier excepción interna → `[]`. Respetamos el
  contrato del framework aunque el orchestrator ya tenga su outer try.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

from rag_anticipate.signals.base import register_signal


# Buckets canónicos donde SÍ queremos empujar sugerencias de densificación.
# 00-Inbox se excluye deliberadamente — las notas ahí están en triaje
# pendiente y van a mover/archive/delete igual, el ping sería ruido.
# 04-Archive también queda fuera implícitamente por no estar en la lista.
_ORPHAN_ALLOWED_PREFIXES: tuple[str, ...] = (
    "01-Projects/",
    "02-Areas/",
    "03-Resources/",
)

# ETL output prefixes — autogenerados desde fuentes externas (Chrome
# history, Gmail dumps, WhatsApp rollups, Calendar, etc.) NO son notas
# humanas y typically tienen 0 wikilinks por diseño (son texto raw del
# bridge / ETL). El signal orphan_surface marca falsos positivos
# repetidos sobre estos archivos — telemetría 7d (audit 2026-04-26):
# `03-Resources/Chrome/2026-04-26.md` (154K chars) seleccionado 10×
# y siempre rebotado por daily_cap → ranura desperdiciada que podría
# usar otro signal real. Excluir explícitamente.
_ORPHAN_EXCLUDE_SUBPREFIXES: tuple[str, ...] = (
    "03-Resources/Chrome/",
    "03-Resources/Gmail/",
    "03-Resources/WhatsApp/",
    "03-Resources/Calendar/",
    "03-Resources/Reminders/",
    "03-Resources/Drive/",
    "03-Resources/Bookmarks/",
    "03-Resources/Spotify/",
    "03-Resources/Screen-Time/",
    "03-Resources/Safari/",
    "03-Resources/Contacts/",
)

# Grace period inferior: no pushear dentro de las primeras 2h tras el save.
# El user todavía tiene la nota abierta/fresca, preguntarle "¿querés
# agregarle links?" mientras está escribiendo es interruptivo.
_ORPHAN_MIN_AGE_HOURS = 2

# Ventana superior: pasadas 24h asumimos que la nota "se enfrió" y el push
# es menos útil (además lo cubre el snooze de 24h: si re-aparece al día
# siguiente ya se silenció).
_ORPHAN_MAX_AGE_HOURS = 24

# Mínimo de caracteres para considerar la nota "sustantiva" y flaggear la
# ausencia de wikilinks como señal. Por debajo de esto son typically
# stubs/TODOs que no van a tener red de links de todas formas.
_ORPHAN_MIN_CHARS = 200

# Fallback regex por si `_extract_wikilinks_from_markdown` de rag.py no
# está disponible. Match laxo — captura `[[algo]]` sin resolver alias ni
# secciones. Suficiente para contar > 0 vs 0 (que es todo lo que pregunta
# esta signal).
_ORPHAN_FALLBACK_RE = re.compile(r"\[\[[^\]]+\]\]")


def _count_outgoing_wikilinks(text: str) -> int:
    """Cuenta wikilinks outgoing en el body de una nota.

    Usa `_extract_wikilinks_from_markdown` de rag.py (que respeta alias y
    descarta la parte post-`|` / post-`#`) si está disponible. Fallback a
    un regex simple si rag.py cambió y la helper se renombró/movió.
    """
    try:
        from rag import _extract_wikilinks_from_markdown

        return len(_extract_wikilinks_from_markdown(text))
    except Exception:
        try:
            return len(_ORPHAN_FALLBACK_RE.findall(text))
        except Exception:
            return 0


def _score_for_chars(chars: int) -> float:
    """Mapea tamaño de la nota a score [0, 1].

    Bands:
    - ≥3000 chars → 0.9 (orphan "grande", high value de densificar)
    - ≥1500 chars → 0.8
    - ≥500 chars  → 0.6 (baseline de nota "sustantiva")
    - ≥200 chars  → 0.4 (pasó el filtro mínimo pero cerca del límite)

    El threshold global del agent (`RAG_ANTICIPATE_MIN_SCORE` default 0.35)
    deja pasar todo el rango por default, pero los 0.4 "mueren" fácil si
    el user sube el threshold o compiten contra una calendar/echo.
    """
    if chars >= 3000:
        return 0.9
    if chars >= 1500:
        return 0.8
    if chars >= 500:
        return 0.6
    return 0.4


@register_signal(name="orphan_surface", snooze_hours=24)
def orphan_surface_signal(now: datetime) -> list:
    """Emite hasta 2 candidates — las 2 notas orphan más grandes en la ventana.

    Silent-fail: cualquier error (vault no accesible, permission denied,
    encoding, etc.) → `[]`. El orchestrator tiene su propio try/except
    como safety net pero el contrato pide que cada signal degrade sola.
    """
    from rag import AnticipatoryCandidate

    try:
        from rag import _resolve_vault_path, is_excluded

        vault = _resolve_vault_path()
        if not isinstance(vault, Path) or not vault.exists():
            return []

        min_age = timedelta(hours=_ORPHAN_MIN_AGE_HOURS)
        max_age = timedelta(hours=_ORPHAN_MAX_AGE_HOURS)

        # Tuple: (chars, file_rel)
        orphans: list[tuple[int, str]] = []

        for md_path in vault.rglob("*.md"):
            try:
                if not md_path.is_file():
                    continue
                rel = md_path.relative_to(vault).as_posix()

                # Bucket allowlist — 00-Inbox/04-Archive/etc. quedan fuera.
                if not any(rel.startswith(p) for p in _ORPHAN_ALLOWED_PREFIXES):
                    continue

                # Sub-prefix denylist — ETL outputs auto-generados.
                if any(rel.startswith(p) for p in _ORPHAN_EXCLUDE_SUBPREFIXES):
                    continue

                # Defensive: exclusiones globales (Claude transcripts,
                # system folders, etc.) aunque estén bajo un bucket allowed.
                try:
                    if is_excluded(rel):
                        continue
                except Exception:
                    # is_excluded roto → no filtramos (mejor falso positivo
                    # que romper la signal completa).
                    pass

                # Ventana de mtime: [now - 24h, now - 2h].
                try:
                    mtime = md_path.stat().st_mtime
                except Exception:
                    continue
                mtime_dt = datetime.fromtimestamp(mtime)
                age = now - mtime_dt
                if age < min_age or age > max_age:
                    continue

                # Leer body — si falla la lectura (encoding bizarro,
                # permission), skippeamos esta nota.
                try:
                    text = md_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                chars = len(text)
                if chars < _ORPHAN_MIN_CHARS:
                    continue

                if _count_outgoing_wikilinks(text) != 0:
                    continue

                orphans.append((chars, rel))
            except Exception:
                # Nota individual malformada — saltar, seguir con las otras.
                continue

        if not orphans:
            return []

        # Más grandes primero. Python sort es estable → empates conservan
        # orden de aparición del rglob, que es determinístico por path.
        orphans.sort(key=lambda t: t[0], reverse=True)

        candidates: list = []
        for chars, file_rel in orphans[:2]:
            title = Path(file_rel).stem
            score = _score_for_chars(chars)
            message = (
                f"🔗 Nota nueva sin links: [[{title}]]\n"
                f"  Tamaño: {chars} chars, 0 wikilinks outgoing.\n"
                f"  ¿Correr `rag wikilinks suggest --path {file_rel}` "
                f"para sugerencias?"
            )
            dedup_key = f"orphan:{file_rel}"
            reason = f"chars={chars} path={file_rel}"
            candidates.append(AnticipatoryCandidate(
                kind="anticipate-orphan_surface",
                score=score,
                message=message,
                dedup_key=dedup_key,
                snooze_hours=24,
                reason=reason,
            ))

        return candidates
    except Exception:
        return []
