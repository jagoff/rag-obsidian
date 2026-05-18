"""Temporal intent detector — Spanish-first regex.

Phase 5 cont de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`).

## Qué hace

"¿qué escribí la última semana sobre RAG?", "notas del mes pasado",
"ideas de enero". Tight Spanish-first patterns: infer
`(start_ts, end_ts)` + a cleaned query stripped of the temporal
phrase so the embedder doesn't waste signal on tokens we already
turned into metadata.

## API

- `detect_temporal_intent(text)` → `((start_ts, end_ts), cleaned)` o `(None, text)`.
- `parse_since(value)` → `start_ts` para `--since` flag (acepta
  `'7d'`/`'2w'`/`'3m'`/`'1y'` relativos o ISO date).

## Lazy imports

Solo `click.BadParameter` (3rd party) — top-level. El módulo es
self-contained: no necesita `rag/__init__.py` deps. Por eso queda
fácil de testear standalone.

## Re-export

`rag/__init__.py` hace `from rag._temporal_intent import *  # noqa`.
Preserva 100% compat con `rag.detect_temporal_intent`,
`rag.parse_since`, `rag._SPANISH_MONTHS`, etc.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta

import click

__all__ = [
    "_SPANISH_MONTHS",
    "_UNIT_DAYS",
    "_TEMPORAL_PATTERNS",
    "_SINCE_REL_RE",
    "_now_dt",
    "_range_last_n_days",
    "_range_this_period",
    "_range_month",
    "detect_temporal_intent",
    "parse_since",
]


_SPANISH_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
_UNIT_DAYS = {
    "dia": 1, "día": 1, "dias": 1, "días": 1,
    "semana": 7, "semanas": 7,
    "mes": 30, "meses": 30,
    "año": 365, "ano": 365, "años": 365, "anos": 365,
}

# Each pattern captures the span to strip from the query via named group `span`.
_TEMPORAL_PATTERNS = [
    # "últimos N días/semanas/meses/años"
    re.compile(
        r"(?P<span>(?:en\s+)?(?:los\s+|las\s+)?[úu]ltim[oa]s?\s+(?P<n>\d+)\s+"
        r"(?P<unit>d[íi]as?|semanas?|meses|a[ñn]os?))",
        re.IGNORECASE,
    ),
    # "hace N días/semanas/meses/años"
    re.compile(
        r"(?P<span>hace\s+(?P<n>\d+)\s+(?P<unit>d[íi]as?|semanas?|meses|a[ñn]os?))",
        re.IGNORECASE,
    ),
    # "última semana/mes/año" (implicit N=1)
    re.compile(
        r"(?P<span>(?:de\s+la\s+|de\s+el\s+|del\s+|en\s+la\s+|en\s+el\s+|la\s+|el\s+)?"
        r"[úu]ltim[oa]\s+(?P<unit>semana|mes|a[ñn]o))",
        re.IGNORECASE,
    ),
    # "semana/mes/año pasad[oa]" (implicit N=1)
    re.compile(
        r"(?P<span>(?:de\s+la\s+|de\s+el\s+|del\s+|en\s+la\s+|en\s+el\s+|la\s+|el\s+)?"
        r"(?P<unit>semana|mes|a[ñn]o)\s+pasad[oa])",
        re.IGNORECASE,
    ),
    # "esta semana/este mes/este año" — start of current calendar period
    re.compile(
        r"(?P<span>(?:de\s+|en\s+)?est[ae]\s+(?P<unit>semana|mes|a[ñn]o))",
        re.IGNORECASE,
    ),
    # "ayer" / "hoy"
    re.compile(r"(?P<span>\b(?P<unit>ayer|hoy)\b)", re.IGNORECASE),
    # Spanish month name — "de enero", "en marzo", bare "enero"
    re.compile(
        r"(?P<span>(?:de\s+|en\s+)?\b(?P<month>enero|febrero|marzo|abril|mayo|junio|"
        r"julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\b)",
        re.IGNORECASE,
    ),
]


def _now_dt() -> datetime:
    """Indirection so tests can freeze time by monkeypatching."""
    return datetime.now()


def _current_now() -> datetime:
    """Root-package override for back-compat with `monkeypatch(rag._now_dt)`.

    `rag.__init__` re-exports this module, so historical tests patch the root
    binding. Internal helpers must honor that patched binding.
    """
    root = sys.modules.get("rag")
    fn = getattr(root, "_now_dt", None) if root is not None else None
    if callable(fn) and fn is not _now_dt:
        return fn()
    return _now_dt()


def _range_last_n_days(n: int) -> tuple[float, float]:
    now = _current_now()
    start = now.timestamp() - n * 86400
    return start, now.timestamp()


def _range_this_period(unit: str) -> tuple[float, float]:
    """Start-of-current calendar period → now. `unit` ∈ semana|mes|año."""
    now = _current_now()
    if unit.startswith("sem"):
        # ISO week: Monday = 0. Start of the current week at 00:00.
        start_dt = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    elif unit.startswith("mes"):
        start_dt = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:  # año
        start_dt = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return start_dt.timestamp(), now.timestamp()


def _range_month(month: int) -> tuple[float, float]:
    """Absolute Spanish month name. If the month is in the future relative to
    today, assume the user meant last year's occurrence (covers "qué escribí
    en noviembre" asked in March).
    """
    now = _current_now()
    year = now.year if month <= now.month else now.year - 1
    start_dt = datetime(year, month, 1)
    if month == 12:
        end_dt = datetime(year + 1, 1, 1)
    else:
        end_dt = datetime(year, month + 1, 1)
    return start_dt.timestamp(), end_dt.timestamp()


def detect_temporal_intent(text: str) -> tuple[tuple[float, float] | None, str]:
    """Return ((start_ts, end_ts), cleaned_query) or (None, text).

    Spanish-first. Tight regex — must not steal "del año de la pera" and
    similar idioms. When a pattern matches, strips the phrase from the
    returned query so the embedder sees only the semantic residue.
    """
    for pat in _TEMPORAL_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        groups = m.groupdict()
        rng: tuple[float, float] | None = None
        unit = groups.get("unit")
        if "n" in groups and groups["n"]:
            n = int(groups["n"])
            days = _UNIT_DAYS.get(unit.lower())
            if days:
                rng = _range_last_n_days(n * days)
        elif unit and unit.lower() in ("ayer",):
            now = _current_now()
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = now.replace(hour=0, minute=0, second=0, microsecond=0)
            rng = (start.timestamp(), end.timestamp())
        elif unit and unit.lower() in ("hoy",):
            now = _current_now()
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            rng = (start.timestamp(), now.timestamp())
        elif groups.get("month"):
            rng = _range_month(_SPANISH_MONTHS[groups["month"].lower()])
        elif unit:
            # última/pasad[oa] / esta — implicit N=1.
            norm = unit.lower()
            low_text = m.group("span").lower()
            if "est" in low_text:  # "esta semana" / "este mes" / "este año"
                rng = _range_this_period(norm)
            else:
                days = 7 if norm.startswith("sem") else (30 if norm.startswith("mes") else 365)
                rng = _range_last_n_days(days)
        if rng is None:
            continue
        cleaned = (text[:m.start("span")] + text[m.end("span"):]).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return rng, cleaned or text
    return None, text


_SINCE_REL_RE = re.compile(r"^(\d+)\s*([dwmy])$", re.IGNORECASE)


def parse_since(value: str) -> float:
    """Parse `--since` flag → start_ts. Accepts:
      - '7d', '2w', '3m', '1y' (relative)
      - ISO date: '2026-01-01' or '2026-01-01T09:30:00'
    Raises click.BadParameter with a helpful message on invalid input.
    """
    s = value.strip()
    m = _SINCE_REL_RE.match(s)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        days = {"d": 1, "w": 7, "m": 30, "y": 365}[unit]
        return _current_now().timestamp() - n * days * 86400
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        raise click.BadParameter(
            f"Formato inválido: {value!r}. Usá '7d', '2w', '3m', '1y' o una fecha ISO (YYYY-MM-DD)."
        )
