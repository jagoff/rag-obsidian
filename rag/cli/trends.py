"""`rag trends` — análisis temático de queries recientes.

Phase 3 cont de modularización (audit perf 2026-05-08, ROI 170).

Agrupa rag_queries de los últimos N días por:
  - Folders: top-level segment de cada path retrieved.
  - Tags: tags hit en el extra_json.
  - Keywords: tokens de las queries (post stopword filter).
  - Sources: breakdown vault / whatsapp / calendar / etc.

Complementa `rag dashboard` (analytics per-query) con vista agregada.

## API

- `_extract_trends(days, top_n)` → dict con folders/tags/keywords/sources.
- `_SPANISH_STOPWORDS` — frozenset compartido.
- `trends_cli` Click command — registrado al final de
  `rag/__init__.py` via `cli.add_command(trends_cli)`.

## Lazy imports

`_ragvec_state_conn` vive en `rag/__init__.py`. Lazy adentro de
`_extract_trends`. `console` también lazy adentro del CLI.

## Re-export

`rag/__init__.py` re-exporta `_extract_trends` + `_SPANISH_STOPWORDS`
para tests que llaman `rag._extract_trends(...)`.
"""

from __future__ import annotations

import json
import re

import click

__all__ = [
    "_SPANISH_STOPWORDS",
    "_extract_trends",
    "trends_cli",
]


_SPANISH_STOPWORDS = frozenset([
    "a", "al", "algo", "algún", "alguna", "algunas", "alguno", "algunos",
    "ante", "antes", "aquí", "así", "cada", "como", "con", "contra", "cual",
    "cuando", "de", "del", "desde", "donde", "dos", "dónde", "el", "ella",
    "ellas", "ellos", "en", "entre", "era", "eran", "eres", "es", "ese",
    "eso", "esos", "esta", "estaba", "estaban", "estamos", "estan", "están",
    "estar", "estas", "este", "esto", "estos", "estoy", "fue", "fueron",
    "ha", "haber", "habia", "había", "hace", "hacer", "han", "hasta", "hay",
    "la", "las", "le", "les", "lo", "los", "más", "me", "mi", "mis", "mucho",
    "muy", "nada", "ni", "no", "nos", "o", "otra", "otras", "otro", "otros",
    "para", "pero", "poco", "por", "porque", "que", "qué", "se", "sea",
    "ser", "si", "sí", "sin", "sobre", "solo", "son", "soy", "su", "sus",
    "también", "te", "tengo", "ti", "tiene", "tienen", "todos", "tu",
    "tus", "un", "una", "unas", "uno", "unos", "vamos", "ver", "vez", "y",
    "ya", "yo",
    "the", "is", "a", "an", "of", "to", "in", "for", "on", "with", "at",
    "by", "this", "that", "it", "as", "are", "be", "from", "was", "were",
    "has", "have", "i", "you", "we", "they", "he", "she", "his", "her",
    "what", "which", "how", "when", "where",
])


def _extract_trends(days: int = 7, top_n: int = 10) -> dict:
    """Aggregate recent queries into trends by folder, tags, and keywords."""
    from collections import Counter  # noqa: PLC0415

    from rag import _ragvec_state_conn  # noqa: PLC0415

    out = {
        "days": days,
        "n_queries": 0,
        "folders": [],
        "tags": [],
        "keywords": [],
        "sources": [],
    }
    try:
        with _ragvec_state_conn() as conn:
            rows = conn.execute(
                f"SELECT q, paths_json, extra_json "
                f"FROM rag_queries "
                f"WHERE ts > datetime('now', '-{int(days)} days') "
                f"  AND q IS NOT NULL AND q != ''"
            ).fetchall()
    except Exception:
        return out
    if not rows:
        return out
    folder_counter: Counter = Counter()
    tag_counter: Counter = Counter()
    kw_counter: Counter = Counter()
    source_counter: Counter = Counter()
    kw_re = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)

    for q, paths_json, extra_json in rows:
        out["n_queries"] += 1
        try:
            paths = json.loads(paths_json) if paths_json else []
        except Exception:
            paths = []
        for p in paths[:5]:
            if not isinstance(p, str) or not p:
                continue
            if "://" in p:
                source_counter[p.split("://", 1)[0]] += 1
            else:
                source_counter["vault"] += 1
                segs = p.split("/")
                if segs:
                    folder_counter[segs[0]] += 1
        try:
            extra = json.loads(extra_json) if extra_json else {}
        except Exception:
            extra = {}
        tags_field = extra.get("tags_hit") if isinstance(extra, dict) else None
        if isinstance(tags_field, list):
            for t in tags_field:
                if isinstance(t, str) and t:
                    tag_counter[t.lower()] += 1
        if isinstance(q, str):
            for tok in kw_re.findall(q.lower()):
                if len(tok) < 3:
                    continue
                if tok in _SPANISH_STOPWORDS:
                    continue
                kw_counter[tok] += 1

    out["folders"] = folder_counter.most_common(top_n)
    out["tags"] = tag_counter.most_common(top_n)
    out["keywords"] = kw_counter.most_common(top_n)
    out["sources"] = source_counter.most_common()
    return out


@click.command("trends")
@click.option("--days", default=7, show_default=True,
              help="Ventana en días para análisis.")
@click.option("--top", default=10, show_default=True,
              help="Cuántos items top mostrar por categoría.")
@click.option("--as-json", "as_json", is_flag=True,
              help="Output JSON machine-readable.")
def trends_cli(days: int, top: int, as_json: bool):
    """Análisis temático de queries recientes.

    Agrupa rag_queries de los últimos --days días por:
      - Folders: top-level segment de cada path retrieved.
      - Tags: tags hit en el extra_json.
      - Keywords: tokens de las queries (post stopword filter).
      - Sources: breakdown vault / whatsapp / calendar / etc.

    Ejemplos:
        rag trends                         # última semana
        rag trends --days 30 --top 20      # mes, top-20
        rag trends --as-json | jq          # pipeline

    Complementa `rag dashboard` (analytics per-query) con vista
    agregada temática.
    """
    from rag import console  # noqa: PLC0415

    report = _extract_trends(days=days, top_n=top)
    if as_json:
        payload = {
            **report,
            "folders": [{"name": n, "count": c} for n, c in report["folders"]],
            "tags": [{"name": n, "count": c} for n, c in report["tags"]],
            "keywords": [{"name": n, "count": c} for n, c in report["keywords"]],
            "sources": [{"name": n, "count": c} for n, c in report["sources"]],
        }
        click.echo(json.dumps(payload))
        return

    console.print()
    console.print(f"[bold cyan]Trends[/bold cyan] "
                  f"[dim]· últimos {days} días · "
                  f"{report['n_queries']} queries[/dim]")
    console.print()

    def _render_section(title: str, items: list, unit: str = ""):
        if not items:
            console.print(f"[bold]{title}[/bold]: [dim]sin data[/dim]")
            return
        console.print(f"[bold]{title}[/bold]")
        max_count = items[0][1] if items else 1
        for name, count in items:
            bar_len = int((count / max_count) * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            console.print(f"  [cyan]{count:4d}[/cyan]  {bar}  {name}{unit}")
        console.print()

    _render_section("Folders", report["folders"])
    _render_section("Tags", report["tags"])
    _render_section("Keywords", report["keywords"])
    _render_section("Sources", report["sources"])
