"""`rag config` — env var dashboard.

Phase 3 cont de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer el CLI command `rag config` + el dict curado `_CONFIG_VARS`
+ el scanner `_collect_env_var_names_from_source` desde
`rag/__init__.py`.

## Approach híbrido

- Dict curado `_CONFIG_VARS` con ~40 vars principales (descriptions,
  tipos, rangos sugeridos).
- Scan del code para detectar todas las env vars `RAG_*`/`OBSIDIAN_RAG_*`
  que NO estén en el curado, y mostrar sólo current value.

Output: tabla Rich default, `--as-json` machine-readable, `--only-set`
filter, `--filter PATTERN` substring search.

## Importante: `_collect_env_var_names_from_source`

Scanea `rag/__init__.py` (el módulo parent) NO el sub-módulo
extraído. El path se computa relativo al `__file__` parent.
"""

from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path

import click

__all__ = [
    "_CONFIG_VARS",
    "_collect_env_var_names_from_source",
    "config_cli",
]


_CONFIG_VARS: tuple[tuple[str, str, str, str], ...] = (
    # (name, default, type, description)
    # — Core
    ("OBSIDIAN_RAG_VAULT", "", "path",
     "Override del vault activo. Gana sobre vaults.json y _DEFAULT_VAULT."),
    ("RAG_TIMEZONE", "America/Argentina/Buenos_Aires", "tz",
     "IANA tz usado por _parse_natural_datetime para ISO con tzinfo."),

    # — LLM models
    ("RAG_LLM_KEEP_ALIVE", "-1", "duration",
     "kwarg keep_alive para chat/generate. MLX in-process: no-op pero se "
     "preserva la firma. '20m', '-1' forever, etc."),
    ("RAG_LOCAL_EMBED", "", "bool",
     "Usa SentenceTransformer in-process para query embed (default ON post-Ola 6)."),
    ("OBSIDIAN_RAG_WEB_CHAT_MODEL", "", "str",
     "Override del chat model para el web server. Default: resolve_chat_model()."),
    ("RAG_LOOKUP_MODEL", "qwen2.5:3b", "str",
     "Modelo usado en el fast-path del adaptive routing."),
    ("RAG_LOOKUP_NUM_CTX", "4096", "int",
     "Context window del fast-path LLM. Bumped desde 2048 (2026-04-22 refuse bug)."),

    # — Retrieve / ranking
    ("RAG_LOOKUP_THRESHOLD", "0.6", "float",
     "Score mínimo del top-1 para disparar el fast-path de adaptive routing."),
    ("RAG_ADAPTIVE_ROUTING", "1", "bool",
     "Activa el pipeline adaptativo — fast-path dispatch + skip reformulate para metadata intents."),
    ("RAG_EXPLORE", "", "bool",
     "ε-exploration: 10% chance de swap top-3 con rank 4-7. Generado counterfactuals para tune online."),
    ("RAG_EXPAND_MIN_TOKENS", "6", "int",
     "Queries con <N tokens skippean expand_queries() (paraphrase)."),
    ("RAG_ANAPHORA_RESOLVER", "1", "bool",
     "Quick Win #1: resolver de anáfora upstream del retrieve para "
     "queries follow-up tipo 'y en Madrid?'. Default ON; \"0\"/\"false\""
     "/\"no\" la apaga."),

    # — Feature #2 & friends
    ("RAG_SCORE_CALIBRATION", "", "bool",
     "Feature #2: aplicar calibración isotónica per-source a rerank scores."),
    ("RAG_AUTO_HARVEST_JUDGE_MODEL", "qwen2.5:7b", "str",
     "Feature #1: modelo LLM-as-judge del auto-harvest nocturno."),
    ("RAG_AUTO_HARVEST_MIN_CONF", "0.8", "float",
     "Feature #1: confidence mínima del judge para insertar row."),

    # — Feature #3
    ("RAG_LLM_INTENT", "", "bool",
     "Feature #3: fallback LLM post-regex cuando classify_intent devuelve 'semantic'."),
    ("RAG_LLM_INTENT_MODEL", "qwen2.5:3b", "str",
     "Feature #3: modelo usado en el fallback."),

    # — Feature #4
    ("RAG_AGENT_UNPRODUCTIVE_CAP", "3", "int",
     "Feature #4: cuántos tool calls improductivos consecutivos disparan el nudge en `rag do`."),

    # — Feature #5
    ("RAG_MMR_DIVERSITY", "", "bool",
     "Feature #5: activa el pass MMR post-rerank para diversidad en top-k."),
    ("RAG_MMR_LAMBDA", "0.7", "float",
     "Feature #5: lambda relevance-vs-diversity. 1.0 = pure relevance (MMR no-op), 0.0 = pure diversity."),

    # — Feature #6
    ("RAG_PPR_TOPIC", "", "bool",
     "Feature #6: Personalized PageRank topic-aware con seed = top-K rerank."),
    ("RAG_PPR_SEED_K", "5", "int",
     "Feature #6: cuántos top del rerank usar como seeds del PPR."),

    # — WhatsApp fast-path
    ("RAG_WA_FAST_PATH", "1", "bool",
     "Fast-path para WhatsApp queries (branches 1 + 2). Workaround hasta calibración activa."),
    ("RAG_WA_FAST_PATH_THRESHOLD", "0.05", "float",
     "Threshold del branch 2 del WA fast-path — detectar queries WA implícitos."),
    ("RAG_WA_SKIP_PARAPHRASE", "1", "bool",
     "Skip expand_queries() cuando caller explícita source='whatsapp' (único)."),

    # — Reranker
    ("RAG_RERANKER_IDLE_TTL", "900", "int",
     "Segundos que el cross-encoder queda resident antes del idle-unload."),
    ("RAG_RERANKER_NEVER_UNLOAD", "", "bool",
     "Pin reranker en MPS VRAM permanentemente. Seteado en web+serve plists."),
    ("RAG_RERANKER_FT_PATH", "", "path",
     "Path a un fine-tuned reranker (override del base bge-reranker-v2-m3)."),
    ("RAG_RERANKER_FT", "", "bool",
     "GC#2.C: cargar LoRA adapter desde ~/.local/share/obsidian-rag/reranker_ft/ "
     "on top del base reranker. Default OFF; fallback silencioso si peft no "
     "está instalado o el adapter no existe."),
    ("RAG_DRAFTS_FT", "", "bool",
     "GC#3: activar el modelo de drafts fine-tuned en /api/draft/preview. "
     "Lee adapter de ~/.local/share/obsidian-rag/drafts_ft/. Default OFF "
     "→ endpoint hace echo del bot_draft_baseline. NO afecta el listener "
     "TS (sigue con qwen2.5:14b en producción)."),
    ("RAG_DRAFTS_FT_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct", "str",
     "Modelo base para el fine-tune de drafts. Override solo para "
     "experimentos — el default debe matchear training time vs runtime "
     "(si entrenaste sobre Qwen2.5-7B-Instruct, runtime tiene que ser el mismo)."),

    # — Memory
    ("RAG_MEMORY_PRESSURE_DISABLE", "", "bool",
     "Desactiva el memory-pressure watchdog (Mac freeze guard)."),
    ("RAG_MEMORY_PRESSURE_THRESHOLD", "85", "int",
     "% de memoria usada que dispara el watchdog (default 85)."),
    ("RAG_MEMORY_PRESSURE_INTERVAL", "60", "int",
     "Intervalo en segundos de sampling del memory-pressure watchdog."),

    # — Semantic cache (GC#1)
    ("RAG_CACHE_ENABLED", "1", "bool",
     "Activar semantic response cache (GC#1) + typo normalization."),
    ("RAG_CACHE_TTL_DEFAULT", "86400", "int",
     "TTL default en segundos para entradas del cache (24h)."),
    ("RAG_CACHE_COSINE", "0.93", "float",
     "Threshold de cosine similarity para el cache hit."),

    # — Entity extraction
    ("RAG_ENTITY_LOOKUP", "1", "bool",
     "Activar handle_entity_lookup() para el intent entity_lookup."),
    ("RAG_EXTRACT_ENTITIES", "1", "bool",
     "Popular rag_entities + rag_entity_mentions durante indexing."),

    # — OCR
    ("RAG_OCR", "1", "bool",
     "OCR en imágenes embebidas durante indexing (Apple Vision, macOS only)."),

    # — Misc
    ("OBSIDIAN_RAG_NO_APPLE", "", "bool",
     "Desactiva integraciones Apple (Calendar, Reminders, Mail, Screen Time)."),
    ("RAG_TRACK_OPENS", "", "bool",
     "Cambia OSC-8 links de file:// a x-rag-open:// para rutear clicks via `rag open`."),
    ("RAG_DEBUG", "", "bool",
     "Emite logs extra de debug al stderr."),
    ("RAG_LOG_QUERY_ASYNC", "1", "bool",
     "Escribir rag_queries async (off-thread) para no bloquear retrieve."),
)


@functools.lru_cache(maxsize=1)
def _collect_env_var_names_from_source() -> set[str]:
    """Scan the rag package for all RAG_*/OBSIDIAN_RAG_*/OLLAMA_* env var refs.

    Returns a set of var names. Used to surface env vars in `rag config`
    output that aren't in the curated `_CONFIG_VARS` — gives them
    minimal coverage (name + current value, no description).

    Best-effort: parses via regex, not AST. Result is cached
    (lru_cache maxsize=1) — same process → same source.

    The config CLI was extracted out of `rag/__init__.py`; new env vars now
    live across `rag/cli/*`, integrations, runtime jobs, and web helpers.
    Scanning the package keeps discovered vars visible after modularization.
    """
    pattern = re.compile(
        r'(?:os\.environ\.get|os\.environ\[)\s*\(?\s*["\']'
        r'((?:OBSIDIAN_RAG|RAG|OLLAMA)_[A-Z][A-Z0-9_]*)["\']'
    )
    package_dir = Path(__file__).parent.parent
    out: set[str] = set()
    try:
        files = list(package_dir.rglob("*.py"))
    except Exception:
        files = [package_dir / "__init__.py"]
    for py in files:
        try:
            src = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        out.update(pattern.findall(src))
    return out


@click.command("config")
@click.option("--only-set", is_flag=True,
              help="Mostrar solo vars con un valor actual en el environment.")
@click.option("--filter", "filter_pattern", default=None,
              help="Substring para filtrar por nombre (case-insensitive).")
@click.option("--as-json", "as_json", is_flag=True,
              help="Output JSON machine-readable.")
def config_cli(only_set: bool, filter_pattern: str | None, as_json: bool):
    """Ver todas las env vars del sistema con sus valores actuales + defaults.

    Útil para debug ('¿qué valor tiene RAG_X ahora mismo?') y onboarding
    (ver todos los knobs expuestos). Muestra primero las ~40 vars curadas
    con descripciones, después un bloque de vars no-documentadas
    detectadas automáticamente del source (name + current value, sin
    description).
    """
    from rich.table import Table  # noqa: PLC0415

    from rag import console  # noqa: PLC0415

    filter_low = (filter_pattern or "").strip().lower()
    curated_names = {name for name, *_ in _CONFIG_VARS}
    all_names_in_src = _collect_env_var_names_from_source()
    uncurated = sorted(all_names_in_src - curated_names)

    entries: list[dict] = []
    for name, default, type_, desc in _CONFIG_VARS:
        cur = os.environ.get(name, "")
        if only_set and not cur:
            continue
        if filter_low and filter_low not in name.lower():
            continue
        entries.append({
            "name": name, "default": default, "type": type_,
            "description": desc, "current": cur,
            "is_set": bool(cur), "curated": True,
        })

    for name in uncurated:
        cur = os.environ.get(name, "")
        if only_set and not cur:
            continue
        if filter_low and filter_low not in name.lower():
            continue
        entries.append({
            "name": name, "default": "", "type": "",
            "description": "", "current": cur,
            "is_set": bool(cur), "curated": False,
        })

    if as_json:
        click.echo(json.dumps(entries, indent=2))
        return

    console.print()
    curated_entries = [e for e in entries if e["curated"]]
    uncurated_entries = [e for e in entries if not e["curated"]]

    if curated_entries:
        table = Table(
            title=f"Env vars curadas ({len(curated_entries)})",
            show_lines=False, header_style="bold",
            title_justify="left", title_style="bold cyan",
        )
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Current", style="green")
        table.add_column("Default", style="dim")
        table.add_column("Description")
        for e in curated_entries:
            cur_disp = e["current"][:40] if e["current"] else "[dim](unset)[/dim]"
            def_disp = e["default"] if e["default"] else "[dim]—[/dim]"
            table.add_row(e["name"], cur_disp, def_disp, e["description"][:80])
        console.print(table)

    if uncurated_entries:
        console.print()
        console.print(
            f"[bold yellow]No-documentadas "
            f"({len(uncurated_entries)})[/bold yellow]  "
            "[dim]— detectadas por scan del source, sin descripción curada[/dim]"
        )
        for e in uncurated_entries:
            cur = f"[green]{e['current'][:50]}[/green]" if e["current"] else "[dim](unset)[/dim]"
            console.print(f"  [cyan]{e['name']:40s}[/cyan]  {cur}")

    console.print()
    n_set = sum(1 for e in entries if e["is_set"])
    console.print(
        f"[bold]Total:[/bold] {len(entries)} vars, "
        f"[green]{n_set} con valor[/green], "
        f"[dim]{len(entries) - n_set} unset[/dim]"
    )
    console.print()
