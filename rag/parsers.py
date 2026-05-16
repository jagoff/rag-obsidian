"""Markdown parsers — extracted from `rag/__init__.py` (Wave 5 split, 2026-05-10).

Pure parsing primitives sobre el texto crudo de notas Obsidian / markdown.
Sin I/O, sin DB, sin LLM, sin red — solo regex + YAML. Lo único que cruza el
borde es `_silent_log` para frontmatter YAML inválido (deferred import abajo
para evitar circularidad con `rag.__init__`).

## Qué incluye

Frontmatter (YAML `---...---`):
- `parse_frontmatter(text)` → dict
- `_normalize_fm_tags(fm)` → list[str]
- `_normalize_fm_aliases(fm)` → list[str]
- `extract_frontmatter_tags(text)` → list[str] (legacy alias)
- `_format_fm_value(v)` → str | None (helper para prefix embed)

Markdown body:
- `clean_md(text)` → str (strip frontmatter + flatten wikilinks)
- `extract_wikilinks(text)` → list[str] (titles únicos, orden preservado)
- `extract_inline_tags(text)` → list[str] (`#tag` body-level, ignora code)
- `extract_tasks(body)` → dict (`open`/`done` counters + `texts` cap N)

## Constantes públicas

- `FM_SEARCHABLE_FIELDS` — allowlist de keys que rinden a `key=value` en el
  prefix del embedding.
- `WIKILINK_RE`, `_INLINE_TAG_RE`, `_FENCED_CODE_RE`, `_INLINE_CODE_RE`,
  `_TASK_RE` — regexes compartidos.
- `_FM_BLACKLIST`, `_FM_VALUE_CHARS`, `_FM_GENERIC_BUDGET` — caps para FM
  generic-pass.
- `_TASKS_TEXT_CAP`, `_TASK_TEXT_CHARS` — caps para tareas en prefix.

## Invariantes

- Silent-fail en YAML inválido: `parse_frontmatter` retorna `{}` + loguea
  via `_silent_log("parse_frontmatter_yaml", exc)` para audit (no raise —
  preserva contrato pre-2026-04-24 en call sites como `_run_index_inner`).
- CRLF + LF aceptados en regex de FM (BUG #26, 2026-04-26).
- Extract paths NO mutan input.

## Re-export

`rag/__init__.py` hace `from rag.parsers import (...)` con nombres explícitos.
Tests + call sites externos (`rag/_surface.py`, `rag/archive.py`,
`rag/conversation_distiller.py`, `web/conversation_writer.py`,
`web/atlas_dashboard.py`, `scripts/*`) siguen importando desde `rag`
gracias al re-export — no requieren cambio.
"""

from __future__ import annotations

import re

import yaml

__all__ = [
    # constantes
    "FM_SEARCHABLE_FIELDS",
    "WIKILINK_RE",
    "_INLINE_TAG_RE",
    "_FENCED_CODE_RE",
    "_INLINE_CODE_RE",
    "_TASK_RE",
    "_TASKS_TEXT_CAP",
    "_TASK_TEXT_CHARS",
    "_FM_BLACKLIST",
    "_FM_VALUE_CHARS",
    "_FM_GENERIC_BUDGET",
    # funciones públicas
    "parse_frontmatter",
    "extract_frontmatter_tags",
    "clean_md",
    "extract_wikilinks",
    "extract_inline_tags",
    "extract_tasks",
    # internas
    "_normalize_fm_tags",
    "_normalize_fm_aliases",
    "_format_fm_value",
]


def parse_frontmatter(text: str) -> dict:
    """Parse full YAML frontmatter as dict. Returns {} if none or invalid.

    Si el YAML es malformado (p.ej. `---\ntags: [unclosed\n---`), `yaml.YAMLError`
    se logea via `_silent_log` para que el indexer pueda detectar notas
    con frontmatter roto en su audit log (2026-04-24 audit: pre-fix el
    error era swallow silencioso → notas indexadas sin tags/aliases/area
    sin aviso al user. Ahora queda trace en silent_errors.jsonl para que
    se pueda correr `rag vault health` y detectar esas notas).

    Intencionalmente NO raise para mantener el contrato existente (callers
    esperan dict vacío en error), pero el log da observabilidad.
    """
    # Audit 2026-04-26 (BUG #26): aceptar CRLF + LF. Pre-fix sólo `\n`
    # → notas con line endings Windows quedaban indexadas SIN tags/
    # aliases/area silentemente.
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = yaml.safe_load(match.group(1))
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError as exc:
        # Deferred import para evitar circular import con rag.__init__.
        from rag import _silent_log  # noqa: PLC0415

        _silent_log("parse_frontmatter_yaml", exc)
        return {}


def _normalize_fm_tags(fm: dict) -> list[str]:
    """Normalize frontmatter tags to list[str].

    Obsidian accepts both `tags: a, b, c` (YAML scalar → str) and `tags: [a, b, c]`
    (list). Iterating the scalar form yields characters — handle both forms here."""
    raw = fm.get("tags")
    if not raw:
        return []
    if isinstance(raw, str):
        return [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if isinstance(raw, (list, tuple, set)):
        return [s for t in raw if (s := str(t).strip())]
    s = str(raw).strip()
    return [s] if s else []


def _normalize_fm_aliases(fm: dict) -> list[str]:
    """Normalize frontmatter ``aliases:`` to list[str].

    Obsidian accepts the list form (``aliases: [Maru, Marucha]``), the YAML
    block list (``aliases:\\n  - Maru\\n  - Marucha``, which PyYAML already
    hydrates into a Python list), and the scalar form (``aliases: Maru``).
    Generalized post 2026-04-22 — before this, the aliases parser only ran
    inside ``99-Mentions/`` dossiers, so a vault note with ``aliases:`` in
    the FM lost that signal at retrieval time.

    Treats ``aliases``, ``alias`` (singular) as equivalent keys since both
    appear in the wild."""
    raw = fm.get("aliases")
    if raw is None:
        raw = fm.get("alias")
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, tuple, set)):
        return [s for a in raw if (s := str(a).strip())]
    # Scalar string: treat as a single alias (no comma-splitting — aliases
    # commonly contain commas in "Last, First" form).
    s = str(raw).strip()
    return [s] if s else []


def extract_frontmatter_tags(text: str) -> list[str]:
    """Extract tags list from YAML frontmatter (kept for backwards compat)."""
    return _normalize_fm_tags(parse_frontmatter(text))


# Fields worth surfacing to both the embedding prefix and chunk metadata.
FM_SEARCHABLE_FIELDS = ("area", "cancion", "familia", "estado", "periodo", "created", "modified")


def clean_md(text: str) -> str:
    """Remove YAML frontmatter, convert wiki-links to plain text."""
    text = re.sub(r"^---\r?\n.*?\r?\n---\r?\n", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    return text.strip()


# Obsidian [[Link]] or [[Link#Section]] or [[Link|Alias]] — capture the target note title.
WIKILINK_RE = re.compile(r"\[\[([^\]|#^]+)(?:[#^][^\]|]*)?(?:\|[^\]]+)?\]\]")


def extract_wikilinks(text: str) -> list[str]:
    """Return unique note titles referenced by Obsidian wikilinks in `text`,
    order preserved. Run on RAW note content (before clean_md strips links).
    Frontmatter `related:` wikilinks are already included via the YAML being
    part of `text`.
    """
    seen: list[str] = []
    for m in WIKILINK_RE.finditer(text):
        title = m.group(1).strip()
        if title and title not in seen:
            seen.append(title)
    return seen


# Inline Obsidian tag `#tag` — same character class as the query-side regex
# at rag.py:3258 (infer_filters). Pre-2026-04-22 this was only matched in
# queries; the indexer ignored body-level `#tag` usage entirely, leaving
# "journal with #idea inline" invisible to tag filtering + embedding.
_INLINE_TAG_RE = re.compile(r"(?<![\w`/])#([A-Za-zÀ-ÿ0-9_][A-Za-zÀ-ÿ0-9_\-/]{1,})")
# Strip backtick-fenced code (```…```) and inline code (`…`) before scanning
# so `#include`, `#!/bin/bash`, `#define`, etc. don't pollute the tag set.
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def extract_inline_tags(text: str) -> list[str]:
    """Return Obsidian-style inline tags from markdown body, order preserved,
    deduped. Ignores tags inside fenced or inline code blocks and refuses to
    treat markdown headers (`# H1`, `## H2`) as tags (the lookbehind requires
    a non-word char *other than whitespace* before the `#`, so `\\n# H1` is
    rejected because the `H` is followed by a space — `# H` is a header, not
    `#H`).

    Obsidian tag conventions (https://help.obsidian.md/tags):
      - `#tag`, `#tag-with-dash`, `#group/subgroup`, `#a1b2`
      - Must start with a letter or underscore (digit-only → rejected)
      - Can contain letters, digits, `_`, `-`, `/`
      - Min 2 chars after the `#` (same floor as the query-side regex)
    """
    # Strip code before scanning.
    scrub = _FENCED_CODE_RE.sub("", text)
    scrub = _INLINE_CODE_RE.sub("", scrub)
    seen: list[str] = []
    for m in _INLINE_TAG_RE.finditer(scrub):
        tag = m.group(1).strip()
        # Header guard: if what follows the `#` is a single token and the
        # original match was at column 0 preceded by a newline (or start),
        # we still want it — the lookbehind already requires non-whitespace.
        # The remaining case is markdown headers: `# H1` → after `#` there's
        # a space, so the regex wouldn't match (needs `#\w`). Good.
        if not tag or tag in seen:
            continue
        # Reject digit-only tags (Obsidian rejects these too).
        stripped = tag.replace("-", "").replace("_", "").replace("/", "")
        if stripped.isdigit():
            continue
        seen.append(tag)
    return seen


# Task checkbox detection — handles bullet styles (`-`, `*`, `+`) and
# numbered lists (`1.`, `2)`), any indent depth, and both lowercase `[x]`
# + uppercase `[X]` for done. The captured group distinguishes open vs done.
_TASK_RE = re.compile(
    r"^\s*(?:[-*+]|\d+[.)])\s+\[([ xX])\]\s+(.+?)\s*$",
    re.MULTILINE,
)

# Max # of open task bodies we surface to the embedding prefix. Keeps the
# prefix from ballooning on notes like "lista de 100 pendientes".
_TASKS_TEXT_CAP = 10
# Per-task body trimmed to this many chars in the prefix — enough to be
# semantically useful, short enough to avoid domination.
_TASK_TEXT_CHARS = 80


def extract_tasks(body: str) -> dict:
    """Parse Obsidian / GFM task list items. Returns
    ``{"open": int, "done": int, "texts": [str, ...]}``.

    ``texts`` is the ordered list of the first N open-task bodies (up to
    ``_TASKS_TEXT_CAP``), trimmed per-item to ``_TASK_TEXT_CHARS``. Done
    tasks never enter ``texts`` — they're fulfilled, not pending, and
    including them would bias "qué tengo pendiente?" retrieval toward
    completed items.
    """
    open_ct = 0
    done_ct = 0
    texts: list[str] = []
    for m in _TASK_RE.finditer(body):
        mark = m.group(1)
        content = m.group(2).strip()
        if mark == " ":
            open_ct += 1
            if content and len(texts) < _TASKS_TEXT_CAP:
                texts.append(content[:_TASK_TEXT_CHARS])
        else:
            done_ct += 1
    return {"open": open_ct, "done": done_ct, "texts": texts}


# Frontmatter generalization — fields to skip from the embedded prefix
# because they're plugin/internal noise, not user intent. Extends the
# original FM_SEARCHABLE_FIELDS allowlist with a blacklist pattern:
# anything that starts with `_`, plus common noisy keys.
#
# The original `FM_SEARCHABLE_FIELDS` allowlist is kept — those fields
# render as `key=value` in the prefix for backward compat. Everything
# NOT in the allowlist / blacklist / already-handled set falls through
# the generalized path below (`_FM_GENERIC_FIELDS_IN_PREFIX`).
_FM_BLACKLIST = frozenset({
    # Obsidian internal (plugin state, positions, etc.)
    "position", "id", "uuid",
    # Obsidian core already-handled keys — extracted explicitly, no need
    # to duplicate them in the generic pass.
    "tags", "aliases", "related", "cssclass", "cssclasses", "publish",
    # Fields with structured meaning for the RAG pipeline (handled elsewhere)
    "contradicts", "archived_at", "archived_from", "archived_reason",
    # Hashes / IDs / timestamps that won't help semantic match
    "hash", "sha", "md5", "checksum",
})
# Per-value cap to stop a single field from dominating the prefix.
_FM_VALUE_CHARS = 200
# Total cap across all generic FM fields (hard budget — once we exhaust
# this the rest are dropped). Keeps a pathological YAML blob from
# exploding the embedding context.
_FM_GENERIC_BUDGET = 800


def _format_fm_value(v) -> str | None:
    """Render a frontmatter value for inclusion in the prefix.

    Rules:
      - ``None``, empty string → ``None`` (caller skips the field)
      - scalar (str/int/float/bool) → ``str(v)`` capped at ``_FM_VALUE_CHARS``
      - list/tuple → comma-joined string-rendered elements, capped
      - dict / other → ``None`` (not worth the complexity)
    """
    if v is None or v == "":
        return None
    if isinstance(v, (str, int, float, bool)):
        s = str(v).strip()
        if not s:
            return None
        return s[:_FM_VALUE_CHARS]
    if isinstance(v, (list, tuple)):
        parts: list[str] = []
        for item in v:
            if item is None:
                continue
            if isinstance(item, (str, int, float, bool)):
                s = str(item).strip()
                if s:
                    parts.append(s)
        if not parts:
            return None
        return ", ".join(parts)[:_FM_VALUE_CHARS]
    return None
