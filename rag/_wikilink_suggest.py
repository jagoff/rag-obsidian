"""Wikilink suggestions — densify Obsidian graph by surfacing un-linked mentions.

Phase 5 cont de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer el sub-sistema de wikilink suggestions desde `rag/__init__.py`.

## Cómo funciona

Pure regex-by-title scan contra el `title_to_paths` index del corpus.
NO LLM, NO embeddings. Skips frontmatter, code blocks, existing
wikilinks, markdown links y HTML tags para no envolver texto que
el user ya linkeó por otro lado.

## API

- `find_wikilink_suggestions(col, note_path)` → list[dict] con
  candidatos (title, target, line, char_offset, context).
- `apply_wikilink_suggestions(note_path, suggestions)` →
  (count, titles_applied). Itera de highest offset a lowest para
  preservar offsets de los earlier matches.

## Lazy imports

`VAULT_PATH` y `_load_corpus` viven en `rag/__init__.py`. Lazy
adentro de cada función.

## Re-export

`rag/__init__.py` hace `from rag._wikilink_suggest import *  # noqa`.
Preserva 100% compat con `rag.find_wikilink_suggestions(...)` y
`rag.apply_wikilink_suggestions(...)`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag import SqliteVecCollection

__all__ = [
    "_WIKILINK_SKIP_PATTERNS",
    "_wikilink_skip_spans",
    "_in_skip_span",
    "find_wikilink_suggestions",
    "apply_wikilink_suggestions",
]


_WIKILINK_SKIP_PATTERNS = [
    re.compile(r"```.*?```", re.DOTALL),                  # fenced code
    re.compile(r"`[^`\n]+`"),                              # inline code
    re.compile(r"!?\[\[[^\]]+\]\]"),                       # existing wikilinks (incl. ![[embed]])
    re.compile(r"\[[^\]\n]+\]\([^\)\n]+\)"),               # markdown links
    re.compile(r"<!--.*?-->", re.DOTALL),                  # HTML comments (audit R2-Wikilinks #2)
    re.compile(r"<[^>\n]+>"),                              # HTML tags
]


def _wikilink_skip_spans(text: str) -> list[tuple[int, int]]:
    """Build the list of (start, end) char ranges to ignore when proposing
    wikilinks. Includes frontmatter at top, code blocks, existing wikilinks,
    markdown links and HTML tags.
    """
    spans: list[tuple[int, int]] = []
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            spans.append((0, end + 4))
    for pat in _WIKILINK_SKIP_PATTERNS:
        for m in pat.finditer(text):
            spans.append(m.span())
    return spans


def _in_skip_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def find_wikilink_suggestions(
    col: "SqliteVecCollection",
    note_path: str,
    min_title_len: int = 4,
    max_per_note: int = 30,
) -> list[dict]:
    """Return wikilink suggestions for one note.

    For each unique title in the corpus whose body matches inside `note_path`
    AND that match isn't already covered by a wikilink/markdown link/code/HTML
    span, propose `[[Title]]`. Case-sensitive, word-boundary anchored.

    Returns [{title, target, line, char_offset, context}, ...].

    Heuristics:
     - `min_title_len`: skip very short titles (high collision risk; "TDD",
       "AI", "X" trigger everywhere).
     - Ambiguous titles (same string maps to multiple paths) are skipped —
       can't know which to link to without user input.
     - Self-links suppressed (target == note_path).
     - Only the FIRST occurrence per title in the note is proposed (Obsidian
       convention: one wikilink per page is enough for graph purposes).
    """
    from rag import VAULT_PATH, _load_corpus  # noqa: PLC0415

    full = (VAULT_PATH / note_path).resolve()
    try:
        full.relative_to(VAULT_PATH.resolve())
    except ValueError:
        return []
    if not full.is_file():
        return []
    raw = full.read_text(encoding="utf-8", errors="ignore")
    skip_spans = _wikilink_skip_spans(raw)

    c = _load_corpus(col)
    title_to_paths = c["title_to_paths"]
    own_title = full.stem

    suggestions: list[dict] = []
    seen_titles: set[str] = set()
    # Sort longest first — if "Claude Code" is a title and so is "Claude",
    # prefer the longer phrase so we don't double-suggest overlapping spans.
    titles_sorted = sorted(title_to_paths.items(), key=lambda kv: -len(kv[0]))
    for title, paths in titles_sorted:
        if len(title) < min_title_len or title in seen_titles:
            continue
        if title == own_title:
            continue
        if len(paths) != 1:
            continue  # ambiguous — skip
        target = next(iter(paths))
        if target == note_path:
            continue
        try:
            pat = re.compile(rf"\b{re.escape(title)}\b")
        except re.error:
            continue
        for m in pat.finditer(raw):
            if _in_skip_span(m.start(), skip_spans):
                continue
            line = raw[:m.start()].count("\n") + 1
            ctx = raw[max(0, m.start() - 60):min(len(raw), m.end() + 60)]
            suggestions.append({
                "title": title,
                "target": target,
                "line": line,
                "char_offset": m.start(),
                "context": re.sub(r"\s+", " ", ctx).strip(),
            })
            seen_titles.add(title)
            break  # one per title per note
        if len(suggestions) >= max_per_note:
            break
    suggestions.sort(key=lambda s: s["char_offset"])
    return suggestions


def apply_wikilink_suggestions(note_path: str, suggestions: list[dict]) -> tuple[int, list[str]]:
    """Wrap each proposed mention with `[[ ]]`. Returns (count, titles_applied).
    Iterates from highest offset to lowest so earlier offsets stay valid.
    Defensive: re-checks the literal text at offset before substituting
    so a stale suggestion (file edited mid-flight) is silently skipped.
    """
    from rag import VAULT_PATH  # noqa: PLC0415

    full = (VAULT_PATH / note_path).resolve()
    try:
        full.relative_to(VAULT_PATH.resolve())
    except ValueError:
        return 0, []
    if not full.is_file() or not suggestions:
        return 0, []
    raw = full.read_text(encoding="utf-8", errors="ignore")
    by_offset = sorted(suggestions, key=lambda s: s["char_offset"], reverse=True)
    applied = 0
    applied_titles: list[str] = []
    for s in by_offset:
        start = s["char_offset"]
        title = s["title"]
        end = start + len(title)
        if raw[start:end] != title:
            continue
        raw = raw[:start] + f"[[{title}]]" + raw[end:]
        applied += 1
        applied_titles.append(title)
    if applied:
        full.write_text(raw, encoding="utf-8")
    return applied, applied_titles
