"""Citation validation and Rich/Obsidian rendering helpers."""
from __future__ import annotations

import os
import re
import urllib.parse

from rich.text import Text

__all__ = [
    "NOTE_LINK_RE",
    "BARE_PATH_RE",
    "URL_LINK_RE",
    "BARE_URL_RE",
    "EXT_RE",
    "CODE_FENCE_RE",
    "INLINE_CODE_RE",
    "BOLD_RE",
    "verify_citations",
    "_link_scheme",
    "_file_link_style",
    "_url_link_style",
    "to_obsidian_url",
    "convert_obsidian_links",
    "render_response",
]

# Allow single-level balanced parens inside the path (Obsidian paths often
# contain literal parens like "02-Areas/Musica/Explorando (otras)/X.md").
NOTE_LINK_RE = re.compile(r"\[([^\]]+)\]\(((?:[^()\n]|\([^()\n]*\))+?\.md)\)")
# command-r often emits just [path.md] without a markdown-link wrapper.
BARE_PATH_RE = re.compile(r"\[([^\[\]\n]+?\.md)\]")
# External web links — checked before NOTE_LINK_RE because URLs can technically
# end in .md (e.g. github raw README.md) and would otherwise be treated as a
# vault-relative path. Bare URL form catches naked https:// in prose.
URL_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
BARE_URL_RE = re.compile(r"https?://[^\s)\]\"'<>]+")
# 2026-04-23: permisivo frente a closing tags malformados — qwen2.5:7b a
# veces emite `</ext>>` (un `<` faltante) o `<</ext>` (un `>` faltante)
# en vez del canónico `<</ext>>`. El frontend web ya usaba un regex
# igual de permisivo (web/static/app.js `<{1,2}\/ext>{1,2}`). Antes,
# el CLI (`rag query --plain`) dejaba esas variantes literales en la
# terminal. Ahora matcheamos las 4 variantes y las renderemos igual.
EXT_RE = re.compile(r"<<ext>>(.*?)<{1,2}\/ext>{1,2}", re.DOTALL)
# Fenced code blocks: ```lang\nbody\n```  — lang is optional.
CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+.\-]*\n?(.*?)\n?```", re.DOTALL)
# Inline code: `literal` (no newlines, no empties). Skipped inside fences
# because fences are extracted before this pass runs.
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
# Bold: **literal** (no nested asterisks, no newlines). command-r emite
# estos marcadores seguido en listas ("**Amplificadores:**") y la terminal
# los muestra raw si no los parseamos.
BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")


def verify_citations(response_text: str, metas: list[dict]) -> list[tuple[str, str]]:
    """Check that every .md reference in the LLM response points at a path
    that was actually retrieved. Returns list of (label, path) for unverified
    citations — empty list means all citations are grounded.

    Recognises both formats:
      - [Label](path.md)     — Markdown link style (phi4/qwen)
      - [path.md]            — bracket-only style (command-r default)
    """
    retrieved = {m.get("file", "") for m in metas}
    issues: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def check(label: str, path: str) -> None:
        decoded = urllib.parse.unquote(path)
        if decoded in retrieved or path in retrieved:
            return
        key = (label, decoded)
        if key in seen:
            return
        seen.add(key)
        issues.append((label, decoded))

    # First, markdown-style links — then strip them so bracket-only scan
    # doesn't double-flag the same path.
    consumed_spans: list[tuple[int, int]] = []
    for m in NOTE_LINK_RE.finditer(response_text):
        check(m.group(1), m.group(2))
        consumed_spans.append(m.span())

    for m in BARE_PATH_RE.finditer(response_text):
        if any(s <= m.start() < e for s, e in consumed_spans):
            continue
        path = m.group(1)
        check(path, path)

    return issues


def _link_scheme() -> str:
    """Return the URI scheme used for OSC 8 note links.

    When RAG_TRACK_OPENS=1, returns "x-rag-open" so that terminals configured
    with a custom URL handler (pointing at `rag open`) record open events in
    behavior.jsonl. Users must register the handler separately — this function
    only switches the scheme; no subprocess or side-effect occurs here.
    When the env var is absent or any other value, returns "file" (default,
    zero behavior change for existing users).
    """
    return "x-rag-open" if os.environ.get("RAG_TRACK_OPENS") == "1" else "file"


def _file_link_style(path: str, base: str) -> str:
    """OSC 8 clickable style (iTerm2/Terminal.app) — opens the note in Obsidian
    when clicked on macOS (file:// URL hands off to the default .md handler).
    When RAG_TRACK_OPENS=1, uses x-rag-open:// instead so a registered URL
    handler can call `rag open <path>` and record the event in behavior.jsonl.
    """
    import rag as _rag
    full = (_rag.VAULT_PATH / path).resolve()
    scheme = _link_scheme()
    return f"{base} link {scheme}://{urllib.parse.quote(str(full))}"


def _url_link_style(url: str, base: str) -> str:
    """OSC 8 clickable style for external https?:// URLs — terminal hands off
    to the default browser when Cmd/Ctrl-clicked.
    """
    return f"{base} link {url}"


def to_obsidian_url(path: str, vault_name: str | None = None) -> str:
    """`obsidian://open?vault=<name>&file=<encoded>` for a vault-relative .md path.

    Used by `--plain` outputs that go to WhatsApp / other chat
    surfaces — those render bare URLs as clickable but ignore markdown link
    syntax. Click on mobile opens the note straight in Obsidian.
    Vault name defaults to `VAULT_PATH.name` (matches how Obsidian names vaults
    by their directory).
    """
    if vault_name is None:
        import rag as _rag
        vault_name = _rag.VAULT_PATH.name
    name = vault_name
    encoded = urllib.parse.quote(path, safe="/")
    return (
        f"obsidian://open?vault={urllib.parse.quote(name, safe='')}"
        f"&file={encoded}"
    )


def convert_obsidian_links(text: str, vault_name: str | None = None) -> str:
    """Replace `[Label](path.md)` and `[path.md]` with `obsidian://` URLs.

    For chat surfaces (WA/TG) where the markdown link syntax shows up as
    literal `[Label](path.md)` text. We emit `Label: obsidian://...` for the
    labeled form and a bare `obsidian://...` for the bracket-only form, so the
    chat client renders the URL clickable while preserving the human label.

    Order matters: NOTE_LINK_RE first, then BARE_PATH_RE on the leftovers, so
    a `[Label](foo.md)` match isn't double-processed by the bracket-only pass.
    """
    consumed: list[tuple[int, int]] = []
    parts: list[tuple[int, int, str]] = []
    for m in NOTE_LINK_RE.finditer(text):
        url = to_obsidian_url(m.group(2), vault_name)
        parts.append((m.start(), m.end(), f"{m.group(1)}: {url}"))
        consumed.append(m.span())
    for m in BARE_PATH_RE.finditer(text):
        if any(s <= m.start() < e for s, e in consumed):
            continue
        url = to_obsidian_url(m.group(1), vault_name)
        parts.append((m.start(), m.end(), url))
    if not parts:
        return text
    parts.sort()
    out: list[str] = []
    pos = 0
    for start, end, repl in parts:  # noqa: F402  (loop var, no relación con import bottom-of-module)
        if start > pos:
            out.append(text[pos:start])
        out.append(repl)
        pos = end
    if pos < len(text):
        out.append(text[pos:])
    return "".join(out)


def render_response(text: str) -> Text:
    """Render LLM response:
       - ```lang\\n...\\n``` → fence stripped, cada línea con gutter dim ("  │ ")
         y contenido bold white (se puede copiar el comando limpio)
       - `inline`         → backticks stripped, bold cyan
       - [Label](path.md) → label bold cyan, path dim cyan + clickable
       - [path.md]        → path bold magenta + clickable (command-r style)
       - <<ext>>...<</ext>> → dim yellow italic (external / inferred content)

    El pipeline extrae primero los fences (que no contienen más markdown),
    después procesa lo de afuera con ext → links → inline code.

    2026-04-29: aplica `replace_iberian_leaks` al texto ANTES de
    procesar markdown. Los LLM locales (qwen2.5:7b, command-r) a veces
    se deslizan al portugués/galego pese al system prompt en español
    rioplatense — el filter es la última barrera. Reportado con
    `rag query "Que tenes de Grecia?"` que devolvió respuesta con
    "primeira", "tua", "falam", "vistes", "primeiramente", "nos braços".
    Aplicar acá garantiza que TODOS los call sites del CLI que
    rendereen output del LLM pasen por el filter (chat, query,
    summarize, history replay).
    """
    from rag.iberian_leak_filter import replace_iberian_leaks
    text = replace_iberian_leaks(text)
    out = Text()

    def emit_plain_or_inline(seg: str, base_style: str | None = None):
        """Segmento sin links/ext/fences — maneja `inline code` y **bold**.
        Se tokeniza en spans ordenados y no solapados para que `x **y** z`
        o `**x `y` z**` se rendereen sin tocar los marcadores del otro tipo.
        """
        spans: list[tuple[int, int, str, str]] = []
        for m in INLINE_CODE_RE.finditer(seg):
            spans.append((m.start(), m.end(), "code", m.group(1)))
        code_ranges = [(s, e) for s, e, *_ in spans]
        for m in BOLD_RE.finditer(seg):
            if any(cs <= m.start() < ce for cs, ce in code_ranges):
                continue   # bold adentro de inline code es literal
            spans.append((m.start(), m.end(), "bold", m.group(1)))
        spans.sort()

        # No bold en respuesta (preferencia del usuario 2026-04-18).
        # Inline code conserva color pero sin bold; **bold** del markdown
        # se renderiza plano con el base_style actual. Las fuentes (ver
        # print_sources) siguen con bold — son UI label, no prosa.
        inline_code_style = (
            "yellow" if base_style and "yellow" in base_style
            else "cyan"
        )
        bold_style = base_style or ""

        pos = 0
        for start, end, kind, content in spans:  # noqa: F402
            if start > pos:
                out.append(seg[pos:start], style=base_style)
            if kind == "code":
                out.append(content, style=inline_code_style)
            else:   # bold
                out.append(content, style=bold_style)
            pos = end
        if pos < len(seg):
            out.append(seg[pos:], style=base_style)

    def emit_links(segment: str, base_style: str | None = None):
        """Segmento con links + inline code. Asume fences y ext ya extraídos.

        Order matters: URLs first (more specific — must contain `://`), then
        note paths. A markdown link whose target is `https://x/foo.md` would
        otherwise be misread as a vault path by NOTE_LINK_RE.
        """
        spans: list[tuple[int, int, str, str, str]] = []  # start, end, label, target, kind
        consumed: list[tuple[int, int]] = []
        for m in URL_LINK_RE.finditer(segment):
            spans.append((m.start(), m.end(), m.group(1), m.group(2), "url-md"))
            consumed.append(m.span())
        for m in NOTE_LINK_RE.finditer(segment):
            if any(s <= m.start() < e for s, e in consumed):
                continue
            spans.append((m.start(), m.end(), m.group(1), m.group(2), "note-md"))
            consumed.append(m.span())
        for m in BARE_PATH_RE.finditer(segment):
            if any(s <= m.start() < e for s, e in consumed):
                continue
            spans.append((m.start(), m.end(), m.group(1), m.group(1), "note-bare"))
            consumed.append(m.span())
        for m in BARE_URL_RE.finditer(segment):
            if any(s <= m.start() < e for s, e in consumed):
                continue
            spans.append((m.start(), m.end(), m.group(0), m.group(0), "url-bare"))
        spans.sort()

        last = 0
        # Sin bold en la respuesta: links conservan color pero plano.
        label_base = "cyan" if not base_style else "yellow"
        path_base = "cyan dim" if not base_style else "yellow dim"
        url_base = "blue" if not base_style else "yellow"
        url_dim = "blue dim" if not base_style else "yellow dim"
        for start, end, label, target, kind in spans:  # noqa: F402
            if start > last:
                emit_plain_or_inline(segment[last:start], base_style=base_style)
            if kind == "url-md":
                # Terminal-native: label clickeable (OSC 8) + URL en dim entre
                # paréntesis. Sin brackets markdown — terminal no renderea md.
                # Si label == target (LLM emitió `[url](url)`), un solo render.
                if label.strip() == target.strip():
                    out.append(target, style=_url_link_style(target, url_base))
                else:
                    out.append(label, style=_url_link_style(target, url_base))
                    out.append(" (", style="dim")
                    out.append(target, style=_url_link_style(target, url_dim))
                    out.append(")", style="dim")
            elif kind == "url-bare":
                out.append(target, style=_url_link_style(target, url_base))
            elif kind == "note-bare":
                out.append(target, style=_file_link_style(target, "magenta"))
            else:  # note-md
                if label.strip() == target.strip():
                    out.append(target, style=_file_link_style(target, label_base))
                else:
                    out.append(label, style=_file_link_style(target, label_base))
                    out.append(" (", style="dim")
                    out.append(target, style=_file_link_style(target, path_base))
                    out.append(")", style="dim")
            last = end
        if last < len(segment):
            emit_plain_or_inline(segment[last:], base_style=base_style)

    def emit_ext_and_links(segment: str):
        """Segmento sin fences — ext markers primero, links después."""
        pos = 0
        for m in EXT_RE.finditer(segment):
            if m.start() > pos:
                emit_links(segment[pos:m.start()])
            out.append("⚠ ", style="bold yellow")
            emit_links(m.group(1).strip(), base_style="yellow dim italic")
            pos = m.end()
        if pos < len(segment):
            emit_links(segment[pos:])

    def emit_code_fence(code: str):
        """Fence stripped: gutter dim + contenido bold white por línea. El
        contenido queda seleccionable/copiable sin los backticks."""
        lines = code.rstrip("\n").split("\n")
        if not lines or (len(lines) == 1 and not lines[0]):
            return
        # Blank line arriba del bloque para respiro visual.
        if len(out) and not str(out).endswith("\n"):
            out.append("\n")
        for ln in lines:
            out.append("  │ ", style="cyan dim")
            out.append(ln + "\n", style="white")

    pos = 0
    for m in CODE_FENCE_RE.finditer(text):
        if m.start() > pos:
            emit_ext_and_links(text[pos:m.start()])
        emit_code_fence(m.group(1))
        pos = m.end()
    if pos < len(text):
        emit_ext_and_links(text[pos:])
    return out
