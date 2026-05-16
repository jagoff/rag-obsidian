"""Markdown URL extraction helpers used by indexing and link search."""
from __future__ import annotations

import re

__all__ = [
    "URL_BARE_RE",
    "URL_MD_RE",
    "URL_CONTEXT_CHARS",
    "_is_media_url",
    "_grab_url_context",
    "extract_urls",
]

# ── URL EXTRACTION ────────────────────────────────────────────────────────────
# URLs in markdown notes show up two ways: as `[anchor](https://...)` (preferred,
# carries human description) and as bare `https://...`. We extract both, dedup
# by URL within a file, and capture ±URL_CONTEXT_CHARS of surrounding prose so
# the URL-finder can match queries like "donde está el link a la doc de X" by
# semantic similarity to that context, not to the URL itself.

URL_BARE_RE = re.compile(r'https?://[^\s\)\]"\'<>`]+', re.IGNORECASE)
URL_MD_RE = re.compile(r'\[([^\]]+)\]\((https?://[^\)\s]+)\)', re.IGNORECASE)
# Trailing punctuation that's almost never part of a real URL.
_URL_TRAILING_PUNCT = ".,;:!?)>\"'`"
URL_CONTEXT_CHARS = 240
# Images and media — almost always noise when the user asks for "links" or
# "documentación de X". Filtered at extraction time. Embedded image references
# in markdown also use `![alt](url)` which never goes through URL_MD_RE
# (notice the leading `!` is not consumed by `[`); but bare image URLs in prose
# are common (CDN links inside copy-pasted docs), so we filter on the path tail.
_IMAGE_EXT_RE = re.compile(
    r"\.(?:png|jpe?g|gif|svg|webp|bmp|ico|tiff?|avif|heic|mp4|webm|mov|mp3|wav|ogg|pdf)"
    r"(?:[?#].*)?$",
    re.IGNORECASE,
)


def _is_media_url(url: str) -> bool:
    return bool(_IMAGE_EXT_RE.search(url))


def _grab_url_context(text: str, start: int, end: int, window: int = URL_CONTEXT_CHARS) -> str:
    """Return up to `window` chars on each side of [start, end), single-line."""
    a = max(0, start - window)
    b = min(len(text), end + window)
    snippet = text[a:b]
    snippet = re.sub(r'\s+', ' ', snippet).strip()
    return snippet[:window * 2 + 100]


def extract_urls(text: str) -> list[dict]:
    """Pull every URL out of a note body. Returns deduped list of
    {url, anchor, line, context}.

    Markdown-style links are scanned first and consumed so a bare-URL pass
    doesn't double-flag the same address inside a `[label](url)`.
    """
    out: list[dict] = []
    seen: set[str] = set()
    consumed: list[tuple[int, int]] = []
    for m in URL_MD_RE.finditer(text):
        anchor, url = m.group(1).strip(), m.group(2).strip()
        consumed.append(m.span())
        if url in seen or _is_media_url(url):
            continue
        seen.add(url)
        out.append({
            "url": url,
            "anchor": anchor[:120],
            "line": text[:m.start()].count("\n") + 1,
            "context": _grab_url_context(text, m.start(), m.end()),
        })
    for m in URL_BARE_RE.finditer(text):
        if any(s <= m.start() < e for s, e in consumed):
            continue
        url = m.group(0).rstrip(_URL_TRAILING_PUNCT)
        if url in seen or _is_media_url(url):
            continue
        seen.add(url)
        out.append({
            "url": url,
            "anchor": "",
            "line": text[:m.start()].count("\n") + 1,
            "context": _grab_url_context(text, m.start(), m.end()),
        })
    return out


