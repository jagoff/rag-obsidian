"""Stale-source detector — surface to the user when a low-confidence
query lexically matches a past conversation whose cited sources no
longer exist in the vault.

The point: when ``rag chat`` is about to say "no encontré nada", check
if the user already asked something similar in the past and got a good
answer citing notes that have since evaporated. If so, surface a hint
pointing to either (a) the past conversation, or (b) the distilled
runbook (if `rag distill-conversations` already ran). This bridges the
gap between "knowledge was here" and "knowledge is now retrievable
again".

Cheap-on-purpose: lexical token overlap (Jaccard) on past queries,
no embeddings, no SQL. Scans ``99-AI/conversations/*.md`` frontmatters
and first user query line per turn. For 121 conversations the pass
costs ~50 ms.
"""
from __future__ import annotations

import re
from pathlib import Path

import rag as _rag

__all__ = [
    "MIN_OVERLAP_DEFAULT",
    "stale_source_hint",
    "find_stale_matches",
]

MIN_OVERLAP_DEFAULT = 0.35  # jaccard floor para considerar "queries similares"
_MAX_HINTS = 3

# Stopwords ES + EN — descarte para que el overlap no se infle con palabras
# vacías. Lista corta, no exhaustiva — sólo lo que aparece seguido en queries
# de productividad/ops.
_STOP = frozenset({
    "a", "al", "como", "con", "cual", "cuales", "de", "del", "donde", "el",
    "en", "es", "esta", "este", "esto", "la", "las", "lo", "los", "mas",
    "me", "mi", "no", "o", "para", "por", "que", "se", "si", "su", "te",
    "un", "una", "uno", "y",
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "have", "how", "i", "in", "is", "it", "of", "on", "or", "the", "to",
    "what", "where", "with",
})


def _tokenize(text: str) -> set[str]:
    """Lower, strip punct, drop stopwords, drop tokens shorter than 3 chars."""
    tokens = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9]+", text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def find_stale_matches(
    query: str,
    vault: Path,
    min_overlap: float = MIN_OVERLAP_DEFAULT,
    max_hints: int = _MAX_HINTS,
) -> list[dict]:
    """Devuelve list of dicts ``{conv_path, query_past, overlap,
    sources_missing, sources_present, distilled_to}`` ordenado por
    overlap descendente.

    Sólo incluye conversations con al menos una source missing —
    aquellas con todas las sources presentes no son "stale" (RAG normal
    debería retrieve la nota directo).
    """
    conv_dir = vault / "04-Archive/99-obsidian-system/99-AI/conversations"
    if not conv_dir.is_dir():
        return []
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    fm_re = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
    quote_re = re.compile(r"^>\s*(.+)$", re.MULTILINE)
    hits: list[dict] = []
    for f in conv_dir.rglob("*.md"):
        try:
            raw = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = fm_re.match(raw)
        if not m:
            continue
        fm_text = m.group(1)
        body = raw[m.end():]
        # Levanto sources del frontmatter (parser flat).
        sources: list[str] = []
        in_sources = False
        distilled_to: str | None = None
        for line in fm_text.splitlines():
            if line.startswith("sources:"):
                in_sources = True
                continue
            if line.startswith("distilled_to:"):
                distilled_to = line.split(":", 1)[1].strip().strip('"').strip("'")
                in_sources = False
                continue
            if in_sources:
                if line.startswith("  - ") or line.startswith("- "):
                    s = line.split("-", 1)[1].strip().strip('"').strip("'")
                    if s and not s.startswith(("whatsapp://", "http://", "https://")):
                        sources.append(s)
                elif line and not line.startswith(" "):
                    in_sources = False
        if not sources:
            continue
        missing = [s for s in sources if not (vault / s).is_file()]
        if not missing:
            continue
        # Por cada blockquote del body (cada `> ...`) computo overlap.
        # Quedate con el mejor para esta conversation.
        best_overlap = 0.0
        best_query = ""
        for qm in quote_re.finditer(body):
            past_q = qm.group(1).strip()
            if not past_q:
                continue
            ov = _jaccard(q_tokens, _tokenize(past_q))
            if ov > best_overlap:
                best_overlap = ov
                best_query = past_q
        if best_overlap < min_overlap:
            continue
        present = [s for s in sources if (vault / s).is_file()]
        hits.append({
            "conv_path": str(f.relative_to(vault)),
            "query_past": best_query,
            "overlap": best_overlap,
            "sources_missing": missing,
            "sources_present": present,
            "distilled_to": distilled_to,
        })
    hits.sort(key=lambda x: x["overlap"], reverse=True)
    return hits[:max_hints]


def stale_source_hint(
    query: str,
    vault: Path | None = None,
    min_overlap: float = MIN_OVERLAP_DEFAULT,
) -> str | None:
    """Return a markdown hint string ready to append to a chat answer when
    confidence is low, or ``None`` if no stale match.

    El formato es compacto — una línea + bullets — para no saturar la
    respuesta. Diseñado para appendear al final del answer del bot,
    NO para reemplazarlo.
    """
    vp = vault or _rag.VAULT_PATH
    hits = find_stale_matches(query, vp, min_overlap=min_overlap)
    if not hits:
        return None
    lines = [
        "",
        "> 💡 **Pregunta similar contestada antes** — las fuentes citadas "
        "ya no existen pero la respuesta está preservada:",
    ]
    for h in hits:
        if h["distilled_to"]:
            target = h["distilled_to"]
            kind = "runbook destilado"
        else:
            target = h["conv_path"]
            kind = f"conversation ({len(h['sources_missing'])} source(s) missing)"
        lines.append(
            f"> · `{target}` — {kind} "
            f"(overlap {h['overlap']:.0%}, “{h['query_past'][:60]}”)"
        )
    lines.append(
        "> Tip: corré `rag distill-conversations --apply` para canonizar "
        "estas respuestas como notas indexables."
    )
    return "\n".join(lines)
