#!/usr/bin/env python3
"""Obsidian RAG — semantic chunking, title prefix, HyDE, hybrid BM25+semantic, reranking."""

import hashlib
import json
import os
import re
import threading
import time
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path

# Reranker is cached locally — go offline to skip HF Hub network check + warnings.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import click
import chromadb
import ollama
import yaml
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import track
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import urllib.parse

NOTE_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\n]+?\.md)\)")
# command-r often emits just [path.md] without a markdown-link wrapper.
BARE_PATH_RE = re.compile(r"\[([^\[\]\n]+?\.md)\]")
EXT_RE = re.compile(r"<<ext>>(.*?)<<\/ext>>", re.DOTALL)


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


def render_response(text: str) -> Text:
    """Render LLM response with:
       - [Note](path.md) → cyan bold + underlined path
       - <<ext>>...<</ext>> → dim yellow (info NOT from notes: filler, intros, general knowledge)
    """
    out = Text()

    def append_with_links(segment: str, base_style: str | None = None):
        last = 0
        for m in NOTE_LINK_RE.finditer(segment):
            if m.start() > last:
                out.append(segment[last:m.start()], style=base_style)
            out.append(m.group(1), style="bold cyan" if not base_style else "bold yellow")
            out.append(" (", style="dim")
            out.append(
                m.group(2),
                style="cyan underline" if not base_style else "yellow underline",
            )
            out.append(")", style="dim")
            last = m.end()
        if last < len(segment):
            out.append(segment[last:], style=base_style)

    pos = 0
    for m in EXT_RE.finditer(text):
        if m.start() > pos:
            append_with_links(text[pos:m.start()])
        # Marker for external/inferred content, rendered dim yellow with a leading icon.
        out.append("⚠ ", style="bold yellow")
        append_with_links(m.group(1).strip(), base_style="yellow dim italic")
        pos = m.end()
    if pos < len(text):
        append_with_links(text[pos:])
    return out

VAULT_PATH = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"
DB_PATH = Path.home() / ".local/share/obsidian-rag/chroma"
LOG_PATH = Path.home() / ".local/share/obsidian-rag/queries.jsonl"


def log_query_event(event: dict) -> None:
    """Append a JSONL event to the local query log (best-effort, never raises)."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        event = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass
EMBED_MODEL = "bge-m3"  # multilingual (ES/EN), 1024-dim
# Chat model preference: first available wins. command-r is RAG-trained +
# citation-native, ideal for this use case. Fallbacks cover slower pulls.
CHAT_MODEL_PREFERENCE = ("command-r:latest", "qwen2.5:14b", "phi4:latest")
HELPER_MODEL = "qwen2.5:3b"      # fast, for internal rewrites (multi-query, HyDE, reformulate)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # cross-encoder, multilingual, MPS-friendly
COLLECTION_NAME = "obsidian_notes_v6"  # v6: larger chunks (150-800), parent-in-metadata

# Deterministic decoding — this is a retrieval tool, not creative writing.
CHAT_OPTIONS = {"temperature": 0, "top_p": 1, "seed": 42}
HELPER_OPTIONS = {"temperature": 0, "top_p": 1, "seed": 42}


_CHAT_MODEL_RESOLVED: str | None = None


def resolve_chat_model() -> str:
    """Pick the first available model from CHAT_MODEL_PREFERENCE.

    Cached per-process so the Ollama `list` call runs at most once. If none
    are installed, raise a clear error pointing at `ollama pull`.
    """
    global _CHAT_MODEL_RESOLVED
    if _CHAT_MODEL_RESOLVED is not None:
        return _CHAT_MODEL_RESOLVED
    try:
        available = {m.model for m in ollama.list().models}
    except Exception:
        available = set()
    for candidate in CHAT_MODEL_PREFERENCE:
        if candidate in available:
            _CHAT_MODEL_RESOLVED = candidate
            return candidate
    raise RuntimeError(
        f"Ningún modelo de CHAT_MODEL_PREFERENCE instalado: {CHAT_MODEL_PREFERENCE}. "
        f"Instalá uno con: ollama pull {CHAT_MODEL_PREFERENCE[0].split(':')[0]}"
    )

MIN_CHUNK = 150    # chars — merge smaller chunks with neighbor
MAX_CHUNK = 800    # chars — bge-m3 accepts ~2048 tokens; 800 chars ≈ 200 tokens
                   # sweet spot: enough context per chunk, prefix doesn't dominate

# Folder prefixes to exclude from indexing entirely (deleted / system folders).
EXCLUDED_PREFIXES = (".trash/", ".obsidian/")


def is_excluded(rel_path: str) -> bool:
    return any(rel_path.startswith(p) for p in EXCLUDED_PREFIXES)
RETRIEVE_K = 20    # candidates from semantic + BM25 each
RERANK_TOP = 5     # final chunks after reranking
# Reranker confidence threshold. bge-reranker-v2-m3 returns sigmoid-ish
# scores for this corpus: irrelevant queries sit around 0.005-0.015, borderline
# around 0.02-0.10, clearly relevant > 0.2. Gate the LLM below this threshold
# to skip hallucinated answers from unrelated chunks.
CONFIDENCE_RERANK_MIN = 0.015

console = Console()


# ── DB ────────────────────────────────────────────────────────────────────────

def get_db() -> chromadb.Collection:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── TEXT PROCESSING ───────────────────────────────────────────────────────────

def parse_frontmatter(text: str) -> dict:
    """Parse full YAML frontmatter as dict. Returns {} if none or invalid."""
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = yaml.safe_load(match.group(1))
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        return {}


def extract_frontmatter_tags(text: str) -> list[str]:
    """Extract tags list from YAML frontmatter (kept for backwards compat)."""
    fm = parse_frontmatter(text)
    tags = fm.get("tags") or []
    return [str(t) for t in tags if t]


# Fields worth surfacing to both the embedding prefix and chunk metadata.
FM_SEARCHABLE_FIELDS = ("area", "cancion", "familia", "estado", "periodo", "created", "modified")


def clean_md(text: str) -> str:
    """Remove YAML frontmatter, convert wiki-links to plain text."""
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    return text.strip()


def build_prefix(note_title: str, folder: str, tags: list[str], fm: dict) -> str:
    """Build the embedding prefix combining path, frontmatter fields, and tags."""
    fm_bits = []
    for field in FM_SEARCHABLE_FIELDS:
        v = fm.get(field)
        if v is None or v == "":
            continue
        fm_bits.append(f"{field}={v}")
    # related links add useful semantic signal (other notes this one points to)
    related = fm.get("related") or []
    if isinstance(related, list) and related:
        # Extract titles from [[wikilinks]] or bare strings
        rel_titles = []
        for r in related[:8]:
            if not isinstance(r, str):
                continue
            m = re.search(r"\[\[([^\]|]+)", r)
            rel_titles.append(m.group(1) if m else r)
        if rel_titles:
            fm_bits.append("related=" + ", ".join(rel_titles))

    header = f"[{folder} | {note_title}"
    if fm_bits:
        header += " | " + " ".join(fm_bits)
    header += "]"
    if tags:
        header += " " + " ".join(f"#{t}" for t in tags)
    return header


PARENT_MAX_CHARS = 1200
_HEADER_RE = re.compile(r"^#{1,6} ", re.MULTILINE)


def _compute_parent(body: str, chunk_start: int, chunk_end: int) -> str:
    """Return the surrounding section (between Markdown headers) as parent
    context, capped at PARENT_MAX_CHARS centered on the chunk.
    """
    # Section start: last header at or before chunk_start.
    section_start = 0
    for m in _HEADER_RE.finditer(body):
        if m.start() > chunk_start:
            break
        section_start = m.start()

    # Section end: next header strictly after section_start.
    m = _HEADER_RE.search(body, section_start + 1)
    section_end = m.start() if m else len(body)

    if section_end - section_start > PARENT_MAX_CHARS:
        half = PARENT_MAX_CHARS // 2
        center = (chunk_start + chunk_end) // 2
        section_start = max(section_start, center - half)
        section_end = min(section_end, center + half)

    return body[section_start:section_end].strip()


def semantic_chunks(
    text: str, note_title: str, folder: str, tags: list[str], fm: dict
) -> list[tuple[str, str, str]]:
    """
    Split text into semantic chunks.
    Returns list of (embed_text, display_text, parent_text).
    Short notes (< MIN_CHUNK) emit one chunk with just the prefix + body so they
    are still retrievable.
    """
    prefix = build_prefix(note_title, folder, tags, fm)

    stripped = text.strip()
    if not stripped:
        return []

    # Short note: one chunk, parent == chunk (nothing larger to expand into).
    if len(stripped) < MIN_CHUNK:
        return [(f"{prefix}\n{stripped}", stripped, stripped)]

    # Split on headers and blank lines
    pieces = re.split(r"(?=\n#{1,3} )|\n\n+", stripped)
    pieces = [p.strip() for p in pieces if p.strip()]

    # Merge chunks that are too small into the previous one
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) < MIN_CHUNK:
            merged[-1] = merged[-1] + "\n\n" + piece
        else:
            merged.append(piece)

    # Hard-split chunks that exceed MAX_CHUNK at sentence boundaries
    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= MAX_CHUNK:
            final.append(chunk)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            buf = ""
            for sent in sentences:
                if len(buf) + len(sent) + 1 > MAX_CHUNK and buf:
                    final.append(buf.strip())
                    buf = sent
                else:
                    buf = (buf + " " + sent).strip() if buf else sent
            if buf:
                final.append(buf.strip())

    # Locate each chunk sequentially in the body so we can compute parent slices
    # deterministically (O(1) slice at query time, no needle-find).
    result = []
    cursor = 0
    for chunk in final:
        if not chunk:
            continue
        # Sequential search — monotonic cursor avoids picking duplicate occurrences.
        idx = stripped.find(chunk, cursor)
        if idx < 0:
            # Fallback for whitespace drift (merge may have inserted extra \n\n).
            idx = stripped.find(chunk[:64], cursor)
            if idx < 0:
                idx = cursor
        end = idx + len(chunk)
        cursor = end
        parent = _compute_parent(stripped, idx, end)
        embed_text = f"{prefix}\n{chunk}"
        result.append((embed_text, chunk, parent))

    if not result:
        body = stripped[:MAX_CHUNK]
        result.append((f"{prefix}\n{body}", body, body))
    return result


# ── EMBEDDING ─────────────────────────────────────────────────────────────────

def embed(texts: list[str]) -> list[list[float]]:
    resp = ollama.embed(model=EMBED_MODEL, input=texts)
    return resp.embeddings


def hyde_embed(question: str) -> list[float]:
    """Generate a short hypothetical note sentence and embed it (1 sentence = fast)."""
    prompt = (
        f'Write ONE sentence as if from personal notes that directly answers: "{question}"\n\nSentence:'
    )
    resp = ollama.chat(
        model=HELPER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=HELPER_OPTIONS,
    )
    return embed([resp.message.content.strip()])[0]


# ── SEARCH ────────────────────────────────────────────────────────────────────

_corpus_cache: dict | None = None


def _tokenize(text: str) -> list[str]:
    # NFD + drop combining marks → "canción" matches "cancion", "día" matches "dia".
    normalized = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    return re.findall(r"\w+", stripped)


def _load_corpus(col: chromadb.Collection) -> dict:
    """Load and cache the full corpus + BM25 index + vocabulary.

    Invalidated when collection size changes. Chunk updates that keep the same
    count won't invalidate — acceptable for chat-mode within a process; fresh
    CLI invocations rebuild from disk anyway.
    """
    global _corpus_cache
    n = col.count()
    if _corpus_cache is not None and _corpus_cache["count"] == n:
        return _corpus_cache

    data = col.get(include=["documents", "metadatas"])
    docs, ids, metas = data["documents"], data["ids"], data["metadatas"]

    bm25 = None
    if docs:
        bm25_texts = [
            f"{m.get('note', '')} {m.get('file', '')} {d}".lower()
            for d, m in zip(docs, metas)
        ]
        tokenized = [_tokenize(t) for t in bm25_texts]
        bm25 = BM25Okapi(tokenized)

    tags: set[str] = set()
    folders: set[str] = set()
    for m in metas:
        for t in (m.get("tags") or "").split(","):
            t = t.strip()
            if t:
                tags.add(t)
        f = m.get("folder")
        if f:
            folders.add(f)

    _corpus_cache = {
        "count": n, "ids": ids, "docs": docs, "metas": metas,
        "bm25": bm25, "tags": tags, "folders": folders,
    }
    return _corpus_cache


def _invalidate_corpus_cache() -> None:
    global _corpus_cache
    _corpus_cache = None


def bm25_search(col: chromadb.Collection, query: str, k: int, folder: str | None, tag: str | None = None) -> list[str]:
    """Keyword search using BM25 over the full collection."""
    c = _load_corpus(col)
    if c["bm25"] is None:
        return []

    scores = c["bm25"].get_scores(_tokenize(query))
    ids, metas = c["ids"], c["metas"]

    if folder or tag:
        valid = [
            i for i, m in enumerate(metas)
            if (not folder or folder in m.get("file", ""))
            and (not tag or tag in m.get("tags", ""))
        ]
        if not valid:
            return []
        top = sorted(valid, key=lambda i: scores[i], reverse=True)[:k]
    else:
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [ids[i] for i in top]


def rrf_merge(sem_ids: list[str], bm25_ids: list[str], rrf_k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion — combines semantic and BM25 rankings."""
    scores: dict[str, float] = {}
    for rank, id_ in enumerate(sem_ids):
        scores[id_] = scores.get(id_, 0) + 1 / (rrf_k + rank + 1)
    for rank, id_ in enumerate(bm25_ids):
        scores[id_] = scores.get(id_, 0) + 1 / (rrf_k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
    return dot / (norm + 1e-8)


_reranker = None


def get_reranker():
    """Lazy-load cross-encoder reranker (downloaded on first use, ~600MB)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        # Use MPS on Apple Silicon, CPU elsewhere; sentence-transformers auto-detects.
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker


def cross_encoder_rerank(
    query: str,
    candidates: list[tuple[str, dict, str]],
    top_k: int,
) -> tuple[list[tuple[str, dict, str]], float]:
    """Rerank with a cross-encoder — scores each (query, chunk) pair jointly.
    Returns (top_k candidates, top_score). top_score is the logit of the best pair —
    use it to judge confidence (> 0 relevant, < 0 irrelevant).
    """
    if not candidates:
        return [], float("-inf")
    reranker = get_reranker()
    pairs = [(query, c[0]) for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    scored = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
    top_score = float(scored[0][1])
    return [c for c, _ in scored[:top_k]], top_score


# ── RETRIEVAL PIPELINE ────────────────────────────────────────────────────────

def expand_queries(question: str) -> list[str]:
    """Generate 2 paraphrases for multi-query retrieval. Returns [original, p1, p2]."""
    prompt = (
        "Reformulá esta pregunta de DOS maneras distintas — distintas palabras clave, "
        "mismo sentido. Devolvé SOLO las dos reformulaciones, una por línea. "
        "Sin numerar, sin explicar.\n\n"
        f"Pregunta: {question}"
    )
    resp = ollama.chat(
        model=HELPER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=HELPER_OPTIONS,
    )
    lines = [l.strip(" -*·") for l in resp.message.content.splitlines() if l.strip()]
    variants = [question] + [l for l in lines if l != question][:2]
    return variants


_INTENT_COUNT_RE = re.compile(r"\b(cu[aá]nt[aos]s?|how many)\b", re.IGNORECASE)
_INTENT_LIST_RE = re.compile(
    r"\b(list[aá](?:me|r)?|dame\s+(?:todas|las\s+notas)|mostr[aá](?:me|r)?\s+notas|qu[eé]\s+notas\s+tengo)\b",
    re.IGNORECASE,
)
_INTENT_RECENT_RE = re.compile(
    r"\b(recientes?|modificad[aos]{1,2}|[uú]ltim[aos]{1,2}\s+notas?|esta\s+semana|este\s+mes|hoy)\b",
    re.IGNORECASE,
)


def classify_intent(
    question: str, known_tags: set[str], known_folders: set[str]
) -> tuple[str, dict]:
    """Detect query intent. Returns (intent, params) where intent is one of
    'count', 'list', 'recent', 'semantic' and params carries extracted filters.

    Intent detection is deliberately strict (regex over known verbs) so natural
    questions stay on the semantic path. Ambiguous queries fall through to
    'semantic'.
    """
    params: dict = {}

    # Tag: explicit #tag only
    for m in re.finditer(r"#([a-z0-9][a-z0-9\-]+)", question.lower()):
        if m.group(1) in {t.lower() for t in known_tags}:
            params["tag"] = m.group(1)
            break

    # Folder: leaf-name literal match (>=5 chars to avoid false positives)
    low = f" {question.lower()} "
    for f in sorted(known_folders, key=len, reverse=True):
        if not f or f == ".":
            continue
        leaf = f.split("/")[-1].lower()
        if len(leaf) < 5:
            continue
        if f" {leaf} " in low:
            params["folder"] = f
            break

    if _INTENT_COUNT_RE.search(question):
        return "count", params
    if _INTENT_LIST_RE.search(question):
        return "list", params
    if _INTENT_RECENT_RE.search(question):
        return "recent", params
    return "semantic", {}


def _filter_files(metas: list[dict], tag: str | None, folder: str | None) -> list[dict]:
    """Return a deduped list of unique-file meta dicts matching filters.
    Always skips excluded prefixes (`.trash/`, `.obsidian/`).
    """
    seen: dict[str, dict] = {}
    for m in metas:
        f = m.get("file", "")
        if f in seen or is_excluded(f):
            continue
        if tag:
            tags_list = [t.strip() for t in (m.get("tags") or "").split(",") if t.strip()]
            if tag not in tags_list:
                continue
        if folder and folder not in f:
            continue
        seen[f] = m
    return list(seen.values())


def handle_count(col: chromadb.Collection, params: dict) -> tuple[int, list[dict]]:
    """COUNT intent — returns (count, matching_files_meta)."""
    c = _load_corpus(col)
    files = _filter_files(c["metas"], params.get("tag"), params.get("folder"))
    return len(files), files


def handle_list(col: chromadb.Collection, params: dict, limit: int = 50) -> list[dict]:
    """LIST intent — returns up to `limit` files matching filters."""
    c = _load_corpus(col)
    files = _filter_files(c["metas"], params.get("tag"), params.get("folder"))
    files.sort(key=lambda m: m.get("file", ""))
    return files[:limit]


def handle_recent(col: chromadb.Collection, params: dict, limit: int = 20) -> list[dict]:
    """RECENT intent — returns files sorted by `modified` frontmatter desc."""
    c = _load_corpus(col)
    files = _filter_files(c["metas"], params.get("tag"), params.get("folder"))
    # Sort by modified date (ISO string sorts lexicographically correctly).
    files.sort(key=lambda m: m.get("modified") or m.get("created") or "", reverse=True)
    return files[:limit]


def render_file_list(title: str, files: list[dict]) -> None:
    """Print a compact table of note name + path + tags."""
    tbl = Table(title=title, show_header=False, box=None, pad_edge=False, padding=(0, 1))
    tbl.add_column(style="bold cyan")
    tbl.add_column(style="cyan dim")
    tbl.add_column(style="dim yellow")
    for m in files:
        tbl.add_row(
            m.get("note", ""),
            m.get("file", ""),
            "#" + " #".join(t.strip() for t in (m.get("tags") or "").split(",") if t.strip()) or "",
        )
    console.print(tbl)


def infer_filters(
    question: str, known_tags: set[str], known_folders: set[str]
) -> tuple[str | None, str | None]:
    """Sniff tag/folder hints. Conservative: only trigger on explicit cues to avoid
    over-filtering the search space.
      - Tag: only if query uses explicit #tag syntax.
      - Folder: only if a distinctive folder leaf name (≥5 chars) appears verbatim.
    """
    low = question.lower()
    # Tag: explicit #tag in query
    tag = None
    for m in re.finditer(r"#([a-z0-9][a-z0-9\-]+)", low):
        t = m.group(1)
        if t in {t.lower() for t in known_tags}:
            tag = t
            break
    # Folder: explicit mention of leaf name (distinctive, ≥5 chars)
    folder = None
    for f in sorted(known_folders, key=len, reverse=True):
        if not f or f == ".":
            continue
        leaf = f.split("/")[-1].lower()
        if len(leaf) < 5:
            continue
        if f" {leaf} " in f" {low} ":
            folder = f
            break
    return folder, tag


def expand_to_parent(chunk_text: str, meta: dict) -> str:
    """Return the pre-computed parent slice stored at index time.
    Falls back to the chunk itself if metadata missing (pre-v5 index).
    """
    parent = meta.get("parent")
    return parent if parent else chunk_text


def get_vocabulary(col: chromadb.Collection) -> tuple[set[str], set[str]]:
    """Collect unique tags and folders from the index for intent-based filtering."""
    c = _load_corpus(col)
    return c["tags"], c["folders"]


# ── UX HELPERS ────────────────────────────────────────────────────────────────

def confidence_badge(score: float) -> tuple[str, str]:
    """Return (emoji, label) based on reranker logit score."""
    if score >= 3.0:
        return ("🟢", f"alta · {score:.1f}")
    if score >= 0.0:
        return ("🟡", f"media · {score:.1f}")
    return ("🔴", f"baja · {score:.1f}")


def score_bar(score: float, width: int = 5) -> str:
    """Visual bar from reranker score. Maps score ∈ [-5, 10] to filled cells."""
    clipped = max(-5.0, min(10.0, score))
    normalized = (clipped + 5.0) / 15.0  # 0..1
    filled = int(round(normalized * width))
    return "■" * filled + "□" * (width - filled)


def render_sources(metas: list[dict], scores: list[float]) -> Table:
    """Compact sources table with score bar + path."""
    tbl = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
    tbl.add_column(style="dim")  # bar
    tbl.add_column(style="bold cyan")  # note name
    tbl.add_column(style="cyan dim")  # path
    seen_files = set()
    for m, s in zip(metas, scores):
        f = m.get("file", "")
        if f in seen_files:
            continue
        seen_files.add(f)
        tbl.add_row(f"{score_bar(s)}  {s:+.1f}", m.get("note", ""), f)
    return tbl


SYSTEM_RULES = (
    "Eres un asistente de consulta sobre las notas personales de Obsidian del "
    "usuario. NO sos un modelo de conocimiento general.\n\n"
    "REGLA 1 — FUENTE ÚNICA: respondé usando SOLO información literalmente "
    "presente en el CONTEXTO. Si la pregunta no está cubierta, respondé "
    "exactamente: 'No tengo esa información en tus notas.' y cortá.\n\n"
    "REGLA 2 — CITAR RUTA: cada vez que menciones una nota por nombre, "
    "acompañala de su ruta en formato markdown: [NombreNota](ruta/relativa.md). "
    "La ruta está en `[ruta: ...]` del header del chunk. Citá al menos la primera vez.\n\n"
    "REGLA 3 — MARCAR EXTERNO: si agregás texto que NO sale textualmente del "
    "contexto (intros, parafraseos, biografía, conectores, opinión, conocimiento "
    "general), envolvelo en `<<ext>>...<</ext>>`. Fuera de esos marcadores TODO "
    "debe ser verificable palabra por palabra en el contexto.\n\n"
    "REGLA 4 — FORMATO: respuesta directa, viñetas para listas, sin intro vacía.\n"
)

SYSTEM_RULES_STRICT = (
    "Eres un asistente de consulta sobre las notas personales de Obsidian del "
    "usuario. NO sos un modelo de conocimiento general.\n\n"
    "REGLA 1 — FUENTE ÚNICA Y LITERAL: respondé usando SOLO información "
    "literalmente presente en el CONTEXTO. PROHIBIDO agregar conocimiento general, "
    "biografía, definiciones externas, intros, conectores o parafraseos que amplíen "
    "lo que dice el contexto. Si la pregunta no está cubierta, respondé exactamente: "
    "'No tengo esa información en tus notas.' y cortá.\n\n"
    "REGLA 2 — CITAR RUTA: cada vez que menciones una nota, acompañala de su ruta "
    "en formato markdown: [NombreNota](ruta/relativa.md). Ruta está en `[ruta: ...]` "
    "del header del chunk. Citá al menos la primera vez.\n\n"
    "REGLA 3 — FORMATO: respuesta directa, viñetas cortas, sin intro. Preferí citar "
    "fragmentos verbatim del contexto antes que reformular.\n"
)


def print_query_header(question: str, result: dict) -> None:
    """Render the question panel + metadata row (filters, variants, confidence)."""
    console.print()
    console.print(Panel(f"[bold white]{question}[/bold white]", border_style="cyan", padding=(0, 1)))
    meta_bits: list[str] = []
    emoji, label = confidence_badge(result["confidence"])
    meta_bits.append(f"{emoji} {label}")
    if result["filters_applied"]:
        parts = [f"{k}={v}" for k, v in result["filters_applied"].items()]
        meta_bits.append(f"filtros: {', '.join(parts)}")
    if len(result["query_variants"]) > 1:
        meta_bits.append(f"{len(result['query_variants'])} variantes")
    meta_bits.append(f"{len({m['file'] for m in result['metas']})} nota(s)")
    console.print(f"  [dim]{' · '.join(meta_bits)}[/dim]")


def print_sources(result: dict) -> None:
    """Render the sources table with score bars."""
    if not result["metas"]:
        return
    console.print()
    console.print(Rule(title="[dim]Fuentes[/dim]", style="dim", characters="╌"))
    console.print(render_sources(result["metas"], result["scores"]))


def reformulate_query(question: str, history: list[dict]) -> str:
    """Rewrite the question as a standalone search query using conversation history."""
    if not history:
        return question

    recent = history[-6:]  # last 3 turns
    history_text = "\n".join(
        f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content'][:200]}"
        for m in recent
    )
    prompt = (
        "Dado este historial de conversación:\n"
        f"{history_text}\n\n"
        f"Y esta nueva pregunta: \"{question}\"\n\n"
        "Reescribe la pregunta como una consulta de búsqueda autónoma y específica "
        "(sin pronombres ambiguos, con contexto completo). "
        "Responde SOLO con la consulta reformulada, sin explicaciones."
    )
    resp = ollama.chat(
        model=HELPER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=HELPER_OPTIONS,
    )
    return resp.message.content.strip().strip('"')


def build_where(folder: str | None, tag: str | None) -> dict | None:
    """Build ChromaDB where filter from folder and/or tag."""
    conditions = []
    if folder:
        conditions.append({"file": {"$contains": folder}})
    if tag:
        conditions.append({"tags": {"$contains": tag}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve(
    col: chromadb.Collection,
    question: str,
    k: int,
    folder: str | None,
    history: list[dict] | None = None,
    tag: str | None = None,
    precise: bool = False,
    multi_query: bool = True,
    auto_filter: bool = True,
) -> dict:
    """Full retrieval pipeline. Returns dict:
       { docs, metas, scores, confidence, search_query, filters_applied, query_variants }

    Pipeline:
      - optional history-aware reformulation (precise)
      - optional auto-filter by tag/folder inferred from the query
      - multi-query: 3 paraphrases, BM25 + semantic for each, union candidates
      - cross-encoder rerank against the original question
      - parent-chunk expansion for final candidates
    """
    if col.count() == 0:
        return {
            "docs": [], "metas": [], "scores": [], "confidence": float("-inf"),
            "search_query": question, "filters_applied": {}, "query_variants": [question],
        }

    # 1. Reformulate vs history (precise mode)
    search_query = question
    if precise and history:
        search_query = reformulate_query(question, history)

    # 2. Auto-filter: sniff tag/folder from the query against index vocabulary
    filters_applied: dict[str, str] = {}
    if auto_filter and not folder and not tag:
        known_tags, known_folders = get_vocabulary(col)
        inferred_folder, inferred_tag = infer_filters(search_query, known_tags, known_folders)
        folder = folder or inferred_folder
        tag = tag or inferred_tag
        if inferred_tag:
            filters_applied["tag"] = inferred_tag
        if inferred_folder:
            filters_applied["folder"] = inferred_folder

    # 3. Multi-query expansion (original + 2 paraphrases)
    variants = [search_query]
    if multi_query:
        try:
            variants = expand_queries(search_query)
        except Exception:
            variants = [search_query]

    # 4. Retrieve per variant, union IDs.
    #    Non-HyDE path: batch-embed all variants in one Ollama call.
    #    HyDE path: each variant needs its own generated hypothetical, keep per-variant.
    where = build_where(folder, tag)
    if precise:
        variant_embeds = [hyde_embed(v) for v in variants]
    else:
        variant_embeds = embed(variants)

    seen_ids: set[str] = set()
    merged_ordered: list[str] = []
    for v, q_embed in zip(variants, variant_embeds):
        sem_kwargs: dict = {
            "query_embeddings": [q_embed],
            "n_results": min(RETRIEVE_K, col.count()),
            "include": ["documents", "metadatas"],
        }
        if where:
            sem_kwargs["where"] = where
        sem_ids = col.query(**sem_kwargs)["ids"][0]
        bm25_ids = bm25_search(col, v, RETRIEVE_K, folder, tag)
        for id_ in rrf_merge(sem_ids, bm25_ids):
            if id_ not in seen_ids:
                seen_ids.add(id_)
                merged_ordered.append(id_)

    if not merged_ordered:
        return {
            "docs": [], "metas": [], "scores": [], "confidence": float("-inf"),
            "search_query": search_query, "filters_applied": filters_applied,
            "query_variants": variants,
        }

    # 5. Fetch candidates
    fetched = col.get(ids=merged_ordered, include=["documents", "metadatas"])
    id_map = {
        id_: (doc, meta)
        for id_, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])
    }
    candidates = [(id_map[id_][0], id_map[id_][1], id_) for id_ in merged_ordered if id_ in id_map]

    # 6. Parent-chunk expansion BEFORE rerank so the reranker scores the same
    #    context the LLM will actually see. Prevents chunk-vs-parent mismatch
    #    where the best-scoring chunk's parent is less relevant than another's.
    expanded = [expand_to_parent(c[0], c[1]) for c in candidates]

    # 7. Cross-encoder rerank against the ORIGINAL question (not paraphrase)
    reranker = get_reranker()
    pairs = [(question, e) for e in expanded]
    scores = reranker.predict(pairs, show_progress_bar=False)
    scored = sorted(
        zip(candidates, expanded, scores),
        key=lambda x: float(x[2]),
        reverse=True,
    )[:k]
    final_scores = [float(s) for _, _, s in scored]
    top_score = final_scores[0] if final_scores else float("-inf")
    docs = [e for _, e, _ in scored]
    metas = [c[1] for c, _, _ in scored]

    return {
        "docs": docs,
        "metas": metas,
        "scores": final_scores,
        "confidence": top_score,
        "search_query": search_query,
        "filters_applied": filters_applied,
        "query_variants": variants,
    }


# ── COMMANDS ──────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """RAG local para notas de Obsidian."""


def file_hash(raw: str) -> str:
    """Stable hash of file contents to detect changes."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _index_single_file(col: chromadb.Collection, path: Path) -> str:
    """(Re)index one markdown file. Returns one of:
    'skipped' (unchanged), 'indexed' (new/updated), 'removed' (file gone),
    'empty' (file exists but produced no chunks).
    Invalidates the corpus cache when changes occur.
    """
    try:
        doc_id_prefix = str(path.relative_to(VAULT_PATH))
    except ValueError:
        return "skipped"  # outside vault
    if is_excluded(doc_id_prefix):
        return "skipped"

    existing = col.get(where={"file": doc_id_prefix}, include=["metadatas"])
    existing_ids = existing["ids"]
    existing_hash = (
        existing["metadatas"][0].get("hash") if existing["metadatas"] else None
    )

    if not path.is_file():
        if existing_ids:
            col.delete(ids=existing_ids)
            _invalidate_corpus_cache()
            return "removed"
        return "empty"

    raw = path.read_text(encoding="utf-8", errors="ignore")
    h = file_hash(raw)
    if existing_hash == h:
        return "skipped"

    if existing_ids:
        col.delete(ids=existing_ids)

    folder = str(path.relative_to(VAULT_PATH).parent)
    fm = parse_frontmatter(raw)
    tags = [str(t) for t in (fm.get("tags") or []) if t]
    text = clean_md(raw)
    chunks = semantic_chunks(text, path.stem, folder, tags, fm)
    if not chunks:
        _invalidate_corpus_cache()
        return "empty"

    ids = [f"{doc_id_prefix}::{i}" for i in range(len(chunks))]
    embed_texts = [c[0] for c in chunks]
    display_texts = [c[1] for c in chunks]
    parent_texts = [c[2] for c in chunks]
    embeddings = embed(embed_texts)

    base_meta = {
        "file": doc_id_prefix, "note": path.stem, "folder": folder,
        "tags": ",".join(tags), "hash": h,
    }
    for field in FM_SEARCHABLE_FIELDS:
        v = fm.get(field)
        if v not in (None, ""):
            base_meta[field] = str(v)

    metadatas = [dict(base_meta, parent=p) for p in parent_texts]
    col.add(
        ids=ids,
        embeddings=embeddings,
        documents=display_texts,
        metadatas=metadatas,
    )
    _invalidate_corpus_cache()
    return "indexed"


@cli.command()
@click.option("--reset", is_flag=True, help="Borrar índice antes de reindexar")
def index(reset: bool):
    """Indexar notas del vault (incremental, detecta cambios por hash)."""
    col = get_db()
    _invalidate_corpus_cache()

    if reset:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        col = client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        console.print("[yellow]Índice borrado.[/yellow]")

    # Build file → chunks_in_db map (once) so we can detect stale/orphan chunks.
    existing_all = col.get(include=["metadatas"])
    file_to_chunks: dict[str, list[tuple[str, str]]] = {}
    for id_, meta in zip(existing_all["ids"], existing_all["metadatas"]):
        file_to_chunks.setdefault(meta.get("file", ""), []).append(
            (id_, meta.get("hash", ""))
        )

    md_files = [
        p for p in VAULT_PATH.rglob("*.md")
        if not is_excluded(str(p.relative_to(VAULT_PATH)))
    ]
    console.print(f"[cyan]Indexando {len(md_files)} notas...[/cyan]")

    indexed_files = set()
    added_chunks = 0
    updated_files = 0
    for path in track(md_files, description="Procesando..."):
        doc_id_prefix = str(path.relative_to(VAULT_PATH))
        folder = str(path.relative_to(VAULT_PATH).parent)
        raw = path.read_text(encoding="utf-8", errors="ignore")
        h = file_hash(raw)

        # Skip unchanged files (hash matches what's stored)
        existing = file_to_chunks.get(doc_id_prefix, [])
        if existing and all(eh == h for _, eh in existing):
            indexed_files.add(doc_id_prefix)
            continue

        # File changed (or new) — remove any stale chunks first
        if existing:
            col.delete(ids=[eid for eid, _ in existing])
            updated_files += 1

        fm = parse_frontmatter(raw)
        tags = [str(t) for t in (fm.get("tags") or []) if t]
        text = clean_md(raw)
        chunks = semantic_chunks(text, path.stem, folder, tags, fm)
        if not chunks:
            indexed_files.add(doc_id_prefix)
            continue

        ids = [f"{doc_id_prefix}::{i}" for i in range(len(chunks))]
        embed_texts = [c[0] for c in chunks]
        display_texts = [c[1] for c in chunks]
        parent_texts = [c[2] for c in chunks]
        embeddings = embed(embed_texts)

        # Metadata carries hash + searchable frontmatter fields for filtering.
        base_meta = {
            "file": doc_id_prefix,
            "note": path.stem,
            "folder": folder,
            "tags": ",".join(tags),
            "hash": h,
        }
        for field in FM_SEARCHABLE_FIELDS:
            v = fm.get(field)
            if v not in (None, ""):
                base_meta[field] = str(v)

        metadatas = []
        for p in parent_texts:
            meta = dict(base_meta)
            meta["parent"] = p
            metadatas.append(meta)

        col.add(
            ids=ids,
            embeddings=embeddings,
            documents=display_texts,
            metadatas=metadatas,
        )
        added_chunks += len(ids)
        indexed_files.add(doc_id_prefix)

    # Orphan cleanup: files in DB that no longer exist on disk
    orphan_files = set(file_to_chunks.keys()) - indexed_files
    orphan_ids: list[str] = []
    for f in orphan_files:
        orphan_ids.extend(eid for eid, _ in file_to_chunks[f])
    if orphan_ids:
        col.delete(ids=orphan_ids)

    console.print(
        f"[green]Listo. {added_chunks} chunks (re)indexados · "
        f"{updated_files} notas actualizadas · "
        f"{len(orphan_files)} huérfanas limpiadas.[/green]"
    )


@cli.command()
@click.option("--debounce", default=3.0, help="Segundos a esperar antes de reindexar un cambio (default: 3)")
def watch(debounce: float):
    """Observa el vault y reindexa incrementalmente al guardar notas."""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    col = get_db()
    pending: set[Path] = set()
    lock = threading.Lock()

    class Handler(FileSystemEventHandler):
        def _queue(self, raw_path: str) -> None:
            if not raw_path.endswith(".md"):
                return
            p = Path(raw_path)
            try:
                p.resolve().relative_to(VAULT_PATH.resolve())
            except ValueError:
                return
            with lock:
                pending.add(p)

        def on_modified(self, event):
            if not event.is_directory:
                self._queue(event.src_path)
        def on_created(self, event):
            if not event.is_directory:
                self._queue(event.src_path)
        def on_deleted(self, event):
            if not event.is_directory:
                self._queue(event.src_path)
        def on_moved(self, event):
            if not event.is_directory:
                self._queue(event.src_path)
                self._queue(event.dest_path)

    observer = Observer()
    observer.schedule(Handler(), str(VAULT_PATH), recursive=True)
    observer.start()

    console.print(f"[cyan]Watching[/cyan] {VAULT_PATH}")
    console.print(f"[dim]debounce={debounce}s · Ctrl+C para salir[/dim]")

    try:
        while True:
            time.sleep(debounce)
            if not pending:
                continue
            with lock:
                batch = list(pending)
                pending.clear()
            for p in batch:
                try:
                    status = _index_single_file(col, p)
                except Exception as e:
                    console.print(f"  [red]error[/red] {p.name}: {e}")
                    continue
                if status == "skipped":
                    continue
                try:
                    rel = p.relative_to(VAULT_PATH)
                except ValueError:
                    rel = p
                color = {"indexed": "green", "removed": "yellow", "empty": "dim"}.get(status, "white")
                console.print(f"  [{color}]{status:>8}[/{color}] {rel}")
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        console.print("[yellow]Watch detenido.[/yellow]")


@cli.command()
@click.argument("question")
@click.option("-k", default=RERANK_TOP, help=f"Chunks finales a usar (default: {RERANK_TOP})")
@click.option("--folder", default=None, help="Filtrar por carpeta (ej: '02-Areas/Musica')")
@click.option("--tag", default=None, help="Filtrar por tag (ej: letra, rock, ai, finanzas)")
@click.option("--hyde", is_flag=True, help="Activa HyDE (mejora con LLMs grandes; con modelos chicos tiende a empeorar)")
@click.option("--no-multi", is_flag=True, help="Desactiva multi-query expansion")
@click.option("--no-auto-filter", is_flag=True, help="Desactiva inferencia de filtros")
@click.option("--raw", is_flag=True, help="Skip LLM — muestra chunks recuperados directo")
@click.option("--loose", is_flag=True, help="Permite prosa externa del LLM (marcada con ⚠)")
@click.option("--force", is_flag=True, help="Llamar al LLM incluso si la confianza del rerank es baja")
def query(
    question: str, k: int, folder: str | None, tag: str | None,
    hyde: bool, no_multi: bool, no_auto_filter: bool,
    raw: bool, loose: bool, force: bool,
):
    """Consulta única contra las notas."""
    col = get_db()
    if col.count() == 0:
        console.print("[red]Índice vacío. Ejecuta: rag index[/red]")
        return

    # Intent routing: aggregate/list/recent queries don't need the retrieval
    # pipeline + LLM — they want a metadata scan. Fall through to semantic
    # otherwise. User-supplied --folder/--tag override classifier params.
    known_tags, known_folders = get_vocabulary(col)
    intent, intent_params = classify_intent(question, known_tags, known_folders)
    if intent != "semantic":
        if folder:
            intent_params["folder"] = folder
        if tag:
            intent_params["tag"] = tag
        params_str = ", ".join(f"{k}={v}" for k, v in intent_params.items() if v) or "sin filtros"
        console.print()
        if intent == "count":
            n, files = handle_count(col, intent_params)
            console.print(
                f"[bold green]{n}[/bold green] nota(s) [dim]({params_str})[/dim]"
            )
            if n and n <= 30:
                render_file_list(f"notas", files)
        elif intent == "list":
            files = handle_list(col, intent_params)
            console.print(
                f"[bold cyan]{len(files)}[/bold cyan] nota(s) [dim]({params_str})[/dim]"
            )
            render_file_list("notas", files)
        elif intent == "recent":
            files = handle_recent(col, intent_params)
            console.print(
                f"[bold cyan]{len(files)}[/bold cyan] nota(s) recientes [dim]({params_str})[/dim]"
            )
            render_file_list("últimas modificadas", files)
        log_query_event({
            "cmd": "query", "q": question, "intent": intent, "params": intent_params,
            "count": len(files) if intent != "count" else n,
        })
        return

    t_start = time.perf_counter()
    with console.status("[dim]buscando…[/dim]", spinner="dots"):
        result = retrieve(
            col, question, k, folder, tag=tag, precise=hyde,
            multi_query=not no_multi, auto_filter=not no_auto_filter,
        )
    t_retrieve = time.perf_counter() - t_start
    if not result["docs"]:
        log_query_event({
            "cmd": "query", "q": question, "filters": result.get("filters_applied"),
            "variants": result.get("query_variants"), "paths": [],
            "top_score": None, "t_retrieve": round(t_retrieve, 2), "answered": False,
        })
        console.print("[yellow]Sin resultados.[/yellow]")
        return

    print_query_header(question, result)

    if raw:
        # Skip LLM — dump retrieved chunks verbatim with their path.
        console.print()
        for d, m, s in zip(result["docs"], result["metas"], result["scores"]):
            path = m.get("file", "")
            note = m.get("note", "")
            console.print(f"[bold cyan]{note}[/bold cyan] [dim]({path}) · {s:+.1f}[/dim]")
            console.print(Markdown(d))
            console.print(Rule(style="dim"))
        print_sources(result)
        return

    # Gate LLM on reranker confidence. Negative top score ≈ rerank found
    # nothing relevant — skipping the LLM avoids hallucinated answers from
    # unrelated chunks. `--force` overrides.
    if result["confidence"] < CONFIDENCE_RERANK_MIN and not force:
        console.print()
        console.print(
            f"[yellow]No tengo esa información en tus notas.[/yellow] "
            f"[dim](top rerank score: {result['confidence']:+.2f} < {CONFIDENCE_RERANK_MIN}; "
            f"usá --force para llamar al LLM igual)[/dim]"
        )
        print_sources(result)
        log_query_event({
            "cmd": "query", "q": question,
            "filters": result.get("filters_applied"),
            "variants": result.get("query_variants"),
            "paths": [m.get("file", "") for m in result["metas"]],
            "scores": [round(float(s), 2) for s in result["scores"]],
            "top_score": round(float(result["confidence"]), 2),
            "t_retrieve": round(t_retrieve, 2), "t_gen": 0.0,
            "answered": False, "gated_low_confidence": True,
        })
        return

    context = "\n\n---\n\n".join(
        f"[nota: {m['note']}] [ruta: {m['file']}]\n{d}"
        for d, m in zip(result["docs"], result["metas"])
    )
    rules = SYSTEM_RULES if loose else SYSTEM_RULES_STRICT
    prompt = (
        f"{rules}\n"
        f"CONTEXTO:\n{context}\n\n"
        f"PREGUNTA: {question}\n\nRESPUESTA:"
    )

    t_gen_start = time.perf_counter()
    with console.status("[dim]pensando…[/dim]", spinner="dots"):
        response = ollama.chat(
            model=resolve_chat_model(),
            messages=[{"role": "user", "content": prompt}],
            options=CHAT_OPTIONS,
        )
    t_gen = time.perf_counter() - t_gen_start
    full = response.message.content

    console.print()
    console.print(render_response(full))

    bad = verify_citations(full, result["metas"])
    if bad:
        console.print()
        console.print("[bold red]⚠ Citas no verificadas:[/bold red]")
        for label, path in bad:
            console.print(f"  [red]• {label} → {path}[/red] [dim](no está en los chunks recuperados)[/dim]")

    log_query_event({
        "cmd": "query",
        "q": question,
        "filters": result.get("filters_applied"),
        "variants": result.get("query_variants"),
        "paths": [m.get("file", "") for m in result["metas"]],
        "scores": [round(float(s), 2) for s in result["scores"]],
        "top_score": round(float(result["confidence"]), 2),
        "t_retrieve": round(t_retrieve, 2),
        "t_gen": round(t_gen, 2),
        "answer_len": len(full),
        "bad_citations": [p for _, p in bad],
        "mode": "raw" if raw else ("loose" if loose else "strict"),
    })

    print_sources(result)


@cli.command()
@click.option("-k", default=RERANK_TOP, help="Chunks finales por turno")
@click.option("--folder", default=None, help="Filtrar por carpeta (ej: '02-Areas/Musica')")
@click.option("--tag", default=None, help="Filtrar por tag (ej: letra, rock, ai, finanzas)")
@click.option("--precise", is_flag=True, help="HyDE + reformulación (más preciso, ~5s extra)")
@click.option("--no-multi", is_flag=True, help="Desactiva multi-query expansion")
@click.option("--no-auto-filter", is_flag=True, help="Desactiva inferencia de filtros")
def chat(
    k: int, folder: str | None, tag: str | None, precise: bool,
    no_multi: bool, no_auto_filter: bool,
):
    """Chat interactivo con tus notas."""
    col = get_db()
    if col.count() == 0:
        console.print("[red]Índice vacío. Ejecuta: rag index[/red]")
        return

    flags = []
    if folder:
        flags.append(f"carpeta: {folder}")
    if tag:
        flags.append(f"tag: #{tag}")
    features = []
    if precise:
        features.append("HyDE")
    if not no_multi:
        features.append("multi-query")
    if not no_auto_filter:
        features.append("auto-filter")
    features.append("rerank")
    subtitle = f"[dim]· {' · '.join(features)}"
    if flags:
        subtitle += f" · {' · '.join(flags)}"
    subtitle += "[/dim]"

    console.print(Panel(
        f"[bold green]RAG Obsidian — Chat[/bold green]\n{subtitle}\n"
        "[dim]/exit o Ctrl+C para salir[/dim]",
        border_style="green",
    ))

    history: list[dict] = []
    first_turn = True
    while True:
        try:
            if not first_turn:
                console.print(Rule(style="dim", characters="╌"))
            first_turn = False
            question = console.input("\n[bold cyan]❯[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Hasta luego.[/dim]")
            break

        if question == "/exit":
            console.print("[dim]Hasta luego.[/dim]")
            break
        if not question:
            continue

        with console.status("[dim]buscando…[/dim]", spinner="dots"):
            result = retrieve(
                col, question, k, folder, history, tag, precise,
                multi_query=not no_multi, auto_filter=not no_auto_filter,
            )
        if not result["docs"]:
            console.print("[yellow]Sin resultados relevantes.[/yellow]")
            continue

        print_query_header(question, result)

        context = "\n\n---\n\n".join(
            f"[nota: {m['note']}] [ruta: {m['file']}]\n{d}"
            for d, m in zip(result["docs"], result["metas"])
        )
        system = f"{SYSTEM_RULES}\nCONTEXTO RELEVANTE:\n{context}"

        history.append({"role": "user", "content": question})
        messages = [{"role": "system", "content": system}] + history[-6:]

        with console.status("[dim]pensando…[/dim]", spinner="dots"):
            response = ollama.chat(
                model=resolve_chat_model(), messages=messages, options=CHAT_OPTIONS
            )
        full = response.message.content

        console.print()
        console.print(render_response(full))
        history.append({"role": "assistant", "content": full})
        print_sources(result)


@cli.command()
@click.option("--file", "queries_file", default="queries.yaml",
              help="YAML con queries golden (default: queries.yaml)")
@click.option("-k", default=5, help="top-k para calcular hit/recall (default: 5)")
@click.option("--hyde", is_flag=True, help="Activa HyDE en la evaluación")
@click.option("--no-multi", is_flag=True, help="Sin multi-query expansion")
def eval(queries_file: str, k: int, hyde: bool, no_multi: bool):
    """Evaluar el retriever contra un set de queries golden.

    Métricas: hit@k (% queries donde alguna nota esperada cae en top-k),
    MRR (mean reciprocal rank del mejor hit), recall@k (fracción de notas
    esperadas recuperadas por query, promediado).
    """
    import pathlib
    path = pathlib.Path(queries_file)
    if not path.is_absolute():
        path = pathlib.Path.cwd() / queries_file
    if not path.is_file():
        console.print(f"[red]No existe {path}[/red]")
        return

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries = data.get("queries") or []
    if not queries:
        console.print(f"[yellow]Sin queries en {path}[/yellow]")
        return

    col = get_db()
    if col.count() == 0:
        console.print("[red]Índice vacío. Ejecuta: rag index[/red]")
        return

    hits = 0
    rr_sum = 0.0
    recall_sum = 0.0
    per_query: list[tuple[str, bool, float, float, list[str]]] = []

    for entry in track(queries, description="Evaluando…"):
        q = entry["question"]
        expected = set(entry.get("expected") or [])
        result = retrieve(
            col, q, k, folder=None, tag=None,
            precise=hyde, multi_query=not no_multi, auto_filter=True,
        )
        retrieved_paths = [m.get("file", "") for m in result["metas"]]
        # dedup while preserving order for MRR
        seen_paths: list[str] = []
        for p in retrieved_paths:
            if p not in seen_paths:
                seen_paths.append(p)
        unique_retrieved = set(seen_paths)

        hit = bool(expected & unique_retrieved)
        hits += 1 if hit else 0

        rr = 0.0
        for rank, p in enumerate(seen_paths, start=1):
            if p in expected:
                rr = 1.0 / rank
                break
        rr_sum += rr

        recall = len(expected & unique_retrieved) / len(expected) if expected else 0.0
        recall_sum += recall

        per_query.append((q, hit, rr, recall, seen_paths))

    n = len(queries)
    hit_at_k = hits / n
    mrr = rr_sum / n
    recall_at_k = recall_sum / n

    # Per-query detail table
    tbl = Table(title=f"Evaluación (k={k})", show_lines=False)
    tbl.add_column("Query", style="cyan", overflow="fold", max_width=50)
    tbl.add_column("Hit", justify="center")
    tbl.add_column("RR", justify="right")
    tbl.add_column("Recall", justify="right")
    for q, hit, rr, rec, _ in per_query:
        tbl.add_row(
            q,
            "[green]✓[/green]" if hit else "[red]✗[/red]",
            f"{rr:.2f}",
            f"{rec:.2f}",
        )
    console.print(tbl)

    # Failed queries: show what the retriever actually returned
    failed = [(q, paths) for q, hit, _, _, paths in per_query if not hit]
    if failed:
        console.print()
        console.print("[bold red]Queries sin hit — top-k recuperado:[/bold red]")
        for q, paths in failed:
            console.print(f"  [yellow]{q}[/yellow]")
            for p in paths[:k]:
                console.print(f"    [dim]· {p}[/dim]")

    # Aggregate
    console.print()
    console.print(
        f"[bold]hit@{k}:[/bold] {hit_at_k:.2%}  ·  "
        f"[bold]MRR:[/bold] {mrr:.3f}  ·  "
        f"[bold]recall@{k}:[/bold] {recall_at_k:.2%}  ·  "
        f"[dim]n={n}[/dim]"
    )


@cli.command()
@click.option("-n", default=20, help="Cantidad de queries a mostrar (default: 20)")
@click.option("--low-confidence", is_flag=True, help="Solo queries con top_score < 0")
def log(n: int, low_confidence: bool):
    """Inspeccionar el log de queries (últimas N)."""
    if not LOG_PATH.is_file():
        console.print(f"[yellow]No hay log aún en {LOG_PATH}[/yellow]")
        return
    entries: list[dict] = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    if low_confidence:
        entries = [e for e in entries if (e.get("top_score") or 0) < 0]
    entries = entries[-n:]
    tbl = Table(title=f"Últimas {len(entries)} queries", show_lines=False)
    tbl.add_column("ts", style="dim")
    tbl.add_column("query", style="cyan", overflow="fold", max_width=45)
    tbl.add_column("score", justify="right")
    tbl.add_column("retr", justify="right")
    tbl.add_column("gen", justify="right")
    tbl.add_column("mode", style="dim")
    for e in entries:
        score = e.get("top_score")
        score_str = f"{score:+.1f}" if isinstance(score, (int, float)) else "-"
        score_style = "green" if (score or 0) >= 3 else ("yellow" if (score or 0) >= 0 else "red")
        tbl.add_row(
            e.get("ts", "")[-8:],
            e.get("q", ""),
            f"[{score_style}]{score_str}[/{score_style}]",
            f"{e.get('t_retrieve', 0):.1f}",
            f"{e.get('t_gen', 0):.1f}",
            e.get("mode", ""),
        )
    console.print(tbl)


@cli.command()
def stats():
    """Estado del índice."""
    col = get_db()
    count = col.count()
    console.print(f"[cyan]Chunks indexados:[/cyan] {count}")
    console.print(f"[cyan]Vault:[/cyan] {VAULT_PATH}")
    console.print(f"[cyan]DB:[/cyan] {DB_PATH}")
    console.print(f"[cyan]Colección:[/cyan] {COLLECTION_NAME}")
    console.print(f"[cyan]Embed model:[/cyan] {EMBED_MODEL}")
    try:
        resolved = resolve_chat_model()
    except Exception as e:
        resolved = f"[red]{e}[/red]"
    console.print(f"[cyan]Chat model:[/cyan] {resolved}  [dim](preference: {', '.join(CHAT_MODEL_PREFERENCE)})[/dim]")
    console.print(f"[cyan]Helper model:[/cyan] {HELPER_MODEL}")
    console.print(f"[cyan]Reranker:[/cyan] {RERANKER_MODEL}")
    console.print(f"[cyan]Pipeline:[/cyan] HyDE + BM25 + RRF + cross-encoder rerank")


if __name__ == "__main__":
    cli()
