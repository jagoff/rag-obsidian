#!/usr/bin/env python3
"""Obsidian RAG — semantic chunking, title prefix, HyDE, hybrid BM25+semantic, reranking."""

import contextlib
import hashlib
import json
import os
import re
import secrets
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

# Allow single-level balanced parens inside the path (Obsidian paths often
# contain literal parens like "02-Areas/Musica/Explorando (otras)/X.md").
NOTE_LINK_RE = re.compile(r"\[([^\]]+)\]\(((?:[^()\n]|\([^()\n]*\))+?\.md)\)")
# command-r often emits just [path.md] without a markdown-link wrapper.
BARE_PATH_RE = re.compile(r"\[([^\[\]\n]+?\.md)\]")
EXT_RE = re.compile(r"<<ext>>(.*?)<<\/ext>>", re.DOTALL)
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


def _file_link_style(path: str, base: str) -> str:
    """OSC 8 clickable style (iTerm2/Terminal.app) — opens the note in Obsidian
    when clicked on macOS (file:// URL hands off to the default .md handler).
    """
    full = (VAULT_PATH / path).resolve()
    return f"{base} link file://{urllib.parse.quote(str(full))}"


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
    """
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

        inline_code_style = (
            "bold yellow" if base_style and "yellow" in base_style
            else "bold cyan"
        )
        bold_style = f"bold {base_style}" if base_style else "bold"

        pos = 0
        for start, end, kind, content in spans:
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
        """Segmento con links + inline code. Asume fences y ext ya extraídos."""
        spans: list[tuple[int, int, str, str]] = []
        consumed: list[tuple[int, int]] = []
        for m in NOTE_LINK_RE.finditer(segment):
            spans.append((m.start(), m.end(), m.group(1), m.group(2)))
            consumed.append(m.span())
        for m in BARE_PATH_RE.finditer(segment):
            if any(s <= m.start() < e for s, e in consumed):
                continue
            spans.append((m.start(), m.end(), m.group(1), m.group(1)))
        spans.sort()

        last = 0
        label_base = "bold cyan" if not base_style else "bold yellow"
        path_base = "cyan dim" if not base_style else "yellow dim"
        for start, end, label, path in spans:
            if start > last:
                emit_plain_or_inline(segment[last:start], base_style=base_style)
            if label == path:
                out.append(path, style=_file_link_style(path, "bold magenta"))
            else:
                out.append(label, style=_file_link_style(path, label_base))
                out.append(" (", style="dim")
                out.append(path, style=_file_link_style(path, path_base))
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
            out.append(ln + "\n", style="bold white")

    pos = 0
    for m in CODE_FENCE_RE.finditer(text):
        if m.start() > pos:
            emit_ext_and_links(text[pos:m.start()])
        emit_code_fence(m.group(1))
        pos = m.end()
    if pos < len(text):
        emit_ext_and_links(text[pos:])
    return out

# Multi-vault: `OBSIDIAN_RAG_VAULT` env var overrides the default iCloud vault.
# Collections are namespaced per vault path so switching doesn't pollute the
# index — fresh vault = fresh collection automatically.
_DEFAULT_VAULT = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/Notes"

# Multi-vault: registry persistente con vaults nombrados + un "current"
# pointer. Precedencia para resolver VAULT_PATH:
#   1. OBSIDIAN_RAG_VAULT env (per-invocation override, gana siempre).
#   2. Registry's "current" vault (rag vault use <name>).
#   3. _DEFAULT_VAULT (iCloud Notes) — para usuarios single-vault que
#      nunca tocan el registry.
# La colección de Chroma se nombra con el hash del path resuelto, así que
# cada vault tiene su propio índice automáticamente. Cero contaminación.
VAULTS_CONFIG_PATH = Path.home() / ".config/obsidian-rag/vaults.json"


def _load_vaults_config() -> dict:
    """Lee el registry de vaults. Estructura:
      {"vaults": {name: absolute_path}, "current": name | None}
    Si no existe o está corrupto, devuelve estructura vacía — el caller
    decide qué hacer (typicamente: caer al default).
    """
    if not VAULTS_CONFIG_PATH.is_file():
        return {"vaults": {}, "current": None}
    try:
        cfg = json.loads(VAULTS_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"vaults": {}, "current": None}
    cfg.setdefault("vaults", {})
    cfg.setdefault("current", None)
    return cfg


def _save_vaults_config(cfg: dict) -> None:
    VAULTS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    VAULTS_CONFIG_PATH.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def _resolve_vault_path() -> Path:
    """Aplica la precedencia de 3 niveles para decidir el vault activo."""
    env = os.environ.get("OBSIDIAN_RAG_VAULT")
    if env:
        return Path(env)
    cfg = _load_vaults_config()
    cur = cfg.get("current")
    vaults = cfg.get("vaults", {})
    if cur and cur in vaults:
        return Path(vaults[cur])
    return _DEFAULT_VAULT


VAULT_PATH = _resolve_vault_path()
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


# ── SESSIONS ──────────────────────────────────────────────────────────────────
# Persist multi-turn conversation so follow-ups ("profundizá", pronouns without
# antecedent) can reference prior turns. Shared across `rag chat`, `rag query
# --continue`, MCP `rag_query(session_id=...)`, and the Telegram bots (which
# pass `tg:<chat_id>` as id).
#
# Storage: one JSON file per session under SESSIONS_DIR. LAST_SESSION_FILE
# holds the most recent id so `--continue` / `--resume` can default to it.
# Caller-supplied ids are allowed (needed for Telegram chat_id), validated
# against SESSION_ID_RE.

SESSIONS_DIR = Path.home() / ".local/share/obsidian-rag/sessions"
LAST_SESSION_FILE = Path.home() / ".local/share/obsidian-rag/last_session"
SESSION_TTL_DAYS = 30
SESSION_MAX_TURNS = 50           # cap per file — keeps JSON small, bounds retrieval context
SESSION_HISTORY_WINDOW = 6       # last N messages fed to reformulate_query / LLM

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def new_session_id() -> str:
    """Short opaque id: hex timestamp + 6 random hex chars (sortable + unique)."""
    return f"{int(time.time()):x}-{secrets.token_hex(3)}"


def _valid_session_id(sid: str) -> bool:
    return bool(SESSION_ID_RE.match(sid))


def session_path(sid: str) -> Path:
    if not _valid_session_id(sid):
        raise ValueError(f"invalid session id: {sid!r}")
    return SESSIONS_DIR / f"{sid}.json"


def load_session(sid: str) -> dict | None:
    """Read a session file. Returns None if missing, invalid id, or unreadable."""
    try:
        p = session_path(sid)
    except ValueError:
        return None
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_session(sess: dict) -> None:
    """Atomic write via tmp file + replace. Also records last-session pointer."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sess["updated_at"] = datetime.now().isoformat(timespec="seconds")
    p = session_path(sess["id"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sess, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    _set_last_session(sess["id"])


def ensure_session(sid: str | None, *, mode: str) -> dict:
    """Return an existing session or a fresh one. Caller-supplied ids are kept
    verbatim if valid; otherwise a new random id is minted.
    """
    if sid:
        existing = load_session(sid)
        if existing:
            return existing
        new_id = sid if _valid_session_id(sid) else new_session_id()
    else:
        new_id = new_session_id()
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "id": new_id,
        "created_at": now,
        "updated_at": now,
        "mode": mode,
        "turns": [],
    }


def append_turn(sess: dict, turn: dict) -> None:
    """Append one turn to the session, capping total stored turns."""
    sess.setdefault("turns", []).append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        **turn,
    })
    if len(sess["turns"]) > SESSION_MAX_TURNS:
        sess["turns"] = sess["turns"][-SESSION_MAX_TURNS:]


def session_history(sess: dict, window: int = SESSION_HISTORY_WINDOW) -> list[dict]:
    """Flatten session turns into `[{role, content}]` for reformulate_query /
    the chat LLM. Returns the last `window` messages.
    """
    msgs: list[dict] = []
    for turn in sess.get("turns", []):
        q = turn.get("q")
        a = turn.get("a")
        if q:
            msgs.append({"role": "user", "content": q})
        if a:
            msgs.append({"role": "assistant", "content": a})
    return msgs[-window:]


def last_session_id() -> str | None:
    try:
        sid = LAST_SESSION_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return sid or None


def _set_last_session(sid: str) -> None:
    try:
        LAST_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_SESSION_FILE.write_text(sid, encoding="utf-8")
    except Exception:
        pass


def list_sessions(limit: int = 20) -> list[dict]:
    """Return recent session summaries (newest first) — id, turn count, first question."""
    if not SESSIONS_DIR.is_dir():
        return []
    out: list[dict] = []
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        turns = s.get("turns", [])
        first_q = next((t.get("q", "") for t in turns if t.get("q")), "")
        out.append({
            "id": s.get("id", p.stem),
            "updated_at": s.get("updated_at", ""),
            "created_at": s.get("created_at", ""),
            "turns": len(turns),
            "first_q": first_q[:80],
            "mode": s.get("mode", ""),
        })
    out.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
    return out[:limit]


def cleanup_sessions(ttl_days: int = SESSION_TTL_DAYS) -> int:
    """Remove session files older than `ttl_days` by mtime. Returns count removed."""
    if not SESSIONS_DIR.is_dir():
        return 0
    cutoff = time.time() - ttl_days * 86400
    removed = 0
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except Exception:
            pass
    return removed


EMBED_MODEL = "bge-m3"  # multilingual (ES/EN), 1024-dim
# Chat model preference: first available wins. command-r is RAG-trained +
# citation-native, ideal for this use case. Fallbacks cover slower pulls.
CHAT_MODEL_PREFERENCE = ("command-r:latest", "qwen2.5:14b", "phi4:latest")
HELPER_MODEL = "qwen2.5:3b"      # fast, for internal rewrites (multi-query, HyDE, reformulate)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # cross-encoder, multilingual, MPS-friendly
_COLLECTION_BASE = "obsidian_notes_v7"  # v7: wikilinks (outlinks) in metadata
# Separate collection for URL-context embeddings — the URL-finder pipeline
# scores against the prose around each link, not the link string itself.
# Versioned independently from the main collection.
_URLS_COLLECTION_BASE = "obsidian_urls_v1"
# Per-vault suffix so multiple vaults don't share a collection.
_vault_slug = hashlib.sha256(str(VAULT_PATH.resolve()).encode()).hexdigest()[:8]
COLLECTION_NAME = (
    _COLLECTION_BASE if str(VAULT_PATH.resolve()) == str(_DEFAULT_VAULT.resolve())
    else f"{_COLLECTION_BASE}_{_vault_slug}"
)
URLS_COLLECTION_NAME = (
    _URLS_COLLECTION_BASE if str(VAULT_PATH.resolve()) == str(_DEFAULT_VAULT.resolve())
    else f"{_URLS_COLLECTION_BASE}_{_vault_slug}"
)

# Deterministic decoding — this is a retrieval tool, not creative writing.
# num_ctx dimensionado al prompt real (system ~500 tokens + 5 chunks × ~300
# tokens + query + answer) — ventana de KV más chica = menos memoria y genera
# más rápido.
# num_predict cap evita respuestas verbose: answers útiles caben en ~600 tok.
CHAT_OPTIONS = {
    "temperature": 0, "top_p": 1, "seed": 42,
    "num_ctx": 4096,
    "num_predict": 768,
}
# Helpers (paraphrases, HyDE) devuelven 1-3 líneas — cap chico acelera.
HELPER_OPTIONS = {
    "temperature": 0, "top_p": 1, "seed": 42,
    "num_ctx": 1024,
    "num_predict": 128,
}

# Keep models resident in VRAM between queries — avoids 2-3s cold reload.
# Ollama accepts -1 (forever, as int) or a duration string like "30m". Default
# to forever for the local-first use case; user can tune via env var.
def _parse_keep_alive(val: str) -> int | str:
    try:
        return int(val)
    except ValueError:
        return val
OLLAMA_KEEP_ALIVE = _parse_keep_alive(os.environ.get("OLLAMA_KEEP_ALIVE", "-1"))


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

def is_excluded(rel_path: str) -> bool:
    """Skip any path whose top-level segment or any parent segment is a
    dotfolder (e.g. .trash/, .obsidian/, .git/). System/hidden by convention.
    """
    return any(seg.startswith(".") for seg in rel_path.split("/") if seg)
RETRIEVE_K = 20    # candidates from semantic + BM25 each
# Cross-encoder rerank escala ~50ms por par en MPS fp32. Con 3 variantes ×
# 2 métodos el pool crudo llega a 70-80 únicos, reranker tarda ~4s. Cappear
# a 40 lleva el rerank a ~2.5s (ahorro ~1.5s) manteniendo hit@5 = RERANK_TOP
# siempre es ≤5, imposible que el top-5 caiga fuera del top-40 RRF salvo
# corner case con score muy plano. Calibrado empíricamente — bajarlo a 30
# arriesga recall en queries ambigüas.
RERANK_POOL_MAX = 40
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


def _collection_name_for_vault(vault_path: Path) -> str:
    """Mismo schema que COLLECTION_NAME: sha256[:8] sufijo salvo default.
    Factorizado para que podamos abrir colecciones de vaults *distintos* al
    activo sin tocar VAULT_PATH ni forzar reimport.
    """
    resolved = vault_path.resolve()
    if str(resolved) == str(_DEFAULT_VAULT.resolve()):
        return _COLLECTION_BASE
    slug = hashlib.sha256(str(resolved).encode()).hexdigest()[:8]
    return f"{_COLLECTION_BASE}_{slug}"


def get_db_for(vault_path: Path) -> chromadb.Collection:
    """Abre la colección Chroma correspondiente a un vault arbitrario.
    Todos los vaults viven en la misma DB (DB_PATH) — la separación es por
    nombre de colección, namespaced por hash del path del vault.
    """
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name=_collection_name_for_vault(vault_path),
        metadata={"hnsw:space": "cosine"},
    )


def _urls_collection_name_for_vault(vault_path: Path) -> str:
    """Mismo schema que URLS_COLLECTION_NAME pero para un vault arbitrario."""
    resolved = vault_path.resolve()
    if str(resolved) == str(_DEFAULT_VAULT.resolve()):
        return _URLS_COLLECTION_BASE
    slug = hashlib.sha256(str(resolved).encode()).hexdigest()[:8]
    return f"{_URLS_COLLECTION_BASE}_{slug}"


# ── Auto-index ────────────────────────────────────────────────────────────────
# Detectar y absorber cambios del vault sin que el usuario tenga que correr
# `rag index` explícito. Dos casos:
#   1. Vault vacío en scope → first-time full index (silent excepto progreso).
#   2. Vault con contenido + archivos modificados desde último check → reindex
#      incremental SOLO de esos archivos. mtime es el filtro barato; hash
#      gate dentro de _index_single_file evita reembedding si nada cambió.
# Persistimos last_check_at por vault para no leer todos los archivos cada
# vez que arranca el chat.

AUTO_INDEX_STATE_PATH = Path.home() / ".local/share/obsidian-rag/auto_index_state.json"


def _auto_index_state_load() -> dict:
    """Lee el state de auto-index. Estructura: {vault_hash: last_check_ts}.
    Tolerante a archivo faltante o corrupto — devuelve {} en cualquier error.
    """
    if not AUTO_INDEX_STATE_PATH.is_file():
        return {}
    try:
        return json.loads(AUTO_INDEX_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _auto_index_state_save(state: dict) -> None:
    try:
        AUTO_INDEX_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        AUTO_INDEX_STATE_PATH.write_text(
            json.dumps(state, indent=2), encoding="utf-8",
        )
    except Exception:
        pass


def _vault_state_key(vault_path: Path) -> str:
    """Clave estable para el state — sha256[:16] del path absoluto."""
    return hashlib.sha256(str(vault_path.resolve()).encode()).hexdigest()[:16]


@contextlib.contextmanager
def _with_vault(vault_path: Path):
    """Context manager que swappea los globals VAULT_PATH /
    COLLECTION_NAME / URLS_COLLECTION_NAME para apuntar a otro vault.

    Usado por _auto_index_vault para reusar _index_single_file (que asume
    que VAULT_PATH es el vault target). Single-threaded — NO usar entre
    threads. Restaura los valores en finally + invalida cache de corpus.
    """
    g = globals()
    saved = {
        "VAULT_PATH": g["VAULT_PATH"],
        "COLLECTION_NAME": g["COLLECTION_NAME"],
        "URLS_COLLECTION_NAME": g["URLS_COLLECTION_NAME"],
    }
    try:
        g["VAULT_PATH"] = vault_path
        g["COLLECTION_NAME"] = _collection_name_for_vault(vault_path)
        g["URLS_COLLECTION_NAME"] = _urls_collection_name_for_vault(vault_path)
        _invalidate_corpus_cache()
        yield
    finally:
        g["VAULT_PATH"] = saved["VAULT_PATH"]
        g["COLLECTION_NAME"] = saved["COLLECTION_NAME"]
        g["URLS_COLLECTION_NAME"] = saved["URLS_COLLECTION_NAME"]
        _invalidate_corpus_cache()


def auto_index_vault(vault_path: Path) -> dict:
    """Detecta cambios en `vault_path` y reindexa lo necesario. Retorna
    {scanned, indexed, removed, kind, took_ms} donde kind ∈
    {"first_time", "incremental", "no_changes"}.

    Estrategia:
      - Vault vacío → first_time, escanea todo + indexa.
      - Vault con contenido → mtime-based incremental: solo lee archivos
        cuyo mtime > last_check_at. Para cada uno, _index_single_file
        ya hace su propio hash gate (skip si content no cambió de verdad).
      - Limpia orphans (archivos en el índice que ya no están en disco).
      - Actualiza last_check_at al final.
    """
    import time as _t
    t0 = _t.perf_counter()
    state = _auto_index_state_load()
    key = _vault_state_key(vault_path)
    last_check = float(state.get(key, 0.0))

    with _with_vault(vault_path):
        col = get_db()
        first_time = col.count() == 0

        # Listar md files (rglob es rápido en APFS, ~50ms para 500 archivos).
        md_files: list[Path] = []
        for p in vault_path.rglob("*.md"):
            try:
                rel = p.relative_to(vault_path)
            except ValueError:
                continue
            if is_excluded(str(rel)):
                continue
            md_files.append(p)

        # mtime-filter SOLO si no es first_time (ahí hay que indexar todo).
        if first_time:
            candidates = md_files
        else:
            candidates = [p for p in md_files if p.stat().st_mtime > last_check]

        indexed = 0
        for p in candidates:
            try:
                status = _index_single_file(col, p, skip_contradict=True)
            except Exception:
                continue
            if status == "indexed":
                indexed += 1

        # Orphans: archivos en el índice que ya no están en disco. Solo
        # vale chequear cuando el vault ya estaba indexado (skip first_time).
        removed = 0
        if not first_time:
            on_disk = {str(p.relative_to(vault_path)) for p in md_files}
            existing = col.get(include=["metadatas"])
            indexed_files = {m.get("file", "") for m in existing["metadatas"]}
            indexed_files.discard("")
            orphans = indexed_files - on_disk
            for orphan in orphans:
                stale = col.get(where={"file": orphan}, include=[])
                if stale["ids"]:
                    col.delete(ids=stale["ids"])
                    removed += 1

    state[key] = _t.time()
    _auto_index_state_save(state)

    if first_time and indexed > 0:
        kind = "first_time"
    elif indexed == 0 and removed == 0:
        kind = "no_changes"
    else:
        kind = "incremental"

    return {
        "scanned": len(md_files),
        "indexed": indexed,
        "removed": removed,
        "kind": kind,
        "took_ms": int((_t.perf_counter() - t0) * 1000),
    }


def get_urls_db() -> chromadb.Collection:
    """Companion collection storing one row per URL found in the vault, with
    its surrounding prose embedded for semantic-by-context lookup.
    """
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name=URLS_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


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


def _index_urls(
    col_urls: chromadb.Collection,
    doc_id_prefix: str,
    raw_text: str,
    note_title: str,
    folder: str,
    tags: list[str],
) -> int:
    """Replace the URL rows for this file with entries extracted from raw_text.
    Returns the count of URL rows written.

    Idempotent: deletes any existing rows whose id starts with `{path}::url::`,
    then re-inserts. Called from `_index_single_file` so URL state stays in
    lockstep with the main chunk index.
    """
    # Delete any prior URL rows for this file. ChromaDB has no prefix delete,
    # so we list ids and filter.
    existing = col_urls.get(where={"file": doc_id_prefix}, include=[])
    if existing.get("ids"):
        col_urls.delete(ids=existing["ids"])

    urls = extract_urls(raw_text)
    if not urls:
        return 0

    ids = [f"{doc_id_prefix}::url::{i}" for i in range(len(urls))]
    contexts = [u["context"] for u in urls]
    embeddings = embed(contexts)
    metas = [
        {
            "file": doc_id_prefix,
            "note": note_title,
            "folder": folder,
            "tags": ",".join(tags),
            "url": u["url"],
            "anchor": u["anchor"],
            "line": u["line"],
        }
        for u in urls
    ]
    col_urls.add(
        ids=ids,
        embeddings=embeddings,
        documents=contexts,
        metadatas=metas,
    )
    return len(urls)


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
    resp = ollama.embed(model=EMBED_MODEL, input=texts, keep_alive=OLLAMA_KEEP_ALIVE)
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
        keep_alive=OLLAMA_KEEP_ALIVE,
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
    title_to_paths: dict[str, set[str]] = {}
    outlinks: dict[str, list[str]] = {}   # path → list of linked note titles
    backlinks: dict[str, set[str]] = {}   # note title → set of paths that link to it

    for m in metas:
        for t in (m.get("tags") or "").split(","):
            t = t.strip()
            if t:
                tags.add(t)
        f = m.get("folder")
        if f:
            folders.add(f)
        path = m.get("file", "")
        title = m.get("note", "")
        if title and path:
            title_to_paths.setdefault(title, set()).add(path)
        if path not in outlinks:
            raw_links = [
                ln.strip() for ln in (m.get("outlinks") or "").split(",") if ln.strip()
            ]
            outlinks[path] = raw_links
            for t in raw_links:
                backlinks.setdefault(t, set()).add(path)

    _corpus_cache = {
        "count": n, "ids": ids, "docs": docs, "metas": metas,
        "bm25": bm25, "tags": tags, "folders": folders,
        "title_to_paths": title_to_paths,
        "outlinks": outlinks,    # path → [linked titles]
        "backlinks": backlinks,  # title → {paths that link to it}
    }
    return _corpus_cache


def _invalidate_corpus_cache() -> None:
    global _corpus_cache
    _corpus_cache = None


def _folder_matches(file_path: str, folder: str) -> bool:
    """Match a file against a folder filter, always widening to 00-Inbox.

    Inbox is a staging folder, not a topic — captures there haven't been
    filed yet, so by construction they can be about *any* topic. Excluding
    them from a folder-filtered search hides fresh context silently. This
    bit us asking "como activo claude peers?" where infer_filters auto-
    applied `folder=03-Resources/Claude` and the inbox note about the
    claude-peers MCP was dropped.
    """
    if folder in file_path:
        return True
    return file_path.startswith(_CAPTURE_FOLDER + "/")


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
            if (not folder or _folder_matches(m.get("file", ""), folder))
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
    """Lazy-load cross-encoder reranker on the best available accelerator.
    Explicit device picks MPS on Apple Silicon — sentence-transformers'
    auto-detect falls back to CPU in some venvs, costing ~3× on rerank.

    NOTE: fp16 on MPS breaks bge-reranker-v2-m3 (collapses all scores to
    ~0.001, verified empirically on 2026-04-13). Keep fp32.
    """
    global _reranker
    if _reranker is None:
        import torch
        from sentence_transformers import CrossEncoder
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device=device)
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

_PROPER_NOUN_RE = re.compile(r"\b([A-ZÁÉÍÓÚÑ][\wáéíóúñ'\-]{2,})\b")
_PARAPHRASE_TOKEN_RE = re.compile(r"\b[\wáéíóúñ'\-]{3,}\b")


def _extract_proper_nouns(text: str) -> set[str]:
    """Tokens que arrancan con mayúscula, ≥3 chars — candidatos a nombre
    propio. Usado para validar que las paráfrasis preservan entidades.
    """
    return {m.group(1) for m in _PROPER_NOUN_RE.finditer(text)}


def expand_queries(question: str) -> list[str]:
    """Generate 2 paraphrases for multi-query retrieval. Returns [original, p1, p2].

    Nombres propios se preservan vía dos frenos: (1) el prompt se lo pide
    explícito con un anti-ejemplo, (2) un guardrail case-insensitive rechaza
    paráfrasis que droppean tokens propios del original. Sin esto, qwen2.5:3b
    generaba 'el actor Adam Jones' / 'el intérprete X', desviando el recall.
    """
    prompt = (
        "Reformulá esta pregunta de DOS maneras distintas — distintas "
        "palabras clave, mismo sentido. Nombres propios (personas, bandas, "
        "productos, acrónimos) van literales, NO agregues calificativos "
        "como 'el actor X' o 'la banda Y'.\n\n"
        "Ejemplo:\n"
        "  'qué usa adam jones?'\n"
        "  → 'qué equipo usa adam jones?'\n"
        "  → 'cuál es el rig de adam jones?'\n\n"
        "Devolvé SOLO las dos reformulaciones, una por línea. Sin numerar, "
        "sin explicar.\n\n"
        f"Pregunta: {question}"
    )
    try:
        resp = ollama.chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=HELPER_OPTIONS,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        content = resp.message.content
    except Exception:
        return [question]

    lines = [l.strip(" -*·") for l in content.splitlines() if l.strip()]
    # Guardrail case-insensitive: cada paráfrasi debe contener todos los
    # tokens propios del original (lowercased). Evita que "RAG" se pierda
    # aunque qwen lo devuelva como "rag" — el issue no es el case, es que
    # la entidad esté presente.
    original_noun_tokens = {n.lower() for n in _extract_proper_nouns(question)}
    kept: list[str] = []
    for l in lines:
        if l == question:
            continue
        if original_noun_tokens:
            para_tokens = {m.group(0).lower() for m in _PARAPHRASE_TOKEN_RE.finditer(l)}
            if not original_noun_tokens.issubset(para_tokens):
                continue
        kept.append(l)
        if len(kept) >= 2:
            break
    return [question] + kept


_RECENCY_RE = re.compile(
    r"\b(?:recient(?:es?|emente)|[uú]ltim[aos]{1,2}|[uú]ltimamente|hoy|ayer|"
    r"esta\s+semana|este\s+mes|este\s+a[nñ]o|del?\s+mes\s+pasado|reci[eé]n|"
    r"en\s+el\s+[uú]ltimo\s+(?:mes|a[nñ]o))\b",
    re.IGNORECASE,
)


def has_recency_cue(question: str) -> bool:
    """True when the query asks for time-sorted results — triggers a recency
    boost in the reranker composite score.
    """
    return bool(_RECENCY_RE.search(question))


def recency_boost(meta: dict, half_life_days: float = 90.0) -> float:
    """Exponential decay based on `modified` frontmatter. Returns a value in
    [0, 1] — 1 if edited today, 0.5 after `half_life_days`, 0 for very old.
    Falls back to `created` if `modified` is missing.
    """
    stamp = meta.get("modified") or meta.get("created") or ""
    if not stamp:
        return 0.0
    try:
        # Accept ISO strings like "2026-04-13T16:50:16-03:00" or date-only.
        from datetime import datetime as _dt
        dt = _dt.fromisoformat(stamp.replace("Z", "+00:00"))
    except Exception:
        return 0.0
    try:
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
    except Exception:
        return 0.0
    import math
    return math.exp(-math.log(2) * age_days / half_life_days)


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
        if folder and not _folder_matches(f, folder):
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
    "acompañala de su ruta. La ruta figura literal en `[ruta: <VALOR>]` al "
    "inicio de cada chunk — usá el VALOR exacto, sin modificarlo. "
    "Formato: [Título](VALOR). "
    "Ejemplo: si un chunk abre con `[ruta: 02-Areas/Salud/postura.md]`, "
    "escribís [postura](02-Areas/Salud/postura.md). "
    "PROHIBIDO: escribir placeholders como 'ruta/relativa.md', 'path.md', "
    "'nombre.md' u otra etiqueta genérica — siempre la ruta real. "
    "Citá al menos la primera vez que nombres la nota.\n\n"
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
    "REGLA 2 — CITAR RUTA: cada vez que menciones una nota, acompañala de su ruta. "
    "La ruta figura literal en `[ruta: <VALOR>]` al inicio de cada chunk — usá "
    "el VALOR exacto, sin modificarlo. Formato: [Título](VALOR). "
    "Ejemplo: si un chunk abre con `[ruta: 02-Areas/Salud/postura.md]`, "
    "escribís [postura](02-Areas/Salud/postura.md). "
    "PROHIBIDO: escribir placeholders como 'ruta/relativa.md', 'path.md' u otra "
    "etiqueta genérica — siempre la ruta real. Citá al menos la primera vez.\n\n"
    "REGLA 3 — FORMATO: respuesta directa, viñetas cortas, sin intro. Preferí citar "
    "fragmentos verbatim del contexto antes que reformular.\n"
)


def print_query_header(question: str, result: dict, show_question: bool = True) -> None:
    """Render the metadata row (filters, variants, confidence), optionally
    preceded by the question in a panel. In chat the user just typed the
    question, so `show_question=False` skips that to reduce noise.
    """
    console.print()
    if show_question:
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


def find_related(
    col: chromadb.Collection,
    source_metas: list[dict],
    limit: int = 5,
) -> list[tuple[dict, int, str]]:
    """Return up to `limit` notes related to the sources.

    Score combines two signals:
      - shared_tags: count of tags shared with the union of source tags
      - graph_hops: 2× if directly linked from source (outlink), 2× if linked
        TO source (backlink), 0 otherwise. Graph edges are strong signal —
        Obsidian users link deliberately.
    Tie-break: same top-level folder as any source.
    Returns tuples of (meta, composite_score, reason) where reason is
    'tags', 'link', or 'tags+link' for the UI.
    """
    source_paths = {m.get("file", "") for m in source_metas}
    source_titles = {m.get("note", "") for m in source_metas if m.get("note")}
    source_tags: set[str] = set()
    source_roots: set[str] = set()
    for m in source_metas:
        for t in (m.get("tags") or "").split(","):
            t = t.strip()
            if t:
                source_tags.add(t)
        f = m.get("file", "")
        if "/" in f:
            source_roots.add(f.split("/", 1)[0])

    c = _load_corpus(col)

    # Graph: collect paths connected to any source via outlink or backlink.
    linked_paths: dict[str, int] = {}  # path → link weight (1 per edge)
    # Outlinks: source → titles → resolve to paths via title_to_paths
    for src in source_metas:
        src_path = src.get("file", "")
        for target_title in c["outlinks"].get(src_path, []):
            for target_path in c["title_to_paths"].get(target_title, set()):
                if target_path in source_paths:
                    continue
                linked_paths[target_path] = linked_paths.get(target_path, 0) + 1
    # Backlinks: notes that link TO a source title
    for title in source_titles:
        for linker_path in c["backlinks"].get(title, set()):
            if linker_path in source_paths:
                continue
            linked_paths[linker_path] = linked_paths.get(linker_path, 0) + 1

    candidates: dict[str, tuple[dict, int, int, str]] = {}
    for m in c["metas"]:
        f = m.get("file", "")
        if f in source_paths or f in candidates or is_excluded(f):
            continue
        tags = [t.strip() for t in (m.get("tags") or "").split(",") if t.strip()]
        shared = len(source_tags.intersection(tags)) if source_tags else 0
        link_hits = linked_paths.get(f, 0)
        if shared < 2 and link_hits == 0:
            continue
        # Graph weight = 2× per edge — deliberate links > tag co-occurrence.
        score = shared + 2 * link_hits
        reason = (
            "tags+link" if shared >= 2 and link_hits > 0
            else "link" if link_hits > 0 else "tags"
        )
        root_match = 1 if f.split("/", 1)[0] in source_roots else 0
        candidates[f] = (m, score, root_match, reason)

    ranked = sorted(
        candidates.values(),
        key=lambda x: (-x[1], -x[2], x[0].get("file", "")),
    )
    return [(m, score, reason) for m, score, _, reason in ranked[:limit]]


INBOX_FOLDER = "00-Inbox"

# Natural-language save triggers. Two cases:
#   - Strong save verbs (guardá, salvá, agendá) → save on their own.
#   - Neutral verbs (creá, armá, generá, escribí, agregá, añadí) → require
#     the word "nota/notas" within ~5 tokens to avoid false positives on
#     generic questions like "creá un resumen".
_STRONG_SAVE_VERB = r"(?:guard[aá]|salv[aá]|agend[aá])"
_NEUTRAL_SAVE_VERB = r"(?:cre[aá]|agreg[aá]|a[nñ]ad[ií]|escrib[íi]|arm[aá]|gener[aá])"
_SAVE_INTENT_RE = re.compile(
    r"\b(?:"
    rf"{_STRONG_SAVE_VERB}\w*"
    rf"|{_NEUTRAL_SAVE_VERB}\w*(?:\s+\w+){{0,5}}\s+notas?"
    r")\b",
    re.IGNORECASE,
)


# Reindex intent — matched in chat loop so the user can ask to reindex
# without leaving the conversation. Strong verbs trigger alone; "actualizá"
# (weaker, ambiguous) needs a vault/notas/índice object nearby.
_REINDEX_STRONG_RE = re.compile(
    r"\b(?:re[\s-]?index(?:ar|á|a|alo|alos|alas)?|reescane(?:ar|á|a)?|refresc(?:ar|á|a))\b",
    re.IGNORECASE,
)
_REINDEX_WEAK_RE = re.compile(r"\bactualiz(?:ar|á|a)\b", re.IGNORECASE)
_REINDEX_OBJECT_RE = re.compile(
    r"\b(?:vault|[íi]ndice|notas?|todo)\b", re.IGNORECASE,
)
_REINDEX_RESET_RE = re.compile(
    r"\b(?:desde\s+cero|de\s+cero|completo|reset(?:ear|á|a)?|rebuild|scratch|borr(?:ar|á|alo)\s+(?:y|e)\s+rehacer)\b",
    re.IGNORECASE,
)


# Link-finder intent — when the user wants the URL itself, not prose about
# the URL. Patterns: "link/url/enlace/doc(umentación) de|para|a X",
# "donde está/tengo el link/url/doc de X", "dame el link/url/enlace de X".
# Tight on purpose — must not steal "qué dice X sobre Y" or generic queries.
_LINK_INTENT_RE = re.compile(
    r"(?:"
    r"\b(?:link|url|enlace|enlaces|links|urls|doc|docs|documentaci[oó]n)\b\s+(?:de|del|para|a|al|sobre)\s+"
    r"|d[oó]nde\s+(?:est[áa]|tengo|guardo|qued[oó]|hab[íi]a)\s+(?:el\s+|la\s+|los\s+|las\s+)?(?:link|url|enlace|doc|documentaci[oó]n)"
    r"|dame\s+(?:el|los|la|las)\s+(?:link|url|enlace|enlaces|links|urls|doc|docs)"
    r"|busc[áa]\s+(?:el|los|la|las)\s+(?:link|url|enlace|doc)"
    r")",
    re.IGNORECASE,
)


def detect_link_intent(text: str) -> tuple[bool, str | None]:
    """Return (matches, query). When matches, `query` is the residual text
    after stripping the trigger phrase — what to actually search for. None
    means "use the whole input verbatim".
    """
    t = text.strip()
    if t.startswith("/links"):
        rest = t[len("/links"):].strip()
        return True, (rest or None)
    if not _LINK_INTENT_RE.search(t):
        return False, None
    return True, None


def detect_reindex_intent(text: str) -> tuple[bool, bool]:
    """Return (matches, reset). `reset=True` when the user asked for a full
    rebuild (`reset`/`desde cero`/`completo`) — otherwise incremental.
    """
    t = text.strip()
    if t.startswith("/reindex"):
        rest = t[len("/reindex"):].strip().lower()
        return True, bool(_REINDEX_RESET_RE.search(rest)) or rest in ("--reset", "reset")
    has_match = bool(_REINDEX_STRONG_RE.search(t)) or (
        bool(_REINDEX_WEAK_RE.search(t)) and bool(_REINDEX_OBJECT_RE.search(t))
    )
    if not has_match:
        return False, False
    return True, bool(_REINDEX_RESET_RE.search(t))


def detect_save_intent(text: str) -> tuple[bool, str | None]:
    """Return (is_save, optional_title). Title extracted after 'llamada/titulada/nota:'
    if present, else None so save_note derives from body.
    """
    t = text.strip()
    if t.startswith("/save"):
        title = t[len("/save"):].strip() or None
        return True, title
    if not _SAVE_INTENT_RE.search(t):
        return False, None
    # Try to extract a title after common prepositions / markers.
    m = re.search(
        r"(?:titulad[ao]|llamad[ao]|con\s+(?:el\s+)?t[ií]tulo|nota\s*[:\-])\s+['\"]?([^'\"\n]+?)['\"]?\s*$",
        t, re.IGNORECASE,
    )
    return True, (m.group(1).strip() if m else None)


def save_note(
    col: chromadb.Collection,
    title: str | None,
    body: str,
    question: str,
    source_metas: list[dict],
    folder: str = INBOX_FOLDER,
) -> Path:
    """Create a new Markdown note in the vault from a chat response.

    Returns the absolute Path written. Also indexes the new note so it
    becomes retrievable immediately.
    """
    # Title: user-supplied → first non-empty line of body → timestamp.
    if title:
        clean_title = title.strip().strip('"').strip("'")
    else:
        first_line = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        # strip markdown bullets / leading # if present
        first_line = re.sub(r"^[#*\-\d.\s]+", "", first_line).strip()
        clean_title = first_line[:80] or datetime.now().strftime("nota-%Y%m%d-%H%M%S")
    # Sanitise for filename
    safe = re.sub(r"[/\\:\n]", " ", clean_title).strip()
    safe = safe or datetime.now().strftime("nota-%Y%m%d-%H%M%S")
    path = VAULT_PATH / folder / f"{safe}.md"

    # Avoid clobbering an existing note — suffix with a timestamp.
    if path.exists():
        path = path.with_name(
            f"{safe} ({datetime.now().strftime('%H%M%S')}).md"
        )

    # Aggregate tags from source notes (union, capped).
    all_tags: list[str] = []
    for m in source_metas:
        for t in (m.get("tags") or "").split(","):
            t = t.strip()
            if t and t not in all_tags:
                all_tags.append(t)
    all_tags = all_tags[:10]

    now = datetime.now().isoformat(timespec="seconds")
    fm_lines = [
        "---",
        f"created: '{now}'",
        f"modified: '{now}'",
    ]
    if all_tags:
        fm_lines.append("tags:")
        for t in all_tags:
            fm_lines.append(f"- {t}")
    # Source notes for traceability.
    if source_metas:
        fm_lines.append("related:")
        for m in source_metas:
            fm_lines.append(f"- '[[{m.get('note', '')}]]'")
    fm_lines.append(f"source_query: {question!r}")
    fm_lines.append("---")
    frontmatter = "\n".join(fm_lines)

    content = f"{frontmatter}\n\n# {clean_title}\n\n{body.strip()}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    # Index immediately so it's searchable right away.
    try:
        _index_single_file(col, path)
    except Exception:
        pass

    return path


def render_related(related: list[tuple[dict, int, str]]) -> None:
    """Print a compact panel of related notes with score + reason badge."""
    if not related:
        return
    console.print()
    console.print(Rule(title="[dim]Relacionadas[/dim]", style="dim", characters="╌"))
    tbl = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
    tbl.add_column(style="dim", justify="right")   # score
    tbl.add_column(style="dim italic")              # reason badge
    tbl.add_column(style="bold magenta")            # note title
    tbl.add_column(style="cyan dim")                # path
    reason_style = {
        "link": "[bold blue]↔[/bold blue]",
        "tags": "[yellow]#[/yellow]",
        "tags+link": "[bold blue]↔[/bold blue][yellow]#[/yellow]",
    }
    for m, score, reason in related:
        tbl.add_row(
            f"×{score}",
            reason_style.get(reason, ""),
            m.get("note", ""),
            m.get("file", ""),
        )
    console.print(tbl)


# ── WIKILINK SUGGESTIONS ──────────────────────────────────────────────────────
# Densifies the Obsidian graph by surfacing mentions of existing note titles
# that aren't yet `[[wikilinked]]`. Pure regex-by-title scan against the
# corpus' `title_to_paths` index — no LLM, no embeddings. Skips frontmatter,
# code blocks, existing wikilinks, markdown links and HTML tags so we never
# wrap text the user already linked elsewhere.

_WIKILINK_SKIP_PATTERNS = [
    re.compile(r"```.*?```", re.DOTALL),                  # fenced code
    re.compile(r"`[^`\n]+`"),                              # inline code
    re.compile(r"!?\[\[[^\]]+\]\]"),                       # existing wikilinks (incl. ![[embed]])
    re.compile(r"\[[^\]\n]+\]\([^\)\n]+\)"),               # markdown links
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
    col: chromadb.Collection,
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


def apply_wikilink_suggestions(note_path: str, suggestions: list[dict]) -> int:
    """Wrap each proposed mention with `[[ ]]`. Returns the count actually
    applied. Iterates from highest offset to lowest so earlier offsets stay
    valid. Defensive: re-checks the literal text at offset before substituting
    so a stale suggestion (file edited mid-flight) is silently skipped.
    """
    full = (VAULT_PATH / note_path).resolve()
    try:
        full.relative_to(VAULT_PATH.resolve())
    except ValueError:
        return 0
    if not full.is_file() or not suggestions:
        return 0
    raw = full.read_text(encoding="utf-8", errors="ignore")
    by_offset = sorted(suggestions, key=lambda s: s["char_offset"], reverse=True)
    applied = 0
    for s in by_offset:
        start = s["char_offset"]
        title = s["title"]
        end = start + len(title)
        if raw[start:end] != title:
            continue
        raw = raw[:start] + f"[[{title}]]" + raw[end:]
        applied += 1
    if applied:
        full.write_text(raw, encoding="utf-8")
    return applied


# ── DUPLICATE DETECTION ──────────────────────────────────────────────────────
# Pairwise cosine over per-note centroid embeddings (mean of the note's chunks
# in the main collection). Numpy for the O(N²/2) sweep — ~500 notes finishes
# in well under a second. A "centroid" is a coarse fingerprint, intentional:
# we want to surface notes that broadly cover the same topic, not just
# notes that share a single phrase.

def _note_centroids(
    col: chromadb.Collection,
    folder: str | None = None,
) -> tuple[list[str], list[dict], "np.ndarray"]:
    """Group all chunks by file, average the embeddings, L2-normalise.

    Returns (file_paths, first_meta_per_file, centroids_matrix). Files with
    zero chunks (shouldn't happen) are skipped silently.
    """
    import numpy as np
    data = col.get(include=["embeddings", "metadatas"])
    by_file: dict[str, dict] = {}
    for emb, meta in zip(data["embeddings"], data["metadatas"]):
        f = meta.get("file", "")
        if not f:
            continue
        if folder and not (f == folder or f.startswith(folder.rstrip("/") + "/")):
            continue
        if f not in by_file:
            by_file[f] = {"embeds": [], "meta": meta}
        by_file[f]["embeds"].append(emb)
    files = sorted(by_file.keys())
    if not files:
        return [], [], np.zeros((0, 0), dtype="float32")
    arr = np.stack([
        np.mean(np.asarray(by_file[f]["embeds"], dtype="float32"), axis=0)
        for f in files
    ])
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    metas = [by_file[f]["meta"] for f in files]
    return files, metas, arr


def find_duplicate_notes(
    col: chromadb.Collection,
    threshold: float = 0.85,
    folder: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Pairs of notes with centroid cosine ≥ threshold.

    Returns descending-by-similarity list of
    {a_path, a_note, b_path, b_note, similarity, snippet_a, snippet_b}.
    The snippets are the first ~200 chars of each note's body for at-a-glance
    comparison in the renderer.
    """
    import numpy as np
    files, metas, arr = _note_centroids(col, folder)
    if len(files) < 2:
        return []
    sims = arr @ arr.T
    # Ignore self and lower triangle: only count each pair once. We mask via
    # boolean — overwriting `sims` with `np.triu(sims, k=1)` would zero out
    # the lower triangle, which a threshold of <= 0 would then accept as a
    # spurious "match".
    mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
    rows, cols = np.where(mask & (sims >= threshold))
    pairs: list[dict] = []
    for r, c in zip(rows, cols):
        a, b = files[int(r)], files[int(c)]
        pa = (VAULT_PATH / a)
        pb = (VAULT_PATH / b)
        snip_a = pa.read_text(encoding="utf-8", errors="ignore")[:200] if pa.is_file() else ""
        snip_b = pb.read_text(encoding="utf-8", errors="ignore")[:200] if pb.is_file() else ""
        pairs.append({
            "a_path": a,
            "b_path": b,
            "a_note": metas[int(r)].get("note", ""),
            "b_note": metas[int(c)].get("note", ""),
            "similarity": round(float(sims[r, c]), 3),
            "snippet_a": re.sub(r"\s+", " ", snip_a).strip(),
            "snippet_b": re.sub(r"\s+", " ", snip_b).strip(),
        })
    pairs.sort(key=lambda p: -p["similarity"])
    return pairs[:limit]


# ── SURFACE: proactive bridge builder ────────────────────────────────────────
# Pares semánticamente cercanos pero topológicamente lejanos en el grafo de
# wikilinks. Surface propone conexiones que el usuario no hizo — generalmente
# porque las dos notas se escribieron en momentos distintos. Composición pura
# de `_note_centroids` + grafo cacheado en `_load_corpus` + LLM helper para
# una oración de "por qué están conectadas".

SURFACE_LOG_PATH = Path.home() / ".local/share/obsidian-rag/surface.jsonl"
SURFACE_SKIP_FOLDERS = ("00-Inbox/", "04-Archive/", "05-Reviews/")


def _build_graph_adj(corpus: dict) -> dict[str, set[str]]:
    """Adyacencia no dirigida del grafo de wikilinks: path → set de paths linkeados.
    Compone outlinks (path → títulos) con title_to_paths (título → paths).
    Cada edge existe en ambas direcciones para tratar el grafo como no dirigido.
    """
    adj: dict[str, set[str]] = {}
    title_to_paths = corpus["title_to_paths"]
    for src_path, titles in corpus["outlinks"].items():
        for t in titles:
            for tgt in title_to_paths.get(t, ()):
                if tgt == src_path:
                    continue
                adj.setdefault(src_path, set()).add(tgt)
                adj.setdefault(tgt, set()).add(src_path)
    return adj


def _hop_set(adj: dict[str, set[str]], start: str, hops: int) -> set[str]:
    """BFS hasta `hops` saltos desde `start`. Retorna el set visitado incluyendo start.
    Usado para determinar distancia mínima: b no está en hop_set(a, N-1) ⇒ dist ≥ N.
    """
    if hops <= 0:
        return {start}
    seen = {start}
    frontier = {start}
    for _ in range(hops):
        nxt: set[str] = set()
        for n in frontier:
            nxt |= adj.get(n, set()) - seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


_SURFACE_MOC_TITLE_RE = re.compile(r"^(MOC|Index|Map)(\s|$|[-_])", re.IGNORECASE)


def _is_moc_note(meta: dict) -> bool:
    """Heurística MOC: título empieza con MOC/Index/Map, o tag #moc, o la nota
    lleva el mismo nombre que su carpeta (convención folder-index).
    """
    title = (meta.get("note") or "").strip()
    if _SURFACE_MOC_TITLE_RE.match(title):
        return True
    tags = {t.strip().lower() for t in (meta.get("tags") or "").split(",") if t.strip()}
    if "moc" in tags:
        return True
    path = meta.get("file") or ""
    parts = path.split("/")
    if len(parts) >= 2 and parts[-1] == f"{parts[-2]}.md":
        return True
    return False


def _note_age_days(meta: dict) -> float | None:
    """Edad en días desde `created` o `modified` del frontmatter.
    None si no hay timestamp parseable — el caller decide qué hacer.
    """
    stamp = meta.get("created") or meta.get("modified") or ""
    if not stamp:
        return None
    try:
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except Exception:
        return None
    try:
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        return max(0.0, (now - dt).total_seconds() / 86400.0)
    except Exception:
        return None


def find_surface_bridges(
    col: chromadb.Collection,
    sim_threshold: float = 0.78,
    min_hops: int = 3,
    top: int = 5,
    skip_young_days: int = 7,
) -> list[dict]:
    """Pares de notas semánticamente cercanas pero lejanas en el grafo.
    Propone los puentes que el usuario no hizo.

    Filtros (AND):
      - cosine(centroid_a, centroid_b) ≥ sim_threshold
      - distancia en el grafo ≥ min_hops  (b ∉ (min_hops−1)-hop de a)
      - ninguna nota en 00-Inbox / 04-Archive / 05-Reviews
      - ninguna nota es MOC (título, tag o folder-index)
      - el par NO comparte ≥2 tags (la conexión ya es explícita vía tags)
      - ambas notas tienen edad ≥ skip_young_days (notas frescas siguen evolucionando)

    Retorna los `top` mejores pares por similitud descendente. Cada dict incluye
    snippets (primeros ~800 chars del body, sin frontmatter) para que el caller
    pueda alimentar un LLM opcional que genere la oración de "por qué conectan".
    """
    import numpy as np
    corpus = _load_corpus(col)
    files, metas, arr = _note_centroids(col)
    if len(files) < 2:
        return []
    adj = _build_graph_adj(corpus)

    def _eligible(idx: int) -> bool:
        p = files[idx]
        if any(p.startswith(pref) for pref in SURFACE_SKIP_FOLDERS):
            return False
        if _is_moc_note(metas[idx]):
            return False
        age = _note_age_days(metas[idx])
        if age is not None and age < skip_young_days:
            return False
        return True

    elig = [i for i in range(len(files)) if _eligible(i)]
    if len(elig) < 2:
        return []

    sims = arr @ arr.T
    hop_cache: dict[str, set[str]] = {}

    def _hops(p: str) -> set[str]:
        if p not in hop_cache:
            hop_cache[p] = _hop_set(adj, p, min_hops - 1)
        return hop_cache[p]

    candidates: list[dict] = []
    for ii, i in enumerate(elig):
        p_i = files[i]
        hops_i = _hops(p_i)
        row = sims[i]
        tags_i = {t.strip() for t in (metas[i].get("tags") or "").split(",") if t.strip()}
        for j in elig[ii + 1:]:
            s = float(row[j])
            if s < sim_threshold:
                continue
            p_j = files[j]
            if p_j in hops_i:
                continue
            tags_j = {t.strip() for t in (metas[j].get("tags") or "").split(",") if t.strip()}
            shared = tags_i & tags_j
            if len(shared) >= 2:
                continue
            candidates.append({
                "a_path": p_i, "b_path": p_j,
                "a_note": metas[i].get("note", ""),
                "b_note": metas[j].get("note", ""),
                "similarity": round(s, 3),
                "shared_tags": sorted(shared),
                "a_age_days": _note_age_days(metas[i]),
                "b_age_days": _note_age_days(metas[j]),
            })

    candidates.sort(key=lambda p: -p["similarity"])
    top_pairs = candidates[:top]

    # Snippets: una lectura por path único, sin frontmatter, colapsando whitespace.
    needed = {p["a_path"] for p in top_pairs} | {p["b_path"] for p in top_pairs}
    snippets: dict[str, str] = {}
    for rel in needed:
        full = VAULT_PATH / rel
        if not full.is_file():
            snippets[rel] = ""
            continue
        try:
            body = full.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            snippets[rel] = ""
            continue
        if body.startswith("---\n"):
            end = body.find("\n---\n", 4)
            if end > 0:
                body = body[end + 5:]
        snippets[rel] = re.sub(r"\s+", " ", body[:800]).strip()
    for p in top_pairs:
        p["a_snippet"] = snippets.get(p["a_path"], "")
        p["b_snippet"] = snippets.get(p["b_path"], "")
    return top_pairs


def _surface_generate_reason(pair: dict) -> str:
    """Una oración en español explicando la conexión. '' si falla o si el modelo
    declara "sin conexión clara" — esos pares quedan como candidatos silenciosos
    (buenos para rankeo, ruidosos para mostrar).

    Usa el chat model (command-r), no el helper: juzgar conexión entre notas es
    una tarea de síntesis donde qwen2.5:3b se va a lo genérico ("ambas hablan
    de música") o refuses con "sin conexión clara" en pares donde command-r sí
    ve la temática real. Mismo criterio que el contradiction radar.
    """
    prompt = (
        "Tenés dos notas de un vault personal. En UNA oración en español "
        "(≤25 palabras), decí por qué están conectadas a nivel de contenido. "
        "No inventes datos; si no hay conexión clara, respondé exactamente "
        "'sin conexión clara'.\n\n"
        f"NOTA A — {pair['a_note']}:\n{pair.get('a_snippet', '')[:600]}\n\n"
        f"NOTA B — {pair['b_note']}:\n{pair.get('b_snippet', '')[:600]}\n\n"
        "CONEXIÓN:"
    )
    try:
        resp = ollama.chat(
            model=resolve_chat_model(),
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "top_p": 1, "seed": 42,
                     "num_ctx": 2048, "num_predict": 80},
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        reason = (resp.message.content or "").strip().split("\n", 1)[0].strip()
    except Exception:
        return ""
    if not reason or "sin conexión clara" in reason.lower():
        return ""
    reason = reason.strip(".") + "."
    return reason if len(reason) > 4 else ""


def _surface_log_run(summary: dict, pairs: list[dict]) -> None:
    """Append-only log: una línea `surface_run` + N líneas `surface_pair`.
    Mismo timestamp en todas para poder agrupar la corrida al leer."""
    try:
        SURFACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().isoformat(timespec="seconds")
        with SURFACE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"ts": ts, "cmd": "surface_run", **summary},
                ensure_ascii=False,
            ) + "\n")
            for p in pairs:
                f.write(json.dumps(
                    {"ts": ts, "cmd": "surface_pair", **p},
                    ensure_ascii=False,
                ) + "\n")
    except Exception:
        pass


def _suggest_tags_for_note(
    col: chromadb.Collection,
    body: str,
    note_title: str,
    max_tags: int = 6,
) -> list[str]:
    """Pure helper: ask the helper LLM to pick tags from existing vault vocab.
    Returns picked list (may be empty). Shared by `rag autotag` and `rag inbox`.
    """
    c = _load_corpus(col)
    vocab = sorted(c["tags"])
    if not vocab or not body.strip():
        return []
    prompt = (
        "Sos un asistente que etiqueta notas personales. Elegí entre 3 y "
        f"{max_tags} tags DEL VOCABULARIO EXISTENTE que mejor describan esta "
        "nota. NO inventes tags nuevos. Devolvé SOLO una lista YAML de "
        "strings, sin explicación.\n\n"
        f"VOCABULARIO ({len(vocab)} tags): {', '.join(vocab)}\n\n"
        f"TÍTULO: {note_title}\n\n"
        f"CONTENIDO:\n{body}\n\n"
        "TAGS:"
    )
    try:
        resp = ollama.chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=HELPER_OPTIONS,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        answer = resp.message.content.strip()
    except Exception:
        return []
    vocab_set = {t.lower() for t in vocab}
    picked: list[str] = []
    for line in answer.splitlines():
        line = line.strip().strip("-*[]").strip().strip(",").strip("'\"")
        if not line:
            continue
        for tok in re.split(r"[,\s]+", line):
            tok = tok.strip("#'\"").lower()
            if tok in vocab_set and tok not in picked:
                picked.append(tok)
        if len(picked) >= max_tags:
            break
    return picked


def _apply_frontmatter_tags(note_path: Path, merged_tags: list[str]) -> bool:
    """Rewrite the note's frontmatter `tags:` block to `merged_tags`.
    Preserves the rest of the YAML verbatim. Returns True on success.
    """
    if not note_path.is_file():
        return False
    raw = note_path.read_text(encoding="utf-8", errors="ignore")
    if raw.startswith("---\n"):
        end = raw.find("\n---\n", 4)
        if end < 0:
            return False
        fm_text = raw[4:end]
        rest = raw[end + 5:]
        new_fm_lines: list[str] = []
        in_tag_block = False
        for line in fm_text.splitlines():
            if in_tag_block and re.match(r"^\s*-\s+", line):
                continue
            in_tag_block = False
            if re.match(r"^tags\s*:", line):
                in_tag_block = True
                continue
            new_fm_lines.append(line)
        new_fm_lines.append("tags:")
        for t in merged_tags:
            new_fm_lines.append(f"- {t}")
        new_raw = "---\n" + "\n".join(new_fm_lines) + "\n---\n" + rest
    else:
        fm_block = (
            "---\ntags:\n" + "\n".join(f"- {t}" for t in merged_tags) + "\n---\n\n"
        )
        new_raw = fm_block + raw
    note_path.write_text(new_raw, encoding="utf-8")
    return True


def _suggest_folder_for_note(
    col: chromadb.Collection,
    note_path: str,
    k: int = 8,
    skip_folder_prefix: str = "00-",
) -> tuple[str, float]:
    """Propose a destination folder by mode of folders among the K most
    semantically similar OTHER notes. Excludes Inbox-style folders (any path
    starting with `skip_folder_prefix`) so we recommend a real home, not "stay
    where you are".

    Returns (folder, confidence) where confidence is the share of neighbors
    that voted for the winner. ("", 0.0) if nothing usable.
    """
    full = (VAULT_PATH / note_path).resolve()
    try:
        full.relative_to(VAULT_PATH.resolve())
    except ValueError:
        return ("", 0.0)
    if not full.is_file():
        return ("", 0.0)
    raw = full.read_text(encoding="utf-8", errors="ignore")
    text = clean_md(raw)[:3000].strip()
    if not text or col.count() == 0:
        return ("", 0.0)
    try:
        q_embed = embed([text])[0]
    except Exception:
        return ("", 0.0)
    n = min(k * 4, col.count())
    res = col.query(
        query_embeddings=[q_embed], n_results=n, include=["metadatas"],
    )
    folders: list[str] = []
    for m in res["metadatas"][0]:
        if m.get("file") == note_path:
            continue
        f = m.get("folder") or ""
        if not f or f.startswith(skip_folder_prefix):
            continue
        folders.append(f)
        if len(folders) >= k:
            break
    if not folders:
        return ("", 0.0)
    from collections import Counter
    best, count = Counter(folders).most_common(1)[0]
    return (best, round(count / len(folders), 3))


def triage_inbox_note(
    col: chromadb.Collection,
    note_path: str,
    max_tags: int = 5,
    dupe_threshold: float = 0.85,
) -> dict:
    """Compose all triage signals for one Inbox note: destination folder, tags
    from vocabulary, wikilink suggestions, near-duplicate flags. Returns a
    plain dict the CLI renderer + the eventual `--apply` path consume.
    """
    full = (VAULT_PATH / note_path).resolve()
    if not full.is_file():
        return {"path": note_path, "error": "not found"}
    raw = full.read_text(encoding="utf-8", errors="ignore")
    fm = parse_frontmatter(raw)
    current_tags = [str(t) for t in (fm.get("tags") or []) if t]
    body = clean_md(raw)[:3000]
    folder, fconf = _suggest_folder_for_note(col, note_path)
    tags = _suggest_tags_for_note(col, body, full.stem, max_tags=max_tags)
    new_tags = [t for t in tags if t not in current_tags]
    wikilinks = find_wikilink_suggestions(col, note_path, max_per_note=10)
    dupes = find_near_duplicates_for(
        col, note_path, threshold=dupe_threshold, limit=3,
    )
    return {
        "path": note_path,
        "current_folder": str(full.parent.relative_to(VAULT_PATH)),
        "folder_suggested": folder,
        "folder_confidence": fconf,
        "tags_current": current_tags,
        "tags_suggested": tags,
        "tags_new": new_tags,
        "wikilinks": wikilinks,
        "duplicates": dupes,
    }


# ── FILING: asistente de inbox que propone destino + upward-link ─────────────
# Fase 1 es dry-run puro: calcula propuestas y las loguea a filing.jsonl. No
# mueve archivos. Los datos acumulados (pares nota→folder aceptados/rechazados)
# alimentan un ranker en fase 2 cuando sumemos confirmación interactiva.

FILING_LOG_PATH = Path.home() / ".local/share/obsidian-rag/filing.jsonl"
# Umbrales de confianza para la etiqueta visual (la decisión final siempre es
# del usuario; el sistema solo colorea):
#   ≥ 0.55  → firm      (≥55% de los K vecinos vota la misma carpeta)
#   0.35-0.55 → tentative
#   < 0.35  → low       (señal insuficiente, sugerir revisión manual)
FILING_CONFIDENCE_FIRM = 0.55
FILING_CONFIDENCE_TENTATIVE = 0.35


def _top_k_neighbors(
    col: chromadb.Collection,
    note_path: str,
    k: int = 8,
    skip_folder_prefix: str = "00-",
) -> list[tuple[dict, float]]:
    """Top-k semantic neighbors (meta, similarity) dedupeado por file, excluye
    la nota misma y cualquier path en Inbox-style folders (00-*).

    Chroma cosine-space: distance = 1 - cos_sim → similarity = max(0, 1-dist).
    """
    full = (VAULT_PATH / note_path).resolve()
    try:
        full.relative_to(VAULT_PATH.resolve())
    except ValueError:
        return []
    if not full.is_file() or col.count() == 0:
        return []
    text = clean_md(full.read_text(encoding="utf-8", errors="ignore"))[:3000].strip()
    if not text:
        return []
    try:
        q_embed = embed([text])[0]
    except Exception:
        return []
    n = min(k * 4, col.count())
    res = col.query(
        query_embeddings=[q_embed], n_results=n,
        include=["metadatas", "distances"],
    )
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    out: list[tuple[dict, float]] = []
    seen_files: set[str] = set()
    for m, d in zip(metas, dists):
        f = m.get("file", "")
        if not f or f == note_path or f in seen_files:
            continue
        if f.startswith(skip_folder_prefix):
            continue
        seen_files.add(f)
        sim = max(0.0, 1.0 - float(d))
        out.append((m, sim))
        if len(out) >= k:
            break
    return out


def _infer_upward_link(
    neighbors: list[tuple[dict, float]],
) -> tuple[str, str]:
    """De los vecinos top-k, elegir target para el upward-link.

    Preferencia: primer MOC detectado (via _is_moc_note, que ya maneja
    title/tag/folder-index). Si no hay MOC entre los vecinos, devolver el
    top-1 como link horizontal (menos ideal pero no deja la nota huérfana).
    Retorna (title, kind) con kind ∈ {"moc", "neighbor", ""}.
    """
    for m, _ in neighbors:
        if _is_moc_note(m):
            return (m.get("note", ""), "moc")
    if neighbors:
        return (neighbors[0][0].get("note", ""), "neighbor")
    return ("", "")


def build_filing_proposal(
    col: chromadb.Collection,
    note_path: str,
    k: int = 8,
    history: list[dict] | None = None,
) -> dict:
    """Compone folder sugerido + upward-link + neighbors para una nota del
    Inbox. Retorna dict plano que CLI + logger consumen igual.

    Compone tres señales:
      - baseline: `_suggest_folder_for_note` (mode-voting sobre vecinos del
        corpus, fuera de Inbox). Lo que ya teníamos en fase 1/2.
      - upward-link: `_infer_upward_link` (MOC entre neighbors o top-1).
      - personalización (fase 3): si hay ≥FILING_PERSONALIZE_MIN_HISTORY
        decisiones pasadas, votamos por k-NN contra ellas y comparamos con
        baseline. Tres estados:
          * "agreed"           — ambos coinciden → boost de confidence.
          * "personalized"     — disagree y la señal personal es más fuerte
                                  → la propuesta cambia.
          * "baseline+history" — disagree pero baseline gana → no cambia
                                  pero marcamos que hay señal histórica.
        Cold-start (history < umbral) → simplemente "baseline".

    `history` opcional: el caller puede precargarlo (ej. el CLI lo hace
    una vez antes del loop) para evitar releer filing.jsonl por nota.
    """
    full = (VAULT_PATH / note_path).resolve()
    if not full.is_file():
        return {"path": note_path, "error": "not_found"}

    neighbors = _top_k_neighbors(col, note_path, k=k)
    base_folder, base_confidence = _suggest_folder_for_note(col, note_path, k=k)
    upward_title, upward_kind = _infer_upward_link(neighbors)

    folder = base_folder
    confidence = base_confidence
    source = "baseline"
    evidence: list[dict] = []

    if history is None:
        history = _load_filing_decisions()
    if len(history) >= FILING_PERSONALIZE_MIN_HISTORY:
        q_embed = _embed_note_body(note_path)
        votes, evidence = _personalized_folder_vote(col, q_embed, history)
        if votes:
            personal_folder = max(votes, key=votes.get)
            total = sum(votes.values())
            personal_conf = (votes[personal_folder] / total) if total else 0.0
            if personal_folder == base_folder and base_folder:
                source = "agreed"
                confidence = min(0.99, base_confidence + FILING_AGREE_BOOST)
            elif personal_conf > base_confidence:
                folder = personal_folder
                confidence = round(personal_conf, 3)
                source = "personalized"
            else:
                source = "baseline+history"

    return {
        "path": note_path,
        "note": full.stem,
        "folder": folder,
        "confidence": confidence,
        "source": source,
        "evidence": evidence,
        "upward_title": upward_title,
        "upward_kind": upward_kind,
        "neighbors": [
            {
                "path": m.get("file", ""),
                "note": m.get("note", ""),
                "sim": round(s, 3),
            }
            for m, s in neighbors[:5]
        ],
    }


def _filing_log_proposal(proposal: dict, decision: str | None = None) -> None:
    """Append-only log. Una línea por propuesta. `decision` es opcional —
    fase 1 no lo setea (dry-run), fase 2 lo setea con accept/reject/edit/skip
    para alimentar el ranker.
    """
    try:
        FILING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().isoformat(timespec="seconds")
        event = {"ts": ts, "cmd": "filing_proposal", **proposal}
        if decision is not None:
            event["decision"] = decision
        with FILING_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ── FILING fase 3: personalización por k-NN sobre decisiones pasadas ─────────
# La idea: cuando proponemos folder para una nota nueva, miramos las notas que
# YA filaste (decision=accept/edit en filing.jsonl). Si las notas más
# semánticamente parecidas a la nueva fueron filadas a un folder X, sesgamos
# la propuesta hacia X. Sin training, sin features manuales — pura memoria
# de tu comportamiento sobre embeddings que ya tenemos.

FILING_PERSONALIZE_MIN_HISTORY = 5      # below this → fallback a baseline puro
FILING_PERSONALIZE_TOP_K = 5            # vecinos en el espacio de decisiones pasadas
FILING_PERSONALIZE_MIN_SIM = 0.30       # piso de cosine para que una decisión "cuente"
FILING_AGREE_BOOST = 0.15               # bump de confidence cuando baseline+personalized coinciden


def _load_filing_decisions(limit: int = 500) -> list[dict]:
    """Lee las últimas N decisiones positivas (accept/edit) de filing.jsonl.

    Solo incluye las que tienen `applied_to` (sea porque se aplicaron en fase
    2 o porque ya las migramos). reject/skip/error/quit se descartan — la
    fase 3 personaliza desde lo que validaste, no desde lo que rechazaste
    (eso vendría en una fase posterior con un ranker logístico real).
    """
    if not FILING_LOG_PATH.is_file():
        return []
    try:
        lines = FILING_LOG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict] = []
    for line in reversed(lines):
        if len(out) >= limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("decision") not in ("accept", "edit"):
            continue
        applied = e.get("applied_to")
        if not applied:
            continue
        out.append({
            "applied_to": applied,
            "target_folder": str(Path(applied).parent),
            "decision": e["decision"],
            "ts": e.get("ts", ""),
        })
    return out


def _embed_note_body(note_path: str) -> list[float] | None:
    """Embed del cuerpo de la nota para lookups de similitud. None si no
    se puede leer o está vacía. Espejo de lo que hace `_top_k_neighbors`
    internamente — extraído acá para que personalización lo reuse.
    """
    full = (VAULT_PATH / note_path).resolve()
    try:
        full.relative_to(VAULT_PATH.resolve())
    except ValueError:
        return None
    if not full.is_file():
        return None
    text = clean_md(full.read_text(encoding="utf-8", errors="ignore"))[:3000].strip()
    if not text:
        return None
    try:
        return embed([text])[0]
    except Exception:
        return None


def _personalized_folder_vote(
    col: chromadb.Collection,
    q_embed: list[float] | None,
    history: list[dict],
    top_k: int = FILING_PERSONALIZE_TOP_K,
) -> tuple[dict[str, float], list[dict]]:
    """Voto k-NN sobre decisiones pasadas. Para la nota query (q_embed),
    encontrar las top_k decisiones más similares y sumar similitud por
    target_folder.

    Retorna (votes, evidence):
      - votes: {folder: similarity_sum}
      - evidence: lista [{applied_to, target_folder, sim}] de los vecinos
        que efectivamente contribuyeron (para mostrar en UI: "3 similares
        en este mismo folder").

    Skip silencioso si no hay history, no hay query embedding, o ningún
    `applied_to` está en los centroides (ej. notas borradas después).
    """
    if not history or q_embed is None:
        return {}, []
    import numpy as np
    files, _metas, arr = _note_centroids(col)
    file_to_idx = {f: i for i, f in enumerate(files)}

    embs: list = []
    info: list[dict] = []
    for d in history:
        idx = file_to_idx.get(d["applied_to"])
        if idx is None:
            continue
        embs.append(arr[idx])
        info.append(d)

    if not embs:
        return {}, []

    decision_arr = np.asarray(embs, dtype=np.float32)
    q = np.asarray(q_embed, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return {}, []
    q = q / n
    # decision_arr ya viene L2-normalizado de _note_centroids → dot = cosine.
    sims = decision_arr @ q

    order = np.argsort(sims)[::-1]
    votes: dict[str, float] = {}
    evidence: list[dict] = []
    for i in order[:top_k]:
        sim = float(sims[i])
        if sim < FILING_PERSONALIZE_MIN_SIM:
            break
        folder = info[i]["target_folder"]
        votes[folder] = votes.get(folder, 0.0) + sim
        evidence.append({
            "applied_to": info[i]["applied_to"],
            "target_folder": folder,
            "sim": round(sim, 3),
        })
    return votes, evidence


# ── FILING apply + undo ──────────────────────────────────────────────────────

FILING_BATCHES_DIR = Path.home() / ".local/share/obsidian-rag/filing_batches"
# Token que marca el upward-link appendeado al pie de la nota — permite al
# undo detectarlo y removerlo limpio. Cualquier edición manual del usuario
# queda por encima del token intacta.
FILING_UPWARD_MARKER = "<!-- rag-file:upward -->"


def _append_upward_link(note_full_path: Path, upward_title: str) -> bool:
    """Agrega `\\n\\n---\\n{marker}\\n↑ [[Title]]\\n` al final de la nota.
    Idempotente: si ya existe un bloque con el marker, lo reemplaza en vez
    de duplicar. Retorna True si se escribió.
    """
    if not note_full_path.is_file() or not upward_title:
        return False
    raw = note_full_path.read_text(encoding="utf-8", errors="ignore")
    block = f"\n\n---\n{FILING_UPWARD_MARKER}\n↑ [[{upward_title}]]\n"
    if FILING_UPWARD_MARKER in raw:
        # Reemplazar el bloque viejo (desde el --- previo al marker hasta EOF).
        idx = raw.find(FILING_UPWARD_MARKER)
        # Buscar el --- que precede al marker (ancla del bloque).
        sep = raw.rfind("\n---\n", 0, idx)
        if sep >= 0:
            raw = raw[:sep] + block
        else:
            raw = raw.rstrip() + block
    else:
        raw = raw.rstrip() + block
    note_full_path.write_text(raw, encoding="utf-8")
    return True


def _remove_upward_link(note_full_path: Path) -> bool:
    """Remueve el bloque agregado por _append_upward_link. Usado por undo.
    Retorna True si había algo que remover.
    """
    if not note_full_path.is_file():
        return False
    raw = note_full_path.read_text(encoding="utf-8", errors="ignore")
    idx = raw.find(FILING_UPWARD_MARKER)
    if idx < 0:
        return False
    sep = raw.rfind("\n---\n", 0, idx)
    if sep < 0:
        return False
    note_full_path.write_text(raw[:sep].rstrip() + "\n", encoding="utf-8")
    return True


def _apply_filing_move(
    col: chromadb.Collection,
    src_rel: str,
    target_folder: str,
    upward_title: str,
) -> dict:
    """Ejecuta el move + upward-link + reindex para UNA nota.

    Retorna entry del batch log: {src, dst, upward_title, upward_written}.
    Levanta si target folder queda fuera del vault (seguridad path traversal).
    """
    src = (VAULT_PATH / src_rel).resolve()
    src.relative_to(VAULT_PATH.resolve())   # ValueError si escapa
    if not src.is_file():
        raise FileNotFoundError(src_rel)
    target_dir = (VAULT_PATH / target_folder).resolve()
    target_dir.relative_to(VAULT_PATH.resolve())
    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / src.name
    if dst.exists():
        raise FileExistsError(str(dst.relative_to(VAULT_PATH)))
    # Mover con shutil para manejar cross-device edge cases (iCloud puede
    # montarse distinto que el home en ciertos contextos).
    import shutil
    shutil.move(str(src), str(dst))
    written = _append_upward_link(dst, upward_title) if upward_title else False
    # Reindex: el hook del ambient agent se dispara sobre saves del Inbox,
    # pero acá venimos del Inbox hacia afuera — skip_contradict para no
    # gatillar un check O(n²) costoso en un apply-batch.
    try:
        _index_single_file(col, dst, skip_contradict=True)
    except Exception:
        pass
    return {
        "src": src_rel,
        "dst": str(dst.relative_to(VAULT_PATH)),
        "upward_title": upward_title,
        "upward_written": written,
    }


def _write_filing_batch(entries: list[dict]) -> Path | None:
    """Persiste un batch de moves para permitir undo atómico. Un archivo
    JSONL por corrida, nombrado por timestamp. Retorna el path o None si no
    hay entries válidos.
    """
    if not entries:
        return None
    FILING_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = FILING_BATCHES_DIR / f"{ts}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return path


def _last_filing_batch() -> Path | None:
    """Devuelve el batch más reciente (por mtime). None si no hay ninguno."""
    if not FILING_BATCHES_DIR.is_dir():
        return None
    batches = sorted(
        FILING_BATCHES_DIR.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return batches[0] if batches else None


def _rollback_filing_batch(col: chromadb.Collection, batch_path: Path) -> list[dict]:
    """Revierte cada entry: remueve upward-link, mueve back del dst al src.
    Retorna lista de resultados por entry: {src, dst, ok, error?}.

    No borra el batch log hasta confirmar: el caller decide qué hacer con él
    (el CLI lo rename a .undone para trazabilidad).
    """
    import shutil
    results: list[dict] = []
    with batch_path.open("r", encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]
    # Revertir en orden inverso por seguridad (si hubo dependencias de path).
    for e in reversed(entries):
        src_rel = e["src"]
        dst_rel = e["dst"]
        r = {"src": src_rel, "dst": dst_rel, "ok": False}
        try:
            dst_full = (VAULT_PATH / dst_rel).resolve()
            dst_full.relative_to(VAULT_PATH.resolve())
            if not dst_full.is_file():
                r["error"] = "dst no existe"
                results.append(r)
                continue
            if e.get("upward_written"):
                _remove_upward_link(dst_full)
            src_full = (VAULT_PATH / src_rel).resolve()
            src_full.relative_to(VAULT_PATH.resolve())
            src_full.parent.mkdir(parents=True, exist_ok=True)
            if src_full.exists():
                r["error"] = "src ocupado — no overwrite"
                results.append(r)
                continue
            shutil.move(str(dst_full), str(src_full))
            try:
                _index_single_file(col, src_full, skip_contradict=True)
            except Exception:
                pass
            r["ok"] = True
        except Exception as ex:
            r["error"] = str(ex)
        results.append(r)
    return results


def find_near_duplicates_for(
    col: chromadb.Collection,
    note_path: str,
    threshold: float = 0.80,
    limit: int = 5,
) -> list[dict]:
    """Notes whose centroid is most similar to `note_path`'s centroid.

    Used by inbox triage to flag "this incoming note may already exist".
    Lower default threshold than `find_duplicate_notes` so partial overlaps
    surface as warnings rather than be silently missed.
    """
    import numpy as np
    files, metas, arr = _note_centroids(col)
    if note_path not in files:
        return []
    idx = files.index(note_path)
    sims = arr @ arr[idx]
    out: list[dict] = []
    for i, s in enumerate(sims):
        if i == idx:
            continue
        s = float(s)
        if s < threshold:
            continue
        out.append({
            "path": files[i],
            "note": metas[i].get("note", ""),
            "similarity": round(s, 3),
        })
    out.sort(key=lambda r: -r["similarity"])
    return out[:limit]


def find_contradictions(
    col: chromadb.Collection,
    question: str,
    answer: str,
    exclude_paths: set[str],
    k: int = 5,
) -> list[dict]:
    """Surface chunks in the vault whose claims contradict `answer`.

    Procedure:
      1. Embed the answer and pull nearest chunks (not the question — we want
         what directly contradicts what was just claimed, not what matches
         the question framing).
      2. Drop chunks already cited in the answer's retrieval (`exclude_paths`).
      3. Rerank the remainder against the original question so the survivors
         are still topically on-point.
      4. Ask the helper (one JSON prompt) which of the top candidates
         actually contradict — strict definition, conservative bias.

    Returns [{path, note, snippet, why}, ...] — empty if nothing genuine.
    Non-fatal: parse errors, short answers, empty vault all return [].
    """
    answer = (answer or "").strip()
    if len(answer) < 40 or col.count() == 0:
        return []
    try:
        ans_embed = embed([answer])[0]
    except Exception:
        return []
    # Pull generously; we'll filter + rerank down. Dedup by path so the
    # counter-set isn't dominated by multiple chunks of the same note.
    n = min(k * 4 + len(exclude_paths) + 5, col.count())
    cand = col.query(
        query_embeddings=[ans_embed], n_results=n,
        include=["documents", "metadatas"],
    )
    filtered: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for d, m in zip(cand["documents"][0], cand["metadatas"][0]):
        path = m.get("file", "")
        if path in exclude_paths or path in seen:
            continue
        seen.add(path)
        filtered.append((expand_to_parent(d, m), m))
        if len(filtered) >= k * 2:
            break
    if not filtered:
        return []
    try:
        reranker = get_reranker()
        pairs = [(question, d) for d, _ in filtered]
        scores = reranker.predict(pairs, show_progress_bar=False)
    except Exception:
        scores = [0.0] * len(filtered)
    ranked = sorted(
        zip(filtered, scores), key=lambda x: float(x[1]), reverse=True
    )[:k]

    numbered = "\n\n".join(
        f"[{i}] nota: {m.get('note','')} (ruta: {m.get('file','')})\n{d[:600]}"
        for i, ((d, m), _) in enumerate(ranked, 1)
    )
    prompt = (
        "Se generó esta RESPUESTA a partir de notas de un vault personal:\n\n"
        f"RESPUESTA: {answer}\n\n"
        "Estos son fragmentos de OTRAS notas del vault que no se usaron en "
        "la respuesta:\n\n"
        f"{numbered}\n\n"
        "Tu tarea: identificar cuáles fragmentos (si hay) CONTRADICEN la "
        "respuesta. Contradicción = afirmaciones incompatibles sobre el "
        "mismo sujeto. NO cuenta: temas distintos, matices, información "
        "complementaria, o perspectivas que el autor podría sostener a la "
        "vez. Sé conservador — mejor perder una contradicción real que "
        "marcar una falsa.\n\n"
        "Respondé SOLO con JSON con esta forma exacta:\n"
        '{"contradictions": [{"index": N, "why": "tensión en <20 palabras"}]}\n'
        'Si no hay contradicciones: {"contradictions": []}'
    )
    # Detector uses the chat model (command-r), NOT the helper. Empirical: on
    # this corpus, qwen2.5:3b is non-deterministic at temp=0 (same case yields
    # FP then empty across runs) and emits malformed JSON often. command-r is
    # RAG-trained, hugs the source text, and reliably returns parseable JSON.
    # Cost: ~5-10s per check — only paid on opt-in --counter, acceptable.
    try:
        resp = ollama.chat(
            model=resolve_chat_model(),
            messages=[{"role": "user", "content": prompt}],
            options=CHAT_OPTIONS,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        raw = resp.message.content.strip()
    except Exception:
        return []
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    hits = data.get("contradictions") or []
    if not isinstance(hits, list):
        return []
    out: list[dict] = []
    for item in hits:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int) or not (1 <= idx <= len(ranked)):
            continue
        (doc, meta), _ = ranked[idx - 1]
        why = str(item.get("why") or "").strip()[:200]
        out.append({
            "path": meta.get("file", ""),
            "note": meta.get("note", ""),
            "snippet": doc[:280].strip(),
            "why": why,
        })
    return out


def render_contradictions(contrad: list[dict]) -> None:
    """Render the counter-evidence block under a generated answer."""
    if not contrad:
        return
    console.print()
    console.print(Rule(
        title="[bold red]⚡ Counter-evidence en tu vault[/bold red]",
        style="red",
    ))
    for c in contrad:
        console.print()
        line = Text()
        line.append("⚠ ", style="bold red")
        line.append(c.get("note", ""), style=_file_link_style(c["path"], "bold red"))
        line.append("  ", style="")
        line.append(c["path"], style=_file_link_style(c["path"], "red dim"))
        console.print(line)
        if c.get("why"):
            console.print(f"   [italic yellow]{c['why']}[/italic yellow]")
        if c.get("snippet"):
            console.print(f"   [dim]{c['snippet']}[/dim]")


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
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    return resp.message.content.strip().strip('"')


def build_where(folder: str | None, tag: str | None) -> dict | None:
    """Build ChromaDB where filter from folder and/or tag.

    Folder is widened with `$or` to include 00-Inbox — rationale in
    `_folder_matches`. If the user *explicitly* filtered to Inbox, the OR
    is a no-op (same set twice), so the extra clause is harmless.
    """
    conditions = []
    if folder:
        conditions.append({"$or": [
            {"file": {"$contains": folder}},
            {"file": {"$contains": _CAPTURE_FOLDER + "/"}},
        ]})
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

    # Sequential — ChromaDB + BM25Okapi both hold a GIL-bound mutex, so
    # ThreadPoolExecutor over these serialises anyway AND adds per-task
    # overhead. Measured: parallel 3× slower than sequential on M3 Max.
    seen_ids: set[str] = set()
    merged_ordered: list[str] = []
    n_results = min(RETRIEVE_K, col.count())
    for v, q_embed in zip(variants, variant_embeds):
        sem_kwargs: dict = {
            "query_embeddings": [q_embed],
            "n_results": n_results,
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

    # 5a. Cap pool antes de fetch/rerank. El reranker es O(n) en pares y
    #     domina la latencia de la query (~50ms/par en MPS fp32). Un pool
    #     de 40 mantiene recall efectivo (top-5 rarísimo que esté fuera
    #     del top-40 RRF) y ahorra 1-2s por query.
    merged_ordered = merged_ordered[:RERANK_POOL_MAX]

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

    # Recency boost: when the query carries a temporal cue ("últimamente",
    # "este mes", "recientes"…), reweight by how fresh each note is. Small
    # additive (<=0.1) — enough to break ties toward recent notes without
    # overwhelming strong reranker signal.
    apply_recency = has_recency_cue(question)
    final_pairs: list[tuple] = []
    for c, e, s in zip(candidates, expanded, scores):
        final = float(s)
        if apply_recency:
            final += 0.1 * recency_boost(c[1])
        final_pairs.append((c, e, final))
    scored = sorted(final_pairs, key=lambda x: x[2], reverse=True)[:k]
    final_scores = [s for _, _, s in scored]
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


# ── Multi-vault retrieval ────────────────────────────────────────────────────
# Wrapper sobre retrieve() que corre la pipeline una vez por cada colección
# y merge-rankea globalmente por score del reranker. Cada vault mantiene su
# propio BM25 + grafo de wikilinks (se construyen por col en _load_corpus),
# y el reranker es el mismo (scores comparables entre cols).
#
# Trade-off conocido: latencia lineal en N de vaults (rerank es el cuello).
# Para 2 vaults ≈ 2x (~4s vs ~2s). Paralelizar con threads no ayuda: el
# reranker en MPS serializa de todas formas + agrega overhead por task.


def resolve_vault_paths(names: list[str] | None) -> list[tuple[str, Path]]:
    """Resuelve una lista de nombres a [(display_name, vault_path)]. Ignora
    nombres no registrados silenciosamente (el caller puede verificar el
    resultado vs input).

    `names=None` → solo el vault activo (resolve_vault_path + su nombre).
    `names=["all"]` → todos los registrados en el registry.
    """
    cfg = _load_vaults_config()
    if names is None:
        # Vault activo según precedencia estándar.
        active = _resolve_vault_path()
        # Intentamos darle un nombre legible.
        env = os.environ.get("OBSIDIAN_RAG_VAULT")
        if env:
            return [(f"env:{active.name}", active)]
        cur = cfg["current"]
        if cur and cur in cfg["vaults"]:
            return [(cur, active)]
        return [(f"default:{active.name}", active)]
    if names == ["all"]:
        names = list(cfg["vaults"].keys())
    out: list[tuple[str, Path]] = []
    for n in names:
        if n in cfg["vaults"]:
            out.append((n, Path(cfg["vaults"][n])))
    return out


def multi_retrieve(
    vaults: list[tuple[str, Path]],
    question: str,
    k: int,
    folder: str | None,
    history: list[dict] | None = None,
    tag: str | None = None,
    precise: bool = False,
    multi_query: bool = True,
    auto_filter: bool = True,
) -> dict:
    """Retrieve cross-vault. Para cada vault de `vaults`:
      1. Abre su colección (get_db_for).
      2. Corre retrieve() ahí.
      3. Anota las metas con `_vault` (display name) + `_vault_path`.
    Merge global por score del reranker, devuelve top-k.

    Return shape compatible con retrieve() para que el resto del pipeline
    (print_query_header, render_sources, etc.) no necesite cambios.
    `filters_applied` refleja los filtros inferidos en el primer vault
    que retornó algo (suficiente para mostrar en el header).
    """
    if not vaults:
        return {
            "docs": [], "metas": [], "scores": [], "confidence": float("-inf"),
            "search_query": question, "filters_applied": {}, "query_variants": [question],
            "vault_scope": [],
        }
    # Camino corto: un solo vault → evitamos el overhead de merge.
    if len(vaults) == 1:
        name, path = vaults[0]
        col = get_db_for(path)
        r = retrieve(
            col, question, k, folder, history, tag, precise,
            multi_query=multi_query, auto_filter=auto_filter,
        )
        # Solo anotamos si hay >=2 en scope el display (innecesario para uno).
        r["vault_scope"] = [name]
        return r

    all_items: list[tuple[float, str, dict]] = []
    variants: list[str] | None = None
    filters_applied: dict = {}
    for name, path in vaults:
        col = get_db_for(path)
        if col.count() == 0:
            continue   # vault no indexado — skip silent
        r = retrieve(
            col, question, k, folder, history, tag, precise,
            multi_query=multi_query, auto_filter=auto_filter,
        )
        if variants is None:
            variants = r["query_variants"]
        for d, m, s in zip(r["docs"], r["metas"], r["scores"]):
            annotated = dict(m)
            annotated["_vault"] = name
            annotated["_vault_path"] = str(path)
            all_items.append((s, d, annotated))
        # Primer vault con filtros inferidos manda (raro que difieran).
        if not filters_applied and r["filters_applied"]:
            filters_applied = r["filters_applied"]

    all_items.sort(key=lambda x: -x[0])
    top = all_items[:k]
    top_score = top[0][0] if top else float("-inf")
    return {
        "docs": [d for _, d, _ in top],
        "metas": [m for _, _, m in top],
        "scores": [s for s, _, _ in top],
        "confidence": top_score,
        "search_query": question,
        "filters_applied": filters_applied,
        "query_variants": variants or [question],
        "vault_scope": [n for n, _ in vaults],
    }


# ── COMMANDS ──────────────────────────────────────────────────────────────────

# Highlight `chat` en `rag --help`. El default formatter de click pinta cada
# subcomando en gris uniforme — `chat` es el daily driver, vale destacarlo.
# Manualmente formateamos las filas para que los ANSI escapes no rompan la
# alineación de la columna (formatter.write_dl mide len() literal y no la
# visible width).
_HIGHLIGHTED_COMMANDS = {"chat"}


class _HighlightGroup(click.Group):
    def format_commands(self, ctx, formatter):
        rows: list[tuple[str, click.Command]] = []
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            rows.append((name, cmd))
        if not rows:
            return
        max_name = max(len(n) for n, _ in rows)
        # Reserva 4 cols (2 indent + 2 gap) además del nombre.
        limit = formatter.width - 4 - max_name
        with formatter.section("Commands"):
            for name, cmd in rows:
                short = cmd.get_short_help_str(limit)
                pad = " " * (max_name - len(name) + 2)
                if name in _HIGHLIGHTED_COMMANDS:
                    display = click.style(name, fg="cyan", bold=True)
                else:
                    display = name
                formatter.write(f"  {display}{pad}{short}\n")


@click.group(cls=_HighlightGroup)
def cli():
    """RAG local para notas de Obsidian."""


def file_hash(raw: str) -> str:
    """Stable hash of file contents to detect changes."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# Sidecar log for contradiction events. Phase 3 (weekly digest) reads this.
# Schema (per-line JSON): {ts, subject_path, contradicts: [{path, why}], helper_raw}.
CONTRADICTION_LOG_PATH = Path.home() / ".local/share/obsidian-rag/contradictions.jsonl"


def find_contradictions_for_note(
    col: chromadb.Collection,
    note_body: str,
    exclude_paths: set[str],
    k: int = 5,
) -> list[dict]:
    """Index-time twin of `find_contradictions`.

    Surfaces chunks in the vault that contradict claims in `note_body`. Called
    when a new or modified note is (re)indexed so we can flag it against prior
    vault content. Uses the note's first paragraph as the rerank anchor (no
    external question). Returns [{path, note, snippet, why}, ...]; empty on
    short bodies, empty vault, parse errors, or no genuine contradictions.
    """
    body = (note_body or "").strip()
    if len(body) < 200 or col.count() == 0:
        return []
    anchor = next(
        (p.strip() for p in body.split("\n\n") if p.strip()),
        body[:400],
    )[:400]
    try:
        body_embed = embed([body[:2000]])[0]
    except Exception:
        return []
    n = min(k * 4 + len(exclude_paths) + 5, col.count())
    cand = col.query(
        query_embeddings=[body_embed], n_results=n,
        include=["documents", "metadatas"],
    )
    filtered: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for d, m in zip(cand["documents"][0], cand["metadatas"][0]):
        path_str = m.get("file", "")
        if path_str in exclude_paths or path_str in seen:
            continue
        seen.add(path_str)
        filtered.append((expand_to_parent(d, m), m))
        if len(filtered) >= k * 2:
            break
    if not filtered:
        return []
    try:
        reranker = get_reranker()
        pairs = [(anchor, d) for d, _ in filtered]
        scores = reranker.predict(pairs, show_progress_bar=False)
    except Exception:
        scores = [0.0] * len(filtered)
    ranked = sorted(
        zip(filtered, scores), key=lambda x: float(x[1]), reverse=True
    )[:k]

    numbered = "\n\n".join(
        f"[{i}] nota: {m.get('note','')} (ruta: {m.get('file','')})\n{d[:600]}"
        for i, ((d, m), _) in enumerate(ranked, 1)
    )
    prompt = (
        "Esta es una nota nueva o modificada del vault:\n\n"
        f"NOTA:\n{body[:1500]}\n\n"
        "Estos son fragmentos de OTRAS notas previas del vault:\n\n"
        f"{numbered}\n\n"
        "Tu tarea: identificar cuáles fragmentos (si hay) CONTRADICEN "
        "afirmaciones de la nota nueva. Contradicción = afirmaciones "
        "incompatibles sobre el mismo sujeto. NO cuenta: temas distintos, "
        "matices, información complementaria, o perspectivas que el autor "
        "podría sostener a la vez. Sé conservador — mejor perder una "
        "contradicción real que marcar una falsa.\n\n"
        "Respondé SOLO con JSON con esta forma exacta:\n"
        '{"contradictions": [{"index": N, "why": "tensión en <20 palabras"}]}\n'
        'Si no hay contradicciones: {"contradictions": []}'
    )
    # Detector uses the chat model (command-r), NOT the helper. Same reasoning
    # as find_contradictions: qwen2.5:3b is non-deterministic and emits
    # malformed JSON; command-r returns clean parseable output. Cost ~5-10s
    # per new note in incremental indexing — bounded by guardrails (skip on
    # full reindex, skip if body < 200 chars).
    helper_raw = ""
    try:
        resp = ollama.chat(
            model=resolve_chat_model(),
            messages=[{"role": "user", "content": prompt}],
            options=CHAT_OPTIONS,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        helper_raw = resp.message.content.strip()
    except Exception:
        return []
    m = re.search(r"\{.*\}", helper_raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    hits = data.get("contradictions") or []
    if not isinstance(hits, list):
        return []
    out: list[dict] = []
    for item in hits:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int) or not (1 <= idx <= len(ranked)):
            continue
        (doc, meta), _ = ranked[idx - 1]
        why = str(item.get("why") or "").strip()[:200]
        out.append({
            "path": meta.get("file", ""),
            "note": meta.get("note", ""),
            "snippet": doc[:280].strip(),
            "why": why,
            "_helper_raw": helper_raw,  # consumed by _log_contradictions, stripped for callers
        })
    return out


def _log_contradictions(
    subject_path: str,
    contrad: list[dict] | None = None,
    skipped: str | None = None,
    helper_raw: str = "",
) -> None:
    """Append a contradiction event to the sidecar log. Best-effort, never raises.

    Schema (agreed with phase-3 digest worker):
      {ts, cmd: "contradict_index", subject_path,
       contradicts: [{path, note, why}], helper_raw, skipped: str|null}
    `skipped` is null when the check ran and produced entries; a string reason
    ("too_short" | "error") when it bailed.
    """
    entries = [
        {
            "path": c["path"],
            "note": c.get("note", ""),
            "why": c.get("why", ""),
        }
        for c in (contrad or [])
    ]
    try:
        CONTRADICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "cmd": "contradict_index",
            "subject_path": subject_path,
            "contradicts": entries,
            "helper_raw": helper_raw,
            "skipped": skipped,
        }
        with CONTRADICTION_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _update_contradicts_frontmatter(path: Path, contradicts: list[str]) -> bool:
    """Write/replace `contradicts: [...]` in the note's YAML frontmatter.

    Preserves all other frontmatter keys and body content verbatim. Creates
    a frontmatter block if none exists. Returns True if the file was written.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    if raw.startswith("---\n"):
        end = raw.find("\n---\n", 4)
        if end < 0:
            return False
        fm_text = raw[4:end]
        rest = raw[end + 5:]
        new_lines: list[str] = []
        in_block = False
        for line in fm_text.splitlines():
            if in_block and re.match(r"^\s*-\s+", line):
                continue
            in_block = False
            if re.match(r"^contradicts\s*:", line):
                in_block = True
                continue
            new_lines.append(line)
        new_lines.append("contradicts:")
        for p in contradicts:
            new_lines.append(f"- {p}")
        new_raw = "---\n" + "\n".join(new_lines) + "\n---\n" + rest
    else:
        fm_block = (
            "---\ncontradicts:\n"
            + "\n".join(f"- {p}" for p in contradicts)
            + "\n---\n\n"
        )
        new_raw = fm_block + raw
    try:
        path.write_text(new_raw, encoding="utf-8")
    except Exception:
        return False
    return True


def _check_and_flag_contradictions(
    col: chromadb.Collection,
    path: Path,
    text: str,
    doc_id_prefix: str,
) -> tuple[str, str] | None:
    """Run the contradiction pipeline and persist findings to FM + sidecar log.

    If contradictions are found, the note's frontmatter is updated in place
    and a CLI alert + log entry are emitted. Returns (new_raw, new_hash) when
    the file was modified so the caller can use the fresh values for indexing;
    returns None on short body, no contradictions, or any failure.
    """
    if len(text) < 200:
        _log_contradictions(doc_id_prefix, skipped="too_short")
        return None
    try:
        contrad = find_contradictions_for_note(col, text, {doc_id_prefix}, k=5)
    except Exception:
        _log_contradictions(doc_id_prefix, skipped="error")
        return None
    if not contrad:
        return None  # ran cleanly, nothing to flag — no log entry
    paths = [c["path"] for c in contrad]
    if not _update_contradicts_frontmatter(path, paths):
        _log_contradictions(doc_id_prefix, skipped="error")
        return None
    try:
        new_raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        _log_contradictions(doc_id_prefix, skipped="error")
        return None
    console.print(
        f"[yellow]⚠[/yellow] [bold]{path.stem}[/bold] "
        f"contradice: {', '.join(paths)}"
    )
    helper_raw = contrad[0].get("_helper_raw", "") if contrad else ""
    _log_contradictions(doc_id_prefix, contrad=contrad, helper_raw=helper_raw)
    return (new_raw, file_hash(new_raw))


# ── AMBIENT AGENT ────────────────────────────────────────────────────────────
# Reactivo al filesystem: cuando una nota nueva aparece (o cambia) en 00-Inbox,
# el agente corre análisis sin LLM (dupes + related + wikilinks) y:
#   - auto-aplica wikilinks (regex determinística, segura)
#   - envía un mensaje Telegram con los findings + cambios
#
# Config (escrita por el bot vía `/enable_ambient`): JSON en
# `~/.local/share/obsidian-rag/ambient.json` con `{chat_id, bot_token}`.
# Sin config → no-op silencioso (no molesta si el usuario no lo habilitó).
#
# State file para idempotencia: `~/.local/share/obsidian-rag/ambient_state.jsonl`
# con `{path, hash, analyzed_at}`. Skip si la misma combinación corrió hace <5min.

AMBIENT_CONFIG_PATH = Path.home() / ".local/share/obsidian-rag/ambient.json"
AMBIENT_STATE_PATH = Path.home() / ".local/share/obsidian-rag/ambient_state.jsonl"
AMBIENT_DEDUP_WINDOW_SEC = 300   # 5 min
AMBIENT_LOG_PATH = Path.home() / ".local/share/obsidian-rag/ambient.jsonl"


def _ambient_config() -> dict | None:
    """Read the ambient config. Returns None if disabled / missing."""
    if not AMBIENT_CONFIG_PATH.is_file():
        return None
    try:
        c = json.loads(AMBIENT_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not c.get("chat_id") or not c.get("bot_token"):
        return None
    if c.get("enabled") is False:
        return None
    return c


def _ambient_should_skip(doc_id_prefix: str, h: str) -> bool:
    """Return True if we already analyzed this exact path+hash recently."""
    if not AMBIENT_STATE_PATH.is_file():
        return False
    try:
        lines = AMBIENT_STATE_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    cutoff = time.time() - AMBIENT_DEDUP_WINDOW_SEC
    for line in reversed(lines[-500:]):   # scan recent tail
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("path") != doc_id_prefix or e.get("hash") != h:
            continue
        try:
            ts = float(e.get("analyzed_at", 0))
        except Exception:
            continue
        if ts >= cutoff:
            return True
    return False


def _ambient_state_record(doc_id_prefix: str, h: str, payload: dict) -> None:
    try:
        AMBIENT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "path": doc_id_prefix, "hash": h,
            "analyzed_at": time.time(), **payload,
        }
        with AMBIENT_STATE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _ambient_log_event(event: dict) -> None:
    try:
        AMBIENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        e = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
        with AMBIENT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _ambient_telegram_send(chat_id: str, bot_token: str, text: str) -> bool:
    """Fire-and-forget Telegram sendMessage. Returns True on 2xx response."""
    import urllib.request
    import urllib.error
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _ambient_hook(
    col: chromadb.Collection,
    path: Path,
    doc_id_prefix: str,
    h: str,
) -> None:
    """Reactive analysis triggered on new/modified Inbox notes.

    No-op when:
      - file is outside 00-Inbox/
      - ambient config missing (bot didn't run /enable_ambient)
      - same path+hash analyzed within 5 min (dedup)
      - frontmatter `ambient: skip`

    Cheap actions (no LLM):
      - apply_wikilink_suggestions  (regex-deterministic)
      - find_near_duplicates_for    (pairwise cosine)
      - find_related                (graph + tags)
    Expensive (tag suggestion, contradiction) NOT run — those live in their
    own pipelines to avoid double-counting LLM cost per save event.
    """
    if not doc_id_prefix.startswith(_CAPTURE_FOLDER + "/"):
        return
    cfg = _ambient_config()
    if cfg is None:
        return
    if _ambient_should_skip(doc_id_prefix, h):
        return

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    fm = parse_frontmatter(raw)
    if fm.get("ambient") == "skip":
        _ambient_state_record(doc_id_prefix, h, {"skipped_reason": "fm_skip"})
        return
    if fm.get("type") in ("morning-brief", "weekly-digest", "prep"):
        # System-generated notes don't deserve ambient pings.
        _ambient_state_record(doc_id_prefix, h, {"skipped_reason": "system_note"})
        return

    # 1. Wikilink suggestions → auto-apply (conservative: the suggester already
    #    skips ambiguous titles, self-links, and short titles by default).
    wikilinks_applied = 0
    try:
        sugs = find_wikilink_suggestions(col, doc_id_prefix, max_per_note=10)
        if sugs:
            wikilinks_applied = apply_wikilink_suggestions(doc_id_prefix, sugs)
    except Exception:
        sugs = []

    # 2. Near-duplicates — flag only (user decides whether to merge).
    try:
        dupes = find_near_duplicates_for(col, doc_id_prefix, threshold=0.85, limit=3)
    except Exception:
        dupes = []

    # 3. Related notes — informational (top 3).
    try:
        # find_related takes a list of metas; build one from our own.
        self_meta = {
            "file": doc_id_prefix, "note": path.stem,
            "folder": str(path.relative_to(VAULT_PATH).parent),
            "tags": ",".join(
                str(t) for t in (fm.get("tags") or []) if t
            ),
        }
        related = find_related(col, [self_meta], limit=5)
    except Exception:
        related = []

    # Build Telegram message — compact, with wikilink-safe format.
    title = path.stem
    lines = [f"🤖 Ambient: [[{title}]]"]
    if wikilinks_applied:
        titles_applied = [s["title"] for s in sugs[:wikilinks_applied]]
        preview = ", ".join(titles_applied[:3])
        extra = "" if len(titles_applied) <= 3 else f" +{len(titles_applied) - 3}"
        lines.append(f"🔗 Linkeé {wikilinks_applied}: {preview}{extra}")
    if dupes:
        lines.append("⚠ Posibles duplicados:")
        for d in dupes[:3]:
            lines.append(f"  · [[{d['note']}]]  sim {d['similarity']:.2f}")
    if related:
        lines.append("📎 Relacionadas:")
        for m, score, reason in related[:3]:
            badge = {"link": "↔", "tags": "#", "tags+link": "↔#"}.get(reason, "")
            lines.append(f"  · [[{m.get('note', '')}]]  ×{score} {badge}")

    msg = "\n".join(lines)
    sent = False
    if len(lines) > 1:
        sent = _ambient_telegram_send(cfg["chat_id"], cfg["bot_token"], msg)

    _ambient_log_event({
        "cmd": "ambient_hook",
        "path": doc_id_prefix,
        "hash": h,
        "wikilinks_applied": wikilinks_applied,
        "wikilinks_proposed": len(sugs),
        "dupes": [{"path": d["path"], "sim": d["similarity"]} for d in dupes],
        "related_count": len(related),
        "telegram_sent": sent,
        "quiet": len(lines) <= 1,
    })
    _ambient_state_record(doc_id_prefix, h, {
        "wikilinks_applied": wikilinks_applied,
        "dupes_count": len(dupes),
        "related_count": len(related),
    })


def _index_single_file(
    col: chromadb.Collection, path: Path, skip_contradict: bool = False,
) -> str:
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

    folder = str(path.relative_to(VAULT_PATH).parent)
    fm = parse_frontmatter(raw)
    tags = [str(t) for t in (fm.get("tags") or []) if t]
    outlinks = extract_wikilinks(raw)  # parsed on raw; clean_md strips the syntax
    text = clean_md(raw)

    # Contradiction check runs before re-embedding so we only pay once. The
    # collection still holds the old version of this note if any — we exclude
    # `doc_id_prefix` from the candidate set regardless.
    if not skip_contradict:
        updated = _check_and_flag_contradictions(col, path, text, doc_id_prefix)
        if updated:
            raw, h = updated
            fm = parse_frontmatter(raw)
            tags = [str(t) for t in (fm.get("tags") or []) if t]
            outlinks = extract_wikilinks(raw)
            # text is derived from clean_md (strips frontmatter), unchanged.

    if existing_ids:
        col.delete(ids=existing_ids)

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
        "outlinks": ",".join(outlinks),
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
    # URL sub-index runs in lockstep — same hash gate, same liveness window.
    try:
        _index_urls(get_urls_db(), doc_id_prefix, raw, path.stem, folder, tags)
    except Exception:
        pass
    _invalidate_corpus_cache()
    # Ambient hook — fires for Inbox saves only, no-op otherwise.
    # Runs AFTER the main indexing so the note is retrievable to itself for
    # related/dupe analysis. Pure best-effort; any failure is logged, never
    # blocks indexing.
    try:
        _ambient_hook(col, path, doc_id_prefix, h)
    except Exception as e:
        _ambient_log_event({
            "cmd": "ambient_hook_error",
            "path": doc_id_prefix,
            "error": str(e)[:200],
        })
    return "indexed"


def _run_index(reset: bool, no_contradict: bool) -> dict:
    """Core indexing logic. Shared by the `rag index` CLI and the in-chat
    natural-language reindex intent. Returns stats dict for callers that
    want to render their own summary.
    """
    col = get_db()
    _invalidate_corpus_cache()
    # Contradiction check only runs in incremental mode and when not opted out.
    check_contradictions = not reset and not no_contradict

    if reset:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        for cname in (COLLECTION_NAME, URLS_COLLECTION_NAME):
            try:
                client.delete_collection(cname)
            except Exception:
                pass
        col = client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        console.print("[yellow]Índice borrado (notas + URLs).[/yellow]")

    col_urls = get_urls_db()
    urls_indexed = 0

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

        fm = parse_frontmatter(raw)
        tags = [str(t) for t in (fm.get("tags") or []) if t]
        outlinks = extract_wikilinks(raw)
        text = clean_md(raw)

        # Before touching the collection for this file, check whether the
        # incoming content contradicts anything already in the vault. When
        # it does, the note's frontmatter is rewritten in place and we pick
        # up the new raw/hash here.
        if check_contradictions:
            updated = _check_and_flag_contradictions(col, path, text, doc_id_prefix)
            if updated:
                raw, h = updated
                fm = parse_frontmatter(raw)
                tags = [str(t) for t in (fm.get("tags") or []) if t]
                outlinks = extract_wikilinks(raw)

        # File changed (or new) — remove any stale chunks first
        if existing:
            col.delete(ids=[eid for eid, _ in existing])
            updated_files += 1

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
            "outlinks": ",".join(outlinks),
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
        # URL sub-index, same hash gate as the chunk index above.
        try:
            urls_indexed += _index_urls(col_urls, doc_id_prefix, raw, path.stem, folder, tags)
        except Exception:
            pass
        indexed_files.add(doc_id_prefix)

    # Orphan cleanup: files in DB that no longer exist on disk
    orphan_files = set(file_to_chunks.keys()) - indexed_files
    orphan_ids: list[str] = []
    for f in orphan_files:
        orphan_ids.extend(eid for eid, _ in file_to_chunks[f])
    if orphan_ids:
        col.delete(ids=orphan_ids)
    # Mirror orphan cleanup in the URL collection.
    for f in orphan_files:
        try:
            existing_urls = col_urls.get(where={"file": f}, include=[])
            if existing_urls.get("ids"):
                col_urls.delete(ids=existing_urls["ids"])
        except Exception:
            pass

    console.print(
        f"[green]Listo. {added_chunks} chunks (re)indexados · "
        f"{updated_files} notas actualizadas · "
        f"{urls_indexed} URLs indexadas · "
        f"{len(orphan_files)} huérfanas limpiadas.[/green]"
    )
    return {
        "added_chunks": added_chunks,
        "updated_files": updated_files,
        "urls_indexed": urls_indexed,
        "orphans": len(orphan_files),
        "total_files": len(md_files),
    }


@cli.command()
@click.option("--reset", is_flag=True, help="Borrar índice antes de reindexar")
@click.option("--no-contradict", is_flag=True, help="Saltear el check de contradicciones en notas nuevas/modificadas")
def index(reset: bool, no_contradict: bool):
    """Indexar notas del vault (incremental, detecta cambios por hash).

    En el camino incremental, cada nota nueva o modificada pasa por un check
    de contradicciones contra el resto del vault. Con `--reset` se omite
    (full reindex haría O(n²) llamadas al helper). `--no-contradict` lo
    saltea también en incremental.
    """
    _run_index(reset=reset, no_contradict=no_contradict)


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
@click.option("--session", "session_id", default=None,
              help="ID de sesión (reanuda si existe, crea si no). Ej: 'tg:12345'.")
@click.option("--continue", "continue_", is_flag=True,
              help="Reanuda la última sesión usada (atajo a --session <última>)")
@click.option("--plain", is_flag=True,
              help="Salida plana sin colores/paneles/sources. Para consumo programático.")
@click.option("--counter", is_flag=True,
              help="Después de responder, buscar chunks del vault que CONTRADIGAN la respuesta")
def query(
    question: str, k: int, folder: str | None, tag: str | None,
    hyde: bool, no_multi: bool, no_auto_filter: bool,
    raw: bool, loose: bool, force: bool,
    session_id: str | None, continue_: bool, plain: bool,
    counter: bool,
):
    """Consulta única contra las notas."""
    col = get_db()
    if col.count() == 0:
        if plain:
            click.echo("Índice vacío. Ejecuta: rag index")
        else:
            console.print("[red]Índice vacío. Ejecuta: rag index[/red]")
        return

    # Session resolution: --continue picks up last_session_id unless --session
    # is also given. If neither is present, we don't touch the session store.
    sess: dict | None = None
    if continue_ and not session_id:
        session_id = last_session_id()
    if session_id is not None or continue_:
        sess = ensure_session(session_id, mode="query") if session_id else None
    history = session_history(sess) if sess else None

    # Reformulate standalone follow-ups when we have prior turns. Do this even
    # outside --hyde/--precise because the whole point of --session is that the
    # new question can be a pronoun-laden fragment.
    effective_question = question
    if history:
        try:
            effective_question = reformulate_query(question, history)
        except Exception:
            effective_question = question

    # Intent routing: aggregate/list/recent queries don't need the retrieval
    # pipeline + LLM — they want a metadata scan. Fall through to semantic
    # otherwise. User-supplied --folder/--tag override classifier params.
    known_tags, known_folders = get_vocabulary(col)
    intent, intent_params = classify_intent(effective_question, known_tags, known_folders)
    if intent != "semantic":
        if folder:
            intent_params["folder"] = folder
        if tag:
            intent_params["tag"] = tag
        params_str = ", ".join(f"{k}={v}" for k, v in intent_params.items() if v) or "sin filtros"
        if not plain:
            console.print()
        if intent == "count":
            n, files = handle_count(col, intent_params)
            if plain:
                click.echo(f"{n} nota(s) ({params_str})")
                for f in files[:30]:
                    click.echo(f["file"])
            else:
                console.print(
                    f"[bold green]{n}[/bold green] nota(s) [dim]({params_str})[/dim]"
                )
                if n and n <= 30:
                    render_file_list(f"notas", files)
        elif intent == "list":
            files = handle_list(col, intent_params)
            if plain:
                click.echo(f"{len(files)} nota(s) ({params_str})")
                for f in files:
                    click.echo(f["file"])
            else:
                console.print(
                    f"[bold cyan]{len(files)}[/bold cyan] nota(s) [dim]({params_str})[/dim]"
                )
                render_file_list("notas", files)
        elif intent == "recent":
            files = handle_recent(col, intent_params)
            if plain:
                click.echo(f"{len(files)} nota(s) recientes ({params_str})")
                for f in files:
                    click.echo(f["file"])
            else:
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
    if plain:
        result = retrieve(
            col, effective_question, k, folder, tag=tag, precise=hyde,
            multi_query=not no_multi, auto_filter=not no_auto_filter,
        )
    else:
        with console.status("[dim]buscando…[/dim]", spinner="dots"):
            result = retrieve(
                col, effective_question, k, folder, tag=tag, precise=hyde,
                multi_query=not no_multi, auto_filter=not no_auto_filter,
            )
    t_retrieve = time.perf_counter() - t_start
    if not result["docs"]:
        log_query_event({
            "cmd": "query", "q": question, "filters": result.get("filters_applied"),
            "variants": result.get("query_variants"), "paths": [],
            "top_score": None, "t_retrieve": round(t_retrieve, 2), "answered": False,
        })
        if plain:
            click.echo("Sin resultados.")
        else:
            console.print("[yellow]Sin resultados.[/yellow]")
        return

    if not plain:
        print_query_header(question, result)

    if raw:
        # Skip LLM — dump retrieved chunks verbatim with their path.
        if plain:
            for d, m, s in zip(result["docs"], result["metas"], result["scores"]):
                click.echo(f"{m.get('note','')} ({m.get('file','')}) · {s:+.1f}")
                click.echo(d)
                click.echo("---")
        else:
            console.print()
            for d, m, s in zip(result["docs"], result["metas"], result["scores"]):
                path = m.get("file", "")
                note = m.get("note", "")
                console.print(f"[bold cyan]{note}[/bold cyan] [dim]({path}) · {s:+.1f}[/dim]")
                console.print(Markdown(d))
                console.print(Rule(style="dim"))
            print_sources(result)
            render_related(find_related(col, result["metas"]))
        return

    # Gate LLM on reranker confidence. Negative top score ≈ rerank found
    # nothing relevant — skipping the LLM avoids hallucinated answers from
    # unrelated chunks. `--force` overrides.
    if result["confidence"] < CONFIDENCE_RERANK_MIN and not force:
        msg = (
            f"No tengo esa información en tus notas. "
            f"(top rerank score: {result['confidence']:+.2f} < {CONFIDENCE_RERANK_MIN}; "
            f"usá --force para llamar al LLM igual)"
        )
        if plain:
            click.echo(msg)
        else:
            console.print()
            console.print(
                f"[yellow]No tengo esa información en tus notas.[/yellow] "
                f"[dim](top rerank score: {result['confidence']:+.2f} < {CONFIDENCE_RERANK_MIN}; "
                f"usá --force para llamar al LLM igual)[/dim]"
            )
            print_sources(result)
            render_related(find_related(col, result["metas"]))
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
        if sess is not None:
            append_turn(sess, {
                "q": question,
                "q_reformulated": effective_question if effective_question != question else None,
                "a": None,
                "paths": [m.get("file", "") for m in result["metas"]],
                "top_score": round(float(result["confidence"]), 3),
                "gated": True,
            })
            save_session(sess)
        return

    context = "\n\n---\n\n".join(
        f"[nota: {m['note']}] [ruta: {m['file']}]\n{d}"
        for d, m in zip(result["docs"], result["metas"])
    )
    rules = SYSTEM_RULES if loose else SYSTEM_RULES_STRICT
    # When we have session history, build a chat-style prompt so the LLM sees
    # the conversation context. Otherwise keep the one-shot prompt unchanged.
    if history:
        messages = (
            [{"role": "system", "content": f"{rules}\nCONTEXTO:\n{context}"}]
            + history
            + [{"role": "user", "content": question}]
        )
    else:
        messages = [{"role": "user", "content": f"{rules}\nCONTEXTO:\n{context}\n\nPREGUNTA: {question}\n\nRESPUESTA:"}]

    t_gen_start = time.perf_counter()
    parts: list[str] = []
    if plain:
        # Stream tokens directly to stdout — no Rich overlays. Consumers (bots,
        # scripts) see the answer forming in real time without ANSI cruft.
        for chunk in ollama.chat(
            model=resolve_chat_model(),
            messages=messages,
            options=CHAT_OPTIONS,
            stream=True,
            keep_alive=OLLAMA_KEEP_ALIVE,
        ):
            tok = chunk.message.content
            parts.append(tok)
            click.echo(tok, nl=False)
        click.echo("")
    else:
        console.print()
        # Stream tokens: perceived latency drops from ~30s to ~3s as the user sees
        # the answer forming. We render unstyled during the stream, then redraw
        # with link/ext styling at the end.
        from rich.live import Live
        with Live("", console=console, refresh_per_second=12, transient=True) as live:
            for chunk in ollama.chat(
                model=resolve_chat_model(),
                messages=messages,
                options=CHAT_OPTIONS,
                stream=True,
                keep_alive=OLLAMA_KEEP_ALIVE,
            ):
                parts.append(chunk.message.content)
                live.update(Text("".join(parts)))
    t_gen = time.perf_counter() - t_gen_start
    full = "".join(parts)
    if not plain:
        console.print(render_response(full))

    bad = verify_citations(full, result["metas"])
    if bad and not plain:
        console.print()
        console.print("[bold red]⚠ Citas no verificadas:[/bold red]")
        for label, path in bad:
            console.print(f"  [red]• {label} → {path}[/red] [dim](no está en los chunks recuperados)[/dim]")

    contrad: list[dict] = []
    if counter:
        def _run_counter():
            return find_contradictions(
                col, question, full,
                exclude_paths={m.get("file", "") for m in result["metas"]},
            )
        if plain:
            contrad = _run_counter()
            if contrad:
                click.echo("")
                click.echo("Counter-evidence:")
                for c in contrad:
                    click.echo(f"  ⚠ {c.get('note','')} ({c['path']})")
                    if c.get("why"):
                        click.echo(f"    {c['why']}")
        else:
            with console.status("[dim]buscando contra-evidencia…[/dim]", spinner="dots"):
                contrad = _run_counter()
            render_contradictions(contrad)

    log_query_event({
        "cmd": "query",
        "q": question,
        "q_reformulated": effective_question if effective_question != question else None,
        "session": sess["id"] if sess else None,
        "filters": result.get("filters_applied"),
        "variants": result.get("query_variants"),
        "paths": [m.get("file", "") for m in result["metas"]],
        "scores": [round(float(s), 2) for s in result["scores"]],
        "top_score": round(float(result["confidence"]), 2),
        "t_retrieve": round(t_retrieve, 2),
        "t_gen": round(t_gen, 2),
        "answer_len": len(full),
        "bad_citations": [p for _, p in bad],
        "contradictions": [{"path": c["path"], "why": c["why"]} for c in contrad] if counter else None,
        "mode": "raw" if raw else ("loose" if loose else "strict"),
        "plain": plain,
    })

    if sess is not None:
        append_turn(sess, {
            "q": question,
            "q_reformulated": effective_question if effective_question != question else None,
            "a": full,
            "paths": [m.get("file", "") for m in result["metas"]],
            "top_score": round(float(result["confidence"]), 3),
            "contradictions": [c["path"] for c in contrad] if contrad else None,
        })
        save_session(sess)

    if not plain:
        print_sources(result)
        render_related(find_related(col, result["metas"]))


def _arrow_select(
    title: str,
    choices: list[str],
    default_idx: int = 0,
) -> int:
    """Menú interactivo con ↑/↓ + Enter. Retorna el índice elegido, o -1
    si el usuario cancela (q / Esc bare / Ctrl-C).

    Fallback no-TTY (pipes, tests, CI): retorna default_idx sin prompt.

    Implementación: ANSI cursor movement manual. Probada predictable —
    Rich.Live con cbreak + auto_refresh dio bugs de doble-render y
    flechas que no actualizaban. Acá controlamos línea por línea:
      - render inicial: imprime N+2 líneas (título + choices + hint).
      - en cada keypress: cursor up N+2, escribe cada línea con \\x1b[2K
        (clear line) + contenido nuevo + \\n.
      - al salir: cursor up N+2, clear N+2 líneas, posiciona cursor
        donde estaba el título (chat panel arranca limpio ahí).
    """
    import sys
    if not sys.stdin.isatty():
        return default_idx

    import select as _select_mod
    import termios
    import tty

    n_choices = len(choices)
    selected = max(0, min(default_idx, n_choices - 1))
    # Layout: 1 (título) + n_choices + 1 (hint) = total_lines.
    total_lines = 1 + n_choices + 1
    hint = "[↑/↓ navegar · Enter elegir · q cancelar]"

    def _line_for(idx: int) -> str:
        """Genera la línea idx-ésima del menú con ANSI styling.
        idx 0 = título; 1..n_choices = choices; n_choices+1 = hint.
        """
        if idx == 0:
            return f"\x1b[1m{title}\x1b[0m"
        if idx == total_lines - 1:
            return f"\x1b[2m{hint}\x1b[0m"
        ci = idx - 1
        label = choices[ci]
        if ci == selected:
            return f"  \x1b[1;36m❯ {label}\x1b[0m"
        return f"    {label}"

    def _draw_initial():
        for i in range(total_lines):
            sys.stdout.write(_line_for(i) + "\n")
        sys.stdout.flush()

    def _redraw():
        # Volver al inicio del bloque, reescribir cada línea limpiándola.
        sys.stdout.write(f"\x1b[{total_lines}A")
        for i in range(total_lines):
            sys.stdout.write("\r\x1b[2K" + _line_for(i) + "\n")
        sys.stdout.flush()

    def _erase_menu():
        sys.stdout.write(f"\x1b[{total_lines}A")
        for _ in range(total_lines):
            sys.stdout.write("\r\x1b[2K\n")
        sys.stdout.write(f"\x1b[{total_lines}A")
        sys.stdout.flush()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    _draw_initial()
    try:
        tty.setcbreak(fd)
        while True:
            # CRÍTICO: os.read(fd, 1) — NO sys.stdin.read(1).
            # sys.stdin es TextIOWrapper con su propio buffer que sigue
            # siendo line-buffered incluso con cbreak en el TTY. read(1)
            # bloquea esperando que llegue un newline. Bypass directo al
            # fd con os.read da un byte tan pronto como esté disponible.
            try:
                b = os.read(fd, 1)
            except (KeyboardInterrupt, OSError):
                _erase_menu()
                return -1
            if not b:
                continue
            ch = b.decode("latin-1")
            if ch == "\x1b":   # ESC o secuencia
                # Pequeño timeout para distinguir ESC bare de inicio de seq.
                r, _, _ = _select_mod.select([fd], [], [], 0.05)
                if r:
                    seq = os.read(fd, 2).decode("latin-1")
                    if seq == "[A":
                        selected = (selected - 1) % n_choices
                    elif seq == "[B":
                        selected = (selected + 1) % n_choices
                    # otras secuencias: ignorar
                else:
                    _erase_menu()
                    return -1   # ESC bare → cancelar
            elif ch in ("\r", "\n"):
                _erase_menu()
                return selected
            elif ch == "q":
                _erase_menu()
                return -1
            elif ch == "k":
                selected = (selected - 1) % n_choices
            elif ch == "j":
                selected = (selected + 1) % n_choices
            elif ch == "\x03":   # Ctrl-C en cbreak (ISIG-aware) puede
                _erase_menu()    # llegar igual via el fd; manejar.
                return -1
            _redraw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _prompt_vault_scope_interactive(cfg: dict) -> list[tuple[str, Path]]:
    """Pide al usuario qué vault(s) usar al arrancar chat. Solo se llama si
    hay ≥2 vaults registrados. Retorna lista de (name, path)."""
    vaults = list(cfg["vaults"].items())
    n = len(vaults)
    default_idx = 0
    if cfg["current"] and cfg["current"] in cfg["vaults"]:
        default_idx = list(cfg["vaults"].keys()).index(cfg["current"])

    choices = [
        f"{name}  ({Path(path).name})"
        for name, path in vaults
    ]
    choices.append(f"ambos / todos  (cross-vault en los {n} a la vez)")

    selected = _arrow_select(
        "¿Sobre qué vault querés conversar?",
        choices,
        default_idx=default_idx,
    )
    if selected < 0:
        # Cancelado → caer al default (vault current).
        selected = default_idx

    if selected < n:
        name, path = vaults[selected]
        console.print(f"[dim]→ vault: [bold magenta]{name}[/bold magenta][/dim]")
        return [(name, Path(path))]
    # Última opción: todos.
    console.print(
        f"[dim]→ vault: [bold magenta]{' + '.join(n for n, _ in vaults)}[/bold magenta] "
        f"[dim](cross-vault)[/dim][/dim]"
    )
    return [(name, Path(path)) for name, path in vaults]


@cli.command()
@click.option("-k", default=RERANK_TOP, help="Chunks finales por turno")
@click.option("--folder", default=None, help="Filtrar por carpeta (ej: '02-Areas/Musica')")
@click.option("--tag", default=None, help="Filtrar por tag (ej: letra, rock, ai, finanzas)")
@click.option("--precise", is_flag=True, help="HyDE + reformulación (más preciso, ~5s extra)")
@click.option("--no-multi", is_flag=True, help="Desactiva multi-query expansion")
@click.option("--no-auto-filter", is_flag=True, help="Desactiva inferencia de filtros")
@click.option("--session", "session_id", default=None,
              help="ID de sesión (reanuda si existe, crea si no). Admite 'tg:<chat_id>' etc.")
@click.option("--resume", is_flag=True, help="Reanuda la última sesión usada")
@click.option("--counter", is_flag=True,
              help="Después de cada respuesta, buscar chunks del vault que la CONTRADIGAN")
@click.option("--vault", "vault_scope", default=None,
              help="Scope de retrieval: nombre(s) separados por coma, o 'all'. "
                   "Omitir = prompt interactivo si hay ≥2 vaults registrados.")
def chat(
    k: int, folder: str | None, tag: str | None, precise: bool,
    no_multi: bool, no_auto_filter: bool,
    session_id: str | None, resume: bool, counter: bool,
    vault_scope: str | None,
):
    """Chat interactivo con tus notas."""
    # col = vault para WRITE operations (/save, /inbox, /reindex,
    # find_related). Siempre apunta al current del registry, aunque el
    # scope de READ incluya múltiples vaults — evita ambigüedad ("guardar
    # dónde").
    col = get_db()

    # Resolver scope de retrieval (READ).
    if vault_scope:
        if vault_scope.strip() == "all":
            vaults_resolved = resolve_vault_paths(["all"])
        else:
            names = [n.strip() for n in vault_scope.split(",") if n.strip()]
            vaults_resolved = resolve_vault_paths(names)
        if not vaults_resolved:
            console.print(
                f"[red]--vault '{vault_scope}' no resolvió ningún vault registrado.[/red]"
            )
            return
    else:
        _vcfg = _load_vaults_config()
        if len(_vcfg["vaults"]) >= 2 and not os.environ.get("OBSIDIAN_RAG_VAULT"):
            vaults_resolved = _prompt_vault_scope_interactive(_vcfg)
        else:
            vaults_resolved = resolve_vault_paths(None)

    if not vaults_resolved:
        console.print("[red]Sin vault para consultar. Registrá uno con `rag vault add`.[/red]")
        return

    # Defer auto_index hasta DESPUÉS de imprimir el Panel — sin esto, una
    # corrida de first-time index (1-2 min para 500 notas) deja la pantalla
    # en negro y parece colgada. Marcamos qué vaults van a indexarse para
    # disparar el ciclo después del banner.
    pending_index: list[tuple[str, Path, bool, int]] = []   # (name, path, is_first_time, n_files)
    for name, vpath in vaults_resolved:
        try:
            col_v = get_db_for(vpath)
            empty = col_v.count() == 0
        except Exception:
            continue
        n_files = sum(1 for _ in vpath.rglob("*.md"))
        if empty and n_files > 0:
            pending_index.append((name, vpath, True, n_files))
        elif not empty:
            pending_index.append((name, vpath, False, n_files))

    # Resolve session: --resume takes last used id; --session takes explicit id;
    # otherwise a fresh id is minted. ensure_session is idempotent on both paths.
    if resume and not session_id:
        session_id = last_session_id()
        if not session_id:
            console.print("[yellow]No hay sesión previa para reanudar. Creando una nueva.[/yellow]")
    sess = ensure_session(session_id, mode="chat")
    resumed = bool(sess["turns"])

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
    if counter:
        features.append("counter")
    subtitle = f"[dim]· {' · '.join(features)}"
    if flags:
        subtitle += f" · {' · '.join(flags)}"
    subtitle += "[/dim]"

    # Vault scope del chat. Uno solo → nombre; dos o más → "A + B" en la label.
    scope_names = [n for n, _ in vaults_resolved]
    if len(scope_names) == 1:
        vault_label = scope_names[0]
    else:
        vault_label = " + ".join(scope_names) + f" [dim](cross-vault)[/dim]"
    vault_line = f"[dim]· vault: [bold magenta]{vault_label}[/bold magenta][/dim]"

    session_line = (
        f"[dim]· sesión: [cyan]{sess['id']}[/cyan]"
        f"{' · reanudada (' + str(len(sess['turns'])) + ' turnos)' if resumed else ' · nueva'}[/dim]"
    )

    console.print(Panel(
        f"[bold green]RAG Obsidian — Chat[/bold green]\n{subtitle}\n"
        f"{vault_line}\n{session_line}\n"
        "[dim]/save · /reindex [reset] · /links <q> · /inbox [apply|undo] · /cls · /exit[/dim]",
        border_style="green",
    ))

    # AHORA sí: ejecutamos auto_index con feedback visible. El usuario vio
    # el banner primero, así que sabe que el chat está vivo aunque la
    # primera nota tarde 1-2 min en indexarse.
    for name, vpath, is_first_time, n_files in pending_index:
        if is_first_time:
            with console.status(
                f"[bold yellow]Primer index de '{name}'[/bold yellow] "
                f"[dim]· {n_files} notas · 1-2 min · embeddings + chunking…[/dim]",
                spinner="dots",
            ):
                stats = auto_index_vault(vpath)
            console.print(
                f"[green]✓ '{name}' indexado:[/green] {stats['indexed']} notas "
                f"[dim]({stats['took_ms'] / 1000:.1f}s)[/dim]"
            )
        else:
            stats = auto_index_vault(vpath)
            if stats["kind"] == "incremental" and (stats["indexed"] or stats["removed"]):
                console.print(
                    f"[green]✓ '{name}':[/green] {stats['indexed']} cambios "
                    f"+ {stats['removed']} orphans removidos "
                    f"[dim]({stats['took_ms']}ms)[/dim]"
                )

    # Re-validar — si todos quedaron vacíos (vault sin .md o índex falló),
    # no tiene sentido seguir.
    total_chunks = 0
    for name, vpath in vaults_resolved:
        try:
            total_chunks += get_db_for(vpath).count()
        except Exception:
            pass
    if total_chunks == 0:
        console.print(
            f"[red]Los vaults seleccionados están vacíos. "
            f"Verificá que tienen archivos .md.[/red]"
        )
        return

    history: list[dict] = session_history(sess, window=SESSION_HISTORY_WINDOW)
    last_assistant = ""
    last_question = ""
    last_sources: list[dict] = []
    # If resuming, seed last_* from final turn so `/save` works immediately.
    if sess["turns"]:
        final = sess["turns"][-1]
        last_assistant = final.get("a", "") or ""
        last_question = final.get("q", "") or ""
    first_turn = True
    while True:
        try:
            if not first_turn:
                console.print(Rule(style="dim", characters="╌"))
            first_turn = False
            question = console.input("\n[bold green]tu › [/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Hasta luego.[/dim]")
            break

        if question == "/exit":
            console.print("[dim]Hasta luego.[/dim]")
            break
        if not question:
            continue

        # /cls — limpia la pantalla y borra la conversación: turnos persistidos,
        # history en memoria, y last_* (no hay respuesta previa para `/save`).
        # Mantiene el session_id para que `--resume` siga apuntando acá, pero
        # arranca vacío (como si fuera primer turno).
        if question == "/cls":
            sess["turns"] = []
            try:
                save_session(sess)
            except Exception:
                pass
            history = []
            last_assistant = ""
            last_question = ""
            last_sources = []
            console.clear()
            console.print("[dim]Conversación borrada. Sesión sigue activa.[/dim]")
            continue

        # /inbox [apply|undo] — filing assistant sobre 00-Inbox, dispatch al
        # mismo callback que `rag file` standalone. Sin args: dry-run.
        # Apply abre el loop interactivo con click.prompt (compone bien con
        # el input de chat). Undo revierte el último batch.
        # Nota: el CLI sigue siendo `rag file` por compatibilidad; el slash
        # del chat usa `/inbox` porque es más representativo del scope
        # ("¿qué hago con mi inbox?").
        if question == "/inbox" or question.startswith("/inbox "):
            rest = question[len("/inbox"):].strip().split()
            do_apply = "apply" in rest
            do_undo = "undo" in rest
            if do_apply and do_undo:
                console.print("[red]No podés combinar apply y undo.[/red]")
                continue
            try:
                file_cmd.callback(
                    path=None, folder="00-Inbox", one=False, limit=20,
                    k=8, do_apply=do_apply, do_undo=do_undo, plain=False,
                )
            except Exception as e:
                console.print(f"[red]Error en /inbox: {e}[/red]")
            continue

        # Link intent — "donde está el link a X", "dame la url de Y",
        # "documentación de Z". Bypasses the LLM and returns URLs directly
        # from the URL sub-index. Checked early so a query that names a
        # vault topic doesn't get prose-paraphrased when the user really
        # wanted the literal link.
        is_link, link_q = detect_link_intent(question)
        if is_link:
            search_q = link_q or question
            # Backfill PRIMERO (fuera del status spinner) — si la URL DB
            # está vacía, su mensaje "indexando ~1 min" tiene que ser
            # visible al usuario, no enterrado bajo el "buscando URLs…".
            try:
                _maybe_backfill_urls()
            except Exception:
                pass
            with console.status("[dim]buscando URLs…[/dim]", spinner="dots"):
                items = find_urls(search_q, k=10)
            render_links(items)
            log_query_event({
                "cmd": "links", "q": search_q, "via": "chat",
                "n_results": len(items),
                "top_url": items[0]["url"] if items else None,
            })
            continue

        # Reindex intent — accepts /reindex or natural-language ("reindexá",
        # "actualizá el vault", "reescaneá las notas", "reset desde cero").
        # Checked BEFORE save intent so a phrase like "reindexá las notas"
        # doesn't trigger the save heuristic via "notas".
        is_reindex, reset = detect_reindex_intent(question)
        if is_reindex:
            label = "completo (--reset)" if reset else "incremental"
            console.print(f"[dim]Reindexando ({label})…[/dim]")
            try:
                stats = _run_index(reset=reset, no_contradict=False)
            except Exception as e:
                console.print(f"[red]Error indexando: {e}[/red]")
                continue
            # Collection identity changes on --reset; refresh local handle so
            # subsequent retrieves hit the new collection. Cache invalidation
            # already happens inside _run_index.
            col = get_db()
            console.print(
                f"[green]✓ Indexado:[/green] {stats['added_chunks']} chunks · "
                f"{stats['updated_files']} actualizadas · "
                f"{stats['orphans']} huérfanas"
            )
            continue

        # Save intent — accepts /save or natural-language ("agregá esto a una
        # nota", "guardá la respuesta", "creá una nota llamada X", etc.)
        is_save, save_title = detect_save_intent(question)
        if is_save:
            if not last_assistant:
                console.print("[yellow]No hay respuesta para guardar todavía.[/yellow]")
                continue
            path = save_note(
                col, save_title, last_assistant, last_question, last_sources,
            )
            rel = path.relative_to(VAULT_PATH)
            console.print(f"[green]✓ Guardado:[/green] [bold cyan]{rel}[/bold cyan]")
            continue

        with console.status("[dim]buscando…[/dim]", spinner="dots"):
            result = multi_retrieve(
                vaults_resolved, question, k, folder, history, tag, precise,
                multi_query=not no_multi, auto_filter=not no_auto_filter,
            )
        if not result["docs"]:
            console.print("[yellow]Sin resultados relevantes.[/yellow]")
            continue

        print_query_header(question, result, show_question=False)

        # Con >1 vault, anotamos el chunk header con el vault de origen así
        # el LLM puede citarlo y vos podés distinguir en la respuesta.
        is_multi = len(vaults_resolved) > 1
        context = "\n\n---\n\n".join(
            (f"[vault: {m.get('_vault', '?')}] " if is_multi else "")
            + f"[nota: {m['note']}] [ruta: {m['file']}]\n{d}"
            for d, m in zip(result["docs"], result["metas"])
        )
        # Prompt shape debe igualar `rag query` exactamente: sin priming
        # (PREGUNTA:/RESPUESTA:) command-r a veces refusea en borderline scores
        # aunque el contexto alcance — reproducido con "adam jones sistema de
        # sonido" (score 0.3, mismo Guitar.md retrieved, query respondía y chat
        # decía "No tengo esa información"). Para el primer turno usamos la
        # misma estructura one-shot que query; con history armamos el formato
        # multi-turno pero manteniendo el priming del contexto igual.
        if history:
            messages = (
                [{"role": "system", "content": f"{SYSTEM_RULES}\nCONTEXTO:\n{context}"}]
                + history
                + [{"role": "user", "content": question}]
            )
        else:
            messages = [{"role": "user", "content": (
                f"{SYSTEM_RULES}\nCONTEXTO:\n{context}\n\n"
                f"PREGUNTA: {question}\n\nRESPUESTA:"
            )}]
        history.append({"role": "user", "content": question})

        console.print()
        parts: list[str] = []
        console.print("[bold cyan]rag › [/bold cyan]", end="")
        from rich.live import Live
        with Live("", console=console, refresh_per_second=12, transient=True) as live:
            for chunk in ollama.chat(
                model=resolve_chat_model(),
                messages=messages,
                options=CHAT_OPTIONS,
                stream=True,
                keep_alive=OLLAMA_KEEP_ALIVE,
            ):
                parts.append(chunk.message.content)
                live.update(Text("".join(parts)))
        full = "".join(parts)
        console.print(render_response(full))
        history.append({"role": "assistant", "content": full})
        history = history[-SESSION_HISTORY_WINDOW:]
        last_assistant = full
        last_question = question
        last_sources = list(result["metas"])

        contrad: list[dict] = []
        if counter:
            with console.status("[dim]buscando contra-evidencia…[/dim]", spinner="dots"):
                contrad = find_contradictions(
                    col, question, full,
                    exclude_paths={m.get("file", "") for m in result["metas"]},
                )
            render_contradictions(contrad)

        append_turn(sess, {
            "q": question,
            "a": full,
            "paths": [m.get("file", "") for m in result["metas"]],
            "top_score": round(float(result["confidence"]), 3),
            "contradictions": [c["path"] for c in contrad] if contrad else None,
        })
        save_session(sess)

        print_sources(result)
        render_related(find_related(col, result["metas"]))


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
    chains = data.get("chains") or []
    if not queries and not chains:
        console.print(f"[yellow]Sin queries ni chains en {path}[/yellow]")
        return

    col = get_db()
    if col.count() == 0:
        console.print("[red]Índice vacío. Ejecuta: rag index[/red]")
        return

    def _score(expected_set: set[str], seen_paths: list[str]) -> tuple[bool, float, float]:
        retrieved_set = set(seen_paths)
        hit = bool(expected_set & retrieved_set)
        rr = 0.0
        for rank, p in enumerate(seen_paths, start=1):
            if p in expected_set:
                rr = 1.0 / rank
                break
        recall = (
            len(expected_set & retrieved_set) / len(expected_set)
            if expected_set else 0.0
        )
        return hit, rr, recall

    def _dedup(paths: list[str]) -> list[str]:
        seen: list[str] = []
        for p in paths:
            if p not in seen:
                seen.append(p)
        return seen

    # ── Single queries ──────────────────────────────────────────────────
    per_query: list[tuple[str, bool, float, float, list[str]]] = []
    if queries:
        for entry in track(queries, description="Evaluando queries…"):
            q = entry["question"]
            expected = set(entry.get("expected") or [])
            result = retrieve(
                col, q, k, folder=None, tag=None,
                precise=hyde, multi_query=not no_multi, auto_filter=True,
            )
            seen_paths = _dedup([m.get("file", "") for m in result["metas"]])
            hit, rr, recall = _score(expected, seen_paths)
            per_query.append((q, hit, rr, recall, seen_paths))

        tbl = Table(title=f"Queries sueltas (k={k})", show_lines=False)
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

        failed = [(q, paths) for q, hit, _, _, paths in per_query if not hit]
        if failed:
            console.print()
            console.print("[bold red]Queries sin hit — top-k recuperado:[/bold red]")
            for q, paths in failed:
                console.print(f"  [yellow]{q}[/yellow]")
                for p in paths[:k]:
                    console.print(f"    [dim]· {p}[/dim]")

        n = len(queries)
        hit_at_k = sum(1 for _, h, _, _, _ in per_query if h) / n
        mrr = sum(r for _, _, r, _, _ in per_query) / n
        recall_at_k = sum(r for _, _, _, r, _ in per_query) / n
        console.print()
        console.print(
            f"[bold]Singles:[/bold] hit@{k} {hit_at_k:.2%}  ·  "
            f"MRR {mrr:.3f}  ·  recall@{k} {recall_at_k:.2%}  ·  "
            f"[dim]n={n}[/dim]"
        )

    # ── Multi-turn chains ───────────────────────────────────────────────
    # Turn 0 runs as-is. For later turns we explicitly call reformulate_query
    # against accumulated history and pass the rewrite as the search query to
    # retrieve(..., precise=False) — keeps history-aware search on while
    # avoiding HyDE (bundled with precise=True, known to hurt on qwen2.5:3b).
    if chains:
        console.print()
        chain_rows: list[tuple[str, str, bool, float, float, list[str]]] = []
        per_chain_success: list[tuple[str, bool]] = []

        for chain in track(chains, description="Evaluando chains…"):
            chain_id = chain.get("id", "<sin id>")
            turns = chain.get("turns") or []
            history: list[dict] = []
            all_hit = True
            for i, turn in enumerate(turns):
                q = turn["question"]
                expected = set(turn.get("expected") or [])
                search_q = q if (i == 0 or not history) else reformulate_query(q, history)
                result = retrieve(
                    col, search_q, k, folder=None, tag=None,
                    precise=False, multi_query=not no_multi, auto_filter=True,
                )
                seen_paths = _dedup([m.get("file", "") for m in result["metas"]])
                hit, rr, recall = _score(expected, seen_paths)
                if not hit:
                    all_hit = False
                chain_rows.append((chain_id, q, hit, rr, recall, seen_paths))
                # Fake assistant turn: top retrieved path anchors the topic so
                # the next reformulation has concrete nouns to resolve pronouns
                # against without a real chat-model call.
                history.append({"role": "user", "content": q})
                top = seen_paths[0] if seen_paths else ""
                history.append({
                    "role": "assistant",
                    "content": f"(contexto recuperado: {top})",
                })
            per_chain_success.append((chain_id, all_hit))

        ctbl = Table(title=f"Chains multi-turno (k={k})", show_lines=False)
        ctbl.add_column("Chain", style="magenta", no_wrap=True)
        ctbl.add_column("Turno", style="cyan", overflow="fold", max_width=44)
        ctbl.add_column("Hit", justify="center")
        ctbl.add_column("RR", justify="right")
        ctbl.add_column("Recall", justify="right")
        for cid, q, hit, rr, rec, _ in chain_rows:
            ctbl.add_row(
                cid,
                q,
                "[green]✓[/green]" if hit else "[red]✗[/red]",
                f"{rr:.2f}",
                f"{rec:.2f}",
            )
        console.print(ctbl)

        failed_turns = [(cid, q, paths) for cid, q, hit, _, _, paths in chain_rows if not hit]
        if failed_turns:
            console.print()
            console.print("[bold red]Turns sin hit — top-k recuperado:[/bold red]")
            for cid, q, paths in failed_turns:
                console.print(f"  [magenta]{cid}[/magenta] [yellow]{q}[/yellow]")
                for p in paths[:k]:
                    console.print(f"    [dim]· {p}[/dim]")

        nt = len(chain_rows)
        nc = len(per_chain_success)
        chain_hit = sum(1 for _, _, h, _, _, _ in chain_rows if h) / nt if nt else 0.0
        chain_mrr = sum(r for _, _, _, r, _, _ in chain_rows) / nt if nt else 0.0
        chain_recall = sum(r for _, _, _, _, r, _ in chain_rows) / nt if nt else 0.0
        chain_success = sum(1 for _, ok in per_chain_success if ok) / nc if nc else 0.0
        console.print()
        console.print(
            f"[bold]Chains:[/bold] hit@{k} {chain_hit:.2%}  ·  "
            f"MRR {chain_mrr:.3f}  ·  recall@{k} {chain_recall:.2%}  ·  "
            f"chain_success {chain_success:.2%}  ·  "
            f"[dim]turns={nt}, chains={nc}[/dim]"
        )


@cli.command()
@click.argument("path")
@click.option("--apply", is_flag=True, help="Escribir los tags al frontmatter (por defecto solo imprime)")
@click.option("--max-tags", default=6, help="Cantidad máxima de tags a sugerir")
def autotag(path: str, apply: bool, max_tags: int):
    """Sugerir tags para una nota usando el vocabulario existente del vault.

    El helper model (qwen2.5:3b) ve los tags ya usados en el índice + el
    contenido de la nota y elige los que encajen. No inventa tags nuevos —
    mantiene consistencia con la taxonomía del vault.
    """
    note_path = VAULT_PATH / path if not path.startswith("/") else Path(path)
    if not note_path.is_file():
        console.print(f"[red]Nota no encontrada:[/red] {note_path}")
        return

    col = get_db()
    c = _load_corpus(col)
    vocab = sorted(c["tags"])
    raw = note_path.read_text(encoding="utf-8", errors="ignore")
    fm = parse_frontmatter(raw)
    current_tags = [str(t) for t in (fm.get("tags") or []) if t]
    body = clean_md(raw)[:3000]

    prompt = (
        "Sos un asistente que etiqueta notas personales. Elegí entre 3 y "
        f"{max_tags} tags DEL VOCABULARIO EXISTENTE que mejor describan esta nota. "
        "NO inventes tags nuevos. Devolvé SOLO una lista YAML de strings, "
        "sin explicación.\n\n"
        f"VOCABULARIO ({len(vocab)} tags): {', '.join(vocab)}\n\n"
        f"TÍTULO: {note_path.stem}\n\n"
        f"CONTENIDO:\n{body}\n\n"
        "TAGS:"
    )

    resp = ollama.chat(
        model=HELPER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=HELPER_OPTIONS,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    answer = resp.message.content.strip()

    # Parse: expect "- tag1\n- tag2" or "[tag1, tag2]" or plain list.
    vocab_set = {t.lower() for t in vocab}
    picked: list[str] = []
    for line in answer.splitlines():
        line = line.strip().strip("-*[]").strip().strip(",").strip("'\"")
        if not line:
            continue
        for tok in re.split(r"[,\s]+", line):
            tok = tok.strip("#'\"").lower()
            if tok in vocab_set and tok not in picked:
                picked.append(tok)
        if len(picked) >= max_tags:
            break

    if not picked:
        console.print("[yellow]El modelo no devolvió tags del vocabulario.[/yellow]")
        console.print(f"[dim]Respuesta raw: {answer[:200]}[/dim]")
        return

    merged = list(dict.fromkeys([*current_tags, *picked]))[:max_tags]
    new_tags = [t for t in picked if t not in current_tags]

    console.print(f"[bold cyan]Nota:[/bold cyan] {path}")
    console.print(f"[dim]Tags actuales:[/dim] {current_tags or '—'}")
    console.print(f"[bold green]Sugeridos:[/bold green] {picked}")
    console.print(f"[bold]Nuevos a añadir:[/bold] {new_tags or '—'}")

    if not apply or not new_tags:
        return

    # Rewrite frontmatter with merged tag list.
    if raw.startswith("---\n"):
        end = raw.find("\n---\n", 4)
        if end < 0:
            console.print("[red]Frontmatter mal formado, no se modifica.[/red]")
            return
        fm_text = raw[4:end]
        rest = raw[end + 5:]
        # Replace or insert tags: key — yaml.safe_dump-ish but we want to
        # preserve the rest of the frontmatter verbatim.
        new_fm_lines: list[str] = []
        skipped_tags = False
        in_tag_block = False
        for line in fm_text.splitlines():
            if in_tag_block and re.match(r"^\s*-\s+", line):
                continue  # drop old tag items
            in_tag_block = False
            if re.match(r"^tags\s*:", line):
                in_tag_block = True
                skipped_tags = True
                continue
            new_fm_lines.append(line)
        new_fm_lines.append("tags:")
        for t in merged:
            new_fm_lines.append(f"- {t}")
        new_raw = "---\n" + "\n".join(new_fm_lines) + "\n---\n" + rest
    else:
        # No frontmatter — prepend one.
        fm_block = "---\ntags:\n" + "\n".join(f"- {t}" for t in merged) + "\n---\n\n"
        new_raw = fm_block + raw

    note_path.write_text(new_raw, encoding="utf-8")
    console.print(f"[green]✓ Frontmatter actualizado.[/green]")
    _index_single_file(col, note_path)


@cli.command()
@click.option("--threshold", default=CONFIDENCE_RERANK_MIN,
              help=f"Queries con top_score ≤ este umbral se consideran 'gap' (default: {CONFIDENCE_RERANK_MIN})")
@click.option("--min-count", default=2, help="Queries mínimas por cluster para reportar")
@click.option("--days", default=60, help="Ventana de log a analizar (default: 60)")
def gaps(threshold: float, min_count: int, days: int):
    """Detectar temas consultados repetidamente sin respuesta en el vault.

    Agrupa queries low-confidence del log (confidence ≤ threshold) por
    similaridad de embedding. Cada cluster con ≥min-count apariciones es un
    candidato a nota nueva.
    """
    if not LOG_PATH.is_file():
        console.print("[yellow]No hay log todavía.[/yellow]")
        return
    from datetime import timedelta as _td
    cutoff = datetime.now() - _td(days=days)

    entries = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        score = e.get("top_score")
        if score is None or score > threshold:
            continue
        try:
            ts = datetime.fromisoformat(e.get("ts", ""))
        except Exception:
            continue
        if ts < cutoff:
            continue
        q = e.get("q", "").strip()
        if q:
            entries.append({"q": q, "ts": ts, "score": score})

    if not entries:
        console.print("[green]No hay gaps persistentes en la ventana analizada.[/green]")
        return

    # Dedupe near-identical queries by lowercase stem, then cluster by
    # embedding similarity (cheap — N low-conf queries is small).
    queries_unique: list[str] = []
    query_counts: dict[str, int] = {}
    for e in entries:
        key = " ".join(_tokenize(e["q"]))
        if key not in query_counts:
            queries_unique.append(e["q"])
            query_counts[key] = 0
        query_counts[key] += 1

    if len(queries_unique) == 1:
        q = queries_unique[0]
        count = list(query_counts.values())[0]
        if count >= min_count:
            console.print(
                f"[yellow]Gap:[/yellow] [bold]{q}[/bold] "
                f"[dim](×{count})[/dim]"
            )
            console.print("[dim]Sugerencia: creá una nota sobre este tema.[/dim]")
        return

    # Cluster by cosine similarity on embeddings.
    embeddings = embed(queries_unique)
    clusters: list[list[int]] = []
    assigned = [False] * len(queries_unique)
    for i in range(len(queries_unique)):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, len(queries_unique)):
            if assigned[j]:
                continue
            if cosine_sim(embeddings[i], embeddings[j]) >= 0.75:
                group.append(j)
                assigned[j] = True
        clusters.append(group)

    # Rank clusters by total query count.
    cluster_rows: list[tuple[int, list[str]]] = []
    for group in clusters:
        total = sum(
            query_counts.get(" ".join(_tokenize(queries_unique[idx])), 0)
            for idx in group
        )
        if total < min_count:
            continue
        sample = [queries_unique[idx] for idx in group]
        cluster_rows.append((total, sample))
    cluster_rows.sort(key=lambda x: -x[0])

    if not cluster_rows:
        console.print("[green]No hay gaps persistentes (todos los grupos < min-count).[/green]")
        return

    console.print(
        f"[bold yellow]Gaps detectados[/bold yellow] "
        f"[dim]({len(cluster_rows)} cluster(s), últimos {days} días)[/dim]"
    )
    console.print()
    for total, samples in cluster_rows:
        console.print(f"  [bold magenta]×{total}[/bold magenta]  {samples[0]}")
        for s in samples[1:4]:
            console.print(f"         [dim]· {s}[/dim]")
        if len(samples) > 4:
            console.print(f"         [dim]· … {len(samples) - 4} variantes más[/dim]")
        console.print()
    console.print(
        "[dim]Cada cluster es un tema que consultaste sin encontrar respuesta — "
        "candidato a una nota nueva.[/dim]"
    )


@cli.command()
@click.argument("query", required=False)
@click.option("--tag", default=None, help="Filtrar por tag")
@click.option("--folder", default=None, help="Filtrar por carpeta")
@click.option("--limit", default=30, help="Cantidad de notas a mostrar")
def timeline(query: str | None, tag: str | None, folder: str | None, limit: int):
    """Ver notas ordenadas por fecha de modificación (más recientes primero).

    Opcionalmente filtrable por tag, carpeta o query semántica. Usa
    frontmatter `modified` — si falta, cae a `created`.
    """
    col = get_db()
    c = _load_corpus(col)
    if query:
        # Rank by semantic similarity first, then show with dates.
        result = retrieve(
            col, query, limit, folder, tag=tag,
            precise=False, multi_query=True, auto_filter=False,
        )
        files = []
        seen = set()
        for m in result["metas"]:
            f = m.get("file", "")
            if f in seen or is_excluded(f):
                continue
            seen.add(f)
            files.append(m)
    else:
        files = _filter_files(c["metas"], tag, folder)

    files.sort(
        key=lambda m: m.get("modified") or m.get("created") or "",
        reverse=True,
    )
    files = files[:limit]

    if not files:
        console.print("[yellow]Sin resultados.[/yellow]")
        return

    tbl = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
    tbl.add_column(style="dim")            # date
    tbl.add_column(style="bold magenta")   # title
    tbl.add_column(style="cyan dim")       # path
    for m in files:
        stamp = (m.get("modified") or m.get("created") or "")[:10]
        tbl.add_row(stamp or "—", m.get("note", ""), m.get("file", ""))
    console.print(tbl)


@cli.command()
@click.argument("note_title")
@click.option("--depth", default=1, help="Niveles de vecindad (1 = solo vecinos directos, 2 = vecinos de vecinos)")
@click.option("--output", default=None, help="Ruta .canvas de salida (default: 00-Inbox/graph-{note}.canvas)")
def graph(note_title: str, depth: int, output: str | None):
    """Exportar un Obsidian canvas con la vecindad de una nota por wikilinks.

    NOTE_TITLE: título exacto (file stem) de la nota semilla. Atravesa outlinks
    y backlinks hasta `--depth`. El canvas resultante puede abrirse en
    Obsidian para ver el clúster conceptual.
    """
    import json as _json
    col = get_db()
    c = _load_corpus(col)

    # Resolver título → primer path (Obsidian permite múltiples notas con mismo
    # stem; tomamos el primero no-excluido).
    paths = [p for p in c["title_to_paths"].get(note_title, set()) if not is_excluded(p)]
    if not paths:
        console.print(f"[red]Nota no encontrada en el índice:[/red] {note_title}")
        return
    seed_path = sorted(paths)[0]

    # BFS outlinks + backlinks
    visited: dict[str, int] = {seed_path: 0}
    edges: set[tuple[str, str]] = set()
    frontier = [seed_path]
    for hop in range(1, depth + 1):
        next_frontier: list[str] = []
        for p in frontier:
            # outlinks
            for target_title in c["outlinks"].get(p, []):
                for tp in c["title_to_paths"].get(target_title, set()):
                    if is_excluded(tp):
                        continue
                    edges.add((p, tp))
                    if tp not in visited:
                        visited[tp] = hop
                        next_frontier.append(tp)
            # backlinks — notes that link to the current node's TITLE
            title = _path_to_title(c, p)
            if title:
                for linker in c["backlinks"].get(title, set()):
                    if is_excluded(linker):
                        continue
                    edges.add((linker, p))
                    if linker not in visited:
                        visited[linker] = hop
                        next_frontier.append(linker)
        frontier = next_frontier
        if not frontier:
            break

    # Build Obsidian canvas JSON (file nodes + edges)
    nodes, canvas_edges = [], []
    path_to_id: dict[str, str] = {}
    for i, (p, hop) in enumerate(sorted(visited.items(), key=lambda kv: (kv[1], kv[0]))):
        node_id = f"n{i}"
        path_to_id[p] = node_id
        # Radial layout: seed at center, rings per hop
        import math
        if hop == 0:
            x, y = 0, 0
        else:
            ring_members = [q for q, h in visited.items() if h == hop]
            idx = ring_members.index(p)
            n = len(ring_members)
            r = hop * 600
            angle = 2 * math.pi * idx / max(1, n)
            x, y = int(r * math.cos(angle)), int(r * math.sin(angle))
        nodes.append({
            "id": node_id,
            "type": "file",
            "file": p,
            "x": x, "y": y,
            "width": 320, "height": 160,
        })
    for i, (a, b) in enumerate(sorted(edges)):
        if a in path_to_id and b in path_to_id:
            canvas_edges.append({
                "id": f"e{i}",
                "fromNode": path_to_id[a], "fromSide": "right",
                "toNode": path_to_id[b], "toSide": "left",
            })

    canvas = {"nodes": nodes, "edges": canvas_edges}

    if output:
        out_path = Path(output)
        if not out_path.is_absolute():
            out_path = VAULT_PATH / output
    else:
        safe = re.sub(r"[/\\:\n]", " ", note_title).strip()
        out_path = VAULT_PATH / INBOX_FOLDER / f"graph-{safe}.canvas"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps(canvas, ensure_ascii=False, indent=2), encoding="utf-8")

    rel = out_path.relative_to(VAULT_PATH) if str(out_path).startswith(str(VAULT_PATH)) else out_path
    console.print(
        f"[green]✓ Canvas generado:[/green] [bold cyan]{rel}[/bold cyan] "
        f"[dim]({len(nodes)} nodos, {len(canvas_edges)} aristas, depth={depth})[/dim]"
    )


# ── Weekly narrative digest (Contradiction Radar phase 3) ─────────────────────

def _iso_week_label(dt: datetime) -> str:
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def _parse_iso_week(week: str) -> tuple[datetime, datetime]:
    """Parse 'YYYY-WNN' to [Monday 00:00, next Monday 00:00) local time."""
    from datetime import timedelta as _td
    m = re.match(r"^(\d{4})-W(\d{1,2})$", week.strip())
    if not m:
        raise click.BadParameter(
            f"Formato inválido: {week!r} (esperado YYYY-WNN, ej. 2026-W15)"
        )
    year, wk = int(m.group(1)), int(m.group(2))
    monday = datetime.strptime(f"{year}-W{wk:02d}-1", "%G-W%V-%u")
    return monday, monday + _td(days=7)


def _collect_week_evidence(
    start: datetime,
    end: datetime,
    vault: Path,
    query_log: Path,
    contradiction_log: Path,
) -> dict:
    """Gather evidence for the weekly digest from three sources.

    Returns dict with keys:
      - recent_notes: vault files modified in [start, end)
      - fm_contradictions: notes whose YAML has `contradicts: [paths]` (snapshot)
      - index_contradictions: sidecar log entries in [start, end) with ≥1 contradict
      - query_contradictions: queries.jsonl entries in window with contradictions
      - low_conf_queries: queries in window with top_score <= CONFIDENCE_RERANK_MIN
    """
    recent: list[dict] = []
    fm_contrad: list[dict] = []
    if vault.is_dir():
        for p in vault.rglob("*.md"):
            try:
                rel = str(p.relative_to(vault))
            except ValueError:
                continue
            if is_excluded(rel):
                continue
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
            except OSError:
                continue
            try:
                raw = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            fm = parse_frontmatter(raw)
            contradicts = fm.get("contradicts") or []
            if isinstance(contradicts, list) and contradicts:
                targets = [str(t) for t in contradicts if t]
                if targets:
                    fm_contrad.append({"path": rel, "targets": targets})
            if start <= mtime < end:
                snippet = clean_md(raw)[:300].strip()
                recent.append({
                    "path": rel,
                    "title": p.stem,
                    "modified": mtime.isoformat(timespec="seconds"),
                    "snippet": snippet,
                })
    recent.sort(key=lambda r: r["modified"], reverse=True)

    def _read_jsonl_in_window(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        out: list[dict] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            try:
                ts = datetime.fromisoformat(e.get("ts", ""))
            except Exception:
                continue
            if start <= ts < end:
                out.append(e)
        return out

    index_contrad: list[dict] = []
    for e in _read_jsonl_in_window(contradiction_log):
        if e.get("cmd") != "contradict_index":
            continue
        entries = e.get("contradicts") or []
        if not entries:
            continue
        index_contrad.append({
            "ts": e.get("ts"),
            "subject_path": e.get("subject_path", ""),
            "targets": [
                {"path": c.get("path", ""), "why": c.get("why", "")}
                for c in entries if isinstance(c, dict)
            ],
        })

    query_contrad: list[dict] = []
    low_conf: list[dict] = []
    for e in _read_jsonl_in_window(query_log):
        if e.get("cmd") != "query":
            continue
        contrad = e.get("contradictions")
        if isinstance(contrad, list) and contrad:
            for c in contrad:
                if isinstance(c, dict) and c.get("path"):
                    query_contrad.append({
                        "ts": e.get("ts"),
                        "q": e.get("q", ""),
                        "path": c.get("path", ""),
                        "why": c.get("why", ""),
                    })
        score = e.get("top_score")
        if isinstance(score, (int, float)) and score <= CONFIDENCE_RERANK_MIN:
            q = (e.get("q") or "").strip()
            if q:
                low_conf.append({
                    "ts": e.get("ts"),
                    "q": q,
                    "top_score": float(score),
                })

    return {
        "recent_notes": recent,
        "fm_contradictions": fm_contrad,
        "index_contradictions": index_contrad,
        "query_contradictions": query_contrad,
        "low_conf_queries": low_conf,
    }


def _render_digest_prompt(week_label: str, ev: dict) -> str:
    parts: list[str] = [
        f"Contexto del vault para la semana {week_label}.",
        "",
    ]
    if ev["recent_notes"]:
        parts.append("Notas creadas/modificadas esta semana:")
        for n in ev["recent_notes"][:40]:
            snippet = n["snippet"].replace("\n", " ")
            parts.append(f"- [[{n['title']}]] ({n['path']}) — {snippet}")
        parts.append("")
    if ev["index_contradictions"]:
        parts.append("Contradicciones flaggeadas al indexar (fase 2):")
        for ic in ev["index_contradictions"][:40]:
            for t in ic["targets"]:
                parts.append(
                    f"- {ic['subject_path']} vs {t['path']} — {t['why']}"
                )
        parts.append("")
    if ev["fm_contradictions"]:
        parts.append(
            "Estado actual de `contradicts:` en frontmatter (snapshot del vault):"
        )
        for fc in ev["fm_contradictions"][:40]:
            parts.append(f"- {fc['path']} → {', '.join(fc['targets'])}")
        parts.append("")
    if ev["query_contradictions"]:
        parts.append("Contradicciones detectadas al responder queries (fase 1):")
        for qc in ev["query_contradictions"][:40]:
            parts.append(
                f"- pregunta {qc['q']!r} → {qc['path']}: {qc['why']}"
            )
        parts.append("")
    if ev["low_conf_queries"]:
        parts.append("Queries con baja confianza (posibles gaps persistentes):")
        for lq in ev["low_conf_queries"][:30]:
            parts.append(f"- {lq['q']!r} (top_score={lq['top_score']:.3f})")
        parts.append("")
    parts.append(
        "Tarea: armá un review en primera persona de qué pasó en mi vault "
        "esta semana: qué conceptos nuevos emergieron, qué posiciones se "
        "movieron vs posiciones previas, qué tensiones se revelaron, qué "
        "gaps persistentes aparecieron. Prosa narrativa, no bullets. "
        "400-600 palabras. Citá notas con [[wikilinks]] de Obsidian. Si "
        "alguno de los contextos está vacío, obvialo sin mencionarlo "
        "explícitamente."
    )
    return "\n".join(parts)


def _generate_digest_narrative(prompt: str) -> str:
    resp = ollama.chat(
        model=resolve_chat_model(),
        messages=[{"role": "user", "content": prompt}],
        options=CHAT_OPTIONS,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    return (resp.message.content or "").strip()


DIGEST_FOLDER = "05-Reviews"


@cli.command()
@click.option("--week", "week_opt", default=None,
              help="Semana ISO YYYY-WNN (default: la semana que termina hoy)")
@click.option("--days", default=7, show_default=True,
              help="Ventana en días (ignorado si se pasa --week)")
@click.option("--dry-run", is_flag=True,
              help="No escribas ni indexes — mostrá el output")
def digest(week_opt: str | None, days: int, dry_run: bool):
    """Weekly narrative digest del vault (Contradiction Radar fase 3).

    Consume notas modificadas, frontmatter `contradicts:`, el sidecar log
    de contradicciones index-time, las contradicciones query-time y las
    queries low-confidence. Genera prosa narrativa con command-r y la
    guarda en `05-Reviews/YYYY-WNN.md` (auto-indexado).
    """
    from datetime import timedelta as _td
    if week_opt:
        try:
            start, end = _parse_iso_week(week_opt)
        except click.BadParameter as e:
            console.print(f"[red]{e.message}[/red]")
            return
        week_label = week_opt
    else:
        end = datetime.now()
        start = end - _td(days=days)
        week_label = _iso_week_label(end)

    ev = _collect_week_evidence(
        start, end, VAULT_PATH, LOG_PATH, CONTRADICTION_LOG_PATH,
    )

    total_signals = (
        len(ev["recent_notes"]) + len(ev["fm_contradictions"])
        + len(ev["index_contradictions"])
        + len(ev["query_contradictions"]) + len(ev["low_conf_queries"])
    )
    if total_signals == 0:
        console.print(
            f"[yellow]Sin evidencia para {week_label}: "
            "0 notas modificadas, 0 contradicciones, 0 queries low-confidence."
            "[/yellow]"
        )
        return

    console.print(
        f"[dim]Evidencia {week_label}:[/dim] "
        f"{len(ev['recent_notes'])} notas · "
        f"{len(ev['index_contradictions'])} contradicciones index · "
        f"{len(ev['fm_contradictions'])} frontmatter · "
        f"{len(ev['query_contradictions'])} query · "
        f"{len(ev['low_conf_queries'])} low-conf"
    )

    prompt = _render_digest_prompt(week_label, ev)
    with console.status(
        "[dim]Generando narrativa con command-r…[/dim]", spinner="dots"
    ):
        narrative = _generate_digest_narrative(prompt)

    if not narrative:
        console.print("[red]El modelo devolvió respuesta vacía. Abortando.[/red]")
        return

    now = datetime.now().isoformat(timespec="seconds")
    fm_lines = [
        "---",
        f"created: '{now}'",
        f"modified: '{now}'",
        "tags:",
        "- review",
        "- weekly-digest",
        f"week: '{week_label}'",
        "---",
    ]
    body = (
        "\n".join(fm_lines)
        + f"\n\n# Review {week_label}\n\n{narrative.strip()}\n"
    )

    if dry_run:
        console.rule(f"[bold]Digest {week_label} (dry-run)[/bold]")
        console.print(body, markup=False, highlight=False)
        return

    path = VAULT_PATH / DIGEST_FOLDER / f"{week_label}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path = path.with_name(
            f"{week_label} ({datetime.now().strftime('%H%M%S')}).md"
        )
    path.write_text(body, encoding="utf-8")

    try:
        col = get_db()
        _index_single_file(col, path)
    except Exception as e:
        console.print(
            f"[yellow]Nota escrita pero auto-index falló: {e}[/yellow]"
        )

    try:
        rel = path.relative_to(VAULT_PATH)
    except ValueError:
        rel = path
    console.print(
        f"[green]✓ Digest guardado:[/green] [bold cyan]{rel}[/bold cyan]"
    )


def _path_to_title(corpus: dict, path: str) -> str | None:
    for title, paths in corpus["title_to_paths"].items():
        if path in paths:
            return title
    return None


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


# ── AGENT LOOP ────────────────────────────────────────────────────────────────

def _agent_tool_search(query: str, k: int = 5) -> str:
    """Buscar chunks relevantes en el vault de Obsidian.

    Args:
        query: Pregunta o tema a buscar, en lenguaje natural.
        k: Cantidad de chunks a devolver (default 5).

    Returns:
        JSON con lista de {note, path, score, content} de los chunks más
        relevantes, ordenados por score descendente.
    """
    col = get_db()
    result = retrieve(
        col, query, k, folder=None, tag=None,
        precise=False, multi_query=True, auto_filter=True,
    )
    out = [
        {
            "note": m.get("note", ""),
            "path": m.get("file", ""),
            "score": round(float(s), 3),
            "content": d[:1000],
        }
        for d, m, s in zip(result["docs"], result["metas"], result["scores"])
    ]
    return json.dumps(out, ensure_ascii=False)


def _agent_tool_read_note(path: str) -> str:
    """Leer el contenido completo de una nota del vault.

    Args:
        path: Ruta relativa al vault, por ej. "02-Areas/Coaching/Ikigai.md".

    Returns:
        Texto markdown completo, o mensaje de error si no existe.
    """
    if not path.endswith(".md"):
        return "Error: el path debe terminar en .md"
    full = (VAULT_PATH / path).resolve()
    try:
        full.relative_to(VAULT_PATH.resolve())
    except ValueError:
        return "Error: path fuera del vault"
    if not full.is_file():
        return f"Error: nota no encontrada: {path}"
    return full.read_text(encoding="utf-8", errors="ignore")


def _agent_tool_list_notes(folder: str | None = None, tag: str | None = None, limit: int = 30) -> str:
    """Listar notas del vault, opcionalmente filtradas por carpeta o tag.

    Args:
        folder: Carpeta vault-relativa para filtrar (ej. "02-Areas/Coaching").
        tag: Tag sin '#' prefix (ej. "coaching").
        limit: Cantidad máxima a devolver.

    Returns:
        JSON con lista de {note, path, tags, modified}.
    """
    col = get_db()
    c = _load_corpus(col)
    files = _filter_files(c["metas"], tag, folder)
    files.sort(key=lambda m: m.get("modified") or "", reverse=True)
    out = [
        {
            "note": m.get("note", ""),
            "path": m.get("file", ""),
            "tags": m.get("tags", ""),
            "modified": m.get("modified", ""),
        }
        for m in files[:limit]
    ]
    return json.dumps(out, ensure_ascii=False)


_AGENT_PENDING_WRITES: list[dict] = []


def _agent_tool_propose_write(path: str, content: str, rationale: str = "") -> str:
    """Proponer crear o sobreescribir una nota del vault. NO escribe
    inmediatamente — registra la propuesta para que el usuario confirme.

    Usá esta tool para crear notas resumen, agregar notas derivadas, o
    guardar outputs. Siempre bajo rutas del vault (00-Inbox/, 01-Projects/,
    02-Areas/, 03-Resources/). Nunca en dotfolders.

    Args:
        path: Ruta relativa al vault. Debe terminar en .md.
        content: Contenido markdown completo (incluye frontmatter si querés).
        rationale: Breve justificación (por qué esta nota).

    Returns:
        Mensaje de confirmación registrado.
    """
    if not path.endswith(".md"):
        return "Error: path debe terminar en .md"
    if is_excluded(path):
        return "Error: path en dotfolder rechazado"
    _AGENT_PENDING_WRITES.append({"path": path, "content": content, "rationale": rationale})
    return f"Propuesta registrada: {path} ({len(content)} chars). El usuario verá la acción antes de ejecutarse."


_AGENT_SYSTEM = (
    "Sos un asistente agéntico sobre un vault de Obsidian personal. "
    "Tenés tools para buscar, leer y listar notas del vault, y para PROPONER "
    "creaciones o modificaciones. No ejecutás writes vos — sólo los proponés "
    "vía `propose_write`; el usuario los confirma después.\n\n"
    "Reglas:\n"
    "1. Antes de proponer writes, usá search/read/list para juntar contexto "
    "   del vault real. No inventes paths ni contenidos.\n"
    "2. Las notas propuestas deben vivir bajo 00-Inbox/, 01-Projects/, "
    "   02-Areas/ o 03-Resources/. Default: 00-Inbox/ para capturas nuevas.\n"
    "3. Incluí frontmatter YAML válido con `created`, `tags` del vocabulario "
    "   existente cuando sea posible, y `related: [[wikilink]]` a notas "
    "   fuente cuando derivás de ellas.\n"
    "4. Cuando terminás, respondé con un resumen breve en lenguaje natural "
    "   de lo que hiciste y qué proponés.\n"
    "5. Si la instrucción es ambigua o no podés completarla con el vault, "
    "   decilo — no inventes."
)


@cli.command()
@click.argument("instruction")
@click.option("--yes", is_flag=True, help="Ejecutar writes propuestos sin confirmar uno por uno")
@click.option("--max-iterations", default=8, help="Cap de iteraciones de tool calling (default 8)")
def do(instruction: str, yes: bool, max_iterations: int):
    """Ejecutar una instrucción agéntica sobre el vault.

    Ejemplos:
        rag do "armá un resumen de todas mis notas sobre ikigai en una nota nueva"
        rag do "listame qué referentes tengo en coaching y proponé una nota índice"
        rag do "buscá notas sin tags y sugerí tags basados en contenido"

    Los writes se proponen primero (`propose_write` tool) y se confirman al
    final uno por uno (o todos con --yes).
    """
    _AGENT_PENDING_WRITES.clear()

    tools = [
        _agent_tool_search,
        _agent_tool_read_note,
        _agent_tool_list_notes,
        _agent_tool_propose_write,
    ]
    tool_fns = {fn.__name__: fn for fn in tools}

    messages = [
        {"role": "system", "content": _AGENT_SYSTEM},
        {"role": "user", "content": instruction},
    ]

    console.print(Panel(f"[bold cyan]{instruction}[/bold cyan]", border_style="cyan"))

    model = resolve_chat_model()
    for it in range(max_iterations):
        with console.status(f"[dim]pensando (iter {it + 1}/{max_iterations})…[/dim]", spinner="dots"):
            resp = ollama.chat(
                model=model,
                messages=messages,
                tools=tools,
                options=CHAT_OPTIONS,
                keep_alive=OLLAMA_KEEP_ALIVE,
            )

        msg = resp.message
        # Persist the assistant turn for the follow-up tool messages.
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])],
        })

        if not msg.tool_calls:
            # Final answer.
            console.print()
            console.print(render_response(msg.content or ""))
            break

        # Execute each tool call.
        for tc in msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments or {}
            # command-r wraps args as {"tool_name": "...", "parameters": {...}}.
            # Unwrap so we can call the Python function directly.
            if isinstance(args, dict) and "parameters" in args and isinstance(args["parameters"], (dict, str)):
                params = args["parameters"]
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except Exception:
                        params = {}
                args = params
            # Drop empty-string optional args so defaults kick in.
            args = {k: v for k, v in args.items() if v not in ("", None)}
            console.print(f"  [dim]→ {name}({', '.join(f'{k}={str(v)[:40]!r}' for k, v in args.items())})[/dim]")
            fn = tool_fns.get(name)
            if not fn:
                result = f"Error: tool '{name}' no existe"
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = f"Error de argumentos en {name}: {e}"
                except Exception as e:
                    result = f"Error ejecutando {name}: {e}"
            messages.append({
                "role": "tool",
                "name": name,
                "content": result if isinstance(result, str) else json.dumps(result),
            })
    else:
        console.print(f"[yellow]⚠ Cap de {max_iterations} iteraciones alcanzado.[/yellow]")

    # Show pending writes, confirm each.
    if _AGENT_PENDING_WRITES:
        console.print()
        console.print(Rule(title="[bold yellow]Writes propuestos[/bold yellow]", style="yellow"))
        col = get_db()
        for i, w in enumerate(_AGENT_PENDING_WRITES, 1):
            console.print()
            console.print(f"[bold]{i}. {w['path']}[/bold]  [dim]({len(w['content'])} chars)[/dim]")
            if w.get("rationale"):
                console.print(f"   [dim italic]{w['rationale']}[/dim italic]")
            console.print(Markdown(w["content"][:600] + ("…" if len(w["content"]) > 600 else "")))

            if yes:
                apply = True
            else:
                try:
                    resp = console.input("   [bold green]Crear?[/bold green] [y/N] ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    console.print("   [yellow]cancelado[/yellow]")
                    break
                apply = resp in ("y", "s", "yes", "si", "sí")

            if apply:
                full = VAULT_PATH / w["path"]
                full.parent.mkdir(parents=True, exist_ok=True)
                if full.exists():
                    full = full.with_name(f"{full.stem} ({datetime.now().strftime('%H%M%S')}).md")
                full.write_text(w["content"], encoding="utf-8")
                try:
                    _index_single_file(col, full)
                except Exception:
                    pass
                console.print(f"   [green]✓ escrito:[/green] {full.relative_to(VAULT_PATH)}")
            else:
                console.print("   [dim]saltado[/dim]")
    else:
        console.print("[dim](sin writes propuestos)[/dim]")


_URLS_BACKFILL_DONE = False


def _maybe_backfill_urls() -> None:
    """Auto-backfill the URL sub-index when it's empty but the main collection
    isn't — covers the "user upgraded past v0 and never ran `rag links
    --rebuild`" path. Idempotent within a process; runs at most once.
    """
    global _URLS_BACKFILL_DONE
    if _URLS_BACKFILL_DONE:
        return
    _URLS_BACKFILL_DONE = True
    try:
        col_urls = get_urls_db()
        if col_urls.count() > 0:
            return
        main = get_db()
        if main.count() == 0:
            return
    except Exception:
        return
    console.print(
        "[dim]URL index vacío — backfill automático (~1 min)…[/dim]"
    )
    try:
        _rebuild_urls_index()
    except Exception as e:
        console.print(f"[yellow]Backfill falló: {e}[/yellow]")


def find_urls(
    query: str,
    k: int = 10,
    folder: str | None = None,
    tag: str | None = None,
) -> list[dict]:
    """Semantic-by-context search over the URL sub-index.

    The collection embeds the prose around each URL (not the URL string), so a
    query like "documentación de claude code" matches notes that introduce or
    annotate the relevant link. Reranks the top candidates against the query
    with the cross-encoder, dedups by URL, returns up to k items.

    First-call autopilot: if the URL collection is empty but the vault is
    indexed, runs `_rebuild_urls_index` silently so the user never has to
    remember `rag links --rebuild` after upgrading.
    """
    _maybe_backfill_urls()
    col_urls = get_urls_db()
    if col_urls.count() == 0:
        return []
    try:
        q_embed = embed([query])[0]
    except Exception:
        return []
    where = build_where(folder, tag)
    n = min(k * 3 + 5, col_urls.count())
    kwargs: dict = {
        "query_embeddings": [q_embed], "n_results": n,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    res = col_urls.query(**kwargs)
    docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
    if not docs:
        return []
    try:
        reranker = get_reranker()
        scores = [float(s) for s in reranker.predict(
            [(query, d) for d in docs], show_progress_bar=False,
        )]
    except Exception:
        scores = [1.0 - float(d) for d in dists]
    items: list[dict] = []
    seen: set[str] = set()
    per_file: dict[str, int] = {}
    PER_FILE_CAP = 2  # diversify so one note can't dominate the top-k
    for d, m, score in sorted(
        zip(docs, metas, scores), key=lambda x: x[2], reverse=True,
    ):
        url = m.get("url", "")
        path_str = m.get("file", "")
        if not url or url in seen:
            continue
        if per_file.get(path_str, 0) >= PER_FILE_CAP:
            continue
        seen.add(url)
        per_file[path_str] = per_file.get(path_str, 0) + 1
        items.append({
            "url": url,
            "anchor": m.get("anchor", ""),
            "path": path_str,
            "note": m.get("note", ""),
            "line": m.get("line", 0),
            "context": d,
            "score": score,
        })
        if len(items) >= k:
            break
    return items


def render_links(items: list[dict], plain: bool = False) -> None:
    """Render link-finder results. Plain mode = bot/script-friendly text."""
    if not items:
        if plain:
            click.echo("Sin URLs.")
        else:
            console.print("[yellow]Sin URLs en el vault para esa consulta.[/yellow]")
        return
    if plain:
        for i, it in enumerate(items, 1):
            click.echo(f"{i}. {it['url']}")
            label = it.get("anchor") or it.get("note", "")
            if label:
                click.echo(f"   {label}")
            click.echo(f"   en {it['path']}:{it.get('line', 0)}")
        return
    console.print()
    console.print(Rule(title="[bold cyan]🔗 Links[/bold cyan]", style="cyan"))
    for i, it in enumerate(items, 1):
        console.print()
        head = Text()
        head.append(f" {i}. ", style="bold cyan")
        # Real OSC 8 hyperlink so Cmd/Ctrl-click opens in the default browser.
        head.append(it["url"], style=f"underline blue link {it['url']}")
        console.print(head)
        if it.get("anchor"):
            console.print(f"    [italic]{it['anchor']}[/italic]")
        loc = Text()
        loc.append("    en ", style="dim")
        loc.append(it.get("note", ""), style=_file_link_style(it["path"], "magenta"))
        loc.append(
            f"  ({it['path']}:{it.get('line', 0)})",
            style=_file_link_style(it["path"], "dim"),
        )
        console.print(loc)
        ctx = (it.get("context") or "").strip()
        if ctx:
            ctx_short = ctx[:240] + ("…" if len(ctx) > 240 else "")
            console.print(f"    [dim]{ctx_short}[/dim]")


def _rebuild_urls_index() -> dict:
    """Re-extract URLs from every note in the vault into the URL collection
    without touching chunk embeddings. Cheap (no LLM, no chunk re-embed) — use
    after upgrades that change URL extraction logic, or to backfill the URL
    index on a vault that was already chunk-indexed before this feature.
    """
    col_urls = get_urls_db()
    md_files = [
        p for p in VAULT_PATH.rglob("*.md")
        if not is_excluded(str(p.relative_to(VAULT_PATH)))
    ]
    console.print(f"[cyan]Re-extrayendo URLs de {len(md_files)} notas...[/cyan]")
    total = 0
    files_with_urls = 0
    for path in track(md_files, description="URLs..."):
        try:
            doc_id_prefix = str(path.relative_to(VAULT_PATH))
            folder = str(path.relative_to(VAULT_PATH).parent)
            raw = path.read_text(encoding="utf-8", errors="ignore")
            fm = parse_frontmatter(raw)
            tags = [str(t) for t in (fm.get("tags") or []) if t]
            n = _index_urls(col_urls, doc_id_prefix, raw, path.stem, folder, tags)
            total += n
            if n:
                files_with_urls += 1
        except Exception:
            continue
    console.print(
        f"[green]Listo. {total} URLs en {files_with_urls} notas.[/green]"
    )
    return {"urls": total, "files": files_with_urls}


@cli.command()
@click.argument("query", required=False)
@click.option("-k", default=10, help="Cantidad de URLs a devolver (default: 10)")
@click.option("--folder", default=None, help="Filtrar por carpeta")
@click.option("--tag", default=None, help="Filtrar por tag")
@click.option("--open", "open_idx", type=int, default=None,
              help="Abrir la URL del rank N en el browser por defecto (macOS: open)")
@click.option("--plain", is_flag=True, help="Salida plana sin colores/paneles")
@click.option("--rebuild", is_flag=True,
              help="Re-extraer URLs de todas las notas sin re-embeddar chunks")
def links(query: str | None, k: int, folder: str | None, tag: str | None,
          open_idx: int | None, plain: bool, rebuild: bool):
    """Buscar URLs en el vault por contexto semántico — respuesta sin LLM."""
    if rebuild:
        _rebuild_urls_index()
        return
    if not query:
        console.print("[yellow]Pasá una query o usá --rebuild.[/yellow]")
        return
    items = find_urls(query, k=k, folder=folder, tag=tag)
    if open_idx is not None:
        if not (1 <= open_idx <= len(items)):
            msg = f"Índice {open_idx} fuera de rango (1-{len(items)})"
            click.echo(msg) if plain else console.print(f"[red]{msg}[/red]")
            return
        url = items[open_idx - 1]["url"]
        import subprocess
        try:
            subprocess.run(["open", url], check=False)
        except Exception as e:
            msg = f"No pude abrir: {e}"
            click.echo(msg) if plain else console.print(f"[red]{msg}[/red]")
            return
        click.echo(f"abierto: {url}") if plain else console.print(f"[green]✓ Abierto:[/green] {url}")
        return
    render_links(items, plain=plain)
    log_query_event({
        "cmd": "links", "q": query, "n_results": len(items),
        "folder": folder, "tag": tag,
        "top_url": items[0]["url"] if items else None,
    })


@cli.command()
@click.option("--threshold", default=0.85, show_default=True,
              help="Cosine mínimo para considerar duplicado (sobre centroides)")
@click.option("--folder", default=None, help="Acotar a este folder")
@click.option("--limit", default=50, show_default=True,
              help="Top N pares a mostrar")
@click.option("--plain", is_flag=True, help="Salida plana sin colores")
def dupes(threshold: float, folder: str | None, limit: int, plain: bool):
    """Buscar pares de notas potencialmente duplicadas (centroides similares)."""
    pairs = find_duplicate_notes(col=get_db(), threshold=threshold, folder=folder, limit=limit)
    if not pairs:
        msg = f"Sin pares con cosine ≥ {threshold}."
        click.echo(msg) if plain else console.print(f"[yellow]{msg}[/yellow]")
        return
    if plain:
        for p in pairs:
            click.echo(f"{p['similarity']:.2f}  {p['a_path']}  ↔  {p['b_path']}")
        return
    console.print()
    console.print(Rule(
        title=f"[bold yellow]🔁 {len(pairs)} par(es) sospechoso(s) (cosine ≥ {threshold})[/bold yellow]",
        style="yellow",
    ))
    for p in pairs:
        console.print()
        console.print(
            f"[bold]{p['similarity']:.3f}[/bold]  "
            f"[magenta]{p['a_note']}[/magenta] "
            f"[dim]({p['a_path']})[/dim]"
        )
        console.print(
            f"      ↔  [magenta]{p['b_note']}[/magenta] "
            f"[dim]({p['b_path']})[/dim]"
        )
        if p.get("snippet_a"):
            console.print(f"      [dim]A:[/dim] [dim italic]{p['snippet_a'][:140]}[/dim italic]")
        if p.get("snippet_b"):
            console.print(f"      [dim]B:[/dim] [dim italic]{p['snippet_b'][:140]}[/dim italic]")
    log_query_event({
        "cmd": "dupes", "threshold": threshold, "folder": folder,
        "n_pairs": len(pairs),
    })


@cli.command()
@click.option("--sim-threshold", default=0.78, show_default=True,
              help="Cosine mínimo entre centroides para considerar un par")
@click.option("--min-hops", default=3, show_default=True,
              help="Distancia mínima en el grafo (≥3 = ni vecinos ni vecinos-de-vecinos)")
@click.option("--top", default=5, show_default=True,
              help="Máximo de puentes a proponer por corrida")
@click.option("--skip-young-days", default=7, show_default=True,
              help="Ignorar notas más nuevas que este umbral (siguen evolucionando)")
@click.option("--no-llm", is_flag=True,
              help="No generar 'por qué' (más rápido, útil para debugging)")
@click.option("--plain", is_flag=True, help="Salida plana sin colores")
def surface(sim_threshold: float, min_hops: int, top: int,
            skip_young_days: int, no_llm: bool, plain: bool):
    """Puentes no hechos: pares de notas cercanas en significado pero lejanas en el grafo.

    Corrida proactiva — ideal como cron nocturno antes del morning brief, o manual
    cuando querés densificar el grafo. Cada corrida loguea a surface.jsonl para
    después alimentar el ranker del feedback loop.
    """
    import time as _time
    t0 = _time.time()
    pairs = find_surface_bridges(
        col=get_db(),
        sim_threshold=sim_threshold,
        min_hops=min_hops,
        top=top,
        skip_young_days=skip_young_days,
    )
    duration_ms = int((_time.time() - t0) * 1000)

    if not pairs:
        msg = f"Sin puentes con cosine ≥ {sim_threshold} y dist ≥ {min_hops}. Bajá --sim-threshold si querés explorar."
        click.echo(msg) if plain else console.print(f"[yellow]{msg}[/yellow]")
        _surface_log_run({
            "n_pairs": 0, "sim_threshold": sim_threshold, "min_hops": min_hops,
            "top": top, "skip_young_days": skip_young_days, "llm": not no_llm,
            "duration_ms": duration_ms,
        }, [])
        return

    if not no_llm:
        for p in pairs:
            p["reason"] = _surface_generate_reason(p)
    duration_ms = int((_time.time() - t0) * 1000)

    _surface_log_run({
        "n_pairs": len(pairs), "sim_threshold": sim_threshold,
        "min_hops": min_hops, "top": top, "skip_young_days": skip_young_days,
        "llm": not no_llm, "duration_ms": duration_ms,
    }, pairs)

    if plain:
        for i, p in enumerate(pairs, 1):
            click.echo(f"{i}. {p['similarity']:.2f}  {p['a_path']}  ↔  {p['b_path']}")
            if p.get("reason"):
                click.echo(f"   {p['reason']}")
        return

    console.print()
    console.print(Rule(
        title=f"[bold magenta]🔗 {len(pairs)} puente(s) sugerido(s) (cosine ≥ {sim_threshold}, hops ≥ {min_hops})[/bold magenta]",
        style="magenta",
    ))
    for i, p in enumerate(pairs, 1):
        console.print()
        console.print(
            f"[bold]{i}.[/bold] [bold]{p['similarity']:.3f}[/bold]  "
            f"[magenta]{p['a_note']}[/magenta] [dim]({p['a_path']})[/dim]"
        )
        console.print(
            f"     ↔  [magenta]{p['b_note']}[/magenta] [dim]({p['b_path']})[/dim]"
        )
        if p.get("reason"):
            console.print(f"     [cyan]→[/cyan] [italic]{p['reason']}[/italic]")
    console.print()
    console.print(f"[dim]{duration_ms}ms · log: {SURFACE_LOG_PATH}[/dim]")


def _filing_source_label(pr: dict) -> str:
    """Etiqueta corta de la procedencia de la propuesta. Vacía si baseline puro
    (no agregamos ruido visual cuando no hay personalización)."""
    src = pr.get("source", "baseline")
    ev = pr.get("evidence", []) or []
    if src == "agreed":
        return f"agreed · {len(ev)} similar(es) en este folder"
    if src == "personalized":
        return f"personalized · {len(ev)} similar(es) confirman"
    if src == "baseline+history":
        # Hay history pero no convence — útil saberlo (el usuario puede edit).
        return f"baseline · history apunta a otros folders"
    return ""


def _render_filing_proposal(i: int, total: int, pr: dict, plain: bool) -> None:
    """Renderer compartido por dry-run y apply. Muestra una propuesta; no pide
    input ni decide — solo layout."""
    if plain:
        if "error" in pr:
            click.echo(f"[{i}/{total}] {pr['path']}  [error: {pr['error']}]")
            return
        conf = pr["confidence"]
        tag = ("firm" if conf >= FILING_CONFIDENCE_FIRM
               else "tentative" if conf >= FILING_CONFIDENCE_TENTATIVE
               else "low")
        click.echo(f"[{i}/{total}] {pr['note']}  ({pr['path']})")
        target = pr["folder"] or "(sin propuesta)"
        line = f"         -> {target}  (conf {conf:.2f}, {tag})"
        src_label = _filing_source_label(pr)
        if src_label:
            line += f"  [{src_label}]"
        click.echo(line)
        if pr["upward_title"]:
            click.echo(f"         ^ [[{pr['upward_title']}]] ({pr['upward_kind']})")
        return
    if "error" in pr:
        console.print(
            f"\n[bold]{i}/{total}[/bold] [red]{pr['path']}[/red] "
            f"[dim](error: {pr['error']})[/dim]"
        )
        return
    console.print(
        f"\n[bold]{i}/{total}[/bold] [bold]{pr['note']}[/bold] "
        f"[dim]({pr['path']})[/dim]"
    )
    conf = pr["confidence"]
    if conf >= FILING_CONFIDENCE_FIRM:
        style, label = "green", f"firm · {conf:.2f}"
    elif conf >= FILING_CONFIDENCE_TENTATIVE:
        style, label = "yellow", f"tentative · {conf:.2f}"
    else:
        style, label = "red", f"low · {conf:.2f}"
    target = pr["folder"] or "(sin propuesta)"
    src_label = _filing_source_label(pr)
    src_style = {
        "agreed · ": "green dim",
        "personalized · ": "magenta dim",
        "baseline · ": "yellow dim",
    }
    src_color = "dim"
    for prefix, color in src_style.items():
        if src_label.startswith(prefix.rstrip()):
            src_color = color
            break
    line = (
        f"   [cyan]→[/cyan] [bold]{target}[/bold] "
        f"[dim]·[/dim] [{style}]{label}[/{style}]"
    )
    if src_label:
        line += f"  [{src_color}]· {src_label}[/{src_color}]"
    console.print(line)
    if pr["upward_title"]:
        badge = "📍 MOC" if pr["upward_kind"] == "moc" else "↔ nearest"
        console.print(
            f"   [cyan]↑[/cyan] [[[magenta]{pr['upward_title']}[/magenta]]] "
            f"[dim]{badge}[/dim]"
        )


@cli.command(name="file")
@click.argument("path", required=False)
@click.option("--folder", default="00-Inbox", show_default=True,
              help="Folder a escanear si no pasás path")
@click.option("--one", is_flag=True, help="Solo la nota más vieja del inbox")
@click.option("--limit", default=20, show_default=True,
              help="Máximo de notas a procesar por corrida")
@click.option("-k", "k", default=8, show_default=True,
              help="Cantidad de vecinos semánticos a considerar")
@click.option("--apply", "do_apply", is_flag=True,
              help="Modo interactivo: mover + upward-link + reindex con confirmación y/n/e/s/q")
@click.option("--undo", "do_undo", is_flag=True,
              help="Revertir el último batch de --apply (lee filing_batches/)")
@click.option("--plain", is_flag=True, help="Salida plana sin colores")
def file_cmd(path: str | None, folder: str, one: bool, limit: int,
             k: int, do_apply: bool, do_undo: bool, plain: bool):
    """Filing asistido de Inbox: destino PARA + upward-link sugeridos.

    Sin flags: dry-run (propone y loguea, no mueve nada).
    --apply:   interactivo — y/n/e(edit)/s(skip)/q(quit) por nota.
    --undo:    revierte el último batch aplicado.

    Contrato de seguridad: --apply mueve archivos solo tras tu 'y'. Cada
    move se persiste en filing_batches/<ts>.jsonl para que --undo sepa
    exactamente qué revertir. Edit permite cambiar el folder destino sin
    volver a sugerir desde cero.
    """
    col = get_db()

    if do_apply and do_undo:
        msg = "No podés combinar --apply y --undo."
        click.echo(msg) if plain else console.print(f"[red]{msg}[/red]")
        return

    # ── UNDO path ────────────────────────────────────────────────────────────
    if do_undo:
        batch = _last_filing_batch()
        if batch is None:
            msg = "No hay batches previos para revertir."
            click.echo(msg) if plain else console.print(f"[yellow]{msg}[/yellow]")
            return
        results = _rollback_filing_batch(col, batch)
        ok = sum(1 for r in results if r.get("ok"))
        fail = len(results) - ok
        # Renombrar el batch a .undone para trazabilidad sin borrar.
        try:
            batch.rename(batch.with_suffix(".undone"))
        except Exception:
            pass
        if plain:
            click.echo(f"undo: {ok} movido(s) de vuelta, {fail} error(es). batch: {batch.name}")
            for r in results:
                status = "✓" if r.get("ok") else f"✗ {r.get('error', '')}"
                click.echo(f"  {status}  {r['dst']} -> {r['src']}")
            return
        console.print()
        console.print(Rule(
            title=f"[bold magenta]↩ undo: {ok} movido(s) de vuelta, {fail} error(es)[/bold magenta]",
            style="magenta",
        ))
        for r in results:
            if r.get("ok"):
                console.print(f"  [green]✓[/green] [dim]{r['dst']}[/dim] → [cyan]{r['src']}[/cyan]")
            else:
                console.print(f"  [red]✗[/red] {r['dst']}  [dim]({r.get('error','?')})[/dim]")
        console.print(f"\n[dim]batch renombrado a {batch.with_suffix('.undone').name}[/dim]")
        return

    # ── Collect target notes ─────────────────────────────────────────────────
    if path:
        full = Path(path) if path.startswith("/") else VAULT_PATH / path
        notes = [full] if full.is_file() else []
        if not notes:
            msg = f"No existe: {full}"
            click.echo(msg) if plain else console.print(f"[red]{msg}[/red]")
            return
    else:
        inbox_dir = VAULT_PATH / folder
        if not inbox_dir.is_dir():
            msg = f"No existe el folder: {inbox_dir}"
            click.echo(msg) if plain else console.print(f"[red]{msg}[/red]")
            return
        notes = sorted(inbox_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
        notes = notes[:1] if one else notes[:limit]

    if not notes:
        msg = f"Sin notas en {folder}."
        click.echo(msg) if plain else console.print(f"[yellow]{msg}[/yellow]")
        return

    # Frontmatter opt-out: file: skip respeta la elección del usuario.
    eligible: list[Path] = []
    skipped_by_fm: list[str] = []
    for p in notes:
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
            fm = parse_frontmatter(raw)
            if fm.get("file") == "skip":
                skipped_by_fm.append(str(p.relative_to(VAULT_PATH)))
                continue
        except Exception:
            pass
        eligible.append(p)

    if skipped_by_fm and not plain:
        console.print(
            f"[dim]Skip por frontmatter `file: skip`: {len(skipped_by_fm)}[/dim]"
        )

    if not eligible:
        msg = "Todas las notas tienen `file: skip` en frontmatter."
        click.echo(msg) if plain else console.print(f"[yellow]{msg}[/yellow]")
        return

    # ── Build proposals ──────────────────────────────────────────────────────
    # Cargamos history una vez por batch — evita releer filing.jsonl N veces.
    # Si está bajo el umbral, igual se pasa (build_filing_proposal lo ignora).
    history = _load_filing_decisions()
    if not plain and len(history) >= FILING_PERSONALIZE_MIN_HISTORY:
        console.print(
            f"[dim]Personalización activa · {len(history)} decisión(es) en history[/dim]"
        )

    proposals: list[tuple[Path, dict]] = []
    for p in eligible:
        rel = str(p.relative_to(VAULT_PATH))
        prop = build_filing_proposal(col, rel, k=k, history=history)
        proposals.append((p, prop))

    # ── Dry-run path: print + log, bail ──────────────────────────────────────
    if not do_apply:
        if not plain:
            console.print()
            console.print(Rule(
                title=f"[bold magenta]📥 {len(proposals)} propuesta(s) de filing[/bold magenta]",
                style="magenta",
            ))
        for i, (_, pr) in enumerate(proposals, 1):
            _render_filing_proposal(i, len(proposals), pr, plain)
            _filing_log_proposal(pr)
        if plain:
            click.echo(f"\nlog: {FILING_LOG_PATH}")
        else:
            console.print(
                f"\n[dim]Dry-run — nada se movió. Log: {FILING_LOG_PATH}[/dim]\n"
                f"[dim]Correr con --apply para mover interactivo.[/dim]"
            )
        return

    # ── Apply path: interactive loop ─────────────────────────────────────────
    if not plain:
        console.print()
        console.print(Rule(
            title=f"[bold magenta]📥 Apply: {len(proposals)} nota(s) · y/n/e/s/q[/bold magenta]",
            style="magenta",
        ))

    batch_entries: list[dict] = []
    for i, (src_path, pr) in enumerate(proposals, 1):
        _render_filing_proposal(i, len(proposals), pr, plain)
        if "error" in pr:
            _filing_log_proposal(pr, decision="error")
            continue

        target = pr["folder"]
        if not target:
            # Sin propuesta de folder → solo edit o skip.
            choice = click.prompt("   [e]dit target / [s]kip / [q]uit",
                                  default="s", show_default=False).strip().lower()
        else:
            choice = click.prompt(
                "   [y]es · [n]o · [e]dit · [s]kip · [q]uit",
                default="y", show_default=False,
            ).strip().lower()

        if choice == "q":
            _filing_log_proposal(pr, decision="quit")
            click.echo("Quit — guardando batch con lo aplicado hasta ahora.") if plain else console.print("[dim]Quit — guardando batch con lo aplicado hasta ahora.[/dim]")
            break
        if choice in ("s", "skip"):
            _filing_log_proposal(pr, decision="skip")
            continue
        if choice in ("n", "no"):
            _filing_log_proposal(pr, decision="reject")
            continue
        if choice in ("e", "edit"):
            new_target = click.prompt("   Nuevo folder (vault-relative, ej 02-Areas/Salud)").strip()
            if not new_target:
                _filing_log_proposal(pr, decision="skip")
                continue
            target = new_target
            decision = "edit"
        else:
            # 'y' o default
            if not target:
                _filing_log_proposal(pr, decision="skip")
                continue
            decision = "accept"

        # Ejecutar move + upward-link + reindex.
        try:
            entry = _apply_filing_move(col, pr["path"], target, pr.get("upward_title") or "")
        except FileExistsError as ex:
            msg = f"   [destino ya existe: {ex}] skip"
            click.echo(msg) if plain else console.print(f"[red]{msg}[/red]")
            _filing_log_proposal(pr, decision="error")
            continue
        except Exception as ex:
            msg = f"   [error: {ex}] skip"
            click.echo(msg) if plain else console.print(f"[red]{msg}[/red]")
            _filing_log_proposal(pr, decision="error")
            continue

        batch_entries.append(entry)
        # Log con decisión + target efectivo (puede diferir de pr["folder"]).
        logged = dict(pr)
        logged["applied_to"] = entry["dst"]
        _filing_log_proposal(logged, decision=decision)
        if plain:
            click.echo(f"   ✓ {entry['src']} -> {entry['dst']}")
        else:
            console.print(f"   [green]✓[/green] movido a [cyan]{entry['dst']}[/cyan]")

    batch_path = _write_filing_batch(batch_entries)
    if plain:
        click.echo(f"\n{len(batch_entries)} aplicada(s).")
        if batch_path:
            click.echo(f"batch: {batch_path}  (rag file --undo para revertir)")
    else:
        console.print()
        if batch_entries:
            console.print(
                f"[green]✓ {len(batch_entries)} aplicada(s).[/green] "
                f"[dim]batch: {batch_path.name if batch_path else '(ninguno)'}[/dim]"
            )
            console.print("[dim]`rag file --undo` revierte este batch.[/dim]")
        else:
            console.print("[yellow]Nada aplicado.[/yellow]")


@cli.command()
@click.option("--folder", default="00-Inbox", show_default=True,
              help="Folder de Inbox a triar")
@click.option("--apply", is_flag=True,
              help="Mover + taggear + linkificar (default = dry-run)")
@click.option("--max-tags", default=5, show_default=True)
@click.option("--limit", default=20, show_default=True,
              help="Máximo de notas a procesar por corrida")
@click.option("--folder-min-conf", default=0.4, show_default=True,
              help="Confianza mínima para mover (--apply)")
@click.option("--no-folder", is_flag=True, help="No sugerir destino")
@click.option("--no-tags", is_flag=True, help="No sugerir tags")
@click.option("--no-wikilinks", is_flag=True, help="No sugerir wikilinks")
def inbox(folder: str, apply: bool, max_tags: int, limit: int,
          folder_min_conf: float, no_folder: bool, no_tags: bool,
          no_wikilinks: bool):
    """Triar notas de Inbox: sugerir destino + tags + wikilinks + duplicados.

    Compone retrieve (vecino semántico para folder), autotag, find_wikilink_
    suggestions y find_near_duplicates_for. Con --apply mueve cada nota a su
    destino sugerido (si la confianza supera --folder-min-conf), agrega los
    tags nuevos al frontmatter, aplica los wikilinks y re-indexa.
    """
    inbox_dir = VAULT_PATH / folder
    if not inbox_dir.is_dir():
        console.print(f"[red]No existe el folder:[/red] {inbox_dir}")
        return
    notes = sorted(inbox_dir.glob("*.md"))[:limit]
    if not notes:
        console.print(f"[yellow]Sin notas en {folder}.[/yellow]")
        return

    col = get_db()
    moved = 0
    tagged = 0
    linkified = 0
    flagged_dupes = 0
    for path in notes:
        rel = str(path.relative_to(VAULT_PATH))
        try:
            t = triage_inbox_note(col, rel, max_tags=max_tags)
        except Exception as e:
            console.print(f"[red]Error en {rel}: {e}[/red]")
            continue

        console.print()
        console.print(Rule(style="dim", characters="╌"))
        console.print(f"[bold cyan]{rel}[/bold cyan]")

        if not no_folder and t.get("folder_suggested"):
            conf = t["folder_confidence"]
            color = "green" if conf >= folder_min_conf else "yellow"
            console.print(
                f"  📁 destino: [{color}]{t['folder_suggested']}[/{color}] "
                f"[dim]({conf:.2f} conf)[/dim]"
            )
        if not no_tags and t.get("tags_new"):
            console.print(
                f"  🏷  tags nuevos: [yellow]{', '.join(t['tags_new'])}[/yellow] "
                f"[dim](actuales: {', '.join(t['tags_current']) or '—'})[/dim]"
            )
        if not no_wikilinks and t.get("wikilinks"):
            wl_titles = [w["title"] for w in t["wikilinks"][:6]]
            extra = len(t["wikilinks"]) - len(wl_titles)
            extras_str = f" [dim]+{extra} más[/dim]" if extra > 0 else ""
            console.print(
                f"  🔗 wikilinks: [magenta]{', '.join(wl_titles)}[/magenta]{extras_str}"
            )
        for d in (t.get("duplicates") or []):
            flagged_dupes += 1
            console.print(
                f"  ⚠  posible duplicado: [red]{d['note']}[/red] "
                f"[dim]({d['path']}, sim {d['similarity']})[/dim]"
            )

        if not apply:
            continue

        # Apply: move (only if confidence ≥ min), tag, linkify, reindex.
        target_path = path
        if not no_folder and t.get("folder_suggested") and t["folder_confidence"] >= folder_min_conf:
            target_dir = VAULT_PATH / t["folder_suggested"]
            target_dir.mkdir(parents=True, exist_ok=True)
            candidate = target_dir / path.name
            if candidate.exists():
                console.print(f"  [red]✗ destino ya existe, no muevo[/red]")
            else:
                import shutil
                shutil.move(str(path), str(candidate))
                # Drop chunks of the old path; the new path will be added below.
                _index_single_file(col, path, skip_contradict=True)
                target_path = candidate
                moved += 1
                console.print(
                    f"  [green]✓ movido a {target_path.relative_to(VAULT_PATH)}[/green]"
                )

        if not no_tags and t.get("tags_new"):
            merged = list(dict.fromkeys([*t["tags_current"], *t["tags_new"]]))[:max_tags]
            if _apply_frontmatter_tags(target_path, merged):
                tagged += 1
                console.print("  [green]✓ tags actualizados[/green]")

        if not no_wikilinks:
            # Recompute against the (possibly new) path; the file content may
            # have shifted offsets after the tag-write above.
            new_rel = str(target_path.relative_to(VAULT_PATH))
            fresh = find_wikilink_suggestions(col, new_rel, max_per_note=10)
            if fresh:
                n = apply_wikilink_suggestions(new_rel, fresh)
                if n:
                    linkified += 1
                    console.print(f"  [green]✓ {n} wikilinks aplicados[/green]")

        # Final reindex picks up moved path + tag/wikilink edits in one pass.
        try:
            _index_single_file(col, target_path, skip_contradict=True)
        except Exception:
            pass

    console.print()
    if apply:
        console.print(
            f"[green]✓ Triaje aplicado:[/green] "
            f"{moved} movidas · {tagged} tags actualizados · "
            f"{linkified} con wikilinks · {flagged_dupes} dupes flaggeados"
        )
    else:
        console.print(
            f"[cyan]Plan generado para {len(notes)} nota(s).[/cyan] "
            f"[dim]Re-correr con --apply para ejecutar.[/dim]"
        )


def _slug(text: str, maxlen: int = 50) -> str:
    """Slugify for filenames: lowercase, alphanum + hyphens, capped."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:maxlen] or "topic"


@cli.command()
@click.argument("topic")
@click.option("-k", default=8, show_default=True,
              help="Chunks de contexto principal a recuperar")
@click.option("--folder", default=None,
              help="Limitar el contexto a este folder (e.g., '02-Areas/Coaching')")
@click.option("--save", is_flag=True,
              help="Guardar el brief en 00-Inbox/ y auto-indexar")
@click.option("--no-urls", is_flag=True, help="Saltear URLs")
@click.option("--no-related", is_flag=True, help="Saltear notas relacionadas (graph)")
@click.option("--plain", is_flag=True,
              help="Salida plana sin colores/streaming overlay")
def prep(topic: str, k: int, folder: str | None, save: bool,
         no_urls: bool, no_related: bool, plain: bool):
    """Preparar un brief de contexto sobre una persona / proyecto / tema.

    Compone retrieve + find_related + find_urls y le pide a command-r que
    arme un brief estructurado en 1ra persona (background, threads abiertos,
    preguntas para explorar, URLs). Sin --save imprime al terminal; con
    --save escribe `00-Inbox/YYYY-MM-DD-prep-<slug>.md` y lo indexa.
    """
    col = get_db()
    if col.count() == 0:
        msg = "Índice vacío. Ejecuta: rag index"
        click.echo(msg) if plain else console.print(f"[red]{msg}[/red]")
        return

    # 1. Main retrieval — direct topic search.
    main = retrieve(
        col, topic, k=k, folder=folder, tag=None,
        precise=False, multi_query=True, auto_filter=True,
    )
    sources_main = main["metas"]
    if not sources_main:
        msg = f"Sin contexto en el vault para '{topic}'."
        click.echo(msg) if plain else console.print(f"[yellow]{msg}[/yellow]")
        return

    # 2. Graph neighbours of the top sources.
    related = [] if no_related else find_related(col, sources_main, limit=6)

    # 3. URLs scoped to topic — surface bookmarks/refs the brief should mention.
    urls = [] if no_urls else find_urls(topic, k=5, folder=folder)

    # 4. Build the context block. Cap chunk length so the brief stays focused.
    chunks_text = "\n\n---\n\n".join(
        f"[{i+1}] [[{m.get('note','')}]] (ruta: {m.get('file','')})\n{d[:600]}"
        for i, (d, m) in enumerate(zip(main["docs"], sources_main))
    )
    related_text = (
        "\n".join(
            f"- [[{m.get('note','')}]] ({m.get('file','')})  ×{score} {reason}"
            for m, score, reason in related[:6]
        )
        or "(sin relacionadas)"
    )
    urls_text = (
        "\n".join(
            f"- {u['url']}  ({u.get('anchor') or u.get('note','')})"
            for u in urls[:5]
        )
        or "(sin URLs en el vault para esto)"
    )

    rules = (
        "Sos un asistente que arma briefs de contexto a partir de las notas "
        "personales del usuario, en 1ra persona. NO inventes información que "
        "no esté en el contexto. Citá las notas con [[Título]] cuando hagas "
        "afirmaciones puntuales. Si no hay info para una sección, escribí "
        "honestamente que no hay nada en el vault."
    )
    prompt = (
        f"{rules}\n\n"
        f"TEMA / PERSONA / PROYECTO: {topic}\n\n"
        f"NOTAS RELEVANTES:\n{chunks_text}\n\n"
        f"NOTAS RELACIONADAS (graph):\n{related_text}\n\n"
        f"URLS RELEVANTES:\n{urls_text}\n\n"
        "Generá el brief en Markdown con esta estructura exacta:\n\n"
        f"# Prep: {topic}\n\n"
        "## Resumen ejecutivo\n"
        "(2-3 líneas, lo más importante para entrar en contexto YA)\n\n"
        "## Background\n"
        "(lo que ya sé sobre esto desde mis notas — un párrafo sólido)\n\n"
        "## Threads abiertos\n"
        "(decisiones/acciones pendientes que detectes en las notas; bullets)\n\n"
        "## Preguntas para explorar\n"
        "(qué preguntas valdría hacer/considerar en una conversación sobre "
        "esto; bullets, máximo 5)\n\n"
        "## URLs y fuentes\n"
        "(URLs relevantes literales + lista de notas citadas como [[wikilinks]])\n\n"
        "Mantenete entre 350 y 550 palabras."
    )

    t_start = time.perf_counter()
    parts: list[str] = []
    if plain:
        for chunk in ollama.chat(
            model=resolve_chat_model(),
            messages=[{"role": "user", "content": prompt}],
            options=CHAT_OPTIONS,
            stream=True,
            keep_alive=OLLAMA_KEEP_ALIVE,
        ):
            tok = chunk.message.content
            parts.append(tok)
            click.echo(tok, nl=False)
        click.echo("")
    else:
        from rich.live import Live
        with console.status("[dim]armando brief…[/dim]", spinner="dots"):
            # Drain the stream silently first, then redraw with link styling.
            for chunk in ollama.chat(
                model=resolve_chat_model(),
                messages=[{"role": "user", "content": prompt}],
                options=CHAT_OPTIONS,
                stream=True,
                keep_alive=OLLAMA_KEEP_ALIVE,
            ):
                parts.append(chunk.message.content)
        full = "".join(parts)
        console.print()
        console.print(render_response(full))

    full = "".join(parts)
    t_gen = time.perf_counter() - t_start

    log_query_event({
        "cmd": "prep", "topic": topic, "folder": folder,
        "n_sources": len(sources_main), "n_related": len(related),
        "n_urls": len(urls), "answer_len": len(full),
        "t_gen": round(t_gen, 2),
    })

    if save:
        date = datetime.now().strftime("%Y-%m-%d")
        slug = _slug(topic)
        target_dir = VAULT_PATH / "00-Inbox"
        target_dir.mkdir(parents=True, exist_ok=True)
        candidate = target_dir / f"{date}-prep-{slug}.md"
        i = 2
        while candidate.exists():
            candidate = target_dir / f"{date}-prep-{slug}-{i}.md"
            i += 1
        source_links = ", ".join(
            f"'[[{m.get('note','')}]]'" for m in sources_main
        )
        fm = (
            "---\n"
            f"created: {datetime.now().isoformat(timespec='seconds')}\n"
            "type: prep\n"
            f"topic: \"{topic}\"\n"
            "tags: [prep]\n"
            f"sources: [{source_links}]\n"
            "---\n\n"
        )
        candidate.write_text(fm + full + "\n", encoding="utf-8")
        try:
            _index_single_file(col, candidate, skip_contradict=True)
        except Exception:
            pass
        rel = candidate.relative_to(VAULT_PATH)
        click.echo(f"guardado: {rel}") if plain else console.print(
            f"\n[green]✓ Guardado:[/green] [bold cyan]{rel}[/bold cyan]"
        )


@cli.group()
def wikilinks():
    """Densificar el grafo: sugerir [[wikilinks]] faltantes."""


@wikilinks.command("suggest")
@click.option("--note", "note_path", default=None,
              help="Solo esta nota (vault-relative)")
@click.option("--folder", default=None, help="Solo notas bajo este folder")
@click.option("--apply", is_flag=True,
              help="Aplicar las sugerencias (default = dry-run)")
@click.option("--max-per-note", default=30, show_default=True,
              help="Cap de sugerencias por nota")
@click.option("--min-len", default=4, show_default=True,
              help="Largo mínimo del título a considerar (filtro de colisiones)")
@click.option("--show", default=5, show_default=True,
              help="Sugerencias por nota a imprimir en dry-run")
def wikilinks_suggest(
    note_path: str | None, folder: str | None, apply: bool,
    max_per_note: int, min_len: int, show: int,
):
    """Encontrar menciones a títulos de notas que NO están wikilinkeadas y
    proponerlas. Skipea código, frontmatter, links existentes, y títulos
    ambiguos (mismo string en varias notas).
    """
    col = get_db()
    if note_path:
        paths = [note_path]
    else:
        c = _load_corpus(col)
        paths = sorted({m.get("file", "") for m in c["metas"] if m.get("file")})
        if folder:
            paths = [p for p in paths if p.startswith(folder.rstrip("/") + "/") or p == folder]

    if not paths:
        console.print("[yellow]No hay notas que procesar.[/yellow]")
        return

    total_suggestions = 0
    notes_with_suggestions = 0
    notes_applied = 0
    by_note: list[tuple[str, list[dict]]] = []
    for path in track(paths, description="Analizando..."):
        try:
            sugs = find_wikilink_suggestions(
                col, path, min_title_len=min_len, max_per_note=max_per_note,
            )
        except Exception:
            continue
        if not sugs:
            continue
        notes_with_suggestions += 1
        total_suggestions += len(sugs)
        by_note.append((path, sugs))

        if apply:
            try:
                n_applied = apply_wikilink_suggestions(path, sugs)
                if n_applied:
                    notes_applied += 1
                    # Re-index the rewritten note so retrieval picks up the new
                    # outlinks. Skip contradiction check — wikilinks aren't a
                    # claim change, just graph noise; saves an LLM call.
                    try:
                        _index_single_file(
                            col, VAULT_PATH / path, skip_contradict=True,
                        )
                    except Exception:
                        pass
            except Exception as e:
                console.print(f"[red]Error aplicando en {path}: {e}[/red]")

    if not apply:
        for path, sugs in by_note:
            console.print(
                f"\n[bold cyan]{path}[/bold cyan] [dim]({len(sugs)} sugerencia(s))[/dim]"
            )
            for s in sugs[:show]:
                console.print(
                    f"  [yellow]{s['title']}[/yellow] → [magenta]{s['target']}[/magenta]"
                    f" [dim]línea {s['line']}[/dim]"
                )
                ctx = s["context"]
                if len(ctx) > 140:
                    ctx = ctx[:140] + "…"
                console.print(f"    [dim]{ctx}[/dim]")
            if len(sugs) > show:
                console.print(f"    [dim]+ {len(sugs) - show} más…[/dim]")
        console.print(
            f"\n[cyan]{total_suggestions} sugerencia(s) en "
            f"{notes_with_suggestions}/{len(paths)} nota(s).[/cyan] "
            f"[dim]Re-correr con --apply para escribir.[/dim]"
        )
    else:
        console.print(
            f"\n[green]✓ Aplicado:[/green] {total_suggestions} wikilinks en "
            f"{notes_applied}/{len(paths)} notas."
        )


# ── QUICK CAPTURE (rag capture) ──────────────────────────────────────────────
# Atomic building block: take a string (from stdin or arg) and land it in the
# vault as a fresh Inbox note. Used directly by the user and indirectly by the
# Telegram voice-to-note path.

_CAPTURE_FOLDER = "00-Inbox"


def _capture_slug(text: str, maxlen: int = 40) -> str:
    """First non-empty line, slugged."""
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "capture")
    return _slug(first, maxlen=maxlen)


def capture_note(
    text: str,
    tags: list[str] | None = None,
    source: str | None = None,
    title: str | None = None,
) -> Path:
    """Create `<vault>/00-Inbox/YYYY-MM-DD-HHMM-<slug>.md` with the given text
    and minimal frontmatter. Returns the written path.

    Idempotent on name collision: appends a ``-2``, ``-3`` suffix.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("capture text is empty")
    now = datetime.now()
    stamp = now.strftime("%Y-%m-%d-%H%M")
    slug = _slug(title, maxlen=40) if title else _capture_slug(text)
    target_dir = VAULT_PATH / _CAPTURE_FOLDER
    target_dir.mkdir(parents=True, exist_ok=True)
    candidate = target_dir / f"{stamp}-{slug}.md"
    i = 2
    while candidate.exists():
        candidate = target_dir / f"{stamp}-{slug}-{i}.md"
        i += 1
    fm_lines = [
        "---",
        f"created: '{now.isoformat(timespec='seconds')}'",
        "type: capture",
    ]
    all_tags = list(dict.fromkeys(["capture", *(tags or [])]))
    fm_lines.append("tags:")
    for t in all_tags:
        fm_lines.append(f"- {t}")
    if source:
        fm_lines.append(f"source: {source}")
    fm_lines.append("---")
    body = "\n".join(fm_lines) + f"\n\n{text.strip()}\n"
    candidate.write_text(body, encoding="utf-8")
    return candidate


@cli.command()
@click.argument("text", required=False)
@click.option("--stdin", "from_stdin", is_flag=True,
              help="Leer texto de stdin (ignora arg posicional)")
@click.option("--tag", "tags", multiple=True,
              help="Tag extra (además de 'capture'). Repetible.")
@click.option("--source", default=None,
              help="Etiqueta de origen (ej: 'voice', 'telegram', 'cli')")
@click.option("--title", default=None,
              help="Título custom (slug del filename). Default: primera línea.")
@click.option("--plain", is_flag=True, help="Salida plana (ruta only)")
def capture(text: str | None, from_stdin: bool, tags: tuple[str, ...],
            source: str | None, title: str | None, plain: bool):
    """Capturar una nota rápida al 00-Inbox/ del vault.

    Uso:
      rag capture "idea suelta"
      echo "nota" | rag capture --stdin --tag voice --source telegram
    """
    if from_stdin:
        import sys
        text = sys.stdin.read()
    if not (text and text.strip()):
        msg = "Sin texto. Pasá arg o --stdin."
        click.echo(msg) if plain else console.print(f"[red]{msg}[/red]")
        return
    try:
        path = capture_note(
            text, tags=list(tags), source=source, title=title,
        )
    except ValueError as e:
        click.echo(str(e)) if plain else console.print(f"[red]{e}[/red]")
        return
    # Auto-index so the captured note is retrievable inmediatly — `rag watch`
    # would pick it up too, but we don't want to wait for its debounce.
    try:
        _index_single_file(get_db(), path, skip_contradict=True)
    except Exception:
        pass
    rel = path.relative_to(VAULT_PATH)
    if plain:
        click.echo(str(rel))
    else:
        console.print(f"[green]✓ Capturado:[/green] [bold cyan]{rel}[/bold cyan]")


# ── MORNING BRIEF (rag morning) ──────────────────────────────────────────────
# Proactive daily brief: what happened yesterday + what's pending + what to
# focus today. Composes recent-notes / inbox / todo-frontmatter / new
# contradictions / low-conf queries. Writes to 05-Reviews/YYYY-MM-DD.md.
# Auto-fires via launchd weekday mornings.

MORNING_FOLDER = "05-Reviews"


def _collect_morning_evidence(
    now: datetime,
    vault: Path,
    query_log: Path,
    contradiction_log: Path,
    lookback_hours: int = 36,
) -> dict:
    """Gather yesterday-ish signals. 36h lookback covers the gap between
    morning runs on different days (skipping a day doesn't lose yesterday).
    """
    from datetime import timedelta as _td
    start = now - _td(hours=lookback_hours)

    recent: list[dict] = []
    inbox: list[dict] = []
    todos: list[dict] = []
    if vault.is_dir():
        for p in vault.rglob("*.md"):
            try:
                rel = str(p.relative_to(vault))
            except ValueError:
                continue
            if is_excluded(rel):
                continue
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime)
            except OSError:
                continue
            try:
                raw = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            fm = parse_frontmatter(raw)
            title = p.stem
            if rel.startswith(f"{_CAPTURE_FOLDER}/"):
                inbox.append({
                    "path": rel, "title": title,
                    "modified": mtime.isoformat(timespec="seconds"),
                    "snippet": clean_md(raw)[:160].strip(),
                })
            if start <= mtime < now and not rel.startswith(f"{MORNING_FOLDER}/"):
                recent.append({
                    "path": rel, "title": title,
                    "modified": mtime.isoformat(timespec="seconds"),
                    "snippet": clean_md(raw)[:220].strip(),
                })
            t = fm.get("todo")
            d = fm.get("due")
            if t or d:
                todos.append({
                    "path": rel, "title": title,
                    "todo": t if t else None,
                    "due": str(d) if d else None,
                })

    new_contrad: list[dict] = []
    if contradiction_log.is_file():
        try:
            lines = contradiction_log.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            try:
                ts = datetime.fromisoformat(e.get("ts", ""))
            except Exception:
                continue
            if ts < start or ts >= now:
                continue
            if e.get("cmd") != "contradict_index":
                continue
            entries = e.get("contradicts") or []
            if not entries:
                continue
            new_contrad.append({
                "subject_path": e.get("subject_path", ""),
                "targets": [
                    {"path": c.get("path", ""), "why": c.get("why", "")}
                    for c in entries if isinstance(c, dict)
                ],
            })

    low_conf: list[dict] = []
    if query_log.is_file():
        try:
            lines = query_log.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            try:
                ts = datetime.fromisoformat(e.get("ts", ""))
            except Exception:
                continue
            if ts < start or ts >= now:
                continue
            if e.get("cmd") != "query":
                continue
            score = e.get("top_score")
            if isinstance(score, (int, float)) and score <= CONFIDENCE_RERANK_MIN:
                q = (e.get("q") or "").strip()
                if q:
                    low_conf.append({"q": q, "top_score": float(score)})

    recent.sort(key=lambda r: r["modified"], reverse=True)
    inbox.sort(key=lambda r: r["modified"], reverse=True)

    return {
        "recent_notes": recent,
        "inbox_pending": inbox,
        "todos": todos,
        "new_contradictions": new_contrad,
        "low_conf_queries": low_conf,
    }


def _render_morning_prompt(date_label: str, ev: dict) -> str:
    parts = [
        f"Generás un morning brief para el {date_label}, en 1ra persona, "
        "en español rioplatense, tono calmo y directo.",
        "",
        "Contexto real del vault (NO inventes lo que no esté acá):",
        "",
    ]
    if ev["recent_notes"]:
        parts.append(f"## Notas modificadas ayer ({len(ev['recent_notes'])}):")
        for r in ev["recent_notes"][:12]:
            parts.append(f"- [[{r['title']}]] ({r['path']}): {r['snippet'][:200]}")
        parts.append("")
    if ev["inbox_pending"]:
        parts.append(f"## En 00-Inbox/ ({len(ev['inbox_pending'])} sin triar):")
        for r in ev["inbox_pending"][:10]:
            parts.append(f"- [[{r['title']}]]: {r['snippet'][:160]}")
        parts.append("")
    if ev["todos"]:
        parts.append(f"## Notas con todo/due ({len(ev['todos'])}):")
        for r in ev["todos"][:10]:
            bits = []
            if r.get("due"):
                bits.append(f"due={r['due']}")
            if r.get("todo"):
                bits.append("todo=Y")
            parts.append(f"- [[{r['title']}]] ({r['path']}) {', '.join(bits)}")
        parts.append("")
    if ev["new_contradictions"]:
        parts.append(
            f"## Contradicciones nuevas detectadas ({len(ev['new_contradictions'])}):"
        )
        for c in ev["new_contradictions"][:5]:
            targets = ", ".join(t["path"] for t in c["targets"][:3])
            parts.append(f"- {c['subject_path']} ↔ {targets}")
        parts.append("")
    if ev["low_conf_queries"]:
        parts.append(f"## Queries sin respuesta buena ({len(ev['low_conf_queries'])}):")
        for q in ev["low_conf_queries"][:6]:
            parts.append(f"- \"{q['q']}\" (score {q['top_score']:+.2f})")
        parts.append("")
    parts.extend([
        "Formato de salida (Markdown, EXACTO):",
        "",
        "## 📬 Ayer en una línea",
        "(1 oración: qué pasó en el vault ayer)",
        "",
        "## 🎯 Foco sugerido para hoy",
        "(3 bullets con [[wikilink]] a la nota relevante si aplica; "
        "priorizar lo urgente/frágil; si no hay nada crítico, decilo honestamente)",
        "",
        "## 🗂 Pendientes que asoman",
        "(si hay inbox o todos, listarlos cortos; si no, omitir la sección)",
        "",
        "## ⚠ Atender",
        "(si hay contradicciones nuevas o gaps persistentes, nombrarlos; "
        "si no, omitir la sección)",
        "",
        "Entre 120 y 280 palabras total. Citá notas con [[Título]].",
    ])
    return "\n".join(parts)


def _generate_morning_narrative(prompt: str) -> str:
    try:
        resp = ollama.chat(
            model=resolve_chat_model(),
            messages=[{"role": "user", "content": prompt}],
            options=CHAT_OPTIONS,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        return (resp.message.content or "").strip()
    except Exception:
        return ""


@cli.command()
@click.option("--dry-run", is_flag=True,
              help="Imprimir el brief sin escribir el archivo")
@click.option("--date", "date_opt", default=None,
              help="Fecha objetivo YYYY-MM-DD (default: hoy)")
@click.option("--lookback-hours", default=36, show_default=True,
              help="Ventana de evidencia hacia atrás")
def morning(dry_run: bool, date_opt: str | None, lookback_hours: int):
    """Brief matutino: qué pasó ayer + qué enfocar hoy.

    Consume notas modificadas, 00-Inbox/ pending, todo/due frontmatter,
    contradicciones index-time del sidecar y queries low-confidence.
    command-r arma un brief de 120-280 palabras. Escribe a
    `05-Reviews/YYYY-MM-DD.md` (auto-indexado) salvo --dry-run.
    """
    if date_opt:
        try:
            target = datetime.fromisoformat(date_opt)
        except ValueError:
            console.print(f"[red]Fecha inválida: {date_opt}[/red]")
            return
    else:
        target = datetime.now()

    ev = _collect_morning_evidence(
        target, VAULT_PATH, LOG_PATH, CONTRADICTION_LOG_PATH,
        lookback_hours=lookback_hours,
    )
    total = (
        len(ev["recent_notes"]) + len(ev["inbox_pending"])
        + len(ev["todos"]) + len(ev["new_contradictions"])
        + len(ev["low_conf_queries"])
    )
    if total == 0:
        console.print(
            "[yellow]Mañana en blanco:[/yellow] sin notas modificadas, "
            "inbox vacío, 0 todos, 0 contradicciones nuevas."
        )
        return

    console.print(
        f"[dim]Evidencia:[/dim] {len(ev['recent_notes'])} recientes · "
        f"{len(ev['inbox_pending'])} inbox · {len(ev['todos'])} todos · "
        f"{len(ev['new_contradictions'])} contrad · "
        f"{len(ev['low_conf_queries'])} low-conf"
    )

    date_label = target.strftime("%Y-%m-%d")
    prompt = _render_morning_prompt(date_label, ev)
    with console.status("[dim]Armando brief…[/dim]", spinner="dots"):
        narrative = _generate_morning_narrative(prompt)
    if not narrative:
        console.print("[red]Modelo devolvió respuesta vacía.[/red]")
        return

    now_iso = datetime.now().isoformat(timespec="seconds")
    fm_lines = [
        "---",
        f"created: '{now_iso}'",
        "type: morning-brief",
        "tags:",
        "- review",
        "- morning-brief",
        f"date: '{date_label}'",
        "---",
    ]
    body = "\n".join(fm_lines) + f"\n\n# Morning brief — {date_label}\n\n{narrative}\n"

    if dry_run:
        console.rule(f"[bold]Morning brief {date_label} (dry-run)[/bold]")
        console.print(body, markup=False, highlight=False)
        return

    path = VAULT_PATH / MORNING_FOLDER / f"{date_label}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path = path.with_name(
            f"{date_label} ({datetime.now().strftime('%H%M%S')}).md"
        )
    path.write_text(body, encoding="utf-8")
    try:
        _index_single_file(get_db(), path, skip_contradict=True)
    except Exception:
        pass
    rel = path.relative_to(VAULT_PATH)
    console.print(f"[green]✓ Brief guardado:[/green] [bold cyan]{rel}[/bold cyan]")


# ── DEAD NOTES (rag dead) ────────────────────────────────────────────────────
# Candidates for archive: notes with 0 graph edges + never retrieved + old mtime.
# Pure Python. Surfaces candidates only — never moves/deletes.


def _note_created_ts(raw: str, mtime: float) -> float:
    """Best-effort creation timestamp for a note. iCloud sync bumps mtimes
    constantly, so mtime-only age is noisy. Prefer frontmatter `created:` when
    parseable, fall back to mtime.
    """
    if not raw.startswith("---"):
        return mtime
    fm = parse_frontmatter(raw)
    for key in ("created", "date"):
        v = fm.get(key)
        if not v:
            continue
        s = str(v).strip().strip("'\"")
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s[:len(fmt) + 2].rstrip("Z"), fmt).timestamp()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(s).timestamp()
        except ValueError:
            continue
    return mtime


def find_dead_notes(
    col: chromadb.Collection,
    vault: Path,
    query_log: Path,
    min_age_days: int = 365,
    query_window_days: int = 180,
    exclude_folders: tuple[str, ...] = (
        "00-Inbox", "04-Archive", "05-Reviews",
    ),
    use_frontmatter_date: bool = True,
) -> list[dict]:
    """Dead-note candidates. AND of: 0 outlinks + 0 backlinks + not retrieved
    in `query_window_days` + age > `min_age_days` + outside `exclude_folders`.

    Age source: frontmatter `created:` if present (more reliable in iCloud
    vaults where sync constantly bumps mtime), else mtime. Disable with
    `use_frontmatter_date=False`.
    """
    from datetime import timedelta as _td
    corpus = _load_corpus(col)
    title_to_paths = corpus["title_to_paths"]
    paths_with_outlinks = {
        p for p, links in corpus["outlinks"].items() if links
    }
    backlinked_paths: set[str] = set()
    for title, sources in corpus["backlinks"].items():
        if sources:
            backlinked_paths.update(title_to_paths.get(title, set()))

    retrieved_paths: set[str] = set()
    if query_log.is_file():
        cutoff = datetime.now() - _td(days=query_window_days)
        try:
            lines = query_log.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            try:
                ts = datetime.fromisoformat(e.get("ts", ""))
            except Exception:
                continue
            if ts < cutoff:
                continue
            paths = e.get("paths") or []
            if isinstance(paths, list):
                for p in paths:
                    if isinstance(p, str) and p:
                        retrieved_paths.add(p)

    files_info: dict[str, tuple[float, float]] = {}  # rel → (age_ts, mtime)
    if vault.is_dir():
        for p in vault.rglob("*.md"):
            try:
                rel = str(p.relative_to(vault))
            except ValueError:
                continue
            if is_excluded(rel):
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            age_ts = mtime
            if use_frontmatter_date:
                try:
                    raw = p.read_text(encoding="utf-8", errors="ignore")
                    age_ts = _note_created_ts(raw, mtime)
                except OSError:
                    pass
            files_info[rel] = (age_ts, mtime)

    age_cutoff = datetime.now().timestamp() - (min_age_days * 86400)
    candidates: list[dict] = []
    for rel, (age_ts, mtime) in files_info.items():
        if any(
            rel == ex or rel.startswith(ex.rstrip("/") + "/")
            for ex in exclude_folders
        ):
            continue
        if age_ts >= age_cutoff:
            continue
        if rel in paths_with_outlinks:
            continue
        if rel in backlinked_paths:
            continue
        if rel in retrieved_paths:
            continue
        age_days = int((datetime.now().timestamp() - age_ts) // 86400)
        candidates.append({
            "path": rel,
            "age_days": age_days,
            "mtime": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            "age_source": "frontmatter" if age_ts != mtime else "mtime",
        })
    candidates.sort(key=lambda r: -r["age_days"])
    return candidates


@cli.command()
@click.option("--min-age-days", default=365, show_default=True,
              help="Edad mínima por mtime")
@click.option("--query-window-days", default=180, show_default=True,
              help="Ventana para considerar que una nota 'fue usada' en queries")
@click.option("--limit", default=50, show_default=True,
              help="Cap del listado")
@click.option("--folder", default=None, help="Acotar a una subcarpeta")
@click.option("--plain", is_flag=True, help="Salida plana (path por línea)")
def dead(min_age_days: int, query_window_days: int, limit: int,
         folder: str | None, plain: bool):
    """Listar notas candidatas a archivar (dead code del vault).

    Criterio AND: 0 outlinks + 0 backlinks + no recuperada en N días +
    mtime > N días + fuera de Inbox/Archive/Reviews. No borra nada.
    """
    col = get_db()
    items = find_dead_notes(
        col, VAULT_PATH, LOG_PATH,
        min_age_days=min_age_days,
        query_window_days=query_window_days,
    )
    if folder:
        prefix = folder.rstrip("/") + "/"
        items = [
            it for it in items
            if it["path"] == folder or it["path"].startswith(prefix)
        ]
    items = items[:limit]

    log_query_event({
        "cmd": "dead", "min_age_days": min_age_days,
        "query_window_days": query_window_days,
        "folder": folder, "n_candidates": len(items),
    })

    if not items:
        msg = "Sin candidatos a dead notes."
        click.echo(msg) if plain else console.print(f"[green]{msg}[/green]")
        return
    if plain:
        for it in items:
            click.echo(f"{it['age_days']}d\t{it['path']}")
        return
    console.print()
    console.print(Rule(
        title=f"[bold yellow]🪦 {len(items)} nota(s) candidatas a archivar[/bold yellow]",
        style="yellow",
    ))
    tbl = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    tbl.add_column("edad", style="yellow", justify="right")
    tbl.add_column("última modif", style="dim")
    tbl.add_column("path", style="cyan")
    for it in items:
        tbl.add_row(f"{it['age_days']}d", it["mtime"][:10], it["path"])
    console.print(tbl)
    console.print(
        "\n[dim]Criterio AND: 0 outlinks · 0 backlinks · "
        f"no recuperada en {query_window_days}d · mtime > {min_age_days}d.[/dim]"
    )


# ── AUTOMATION (launchd) ─────────────────────────────────────────────────────
# Two services keep the RAG alive without manual rituals:
#  - obsidian-rag-watch: runs `rag watch` (auto-reindex on vault changes).
#  - obsidian-rag-digest: weekly `rag digest`, Sunday 22:00 local.
# Both managed by `rag setup` / `rag setup --remove`. Idempotent reload on each
# install. Logs land in ~/.local/share/obsidian-rag/ for tail-ability.

_LAUNCH_AGENTS_DIR = Path.home() / "Library/LaunchAgents"
_RAG_LOG_DIR = Path.home() / ".local/share/obsidian-rag"


def _rag_binary() -> str:
    """Best-effort path to the installed `rag` binary. Default uv tool path
    first; fall back to PATH lookup. The launchd service runs without our
    interactive PATH so we resolve it once at install time.
    """
    candidates = [
        Path.home() / ".local/bin/rag",
        Path("/usr/local/bin/rag"),
        Path("/opt/homebrew/bin/rag"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    import shutil
    found = shutil.which("rag")
    return found or str(candidates[0])


def _watch_plist(rag_bin: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-watch</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>watch</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/watch.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/watch.error.log</string>
</dict>
</plist>
"""


def _digest_plist(rag_bin: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-digest</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>digest</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>0</integer>
    <key>Hour</key><integer>22</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/digest.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/digest.error.log</string>
</dict>
</plist>
"""


def _morning_plist(rag_bin: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.fer.obsidian-rag-morning</string>
  <key>ProgramArguments</key>
  <array>
    <string>{rag_bin}</string>
    <string>morning</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>{Path.home()}</string>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
    <key>NO_COLOR</key><string>1</string>
    <key>TERM</key><string>dumb</string>
  </dict>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardOutPath</key><string>{_RAG_LOG_DIR}/morning.log</string>
  <key>StandardErrorPath</key><string>{_RAG_LOG_DIR}/morning.error.log</string>
</dict>
</plist>
"""


def _services_spec(rag_bin: str) -> list[tuple[str, str, str]]:
    """Return [(label, plist_filename, plist_xml), ...]."""
    return [
        ("com.fer.obsidian-rag-watch", "com.fer.obsidian-rag-watch.plist",
         _watch_plist(rag_bin)),
        ("com.fer.obsidian-rag-digest", "com.fer.obsidian-rag-digest.plist",
         _digest_plist(rag_bin)),
        ("com.fer.obsidian-rag-morning", "com.fer.obsidian-rag-morning.plist",
         _morning_plist(rag_bin)),
    ]


@cli.command()
@click.option("--remove", is_flag=True,
              help="Desinstalar los servicios en lugar de instalarlos")
def setup(remove: bool):
    """Instalar (o desinstalar) los servicios launchd que mantienen el RAG vivo
    sin intervención: `rag watch` (auto-reindex) y `rag digest` (semanal).
    Idempotente — re-correr lo recarga.
    """
    import subprocess
    rag_bin = _rag_binary()
    if not Path(rag_bin).is_file():
        console.print(f"[red]No encuentro el binario `rag`:[/red] {rag_bin}")
        console.print("[dim]Instalá primero: uv tool install --reinstall --editable .[/dim]")
        return
    _LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    _RAG_LOG_DIR.mkdir(parents=True, exist_ok=True)

    for label, fname, content in _services_spec(rag_bin):
        plist_path = _LAUNCH_AGENTS_DIR / fname
        # Always unload first so a stale version doesn't linger after reinstall.
        if plist_path.exists():
            subprocess.run(
                ["launchctl", "unload", str(plist_path)],
                check=False, capture_output=True,
            )
        if remove:
            if plist_path.exists():
                plist_path.unlink()
                console.print(f"[green]✓[/green] removido: {label}")
            else:
                console.print(f"[dim]· no estaba instalado: {label}[/dim]")
            continue
        plist_path.write_text(content, encoding="utf-8")
        try:
            subprocess.run(
                ["launchctl", "load", str(plist_path)],
                check=True, capture_output=True,
            )
            console.print(f"[green]✓[/green] cargado: [bold]{label}[/bold]")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="ignore") if e.stderr else ""
            console.print(f"[red]✗[/red] falló cargar {label}: {stderr.strip()}")

    if not remove:
        console.print()
        console.print(
            f"[dim]Logs en {_RAG_LOG_DIR}/{{watch,digest}}.{{log,error.log}}[/dim]"
        )


@cli.group()
def ambient():
    """Ambient Agent: reacciona a capturas en 00-Inbox con wikilinks+dupes+related."""


@ambient.command("status")
def ambient_status():
    """¿Está habilitado? ¿Con qué chat_id?"""
    cfg = _ambient_config()
    if cfg is None:
        if not AMBIENT_CONFIG_PATH.is_file():
            console.print("[yellow]Deshabilitado[/yellow] — no hay config. Enviá /enable_ambient al bot.")
        else:
            console.print(
                "[yellow]Deshabilitado[/yellow] — config existe pero está mal o tiene enabled=false."
            )
        return
    console.print(
        f"[green]Habilitado[/green] · chat_id={cfg['chat_id']} · "
        f"bot_token={cfg['bot_token'][:10]}…"
    )
    if AMBIENT_STATE_PATH.is_file():
        try:
            n = sum(1 for _ in AMBIENT_STATE_PATH.open())
            console.print(f"[dim]State: {n} análisis registrados[/dim]")
        except Exception:
            pass


@ambient.command("disable")
def ambient_disable():
    """Deshabilitar ambient (deja el config pero con enabled=false)."""
    if not AMBIENT_CONFIG_PATH.is_file():
        console.print("[dim]No había config.[/dim]")
        return
    try:
        c = json.loads(AMBIENT_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        c = {}
    c["enabled"] = False
    AMBIENT_CONFIG_PATH.write_text(json.dumps(c, indent=2), encoding="utf-8")
    console.print("[green]✓[/green] ambient deshabilitado. Re-habilitar con /enable_ambient en el bot.")


@ambient.command("test")
@click.argument("path", required=False)
def ambient_test(path: str | None):
    """Simular análisis sobre una nota existente (dry-run en vivo, sí escribe)."""
    col = get_db()
    if path is None:
        inbox = VAULT_PATH / _CAPTURE_FOLDER
        candidates = sorted(inbox.glob("*.md"))[:1]
        if not candidates:
            console.print(f"[yellow]Sin notas en {inbox}[/yellow]")
            return
        p = candidates[0]
    else:
        p = VAULT_PATH / path if not path.startswith("/") else Path(path)
    if not p.is_file():
        console.print(f"[red]No existe:[/red] {p}")
        return
    doc_id_prefix = str(p.relative_to(VAULT_PATH))
    raw = p.read_text(encoding="utf-8", errors="ignore")
    h = file_hash(raw)
    console.print(f"[cyan]Triggereando ambient hook sobre:[/cyan] {doc_id_prefix}")
    _ambient_hook(col, p, doc_id_prefix, h)
    console.print("[green]✓[/green] Listo. Ver log: tail ~/.local/share/obsidian-rag/ambient.jsonl")


@ambient.command("log")
@click.option("-n", default=10, show_default=True, help="Últimos N eventos")
def ambient_log(n: int):
    """Tail del log de eventos ambient."""
    if not AMBIENT_LOG_PATH.is_file():
        console.print(f"[yellow]Aún no hay log en {AMBIENT_LOG_PATH}[/yellow]")
        return
    lines = AMBIENT_LOG_PATH.read_text(encoding="utf-8").splitlines()[-n:]
    for line in lines:
        try:
            e = json.loads(line)
        except Exception:
            continue
        ts = e.get("ts", "")[-8:]
        applied = e.get("wikilinks_applied", 0)
        dupes = len(e.get("dupes") or [])
        rel = e.get("related_count", 0)
        sent = "📨" if e.get("telegram_sent") else ("🔇" if e.get("quiet") else "✗")
        console.print(
            f"[dim]{ts}[/dim] {sent} "
            f"[cyan]{e.get('path', '—')}[/cyan] "
            f"[yellow]links={applied}[/yellow] "
            f"[magenta]dupes={dupes}[/magenta] "
            f"[blue]rel={rel}[/blue]"
        )


@cli.command()
def stats():
    """Estado del índice."""
    col = get_db()
    count = col.count()
    try:
        urls_count = get_urls_db().count()
    except Exception:
        urls_count = 0
    console.print(f"[cyan]Chunks indexados:[/cyan] {count}")
    console.print(f"[cyan]URLs indexadas:[/cyan] {urls_count}  [dim]({URLS_COLLECTION_NAME})[/dim]")
    # Identificar la fuente del vault (env / registry / default) ayuda al
    # debug en setups multi-vault.
    vault_src = "default"
    if os.environ.get("OBSIDIAN_RAG_VAULT"):
        vault_src = "env"
    else:
        _vcfg = _load_vaults_config()
        if _vcfg["current"] and _vcfg["current"] in _vcfg["vaults"]:
            vault_src = f"registry · {_vcfg['current']}"
    console.print(f"[cyan]Vault:[/cyan] {VAULT_PATH}  [dim]({vault_src})[/dim]")
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


@cli.group()
def vault():
    """Multi-vault: registrar / cambiar / listar vaults de Obsidian.

    El registry vive en ~/.config/obsidian-rag/vaults.json. Cada vault
    obtiene su propia colección de Chroma automáticamente (namespacing
    por hash del path) — switchear no contamina ni cruza datos.

    Precedencia para resolver el vault activo:
      1. OBSIDIAN_RAG_VAULT env var (override per-invocación, gana siempre).
      2. `vault use <name>` (el "current" del registry, persistente).
      3. Default iCloud Notes (legacy, para usuarios single-vault).
    """


@vault.command("add")
@click.argument("name")
@click.argument("path", type=click.Path(
    exists=True, file_okay=False, dir_okay=True, resolve_path=True,
))
def vault_add(name: str, path: str):
    """Registrar un vault con un nombre. Si es el primero, queda activo."""
    cfg = _load_vaults_config()
    if name in cfg["vaults"] and cfg["vaults"][name] != path:
        console.print(
            f"[yellow]Sobreescribiendo[/yellow] '{name}': "
            f"{cfg['vaults'][name]} → {path}"
        )
    cfg["vaults"][name] = path
    if not cfg["current"]:
        cfg["current"] = name
        marker = " (activo)"
    else:
        marker = ""
    _save_vaults_config(cfg)
    console.print(f"[green]✓[/green] vault [bold]{name}[/bold] → {path}{marker}")


@vault.command("list")
def vault_list():
    """Listar vaults registrados, marcando el activo."""
    cfg = _load_vaults_config()
    if not cfg["vaults"]:
        console.print(
            "[dim]Sin vaults registrados.[/dim] "
            "Usá [bold]rag vault add <name> <path>[/bold] para empezar."
        )
        console.print(f"[dim]Default actual: {_DEFAULT_VAULT}[/dim]")
        return
    cur = cfg["current"]
    env = os.environ.get("OBSIDIAN_RAG_VAULT")
    for name, path in cfg["vaults"].items():
        marker = "[green]→[/green]" if name == cur else "  "
        console.print(f"  {marker} [bold]{name}[/bold]  [dim]{path}[/dim]")
    if env:
        console.print(
            f"\n[yellow]⚠ OBSIDIAN_RAG_VAULT está seteado[/yellow] "
            f"[dim]({env})[/dim] — overridea el registry."
        )


@vault.command("use")
@click.argument("name")
def vault_use(name: str):
    """Cambiar al vault NAME (persistente). Afecta a futuras invocaciones."""
    cfg = _load_vaults_config()
    if name not in cfg["vaults"]:
        registered = ", ".join(cfg["vaults"]) or "(ninguno)"
        console.print(
            f"[red]vault '{name}' no registrado.[/red] "
            f"Registrados: {registered}"
        )
        return
    cfg["current"] = name
    _save_vaults_config(cfg)
    path = cfg["vaults"][name]
    console.print(f"[green]✓[/green] vault activo: [bold]{name}[/bold]  [dim]({path})[/dim]")
    if os.environ.get("OBSIDIAN_RAG_VAULT"):
        console.print(
            "[yellow]⚠[/yellow] OBSIDIAN_RAG_VAULT está seteado — "
            "lo seguirá overrideando hasta que lo desetees."
        )


@vault.command("current")
def vault_current():
    """Mostrar el vault que se va a usar y por qué."""
    env = os.environ.get("OBSIDIAN_RAG_VAULT")
    if env:
        console.print(f"[bold]env[/bold] OBSIDIAN_RAG_VAULT → [cyan]{env}[/cyan]")
        return
    cfg = _load_vaults_config()
    cur = cfg["current"]
    if cur and cur in cfg["vaults"]:
        console.print(
            f"[bold]registry[/bold] [bold]{cur}[/bold] → "
            f"[cyan]{cfg['vaults'][cur]}[/cyan]"
        )
        return
    console.print(f"[bold]default[/bold] → [cyan]{_DEFAULT_VAULT}[/cyan]")


@vault.command("remove")
@click.argument("name")
def vault_remove(name: str):
    """Quitar un vault del registry. NO borra archivos del disco."""
    cfg = _load_vaults_config()
    if name not in cfg["vaults"]:
        console.print(f"[red]vault '{name}' no registrado.[/red]")
        return
    del cfg["vaults"][name]
    if cfg["current"] == name:
        cfg["current"] = next(iter(cfg["vaults"]), None)
    _save_vaults_config(cfg)
    if cfg["current"]:
        console.print(
            f"[green]✓[/green] '{name}' removido. "
            f"Activo ahora: [bold]{cfg['current']}[/bold]"
        )
    else:
        console.print(
            f"[green]✓[/green] '{name}' removido. "
            f"Sin current — caerá al default."
        )


@cli.group()
def session():
    """Administrar sesiones conversacionales (list / show / clear / cleanup)."""


@session.command("list")
@click.option("-n", "limit", default=20, help="Máximo de sesiones a listar (default: 20)")
def session_list(limit: int):
    """Lista las sesiones recientes."""
    rows = list_sessions(limit=limit)
    if not rows:
        console.print("[dim]No hay sesiones.[/dim]")
        return
    t = Table(title=f"Sesiones ({len(rows)})", show_header=True, header_style="bold")
    t.add_column("id", style="cyan")
    t.add_column("turns", justify="right")
    t.add_column("modo", style="dim")
    t.add_column("actualizada", style="dim")
    t.add_column("primer turno")
    for r in rows:
        t.add_row(
            r["id"], str(r["turns"]), r.get("mode", ""),
            r.get("updated_at", ""), r.get("first_q", ""),
        )
    console.print(t)


@session.command("show")
@click.argument("sid")
def session_show(sid: str):
    """Muestra los turnos de una sesión."""
    s = load_session(sid)
    if not s:
        console.print(f"[red]Sesión no encontrada: {sid}[/red]")
        return
    console.print(Panel(
        f"[bold cyan]{s['id']}[/bold cyan]  [dim]· {len(s.get('turns', []))} turnos · "
        f"{s.get('mode', '')}  · creada {s.get('created_at', '')}[/dim]",
        border_style="cyan",
    ))
    for i, turn in enumerate(s.get("turns", []), 1):
        console.print()
        console.print(f"[bold]{i}. ❯[/bold] {turn.get('q', '')}")
        if turn.get("q_reformulated"):
            console.print(f"   [dim italic]reformulada: {turn['q_reformulated']}[/dim italic]")
        if turn.get("a"):
            console.print(render_response(turn["a"]))
        paths = turn.get("paths") or []
        if paths:
            console.print(f"   [dim]sources: {', '.join(paths[:3])}{'…' if len(paths) > 3 else ''}[/dim]")


@session.command("clear")
@click.argument("sid")
@click.option("--yes", is_flag=True, help="No pedir confirmación")
def session_clear(sid: str, yes: bool):
    """Borra una sesión por id."""
    try:
        p = session_path(sid)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return
    if not p.exists():
        console.print(f"[yellow]No existe:[/yellow] {sid}")
        return
    if not yes:
        try:
            confirm = console.input(f"Borrar sesión [cyan]{sid}[/cyan]? [y/N] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            console.print("[dim]cancelado[/dim]")
            return
        if confirm != "y":
            console.print("[dim]cancelado[/dim]")
            return
    p.unlink()
    console.print(f"[green]✓ borrada:[/green] {sid}")


@session.command("cleanup")
@click.option("--days", default=SESSION_TTL_DAYS, show_default=True,
              help="Borrar sesiones más viejas que N días (mtime)")
def session_cleanup(days: int):
    """Purga sesiones viejas."""
    removed = cleanup_sessions(ttl_days=days)
    console.print(f"[cyan]{removed}[/cyan] sesión(es) borrada(s) (TTL {days}d)")


if __name__ == "__main__":
    cli()
