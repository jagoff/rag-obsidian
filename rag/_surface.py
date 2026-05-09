"""Surface — proactive bridge builder + PageRank + filing helpers.

Phase 5 cont de modularización (audit perf 2026-05-08, ver
`99-obsidian/99-AI/system/perf-modularization-2026-05-08/plan.md`):
extraer el sub-sistema "surface" desde `rag/__init__.py`.

## Qué vive acá

- **PageRank infrastructure**:
  - `_build_graph_adj` — adyacencia no dirigida desde outlinks +
    title_to_paths.
  - `_graph_pagerank` — power iteration vanilla.
  - `_personalized_pagerank` — topic-aware (PPR) con teleport sesgado
    a seeds.
  - `get_pagerank` — wrapper cacheado, invalidate keyed por
    `collection_id`.
- **Surface bridge builder**:
  - `find_surface_bridges` — pares semánticamente cercanos pero
    topológicamente lejanos (N-hop apart en el grafo de wikilinks).
  - `_surface_generate_reason` — LLM helper que arma "por qué
    conectan" en una oración.
  - `_surface_log_run` — append SQL.
- **Inbox triage helpers** (usados por filing + triage):
  - `_is_moc_note` — heurística MOC (title/tag/folder-index).
  - `_note_age_days`, `_hop_set` — utilidades del grafo.
  - `_suggest_tags_for_note` — LLM helper sobre vocab existente.
  - `_apply_frontmatter_tags` — rewrite del bloque `tags:`.
  - `_suggest_folder_for_note` — mode-vote sobre folders de vecinos.
  - `triage_inbox_note` — orquestador de signals (folder + tags +
    wikilinks + dupes).

## PageRank cache — cross-module rebind

`_pagerank_cache` + `_pagerank_cache_cid` viven acá pero los
manipulamos vía `rag.X` para que `_invalidate_corpus_cache()`
(en `rag/__init__.py`) pueda invalidarlos sin import explícito.
Pattern: `import rag as _rag; _rag._pagerank_cache = new_rank`
funciona porque Python permite atributo-assignment sobre un
módulo. Y `from rag._surface import *` re-exporta las constantes
al namespace `rag`, así que `rag._pagerank_cache` es el mismo
objeto que `rag._surface._pagerank_cache`.

## Lazy imports

Deps en `_load_corpus`, `_corpus_cache_lock`, `_note_centroids`,
`VAULT_PATH`, `embed`, `clean_md`, `_chat_capped_client`,
`resolve_chat_model`, `chat_keep_alive`, `_helper_client`,
`HELPER_MODEL`, `HELPER_OPTIONS`, `LLM_KEEP_ALIVE`,
`_ragvec_state_conn`, `_sql_append_event`, `_sql_write_with_retry`,
`_log_surface_event_background_default`, `_enqueue_background_sql`,
`parse_frontmatter`, `_normalize_fm_tags`, `find_wikilink_suggestions`,
`find_near_duplicates_for` — todos en `rag/__init__.py`. Lazy
adentro de funciones evita circular import.

`_map_surface_row` se importa top-level desde `rag._row_mappers`
(módulo hoja sin deps al parent).

## Re-export

`rag/__init__.py` hace `from rag._surface import *  # noqa`.
Preserva 100% compat con `rag.find_surface_bridges`,
`rag.get_pagerank`, `rag._is_moc_note`, etc.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rag._row_mappers import _map_surface_row

if TYPE_CHECKING:
    from rag import SqliteVecCollection

__all__ = [
    "SURFACE_LOG_PATH",
    "SURFACE_SKIP_FOLDERS",
    "_pagerank_cache",
    "_pagerank_cache_cid",
    "_build_graph_adj",
    "_graph_pagerank",
    "_personalized_pagerank",
    "get_pagerank",
    "_hop_set",
    "_SURFACE_MOC_TITLE_RE",
    "_is_moc_note",
    "_note_age_days",
    "find_surface_bridges",
    "_surface_generate_reason",
    "_surface_log_run",
    "_suggest_tags_for_note",
    "_apply_frontmatter_tags",
    "_suggest_folder_for_note",
    "triage_inbox_note",
]


SURFACE_LOG_PATH = Path.home() / ".local/share/obsidian-rag/surface.jsonl"
SURFACE_SKIP_FOLDERS = ("00-Inbox/", "04-Archive/")


def _build_graph_adj(corpus: dict) -> dict[str, set[str]]:
    """Adyacencia no dirigida del grafo de wikilinks: path → set de paths linkeados.
    Compone outlinks (path → títulos) con title_to_paths (título → paths).
    Cada edge existe en ambas direcciones para tratar el grafo como no dirigido.

    Memoized on the corpus dict — `retrieve()` calls this path twice per turn
    (once inside `get_pagerank`, once in graph expansion) plus `surface` and
    other callers. Adj is a pure function of the corpus; the corpus cache
    already invalidates on `col.count()` delta, so piggybacking is safe.
    """
    cached = corpus.get("_adj_cache")
    if cached is not None:
        return cached
    adj: dict[str, set[str]] = {}
    title_to_paths = corpus["title_to_paths"]
    for src_path, titles in corpus["outlinks"].items():
        for t in titles:
            for tgt in title_to_paths.get(t, ()):
                if tgt == src_path:
                    continue
                adj.setdefault(src_path, set()).add(tgt)
                adj.setdefault(tgt, set()).add(src_path)
    corpus["_adj_cache"] = adj
    return adj


def _graph_pagerank(
    adj: dict[str, set[str]], damping: float = 0.85, iterations: int = 20,
) -> dict[str, float]:
    """Compute PageRank over the wikilink adjacency graph.

    Simple power iteration — O(iterations * edges). With ~2k notes and
    ~5k edges, this takes <10ms. Returns path → score (sums to 1.0).
    """
    nodes = list(adj.keys())
    if not nodes:
        return {}
    n = len(nodes)
    rank = {node: 1.0 / n for node in nodes}
    for _ in range(iterations):
        new_rank: dict[str, float] = {}
        for node in nodes:
            incoming = 0.0
            for neighbor in adj.get(node, set()):
                out_degree = len(adj.get(neighbor, set())) or 1
                incoming += rank.get(neighbor, 0.0) / out_degree
            new_rank[node] = (1 - damping) / n + damping * incoming
        rank = new_rank
    return rank


def _personalized_pagerank(
    adj: dict[str, set[str]],
    seed_paths: list[str] | set[str],
    *,
    damping: float = 0.85,
    iterations: int = 15,
) -> dict[str, float]:
    """Topic-aware PageRank — power iteration with teleport biased to seeds.

    Feature #6 del 2026-04-23. Classic PageRank assumes uniform random
    teleport (1/n per node); personalized PageRank teleports to a small
    `seed_paths` set, making authority scores topic-specific. Seed paths
    come from the top-K of the cross-encoder rerank — so `ppr[path]` is
    how much wikilink authority `path` has WITHIN the topic of the query.

    Complexity matches the classic version: O(iterations * edges). With
    15 iterations (down from 20 for PageRank; PPR converges faster on
    small seed sets) and ~5k edges, <7ms in practice. Returns path →
    score (sums to ~1.0 subject to numerical stability).

    Seeds not in the adjacency are ignored. If all seeds are unknown or
    the seed list is empty, falls back to uniform teleport (= plain
    PageRank, same as _graph_pagerank).
    """
    nodes = list(adj.keys())
    if not nodes:
        return {}
    len(nodes)
    # Filter seed set to known nodes; drop unknowns silently.
    if isinstance(seed_paths, (list, tuple)):
        seed_set = {p for p in seed_paths if p in adj}
    else:
        seed_set = {p for p in seed_paths if p in adj}
    if not seed_set:
        # Audit 2026-04-26 M3: pre-fix devolvía global PageRank acá. Pero
        # el caller (`retrieve()` línea ~19979) distinguía PPR vs global
        # via `if ppr_score else pagerank` — si esto retorna global, la
        # distinción se pierde silenciosamente y el ranker usa "global"
        # creyendo que es PPR. Devolvemos `{}` para que el caller caiga
        # explícitamente al global por su path normal.
        return {}
    # Personalization vector: 1/|seed| on seeds, 0 elsewhere.
    seed_weight = 1.0 / len(seed_set)
    teleport = {node: (seed_weight if node in seed_set else 0.0) for node in nodes}
    # Initial rank: same as teleport (biased toward seeds).
    rank = dict(teleport)
    for _ in range(iterations):
        new_rank: dict[str, float] = {}
        for node in nodes:
            incoming = 0.0
            for neighbor in adj.get(node, set()):
                out_degree = len(adj.get(neighbor, set())) or 1
                incoming += rank.get(neighbor, 0.0) / out_degree
            new_rank[node] = (1 - damping) * teleport[node] + damping * incoming
        rank = new_rank
    return rank


# Module-level cache for PageRank — invalidated with corpus cache.
# `_invalidate_corpus_cache` (en `rag/__init__.py`) muta estas vars vía
# `rag._pagerank_cache = None` (se accede por re-export, mismo objeto).
_pagerank_cache: dict[str, float] | None = None
_pagerank_cache_cid: str | None = None


def get_pagerank(col: "SqliteVecCollection") -> dict[str, float]:
    """Cached PageRank over the wikilink graph. Rebuilt when corpus changes.

    Invalidation key is the corpus `collection_id` (sqlite-vec's stable
    UUID — rotates on delete+create). The previous implementation keyed
    on `id(corpus)` which is the memory address of the dict; that address
    changes every time `_load_corpus` rebuilds even when the content is
    identical, causing needless PageRank recomputation on every retrieve
    after any cache-invalidation nudge.
    """
    import rag as _rag  # noqa: PLC0415

    corpus = _rag._load_corpus(col)
    cid = corpus.get("collection_id") or ""
    with _rag._corpus_cache_lock:
        cached = getattr(_rag, "_pagerank_cache", None)
        cached_cid = getattr(_rag, "_pagerank_cache_cid", None)
        if cached is not None and cached_cid == cid:
            return cached
    adj = _build_graph_adj(corpus)
    new_rank = _graph_pagerank(adj)
    with _rag._corpus_cache_lock:
        _rag._pagerank_cache = new_rank
        _rag._pagerank_cache_cid = cid
    return new_rank


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
    col: "SqliteVecCollection",
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
      - ninguna nota en 00-Inbox / 04-Archive (incluye reviews/)
      - ninguna nota es MOC (título, tag o folder-index)
      - el par NO comparte ≥2 tags (la conexión ya es explícita vía tags)
      - ambas notas tienen edad ≥ skip_young_days (notas frescas siguen evolucionando)

    Retorna los `top` mejores pares por similitud descendente. Cada dict incluye
    snippets (primeros ~800 chars del body, sin frontmatter) para que el caller
    pueda alimentar un LLM opcional que genere la oración de "por qué conectan".
    """
    from rag import VAULT_PATH, _load_corpus, _note_centroids  # noqa: PLC0415

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
    from rag import (  # noqa: PLC0415
        _chat_capped_client,
        chat_keep_alive,
        resolve_chat_model,
    )

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
        resp = _chat_capped_client().chat(
            model=resolve_chat_model(),
            messages=[{"role": "user", "content": prompt}],
            # num_ctx=4096 matches CHAT_OPTIONS / _WEB_CHAT_NUM_CTX. Previous
            # 2048 value was invalidating the web chat's KV cache every time
            # the watch thread indexed a note and hit this path.
            options={"temperature": 0, "top_p": 1, "seed": 42,
                     "num_ctx": 4096, "num_predict": 80},
            keep_alive=chat_keep_alive(),
        )
        reason = (resp.message.content or "").strip().split("\n", 1)[0].strip()
    except Exception:
        return ""
    if not reason or "sin conexión clara" in reason.lower():
        return ""
    reason = reason.strip(".") + "."
    return reason if len(reason) > 4 else ""


def _surface_log_run(summary: dict, pairs: list[dict]) -> None:
    """Append-only log: una línea `surface_run` + N líneas `surface_pair` en
    rag_surface_log. Mismo timestamp en todas para poder agrupar la corrida
    al leer. SQL-only since T10."""
    from rag import (  # noqa: PLC0415
        _enqueue_background_sql,
        _log_surface_event_background_default,
        _ragvec_state_conn,
        _sql_append_event,
        _sql_write_with_retry,
    )

    ts = datetime.now().isoformat(timespec="seconds")

    def _do() -> None:
        with _ragvec_state_conn() as conn:
            _sql_append_event(
                conn, "rag_surface_log",
                _map_surface_row({"ts": ts, "cmd": "surface_run", **summary}),
            )
            for p in pairs:
                _sql_append_event(
                    conn, "rag_surface_log",
                    _map_surface_row({"ts": ts, "cmd": "surface_pair", **p}),
                )
    if _log_surface_event_background_default():
        _enqueue_background_sql(_do, "surface_sql_write_failed")
    else:
        _sql_write_with_retry(_do, "surface_sql_write_failed")


def _suggest_tags_for_note(
    col: "SqliteVecCollection",
    body: str,
    note_title: str,
    max_tags: int = 6,
) -> list[str]:
    """Pure helper: ask the helper LLM to pick tags from existing vault vocab.
    Returns picked list (may be empty). Shared by `rag autotag` and `rag inbox`.
    """
    from rag import (  # noqa: PLC0415
        HELPER_MODEL,
        HELPER_OPTIONS,
        LLM_KEEP_ALIVE,
        _helper_client,
        _load_corpus,
    )

    c = _load_corpus(col)
    vocab = sorted(c["tags"])  # noqa: F811 — local var shadows re-exported CLI symbol
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
        resp = _helper_client().chat(
            model=HELPER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=HELPER_OPTIONS,
            keep_alive=LLM_KEEP_ALIVE,
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
    col: "SqliteVecCollection",
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
    from rag import VAULT_PATH, clean_md, embed  # noqa: PLC0415

    full = (VAULT_PATH / note_path).resolve()
    try:
        full.relative_to(VAULT_PATH.resolve())
    except ValueError:
        return ("", 0.0)
    if not full.is_file():
        return ("", 0.0)
    raw = full.read_text(encoding="utf-8", errors="ignore")
    text = clean_md(raw)[:3000].strip()
    col_total = int(col.count())
    if not text or col_total == 0:
        return ("", 0.0)
    try:
        q_embed = embed([text])[0]
    except Exception:
        return ("", 0.0)
    n = min(k * 4, col_total)
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
    from collections import Counter  # noqa: PLC0415
    best, count = Counter(folders).most_common(1)[0]
    return (best, round(count / len(folders), 3))


def triage_inbox_note(
    col: "SqliteVecCollection",
    note_path: str,
    max_tags: int = 5,
    dupe_threshold: float = 0.85,
) -> dict:
    """Compose all triage signals for one Inbox note: destination folder, tags
    from vocabulary, wikilink suggestions, near-duplicate flags. Returns a
    plain dict the CLI renderer + the eventual `--apply` path consume.
    """
    from rag import (  # noqa: PLC0415
        VAULT_PATH,
        _normalize_fm_tags,
        clean_md,
        find_near_duplicates_for,
        find_wikilink_suggestions,
        parse_frontmatter,
    )

    full = (VAULT_PATH / note_path).resolve()
    if not full.is_file():
        return {"path": note_path, "error": "not found"}
    raw = full.read_text(encoding="utf-8", errors="ignore")
    fm = parse_frontmatter(raw)
    current_tags = _normalize_fm_tags(fm)
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
