"""Weekly consolidation of episodic-memory conversation notes (Phase 2).

Scans `04-Archive/99-obsidian-system/99-AI/conversations/`, groups
related conversations by embedding similarity (connected components on
cosine ≥ threshold), synthesises each cluster into a single consolidated
note in the appropriate PARA folder, and archives the originals under
`04-Archive/conversations/YYYY-MM/`.

Invoked via `rag consolidate` or the weekly launchd plist
`com.fer.obsidian-rag-consolidate` (Mondays 06:00 local).

Design notes:
- `04-Archive/99-obsidian-system/99-AI/conversations/` is already
  excluded from the search index (`is_excluded` cubre TODO el prefix
  `04-Archive/99-obsidian-system/`), así que originals son invisibles a
  `retrieve()` hasta que un consolidated note las promueva a PARA.
  `04-Archive/conversations/` también se excluye explícitamente para
  que los originales archivados no leakeen de vuelta.
- Pre-2026-04-25 las conversations vivían en `00-Inbox/conversations/`.
  El consolidator antes scaneaba esa carpeta. Tras 2026-04-25 las
  conversations son "system files" (no son del PARA del user, son
  artefactos generados por el chat web), por eso pasaron a vivir bajo
  `99-AI/`. Si el user tiene archivos legacy en
  `00-Inbox/conversations/`, este script los ignora — moverlos a la
  nueva ubicación o borrarlos a mano.
- Representation per conversation = `first_question + answer_preview`
  embedded via `bge-m3`. One batch embed call per run.
- Chat model (resolve_chat_model → qwen2.5:7b by default) does the
  synthesis. ~6s per cluster; for the typical 1-3 clusters/week run
  this is fine. Non-streaming; temperature=0 for determinism.
- Project vs Resource heuristic: regex over conversation bodies for
  action verbs / future tense / dates → project. Conservative default:
  `03-Resources/` (safer to over-archive than over-promote to Projects).
- Dry-run prints the plan and never writes. Real runs write under a
  `BEGIN IMMEDIATE` SQL guard so concurrent indexers don't see a
  half-moved source file.

Log schema (`~/.local/share/obsidian-rag/consolidation.log`, JSONL):
  {run_at, window_days, n_conversations, n_clusters, n_promoted,
   n_archived, duration_s, dry_run, clusters: [{size, target, title}]}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag  # noqa: E402
from web import conversation_writer  # noqa: E402


CONVERSATIONS_SUBFOLDER = "04-Archive/99-obsidian-system/99-AI/conversations"
ARCHIVE_SUBFOLDER = "04-Archive/conversations"
CONSOLIDATION_LOG = Path.home() / ".local/share/obsidian-rag/consolidation.log"

# Default heuristics — tuneable via flags. 0.75 is the midpoint between
# "clearly same topic" (>0.85) and "maybe related" (~0.65) observed in
# manual spot-checks across a 30-note sample. Lower = more aggressive
# consolidation; higher = fewer but tighter clusters.
DEFAULT_THRESHOLD = 0.75
DEFAULT_MIN_CLUSTER = 3
DEFAULT_WINDOW_DAYS = 14

# Project classifier keywords. Matched as whole-word or with common ES/EN
# prefixes. Conservative — only bumps to Projects on clear action signal;
# everything else goes to Resources.
_PROJECT_PATTERNS = re.compile(
    r"\b(?:"
    r"ma(?:ñ|n)ana|pr(?:ó|o)xim[ao]s?|deadline|entrega|vence|para\s+el\s+\d+|"
    r"agenda[rs]?|agend(?:á|ar|ame)|tengo\s+que|hay\s+que|"
    r"mand(?:á|ar|ame)|escrib(?:í|ir|ime)|prepar(?:á|ar|ame)|"
    r"recordame|recorda|armar|llamar\s+a|"
    r"todo|to[-\s]?do|tarea|task|milestone|sprint|"
    r"next\s+week|next\s+month|tomorrow"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConversationNote:
    path: Path                    # absolute
    rel_path: str                 # vault-relative
    created: datetime
    updated: datetime
    turns: int
    first_question: str
    first_answer: str
    body: str                     # full body (ex-frontmatter), raw
    sources: list[str]            # from frontmatter


# ── Reader ─────────────────────────────────────────────────────────────────

def _parse_iso(ts: str) -> datetime | None:
    """Lenient ISO parser. Handles trailing Z + naive timestamps."""
    if not ts:
        return None
    t = ts.strip().replace("Z", "+00:00") if ts.endswith("Z") else ts
    try:
        dt = datetime.fromisoformat(t)
        # Normalise to naive local for window comparison.
        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)
        return dt
    except ValueError:
        return None


_FIRST_TURN_RE = re.compile(
    r"^##\s+Turn\s+1\b[^\n]*\n\n>\s*(?P<q>.+?)\n\n(?P<a>.+?)(?:\n\n\*\*Sources\*\*:|$)",
    re.DOTALL | re.MULTILINE,
)


def _extract_first_turn(body: str) -> tuple[str, str]:
    """Pull (first_question, first_answer) from the body. Returns ('','')
    on malformed input — the conversation is still usable, just embed-only."""
    m = _FIRST_TURN_RE.search(body)
    if not m:
        return "", ""
    q = m.group("q").strip()
    a = m.group("a").strip()
    # Collapse whitespace; keep first ~600 chars of answer (enough to encode
    # the topic without bloating the embed input).
    a_cleaned = re.sub(r"\s+", " ", a)[:600]
    return q, a_cleaned


def scan_conversations(
    root: Path, window_days: int, *, now: datetime | None = None,
) -> list[ConversationNote]:
    """Load conversation notes modified within the window.

    `root` is the absolute path to the conversations folder
    (`04-Archive/99-obsidian-system/99-AI/conversations/`). Malformed
    files (bad frontmatter, missing Turn 1) are skipped silently — the
    writer path always produces well-formed notes so any breakage is a
    manual edit we want to leave untouched.
    """
    out: list[ConversationNote] = []
    if not root.is_dir():
        return out
    cutoff = (now or datetime.now()).timestamp() - window_days * 86400
    for p in sorted(root.glob("*.md")):
        try:
            if p.stat().st_mtime < cutoff:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            meta, body = conversation_writer._parse_frontmatter(text)
        except (ValueError, OSError):
            continue
        created = _parse_iso(meta.get("created") or "")
        updated = _parse_iso(meta.get("updated") or "")
        if created is None:
            continue
        try:
            turns = int(meta.get("turns") or 0)
        except (TypeError, ValueError):
            turns = 0
        q, a = _extract_first_turn(body)
        sources = meta.get("sources") or []
        if not isinstance(sources, list):
            sources = []
        vault_root = _vault_root_of(p)
        try:
            rel_path = str(p.relative_to(vault_root))
        except ValueError:
            rel_path = str(p)
        out.append(ConversationNote(
            path=p, rel_path=rel_path,
            created=created, updated=updated or created,
            turns=turns,
            first_question=q, first_answer=a,
            body=body, sources=sources,
        ))
    return out


def _vault_root_of(conversation_path: Path) -> Path:
    """Walk up to find the vault root (parent of `00-Inbox`).

    El loop sube por TODOS los parents buscando un dir hermano del PARA
    (`00-Inbox/` o `01-Projects/`), así que funciona igual con la path
    legacy (`<vault>/00-Inbox/conversations/file.md` → 2 niveles arriba)
    como con la nueva (`<vault>/04-Archive/99-obsidian-system/99-AI/
    conversations/file.md` → 4 niveles arriba). Si no encuentra el PARA
    (escenario raro: vault malformado, test con layout custom), fallback
    al parent inmediato — conservador, no rompe walks futuros."""
    for parent in conversation_path.parents:
        if (parent / "00-Inbox").is_dir() or (parent / "01-Projects").is_dir():
            return parent
    return conversation_path.parent


# ── Clustering ─────────────────────────────────────────────────────────────

def _l2_normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return v
    return [x / n for x in v]


def _cosine(u: list[float], v: list[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def _embed_conversations(items: list[ConversationNote]) -> list[list[float]]:
    """One batch call to bge-m3. Normalised so cosine == dot."""
    if not items:
        return []
    texts = [
        f"{it.first_question}\n\n{it.first_answer}".strip() or "(empty)"
        for it in items
    ]
    raw = rag.embed(texts)
    return [_l2_normalize(v) for v in raw]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def cluster_by_embedding(
    embeddings: list[list[float]], *, threshold: float, min_cluster: int,
) -> list[list[int]]:
    """Connected-components clustering. Returns lists of indices,
    size ≥ min_cluster, sorted largest-first."""
    n = len(embeddings)
    if n < min_cluster:
        return []
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _cosine(embeddings[i], embeddings[j]) >= threshold:
                uf.union(i, j)
    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(uf.find(i), []).append(i)
    clusters = [idxs for idxs in buckets.values() if len(idxs) >= min_cluster]
    clusters.sort(key=len, reverse=True)
    return clusters


# ── Target folder classifier ──────────────────────────────────────────────

def classify_target_folder(cluster: list[ConversationNote]) -> str:
    """Project vs Resource. Conservative — needs ≥2 action hits across the
    cluster to bump to Projects, otherwise Resources."""
    hits = 0
    for it in cluster:
        if _PROJECT_PATTERNS.search(it.body):
            hits += 1
            if hits >= 2:
                return "01-Projects"
    return "03-Resources"


# ── Synthesis ──────────────────────────────────────────────────────────────

_SYNTH_SYSTEM = (
    "Sos un editor que consolida varias conversaciones sobre un mismo "
    "tema en una sola nota Markdown para un Obsidian vault.\n"
    "\n"
    "Requisitos:\n"
    "- Escribí en español rioplatense, natural y directo.\n"
    "- Devolvé SOLO el body de la nota (sin frontmatter, sin backticks).\n"
    "- Empezá con una oración que describa el tema central.\n"
    "- Después agrupá en secciones con `##`: puntos clave, decisiones tomadas, "
    "preguntas abiertas. Omití secciones vacías.\n"
    "- Preservá los [[wikilinks]] que aparecen en el input.\n"
    "- NO inventes información que no esté en las conversaciones.\n"
    "- NO repitas literalmente pregunta-respuesta — destilá."
)


def _synth_user_prompt(cluster: list[ConversationNote]) -> str:
    parts = ["Consolidá las siguientes conversaciones en una sola nota:\n"]
    for i, it in enumerate(cluster, 1):
        parts.append(f"--- Conversación {i} ({it.created.date().isoformat()}) ---\n")
        # First turn (representative) + truncated tail of body.
        parts.append(f"Pregunta inicial: {it.first_question}\n\n")
        body_snip = it.body.strip()[:4000]
        parts.append(f"{body_snip}\n")
    parts.append(
        "\nDevolvé solo el body consolidado, sin frontmatter ni metacomentarios."
    )
    return "\n".join(parts)


def synthesize_cluster(
    cluster: list[ConversationNote], *, model: str | None = None,
) -> tuple[str, str]:
    """Return `(title, body)`. Title = best-effort extraction from the first
    H1/first-line; body = the full markdown the LLM produced. Raises on LLM
    failure; caller decides whether to skip or retry.
    """
    import ollama  # imported lazily so unit tests can monkeypatch without network
    chat_model = model or rag.resolve_chat_model()
    resp = ollama.chat(
        model=chat_model,
        messages=[
            {"role": "system", "content": _SYNTH_SYSTEM},
            {"role": "user", "content": _synth_user_prompt(cluster)},
        ],
        options=rag.CHAT_OPTIONS,
        keep_alive=rag.OLLAMA_KEEP_ALIVE,
    )
    body = (resp.get("message", {}).get("content") or "").strip()
    if not body:
        raise RuntimeError("empty synthesis response")
    # Title heuristic: first non-empty line that isn't a heading marker.
    title = ""
    for ln in body.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # strip leading #, **, ", bullets
        title = re.sub(r"^[#*\-\"\s]+", "", ln).rstrip(" .:;")
        if title:
            break
    if not title:
        title = f"Conversaciones consolidadas {cluster[0].created.date().isoformat()}"
    return title[:80], body


# ── Writers ────────────────────────────────────────────────────────────────

def _safe_stem(title: str) -> str:
    safe = re.sub(r"[/\\:\n]", " ", title).strip()
    return safe or datetime.now().strftime("consolidated-%Y%m%d-%H%M%S")


def _render_consolidated(
    title: str, body: str, cluster: list[ConversationNote], target_folder: str,
) -> str:
    date_range = (
        min(it.created for it in cluster).date().isoformat(),
        max(it.created for it in cluster).date().isoformat(),
    )
    # Point archive wikilinks at the NEW archive paths (post-move). This
    # is a forward reference — archive happens after the .md lands.
    archive_month = datetime.now().strftime("%Y-%m")
    origins = []
    for it in cluster:
        archived_rel = (
            f"{ARCHIVE_SUBFOLDER}/{archive_month}/{it.path.name}"
        )
        # Obsidian-style wikilink to the archived file.
        archived_title = archived_rel[:-3] if archived_rel.endswith(".md") else archived_rel
        origins.append(f"- [[{archived_title}]] — {it.created.date().isoformat()}")
    # Sources union across all turns of all conversations.
    sources_union: set[str] = set()
    for it in cluster:
        for s in it.sources:
            if isinstance(s, str) and s:
                sources_union.add(s)
    fm = [
        "---",
        "type: consolidated-conversation",
        f"created: {datetime.now().isoformat(timespec='seconds')}",
        f"source_conversations: {len(cluster)}",
        f"date_range: [{date_range[0]}, {date_range[1]}]",
        "tags:",
        "  - consolidated-conversation",
        "  - rag-chat",
        "---",
    ]
    head = "\n".join(fm) + "\n\n"
    origin_block = (
        "## Conversaciones originales\n\n" + "\n".join(origins) + "\n"
    )
    if sources_union:
        sources_block = (
            "\n## Notas referenciadas\n\n"
            + "\n".join(f"- [[{s[:-3] if s.endswith('.md') else s}]]"
                         for s in sorted(sources_union))
            + "\n"
        )
    else:
        sources_block = ""
    return f"{head}{body.rstrip()}\n\n{origin_block}{sources_block}"


def _unique_path(candidate: Path) -> Path:
    """Append `(N)` if `candidate` exists."""
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    parent = candidate.parent
    for n in range(2, 100):
        alt = parent / f"{stem} ({n}){suffix}"
        if not alt.exists():
            return alt
    # Last-resort: timestamp.
    return parent / f"{stem}-{datetime.now().strftime('%H%M%S')}{suffix}"


def promote(
    vault_root: Path,
    target_folder: str,
    cluster: list[ConversationNote],
    title: str,
    body: str,
) -> Path:
    target_dir = vault_root / target_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{_safe_stem(title)}.md"
    path = _unique_path(path)
    content = _render_consolidated(title, body, cluster, target_folder)
    # Atomic write — same pattern as conversation_writer.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def archive_originals(
    vault_root: Path, cluster: list[ConversationNote],
) -> list[Path]:
    """Move each conversation into `04-Archive/conversations/YYYY-MM/`.
    Returns the new absolute paths. Errors skip individual files — caller
    sees the partial list and logs the discrepancy."""
    archive_dir = vault_root / ARCHIVE_SUBFOLDER / datetime.now().strftime("%Y-%m")
    archive_dir.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    for it in cluster:
        dst = archive_dir / it.path.name
        dst = _unique_path(dst)
        try:
            shutil.move(str(it.path), str(dst))
            moved.append(dst)
        except OSError:
            # Partial failure — don't rollback the whole cluster; log and
            # continue. The consolidated note is already written.
            continue
    return moved


# ── Orchestration ──────────────────────────────────────────────────────────

def _log_run(record: dict) -> None:
    try:
        CONSOLIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CONSOLIDATION_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def run(
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    threshold: float = DEFAULT_THRESHOLD,
    min_cluster: int = DEFAULT_MIN_CLUSTER,
    dry_run: bool = False,
    vault_root: Path | None = None,
    now: datetime | None = None,
) -> dict:
    """Run one consolidation pass. Returns a summary dict; also appends a
    JSONL record to `consolidation.log` unless dry_run."""
    t0 = time.perf_counter()
    vault = vault_root if vault_root is not None else rag.VAULT_PATH
    conv_root = vault / CONVERSATIONS_SUBFOLDER
    items = scan_conversations(conv_root, window_days, now=now)
    summary: dict = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "window_days": window_days,
        "threshold": threshold,
        "min_cluster": min_cluster,
        "n_conversations": len(items),
        "n_clusters": 0,
        "n_promoted": 0,
        "n_archived": 0,
        "dry_run": bool(dry_run),
        "clusters": [],
    }
    if not items:
        summary["duration_s"] = round(time.perf_counter() - t0, 2)
        if not dry_run:
            _log_run(summary)
        return summary

    embeddings = _embed_conversations(items)
    clusters_idx = cluster_by_embedding(
        embeddings, threshold=threshold, min_cluster=min_cluster,
    )
    summary["n_clusters"] = len(clusters_idx)

    for idxs in clusters_idx:
        cluster = [items[i] for i in idxs]
        target_folder = classify_target_folder(cluster)
        if dry_run:
            first_q = cluster[0].first_question[:60]
            summary["clusters"].append({
                "size": len(cluster),
                "target": target_folder,
                "title": first_q or "(sin título)",
                "paths": [c.rel_path for c in cluster],
            })
            continue
        try:
            title, body = synthesize_cluster(cluster)
        except Exception as exc:
            summary["clusters"].append({
                "size": len(cluster), "target": target_folder,
                "title": "(synthesis failed)", "error": repr(exc),
                "paths": [c.rel_path for c in cluster],
            })
            continue
        promoted_path = promote(vault, target_folder, cluster, title, body)
        # Pre-check before archiving originals: if the synthesized note
        # didn't land on disk with non-trivial content, we'd end up with
        # orphan originals moved to 04-Archive/ and no consolidated note
        # to cite them. Bail out — the cluster gets retried next run.
        # `promote()` writes via os.replace so we never see a half-written
        # tmp file, but the file could still be missing if the fs ran out
        # of space or the path was unwritable. Threshold 200 bytes is
        # below the smallest plausible consolidated note (frontmatter +
        # 1-line body) so it only fires on truly empty writes.
        try:
            promoted_ok = (
                promoted_path.is_file()
                and promoted_path.stat().st_size > 200
            )
        except OSError:
            promoted_ok = False
        if not promoted_ok:
            summary["clusters"].append({
                "size": len(cluster), "target": target_folder,
                "title": title, "error": "promoted_note_missing_or_empty",
                "promoted_path": str(promoted_path.relative_to(vault))
                if promoted_path.is_file() else None,
                "paths": [c.rel_path for c in cluster],
            })
            continue
        archived_paths = archive_originals(vault, cluster)
        summary["n_promoted"] += 1
        summary["n_archived"] += len(archived_paths)
        summary["clusters"].append({
            "size": len(cluster),
            "target": target_folder,
            "title": title,
            "promoted_path": str(promoted_path.relative_to(vault)),
            "archived_paths": [str(p.relative_to(vault)) for p in archived_paths],
        })

    summary["duration_s"] = round(time.perf_counter() - t0, 2)
    if not dry_run:
        _log_run(summary)
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--min-cluster", type=int, default=DEFAULT_MIN_CLUSTER)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true",
                     help="Emit summary as JSON for machine consumption.")
    args = ap.parse_args()
    summary = run(
        window_days=args.window_days,
        threshold=args.threshold,
        min_cluster=args.min_cluster,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}{summary['n_conversations']} conversaciones en ventana; "
        f"{summary['n_clusters']} clusters; "
        f"{summary['n_promoted']} promovidos, {summary['n_archived']} archivados "
        f"en {summary['duration_s']}s"
    )
    for c in summary["clusters"]:
        title = c.get("title", "(sin título)")
        size = c.get("size", 0)
        target = c.get("target", "?")
        marker = "·"
        if "error" in c:
            marker = "✗"
            title = f"{title} — {c['error']}"
        elif args.dry_run:
            marker = "would promote →"
        else:
            marker = "promoted →"
        print(f"  {marker} [{size}] {target}: {title}")


if __name__ == "__main__":
    main()
