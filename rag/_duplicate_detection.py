"""Duplicate detection — pairwise cosine over per-note centroid embeddings.

Phase 5 cont de modularización (audit perf 2026-05-08, ROI 520).

## Cómo funciona

Pairwise cosine over per-note centroid embeddings (mean of the note's
chunks in the main collection). Numpy para el O(N²/2) sweep — ~500
notes finishes in well under a second. A "centroid" es un coarse
fingerprint, intentional: surfaceamos notas que broadly cubren el
mismo topic, no solo que comparten una phrase.

## API

- `_note_centroids(col, folder, skip_folders, vault_only)` →
  (files, metas, centroids_matrix). Used by both find_duplicate_notes
  AND find_near_duplicates_for AND surface (via lazy import).
- `find_duplicate_notes(col, threshold=0.85)` → list of pair dicts.
- `find_near_duplicates_for(col, note_path, threshold=0.80)` →
  list of similar notes for inbox triage.

## Lazy imports

`normalize_source`, `VAULT_PATH` viven en `rag/__init__.py`. Lazy
adentro de cada función. `numpy` es 3rd party (top-level OK).

## Re-export

`rag/__init__.py` hace `from rag._duplicate_detection import *`.
Preserva 100% compat con `rag.find_duplicate_notes`,
`rag.find_near_duplicates_for`, `rag._note_centroids`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from rag import SqliteVecCollection

__all__ = [
    "_note_centroids",
    "find_duplicate_notes",
    "find_near_duplicates_for",
]


def _note_centroids(
    col: "SqliteVecCollection",
    folder: str | None = None,
    skip_folders: tuple[str, ...] = (),
    *,
    vault_only: bool = True,
) -> tuple[list[str], list[dict], "np.ndarray"]:
    """Group all chunks by file, average the embeddings, L2-normalise.

    Returns (file_paths, first_meta_per_file, centroids_matrix). Files with
    zero chunks (shouldn't happen) are skipped silently. `skip_folders` is
    a tuple of path prefixes to exclude (e.g. `("04-Archive/",)` so dupes
    don't pair a live note with its archived copy).

    `vault_only` (default True) skips cross-source chunks whose ``source``
    metadata is anything other than ``"vault"`` (calendar, reminders,
    whatsapp, gmail, contacts, calls, safari, messages …).  Those chunks
    share the same collection but are not user vault notes, so they produce
    spurious cosine-1.000 pairs for recurring calendar events and similar
    structural duplicates that the user never sees in Obsidian.
    """
    import numpy as np  # noqa: PLC0415

    from rag import normalize_source  # noqa: PLC0415

    data = col.get(include=["embeddings", "metadatas"])
    by_file: dict[str, dict] = {}
    for emb, meta in zip(data["embeddings"], data["metadatas"]):
        f = meta.get("file", "")
        if not f:
            continue
        if vault_only and normalize_source(meta.get("source")) != "vault":
            continue
        if folder and not (f == folder or f.startswith(folder.rstrip("/") + "/")):
            continue
        if any(f == p.rstrip("/") or f.startswith(p) for p in skip_folders):
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
    col: "SqliteVecCollection",
    threshold: float = 0.85,
    folder: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Pairs of notes with centroid cosine ≥ threshold.

    Returns descending-by-similarity list of
    {a_path, a_note, b_path, b_note, similarity, snippet_a, snippet_b}.
    The snippets are the first ~200 chars of each note's body for at-a-glance
    comparison in the renderer. `04-Archive/` is excluded by default because
    archived notes keep their content and surface as near-duplicates of their
    pre-archive originals — noise on every run otherwise.
    """
    import numpy as np  # noqa: PLC0415

    from rag import VAULT_PATH  # noqa: PLC0415

    files, metas, arr = _note_centroids(col, folder, skip_folders=("04-Archive/",))
    if len(files) < 2:
        return []
    sims = arr @ arr.T
    # Ignore self and lower triangle: only count each pair once.
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


def find_near_duplicates_for(
    col: "SqliteVecCollection",
    note_path: str,
    threshold: float = 0.80,
    limit: int = 5,
) -> list[dict]:
    """Notes whose centroid is most similar to `note_path`'s centroid.

    Used by inbox triage to flag "this incoming note may already exist".
    Lower default threshold than `find_duplicate_notes` so partial overlaps
    surface as warnings rather than be silently missed.
    """
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
