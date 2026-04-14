"""MCP server exposing the Obsidian RAG as tools for Claude Code.

Runs over stdio. Registered in `~/.claude.json` or Claude Code settings.
Tools:
  - rag_query:      retrieve top-k parent-expanded chunks for a question
  - rag_read_note:  read a full note from the vault by relative path
  - rag_list_notes: list notes filtered by folder and/or tag
  - rag_stats:      index metadata (chunk count, models, collection)
"""

from __future__ import annotations

import os

# Silence sentence-transformers / HF output that would corrupt MCP stdio.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

from pathlib import Path

from mcp.server.fastmcp import FastMCP

import rag

mcp = FastMCP("obsidian-rag")


@mcp.tool()
def rag_query(
    question: str,
    k: int = 5,
    folder: str | None = None,
    tag: str | None = None,
    multi_query: bool = True,
) -> list[dict]:
    """Retrieve the most relevant chunks from the Obsidian vault.

    Returns parent-expanded chunks (the surrounding Markdown section), each
    annotated with its source note, relative path, and cross-encoder score.
    The LLM (Claude) should read `content` and cite `path` in its answer.

    Args:
        question: Natural-language query, Spanish or English.
        k: Number of chunks to return (default 5, max 15).
        folder: Optional folder filter, e.g. "02-Areas/Coaching".
        tag: Optional tag filter (no '#' prefix), e.g. "coaching".
        multi_query: Expand query into 3 paraphrases for better recall.
    """
    col = rag.get_db()
    if col.count() == 0:
        return []
    k = max(1, min(k, 15))
    result = rag.retrieve(
        col, question, k, folder,
        tag=tag, precise=False, multi_query=multi_query, auto_filter=True,
    )
    out = []
    for doc, meta, score in zip(result["docs"], result["metas"], result["scores"]):
        out.append({
            "note": meta.get("note", ""),
            "path": meta.get("file", ""),
            "folder": meta.get("folder", ""),
            "tags": meta.get("tags", ""),
            "score": round(float(score), 2),
            "content": doc,
        })
    return out


@mcp.tool()
def rag_read_note(path: str) -> str:
    """Read the full contents of a note from the vault.

    Args:
        path: Vault-relative path, e.g. "02-Areas/Coaching/Autoridad.md".
              Must end in .md and not escape the vault root.
    """
    if not path.endswith(".md"):
        return "Error: path must end in .md"
    full = (rag.VAULT_PATH / path).resolve()
    try:
        full.relative_to(rag.VAULT_PATH.resolve())
    except ValueError:
        return "Error: path escapes the vault root"
    if not full.is_file():
        return f"Error: note not found at {path}"
    return full.read_text(encoding="utf-8", errors="ignore")


@mcp.tool()
def rag_list_notes(
    folder: str | None = None,
    tag: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """List notes in the index, optionally filtered by folder and/or tag.

    Useful for aggregate queries ("what notes do I have about X?") where
    retrieval-by-relevance is the wrong tool.

    Args:
        folder: Only include notes under this folder.
        tag: Only include notes carrying this tag.
        limit: Max number of unique notes to return (default 100).
    """
    col = rag.get_db()
    c = rag._load_corpus(col)
    seen: dict[str, dict] = {}
    for m in c["metas"]:
        file_ = m.get("file", "")
        if file_ in seen:
            continue
        if folder and folder not in file_:
            continue
        tags_str = m.get("tags", "")
        if tag and tag not in [t.strip() for t in tags_str.split(",") if t.strip()]:
            continue
        seen[file_] = {
            "note": m.get("note", ""),
            "path": file_,
            "folder": m.get("folder", ""),
            "tags": tags_str,
        }
        if len(seen) >= limit:
            break
    return list(seen.values())


@mcp.tool()
def rag_stats() -> dict:
    """Return indexing metadata: chunk count, models, collection name."""
    col = rag.get_db()
    return {
        "chunks": col.count(),
        "collection": rag.COLLECTION_NAME,
        "embed_model": rag.EMBED_MODEL,
        "reranker": rag.RERANKER_MODEL,
        "vault_path": str(rag.VAULT_PATH),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
