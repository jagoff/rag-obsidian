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


from mcp.server.fastmcp import FastMCP

import rag

mcp = FastMCP("obsidian-rag")

# MCP server es persistent pero larga vida por conexión — warmup async paga el
# reranker cold load mientras el cliente MCP hace handshake, y deja bge-m3
# caliente para el primer rag_query.
rag.warmup_async()


@mcp.tool()
def rag_query(
    question: str,
    k: int = 5,
    folder: str | None = None,
    tag: str | None = None,
    multi_query: bool = True,
    session_id: str | None = None,
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
        session_id: Optional persistent conversation id. When set, prior turns
            on the same id are used to reformulate follow-ups (so "profundizá"
            or pronoun-laden fragments become standalone queries), and this
            turn is appended to the session history. Accepts any short
            identifier matching [A-Za-z0-9_.:-]{1,64} (e.g. "tg:123", "mcp-x").
    """
    col = rag.get_db()
    if col.count() == 0:
        return []
    k = max(1, min(k, 15))

    sess = rag.ensure_session(session_id, mode="mcp") if session_id else None
    history = rag.session_history(sess) if sess else None
    effective_question = question
    pre_variants: list[str] | None = None
    if history and multi_query:
        try:
            effective_question, pre_variants = rag.reformulate_and_expand(
                question, history
            )
        except Exception:
            effective_question = question
    elif history:
        try:
            effective_question = rag.reformulate_query(question, history)
        except Exception:
            effective_question = question

    result = rag.retrieve(
        col, effective_question, k, folder,
        tag=tag, precise=False, multi_query=multi_query, auto_filter=True,
        variants=pre_variants,
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

    if sess is not None:
        # MCP side: we don't have the final Claude answer (Claude is what's
        # calling us). Persist the user turn + retrieved paths; Claude's reply
        # is outside our visibility. Follow-up turns can still reformulate
        # against the question history even without the answers.
        rag.append_turn(sess, {
            "q": question,
            "q_reformulated": effective_question if effective_question != question else None,
            "a": None,
            "paths": [m.get("file", "") for m in result["metas"]],
            "top_score": round(float(result["confidence"]), 3) if result.get("confidence") is not None else None,
        })
        rag.save_session(sess)

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
def rag_links(
    query: str,
    k: int = 5,
    folder: str | None = None,
    tag: str | None = None,
) -> list[dict]:
    """Find URLs in the vault by semantic context match.

    Designed for "where is the link to X" type questions. Returns the literal
    URLs along with their source note, anchor text (when written as
    `[anchor](url)`), line number, and the surrounding prose used for ranking.
    Bypasses the chat LLM — no paraphrasing of URLs.

    Args:
        query: Natural-language description of what the link is about.
        k: Number of URLs to return (default 5, max 30).
        folder: Optional folder filter, e.g. "03-Resources".
        tag: Optional tag filter (no '#' prefix).
    """
    k = max(1, min(k, 30))
    items = rag.find_urls(query, k=k, folder=folder, tag=tag)
    return [
        {
            "url": it["url"],
            "anchor": it.get("anchor", ""),
            "path": it["path"],
            "note": it.get("note", ""),
            "line": it.get("line", 0),
            "context": it.get("context", ""),
            "score": round(float(it.get("score", 0.0)), 3),
        }
        for it in items
    ]


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
