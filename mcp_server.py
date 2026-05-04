"""MCP server exposing the Obsidian RAG as tools for Claude Code.

Runs over stdio. Registered in `~/.claude.json` or Claude Code settings.

Read-only tools (always safe):
  - rag_query:      retrieve top-k parent-expanded chunks for a question
  - rag_read_note:  read a full note from the vault by relative path
  - rag_list_notes: list notes filtered by folder and/or tag
  - rag_links:      find URLs by semantic context (no LLM paraphrasing)
  - rag_stats:      index metadata (chunk count, models, collection)
  - rag_followup:   list open loops (todos sin cerrar) in the vault

Write tools (create files / Apple DB rows — use when the user asks to
"capture/save/add/create" something, not for read queries):
  - rag_capture:          write a quick note to 00-Inbox/ (idempotent name)
  - rag_save_note:        write a note with explicit title/folder (generic)
  - rag_create_reminder:  add an Apple Reminder (auto-creates if date is clear)
  - rag_create_event:     add an Apple Calendar event (auto-creates if date clear)
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time

# Regex que matchea exactamente lo que el docstring de `rag_query` declara
# aceptar como `session_id`. Compilado a nivel de módulo así no pagamos el
# parse en cada tool call. Audit 2026-04-25 R2-2 #4: pre-fix no había
# validación, un cliente podía pasar "../../etc/passwd" o un string de 10k
# caracteres y disparábamos path-traversal o DoS al persistir la sesión.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")

# Silence sentence-transformers / HF output that would corrupt MCP stdio.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")


from mcp.server.fastmcp import FastMCP

mcp = FastMCP("obsidian-rag")


# Allowlist opcional de tools — controlada por `RAG_MCP_TOOLS` (CSV).
# Default (var ausente o vacía) = todas las tools registradas. Acepta el
# nombre completo (`rag_query`) o el sufijo corto (`query`). El harness
# (.devin/mcp-profiles/) inyecta este env var para reducir la superficie
# del MCP cuando el usuario activa un profile acotado. Diseñado para que
# nada cambie si el var no está seteado — backwards-compatible.
def _allowed_tools() -> set[str] | None:
    raw = os.environ.get("RAG_MCP_TOOLS", "").strip()
    if not raw:
        return None
    return {t.strip() for t in raw.split(",") if t.strip()}


def _terse_mode() -> bool:
    """`RAG_MCP_TOOLS_TERSE=1` recorta los docstrings al primer párrafo.

    Los docstrings completos van como descriptores al system prompt del
    cliente (Claude Code / Devin). Algunas tools tienen 30+ líneas
    explicando edge cases, validaciones y formato — útil para el
    desarrollador, ruido para el agente. En modo terse dejamos solo el
    primer párrafo (la línea sumario + descripción inmediata).
    """
    return os.environ.get("RAG_MCP_TOOLS_TERSE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _trim_docstring(fn):
    """Reescribe `fn.__doc__` dejando solo el primer párrafo.

    Mutación in-place: FastMCP lee `fn.__doc__` cuando registra la tool,
    así que basta con reasignar antes de pasarla a `mcp.tool()`. No
    cambia el comportamiento runtime — solo la metadata expuesta.
    """
    doc = fn.__doc__ or ""
    # Primer párrafo: hasta la primera línea en blanco. Conservamos los
    # \n internos del párrafo para que el sumario multi-línea se vea bien.
    paragraphs = doc.strip().split("\n\n", 1)
    if paragraphs:
        fn.__doc__ = paragraphs[0].strip()
    return fn


_ALLOWED_TOOLS = _allowed_tools()
_TERSE = _terse_mode()
_REGISTERED_TOOLS: list[str] = []
_SKIPPED_TOOLS: list[str] = []


def _maybe_tool(fn):
    """Wrapper sobre `mcp.tool()` que respeta `RAG_MCP_TOOLS` y `RAG_MCP_TOOLS_TERSE`.

    Si el env var `RAG_MCP_TOOLS` no está seteado o lista la tool (por
    nombre completo o por sufijo sin `rag_`), la registra normalmente.
    Caso contrario la deja como función Python sin exponer al protocolo MCP.

    Si `RAG_MCP_TOOLS_TERSE=1`, antes de registrar trim el docstring al
    primer párrafo — reduce ~50% los tokens del descriptor sin afectar
    funcionalidad.
    """
    name = fn.__name__
    short = name[4:] if name.startswith("rag_") else name
    if _ALLOWED_TOOLS is None or name in _ALLOWED_TOOLS or short in _ALLOWED_TOOLS:
        _REGISTERED_TOOLS.append(name)
        if _TERSE:
            fn = _trim_docstring(fn)
        return mcp.tool()(fn)
    _SKIPPED_TOOLS.append(name)
    return fn

# Lazy-import `rag` — Claude Code spawns one MCP server per session, and
# importing rag.py pulls in torch + sentence-transformers + sqlite-vec (~4 GB
# RSS per instance). Three idle sessions = 12 GB wasted and OOM thrash when
# command-r (19 GB) tries to load. Defer until a tool is actually called;
# idle MCPs then stay at ~50 MB.
_rag = None
_rag_lock = threading.Lock()

# Idle self-exit — when Claude Code sessions linger for hours without
# invoking a tool (sometimes days, if a terminal stays open), orphaned MCP
# children pile up holding VM. After idle threshold we `os._exit(0)` and
# Claude Code transparently respawns us on the next tool call (≤1s cost
# per respawn vs. ~4 GB RAM held hostage). Thresholds:
#   - rag not loaded:  2h (nobody ever used this session → keep alive longer
#                          in case user is about to start working)
#   - rag loaded:     30m (heavy libs resident → exit eagerly)
_last_call = time.time()
_IDLE_HOT_SECONDS = 30 * 60
_IDLE_COLD_SECONDS = 2 * 3600


def _touch() -> None:
    """Record tool activity. Called from every tool handler."""
    global _last_call
    _last_call = time.time()


def _idle_killer() -> None:
    while True:
        time.sleep(300)  # check every 5m
        idle = time.time() - _last_call
        threshold = _IDLE_HOT_SECONDS if _rag is not None else _IDLE_COLD_SECONDS
        if idle > threshold:
            os._exit(0)


threading.Thread(target=_idle_killer, daemon=True, name="mcp-idle-killer").start()


def _load_rag():
    global _rag
    if _rag is not None:
        return _rag
    with _rag_lock:
        if _rag is None:
            import rag as _r
            _rag = _r
    return _rag


@_maybe_tool
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
    _touch()
    # Audit 2026-04-25 R2-2 #4: validar el formato del session_id antes de
    # tocar el filesystem. El docstring declara `[A-Za-z0-9_.:-]{1,64}` pero
    # pre-fix no había chequeo, así que un cliente podía mandar:
    #   - "../../etc/passwd" → potencial path traversal en `ensure_session`
    #   - "x" * 10000        → DoS escribiendo archivos enormes de sesión
    #   - "<script>...</script>" → XSS si después se renderea en la web UI
    # Devolvemos lista vacía (mismo shape que "sin resultados") en vez de
    # raisear: el transport MCP renderiza excepciones como "Internal error"
    # sin contexto, así que el cliente no ganaría nada con un raise. Un
    # empty result le da la pista de que algo falló y puede reintentar con
    # un id válido.
    if session_id is not None and not _SESSION_ID_RE.match(session_id):
        return []
    rag = _load_rag()
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
        except Exception as e:
            # Audit 2026-04-25 R2-2 #3: pre-fix silenciaba todos los errores
            # acá sin log → si reformulate cuelga (ollama timeout, OOM,
            # cambio de schema en el JSON parser, etc.) el cliente MCP veía
            # resultados sin reformulación pero nadie tenía forma de
            # diagnosticar. Ahora loggeamos via _silent_log que escribe a
            # ~/.local/share/obsidian-rag/silent_errors.jsonl.
            try:
                rag._silent_log("mcp_reformulate_and_expand", e)
            except Exception:
                # Last-ditch — el log puede fallar si telemetry.db está
                # locked o el queue está lleno. No queremos que un fallo de
                # observabilidad rompa el query.
                pass
            effective_question = question
    elif history:
        try:
            effective_question = rag.reformulate_query(question, history)
        except Exception as e:
            # Audit 2026-04-25 R2-2 #3: idem caso anterior pero para el
            # path single-query (sin expansión a paráfrasis).
            try:
                rag._silent_log("mcp_reformulate_query", e)
            except Exception:
                pass
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


@_maybe_tool
def rag_read_note(path: str) -> str:
    """Read the full contents of a note from the vault.

    Args:
        path: Vault-relative path, e.g. "02-Areas/Coaching/Autoridad.md".
              Must end in .md and not escape the vault root.

    Returns the note content as a string. On any error, returns a string
    that starts with "Error: " followed by a human-readable reason — the
    MCP client (Claude Code / Devin / Cursor) sees this as the tool
    output and can detect the prefix to know it failed.

    Defensive catches added 2026-04-25 (audit R2-2 #2):
    - VAULT_PATH does not exist (iCloud not synced, dev box without vault)
    - ``.resolve()`` raises ``OSError`` (broken symlinks, permission denied)
    - ``read_text()`` raises ``OSError`` (file replaced mid-read, FS error)

    Pre-fix any of these caused an uncaught exception that bubbled up to
    the MCP transport as "Internal error" — the LLM saw no useful hint.
    """
    _touch()
    if not path.endswith(".md"):
        return "Error: path must end in .md"
    rag = _load_rag()
    if not rag.VAULT_PATH.exists():
        return f"Error: vault not found at {rag.VAULT_PATH}"
    try:
        full = (rag.VAULT_PATH / path).resolve()
        full.relative_to(rag.VAULT_PATH.resolve())
    except (ValueError, OSError) as e:
        return f"Error: path invalid or escapes the vault root ({e})"
    if not full.is_file():
        return f"Error: note not found at {path}"
    try:
        return full.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        return f"Error: failed to read {path} ({e})"


@_maybe_tool
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
    _touch()
    rag = _load_rag()
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


@_maybe_tool
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
    _touch()
    k = max(1, min(k, 30))
    rag = _load_rag()
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


@_maybe_tool
def rag_stats() -> dict:
    """Return indexing metadata: chunk count, models, collection name."""
    _touch()
    rag = _load_rag()
    col = rag.get_db()
    return {
        "chunks": col.count(),
        "collection": rag.COLLECTION_NAME,
        "embed_model": rag.EMBED_MODEL,
        "reranker": rag.RERANKER_MODEL,
        "vault_path": str(rag.VAULT_PATH),
    }


# ── Write-side tools (2026-04-22) ────────────────────────────────────────────
# These are distinct from the read tools above because they have side effects
# on the filesystem (capture / save_note) or Apple data stores (Reminders /
# Calendar). The idle-killer semantics still apply (30m hot, 2h cold) — a
# capture/save loads `rag` and auto-indexes the new note for retrievability.
#
# Design notes:
#   - No confirmation UI: the MCP host (Claude Code) is the one deciding to
#     call these, and it already carries the user's instruction. We don't
#     add a second prompt.
#   - `rag_capture` and `rag_save_note` are *not* equivalent: capture is
#     opinionated (always 00-Inbox, auto-stamped filename, frontmatter tags
#     include "capture"), save_note is flexible (user picks folder + title).
#   - Reminders + Calendar wrap `propose_reminder` / `propose_calendar_event`
#     which return a JSON string with either `created:true` or
#     `needs_clarification:true`. We return the parsed dict — the caller
#     decides what to do (ask the user for a clearer date, or report
#     success).


@_maybe_tool
def rag_capture(
    text: str,
    tags: list[str] | None = None,
    source: str | None = None,
    title: str | None = None,
) -> dict:
    """Write a quick note to the Obsidian vault's 00-Inbox/ folder.

    This is the MCP equivalent of `rag capture "text"` — use it when the
    user says "capturá esto", "anotá esto", "guardalo en el inbox", or
    otherwise wants an idea landed in the vault without fussing over
    folder/title. Filename is auto-stamped (YYYY-MM-DD-HHMM-<slug>.md)
    with collision suffixing (-2, -3, …).

    After writing, the note is immediately indexed so subsequent
    `rag_query` calls can retrieve it without waiting for `rag watch`.

    Args:
        text: Body of the note (non-empty). Markdown is preserved.
        tags: Optional tag list (no '#' prefix). "capture" is added
            automatically.
        source: Optional source tag (e.g. "voice", "chat", "mcp") — ends
            up in the frontmatter for later filtering.
        title: Optional explicit title. When None, the first non-empty
            line of the text is slugified.

    Returns:
        {"path": "00-Inbox/YYYY-MM-DD-HHMM-<slug>.md", "created": true}
        On empty text: {"created": false, "error": "<reason>"}.
    """
    _touch()
    rag = _load_rag()
    if not text or not text.strip():
        return {"created": False, "error": "empty text"}
    try:
        path = rag.capture_note(
            text, tags=list(tags) if tags else None,
            source=source, title=title,
        )
    except ValueError as exc:
        return {"created": False, "error": str(exc)}
    # Index immediately so the note is retrievable right away.
    try:
        rag._index_single_file(rag.get_db(), path, skip_contradict=True)
    except Exception:
        # Non-fatal — `rag watch` or next `rag index` will catch it.
        pass
    rel = path.relative_to(rag.VAULT_PATH)
    return {"path": str(rel), "created": True}


@_maybe_tool
def rag_save_note(
    text: str,
    title: str,
    folder: str = "00-Inbox",
    tags: list[str] | None = None,
) -> dict:
    """Write a note to an explicit vault folder with an explicit title.

    Use this when the user names the destination (e.g. "guardalo en
    02-Areas/Salud" or "creá una nota llamada 'Postura al andar'"). For
    unopinionated quick captures, prefer `rag_capture`.

    The folder is created if it doesn't exist. The filename is <title>.md
    with collision suffixing (-2, -3, …) on the basename. Frontmatter:
    `created`, `type: note`, and the tag list (no "capture" default).

    Args:
        text: Body of the note.
        title: Explicit title — used verbatim as H1 and slugified for
            filename. Must be non-empty.
        folder: Vault-relative folder (e.g. "02-Areas/Salud",
            "03-Resources/Ideas"). Must not escape the vault root.
            Defaults to "00-Inbox" for parity with `rag_capture`.
        tags: Optional tag list (no '#' prefix), no defaults added.

    Returns:
        {"path": "<folder>/<slug>.md", "created": true}
        On invalid input: {"created": false, "error": "<reason>"}.
    """
    _touch()
    rag = _load_rag()
    if not text or not text.strip():
        return {"created": False, "error": "empty text"}
    if not title or not title.strip():
        return {"created": False, "error": "empty title"}
    # Folder sanity — reject absolute paths + traversal before we
    # normalize. Stripping slashes first would silently accept
    # "/etc/passwd" as "etc/passwd", which the VAULT_PATH / resolve
    # guard below usually catches but we want early-reject UX.
    folder_raw = (folder or "00-Inbox").strip()
    if folder_raw.startswith("/") or folder_raw.startswith("~"):
        return {"created": False, "error": "invalid folder (absolute path)"}
    folder = folder_raw.strip("/")
    if not folder or ".." in folder.split("/"):
        return {"created": False, "error": "invalid folder"}

    target_dir = rag.VAULT_PATH / folder
    try:
        target_dir.resolve().relative_to(rag.VAULT_PATH.resolve())
    except ValueError:
        return {"created": False, "error": "folder escapes vault root"}
    target_dir.mkdir(parents=True, exist_ok=True)

    slug = rag._slug(title, maxlen=80)
    candidate = target_dir / f"{slug}.md"
    i = 2
    while candidate.exists():
        candidate = target_dir / f"{slug}-{i}.md"
        i += 1

    from datetime import datetime
    now = datetime.now()
    fm_lines = [
        "---",
        f"created: '{now.isoformat(timespec='seconds')}'",
        "type: note",
    ]
    if tags:
        fm_lines.append("tags:")
        for t in tags:
            t = str(t).lstrip("#").strip()
            if t:
                fm_lines.append(f"  - {t}")
    fm_lines.append("---")
    body = f"# {title.strip()}\n\n{text.strip()}\n"
    candidate.write_text("\n".join(fm_lines) + "\n" + body, encoding="utf-8")

    # Index immediately.
    try:
        rag._index_single_file(rag.get_db(), candidate, skip_contradict=True)
    except Exception:
        pass

    rel = candidate.relative_to(rag.VAULT_PATH)
    return {"path": str(rel), "created": True}


@_maybe_tool
def rag_create_reminder(
    title: str,
    when: str = "",
    reminder_list: str | None = None,
    priority: int | None = None,
    notes: str | None = None,
    recurrence: str | None = None,
) -> dict:
    """Create an Apple Reminder. Requires macOS + Reminders.app access.

    Use when the user says "recordame X", "acordame de Y mañana", "ponete
    una alarma", etc. If `when` parses unambiguously (dateparser on a
    Spanish/English natural phrase), the reminder is created directly;
    otherwise the tool returns `needs_clarification:true` so the caller
    can ask the user for a more specific date.

    Args:
        title: Reminder text (what you need to do).
        when: Natural-language date/time ("mañana 10am", "jueves 4pm",
            "en 2 horas"). Leave empty for a reminder without due date.
        reminder_list: Target list name. None = Reminders.app default.
        priority: 1 (high), 5 (medium), 9 (low). None = no priority.
        notes: Extra text for the body.
        recurrence: Natural-language recurrence ("todos los lunes",
            "cada 2 semanas"). None = one-shot.

    Returns:
        Parsed JSON from propose_reminder. Shape depends on outcome:
          - Created: {"kind":"reminder", "created":true, "reminder_id":"...",
                      "fields":{...}}
          - Ambiguous date: {"kind":"reminder", "needs_clarification":true,
                             "proposal_id":"prop-uuid", "fields":{...}}
          - Failure: {"kind":"reminder", "created":false, "error":"...",
                      "fields":{...}}
    """
    _touch()
    rag = _load_rag()
    import json as _json
    raw = rag.propose_reminder(
        title=title,
        when=when,
        list=reminder_list,
        priority=priority,
        notes=notes,
        recurrence_text=recurrence,
    )
    try:
        return _json.loads(raw)
    except Exception as exc:
        return {"kind": "reminder", "created": False,
                "error": f"json decode failed: {exc}",
                "raw": raw}


@_maybe_tool
def rag_create_event(
    title: str,
    start: str,
    end: str | None = None,
    calendar: str | None = None,
    location: str | None = None,
    notes: str | None = None,
    all_day: bool = False,
    recurrence: str | None = None,
) -> dict:
    """Create an Apple Calendar event. Requires macOS + Calendar.app access.

    Use when the user says "agendá X el jueves", "cumple de Y el 15",
    "reunión mañana 10am", etc. Auto-flips to all-day when `start` has
    no time marker (e.g. "el miércoles", "cumpleaños de X el viernes").
    Dedups against existing events on the same day/time window, so
    repeated calls don't litter the calendar.

    Args:
        title: Event summary.
        start: Natural-language date/time ("jueves 4pm" → timed, "el
            jueves" → all-day). Required; empty string returns
            needs_clarification.
        end: End date/time. None + all_day → start + 1 day.
            None + timed → start + 1 hour.
        calendar: Target calendar name. None = first writable (typically
            iCloud).
        location: Free-text location.
        notes: Event description / notes.
        all_day: Force all-day. Auto-flips True when `start` has no
            time marker; pass False explicitly to override.
        recurrence: Natural-language recurrence.

    Returns:
        Parsed JSON from propose_calendar_event. Same shape as
        rag_create_reminder (created / needs_clarification / error).
    """
    _touch()
    rag = _load_rag()
    import json as _json
    raw = rag.propose_calendar_event(
        title=title,
        start=start,
        end=end,
        calendar=calendar,
        location=location,
        notes=notes,
        all_day=all_day,
        recurrence_text=recurrence,
    )
    try:
        return _json.loads(raw)
    except Exception as exc:
        return {"kind": "event", "created": False,
                "error": f"json decode failed: {exc}",
                "raw": raw}


@_maybe_tool
def rag_followup(
    days: int = 30,
    status: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """List open loops in the vault — things you said you'd do that aren't
    closed.

    Scans frontmatter todos, unchecked `- [ ]` checkboxes, and imperative
    clauses ("tengo que X", "pendiente Y", "revisar Z") in recently
    modified notes. Each loop is classified as resolved / stale / activo
    based on whether there's semantic evidence of follow-through in
    later notes.

    Use when the user asks "qué tengo pendiente?", "loops abiertos",
    "qué quedó sin cerrar?".

    Args:
        days: Lookback window in days (default 30).
        status: Filter by status — 'stale', 'activo', 'resolved', or None
            for all.
        limit: Max items to return (default 50).

    Returns:
        List of {"status", "age_days", "kind", "source_note", "loop_text",
        ...} dicts. Empty list if no loops found.
    """
    _touch()
    rag = _load_rag()
    col = rag.get_db()
    items = rag.find_followup_loops(col, rag.VAULT_PATH, days=days)
    if status:
        items = [it for it in items if it.get("status") == status]
    return items[: max(1, int(limit))]


def main() -> None:
    # Log al stderr qué tools quedaron registradas. Útil para debug del
    # harness: si `RAG_MCP_TOOLS` está mal escrito todas las tools quedan
    # fuera y el MCP arranca vacío. El log queda en el stderr del MCP, que
    # Claude Code y Devin redirigen a sus logs de sesión.
    if _ALLOWED_TOOLS is not None:
        sys.stderr.write(
            f"[obsidian-rag-mcp] RAG_MCP_TOOLS allowlist activa "
            f"(registered={_REGISTERED_TOOLS} skipped={_SKIPPED_TOOLS})\n"
        )
        if not _REGISTERED_TOOLS:
            sys.stderr.write(
                "[obsidian-rag-mcp] WARN: el allowlist no matcheó ninguna tool. "
                "Revisar valores en RAG_MCP_TOOLS.\n"
            )
    mcp.run()


if __name__ == "__main__":
    main()
