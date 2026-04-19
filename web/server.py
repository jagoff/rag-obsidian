"""Web UI mínima para `rag chat`.

Espeja el pipeline del CLI (multi_retrieve → command-r streaming → sources)
sobre HTTP + SSE. Sin build step en el frontend; vanilla JS contra este
endpoint. Sesiones persistidas en el mismo store que el CLI — el session_id
de la web es `web:<uuid>` así no colisiona con `tg:<chat_id>` ni con los
ids del chat interactivo.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# rag.py vive en el root del proyecto; lo importamos como módulo.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ollama  # noqa: E402

from rag import (  # noqa: E402
    CHAT_OPTIONS,
    CONTRADICTION_LOG_PATH,
    EVAL_LOG_PATH,
    LOG_PATH,
    _LOG_QUEUE,
    RAG_STATE_SQL,
    _log_sql_state_error,
    _map_cpu_row,
    _map_memory_row,
    _ragvec_state_conn,
    _sql_append_event,
    MORNING_FOLDER,
    OLLAMA_KEEP_ALIVE,
    SESSION_HISTORY_WINDOW,
    SYSTEM_RULES,
    SYSTEM_RULES_WEB,
    _collect_screentime,
    _fmt_hm,
    VAULT_PATH,
    WHATSAPP_BOT_JID,
    WHATSAPP_DB_PATH,
    _apple_enabled,
    _build_tasks_system_rules as _rag_build_tasks_system_rules,
    _collect_scoped_tasks_evidence_multi as _rag_collect_scoped_tasks_evidence_multi,
    _collect_today_evidence,
    _fetch_calendar_ahead,
    _fetch_chrome_bookmarks_used,
    _fetch_drive_evidence,
    _fetch_reminders_due,
    _fetch_vault_activity,
    _fetch_weather_forecast,
    _fetch_whatsapp_unread,
    _format_scoped_tasks_context as _rag_format_scoped_tasks_context,
    _generate_today_narrative,
    _icalbuddy_path,
    _load_corpus,
    _load_vaults_config,
    _path_to_title,
    clean_md,
    is_excluded,
    parse_frontmatter,
    _pendientes_collect,
    _pendientes_urgent,
    _render_today_prompt,
    _tasks_services_consulted as _rag_tasks_services_consulted,
    append_turn,
    ensure_session,
    find_followup_loops,
    get_db,
    BEHAVIOR_LOG_PATH,
    get_pagerank,
    log_behavior_event,
    log_query_event,
    multi_retrieve,
    new_turn_id,
    record_feedback,
    reformulate_query,
    resolve_chat_model,
    resolve_vault_paths,
    save_session,
    session_history,
)

from web.conversation_writer import write_turn, TurnData  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="obsidian-rag web", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Chat model for /chat. Delegated to `resolve_chat_model()` so it tracks
# whatever rag.py decides is best on this host (command-r > qwen2.5:14b >
# phi4). command-r:35b prefill is slower (~5-10s on a 5k-char context)
# but is the only model in the local set that reliably:
#   - stays in Spanish instead of drifting to Portuguese / Italian
#   - resolves pronouns from conversation history ("ella", "eso") against
#     prior turns instead of treating follow-ups as standalone
#   - follows the "no inline citations / no refusal" rules of this prompt
# qwen2.5:3b was used here for a while as a speed optimization (~600ms
# prefill, ~10× faster) but failed those three constraints consistently.
# Set OBSIDIAN_RAG_WEB_CHAT_MODEL to override if you need speed over quality.
WEB_CHAT_MODEL = os.environ.get("OBSIDIAN_RAG_WEB_CHAT_MODEL") or None


def _resolve_web_chat_model() -> str:
    """Return the env-override model if set, else whatever rag.py picks."""
    return WEB_CHAT_MODEL or resolve_chat_model()


# Module-level system prompt. Kept BYTE-IDENTICAL across all /api/chat
# requests so ollama's prefix cache hits — when the system message tokens
# match a prior request, llama.cpp skips KV recomputation for those ~900
# tokens. Cached prefill is ~50× faster (0.23ms/tok vs 12.7ms/tok measured
# on command-r:35b), saving up to 10s on the first user-facing token.
# Any per-request variation (context, signals, history) goes in the user
# message AFTER this static block.
_WEB_SYSTEM_PROMPT = (
    "Eres un asistente de consulta sobre las notas personales de "
    "Obsidian del usuario. NO sos un modelo de conocimiento general.\n\n"
    "REGLA 0 — IDIOMA: respondé SIEMPRE en español rioplatense. "
    "TOTALMENTE PROHIBIDO emitir tokens en chino, japonés, coreano, "
    "árabe, ruso, alemán, portugués, italiano, francés o cualquier "
    "idioma que no sea español. Si el contexto recuperado contiene "
    "fragmentos en otros idiomas (ej. citas en inglés, nombres "
    "propios, código), citalos textualmente entre comillas pero el "
    "resto de tu respuesta TIENE QUE estar en español. Si la "
    "pregunta del usuario está en otro idioma, traducila a español "
    "mentalmente y respondé en español. Esta regla es ABSOLUTA — "
    "ni siquiera caracteres sueltos en otro alfabeto (汉字, "
    "русский, etc.) están permitidos en tu output.\n\n"
    "REGLA 1 — ENGÁNCHATE CON EL CONTEXTO QUE RECIBÍS:\n"
    "  • El CONTEXTO abajo es lo que el retriever consideró más "
    "relacionado con la pregunta. Tu trabajo es leerlo y resumir lo "
    "que aporta, siempre. Incluso si los chunks son breves, parciales "
    "o tangenciales — engancháte con ellos.\n"
    "  • Preguntas tipo '¿tengo algo sobre X?' / '¿qué hay de X?' / "
    "'¿tenés notas para X?' se responden afirmativamente apenas el "
    "CONTEXTO mencione X en título o cuerpo. Listá brevemente lo que "
    "hay.\n"
    "  • Si el CONTEXTO es genuinamente pobre o tangencial para lo "
    "que se pregunta, describí lo que sí aparece y aclaralo en prosa "
    "('las notas que encontré mencionan X pero no detallan Y'). "
    "NUNCA uses fórmulas de refusal del tipo 'no tengo información' "
    "— esa vía de escape está PROHIBIDA en este asistente, siempre "
    "hay que devolver el mejor resumen posible del CONTEXTO.\n"
    "  • Fuera del CONTEXTO no inventes (eso queda para REGLA 3 con "
    "su marcador `<<ext>>`).\n\n"
    "REGLA 2 — NO CITAR NOTAS INLINE: la UI ya muestra la lista de "
    "fuentes recuperadas debajo de cada respuesta (nota, score, "
    "ruta). Por lo tanto TOTALMENTE PROHIBIDO escribir:\n"
    "  • markdown links tipo `[Título](ruta.md)`\n"
    "  • nombres de archivo con extensión (`algo.md`)\n"
    "  • rutas con carpetas PARA (`03-Resources/…`, `02-Areas/…`, etc.)\n"
    "  • el nombre completo de la nota como título dentro de la prosa\n"
    "Podés referirte implícitamente: 'según tus notas', 'en tu nota "
    "sobre X', 'tenés una nota al respecto'. Todo lo demás se ve en "
    "la lista de fuentes — no la dupliques en el cuerpo.\n\n"
    "REGLA 3 — MARCAR EXTERNO: si agregás texto que NO sale "
    "textualmente del contexto (intros, parafraseos, conectores, "
    "opinión, conocimiento general), envolvelo en `<<ext>>...<</ext>>`. "
    "Fuera de esos marcadores TODO debe ser verificable palabra por "
    "palabra en el contexto.\n\n"
    "REGLA 4 — FORMATO: 2-4 oraciones o lista corta de viñetas. Dato "
    "clave en la primera oración; contexto mínimo (qué hace, cómo se "
    "invoca) después. Si la pregunta apunta a un comando, "
    "herramienta, parámetro o valor puntual Y el CONTEXTO incluye "
    "su uso (firma, parámetros, ejemplo, en qué MCP/server vive), "
    "ese uso es OBLIGATORIO en la respuesta — no te quedes en un "
    "token mínimo cuando el chunk lo explica.\n\n"
    "REGLA 4.5 — PRESERVAR LINKS DEL CONTENIDO: si el CONTEXTO "
    "contiene URLs (http://, https://) o wikilinks ([[Nota]]) que "
    "viven DENTRO del cuerpo de una nota (no son la ruta del chunk), "
    "copialos LITERAL en la respuesta — son clickeables y útiles para "
    "el usuario. Aclaración respecto a REGLA 2: REGLA 2 prohíbe "
    "citar las notas-fuente (la lista de abajo ya las muestra). Los "
    "links/URLs que aparecen como contenido de una nota son data, "
    "no citas; tienen que aparecer en la respuesta. Ejemplo: si un "
    "chunk dice 'tutorial: https://x.com/y', escribí 'tutorial: "
    "https://x.com/y', no 'tutorial disponible'.\n\n"
    "REGLA 4.6 — LINK A DOCS OFICIALES (opcional, condicionado): si el "
    "CONTEXTO sobre una herramienta/producto se queda corto Y la "
    "pregunta se beneficiaría de docs oficiales (ej: 'cómo configuro X', "
    "'qué features tiene Y'), podés ofrecer UN link a la documentación "
    "oficial al final, envuelto en `<<ext>>...<</ext>>`. Restricciones "
    "duras:\n"
    "  • Sólo el dominio raíz canónico (ej: https://omnifocus.com, "
    "https://obsidian.md). NUNCA inventes paths profundos "
    "(/docs/v3/foo) — si no estás 100% seguro de la URL exacta, usá "
    "el root.\n"
    "  • Una sola línea, formato: `<<ext>>Más info: <URL></ext>>`.\n"
    "  • NO ofrecer link si el CONTEXTO ya cubre la pregunta, ni para "
    "consultas sobre las notas mismas (vault, tags, búsqueda).\n\n"
    "REGLA 5 — SEGUÍ EL HILO: esta es una conversación, no preguntas "
    "sueltas. Los mensajes previos (user/assistant de arriba) son "
    "contexto vivo. Si la pregunta nueva usa pronombres ('ella', "
    "'eso', 'ahí'), referencias elípticas ('y de X?', 'profundizá'), "
    "o asume un tema del turn anterior, resolvé la referencia usando "
    "el historial y respondé sobre ESE tema. No trates la pregunta "
    "como si empezara de cero.\n"
)


@app.on_event("startup")
def _warmup() -> None:
    """Hydrate the home cache and kick the bg prewarmer; DO NOT pin the
    chat model or reranker on boot.

    The previous version called `ollama.chat(model=command-r, ...)` on
    startup which pinned ~19 GB of unified memory even when the user was
    only browsing /dashboard or /. On a 36 GB Mac that's enough wired
    memory to starve the kernel and cause host freezes. Warmup of the
    expensive bits (reranker MPS init, chat model load) now happens
    on-demand from the first /api/chat request — 2-3s extra cost on one
    query in exchange for not holding the machine hostage when idle.

    Corpus/BM25/PageRank are cheap (RAM only, no VRAM), so we still
    preload them so the first retrieve is fast.
    """
    import threading

    _load_home_cache()
    _ensure_home_prewarmer()
    _ensure_chat_model_prewarmer()

    def _do_warmup() -> None:
        try:
            from rag import get_db_for
            for _name, path in resolve_vault_paths(None):
                try:
                    col = get_db_for(path)
                    if col.count():
                        _load_corpus(col)
                        get_pagerank(col)
                except Exception:
                    pass
                break
        except Exception:
            pass

    threading.Thread(target=_do_warmup, daemon=True).start()

    def _idle_sweeper() -> None:
        """Evict the reranker from MPS after `_RERANKER_IDLE_TTL` of no
        activity. Keeps the Mac responsive when the user walks away from
        chat — 2-3 GB of unified memory freed on idle.

        After eviction, schedules a deferred pre-warm 60s later so the
        reranker is hot again before the likely next request (user returns
        to chat after a short break). The pre-warm runs in its own daemon
        thread and MUST NOT block the sweeper loop.

        When `RAG_RERANKER_NEVER_UNLOAD=1` (env var) the sweeper loop
        still runs but skips `maybe_unload_reranker()` entirely — the
        reranker stays pinned in MPS VRAM at all times. Use this on a
        36 GB unified-memory Mac where the ~2-3 GB cost is acceptable and
        eliminating the 9s cold-reload hit after idle eviction is worth it.
        """
        from rag import maybe_unload_reranker, get_reranker
        _never_unload = os.environ.get("RAG_RERANKER_NEVER_UNLOAD", "").strip() not in ("", "0", "false", "no")
        if _never_unload:
            print("[idle-sweep] RAG_RERANKER_NEVER_UNLOAD=1 — reranker pinned, sweeper disabled", flush=True)
        while True:
            try:
                time.sleep(120)
                if _never_unload:
                    continue
                if maybe_unload_reranker():
                    print("[idle-sweep] reranker unloaded from MPS", flush=True)

                    def _deferred_rewarm() -> None:
                        time.sleep(60)
                        try:
                            get_reranker()
                            print("[idle-sweep] reranker pre-warmed (deferred)", flush=True)
                        except Exception:
                            pass

                    threading.Thread(
                        target=_deferred_rewarm, name="reranker-rewarm", daemon=True
                    ).start()
            except Exception:
                pass

    threading.Thread(target=_idle_sweeper, name="idle-sweeper", daemon=True).start()


@app.get("/")
def home_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "home.html")


@app.get("/chat")
def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    # None → vault activo; "all" → todos los registrados; "name" → ese puntual.
    vault_scope: str | None = None


class FeedbackRequest(BaseModel):
    turn_id: str
    rating: int                   # +1 thumbs up / -1 thumbs down
    q: str | None = None
    paths: list[str] | None = None
    session_id: str | None = None
    reason: str | None = None     # short free-text, optional (≤200 chars after trim)


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest) -> dict:
    """Record a +1/-1 rating for a chat turn into feedback.jsonl.

    The rating signal feeds ranker tuning (`rag tune`) and the golden
    cache that biases future retrievals. Client sends the turn_id from
    the `done` SSE event plus the question and the source paths it saw,
    so we don't need to re-scan queries.jsonl on every click.

    Optional `reason` lets the user jot a short note when clicking 👎
    ("genérico", "faltó la nota X", "mezcla temas") — grouped later by
    `rag insights` to surface common failure modes.
    """
    if not req.turn_id or req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="turn_id + rating ±1 requeridos")
    reason = (req.reason or "").strip()[:200] or None
    record_feedback(
        turn_id=req.turn_id,
        rating=req.rating,
        q=(req.q or "").strip(),
        paths=req.paths or [],
        reason=reason,
        session_id=req.session_id,
    )
    return {"ok": True}


# ── /api/behavior ─────────────────────────────────────────────────────────────
# In-memory token bucket for rate limiting: 120 events/min per IP.
# Stored as {ip: [timestamp, ...]} with a 60s sliding window. No new deps.
import collections as _collections

_BEHAVIOR_BUCKETS: dict[str, list[float]] = _collections.defaultdict(list)
_BEHAVIOR_RATE_LIMIT = 120
_BEHAVIOR_RATE_WINDOW = 60.0  # seconds

_BEHAVIOR_KNOWN_EVENTS = frozenset({
    "open", "open_external", "positive_implicit",
    "negative_implicit", "kept", "deleted", "save",
})
_BEHAVIOR_SESSION_RE = re.compile(r"^[A-Za-z0-9_.@:-]{1,80}$")


class BehaviorRequest(BaseModel):
    source: str
    event: str
    query: str | None = None
    path: str | None = None
    rank: int | None = None
    dwell_ms: int | None = None
    session: str | None = None


@app.post("/api/behavior")
def submit_behavior(req: BehaviorRequest, request: Request) -> dict:
    """Record a user-behavior event from the web dashboard into behavior.jsonl."""
    # Validate source
    if req.source != "web":
        raise HTTPException(status_code=400, detail="source must be 'web'")

    # Validate event
    if req.event not in _BEHAVIOR_KNOWN_EVENTS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown event '{req.event}'; valid: {sorted(_BEHAVIOR_KNOWN_EVENTS)}",
        )

    # Validate session regex if present
    if req.session is not None and not _BEHAVIOR_SESSION_RE.match(req.session):
        raise HTTPException(status_code=400, detail="session id format invalid")

    # Validate path stays inside vault (no traversal)
    if req.path is not None:
        p = req.path
        if p.startswith("/") or ".." in p.split("/"):
            raise HTTPException(status_code=400, detail="path must be vault-relative")
        try:
            resolved = (VAULT_PATH / p).resolve()
            VAULT_PATH.resolve()  # ensure VAULT_PATH itself resolves
            resolved.relative_to(VAULT_PATH.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="path escapes vault root")

    # Rate limit: 120 events/60s per client IP
    client_ip = (request.client.host if request.client else "unknown")
    now = time.time()
    bucket = _BEHAVIOR_BUCKETS[client_ip]
    # Prune old entries
    cutoff = now - _BEHAVIOR_RATE_WINDOW
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= _BEHAVIOR_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    bucket.append(now)

    try:
        log_behavior_event({
            "source": req.source,
            "event": req.event,
            "query": req.query,
            "path": req.path,
            "rank": req.rank,
            "dwell_ms": req.dwell_ms,
            "session": req.session,
        })
    except Exception as exc:
        # Never 500 on I/O failure — degrade gracefully
        print(f"[behavior] write error: {exc}", flush=True)
        raise HTTPException(status_code=503, detail="event log unavailable")

    return {"ok": True}


def _ollama_alive(timeout: float = 2.0) -> bool:
    """Fast probe: does the ollama daemon answer `/api/tags` within `timeout`s?
    When the daemon hits its stuck-load state the HTTP listener accepts but
    never replies — /api/chat then hangs forever. A short /api/tags probe
    catches that state without waiting for a model load.
    """
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _ollama_restart_if_stuck() -> bool:
    """Heal the ollama daemon via `brew services restart`. Returns True on
    successful restart. Blocks 3-5s for the bounce.
    """
    try:
        subprocess.run(
            ["/opt/homebrew/bin/brew", "services", "restart", "ollama"],
            check=True, capture_output=True, timeout=30,
        )
        # Wait up to 10s for the daemon to accept traffic again.
        for _ in range(20):
            if _ollama_alive(timeout=1.0):
                return True
            time.sleep(0.5)
    except Exception:
        pass
    return False


@app.post("/api/ollama/restart")
def ollama_restart() -> dict:
    """Panic button #2: brew-services-restart the ollama daemon. Use when
    /api/chat is hanging forever (stuck-load state: daemon accepts HTTP
    but never streams a reply). Blocks ~5-10s.
    """
    ok = _ollama_restart_if_stuck()
    return {"ok": ok, "alive": _ollama_alive()}


@app.post("/api/ollama/unload")
def ollama_unload() -> dict:
    """Panic button: evict every loaded model from ollama + drop the
    reranker from MPS. Frees 15-25 GB of wired unified memory in one
    call when the Mac starts beachballing.

    Drops each loaded model with keep_alive=0 (ollama's immediate-evict
    sentinel) and clears the process-local reranker. Next chat query
    pays 3-8s cold reload but the host stays responsive until then.
    """
    freed = []
    try:
        ps = ollama.ps()
        for m in getattr(ps, "models", []) or []:
            name = getattr(m, "model", None) or getattr(m, "name", None)
            if not name:
                continue
            try:
                # A bare generate with keep_alive=0 tells ollama to unload
                # immediately after the (empty) prompt returns.
                ollama.generate(model=name, prompt="", keep_alive=0)
                freed.append(name)
            except Exception as exc:
                freed.append(f"{name} (fail: {exc})")
    except Exception as exc:
        return {"ok": False, "error": str(exc), "freed": freed}
    try:
        from rag import maybe_unload_reranker, _reranker_last_use  # noqa: F401
        import rag as _rag
        _rag._reranker_last_use = 0.0   # force eviction regardless of idle_ttl
        reranker_dropped = maybe_unload_reranker()
    except Exception:
        reranker_dropped = False
    return {"ok": True, "freed_models": freed, "reranker_dropped": reranker_dropped}


class ReminderCreateRequest(BaseModel):
    text: str
    due: str | None = None   # "tomorrow" or None
    list: str | None = None  # Reminders list name; None → default


@app.post("/api/reminders/create")
def create_reminder(req: ReminderCreateRequest) -> dict:
    """Create an Apple Reminder. Called from the daily brief's "Para
    mañana" items — each <li> has an inline enter-arrow that posts here.
    `due="tomorrow"` is the expected default from the UI; server computes
    the concrete time inside AppleScript to stay locale-safe.
    """
    from rag import _create_reminder  # noqa: PLC0415
    ok, res = _create_reminder(req.text, due_token=req.due, list_name=req.list)
    if not ok:
        raise HTTPException(status_code=400, detail=res)
    # Bust the home cache so the new reminder shows up in the reminders
    # panel on next load without waiting for SWR.
    try:
        _HOME_STATE["ts"] = 0.0
    except Exception:
        pass
    return {"ok": True, "id": res}


class ReminderCompleteRequest(BaseModel):
    reminder_id: str


@app.post("/api/reminders/complete")
def complete_reminder(req: ReminderCompleteRequest) -> dict:
    """Mark an Apple Reminder as completed by its stable id.

    Called from the home page when the user ticks a reminder. Invalidates
    the home cache so the next /api/home doesn't return the now-completed
    item from SWR.
    """
    from rag import _complete_reminder  # noqa: PLC0415
    ok, msg = _complete_reminder(req.reminder_id)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    # Bust the home cache so the reminder disappears on next load without
    # waiting for the 60s SWR cycle.
    try:
        _HOME_STATE["ts"] = 0.0
    except Exception:
        pass
    return {"ok": True, "message": msg}


@app.get("/api/model")
def get_chat_model() -> dict:
    """Return the chat model that /api/chat would use right now."""
    return {"model": _resolve_web_chat_model()}


@app.get("/api/session/{sid}")
def get_session_info(sid: str) -> dict:
    """Return session summary for the chat UI `/session` slash command."""
    from rag import load_session  # noqa: PLC0415
    sess = load_session(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="session no encontrada")
    turns = sess.get("turns", [])
    return {
        "id": sess.get("id", sid),
        "mode": sess.get("mode", ""),
        "created_at": sess.get("created_at", ""),
        "updated_at": sess.get("updated_at", ""),
        "turns": len(turns),
        "first_q": (turns[0].get("q", "") if turns else "")[:120],
        "last_q": (turns[-1].get("q", "") if turns else "")[:120],
    }


class SaveRequest(BaseModel):
    session_id: str
    title: str | None = None


@app.post("/api/save")
def save_conversation(req: SaveRequest) -> dict:
    """Persist a chat session into the vault as a single Markdown note.

    The CLI's `/save` only saves the last turn; here we flatten the whole
    session so the user can export a back-and-forth thread. Lands in
    00-Inbox with source_metas aggregated across all turns.
    """
    from rag import get_db_for, load_session, save_note  # noqa: PLC0415
    sess = load_session(req.session_id)
    if not sess or not sess.get("turns"):
        raise HTTPException(status_code=404, detail="session vacía o inexistente")
    turns = sess["turns"]
    body_parts: list[str] = []
    for t in turns:
        q = (t.get("q") or "").strip()
        a = (t.get("a") or "").strip()
        if not q and not a:
            continue
        body_parts.append(f"**Pregunta:** {q}\n\n{a}")
    body = "\n\n---\n\n".join(body_parts)
    question = turns[0].get("q", "") or "chat"
    vaults = resolve_vault_paths(None)
    if not vaults:
        raise HTTPException(status_code=400, detail="no hay vault activo")
    col = get_db_for(vaults[0][1])
    seen: set[str] = set()
    source_metas: list[dict] = []
    for t in turns:
        for p in (t.get("paths") or []):
            if p and p not in seen:
                seen.add(p)
                source_metas.append({
                    "file": p,
                    "note": Path(p).stem,
                    "tags": "",
                })
    path = save_note(col, req.title, body, question, source_metas)
    try:
        rel = str(path.relative_to(vaults[0][1]))
    except ValueError:
        rel = str(path)
    return {"ok": True, "path": rel}


@app.post("/api/reindex")
def trigger_reindex() -> dict:
    """Fire-and-forget incremental reindex. `--reset` not exposed via web."""
    try:
        subprocess.Popen(
            ["rag", "index"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="rag CLI no encontrado en PATH")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "message": "index lanzado en background"}


class FollowupsRequest(BaseModel):
    session_id: str


@app.post("/api/followups")
def followups(req: FollowupsRequest) -> dict:
    """Generate 3 follow-up question chips from the last turn.

    Uses the helper model (qwen2.5:3b) with deterministic temp to keep
    suggestions tight and cheap (~400-800ms on M-series). Returns empty
    list on any error so the UI fails silently.
    """
    from rag import load_session, get_db  # noqa: PLC0415
    sess = load_session(req.session_id)
    if not sess or not sess.get("turns"):
        return {"followups": []}
    last = sess["turns"][-1]
    q = (last.get("q") or "").strip()
    a = (last.get("a") or "").strip()[:800]
    paths = (last.get("paths") or [])[:3]
    snippets: list[str] = []
    if paths:
        try:
            col = get_db()
            for p in paths:
                res = col.get(where={"file": p}, include=["documents", "metadatas"])
                docs = res.get("documents") or []
                metas = res.get("metadatas") or []
                if not docs:
                    continue
                title = (metas[0].get("note") if metas else None) or Path(p).stem
                body = (docs[0] or "").strip().replace("\n", " ")
                snippets.append(f"- {title}: {body[:280]}")
        except Exception:
            snippets = []
    ctx_bits = [f"Pregunta previa: {q}", f"Respuesta: {a}"]
    if snippets:
        ctx_bits.append("Fragmentos de las notas que aparecieron:\n" + "\n".join(snippets))
    ctx = "\n\n".join(ctx_bits)
    prompt = (
        "Sugerí 3 preguntas de seguimiento concretas que el usuario podría "
        "hacer para profundizar usando su vault de Obsidian. Las preguntas "
        "DEBEN anclarse en hechos, nombres, herramientas o conceptos que "
        "aparezcan literalmente en los fragmentos de arriba — no inventes "
        "ángulos no presentes. Cada pregunta ≤70 caracteres, en español "
        "rioplatense (tuteo). Devolvé SOLO un JSON con la forma "
        '{"followups": ["...", "...", "..."]}. Sin texto extra.\n\n'
        f"{ctx}\n\nJSON:"
    )
    try:
        resp = ollama.chat(
            model=resolve_chat_model(),
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "seed": 42, "num_predict": 220, "num_ctx": 2048},
            format="json",
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        raw = (resp.message.content or "").strip() or "{}"
        data = json.loads(raw)
        if isinstance(data, dict):
            arr = data.get("followups") or data.get("questions") or \
                  next((v for v in data.values() if isinstance(v, list)), [])
        elif isinstance(data, list):
            arr = data
        else:
            arr = []
        out: list[str] = []
        for s in arr:
            if not isinstance(s, str):
                continue
            s = s.strip().strip('"').strip("'")
            if s and len(s) <= 90:
                out.append(s)
            if len(out) >= 3:
                break
        return {"followups": out}
    except Exception:
        return {"followups": []}


class TTSRequest(BaseModel):
    text: str
    voice: str = "Monica"


@app.post("/api/tts")
def tts(req: TTSRequest):
    """Render text with macOS `say` (default voice Mónica) to WAV bytes.

    Local-only: no cloud. WAV chosen over AIFF because browser support is
    universal. Text capped and markdown-stripped so the voice doesn't
    read out backticks and brackets.
    """
    import tempfile  # noqa: PLC0415
    from fastapi.responses import Response  # noqa: PLC0415
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="texto vacío")
    text = text[:1500]
    clean = re.sub(r"```[\s\S]*?```", " ", text)
    clean = re.sub(r"`[^`]*`", " ", clean)
    clean = re.sub(r"[#*_\[\]<>]", "", clean)
    clean = re.sub(r"\(https?://[^)]+\)", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean:
        raise HTTPException(status_code=400, detail="texto vacío post-limpieza")
    voice = req.voice if re.match(r"^[A-Za-zÀ-ÿ]{1,32}$", req.voice or "") else "Monica"
    fd = None
    out_path = None
    try:
        fh = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_path = fh.name
        fh.close()
        subprocess.run(
            ["say", "-v", voice, "--file-format=WAVE",
             "--data-format=LEI16@22050", "-o", out_path, clean],
            check=True, capture_output=True, timeout=45,
        )
        data = Path(out_path).read_bytes()
        return Response(content=data, media_type="audio/wav")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", errors="replace")[:200]
        raise HTTPException(status_code=500, detail=f"say falló: {detail}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="say timeout")
    finally:
        if out_path:
            try:
                Path(out_path).unlink(missing_ok=True)
            except Exception:
                pass


@app.get("/api/vaults")
def list_vaults() -> dict:
    """Expone el registry + vault activo para el picker de la UI."""
    cfg = _load_vaults_config()
    active = resolve_vault_paths(None)
    active_name = active[0][0] if active else None
    registered = sorted(cfg.get("vaults", {}).keys())
    # Si el activo no está en el registry (p.ej. OBSIDIAN_RAG_VAULT apuntando
    # a uno no registrado), lo incluimos igual para que el picker lo muestre.
    if active_name and active_name not in registered:
        registered = [active_name] + registered
    return {
        "active": active_name,
        "registered": registered,
        "current": cfg.get("current"),
    }


@app.get("/api/history")
def query_history(limit: int = 200) -> dict:
    """Tail of `queries.jsonl` for terminal-style up-arrow nav in the web UI.

    Returns oldest→newest after deduping consecutive identical questions.
    Filters to chat-bound commands so /save, /reindex, internal eval runs,
    etc. don't pollute the history list.
    """
    limit = max(1, min(int(limit or 200), 1000))
    if not LOG_PATH.is_file():
        return {"history": []}
    keep_cmds = {"query", "chat", "ask"}
    out: list[str] = []
    try:
        with LOG_PATH.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                cmd = rec.get("cmd") or ""
                if cmd and cmd not in keep_cmds:
                    continue
                q = (rec.get("q") or "").strip()
                if not q:
                    continue
                if out and out[-1] == q:
                    continue
                out.append(q)
    except OSError:
        return {"history": []}
    return {"history": out[-limit:]}


def _resolve_scope(scope: str | None) -> list[tuple[str, "Path"]]:
    if scope is None or scope == "":
        return resolve_vault_paths(None)
    if scope == "all":
        return resolve_vault_paths(["all"])
    return resolve_vault_paths([scope])


class _InlineCitationStripper:
    """Streaming filter that elides `[Title](path.md)` from LLM output.

    Command-r ignores "don't cite inline" instructions about ~15-20% of
    the time even when the prompt is unambiguous. Since the UI renders a
    dedicated sources block below every answer, inline citations show
    the same note twice. This filter is a safety net: it buffers any
    text starting at an open bracket until we know whether the bracket
    opens a `[…](…md)` link or something else. If it does, we drop the
    whole pattern; if it doesn't, we flush the buffered text unchanged.

    Kept mid-module (not inside the chat handler) so tests can import it
    in isolation.
    """

    _MAX_TITLE = 200   # guard against runaway buffers on malformed output
    _MAX_TARGET = 400

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        i = 0
        n = len(self._buf)
        while i < n:
            c = self._buf[i]
            if c != "[":
                out.append(c)
                i += 1
                continue
            # Try to resolve a `[title](target)` pattern starting here.
            close_br = self._buf.find("]", i + 1)
            if close_br == -1:
                if n - i > self._MAX_TITLE:
                    # Not a link — too long to be a title. Flush.
                    out.append(self._buf[i])
                    i += 1
                    continue
                break  # incomplete; wait for more.
            if close_br + 1 >= n:
                break  # don't know yet what follows `]`.
            if self._buf[close_br + 1] != "(":
                # Plain `[…]` text, not a link. Emit and continue.
                out.append(self._buf[i:close_br + 1])
                i = close_br + 1
                continue
            close_pa = self._buf.find(")", close_br + 2)
            if close_pa == -1:
                if n - (close_br + 2) > self._MAX_TARGET:
                    out.append(self._buf[i:close_br + 2])
                    i = close_br + 2
                    continue
                break  # incomplete link tail; wait for more.
            target = self._buf[close_br + 2:close_pa]
            if target.endswith(".md"):
                # Drop the entire `[title](path.md)`.
                i = close_pa + 1
                # Optional: also elide a trailing ", " or " (" that was
                # a separator between consecutive inline citations.
                while i < n and self._buf[i] in (" ", ","):
                    i += 1
            else:
                # External URL or non-note target — keep as-is.
                out.append(self._buf[i:close_pa + 1])
                i = close_pa + 1
        self._buf = self._buf[i:]
        return "".join(out)

    def flush(self) -> str:
        tail = self._buf
        self._buf = ""
        return tail


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _source_payload(meta: dict, score: float) -> dict:
    return {
        "file": meta.get("file", ""),
        "note": meta.get("note", ""),
        "folder": meta.get("folder", ""),
        "score": round(float(score), 3),
    }


def _confidence_badge(score: float) -> tuple[str, str]:
    if score >= 3.0:
        return ("🟢", f"alta · {score:.1f}")
    if score >= 0.0:
        return ("🟡", f"media · {score:.1f}")
    return ("🔴", f"baja · {score:.1f}")


def _score_bar(score: float, width: int = 5) -> str:
    clipped = max(-5.0, min(10.0, score))
    normalized = (clipped + 5.0) / 15.0
    filled = int(round(normalized * width))
    return "■" * filled + "□" * (width - filled)


# ── Intent detection: tasks / agenda / pendientes ────────────────────────────
# Short-circuits the normal RAG retrieve when the question is about what the
# user has to do. RAG over vault notes produces hallucinations ("pedido de
# speeds") because the vault doesn't store live calendar/reminders; the
# services layer does.
#
# Tasks-mode is scope-aware: "mañana" queries fetch tomorrow's reminders +
# calendar, "esta semana" fetches 7 days, default fetches 2 weeks. Without
# this, a "qué tengo para mañana" query got today's reminders and the LLM
# correctly concluded "nada para mañana" — the fix is to bring the right
# window, not to patch the LLM.

_TASKS_INTENT_TOKENS = re.compile(
    r"\b(pendientes?|agenda|recordatorios?|to-?dos?|tasks?|reminders?|"
    r"tareas?|reuniones?|meetings?|prioridad(?:es)?|prioriz|"
    r"compromisos?|obligaciones?|deberes?)",
    re.IGNORECASE,
)
_TASKS_INTENT_PHRASES = re.compile(
    # "qué tengo/debo/hay/etc" REQUIERE qualifier de tarea/tiempo después
    # — la versión laxa anterior matcheaba "qué hay sobre X" / "qué tengo
    # en notas sobre Y" y mis-routeaba topic queries al fetcher de agenda.
    r"(qu[eé]\s+(tengo|debo|me\s+falta|me\s+queda|hay)\s+"
    r"(para|que\s+(hacer|recordar|completar|priorizar)|"
    r"hacer|recordar|completar|"
    r"hoy|ma[nñ]ana|esta\s+semana|pendiente|priorizar|gestionar|organizar)|"
    r"what\s+do\s+i\s+(have|need)\s+(to\s+)?(do|complete|finish)|"
    r"(tengo|hay)\s+(algo|cosas?|eventos?|citas?)\s+(pendiente|hoy|ma[nñ]ana|esta\s+semana)|"
    r"organiz(ar|ame)\s+(el\s+d[ií]a|la\s+semana))",
    re.IGNORECASE,
)


def _is_tasks_query(q: str) -> bool:
    if not q:
        return False
    return bool(_TASKS_INTENT_TOKENS.search(q) or _TASKS_INTENT_PHRASES.search(q))


# Anaphora cues that mean "this question depends on the prior turn" — pronouns,
# demonstratives, ellipsis markers. If a follow-up has any of these, route it
# through reformulate_query so retrieval gets the resolved entity. Otherwise
# the original wording is already a self-contained search target.
#
# Deliberately excluded: `el`/`la`/`lo` — they're common articles and would
# fire on virtually every Spanish sentence, causing the reformat call on fully
# self-contained questions (empirically regressed quality by ~0.15 rerank).
_FOLLOWUP_CUES = re.compile(
    r"\b(eso|esa|ese|esto|esta|este|aquello|ella|[eé]l\b(?!\s+[A-ZÁÉÍÓÚ])|"
    r"ahí|ahi|allí|alli|allá|alla|"
    r"it\b|this\b|that\b|he\b|she\b|"
    r"y\s+(de|sobre|con|para|en)|"
    r"profundizá|profundiza|ampliá|amplia|seguí|segui|continuá|continua|"
    r"más\s+(sobre|de|al\s+respecto)|mas\s+(sobre|de|al\s+respecto))\b",
    re.IGNORECASE,
)

# Leading-pronoun pattern: question starts with a pronoun/demonstrative that
# requires prior context to resolve. Only checked at position 0 (after strip).
_LEADING_PRONOUN_RE = re.compile(
    r"^(eso|esa|ese|esto|esta|este|ella|[eé]l|it|this|that|he|she)\b",
    re.IGNORECASE,
)

# Short interrogative without an explicit subject — "qué hay de X?", "cuál
# es?", "cómo funciona?" where X ≤ 1 token. Only fires for ≤5-word questions
# so long complete questions pass through.
_SHORT_INTERROGATIVE_RE = re.compile(
    r"^(qué|que|cuál|cual|cómo|como|por\s+qué|por\s+que|"
    r"what|how|which|why)\b",
    re.IGNORECASE,
)


def _looks_like_followup(q: str) -> bool:
    """Heuristic: should we pay the reformulate_query LLM call (~1s)?

    Fires when the question is contextually incomplete and needs prior-turn
    resolution. Three cases:
      (a) ≤2 words — genuine ellipsis: "y eso?", "más?", "cuál?"
      (b) Starts with a pronoun/demonstrative: "eso que dijiste", "ella cómo?"
      (c) ≤5-word question starting with an interrogative word (incomplete
          subject implied by context): "qué hay de eso?"

    Long standalone questions (≥4 words without leading cues) skip the call —
    the original query is already a good search target. Avoids paying 0.6-2s
    reformulate on "resumime los sprints", "qué dice la nota sobre Grecia", etc.
    """
    if not q:
        return False
    words = q.split()
    # (a) very short — almost certainly anaphoric
    if len(words) <= 2:
        return True
    # (b) starts with a pronoun/demonstrative that needs antecedent resolution
    stripped = q.strip().lstrip("¿").strip()
    if _LEADING_PRONOUN_RE.match(stripped):
        return True
    # (c) short interrogative (≤5 words) — subject likely implicit from context
    if len(words) <= 5 and _SHORT_INTERROGATIVE_RE.match(stripped):
        return True
    # (d) explicit anaphora cue anywhere in the question
    return bool(_FOLLOWUP_CUES.search(q))


# Pronouns / demonstratives that, if still present after reformulation,
# signal the helper model failed to resolve the antecedent. Triggers the
# concat-with-last-user-turn fallback. `el`/`la`/`lo` excluded because
# they're also common articles and would fire on every sentence.
_UNRESOLVED_PRONOUN_RE = re.compile(
    r"\b(ella|él|eso|esa|ese|esto|esta|este|aquello|"
    r"ahí|ahi|allí|alli|allá|alla)\b",
    re.IGNORECASE,
)


_SCOPE_N_DAYS = re.compile(
    r"\bpr[oó]xim[oa]s?\s+(\d+)\s+d[ií]as?\b|\bnext\s+(\d+)\s+days?\b",
    re.IGNORECASE,
)
_SCOPE_WEEK = re.compile(
    r"\besta\s+semana\b|\bpr[oó]xima\s+semana\b|\bsemana\s+que\s+viene\b|"
    r"\bthis\s+week\b|\bnext\s+week\b",
    re.IGNORECASE,
)
_SCOPE_TOMORROW = re.compile(r"\bma(?:ñ|n)ana\b|\btomorrow\b", re.IGNORECASE)
_SCOPE_TODAY = re.compile(
    r"\bhoy\b|\bahora\b|\btoday\b|\beste\s+momento\b", re.IGNORECASE,
)


def _detect_time_scope(q: str) -> dict:
    """Return the time scope carried by the query. Priority: explicit N days
    beats week beats tomorrow beats today; unmarked queries get a 14-day
    reminders window + 7-day calendar window (user asking "qué tengo
    pendiente" wants a broad sweep).
    """
    m = _SCOPE_N_DAYS.search(q)
    if m:
        n = int(m.group(1) or m.group(2) or 0)
        n = max(1, min(30, n))
        return {
            "key": "n_days", "label": f"próximos {n} días",
            "reminders_horizon": n + 1, "calendar_ahead": n,
        }
    if _SCOPE_WEEK.search(q):
        return {
            "key": "week", "label": "esta semana",
            "reminders_horizon": 8, "calendar_ahead": 7,
        }
    if _SCOPE_TOMORROW.search(q):
        return {
            "key": "tomorrow", "label": "mañana",
            "reminders_horizon": 2, "calendar_ahead": 1,
        }
    if _SCOPE_TODAY.search(q):
        return {
            "key": "today", "label": "hoy",
            "reminders_horizon": 1, "calendar_ahead": 0,
        }
    return {
        "key": "general", "label": "próximas 2 semanas",
        "reminders_horizon": 14, "calendar_ahead": 7,
    }


def _fetch_calendar_ahead(days_ahead: int, max_events: int = 40) -> list[dict]:
    """Events from today through today+days_ahead via icalBuddy with relative
    date labels ("today" / "tomorrow" / "day after tomorrow" / YYYY-MM-DD).
    Returns [{date_label, title, time_range}]. Empty list on any failure —
    the brief layer tolerates silent missing calendar.
    """
    if not _apple_enabled() or days_ahead < 0:
        return []
    icb = _icalbuddy_path()
    if not icb:
        return []
    query = "eventsToday" if days_ahead == 0 else f"eventsToday+{days_ahead}"
    try:
        res = subprocess.run(
            [icb, "-nc", "-iep", "title,datetime", "-b", "", query],
            capture_output=True, text=True, timeout=10.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if res.returncode != 0 or not (res.stdout or "").strip():
        return []

    events: list[dict] = []
    current: dict | None = None
    for raw in (res.stdout or "").splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if not line.startswith(" ") and not line.startswith("\t"):
            if current and current.get("title"):
                events.append(current)
            current = {"title": line.strip(), "date_label": "", "time_range": ""}
            continue
        stripped = line.strip()
        if current is None:
            continue
        m = re.search(
            r"(\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?)\s*-\s*(\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?)",
            stripped,
        )
        if m:
            current["time_range"] = f"{m.group(1)}–{m.group(2)}"
            # date label is everything before the "at HH:MM" span
            label = stripped.split(" at ", 1)[0].strip()
            if label and label != stripped:
                current["date_label"] = label
        else:
            # date-only line (all-day event)
            current["date_label"] = stripped
    if current and current.get("title"):
        events.append(current)
    return events[:max_events]


def _collect_scoped_evidence(col, now: datetime, scope: dict) -> dict:
    """`_pendientes_collect` under the hood for services that are scope-free
    (gmail, mail, whatsapp, loops, contradictions), then override `reminders`
    and `calendar` with scope-aware fetches. WhatsApp always stays on — user
    wants the recent chat context attached even to non-WA questions.
    """
    ev = _pendientes_collect(col, now, days=14)
    from contextlib import suppress
    with suppress(Exception):
        ev["reminders"] = _fetch_reminders_due(
            now,
            horizon_days=int(scope["reminders_horizon"]),
            max_items=40,
        )
    with suppress(Exception):
        ev["calendar_range"] = _fetch_calendar_ahead(int(scope["calendar_ahead"]))
    # Ensure whatsapp is always populated even if `_pendientes_collect`
    # skipped it silently (bridge transiently down, etc.).
    if not ev.get("whatsapp"):
        with suppress(Exception):
            ev["whatsapp"] = _fetch_whatsapp_unread(hours=24, max_chats=8)
    return ev


def _format_pendientes_context(
    ev: dict, urgent: list[str], now: datetime, scope: dict,
) -> str:
    """Markdown evidence blob for the LLM. Scope label is front-and-centre so
    the model filters to the right window. WhatsApp is always included —
    even a non-WA question benefits from seeing recent chat context.
    """
    lines: list[str] = [
        f"# Evidencia de servicios · {now.strftime('%Y-%m-%d %H:%M')}",
        f"**Ventana temporal pedida:** {scope['label']}",
    ]

    if urgent:
        lines.append(f"\n## 🚨 Urgente / Overdue ({len(urgent)})")
        lines.extend(f"- {u}" for u in urgent)

    cal = ev.get("calendar_range") or ev.get("calendar") or []
    if cal:
        lines.append(f"\n## 📅 Calendar ({len(cal)} eventos)")
        for e in cal[:20]:
            when_bits = [b for b in (e.get("date_label"), e.get("time_range")) if b]
            if not when_bits and e.get("start"):
                when_bits = [e["start"]]
            when = " · ".join(when_bits) or "sin hora"
            lines.append(f"- {when} — {e.get('title','')}")

    rem = ev.get("reminders") or []
    if rem:
        lines.append(f"\n## 📌 Apple Reminders ({len(rem)})")
        for bucket in ("overdue", "today", "upcoming", "undated"):
            bucket_items = [x for x in rem if (x.get("bucket") or "") == bucket]
            for r in bucket_items[:10]:
                due = f" (due {r['due']})" if r.get("due") else ""
                lst = f" [{r['list']}]" if r.get("list") else ""
                lines.append(f"- **{bucket}**: {r.get('name','')}{due}{lst}")

    gm = ev.get("gmail") or {}
    awaiting = gm.get("awaiting_reply") or []
    if awaiting:
        lines.append(f"\n## 📧 Gmail awaiting reply ({len(awaiting)})")
        for m in awaiting[:8]:
            lines.append(
                f"- {m.get('days_old',0)}d · {m.get('subject','')} "
                f"— {m.get('from','')}"
            )

    mail = ev.get("mail_unread") or []
    if mail:
        lines.append(f"\n## 📬 Apple Mail unread 36h ({len(mail)})")
        for m in mail[:5]:
            vip = "VIP · " if m.get("is_vip") else ""
            lines.append(f"- {vip}{m.get('subject','')} — {m.get('sender','')}")

    wa = ev.get("whatsapp") or []
    if wa:
        total = sum(int(w.get("count", 0)) for w in wa)
        lines.append(
            f"\n## 💬 WhatsApp últimas 24h ({len(wa)} chats · {total} msgs)"
        )
        for w in wa[:10]:
            snip = (w.get("last_snippet") or "").strip()[:100]
            snip_part = f' — "{snip}"' if snip else ""
            lines.append(
                f"- {w.get('name','?')} ({w.get('count',0)} msgs){snip_part}"
            )

    stale = ev.get("loops_stale") or []
    activo = ev.get("loops_activo") or []
    if stale or activo:
        lines.append(
            f"\n## 📁 Vault loops ({len(stale)} stale · {len(activo)} activo)"
        )
        for it in stale[:5]:
            src = Path(it["source_note"]).stem
            lines.append(
                f"- stale {it.get('age_days',0)}d: "
                f"{it.get('loop_text','')[:100]} [[{src}]]"
            )
        for it in activo[:5]:
            src = Path(it["source_note"]).stem
            lines.append(
                f"- activo {it.get('age_days',0)}d: "
                f"{it.get('loop_text','')[:100]} [[{src}]]"
            )

    contrad = ev.get("contradictions") or []
    if contrad:
        lines.append(f"\n## ⚠ Contradicciones recientes ({len(contrad)})")
        for c in contrad[:3]:
            tgt = ", ".join(t.get("path", "") for t in c.get("targets", [])[:2])
            lines.append(f"- {c.get('subject_path','')} ↔ {tgt}")

    return "\n".join(lines)


def _pendientes_services_consulted(ev: dict) -> list[str]:
    """Labels of services that returned data — used for UI meta."""
    out: list[str] = []
    if (ev.get("gmail") or {}).get("awaiting_reply"):
        out.append("Gmail")
    if ev.get("mail_unread"):
        out.append("Apple Mail")
    if ev.get("whatsapp"):
        out.append("WhatsApp")
    if ev.get("reminders"):
        out.append("Reminders")
    if ev.get("calendar_range") or ev.get("calendar"):
        out.append("Calendar")
    if ev.get("loops_stale") or ev.get("loops_activo"):
        out.append("Vault loops")
    return out


def _build_tasks_system_rules(scope: dict) -> str:
    """SYSTEM prompt with the concrete time-window embedded. Keeping it per-
    call (instead of a global constant) avoids the 'Sin pendientes' parroting
    failure mode — the model now has to filter by scope, not repeat a fixed
    fallback string.
    """
    return (
        "Sos un asistente personal que organiza la agenda del usuario a "
        "partir de evidencia EN VIVO de sus servicios (Calendar, Reminders, "
        "Gmail, Apple Mail, WhatsApp, loops del vault).\n\n"
        f"VENTANA TEMPORAL: {scope['label']}.\n\n"
        "Reglas estrictas:\n"
        "1. Usá SOLO la evidencia del CONTEXTO. No inventes tareas, "
        "reuniones ni plazos que no estén literalmente listados.\n"
        "2. Filtrá los items a la ventana temporal indicada. "
        "Items de Reminders fuera de la ventana → omitilos.\n"
        "3. Respondé en español, estructurado con encabezados markdown y "
        "bullets. Omití secciones que queden vacías después del filtro.\n"
        "4. Orden: 🚨 Urgente/overdue → 📅 Agenda (Calendar) → "
        "📌 Reminders → 📧 Bandeja (mails) → 💬 WhatsApp → 📁 Vault loops.\n"
        "5. Sé conciso: una línea por item. Para eventos de Calendar incluí "
        "día + hora + título.\n"
        "6. Si tras filtrar por la ventana no queda NADA, respondé "
        "exactamente: 'No tengo nada trackeado para esa ventana.' y parate "
        "ahí — NO copies la frase 'Sin pendientes trackeados' del contexto."
    )


def _gen_tasks_response(sess: dict, question: str, history: list[dict]):
    """SSE generator for the tasks/agenda intent branch. Collects scope-aware
    evidence from ALL registered vaults (home + work) via rag's multi-vault
    collector, builds the structured context, streams the LLM answer.

    Delegates formatting + prompt rules to `rag.py` so web and `rag serve`
    (WhatsApp/bot surface) produce identical output — same sections, same
    `[home]`/`[work]` loop labels, same "SIEMPRE mostrar" prompt semantics.
    """
    now = datetime.now()
    scope = _detect_time_scope(question)
    try:
        all_vaults = resolve_vault_paths(["all"])
    except Exception:
        all_vaults = []
    try:
        if all_vaults:
            ev = _rag_collect_scoped_tasks_evidence_multi(all_vaults, now, scope)
        else:
            ev = _collect_scoped_evidence(get_db(), now, scope)
    except Exception as exc:
        yield _sse("error", {"message": f"servicios fallaron: {exc}"})
        return
    urgent = _pendientes_urgent(ev, now)
    services = (
        _rag_tasks_services_consulted(ev) if all_vaults
        else _pendientes_services_consulted(ev)
    )
    context = (
        _rag_format_scoped_tasks_context(ev, urgent, now, scope) if all_vaults
        else _format_pendientes_context(ev, urgent, now, scope)
    )
    system_rules = _rag_build_tasks_system_rules(scope)

    meta_bits: list[str] = [f"📋 ventana: {scope['label']}"]
    if services:
        meta_bits.append(", ".join(services))
    else:
        meta_bits.append("sin datos de servicios")
    if urgent:
        meta_bits.append(f"🚨 {len(urgent)} urgente(s)")
    yield _sse("meta", {"bits": meta_bits})

    if history:
        messages = (
            [{"role": "system", "content": f"{system_rules}\nCONTEXTO:\n{context}"}]
            + history
            + [{"role": "user", "content": question}]
        )
    else:
        messages = [{"role": "user", "content": (
            f"{system_rules}\nCONTEXTO:\n{context}\n\n"
            f"PREGUNTA: {question}\n\nRESPUESTA:"
        )}]

    parts: list[str] = []
    # Tasks layout with 6+ sections + per-item bullets overflows the default
    # 384 num_predict cap — LLM stops mid-"Vault loops". Bump to 800 so the
    # full brief lands even with both vaults' loops expanded.
    tasks_options = {**CHAT_OPTIONS, "num_predict": 800}
    try:
        for chunk in ollama.chat(
            model=resolve_chat_model(),
            messages=messages,
            options=tasks_options,
            stream=True,
            keep_alive=OLLAMA_KEEP_ALIVE,
        ):
            delta = chunk.message.content or ""
            if delta:
                parts.append(delta)
                yield _sse("token", {"delta": delta})
    except Exception as exc:
        yield _sse("error", {"message": f"LLM falló: {exc}"})
        return

    full = "".join(parts)
    turn_id = new_turn_id()
    append_turn(sess, {
        "q": question,
        "a": full,
        "paths": [],
        "top_score": 0.0,
        "turn_id": turn_id,
        "mode": "tasks",
    })
    save_session(sess)
    log_query_event({
        "cmd": "web.tasks",
        "turn_id": turn_id,
        "session": sess["id"],
        "q": question,
        "scope": scope["key"],
        "scope_label": scope["label"],
        "services": services,
        "n_urgent": len(urgent),
        "n_reminders": len(ev.get("reminders") or []),
        "n_calendar": len(
            ev.get("calendar_range") or ev.get("calendar") or []
        ),
        "n_whatsapp": len(ev.get("whatsapp") or []),
        "n_loops_activo": len(ev.get("loops_activo") or []),
        "n_loops_stale": len(ev.get("loops_stale") or []),
    })

    yield _sse("done", {"turn_id": turn_id, "mode": "tasks"})


_TODAY_FM_SPLIT = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _persist_today_brief(date_label: str, narrative: str) -> None:
    """Write the regenerated brief to `05-Reviews/YYYY-MM-DD-evening.md` so
    subsequent `/api/home` calls hit the cached path. Mirrors the CLI
    `rag today` write — same frontmatter shape so contradiction detection
    and ambient skip rules treat both files the same. Silent-fail.
    """
    from contextlib import suppress

    now_iso = datetime.now().isoformat(timespec="seconds")
    fm_lines = [
        "---",
        f"created: '{now_iso}'",
        "type: evening-brief",
        "tags:",
        "- review",
        "- evening-brief",
        f"date: '{date_label}'",
        "source: web",
        "---",
    ]
    body = "\n".join(fm_lines) + f"\n\n# Evening brief — {date_label}\n\n{narrative}\n"
    path = VAULT_PATH / MORNING_FOLDER / f"{date_label}-evening.md"
    with suppress(Exception):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


def _today_cached_narrative(date_label: str) -> str | None:
    """Return narrative from `05-Reviews/YYYY-MM-DD-evening.md` if present.
    CLI `rag today` writes this file, so we avoid re-running the LLM when the
    brief was already generated today.
    """
    path = VAULT_PATH / MORNING_FOLDER / f"{date_label}-evening.md"
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    body = _TODAY_FM_SPLIT.sub("", raw, count=1).strip()
    # Strip the "# Evening brief — YYYY-MM-DD" header if present
    lines = body.splitlines()
    if lines and lines[0].startswith("# Evening brief"):
        body = "\n".join(lines[1:]).lstrip()
    return body or None


def _fetch_pagerank_top(col, n: int = 5) -> list[dict]:
    """Top-n notes by wikilink PageRank authority. Normalized pr (pr/max_pr).

    Reuses the module-level cache in rag.get_pagerank — cost is a dict sort
    after the first call. Silent-fail via caller suppressor.
    """
    pr_map = get_pagerank(col)
    if not pr_map:
        return []
    corpus = _load_corpus(col)
    ranked = sorted(pr_map.items(), key=lambda kv: -kv[1])[:n]
    max_pr = ranked[0][1] or 1.0
    out: list[dict] = []
    for rank, (path, pr) in enumerate(ranked, start=1):
        title = _path_to_title(corpus, path) or Path(path).stem
        out.append({
            "title": title,
            "path": path,
            "pr": round(pr / max_pr, 4),
            "rank": rank,
        })
    return out


def _fetch_chrome_top_week(n: int = 5) -> list[dict]:
    """Top-n Chrome pages visited in the last 7 days.

    Chrome's `History` SQLite db is held open (locked) by the running
    browser — copy to a tmp file before reading. Chrome's `visit_time` is
    microseconds since 1601-01-01 UTC; convert to unix epoch for the iso
    stamp. Filters out chrome://, file://, and localhost.

    Known limitations (document, not fix — this endpoint is ambient signal,
    not a source of truth):
    - **Multi-profile**: only reads `Default/`. Chrome stores per-profile
      histories in sibling dirs (`Profile 1/History`, `Profile 2/History`, …).
      Users with multiple profiles get a partial top-n. `Local State` JSON
      would list them if we ever want to aggregate.
    - **WAL mode**: Chrome uses SQLite WAL; recent visits may live in
      `History-wal` until checkpointed. Copying only `History` misses the
      tail of "right now". Acceptable for a 7-day window.
    - **Guest mode / Incognito**: never written to disk by design — invisible
      to this fetcher, correctly so.
    """
    import shutil
    import sqlite3
    import tempfile

    src = Path.home() / "Library/Application Support/Google/Chrome/Default/History"
    if not src.is_file():
        return []

    CHROME_EPOCH_OFFSET = 11_644_473_600  # seconds between 1601-01-01 and 1970-01-01
    now_ts = time.time()
    week_ago_chrome = int((now_ts - 7 * 86_400 + CHROME_EPOCH_OFFSET) * 1_000_000)

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=True) as tmp:
        shutil.copyfile(src, tmp.name)
        conn = sqlite3.connect(f"file:{tmp.name}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT u.url AS url, u.title AS title,
                       COUNT(v.id) AS visit_count,
                       MAX(v.visit_time) AS last_visit
                FROM urls u
                JOIN visits v ON v.url = u.id
                WHERE v.visit_time >= ?
                  AND u.url NOT LIKE 'chrome://%'
                  AND u.url NOT LIKE 'chrome-extension://%'
                  AND u.url NOT LIKE 'file://%'
                  AND u.url NOT LIKE '%://localhost%'
                  AND u.url NOT LIKE '%://127.0.0.1%'
                GROUP BY u.url
                ORDER BY visit_count DESC
                LIMIT ?
                """,
                (week_ago_chrome, n),
            ).fetchall()
        finally:
            conn.close()

    out: list[dict] = []
    for r in rows:
        last_unix = (r["last_visit"] / 1_000_000) - CHROME_EPOCH_OFFSET
        out.append({
            "title": r["title"] or "",
            "url": r["url"],
            "visit_count": int(r["visit_count"]),
            "last_visit_iso": datetime.fromtimestamp(last_unix).isoformat(timespec="seconds"),
        })
    return out


def _fetch_youtube_watched(n: int = 5, hours: int = 168) -> list[dict]:
    """Most-recent YouTube video pages opened in Chrome, last `hours` window.

    Same Chrome History DB + tmp-copy pattern as `_fetch_chrome_top_week`.
    Matches `youtube.com/watch`, `m.youtube.com/watch`, and `youtu.be/<id>`.
    Deduplicates by extracted video id (`v=...` for /watch, path segment
    for youtu.be) so re-opening the same video doesn't clutter the panel.
    Titles in Chrome history are the <title> tag at page load — the ad
    slate title occasionally leaks in instead of the real video title;
    acceptable noise for an ambient signal.
    """
    import shutil
    import sqlite3
    import tempfile
    from urllib.parse import urlparse, parse_qs

    src = Path.home() / "Library/Application Support/Google/Chrome/Default/History"
    if not src.is_file():
        return []

    CHROME_EPOCH_OFFSET = 11_644_473_600
    now_ts = time.time()
    window_start_chrome = int((now_ts - hours * 3_600 + CHROME_EPOCH_OFFSET) * 1_000_000)

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=True) as tmp:
        shutil.copyfile(src, tmp.name)
        conn = sqlite3.connect(f"file:{tmp.name}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT u.url AS url, u.title AS title,
                       COUNT(v.id) AS visit_count,
                       MAX(v.visit_time) AS last_visit
                FROM urls u
                JOIN visits v ON v.url = u.id
                WHERE v.visit_time >= ?
                  AND (
                       u.url LIKE '%://www.youtube.com/watch%'
                    OR u.url LIKE '%://youtube.com/watch%'
                    OR u.url LIKE '%://m.youtube.com/watch%'
                    OR u.url LIKE '%://youtu.be/%'
                  )
                GROUP BY u.url
                ORDER BY last_visit DESC
                LIMIT 50
                """,
                (window_start_chrome,),
            ).fetchall()
        finally:
            conn.close()

    seen_ids: set[str] = set()
    out: list[dict] = []
    for r in rows:
        url = r["url"]
        try:
            parsed = urlparse(url)
            if parsed.netloc.endswith("youtu.be"):
                vid = parsed.path.strip("/").split("/")[0] or url
            else:
                vid = (parse_qs(parsed.query).get("v") or [url])[0]
        except Exception:
            vid = url
        if vid in seen_ids:
            continue
        seen_ids.add(vid)
        # "Video Title - YouTube" → "Video Title"; Chrome appends " - YouTube"
        # to the page <title> for every watch page.
        raw_title = (r["title"] or "").strip()
        title = raw_title[:-len(" - YouTube")].rstrip() if raw_title.endswith(" - YouTube") else raw_title
        last_unix = (r["last_visit"] / 1_000_000) - CHROME_EPOCH_OFFSET
        out.append({
            "title": title or url,
            "url": url,
            "video_id": vid,
            "visit_count": int(r["visit_count"]),
            "last_visit_iso": datetime.fromtimestamp(last_unix).isoformat(timespec="seconds"),
        })
        if len(out) >= n:
            break
    return out


# Baselines from CLAUDE.md (2026-04-16 post-quick-wins floor). Hardcoded
# because they're the reference against which the UI renders drift — changing
# the baseline needs a human decision, not an automatic update from the latest
# `rag eval` entry.
_EVAL_BASELINE = {
    "singles_hit5": 0.9048,
    "chains_hit5": 0.76,
    "chains_mrr": 0.58,
    "chain_success": 0.5556,
}


def _fetch_eval_trend(n: int = 10) -> dict | None:
    """Tail the last `n` entries of eval.jsonl and pair them with the
    hardcoded baseline. Silent-fail (returns None) if the log is missing —
    the CLI writes it only after the first `rag eval` run.
    """
    if not EVAL_LOG_PATH.is_file():
        return None
    history: list[dict] = []
    try:
        with EVAL_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    history.append(json.loads(line))
                except Exception:
                    continue
    except OSError:
        return None
    if not history:
        return None
    history = history[-n:]
    return {
        "latest": history[-1],
        "baseline": dict(_EVAL_BASELINE),
        "history": history,
    }


# Followup aging runs the full `find_followup_loops` pipeline, which costs one
# LLM judge call per open loop. Cheap per-loop but the list scales with vault
# size — 6h cache keeps /api/home under ~2s while still refreshing a few
# times per working day.
_FOLLOWUP_AGING_CACHE: dict = {"ts": 0.0, "payload": None}
_FOLLOWUP_AGING_TTL = 6 * 3600


def _fetch_followup_aging() -> dict | None:
    """Bucketize open-loops-by-age for the /api/home aging widget.

    Buckets: 0-7d, 8-30d, stale_30plus (age≥31 OR status=='stale'). Uses a
    90d window on `find_followup_loops` so the 30+ bucket actually sees aged
    items (the default CLI window is 30d). Resolved loops are dropped —
    they are, by definition, no longer aging.
    """
    now_ts = time.time()
    cached = _FOLLOWUP_AGING_CACHE
    if cached["payload"] is not None and now_ts - cached["ts"] < _FOLLOWUP_AGING_TTL:
        return cached["payload"]

    try:
        col = get_db()
        loops = find_followup_loops(col, VAULT_PATH, days=90)
    except Exception:
        return cached["payload"]  # serve last-known-good on failure

    open_loops = [it for it in loops if it.get("status") != "resolved"]
    buckets = {"0_7": 0, "8_30": 0, "stale_30plus": 0}
    for it in open_loops:
        age = int(it.get("age_days") or 0)
        status = it.get("status") or ""
        if age >= 31 or status == "stale":
            buckets["stale_30plus"] += 1
        elif age >= 8:
            buckets["8_30"] += 1
        else:
            buckets["0_7"] += 1

    sample = sorted(open_loops, key=lambda it: -int(it.get("age_days") or 0))[:5]
    sample_out = [
        {
            "note": it.get("source_note") or "",
            "loop": (it.get("loop_text") or "")[:140],
            "age_days": int(it.get("age_days") or 0),
            "status": it.get("status") or "",
        }
        for it in sample
    ]

    payload = {
        "buckets": buckets,
        "total": len(open_loops),
        "sample": sample_out,
    }
    _FOLLOWUP_AGING_CACHE["ts"] = now_ts
    _FOLLOWUP_AGING_CACHE["payload"] = payload
    return payload


def _fetch_drive_recent(now: datetime, max_items: int = 5) -> list[dict]:
    """Files modified in Google Drive in the last 48h. Thin wrapper over
    rag._fetch_drive_evidence — keeps the home panel's window smaller than
    morning's 5d so "recent" on the dashboard means actionable-today, not
    week-in-review. Silent-fail via the underlying fetcher → [].
    """
    ev = _fetch_drive_evidence(now, days=2, max_items=max_items) or {}
    return ev.get("files") or []


def _fetch_whatsapp_unreplied(hours: int = 48, max_chats: int = 5) -> list[dict]:
    """Chats whose **last** message is inbound and sits without a reply.

    Distinct from `_fetch_whatsapp_unread` (counts all inbound in window):
    this one only surfaces conversations where *you* owe the next move.
    A chat with 20 inbound-then-replied messages does not appear here; a
    chat with 1 inbound-and-ignored message does.

    Uses a window function to pick the latest row per chat, then filters
    to `is_from_me = 0`. Window: `hours` bounds how stale a last-inbound
    is allowed to be — anything older is almost certainly abandoned and
    pollutes the panel.

    Schema note: the bridge stores timestamps in RFC3339-ish text; SQLite's
    `datetime()` handles it but raw string comparison would not — keep the
    `datetime(timestamp)` wrappers.
    """
    if not WHATSAPP_DB_PATH.is_file():
        return []
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{WHATSAPP_DB_PATH}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return []
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            WITH last_msg AS (
              SELECT chat_jid, content, is_from_me, timestamp,
                     ROW_NUMBER() OVER (
                       PARTITION BY chat_jid
                       ORDER BY datetime(timestamp) DESC
                     ) AS rn
              FROM messages
              WHERE chat_jid != ?
                AND chat_jid NOT LIKE '%status@broadcast'
                AND datetime(timestamp) > datetime('now', ?)
            )
            SELECT lm.chat_jid   AS jid,
                   c.name        AS name,
                   lm.content    AS last_content,
                   lm.timestamp  AS last_ts
            FROM last_msg lm
            LEFT JOIN chats c ON c.jid = lm.chat_jid
            WHERE lm.rn = 1 AND lm.is_from_me = 0
            ORDER BY datetime(lm.timestamp) DESC
            LIMIT ?
            """,
            (WHATSAPP_BOT_JID, f"-{int(hours)} hours", int(max_chats) * 3),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()

    now_ts = time.time()
    out: list[dict] = []
    for r in rows:
        raw_name = (r["name"] or "").strip()
        jid_prefix = (r["jid"] or "").split("@")[0]
        display_name = raw_name or jid_prefix
        # Same "real name" gate as _fetch_whatsapp_unread: digit-only names
        # are unresolved @lid participants, unhelpful to surface.
        if not any(ch.isalpha() for ch in display_name):
            continue
        snippet = (r["last_content"] or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "…"
        # Bridge stores naive local timestamps; `fromisoformat` parses either
        # naive or tz-aware and `.timestamp()` does the right thing for both.
        try:
            last_dt = datetime.fromisoformat((r["last_ts"] or "").replace("Z", "+00:00"))
            hours_waiting = max(0.0, (now_ts - last_dt.timestamp()) / 3600.0)
        except Exception:
            hours_waiting = 0.0
        out.append({
            "jid": r["jid"],
            "name": display_name,
            "last_snippet": snippet,
            "hours_waiting": round(hours_waiting, 1),
        })
        if len(out) >= max_chats:
            break
    return out


# MOZE (Money app) export — user drops `MOZE_YYYYMMDD_HHMMSS.csv` into iCloud
# Backup. We pick the newest file and parse locally; no network, no API.
# Dates are MM/DD/YYYY (US format); Price uses ES decimals ("2026,74").
# Expenses are stored as negative numbers — abs() for display.
_FINANCE_BACKUP_DIR = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/Backup"
_FINANCE_CACHE: dict = {"key": None, "payload": None}


def _fetch_finance(now: datetime | None = None) -> dict | None:
    """Parse the latest MOZE_*.csv export into a home-panel summary.

    Returns None (silent-fail) if: no CSV found, iCloud folder missing,
    or no rows parseable. Cached by (path, mtime) — re-exports invalidate.
    """
    import calendar
    import csv as _csv
    from collections import Counter

    now = now or datetime.now()
    try:
        csvs = sorted(
            _FINANCE_BACKUP_DIR.glob("MOZE_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        return None
    if not csvs:
        return None
    src = csvs[0]
    try:
        mtime = src.stat().st_mtime
    except Exception:
        return None
    key = (str(src), mtime)
    if _FINANCE_CACHE["key"] == key:
        return _FINANCE_CACHE["payload"]

    def pnum(s: str) -> float:
        s = (s or "").strip().replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return 0.0

    rows: list[tuple[datetime, dict]] = []
    try:
        with src.open(newline="", encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                raw = (r.get("Date") or "").strip()
                if not raw:
                    continue
                try:
                    d = datetime.strptime(raw, "%m/%d/%Y")
                except ValueError:
                    continue
                rows.append((d, r))
    except Exception:
        return None
    if not rows:
        return None

    def ym(d: datetime) -> tuple[int, int]:
        return (d.year, d.month)

    this = ym(now)
    anchor = datetime(now.year, now.month, 1)
    prev = (anchor.year - 1, 12) if anchor.month == 1 else (anchor.year, anchor.month - 1)

    expenses_this: dict[str, float] = {}
    expenses_prev: dict[str, float] = {}
    income_this: dict[str, float] = {}
    by_cat: dict[str, Counter] = {}

    for d, r in rows:
        cur = (r.get("Currency") or "").strip()
        typ = (r.get("Type") or "").strip()
        amt = abs(pnum(r.get("Price")))
        ymk = ym(d)
        if typ == "Expense":
            if ymk == this:
                expenses_this[cur] = expenses_this.get(cur, 0.0) + amt
                cat = (r.get("Main Category") or "—").strip() or "—"
                by_cat.setdefault(cur, Counter())[cat] += amt
            elif ymk == prev:
                expenses_prev[cur] = expenses_prev.get(cur, 0.0) + amt
        elif typ == "Income" and ymk == this:
            income_this[cur] = income_this.get(cur, 0.0) + amt

    rows.sort(key=lambda t: t[0], reverse=True)
    latest: list[dict] = []
    for d, r in rows:
        if len(latest) >= 5:
            break
        if (r.get("Type") or "").strip() not in ("Expense", "Income"):
            continue
        latest.append({
            "date": d.strftime("%Y-%m-%d"),
            "type": (r.get("Type") or "").strip(),
            "category": (r.get("Main Category") or "").strip(),
            "name": (r.get("Name") or "").strip(),
            "store": (r.get("Store") or "").strip(),
            "amount": pnum(r.get("Price")),
            "currency": (r.get("Currency") or "").strip(),
        })

    days_elapsed = now.day
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    ars_this = expenses_this.get("ARS", 0.0)
    ars_prev = expenses_prev.get("ARS", 0.0)
    run_rate = ars_this / days_elapsed if days_elapsed else 0.0
    projected = run_rate * days_in_month
    delta_pct = ((ars_this - ars_prev) / ars_prev * 100.0) if ars_prev else None

    top_cats: list[dict] = []
    for name, amt in (by_cat.get("ARS") or Counter()).most_common(5):
        top_cats.append({
            "name": name,
            "amount": amt,
            "share": (amt / ars_this) if ars_this else 0.0,
        })

    usd_this = expenses_this.get("USD", 0.0) + expenses_this.get("USDB", 0.0)
    usd_prev = expenses_prev.get("USD", 0.0) + expenses_prev.get("USDB", 0.0)

    payload = {
        "source_file": src.name,
        "source_mtime": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
        "month_label": now.strftime("%Y-%m"),
        "days_elapsed": days_elapsed,
        "days_in_month": days_in_month,
        "ars": {
            "this_month": ars_this,
            "prev_month": ars_prev,
            "delta_pct": delta_pct,
            "run_rate_daily": run_rate,
            "projected": projected,
            "income": income_this.get("ARS", 0.0),
            "top_categories": top_cats,
        },
        "usd": {
            "this_month": usd_this,
            "prev_month": usd_prev,
        },
        "latest": latest,
    }
    _FINANCE_CACHE["key"] = key
    _FINANCE_CACHE["payload"] = payload
    return payload


# Cold `_home_compute` is 30–40s (12-fetcher fan-out, slowest = pendientes
# at ~25s). The user must NEVER block on that. Strategy:
#   1. Background pre-warmer thread recomputes every BG_INTERVAL, populating
#      _HOME_STATE["payload"] under lock.
#   2. /api/home reads from _HOME_STATE — instant 2ms regardless of TTL.
#   3. Stale-while-revalidate: if payload age > SOFT_TTL, serve stale and
#      kick a one-shot refresh thread (no double-compute thanks to the
#      `computing` flag).
#   4. First request before the pre-warmer finishes blocks once (cold path),
#      then every subsequent request is instant.
import threading
from fastapi.responses import Response

_HOME_LOCK = threading.Lock()
# Cache serialized bytes too — `/api/home` then becomes a raw bytes write
# (no per-request 35KB json.dumps under the GIL).
_HOME_STATE: dict = {
    "payload": None,
    "body": None,
    "ts": 0.0,
    "computing": False,
}
_HOME_SOFT_TTL = 120.0      # serve cached without bg-refresh under this age
# Pre-warmer cadence — bumped 25s → 300s on 2026-04-17 after chat latency
# blew up to 30-160s. Each cycle fans out 12 channel fetchers and can eat
# 6-27s of ollama + MPS + disk, which starves concurrent /api/chat requests
# waiting for the same ollama daemon. 5-minute refresh is plenty for a
# dashboard — the home page SWRs on visit anyway.
_HOME_BG_INTERVAL = 300.0
# Live chat-in-flight counter. While > 0, the prewarmer skips its cycle so
# chat gets exclusive use of ollama + the reranker. Simple counter (not a
# lock) because multiple concurrent chats are fine — they serialise on
# ollama anyway, and we only care about "nobody chatting" vs "someone is".
_CHAT_INFLIGHT = 0
_CHAT_INFLIGHT_LOCK = threading.Lock()
# Persist cache to disk so service restarts don't force a 30-45s cold
# compute on the next user visit. Disk payload is served immediately
# (stale OK — SWR refreshes in bg within seconds).
_HOME_CACHE_PATH = Path.home() / ".local/share/obsidian-rag/home_cache.json"
_HOME_DISK_TTL = 6 * 3600   # ignore disk cache older than this (stale-but-useful window)

_WARMING_BODY = json.dumps({
    "warming": True,
    "today": {"narrative": "", "narrative_source": "warming",
              "brief_path": None,
              "counts": {"recent_notes": 0, "inbox_today": 0, "todos": 0,
                         "new_contradictions": 0, "low_conf_queries": 0,
                         "total": 0},
              "evidence": {}},
    "urgent": [], "signals": {}, "tomorrow_calendar": [],
    "weather_forecast": None,
}).encode()


def _home_refresh(regenerate: bool = False) -> None:
    """Compute + publish under lock; never run two computes in parallel.
    Silent-fail keeps last good payload visible if a fetcher dependency
    (icalBuddy, whisper, ollama) is momentarily down.
    """
    with _HOME_LOCK:
        if _HOME_STATE["computing"]:
            return
        _HOME_STATE["computing"] = True
    t0 = time.time()
    try:
        payload = _home_compute(regenerate)
        body = json.dumps(payload, default=str).encode()
    except Exception:
        return
    finally:
        with _HOME_LOCK:
            _HOME_STATE["computing"] = False
    now = time.time()
    _HOME_STATE["payload"] = payload
    _HOME_STATE["body"] = body
    _HOME_STATE["ts"] = now
    print(f"[home-refresh] compute={(now - t0):.2f}s regen={regenerate}", file=sys.stderr)
    _persist_home_cache(body, now)


def _persist_home_cache(body: bytes, ts: float) -> None:
    """Write the serialized payload + timestamp to disk so a service
    restart can serve it immediately (instead of forcing a 30-45s cold
    compute on the next visit). Atomic rename so a crash mid-write never
    leaves a half-file. Silent-fail — the cache is optional.
    """
    from contextlib import suppress

    with suppress(Exception):
        _HOME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _HOME_CACHE_PATH.with_suffix(".json.tmp")
        tmp.write_bytes(b'{"ts":' + str(ts).encode()
                        + b',"body":' + body + b'}')
        tmp.replace(_HOME_CACHE_PATH)


def _load_home_cache() -> None:
    """Hydrate `_HOME_STATE` from disk on server startup. If the file is
    present and younger than `_HOME_DISK_TTL`, the first user visit sees
    stale-but-useful data immediately; SWR refreshes in the background.
    """
    if not _HOME_CACHE_PATH.is_file():
        return
    try:
        raw = _HOME_CACHE_PATH.read_bytes()
        data = json.loads(raw)
        ts = float(data.get("ts", 0))
        body_obj = data.get("body")
        if not ts or body_obj is None:
            return
        age = time.time() - ts
        if age > _HOME_DISK_TTL:
            return
        body_bytes = json.dumps(body_obj, default=str).encode()
        _HOME_STATE["body"] = body_bytes
        _HOME_STATE["payload"] = body_obj
        _HOME_STATE["ts"] = ts
        print(f"[home-cache] hydrated from disk age={age:.0f}s size={len(body_bytes)}B",
              file=sys.stderr)
    except Exception as exc:
        print(f"[home-cache] load failed: {exc}", file=sys.stderr)


@app.get("/api/home")
def home_api(regenerate: bool = False) -> Response:
    _ensure_home_prewarmer()
    # Explicit user action — recompute LLM brief synchronously.
    if regenerate:
        _home_refresh(regenerate=True)
        body = _HOME_STATE["body"] or _WARMING_BODY
        return Response(content=body, media_type="application/json")

    body = _HOME_STATE["body"]
    age = time.time() - _HOME_STATE["ts"] if body else float("inf")

    if body and age < _HOME_SOFT_TTL:
        return Response(content=body, media_type="application/json")
    if body:
        # SWR: return stale immediately, refresh in bg
        threading.Thread(
            target=_home_refresh, name="home-swr", daemon=True,
        ).start()
        return Response(content=body, media_type="application/json")
    # First call ever (pre-warmer still racing). Return placeholder so
    # the frontend skeleton stays up; auto-refresh picks up real payload.
    threading.Thread(
        target=_home_refresh, name="home-cold", daemon=True,
    ).start()
    return Response(content=_WARMING_BODY, media_type="application/json")


_HOME_PREWARMER_STARTED = False

# Opt-in prewarmer. Default OFF because the fan-out (12 channel fetchers
# including ollama-based signals) holds the ollama daemon mid-cycle and
# starves concurrent /api/chat requests — even a 300s interval isn't
# enough because a chat arriving mid-cycle still waits for the 7-27s
# compute to finish before its embed call gets served. The home page
# already uses SWR: first visit pays a cold compute (~10s), subsequent
# visits are instant with a bg refresh. That's the right trade.
# Set OBSIDIAN_RAG_HOME_PREWARM=1 to re-enable for display-only deploys.
_HOME_PREWARM_ENABLED = os.environ.get("OBSIDIAN_RAG_HOME_PREWARM") == "1"

def _ensure_home_prewarmer() -> None:
    """Start the pre-warmer loop iff opt-in. Called from startup and from
    /api/home. Idempotent.
    """
    global _HOME_PREWARMER_STARTED
    if _HOME_PREWARMER_STARTED or not _HOME_PREWARM_ENABLED:
        _HOME_PREWARMER_STARTED = True  # poison so we don't retry the check
        return
    _HOME_PREWARMER_STARTED = True

    def loop() -> None:
        while True:
            try:
                # Skip the cycle while a chat request is in flight — each
                # home-compute fans out 12 channel fetchers (some hitting
                # ollama for helper-model judgments) and would queue behind
                # or evict the chat model, turning a 3s chat into 60s+.
                if _CHAT_INFLIGHT > 0:
                    time.sleep(10)
                    continue
                _home_refresh(regenerate=False)
            except Exception:
                pass
            time.sleep(_HOME_BG_INTERVAL)

    threading.Thread(target=loop, name="home-prewarmer", daemon=True).start()


# Chat-model prewarmer. Root cause: ollama evicts the /chat model from VRAM
# between requests despite keep_alive=-1 when memory pressure builds up
# (observed: qwen2.5:7b dropped after ~5min idle while qwen2.5:3b + bge-m3
# + reranker stayed resident). Each cold reload = 4-7s prefill penalty on
# the next /api/chat, turning warm 2.9s into 10s+.
#
# Fix: every _CHAT_PREWARM_INTERVAL seconds, send a 1-token ping to the
# resolved chat model with keep_alive=-1. This re-pins the model in VRAM
# even if ollama decided to evict during idle. Skipped while a real chat
# is in flight (the chat itself keeps the model warm — no need to duplicate).
#
# Cost: ~100-200ms of ollama compute per cycle, ~5GB VRAM pinned continuously.
# Safe on 36GB unified memory with command-r NOT also resident.
_CHAT_PREWARM_INTERVAL = int(os.environ.get("OBSIDIAN_RAG_CHAT_PREWARM_INTERVAL", "240"))
_CHAT_PREWARMER_STARTED = False


def _ensure_chat_model_prewarmer() -> None:
    """Start the chat-model prewarm loop once. Idempotent."""
    global _CHAT_PREWARMER_STARTED
    if _CHAT_PREWARMER_STARTED:
        return
    _CHAT_PREWARMER_STARTED = True

    def loop() -> None:
        # First cycle runs after a short startup delay so the app finishes
        # booting (corpus cache, home hydration) before we pin the LLM.
        # Subsequent cycles run every _CHAT_PREWARM_INTERVAL.
        time.sleep(15)
        while True:
            try:
                if _CHAT_INFLIGHT > 0:
                    time.sleep(15)
                    continue
                model = _resolve_web_chat_model()
                # Tiny ping — 1 token is enough to re-pin KV cache + model weights.
                # options=keep_alive=-1 instructs ollama to hold the model forever,
                # overriding its internal eviction pressure.
                ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": "."}],
                    options={"num_predict": 1, "temperature": 0, "seed": 42},
                    stream=False,
                    keep_alive=-1,
                )
                print(f"[chat-prewarm] {model} pinned", flush=True)
            except Exception as exc:
                # Silent fail: ollama down, model not loaded, network blip.
                # Next cycle retries. Never crash the daemon thread.
                print(f"[chat-prewarm] skipped: {exc}", flush=True)
            time.sleep(_CHAT_PREWARM_INTERVAL)

    threading.Thread(target=loop, name="chat-model-prewarmer", daemon=True).start()


# ── Response cache ──────────────────────────────────────────────────────
# LRU de respuestas completas de /api/chat para queries exactas repetidas.
# Cuando el mismo user query llega dentro del TTL y el vault no cambió,
# servimos la respuesta cacheada como SSE replay (<100ms wall) en lugar de
# re-correr retrieve + LLM (2-3s).
#
# Cache key: sha256(question|vault_scope|chat_model|vault_chunks_count)[:16].
# Incluir el chunks count efectivamente invalida el cache cuando el vault
# gana/pierde notas (count cambia → key cambia). TTL secundario 5min para
# invalidar respuestas sin cambio de vault pero con información que puede
# haber envejecido (ej. `rag tune` refinó pesos → retrieval diferente).
#
# No cacheamos cuando:
#   - hay history (follow-ups dependen del turno previo, el key no refleja eso)
#   - el response fue vacío/error
#   - _wa_in_query matcheó (WA data change rápidamente, datos frescos importan)
_CHAT_CACHE: "OrderedDict[str, dict]" = __import__("collections").OrderedDict()
_CHAT_CACHE_MAX = 100
_CHAT_CACHE_TTL = 300.0  # 5 min
_CHAT_CACHE_LOCK = threading.Lock()


def _chat_cache_key(question: str, vault_scope: str, model: str, vault_chunks: int) -> str:
    raw = f"{question.strip().lower()}|{vault_scope}|{model}|{vault_chunks}"
    return __import__("hashlib").sha256(raw.encode()).hexdigest()[:16]


def _chat_cache_get(key: str) -> dict | None:
    with _CHAT_CACHE_LOCK:
        entry = _CHAT_CACHE.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > _CHAT_CACHE_TTL:
            del _CHAT_CACHE[key]
            return None
        _CHAT_CACHE.move_to_end(key)
        return dict(entry)


def _chat_cache_put(key: str, payload: dict) -> None:
    with _CHAT_CACHE_LOCK:
        _CHAT_CACHE[key] = {"ts": time.time(), **payload}
        _CHAT_CACHE.move_to_end(key)
        while len(_CHAT_CACHE) > _CHAT_CACHE_MAX:
            _CHAT_CACHE.popitem(last=False)


def _home_compute(regenerate: bool = False) -> dict:
    """Centralizer — aggregates every information channel the user cares
    about into one JSON payload. Powers the home page.

    Channels:
    - today narrative (rag today) + vault evidence for the day
    - live signals (reminders, Apple Mail, Gmail, WhatsApp, calendar, weather)
    - derived signals (open loops, contradictions, low-confidence queries)
    - tomorrow agenda + multi-day weather forecast

    Per-channel silent-fail: if one source errors (e.g. Gmail OAuth expired,
    icalBuddy not installed), that key is missing/empty but the rest renders.
    `regenerate=true` forces a fresh LLM narrative (~10s); default reuses the
    cached brief from `05-Reviews/<date>-evening.md` if present.
    """
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import suppress

    now = datetime.now()
    date_label = now.strftime("%Y-%m-%d")
    col = get_db()

    # ── Fan out: vault evidence + live channels + forecast run in parallel.
    # `_pendientes_collect` internally fans out its 9 fetchers too, so the
    # critical path across the whole page is max(today_evidence, slowest
    # single fetcher, tomorrow_calendar, weather_forecast) instead of the sum.
    # Don't use `with ThreadPoolExecutor(...) as pool:` — its __exit__ calls
    # `shutdown(wait=True)` and blocks the whole endpoint until every future
    # finishes, defeating the per-future timeouts below. We submit, collect
    # with timeouts, and then `shutdown(wait=False)` to return fast while
    # any straggler (e.g. cold `_fetch_followup_aging` doing LLM-judge per
    # loop) keeps running in the background to warm its own cache.
    pool = ThreadPoolExecutor(max_workers=14, thread_name_prefix="home")
    timings: dict[str, float] = {}
    t_submit = time.time()

    def _timed(name: str, fn, *args):
        t0 = time.time()
        try:
            return fn(*args)
        finally:
            timings[name] = time.time() - t0

    try:
        fut_today       = pool.submit(
            _timed, "today", _collect_today_evidence,
            now, VAULT_PATH, LOG_PATH, CONTRADICTION_LOG_PATH,
        )
        fut_signals     = pool.submit(_timed, "signals", _pendientes_collect, col, now, 14)
        fut_tomorrow    = pool.submit(_timed, "tomorrow", _fetch_calendar_ahead, 1, 10)
        fut_forecast    = pool.submit(_timed, "forecast", _fetch_weather_forecast)
        fut_pagerank    = pool.submit(_timed, "pagerank", _fetch_pagerank_top, col, 5)
        fut_chrome      = pool.submit(_timed, "chrome", _fetch_chrome_top_week, 5)
        fut_eval        = pool.submit(_timed, "eval", _fetch_eval_trend)
        fut_followup    = pool.submit(_timed, "followup", _fetch_followup_aging)
        fut_drive       = pool.submit(_timed, "drive", _fetch_drive_recent, now, 5)
        fut_wa_unreplied = pool.submit(_timed, "wa_unreplied", _fetch_whatsapp_unreplied, 48, 5)
        fut_bookmarks   = pool.submit(_timed, "bookmarks", _fetch_chrome_bookmarks_used, 48, 5)
        fut_vaults      = pool.submit(_timed, "vaults", _fetch_vault_activity, 48, 5)
        fut_finance     = pool.submit(_timed, "finance", _fetch_finance, now)
        fut_youtube     = pool.submit(_timed, "youtube", _fetch_youtube_watched, 5, 168)

        try:
            today_ev = fut_today.result(timeout=30)
        except Exception as exc:
            pool.shutdown(wait=False, cancel_futures=True)
            raise HTTPException(status_code=500, detail=str(exc))

        # Pendientes itself fans out 9 fetchers; cold path measured ~25s.
        # Bump cap to 45s — the user never waits on this since pre-warmer
        # eats the cold compute. If it still times out, the next cycle
        # repopulates and SWR keeps stale signals visible meanwhile.
        signals = {}
        with suppress(Exception):
            signals = fut_signals.result(timeout=45)

        tomorrow_calendar = []
        with suppress(Exception):
            tomorrow_calendar = fut_tomorrow.result(timeout=10) or []

        weather_forecast = None
        with suppress(Exception):
            weather_forecast = fut_forecast.result(timeout=10)

        pagerank_top: list[dict] = []
        with suppress(Exception):
            pagerank_top = fut_pagerank.result(timeout=10) or []

        chrome_top_week: list[dict] = []
        with suppress(Exception):
            chrome_top_week = fut_chrome.result(timeout=5) or []

        eval_trend: dict | None = None
        with suppress(Exception):
            eval_trend = fut_eval.result(timeout=5)

        # followup_aging has its own 6h cache + LLM-judge per loop on cold.
        # If not ready within 2s, skip it this cycle — the bg thread keeps
        # computing and the next prewarmer pass (25s later) picks up the
        # warm cache. This shaves ~15s off the cold critical path.
        followup_aging: dict | None = None
        with suppress(Exception):
            followup_aging = fut_followup.result(timeout=2)

        drive_recent: list[dict] = []
        with suppress(Exception):
            drive_recent = fut_drive.result(timeout=10) or []

        whatsapp_unreplied: list[dict] = []
        with suppress(Exception):
            whatsapp_unreplied = fut_wa_unreplied.result(timeout=10) or []

        chrome_bookmarks: list[dict] = []
        with suppress(Exception):
            chrome_bookmarks = fut_bookmarks.result(timeout=5) or []

        vault_activity: dict = {}
        with suppress(Exception):
            vault_activity = fut_vaults.result(timeout=10) or {}

        finance: dict | None = None
        with suppress(Exception):
            finance = fut_finance.result(timeout=5)

        youtube_watched: list[dict] = []
        with suppress(Exception):
            youtube_watched = fut_youtube.result(timeout=5) or []

        signals["pagerank_top"] = pagerank_top
        signals["chrome_top_week"] = chrome_top_week
        signals["eval_trend"] = eval_trend
        signals["followup_aging"] = followup_aging
        signals["drive_recent"] = drive_recent
        signals["whatsapp_unreplied"] = whatsapp_unreplied
        signals["chrome_bookmarks"] = chrome_bookmarks
        signals["vault_activity"] = vault_activity
        signals["finance"] = finance
        signals["youtube_watched"] = youtube_watched
    finally:
        # Detach stragglers; they continue warming caches for the next call.
        pool.shutdown(wait=False)
        if timings:
            ranked = sorted(timings.items(), key=lambda kv: -kv[1])
            summary = " ".join(f"{k}={v:.1f}s" for k, v in ranked if v >= 0.2)
            if summary:
                print(f"[home-compute] {summary} total={time.time() - t_submit:.1f}s",
                      file=sys.stderr)

    today_total = (
        len(today_ev.get("recent_notes") or [])
        + len(today_ev.get("inbox_today") or [])
        + len(today_ev.get("todos") or [])
        + len(today_ev.get("new_contradictions") or [])
        + len(today_ev.get("low_conf_queries") or [])
    )

    narrative = "" if today_total == 0 else (_today_cached_narrative(date_label) or "")
    narrative_source = "cached" if narrative else "none"
    # Only call the LLM when the caller explicitly asks. Default path stays
    # fast — if no cached brief exists yet, the UI shows "pendiente" and
    # offers a button that re-hits with regenerate=true.
    if today_total > 0 and regenerate:
        prompt = _render_today_prompt(date_label, today_ev)
        narrative = _generate_today_narrative(prompt)
        narrative_source = "generated" if narrative else "error"
        if narrative:
            _persist_today_brief(date_label, narrative)

    brief_rel: str | None = None
    path = VAULT_PATH / MORNING_FOLDER / f"{date_label}-evening.md"
    if path.is_file():
        with suppress(ValueError):
            brief_rel = str(path.relative_to(VAULT_PATH))

    urgent: list[str] = []
    with suppress(Exception):
        urgent = _pendientes_urgent(signals, now)

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "date": date_label,
        "today": {
            "narrative": narrative,
            "narrative_source": narrative_source,
            "brief_path": brief_rel,
            "counts": {
                "recent_notes": len(today_ev.get("recent_notes") or []),
                "inbox_today": len(today_ev.get("inbox_today") or []),
                "todos": len(today_ev.get("todos") or []),
                "new_contradictions": len(today_ev.get("new_contradictions") or []),
                "low_conf_queries": len(today_ev.get("low_conf_queries") or []),
                "total": today_total,
            },
            "evidence": today_ev,
        },
        "urgent": urgent,
        "signals": signals,
        "tomorrow_calendar": tomorrow_calendar,
        "weather_forecast": weather_forecast,
    }


@app.get("/api/pendientes")
def pendientes_api(days: int = 14) -> dict:
    """Raw JSON view of the services layer. Useful for a dedicated pendientes
    card in the UI or for third-party clients; the chat endpoint uses the
    same `_pendientes_collect` internally when intent matches.
    """
    col = get_db()
    now = datetime.now()
    try:
        ev = _pendientes_collect(col, now, days=days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    urgent = _pendientes_urgent(ev, now)
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "days": days,
        "urgent": urgent,
        "services_consulted": _pendientes_services_consulted(ev),
        "evidence": ev,
    }


def _persist_conversation_turn(
    vault_root: Path,
    session_id: str,
    question: str,
    answer: str,
    metas: list[dict],
    scores: list[float],
    confidence: float,
) -> None:
    try:
        turn = TurnData(
            question=question,
            answer=answer,
            sources=[
                {"file": m.get("file", ""), "score": float(s)}
                for m, s in zip(metas, scores)
            ],
            confidence=float(confidence),
            timestamp=datetime.now(timezone.utc),
        )
        path = write_turn(vault_root, session_id, turn)
        _LOG_QUEUE.put((
            LOG_PATH,
            json.dumps({
                "kind": "conversation_turn_written",
                "session_id": session_id,
                "path": str(path.relative_to(vault_root)),
            }) + "\n",
        ))
    except Exception as exc:
        _LOG_QUEUE.put((
            LOG_PATH,
            json.dumps({
                "kind": "conversation_turn_error",
                "session_id": session_id,
                "error": repr(exc),
            }) + "\n",
        ))


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty question")

    sid = req.session_id or f"web:{uuid.uuid4().hex[:12]}"
    sess = ensure_session(sid, mode="chat")
    vaults = _resolve_scope(req.vault_scope)
    if not vaults:
        raise HTTPException(status_code=400, detail=f"vault '{req.vault_scope}' no encontrado")

    def gen():
        _t0 = time.perf_counter()
        yield _sse("session", {"id": sess["id"]})

        history = session_history(sess, window=SESSION_HISTORY_WINDOW)

        # Tasks / agenda intent → bypass vault RAG (which hallucinates on
        # "what do I have tomorrow" queries because the vault doesn't hold
        # live calendar/reminders) and pull from the services layer instead.
        if _is_tasks_query(question):
            yield from _gen_tasks_response(sess, question, history)
            return

        # Response cache — sirve respuesta cacheada si (question, scope, model,
        # vault_count) matchea + TTL viva + NO es follow-up. Skipped cuando hay
        # history porque la respuesta depende del contexto de la conversación,
        # que el key actual no refleja.
        if not history:
            try:
                from rag import get_db_for
                _vault_chunks = get_db_for(vaults[0][1]).count() if vaults else 0
            except Exception:
                _vault_chunks = 0
            _cache_key = _chat_cache_key(
                question, req.vault_scope or "", _resolve_web_chat_model(), _vault_chunks
            )
            _cached = _chat_cache_get(_cache_key)
            if _cached:
                # Replay completo como SSE. El UI no distingue cached de live.
                yield _sse("status", {"stage": "cached"})
                yield _sse("sources", {
                    "items": _cached["sources_items"],
                    "confidence": _cached["top_score"],
                })
                # Stream el texto en chunks chicos para mantener la ilusión
                # de streaming (el cliente ya espera SSE tokens). 40 chars ≈
                # 10 tokens — UI renderea suave sin el feel "pegote instantáneo".
                _text = _cached["text"]
                for i in range(0, len(_text), 40):
                    yield _sse("token", {"delta": _text[i:i+40]})
                _cached_total_ms = int((time.perf_counter() - _t0) * 1000)
                yield _sse("done", {
                    "turn_id": new_turn_id(),
                    "top_score": _cached["top_score"],
                    "total_ms": _cached_total_ms,
                    "retrieve_ms": 0,
                    "ttft_ms": _cached_total_ms,
                    "llm_ms": 0,
                    "cached": True,
                })
                print(f"[chat-cache] HIT {_cache_key} total={_cached_total_ms}ms", flush=True)
                # Todavía appendeamos al session history — la conversación continúa normal
                append_turn(sess, {
                    "q": question,
                    "a": _text,
                    "paths": [s.get("file", "") for s in _cached["sources_items"]],
                    "top_score": _cached["top_score"],
                    "turn_id": new_turn_id(),
                })
                save_session(sess)
                return

        # Fail fast if ollama is in the "stuck-load" state — accepts HTTP
        # but never responds. Without this probe, the chat hangs on the
        # first embed call for ~5 minutes until the user gives up. The
        # probe is /api/tags which returns instantly on a healthy daemon.
        if not _ollama_alive(timeout=2.0):
            yield _sse("error", {
                "message": "Ollama no responde. Probá: curl -X POST "
                "http://localhost:8765/api/ollama/restart",
            })
            return

        # Kick off WhatsApp fetch in parallel with retrieve so the SQLite
        # round-trip (25-180ms) overlaps with the heavier retrieval work
        # instead of stacking sequentially before the LLM call.
        from concurrent.futures import ThreadPoolExecutor
        _wa_executor = ThreadPoolExecutor(max_workers=1)
        _t_wa_start = time.perf_counter()
        _wa_future = _wa_executor.submit(_fetch_whatsapp_unread, 24, 8)

        # Conversation-aware reformulation is a qwen2.5:3b call that costs
        # 0.6-2s. Only worth paying for short follow-ups that rely on prior
        # context ("y eso?", "más sobre ella"). Standalone questions ≥4 words
        # without anaphora cues skip it — the original query is already a
        # good search target.
        #
        # Two-layer strategy when a follow-up IS detected:
        #   1. reformulate_query rewrites the pronoun using history.
        #   2. If the output STILL contains an unresolved pronoun (helper
        #      got confused by noisy history — e.g. a prior turn cited a
        #      different entity by mistake), fall back to concatenating
        #      the last user turn with the current one. Empirically that
        #      concat anchors retrieval to the right entity without
        #      another LLM call. Example: "tenes algo mas sobre ella?"
        #      after "tenes notas sobre Grecia?" → helper may still emit
        #      "algo mas sobre ella" → concat produces "tenes notas sobre
        #      Grecia? tenes algo mas sobre ella?" which retrieves the
        #      Grecia notes at top (~+0.20 rerank score).
        yield _sse("status", {"stage": "retrieving"})

        _t_reform_start = time.perf_counter()
        # Follow-up resolution: antes llamábamos a reformulate_query (qwen2.5:3b,
        # ~1-2s de LLM call + re-eviction pressure sobre qwen2.5:7b). El comment
        # original admite que el fallback (concat "{last_user_q} {current_q}")
        # empíricamente lograba mismo rerank score con recall igual o mejor
        # (~+0.20 score al anclar con la entidad del turno previo). Saltamos
        # reform directo al concat: 2s → 0ms, y qwen2.5:3b ya no se carga en
        # el path de /chat, aliviando VRAM pressure sobre el 7b pinned.
        search_question = question
        _reform_fired = False
        _reform_used_concat = False
        if history and _looks_like_followup(question):
            _reform_fired = True
            last_user_q = next(
                (m["content"] for m in reversed(history) if m.get("role") == "user"),
                None,
            )
            if last_user_q:
                search_question = f"{last_user_q} {question}"
                _reform_used_concat = True
        _t_reform_end = time.perf_counter()

        try:
            _t_retrieve_start = time.perf_counter()
            # multi_query=False: the qwen2.5:3b paraphrase call costs 1-3s
            # and only marginally improves recall on chat-style questions.
            # rerank_pool=4: prior note said pool=5 averages ~210ms; pool=6+
            # jumps to 1.3-3s due to batch-padding misalignment on MPS.
            # Re-verifying at pool=4 — bge-reranker-v2-m3 batch size 4 aligns
            # naturally and may shave another ~50ms vs pool=5. Top-4 quality
            # holds because RRF already picked the winners; the reranker only
            # refines order. Measured 2026-04-17. Reverted if eval regresses
            # below singles hit@5 76.19 (CI lower bound) or chains
            # chain_success < 16.67.
            result = multi_retrieve(
                vaults, search_question, 1, None, history, None, False,
                multi_query=False, auto_filter=True, date_range=None,
                rerank_pool=2,
            )
            _t_retrieve_end = time.perf_counter()
        except Exception as exc:
            yield _sse("error", {"message": f"retrieve falló: {exc}"})
            return

        if not result["docs"]:
            yield _sse("empty", {"message": "Sin resultados relevantes."})
            return

        # Retrieval signals — previously surfaced as a UI meta bar
        # ("🟡 media · 0.8 · N variantes · M nota(s)"). Removed from the
        # UI per user request; instead we pass these signals as context
        # to the LLM so it can frame the answer (e.g. hedge when the
        # confidence is low, or acknowledge when only one note exists).
        _conf = float(result["confidence"])
        _, _conf_label = _confidence_badge(_conf)
        _n_notes = len({m["file"] for m in result["metas"]})
        _n_variants = len(result.get("query_variants", []))
        _filters = result.get("filters_applied") or {}
        retrieval_signals = {
            "confidence_label": _conf_label,
            "confidence_score": round(_conf, 2),
            "n_notes": _n_notes,
            "n_variants": _n_variants,
            "filters": _filters,
        }

        yield _sse("sources", {
            "items": [
                {**_source_payload(m, s), "bar": _score_bar(float(s))}
                for m, s in zip(result["metas"], result["scores"])
            ],
            "confidence": round(float(result["confidence"]), 3),
        })

        is_multi = len(vaults) > 1
        # Cap each chunk at 500 chars on the fast path. With 4 chunks that's
        # ~2000 chars ≈ 500 tokens of variable context — small enough that
        # uncached prefill stays under ~6s on command-r:35b (12.7ms/tok),
        # large enough that REGLA 4's "obligatory usage block" still lands.
        _WEB_CHUNK_CAP = 500
        context = "\n\n---\n\n".join(
            (f"[vault: {m.get('_vault', '?')}] " if is_multi else "")
            + f"[nota: {m['note']}] [ruta: {m['file']}]\n{d[:_WEB_CHUNK_CAP]}"
            for d, m in zip(result["docs"], result["metas"])
        )

        # Collect the WA fetch we kicked off before retrieve. By now it's
        # almost always done (retrieve dominates), so .result() is a no-op.
        # Telemetry: log the post-retrieve wait, not wall time — so a 0ms
        # `wa_wait` means the SQLite read overlapped fully with retrieve.
        _t_wa_wait_start = time.perf_counter()
        try:
            wa_recent = _wa_future.result(timeout=2.0)
        except Exception:
            wa_recent = []
        finally:
            _wa_executor.shutdown(wait=False)
        _t_wa_end = time.perf_counter()
        _t_wa_wait_ms = int((_t_wa_end - _t_wa_wait_start) * 1000)
        # WA block gated on query intent — antes se inyectaba en CADA chat,
        # agregando ~500-1000ms de prefill innecesario. Ahora sólo se incluye
        # si la pregunta menciona WhatsApp explícitamente. El fetch paralelo
        # se sigue haciendo (telemetry + /api/home consumption) — sólo el
        # prompt injection es el que filtra.
        _wa_in_query = bool(
            re.search(r"\b(whatsapp|\bwa\b|mensaje|chat de|último[s]? chat)", question, re.IGNORECASE)
        )
        if wa_recent and _wa_in_query:
            total_msgs = sum(int(w.get("count", 0)) for w in wa_recent)
            wa_block_lines = [
                f"[contexto auxiliar: WhatsApp últimas 24h · "
                f"{len(wa_recent)} chats · {total_msgs} msgs]"
            ]
            for w in wa_recent[:8]:
                snip = (w.get("last_snippet") or "").strip()[:100]
                snip_part = f' — "{snip}"' if snip else ""
                wa_block_lines.append(
                    f"- {w.get('name','?')} ({w.get('count',0)} "
                    f"msgs){snip_part}"
                )
            context = context + "\n\n---\n\n" + "\n".join(wa_block_lines)

        # Retrieval signals block removido 2026-04-18: costaba ~300 chars
        # = ~75 tokens = ~150ms de prefill uncached en CADA request. El
        # hedging que calibraba (no-refusal, humildad en confianza baja,
        # engancharse con contexto débil) ya está expresado en REGLA 1 +
        # REGLA 4 del _WEB_SYSTEM_PROMPT (que está byte-identical + cached).
        # Empíricamente el modelo respeta esas reglas sin repetir las
        # señales cada turno — redundante con el system prompt.
        # retrieval_signals dict sigue disponible para telemetría
        # (_n_notes, confidence) pero no se inyecta al prompt.

        # Build messages so the system prompt is BYTE-IDENTICAL across all
        # requests — that's the whole game for ollama prefix caching. The
        # context (which changes per query) goes in the user message AFTER
        # any history. With history, this puts the cacheable prefix as:
        #   [system (cached)] + [history turns (partially cached)] + user
        user_content = f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}\n\nRESPUESTA:"
        messages = (
            [{"role": "system", "content": _WEB_SYSTEM_PROMPT}]
            + (history or [])
            + [{"role": "user", "content": user_content}]
        )

        # Log context size for prefill analysis
        _ctx_chars = sum(len(m.get("content","")) for m in messages)

        # Fast-path LLM options.
        # num_ctx: measured from web.log (75 timing rows, 2026-04-17):
        #   P50 ctx ≈ 4889 chars / 1.35 char-per-token ≈ 3622 tok
        #   P95 ctx ≈ 6065 chars ≈ 4492 tok
        #   P99 ctx ≈ 7751 chars ≈ 5742 tok
        # Budget breakdown per request:
        #   system prompt ~1100 tok + retrieved context ~2500–4500 tok
        #   + history (trimmed) + current query
        # 2560 was silently truncating every request above ~1900 tok of
        # retrieved context. 5120 covers P99 (5742 tok) with ~10% headroom.
        # Don't raise further without evidence — larger ctx has real prefill
        # cost on command-r:35b.
        # num_predict 256: REGLA 5 (web prompt) pide 2-4 oraciones, lo que
        # aterriza en ~80-180 tok; con la nueva REGLA 4 (preservar URLs) las
        # respuestas pueden llegar a ~200 tok. Cap previo de 60 cortaba a
        # media oración. Decode budget: qwen2.5:7b ≈ 30 tok/s → 256 tok ≈ 8s.
        # Prefix caching is automatic in ollama when system prompts match
        # byte-for-byte across requests — see _WEB_SYSTEM_PROMPT for the
        # stable definition that keeps the cache warm. cache_prompt was
        # tested as an explicit option but is not in ollama's accepted set
        # and made some calls hang silently — leave it out.
        _WEB_CHAT_OPTIONS = {
            **CHAT_OPTIONS,
            "num_ctx": 4096,
            "num_predict": 256,
        }

        yield _sse("status", {"stage": "generating"})

        # MPS flush sleep removed 2026-04-18: the 20-40s GPU contention
        # it defended against was actually caused by num_ctx mismatch
        # (5120 vs 4096 loaded) forcing ollama KV reinit — fixed at commit
        # 79f6b8e. Without the sleep, warm /api/chat drops by 200ms cleanly.
        # If prefill variance returns to 20s+ range, first suspect num_ctx
        # drift, not MPS contention.

        parts: list[str] = []
        stripper = _InlineCitationStripper()
        _web_model = _resolve_web_chat_model()
        print(
            f"[chat-model-keepalive] model={_web_model} keep_alive={OLLAMA_KEEP_ALIVE}"
            f" num_ctx={_WEB_CHAT_OPTIONS['num_ctx']}",
            flush=True,
        )
        _t_llm_start = time.perf_counter()
        _first_token_logged = False
        try:
            for chunk in ollama.chat(
                model=_web_model,
                messages=messages,
                options=_WEB_CHAT_OPTIONS,
                stream=True,
                keep_alive=OLLAMA_KEEP_ALIVE,
            ):
                delta = chunk.message.content or ""
                if not delta:
                    continue
                if not _first_token_logged:
                    _t_first_token = time.perf_counter()
                    _first_token_logged = True
                filtered = stripper.feed(delta)
                if filtered:
                    parts.append(filtered)
                    yield _sse("token", {"delta": filtered})
            # Flush any tail that was held back waiting for a close-paren.
            tail = stripper.flush()
            if tail:
                parts.append(tail)
                yield _sse("token", {"delta": tail})
        except Exception as exc:
            yield _sse("error", {"message": f"LLM falló: {exc}"})
            return

        full = "".join(parts)
        _t_done = time.perf_counter()

        # Timing breakdown for diagnostics
        _t_reform_ms = int((_t_reform_end - _t_reform_start) * 1000)
        _t_retrieve_ms = int((_t_retrieve_end - _t_retrieve_start) * 1000)
        _t_wa_ms = _t_wa_wait_ms
        _t_llm_prefill_ms = int((_t_first_token - _t_llm_start) * 1000) if _first_token_logged else -1
        _t_llm_decode_ms = int((_t_done - (_t_first_token if _first_token_logged else _t_llm_start)) * 1000)
        _t_total_ms = int((_t_done - _t0) * 1000)
        _t_ttft_ms = int(((_t_first_token if _first_token_logged else _t_done) - _t0) * 1000)
        _tok_count = len(full.split())
        _q_words = len(question.split())
        _reform_outcome = (
            "concat" if _reform_used_concat
            else "rewritten" if _reform_fired
            else "skipped"
        )
        print(
            f"[chat-timing] model={_resolve_web_chat_model()} retrieve={_t_retrieve_ms}ms "
            f"reform={_t_reform_ms}ms reform_outcome={_reform_outcome} q_words={_q_words} "
            f"wa={_t_wa_ms}ms llm_prefill={_t_llm_prefill_ms}ms "
            f"llm_decode={_t_llm_decode_ms}ms ttft={_t_ttft_ms}ms "
            f"total={_t_total_ms}ms ctx_chars={_ctx_chars} tokens≈{_tok_count} "
            f"confidence={result['confidence']:.3f} "
            f"variants={len(result.get('query_variants',[]))}",
            flush=True,
        )

        turn_id = new_turn_id()
        append_turn(sess, {
            "q": question,
            "a": full,
            "paths": [m.get("file", "") for m in result["metas"]],
            "top_score": round(float(result["confidence"]), 3),
            "turn_id": turn_id,
        })
        save_session(sess)
        log_query_event({
            "cmd": "web",
            "turn_id": turn_id,
            "session": sess["id"],
            "q": question,
            "paths": [m.get("file", "") for m in result["metas"]],
            "scores": [round(float(s), 2) for s in result["scores"]],
            "top_score": round(float(result["confidence"]), 2),
        })

        yield _sse("done", {
            "turn_id": turn_id,
            "top_score": round(float(result["confidence"]), 3),
            "total_ms": _t_total_ms,
            "retrieve_ms": _t_retrieve_ms,
            "ttft_ms": _t_ttft_ms,
            "llm_ms": _t_llm_decode_ms,
        })

        threading.Thread(
            target=_persist_conversation_turn,
            args=(
                VAULT_PATH,
                sess["id"],
                req.question,
                full,
                result["metas"],
                result["scores"],
                result["confidence"],
            ),
            daemon=True,
            name=f"conv-writer-{turn_id[:8]}",
        ).start()

        # Cache store — sólo queries sin history (no follow-ups) y con
        # respuesta no-vacía. Keya con el vault_chunks count computado en el
        # cache-get path arriba (mismo scope/model/vault state → misma key).
        if not history and full.strip():
            try:
                _sources_items = [
                    {**_source_payload(m, s), "bar": _score_bar(float(s))}
                    for m, s in zip(result["metas"], result["scores"])
                ]
                _chat_cache_put(_cache_key, {
                    "text": full,
                    "sources_items": _sources_items,
                    "top_score": round(float(result["confidence"]), 3),
                })
                print(f"[chat-cache] PUT {_cache_key} len={len(full)}", flush=True)
            except Exception as _cache_err:
                print(f"[chat-cache] put failed: {_cache_err}", flush=True)

    def guarded():
        """Increment/decrement the global chat-in-flight counter around
        the real generator. The home-prewarmer loop checks this counter
        and skips cycles while > 0 so chat gets exclusive ollama time.
        Decrement runs in `finally`, which executes on normal completion
        AND when the client disconnects (generator .close()).
        """
        global _CHAT_INFLIGHT
        with _CHAT_INFLIGHT_LOCK:
            _CHAT_INFLIGHT += 1
        try:
            yield from gen()
        finally:
            with _CHAT_INFLIGHT_LOCK:
                _CHAT_INFLIGHT = max(0, _CHAT_INFLIGHT - 1)

    return StreamingResponse(guarded(), media_type="text/event-stream")


@app.get("/dashboard")
def dashboard_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")


def _collect_screentime_daily(
    st_start: datetime, st_end: datetime, st_days: int,
) -> list[dict]:
    """Daily screentime buckets via a single SQL scan of knowledgeC.db.

    Replaces N separate `_collect_screentime` calls (one per day). Groups
    /app/usage sessions by local date in SQL so we touch the db once.
    """
    import sqlite3
    from rag import SCREENTIME_DB, _SCREENTIME_COCOA_OFFSET  # noqa: PLC0415

    path = SCREENTIME_DB
    daily: list[dict] = []
    if not path.is_file():
        return daily
    start_ts = st_start.timestamp() - _SCREENTIME_COCOA_OFFSET
    end_ts = st_end.timestamp() - _SCREENTIME_COCOA_OFFSET
    try:
        uri = f"file:{path}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            rows = conn.execute(
                """
                SELECT date(ZSTARTDATE + ?, 'unixepoch', 'localtime') AS day,
                       SUM(ZENDDATE - ZSTARTDATE) AS secs
                FROM ZOBJECT
                WHERE ZSTREAMNAME = '/app/usage'
                  AND ZSTARTDATE >= ?
                  AND ZSTARTDATE < ?
                  AND (ZENDDATE - ZSTARTDATE) >= 5
                GROUP BY day
                ORDER BY day
                """,
                (_SCREENTIME_COCOA_OFFSET, start_ts, end_ts),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return daily
    by_day = {day: int(secs or 0) for day, secs in rows if day}
    for d in range(st_days):
        day_start = (st_end - timedelta(days=st_days - 1 - d)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        key = day_start.strftime("%Y-%m-%d")
        daily.append({"day": key, "secs": by_day.get(key, 0)})
    return daily


# TTL cache for /api/dashboard. Polling clients hit this every minute;
# aggregating 7 JSONL files + 8 sqlite queries per hit is wasteful when
# the underlying logs barely move between polls. SSE stream pushes live
# deltas, so a 30s cache is invisible to the user.
_DASHBOARD_CACHE: dict[int, tuple[float, dict]] = {}
_DASHBOARD_TTL = 30.0


@app.get("/api/dashboard")
def dashboard_api(days: int = 30) -> dict:
    now_ts = time.time()
    hit = _DASHBOARD_CACHE.get(days)
    if hit and now_ts - hit[0] < _DASHBOARD_TTL:
        return hit[1]
    payload = _dashboard_compute(days)
    _DASHBOARD_CACHE[days] = (now_ts, payload)
    return payload


def _dashboard_compute(days: int = 30) -> dict:
    """Aggregate telemetry into dashboard metrics.

    When RAG_STATE_SQL=1: pull rows from the rag_* SQL tables. Falls back to
    JSONL scan on any exception during the cutover observation window (T10
    removes the fallback). Flag OFF = JSONL-only, unchanged behaviour.
    """
    if RAG_STATE_SQL:
        try:
            return _dashboard_compute_sql(days)
        except Exception as exc:
            _log_sql_state_error("dashboard_sql_compute_failed", err=repr(exc))
            # Fall through to JSONL during the 7-day cutover.
    return _dashboard_compute_jsonl(days)


# ── Dashboard event-row helpers ─────────────────────────────────────────────
# The SQL and JSONL readers converge on a common "event dict" shape (matches
# the JSONL line schema) before the aggregator runs, so only the data source
# differs. `_dashboard_aggregate()` is the single source of truth for every
# field the dashboard JS consumes.


def _dashboard_sql_row_to_event(row, json_cols: tuple[str, ...] = ()) -> dict:
    """Rehydrate a sqlite3.Row into the JSONL-style event dict.

    - `*_json` columns whose base name (`_json` suffix stripped) is listed in
      `json_cols` are decoded and re-keyed to their JSONL equivalent (e.g.
      `paths_json` → `paths`).
    - `extra_json` is decoded and its keys merged into the top level (never
      overwriting an explicit column).
    - Other `*_json` columns are decoded in place if `json_cols` doesn't map
      them — safe default so aggregate callers never see raw JSON strings.
    """
    ev: dict = {}
    try:
        keys = row.keys()
    except Exception:
        return ev
    for k in keys:
        v = row[k]
        if v is None:
            continue
        if k == "extra_json":
            try:
                extra = json.loads(v)
                if isinstance(extra, dict):
                    for ek, ev_val in extra.items():
                        ev.setdefault(ek, ev_val)
            except Exception:
                pass
            continue
        if k.endswith("_json"):
            base = k[:-5]
            try:
                decoded = json.loads(v)
            except Exception:
                decoded = None
            target = base if base in json_cols else k
            if decoded is not None:
                ev[target] = decoded
            continue
        ev[k] = v
    return ev


def _dashboard_compute_sql(days: int = 30) -> dict:
    """SQL-path reader: fetch rag_* rows since cutoff, normalise to JSONL-shape
    event dicts, then call the shared aggregator.

    Tables read: rag_queries, rag_feedback, rag_ambient, rag_contradictions,
    rag_filing_log, rag_tune, rag_surface_log. `all_queries` is the unwindowed
    snapshot of rag_queries (used only for the total_queries_all_time KPI).
    """
    from rag import _ragvec_state_conn, _sql_query_window  # local import keeps module-load light
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_iso = cutoff.isoformat(timespec="seconds")

    # Well-earlier floor — we still need "all time" count + full feedback for
    # corrective-miss/neg-reason surfacing (JSONL path uses the full file).
    ancient_iso = "0000-01-01T00:00:00"

    with _ragvec_state_conn() as conn:
        q_rows_window = _sql_query_window(conn, "rag_queries", cutoff_iso)
        # all_queries: full history — same semantics as JSONL (unbounded scan).
        q_rows_all = _sql_query_window(conn, "rag_queries", ancient_iso)
        fb_rows_all = _sql_query_window(conn, "rag_feedback", ancient_iso)
        amb_rows = _sql_query_window(conn, "rag_ambient", cutoff_iso)
        contra_rows = _sql_query_window(conn, "rag_contradictions", cutoff_iso)
        filing_rows = _sql_query_window(conn, "rag_filing_log", cutoff_iso)
        tune_rows = _sql_query_window(conn, "rag_tune", ancient_iso)
        surface_rows = _sql_query_window(conn, "rag_surface_log", ancient_iso)

    _Q_JSON = ("variants", "paths", "scores", "filters", "bad_citations")
    _F_JSON = ("paths",)
    _C_JSON = ("contradicts",)
    _FIL_JSON = ("neighbors",)
    _T_JSON = ("baseline", "best")

    all_queries = [_dashboard_sql_row_to_event(r, _Q_JSON) for r in q_rows_all]
    queries = [_dashboard_sql_row_to_event(r, _Q_JSON) for r in q_rows_window]
    fb_entries = [_dashboard_sql_row_to_event(r, _F_JSON) for r in fb_rows_all]
    # rag_ambient stores extra fields in `payload_json`, not `extra_json`.
    ambient_entries: list[dict] = []
    for r in amb_rows:
        ev = {}
        try:
            keys = r.keys()
        except Exception:
            keys = []
        for k in keys:
            v = r[k]
            if v is None:
                continue
            if k == "payload_json":
                try:
                    payload = json.loads(v)
                    if isinstance(payload, dict):
                        for pk, pv in payload.items():
                            ev.setdefault(pk, pv)
                except Exception:
                    pass
                continue
            ev[k] = v
        ambient_entries.append(ev)
    contra_entries = [_dashboard_sql_row_to_event(r, _C_JSON) for r in contra_rows]
    # Flatten `singles`/`chains` back onto each tune entry so the JSONL aggregator
    # sees `baseline`/`best` dicts as-is.
    tune_entries = [_dashboard_sql_row_to_event(r, _T_JSON) for r in tune_rows]
    # Filing rows — add an implicit `cmd` marker so the aggregator counts them
    # as filing events (it treats every entry in filing_recent as a filing).
    filing_recent = [_dashboard_sql_row_to_event(r, _FIL_JSON) for r in filing_rows]
    # Surface rows — aggregator counts events by `cmd in {"surface_pair","surface_run"}`,
    # and rag_surface_log preserves `cmd` as a column.
    surface_entries = [_dashboard_sql_row_to_event(r) for r in surface_rows]

    return _dashboard_aggregate(
        queries=queries,
        all_queries=all_queries,
        fb_entries=fb_entries,
        ambient_entries=ambient_entries,
        contra_entries=contra_entries,
        filing_recent=filing_recent,
        tune_entries=tune_entries,
        surface_entries=surface_entries,
        days=days,
    )


def _dashboard_compute_jsonl(days: int = 30) -> dict:
    """Aggregate all JSONL logs into dashboard metrics."""
    data_dir = Path.home() / ".local/share/obsidian-rag"
    cutoff = datetime.now() - timedelta(days=days)

    def _read_jsonl(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    def _ts_local(e: dict) -> datetime | None:
        raw = e.get("ts") or e.get("timestamp")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except Exception:
            return None

    all_queries = _read_jsonl(data_dir / "queries.jsonl")
    queries = [e for e in all_queries if (t := _ts_local(e)) and t >= cutoff]
    fb_entries = _read_jsonl(data_dir / "feedback.jsonl")
    ambient_entries = _read_jsonl(data_dir / "ambient.jsonl")
    contra_entries = _read_jsonl(data_dir / "contradictions.jsonl")
    surface_entries = _read_jsonl(data_dir / "surface.jsonl")
    filing_entries = _read_jsonl(data_dir / "filing.jsonl")
    filing_recent = [e for e in filing_entries if (t := _ts_local(e)) and t >= cutoff]
    tune_entries = _read_jsonl(data_dir / "tune.jsonl")

    return _dashboard_aggregate(
        queries=queries,
        all_queries=all_queries,
        fb_entries=fb_entries,
        ambient_entries=ambient_entries,
        contra_entries=contra_entries,
        filing_recent=filing_recent,
        tune_entries=tune_entries,
        surface_entries=surface_entries,
        days=days,
    )


def _dashboard_aggregate(
    *,
    queries: list[dict],
    all_queries: list[dict],
    fb_entries: list[dict],
    ambient_entries: list[dict],
    contra_entries: list[dict],
    filing_recent: list[dict],
    tune_entries: list[dict],
    surface_entries: list[dict],
    days: int,
) -> dict:
    """Build the full dashboard payload from normalised event dicts.

    This is the shared body both the JSONL and SQL paths call with identical
    semantics. The dashboard JS key-contract (`queries_per_day`, `hours`,
    `cmds`, `sources`, `score_distribution`, `hot_topics`, `chat_keywords`,
    `feedback`, etc.) is fixed here — don't drop keys without updating
    web/static/dashboard.js.
    """
    import statistics
    from collections import defaultdict

    data_dir = Path.home() / ".local/share/obsidian-rag"
    cutoff = datetime.now() - timedelta(days=days)

    def _ts(e: dict) -> datetime | None:
        raw = e.get("ts") or e.get("timestamp")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except Exception:
            return None

    def _pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = int(len(s) * p / 100)
        return round(s[min(idx, len(s) - 1)], 3)

    # ── Queries ──────────────────────────────────────────────────────
    scores = [float(e["top_score"]) for e in queries if isinstance(e.get("top_score"), (int, float))]
    t_retrieves = [float(e["t_retrieve"]) for e in queries if isinstance(e.get("t_retrieve"), (int, float))]
    t_gens = [float(e["t_gen"]) for e in queries if isinstance(e.get("t_gen"), (int, float)) and e["t_gen"] > 0]
    t_totals = [
        float(e["t_retrieve"]) + float(e["t_gen"])
        for e in queries
        if isinstance(e.get("t_retrieve"), (int, float)) and isinstance(e.get("t_gen"), (int, float)) and e["t_gen"] > 0
    ]

    # Per-day counts
    queries_per_day: dict[str, int] = defaultdict(int)
    latency_per_day: dict[str, list[float]] = defaultdict(list)
    for e in queries:
        t = _ts(e)
        if t:
            day = t.strftime("%Y-%m-%d")
            queries_per_day[day] += 1
            tr = e.get("t_retrieve")
            tg = e.get("t_gen")
            if isinstance(tr, (int, float)) and isinstance(tg, (int, float)) and tg > 0:
                latency_per_day[day].append(float(tr) + float(tg))

    # Hourly heatmap
    hours = defaultdict(int)
    for e in queries:
        t = _ts(e)
        if t:
            hours[t.hour] += 1

    # Command breakdown
    cmds = defaultdict(int)
    for e in queries:
        cmds[e.get("cmd", "?")] += 1

    # Source breakdown. `cmd: serve*` is exclusively the rag-serve fast path
    # (only the WhatsApp listener uses it), so any serve event is whatsapp
    # regardless of session prefix.
    sources = defaultdict(int)
    for e in queries:
        sid = e.get("session") or ""
        cmd = e.get("cmd") or ""
        if sid.startswith("wa:") or cmd.startswith("serve"):
            sources["whatsapp"] += 1
        elif sid.startswith("web:"):
            sources["web"] += 1
        elif sid.startswith("tg:"):
            sources["telegram"] += 1
        else:
            sources["cli"] += 1

    # Score distribution (10 buckets)
    score_buckets = [0] * 10
    for s in scores:
        idx = min(int(max(0.0, min(s, 1.0)) * 10), 9)
        score_buckets[idx] += 1

    # Hot topics
    topics: dict[str, int] = defaultdict(int)
    for e in queries:
        q = e.get("q", "")
        if q:
            words = q.lower().split()[:3]
            topics[" ".join(words)] += 1
    hot_topics = sorted(topics.items(), key=lambda x: -x[1])[:15]

    # Chat keyword cloud — per-word frequency over user-initiated queries
    # (chat/web/whatsapp), not ambient/eval synthetic traffic. Tokenises
    # on word boundaries, strips accents for collation (so "qué" and "que"
    # collapse), drops stopwords + words shorter than 3 chars. Returns the
    # top 60 for the UI cloud; the renderer sizes by log(count).
    _ES_EN_STOPWORDS = {
        "que", "de", "la", "el", "en", "y", "a", "los", "las", "un", "una",
        "del", "se", "por", "con", "para", "mi", "mis", "me", "tu", "tus",
        "su", "sus", "al", "lo", "le", "les", "es", "esta", "este", "esto",
        "esa", "ese", "eso", "como", "cuando", "donde", "quien", "cual",
        "cuales", "sobre", "entre", "desde", "hasta", "sin", "pero", "o",
        "u", "ni", "mas", "muy", "ya", "si", "no", "tambien", "hay", "fue",
        "son", "soy", "ser", "estoy", "estan", "estar", "he", "has", "ha",
        "hemos", "han", "puedo", "puede", "pueden", "podria", "podrian",
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had", "and", "or", "but",
        "if", "then", "else", "when", "where", "what", "which", "who",
        "how", "why", "to", "of", "in", "on", "for", "with", "from",
        "by", "as", "at", "into", "out", "up", "down", "off", "over",
        "this", "that", "these", "those", "it", "its", "i", "you", "he",
        "she", "we", "they", "me", "my", "your", "our", "their",
        "algo", "nada", "todo", "todos", "otra", "otro", "otros", "otras",
        "solo", "sólo", "tan", "tanto", "tener", "tiene", "tienes",
        "hacer", "hace", "haces", "hice", "hicimos", "voy", "vas", "va",
        "vamos", "van", "dame", "darme", "tengo", "tenes", "tenés",
        "estas", "estos", "esas", "esos", "cuantos", "cuantas",
        "hola", "gracias", "porfa", "plis", "oka", "dale", "listo",
        "quiero", "quieres", "quisiera", "necesito", "necesitas",
        "decime", "dime", "decir", "dijo", "dije", "saber", "sabes",
    }
    _KEYWORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñü0-9][A-Za-zÁÉÍÓÚÑáéíóúñü0-9_\-']{2,}")
    def _fold(s: str) -> str:
        import unicodedata
        nfd = unicodedata.normalize("NFD", s.lower())
        return "".join(ch for ch in nfd if not unicodedata.combining(ch))
    keyword_counts: dict[str, int] = defaultdict(int)
    for e in queries:
        sid = e.get("session") or ""
        cmd = e.get("cmd") or ""
        # User-initiated surfaces only — exclude eval/ambient/morning/etc.
        if not (
            sid.startswith(("wa:", "web:", "tg:"))
            or cmd in ("query", "chat", "web", "serve", "serve-web")
        ):
            continue
        q = (e.get("q") or "").strip()
        if not q:
            continue
        for match in _KEYWORD_RE.findall(q):
            folded = _fold(match)
            if len(folded) < 3 or folded in _ES_EN_STOPWORDS:
                continue
            keyword_counts[folded] += 1
    chat_keywords = [
        {"word": w, "count": c}
        for w, c in sorted(keyword_counts.items(), key=lambda x: -x[1])[:60]
        if c >= 2  # skip hapax — noise dominates
    ]

    # Latency time series (daily p50)
    latency_ts = {}
    for day, vals in sorted(latency_per_day.items()):
        latency_ts[day] = {
            "p50": round(statistics.median(vals), 2) if vals else 0,
            "p95": _pct(vals, 95),
            "count": len(vals),
        }

    # ── Health metrics ───────────────────────────────────────────────
    # Only count queries that actually go through retrieval (have scores)
    retrieval_queries = [e for e in queries if isinstance(e.get("top_score"), (int, float)) and e.get("cmd") in ("query", "chat", "web", None)]
    n_retrieval = len(retrieval_queries)

    gated = sum(1 for e in queries if e.get("gated_low_confidence"))
    gate_rate = round(gated / n_retrieval * 100, 1) if n_retrieval else 0

    bad_citation_queries = sum(1 for e in queries if e.get("bad_citations"))
    bad_citation_rate = round(bad_citation_queries / n_retrieval * 100, 1) if n_retrieval else 0
    bad_citation_total = sum(len(e.get("bad_citations", [])) for e in queries)

    # Score quality bands
    retrieval_scores = [float(e["top_score"]) for e in retrieval_queries if isinstance(e.get("top_score"), (int, float))]
    n_scored = len(retrieval_scores)
    score_high = sum(1 for s in retrieval_scores if s >= 0.3)     # strong match
    score_mid = sum(1 for s in retrieval_scores if 0.05 <= s < 0.3)  # acceptable
    score_low = sum(1 for s in retrieval_scores if s < 0.05)      # weak/miss

    # Score trend per day (daily mean)
    score_per_day: dict[str, list[float]] = defaultdict(list)
    for e in retrieval_queries:
        t = _ts(e)
        s = e.get("top_score")
        if t and isinstance(s, (int, float)):
            score_per_day[t.strftime("%Y-%m-%d")].append(float(s))
    score_trend = {
        day: round(statistics.mean(vals), 3)
        for day, vals in sorted(score_per_day.items())
    }

    # Latency ratio (retrieve vs generate dominance)
    avg_ret = statistics.mean(t_retrieves) if t_retrieves else 0
    avg_gen = statistics.mean(t_gens) if t_gens else 0
    retrieve_pct = round(avg_ret / (avg_ret + avg_gen) * 100, 1) if (avg_ret + avg_gen) > 0 else 0

    # Answer length trend (is the LLM being verbose or terse?)
    answer_lens = [e["answer_len"] for e in queries if isinstance(e.get("answer_len"), (int, float)) and e["answer_len"] > 0]
    avg_answer_len = round(statistics.mean(answer_lens)) if answer_lens else 0

    health = {
        "retrieval_queries": n_retrieval,
        "gate_rate": gate_rate,
        "gated_count": gated,
        "bad_citation_rate": bad_citation_rate,
        "bad_citation_queries": bad_citation_queries,
        "bad_citation_total": bad_citation_total,
        "score_high": score_high,
        "score_mid": score_mid,
        "score_low": score_low,
        "score_high_pct": round(score_high / n_scored * 100, 1) if n_scored else 0,
        "score_mid_pct": round(score_mid / n_scored * 100, 1) if n_scored else 0,
        "score_low_pct": round(score_low / n_scored * 100, 1) if n_scored else 0,
        "score_trend": score_trend,
        "retrieve_pct": retrieve_pct,
        "generate_pct": round(100 - retrieve_pct, 1),
        "avg_answer_len": avg_answer_len,
    }

    # ── Feedback ─────────────────────────────────────────────────────
    fb_pos = sum(1 for e in fb_entries if e.get("rating") == 1)
    fb_neg = sum(1 for e in fb_entries if e.get("rating") == -1)

    # Actionable feedback for RAG improvement (uses the *full* fb log, not windowed —
    # negative signals are rare and we want all of them).
    fb_recent = [e for e in fb_entries if (t := _ts(e)) and t >= cutoff]
    neg_path_counts: dict[str, int] = defaultdict(int)
    pos_path_counts: dict[str, int] = defaultdict(int)
    for e in fb_entries:
        rating = e.get("rating")
        for p in e.get("paths") or []:
            if rating == -1:
                neg_path_counts[p] += 1
            elif rating == 1:
                pos_path_counts[p] += 1

    # Corrective paths: user-supplied "the right note was here" — pure retrieval misses,
    # i.e. golden-eval candidates. Most actionable signal in the whole pipeline.
    corrective_misses: list[dict] = []
    for e in fb_entries:
        cp = e.get("corrective_path")
        if not cp:
            continue
        if cp in (e.get("paths") or []):
            continue
        corrective_misses.append({
            "ts": e.get("ts", ""),
            "q": (e.get("q") or "")[:140],
            "missing_path": cp,
            "retrieved": list((e.get("paths") or [])[:3]),
        })
    corrective_misses.sort(key=lambda x: x["ts"], reverse=True)

    # Recent negative reasons — qualitative why-it-failed signal.
    neg_reasons: list[dict] = []
    for e in fb_entries:
        if e.get("rating") != -1 or not e.get("reason"):
            continue
        neg_reasons.append({
            "ts": e.get("ts", ""),
            "q": (e.get("q") or "")[:140],
            "reason": e["reason"][:200],
            "paths": list((e.get("paths") or [])[:2]),
        })
    neg_reasons.sort(key=lambda x: x["ts"], reverse=True)

    # Calibration mismatch: cross feedback w/ queries by turn_id.
    # - false_confident = retrieved with high score but user said -1
    # - false_gated     = gate refused (low score) but user later said +1 / supplied corrective_path
    queries_by_turn = {e.get("turn_id"): e for e in queries if e.get("turn_id")}
    false_confident = 0
    false_gated = 0
    for fb in fb_entries:
        q_ev = queries_by_turn.get(fb.get("turn_id"))
        if not q_ev:
            continue
        ts = q_ev.get("top_score")
        if not isinstance(ts, (int, float)):
            continue
        if fb.get("rating") == -1 and ts >= 0.20:
            false_confident += 1
        if fb.get("rating") == 1 and ts < 0.05:
            false_gated += 1
        if fb.get("corrective_path") and ts < 0.05:
            false_gated += 1

    # Feedback per day (positive vs negative trend)
    fb_pos_per_day: dict[str, int] = defaultdict(int)
    fb_neg_per_day: dict[str, int] = defaultdict(int)
    for e in fb_recent:
        t = _ts(e)
        if not t:
            continue
        day = t.strftime("%Y-%m-%d")
        if e.get("rating") == 1:
            fb_pos_per_day[day] += 1
        elif e.get("rating") == -1:
            fb_neg_per_day[day] += 1

    feedback_actionable = {
        "total": len(fb_entries),
        "recent_pos": sum(1 for e in fb_recent if e.get("rating") == 1),
        "recent_neg": sum(1 for e in fb_recent if e.get("rating") == -1),
        "net_satisfaction": (
            round((fb_pos - fb_neg) / (fb_pos + fb_neg) * 100, 1)
            if (fb_pos + fb_neg) > 0 else None
        ),
        "top_negative_paths": [
            {"path": p, "count": c, "pos_count": pos_path_counts.get(p, 0)}
            for p, c in sorted(neg_path_counts.items(), key=lambda x: -x[1])[:8]
        ],
        "top_positive_paths": [
            {"path": p, "count": c}
            for p, c in sorted(pos_path_counts.items(), key=lambda x: -x[1])[:8]
        ],
        "corrective_misses": corrective_misses[:10],
        "n_corrective_misses": len(corrective_misses),
        "negative_reasons": neg_reasons[:10],
        "false_confident": false_confident,
        "false_gated": false_gated,
        "per_day_pos": dict(sorted(fb_pos_per_day.items())),
        "per_day_neg": dict(sorted(fb_neg_per_day.items())),
    }

    # ── Ambient ──────────────────────────────────────────────────────
    # ambient_entries is already provided by the caller; filter to window here.
    ambient_recent = [e for e in ambient_entries if (t := _ts(e)) and t >= cutoff]
    ambient_wikilinks = sum(e.get("wikilinks_applied", 0) for e in ambient_recent)
    ambient_per_day: dict[str, int] = defaultdict(int)
    for e in ambient_recent:
        t = _ts(e)
        if t:
            ambient_per_day[t.strftime("%Y-%m-%d")] += 1

    # ── Contradictions ───────────────────────────────────────────────
    contra_recent = [e for e in contra_entries if (t := _ts(e)) and t >= cutoff]
    contra_found = [e for e in contra_recent if e.get("contradicts") and not e.get("skipped")]
    contra_per_day: dict[str, int] = defaultdict(int)
    for e in contra_found:
        t = _ts(e)
        if t:
            contra_per_day[t.strftime("%Y-%m-%d")] += 1

    # ── Surface ──────────────────────────────────────────────────────
    surface_pairs = [e for e in surface_entries if e.get("cmd") == "surface_pair"]
    surface_runs = [e for e in surface_entries if e.get("cmd") == "surface_run"]

    # ── Filing ───────────────────────────────────────────────────────
    # filing_recent is already windowed by the caller.

    # ── Tune ─────────────────────────────────────────────────────────
    tune_history = []
    for e in tune_entries:
        t = _ts(e)
        tune_history.append({
            "ts": e.get("ts"),
            "samples": e.get("samples"),
            "baseline_hit": e.get("baseline", {}).get("hit"),
            "baseline_mrr": e.get("baseline", {}).get("mrr"),
            "best_hit": e.get("best", {}).get("hit"),
            "best_mrr": e.get("best", {}).get("mrr"),
            "delta": e.get("delta"),
        })

    # ── Sessions ─────────────────────────────────────────────────────
    sessions_dir = data_dir / "sessions"
    session_count = 0
    session_turns_total = 0
    wa_sessions = 0
    if sessions_dir.is_dir():
        for sf in sessions_dir.glob("*.json"):
            try:
                sess = json.loads(sf.read_text(encoding="utf-8"))
                session_count += 1
                turns = sess.get("turns", [])
                session_turns_total += len(turns)
                if sess.get("id", "").startswith("wa:"):
                    wa_sessions += 1
            except Exception:
                continue

    # ── Screen Time (macOS knowledgeC.db) ────────────────────────────
    # Cap at 7 days — dashboard day-pickers of 30/90 don't help: knowledge
    # store only keeps ~7-10d of /app/usage before CoreDuet aggregates it
    # away. Querying a wider window returns gaps, not more data.
    st_days = min(max(int(days), 1), 7)
    st_end = datetime.now()
    st_start = st_end - timedelta(days=st_days)
    screentime_agg = _collect_screentime(st_start, st_end)
    screentime_daily = _collect_screentime_daily(st_start, st_end, st_days)
    screentime_total_str = (
        _fmt_hm(int(screentime_agg.get("total_secs") or 0))
        if screentime_agg.get("available") else ""
    )

    # ── Index stats ──────────────────────────────────────────────────
    index_stats = {}
    try:
        col = get_db()
        n_chunks = col.count()
        corpus = _load_corpus(col)
        pr = get_pagerank(col)
        top_pr = sorted(pr.items(), key=lambda x: -x[1])[:5] if pr else []
        index_stats = {
            "chunks": n_chunks,
            "notes": len(corpus.get("title_to_paths", {})),
            "tags": len(corpus.get("tags", set())),
            "folders": len(corpus.get("folders", set())),
            "top_pagerank": [{"path": p, "score": round(s, 4)} for p, s in top_pr],
        }
    except Exception:
        pass

    return {
        "period_days": days,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "kpis": {
            "total_queries": len(queries),
            "total_queries_all_time": len(all_queries),
            "avg_latency": round(statistics.mean(t_totals), 2) if t_totals else None,
            "avg_retrieve": round(statistics.mean(t_retrieves), 2) if t_retrieves else None,
            "avg_generate": round(statistics.mean(t_gens), 2) if t_gens else None,
            "feedback_positive": fb_pos,
            "feedback_negative": fb_neg,
            "sessions": session_count,
            "wa_sessions": wa_sessions,
            "ambient_hooks": len(ambient_recent),
            "ambient_wikilinks": ambient_wikilinks,
            "contradictions_found": len(contra_found),
            "surface_pairs": len(surface_pairs),
            "filings": len(filing_recent),
        },
        "health": health,
        "index": index_stats,
        "queries_per_day": dict(sorted(queries_per_day.items())),
        "latency_per_day": latency_ts,
        "hours": {str(h): hours.get(h, 0) for h in range(24)},
        "cmds": dict(sorted(cmds.items(), key=lambda x: -x[1])),
        "sources": dict(sources),
        "score_distribution": score_buckets,
        "score_stats": {
            "p50": _pct(scores, 50),
            "p95": _pct(scores, 95),
            "min": round(min(scores), 3) if scores else 0,
            "max": round(max(scores), 3) if scores else 0,
        },
        "latency_stats": {
            "retrieve_p50": _pct(t_retrieves, 50),
            "retrieve_p95": _pct(t_retrieves, 95),
            "generate_p50": _pct(t_gens, 50),
            "generate_p95": _pct(t_gens, 95),
            "total_p50": _pct(t_totals, 50),
            "total_p95": _pct(t_totals, 95),
        },
        "hot_topics": [{"topic": t, "count": c} for t, c in hot_topics],
        "chat_keywords": chat_keywords,
        "ambient_per_day": dict(sorted(ambient_per_day.items())),
        "contradictions_per_day": dict(sorted(contra_per_day.items())),
        "tune_history": tune_history,
        "surface_runs": len(surface_runs),
        "filing_confidence": [round(e.get("confidence", 0), 2) for e in filing_recent],
        "feedback": feedback_actionable,
        "screentime": {
            "available": screentime_agg.get("available", False),
            "window_days": st_days,
            "total_secs": int(screentime_agg.get("total_secs") or 0),
            "total_label": screentime_total_str,
            "top_apps": screentime_agg.get("top_apps") or [],
            "categories": screentime_agg.get("categories") or {},
            "daily": screentime_daily,
        },
    }


# ── Real-time stream ─────────────────────────────────────────────────────────
# Tail JSONL logs and emit SSE events. The browser polls /api/dashboard for the
# heavy aggregations; this endpoint pushes deltas as they happen so KPIs and
# the live feed update without waiting for the next poll.

_STREAM_FILES: dict[str, str] = {
    "query": "queries.jsonl",
    "feedback": "feedback.jsonl",
    "ambient": "ambient.jsonl",
    "contradiction": "contradictions.jsonl",
}


@app.get("/api/dashboard/stream")
async def dashboard_stream() -> StreamingResponse:
    """SSE: tail JSONL logs and push new events as they arrive."""
    data_dir = Path.home() / ".local/share/obsidian-rag"
    paths = {kind: data_dir / fname for kind, fname in _STREAM_FILES.items()}
    # Start at current EOF so the client only sees new events.
    offsets: dict[str, int] = {
        kind: (p.stat().st_size if p.is_file() else 0) for kind, p in paths.items()
    }

    async def gen():
        last_heartbeat = time.time()
        # Initial hello so the client can flip the indicator immediately.
        yield _sse("hello", {"t": time.time(), "tracking": list(paths.keys())})
        try:
            while True:
                for kind, path in paths.items():
                    if not path.is_file():
                        continue
                    try:
                        cur_size = path.stat().st_size
                    except OSError:
                        continue
                    prev = offsets.get(kind, 0)
                    if cur_size < prev:
                        # Truncated/rotated — reset and skip.
                        offsets[kind] = cur_size
                        continue
                    if cur_size == prev:
                        continue
                    try:
                        with path.open("rb") as f:
                            f.seek(prev)
                            chunk = f.read(cur_size - prev)
                    except OSError:
                        continue
                    offsets[kind] = cur_size
                    for line in chunk.decode("utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except Exception:
                            continue
                        yield _sse(kind, _stream_payload(kind, ev))
                now = time.time()
                if now - last_heartbeat >= 15:
                    yield _sse("heartbeat", {"t": now})
                    last_heartbeat = now
                await asyncio.sleep(1.5)
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _stream_payload(kind: str, ev: dict) -> dict:
    """Trim raw log entries to the fields the dashboard needs."""
    if kind == "query":
        sid = ev.get("session") or ""
        # Whatsapp listener uses session ids like `wa:<jid>` OR custom strings
        # like `<runtime>:<vault>` (rag serve from whatsapp-listener) — treat
        # `serve` cmd as whatsapp by default since that's the only consumer.
        cmd = ev.get("cmd") or ""
        if sid.startswith("wa:") or cmd.startswith("serve"):
            source = "whatsapp"
        elif sid.startswith("web:"):
            source = "web"
        elif sid.startswith("tg:"):
            source = "telegram"
        else:
            source = "cli"
        tr = ev.get("t_retrieve")
        tg = ev.get("t_gen")
        latency = (
            round(float(tr) + float(tg), 2)
            if isinstance(tr, (int, float)) and isinstance(tg, (int, float))
            else None
        )
        # phase: received → in-flight, error → failed, anything else → completed
        if cmd == "serve.received":
            phase = "in_flight"
        elif cmd == "serve.error":
            phase = "error"
        else:
            phase = "done"
        return {
            "ts": ev.get("ts"),
            "q": (ev.get("q") or "")[:160],
            "score": ev.get("top_score"),
            "latency": latency,
            "source": source,
            "cmd": cmd,
            "phase": phase,
            "error": (ev.get("error") or "")[:200] if ev.get("error") else None,
            "gated": bool(ev.get("gated_low_confidence")),
            "bad_citations": len(ev.get("bad_citations") or []),
            "n_paths": len(ev.get("paths") or []),
        }
    if kind == "feedback":
        return {
            "ts": ev.get("ts"),
            "rating": ev.get("rating"),
            "q": (ev.get("q") or "")[:160],
            "reason": (ev.get("reason") or "")[:200],
            "corrective_path": ev.get("corrective_path"),
            "n_paths": len(ev.get("paths") or []),
        }
    if kind == "ambient":
        return {
            "ts": ev.get("ts"),
            "path": ev.get("path"),
            "wikilinks_applied": ev.get("wikilinks_applied", 0),
        }
    if kind == "contradiction":
        return {
            "ts": ev.get("ts"),
            "path": ev.get("path"),
            "contradicts": ev.get("contradicts"),
            "skipped": bool(ev.get("skipped")),
        }
    return ev


# ── RAG-system memory sampler ──────────────────────────────────────────
# Tracks RSS of processes that are part of the rag stack we built:
#   rag        — obsidian-rag python (watch, morning, today, digest, web,
#                mcp, chat, query, launchd-spawned jobs, their children)
#   ollama     — ollama serve + per-model runners (LLM + embeddings)
#   sqlite-vec — sqlite-vec-gui streamlit inspector pointed at ragvec.db
#   whatsapp   — whatsapp-bridge, whatsapp-listener, whatsapp-mcp, vault-sync
#
# System-wide processes (browser, Claude Code, unrelated python) are
# intentionally excluded — this chart answers "how much RAM does OUR
# stack use?", not "what's eating the machine?".
#
# Zero-dep: parses `ps -axo rss=,command=` on the full command line so
# we can match interpreter path + script path (short `comm=` collapses
# everything to `python3`). RSS on macOS double-counts shared memory
# across helper processes, so absolute numbers overstate real usage;
# trends are what the chart is for.
from collections import deque

_MEMORY_STATE_PATH = Path.home() / ".local/share/obsidian-rag/rag_memory.jsonl"
_MEMORY_BUFFER_MAX = 1440  # 24h @ 60s
_MEMORY_SAMPLE_INTERVAL = 60.0
_MEMORY_BUFFER: deque = deque(maxlen=_MEMORY_BUFFER_MAX)
_MEMORY_LOCK = threading.Lock()

_MEMORY_CATEGORIES = ("rag", "ollama", "sqlite-vec", "whatsapp")

# Order matters: first regex to match claims the process. `whatsapp`
# goes before `rag` because whatsapp-mcp lives under a path that also
# matches obsidian-rag venvs in edge cases; `ollama` before `rag`
# likewise for safety.
_RAG_PROC_MATCHERS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("whatsapp",   re.compile(r"whatsapp-(bridge|listener|vault-sync|mcp)")),
    ("ollama",     re.compile(r"(?:^|/)ollama(?:\s|$)")),
    ("sqlite-vec", re.compile(r"sqlite-vec-gui\b")),
    ("rag",        re.compile(r"obsidian-rag")),
)


def _classify_rag_proc(cmd: str) -> str | None:
    for cat, rx in _RAG_PROC_MATCHERS:
        if rx.search(cmd):
            return cat
    return None


def _rag_proc_label(cmd: str, cat: str) -> str:
    """Friendly, aggregatable label for the `top` list."""
    if cat == "rag":
        if "obsidian-rag-mcp" in cmd:
            return "obsidian-rag-mcp"
        if "web/server.py" in cmd:
            return "rag web"
        m = re.search(r"/rag\s+(\w[\w-]*)", cmd)
        if m:
            return f"rag {m.group(1)}"
        if "resource_tracker" in cmd:
            return "rag (resource_tracker)"
        return "rag (python)"
    if cat == "ollama":
        if "ollama runner" in cmd:
            m = re.search(r"sha256-([0-9a-f]{8,})", cmd)
            return f"ollama runner ({m.group(1)[:8]})" if m else "ollama runner"
        if "ollama serve" in cmd:
            return "ollama serve"
        return "ollama"
    if cat == "sqlite-vec":
        return "sqlite-vec-gui"
    if cat == "whatsapp":
        for tag in ("whatsapp-bridge", "whatsapp-listener", "whatsapp-vault-sync", "whatsapp-mcp"):
            if tag in cmd:
                return tag
        return "whatsapp"
    return cmd.split()[0].rsplit("/", 1)[-1]


def _read_vm_stat() -> dict:
    """Parse `vm_stat` to surface pressure + free pages."""
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=2, check=False,
        ).stdout
    except Exception:
        return {}
    page_size = 4096
    stats: dict[str, float] = {}
    for line in out.splitlines():
        if "page size of" in line:
            m = re.search(r"page size of (\d+)", line)
            if m:
                page_size = int(m.group(1))
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip().rstrip(".")
        try:
            stats[key.strip()] = float(val)
        except ValueError:
            continue
    mb = lambda pages: round(pages * page_size / (1024 * 1024), 1)
    return {
        "free_mb": mb(stats.get("Pages free", 0)),
        "active_mb": mb(stats.get("Pages active", 0)),
        "inactive_mb": mb(stats.get("Pages inactive", 0)),
        "wired_mb": mb(stats.get("Pages wired down", 0)),
        "compressed_mb": mb(stats.get("Pages occupied by compressor", 0)),
    }


def _sample_memory() -> dict | None:
    """One sample: RSS totals per rag-stack category + top processes + vm_stat."""
    try:
        out = subprocess.run(
            ["ps", "-axo", "rss=,command="],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except Exception:
        return None

    by_cat: dict[str, float] = {k: 0.0 for k in _MEMORY_CATEGORIES}
    procs: dict[str, dict] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            rss_kb = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1]
        cat = _classify_rag_proc(cmd)
        if cat is None:
            continue
        rss_mb = rss_kb / 1024.0
        by_cat[cat] += rss_mb
        label = _rag_proc_label(cmd, cat)
        slot = procs.get(label)
        if slot is None:
            procs[label] = {"name": label, "mb": rss_mb, "cat": cat}
        else:
            slot["mb"] += rss_mb

    total = round(sum(by_cat.values()), 1)
    by_cat_rounded = {k: round(v, 1) for k, v in by_cat.items()}
    top = sorted(procs.values(), key=lambda d: d["mb"], reverse=True)[:10]
    top_list = [{"name": d["name"], "mb": round(d["mb"], 1), "cat": d["cat"]} for d in top]

    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "total_mb": total,
        "by_category": by_cat_rounded,
        "top": top_list,
        "vm": _read_vm_stat(),
    }


def _memory_load_history() -> None:
    """Warm the ring buffer from the JSONL sidecar (last N lines)."""
    if not _MEMORY_STATE_PATH.exists():
        return
    try:
        with _MEMORY_STATE_PATH.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()[-_MEMORY_BUFFER_MAX:]
    except Exception:
        return
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            _MEMORY_BUFFER.append(json.loads(raw))
        except Exception:
            continue


def _memory_persist(sample: dict) -> None:
    if RAG_STATE_SQL:
        try:
            with _ragvec_state_conn() as conn:
                _sql_append_event(conn, "rag_memory_metrics",
                                   _map_memory_row(sample))
            return
        except Exception as exc:
            _log_sql_state_error("memory_sql_write_failed", err=repr(exc))
    try:
        _MEMORY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _MEMORY_STATE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample, separators=(",", ":")) + "\n")
    except Exception:
        pass


def _memory_trim_file() -> None:
    """Keep the JSONL file bounded at ~2x buffer so it doesn't grow forever."""
    try:
        if not _MEMORY_STATE_PATH.exists():
            return
        with _MEMORY_STATE_PATH.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        cap = _MEMORY_BUFFER_MAX * 2
        if len(lines) <= cap:
            return
        tail = lines[-cap:]
        with _MEMORY_STATE_PATH.open("w", encoding="utf-8") as fh:
            fh.writelines(tail)
    except Exception:
        pass


@app.on_event("startup")
def _start_memory_sampler() -> None:
    _memory_load_history()

    def loop() -> None:
        trim_counter = 0
        while True:
            sample = _sample_memory()
            if sample:
                with _MEMORY_LOCK:
                    _MEMORY_BUFFER.append(sample)
                _memory_persist(sample)
                trim_counter += 1
                if trim_counter >= 60:  # once an hour
                    _memory_trim_file()
                    trim_counter = 0
            time.sleep(_MEMORY_SAMPLE_INTERVAL)

    threading.Thread(target=loop, name="memory-sampler", daemon=True).start()


@app.get("/api/system-memory")
def system_memory_api(minutes: int = 360) -> dict:
    """Return the last `minutes` of samples (default 6h) + a fresh current.

    Categories: python, browser, ollama, node, claude, other (all MB).
    Used for initial backfill; the live stream is `/api/system-memory/stream`.
    """
    minutes = max(5, min(minutes, 1440))
    cutoff = datetime.now() - timedelta(minutes=minutes)
    with _MEMORY_LOCK:
        all_samples = list(_MEMORY_BUFFER)

    samples = []
    for s in all_samples:
        ts = s.get("ts") or ""
        try:
            if datetime.fromisoformat(ts) >= cutoff:
                samples.append(s)
        except Exception:
            continue

    current = _sample_memory()
    return {
        "categories": list(_MEMORY_CATEGORIES),
        "samples": samples,
        "current": current,
        "interval_s": _MEMORY_SAMPLE_INTERVAL,
        "live_interval_s": _MEMORY_LIVE_INTERVAL,
        "window_minutes": minutes,
    }


# Live SSE stream. Emits a fresh sample every `_MEMORY_LIVE_INTERVAL` seconds
# so the chart updates in real time instead of waiting for the 60s persistent
# tick. The persistent 60s sampler (for the JSONL sidecar + historical
# buffer) runs independently — live ticks are NOT persisted to avoid
# bloating the JSONL at 2s cadence.
_MEMORY_LIVE_INTERVAL = 2.0


@app.get("/api/system-memory/stream")
async def system_memory_stream() -> StreamingResponse:
    async def gen():
        # Prime with one sample immediately so the client doesn't wait.
        first = await asyncio.to_thread(_sample_memory)
        if first:
            yield _sse("sample", first)
        while True:
            await asyncio.sleep(_MEMORY_LIVE_INTERVAL)
            sample = await asyncio.to_thread(_sample_memory)
            if not sample:
                continue
            yield _sse("sample", sample)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── RAG-stack CPU sampler ──────────────────────────────────────────────
# Same scope as the memory sampler (rag, ollama, sqlite-vec, whatsapp).
# macOS `ps -o %cpu` is a decaying average, which smears transient load
# — instead we compute instantaneous CPU% from deltas of cumulative CPU
# time across two snapshots (`ps -axo pid,cputime,command`). Each caller
# (live SSE stream, persistent sampler, etc.) owns its own prev-state
# dict so concurrent streams don't fight over the baseline.
#
# Returned percentages are in "% of one core" (can exceed 100 on
# multi-threaded processes — ollama runners routinely peg several
# cores during generation).
_CPU_STATE_PATH = Path.home() / ".local/share/obsidian-rag/rag_cpu.jsonl"
_CPU_BUFFER_MAX = 1440  # 24h @ 60s
_CPU_SAMPLE_INTERVAL = 60.0
_CPU_LIVE_INTERVAL = 2.0
_CPU_BUFFER: deque = deque(maxlen=_CPU_BUFFER_MAX)
_CPU_LOCK = threading.Lock()
_CPU_NCORES = os.cpu_count() or 1


def _parse_cputime(s: str) -> float | None:
    """Parse macOS ps cputime — `M:SS.FF` or `H:MM:SS.FF` — into seconds."""
    try:
        parts = s.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, TypeError):
        pass
    return None


def _sample_cpu(prev_state: dict) -> dict | None:
    """Instantaneous CPU% per rag-stack category via cputime delta.

    `prev_state` holds the previous snapshot across calls:
        {"ts": monotonic_seconds, "per_pid": {pid: cputime_s}}
    First invocation only primes the state and returns None.
    """
    now_mono = time.monotonic()
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,cputime=,command="],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except Exception:
        return None

    per_pid_cpu: dict[int, float] = {}
    per_pid_cmd: dict[int, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cpu_s = _parse_cputime(parts[1])
        if cpu_s is None:
            continue
        per_pid_cpu[pid] = cpu_s
        per_pid_cmd[pid] = parts[2]

    prev_per_pid = prev_state.get("per_pid") or {}
    prev_ts = prev_state.get("ts")
    prev_state["ts"] = now_mono
    prev_state["per_pid"] = per_pid_cpu

    if prev_ts is None:
        return None
    interval = now_mono - prev_ts
    if interval <= 0:
        return None

    by_cat: dict[str, float] = {k: 0.0 for k in _MEMORY_CATEGORIES}
    procs: dict[str, dict] = {}
    for pid, cpu_s in per_pid_cpu.items():
        prev_cpu = prev_per_pid.get(pid)
        if prev_cpu is None:
            continue
        delta = cpu_s - prev_cpu
        if delta < 0:
            continue  # clock anomaly or pid reuse — skip
        pct = (delta / interval) * 100.0
        if pct < 0.05:
            continue
        cmd = per_pid_cmd[pid]
        cat = _classify_rag_proc(cmd)
        if cat is None:
            continue
        by_cat[cat] += pct
        label = _rag_proc_label(cmd, cat)
        slot = procs.get(label)
        if slot is None:
            procs[label] = {"name": label, "pct": pct, "cat": cat}
        else:
            slot["pct"] += pct

    total = round(sum(by_cat.values()), 1)
    by_cat_rounded = {k: round(v, 1) for k, v in by_cat.items()}
    top = sorted(procs.values(), key=lambda d: d["pct"], reverse=True)[:10]
    top_list = [{"name": d["name"], "pct": round(d["pct"], 1), "cat": d["cat"]} for d in top]

    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "total_pct": total,
        "by_category": by_cat_rounded,
        "top": top_list,
        "ncores": _CPU_NCORES,
        "interval_s": round(interval, 2),
    }


def _cpu_load_history() -> None:
    if not _CPU_STATE_PATH.exists():
        return
    try:
        with _CPU_STATE_PATH.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()[-_CPU_BUFFER_MAX:]
    except Exception:
        return
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            _CPU_BUFFER.append(json.loads(raw))
        except Exception:
            continue


def _cpu_persist(sample: dict) -> None:
    if RAG_STATE_SQL:
        try:
            with _ragvec_state_conn() as conn:
                _sql_append_event(conn, "rag_cpu_metrics",
                                   _map_cpu_row(sample))
            return
        except Exception as exc:
            _log_sql_state_error("cpu_sql_write_failed", err=repr(exc))
    try:
        _CPU_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CPU_STATE_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample, separators=(",", ":")) + "\n")
    except Exception:
        pass


def _cpu_trim_file() -> None:
    try:
        if not _CPU_STATE_PATH.exists():
            return
        with _CPU_STATE_PATH.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        cap = _CPU_BUFFER_MAX * 2
        if len(lines) <= cap:
            return
        with _CPU_STATE_PATH.open("w", encoding="utf-8") as fh:
            fh.writelines(lines[-cap:])
    except Exception:
        pass


@app.on_event("startup")
def _start_cpu_sampler() -> None:
    _cpu_load_history()

    def loop() -> None:
        state: dict = {}
        # Prime the baseline, then wait a full interval before the first
        # real sample so the delta is meaningful.
        _sample_cpu(state)
        trim_counter = 0
        while True:
            time.sleep(_CPU_SAMPLE_INTERVAL)
            sample = _sample_cpu(state)
            if sample:
                with _CPU_LOCK:
                    _CPU_BUFFER.append(sample)
                _cpu_persist(sample)
                trim_counter += 1
                if trim_counter >= 60:
                    _cpu_trim_file()
                    trim_counter = 0

    threading.Thread(target=loop, name="cpu-sampler", daemon=True).start()


@app.get("/api/system-cpu")
def system_cpu_api(minutes: int = 360) -> dict:
    """Last `minutes` of rag-stack CPU samples + a freshly computed current.

    Values are `% of one core` per category. Can exceed 100 per category
    (sum across multithreaded procs). `ncores` is included so the client
    can render a secondary scale as % of total CPU if desired.
    """
    minutes = max(5, min(minutes, 1440))
    cutoff = datetime.now() - timedelta(minutes=minutes)
    with _CPU_LOCK:
        all_samples = list(_CPU_BUFFER)

    samples = []
    for s in all_samples:
        ts = s.get("ts") or ""
        try:
            if datetime.fromisoformat(ts) >= cutoff:
                samples.append(s)
        except Exception:
            continue

    # Snapshot current via a short-gap sample (sleep in a worker thread
    # would block this sync handler; just use a small synchronous window).
    probe: dict = {}
    _sample_cpu(probe)
    time.sleep(_CPU_LIVE_INTERVAL)
    current = _sample_cpu(probe)

    return {
        "categories": list(_MEMORY_CATEGORIES),
        "samples": samples,
        "current": current,
        "interval_s": _CPU_SAMPLE_INTERVAL,
        "live_interval_s": _CPU_LIVE_INTERVAL,
        "window_minutes": minutes,
        "ncores": _CPU_NCORES,
    }


@app.get("/api/system-cpu/stream")
async def system_cpu_stream() -> StreamingResponse:
    async def gen():
        state: dict = {}
        # Prime baseline, wait one interval, then stream.
        await asyncio.to_thread(_sample_cpu, state)
        while True:
            await asyncio.sleep(_CPU_LIVE_INTERVAL)
            sample = await asyncio.to_thread(_sample_cpu, state)
            if not sample:
                continue
            yield _sse("sample", sample)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
