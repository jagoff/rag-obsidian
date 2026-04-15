"""Web UI mínima para `rag chat`.

Espeja el pipeline del CLI (multi_retrieve → command-r streaming → sources)
sobre HTTP + SSE. Sin build step en el frontend; vanilla JS contra este
endpoint. Sesiones persistidas en el mismo store que el CLI — el session_id
de la web es `web:<uuid>` así no colisiona con `tg:<chat_id>` ni con los
ids del chat interactivo.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# rag.py vive en el root del proyecto; lo importamos como módulo.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ollama  # noqa: E402

from rag import (  # noqa: E402
    CHAT_OPTIONS,
    OLLAMA_KEEP_ALIVE,
    SESSION_HISTORY_WINDOW,
    SYSTEM_RULES,
    _load_vaults_config,
    append_turn,
    ensure_session,
    log_query_event,
    multi_retrieve,
    new_turn_id,
    resolve_chat_model,
    resolve_vault_paths,
    save_session,
    session_history,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="obsidian-rag web", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    # None → vault activo; "all" → todos los registrados; "name" → ese puntual.
    vault_scope: str | None = None


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


def _resolve_scope(scope: str | None) -> list[tuple[str, "Path"]]:
    if scope is None or scope == "":
        return resolve_vault_paths(None)
    if scope == "all":
        return resolve_vault_paths(["all"])
    return resolve_vault_paths([scope])


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
        yield _sse("session", {"id": sess["id"]})

        history = session_history(sess, window=SESSION_HISTORY_WINDOW)
        try:
            result = multi_retrieve(
                vaults, question, 6, None, history, None, False,
                multi_query=True, auto_filter=True, date_range=None,
            )
        except Exception as exc:
            yield _sse("error", {"message": f"retrieve falló: {exc}"})
            return

        if not result["docs"]:
            yield _sse("empty", {"message": "Sin resultados relevantes."})
            return

        emoji, label = _confidence_badge(float(result["confidence"]))
        meta_bits: list[str] = [f"{emoji} {label}"]
        if result.get("filters_applied"):
            parts = [f"{k}={v}" for k, v in result["filters_applied"].items()]
            meta_bits.append(f"filtros: {', '.join(parts)}")
        if len(result.get("query_variants", [])) > 1:
            meta_bits.append(f"{len(result['query_variants'])} variantes")
        meta_bits.append(f"{len({m['file'] for m in result['metas']})} nota(s)")
        yield _sse("meta", {"bits": meta_bits})

        yield _sse("sources", {
            "items": [
                {**_source_payload(m, s), "bar": _score_bar(float(s))}
                for m, s in zip(result["metas"], result["scores"])
            ],
            "confidence": round(float(result["confidence"]), 3),
        })

        is_multi = len(vaults) > 1
        context = "\n\n---\n\n".join(
            (f"[vault: {m.get('_vault', '?')}] " if is_multi else "")
            + f"[nota: {m['note']}] [ruta: {m['file']}]\n{d}"
            for d, m in zip(result["docs"], result["metas"])
        )
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

        parts: list[str] = []
        try:
            for chunk in ollama.chat(
                model=resolve_chat_model(),
                messages=messages,
                options=CHAT_OPTIONS,
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
        })

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
