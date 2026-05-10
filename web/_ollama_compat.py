"""Ollama-compat shim in-process — endpoint para que el listener WA hable
MLX nativo en lugar de a un daemon Ollama separado.

## Por qué existe (GC-C, 2026-05-10)

El listener TS (`whatsapp-listener/listener.ts`) habla Ollama API en ~9
sitios (drafts, tagging OCR, calendar helpers, send drafts, etc). Default
apunta a `OLLAMA_URL=http://localhost:11434` — Ollama daemon real con sus
propios modelos cargados (qwen2.5:14b/7b/3b, qwen3-embedding). Eso duplica
modelos en RAM (8.8 GB del DraftHelperModel medido el 2026-05-10) y rompe
la regla MLX-first del proyecto.

Este módulo expone endpoints `/ollama/api/chat` y `/ollama/api/tags` en el
mismo `web/server.py` que ya tiene MLX in-process cargado. El listener
queda con una sola variable env override (`OLLAMA_URL=http://localhost:8765/ollama`)
sin tocar TS — los call sites siguen posteando a `${OLLAMA_URL}/api/chat`.

Beneficios cascada:
- Una sola instancia del modelo en RAM (compartida con chat web + drafts WA).
- Determinismo igual al chat web (`HELPER_OPTIONS = {temperature: 0, seed: 42}`).
- Telemetría unificada: cada draft se logea a `rag_queries` con
  `cmd='listener.<helper_kind>'` (visible en el dashboard + audit).
- DPO loop destrabable: pares (draft, edit) llegan a `rag_draft_decisions`
  vía mismo path que el chat web.

## Compatibilidad con Ollama API

Cubre el subset que el listener consume hoy:

- `POST /api/chat` con `{model, messages, stream, options, format}`.
- `GET /api/tags` (usado como healthcheck).
- `GET /` (liveness).

NO cubre `/api/embed` (el listener no lo invoca; embeddings van vía
`rag` CLI). NO cubre `/api/generate` (no usado).

Streaming: NDJSON line-per-token (Ollama format), terminal con `done: true`
+ token counts. Non-stream: 1 JSON con `message.content` completo.

## Mapping de modelos

`rag.llm_backend.MLX_MODEL_ALIAS` resuelve los nombres Ollama
(`qwen2.5:7b`, `qwen2.5:14b-instruct`, etc) a los MLX HF ids. Modelos no
mapeados devuelven 503 explícito (el listener tiene fallback chain).

## Logging

Cada call escribe a `rag_queries` con cmd=`listener.<short_name>`,
trace_id derivado de `X-Listener-Trace` header (si presente) o nuevo uuid.
El payload se trunca a `_LOG_Q_MAX = 400` chars para no inflar la DB.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

__all__ = [
    "build_chat_response",
    "stream_chat_ndjson",
    "list_tags",
    "shim_liveness",
    "ListenerChatRequest",
]


# Truncate query payload before logging — listener system prompts pueden
# tener miles de chars (vault context + few-shot + dossier), no queremos
# eso en `rag_queries.q` que ya guarda hashes.
_LOG_Q_MAX = 400

# Default num_ctx para drafts del listener. Override per-call via
# `options.num_ctx`. 4096 mismo que CHAT_OPTIONS — el listener inyecta
# vault context + history + few-shot, fácil llega a 3-3.5K tokens.
_DEFAULT_NUM_CTX = 4096
_DEFAULT_NUM_PREDICT = 384


def _now_iso() -> str:
    """ISO-8601 con sufijo Z (Ollama format)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ollama_options_to_chat_options(options: dict | None, top_level: dict | None) -> dict:
    """Translate Ollama `options` dict + top-level fields → opts dict (consumido por `_mlx_chat`).

    Ollama API accepts both `options.{temperature,top_p,seed,num_ctx,num_predict,stop}`
    AND a few top-level fields (`stop`). We normalize both.

    Falls back to MLX defaults (HELPER_OPTIONS-shape) cuando el campo no
    está presente — temp=0/seed=42 garantiza determinismo en drafts.

    Devuelve **dict** (no `ChatOptions` dataclass) porque `_mlx_chat` /
    `_chat_stream_dispatch` construyen el `ChatOptions` internamente; pasarles
    una dataclass cae al fallback `dict(options_dict)` que falla con
    `'ChatOptions' object is not iterable` (no `__iter__`).
    """
    options = options or {}
    top_level = top_level or {}

    # Stop tokens: Ollama acepta str o list. Normalizo a list (JSON-serializable).
    stop = options.get("stop") or top_level.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    elif isinstance(stop, tuple):
        stop = list(stop)
    elif not isinstance(stop, list):
        stop = []

    return {
        "temperature": float(options.get("temperature", 0.0) or 0.0),
        "seed": int(options.get("seed", 42) or 42),
        "num_ctx": int(options.get("num_ctx", _DEFAULT_NUM_CTX) or _DEFAULT_NUM_CTX),
        "num_predict": int(options.get("num_predict", _DEFAULT_NUM_PREDICT) or _DEFAULT_NUM_PREDICT),
        "top_p": float(options.get("top_p", 1.0) or 1.0),
        "stop": stop,
    }


def _resolve_or_503(model: str) -> tuple[str, str]:
    """Return `(short_name, mlx_id)` o levanta HTTPException(503/400).

    Usa `MLX_MODEL_ALIAS` ya existente. Si el alias no está registrado
    devolvemos 503 con sugerencias — el listener tiene fallback chain
    (DRAFT_HELPER_MODEL → DRAFT_HELPER_FALLBACK_MODEL).
    """
    from fastapi import HTTPException  # noqa: PLC0415
    from rag.llm_backend import MLX_MODEL_ALIAS, to_mlx, to_short_name  # noqa: PLC0415

    if not model or not isinstance(model, str):
        raise HTTPException(400, "missing or invalid `model` field")

    mlx_id = to_mlx(model)
    # `to_mlx` devuelve el input as-is si no encuentra alias. Si sigue sin
    # prefijo `mlx-community/` significa que NO está mapeado.
    if not mlx_id.startswith("mlx-community/"):
        known = sorted(MLX_MODEL_ALIAS)
        raise HTTPException(
            503,
            f"model `{model}` not mapped to MLX. Known aliases: {known}. "
            f"Add to `MLX_MODEL_ALIAS` in rag/llm_backend.py.",
        )
    return to_short_name(mlx_id), mlx_id


def _log_to_rag_queries(
    *,
    cmd: str,
    trace_id: str,
    short_name: str,
    messages: list[dict],
    response_text: str,
    t_gen_s: float,
    options_dict: dict,
    error: str | None = None,
) -> None:
    """Best-effort write a `rag_queries` table — drafts del listener
    aparecen en el mismo dashboard que chats web.

    Errores en log NO propagan al caller (silent + telemetry).
    """
    try:
        from rag import log_query_event  # noqa: PLC0415

        # Concat de la última question del messages array (cap _LOG_Q_MAX).
        # Sistema messages se omiten — son context, no la query del user.
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        q_truncated = (last_user or "")[:_LOG_Q_MAX]

        # `log_query_event` mappea campos conocidos a columnas dedicadas y todo
        # lo extra cae a `extra_json` — model+options+error van por ahí.
        log_query_event({
            "cmd": cmd,
            "trace_id": trace_id,
            "q": q_truncated,
            "mode": "listener",
            "t_gen": t_gen_s,
            "answer_len": len(response_text or ""),
            "model": short_name,
            "options": {
                k: options_dict.get(k)
                for k in ("temperature", "seed", "num_ctx", "num_predict")
                if k in options_dict
            },
            **({"error": error} if error else {}),
        })
    except Exception as exc:
        # No tirar excepción — el endpoint debe responder al listener
        # incluso si el log a SQL fallo.
        try:
            from rag import _silent_log  # noqa: PLC0415
            _silent_log("ollama_compat_log_failed", exc)
        except Exception:
            pass


# ── Public surface (used by web/server.py endpoints) ──────────────────────


class ListenerChatRequest:
    """Schema Ollama-compat. NO uso pydantic porque acepta campos
    arbitrarios (Ollama es lax con esto). FastAPI body parsing acepta
    `dict` directo.
    """

    pass  # placeholder for type hints; actual parsing lives en endpoint.


def build_chat_response(
    payload: dict,
    *,
    trace_id: str | None = None,
) -> dict:
    """Non-streaming Ollama-compat `/api/chat` response.

    Args:
        payload: Ollama-format body con `{model, messages, options, format}`.
        trace_id: opcional, propagado del header `X-Listener-Trace`. Si
            None, generamos uno.

    Returns:
        Dict Ollama-format con `{model, created_at, message, done,
        done_reason, total_duration, eval_count, ...}`.

    Raises:
        HTTPException(400) si payload mal-formado.
        HTTPException(503) si modelo no mapeado a MLX.
    """
    from fastapi import HTTPException  # noqa: PLC0415
    from rag import _mlx_chat  # noqa: PLC0415

    if not isinstance(payload, dict):
        raise HTTPException(400, "body must be JSON object")

    model_in = payload.get("model")
    messages = payload.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "missing or empty `messages`")

    short_name, mlx_id = _resolve_or_503(model_in)
    options = payload.get("options") or {}
    fmt = payload.get("format")  # `"json"` o None
    chat_opts = _ollama_options_to_chat_options(options, payload)

    trace_id = trace_id or uuid.uuid4().hex[:16]
    cmd = f"listener.{short_name}"

    t0 = time.time()
    response_text = ""
    err: str | None = None
    try:
        resp = _mlx_chat(
            model=short_name,  # `_mlx_chat` resuelve internamente
            messages=messages,
            options=chat_opts,
            format=fmt if fmt in ("json",) else None,
        )
        # `resp` es ChatResponse (pydantic BaseModel). Acceso atributo.
        msg = resp.message
        response_text = msg.content or ""
    except HTTPException:
        raise
    except Exception as exc:
        err = repr(exc)
        # Devolvemos 500 con mensaje JSON Ollama-shape — el listener tiene
        # try/catch que loguea y cae al fallback chain.
        raise HTTPException(500, f"MLX chat failed: {exc}") from exc
    finally:
        t_gen_s = time.time() - t0
        _log_to_rag_queries(
            cmd=cmd,
            trace_id=trace_id,
            short_name=short_name,
            messages=messages,
            response_text=response_text,
            t_gen_s=t_gen_s,
            options_dict={
                "temperature": chat_opts["temperature"],
                "seed": chat_opts["seed"],
                "num_ctx": chat_opts["num_ctx"],
                "num_predict": chat_opts["num_predict"],
            },
            error=err,
        )

    elapsed_ns = int((time.time() - t0) * 1e9)
    # Ollama response shape (subset que el listener parsea):
    return {
        "model": model_in,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": response_text},
        "done": True,
        "done_reason": "stop",
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": 0,  # MLX backend no expone token counts hoy
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": elapsed_ns,
    }


async def stream_chat_ndjson(
    payload: dict,
    *,
    trace_id: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """Streaming Ollama-compat — yield NDJSON líneas.

    Cada token de MLX `chat_stream` se emite como `{message: {role,
    content}, done: false}` + un final `{done: true, eval_count, ...}`.

    Errores en mid-stream emiten `{error: "..."}` y cierran.
    """
    from fastapi import HTTPException  # noqa: PLC0415
    from rag import _chat_stream_dispatch  # noqa: PLC0415

    if not isinstance(payload, dict):
        raise HTTPException(400, "body must be JSON object")
    model_in = payload.get("model")
    messages = payload.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "missing or empty `messages`")

    short_name, _mlx_id = _resolve_or_503(model_in)
    options = payload.get("options") or {}
    fmt = payload.get("format")
    chat_opts = _ollama_options_to_chat_options(options, payload)

    trace_id = trace_id or uuid.uuid4().hex[:16]
    cmd = f"listener.{short_name}"

    t0 = time.time()
    full_text: list[str] = []
    err: str | None = None
    try:
        for chunk in _chat_stream_dispatch(
            model=short_name,
            messages=messages,
            options=chat_opts,
            format=fmt if fmt in ("json",) else None,
        ):
            # `chunk` es ChatResponse pydantic. piece = content del delta.
            piece = (chunk.message.content or "") if chunk.message else ""
            done = bool(getattr(chunk, "done", False))
            if piece:
                full_text.append(piece)
                yield (
                    json.dumps(
                        {
                            "model": model_in,
                            "created_at": _now_iso(),
                            "message": {"role": "assistant", "content": piece},
                            "done": False,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode("utf-8")
            if done:
                break
    except Exception as exc:
        err = repr(exc)
        yield (
            json.dumps({"error": f"MLX stream failed: {exc}"}, ensure_ascii=False) + "\n"
        ).encode("utf-8")
    finally:
        t_gen_s = time.time() - t0
        elapsed_ns = int(t_gen_s * 1e9)
        # Final chunk con `done: true` + duraciones (Ollama format).
        yield (
            json.dumps(
                {
                    "model": model_in,
                    "created_at": _now_iso(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop" if not err else "error",
                    "total_duration": elapsed_ns,
                    "load_duration": 0,
                    "prompt_eval_count": 0,
                    "prompt_eval_duration": 0,
                    "eval_count": 0,
                    "eval_duration": elapsed_ns,
                },
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        _log_to_rag_queries(
            cmd=cmd,
            trace_id=trace_id,
            short_name=short_name,
            messages=messages,
            response_text="".join(full_text),
            t_gen_s=t_gen_s,
            options_dict={
                "temperature": chat_opts["temperature"],
                "seed": chat_opts["seed"],
                "num_ctx": chat_opts["num_ctx"],
                "num_predict": chat_opts["num_predict"],
                "stream": True,
            },
            error=err,
        )


def list_tags() -> dict:
    """Ollama-compat `/api/tags`. Listener lo usa como healthcheck. Listamos
    los aliases que MLX_MODEL_ALIAS resuelve.
    """
    from rag.llm_backend import MLX_MODEL_ALIAS  # noqa: PLC0415

    models = []
    for alias, mlx_id in MLX_MODEL_ALIAS.items():
        kind = "embed" if "embedding" in mlx_id.lower() else "chat"
        models.append(
            {
                "name": alias,
                "model": alias,
                "modified_at": _now_iso(),
                "size": 0,
                "digest": "",
                "details": {
                    "parent_model": mlx_id,
                    "format": "mlx",
                    "family": kind,
                },
            }
        )
    return {"models": models}


def shim_liveness() -> dict:
    """Healthcheck root del shim — útil para monitoreo + debug."""
    from rag.llm_backend import MLX_MODEL_ALIAS  # noqa: PLC0415

    return {
        "shim": "obsidian-rag-ollama-compat",
        "backend": "mlx-in-process",
        "models_mapped": len(MLX_MODEL_ALIAS),
        "ts": _now_iso(),
    }
