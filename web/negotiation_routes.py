"""FastAPI routes for the WA negotiation foundation API."""
from __future__ import annotations

import inspect
import json
import time
from typing import Annotated, Any, Callable

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel, Field

__all__ = [
    "NegotiationCreateRequest",
    "NegotiationProcessDueRequest",
    "NegotiationQueueSendRequest",
    "NegotiationTransitionRequest",
    "create_negotiation",
    "list_negotiations",
    "list_pending_sends",
    "process_due_sends",
    "queue_negotiation_send",
    "transition_negotiation",
    "register_negotiation_routes",
]


class NegotiationCreateRequest(BaseModel):
    user_intent: str = Field(..., min_length=1, max_length=4000)
    target_jid: str = Field(..., min_length=1, max_length=200)
    target_name: str | None = Field(None, max_length=200)
    perimeter: dict[str, Any] | None = None
    confidence_threshold: float | None = Field(None, ge=0.0, le=1.0)
    max_messages: int | None = Field(None, ge=1)
    style_seed_jid: str | None = Field(None, max_length=200)


class NegotiationTransitionRequest(BaseModel):
    transition: str = Field(..., min_length=1, max_length=120)
    closure_summary: str | None = Field(None, max_length=4000)
    side_effect: dict[str, Any] | None = None


class NegotiationQueueSendRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    typing_simulation_ms: int | None = Field(None, ge=0, le=120_000)
    send_after_ts: float | None = Field(None, ge=0)
    delay_seconds: int | None = Field(None, ge=0, le=60 * 60 * 24 * 30)


class NegotiationProcessDueRequest(BaseModel):
    dry_run: bool = True
    limit: int = Field(10, ge=1, le=100)
    now_ts: float | None = Field(None, ge=0)


def _json_obj(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_item(item: Any) -> dict[str, Any]:
    data = dict(item) if not isinstance(item, dict) else dict(item)
    perimeter = _json_obj(data.get("perimeter_json"))
    if perimeter is not None and "perimeter" not in data:
        data["perimeter"] = perimeter
    side_effect = _json_obj(data.get("side_effect_json"))
    if side_effect is not None and "side_effect" not in data:
        data["side_effect"] = side_effect
    return data


def _list_negotiations_via_sql(
    *,
    status: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    import rag  # noqa: PLC0415

    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    with rag._ragvec_state_conn() as conn:
        cur = conn.execute(
            f"SELECT * FROM rag_negotiations{where}"
            " ORDER BY updated_at DESC LIMIT ?",
            params,
        )
        cols = [col[0] for col in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _list_negotiations(
    *,
    status: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    from rag_negotiations import crud  # noqa: PLC0415

    list_fn = getattr(crud, "list_negotiations", None)
    if callable(list_fn):
        try:
            sig = inspect.signature(list_fn)
            params = sig.parameters
            kwargs: dict[str, Any] = {}
            if status and "status" not in params:
                items = _list_negotiations_via_sql(status=status, limit=limit)
            else:
                if status:
                    kwargs["status"] = status
                if "limit" in params:
                    kwargs["limit"] = limit
                items = list_fn(**kwargs)
        except (TypeError, ValueError):
            items = _list_negotiations_via_sql(status=status, limit=limit)
        except Exception:
            items = _list_negotiations_via_sql(status=status, limit=limit)

        if items is not None:
            return [_normalize_item(item) for item in list(items)][:limit]

    return [
        _normalize_item(item)
        for item in _list_negotiations_via_sql(status=status, limit=limit)
    ]


def _list_pending_sends_via_sql(
    *,
    negotiation_id: int,
    limit: int,
) -> list[dict[str, Any]]:
    import rag  # noqa: PLC0415

    with rag._ragvec_state_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM rag_negotiation_pending_sends "
            "WHERE negotiation_id = ? "
            "ORDER BY queued_at DESC, id DESC "
            "LIMIT ?",
            (int(negotiation_id), int(limit)),
        )
        cols = [col[0] for col in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def list_negotiations(
    status: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> dict[str, Any]:
    status_filter = (status or "").strip() or None
    try:
        items = _list_negotiations(status=status_filter, limit=int(limit))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="negotiations unavailable") from exc
    return {"ok": True, "items": items}


def create_negotiation(req: NegotiationCreateRequest) -> dict[str, Any]:
    from rag_negotiations import crud  # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "user_intent": req.user_intent,
        "target_jid": req.target_jid,
        "target_name": req.target_name,
        "perimeter": req.perimeter,
        "max_messages": req.max_messages,
        "style_seed_jid": req.style_seed_jid,
    }
    if req.confidence_threshold is not None:
        kwargs["confidence_threshold"] = req.confidence_threshold
    try:
        neg_id = crud.create_negotiation(**kwargs)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="negotiation create failed") from exc
    if neg_id is None:
        raise HTTPException(status_code=503, detail="negotiation create failed")
    return {"ok": True, "id": int(neg_id)}


def list_pending_sends(
    negotiation_id: int,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> dict[str, Any]:
    from rag_negotiations import crud  # noqa: PLC0415

    current = crud.get_negotiation(negotiation_id)
    if current is None:
        raise HTTPException(status_code=404, detail="negotiation not found")
    try:
        items = _list_pending_sends_via_sql(
            negotiation_id=negotiation_id,
            limit=int(limit),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="pending sends unavailable") from exc
    return {"ok": True, "items": items}


def _send_after_ts(req: NegotiationQueueSendRequest) -> float | None:
    if req.send_after_ts is not None:
        return float(req.send_after_ts)
    if req.delay_seconds is not None:
        return time.time() + int(req.delay_seconds)
    return None


def queue_negotiation_send(
    negotiation_id: int,
    req: NegotiationQueueSendRequest,
) -> dict[str, Any]:
    from rag_negotiations import crud, state_machine  # noqa: PLC0415

    current = crud.get_negotiation(negotiation_id)
    if current is None:
        raise HTTPException(status_code=404, detail="negotiation not found")
    status = str(current.get("status") or "")
    if status in state_machine.TERMINAL_STATES:
        raise HTTPException(status_code=400, detail=f"negotiation is terminal: {status}")

    try:
        send_id = crud.enqueue_send(
            negotiation_id=negotiation_id,
            content=req.content.strip(),
            typing_simulation_ms=req.typing_simulation_ms,
            send_after_ts=_send_after_ts(req),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="queue send failed") from exc
    if send_id is None:
        raise HTTPException(status_code=503, detail="queue send failed")
    return {
        "ok": True,
        "id": int(send_id),
        "negotiation_id": negotiation_id,
        "status": "pending",
    }


def transition_negotiation(
    negotiation_id: int,
    req: NegotiationTransitionRequest,
) -> dict[str, Any]:
    from rag_negotiations import crud, state_machine  # noqa: PLC0415

    current = crud.get_negotiation(negotiation_id)
    if current is None:
        raise HTTPException(status_code=404, detail="negotiation not found")

    from_status = str(current.get("status") or "")
    try:
        to_status = state_machine.transition(from_status, req.transition)
    except state_machine.InvalidTransitionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        ok = crud.update_status(
            negotiation_id,
            to_status,
            closure_summary=req.closure_summary,
            side_effect=req.side_effect,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="negotiation transition failed") from exc
    if not ok:
        raise HTTPException(status_code=503, detail="negotiation transition failed")

    return {
        "ok": True,
        "id": negotiation_id,
        "from_status": from_status,
        "to_status": to_status,
    }


def process_due_sends(req: NegotiationProcessDueRequest) -> dict[str, Any]:
    from rag.integrations.whatsapp import send as wa_send  # noqa: PLC0415
    from rag_negotiations import crud  # noqa: PLC0415

    try:
        due = crud.dequeue_due(
            now_ts=req.now_ts,
            limit=int(req.limit),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="dequeue due sends failed") from exc

    processed: list[dict[str, Any]] = []
    for item in due:
        send_id = int(item.get("id") or 0)
        neg_id = int(item.get("negotiation_id") or 0)
        negotiation = crud.get_negotiation(neg_id)
        target_jid = str((negotiation or {}).get("target_jid") or "")
        content = str(item.get("content") or "")
        if req.dry_run:
            processed.append({
                "id": send_id,
                "negotiation_id": neg_id,
                "target_jid": target_jid,
                "status": "due",
                "would_send": bool(target_jid and content),
            })
            continue
        if not negotiation or not target_jid or not content:
            crud.mark_send(send_id, status="failed")
            processed.append({
                "id": send_id,
                "negotiation_id": neg_id,
                "status": "failed",
                "reason": "missing negotiation target or content",
            })
            continue
        ok, reason = wa_send._whatsapp_send_to_jid_detailed(
            target_jid,
            content,
            anti_loop=True,
        )
        final_status = "sent" if ok else "failed"
        crud.mark_send(send_id, status=final_status)
        if ok:
            crud.append_turn(
                negotiation_id=neg_id,
                direction="out",
                content=content,
                pause_simulated_ms=item.get("typing_simulation_ms"),
            )
            crud.increment_message_count(neg_id, sent=True)
        processed.append({
            "id": send_id,
            "negotiation_id": neg_id,
            "target_jid": target_jid,
            "status": final_status,
            "reason": reason,
        })
    return {
        "ok": True,
        "dry_run": bool(req.dry_run),
        "count": len(processed),
        "items": processed,
    }


def register_negotiation_routes(
    app,
    require_admin_token: Callable | None = None,
) -> dict[str, object]:
    admin_dep = [Depends(require_admin_token)] if require_admin_token is not None else []
    app.get("/api/negotiations")(list_negotiations)
    app.post("/api/negotiations", dependencies=admin_dep)(create_negotiation)
    app.get("/api/negotiations/{negotiation_id}/pending-sends")(list_pending_sends)
    app.post(
        "/api/negotiations/{negotiation_id}/queue-send",
        dependencies=admin_dep,
    )(queue_negotiation_send)
    app.post(
        "/api/negotiations/process-due-sends",
        dependencies=admin_dep,
    )(process_due_sends)
    app.post(
        "/api/negotiations/{negotiation_id}/transition",
        dependencies=admin_dep,
    )(transition_negotiation)
    return {
        name: globals()[name]
        for name in __all__
        if name != "register_negotiation_routes"
    }
