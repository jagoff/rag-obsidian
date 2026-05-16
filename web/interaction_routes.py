"""Interaction telemetry routes: feedback, mood, drafts, briefs, and behavior."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime as _dt
from datetime import timedelta as _td
from pathlib import Path
from typing import Callable, ContextManager

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field, field_validator

__all__ = [
    "FeedbackRequest",
    "submit_feedback",
    "_VALID_MOOD_LABELS",
    "MoodRequest",
    "submit_mood",
    "cross_source_patterns",
    "mood_history",
    "_VALID_DRAFT_DECISIONS_WEB",
    "DraftDecisionPayload",
    "submit_draft_decision",
    "DraftPreviewPayload",
    "submit_draft_preview",
    "_VALID_ANTICIPATE_RATINGS",
    "AnticipateFeedbackPayload",
    "submit_anticipate_feedback",
    "_VALID_BRIEF_RATINGS",
    "BriefFeedbackPayload",
    "submit_brief_feedback",
    "_BEHAVIOR_KNOWN_EVENTS",
    "_BEHAVIOR_KNOWN_SOURCES",
    "_BEHAVIOR_SESSION_RE",
    "BehaviorRequest",
    "submit_behavior",
    "InteractionRouteDeps",
    "register_interaction_routes",
]

_CHAT_SESSION_RE = re.compile(r"^[A-Za-z0-9_.:@\-]{1,80}$")
_TURN_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


@dataclass(frozen=True)
class InteractionRouteDeps:
    """Callbacks supplied by the live server module at registration time."""

    rate_limit_behavior: Callable[[Request], None]
    record_feedback: Callable[..., None]
    vault_path: Callable[[], Path]
    ragvec_state_conn: Callable[[], ContextManager]
    log_behavior_event: Callable[[dict], None]


_DEPS: InteractionRouteDeps | None = None


def _deps() -> InteractionRouteDeps:
    if _DEPS is None:
        raise RuntimeError("interaction routes not registered")
    return _DEPS


def _rate_limit_behavior(request: Request) -> None:
    _deps().rate_limit_behavior(request)


def _empty_patterns(days: int, lags: tuple[int, ...]) -> dict:
    return {
        "n_findings": 0,
        "top": [],
        "by_severity": {},
        "metrics_with_data": [],
        "days_range": days,
        "lags_tested": list(lags),
    }


class FeedbackRequest(BaseModel):
    turn_id: str
    rating: int
    q: str | None = Field(None, max_length=2000)
    paths: list[str] | None = Field(None, max_length=50)
    session_id: str | None = None
    reason: str | None = Field(None, max_length=500)
    corrective_path: str | None = Field(None, max_length=512)

    @field_validator("turn_id")
    @classmethod
    def _check_turn_id(cls, v: str) -> str:
        if not _TURN_ID_RE.match(v):
            raise ValueError("invalid turn_id format")
        return v

    @field_validator("session_id")
    @classmethod
    def _check_session(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if not _CHAT_SESSION_RE.match(v):
            raise ValueError("invalid session_id format")
        return v


def submit_feedback(req: FeedbackRequest) -> dict:
    """Record a +1/-1 rating for a chat turn."""
    if not req.turn_id or req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="turn_id + rating +/-1 requeridos")
    corrective_path = (req.corrective_path or "").strip() or None
    if corrective_path and "://" in corrective_path:
        corrective_path = None
    reason = "corrective" if corrective_path else ((req.reason or "").strip()[:200] or None)
    _deps().record_feedback(
        turn_id=req.turn_id,
        rating=req.rating,
        q=(req.q or "").strip(),
        paths=req.paths or [],
        reason=reason,
        corrective_path=corrective_path,
        session_id=req.session_id,
    )
    return {"ok": True}


_VALID_MOOD_LABELS: frozenset[str] = frozenset({"good", "meh", "bad"})


class MoodRequest(BaseModel):
    mood: str = Field(..., max_length=10)
    notes: str | None = Field(None, max_length=500)

    @field_validator("mood")
    @classmethod
    def _check_mood(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in _VALID_MOOD_LABELS:
            raise ValueError(f"mood must be one of {sorted(_VALID_MOOD_LABELS)}")
        return v


def submit_mood(req: MoodRequest) -> dict:
    from rag.integrations.pillow_sleep import record_self_report_mood

    notes = (req.notes or "").strip()[:500] or None
    result = record_self_report_mood(req.mood, notes=notes)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "invalid mood"))
    return result


def cross_source_patterns(days: int = 30, lags: str = "0,1,7", top: int = 10) -> dict:
    try:
        days_raw = int(days) if days is not None else 30
    except (TypeError, ValueError):
        days_raw = 30
    days_clamped = max(7, min(days_raw, 90))
    try:
        lag_list = tuple(
            sorted(
                set(
                    max(0, min(int(x.strip()), 14))
                    for x in lags.split(",")
                    if x.strip()
                )
            )
        ) or (0,)
    except (AttributeError, TypeError, ValueError):
        lag_list = (0, 1, 7)
    try:
        top_clamped = max(1, min(int(top), 50))
    except (TypeError, ValueError):
        top_clamped = 10

    try:
        from rag.cross_source_patterns import patterns_summary

        return patterns_summary(days=days_clamped, top=top_clamped, lags=lag_list)
    except Exception as exc:
        print(f"[patterns] failed: {exc}", file=sys.stderr)
        return _empty_patterns(days_clamped, lag_list)


def mood_history(days: int = 30) -> dict:
    try:
        days_raw = int(days) if days is not None else 30
    except (TypeError, ValueError):
        days_raw = 30
    days = max(1, min(days_raw, 90))
    try:
        from rag import mood as _mood
    except ImportError:
        return {"days": [], "histogram": [], "total_days_with_data": 0}

    try:
        recent = _mood.get_recent_scores(days=days)
    except Exception:
        recent = []

    today_dt = _dt.strptime(_mood._today_local(), "%Y-%m-%d")
    by_date_map = {r.get("date"): r for r in recent if r.get("date")}
    daily: list[dict] = []
    histogram_acc: dict[str, dict[str, float]] = {}

    for offset in range(days - 1, -1, -1):
        date = (today_dt - _td(days=offset)).strftime("%Y-%m-%d")
        row = by_date_map.get(date)
        if row is None or row.get("n_signals", 0) == 0:
            daily.append({
                "date": date,
                "score": float(row.get("score", 0.0)) if row else 0.0,
                "n_signals": 0,
                "sources_used": [],
                "by_source": [],
            })
            continue
        try:
            signals = _mood._read_signals_for_date(date)
        except Exception:
            signals = []
        if not signals:
            daily.append({
                "date": date,
                "score": float(row.get("score", 0.0)),
                "n_signals": 0,
                "sources_used": [],
                "by_source": [],
            })
            continue

        by_source: dict[str, dict[str, float]] = {}
        for signal in signals:
            src = signal.get("source") or "unknown"
            value = float(signal.get("value", 0.0))
            weight = float(signal.get("weight", 1.0))
            entry = by_source.setdefault(
                src,
                {"source": src, "contrib_signed": 0.0, "contrib_weight": 0.0, "n_signals": 0},
            )
            entry["contrib_signed"] += value * weight
            entry["contrib_weight"] += abs(value) * weight
            entry["n_signals"] += 1

        total_weight = sum(e["contrib_weight"] for e in by_source.values()) or 1.0
        by_source_list = []
        for src, entry in by_source.items():
            by_source_list.append({
                "source": src,
                "n_signals": entry["n_signals"],
                "contrib": round(entry["contrib_signed"], 3),
                "pct": round(entry["contrib_weight"] / total_weight * 100.0, 1),
            })
            agg = histogram_acc.setdefault(
                src,
                {"source": src, "total_contrib": 0.0, "total_weight": 0.0, "days_active": 0, "n_signals": 0},
            )
            agg["total_contrib"] += entry["contrib_signed"]
            agg["total_weight"] += entry["contrib_weight"]
            agg["days_active"] += 1
            agg["n_signals"] += entry["n_signals"]
        by_source_list.sort(key=lambda x: abs(x["contrib"]), reverse=True)
        daily.append({
            "date": date,
            "score": round(float(row.get("score", 0.0)), 3),
            "n_signals": int(row.get("n_signals", 0)),
            "sources_used": sorted(by_source.keys()),
            "by_source": by_source_list,
        })

    grand_total_w = sum(a["total_weight"] for a in histogram_acc.values()) or 1.0
    histogram = [
        {
            "source": src,
            "total_contrib": round(a["total_contrib"], 3),
            "pct": round(a["total_weight"] / grand_total_w * 100.0, 1),
            "days_active": a["days_active"],
            "n_signals": a["n_signals"],
        }
        for src, a in histogram_acc.items()
    ]
    histogram.sort(key=lambda x: abs(x["total_contrib"]), reverse=True)
    return {
        "days": daily,
        "histogram": histogram,
        "total_days_with_data": sum(1 for d in daily if d["n_signals"] > 0),
        "range_days": days,
    }


_VALID_DRAFT_DECISIONS_WEB: frozenset[str] = frozenset({
    "approved_si",
    "approved_editar",
    "rejected",
    "expired",
})


class DraftDecisionPayload(BaseModel):
    draft_id: str = Field(..., max_length=120)
    contact_jid: str = Field(..., max_length=120)
    contact_name: str | None = Field(None, max_length=200)
    original_msgs: list[dict] = Field(default_factory=list, max_length=50)
    bot_draft: str = Field("", max_length=8000)
    decision: str
    sent_text: str | None = Field(None, max_length=8000)
    extra: dict | None = None

    @field_validator("decision")
    @classmethod
    def _check_decision(cls, v: str) -> str:
        if v not in _VALID_DRAFT_DECISIONS_WEB:
            raise ValueError(f"decision must be one of {sorted(_VALID_DRAFT_DECISIONS_WEB)}")
        return v


def submit_draft_decision(req: DraftDecisionPayload, request: Request) -> dict:
    _rate_limit_behavior(request)
    from rag import _record_draft_decision

    try:
        row_id = _record_draft_decision(
            draft_id=req.draft_id,
            contact_jid=req.contact_jid,
            contact_name=req.contact_name,
            original_msgs=req.original_msgs or [],
            bot_draft=req.bot_draft or "",
            decision=req.decision,
            sent_text=req.sent_text,
            extra=req.extra,
        )
    except Exception as exc:
        return {"ok": False, "reason": f"helper raised: {exc}"}
    if row_id is None:
        return {"ok": False, "reason": "write failed (telemetry unavailable)"}
    return {"ok": True, "id": int(row_id)}


class DraftPreviewPayload(BaseModel):
    original_conversation: str = Field("", max_length=8000)
    bot_draft_baseline: str = Field("", max_length=8000)


def submit_draft_preview(req: DraftPreviewPayload, request: Request) -> dict:
    _rate_limit_behavior(request)
    from rag import (
        _drafts_ft_adapter_available,
        _drafts_ft_enabled,
        generate_draft_preview,
    )

    ft_active = bool(_drafts_ft_enabled() and _drafts_ft_adapter_available())
    try:
        preview = generate_draft_preview(
            original_conversation=req.original_conversation,
            bot_draft_baseline=req.bot_draft_baseline,
        )
    except Exception as exc:
        preview = req.bot_draft_baseline
        ft_active = False
        try:
            import rag as _rag_mod  # noqa: PLC0415

            _rag_mod._silent_log("drafts_ft_generate_failed", exc)
        except ImportError:
            pass
    return {"ok": True, "preview": preview, "ft_active": ft_active}


_VALID_ANTICIPATE_RATINGS: frozenset[str] = frozenset({"positive", "negative", "mute"})


class AnticipateFeedbackPayload(BaseModel):
    dedup_key: str = Field(..., max_length=200)
    rating: str
    reason: str = Field("", max_length=500)

    @field_validator("rating")
    @classmethod
    def _check_rating(cls, v: str) -> str:
        if v not in _VALID_ANTICIPATE_RATINGS:
            raise ValueError(f"rating must be one of {sorted(_VALID_ANTICIPATE_RATINGS)}")
        return v


def submit_anticipate_feedback(req: AnticipateFeedbackPayload, request: Request) -> dict:
    _rate_limit_behavior(request)
    from rag_anticipate.feedback import record_feedback as _rec_feedback

    try:
        ok = _rec_feedback(req.dedup_key, req.rating, reason=req.reason or "", source="wa")
    except Exception as exc:
        return {"ok": False, "reason": f"helper raised: {exc}"}
    if not ok:
        return {"ok": False, "reason": "write failed (telemetry unavailable)"}
    return {"ok": True}


_VALID_BRIEF_RATINGS: frozenset[str] = frozenset({"positive", "negative", "mute"})


class BriefFeedbackPayload(BaseModel):
    dedup_key: str = Field(..., max_length=400)
    rating: str
    reason: str = Field("", max_length=500)

    @field_validator("rating")
    @classmethod
    def _check_rating(cls, v: str) -> str:
        if v not in _VALID_BRIEF_RATINGS:
            raise ValueError(f"rating must be one of {sorted(_VALID_BRIEF_RATINGS)}")
        return v


def submit_brief_feedback(req: BriefFeedbackPayload, request: Request) -> dict:
    _rate_limit_behavior(request)
    from rag import _record_brief_feedback

    try:
        row_id = _record_brief_feedback(
            dedup_key=req.dedup_key,
            rating=req.rating,
            reason=req.reason or "",
            source="wa",
        )
    except Exception as exc:
        return {"ok": False, "reason": f"helper raised: {exc}"}
    if row_id is None:
        return {"ok": False, "reason": "write failed (telemetry unavailable)"}
    return {"ok": True, "id": int(row_id)}


_BEHAVIOR_KNOWN_EVENTS = frozenset({
    "open",
    "open_external",
    "positive_implicit",
    "negative_implicit",
    "kept",
    "deleted",
    "save",
    "copy",
    "query_response",
})
_BEHAVIOR_KNOWN_SOURCES = frozenset({"web", "whatsapp"})
_BEHAVIOR_SESSION_RE = re.compile(r"^[A-Za-z0-9_.@:-]{1,80}$")


class BehaviorRequest(BaseModel):
    source: str
    event: str
    query: str | None = None
    path: str | None = None
    rank: int | None = None
    dwell_ms: int | None = None
    session: str | None = None
    paths_json: str | None = None


def submit_behavior(req: BehaviorRequest, request: Request) -> dict:
    deps = _deps()
    if req.source not in _BEHAVIOR_KNOWN_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown source '{req.source}'; valid: {sorted(_BEHAVIOR_KNOWN_SOURCES)}",
        )
    if req.event not in _BEHAVIOR_KNOWN_EVENTS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown event '{req.event}'; valid: {sorted(_BEHAVIOR_KNOWN_EVENTS)}",
        )

    paths_list_validated: list[str] | None = None
    if req.paths_json is not None:
        try:
            parsed = json.loads(req.paths_json)
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(status_code=422, detail="paths_json must be a JSON-encoded array")
        if not isinstance(parsed, list):
            raise HTTPException(status_code=422, detail="paths_json must decode to a JSON array")
        cleaned: list[str] = []
        for path in parsed:
            if not isinstance(path, str) or not path:
                raise HTTPException(status_code=422, detail="paths_json entries must be non-empty strings")
            if "://" in path:
                raise HTTPException(
                    status_code=422,
                    detail="paths_json entries must be vault-relative (no URI schemes)",
                )
            cleaned.append(path)
        paths_list_validated = cleaned

    if req.session is not None and not _BEHAVIOR_SESSION_RE.match(req.session):
        raise HTTPException(status_code=400, detail="session id format invalid")

    if req.path is not None:
        path = req.path
        if "://" in path:
            raise HTTPException(status_code=400, detail="path must be vault-relative (no URI schemes)")
        if path.startswith("/") or ".." in path.split("/"):
            raise HTTPException(status_code=400, detail="path must be vault-relative")
        try:
            vault_path = deps.vault_path()
            resolved = (vault_path / path).resolve()
            resolved.relative_to(vault_path.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="path escapes vault root")

    _rate_limit_behavior(request)

    dwell_s = (req.dwell_ms / 1000.0) if req.dwell_ms is not None else None
    query_val = req.query
    if (query_val is None or query_val == "") and req.session:
        try:
            with deps.ragvec_state_conn() as conn:
                row = conn.execute(
                    "SELECT q FROM rag_queries WHERE session = ? AND q IS NOT NULL "
                    "AND q <> '' ORDER BY ts DESC LIMIT 1",
                    (req.session,),
                ).fetchone()
                if row and row[0]:
                    query_val = row[0]
        except sqlite3.Error:
            pass

    original_query_id: int | None = None
    if req.event == "open" and req.session:
        try:
            with deps.ragvec_state_conn() as conn:
                row = conn.execute(
                    "SELECT id FROM rag_queries WHERE session = ? "
                    "AND q IS NOT NULL AND q != '' "
                    "ORDER BY ts DESC LIMIT 1",
                    (req.session,),
                ).fetchone()
                if row:
                    original_query_id = int(row[0])
        except sqlite3.Error:
            pass

    try:
        event_payload: dict = {
            "source": req.source,
            "event": req.event,
            "query": query_val,
            "path": req.path,
            "rank": req.rank,
            "dwell_s": dwell_s,
            "session": req.session,
        }
        if paths_list_validated:
            event_payload["paths_json"] = paths_list_validated
        if original_query_id is not None:
            event_payload["original_query_id"] = original_query_id
        deps.log_behavior_event(event_payload)
    except Exception as exc:
        print(f"[behavior] write error: {exc}", flush=True)
        raise HTTPException(status_code=503, detail="event log unavailable")

    return {"ok": True}


def register_interaction_routes(app, deps: InteractionRouteDeps) -> dict[str, object]:
    global _DEPS
    _DEPS = deps
    app.post("/api/feedback")(submit_feedback)
    app.post("/api/mood")(submit_mood)
    app.get("/api/patterns")(cross_source_patterns)
    app.get("/api/mood/history")(mood_history)
    app.post("/api/draft/decision")(submit_draft_decision)
    app.post("/api/draft/preview")(submit_draft_preview)
    app.post("/api/anticipate/feedback")(submit_anticipate_feedback)
    app.post("/api/brief/feedback")(submit_brief_feedback)
    app.post("/api/behavior")(submit_behavior)
    return {name: globals()[name] for name in __all__ if name != "register_interaction_routes"}
