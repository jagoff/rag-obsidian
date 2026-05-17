"""Tests for web/server.py Pydantic request models — parse-time validation.

Not a full /api/* integration suite; just verifies the size caps on
FeedbackRequest + ChatRequest reject oversized payloads with 422
before the handler runs. These caps are defence against a malicious
or buggy client posting multi-MB reason/query strings that would
otherwise sit in memory (FastAPI allocates the full body before
dispatch).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# web.server imports rag (via `from rag import ...`), which has heavy
# side-effects on module init. Skip if it can't import (e.g. during a
# pure-rag test run without web deps).
_web_server = pytest.importorskip("web.server")
FeedbackRequest = _web_server.FeedbackRequest
ChatRequest = _web_server.ChatRequest
app = _web_server.app


# ── FeedbackRequest caps ──────────────────────────────────────────────────────


def _valid_feedback_base() -> dict:
    return {"turn_id": "abc123", "rating": 1}


def test_feedback_accepts_minimal_valid_payload():
    req = FeedbackRequest(**_valid_feedback_base())
    assert req.turn_id == "abc123"
    assert req.rating == 1
    assert req.reason is None


def test_feedback_rejects_oversized_reason():
    """reason > 500 chars must raise ValidationError at parse time.
    Downstream trims to 200; the Pydantic cap is higher to accept
    padded input that the handler strips."""
    payload = {**_valid_feedback_base(), "reason": "x" * 501}
    with pytest.raises(ValidationError) as exc_info:
        FeedbackRequest(**payload)
    # Pydantic v2 error type is "string_too_long".
    assert "string_too_long" in str(exc_info.value)


def test_feedback_accepts_reason_at_cap():
    payload = {**_valid_feedback_base(), "reason": "x" * 500}
    req = FeedbackRequest(**payload)
    assert req.reason is not None
    assert len(req.reason) == 500


def test_feedback_rejects_oversized_q():
    payload = {**_valid_feedback_base(), "q": "x" * 2001}
    with pytest.raises(ValidationError) as exc_info:
        FeedbackRequest(**payload)
    assert "string_too_long" in str(exc_info.value)


def test_feedback_rejects_too_many_paths():
    """paths list capped at 50 items — a real retrieve tops out around
    k=20, 50 is generous but prevents a multi-thousand paths payload
    from being allocated."""
    payload = {**_valid_feedback_base(), "paths": [f"p{i}.md" for i in range(51)]}
    with pytest.raises(ValidationError) as exc_info:
        FeedbackRequest(**payload)
    assert "too_long" in str(exc_info.value).lower()


def test_feedback_accepts_paths_at_cap():
    payload = {**_valid_feedback_base(), "paths": [f"p{i}.md" for i in range(50)]}
    req = FeedbackRequest(**payload)
    assert req.paths is not None
    assert len(req.paths) == 50


# ── FeedbackRequest.corrective_path (2026-04-22) ─────────────────────────────
# Mirrors the CLI corrective prompt (rag.py:~18997). Pre-fix the web had no
# way to surface the positive signal — rag_feedback had 0 web-sourced rows
# with corrective_path, starving the reranker fine-tune of clean pairs from
# 80% of traffic.


def test_feedback_accepts_corrective_path():
    payload = {**_valid_feedback_base(), "rating": -1,
               "corrective_path": "02-Areas/Salud/postura.md"}
    req = FeedbackRequest(**payload)
    assert req.corrective_path == "02-Areas/Salud/postura.md"


def test_feedback_corrective_path_optional():
    """Corrective_path is always optional — a bare rating must still parse."""
    req = FeedbackRequest(**_valid_feedback_base())
    assert req.corrective_path is None


def test_feedback_rejects_oversized_corrective_path():
    """Cap at 512 chars — longest realistic vault path is ~300 but we want
    a buffer without letting a multi-KB payload through."""
    payload = {**_valid_feedback_base(), "rating": -1,
               "corrective_path": "x" * 513}
    with pytest.raises(ValidationError) as exc_info:
        FeedbackRequest(**payload)
    assert "string_too_long" in str(exc_info.value)


def test_feedback_accepts_corrective_path_at_cap():
    payload = {**_valid_feedback_base(), "rating": -1,
               "corrective_path": "x" * 512}
    req = FeedbackRequest(**payload)
    assert req.corrective_path is not None
    assert len(req.corrective_path) == 512


# ── ChatRequest caps (pre-existing) ──────────────────────────────────────────


def test_chat_rejects_oversized_question():
    """ChatRequest.question cap raised from 64000 to 512000 on 2026-05-17.

    Large work-mode pastes are compacted inside the handler before retrieval
    and model calls. Test uses 512001 to cross the transport cap — bump
    here AND in web/server.py's
    `_CHAT_QUESTION_MAX` if you raise it again.
    Implemented via a field_validator, so Pydantic error type is
    'value_error' (not 'string_too_long' like the Field-based caps)."""
    with pytest.raises(ValidationError) as exc_info:
        ChatRequest(question="x" * 512001)
    # field_validator raises ValueError → Pydantic type is "value_error"
    # (not "string_too_long" like the Field-based caps above).
    assert "too long" in str(exc_info.value).lower()


def test_chat_accepts_question_at_cap():
    req = ChatRequest(question="x" * 512000)
    assert len(req.question) == 512000


def test_chat_accepts_large_paste_above_runtime_budget():
    req = ChatRequest(question="x" * 100000)
    assert len(req.question) == 100000


def test_compact_long_chat_question_keeps_head_and_tail():
    text = "A" * 40000 + "MIDDLE" + "Z" * 40000
    compacted, changed = _web_server._compact_long_chat_question(
        text, max_chars=1000,
    )
    assert changed is True
    assert len(compacted) <= 1000
    assert compacted.startswith("A" * 100)
    assert compacted.endswith("Z" * 100)
    assert "contenido pegado muy largo" in compacted


def test_api_chat_oversized_question_does_not_echo_input():
    resp = TestClient(app).post("/api/chat", json={"question": "x" * 512001})
    assert resp.status_code == 422
    body = resp.text
    assert "question too long (>512000 chars)" in body
    assert len(body) < 1000
    assert "x" * 100 not in body


def test_chat_rejects_empty_question():
    """Empty question is rejected by field_validator."""
    with pytest.raises(ValidationError):
        ChatRequest(question="")


@pytest.mark.parametrize(
    "bad_session_id",
    [
        "web:" + ("x" * 120),
        "web:bad/path",
        "web:bad space",
    ],
)
def test_chat_invalid_session_id_degrades_to_new_session(bad_session_id):
    """Stale browser sessionStorage should not make /api/chat return 422.

    The handler will mint a fresh ``web:<uuid>`` when the parsed value is
    None; the lower session storage layer still validates before disk I/O.
    """
    req = ChatRequest(question="hola", session_id=bad_session_id)
    assert req.session_id is None
