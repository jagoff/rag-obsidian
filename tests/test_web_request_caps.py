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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# web.server imports rag (via `from rag import ...`), which has heavy
# side-effects on module init. Skip if it can't import (e.g. during a
# pure-rag test run without web deps).
_web_server = pytest.importorskip("web.server")
FeedbackRequest = _web_server.FeedbackRequest
ChatRequest = _web_server.ChatRequest


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


# ── ChatRequest caps (pre-existing) ──────────────────────────────────────────


def test_chat_rejects_oversized_question():
    """ChatRequest.question cap at 8000 was already in place pre-2026-04-20
    audit (via a field_validator, not Field(max_length=...)); this test
    guards against accidental removal during refactors."""
    with pytest.raises(ValidationError) as exc_info:
        ChatRequest(question="x" * 8001)
    # field_validator raises ValueError → Pydantic type is "value_error"
    # (not "string_too_long" like the Field-based caps above).
    assert "too long" in str(exc_info.value).lower()


def test_chat_accepts_question_at_cap():
    req = ChatRequest(question="x" * 8000)
    assert len(req.question) == 8000


def test_chat_rejects_empty_question():
    """Empty question is rejected by field_validator."""
    with pytest.raises(ValidationError):
        ChatRequest(question="")
