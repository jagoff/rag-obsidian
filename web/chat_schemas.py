"""Pydantic schemas and validation rules for the web chat API."""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from web.chat_uploads import safe_chat_upload_filename

CHAT_SESSION_RE = re.compile(r"^[A-Za-z0-9_.:@\-]{1,80}$")
TURN_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
CHAT_QUESTION_MAX = 512_000
CHAT_QUESTION_RUNTIME_MAX = 32_000
CHAT_ATTACHMENT_TEXT_MAX = 13_000
VALID_CHAT_MODES = {"work", "auto", "fast", "deep"}


class ChatAttachment(BaseModel):
    name: str = Field(default="archivo", max_length=220)
    content_type: str | None = Field(default=None, max_length=160)
    size: int | None = Field(default=None, ge=0, le=25 * 1024 * 1024)
    text: str | None = Field(default="", max_length=CHAT_ATTACHMENT_TEXT_MAX)
    path: str | None = Field(default=None, max_length=600)
    vault_path: str | None = Field(default=None, max_length=600)
    truncated: bool = False

    @field_validator("name")
    @classmethod
    def _check_attachment_name(cls, v: str) -> str:
        v = safe_chat_upload_filename(v or "archivo")
        return v or "archivo"

    @field_validator("path", "vault_path")
    @classmethod
    def _check_attachment_path(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        v = v.strip()
        if "\x00" in v:
            raise ValueError("invalid attachment path")
        if len(v) > 600:
            raise ValueError("attachment path too long")
        return v


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    vault_scope: str | None = None
    redo_turn_id: str | None = Field(None, max_length=80)
    hint: str | None = Field(None, max_length=500)
    mode: str | None = Field(None, max_length=16)
    folder: str | None = Field(None, max_length=500)
    path: str | None = Field(None, max_length=512)
    force: bool = False
    attachments: list[ChatAttachment] = Field(default_factory=list, max_length=6)

    @field_validator("question")
    @classmethod
    def _check_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must be non-empty")
        if len(v) > CHAT_QUESTION_MAX:
            raise ValueError(f"question too long (>{CHAT_QUESTION_MAX} chars)")
        return v

    @field_validator("redo_turn_id")
    @classmethod
    def _check_redo_turn_id(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if not TURN_ID_RE.match(v):
            raise ValueError("invalid redo_turn_id format")
        return v

    @field_validator("session_id")
    @classmethod
    def _check_session(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if not CHAT_SESSION_RE.match(v):
            # Stale browser sessionStorage should degrade to a new session
            # instead of making the whole turn fail with HTTP 422.
            return None
        return v

    @field_validator("vault_scope")
    @classmethod
    def _check_vault_scope(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if len(v) > 80 or not re.match(r"^[A-Za-z0-9_\-]{1,80}$", v):
            raise ValueError("invalid vault_scope format")
        return v

    @field_validator("folder")
    @classmethod
    def _check_folder(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        v = v.strip().strip("/")
        if not v:
            return None
        if "://" in v or v.startswith("/") or ".." in v.split("/"):
            raise ValueError("folder must be vault-relative (no URI/traversal)")
        return v

    @field_validator("path")
    @classmethod
    def _check_path(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        v = v.strip()
        if not v:
            return None
        if "://" in v or v.startswith("/") or ".." in v.split("/"):
            raise ValueError("path must be vault-relative (no URI/traversal)")
        return v


__all__ = [
    "CHAT_ATTACHMENT_TEXT_MAX",
    "CHAT_QUESTION_MAX",
    "CHAT_QUESTION_RUNTIME_MAX",
    "CHAT_SESSION_RE",
    "ChatAttachment",
    "ChatRequest",
    "TURN_ID_RE",
    "VALID_CHAT_MODES",
]
