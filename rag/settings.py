"""Centralised environment settings for the RAG server.

All os.environ lookups happen once at import time.  This avoids scattered
magic strings, makes defaults discoverable, and gives us type hints for
 every knob.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _bool_env(name: str, default: bool = True) -> bool:
    """Parse a truthy/falsy env var.  '0', 'false', 'no' → False."""
    val = os.environ.get(name, "1" if default else "0").strip().lower()
    return val not in ("0", "false", "no")


def _bool_env_opt(name: str, default: bool = False) -> bool:
    """Parse an opt-in env var.  '1', 'true', 'yes' → True."""
    val = os.environ.get(name, "1" if default else "0").strip().lower()
    return val in ("1", "true", "yes")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _str_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True, slots=True)
class RAGSettings:
    """Immutable snapshot of the process environment at import time."""

    # ---- web / server ----
    bind_host: str = _str_env("OBSIDIAN_RAG_BIND_HOST", "127.0.0.1") or "127.0.0.1"
    threadpool_tokens: int = _int_env("RAG_WEB_THREADPOOL_TOKENS", 100)
    prompt_version: str = _str_env("RAG_WEB_PROMPT_VERSION", "v2")
    rerank_pool: int = _int_env("RAG_WEB_RERANK_POOL", 3)
    chat_cache_ttl: float = _float_env("RAG_WEB_CHAT_CACHE_TTL", 86_400.0)
    chat_cache_max: int = _int_env("RAG_WEB_CHAT_CACHE_MAX", 100)
    sse_max_per_ip: int = _int_env("RAG_SSE_MAX_PER_IP", 3)

    # ---- LLM backend ----
    llm_backend: str = _str_env("RAG_LLM_BACKEND", "mlx")
    mlx_keep_fallback: bool = _bool_env_opt("RAG_MLX_KEEP_FALLBACK")

    # ---- context / retrieval ----
    context_purge_stale: bool = _bool_env("RAG_CONTEXT_PURGE_STALE")
    context_dedup_by_file: bool = _bool_env("RAG_CONTEXT_DEDUP_BY_FILE")

    # ---- NLI / grounding ----
    nli_grounding_budget_s: float = _float_env("RAG_NLI_GROUNDING_BUDGET_S", 4.0)
    critique_enabled: bool = _bool_env_opt("RAG_CRITIQUE_ENABLED")

    # ---- PII / redaction ----
    pii_redact_user_output: bool = _bool_env("RAG_PII_REDACT_USER_OUTPUT")

    # ---- metrics / logging ----
    metrics_async: bool = _bool_env_opt("RAG_METRICS_ASYNC")

    # ---- auto-fix worker ----
    auto_fix_worker: bool = _bool_env_opt("RAG_AUTO_FIX_WORKER")
    auto_fix_hourly_cap: int = _int_env("RAG_AUTO_FIX_HOURLY_CAP", 12)

    # ---- prewarm ----
    home_prewarm: bool = _bool_env("OBSIDIAN_RAG_HOME_PREWARM")
    chat_prewarm_interval: int = _int_env("OBSIDIAN_RAG_CHAT_PREWARM_INTERVAL", 240)

    # ---- history / routing ----
    history_summary: bool = _bool_env_opt("RAG_HISTORY_SUMMARY_ON")
    tool_llm_decide: bool = _bool_env_opt("RAG_WEB_TOOL_LLM_DECIDE")

    # ---- static / LAN / tunnel ----
    static_no_cache: bool = _bool_env_opt("OBSIDIAN_RAG_STATIC_NO_CACHE")
    allow_lan: bool = _bool_env_opt("OBSIDIAN_RAG_ALLOW_LAN")
    allow_tunnel: bool = _bool_env_opt("OBSIDIAN_RAG_ALLOW_TUNNEL")

    # ---- chat / OCR ----
    web_chat_model: str | None = _str_env("OBSIDIAN_RAG_WEB_CHAT_MODEL") or None
    ocr_historical_max_days: int = _int_env("RAG_OCR_HISTORICAL_MAX_DAYS", 30)
    anticipate_disabled: bool = _bool_env_opt("RAG_ANTICIPATE_DISABLED")


# Single global instance — read once, used everywhere.
settings = RAGSettings()
