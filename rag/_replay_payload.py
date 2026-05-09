"""Replay payload — hashes + opt-in raw payloads para rag_queries.extra_json.

Phase 5 cont de modularización (audit perf 2026-05-08, ROI 155).

Persiste en `extra_json` de `rag_queries` los campos necesarios para
replay determinístico de queries: hashes siempre (minimal overhead),
payloads raw sólo cuando `RAG_LOG_REPLAY_PAYLOAD=1` (privacy-gated,
default OFF).

## Por qué separar hash de payload

- Los hashes (16 chars c/u) son inertes para la privacidad y habilitan
  dedup de queries idénticas en análisis downstream.
- Los payloads raw (`response_text`, `history_snapshot`) pueden contener
  PII del vault — se gatean con opt-in explícito.

## Storage cost estimado

- Flags OFF (default): ~30 KB/sem (3x16-char hashes por row).
- Flags ON: ~3.5 MB/sem (~50 queries/dia x 2.3 KB/row promedio).

## Env vars

  RAG_LOG_REPLAY_PAYLOAD=1  — activa response_text + history_snapshot
  RAG_LOG_RERANK_RAW=1      — activa rerank_logits_raw (list[float])

## API

- `_REPLAY_RESPONSE_MAX_BYTES`, `_REPLAY_HISTORY_MAX_BYTES` — caps de bytes.
- `_replay_payload_enabled()` → bool.
- `_rerank_raw_enabled()` → bool.
- `_truncate_for_replay(text, max_bytes)` → (text, truncated).
- `_replay_hash(text)` → sha256[:16].
- `_build_replay_fields(*, response, history, prompt, corpus_hash, rerank_logits)` → dict.

## Lazy imports

Sin deps de `rag/__init__.py` — solo stdlib (os, json, hashlib). Por eso
queda fácil de testear standalone.

## Re-export

`rag/__init__.py` hace `from rag._replay_payload import *` con `__all__`
explícito. Preserva 100% compat con `rag._replay_hash`,
`rag._build_replay_fields`, etc.
"""

from __future__ import annotations

import hashlib
import json
import os

__all__ = [
    "_REPLAY_HISTORY_MAX_BYTES",
    "_REPLAY_RESPONSE_MAX_BYTES",
    "_build_replay_fields",
    "_replay_hash",
    "_replay_payload_enabled",
    "_rerank_raw_enabled",
    "_truncate_for_replay",
]


_REPLAY_RESPONSE_MAX_BYTES: int = 8_192
_REPLAY_HISTORY_MAX_BYTES: int = 4_096


def _replay_payload_enabled() -> bool:
    """True cuando RAG_LOG_REPLAY_PAYLOAD=1 -- activa storage de
    response_text + history_snapshot en extra_json.

    Default OFF -- los payloads pueden contener PII del vault.
    """
    val = os.environ.get("RAG_LOG_REPLAY_PAYLOAD", "").strip().lower()
    return val in ("1", "true", "yes")


def _rerank_raw_enabled() -> bool:
    """True cuando RAG_LOG_RERANK_RAW=1 -- activa storage de
    rerank_logits_raw (list[float]) en extra_json.

    Default OFF -- util solo para debugging de regresiones del ranker.
    """
    val = os.environ.get("RAG_LOG_RERANK_RAW", "").strip().lower()
    return val in ("1", "true", "yes")


def _truncate_for_replay(text: str, max_bytes: int) -> "tuple[str, bool]":
    """Trunca text a max_bytes (UTF-8) si supera el limite.

    Retorna (texto_truncado, fue_truncado). La truncacion preserva
    codepoints UTF-8 completos (no corta en medio de un multibyte).
    """
    if not text:
        return ("", False)
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return (text, False)
    truncated_text = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return (truncated_text, True)


def _replay_hash(text: str) -> str:
    """sha256[:16] de text (UTF-8). Retorna '' si text esta vacio."""
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _build_replay_fields(
    *,
    response: str | None = None,
    history: list | None = None,
    prompt: str | None = None,
    corpus_hash: str = "",
    rerank_logits: list | None = None,
) -> dict:
    """Construye el subset de campos de replay payload para extra_json.

    Campos ALWAYS ON (hashes, ~48 chars total):
      corpus_hash        -- hash del corpus snapshot
      prompt_hash        -- sha256[:16] del system+user prompt final
      response_hash      -- sha256[:16] del response del LLM

    Campos OPT-IN (RAG_LOG_REPLAY_PAYLOAD=1):
      response_text      -- raw response capped 8 KB
      response_truncated -- bool, True si excedio 8 KB
      history_snapshot   -- turns previos capped 4 KB
      history_truncated  -- bool, True si excedio 4 KB

    Campos OPT-IN (RAG_LOG_RERANK_RAW=1):
      rerank_logits_raw  -- list[float]

    Siempre retorna un dict (puede estar vacio si todos los inputs son
    None/vacios).
    """
    out: dict = {}

    if corpus_hash:
        out["corpus_hash"] = corpus_hash

    if prompt:
        out["prompt_hash"] = _replay_hash(prompt)

    response_str = response or ""
    if response_str:
        out["response_hash"] = _replay_hash(response_str)
        if _replay_payload_enabled():
            text_capped, truncated = _truncate_for_replay(
                response_str, _REPLAY_RESPONSE_MAX_BYTES,
            )
            out["response_text"] = text_capped
            out["response_truncated"] = truncated

    if history:
        try:
            history_json = json.dumps(history, ensure_ascii=False)
        except Exception:
            history_json = ""
        if history_json:
            out["history_hash"] = _replay_hash(history_json)
            if _replay_payload_enabled():
                snapshot_capped, hist_truncated = _truncate_for_replay(
                    history_json, _REPLAY_HISTORY_MAX_BYTES,
                )
                try:
                    out["history_snapshot"] = json.loads(snapshot_capped)
                except Exception:
                    out["history_snapshot"] = snapshot_capped
                out["history_truncated"] = hist_truncated

    if rerank_logits is not None and _rerank_raw_enabled():
        out["rerank_logits_raw"] = [round(float(x), 6) for x in rerank_logits]

    return out
