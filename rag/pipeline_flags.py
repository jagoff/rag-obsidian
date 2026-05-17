"""Runtime flags for retrieval and post-processing pipeline choices."""
from __future__ import annotations

import os

FALSY_ENV_VALUES = {"0", "false", "no"}

EXPAND_SKIP_INTENTS: frozenset[str] = frozenset({"comparison", "synthesis"})
GRAPH_ALWAYS_INTENTS: frozenset[str] = frozenset({"comparison", "synthesis"})
METADATA_ONLY_INTENTS: frozenset[str] = frozenset(
    {"count", "list", "recent", "agenda", "entity_lookup"}
)


def nli_grounding_enabled() -> bool:
    return os.environ.get("RAG_NLI_GROUNDING", "").strip() not in ("", *FALSY_ENV_VALUES)


def nli_contradicts_threshold() -> float:
    try:
        return float(os.environ.get("RAG_NLI_CONTRADICTS_THRESHOLD", "0.7"))
    except (ValueError, TypeError):
        return 0.7


def nli_skip_intents() -> frozenset[str]:
    intents_str = os.environ.get("RAG_NLI_SKIP_INTENTS", "count,list,recent,agenda")
    return frozenset(s.strip() for s in intents_str.split(",") if s.strip())


def nli_max_claims() -> int:
    try:
        return int(os.environ.get("RAG_NLI_MAX_CLAIMS", "20"))
    except (ValueError, TypeError):
        return 20


def lookup_threshold() -> float:
    return float(os.environ.get("RAG_LOOKUP_THRESHOLD", "0.6"))


def lookup_model_placeholder() -> str:
    return (
        os.environ.get("RAG_LOOKUP_MODEL", "").strip()
        or "qwen2.5:3b"
    )


def lookup_num_ctx() -> int:
    return int(os.environ.get("RAG_LOOKUP_NUM_CTX", "4096"))


def rerank_pool_by_intent() -> dict[str, int]:
    return {
        "comparison": int(os.environ.get("RAG_COMPARISON_POOL", "30")),
        "synthesis": int(os.environ.get("RAG_SYNTHESIS_POOL", "30")),
    }


def adaptive_routing() -> bool:
    val = os.environ.get("RAG_ADAPTIVE_ROUTING", "").strip().lower()
    return val not in FALSY_ENV_VALUES


def should_skip_reformulate(
    intent: str,
    metadata_only_intents: frozenset[str] = METADATA_ONLY_INTENTS,
) -> bool:
    if not adaptive_routing():
        return False
    return intent in metadata_only_intents


def entity_lookup_enabled() -> bool:
    val = os.environ.get("RAG_ENTITY_LOOKUP", "").strip().lower()
    return val not in FALSY_ENV_VALUES


__all__ = [
    "EXPAND_SKIP_INTENTS",
    "FALSY_ENV_VALUES",
    "GRAPH_ALWAYS_INTENTS",
    "METADATA_ONLY_INTENTS",
    "adaptive_routing",
    "entity_lookup_enabled",
    "lookup_model_placeholder",
    "lookup_num_ctx",
    "lookup_threshold",
    "nli_contradicts_threshold",
    "nli_grounding_enabled",
    "nli_max_claims",
    "nli_skip_intents",
    "rerank_pool_by_intent",
    "should_skip_reformulate",
]
