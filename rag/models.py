"""Model tier registry — single source of truth.

6 tiers, env-var driven, hot-reload via per-tier callbacks. Diseñado para
que cambiar de modelo sea trivial:

    rag model set chat qwen2.5:14b      # CLI
    rag.models.swap("chat", "qwen2.5:14b")  # programático

## Tiers

| tier   | env var              | default                                       | constraint                       |
|--------|----------------------|-----------------------------------------------|----------------------------------|
| chat   | `RAG_CHAT_MODEL`     | `qwen2.5:7b`                                  | alias en `MLX_MODEL_ALIAS`       |
| helper | `RAG_HELPER_MODEL`   | `qwen2.5:3b`                                  | alias en `MLX_MODEL_ALIAS`       |
| embed  | `RAG_EMBED_MODEL`    | `qwen3-embedding:0.6b`                        | dim 1024 (vault index lockstep)  |
| rerank | `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` (PT) o Qwen3 (MLX)  | depende `RAG_RERANKER_BACKEND`   |
| stt    | `RAG_STT_MODEL`      | `small`                                       | alias en `_WHISPER_NAME_TO_HF`   |
| vlm    | `RAG_VLM_MODEL`      | `mlx-community/granite-vision-3.2-2b-4bit`    | mlx-vlm compatible               |

## Hot-reload

`swap(tier, model)` corre los reload hooks registrados para ese tier
(invalida caches in-process, descarga modelo viejo del backend MLX).
Cada módulo dueño registra su hook al importar (chat/helper/embed en
`rag/__init__.py`, stt en `rag/whisper.py`, vlm en `rag/ocr.py`).

## Persistencia

Sólo env vars. `set_env(tier, model)` mutar el env del proceso actual.
Para que sobreviva restarts de daemons, el CLI `rag model set --persist`
re-escribe los plists relevantes y dispara `launchctl kickstart`.

## Per-feature overrides preservados

`RAG_LOOKUP_MODEL`, `RAG_LLM_INTENT_MODEL`, `RAG_DRAFTS_FT_BASE_MODEL_MLX`
y `RAG_MLX_RERANKER_MODEL` siguen funcionando como overrides puntuales.
Si NO están seteados, caen al tier correspondiente (helper / rerank).
"""

from __future__ import annotations

import os
from typing import Callable

__all__ = [
    "TIERS",
    "DEFAULTS",
    "ENV_VARS",
    "all_active",
    "get",
    "list_available",
    "register_reload_hook",
    "reset_env",
    "set_env",
    "swap",
    "validate",
]


TIERS: tuple[str, ...] = ("chat", "helper", "embed", "rerank", "stt", "vlm")

DEFAULTS: dict[str, str] = {
    "chat":   "qwen2.5:7b",
    "helper": "qwen2.5:3b",
    "embed":  "qwen3-embedding:0.6b",
    "rerank": "BAAI/bge-reranker-v2-m3",
    "stt":    "small",
    "vlm":    "mlx-community/granite-vision-3.2-2b-4bit",
}

ENV_VARS: dict[str, str] = {tier: f"RAG_{tier.upper()}_MODEL" for tier in TIERS}

# Whitelist de embedders compatibles con el index vigente (dim 1024).
# Cambiar a un modelo fuera de esta lista requiere `rag index --full` y
# bump de `_COLLECTION_BASE` en `rag/__init__.py` — bloqueado por default.
_EMBED_WHITELIST: frozenset[str] = frozenset({
    "qwen3-embedding:0.6b",
    "mlx-community/Qwen3-Embedding-0.6B-8bit",
})

_RELOAD_HOOKS: dict[str, list[Callable[[str, str], None]]] = {
    tier: [] for tier in TIERS
}


def get(tier: str) -> str:
    """Modelo activo para `tier`. Env var → default. Re-evaluado cada call."""
    if tier not in TIERS:
        raise ValueError(f"Tier desconocido {tier!r}; válidos: {TIERS}")
    val = os.environ.get(ENV_VARS[tier], "").strip()
    return val or DEFAULTS[tier]


def all_active() -> dict[str, str]:
    """Snapshot {tier: modelo}."""
    return {tier: get(tier) for tier in TIERS}


def set_env(tier: str, model: str) -> str:
    """Setea env var del tier in-process. Devuelve el modelo previo."""
    if tier not in TIERS:
        raise ValueError(f"Tier desconocido {tier!r}")
    prev = get(tier)
    os.environ[ENV_VARS[tier]] = model
    return prev


def reset_env(tier: str) -> str:
    """Borra el override del tier (vuelve al default). Dispara reload hooks.

    Devuelve modelo previo.
    """
    if tier not in TIERS:
        raise ValueError(f"Tier desconocido {tier!r}")
    prev = get(tier)
    os.environ.pop(ENV_VARS[tier], None)
    new = get(tier)
    if prev != new:
        for hook in _RELOAD_HOOKS[tier]:
            try:
                hook(prev, new)
            except Exception as exc:
                import sys
                print(f"[models] reload hook failed ({tier}): {exc}", file=sys.stderr)
    return prev


def register_reload_hook(tier: str, fn: Callable[[str, str], None]) -> None:
    """Registrar callback `fn(old_model, new_model)` que corre tras swap()."""
    if tier not in _RELOAD_HOOKS:
        raise ValueError(f"Tier desconocido {tier!r}")
    _RELOAD_HOOKS[tier].append(fn)


def swap(tier: str, model: str, *, preload: bool = False, unsafe: bool = False) -> str:
    """Hot-swap del modelo de un tier.

    1. Valida (a menos que `unsafe=True`).
    2. Set env var.
    3. Corre reload hooks (unloadea modelo viejo, invalida caches).
    4. Opcional `preload=True`: warm-up del modelo nuevo.

    Devuelve el modelo previo. Lanza `ValueError` si la validación falla.
    """
    if not unsafe:
        err = validate(tier, model)
        if err:
            raise ValueError(err)
    old = set_env(tier, model)
    for hook in _RELOAD_HOOKS[tier]:
        try:
            hook(old, model)
        except Exception as exc:
            # Hooks no deben bloquear el swap, pero loggeamos
            import sys
            print(f"[models] reload hook failed ({tier}): {exc}", file=sys.stderr)
    if preload:
        _preload(tier, model)
    return old


def list_available(tier: str) -> dict[str, list[str]]:
    """Lista modelos para un tier separados en `cached` / `known`.

    `cached`: snapshots ya bajados a `~/.cache/huggingface/hub/`.
    `known`:  aliases conocidos pero no cacheados (descargables on-demand).
    """
    if tier not in TIERS:
        raise ValueError(f"Tier desconocido {tier!r}")
    if tier in ("chat", "helper"):
        return _list_mlx_chat_helper(tier)
    if tier == "embed":
        return _list_with_cache(_EMBED_WHITELIST)
    if tier == "rerank":
        return _list_rerank()
    if tier == "stt":
        return _list_stt()
    if tier == "vlm":
        return _list_vlm()
    return {"cached": [], "known": []}


def validate(tier: str, model: str) -> str | None:
    """Devuelve mensaje de error si `model` no es válido para `tier`. None = OK."""
    if tier not in TIERS:
        return f"Tier desconocido {tier!r}; válidos: {TIERS}"
    if not model or not model.strip():
        return f"Modelo vacío para tier {tier!r}"
    model = model.strip()
    if tier == "embed":
        if model not in _EMBED_WHITELIST:
            return (
                f"Embed model {model!r} no está en la whitelist "
                f"(dim 1024 obligatoria para `_COLLECTION_BASE=obsidian_notes_v12`). "
                f"Cambiar a un embedder con dim distinto invalida el index — "
                f"forzá con `--unsafe` y corré `rag index --full` después."
            )
    if tier == "stt":
        from rag.whisper import _WHISPER_NAME_TO_HF
        if model not in _WHISPER_NAME_TO_HF and "/" not in model:
            return (
                f"Whisper alias desconocido: {model!r}. "
                f"Conocidos: {sorted(_WHISPER_NAME_TO_HF)}. "
                f"O pasá un HF repo full (`mlx-community/whisper-...`)."
            )
    if tier in ("chat", "helper"):
        from rag.llm_backend import MLX_MODEL_ALIAS
        if not model.startswith("mlx-community/") and model not in MLX_MODEL_ALIAS:
            return (
                f"Alias {model!r} no está en `MLX_MODEL_ALIAS`. "
                f"Aliases: {sorted(MLX_MODEL_ALIAS)}. "
                f"O pasá HF id (`mlx-community/...`)."
            )
    if tier == "vlm":
        if not model.startswith("mlx-community/") and "/" not in model:
            return f"VLM esperado como HF repo (`mlx-community/...`), recibido: {model!r}"
    return None


# ── Internals ──────────────────────────────────────────────────────────────


def _hf_cache_ids() -> set[str]:
    """Set de HF ids ya descargados a `~/.cache/huggingface/hub/`.

    Match laxo: cualquier `models--<org>--<name>` cuenta como `<org>/<name>`.
    """
    from pathlib import Path
    root = Path.home() / ".cache" / "huggingface" / "hub"
    if not root.exists():
        return set()
    out: set[str] = set()
    for entry in root.iterdir():
        name = entry.name
        if not name.startswith("models--"):
            continue
        parts = name[len("models--"):].split("--", 1)
        if len(parts) == 2:
            out.add(f"{parts[0]}/{parts[1]}")
    return out


def _list_mlx_chat_helper(tier: str) -> dict[str, list[str]]:
    from rag.llm_backend import MLX_MODEL_ALIAS
    helper_aliases = {"qwen2.5:3b", "qwen3:4b"}
    chat_aliases = {
        "qwen2.5:7b", "qwen2.5:14b", "qwen3:30b-a3b",
        "command-r", "command-r:latest",
    }
    wanted = chat_aliases if tier == "chat" else helper_aliases
    cached_hf = _hf_cache_ids()
    cached: list[str] = []
    known: list[str] = []
    for alias in sorted(wanted):
        hf_id = MLX_MODEL_ALIAS.get(alias, "")
        if hf_id and hf_id in cached_hf:
            cached.append(alias)
        else:
            known.append(alias)
    return {"cached": cached, "known": known}


def _list_with_cache(aliases: frozenset[str]) -> dict[str, list[str]]:
    cached_hf = _hf_cache_ids()
    cached: list[str] = []
    known: list[str] = []
    for alias in sorted(aliases):
        hf_id = alias if "/" in alias else None
        if hf_id is None:
            from rag.llm_backend import MLX_MODEL_ALIAS
            hf_id = MLX_MODEL_ALIAS.get(alias, "")
        if hf_id and hf_id in cached_hf:
            cached.append(alias)
        else:
            known.append(alias)
    return {"cached": cached, "known": known}


def _list_rerank() -> dict[str, list[str]]:
    aliases = {
        "BAAI/bge-reranker-v2-m3",
        "qwen3-reranker:0.6b",
        "qwen3-reranker:4b",
        "qwen3-reranker:8b",
    }
    cached_hf = _hf_cache_ids()
    cached: list[str] = []
    known: list[str] = []
    try:
        from rag.mlx_reranker import MLX_RERANKER_ALIASES as _aliases_map
    except Exception:
        _aliases_map = {}
    for alias in sorted(aliases):
        hf_id = alias if "/" in alias else _aliases_map.get(alias, "")
        if hf_id and hf_id in cached_hf:
            cached.append(alias)
        else:
            known.append(alias)
    return {"cached": cached, "known": known}


def _list_stt() -> dict[str, list[str]]:
    from rag.whisper import _WHISPER_NAME_TO_HF
    cached_hf = _hf_cache_ids()
    cached: list[str] = []
    known: list[str] = []
    for alias, hf_id in sorted(_WHISPER_NAME_TO_HF.items()):
        if hf_id in cached_hf:
            cached.append(alias)
        else:
            known.append(alias)
    return {"cached": cached, "known": known}


def _list_vlm() -> dict[str, list[str]]:
    aliases = [
        "mlx-community/granite-vision-3.2-2b-4bit",
        "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    ]
    cached_hf = _hf_cache_ids()
    cached: list[str] = []
    known: list[str] = []
    for alias in aliases:
        if alias in cached_hf:
            cached.append(alias)
        else:
            known.append(alias)
    return {"cached": cached, "known": known}


def _preload(tier: str, model: str) -> None:
    """Best-effort warm-up post-swap. Silent fail."""
    try:
        if tier in ("chat", "helper"):
            from rag.llm_backend import MLXBackend, to_mlx
            MLXBackend()._load(to_mlx(model))
    except Exception:
        pass
