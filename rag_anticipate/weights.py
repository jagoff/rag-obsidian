"""User-configurable signal weights for the Anticipatory Agent.

Cada `AnticipatoryCandidate` tiene un `kind` (e.g. `anticipate-calendar`,
`anticipate-echo`, `anticipate-anniversary`, ...) y un `score` ∈ [0, 1]. Antes
del top-1 pick se multiplica el score por el weight del kind (default 1.0
implícito si el kind no está configurado). Esto permite priorizar/desprioritizar
tipos de notif sin tocar code — sólo editás un JSON.

Storage: `~/.local/share/obsidian-rag/anticipate_weights.json`
Schema:  `{"anticipate-calendar": 1.5, "anticipate-echo": 0.7, ...}`
Range:   [0.0, 5.0] por weight (clamp soft via filtering, no exception).

Silent-fail por todos lados: archivo inexistente / JSON malformado / values
fuera de rango → se ignoran y se devuelve dict vacío (= todos los weights = 1.0
implícitos). Nunca tumba el orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path

WEIGHTS_PATH = Path.home() / ".local/share/obsidian-rag/anticipate_weights.json"


def load_weights() -> dict[str, float]:
    """Lee `WEIGHTS_PATH`. Default = `{}` (sin weights = 1.0 implícito).

    Schema: `{"anticipate-calendar": 1.5, "anticipate-echo": 0.7, ...}`.
    Silent-fail → `{}` ante archivo inexistente, JSON inválido, root no-dict, etc.
    Filtra silenciosamente claves no-string y valores fuera de `[0.0, 5.0]`.
    """
    if not WEIGHTS_PATH.is_file():
        return {}
    try:
        data = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        out: dict[str, float] = {}
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool):
                if 0.0 <= float(v) <= 5.0:
                    out[k] = float(v)
        return out
    except Exception:
        return {}


def save_weights(weights: dict[str, float]) -> bool:
    """Persist `weights` a `WEIGHTS_PATH` atómicamente (write tmp + rename).

    Filtra valores inválidos (out-of-range, type mismatch) antes de escribir.
    Returns `True` si OK, `False` si falló alguna fase (mkdir / write / rename).
    """
    try:
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        clean: dict[str, float] = {}
        for k, v in weights.items():
            if isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool):
                if 0.0 <= float(v) <= 5.0:
                    clean[k] = float(v)
        tmp = WEIGHTS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(WEIGHTS_PATH)
        return True
    except Exception:
        return False


def set_weight(kind: str, weight: float) -> bool:
    """Update single `kind` weight. Returns `False` si está fuera de `[0, 5]`."""
    if not (0.0 <= weight <= 5.0):
        return False
    w = load_weights()
    w[kind] = float(weight)
    return save_weights(w)


def remove_weight(kind: str) -> bool:
    """Remove `kind` weight (revierte a default 1.0 implícito). Idempotente."""
    w = load_weights()
    if kind in w:
        del w[kind]
        return save_weights(w)
    return True


def apply_weight(kind: str, score: float) -> float:
    """Returns `score * weight(kind)` clamped a `[0.0, 1.0]`.

    Si el `kind` no tiene weight configurado se asume 1.0 (no-op).
    """
    w = load_weights().get(kind, 1.0)
    return max(0.0, min(1.0, score * w))


def list_weights() -> list[tuple[str, float]]:
    """Return `[(kind, weight), ...]` sorted by `kind` (lexicográfico)."""
    return sorted(load_weights().items())
