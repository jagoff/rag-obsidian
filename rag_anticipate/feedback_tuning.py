"""Phase 2.A — Feedback tuning del Anticipatory Agent.

El user reacciona a los pushes proactivos con `👍` / `👎` / `🔇`. Esos
ratings se persisten en `rag_anticipate_feedback` (vía
`rag_anticipate.feedback.record_feedback`). Este módulo lee esa tabla
+ el log de candidates (`rag_anticipate_candidates`) para inferir,
por kind, si el threshold actual del agent debería:

- **Subir** (más exigente, menos volumen) si el user mutea ese kind
  más del 50 % de las veces que recibe feedback (con al menos 3 mutes).
- **Bajar** (menos exigente, más volumen) si el user reacciona positivo
  más del 80 % de las veces (mute_ratio < 0.2 con al menos 3 positivos).

El delta se capea a `[-0.2, +0.2]` para no descalibrar bruscamente el
agent ante muestreo chico (ej. 3 mutes seguidos un día malo no deben
bloquear el kind para siempre).

## API

```python
delta = compute_kind_threshold_adjustment("anticipate-calendar")
# delta ∈ [-0.2, +0.2]
effective_threshold = base_threshold + delta
```

## Cache TTL 1h

El cómputo joinea `rag_anticipate_feedback` × `rag_anticipate_candidates`
en una ventana de 30 días. No tiene sentido re-ejecutarlo en cada tick
del daemon (cada 10 min) — los ratios cambian de a poco. Cache en
memoria con TTL 1h (3600 s). Invalidación: el daemon corre `rag
anticipate` cada 10 min en un proceso fresco vía launchd, así que la
"cache" en memoria del proceso se rebuilda cada vez que arranca el
proceso. Para tests + CLI manual donde el proceso vive más, el TTL
asegura que un feedback recién agregado se vea dentro de 1h.

## Kill-switch

`RAG_ANTICIPATE_FEEDBACK_TUNING=0` (o `false`) → siempre devuelve 0.0
sin tocar SQL. Default ON.

## Silent-fail

Cualquier excepción interna (DB inaccesible, schema corrupto) → return
0.0. Mismo principio que el resto del agent: prefiere un push extra
sobre tumbar todo el flujo.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

# Cap del delta para evitar runaway tuning con muestras chicas.
_DELTA_CAP_ABS = 0.2

# Mínimo de mutes/positives requeridos para activar el tuning. Evita
# que 1-2 reacciones random muevan el threshold.
_MIN_SAMPLES = 3

# Umbrales de ratio mute/(mute+positive) que disparan ajuste.
_MUTE_RATIO_HIGH = 0.5  # > → más exigente (delta positivo)
_MUTE_RATIO_LOW = 0.2   # < → más permisivo (delta negativo)

# Ventana de feedback considerada (días).
_WINDOW_DAYS = 30

# TTL del cache en memoria (segundos).
_CACHE_TTL_SECONDS = 3600.0

# Cache: {kind: (delta, expires_at_monotonic)}
_cache: dict[str, tuple[float, float]] = {}


def _is_disabled() -> bool:
    raw = os.environ.get("RAG_ANTICIPATE_FEEDBACK_TUNING", "1").strip().lower()
    return raw in ("0", "false", "no", "off")


def _query_kind_counts(kind: str) -> tuple[int, int, int]:
    """Cuenta (positive, negative, mute) per `kind` joineando feedback ×
    candidates en la ventana `_WINDOW_DAYS`.

    Returns (positive, negative, mute). En cualquier error retorna
    (0, 0, 0) — silent-fail.
    """
    cutoff = (datetime.now() - timedelta(days=_WINDOW_DAYS)).isoformat(timespec="seconds")
    try:
        import rag
        from rag_anticipate.feedback import _ensure_feedback_table
        with rag._ragvec_state_conn() as conn:
            # `rag_anticipate_candidates` la crea el bootstrap de tablas
            # de telemetría (ver `_ensure_telemetry_tables` en
            # rag/__init__.py). `rag_anticipate_feedback` se crea on-demand.
            _ensure_feedback_table(conn)
            rows = conn.execute(
                "SELECT f.rating, COUNT(*)"
                " FROM rag_anticipate_feedback f"
                " JOIN rag_anticipate_candidates c"
                "   ON c.dedup_key = f.dedup_key"
                " WHERE c.kind = ?"
                "   AND f.ts >= ?"
                " GROUP BY f.rating",
                (kind, cutoff),
            ).fetchall()
    except Exception:
        return (0, 0, 0)

    pos = neg = mute = 0
    for rating, count in rows:
        if rating == "positive":
            pos = int(count)
        elif rating == "negative":
            neg = int(count)
        elif rating == "mute":
            mute = int(count)
    return (pos, neg, mute)


def _delta_from_counts(positive: int, negative: int, mute: int) -> float:
    """Lógica pura — recibe counts, devuelve delta ∈ [-_DELTA_CAP_ABS, +_DELTA_CAP_ABS].

    - Si el sample size (mute + positive) es chico → 0.
    - Si mute_ratio > 0.5 con ≥3 mutes → delta positivo (más exigente).
    - Si mute_ratio < 0.2 con ≥3 positives → delta negativo (más permisivo).
    - En zona "tibia" → 0.

    El delta crece linealmente con la distancia al umbral, capeado a
    ±_DELTA_CAP_ABS (default 0.2).
    """
    sample = mute + positive
    if sample <= 0:
        return 0.0
    ratio = mute / float(sample)

    if ratio > _MUTE_RATIO_HIGH and mute >= _MIN_SAMPLES:
        # Crece desde +0.10 (en ratio=0.5) hasta +0.20 (en ratio=1.0).
        delta = 0.10 + (ratio - _MUTE_RATIO_HIGH) * 0.4
        return min(_DELTA_CAP_ABS, delta)

    if ratio < _MUTE_RATIO_LOW and positive >= _MIN_SAMPLES:
        # Decrece desde -0.10 (en ratio=0.2) hasta -0.20 (en ratio=0.0).
        delta = -0.10 - (_MUTE_RATIO_LOW - ratio) * 0.5
        return max(-_DELTA_CAP_ABS, delta)

    return 0.0


def compute_kind_threshold_adjustment(kind: str, *, use_cache: bool = True) -> float:
    """Devuelve el delta a aplicar al threshold base de `kind`.

    Args:
        kind: ej. `"anticipate-calendar"`, `"anticipate-echo"`. Si está
            vacío → 0.
        use_cache: pasar `False` desde tests para skipear la cache.

    Returns:
        Delta float en `[-0.2, +0.2]`. Default 0.0 cuando hay poco
        feedback o el feature está deshabilitado.

    Silent-fail: cualquier excepción interna → 0.0.
    """
    if not kind:
        return 0.0
    if _is_disabled():
        return 0.0

    now_mono = time.monotonic()
    if use_cache:
        cached = _cache.get(kind)
        if cached is not None:
            delta, expires_at = cached
            if now_mono < expires_at:
                return delta

    try:
        positive, negative, mute = _query_kind_counts(kind)
        delta = _delta_from_counts(positive, negative, mute)
    except Exception:
        delta = 0.0

    if use_cache:
        _cache[kind] = (delta, now_mono + _CACHE_TTL_SECONDS)
    return delta


def kind_feedback_summary(kind: str) -> dict:
    """Helper para CLI/dashboard — devuelve counts + ratio + delta.

    Útil para `rag anticipate feedback stats` per-kind. Silent-fail:
    en error retorna shape vacío con ceros.
    """
    try:
        positive, negative, mute = _query_kind_counts(kind)
    except Exception:
        positive = negative = mute = 0
    sample = positive + mute
    ratio = (mute / sample) if sample > 0 else 0.0
    return {
        "kind": kind,
        "positive": positive,
        "negative": negative,
        "mute": mute,
        "total": positive + negative + mute,
        "mute_ratio": round(ratio, 3),
        "delta": round(_delta_from_counts(positive, negative, mute), 3),
    }


def all_kinds_feedback_summary() -> list[dict]:
    """Lista todos los kinds que tienen al menos 1 candidate logueado y
    devuelve el summary de cada uno. Sorted por `total` desc.

    Para `rag anticipate feedback stats` global.
    """
    try:
        import rag
        with rag._ragvec_state_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT kind FROM rag_anticipate_candidates "
                " ORDER BY kind"
            ).fetchall()
    except Exception:
        return []
    return sorted(
        (kind_feedback_summary(k) for (k,) in rows),
        key=lambda d: d["total"],
        reverse=True,
    )


def reset_cache() -> None:
    """Limpia la cache. Tests + manual debug."""
    _cache.clear()


__all__ = [
    "compute_kind_threshold_adjustment",
    "kind_feedback_summary",
    "all_kinds_feedback_summary",
    "reset_cache",
]
