"""Phase 2.D — User-configurable signal weights por kind (versión SQL).

Multiplicador per-kind que el orchestrator aplica al score base de cada
candidate antes de filtrar por threshold:

    effective_score = candidate.score * weight(candidate.kind)

Default 1.0 (no-op) cuando el kind no tiene override. Tabla nueva
`rag_anticipate_kind_weights` — schema:

```sql
CREATE TABLE IF NOT EXISTS rag_anticipate_kind_weights (
    kind         TEXT PRIMARY KEY,
    weight       REAL NOT NULL DEFAULT 1.0,
    last_updated TEXT NOT NULL
);
```

## ¿Por qué SQL y no JSON?

El módulo legacy `rag_anticipate.weights` persiste en JSON
(`~/.local/share/obsidian-rag/anticipate_weights.json`) y se mantiene
por compatibilidad con tests + dashboards Phase 1. Phase 2.D quiere SQL
porque:

- Single-source-of-truth con el resto de la telemetría del agent.
- Soporta queries cross-tabla (ej. join con `rag_anticipate_feedback`
  para correlación weight × engagement).
- `last_updated` libre para auditoría de cambios.

Se mantiene también la API JSON-based para compatibilidad. El wire-up
del orchestrator consulta SQL primero; si no hay weight, cae al JSON
como fallback (legacy). Si no hay ninguno → 1.0.

## Range

`[0.0, 5.0]` por weight (mismo que la API JSON). Fuera de rango → set
falla con `False`. El producto `score × weight` se clamea a `[0, 1]`
(consistencia con el threshold `[0, 1]`).

## Silent-fail

Cualquier excepción (DB lock, schema corrupto) → return safe defaults
(1.0 para `get`, `False` para writes, `[]` para list).
"""

from __future__ import annotations

from datetime import datetime

# Range hard caps. Mismo que la API JSON existente.
_MIN_WEIGHT = 0.0
_MAX_WEIGHT = 5.0

# DDL idempotente. Misma forma que `rag_anticipate.feedback._FEEDBACK_DDL`.
_KIND_WEIGHTS_DDL: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS rag_anticipate_kind_weights ("
    " kind TEXT PRIMARY KEY,"
    " weight REAL NOT NULL DEFAULT 1.0,"
    " last_updated TEXT NOT NULL"
    ")",
)


def _ensure_table(conn) -> None:
    """Crea `rag_anticipate_kind_weights` si no existe. Idempotente."""
    for stmt in _KIND_WEIGHTS_DDL:
        conn.execute(stmt)


def _is_valid_weight(w: float) -> bool:
    return _MIN_WEIGHT <= float(w) <= _MAX_WEIGHT


def set_kind_weight(kind: str, weight: float) -> bool:
    """UPSERT del weight para `kind`. Returns `False` si el input es
    inválido (kind vacío, weight fuera de rango) o si el write falló."""
    if not kind or not isinstance(kind, str):
        return False
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
        return False
    if not _is_valid_weight(weight):
        return False
    ts = datetime.now().isoformat(timespec="seconds")
    try:
        import rag
        with rag._ragvec_state_conn() as conn:
            _ensure_table(conn)
            conn.execute(
                "INSERT INTO rag_anticipate_kind_weights (kind, weight, last_updated)"
                " VALUES (?, ?, ?)"
                " ON CONFLICT(kind) DO UPDATE SET"
                "   weight = excluded.weight,"
                "   last_updated = excluded.last_updated",
                (kind, float(weight), ts),
            )
            conn.commit()
        return True
    except Exception:
        return False


def _lookup_kind_weight(kind: str) -> float | None:
    """Lookup raw — devuelve `None` si no hay row para `kind` o el read
    falla. Útil para callers que necesitan distinguir "no override
    configurado" vs "override = 1.0" (ej. `apply_kind_weight` cae al
    fallback JSON sólo en el primer caso)."""
    if not kind:
        return None
    try:
        import rag
        with rag._ragvec_state_conn() as conn:
            _ensure_table(conn)
            row = conn.execute(
                "SELECT weight FROM rag_anticipate_kind_weights WHERE kind = ?",
                (kind,),
            ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    try:
        w = float(row[0])
    except Exception:
        return None
    if not _is_valid_weight(w):
        return None
    return w


def get_kind_weight(kind: str, *, default: float = 1.0) -> float:
    """Devuelve el weight de `kind`. Default 1.0 (no-op multiplier).

    Silent-fail: si la tabla no existe / la row no existe / el read
    falla → return `default`.
    """
    w = _lookup_kind_weight(kind)
    return w if w is not None else default


def list_kind_weights() -> list[dict]:
    """Returns `[{kind, weight, last_updated}, ...]` sorted por kind asc."""
    try:
        import rag
        with rag._ragvec_state_conn() as conn:
            _ensure_table(conn)
            rows = conn.execute(
                "SELECT kind, weight, last_updated"
                " FROM rag_anticipate_kind_weights"
                " ORDER BY kind ASC"
            ).fetchall()
    except Exception:
        return []
    return [
        {"kind": k, "weight": float(w), "last_updated": ts}
        for (k, w, ts) in rows
    ]


def reset_kind_weight(kind: str) -> bool:
    """Borra el override para `kind` (vuelve al default 1.0). Idempotente
    — si no había override, retorna True igual."""
    if not kind:
        return False
    try:
        import rag
        with rag._ragvec_state_conn() as conn:
            _ensure_table(conn)
            conn.execute(
                "DELETE FROM rag_anticipate_kind_weights WHERE kind = ?",
                (kind,),
            )
            conn.commit()
        return True
    except Exception:
        return False


def apply_kind_weight(kind: str, score: float) -> float:
    """Returns `score * weight(kind)` clampeado a `[0.0, 1.0]`.

    Si el kind no tiene weight configurado en SQL, intenta caer al
    fallback legacy (JSON `~/.local/share/obsidian-rag/anticipate_weights.json`).
    Si tampoco está allí, multiplier = 1.0.
    """
    w = _lookup_kind_weight(kind)
    if w is None:
        # Fallback al JSON store legacy (Phase 1). Silent-fail al 1.0.
        try:
            from rag_anticipate.weights import load_weights as _legacy_load
            legacy = _legacy_load()
            w = float(legacy.get(kind, 1.0))
        except Exception:
            w = 1.0
    try:
        result = float(score) * float(w)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, result))


__all__ = [
    "set_kind_weight",
    "get_kind_weight",
    "list_kind_weights",
    "reset_kind_weight",
    "apply_kind_weight",
]
